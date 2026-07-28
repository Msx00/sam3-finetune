import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR

from lora_layers import LoRALinear
from models.moe_injector import HierarchicalMoEController
from models.moe_lora import RoutedMoELinear
from models.training_stages import STAGE_ACTIVE_LOSSES, STAGE_CHECKPOINTS, StageTrainingManager


class DummyLayer(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.projection = LoRALinear(nn.Linear(dim, dim), rank=2, alpha=2)
        self.norm = nn.LayerNorm(dim)


class DummyModel(nn.Module):
    def __init__(self, controller):
        super().__init__()
        self.encoder_lora = LoRALinear(nn.Linear(4, 4), rank=2, alpha=2)
        self.transformer = nn.Module()
        self.transformer.decoder = nn.Module()
        self.transformer.decoder.layers = nn.ModuleList([DummyLayer() for _ in range(6)])
        self.add_module("moe_controller", controller)


def build_components(with_svanet=True):
    controller = HierarchicalMoEController(
        embed_dim=4, rank=2, alpha=2, dropout=0,
        router_hidden_dim=4, routing_mode="soft", temperature=1, dac=False,
    )
    model = DummyModel(controller)
    svanet = nn.Sequential(nn.Conv2d(3, 2, 1)) if with_svanet else None
    return model, controller, svanet


def test_routed_projection_preserves_shared_lora():
    base = LoRALinear(nn.Linear(4, 4), rank=2, alpha=2)
    controller = HierarchicalMoEController(
        embed_dim=4, rank=2, alpha=2, dropout=0,
        router_hidden_dim=4, routing_mode="soft", temperature=1, dac=False,
    )
    routed = RoutedMoELinear(base, controller)
    values = torch.randn(2, 3, 4)
    assert routed.base_linear is base
    assert torch.allclose(routed(values), base(values))


def test_all_stage_freeze_policies_and_optimizer_groups(tmp_path):
    expected = {
        1: {"shared_lora"},
        2: {"router", "auxiliary_head"},
        3: {"expert_lora"},
        4: {"svanet"},
        5: {"router", "auxiliary_head", "expert_lora", "decoder", "svanet"},
    }
    for stage in range(1, 6):
        model, controller, svanet = build_components()
        manager = StageTrainingManager(
            model, controller, stage,
            optimizer_config={
                "learning_rate": 1e-3, "router_lr": 2e-3,
                "expert_lora_lr": 3e-3, "svanet_lr": 4e-3,
            },
            svanet_adapter=svanet,
        )
        assert manager.trainable_group_names() == expected[stage]
        assert manager.checkpoint_name == STAGE_CHECKPOINTS[stage]
        base_losses = {name: 1.0 for name in set().union(*STAGE_ACTIVE_LOSSES.values())}
        active_weights = manager.active_loss_weights(base_losses)
        assert {name for name, value in active_weights.items() if value} == STAGE_ACTIVE_LOSSES[stage]
        optimizer = manager.build_optimizer()
        assert {group["group_name"] for group in optimizer.param_groups} == expected[stage]
        optimized = {
            id(parameter) for group in optimizer.param_groups for parameter in group["params"]
        }
        assert all(parameter.requires_grad for group in optimizer.param_groups for parameter in group["params"])
        assert not any(
            id(parameter) in optimized
            for parameters in manager.groups.values()
            for parameter in parameters if not parameter.requires_grad
        )

        if stage == 3:
            path = tmp_path / manager.checkpoint_name
            scheduler = StepLR(optimizer, step_size=1)
            manager.save_checkpoint(
                path, optimizer, epoch=1, best_loss=0.5, scheduler=scheduler,
                selected_patient_ids={"mr_patient_ids": [1]},
                area_thresholds={"small_max": 0.01},
                boundary_thresholds={"mr": {"contrast_low": 0.2}},
                config={"training": {"stage": 3}},
            )
            assert path.is_file()
            payload = manager.load_checkpoint(path, allowed_stages={3})
            required = {
                "model_state", "router_state", "expert_lora_state",
                "shared_lora_state", "svanet_state", "optimizer_state",
                "scheduler_state", "epoch", "stage", "best_metric",
                "selected_patient_ids", "area_thresholds",
                "boundary_thresholds", "config",
            }
            assert required <= payload.keys()
            resumed = manager.resume(path, optimizer, scheduler)
            assert resumed["stage"] == 3
            try:
                manager.load_checkpoint(path, allowed_stages={2})
                raise AssertionError("stage mismatch was not rejected")
            except ValueError as error:
                assert "incompatible" in str(error)
