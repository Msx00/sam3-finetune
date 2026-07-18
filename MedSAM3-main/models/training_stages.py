"""Five-stage training control for Hierarchical LoRA-MoE-SAM3."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Set

import torch
from torch import nn
from torch.optim import AdamW


STAGE_NAMES = {
    1: "baseline",
    2: "router",
    3: "moe",
    4: "svanet",
    5: "joint",
}

STAGE_CHECKPOINTS = {
    stage: f"stage{stage}_{name}_best.pt" for stage, name in STAGE_NAMES.items()
}

STAGE_ACTIVE_LOSSES = {
    1: {"sam3_loss", "aux_loss"},
    2: {
        "aux_loss", "modality_loss", "area_loss", "area_reg_loss",
        "boundary_router_loss",
    },
    3: {"sam3_loss", "boundary_seg_loss", "load_balance_loss"},
    4: {"refine_loss"},
    5: {
        "sam3_loss", "aux_loss", "modality_loss", "area_loss",
        "area_reg_loss", "boundary_router_loss", "boundary_seg_loss",
        "load_balance_loss", "refine_loss",
    },
}


def _parameter_ids(module: Optional[nn.Module]) -> Set[int]:
    return set() if module is None else {id(parameter) for parameter in module.parameters()}


class StageTrainingManager:
    """Own parameter ownership, freezing, loss gates and optimizer groups."""

    def __init__(
        self,
        model: nn.Module,
        controller: nn.Module,
        stage: int,
        optimizer_config: Mapping[str, float],
        svanet_adapter: Optional[nn.Module] = None,
        stage_config: Optional[Mapping[str, object]] = None,
    ) -> None:
        if stage not in STAGE_NAMES:
            raise ValueError(f"training stage must be 1..5, got {stage}")
        self.model = model
        self.controller = controller
        self.svanet_adapter = svanet_adapter
        self.stage = int(stage)
        self.optimizer_config = dict(optimizer_config)
        self.stage_config = dict(stage_config or {})
        self.groups = self._collect_parameter_groups()
        self.apply_freeze_policy()

    def _collect_parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        controller_ids = _parameter_ids(self.controller)
        svanet_ids = _parameter_ids(self.svanet_adapter)
        expert_ids = _parameter_ids(getattr(self.controller, "expert_pool", None))
        router_module = getattr(self.controller, "router", None)
        router_ids = _parameter_ids(router_module)
        ratio_head = getattr(getattr(router_module, "area_router", None), "ratio_head", None)
        auxiliary_ids = _parameter_ids(ratio_head)
        router_ids -= auxiliary_ids

        shared_ids: Set[int] = set()
        for module in self.model.modules():
            if module.__class__.__name__ == "LoRALinear" and hasattr(module, "lora"):
                shared_ids.update(_parameter_ids(module.lora))
        shared_ids -= controller_ids

        decoder_ids: Set[int] = set()
        decoder = getattr(getattr(self.model, "transformer", None), "decoder", None)
        if decoder is not None:
            for layer in list(decoder.layers)[3:6]:
                decoder_ids.update(_parameter_ids(layer))
        decoder_ids -= controller_ids | shared_ids

        all_named = list(self.model.named_parameters())
        if self.svanet_adapter is not None:
            all_named += [
                (f"svanet_adapter.{name}", parameter)
                for name, parameter in self.svanet_adapter.named_parameters()
            ]
        id_to_param = {id(parameter): parameter for _, parameter in all_named}
        group_ids = {
            "shared_lora": shared_ids,
            "expert_lora": expert_ids,
            "router": router_ids,
            "auxiliary_head": auxiliary_ids,
            "svanet": svanet_ids,
            "decoder": decoder_ids,
        }
        claimed: Set[int] = set()
        groups: Dict[str, List[nn.Parameter]] = {}
        for name, ids in group_ids.items():
            overlap = claimed & ids
            if overlap:
                raise RuntimeError(f"Parameters assigned to multiple groups: {name}")
            claimed.update(ids)
            groups[name] = [id_to_param[item] for item in ids if item in id_to_param]
        return groups

    def trainable_group_names(self) -> Set[str]:
        policy = {
            1: {"shared_lora"},
            2: {"router", "auxiliary_head"},
            3: {"expert_lora"},
            4: {"svanet"},
            5: {"router", "auxiliary_head", "expert_lora", "decoder", "svanet"},
        }[self.stage]
        if self.stage == 1 and self.stage_config.get("train_decoder_stage1", False):
            policy.add("decoder")
        if self.stage == 3 and self.stage_config.get("train_router_stage3", False):
            policy.update({"router", "auxiliary_head"})
        if self.stage == 5 and self.stage_config.get("train_shared_lora_stage5", False):
            policy.add("shared_lora")
        return policy

    def apply_freeze_policy(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        if self.svanet_adapter is not None:
            for parameter in self.svanet_adapter.parameters():
                parameter.requires_grad = False
        for name in self.trainable_group_names():
            for parameter in self.groups[name]:
                parameter.requires_grad = True
        if not any(parameter.requires_grad for group in self.groups.values() for parameter in group):
            raise RuntimeError(f"Stage {self.stage} has no trainable parameters")

    def set_module_modes(self, training: bool = True) -> None:
        if not training:
            self.model.eval()
            if self.svanet_adapter is not None:
                self.svanet_adapter.eval()
            return
        if self.stage == 4:
            self.model.eval()
        else:
            self.model.train()
        if self.svanet_adapter is not None:
            self.svanet_adapter.train(self.stage in {4, 5})

    def active_loss_weights(self, base_weights: Mapping[str, float]) -> Dict[str, float]:
        active = set(STAGE_ACTIVE_LOSSES[self.stage])
        if self.stage == 1 and not self.stage_config.get("use_aux_loss_stage1", True):
            active.discard("aux_loss")
        if self.stage == 3 and self.stage_config.get("use_router_losses_stage3", False):
            active.update({"modality_loss", "area_loss", "area_reg_loss", "boundary_router_loss"})
        return {name: float(value) if name in active else 0.0 for name, value in base_weights.items()}

    def build_optimizer(self) -> AdamW:
        learning_rates = {
            "shared_lora": "shared_lora_lr",
            "expert_lora": "expert_lora_lr",
            "router": "router_lr",
            "auxiliary_head": "auxiliary_head_lr",
            "svanet": "svanet_lr",
            "decoder": "decoder_lr",
        }
        default_lr = float(self.optimizer_config.get("learning_rate", 5e-5))
        parameter_groups = []
        for name in sorted(self.trainable_group_names()):
            parameters = [parameter for parameter in self.groups[name] if parameter.requires_grad]
            if parameters:
                parameter_groups.append({
                    "params": parameters,
                    "lr": float(self.optimizer_config.get(learning_rates[name], default_lr)),
                    "group_name": name,
                })
        optimizer = AdamW(
            parameter_groups,
            lr=default_lr,
            weight_decay=float(self.optimizer_config.get("weight_decay", 0.01)),
        )
        optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        frozen = {id(parameter) for group in self.groups.values() for parameter in group if not parameter.requires_grad}
        if optimized & frozen:
            raise RuntimeError("Frozen parameters leaked into the stage optimizer")
        return optimizer

    def summary(self) -> Dict[str, Dict[str, int | float | bool]]:
        return {
            name: {
                "tensors": len(parameters),
                "parameters": sum(parameter.numel() for parameter in parameters),
                "trainable": any(parameter.requires_grad for parameter in parameters),
            }
            for name, parameters in self.groups.items()
        }

    def print_trainable_parameter_groups(self) -> None:
        print(f"Training stage {self.stage}: {STAGE_NAMES[self.stage]}")
        lr_keys = {
            "shared_lora": "shared_lora_lr", "expert_lora": "expert_lora_lr",
            "router": "router_lr", "auxiliary_head": "auxiliary_head_lr",
            "svanet": "svanet_lr", "decoder": "decoder_lr",
        }
        default_lr = float(self.optimizer_config.get("learning_rate", 5e-5))
        for name, values in self.summary().items():
            status = "train" if values["trainable"] else "frozen"
            lr = float(self.optimizer_config.get(lr_keys[name], default_lr))
            print(
                f"  {name}: {status}, tensors={values['tensors']}, "
                f"parameters={values['parameters']:,}, lr={lr:.3e}, "
                f"requires_grad={values['trainable']}"
            )

    def print_summary(self) -> None:
        """Backward-compatible alias."""
        self.print_trainable_parameter_groups()

    @property
    def checkpoint_name(self) -> str:
        return STAGE_CHECKPOINTS[self.stage]

    def save_checkpoint(
        self,
        path: str | Path,
        optimizer: AdamW,
        epoch: int,
        best_loss: float,
        scheduler: Optional[object] = None,
        selected_patient_ids: Optional[Mapping[str, object]] = None,
        area_thresholds: Optional[Mapping[str, object]] = None,
        boundary_thresholds: Optional[Mapping[str, object]] = None,
        config: Optional[Mapping[str, object]] = None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        shared_names = {
            id(parameter) for parameter in self.groups["shared_lora"]
        }
        shared_state = {
            name: parameter.detach().cpu()
            for name, parameter in self.model.named_parameters()
            if id(parameter) in shared_names
        }
        payload = {
            "format_version": 2,
            "stage": self.stage,
            "stage_name": STAGE_NAMES[self.stage],
            "epoch": int(epoch),
            "best_metric": float(best_loss),
            "best_loss": float(best_loss),
            "model_state": self.model.state_dict(),
            "router_state": self.controller.router.state_dict(),
            "expert_lora_state": self.controller.expert_pool.state_dict(),
            "shared_lora_state": shared_state,
            "controller_state": self.controller.state_dict(),
            "svanet_state": None if self.svanet_adapter is None else self.svanet_adapter.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "selected_patient_ids": dict(selected_patient_ids or {}),
            "area_thresholds": dict(area_thresholds or {}),
            "boundary_thresholds": dict(boundary_thresholds or {}),
            "config": dict(config or {}),
        }
        torch.save(payload, path)

    def load_checkpoint(
        self, path: str | Path, allowed_stages: Optional[Iterable[int]] = None
    ) -> Dict[str, object]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"Unsupported stage checkpoint: {path}")
        source_stage = payload.get("stage")
        if allowed_stages is not None and source_stage not in set(allowed_stages):
            raise ValueError(
                f"Checkpoint stage {source_stage} is incompatible with allowed "
                f"stages {sorted(set(allowed_stages))}: {path}"
            )
        if payload.get("model_state"):
            self.model.load_state_dict(payload["model_state"], strict=False)
        named_parameters = dict(self.model.named_parameters())
        for name, value in (payload.get("shared_lora_state") or {}).items():
            if name in named_parameters:
                named_parameters[name].data.copy_(value.to(named_parameters[name]))
        if payload.get("controller_state"):
            self.controller.load_state_dict(payload["controller_state"], strict=False)
        if self.svanet_adapter is not None and payload.get("svanet_state"):
            self.svanet_adapter.load_state_dict(payload["svanet_state"], strict=False)
        return payload

    def load_dependencies(self, model_config: Mapping[str, object]) -> List[str]:
        dependencies = {
            1: [],
            2: ["stage1_checkpoint"],
            3: ["stage1_checkpoint", "stage2_checkpoint"],
            4: ["stage3_checkpoint"],
            5: ["stage3_checkpoint", "stage4_checkpoint"],
        }[self.stage]
        loaded = []
        for key in dependencies:
            value = model_config.get(key)
            if not value:
                continue
            path = Path(str(value))
            if not path.is_file():
                raise FileNotFoundError(f"Configured {key} does not exist: {path}")
            expected_stage = int(key.removeprefix("stage").removesuffix("_checkpoint"))
            self.load_checkpoint(path, allowed_stages={expected_stage})
            loaded.append(str(path))
        return loaded

    def resume(
        self,
        path: str | Path,
        optimizer: AdamW,
        scheduler: Optional[object] = None,
    ) -> Dict[str, object]:
        payload = self.load_checkpoint(path, allowed_stages={self.stage})
        if payload.get("optimizer_state"):
            optimizer.load_state_dict(payload["optimizer_state"])
        scheduler_state = payload.get("scheduler_state")
        if scheduler is not None and scheduler_state:
            scheduler.load_state_dict(scheduler_state)
        elif scheduler_state and scheduler is None:
            raise ValueError(
                "Checkpoint contains scheduler state but scheduler is disabled"
            )
        return payload
