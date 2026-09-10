import torch
from torch import nn
from torch.nn import functional as F
from types import SimpleNamespace

from models.moe_injector import (
    HierarchicalMoEController,
    _resolve_backbone_fpn_capture,
    inject_hierarchical_moe,
)
from models.moe_lora import RoutedMoELinear


class _DummyAttention(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(self, name, nn.Linear(dim, dim))


class _DummyLayer(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.self_attn = _DummyAttention(dim)
        self.ca_text = _DummyAttention(dim)
        self.cross_attn = _DummyAttention(dim)


class _DummyDecoder(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.d_model = dim
        self.dac = False
        self.layers = nn.ModuleList([_DummyLayer(dim) for _ in range(6)])


class _DummyModel(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.decoder = _DummyDecoder(dim)


class _DummyRoutingSource(nn.Module):
    def forward(self, memory, memory_spatial_shapes):
        return memory, memory_spatial_shapes


def _controller(**kwargs):
    defaults = dict(
        embed_dim=8,
        rank=2,
        alpha=4,
        dropout=0,
        router_hidden_dim=8,
        routing_mode="soft",
        temperature=1,
        dac=False,
    )
    defaults.update(kwargs)
    return HierarchicalMoEController(**defaults)


def test_injector_respects_layers_and_exact_target_modules():
    model = _DummyModel()
    controller = inject_hierarchical_moe(
        model,
        {
            "embed_dim": 8,
            "rank": 2,
            "decoder_layers": [4, 6],
            "target_modules": ["cross_attn.q_proj", "cross_attn.v_proj"],
            "router_source_layer": 3,
            "routing_feature_source": "decoder_memory",
            "residual_scale_init": 0.1,
        },
        verbose=False,
    )
    for layer_number, layer in enumerate(model.transformer.decoder.layers, start=1):
        expected = layer_number in {4, 6}
        assert isinstance(layer.cross_attn.q_proj, RoutedMoELinear) is expected
        assert isinstance(layer.cross_attn.v_proj, RoutedMoELinear) is expected
        assert not isinstance(layer.cross_attn.k_proj, RoutedMoELinear)
        assert not isinstance(layer.self_attn.q_proj, RoutedMoELinear)
    assert len(controller.expert_pool.area_residual_scales) == 4
    for scale in controller.expert_pool.area_residual_scales.values():
        assert torch.allclose(scale.detach(), torch.tensor(0.1))


def test_backbone_fpn_index_matches_sam3_scalp_and_feature_levels():
    model = nn.Module()
    model.num_feature_levels = 1
    model.backbone = nn.Module()
    model.backbone.scalp = 1
    model.backbone.vision_backbone = nn.Module()
    model.backbone.vision_backbone.convs = nn.ModuleList(
        [nn.Identity() for _ in range(4)]
    )
    module, index = _resolve_backbone_fpn_capture(model)
    assert index == 4 - 1 - 1 == 2
    assert module is model.backbone.vision_backbone.convs[2]


def test_backbone_fpn_routes_ignore_prompt_memory_and_align_img_ids():
    controller = _controller(routing_feature_source="backbone_fpn")
    controller.eval()
    source = _DummyRoutingSource()
    spatial_shapes = torch.tensor([[4, 4]])
    raw_fpn = torch.randn(3, 8, 4, 4)
    input_batch = SimpleNamespace(
        find_inputs=[SimpleNamespace(img_ids=torch.tensor([2, 0]))]
    )

    controller.begin_model_forward(None, (input_batch,), {})
    controller.capture_backbone_fpn(None, (), raw_fpn)
    first_q = torch.randn(4, 2, 8)
    controller.route_after_layer3(
        source,
        (),
        {
            "memory": torch.randn(16, 2, 8),
            "memory_spatial_shapes": spatial_shapes,
        },
        (first_q, None),
    )
    first = controller.current_routes
    assert torch.allclose(controller.last_local_feature, raw_fpn[[2, 0]])

    controller.begin_model_forward(None, (input_batch,), {})
    controller.capture_backbone_fpn(None, (), raw_fpn)
    controller.route_after_layer3(
        source,
        (),
        {
            # Simulate a completely different prompt-fused encoder memory.
            "memory": torch.randn(16, 2, 8) * 100,
            "memory_spatial_shapes": spatial_shapes,
        },
        (torch.randn(4, 2, 8) * 100, None),
    )
    second = controller.current_routes
    for key in (
        "modality_logits",
        "area_logits_all",
        "boundary_logits_all",
        "area_joint",
        "boundary_joint",
        "image_locator_logits",
    ):
        assert torch.allclose(first[key], second[key])
    assert not torch.allclose(first["coarse_mask_p3"], second["coarse_mask_p3"])


def test_backbone_fpn_route_fails_if_raw_feature_was_not_captured():
    controller = _controller(routing_feature_source="backbone_fpn")
    input_batch = SimpleNamespace(
        find_inputs=[SimpleNamespace(img_ids=torch.tensor([0]))]
    )
    controller.begin_model_forward(None, (input_batch,), {})
    try:
        controller.route_after_layer3(
            _DummyRoutingSource(),
            (),
            {
                "memory": torch.randn(16, 1, 8),
                "memory_spatial_shapes": torch.tensor([[4, 4]]),
            },
            (torch.randn(4, 1, 8), None),
        )
    except RuntimeError as error:
        assert "was not captured" in str(error)
    else:
        raise AssertionError("missing raw backbone feature must fail fast")


def test_projection_scales_are_independent_between_expert_families():
    controller = _controller()
    routed = RoutedMoELinear(
        nn.Linear(8, 8),
        controller,
        projection_key="test_projection",
        residual_scale_init=0.25,
    )
    area_joint = torch.zeros(1, 2, 3)
    area_joint[0, 0, 0] = 1
    boundary_joint = torch.zeros_like(area_joint)
    controller.current_routes = {
        "area_joint": area_joint,
        "boundary_joint": boundary_joint,
    }
    torch.nn.init.normal_(
        controller.expert_pool.experts["MR_area_small"].B, std=0.1
    )
    values = torch.randn(1, 3, 8)
    base = routed.base_linear(values)
    area = controller.expert_pool.forward_family(values, area_joint, "area")
    assert torch.allclose(routed(values), base + 0.25 * area)
    with torch.no_grad():
        controller.expert_pool.area_residual_scales["test_projection"].fill_(10)
    assert torch.allclose(routed(values), base + area)
    assert (
        controller.expert_pool.area_residual_scales["test_projection"]
        is not controller.expert_pool.boundary_residual_scales["test_projection"]
    )


def test_child_supervision_selects_ground_truth_modality_branch():
    controller = _controller(router_regularizer="batch_prior")
    q3 = torch.randn(4, 2, 8)
    local = torch.randn(2, 8, 6, 6)
    routes = controller.router(q3, local)
    controller.current_routes = routes
    controller.set_routing_targets(
        {
            "modality": torch.tensor([0, 1]),
            "area": torch.tensor([2, 0]),
            "boundary": torch.tensor([1, 2]),
        }
    )
    losses = controller.routing_supervision_losses()
    batch = torch.arange(2)
    expected_area = F.cross_entropy(
        routes["area_logits_all"][batch, torch.tensor([0, 1])],
        torch.tensor([2, 0]),
    )
    assert torch.allclose(losses["area_loss"], expected_area)
    assert torch.isfinite(losses["load_balance_loss"])
    losses["area_loss"].backward()
    conditional_head = controller.router.area_router.conditional_classifier.net[-1]
    assert conditional_head.weight.grad is not None
    assert float(conditional_head.weight.grad.abs().sum()) > 0


def test_selected_child_logits_follow_current_parent_route():
    controller = _controller(routing_mode="top1")
    routes = controller.router(torch.randn(4, 2, 8), torch.randn(2, 8, 6, 6))
    selected_parent = routes["modality"].detach().argmax(dim=-1)
    batch = torch.arange(2)
    assert torch.allclose(
        routes["area_selected_logits"],
        routes["area_logits_all"][batch, selected_parent],
    )

    controller.train()
    controller.set_routing_targets(
        {"modality": 1 - selected_parent},
        {"modality": torch.ones(2, dtype=torch.bool)},
    )
    taught = controller.apply_teacher_routing(routes)
    taught_parent = taught["modality"].argmax(dim=-1)
    assert torch.allclose(
        taught["area_selected_logits"],
        routes["area_logits_all"][batch, taught_parent],
    )


def test_teacher_parent_and_child_form_one_coherent_joint_route():
    controller = _controller(
        routing_mode="top1",
        confidence_routing=True,
        confidence_low_threshold=1.0,
        confidence_high_threshold=1.0,
    )
    controller.train()
    routes = controller.router(torch.randn(4, 2, 8), torch.randn(2, 8, 6, 6))
    assert torch.count_nonzero(routes["area_joint"].detach()) == 0
    controller.set_routing_targets(
        {
            "modality": torch.tensor([0, 1]),
            "area": torch.tensor([2, 0]),
            "boundary": torch.tensor([1, 2]),
        },
        {
            "modality": torch.tensor([True, True]),
            "area": torch.tensor([True, True]),
            "boundary": torch.tensor([True, True]),
        },
    )
    taught = controller.apply_teacher_routing(routes)
    assert torch.equal(taught["area_joint"].detach().flatten(1).sum(dim=1), torch.ones(2))
    assert torch.equal(taught["area_routing_policy"], torch.full((2,), 2))
    assert torch.equal(
        taught["area_joint"].detach().flatten(1).argmax(dim=1), torch.tensor([2, 3])
    )
    assert torch.equal(
        taught["boundary_joint"].detach().flatten(1).argmax(dim=1),
        torch.tensor([1, 5]),
    )
