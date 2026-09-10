import torch

from models.moe_lora import ExpertPool
from models.router import HierarchicalRouter


def test_router_forward_shapes_probabilities_and_nan():
    router = HierarchicalRouter(
        embed_dim=8, hidden_dim=8, routing_mode="top1", temperature=1.0
    )
    q3 = torch.randn(4, 2, 8)
    local = torch.randn(2, 8, 6, 6)
    output = router(q3, local)
    assert output["modality_logits"].shape == (2, 2)
    assert output["area_logits"].shape == (2, 3)
    assert output["boundary_logits"].shape == (2, 3)
    assert output["area_logits_all"].shape == (2, 2, 3)
    assert output["boundary_logits_all"].shape == (2, 2, 3)
    assert output["image_locator_logits"].shape == (2, 1, 6, 6)
    assert output["coarse_mask_p3"].shape == (2, 4, 6, 6)
    for name in ("modality_soft", "area_soft", "boundary_soft"):
        assert torch.isfinite(output[name]).all()
        assert torch.allclose(output[name].sum(dim=-1), torch.ones(2))
    assert torch.allclose(
        output["area_joint_soft"],
        output["modality_soft"][:, :, None] * output["area_soft_all"],
    )
    assert torch.allclose(
        output["boundary_joint_soft"],
        output["modality_soft"][:, :, None] * output["boundary_soft_all"],
    )


def test_image_locator_routes_are_independent_of_prompt_query():
    router = HierarchicalRouter(
        embed_dim=8, hidden_dim=8, routing_mode="soft", use_image_locator=True
    ).eval()
    local = torch.randn(2, 8, 6, 6)
    first = router(torch.randn(4, 2, 8), local)
    second = router(torch.randn(4, 2, 8) * 100, local)
    for key in (
        "modality_logits", "area_logits_all", "boundary_logits_all",
        "area_joint", "boundary_joint", "image_locator_logits",
    ):
        assert torch.allclose(first[key], second[key])
    assert not torch.allclose(first["coarse_mask_p3"], second["coarse_mask_p3"])


def test_independent_router_checkpoint_can_warm_start_conditional_router():
    independent = HierarchicalRouter(
        embed_dim=8, hidden_dim=8, conditional_hierarchy=False
    )
    conditional = HierarchicalRouter(
        embed_dim=8, hidden_dim=8, conditional_hierarchy=True
    )
    incompatible = conditional.load_state_dict(independent.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all("conditional_classifier" in key for key in incompatible.missing_keys)


def test_confidence_policy_supports_top1_top2_and_shared_fallback():
    q3 = torch.randn(4, 2, 8)
    local = torch.randn(2, 8, 6, 6)

    top1_router = HierarchicalRouter(
        embed_dim=8, hidden_dim=8, confidence_routing=True,
        confidence_low_threshold=0.0, confidence_high_threshold=0.0,
    )
    top1 = top1_router(q3, local)
    assert torch.equal(top1["area_routing_policy"], torch.full((2,), 2))
    assert torch.equal(
        (top1["area_joint"].detach() != 0).flatten(1).sum(dim=1),
        torch.ones(2, dtype=torch.long),
    )

    top2_router = HierarchicalRouter(
        embed_dim=8, hidden_dim=8, confidence_routing=True,
        confidence_low_threshold=0.0, confidence_high_threshold=1.0,
        confidence_top_k=2,
    )
    top2 = top2_router(q3, local)
    assert torch.equal(top2["area_routing_policy"], torch.ones(2, dtype=torch.long))
    assert torch.equal(
        (top2["area_joint"].detach() != 0).flatten(1).sum(dim=1),
        torch.full((2,), 2, dtype=torch.long),
    )

    fallback_router = HierarchicalRouter(
        embed_dim=8, hidden_dim=8, confidence_routing=True,
        confidence_low_threshold=1.0, confidence_high_threshold=1.0,
    )
    fallback = fallback_router(q3, local)
    assert torch.equal(fallback["area_routing_policy"], torch.zeros(2, dtype=torch.long))
    assert torch.count_nonzero(fallback["area_joint"].detach()) == 0


def test_ratio_head_is_reachable_from_routed_forward_graph():
    router = HierarchicalRouter(
        embed_dim=8, hidden_dim=8, routing_mode="top1", temperature=1.0
    )
    q3 = torch.randn(4, 2, 8)
    local = torch.randn(2, 8, 6, 6)
    routes = router(q3, local)
    ratio_bias = router.area_router.ratio_head[-1].bias

    # DDP with find_unused_parameters=True traverses the graph returned by the
    # model.  ratio_head must be reachable through the routed area weights even
    # though that dependency intentionally contributes a zero gradient.
    route_objective = (routes["area"] * torch.tensor([1.0, 2.0, 3.0])).sum()
    route_grad = torch.autograd.grad(
        route_objective, ratio_bias, retain_graph=True, allow_unused=True
    )[0]
    assert route_grad is not None
    assert torch.count_nonzero(route_grad) == 0

    # The actual area-ratio regression path must still provide real gradients.
    ratio_loss = torch.nn.functional.smooth_l1_loss(
        routes["area_ratio_pred"], torch.zeros_like(routes["area_ratio_pred"])
    )
    ratio_loss.backward()
    assert ratio_bias.grad is not None
    assert torch.isfinite(ratio_bias.grad).all()
    assert float(ratio_bias.grad.abs().sum()) > 0


def test_top1_expert_and_backward_are_sparse():
    router = HierarchicalRouter(
        embed_dim=8, hidden_dim=8, routing_mode="top1", temperature=1.0
    )
    pool = ExpertPool(8, 8, rank=2, alpha=4)
    q3 = torch.randn(4, 1, 8)
    local = torch.randn(1, 8, 6, 6)
    routes = router(q3, local)
    modality_index = int(routes["modality"].argmax(dim=-1).item())
    area_index = int(routes["area"].argmax(dim=-1).item())
    boundary_index = int(routes["boundary"].argmax(dim=-1).item())
    modality = ("MR", "US")[modality_index]
    area_name = ("small", "medium", "large")[area_index]
    boundary_name = ("clear", "fuzzy", "complex")[boundary_index]
    selected = {
        f"{modality}_area_{area_name}",
        f"{modality}_boundary_{boundary_name}",
    }
    for name in selected:
        torch.nn.init.normal_(pool.experts[name].B, std=0.02)
    values = torch.randn(1, 5, 8)
    output = pool.forward_family(values, routes["area_joint"], "area")
    output = output + pool.forward_family(values, routes["boundary_joint"], "boundary")
    assert output.shape == values.shape
    output.square().mean().backward()
    for name, expert in pool.experts.items():
        if name in selected:
            assert expert.B.grad is not None
            assert float(expert.B.grad.norm()) > 0
        else:
            assert expert.A.grad is None
            assert expert.B.grad is None
    assert any(
        parameter.grad is not None and float(parameter.grad.norm()) > 0
        for parameter in router.parameters()
    )
