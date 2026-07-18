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
    assert output["coarse_mask_p3"].shape == (2, 4, 6, 6)
    for name in ("modality_soft", "area_soft", "boundary_soft"):
        assert torch.isfinite(output[name]).all()
        assert torch.allclose(output[name].sum(dim=-1), torch.ones(2))


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
