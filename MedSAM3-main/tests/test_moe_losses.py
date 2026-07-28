import torch

from models.moe_losses import HierarchicalMoELoss


def test_native_sam3_core_replaces_custom_sam3_loss_and_keeps_gradient():
    loss_module = HierarchicalMoELoss({"sam3_loss": 2.0})
    native_core = torch.tensor(3.25, requires_grad=True)
    final_logits = torch.randn(2, 8, 8, requires_grad=True)
    gt_masks = torch.randint(0, 2, (2, 8, 8), dtype=torch.float32)
    routes = {
        "area_logits": torch.zeros(2, 3),
        "area_ratio_pred": torch.zeros(2),
    }
    zero = torch.tensor(0.0)
    routing_losses = {
        "modality_loss": zero,
        "area_loss": zero,
        "boundary_router_loss": zero,
        "load_balance_loss": zero,
    }

    total, components = loss_module(
        sam3_core_loss=native_core,
        final_logits=final_logits,
        gt_masks=gt_masks,
        aux_logits=None,
        routes=routes,
        routing_losses=routing_losses,
        area_ratio_gt=torch.zeros(2),
    )

    assert components["sam3_loss"] is native_core
    assert torch.allclose(total, native_core * 2.0)
    total.backward()
    assert torch.allclose(native_core.grad, torch.tensor(2.0))
    assert final_logits.grad is not None
    assert torch.count_nonzero(final_logits.grad) == 0


def test_native_sam3_core_must_be_scalar_tensor():
    loss_module = HierarchicalMoELoss({})
    routes = {
        "area_logits": torch.zeros(1, 3),
        "area_ratio_pred": torch.zeros(1),
    }
    routing_losses = {
        name: torch.tensor(0.0)
        for name in (
            "modality_loss",
            "area_loss",
            "boundary_router_loss",
            "load_balance_loss",
        )
    }
    common = {
        "final_logits": torch.zeros(0, 4, 4),
        "gt_masks": torch.zeros(0, 4, 4),
        "aux_logits": None,
        "routes": routes,
        "routing_losses": routing_losses,
        "area_ratio_gt": torch.zeros(1),
    }

    try:
        loss_module(sam3_core_loss=1.0, **common)
    except TypeError:
        pass
    else:
        raise AssertionError("A non-tensor native core loss must be rejected")

    try:
        loss_module(sam3_core_loss=torch.zeros(2), **common)
    except ValueError:
        pass
    else:
        raise AssertionError("A non-scalar native core loss must be rejected")
