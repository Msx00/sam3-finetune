import torch
from torch import nn

from models.moe_losses import HierarchicalMoELoss, dice_bce_with_logits
from models.svanet_roi_adapter import SvANetROIAdapter


class TinySvANet(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 2, kernel_size=1, bias=False)
        nn.init.constant_(self.projection.weight[0], -0.25)
        nn.init.constant_(self.projection.weight[1], 0.25)

    def forward(self, images):
        return self.projection(images)


def test_native_sam3_loss_is_preserved_in_hierarchical_objective():
    native_loss = torch.tensor(3.0, requires_grad=True)
    final_logits = torch.zeros(1, 4, 4, requires_grad=True)
    gt_masks = torch.zeros(1, 4, 4)
    routes = {
        "area_logits": torch.zeros(1, 3),
        "area_ratio_pred": torch.zeros(1),
    }
    zero = torch.tensor(0.0)
    routing_losses = {
        "modality_loss": zero,
        "area_loss": zero,
        "boundary_router_loss": zero,
        "load_balance_loss": zero,
    }
    objective = HierarchicalMoELoss(
        {
            "native_sam3_loss": 2.0,
            "sam3_loss": 0.0,
            "boundary_seg_loss": 0.0,
        }
    )
    total, components = objective(
        final_logits=final_logits,
        gt_masks=gt_masks,
        aux_logits=None,
        routes=routes,
        routing_losses=routing_losses,
        area_ratio_gt=torch.zeros(1),
        native_sam3_loss=native_loss,
    )
    assert torch.allclose(total, torch.tensor(6.0))
    assert components["native_sam3_loss"] is native_loss
    total.backward()
    assert torch.allclose(native_loss.grad, torch.tensor(2.0))


def test_soft_gate_fusion_backpropagates_to_sam_router_and_svanet():
    svanet = TinySvANet()
    adapter = SvANetROIAdapter(
        svanet,
        input_size=(4, 4),
        roi_expand_ratio=0.0,
        min_roi_size=2,
        paste_mode="soft_gate",
        outside_roi="sam3",
        train_trigger="gt_or_pred",
    )
    adapter.train()
    images = torch.ones(1, 1, 8, 8)
    sam3_logits = torch.full((1, 8, 8), -2.0, requires_grad=True)
    area_logits = torch.tensor([[2.0, 0.0, 0.0]], requires_grad=True)
    gt_masks = torch.zeros(1, 8, 8)
    gt_masks[:, 3:5, 3:5] = 1.0

    output = adapter(
        images=images,
        sam3_logits=sam3_logits,
        area_logits=area_logits,
        area_labels=torch.tensor([0]),
        gt_masks=gt_masks,
        use_gt_roi=True,
    )
    gate = output["fusion_gates"][0]
    assert 0.5 < float(gate.detach()) < 1.0
    assert not torch.allclose(output["final_logits"], sam3_logits)

    final_loss = dice_bce_with_logits(output["final_logits"], gt_masks)
    (final_loss + output["refine_loss"]).backward()
    assert sam3_logits.grad is not None
    assert float(sam3_logits.grad.abs().sum()) > 0
    assert area_logits.grad is not None
    assert float(area_logits.grad.abs().sum()) > 0
    assert svanet.projection.weight.grad is not None
    assert float(svanet.projection.weight.grad.abs().sum()) > 0


def test_direct_refine_loss_is_reserved_for_small_targets():
    adapter = SvANetROIAdapter(
        TinySvANet(),
        input_size=(4, 4),
        roi_expand_ratio=0.0,
        min_roi_size=2,
        paste_mode="soft_gate",
        outside_roi="sam3",
        train_trigger="all",
    )
    adapter.train()
    gt_masks = torch.zeros(1, 8, 8)
    gt_masks[:, 2:6, 2:6] = 1.0
    output = adapter(
        images=torch.ones(1, 1, 8, 8),
        sam3_logits=torch.zeros(1, 8, 8),
        area_logits=torch.tensor([[0.0, 3.0, 0.0]]),
        area_labels=torch.tensor([1]),
        gt_masks=gt_masks,
        use_gt_roi=True,
    )
    assert torch.allclose(output["refine_loss"], torch.tensor(0.0))
