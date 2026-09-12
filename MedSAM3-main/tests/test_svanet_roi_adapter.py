"""Focused tests for safe SvANet ROI proposal and fusion."""

import torch
from torch import nn

from models.svanet_roi_adapter import SvANetROIAdapter


class ConstantSvANet(nn.Module):
    def __init__(self, foreground_logit: float = 4.0) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.foreground_logit = float(foreground_logit)
        self.call_count = 0

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        batch, _, height, width = images.shape
        background = self.anchor.expand(batch, 1, height, width)
        foreground = background + self.foreground_logit
        return torch.cat((background, foreground), dim=1)


def _small_area_logits(batch: int = 1) -> torch.Tensor:
    return torch.tensor([[5.0, -2.0, -3.0]]).expand(batch, -1).clone()


def _adapter(model: nn.Module, **kwargs) -> SvANetROIAdapter:
    adapter = SvANetROIAdapter(
        model,
        input_size=(4, 4),
        roi_expand_ratio=0.0,
        min_roi_size=1,
        **kwargs,
    )
    adapter.eval()
    return adapter


def test_safe_default_blends_roi_and_preserves_sam3_outside() -> None:
    model = ConstantSvANet(foreground_logit=4.0)
    adapter = _adapter(model)
    images = torch.randn(1, 3, 8, 8)
    sam3 = torch.full((1, 8, 8), -4.0)
    sam3[:, 2:6, 3:7] = 6.0

    output = adapter(images, sam3, _small_area_logits())

    assert output["roi_sources"] == ["sam3_mask"]
    assert output["roi_boxes"] == [(3, 2, 7, 6)]
    # Default fusion is 0.5 * SAM3 + 0.5 * SvANet.
    assert torch.allclose(output["final_logits"][:, 2:6, 3:7], torch.full((1, 4, 4), 5.0))
    outside = torch.ones_like(sam3, dtype=torch.bool)
    outside[:, 2:6, 3:7] = False
    assert torch.equal(output["final_logits"][outside], sam3[outside])


def test_empty_sam3_uses_image_locator_before_box_or_skip() -> None:
    model = ConstantSvANet()
    adapter = _adapter(model)
    images = torch.randn(1, 3, 8, 8)
    sam3 = torch.full((1, 8, 8), -10.0)
    locator = torch.full((1, 1, 4, 4), -10.0)
    locator[:, :, 1:3, 1:3] = 10.0

    output = adapter(
        images,
        sam3,
        _small_area_logits(),
        locator_logits=locator,
    )

    assert output["roi_sources"] == ["locator_fallback"]
    assert output["batch_stats"]["locator_fallback_count"] == 1
    assert output["skipped_indices"] == []
    assert model.call_count == 1


def test_no_reliable_roi_skips_refinement_instead_of_using_full_image() -> None:
    model = ConstantSvANet()
    adapter = _adapter(
        model,
        empty_mask_fallback="locator_then_box_then_skip",
        min_component_pixels=4,
    )
    images = torch.randn(1, 3, 8, 8)
    sam3 = torch.full((1, 8, 8), -10.0)
    sam3[:, 2, 2] = 10.0  # non-empty, but intentionally unreliable

    output = adapter(images, sam3, _small_area_logits())

    assert output["trigger_mask"].tolist() == [False]
    assert output["confidence_qualified_trigger_mask"].tolist() == [True]
    assert output["refined_indices"] == []
    assert output["skipped_indices"] == [0]
    assert output["batch_stats"]["no_reliable_roi_skip_count"] == 1
    assert output["batch_stats"]["full_image_fallback_count"] == 0
    assert torch.equal(output["final_logits"], sam3)
    assert model.call_count == 0


def test_low_area_router_confidence_can_skip_refinement() -> None:
    model = ConstantSvANet()
    adapter = _adapter(model, min_area_confidence=0.6)
    images = torch.randn(1, 3, 8, 8)
    sam3 = torch.full((1, 8, 8), 5.0)
    uncertain_small = torch.tensor([[0.1, 0.0, -0.1]])

    output = adapter(images, sam3, uncertain_small)

    assert output["requested_trigger_mask"].tolist() == [True]
    assert output["trigger_mask"].tolist() == [False]
    assert output["low_area_confidence_mask"].tolist() == [True]
    assert output["skipped_indices"] == [0]
    assert model.call_count == 0


def test_residual_fusion_and_legacy_destructive_mode_remain_available() -> None:
    images = torch.randn(1, 3, 8, 8)
    sam3 = torch.full((1, 8, 8), -4.0)
    sam3[:, 2:6, 2:6] = 2.0

    residual = _adapter(
        ConstantSvANet(foreground_logit=4.0),
        paste_mode="residual",
        residual_scale=0.25,
    )
    residual_output = residual(images, sam3, _small_area_logits())
    assert torch.allclose(
        residual_output["final_logits"][:, 2:6, 2:6],
        torch.full((1, 4, 4), 3.0),
    )

    legacy = _adapter(
        ConstantSvANet(foreground_logit=4.0),
        empty_mask_fallback="box_then_full_image",
        paste_mode="replace_roi",
        outside_roi="zero",
    )
    empty_sam3 = torch.full((1, 8, 8), -4.0)
    legacy_output = legacy(
        images,
        empty_sam3,
        _small_area_logits(),
        box_prompts=[torch.tensor([[2.0, 2.0, 6.0, 6.0]])],
    )
    outside = torch.ones_like(empty_sam3, dtype=torch.bool)
    outside[:, 2:6, 2:6] = False
    assert legacy_output["roi_sources"] == ["box_fallback"]
    assert torch.all(legacy_output["final_logits"][outside] == -20.0)


def test_four_dimensional_gt_mask_is_normalized_before_roi_selection() -> None:
    model = ConstantSvANet()
    adapter = _adapter(model)
    adapter.train()
    images = torch.randn(1, 3, 8, 8)
    sam3 = torch.full((1, 8, 8), -10.0)
    gt = torch.zeros(1, 1, 8, 8)
    gt[:, :, 2:6, 3:7] = 1.0

    output = adapter(
        images,
        sam3,
        _small_area_logits(),
        area_labels=torch.tensor([0]),
        teacher_area_mask=torch.tensor([True]),
        gt_masks=gt,
        use_gt_roi=True,
    )

    assert output["roi_boxes"] == [(3, 2, 7, 6)]
    assert torch.isfinite(output["refine_loss"])


def test_adapter_rejects_misaligned_router_and_teacher_shapes() -> None:
    adapter = _adapter(ConstantSvANet())
    images = torch.randn(2, 3, 8, 8)
    sam3 = torch.randn(2, 8, 8)

    for kwargs in (
        {"area_logits": torch.randn(2, 2)},
        {
            "area_logits": torch.randn(2, 3),
            "area_labels": torch.tensor([0]),
        },
        {
            "area_logits": torch.randn(2, 3),
            "gt_masks": torch.zeros(1, 8, 8),
        },
    ):
        try:
            adapter(images=images, sam3_logits=sam3, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid shapes must fail closed: {kwargs}")


def test_roi_chunking_matches_single_forward_for_values_and_gradients() -> None:
    def run(roi_chunk_size: int, max_roi_per_step: int = 0):
        model = ConstantSvANet(foreground_logit=2.0)
        adapter = _adapter(
            model, roi_chunk_size=roi_chunk_size, max_roi_per_step=max_roi_per_step
        )
        images = torch.randn(4, 3, 8, 8, requires_grad=True)
        sam3 = torch.full((4, 8, 8), -4.0)
        sam3[:, 2:6, 3:7] = 6.0
        gt = torch.zeros(4, 8, 8)
        gt[:, 2:6, 3:7] = 1.0
        output = adapter(
            images,
            sam3,
            _small_area_logits(4),
            area_labels=torch.zeros(4, dtype=torch.long),
            gt_masks=gt,
            teacher_area_mask=torch.ones(4, dtype=torch.bool),
            use_gt_roi=True,
        )
        assert model.call_count == (4 if roi_chunk_size else 1)
        loss = output["refine_loss"]
        loss.backward()
        return output, loss.detach(), model.anchor.grad.detach().clone()

    reference_output, reference_loss, reference_grad = run(0)
    chunked_output, chunked_loss, chunked_grad = run(1)

    assert torch.equal(chunked_output["final_logits"], reference_output["final_logits"])
    assert torch.equal(chunked_loss, reference_loss)
    assert torch.equal(chunked_grad, reference_grad)


def test_max_roi_per_step_caps_and_reports_skipped_crops() -> None:
    model = ConstantSvANet()
    adapter = _adapter(model, max_roi_per_step=1)
    images = torch.randn(3, 3, 8, 8)
    sam3 = torch.full((3, 8, 8), -10.0)
    sam3[:, 2:6, 3:7] = 6.0

    output = adapter(images, sam3, _small_area_logits(3))

    assert model.call_count == 1
    assert output["trigger_mask"].sum().item() == 1
    assert output["batch_stats"]["trigger_count"] == 1
    assert len(output["skipped_indices"]) == 2
