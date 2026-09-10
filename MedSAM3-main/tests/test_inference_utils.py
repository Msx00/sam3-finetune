from types import SimpleNamespace

import pytest
import torch

from models.inference_utils import normalized_xyxy_prompts, route_predictions


@pytest.mark.parametrize("batch_first", [True, False])
def test_normalized_xyxy_prompts_accepts_both_sam3_box_layouts(batch_first):
    boxes = torch.tensor(
        [
            [
                [0.5, 0.5, 0.4, 0.2],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.3, 0.4, 0.2, 0.4],
                [0.7, 0.8, 0.2, 0.2],
                [0.0, 0.0, 0.0, 0.0],
            ],
        ]
    )
    mask = torch.tensor([[False, True, True], [False, False, True]])
    find_input = SimpleNamespace(
        input_boxes=boxes if batch_first else boxes.transpose(0, 1),
        input_boxes_mask=mask,
    )

    prompts = normalized_xyxy_prompts(find_input)

    assert len(prompts) == 2
    assert torch.allclose(prompts[0], torch.tensor([[0.3, 0.4, 0.7, 0.6]]))
    assert torch.allclose(
        prompts[1],
        torch.tensor([[0.2, 0.2, 0.4, 0.6], [0.6, 0.7, 0.8, 0.9]]),
    )


def test_normalized_xyxy_prompts_rejects_incompatible_shapes():
    find_input = SimpleNamespace(
        input_boxes=torch.zeros(3, 4, 4),
        input_boxes_mask=torch.zeros(2, 1, dtype=torch.bool),
    )

    with pytest.raises(ValueError, match="incompatible shapes"):
        normalized_xyxy_prompts(find_input)


def test_route_predictions_uses_conditional_branch_and_reports_fallback():
    routes = {
        "modality_logits": torch.tensor([[0.0, 5.0]]),
        "modality_soft": torch.tensor([[0.1, 0.9]]),
        # Marginal prediction intentionally disagrees with the selected US
        # conditional branch; reporting must follow the hierarchy.
        "area_soft": torch.tensor([[0.8, 0.1, 0.1]]),
        "area_soft_all": torch.tensor([[[0.9, 0.05, 0.05], [0.1, 0.2, 0.7]]]),
        "boundary_soft": torch.tensor([[0.7, 0.2, 0.1]]),
        "boundary_soft_all": torch.tensor([[[0.8, 0.1, 0.1], [0.2, 0.6, 0.2]]]),
        "area_routing_confidence": torch.tensor([0.42]),
        "boundary_routing_confidence": torch.tensor([0.31]),
        "area_routing_policy": torch.tensor([1]),
        "boundary_routing_policy": torch.tensor([0]),
        "area_joint": torch.tensor([[[0.0, 0.0, 0.0], [0.2, 0.0, 0.8]]]),
        "boundary_joint": torch.zeros(1, 2, 3),
    }

    prediction = route_predictions(routes, 0)

    assert prediction["modality"] == "US"
    assert prediction["area"] == "large"
    assert prediction["boundary"] == "fuzzy"
    assert prediction["area_policy"] == "topk"
    assert prediction["boundary_policy"] == "shared"
    assert prediction["boundary_expert"] is None
    assert prediction["boundary_active_experts"] == []
    assert len(prediction["area_active_experts"]) == 2
    assert abs(prediction["area_confidence"] - 0.42) < 1e-6
