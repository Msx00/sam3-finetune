from types import SimpleNamespace

import pytest
import torch

from models.inference_utils import normalized_xyxy_prompts


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
