import torch

from data.patient_dataset import add_box_noise_xyxy


def test_box_noise_is_reproducible_clipped_and_valid():
    boxes = torch.tensor([[10.0, 20.0, 50.0, 80.0]])
    generator = torch.Generator().manual_seed(7)

    noisy = add_box_noise_xyxy(
        boxes,
        width=100,
        height=90,
        box_noise_std=10.0,
        box_noise_max=3.0,
        generator=generator,
    )

    assert torch.all((noisy - boxes).abs() <= 3.0 + 1e-6)
    assert 0 <= noisy[0, 0] < noisy[0, 2] <= 100
    assert 0 <= noisy[0, 1] < noisy[0, 3] <= 90


def test_zero_box_noise_returns_original_values():
    boxes = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert torch.equal(add_box_noise_xyxy(boxes, 10, 10, 0.0, 2.0), boxes)
