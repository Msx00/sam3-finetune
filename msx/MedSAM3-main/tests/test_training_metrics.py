import torch

from models.training_metrics import EpochStatistics


def test_router_expert_and_patient_macro_metrics():
    statistics = EpochStatistics()
    routes = {
        "modality_logits": torch.tensor([[5.0, 0.0], [0.0, 5.0]]),
        "area_logits": torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]]),
        "boundary_logits": torch.tensor([[5.0, 0.0, 0.0], [0.0, 0.0, 5.0]]),
        "modality": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "area": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "boundary": torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        "teacher_modality_mask": torch.tensor([True, False]),
        "teacher_area_mask": torch.tensor([True, False]),
        "teacher_boundary_mask": torch.tensor([True, False]),
    }
    targets = {
        "modality": torch.tensor([0, 1]),
        "area": torch.tensor([0, 1]),
        "boundary": torch.tensor([0, 2]),
    }
    statistics.update_router(routes, targets)
    statistics.update_losses({"total_loss": torch.tensor(2.0)})

    gt = torch.tensor([[[1.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]])
    perfect = torch.tensor([[[20.0, -20.0], [-20.0, -20.0]], [[20.0, -20.0], [-20.0, -20.0]]])
    metadata = [
        {"patient_id": 1, "modality": "MR", "area_label": 0, "boundary_label": 0},
        {"patient_id": 2, "modality": "US", "area_label": 1, "boundary_label": 2},
    ]
    statistics.update_segmentation(perfect, perfect, gt, metadata)
    report = statistics.report(configured_teacher_ratio=0.5)
    assert report["router"]["modality_accuracy"] == 1.0
    assert report["router"]["actual_teacher_forcing_ratio"] == 0.5
    assert report["experts"]["MR_area_small"] == 1
    assert report["experts"]["US_boundary_complex"] == 1
    assert report["segmentation"]["patient_macro_dice"] == 1.0
    assert report["segmentation"]["small_dice"] == 1.0


def test_patient_macro_does_not_weight_patients_by_slice_count():
    statistics = EpochStatistics()
    gt = torch.zeros(3, 2, 2)
    gt[:, 0, 0] = 1
    good = torch.full((2, 2), -20.0)
    good[0, 0] = 20.0
    bad = torch.full((2, 2), -20.0)
    logits = torch.stack((good, bad, good))
    metadata = [
        {"patient_id": 1, "modality": "MR", "area_label": 0, "boundary_label": 0},
        {"patient_id": 1, "modality": "MR", "area_label": 0, "boundary_label": 0},
        {"patient_id": 2, "modality": "MR", "area_label": 0, "boundary_label": 0},
    ]
    statistics.update_segmentation(logits, logits, gt, metadata)
    report = statistics.segmentation_report()
    assert abs(report["slice_dice"] - 2 / 3) < 1e-8
    assert abs(report["patient_macro_dice"] - 0.75) < 1e-8
