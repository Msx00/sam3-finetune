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


def test_epoch_statistics_mid_epoch_state_round_trip():
    original = EpochStatistics()
    original.update_losses({"total_loss": torch.tensor(2.0)}, weight=3)
    original.teacher_used = 2
    original.teacher_total = 3
    original.slice_records.append({
        "patient_id": 7,
        "modality": "MR",
        "area_label": 0,
        "boundary_label": 1,
        "base_dice": 0.5,
        "base_iou": 1 / 3,
        "final_dice": 0.75,
        "final_iou": 0.6,
    })

    restored = EpochStatistics()
    restored.load_state_dict(original.state_dict())

    assert restored.loss_sums == original.loss_sums
    assert restored.loss_weight == 3
    assert restored.teacher_used == 2
    assert restored.teacher_total == 3
    assert restored.slice_records == original.slice_records
    assert restored.report(0.0) == original.report(0.0)


def test_shared_fallback_is_not_counted_as_first_expert():
    statistics = EpochStatistics()
    routes = {
        "modality_logits": torch.tensor([[5.0, 0.0], [0.0, 5.0]]),
        "area_logits": torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]]),
        "boundary_logits": torch.tensor([[5.0, 0.0, 0.0], [0.0, 0.0, 5.0]]),
        "modality": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "area": torch.zeros(2, 3),
        "boundary": torch.zeros(2, 3),
        "area_joint": torch.zeros(2, 2, 3),
        "boundary_joint": torch.zeros(2, 2, 3),
        "area_routing_confidence": torch.tensor([0.2, 0.3]),
        "boundary_routing_confidence": torch.tensor([0.1, 0.4]),
        "area_routing_policy": torch.zeros(2, dtype=torch.long),
        "boundary_routing_policy": torch.zeros(2, dtype=torch.long),
    }
    targets = {
        "modality": torch.tensor([0, 1]),
        "area": torch.tensor([0, 1]),
        "boundary": torch.tensor([0, 2]),
    }

    statistics.update_router(routes, targets)
    report = statistics.report()

    assert sum(report["experts"].values()) == 0
    assert report["router"]["area_shared_count"] == 2
    assert report["router"]["area_shared_ratio"] == 1.0
    assert abs(report["router"]["area_routing_confidence"] - 0.25) < 1e-7
