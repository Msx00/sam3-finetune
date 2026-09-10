"""Unit tests for deployment-aligned prompt scheduling."""
import torch

from data.patient_dataset import PatientDataset, PatientSliceRecord


def _dataset(*, training=True, mode="image_only") -> PatientDataset:
    dataset = PatientDataset.__new__(PatientDataset)
    dataset.training = training
    dataset.current_epoch = 0
    dataset.prompt_seed = 17
    dataset.modality = "MR"
    dataset.prompt_curriculum_enabled = True
    dataset.fixed_prompt_text = "prostate"
    dataset.prompt_decay_epochs = 10
    dataset.prompt_evaluation_mode = "image_only"
    dataset.prompt_start_probabilities = {
        "image_only": float(mode == "image_only"),
        "text": float(mode == "text"),
        "coarse_box": float(mode == "coarse_box"),
        "accurate_box": float(mode == "accurate_box"),
    }
    dataset.prompt_end_probabilities = dict(dataset.prompt_start_probabilities)
    dataset.coarse_box_expand_min = 0.2
    dataset.coarse_box_expand_max = 0.4
    dataset.coarse_box_jitter_std = 0.1
    dataset.coarse_box_jitter_max = 5.0
    dataset.box_noise_std = 0.1
    dataset.box_noise_max = 5.0
    dataset.base_dataset = type(
        "Base", (), {"images": {1: {"width": 100, "height": 80}}}
    )()
    return dataset


def _record() -> PatientSliceRecord:
    return PatientSliceRecord(
        image_path="unused.png",
        relative_file_name="1/slice_1.png",
        patient_id=1,
        slice_id="slice_1",
        slice_index=1,
        modality="MR",
        split="train",
        base_dataset_index=0,
        image_id=1,
        box_prompt=[[20.0, 15.0, 60.0, 55.0]],
        text_prompt="prostate gland",
    )


def test_image_only_mode_has_no_sample_specific_box_or_text() -> None:
    dataset = _dataset(training=False)
    mode, text, boxes = dataset._resolve_prompt(3, _record())
    assert mode == "image_only"
    assert text == "prostate"
    assert boxes.shape == (0, 4)


def test_coarse_box_is_valid_larger_and_deterministic() -> None:
    dataset = _dataset(mode="coarse_box")
    first = dataset._resolve_prompt(7, _record())
    second = dataset._resolve_prompt(7, _record())
    assert first[0] == "coarse_box"
    assert torch.equal(first[2], second[2])
    box = first[2][0]
    assert 0 <= box[0] < box[2] <= 100
    assert 0 <= box[1] < box[3] <= 80
    assert (box[2] - box[0]) > 40
    assert (box[3] - box[1]) > 40


def test_prompt_probability_schedule_reaches_image_only_endpoint() -> None:
    dataset = _dataset()
    dataset.prompt_start_probabilities = {
        "image_only": 0.2,
        "text": 0.2,
        "coarse_box": 0.4,
        "accurate_box": 0.2,
    }
    dataset.prompt_end_probabilities = {
        "image_only": 0.7,
        "text": 0.1,
        "coarse_box": 0.2,
        "accurate_box": 0.0,
    }
    dataset.set_epoch(10)
    assert dataset._prompt_probabilities() == dataset.prompt_end_probabilities
