import numpy as np
from PIL import Image

from data.patient_dataset import PatientDataset, PatientSliceRecord


def test_apply_thresholds_updates_existing_selected_records(tmp_path):
    image_path = tmp_path / "slice_0.png"
    mask_path = tmp_path / "mask_0.png"
    image = np.zeros((16, 16), dtype=np.uint8)
    image[6:10, 6:10] = 200
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[6:10, 6:10] = 255
    Image.fromarray(image).save(image_path)
    Image.fromarray(mask).save(mask_path)

    dataset = PatientDataset.__new__(PatientDataset)
    dataset.modality = "MR"
    dataset.records = [
        PatientSliceRecord(
            image_path=str(image_path),
            relative_file_name="1/slice_0.png",
            patient_id=1,
            slice_id="slice_0",
            slice_index=0,
            modality="MR",
            split="train",
            base_dataset_index=0,
            image_id=1,
            mask_path=str(mask_path),
            area_ratio=16 / 256,
        )
    ]
    dataset.apply_thresholds(
        {"small_max": 0.1, "medium_max": 0.2},
        {
            "boundary_band_width": 3,
            "mr": {"contrast_low": 0.0, "complexity_high": 1e6},
            "us": {"contrast_low": 0.0, "complexity_high": 1e6},
        },
    )
    record = dataset.records[0]
    assert record.area_label == 0
    assert record.boundary_label == 0
    assert record.boundary_contrast >= 0
    assert record.boundary_complexity > 0
