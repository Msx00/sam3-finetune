import math
import pytest

from test_moe_sam3_prompts import (
    patient_macro_summary_rows,
    summarize,
    summarize_patients,
)


def _row(modality, patient_id, slice_id, dice, hd95):
    return {
        "prompt_mode": "text",
        "modality": modality,
        "patient_id": patient_id,
        "slice_id": slice_id,
        "gt_available": True,
        "gt_foreground": True,
        "missing_prediction": False,
        "is_small_target": False,
        "dice": dice,
        "hd95": hd95,
        "iou": dice,
        "precision": dice,
        "recall": dice,
    }


def test_patient_macro_gives_each_patient_equal_weight_and_splits_modalities():
    rows = [
        _row("MR", 1, "slice_1", 1.0, 10.0),
        _row("MR", 1, "slice_2", 0.0, 20.0),
        _row("MR", 2, "slice_1", 1.0, math.nan),
        _row("US", 3, "slice_1", 0.2, 30.0),
    ]

    patient_rows = summarize_patients(rows)
    summary = summarize(rows, patient_rows)
    text = summary["by_prompt_mode"]["text"]

    # Slice macro is (1 + 0 + 1 + .2) / 4, whereas patient macro is
    # ((1 + 0) / 2 + 1 + .2) / 3.
    assert text["all_samples"]["dice"] == pytest.approx(0.55)
    assert text["patient_macro"]["dice"] == pytest.approx(1.7 / 3.0)
    assert text["by_modality"]["MR"]["patient_macro"]["dice"] == pytest.approx(0.75)
    assert text["by_modality"]["US"]["patient_macro"]["dice"] == pytest.approx(0.2)

    # MR patient 2 has no finite HD95, so it is counted as a patient but is
    # excluded from the HD95 mean and num_valid_hd95 records that fact.
    assert text["patient_macro"]["hd95"] == pytest.approx(22.5)
    assert text["patient_macro"]["num_patients_evaluated"] == 3
    assert text["patient_macro"]["num_valid_hd95"] == 2

    flat = patient_macro_summary_rows(summary)
    all_rows = [row for row in flat if row["target_scope"] == "all"]
    assert {row["modality"] for row in all_rows} == {"overall", "MR", "US"}
