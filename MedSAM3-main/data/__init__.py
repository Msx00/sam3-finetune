"""Dataset wrappers specific to the hierarchical training entry."""

from .patient_dataset import PatientDataset, PatientSliceRecord

__all__ = ["PatientDataset", "PatientSliceRecord"]
