from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split

from duckietown_seg.data.transforms import SegmentationResize, image_to_tensor, mask_to_tensor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MASK_EXTENSIONS = {".png", ".bmp", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class SamplePair:
    image_path: Path
    mask_path: Path

    @property
    def sample_id(self) -> str:
        return self.image_path.stem


class DuckietownLaneSegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset for Duckietown lane segmentation with explicit pairing validation."""

    def __init__(
        self,
        dataset_root: str | Path,
        image_dirname: str,
        mask_dirname: str,
        image_size: Sequence[int],
        num_classes: int = 4,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.image_dir = self.dataset_root / image_dirname
        self.mask_dir = self.dataset_root / mask_dirname
        self.num_classes = num_classes
        self.resize = SegmentationResize(image_size)
        self.samples = self._build_pairs()

    def _iter_files(self, directory: Path, allowed_extensions: Iterable[str]) -> dict[str, Path]:
        if not directory.exists():
            raise FileNotFoundError(f"Missing directory: {directory}")
        files = [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in allowed_extensions]
        if not files:
            raise FileNotFoundError(f"No files found in {directory}")
        mapping: dict[str, Path] = {}
        for path in sorted(files):
            stem = path.stem
            if stem in mapping:
                raise ValueError(f"Duplicate sample id '{stem}' found in {directory}")
            mapping[stem] = path
        return mapping

    def _build_pairs(self) -> list[SamplePair]:
        images = self._iter_files(self.image_dir, IMAGE_EXTENSIONS)
        masks = self._iter_files(self.mask_dir, MASK_EXTENSIONS)

        image_ids = set(images)
        mask_ids = set(masks)
        if image_ids != mask_ids:
            missing_masks = sorted(image_ids - mask_ids)
            missing_images = sorted(mask_ids - image_ids)
            raise ValueError(
                "Image-mask pairing mismatch detected. "
                f"Missing masks for ids: {missing_masks[:10]}. "
                f"Missing images for ids: {missing_images[:10]}."
            )

        return [SamplePair(image_path=images[sample_id], mask_path=masks[sample_id]) for sample_id in sorted(image_ids)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        mask = Image.open(sample.mask_path).convert("L")
        image, mask = self.resize(image, mask)
        image_tensor = image_to_tensor(image)
        mask_tensor = mask_to_tensor(mask)
        self._validate_mask(mask_tensor, sample.sample_id)
        return image_tensor, mask_tensor

    def _validate_mask(self, mask: torch.Tensor, sample_id: str) -> None:
        min_value = int(mask.min().item())
        max_value = int(mask.max().item())
        if min_value < 0 or max_value >= self.num_classes:
            raise ValueError(
                f"Mask '{sample_id}' contains invalid class ids outside [0, {self.num_classes - 1}]: "
                f"min={min_value}, max={max_value}"
            )


def build_train_val_datasets(
    dataset: DuckietownLaneSegmentationDataset,
    val_fraction: float,
    split_seed: int,
) -> tuple[Subset[DuckietownLaneSegmentationDataset], Subset[DuckietownLaneSegmentationDataset]]:
    """Create deterministic train/val subsets."""

    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}.")
    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Validation split is too large for the dataset size.")
    generator = torch.Generator().manual_seed(split_seed)
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)
    return train_subset, val_subset

