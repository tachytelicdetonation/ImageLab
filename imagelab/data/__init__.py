"""Optional data batteries — none of the framework requires them, but trustworthy
comparisons do: the manifest layout, the per-item-stable train/val split, and the
single-sample overfit clamp are why metrics stay comparable across a whole ledger."""

from imagelab.data.dataset import (  # noqa: F401
    ManifestImageDataset, OverfitDataset, record_split,
)
