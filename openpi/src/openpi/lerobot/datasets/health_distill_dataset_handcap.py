#!/usr/bin/env python
"""Random multi-health Handcap dataset for wrist-force to tactile distillation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import torch

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset_handcap import LeRobotDatasetHandcap


class MultiHealthLeRobotDatasetHandcap(torch.utils.data.Dataset):
    """A flattened collection of Handcap datasets with a health id per item."""

    def __init__(
        self,
        repo_ids: Sequence[str],
        roots: Sequence[str | Path],
        *,
        health_labels: Sequence[str] | None = None,
        episodes: dict | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerances_s: dict | None = None,
        download_videos: bool = True,
        video_backend: str | None = None,
    ):
        super().__init__()
        if len(repo_ids) != len(roots):
            raise ValueError("repo_ids and roots must have the same length.")
        if len(repo_ids) < 2:
            raise ValueError("At least two health datasets are required.")
        if health_labels is not None and len(health_labels) != len(repo_ids):
            raise ValueError("health_labels must match repo_ids length.")

        self.repo_ids = tuple(repo_ids)
        self.roots = tuple(Path(root) for root in roots)
        self.health_labels = tuple(health_labels or [str(i) for i in range(len(repo_ids))])
        self.num_healths = len(self.repo_ids)
        tolerances_s = tolerances_s if tolerances_s else dict.fromkeys(self.repo_ids, 0.0001)

        self._datasets = [
            LeRobotDatasetHandcap(
                repo_id,
                root=root,
                episodes=episodes[repo_id] if episodes else None,
                image_transforms=image_transforms,
                delta_timestamps=delta_timestamps,
                tolerance_s=tolerances_s[repo_id],
                download_videos=download_videos,
                video_backend=video_backend,
            )
            for repo_id, root in zip(self.repo_ids, self.roots, strict=True)
        ]
        self.health_lengths = tuple(len(dataset) for dataset in self._datasets)
        self.health_offsets = []
        total = 0
        for length in self.health_lengths:
            self.health_offsets.append(total)
            total += length
        self.num_frames = total
        self.stats = aggregate_stats([dataset.meta.stats for dataset in self._datasets])

    @property
    def fps(self) -> int:
        return self._datasets[0].fps

    @property
    def meta(self):
        return self._datasets[0].meta

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> dict:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        health_id = 0
        local_idx = int(idx)
        for candidate_health_id, offset in enumerate(self.health_offsets):
            length = self.health_lengths[candidate_health_id]
            if offset <= idx < offset + length:
                health_id = candidate_health_id
                local_idx = int(idx - offset)
                break

        item = self._datasets[health_id][local_idx]
        item["health_id"] = torch.tensor(health_id, dtype=torch.int64)
        item["health_label"] = self.health_labels[health_id]
        item["health_source_index"] = torch.tensor(local_idx, dtype=torch.int64)
        return item

