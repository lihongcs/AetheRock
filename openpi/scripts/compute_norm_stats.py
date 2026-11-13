# Modified from OpenPI by the AetheRock project for ForceVT normalization data.

"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import dataclasses

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def _maybe_override_health_paths(
    config: _config.TrainConfig,
    health_data_roots: tuple[str, ...] | None,
    health_repo_ids: tuple[str, ...] | None,
    health_sampling_weights: tuple[float, ...] | None,
    data_repo_id: str | None,
    health_labels: tuple[str, ...] | None,
) -> _config.TrainConfig:
    if (
        any(value is not None for value in (health_data_roots, health_repo_ids, health_sampling_weights, health_labels))
    ) and not hasattr(config.data, "health_data_roots"):
        raise ValueError("--health-data-roots can only be used with multi-health Handcap data configs.")

    data = config.data
    if health_data_roots is not None:
        data = dataclasses.replace(data, health_data_roots=health_data_roots)
    if health_repo_ids is not None:
        data = dataclasses.replace(data, health_repo_ids=health_repo_ids)
    if health_labels is not None:
        data = dataclasses.replace(data, health_labels=health_labels)
    if health_sampling_weights is not None:
        data = dataclasses.replace(data, health_sampling_weights=health_sampling_weights)
    if data_repo_id is not None:
        data = dataclasses.replace(data, repo_id=data_repo_id)
    return dataclasses.replace(config, data=data)


def main(
    config_name: str,
    max_frames: int | None = None,
    health_data_roots: tuple[str, ...] | None = None,
    health_repo_ids: tuple[str, ...] | None = None,
    health_sampling_weights: tuple[float, ...] | None = None,
    data_repo_id: str | None = None,
    health_labels: tuple[str, ...] | None = None,
):
    config = _config.get_config(config_name)
    config = _maybe_override_health_paths(
        config, health_data_roots, health_repo_ids, health_sampling_weights, data_repo_id, health_labels
    )
    data_config = config.data.create(config.assets_dirs, config.model)

    output_path = config.assets_dirs / data_config.repo_id
    if (output_path / "norm_stats.json").exists():
        print(f"Norm stats already exist at {output_path / 'norm_stats.json'}. Skipping computation.")
        return

    data_loader, num_batches = create_torch_dataloader(
        data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
