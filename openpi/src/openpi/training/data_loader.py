# Modified from OpenPI by the AetheRock project for HandCap data loading.

from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import sys
import pathlib
# 将 openpi 模块所在目录加入系统路径，使内部通过 import lerobot 能够直接寻址到本地拷贝的 lerobot
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import lerobot.datasets.dataset_metadata_handcap as dataset_metadata_handcap
import lerobot.datasets.health_distill_dataset_handcap as health_distill_dataset_handcap
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


def _unwrap_transformed_dataset(dataset: Dataset) -> Dataset:
    while isinstance(dataset, TransformedDataset):
        dataset = dataset._dataset
    return dataset


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    if not getattr(data_config, "use_health_distill_multi_handcap", False):
        raise ValueError("This release only includes ForceVT data configurations.")
    health_repo_ids = tuple(data_config.health_repo_ids)
    health_roots = tuple(data_config.health_data_roots)
    if not health_roots:
        raise ValueError("ForceVT training requires the tactile fidelity dataset roots.")
    dataset_meta = dataset_metadata_handcap.LeRobotDatasetMetadataHandcap(health_repo_ids[0], root=health_roots[0])
    delta_timestamps = {
        key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
    }
    dataset = health_distill_dataset_handcap.MultiHealthLeRobotDatasetHandcap(
        health_repo_ids,
        health_roots,
        health_labels=tuple(data_config.health_labels),
        delta_timestamps=delta_timestamps,
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    raw_dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(raw_dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    batch_sampler = None
    if getattr(data_config, "use_health_distill_multi_handcap", False):
        if sampler is not None:
            raise NotImplementedError("Distributed sampling is not implemented for health distillation datasets.")
        health_dataset = _unwrap_transformed_dataset(raw_dataset)
        batch_sampler = BalancedHealthBatchSampler(
            health_lengths=tuple(getattr(health_dataset, "health_lengths")),
            health_offsets=tuple(getattr(health_dataset, "health_offsets")),
            health_sampling_weights=tuple(getattr(data_config, "health_sampling_weights", ())),
            batch_size=local_batch_size,
            shuffle=shuffle,
            seed=seed,
        )

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(batch_sampler is None and sampler is None and shuffle),
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


class BalancedHealthBatchSampler(torch.utils.data.Sampler):
    """Yields cross-health batches while cycling through each source fairly."""

    def __init__(
        self,
        *,
        health_lengths: Sequence[int],
        health_offsets: Sequence[int],
        health_sampling_weights: Sequence[float] = (),
        batch_size: int,
        shuffle: bool,
        seed: int,
    ):
        if len(health_lengths) != len(health_offsets):
            raise ValueError("health_lengths and health_offsets must have the same length.")
        if any(length <= 0 for length in health_lengths):
            raise ValueError("All health datasets must contain at least one frame.")
        self.health_lengths = tuple(int(length) for length in health_lengths)
        self.health_offsets = tuple(int(offset) for offset in health_offsets)
        self.num_healths = len(self.health_lengths)
        self.batch_size = int(batch_size)
        weights = tuple(float(weight) for weight in health_sampling_weights) or tuple(1.0 for _ in self.health_lengths)
        if len(weights) != self.num_healths:
            raise ValueError("health_sampling_weights must have the same length as health_lengths.")
        if any(weight <= 0.0 for weight in weights):
            raise ValueError("health_sampling_weights must be positive.")
        self.health_sampling_weights = weights
        self.samples_per_health = self._allocate_samples_per_health(self.batch_size, weights)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        batches_per_epoch = max(
            1,
            max(
                (length + samples - 1) // samples
                for length, samples in zip(self.health_lengths, self.samples_per_health, strict=True)
            ),
        )
        # Health sampling cycles through shuffled per-source permutations, so a
        # short real epoch only forces PyTorch to tear down/re-prime the iterator.
        # Keep a long virtual epoch to avoid multi-minute prefetch stalls during
        # long training runs.
        self.batches_per_epoch = max(10_000, batches_per_epoch) if self.shuffle else batches_per_epoch

    @staticmethod
    def _allocate_samples_per_health(batch_size: int, weights: Sequence[float]) -> tuple[int, ...]:
        if batch_size < len(weights):
            raise ValueError(
                f"Health distillation batch size ({batch_size}) must be at least the number of health datasets ({len(weights)})."
            )
        total_weight = float(sum(weights))
        raw_counts = [batch_size * weight / total_weight for weight in weights]
        counts = [max(1, int(count)) for count in raw_counts]
        while sum(counts) > batch_size:
            idx = max(range(len(counts)), key=lambda i: counts[i] - raw_counts[i])
            if counts[idx] <= 1:
                raise ValueError("Could not allocate at least one sample for each health dataset.")
            counts[idx] -= 1
        while sum(counts) < batch_size:
            idx = max(range(len(counts)), key=lambda i: raw_counts[i] - counts[i])
            counts[idx] += 1
        return tuple(counts)

    @staticmethod
    def _take_cycled_indices(
        *,
        order: list[int],
        cursor: int,
        length: int,
        count: int,
        generator: torch.Generator | None,
    ) -> tuple[list[int], list[int], int]:
        indices: list[int] = []
        while len(indices) < count:
            take = min(count - len(indices), length - cursor)
            indices.extend(order[cursor : cursor + take])
            cursor += take
            if cursor == length:
                if generator is None:
                    order = list(range(length))
                else:
                    order = torch.randperm(length, generator=generator).tolist()
                cursor = 0
        return indices, order, cursor

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        self.epoch += 1

        cursors = [0] * self.num_healths
        orders = [
            torch.randperm(length, generator=generator).tolist() if self.shuffle else list(range(length))
            for length in self.health_lengths
        ]
        for _ in range(self.batches_per_epoch):
            batch = []
            for health_id, (offset, length) in enumerate(zip(self.health_offsets, self.health_lengths, strict=True)):
                local_indices, orders[health_id], cursors[health_id] = self._take_cycled_indices(
                    order=orders[health_id],
                    cursor=cursors[health_id],
                    length=length,
                    count=self.samples_per_health[health_id],
                    generator=generator if self.shuffle else None,
                )
                batch.extend(offset + local_idx for local_idx in local_indices)
            if self.shuffle:
                order = torch.randperm(len(batch), generator=generator).tolist()
                batch = [batch[i] for i in order]
            yield batch

    def __len__(self):
        return self.batches_per_epoch


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        batch_sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if batch_sampler is None and len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        data_loader_kwargs = {
            "num_workers": num_workers,
            "multiprocessing_context": mp_context,
            "persistent_workers": num_workers > 0,
            "collate_fn": _collate_fn,
            "worker_init_fn": _worker_init_fn,
            "generator": generator,
        }
        if num_workers > 0:
            prefetch_factor = int(os.environ.get("OPENPI_DATALOADER_PREFETCH_FACTOR", "2"))
            if prefetch_factor <= 0:
                raise ValueError("OPENPI_DATALOADER_PREFETCH_FACTOR must be positive.")
            data_loader_kwargs["prefetch_factor"] = prefetch_factor
        if batch_sampler is not None:
            self._data_loader = torch.utils.data.DataLoader(
                typing.cast(torch.utils.data.Dataset, dataset),
                batch_sampler=batch_sampler,
                **data_loader_kwargs,
            )
        else:
            self._data_loader = torch.utils.data.DataLoader(
                typing.cast(torch.utils.data.Dataset, dataset),
                batch_size=local_batch_size,
                shuffle=(sampler is None and shuffle),
                sampler=sampler,
                drop_last=True,
                **data_loader_kwargs,
            )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
