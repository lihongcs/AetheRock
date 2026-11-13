from collections.abc import Callable, Mapping, Sequence
import dataclasses
import functools
import pathlib
import re
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable, Any

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class HandcapRepackTransform(DataTransformFn):
    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        structure = {
            "observation/wrist_image": "observation.images.wrist",
            "observation/left_tactile": "observation.tactiles.left",
            "observation/right_tactile": "observation.tactiles.right",
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
        }

        flat_item = flatten_dict(data)
        output = jax.tree.map(lambda k: flat_item[k], structure)

        for key in ("observation.force", "observation.forces", "observation/force", "force"):
            if key in flat_item:
                output["observation/force"] = flat_item[key]
                break
        else:
            left_key = "observation.forces.left"
            right_key = "observation.forces.right"
            if left_key in flat_item and right_key in flat_item:
                left = np.asarray(flat_item[left_key])
                right = np.asarray(flat_item[right_key])
                # Make sure they have at least 1 dimension
                if left.ndim == 0: left = left[None]
                if right.ndim == 0: right = right[None]
                output["observation/force"] = np.concatenate([left, right], axis=-1)
            elif left_key in flat_item:
                output["observation/force"] = flat_item[left_key]

        for metadata_key in ("health_id",):
            if metadata_key in flat_item:
                output[metadata_key] = flat_item[metadata_key]

        return output


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean, stats.std
        if (dim := mean.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] - mean) / (std + 1e-6), x[..., dim:]], axis=-1)
        mean, std = mean[..., : x.shape[-1]], std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate(
                [(x[..., :dim] - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0, x[..., dim:]],
                axis=-1,
            )
        q01, q99 = q01[..., : x.shape[-1]], q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        return data


@functools.lru_cache(maxsize=16)
def _load_reference_image(path: str, height: int, width: int) -> np.ndarray:
    from PIL import Image

    image_path = pathlib.Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Tactile reference frame not found: {image_path}")
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    image = Image.open(image_path).convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), resampling)
    return np.asarray(image, dtype=np.uint8)


@dataclasses.dataclass(frozen=True)
class TactileReferenceMask(DataTransformFn):
    """Randomly replace a tactile image region with a reference no-contact frame."""

    refer_dir: str
    mask_ratio: float = 0.3
    sample_ratio: float = 1.0
    left_key: str = "observation/left_tactile"
    right_key: str = "observation/right_tactile"

    def __post_init__(self):
        if not 0.0 <= self.mask_ratio <= 1.0:
            raise ValueError(f"mask_ratio must be in [0, 1], got {self.mask_ratio}")
        if not 0.0 <= self.sample_ratio <= 1.0:
            raise ValueError(f"sample_ratio must be in [0, 1], got {self.sample_ratio}")

    def __call__(self, data: DataDict) -> DataDict:
        if self.mask_ratio <= 0.0 or self.sample_ratio <= 0.0:
            return data
        refer_dir = pathlib.Path(self.refer_dir)
        selection = self._sample_selection(data)
        if selection is False:
            return data
        if self.left_key in data:
            data[self.left_key] = self._mask_image(data[self.left_key], self._reference_path(refer_dir, "left"), selection)
        if self.right_key in data:
            data[self.right_key] = self._mask_image(data[self.right_key], self._reference_path(refer_dir, "right"), selection)
        return data

    def _sample_selection(self, data: DataDict) -> bool | np.ndarray:
        for key in (self.left_key, self.right_key):
            if key not in data:
                continue
            array = np.asarray(data[key])
            if array.ndim == 4:
                count = array.shape[0]
                selected_count = int(round(count * self.sample_ratio))
                if selected_count <= 0:
                    return False
                if selected_count >= count:
                    return np.ones(count, dtype=bool)
                selected = np.zeros(count, dtype=bool)
                selected[np.random.choice(count, size=selected_count, replace=False)] = True
                return selected
            break
        return bool(np.random.random() < self.sample_ratio)

    def _reference_path(self, refer_dir: pathlib.Path, side: str) -> pathlib.Path:
        candidates = {
            "left": ("lefttactile.png", "left_tactile.png", "left.png"),
            "right": ("righttactile.png", "right_tactile.png", "right.png"),
        }[side]
        for name in candidates:
            path = refer_dir / name
            if path.exists():
                return path
        return refer_dir / candidates[0]

    def _mask_image(self, image: np.ndarray, reference_path: pathlib.Path, selection: bool | np.ndarray) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim == 4:
            output = np.array(array, copy=True)
            selected = np.asarray(selection, dtype=bool)
            if selected.shape != (output.shape[0],):
                selected = np.full(output.shape[0], bool(np.any(selected)), dtype=bool)
            for idx in range(output.shape[0]):
                if not selected[idx]:
                    continue
                output[idx] = self._mask_single_image(output[idx], reference_path)
            return output
        if selection is not True:
            return image
        return self._mask_single_image(array, reference_path)

    def _mask_single_image(self, image: np.ndarray, reference_path: pathlib.Path) -> np.ndarray:
        output = np.array(image, copy=True)
        is_chw = output.ndim == 3 and output.shape[0] in (1, 3) and output.shape[-1] not in (1, 3)
        if is_chw:
            output_hwc = np.moveaxis(output, 0, -1)
        else:
            output_hwc = output
        if output_hwc.ndim != 3:
            return output

        height, width, channels = output_hwc.shape
        ref = _load_reference_image(str(reference_path), height, width)
        if channels == 1:
            ref = ref[..., :1]
        elif channels < ref.shape[-1]:
            ref = ref[..., :channels]
        elif channels > ref.shape[-1]:
            pad = np.repeat(ref[..., -1:], channels - ref.shape[-1], axis=-1)
            ref = np.concatenate([ref, pad], axis=-1)
        ref = self._convert_reference_dtype(ref, output_hwc)

        y0, x0, mask_h, mask_w = self._random_region(height, width)
        output_hwc[y0 : y0 + mask_h, x0 : x0 + mask_w] = ref[y0 : y0 + mask_h, x0 : x0 + mask_w]

        if is_chw:
            return np.moveaxis(output_hwc, -1, 0)
        return output_hwc

    def _convert_reference_dtype(self, reference: np.ndarray, target: np.ndarray) -> np.ndarray:
        if np.issubdtype(target.dtype, np.floating):
            ref = reference.astype(np.float32)
            if np.nanmax(target) <= 1.0:
                ref = ref / 255.0
            return ref.astype(target.dtype)
        return reference.astype(target.dtype, copy=False)

    def _random_region(self, height: int, width: int) -> tuple[int, int, int, int]:
        area = max(1, int(round(height * width * self.mask_ratio)))
        aspect = float(2 ** np.random.uniform(-1.0, 1.0))
        mask_h = max(1, min(height, int(round(np.sqrt(area / aspect)))))
        mask_w = max(1, min(width, int(round(area / mask_h))))
        y0 = int(np.random.randint(0, height - mask_h + 1))
        x0 = int(np.random.randint(0, width - mask_w + 1))
        return y0, x0, mask_h, mask_w


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str] | Any

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        prompt = None

        if hasattr(self.tasks, "index") and hasattr(self.tasks, "task_index"): # pandas DataFrame
            # The index is the task string, and the column is 'task_index'
            matches = self.tasks[self.tasks["task_index"] == task_index]
            if len(matches) > 0:
                prompt = str(matches.index[0])
        elif isinstance(self.tasks, dict):
            prompt = self.tasks.get(task_index)

        if prompt is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
