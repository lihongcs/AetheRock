import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 4:
        image = image[-1]
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(255 * image, 0, 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _parse_temporal_grid(image, grid_shape: tuple[int, int] = (2, 2)) -> np.ndarray:
    frames = np.asarray(image)
    rows, cols = grid_shape
    if rows <= 0 or cols <= 0:
        raise ValueError("grid_shape dimensions must be positive.")

    num_cells = rows * cols
    if frames.ndim <= 3:
        parsed = _parse_image(frames)
        parsed_frames = [parsed] * num_cells
    else:
        if frames.shape[0] < num_cells:
            pad = np.repeat(frames[:1], num_cells - frames.shape[0], axis=0)
            frames = np.concatenate([pad, frames], axis=0)
        parsed_frames = [_parse_image(frame) for frame in frames[-num_cells:]]

    target_h, target_w = parsed_frames[-1].shape[:2]
    grid_rows = []
    for row in range(rows):
        start = row * cols
        grid_rows.append(np.concatenate(parsed_frames[start : start + cols], axis=1))
    grid = np.concatenate(grid_rows, axis=0)
    if grid.shape[:2] != (target_h, target_w):
        from PIL import Image as PILImage

        resampling = getattr(PILImage, "Resampling", PILImage).BILINEAR
        grid = np.asarray(PILImage.fromarray(grid).resize((target_w, target_h), resampling))
    return grid


def _force_from_value(value, force_dim: int) -> np.ndarray:
    force = np.asarray(value, dtype=np.float32)
    if force.ndim == 0:
        force = np.array([force], dtype=np.float32)
    if force.shape[-1] < force_dim:
        pad_width = [(0, 0)] * force.ndim
        pad_width[-1] = (0, force_dim - force.shape[-1])
        force = np.pad(force, pad_width)
    return force[..., :force_dim]


def _extract_force(data: dict, state: np.ndarray, force_dim: int, action_base_dim: int) -> np.ndarray:
    for key in (
        "observation/force",
        "observation/forces",
        "observation.force",
        "observation.forces",
        "force",
    ):
        if key in data:
            return _force_from_value(data[key], force_dim)

    left = None
    right = None
    for key in ("observation/force_left", "observation.forces.left", "observation/forces/left"):
        if key in data:
            left = _force_from_value(data[key], 1)
            break
    for key in ("observation/force_right", "observation.forces.right", "observation/forces/right"):
        if key in data:
            right = _force_from_value(data[key], 1)
            break
    if left is not None and right is not None:
        return _force_from_value(np.concatenate([left, right], axis=-1), force_dim)

    state = np.asarray(state)
    if state.shape[-1] >= action_base_dim + force_dim:
        return _force_from_value(state[..., action_base_dim : action_base_dim + force_dim], force_dim)

    return np.zeros((force_dim,), dtype=np.float32)


def _append_force_to_state(state: np.ndarray, force: np.ndarray, action_base_dim: int) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if state.shape[-1] >= action_base_dim + force.shape[-1]:
        return state
    return np.concatenate([state[..., :action_base_dim], force.astype(state.dtype)], axis=-1)


def _ensure_action_force(actions: np.ndarray, force_dim: int, action_base_dim: int) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    output_dim = action_base_dim + force_dim
    if actions.shape[-1] >= output_dim:
        return actions[..., :output_dim]
    if actions.shape[-1] < action_base_dim:
        actions = transforms.pad_to_dim(actions, action_base_dim, axis=-1)
    force_pad_shape = actions.shape[:-1] + (force_dim,)
    force_pad = np.zeros(force_pad_shape, dtype=actions.dtype)
    return np.concatenate([actions[..., :action_base_dim], force_pad], axis=-1)


@dataclasses.dataclass(frozen=True)
class HandcapInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # The action dimension of the model. Will be used to pad state and actions for pi0 model (not pi0-FAST).
    # Do not change this for your own dataset.
    action_dim: int

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType = _model.ModelType.PI0
    include_tactile: bool = True
    force_predict: bool = False
    force_guide: bool = False
    action_base_dim: int = 10
    force_dim: int = 2
    force_observation_current_only: bool = False
    tactile_temporal_grid: bool = False
    tactile_grid_shape: tuple[int, int] = (2, 2)

    def __call__(self, data: dict) -> dict:
        # We only mask padding for pi0 model, not pi0-FAST. Do not change this for your own dataset.
        mask_padding = self.model_type == _model.ModelType.PI0

        # We pad the proprioceptive input to the action dimension of the model.
        # For pi0-FAST, we don't pad the state. For Libero, we don't need to differentiate
        # since the pi0-FAST action_dim = 7, which is < state_dim = 8, so pad is skipped.
        # Keep this for your own dataset, but if your dataset stores the proprioceptive input
        # in a different key than "observation/state", you should change it below.
        raw_state = np.asarray(data["observation/state"], dtype=np.float32)
        force = _extract_force(data, raw_state, self.force_dim, self.action_base_dim)
        current_force = force[0] if force.ndim > 1 else force
        
        if self.force_predict or self.force_guide:
            raw_state = _append_force_to_state(raw_state, current_force, self.action_base_dim)
        state = transforms.pad_to_dim(raw_state, self.action_dim)

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        wrist_image = _parse_image(data["observation/wrist_image"])
        images = {"wrist_0_rgb": wrist_image}
        image_mask = {"wrist_0_rgb": np.True_}

        if self.include_tactile:
            if self.tactile_temporal_grid:
                left_tactile = _parse_temporal_grid(data["observation/left_tactile"], self.tactile_grid_shape)
                right_tactile = _parse_temporal_grid(data["observation/right_tactile"], self.tactile_grid_shape)
            else:
                left_tactile = _parse_image(data["observation/left_tactile"])
                right_tactile = _parse_image(data["observation/right_tactile"])
            images.update(
                {
                    "left_tactile_0_rgb": left_tactile,
                    "right_tactile_0_rgb": right_tactile,
                }
            )
            image_mask.update(
                {
                    "left_tactile_0_rgb": np.True_,
                    "right_tactile_0_rgb": np.True_,
                }
            )

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": state,
            "force": current_force if self.force_observation_current_only else force,
            "image": images,
            "image_mask": image_mask,
        }
        for metadata_key in ("health_id",):
            if metadata_key in data:
                inputs[metadata_key] = np.asarray(data[metadata_key])

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if self.force_predict:
                if force.ndim > 1 and force.shape[0] == actions.shape[0]:
                    # force sequence is available, concatenate with actions
                    actions = np.concatenate([actions[..., :self.action_base_dim], force], axis=-1)
                else:
                    # fallback to padding with zero
                    actions = _ensure_action_force(actions, self.force_dim, self.action_base_dim)
            actions = transforms.pad_to_dim(actions, self.action_dim)
            inputs["actions"] = actions

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class HandcapOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    output_action_dim: int = 10
    force_predict: bool = False
    action_base_dim: int = 10
    force_dim: int = 2

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        actions = np.asarray(data["actions"][:, : self.output_action_dim])
        outputs = {"actions": actions}
        if self.force_predict:
            outputs["robot_actions"] = actions[:, : self.action_base_dim]
            outputs["gripper_pred"] = actions[:, self.action_base_dim - 1 : self.action_base_dim]
            outputs["force_pred"] = actions[:, self.action_base_dim : self.action_base_dim + self.force_dim]
        return outputs
