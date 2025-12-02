import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def make_calvin_example() -> dict:
    """Creates a random input example for the Calvin policy."""
    return {
        # State observations
        "state.ee_pos": np.random.rand(3),
        "state.ee_rot": np.random.rand(3),
        # "state.joints": np.random.rand(7),
        "state.gripper": np.random.rand(1),
        # Image observations
        "video.image_base": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "video.image_wrist": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        # Language prompt
        "task": "push the pink block to the left",
    }


@dataclasses.dataclass(frozen=True)
class CalvinInputs(transforms.DataTransformFn):
    """
    This class converts inputs from the Calvin dataset format to the format expected by the model.
    It is used for both training and inference.
    """
    
    # # Determines which model will be used.
    # model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Parse the main and wrist camera images.
        base_image = _parse_image(data["image"]["base"])
        wrist_image = _parse_image(data["image"]["wrist"])

        # Concatenate all parts of the robot's state into a single vector.
        # The order is: end-effector position (3), rotation (3), joint positions (7), and gripper state (1).
        # Total state dimension = 3 + 3 + 1 = 7.
        state = np.concatenate(
            [
                data["state"]["ee_pos"],
                data["state"]["ee_rot"],
                data["state"]["gripper"],
            ],
            axis=-1,
        )

        # Create the 'inputs' dictionary that the model expects.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            actions = np.concatenate(
                [
                    data["actions"]["delta_ee_pos"],
                    data["actions"]["delta_ee_rot"],
                    data["actions"]["gripper"],
                ],
                axis=-1,
            )

            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class CalvinOutputs(transforms.DataTransformFn):
    """
    This class converts the model's output actions back to the dataset-specific format.
    It is primarily used during inference.
    """

    def __call__(self, data: dict) -> dict:
        # The model outputs a padded action tensor. We only need the first 7 dimensions
        # which correspond to the actual action space of the Calvin environment.
        return {"actions": np.asarray(data["actions"][:, :7])}