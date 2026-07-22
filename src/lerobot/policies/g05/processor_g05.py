# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea
# Modified for LeRobot in 2026.

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from torch import Tensor
from torch.nn import functional

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    ObservationProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RelativeActionsProcessorStep,
    batch_to_transition,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)
from lerobot.utils.import_utils import _transformers_available

from .configuration_g05 import G05Config
from .modular_g05 import (
    G05_ACTION_DIM_PAD_MASK,
    G05_ACTION_PAD_MASK,
    G05_ACTION_TOKEN_IDS,
    G05_ATTENTION_MASK,
    G05_IMAGES,
    G05_INPUT_IDS,
    G05_LABELS,
    G05_PREFIX_LENGTH,
)


@dataclass
class G05LiberoObservationStep(ObservationProcessorStep):
    """Match the released G0.5 LIBERO observation boundary."""

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        observation = observation.copy()
        for key, image in observation.items():
            if key.startswith("observation.images."):
                # LIBERO returns both cameras upside down relative to the training data.
                observation[key] = torch.flip(image, dims=(-2, -1))

        robot_state = observation.pop("observation.robot_state", None)
        if robot_state is None:
            return observation
        position = robot_state["eef"]["pos"]
        axis_angle = self._quat2axisangle(robot_state["eef"]["quat"])
        gripper = robot_state["gripper"]["qpos"][..., :1]
        observation[OBS_STATE] = torch.cat((position, axis_angle, gripper), dim=-1).float()
        return observation

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features = {feature_type: values.copy() for feature_type, values in features.items()}
        features[PipelineFeatureType.STATE] = {
            OBS_STATE: PolicyFeature(type=PipelineFeatureType.STATE, shape=(7,))
        }
        return features

    @staticmethod
    def _quat2axisangle(quaternion: Tensor) -> Tensor:
        """Convert LIBERO xyzw quaternions to the axis-angle representation used in training."""
        scalar = quaternion[..., 3:].clamp(-1, 1)
        denominator = torch.sqrt((1 - scalar.square()).clamp_min(0))
        scale = 2 * torch.acos(scalar) / denominator.clamp_min(1e-8)
        return quaternion[..., :3] * torch.where(denominator > 1e-8, scale, scale.new_zeros(()))


@dataclass
class G05LiberoActionStep(ProcessorStep):
    """Convert the trained [0, 1] gripper convention to LIBERO's {-1, +1} commands."""

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        action = transition.get(TransitionKey.ACTION)
        if action is not None:
            action = action.clone()
            # G0.5 encodes 0 as closed and 1 as open. LIBERO uses the opposite
            # actuator signs: +1 closes and -1 opens.
            action[..., -1] = torch.where(action[..., -1] > 0.5, -1.0, 1.0)
            transition[TransitionKey.ACTION] = action
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_g05_libero_pre_post_processors() -> tuple[PolicyProcessorPipeline, PolicyProcessorPipeline]:
    """Build the environment-side adapters required by the released LIBERO checkpoint."""
    return (
        PolicyProcessorPipeline(steps=[G05LiberoObservationStep()]),
        PolicyProcessorPipeline(steps=[G05LiberoActionStep()]),
    )


def _normalize(values: Tensor, low: list, high: list) -> Tensor:
    if not low or not high:
        return values
    lower, upper = values.new_tensor(low), values.new_tensor(high)
    return (2 * (values - lower) / (upper - lower).clamp_min(1e-6) - 1).clamp(-1, 1)


def _unnormalize(values: Tensor, low: list, high: list) -> Tensor:
    if not low or not high:
        return values
    lower, upper = values.new_tensor(low), values.new_tensor(high)
    return (values + 1) * (upper - lower) / 2 + lower


def _apply_normalization(
    values: Tensor, specs: list[dict], *, inverse: bool, step_index: int | None = None
) -> Tensor:
    """Apply the legacy per-part normalizer exactly at the LeRobot boundary."""
    if not specs:
        return values
    output: list[Tensor] = []
    offset = 0
    for spec in specs:
        stats = {name: values.new_tensor(value) for name, value in spec["stats"].items()}
        width = next(iter(stats.values())).shape[-1]
        current = values[..., offset : offset + width]
        offset += width
        if next(iter(stats.values())).ndim >= 2:
            if current.ndim >= 3:
                horizon = current.shape[-2]
                if horizon > next(iter(stats.values())).shape[-2]:
                    raise ValueError("G0.5 action horizon exceeds the published normalization statistics")
                stats = {name: value[..., :horizon, :] for name, value in stats.items()}
            elif step_index is not None:
                index = min(step_index, next(iter(stats.values())).shape[-2] - 1)
                stats = {name: value[..., index, :] for name, value in stats.items()}

        mode = spec["mode"]
        base_mode = mode.removesuffix("-tail")
        if base_mode == "z-score":
            mean, std = stats["mean"], stats["std"]
            scale = 1 / (std + 1e-8)
            shift = -mean / (std + 1e-8)
            constant = std < 1e-4
            scale = torch.where(constant, torch.ones_like(scale), scale)
            shift = torch.where(constant, -mean, shift)
        elif base_mode == "q01/q99":
            low, high = stats["q01"], stats["q99"]
            value_range = high - low
            constant = value_range < 1e-4
            value_range = torch.where(constant, torch.full_like(value_range, 2), value_range)
            scale = 2 / value_range
            shift = -1 - scale * low
            shift = torch.where(constant, -low, shift)
        else:
            raise ValueError(f"unsupported G0.5 normalization mode: {mode}")

        if inverse:
            current = (current - shift) / scale
        if mode.endswith("-tail"):
            q01, q99, mean = stats["q01"], stats["q99"], stats["mean"]
            degenerate = (q99 <= q01) | (mean <= q01) | (mean >= q99)
            c_pos = torch.where(degenerate, torch.ones_like(mean), 0.075 * (q99 - mean))
            c_neg = torch.where(degenerate, torch.ones_like(mean), 0.075 * (mean - q01))
            if inverse:
                positive = q99 + c_pos * torch.expm1(((current - q99) / c_pos).clamp_min(0))
                negative = q01 - c_neg * torch.expm1(((q01 - current) / c_neg).clamp_min(0))
            else:
                positive = q99 + c_pos * torch.log1p(((current - q99) / c_pos).clamp_min(0))
                negative = q01 - c_neg * torch.log1p(((q01 - current) / c_neg).clamp_min(0))
            current = torch.where(current > q99, positive, torch.where(current < q01, negative, current))
        if not inverse:
            current = (current * scale + shift).clamp(-5, 5).nan_to_num()
        output.append(current)
    if offset != values.shape[-1]:
        raise ValueError("G0.5 normalization specs do not cover the physical tensor")
    return torch.cat(output, dim=-1)


def _pad_last_dim(values: Tensor, indices: list[int], target_dim: int) -> tuple[Tensor, Tensor]:
    physical_dim = values.shape[-1]
    indices = indices or list(range(physical_dim))
    if len(indices) != physical_dim or len(set(indices)) != len(indices):
        raise ValueError("G0.5 layout indices must uniquely map every physical dimension")
    if min(indices, default=0) < 0 or max(indices, default=-1) >= target_dim:
        raise ValueError("G0.5 layout index exceeds the checkpoint dimension")
    output = values.new_zeros(*values.shape[:-1], target_dim)
    output[..., indices] = values
    dimension_is_pad = torch.ones(values.shape[0], target_dim, dtype=torch.bool, device=values.device)
    dimension_is_pad[:, indices] = False
    return output, dimension_is_pad


@ProcessorStepRegistry.register(name="g05_prepare_inputs")
@dataclass
class G05PrepareInputsStep(ProcessorStep):
    tokenizer_path: str
    action_tokenizer_path: str
    camera_keys: list[str]
    dummy_camera_keys: list[str]
    image_size: tuple[int, int]
    patch_size: int
    spatial_merge_size: int
    n_obs_steps: int
    internal_state_dim: int
    internal_action_dim: int
    state_indices: list[int]
    action_indices: list[int]
    state_norm_low: list
    state_norm_high: list
    action_norm_low: list
    action_norm_high: list
    state_normalization: list[dict]
    action_normalization: list[dict]
    embodiment: str
    max_task_tokens: int
    max_prompt_length: int
    image_token_id: int
    vision_start_token_id: int
    vision_end_token_id: int
    state_token_id: int
    eov_token_id: int
    pad_token_id: int
    eos_token_id: int
    camera_order: list[str] | None = None

    def __post_init__(self) -> None:
        if not _transformers_available:
            raise ImportError("G0.5 processors require transformers; install lerobot[g05]")
        if not self.tokenizer_path:
            raise ValueError("tokenizer_path must resolve inside the G0.5 artifact")
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, local_files_only=True)
        self._action_tokenizer = None
        if self.action_tokenizer_path:
            from .action_tokenizer import G05ActionCodecModel, G05ActionTokenizer

            codec = G05ActionCodecModel.from_pretrained(
                self.action_tokenizer_path,
                local_files_only=True,
            )
            self._action_tokenizer = G05ActionTokenizer(codec, self._tokenizer)

    def get_config(self) -> dict[str, Any]:
        return {
            item.name: (
                "" if item.name in {"tokenizer_path", "action_tokenizer_path"} else getattr(self, item.name)
            )
            for item in fields(self)
        }

    def _prepare_images(self, observation: dict[str, Any], state: Tensor) -> Tensor:
        images: list[Tensor] = []
        camera_order = self.camera_order or [*self.camera_keys, *self.dummy_camera_keys]
        for key in camera_order:
            value = observation.get(key)
            if value is None:
                if key not in self.dummy_camera_keys:
                    raise ValueError(f"missing required G0.5 camera {key!r}")
                value = state.new_zeros(
                    state.shape[0], self.n_obs_steps, 3, self.image_size[0], self.image_size[1]
                )
            if value.ndim == 4:
                value = value.unsqueeze(1)
            if value.ndim != 5:
                raise ValueError(f"camera {key!r} must have shape [B,T,C,H,W] or [B,C,H,W]")
            if value.shape[1] != self.n_obs_steps:
                raise ValueError(f"camera {key!r} has {value.shape[1]} frames; expected {self.n_obs_steps}")
            batch_size, steps, channels, height, width = value.shape
            value = value.float()
            if value.max().item() > 1.5:
                value = value / 255
            if (height, width) != tuple(self.image_size):
                value = functional.interpolate(
                    value.reshape(batch_size * steps, channels, height, width),
                    size=self.image_size,
                    mode="bilinear",
                    align_corners=False,
                ).reshape(batch_size, steps, channels, *self.image_size)
            images.append(value * 2 - 1)
        return torch.stack(images, dim=1)

    def _prompt_ids(self, tasks: list[str], num_images: int) -> tuple[Tensor, Tensor]:
        image_tokens = (self.image_size[0] // self.patch_size // self.spatial_merge_size) * (
            self.image_size[1] // self.patch_size // self.spatial_merge_size
        )
        rows: list[list[int]] = []
        for task in tasks:
            row = self._tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
            for _ in range(num_images * self.n_obs_steps):
                row += [self.vision_start_token_id]
                row += [self.image_token_id] * image_tokens
                row += [self.vision_end_token_id]
            # Static and dynamic template segments are tokenized separately in G0.5.
            row += self._tokenizer.encode("Embodiment: ", add_special_tokens=False)
            row += self._tokenizer.encode(self.embodiment, add_special_tokens=False)
            row += self._tokenizer.encode("; Task: ", add_special_tokens=False)
            task_ids = self._tokenizer.encode(task.strip(), add_special_tokens=False)
            row += task_ids[: self.max_task_tokens]
            row += self._tokenizer.encode(" State: ", add_special_tokens=False)
            row += [self.state_token_id] * self.n_obs_steps
            row += self._tokenizer.encode(
                ";<|im_end|>\n<|im_start|>robot\nAction: ", add_special_tokens=False
            )
            row += [self.eov_token_id]
            if len(row) > self.max_prompt_length:
                raise ValueError("G0.5 prompt exceeds max_prompt_length")
            rows.append(row)
        width = max(map(len, rows))
        ids = torch.full((len(rows), width), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.bool)
        for index, row in enumerate(rows):
            # G0.5 right-aligns prefix batches, so shorter prompts are left-padded.
            ids[index, width - len(row) :] = torch.tensor(row)
            mask[index, width - len(row) :] = True
        return ids, mask

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = dict(transition.get(TransitionKey.OBSERVATION) or {})
        complementary = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})
        state = observation.get(OBS_STATE)
        if state is None:
            raise ValueError("G0.5 requires observation.state")
        state = state.float()
        state = (
            _apply_normalization(state, self.state_normalization, inverse=False)
            if self.state_normalization
            else _normalize(state, self.state_norm_low, self.state_norm_high)
        )
        state, _ = _pad_last_dim(state, self.state_indices, self.internal_state_dim)
        observation[OBS_STATE] = state
        complementary[G05_IMAGES] = self._prepare_images(observation, state)
        action_dimension_is_pad = torch.ones(
            state.shape[0], self.internal_action_dim, dtype=torch.bool, device=state.device
        )
        action_dimension_is_pad[:, self.action_indices] = False
        complementary[G05_ACTION_DIM_PAD_MASK] = action_dimension_is_pad

        tasks = complementary.get("task")
        if isinstance(tasks, str):
            tasks = [tasks]
        if not isinstance(tasks, (list, tuple)) or not all(isinstance(task, str) for task in tasks):
            raise ValueError("G0.5 requires one task string per batch item")
        ids, mask = self._prompt_ids(list(tasks), len(self.camera_keys) + len(self.dummy_camera_keys))
        complementary[G05_INPUT_IDS] = ids.to(state.device)
        complementary[G05_ATTENTION_MASK] = mask.to(state.device)

        action = transition.get(TransitionKey.ACTION)
        if action is not None:
            action = action.float()
            action = (
                _apply_normalization(action, self.action_normalization, inverse=False)
                if self.action_normalization
                else _normalize(action, self.action_norm_low, self.action_norm_high)
            )
            action, dimension_is_pad = _pad_last_dim(action, self.action_indices, self.internal_action_dim)
            transition[TransitionKey.ACTION] = action
            complementary[G05_ACTION_DIM_PAD_MASK] = dimension_is_pad
            complementary[G05_ACTION_PAD_MASK] = torch.zeros(
                action.shape[:2], dtype=torch.bool, device=action.device
            )
            if self._action_tokenizer is not None:
                codec_device = next(self._action_tokenizer.model.parameters()).device
                if codec_device != action.device:
                    self._action_tokenizer.model.to(action.device)
                action_token_ids = self._action_tokenizer.encode(action)
                complementary[G05_ACTION_TOKEN_IDS] = action_token_ids
                prefix_length = ids.shape[1]
                eos = torch.full((ids.shape[0], 1), self.eos_token_id, dtype=torch.long, device=action.device)
                suffix = torch.cat((action_token_ids, eos), dim=-1)
                complementary[G05_INPUT_IDS] = torch.cat((ids.to(action.device), suffix), dim=-1)
                complementary[G05_ATTENTION_MASK] = torch.cat(
                    (mask.to(action.device), torch.ones_like(suffix, dtype=torch.bool)), dim=-1
                )
                complementary[G05_LABELS] = torch.cat(
                    (torch.full_like(ids, -100, device=action.device), suffix), dim=-1
                )
                complementary[G05_PREFIX_LENGTH] = prefix_length

        transition[TransitionKey.OBSERVATION] = observation
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register(name="g05_restore_action")
@dataclass
class G05RestoreActionStep(ProcessorStep):
    action_norm_low: list
    action_norm_high: list
    action_normalization: list[dict] | None = None
    action_horizon: int | None = None

    def __post_init__(self) -> None:
        self._step_index = 0

    def reset(self) -> None:
        self._step_index = 0

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            raise ValueError("G0.5 postprocessor requires an action tensor")
        if self.action_normalization:
            step_index = self._step_index
            if self.action_horizon is not None:
                step_index %= self.action_horizon
            transition[TransitionKey.ACTION] = _apply_normalization(
                action,
                self.action_normalization,
                inverse=True,
                step_index=step_index,
            )
            if action.ndim < 3:
                self._step_index += 1
            return transition
        low, high = self.action_norm_low, self.action_norm_high
        if low and isinstance(low[0], list) and action.ndim < 3:
            index = min(self._step_index, len(low) - 1)
            low, high = low[index], high[index]
            self._step_index += 1
        transition[TransitionKey.ACTION] = _unnormalize(action, low, high)
        return transition

    def get_config(self) -> dict[str, Any]:
        return {
            "action_norm_low": self.action_norm_low,
            "action_norm_high": self.action_norm_high,
            "action_normalization": self.action_normalization,
            "action_horizon": self.action_horizon,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_g05_pre_post_processors(
    config: G05Config,
    dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    *,
    tokenizer_path: str | Path | None = None,
    action_tokenizer_path: str | Path | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    del dataset_stats
    if config.state_token_id is None or config.eov_token_id is None:
        raise ValueError("converted G0.5 config must define state_token_id and eov_token_id")
    steps = make_default_policy_processor_steps(config, None)
    relative_step = RelativeActionsProcessorStep(
        enabled=bool(config.relative_action_mask),
        action_names=["relative" if value else "absolute" for value in config.relative_action_mask] or None,
        exclude_joints=["absolute"],
    )
    prepare = G05PrepareInputsStep(
        tokenizer_path=str(tokenizer_path or ""),
        action_tokenizer_path=str(action_tokenizer_path or ""),
        camera_keys=config.camera_keys,
        dummy_camera_keys=config.dummy_camera_keys,
        image_size=config.image_size,
        patch_size=config.vision_patch_size,
        spatial_merge_size=config.vision_spatial_merge_size,
        n_obs_steps=config.n_obs_steps,
        internal_state_dim=config.internal_state_dim,
        internal_action_dim=config.internal_action_dim,
        state_indices=config.state_indices,
        action_indices=config.action_indices,
        state_norm_low=config.state_norm_low,
        state_norm_high=config.state_norm_high,
        action_norm_low=config.action_norm_low,
        action_norm_high=config.action_norm_high,
        state_normalization=config.state_normalization,
        action_normalization=config.action_normalization,
        embodiment=config.embodiment,
        max_task_tokens=config.max_task_tokens,
        max_prompt_length=config.max_prompt_length,
        image_token_id=config.image_token_id,
        vision_start_token_id=config.vision_start_token_id,
        vision_end_token_id=config.vision_end_token_id,
        state_token_id=config.state_token_id,
        eov_token_id=config.eov_token_id,
        pad_token_id=config.pad_token_id,
        eos_token_id=config.eos_token_id,
        camera_order=config.camera_order,
    )
    return make_policy_processor_pipelines(
        input_steps=[
            steps.rename_observations,
            steps.add_batch_dim,
            relative_step,
            steps.to_device,
            prepare,
        ],
        output_steps=[
            G05RestoreActionStep(
                action_norm_low=config.action_norm_low,
                action_norm_high=config.action_norm_high,
                action_normalization=config.action_normalization,
                action_horizon=config.n_action_steps,
            ),
            AbsoluteActionsProcessorStep(
                enabled=bool(config.relative_action_mask), relative_step=relative_step
            ),
            steps.to_cpu,
        ],
    )


def make_g05_pre_post_processors_from_pretrained(
    config: G05Config,
    pretrained_path: str,
    *,
    revision: str | None = None,
    preprocessor_config_filename: str = f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
    postprocessor_config_filename: str = f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Load processors while resolving the tokenizer inside the same artifact."""
    artifact_root = Path(pretrained_path)
    if not artifact_root.is_dir():
        artifact_root = Path(
            snapshot_download(
                pretrained_path,
                revision=revision,
                allow_patterns=[
                    f"{config.tokenizer_subdir}/*",
                    f"{config.action_tokenizer_subdir}/*",
                    preprocessor_config_filename,
                    postprocessor_config_filename,
                    "*.safetensors",
                ],
            )
        )
    tokenizer_path = artifact_root / config.tokenizer_subdir
    action_tokenizer_path = artifact_root / config.action_tokenizer_subdir
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(
            f"self-contained G0.5 artifact is missing tokenizer directory: {tokenizer_path}"
        )
    if not action_tokenizer_path.is_dir():
        raise FileNotFoundError(
            f"self-contained G0.5 artifact is missing action tokenizer directory: {action_tokenizer_path}"
        )
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=artifact_root,
        config_filename=preprocessor_config_filename,
        overrides={
            "g05_prepare_inputs": {
                "tokenizer_path": str(tokenizer_path),
                "action_tokenizer_path": str(action_tokenizer_path) if config.discrete_action else "",
            }
        },
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=artifact_root,
        config_filename=postprocessor_config_filename,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    restore_step = next(
        (step for step in postprocessor.steps if isinstance(step, G05RestoreActionStep)), None
    )
    if restore_step is not None:
        # Older converted artifacts predate this serialized field.
        restore_step.action_horizon = config.n_action_steps
    relative_step = next(
        (step for step in preprocessor.steps if isinstance(step, RelativeActionsProcessorStep)), None
    )
    if relative_step is not None:
        for step in postprocessor.steps:
            if isinstance(step, AbsoluteActionsProcessorStep):
                step.relative_step = relative_step
    return preprocessor, postprocessor
