# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea
# Modified for LeRobot in 2026.

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE


def _default_layer_types() -> list[str]:
    return ["linear_attention" if index % 4 != 3 else "full_attention" for index in range(24)]


@PreTrainedConfig.register_subclass("g05")
@dataclass
class G05Config(PreTrainedConfig):
    """Serializable G0.5 architecture and data-boundary contract.

    Values that vary between published checkpoints are stored in the converted
    artifact. Runtime code never infers an embodiment from a checkpoint name.
    """

    chunk_size: int = 32
    n_action_steps: int = 16
    n_obs_steps: int = 1
    image_size: tuple[int, int] = (256, 256)
    camera_keys: list[str] = field(default_factory=list)
    dummy_camera_keys: list[str] = field(default_factory=list)
    camera_order: list[str] = field(default_factory=list)

    internal_action_dim: int = 27
    internal_state_dim: int = 27
    action_indices: list[int] = field(default_factory=list)
    state_indices: list[int] = field(default_factory=list)
    action_norm_low: list = field(default_factory=list)
    action_norm_high: list = field(default_factory=list)
    state_norm_low: list = field(default_factory=list)
    state_norm_high: list = field(default_factory=list)
    action_normalization: list[dict] = field(default_factory=list)
    state_normalization: list[dict] = field(default_factory=list)
    relative_action_mask: list[bool] = field(default_factory=list)
    embodiment: str = "unknown"

    vocab_size: int = 252189
    pad_token_id: int = 0
    eos_token_id: int = 248044
    image_token_id: int = 248056
    vision_start_token_id: int = 248053
    vision_end_token_id: int = 248054
    state_token_id: int | None = None
    eov_token_id: int | None = None
    max_task_tokens: int = 200
    max_prompt_length: int = 1200

    text_hidden_size: int = 2048
    text_intermediate_size: int = 6144
    text_num_layers: int = 24
    text_num_heads: int = 8
    text_num_kv_heads: int = 2
    text_head_dim: int = 256
    text_layer_types: list[str] = field(default_factory=_default_layer_types)
    rope_theta: float = 10_000_000.0
    mrope_section: tuple[int, int, int] = (11, 11, 10)

    vision_depth: int = 24
    vision_hidden_size: int = 1024
    vision_intermediate_size: int = 4096
    vision_num_heads: int = 16
    vision_patch_size: int = 16
    vision_temporal_patch_size: int = 2
    vision_spatial_merge_size: int = 2

    expert_hidden_size: int = 1024
    expert_intermediate_size: int = 4096
    expert_num_layers: int = 24
    expert_num_heads: int = 8
    expert_num_kv_heads: int = 2
    expert_head_dim: int = 256

    num_inference_steps: int = 10
    flow_sig_min: float = 0.001
    flow_sampling: str = "beta"
    flow_beta_alpha: float = 1.5
    flow_beta_beta: float = 1.0
    num_flow_samples: int = 1
    flow_joint_training: bool = True
    fm_loss_weight: float = 1.0
    action_token_loss_weight: float = 0.0
    action_token_start_id: int | None = None
    action_token_end_id: int | None = None
    predict_cot: bool = False
    discrete_action: bool = False
    action_attend_cot: bool = False

    dtype: str = "bfloat16"

    optimizer_lr: float = 1e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 100_000
    scheduler_decay_lr: float = 1e-6

    # Conversion records these values rather than leaving runtime references to
    # Hydra configs or dataset-stat files.
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )
    tokenizer_subdir: str = "processor"
    action_tokenizer_subdir: str = "action_tokenizer"
    source_variant: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.chunk_size <= 0 or self.n_action_steps <= 0:
            raise ValueError("chunk_size and n_action_steps must be positive")
        if self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps cannot exceed chunk_size")
        if self.n_obs_steps <= 0:
            raise ValueError("n_obs_steps must be positive")
        if self.internal_action_dim <= 0 or self.internal_state_dim <= 0:
            raise ValueError("internal state and action dimensions must be positive")
        if self.max_task_tokens <= 0 or self.max_prompt_length <= 0:
            raise ValueError("prompt token limits must be positive")
        if len(self.text_layer_types) != self.text_num_layers:
            raise ValueError("text_layer_types must contain one entry per text layer")
        if any(kind not in {"linear_attention", "full_attention"} for kind in self.text_layer_types):
            raise ValueError("unsupported Qwen3.5 layer type")
        if self.flow_sampling not in {"beta", "uniform"}:
            raise ValueError("flow_sampling must be 'beta' or 'uniform'")
        if self.num_flow_samples <= 0:
            raise ValueError("num_flow_samples must be positive")
        if self.dtype not in {"bfloat16", "float32"}:
            raise ValueError("dtype must be 'bfloat16' or 'float32'")
        if (self.action_token_start_id is None) != (self.action_token_end_id is None):
            raise ValueError("action token range must define both start and end")
        action_dim = self.output_features.get(ACTION)
        if self.relative_action_mask and (
            action_dim is None or len(self.relative_action_mask) != action_dim.shape[-1]
        ):
            raise ValueError("relative_action_mask must match the physical action dimension")
        duplicate_cameras = set(self.camera_keys) & set(self.dummy_camera_keys)
        if duplicate_cameras:
            raise ValueError(f"cameras cannot be both real and dummy: {sorted(duplicate_cameras)}")
        known_cameras = set(self.camera_keys) | set(self.dummy_camera_keys)
        if self.camera_order and (
            set(self.camera_order) != known_cameras or len(self.camera_order) != len(known_cameras)
        ):
            raise ValueError("camera_order must contain every real and dummy camera exactly once")

    def validate_features(self) -> None:
        state = self.input_features.get(OBS_STATE)
        action = self.output_features.get(ACTION)
        if state is None or state.type is not FeatureType.STATE:
            raise ValueError(f"G0.5 requires a {OBS_STATE!r} state feature")
        if action is None or action.type is not FeatureType.ACTION:
            raise ValueError(f"G0.5 requires an {ACTION!r} action feature")
        if state.shape[-1] > self.internal_state_dim:
            raise ValueError("physical state dimension exceeds the checkpoint state dimension")
        if action.shape[-1] > self.internal_action_dim:
            raise ValueError("physical action dimension exceeds the checkpoint action dimension")

        visual_keys = {
            key for key, feature in self.input_features.items() if feature.type is FeatureType.VISUAL
        }
        missing = set(self.camera_keys) - visual_keys
        if missing:
            raise ValueError(f"missing configured G0.5 cameras: {sorted(missing)}")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list[int]:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
