# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea
# Modified for LeRobot in 2026.

"""Convert an official G0.5 checkpoint into a self-contained LeRobot artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import save_torch_state_dict
from torch import Tensor
from transformers import AutoTokenizer

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE

from .action_tokenizer import G05ActionCodecConfig, G05ActionCodecModel
from .configuration_g05 import G05Config
from .modeling_g05 import G05Policy
from .processor_g05 import make_g05_pre_post_processors


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise ImportError("checkpoint conversion requires PyYAML") from error
    with path.open() as stream:
        return yaml.safe_load(stream)


def _checkpoint_path(variant_dir: Path) -> Path:
    candidates = [variant_dir / "model.pt", variant_dir / "checkpoints" / "model_state_dict.pt"]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one G0.5 model checkpoint under {variant_dir}")
    return matches[0]


def _model_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if not isinstance(checkpoint, dict) or not all(
        isinstance(value, torch.Tensor) for value in checkpoint.values()
    ):
        raise ValueError("legacy checkpoint does not contain a tensor-only model_state_dict")
    return checkpoint


def _group_offsets(parts_meta: dict[str, int]) -> dict[str, int]:
    offsets: dict[str, int] = {}
    offset = 0
    for name, width in parts_meta.items():
        offsets[name] = offset
        offset += int(width)
    return offsets


def _layout_indices(
    shape_items: list[dict[str, Any]], merge_spec: dict[str, list[str]], parts_meta: dict[str, int]
) -> list[int]:
    offsets = _group_offsets(parts_meta)
    indices: list[int] = []
    for item in shape_items:
        source_name = item["key"]
        group = next((name for name, members in merge_spec.items() if source_name in members), None)
        if group is None:
            raise ValueError(f"shape key {source_name!r} is absent from the merger spec")
        width = int(item["shape"] if isinstance(item["shape"], int) else item["shape"][-1])
        if width > int(parts_meta[group]):
            raise ValueError(f"shape key {source_name!r} does not fit group {group!r}")
        indices.extend(range(offsets[group], offsets[group] + width))
    return indices


def _select_embodiment(hydra: dict[str, Any], stats: dict[str, Any], requested: str | None) -> str:
    embodiment = requested or hydra.get("eval_embodiment")
    if embodiment is None and len(stats) == 1:
        embodiment = next(iter(stats))
    if embodiment is None:
        raise ValueError("this checkpoint contains multiple embodiments; pass --embodiment explicitly")
    if embodiment not in stats:
        raise ValueError(f"embodiment {embodiment!r} has no published statistics; available: {sorted(stats)}")
    return embodiment


def _droid_shape_meta() -> dict[str, list[dict[str, Any]]]:
    """Schema published with G0.5-DROID, whose Hydra sidecar retained an oc.load reference."""
    return {
        "action": [
            {"key": "right_arm", "lerobot_key": "action", "shape": 7},
            {"key": "right_gripper", "lerobot_key": "action", "shape": 1},
        ],
        "state": [
            {"key": "right_arm", "lerobot_key": "observation.state", "shape": 7},
            {"key": "right_gripper", "lerobot_key": "observation.state", "shape": 1},
        ],
        "images": [
            {
                "key": "exterior_rgb",
                "lerobot_key": "observation.images.exterior_1_left",
                "camera_type": "exterior",
            },
            {
                "key": "right_wrist_rgb",
                "lerobot_key": "observation.images.wrist_left",
                "camera_type": "wrist_left",
            },
            {"key": "left_wrist_rgb", "camera_type": "wrist_right", "dummy": True},
        ],
    }


def _embodiment_metadata(hydra: dict[str, Any], embodiment: str) -> tuple[dict[str, Any], dict[str, Any]]:
    data = hydra["data"]
    if isinstance(data, dict):
        processor = data.get("processors", {}).get(embodiment, {})
        dataset = data.get("embodiment_datasets", {}).get(embodiment, {})
        shape_meta = processor.get("shape_meta") or dataset.get("shape_meta")
        if shape_meta is not None:
            return shape_meta, processor
    if embodiment == "Droid_Franka":
        return _droid_shape_meta(), {
            "norm_default_mode": "q01/q99",
            "action_state_transforms": [
                {
                    "_target_": "g05.data_processor.transforms.relative_action.RelativeJointTransform",
                    "keys": ["right_arm"],
                }
            ],
        }
    raise ValueError(f"the published metadata has no shape schema for {embodiment!r}")


def _merge_spec(processor: dict[str, Any], parts_meta: dict[str, int]) -> dict[str, list[str]]:
    configured = processor["action_state_merger"].get("merge_spec")
    if isinstance(configured, dict):
        return configured
    # Some post-training sidecars retained an oc.load expression. The action
    # tokenizer's canonical group names make the intended mapping unambiguous.
    spec = {name: [name] for name in parts_meta}
    for side in ("left", "right"):
        control = f"{side}_control"
        if control in spec:
            spec[control].extend([f"{side}_arm", f"{side}_ee_pose"])
    return spec


def _relative_action_mask(
    processor: dict[str, Any], action_items: list[dict[str, Any]], state_items: list[dict[str, Any]]
) -> list[bool]:
    relative_keys: set[str] = set()
    for transform in processor.get("action_state_transforms", []):
        if str(transform.get("_target_", "")).endswith("RelativeJointTransform"):
            relative_keys.update(transform["keys"])
    if not relative_keys:
        return []
    action_layout = [(item["key"], int(item["shape"])) for item in action_items]
    state_layout = [(item["key"], int(item["shape"])) for item in state_items]
    if action_layout != state_layout:
        raise ValueError("relative actions require matching physical action and state layouts")
    unknown = relative_keys - {name for name, _ in action_layout}
    if unknown:
        raise ValueError(f"relative-action keys are absent from the shape schema: {sorted(unknown)}")
    return [name in relative_keys for name, width in action_layout for _ in range(width)]


def _normalization_specs(
    stats: dict[str, Any],
    shape_items: list[dict[str, Any]],
    *,
    stepwise: bool,
    default_mode: str,
    exception_modes: dict[str, str],
) -> list[dict[str, Any]]:
    prefix = "stepwise_" if stepwise else "global_"
    specs: list[dict[str, Any]] = []
    for item in shape_items:
        mode = exception_modes.get(item["key"], default_mode)
        if mode not in {"q01/q99", "z-score", "z-score-tail"}:
            raise ValueError(f"normalization mode {mode!r} for {item['key']!r} is not supported")
        item_stats = stats[item["key"]]
        names = {"q01", "q99"} if mode == "q01/q99" else {"mean", "std"}
        if mode.endswith("-tail"):
            names.update(("q01", "q99"))
        selected = {name: item_stats[prefix + name] for name in sorted(names)}
        specs.append({"mode": mode, "stats": selected})
    return specs


def _action_tokens(model_config: dict[str, Any]) -> list[str]:
    tokenizer_config = model_config["AT_CONFIG"]
    architecture = tokenizer_config["model_arch"]
    codebook_size = int(architecture["codebook_size"])
    tokens = [f"<action{index:04d}>" for index in range(codebook_size)]
    if tokenizer_config.get("use_group_markers", False):
        parts = tokenizer_config["parts_meta"]
        rule_patterns = tuple(tokenizer_config.get("rule_based_key_patterns", []))
        rule = [name for name in parts if any(pattern in name for pattern in rule_patterns)]
        learned = [name for name in parts if name not in rule]
        residuals = int(architecture["n_codebooks"])
        tokens += [f"<{name}_{level}>" for level in range(residuals) for name in learned]
        tokens += [f"<{name}>" for name in rule]
    return tokens


def _save_action_tokenizer(
    checkpoint_path: Path,
    tokenizer_config: dict[str, Any],
    output_dir: Path,
) -> None:
    """Convert the legacy pickled codec into a standalone HF model directory."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
    source = checkpoint.get("model_state_dict", checkpoint)
    source = {key.removeprefix("model."): value.contiguous() for key, value in source.items()}
    frontend_fields = {
        name: tokenizer_config[name]
        for name in (
            "parts_meta",
            "rule_based_key_patterns",
            "rule_based_min_block_len",
            "rule_based_binarize_threshold",
            "num_residuals",
            "use_group_markers",
            "absent_key_fill_value",
        )
        if name in tokenizer_config
    }
    config = G05ActionCodecConfig(**tokenizer_config["model_arch"], **frontend_fields)
    with torch.device("meta"):
        expected = G05ActionCodecModel(config).state_dict()
    missing = set(expected) - set(source)
    unexpected = set(source) - set(expected)
    bad_shapes = {
        key: (tuple(source[key].shape), tuple(expected[key].shape))
        for key in set(expected) & set(source)
        if source[key].shape != expected[key].shape
    }
    if missing or unexpected or bad_shapes:
        raise ValueError(
            f"ActionCodec weight mapping failed: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}, bad_shapes={bad_shapes}"
        )
    output_dir.mkdir(parents=True)
    config.save_pretrained(output_dir)
    save_torch_state_dict(source, output_dir, max_shard_size="5GB")


def _build_config(
    hydra: dict[str, Any],
    stats: dict[str, Any],
    state: dict[str, Tensor],
    n_action_steps: int,
    embodiment: str | None = None,
) -> tuple[G05Config, list[str]]:
    model = hydra["model"]
    architecture = model["model_arch"]
    processor = model["processor"]
    embodiment = _select_embodiment(hydra, stats, embodiment)
    shape_meta, embodiment_processor = _embodiment_metadata(hydra, embodiment)
    action_items, state_items, image_items = (
        shape_meta["action"],
        shape_meta["state"],
        shape_meta["images"],
    )
    tokenizer_config = architecture["AT_CONFIG"]
    if not isinstance(tokenizer_config, dict):
        tokenizer_config = hydra["tokenizer"]["vq_config"]
    parts_meta = tokenizer_config["parts_meta"]
    merge_spec = _merge_spec(processor, parts_meta)
    action_indices = _layout_indices(action_items, merge_spec, parts_meta)
    state_indices = _layout_indices(state_items, merge_spec, parts_meta)
    physical_action_dim = len(action_indices)
    physical_state_dim = len(state_indices)
    embodiment_stats = stats[embodiment]
    # The published policy is served through model.processor, whose normalization
    # contract can intentionally differ from the dataset's training transforms.
    # Reusing the dataset processor here changes both proprio inputs and decoded
    # actions (LIBERO is q01/q99 at inference, while its dataset config is z-score).
    default_mode = processor.get("norm_default_mode", "q01/q99")
    exception_mode = processor.get("norm_exception_mode", {})
    action_normalization = _normalization_specs(
        embodiment_stats["action"],
        action_items,
        stepwise=bool(processor["use_stepwise_action_norm"]),
        default_mode=default_mode,
        exception_modes=exception_mode.get("action", {}),
    )
    state_normalization = _normalization_specs(
        embodiment_stats["state"],
        state_items,
        stepwise=False,
        default_mode=default_mode,
        exception_modes=exception_mode.get("state", {}),
    )

    image_size = tuple(next(iter(processor["camera_size_config"].values())))
    camera_keys: list[str] = []
    dummy_camera_keys: list[str] = []
    camera_order: list[str] = []
    for index, item in enumerate(image_items):
        if item.get("dummy", False):
            key = f"observation.images.g05_dummy_{index}"
            dummy_camera_keys.append(key)
        else:
            key = item["lerobot_key"]
            camera_keys.append(key)
        camera_order.append(key)
    output_camera_count = int(processor["num_output_cameras"])
    while len(camera_order) < output_camera_count:
        key = f"observation.images.g05_dummy_{len(camera_order)}"
        dummy_camera_keys.append(key)
        camera_order.append(key)
    if len(camera_order) != output_camera_count:
        raise ValueError("shape metadata contains more cameras than the model processor accepts")
    vocab_size = int(state["model.vlm.input_proj.weight"].shape[0])
    action_tokens = _action_tokens({**architecture, "AT_CONFIG": tokenizer_config})
    base_tokenizer_size = vocab_size - len(action_tokens) - 2
    if base_tokenizer_size <= 0:
        raise ValueError("invalid checkpoint vocabulary layout")
    eov_token_id = base_tokenizer_size + len(action_tokens)
    state_token_id = eov_token_id + 1
    if state_token_id + 1 != vocab_size:
        raise ValueError("checkpoint vocabulary is not action tokens + EOV + state")

    vlm, expert, vision, fm = (
        architecture["vlm"],
        architecture["action_expert"],
        architecture["vision"],
        architecture["fm"],
    )
    optimizer_lr = float(model.get("learning_rate") or 1e-5)
    supported_fm_contract = {
        "time_convention": "pi_convention",
        "padding_action_weight": 0.0,
        "zero_pad_action_target": False,
        "action_causal": False,
        "final_action_clip_value": None,
    }
    for name, expected_value in supported_fm_contract.items():
        if fm.get(name) != expected_value:
            raise ValueError(
                f"unsupported G0.5 flow setting {name}={fm.get(name)!r}; expected {expected_value!r}"
            )
    if architecture.get("ae_vlm_condition_mode") != "cross_attn_only":
        raise ValueError("only the published cross_attn_only action conditioning is supported")
    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(physical_state_dim,)),
        **{key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, *image_size)) for key in camera_keys},
    }
    config = G05Config(
        input_features=input_features,
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(physical_action_dim,))},
        chunk_size=(
            int(architecture["horizon_steps"]) if isinstance(architecture["horizon_steps"], int) else 32
        ),
        n_action_steps=n_action_steps,
        n_obs_steps=(
            int(architecture["num_obs_steps"]) if isinstance(architecture["num_obs_steps"], int) else 1
        ),
        image_size=image_size,
        camera_keys=camera_keys,
        dummy_camera_keys=dummy_camera_keys,
        camera_order=camera_order,
        internal_action_dim=int(architecture["action_dim"]),
        internal_state_dim=int(architecture["proprio_dim"]),
        action_indices=action_indices,
        state_indices=state_indices,
        action_normalization=action_normalization,
        state_normalization=state_normalization,
        relative_action_mask=_relative_action_mask(embodiment_processor, action_items, state_items),
        embodiment=embodiment,
        vocab_size=vocab_size,
        pad_token_id=int(architecture["pad_token_id"]),
        eos_token_id=int(architecture["eos_token_id"]),
        image_token_id=int(architecture["image_token_index"]),
        state_token_id=state_token_id,
        eov_token_id=eov_token_id,
        max_prompt_length=int(architecture["max_chunk_token_length"]),
        text_hidden_size=int(vlm["hidden_size"]),
        text_intermediate_size=int(vlm["intermediate_size"]),
        text_num_layers=int(vlm["num_hidden_layers"]),
        text_num_heads=int(vlm["num_attention_heads"]),
        text_num_kv_heads=int(vlm["num_key_value_heads"]),
        text_head_dim=int(vlm["head_dim"]),
        text_layer_types=list(vlm["layer_types"]),
        rope_theta=float(vlm["rope_parameters"]["rope_theta"]),
        mrope_section=tuple(vlm["rope_parameters"]["mrope_section"]),
        vision_depth=int(vision["depth"]),
        vision_hidden_size=int(vision["hidden_size"]),
        vision_intermediate_size=int(vision["intermediate_size"]),
        vision_num_heads=int(vision["num_heads"]),
        vision_patch_size=int(vision["patch_size"]),
        vision_temporal_patch_size=int(vision["temporal_patch_size"]),
        vision_spatial_merge_size=int(vision["spatial_merge_size"]),
        expert_hidden_size=int(expert["hidden_size"]),
        expert_intermediate_size=int(expert["intermediate_size"]),
        expert_num_layers=int(expert["num_hidden_layers"]),
        expert_num_heads=int(expert["num_attention_heads"]),
        expert_num_kv_heads=int(expert["num_key_value_heads"]),
        expert_head_dim=int(expert["head_dim"]),
        num_inference_steps=int(fm["num_inference_steps"]),
        flow_sig_min=float(fm["flow_sig_min"]),
        flow_sampling=str(fm["flow_sampling"]),
        num_flow_samples=int(fm.get("num_flow_samples", 1)),
        flow_joint_training=bool(fm["joint_training"]),
        fm_loss_weight=float(fm["fm_weight"]),
        action_token_loss_weight=float(architecture["ar"]["ce_weight"]),
        action_token_start_id=base_tokenizer_size,
        action_token_end_id=base_tokenizer_size + len(action_tokens),
        predict_cot=bool(architecture["predict_cot"]),
        discrete_action=bool(architecture["discrete_action"]),
        action_attend_cot=bool(architecture["action_attend_cot"]),
        optimizer_lr=optimizer_lr,
        optimizer_betas=tuple(model.get("betas") or (0.9, 0.95)),
        optimizer_weight_decay=float(model.get("weight_decay") or 0.0),
        optimizer_grad_clip_norm=float(model.get("max_grad_norm") or 1.0),
        scheduler_warmup_steps=int(model.get("warmup_steps") or 0),
        scheduler_decay_steps=int(model.get("max_steps") or 100_000),
        scheduler_decay_lr=optimizer_lr * float(model.get("lr_min_ratio") or 0.1),
        source_variant=None,
    )
    return config, action_tokens


def _remap_weights(source: dict[str, Tensor]) -> dict[str, Tensor]:
    remapped: dict[str, Tensor] = {}
    for key, value in source.items():
        target = key
        target = target.replace("model.vlm.input_proj.", "model.vlm.embed_tokens.")
        target = target.replace("model.vlm.output_proj.", "model.output_proj.")
        target = target.replace("model.proprio_embedder.mlp.", "model.proprio_embedder.")
        if target in remapped:
            raise ValueError(f"multiple source weights map to {target}")
        remapped[target] = value.contiguous()
    return remapped


def convert(args: argparse.Namespace) -> None:
    variant_dir = args.legacy_root / args.variant
    metadata_dir = args.metadata_root / args.variant
    hydra = _load_yaml(metadata_dir / ".hydra" / "config.yaml")
    with (metadata_dir / "dataset_stats.json").open() as stream:
        stats = json.load(stream)
    checkpoint_path = _checkpoint_path(variant_dir)
    source_state = _model_state(checkpoint_path)
    config, action_tokens = _build_config(hydra, stats, source_state, args.n_action_steps, args.embodiment)
    config.source_variant = args.variant

    base_tokenizer = AutoTokenizer.from_pretrained(args.processor_dir, local_files_only=True)
    expected_base_size = config.vocab_size - len(action_tokens) - 2
    if len(base_tokenizer) != expected_base_size:
        raise ValueError(
            f"base tokenizer has {len(base_tokenizer)} tokens; checkpoint expects {expected_base_size}"
        )
    base_tokenizer.add_tokens(action_tokens + ["<EOV>", "<state>"], special_tokens=True)
    if len(base_tokenizer) != config.vocab_size:
        raise ValueError("reconstructed tokenizer vocabulary does not match model embeddings")

    remapped = _remap_weights(source_state)
    with torch.device("meta"):
        expected = G05Policy(config).state_dict()
    missing = set(expected) - set(remapped)
    unexpected = set(remapped) - set(expected)
    bad_shapes = {
        key: (tuple(remapped[key].shape), tuple(expected[key].shape))
        for key in set(expected) & set(remapped)
        if remapped[key].shape != expected[key].shape
    }
    # Tied language-model output weights are serialized once by LeRobot.
    missing.discard("model.output_proj.weight")
    if missing or unexpected or bad_shapes:
        raise ValueError(
            f"weight mapping failed: missing={sorted(missing)[:20]}, "
            f"unexpected={sorted(unexpected)[:20]}, bad_shapes={bad_shapes}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    tokenizer_config = hydra["model"]["model_arch"]["AT_CONFIG"]
    if not isinstance(tokenizer_config, dict):
        tokenizer_config = hydra["tokenizer"]["vq_config"]
    action_tokenizer_checkpoint = args.action_tokenizer_checkpoint or args.legacy_root / "action_tokenizer.pt"
    if not action_tokenizer_checkpoint.is_file():
        raise FileNotFoundError(f"missing G0.5 ActionCodec checkpoint: {action_tokenizer_checkpoint}")
    _save_action_tokenizer(
        action_tokenizer_checkpoint,
        tokenizer_config,
        args.output_dir / config.action_tokenizer_subdir,
    )
    processor_output = args.output_dir / config.tokenizer_subdir
    base_tokenizer.save_pretrained(processor_output)
    for filename in ("config.json", "preprocessor_config.json", "video_preprocessor_config.json"):
        source = args.processor_dir / filename
        if source.is_file():
            shutil.copy2(source, processor_output / filename)
    config.save_pretrained(args.output_dir)
    preprocessor, postprocessor = make_g05_pre_post_processors(config, tokenizer_path=processor_output)
    preprocessor.save_pretrained(args.output_dir, config_filename="policy_preprocessor.json")
    postprocessor.save_pretrained(args.output_dir, config_filename="policy_postprocessor.json")
    save_torch_state_dict(
        remapped,
        args.output_dir,
        max_shard_size="20GB",
        shared_tensors_to_discard=["model.output_proj.weight"],
    )

    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    provenance = {
        "format": "lerobot-g05-v1",
        "source_variant": args.variant,
        "embodiment": config.embodiment,
        "source_sha256": digest.hexdigest(),
        "source_filename": checkpoint_path.name,
        "action_tokenizer_filename": action_tokenizer_checkpoint.name,
        "self_contained": True,
    }
    (args.output_dir / "conversion.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--processor-dir", type=Path, required=True)
    parser.add_argument("--action-tokenizer-checkpoint", type=Path)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--embodiment")
    parser.add_argument("--n-action-steps", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    convert(parser.parse_args())


if __name__ == "__main__":
    main()
