# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0

import pytest
import torch

pytest.importorskip("transformers")

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.g05.action_tokenizer import (
    G05ActionCodecConfig,
    G05ActionCodecModel,
    G05ActionTokenizer,
)
from lerobot.policies.g05.configuration_g05 import G05Config
from lerobot.policies.g05.convert_g05_checkpoint import _action_tokens
from lerobot.policies.g05.modeling_g05 import (
    G05_ATTENTION_MASK,
    G05_IMAGES,
    G05_INPUT_IDS,
    G05Policy,
)
from lerobot.policies.g05.modular_g05 import G05_ACTION_DIM_PAD_MASK, G05_LABELS, G05_PREFIX_LENGTH
from lerobot.policies.g05.processor_g05 import (
    G05LiberoActionStep,
    G05LiberoObservationStep,
    G05PrepareInputsStep,
    G05RestoreActionStep,
    _apply_normalization,
)
from lerobot.types import TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE


def _features() -> tuple[dict, dict]:
    return (
        {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
            "observation.images.cam": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
        },
        {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
    )


def _tiny_config() -> G05Config:
    input_features, output_features = _features()
    return G05Config(
        input_features=input_features,
        output_features=output_features,
        device="cpu",
        camera_keys=["observation.images.cam"],
        image_size=(32, 32),
        internal_action_dim=4,
        internal_state_dim=4,
        action_indices=[1, 3],
        state_indices=[0, 1, 3],
        vocab_size=100,
        image_token_id=2,
        state_token_id=3,
        eov_token_id=4,
        eos_token_id=5,
        text_hidden_size=32,
        text_intermediate_size=64,
        text_num_layers=4,
        text_num_heads=2,
        text_num_kv_heads=1,
        text_head_dim=16,
        text_layer_types=["linear_attention"] * 3 + ["full_attention"],
        mrope_section=(2, 1, 1),
        vision_depth=1,
        vision_hidden_size=32,
        vision_intermediate_size=64,
        vision_num_heads=2,
        expert_hidden_size=32,
        expert_intermediate_size=64,
        expert_num_layers=4,
        expert_num_heads=2,
        expert_num_kv_heads=1,
        expert_head_dim=16,
        chunk_size=2,
        n_action_steps=1,
        num_inference_steps=2,
        dtype="float32",
    )


def test_config_delta_indices_and_validation() -> None:
    config = _tiny_config()
    assert config.observation_delta_indices == [0]
    assert config.action_delta_indices == [0, 1]
    config.validate_features()

    config.n_action_steps = 3
    with pytest.raises(ValueError, match="cannot exceed"):
        config.__post_init__()


def test_model_forward_backward_and_action_queue() -> None:
    policy = G05Policy(_tiny_config())
    batch = {
        G05_INPUT_IDS: torch.tensor([[2, 3, 6]]),
        G05_ATTENTION_MASK: torch.ones(1, 3, dtype=torch.bool),
        G05_IMAGES: torch.randn(1, 1, 1, 3, 32, 32),
        OBS_STATE: torch.tensor([[1.0, 2.0, 0.0, 3.0]]),
        ACTION: torch.randn(1, 2, 4),
    }
    loss, logs = policy(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(logs) == {
        "fm_loss",
        "action_token_loss",
        "cot_loss",
        "action_token_accuracy",
        "cot_accuracy",
    }
    noise = torch.zeros(1, 2, 4)
    assert policy.predict_action_chunk(batch, noise=noise).shape == (1, 2, 2)
    batch[G05_ACTION_DIM_PAD_MASK] = torch.tensor([[True, False, True, False]])
    padded_actions = policy.model.sample_actions(batch, noise=torch.ones_like(noise))
    torch.testing.assert_close(padded_actions[..., [0, 2]], torch.zeros(1, 2, 2))
    assert policy.select_action(batch, noise=noise).shape == (1, 2)
    policy.reset()
    assert not policy._action_queue


def test_discrete_action_loss_reuses_prefix_cache() -> None:
    config = _tiny_config()
    config.discrete_action = True
    config.action_token_loss_weight = 1.0
    config.action_token_start_id = 6
    config.action_token_end_id = 10
    policy = G05Policy(config)
    batch = {
        G05_INPUT_IDS: torch.tensor([[2, 3, 6, 7, 5]]),
        G05_ATTENTION_MASK: torch.ones(1, 5, dtype=torch.bool),
        G05_IMAGES: torch.randn(1, 1, 1, 3, 32, 32),
        G05_LABELS: torch.tensor([[-100, -100, -100, 7, 5]]),
        G05_PREFIX_LENGTH: 3,
        OBS_STATE: torch.tensor([[1.0, 2.0, 0.0, 3.0]]),
        ACTION: torch.randn(1, 2, 4),
    }

    loss, logs = policy(batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert logs["action_token_loss"] > 0


def test_processor_camera_layout_and_stepwise_action_stats(monkeypatch) -> None:
    class FakeTokenizer:
        @staticmethod
        def encode(text, add_special_tokens=False):
            del add_special_tokens
            return [10 + ord(char) % 10 for char in text]

    from transformers import AutoTokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: FakeTokenizer())
    step = G05PrepareInputsStep(
        tokenizer_path="artifact/processor",
        action_tokenizer_path="",
        camera_keys=["observation.images.cam"],
        dummy_camera_keys=["observation.images.dummy"],
        image_size=(32, 32),
        patch_size=16,
        spatial_merge_size=2,
        n_obs_steps=1,
        internal_state_dim=4,
        internal_action_dim=4,
        state_indices=[0, 1, 3],
        action_indices=[1, 3],
        state_norm_low=[],
        state_norm_high=[],
        action_norm_low=[[-1.0, -2.0], [-3.0, -4.0]],
        action_norm_high=[[1.0, 2.0], [3.0, 4.0]],
        state_normalization=[],
        action_normalization=[],
        embodiment="test",
        max_task_tokens=200,
        max_prompt_length=512,
        image_token_id=2,
        vision_start_token_id=7,
        vision_end_token_id=8,
        state_token_id=3,
        eov_token_id=4,
        pad_token_id=0,
        eos_token_id=5,
    )
    transition = {
        TransitionKey.OBSERVATION: {
            OBS_STATE: torch.tensor([[1.0, 2.0, 3.0]]),
            "observation.images.cam": torch.zeros(1, 3, 24, 24),
        },
        TransitionKey.ACTION: torch.zeros(1, 2, 2),
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["test task"]},
    }
    output = step(transition)
    complementary = output[TransitionKey.COMPLEMENTARY_DATA]
    assert output[TransitionKey.OBSERVATION][OBS_STATE].shape == (1, 4)
    assert output[TransitionKey.ACTION].shape == (1, 2, 4)
    assert complementary[G05_IMAGES].shape == (1, 2, 1, 3, 32, 32)
    assert complementary[G05_INPUT_IDS].eq(2).sum() == 2
    torch.testing.assert_close(
        complementary[G05_ACTION_DIM_PAD_MASK], torch.tensor([[True, False, True, False]])
    )

    ids, mask = step._prompt_ids(["a", "longer task"], num_images=2)
    assert not mask[0, 0]
    assert mask[1, 0]
    assert ids[0, 0] == step.pad_token_id

    restore = G05RestoreActionStep(
        action_norm_low=[[-1.0, -2.0], [-3.0, -4.0]],
        action_norm_high=[[1.0, 2.0], [3.0, 4.0]],
    )
    first = restore({TransitionKey.ACTION: torch.zeros(2)})[TransitionKey.ACTION]
    second = restore({TransitionKey.ACTION: torch.zeros(2)})[TransitionKey.ACTION]
    torch.testing.assert_close(first, torch.zeros(2))
    torch.testing.assert_close(second, torch.zeros(2))
    restore.reset()


def test_mrope_positions_use_image_grid_and_ignore_left_padding() -> None:
    config = _tiny_config()
    config.image_size = (64, 64)
    policy = G05Policy(config)
    input_ids = torch.tensor([[0, 7, 2, 2, 2, 2, 8]])
    attention_mask = input_ids.ne(0)

    positions = policy.model.build_mrope_position_ids(input_ids, attention_mask)

    torch.testing.assert_close(positions[:, 0, 0], torch.zeros(3, dtype=torch.long))
    torch.testing.assert_close(positions[:, 0, 1], torch.zeros(3, dtype=torch.long))
    torch.testing.assert_close(
        positions[:, 0, 2:6],
        torch.tensor([[1, 1, 1, 1], [1, 1, 2, 2], [1, 2, 1, 2]]),
    )
    torch.testing.assert_close(positions[:, 0, 6], torch.full((3,), 3, dtype=torch.long))


def test_action_token_order_matches_g05_vocab_layout() -> None:
    model_config = {
        "AT_CONFIG": {
            "parts_meta": {
                "left_control": 9,
                "left_gripper": 1,
                "right_control": 9,
                "right_gripper": 1,
            },
            "rule_based_key_patterns": ["gripper"],
            "use_group_markers": True,
            "model_arch": {"codebook_size": 4096, "n_codebooks": 4},
        }
    }
    tokens = _action_tokens(model_config)
    assert len(tokens) == 4106
    assert tokens[0] == "<action0000>"
    assert tokens[4096:4100] == [
        "<left_control_0>",
        "<right_control_0>",
        "<left_control_1>",
        "<right_control_1>",
    ]
    assert tokens[-2:] == ["<left_gripper>", "<right_gripper>"]


def test_action_codec_grouped_token_roundtrip() -> None:
    codec_config = G05ActionCodecConfig(
        max_component_dim=3,
        horizon=4,
        horizon_patch_size=2,
        conv_in_action_kernel=2,
        encoder_channels=8,
        latent_dim=4,
        c_mults=[1],
        strides=[[1, 1]],
        transformer_depths=[1],
        num_heads=1,
        dim_heads=32,
        use_block_dct=True,
        block_dct_block_size=2,
        n_codebooks=1,
        codebook_size=16,
        codebook_dim=2,
        parts_meta={"control": 2, "gripper": 1},
        num_residuals=1,
    )
    tokenizer = G05ActionTokenizer(G05ActionCodecModel(codec_config))
    action = torch.tensor([[[0.1, 0.2, -1.0], [0.3, 0.4, -1.0], [0.5, 0.6, 1.0], [0.7, 0.8, 1.0]]])

    token_ids = tokenizer.encode_action_indices(action)
    decoded = tokenizer.decode_action_indices(token_ids)

    assert token_ids.shape == (1, 7)
    assert token_ids[0, 0] == tokenizer.marker_indices["<control_0>"]
    assert token_ids[0, -2] == tokenizer.marker_indices["<gripper>"]
    torch.testing.assert_close(decoded[..., -1], action[..., -1])


def test_legacy_normalization_modes_and_stepwise_restore() -> None:
    specs = [
        {
            "mode": "z-score-tail",
            "stats": {
                "mean": [0.0],
                "std": [2.0],
                "q01": [-1.0],
                "q99": [1.0],
            },
        },
        {"mode": "q01/q99", "stats": {"q01": [0.0], "q99": [4.0]}},
    ]
    values = torch.tensor([[[-1.0, 1.0], [1.0, 3.0]]])
    normalized = _apply_normalization(values, specs, inverse=False)
    torch.testing.assert_close(normalized, torch.tensor([[[-0.5, -0.5], [0.5, 0.5]]]))
    torch.testing.assert_close(_apply_normalization(normalized, specs, inverse=True), values)

    restore = G05RestoreActionStep(
        action_norm_low=[],
        action_norm_high=[],
        action_horizon=2,
        action_normalization=[
            {
                "mode": "q01/q99",
                "stats": {"q01": [[0.0], [2.0]], "q99": [[2.0], [6.0]]},
            }
        ],
    )
    first = restore({TransitionKey.ACTION: torch.zeros(1)})[TransitionKey.ACTION]
    second = restore({TransitionKey.ACTION: torch.zeros(1)})[TransitionKey.ACTION]
    third = restore({TransitionKey.ACTION: torch.zeros(1)})[TransitionKey.ACTION]
    torch.testing.assert_close(first, torch.tensor([1.0]))
    torch.testing.assert_close(second, torch.tensor([4.0]))
    torch.testing.assert_close(third, first)


def test_libero_boundary_matches_g05_state_and_gripper_contract() -> None:
    observation = {
        "observation.images.image": torch.arange(12).reshape(1, 1, 3, 4),
        "observation.robot_state": {
            "eef": {
                "pos": torch.tensor([[1.0, 2.0, 3.0]]),
                "quat": torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
            },
            "gripper": {"qpos": torch.tensor([[0.03, -0.03]])},
        },
    }
    processed = G05LiberoObservationStep().observation(observation)

    torch.testing.assert_close(
        processed[OBS_STATE], torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.03]])
    )
    torch.testing.assert_close(
        processed["observation.images.image"],
        torch.flip(observation["observation.images.image"], dims=(-2, -1)),
    )

    transition = {TransitionKey.ACTION: torch.tensor([[0.0, 0.49], [0.0, 0.51]])}
    action = G05LiberoActionStep()(transition)[TransitionKey.ACTION]
    torch.testing.assert_close(action[..., -1], torch.tensor([1.0, -1.0]))
