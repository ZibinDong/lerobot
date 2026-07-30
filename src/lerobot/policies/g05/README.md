# G0.5

## Best Validated LIBERO Configuration

The best complete four-suite result currently uses the converted 100k checkpoint with strict FP32 inference, eager text/action attention, eager vision attention, and the default LIBERO episode lengths.

Local model artifact:

```text
/efm-vepfs/group-pretrain/dzb/lerobot/outputs/checkpoints/g05-libero-fmonly-step-100000
```

Checkpoint source SHA256:

```text
432f225c7647bdab77bfa2c2b397dc38cf8631bd70c92b85b261ff2817c1e795
```

Key policy settings:

```text
dtype=float32
use_amp=false
attn_implementation=eager
vision_attn_implementation=eager
NVIDIA_TF32_OVERRIDE=0
discrete_action=false
predict_cot=false
action_attend_cot=false
```

Evaluation protocol:

- 10 tasks per suite and 50 episodes per task
- `seed=0`
- 32-step action chunks, with 10 actions executed before replanning
- 10 flow-matching Euler steps
- Default LeRobot LIBERO episode lengths

## Results

| Suite          | Success rate | Successes | Episodes |
| -------------- | -----------: | --------: | -------: |
| libero_spatial |        97.8% |       489 |      500 |
| libero_object  |        99.8% |       499 |      500 |
| libero_goal    |        97.4% |       487 |      500 |
| libero_10      |        97.2% |       486 |      500 |
| **Overall**    |   **98.05%** |  **1961** | **2000** |

Evaluation outputs:

```text
outputs/eval/g05-libero-step100000-fp32-eager-4suites-50episodes-seed0/
```

## Reproduction

Run the following once for each suite by setting `SUITE` to `libero_spatial`, `libero_object`, `libero_goal`, or `libero_10`:

```bash
export NVIDIA_TF32_OVERRIDE=0
export SUITE=libero_spatial

uv run lerobot-eval \
    --policy.path=outputs/checkpoints/g05-libero-fmonly-step-100000 \
    --policy.device=cuda \
    --policy.dtype=float32 \
    --policy.use_amp=false \
    --policy.attn_implementation=eager \
    --policy.vision_attn_implementation=eager \
    --policy.discrete_action=false \
    --policy.predict_cot=false \
    --policy.action_attend_cot=false \
    --env.type=libero \
    --env.task="$SUITE" \
    --env.camera_name_mapping='{"agentview_image":"image","robot0_eye_in_hand_image":"wrist_image"}' \
    --env.observation_height=256 \
    --env.observation_width=256 \
    --eval.batch_size=50 \
    --eval.n_episodes=50 \
    --eval.use_async_envs=true \
    --seed=0 \
    --output_dir="outputs/eval/g05-step100000-fp32-eager-${SUITE}"
```
