# G0.5

本文档记录 LeRobot G0.5 集成、旧格式 checkpoint 转换、SO100
部署方法，以及 2026-07-30 至 2026-07-31 实机测试中已经解决和仍未解决的问题。

## 当前状态

当前代码已经完成以下工作：

- 可将原始 GalaxeaFM/G0.5 checkpoint 转换为自包含的 LeRobot artifact。
- converter 根据 checkpoint 的
  `continuous_action`、`discrete_action` 和 `return_continuous_action`
  自动选择推理 head，不再写死为 continuous/flow matching。
- `predict_cot` 是运行时是否生成 CoT 的统一开关。
- 支持 checkpoint 中保存的 CoT prompt 和 AR sampling 参数。
- 支持 SO100 旧部署代码使用的关节坐标系变换。
- converted artifact 可被 `strict=True` 加载，并可完成离线 dummy inference。
- G0.5 针对性单元测试和 Ruff 检查已通过。

截至 2026-07-31，SO100 的 AR 离散动作部署已经完成以下验证：

- 机械臂、两路相机和旧校准文件均已成功接入。
- ActionCodec parts 顺序错误已经修复；转换权重可正确映射 SO100 的六维动作。
- 使用官方 `lerobot-rollout`、greedy AR、10 度安全限幅和
  `torch.compile` 的 120 秒实机命令能够成功启动并执行 rollout。
- 下面记录的命令是当前实机验证基线；旧的 3 度、20 秒测试仅保留为历史记录。

## 2026-07-31 实机验证命令

以下命令已在本机成功完成模型加载、`torch.compile` 推理和 SO100 实机
rollout。`return_to_initial_position=true` 会在 rollout 结束后返回命令启动时
记录的关节位置，而不是自动返回 G0.5 的 home pose。

```bash
cd /home/galaxea/zibin.dong/lerobot

uv run lerobot-rollout \
  --strategy.type=base \
  --policy.path=/home/galaxea/zibin.dong/lerobot/outputs/checkpoints/g05-so100-opensource-16node-resampled30fps-lerobot \
  --policy.predict_cot=true \
  --policy.discrete_action=true \
  --policy.continuous_action=false \
  --policy.ar_do_sample=false \
  --robot.type=so100_follower \
  --robot.id=g05_so100_follower \
  --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5970073357-if00 \
  --robot.use_degrees=true \
  --robot.max_relative_target=10 \
  --robot.disable_torque_on_disconnect=false \
  --robot.cameras="{exterior_rgb: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: 200}, right_wrist_rgb: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30, fourcc: MJPG, backend: 200}}" \
  --task="pick up the red block and put it into the blue bowl" \
  --fps=30 \
  --duration=120 \
  --return_to_initial_position=true \
  --device=cuda \
  --use_torch_compile=true
```

首次动作块会包含 TorchInductor 编译开销，后续推理复用编译缓存。

## 本次代码更新

### Checkpoint 自动识别

`convert_g05_checkpoint.py` 现在从原始 Hydra 配置读取真实的模型契约：

- `continuous_action`
- `discrete_action`
- `return_continuous_action`
- `predict_cot`
- `action_attend_cot`
- AR `do_sample`
- AR `temperature`
- AR `top_k`
- AR `top_p`
- AR `repetition_penalty`
- AR `no_repeat_ngram_size`

推理 head 的选择规则如下：

| Checkpoint 配置 | LeRobot `inference_action_head` |
| --- | --- |
| 只有 continuous | `fm` |
| 只有 discrete | `ar` |
| continuous + discrete，`return_continuous_action=true` | `fm` |
| continuous + discrete，`return_continuous_action=false` | `ar` |

如果 checkpoint 两种 action 都未启用，converter 会直接报错，而不是猜测。

### CoT 推理

converter 根据 checkpoint 的 `samples_builder` 恢复部署时使用的 CoT
instruction。目前支持：

- `SubtaskCoTBuilder`
- `TaskAsSubtaskCoTBuilder`
- `FutureSubtaskCoTBuilder`
- `BBoxCoTBuilder`
- `BBoxSubtaskCoTBuilder`
- `Trace2DCoTBuilder`
- `SubtaskActionHintCoTBuilder`

`predict_cot=true` 时：

1. processor 构造原始 checkpoint 对应的 CoT prompt。
2. VLM 先自回归生成 CoT。
3. 离散 action 模式消费 EOV transition。
4. 再生成连续的一段 ActionCodec token。

`predict_cot=false` 时，processor 直接构造 `Action: <EOV>` 前缀，不生成
CoT。CLI override 会同时更新 policy config 和 serialized processor，避免二者
不一致：

```bash
--policy.predict_cot=false
```

### SO100 坐标系

原始 SO100 client 在机械臂物理坐标和训练坐标之间使用：

```text
signs   = [1, -1, 1, 1, 1, 1]
offsets = [0, 90, 90, 0, 0, 0]
```

状态进入模型前：

```text
model_state = signs * arm_state + offsets
```

动作送给机械臂前：

```text
arm_action = signs * (model_action - offsets)
```

converter 现在为六维 `so100`/`so101` artifact 自动写入这组参数。processor
新增：

- `G05StateFrameTransformStep`
- `G05ActionFrameTransformStep`

因此使用 converted artifact 时，不要在外部 client 再做一次相同变换。

### 测试覆盖

新增或扩展的测试覆盖：

- CoT builder 到 prompt 的映射。
- checkpoint action flags 到推理 head 的映射。
- SO100 坐标变换可逆性。
- `predict_cot` prompt 与 direct-action prompt。
- AR 推理中 CoT、EOV transition 和 action token 的顺序。
- AR sampling 配置。
- serialized processor 在 CLI override 后与 policy config 保持一致。

涉及的主要文件：

```text
src/lerobot/policies/g05/configuration_g05.py
src/lerobot/policies/g05/convert_g05_checkpoint.py
src/lerobot/policies/g05/modeling_g05.py
src/lerobot/policies/g05/processor_g05.py
tests/policies/g05/test_g05.py
docs/source/g05.mdx
```

## 环境安装

优先使用 `uv`，不要使用原始 `pip`：

```bash
cd /home/galaxea/zibin.dong/lerobot
uv venv
source .venv/bin/activate
uv pip install -e ".[all]"
```

不指定 `-i` 时使用 uv/Python 的正常默认源。若 shell 中曾配置镜像，先检查：

```bash
env | grep -E '^(UV|PIP).*INDEX'
```

如果输出中存在 `UV_INDEX_URL`、`UV_DEFAULT_INDEX`、`PIP_INDEX_URL` 或
`PIP_EXTRA_INDEX_URL`，先删除对应的 shell 配置或在当前终端 `unset`，再执行
安装。`uv pip` 没有 `config list` 子命令。

本次安装中遇到过两个环境问题：

1. 清华镜像一度无法解析 `num2words>=0.5.14,<0.6.0`，导致
   `lerobot[all]` 被判定为不可满足。随后重试可以正常解析 315 个包，说明并非
   G0.5 或 LeRobot 的依赖声明错误。
2. `nvidia-cublas-cu12` 解压时报 `No space left on device`。真正写满的是
   `/home/galaxea/.cache/uv` 所在文件系统；已删除的文件仍在桌面回收站中，
   清空回收站后空间才真正释放。

## 转换旧 SO100 checkpoint

### 本次使用的路径

原始 checkpoint：

```text
/home/galaxea/galaxeafm-opensource/checkpoints/so100-so100-opensource-16node_resampled30fps
```

LeRobot artifact：

```text
/home/galaxea/zibin.dong/lerobot/outputs/checkpoints/g05-so100-opensource-16node-resampled30fps-lerobot
```

转换命令：

```bash
cd /home/galaxea/zibin.dong/lerobot

uv run python -m lerobot.policies.g05.convert_g05_checkpoint \
  --legacy-root /home/galaxea/galaxeafm-opensource/checkpoints \
  --metadata-root /home/galaxea/galaxeafm-opensource/checkpoints \
  --processor-dir /home/galaxea/galaxeafm-opensource/checkpoints/so100-so100-opensource-16node_resampled30fps/hf_processor \
  --variant so100-so100-opensource-16node_resampled30fps \
  --embodiment so100 \
  --n-action-steps 32 \
  --output-dir outputs/checkpoints/g05-so100-opensource-16node-resampled30fps-lerobot
```

converter 使用 `exist_ok=False` 创建输出目录。重新转换时必须换一个新的
`--output-dir`，或在确认不再需要旧 artifact 后手动处理旧目录。

### 本次 checkpoint 的转换结果

`conversion.json` 记录：

```text
format:             lerobot-g05-v1
source_variant:     so100-so100-opensource-16node_resampled30fps
embodiment:         so100
source_filename:    model_state_dict.pt
source_sha256:      e96b0777a2e3f4662bdf5c5ed1ca765260855afe70b0a053169f76dce940f7a7
self_contained:     true
```

该 checkpoint 的关键运行配置为：

```text
inference_action_head = ar
continuous_action     = false
discrete_action       = true
predict_cot           = true
action_attend_cot     = true
cot_prompt            = "predict bbox, subtask and action"
chunk_size            = 32
n_action_steps        = 32
internal state/action = 27 dimensions
physical state/action = 6 dimensions
physical indices      = 10..15
```

转换后目录必须整体保留：

```text
config.json
model.safetensors
conversion.json
policy_preprocessor.json
policy_postprocessor.json
processor/
action_tokenizer/
```

不要只复制 `model.safetensors`。本次 artifact 约 11 GB，位于 Git ignored 的
`outputs/` 下，不提交到代码仓库。

## 加载和离线使用

```python
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.g05.configuration_g05 import G05Config
from lerobot.policies.g05.modeling_g05 import G05Policy

path = (
    "/home/galaxea/zibin.dong/lerobot/outputs/checkpoints/"
    "g05-so100-opensource-16node-resampled30fps-lerobot"
)
config = G05Config.from_pretrained(path)
policy = G05Policy.from_pretrained(path, config=config, strict=True)
preprocessor, postprocessor = make_pre_post_processors(
    config,
    pretrained_path=path,
)
```

离线检查时至少确认：

- `strict=True` 加载无 missing/unexpected weights。
- 输入包含 `observation.state`、`exterior_rgb` 和任务文本。
- `right_wrist_rgb` 存在时映射到正确的腕部相机。
- 缺失的 optional camera 由 processor 补黑帧。
- `predict_cot=true` 时能生成有效 CoT 和 action token。
- 输出 shape 为 `[batch, 32, 6]`，最终为物理机械臂绝对位置。

## SO100 硬件部署手册

### 1. 检查设备

本次使用：

```text
robot:
  /dev/serial/by-id/usb-1a86_USB_Single_Serial_5970073357-if00

cameras:
  exterior_rgb    -> /dev/video2
  right_wrist_rgb -> /dev/video0
```

设备编号可能在重启或重新插拔后变化，运行前重新检查：

```bash
ls -l /dev/serial/by-id/
v4l2-ctl --list-devices
```

不要只根据 `/dev/video0`、`/dev/video2` 的编号猜测相机语义，应先看图确认。

### 2. 校准

本次复用的 calibration id：

```text
g05_so100_follower
```

对应文件：

```text
/home/galaxea/.cache/huggingface/lerobot/calibration/robots/so_follower/g05_so100_follower.json
```

rollout 时必须带同一个 robot id：

```bash
--robot.id=g05_so100_follower
```

如果硬件、舵机方向或机械零位发生变化，不应继续复用旧文件，应重新校准：

```bash
uv run lerobot-calibrate \
  --robot.type=so100_follower \
  --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5970073357-if00 \
  --robot.id=g05_so100_follower
```

### 3. 起始姿态

原始部署代码根据训练状态均值使用以下物理臂姿态作为 home：

```text
[3.1, -34.3, 31.5, 55.9, -12.3, 13.4]
```

顺序为：

```text
shoulder_pan
shoulder_lift
elbow_flex
wrist_flex
wrist_roll
gripper
```

本次测试曾尝试移动到该姿态，但实际停在约：

```text
[3.3, -34.3, 42.1, 55.7, -12.2, 13.2]
```

其中 elbow 与目标仍相差约 10.6 度。因此本次实机结果包含起始姿态未完全
对齐的影响。当前 LeRobot rollout 没有复刻原 client 的自动 homing 流程。

### 4. 实机 rollout

下面是本次 20 秒测试所使用配置的可复现版本，**不是已经验证有效的推荐参数**：

```bash
uv run lerobot-rollout \
  --strategy.type=base \
  --policy.path=/home/galaxea/zibin.dong/lerobot/outputs/checkpoints/g05-so100-opensource-16node-resampled30fps-lerobot \
  --policy.predict_cot=true \
  --robot.type=so100_follower \
  --robot.id=g05_so100_follower \
  --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5970073357-if00 \
  --robot.use_degrees=true \
  --robot.max_relative_target=3 \
  --robot.disable_torque_on_disconnect=false \
  --robot.cameras="{exterior_rgb: {type: opencv, index_or_path: /dev/video2, width: 1280, height: 720, fps: 30, fourcc: MJPG}, right_wrist_rgb: {type: opencv, index_or_path: /dev/video0, width: 1280, height: 720, fps: 30, fourcc: MJPG}}" \
  --task="pick up the red block and put it into the blue bowl" \
  --fps=30 \
  --duration=20 \
  --return_to_initial_position=false \
  --device=cuda
```

重要安全说明：

- `--robot.disable_torque_on_disconnect=false` 会在程序退出后保持舵机力矩，
  这是为了避免机械臂一停程序就松掉倒下。使用者必须知道此时机械臂仍上力。
- `--return_to_initial_position=false` 会保留最后姿态，不会自动回到启动姿态。
- 不要在机械臂周围有人或障碍物时直接运行未验证 policy。
- `max_relative_target` 是 LeRobot 当前的逐关节限幅，不等同于原 client 的
  整体 delta 向量缩放，不能仅靠调整这个数值完成控制链路对齐。

## 已完成的验证

代码验证：

```bash
uv run pytest tests/policies/g05/test_g05.py -q
uv run ruff check \
  src/lerobot/policies/g05/configuration_g05.py \
  src/lerobot/policies/g05/convert_g05_checkpoint.py \
  src/lerobot/policies/g05/modeling_g05.py \
  src/lerobot/policies/g05/processor_g05.py \
  tests/policies/g05/test_g05.py
```

本次针对性测试结果为 23 passed。另已完成：

- 旧 checkpoint 转换成功。
- 11 GB artifact `strict=True` 加载成功。
- dummy observation 离线推理成功。
- CoT 文本可生成，包含 red object、blue bowl 和对应 subtask。
- 离散 ActionCodec token 可生成并解码为 32 步动作。
- SO100 旧 calibration 文件可加载。
- 串口和两路相机均可连接。
- 实机 rollout 可执行到结束。

这些结果只证明代码路径可运行，不证明实机动作与原始部署等价。

## 已解决的问题

### Converter 错误地假设 continuous action

旧 converter 对 action head 有硬编码倾向，无法正确处理当前 checkpoint 的
`continuous_action=false`、`discrete_action=true`。现已改为读取 checkpoint
真实配置；本 checkpoint 转换后使用 `inference_action_head=ar`。

### CoT 开关和 processor 不同步

仅修改 policy config 会使 serialized processor 仍使用旧 prompt。现在
`--policy.predict_cot` override 会同步更新 `g05_prepare_inputs` 的
`predict_cot`、`cot_prompt` 和 `append_eov`。

### SO100 状态和动作坐标系不一致

原 client 的 sign/offset 变换之前不在 LeRobot processor 中，容易导致模型
看到错误 proprio 或输出错误物理关节角。现在 converter 和 pre/postprocessor
已经携带这一变换，并有可逆性测试。

### 环境安装失败

- `num2words` 的问题来自镜像索引解析，不是 G0.5 依赖本身。
- CUDA wheel 解压失败来自磁盘空间和未清空的回收站。

### 程序退出后机械臂松掉

SO follower 默认在 disconnect 时关闭力矩。需要保持姿态的临时测试使用：

```text
--robot.disable_torque_on_disconnect=false
```

这只是退出行为配置，不是推理效果修复。

## 尚未解决的问题

### 1. 原始推理链路尚未完成逐数值等价验证

尚未用同一张 exterior 图、同一张 wrist 图、同一组关节状态和同一 task，
同时输入原始 G0.5 pipeline 与 LeRobot pipeline，并逐项比较：

- resize/颜色/归一化后的 camera tensor
- padded/normalized state
- prompt token ids
- CoT token
- action token
- 解码和反归一化后的 32 步物理动作

因此当前不能声称 LeRobot policy 与原始部署的模型输入输出完全一致。

### 2. 控制频率存在冲突

checkpoint 名称和数据配置指向 resampled 30 fps，但原始
`so100_policy_client.py` 的默认 `action_fps` 是 15 Hz，并明确将其作为控制
频率。此次 LeRobot 实机测试使用 30 Hz。

同一 32 步 chunk：

- 15 Hz 约执行 2.13 秒。
- 30 Hz 约执行 1.07 秒。

这会直接改变动作轨迹的时间尺度。应先确认训练/原部署实际使用的执行频率，
再确定 LeRobot 的 `--fps`，不能根据 checkpoint 文件名直接猜测。

ActionCodec 本身不使用 fps；这里的问题是动作发送节奏，不是 codec 解码。

### 3. 安全限幅算法不一致

原 client 的 `clip_action` 在任一关节超过阈值时，对**整个关节 delta
向量同比例缩放**，保持关节空间方向：

```text
delta *= max_deg / max(abs(delta))
```

LeRobot `ensure_safe_goal_position` 当前对每个关节分别 clamp。此次使用
`max_relative_target=3` 时，模型输出会被逐轴裁剪，可能严重改变协同轨迹方向。

需要为这个部署实现原 client 等价的向量限幅，或在 policy/robot 之间增加专用
controller adapter。

### 4. 串口控制线程不一致

原 client 使用 `FollowerArm` 后台线程独占串口：

- 持续发送最后一个目标。
- 持续读取机械臂状态。
- policy producer 只按 action fps 更新 target。
- 模型重新计算 chunk 时，控制线程仍在运行。

当前 LeRobot sync rollout 在主循环中执行 observation、policy inference 和
send action。模型每次重新计算 chunk 约需 2.4 秒，日志会报告约 0.4 Hz 的慢
循环；之后才连续消费缓存动作。虽然舵机自身会保持最后目标，但该时序仍未证明
与原 client 等价。

### 5. 起始姿态没有可靠对齐

原 client 启动时会自动 home 到训练分布中心。本次 home 未完全到位，elbow
约差 10.6 度。LeRobot rollout 尚未集成相同的 homing、容差、超时和失败终止
逻辑。

### 6. 相机采集参数没有对齐

原 client 直接使用 OpenCV camera 默认采集参数，dummy shape 是 480x640。
本次测试使用 1280x720 MJPG，再 resize 到 256x256。视场、宽高比、曝光和
畸变均可能与训练/原部署不同。相机语义映射虽然是正确的：

```text
/dev/video2 -> exterior_rgb
/dev/video0 -> right_wrist_rgb
```

但像素输入尚未做同帧对照。

### 7. AR 推理具有随机性

原 checkpoint 配置为 sampling 模式：

```text
do_sample=true
temperature=0.7
top_k=128
top_p=0.95
repetition_penalty=1.2
no_repeat_ngram_size=10
```

converter 已恢复这些参数，但单次实机 rollout 的 CoT 和 action 会有随机性。
排查等价性时应先固定随机种子，或临时用 greedy 做原版/LeRobot 的确定性
对照，之后再恢复 sampling。

### 8. 实机任务尚未成功

测试任务：

```text
pick up the red block and put it into the blue bowl
```

结果很差，未完成抓取和放置。现阶段最可能的部署问题集中在控制频率、限幅、
后台控制线程、home 和输入像素对齐；在这些问题完成排查前，不应把失败归因于
模型能力，也不应继续用反复启动机械臂的方式定位问题。

## 推荐的下一步

下一次继续时应按以下顺序进行：

1. 保存一组真实但静态的双相机图像和六维 state。
2. 固定随机种子并关闭硬件写入。
3. 对原 pipeline 和 LeRobot pipeline 做同输入逐项数值比较。
4. 修复所有模型输入/输出差异。
5. 实现与原 client 等价的 15/30 Hz 可配置调度、整体向量限幅和后台持有线程。
6. 增加可靠 homing，未达到容差时禁止进入 rollout。
7. 增加不写机械臂的 dry-run/日志模式，先记录 CoT、raw action chunk 和限幅后
   action，再进行一次短时实机测试。

## 已验证的 LIBERO 配置

现有 LIBERO 最佳完整四套件结果使用 converted 100k checkpoint、严格 FP32、
eager text/action attention、eager vision attention，以及默认 LIBERO episode
长度。

关键配置：

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

结果：

| Suite | Success rate | Successes | Episodes |
| --- | ---: | ---: | ---: |
| libero_spatial | 97.8% | 489 | 500 |
| libero_object | 99.8% | 499 | 500 |
| libero_goal | 97.4% | 487 | 500 |
| libero_10 | 97.2% | 486 | 500 |
| **Overall** | **98.05%** | **1961** | **2000** |

复现单个 suite：

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
