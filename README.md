# supermimic-rl

按 [实施指导方案](SuperMimic_复现与_Isaac_Lab_接入实施指导方案.md) 实现的可微分世界模型 BPTT 强化学习基础库。采用类似 RSL-RL 的 Runner / Algorithm / Models / Storage 职责划分，独立依赖 PyTorch。

当前支持 toy 和宇树 G1 的 Isaac Lab 单动作跟踪训练：真实环境采集 → 连续 8 步模型拟合 → 32 步可微想象 → actor / critic 更新。G1 复用本仓库新增的官方 mimic 资产和参考加载器，使用独立 SuperMimic 任务配置。已接入任务级训练，但尚不代表论文结果复现；多运动库与返回率起点采样仍待实现。

G1 完整字段、论文差异、时序和验收说明见 [实施方案第 14 节](SuperMimic_复现与_Isaac_Lab_接入实施指导方案.md#14-宇树官方-mimic--g1-isaac-lab-的实际接入)。

## G1 Isaac Lab 训练

在已安装 Isaac Lab / Isaac Sim 的环境中，先将官方 CSV 转为 **50 Hz** NPZ：

```bash
python scripts/mimic/csv_to_npz.py \
  -f source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/dance_102/G1_Take_102.bvh_60hz.csv \
  --input_fps 60 --output_fps 50 --output_name /tmp/g1_dance_50hz.npz \
  --convert-only --headless --device cuda:0
```

先用少量环境验证，再打开默认 DR/noise 训练：

```bash
python scripts/supermimic/train.py --motion-file /tmp/g1_dance_50hz.npz \
  --num-envs 4 --check-only --no-dr --no-observation-noise \
  --episode-length-s 0.24 --log-dir runs/g1_check --headless --device cuda:0

python scripts/supermimic/train.py --motion-file /tmp/g1_dance_50hz.npz \
  --num-envs 64 --iterations 1000 --log-dir runs/g1_train --headless --device cuda:0
```

任务默认 `Unitree-G1-29dof-SuperMimic-Dance-102`，另支持 `Unitree-G1-29dof-SuperMimic-Gangnam-Style`，必须配套相应参考。机器人资产可用 `UNITREE_ROS_DIR` 或训练参数 `--robot-urdf` 指定。脚本优先导入本仓库的官方代码，保留原 PPO 配置。

G1 使用 93 维观测的 10 帧历史、10 帧参考、32 维目标 latent、真实物理 context 及解析奖励/终止头。`debug` 默认适合先做小规模验证；`--profile paper` 提供论文网络宽度和主要优化器配置，需要更大显存，且仍有调度、编码和采样偏离，详见方案。

`--resume runs/g1_train/model_latest.pt` 恢复模型、optimizer、replay 和计数器，参数必须保持一致。PhysX 恢复后开始新 episode，不能精确恢复接触/求解器状态；评估应单独创建环境。运行目录保存 `metrics.jsonl`、`task_spec.json`、`train_config.json`、`env_config.yaml` 和含 motion SHA256 的元数据。

本机已用 RTX 5070 Ti / Isaac Sim 5.1 完成 4 环境短测，包括强制 timeout reset、开启 DR/noise、8 步模型拟合及 32 步 actor/critic 更新。此结果验证接入可运行，不表示运动跟踪已经收敛。

## 单动作收敛实验

新增 `--profile single_motion`，作为 16 GB GPU 上的工程起点，尚未完成长期收敛验证。建议先用一段 2–5 秒参考、128 个环境，并关闭 DR、传感噪声和 reset 位姿/速度扰动。采集动作仍保留 0.1 标准差的探索噪声。

本机已用该网络和完整 batch 在 128 个 G1 环境中完成两轮资源/链路验证（短测将预热临时设为 1 次 update）；正式配置仍使用 100 次预热。相关回归测试共 35 项通过。

| 参数 | `single_motion` |
| --- | --- |
| 模型窗口 / 想象长度 | 8 / 32 |
| actor / 目标编码器 | (512, 256, 128)，目标 latent=32 |
| 世界模型 / critic | (512,512,512) / (256,128) |
| 模型 batch / 每次更新拟合 batch 数 | 512 / 10 |
| 想象 batch | 128 |
| 模型预热 | 前 100 次算法 update 只训练模型；之后开启 actor/critic |
| actor / critic / 模型学习率 | 1e-4 / 2e-4 / 3e-4，保持常数 |
| 每环境每轮采集 / replay | 16 / 256 vector steps |
| critic | 5 epochs × 4 minibatches，γ=.99、λ=.95 |
| 梯度裁剪 | global norm=1；actor 再按元素裁剪到 ±.1 |

例如从官方 CSV 选取前 5 秒：

```bash
mkdir -p data
python scripts/mimic/csv_to_npz.py \
  -f source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/dance_102/G1_Take_102.bvh_60hz.csv \
  --frame_range 1 301 --input_fps 60 --output_fps 50 \
  --output_name data/g1_dance_5s_50hz.npz --convert-only --headless --device cuda:0

python scripts/supermimic/train.py --profile single_motion \
  --motion-file data/g1_dance_5s_50hz.npz --num-envs 128 --iterations 10000 \
  --episode-length-s 5 --no-dr --no-observation-noise --no-reset-perturbations \
  --log-dir runs/g1_single_motion_v2_seed1 --log-interval 10 --seed 1 \
  --headless --device cuda:0

python scripts/supermimic/train.py \
  --profile single_motion \
  --motion-file /home/yi/aaai/supermimic/supermimic-rl/datasets/g1_dance_50hz.npz \
  --num-envs 4096 \
  --model-batch-size 1024 \
  --imagination-batch-size 256 \
  --iterations 10000 \
  --episode-length-s 1000000 \
  --no-dr --no-observation-noise --no-reset-perturbations \
  --log-interval 10 --seed 1 \
  --headless --device cuda:0
```

先检查 1000–3000 轮的趋势，再决定是否继续；10000 轮是实验预算，不是保证收敛的轮数。100 轮模型预热仍使用当前策略采集真实数据，日志显示 `world model warmup`，尚无 actor/critic loss 是正常现象。`--check-only` 会跳过该预热以验证完整链路。

显存不足时，在新实验启动时降为 `--num-envs 64 --model-batch-size 256 --imagination-batch-size 64`。还可通过 `--world-model-batches / --actor-warmup-updates / --actor-lr / --critic-lr / --world-model-lr` 调整。严格续训仍要求算法参数一致。

收敛判断同时看：`Train/mean_step_reward` 上升、`Train/episode_failure_rate` 下降、`Metrics/motion/error_body_pos` 等跟踪误差下降，以及 `WorldModel/replay_normalized_state_mse_8/32` 稳定或下降。后两项 replay 指标不是独立验证集误差。只见 `Imagination/return` 上升而真实奖励不改善，说明需要检查模型被策略利用的可能性。若 1000–3000 轮完全无改善，应诊断模型误差与数据覆盖，而不是只延长训练。

训练从随机参考帧 reset，片段末端算 timeout，所以平均 episode 长度不能直接作为整段完成率。应额外用 `play.py --start-frame 0 --steps 1250 --headless` 检查从首帧连续执行；其结尾分别打印 failures 和 timeouts。初始片段跑通后，再在新实验中逐项加入 reset 扰动、传感噪声和 DR。

**奖励尺度修订**：核对论文 Table A.6 下方说明后，G1 `TaskSpec` 升至 v2，所有奖励项乘控制周期 dt=.02，理想单步奖励上限为 .1。v1 的回报数值是 v2 的 50 倍；旧训练 checkpoint 不用于该正式实验续训，需新建训练目录。v1 actor 仍可用 play 推理，但奖励输出按 v2 单位报告。预热、较小网络与这些实验参数是工程选择，详见[论文附录 C/D/G](https://arxiv.org/pdf/2610.28480)。

## 播放最新 G1 策略

默认从 `runs/g1_supermimic` 及其子目录中，按检查点文件修改时间选择最新的 `model_latest.pt` 或 `model_<iteration>.pt`，自动读取训练保存的任务、参考动作和网络结构：

```bash
python scripts/supermimic/play.py --real-time
```

指定训练目录或检查点：

```bash
python scripts/supermimic/play.py --log-dir runs/g1_train --real-time
python scripts/supermimic/play.py --checkpoint runs/g1_train/model_100.pt
# 无窗口短测
python scripts/supermimic/play.py --log-dir runs/g1_train --headless --steps 200 --device cuda:0
```

默认显示一个机器人，从参考第 0 帧开始，每次 reset 回到该起点，并由相机跟随。策略使用训练时相同的 10 帧观测历史和参考目标；默认关闭 DR、reset 扰动与传感噪声，动作没有采集期探索噪声。可用 `--randomize`、`--observation-noise` 开启相应选项，`--start-frame` 修改参考起点，`--fixed-camera` 固定相机，`--num-envs` 修改环境数。

GUI 默认运行到关闭窗口；`--steps N` 限定步数，headless 未指定时默认 1000 步。只加载 actor，不构建世界模型、critic、optimizer 或 replay，也不会写入训练日志/覆盖检查点。脚本启动时打印所选路径、训练轮数、任务和参考文件。

参考文件移动后，用 `--motion-file /new/path/motion.npz` 覆盖；早期检查点若没有配套元数据，可显式提供 `--task` 与 `--motion-file`。现有 G1 检查点支持从权重推断网络结构，新检查点直接保存策略结构及任务/资产信息。播放时会核对 TaskSpec、关节顺序和动作缩放。

## 运行

Python >= 3.10，PyTorch >= 2.2。已有 PyTorch 环境可以直接执行：

```bash
python -m supermimic.train --iterations 20 --num-envs 16 --log-dir runs/toy
python -m supermimic.train --iterations 10 --num-envs 16 --log-dir runs/toy --resume runs/toy/model_latest.pt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -v
```

也可以 `pip install -e .` 后使用 `supermimic-train`。CUDA 环境添加 `--device cuda:0`。命令行默认使用单 CPU 线程，避免小批量 toy 计算的线程调度开销。

`metrics.jsonl` 记录模型、actor、critic 损失与裁剪前梯度范数、软存活权重、真实平均单步奖励、想象目标值、有效窗口比例、采集吞吐和阶段耗时。真实奖励与想象 return 的时间尺度不同，不应直接相减。`replay_state_mse_1/8/32` 是真实回放动作下的自回归预测误差；数据不足以构成相应连续窗口时不输出。它们不是独立验证集指标。

## 训练日志

参考 RSL-RL 的独立 Logger 分工，toy 与 G1 共用 `TrainingLogger`。默认每轮打印训练摘要，包含迭代进度、累计真实环境步数、总吞吐、采集/训练耗时、ETA、模型/actor/critic 损失与梯度范数、实际单步奖励、最近 100 个完整 episode 的回报和长度、想象 return、软存活率及回放有效窗口比例。尚无完整 episode 时不输出 episode 平均值；没有有效训练窗口时显示 `replay warmup`。

```text
########################################################################################
                           SuperMimic learning iteration 10/1000
                               Status: training
              Total environment steps: ...
                     World model loss: ...
                           Actor loss: ...
                          Critic loss: ...
                  Mean episode return: ...
          Mean episode length (steps): ...
                        Elapsed / ETA: ... / ...
```

设置 `log_dir` 后保存：

| 文件 | 内容 |
| --- | --- |
| `training.log` | 与控制台同样的可读摘要 |
| `metrics.jsonl` | 每轮完整标量，便于脚本分析 |
| `events.out.tfevents.*` | TensorBoard 曲线（默认启用） |
| `train_config.json` / `task_spec.json` | 当前算法、日志与任务配置 |

TensorBoard 为可选依赖；`pip install -e '.[logging]'` 可安装。缺少依赖时会提示，并继续使用控制台/JSONL。启动查看：

```bash
tensorboard --logdir runs --port 6006
```

标签按 `Loss/`、`Grad/`、`WorldModel/`、`Train/`、`Imagination/`、`Replay/`、`LearningRate/`、`Policy/`、`Perf/` 组织。其中 `Train/mean_reward` 是已完成 episode 的真实未折扣回报，`Train/mean_step_reward` 是本轮真实单步奖励均值；想象目标单独记为 `Imagination/return`。G1 的 Isaac Lab episode extras（如 `Episode_Reward/`、`Metrics/`）也会写入。

两个训练入口均支持：

```bash
# 每 10 轮打印/保存摘要，JSONL 与 TensorBoard 仍然每轮记录。
python -m supermimic.train --iterations 100 --log-interval 10
# 只用控制台和 JSONL，不创建 TensorBoard events。
python -m supermimic.train --logger json
# 关闭控制台迭代摘要；文件继续记录，命令行末尾仍输出最终结果。
python -m supermimic.train --quiet
```

也可设置 `RunnerConfig.logger / console_log / log_interval / episode_stats_window / log_flush_secs`。日志配置可在恢复时修改，不影响算法配置校验。episode 的未完成累计量和最近完成记录随检查点保存；toy 精确恢复后继续累计，PhysX 从新 episode 恢复时清空未完成累计量。旧检查点可加载，episode 统计从恢复后重新累计。重复调用 `learn()` 或异常退出时会关闭并 flush writer。

## 接口与模块

```python
from supermimic.configs import AlgorithmConfig, RunnerConfig
from supermimic.envs import ToyAdapter
from supermimic.models import WorldModel
from supermimic.models.heads import ToyObservation, ToyReward, ToyTermination
from supermimic.runners import SuperMimicRunner

cfg = RunnerConfig(algorithm=AlgorithmConfig())
env = ToyAdapter(num_envs=16, noise_std=cfg.algorithm.observation_noise_std)
model = WorldModel(env.spec, cfg.hidden_dims,
                   ToyObservation(), ToyReward(), ToyTermination())
runner = SuperMimicRunner(env, cfg, log_dir="runs/toy", world_model=model)
runner.learn(num_learning_iterations=20)
runner.save("runs/toy/checkpoint.pt")
runner.load("runs/toy/checkpoint.pt")
metrics = runner.evaluate(n_steps=128, seed=0)
policy = runner.get_inference_policy()
view = env.get_step_view()
actions = policy(view.obs_history, view.goal)
```

Runner 也接受具有相同字段的配置字典。`learn` 的迭代数表示追加训练轮数。每个 iteration 先采集 `num_steps_per_env` 步，再执行 `updates_per_iteration` 次模型拟合与策略/critic 更新；没有有效模型窗口时继续预热采集。

| 模块 | 主要接口与职责 |
| --- | --- |
| `SuperMimicRunner` | `learn / evaluate / save / load / get_inference_policy`，编排、JSON 日志、检查点 |
| `SuperMimicAlgorithm` | `collect / fit_world_model / imagine / update_actor / update_critic / update`，三套独立 optimizer |
| `ReplayBuffer` | `append / sample_windows / sample_seeds / state_dict`，按时间和环境保留真实转移 |
| `Actor` | `act(history, goal)`，目标编码器与确定性 tanh 动作解码器 |
| `Critic` | `value(state, goal, context)`，训练时特权输入 |
| `WorldModel` | `transition / observe_clean / reward / termination_prob`，可学习或解析头 |
| `EnvAdapter` | `reset / get_step_view / step / get_context / get_full_state`，原子快照及真实执行动作 |

张量单步为 `[B, feature]`，序列为 `[B, time, feature]`。默认历史长度 10，模型窗口 8，想象长度 32，γ=0.99，λ=0.95。actor 仅接收历史与目标；context 仅进入模型和 critic。模型规模是可配置的小型 MLP 工程起点。

## 更新规则

- 模型拟合以真实初始状态开始，后续自回归使用预测状态与回放记录的执行动作。损失沿时间求和、沿 batch 平均；状态采用特征维均方误差。学习型头增加观测 MSE、奖励 MSE 和终止 BCE；解析头不加入对应监督损失。
- actor 最大化 `Σ γ^t c_t r_(t+1) + γ^H c_H V_H`，其中 `c_(t+1) = c_t (1-d_(t+1))`。世界模型与 critic 在构图时冻结参数，保留对状态和动作的输入梯度。整段想象只做一次 actor backward。
- 想象从回放的真实含噪历史开始，每步把预测的干净观测加外生噪声后压入历史。当前观测不会重复入队。
- critic 使用同次想象中固定的 GAE returns；状态、目标、context 和 targets 全部 detach，随后进行多 epoch / minibatch 拟合。没有 target critic。
- replay 分开保存失败终止与时间截断，窗口最后一步允许 done，此前不允许 done；还检查 episode 与 episode 内 context 一致性。时间截断不会监督为失败终止。

## 接入新任务 / Isaac Lab

实现 [EnvAdapter](supermimic/envs/protocol.py) 并提供版本化 `TaskSpec`。`StepView` 所有字段必须来自同一时刻并拥有稳定生命周期；返回快照或 clone，避免后续仿真原地写入。`EnvStep.terminal` 必须是动作执行后、自动 reset 前的真实状态；`EnvStep.current` 是 reset 后下一次策略使用的状态。`action_exec` 为动作变换后实际执行动作的归一化表示，缩放与编码必须与世界模型一致。

提供 `goal_provider(seed, offset)`，通过 `motion_id / frame_id + offset` 查询参考目标，offset 包含末端 H。该目标在 t 时刻给策略，并用于评价 t→t+1 的转移。toy 目标在 episode 内固定；运动跟踪必须提供自己的参考推进，不能直接复用 `stationary_goal`。

按任务替换状态编码及世界模型转移，提供可微观测、奖励与软终止模块，或使用默认学习型头。`WorldModel` 未传入的头会自动启用监督学习；解析头应为 `nn.Module`，函数签名见 [模型实现](supermimic/models/world_model.py)。姿态归一化、局部坐标系、接触、动作缩放、关节顺序必须由任务明确实现。

Runner 可注入自定义 `world_model / actor / critic / goal_provider`，无需依赖 PPO 内部接口。其职责和调用方式参考 [RSL-RL Runner](https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/runners/on_policy_runner.py)，数据协议针对模型学习扩展，不是现有 `RslRlVecEnvWrapper` 的直接替代。

检查点保存三组网络与 optimizer、完整 replay、环境状态、配置、TaskSpec、Python/PyTorch/CUDA 随机数和迭代计数。严格续训要求配置一致，并要求环境 `state_dict/load_state_dict` 能完整恢复仿真内部状态；仅恢复机器人姿态不足以保证真实物理仿真精确续训。当前 CPU toy 测试可逐参数精确恢复。推理迁移可单独保存 actor 的 `state_dict`。只加载可信检查点。

`evaluate` 使用无探索噪声的确定性动作，结束后恢复环境和随机状态。观测噪声与 DR 仍由适配器设置决定；固定无噪评估需显式构建相应环境配置。toy 使用共享数学公式校验观测与奖励，失败标签是位置越界，想象期通过 sigmoid 平滑。

## 验证范围

测试覆盖软存活目标手算、GAE λ 边界、软终止梯度、动作 Jacobian 与完整 32 步 actor 梯度有限差分、后期奖励到早期动作的传播、模型与 critic 参数冻结、学习型头更新、回放环形覆盖和 reset 边界、DR context、终止前状态、toy 奖励/观测一致性、检查点恢复、推理接口及预热。

默认配置和短期训练只验证算法闭环与数值可运行性。下一阶段需要多运动库、返回率起点采样、独立验证数据及真实环境长期回报对照，以判断模型误差和策略学习效果。
