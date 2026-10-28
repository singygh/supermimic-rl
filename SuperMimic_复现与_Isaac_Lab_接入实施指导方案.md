# SuperMimic 复现与 Isaac Lab 接入实施指导方案

> 版本：2026-10-28。本文面向先复现训练逻辑、再接入 Isaac Lab 的实现路线。标记 **[论文]** 表示可在 SuperMimic 论文中直接找到的设定；**[建议]** 表示为工程实现、调试或跨仿真器迁移提出的设计，不能当作论文原始实现。所有张量维度、单位、坐标系和奖励时序都应以实际任务配置为准。

## 1. 目标与边界

**目标**：用普通刚体仿真器收集转移，在线训练可微世界模型，再通过该模型的短时想象轨迹训练确定性动作策略与 critic。仿真器只提供数据和评估，不参与反向传播。先在可控的小任务上验证梯度、时序和数据一致性，再扩展至运动跟踪与 Isaac Lab。

**架构取舍**：借鉴 RSL-RL 的 Runner / Algorithm / Models / Storage / Environment 分工、配置和日志方式；不要把核心更新写成 PPO。PPO 的 log-prob、clip ratio、熵奖励和传统 on-policy rollout storage 不是 SuperMimic 的策略梯度。需要的是可重采样的**带状态与 DR context 的回放缓冲区**，以及穿过冻结参数的世界模型所构成的 **32-step BPTT 计算图**。RSL-RL 当前概览对这些模块职责的描述见[官方文档](https://github.com/leggedrobotics/rsl_rl/blob/main/docs/guide/overview.rst)。

**复现层级**：

1. 算法层复现：三阶段循环、8-step 模型拟合、32-step BPTT、软存活权重、critic/GAE；可先用小型仿真任务。
2. 任务层复现：匹配 G1、参考运动、动作缩放、奖励、终止、观测和 DR；这决定与论文指标是否可比。
3. 结果层复现：再考虑论文网络规模、数据集、批量、训练时长和硬件。Isaac Lab 迁移先追求定义一致与可训练性，不直接承诺论文的速度或真实机器人结果。

## 2. 论文事实与建议设计的对照

| 事项 | 论文设定 | 本方案建议 |
|---|---|---|
| 主循环 | 每轮仿真采集、模型拟合、策略/critic 更新，世界模型持续在线更新 | `SuperMimicRunner` 只负责编排与检查点，计算封装在 `SuperMimicAlgorithm` |
| 模型拟合 | 从回放数据取 `k=8` 的连续窗口，以真实初始状态和记录动作自回归预测 | 统一 `ReplayWindow` 数据结构，跨终止/重置的窗口一律拒绝或显式截断 |
| 想象优化 | 从回放状态开始，`H=32`，策略通过固定参数的世界模型获取解析梯度 | 专用 `imagine_and_update_actor()`；为显存可调批量、必要时使用经验证的重计算 |
| 世界模型 | 预测下一状态，观测/奖励/终止由可学习或解析头提供；DR context 进入动力学/观测 | 将物理状态、观测映射、奖励和软终止拆成可测试模块 |
| Critic | 对想象轨迹用 GAE return 拟合；没有 EMA target critic | 单独 optimizer；目标生成后 `detach`；actor 更新时冻结 critic **参数**但允许输入梯度 |
| 部署信息 | actor 只接收带噪本体观测历史和目标；critic 可接收状态、目标与 context | 用明确的 `ActorInput` / `CriticInput` 类型防止特权量泄漏 |
| Isaac Lab | 论文主要使用 MuJoCo Warp，不给出 Isaac Lab 适配层 | 独立 `IsaacLabEnvAdapter`，不要假定现成 RSL-RL wrapper 已暴露完整物理状态与 DR 参数 |

论文依据：[SuperMimic 正文、式 (1)–(3)、附录 A–D](https://arxiv.org/pdf/2610.28480)。

## 3. 推荐目录结构

```text
supermimic/
├── configs/
│   ├── algorithm.py          # k/H、批量、优化器、权重、阶段开关
│   ├── task.py               # 状态/观测/奖励/动作/终止规格
│   └── env_isaaclab.py       # Isaac Lab 任务与物理步长配置
├── runners/
│   └── supermimic_runner.py  # 生命周期、迭代编排、日志、评估、保存/恢复
├── algorithms/
│   └── supermimic.py         # collect / fit_world_model / update_actor_critic
├── storage/
│   ├── replay_buffer.py      # 环形缓存、连续窗口、初始状态采样
│   └── batches.py            # TransitionBatch / ReplayWindow / ImaginedRollout
├── models/
│   ├── world_model.py        # f_ω(s,a,ξ)
│   ├── actor.py              # 目标编码器 + 共享动作解码器
│   ├── critic.py             # V_ψ(s,g,ξ)
│   └── heads.py              # 可微观测、奖励、终止
├── envs/
│   ├── protocol.py           # EnvAdapter 协议及张量契约
│   ├── toy_adapter.py        # 小环境验证
│   └── isaaclab_adapter.py   # Isaac Lab 具体映射
├── tasks/
│   ├── motion_library.py     # 参考帧、目标序列、时间推进
│   ├── state_codec.py        # 仿真状态 <-> 模型状态；坐标系与归一化
│   └── reward_terms.py       # 仿真与想象共用的奖励数学定义
├── tests/
│   ├── test_buffer_boundaries.py
│   ├── test_gradients.py
│   ├── test_returns.py
│   └── test_adapter_contract.py
└── train.py                  # 配置装配与启动
```

以上目录全部是 **[建议]**。如果以后要嵌入 RSL-RL，可保留相同职责，但先固定版本并做一层薄适配，不依赖其 PPO 内部私有方法或 storage 语义。

## 4. 核心数据契约与接口

### 4.1 时间与张量约定

统一采用 `[batch, time, feature]`，单步张量 `[batch, feature]`。`s_t` 是动作执行前的可恢复完整状态；`a_t^exec` 是仿真器实际执行的动作；`r_{t+1}`、`done_{t+1}` 和 `s_{t+1}` 均由该动作产生。`g_t` 是在时刻 `t` 给 actor 的目标，`ξ` 在一个 episode 内固定（若环境实现逐步变化，必须把变化纳入状态或模型输入）。`dt`、动作缩放、关节顺序、四元数约定、局部坐标系与目标帧索引写入一份版本化 `TaskSpec`。

```python
@dataclass
class TransitionBatch:
    state_t: Tensor         # [B, Ds]，完整仿真/模型状态
    clean_obs_t: Tensor     # [B, Do]，无传感噪声
    obs_history_t: Tensor   # [B, 10, Do]，实测历史，可含噪声
    goal_t: Tensor          # [B, ...]，含未来参考帧时须记录帧索引
    action_exec_t: Tensor   # [B, Da]，含采集时动作噪声及实际限幅
    reward_tp1: Tensor     # [B]
    next_state: Tensor     # [B, Ds]
    next_clean_obs: Tensor # [B, Do]
    terminated_tp1: Tensor # [B]，任务终止标签
    truncated_tp1: Tensor  # [B]，时间限制；单独存储
    context: Tensor        # [B, Dxi]，真实采样的 DR 参数
    episode_id: Tensor     # [B]，窗口边界校验
    motion_id: Tensor      # [B]，参考序列索引
    frame_id: Tensor       # [B]，当前参考帧索引
```

**[建议]** `ReplayWindow` 含 `state_0 [B,Ds]`、`actions [B,8,Da]`、真实 `states/rewards/clean_obs/termination` 对齐到 `t+1...t+8`、`goals`、`context` 和 `valid_mask`。`ImaginationSeed` 含 `state_0`、对应的 10 帧历史、目标/参考索引及 `context`。回放缓存只能存真实仿真数据，不能把模型想象轨迹混入世界模型的监督样本。

### 4.2 Runner / Algorithm / ReplayBuffer / Models / EnvAdapter

| 模块 | 最小接口（建议） | 不变量与责任 |
|---|---|---|
| `SuperMimicRunner` | `learn(n_iterations)`, `evaluate()`, `save(path)`, `load(path)` | 管理随机种子、初始化、采集/更新节奏、指标、检查点；不直接写损失 |
| `SuperMimicAlgorithm` | `collect(env, n_steps)`, `fit_world_model(batch)`, `imagine(seed)`, `update_actor(rollout)`, `update_critic(rollout)` | 三套 optimizer 与梯度域隔离；保存参数、optimizer、调度器、计数器 |
| `ReplayBuffer` | `append(step)`, `sample_windows(k,batch)`, `sample_seeds(batch)`, `state_dict()` | 按 env/episode 连续采样，排除跨 reset 窗口；保留执行动作与 context |
| `WorldModel` | `transition(s,a,xi)->s_next`, `observe_clean(s,xi)->o`, `reward(s_next,a,g)->r`, `termination_prob(s_next,g)->d_hat` | 输出对输入保持可微；各头用同一时序定义；训练与想象使用相同状态编码 |
| `Actor` | `act(obs_history,goal)->action` | 只使用部署时可获得的信息；可选目标编码器/共享解码器 |
| `Critic` | `value(state,goal,context)->scalar` | 训练期允许完整状态与 context；不共享 actor 的特权输入 |
| `EnvAdapter` | `reset(ids)->StepView`, `get_step_view()->StepView`, `step(action_exec)->StepView`, `get_context()->xi`, `get_full_state()->s` | 原子地返回同一时间步的数据；隔离仿真器 API、自动 reset 和状态提取差异 |

`StepView` 应同时返回 `terminated` 与 `truncated`、`episode_id`、参考目标索引、无噪观测和实际给 actor 的带噪历史。若 Isaac Lab 环境在 `step()` 中自动 reset，适配层必须在 reset 覆盖前捕获 **terminal next state**，否则会把新 episode 的初始状态误当作旧 episode 的下一状态。此处是 **[建议]** 的接口要求，不是论文给出的 API。

### 4.3 观测、状态、历史与 DR context

**[论文]** actor 的单帧本体观测含 29 维默认姿态相对关节角、29 维关节速度、3 维基座角速度、3 维投影重力和 29 维上一动作；策略使用最近 10 帧观测。运动目标编码器看未来 10 帧参考。论文的世界模型预测**无传感噪声**观测，想象 rollout 每步再采样噪声后喂给 actor。不能把 actor 观测误当成完整动力学状态。见[附录 A、C](https://arxiv.org/pdf/2610.28480)。

**[建议]** 定义 `state_codec` 为唯一的坐标变换入口，完整状态应足以推进模型、计算奖励和终止，并能确定下一步无噪观测。至少审计 root 位姿/速度、关节态、上一动作、接触或用于解析奖励的运动学量；具体字段随任务而定。历史更新用队列：初始化想象轨迹时从回放缓冲区读取真实最近 10 帧；在每个模型步把 `observe_clean(s_hat_t, ξ) + η_t` 压入队列并移除最旧帧。episode reset 时用规定的填充方式重建历史，避免跨 episode 污染。`η_t` 是外生噪声样本，本身不需要求导，但加法后的观测必须仍对 `s_hat_t` 可微。

**[论文]** DR context `ξ` 包含摩擦系数、关节标定偏差和躯干质心偏移，进入世界模型；部署 actor 不读取它。论文省略 root 速度突变式扰动，因为确定性模型无法解释未记录的跳变。论文附录 A.3 给出采样范围。**[建议]** 在 Isaac Lab 中逐项核对随机化器实际应用的物理参数，并在采集当下记录最终生效值；新加随机化若无法从 `s,a,ξ` 解释，先禁用或显式扩充 `ξ`/状态。训练集与验证集可按 DR 参数、运动片段分层，以诊断外推。

## 5. 端到端训练循环

```python
for iteration in range(num_iterations):
    # 1) 仿真采集：actor 用观测历史和目标；仿真执行加噪且限幅后的动作。
    #    记录 action_exec、s_t、s_{t+1}、r_{t+1}、终止、ξ 和参考索引。
    steps = algorithm.collect(env_adapter, n_collect_steps)
    replay.append(steps)

    # 2) 模型拟合：每轮从回放数据抽连续 8-step 窗口。
    for _ in range(n_world_model_batches):
        windows = replay.sample_windows(k=8, batch_size=wm_batch_size)
        wm_metrics = algorithm.fit_world_model(windows)

    # 3) 从真实回放状态和历史启动想象，冻结模型权重，更新 actor。
    seeds = replay.sample_seeds(batch_size=imagination_batch_size)
    imagined = algorithm.imagine(seeds, horizon=32)
    actor_metrics = algorithm.update_actor(imagined)

    # 4) 用同次想象轨迹的固定 GAE targets 更新 critic。
    critic_metrics = algorithm.update_critic(imagined)
    runner.log_and_checkpoint(...)
```

这是 **[建议]** 的接口伪代码，时序遵循论文 Algorithm A.1。真实实现中，`imagine()` 与 `update_actor()` 通常合在一次保留计算图的调用里；critic 目标应在需要的价值预测完成后 `detach`，且不能在 actor 的 backward 之前改写它所依赖的参数。`collect` 的步数、模型批次数、更新/采集比等由配置给出。**[论文]** 默认 `k=8`、`H=32`、`γ=0.99`、`λ=0.95`，每个模型更新对应一个策略更新；其他附录 D.2 超参数只能在任务与硬件匹配后作为对照起点。

可选的返回率采样：**[论文]** 参考动作时间线以 1 秒分箱，reset 起点混合 50% 均匀、40% 按低回报 softmax、10% 历史困难权重。**[建议]** 把它放在 `MotionStartSampler`，等基础训练与奖励对齐后再开启，避免采样变化掩盖算法错误。

## 6. World Model：8-step 自回归拟合

**[论文]** 用窗口起点真实状态 `s_0` 初始化 `s_hat_0`。对于 `j=0...7`，输入**回放记录的实际执行动作** `a_j^exec` 和该 episode 的 `ξ`，得到 `s_hat_{j+1}=f_ω(s_hat_j,a_j^exec,ξ)`；下一步继续用 `s_hat_{j+1}`，不能每步换回真实状态。损失对预测状态、奖励、无噪观测和硬终止标签求和；解析头不加对应监督项。可写为：

```text
L_WM = mean_{batch} sum_{j=1..8} mask_j * [
    ||s_hat_j - s_j||²
  + (r_hat_j - r_j)²                    # 仅学习型奖励头
  + ||o_clean_hat_j - o_clean_j||²     # 仅学习型观测头
  + BCE(d_hat_j, d_j)                   # 仅学习型终止头
]
```

**[建议]** 首先做一个可观察、可微、单位明确的小状态模型；再加入局部坐标增量、姿态重归一化或残差结构等复杂实现。各损失项可按量纲归一化，但记录该变更，因为论文公式的各项是直接相加。验证集同时报告 1、8、32 步误差，尤其是接触切换和低回报状态。仅低单步误差不足以证明 32 步策略梯度可信。

## 7. 32-step BPTT：可微奖励与软终止

每条想象轨迹从回放 `s_0, ξ, obs_history_0, motion/frame_id` 出发。第 `t` 步取目标 `g_t`，由 `s_hat_t` 构造无噪观测并加采样噪声，actor 产生 `a_t`，世界模型产生 `s_hat_{t+1}`，再计算与 `t+1` 对齐的 `r_hat_{t+1}` 和 `d_hat_{t+1}`。维护 `c_hat_0=1`，`c_hat_{t+1}=c_hat_t(1-d_hat_{t+1})`。**[论文]** 最大化：

```text
J_hat(θ) = E[ Σ_{t=0}^{H-1} γ^t c_hat_t r_hat_{t+1}
             + γ^H c_hat_H V_ψ(s_hat_H, g_H, ξ) ],   H=32
L_actor = -J_hat
```

`r_hat` 必须由可微运算构成，并和仿真奖励保持同一公式、单位、目标帧与时间索引；终止概率 `d_hat∈[0,1]` 是想象期的软门控，而仿真采集存硬 `done`。不要在想象轨迹上用布尔 `if done: break` 替代 `c_hat`，也不要将 `d_hat`、`r_hat` 或中间 `s_hat` 转成 Python 标量。可给 `d_hat` 做数值范围检查，但避免破坏主要梯度。

`total_loss.backward()` 会把各时刻奖励及末端价值通过共享的世界模型转移链回传到较早动作与同一组 actor 参数，梯度自动累加。比如第 3 步的损失可影响第 1–3 步动作；第 2 步可影响第 1–2 步；这些贡献不是额外执行三次 backward。每次更新前 `actor_optimizer.zero_grad(set_to_none=True)`；做一次 `L_actor.backward()` 和裁剪/更新。**禁止**在 32 步中途无意 `detach(s_hat)`、把模型前向包在 `torch.no_grad()`、或把可微状态改为 NumPy。显存优化若使用截断 BPTT，应明确标为偏离论文并单独评估。

### 冻结权重，但保持输入梯度

策略更新时 **[论文]** 世界模型权重固定，链式法则仍须穿过它对状态、动作的 Jacobian。**[建议]** 可在 actor 更新阶段对世界模型和 critic 的 `parameters()` 临时设置 `requires_grad_(False)`，前向仍在正常 autograd 模式；退出后恢复标志。`s_hat_next = world_model(s_hat, action, xi)` 必须保有 `grad_fn`，`∂s_hat_next/∂action` 不能为零或缺失。`torch.no_grad()` 和 `output.detach()` 会截断这条链，不能用于 actor 的想象前向。critic 的参数在 actor 更新阶段冻结，但末端 `V_ψ(s_hat_H,...)` 对 `s_hat_H` 的输入梯度仍需保留。critic 更新使用独立的固定 targets 和 optimizer，检查 actor 更新后世界模型及 critic 参数梯度为空或未被 optimizer 使用。

## 8. Critic 与 GAE

**[论文]** critic 输入为想象状态、目标与 context，估计 32 步之后的价值。只在想象轨迹上计算 GAE return，从 `R_hat_H = V_H` 倒推：

```text
R_hat_t = r_hat_{t+1} + γ(1-d_hat_{t+1})
          * [(1-λ)V_{t+1} + λ R_hat_{t+1}],  t=H-1,...,0
L_V = mean_t (V_ψ(s_hat_t,g_t,ξ) - stopgrad(R_hat_t))²
```

**[建议]** 在 actor 更新的想象结果上一次性生成目标，`detach` 整条目标及 critic 拟合输入，随后多 epoch / minibatch 训练 critic，避免重复构造 32 步 actor 计算图。时间限制 `truncated` 与失败终止 `terminated` 的区别只作用于仿真数据和 reset 处理；不能把时间限制误标为失败终止。论文默认 `λ=0.95`，critic 每次更新 5 个 epoch、每个 epoch 4 个打乱的小批次（附录 D.2）。先以公式的纯张量实现验证，再加入批量化和编译。

## 9. Isaac Lab 接入路径

1. **先锁定版本与任务定义。** 保存 Isaac Lab、RSL-RL、PyTorch 与任务配置版本；定义仿真步长、控制降采样、动作缩放和 PD 增益，并把它们写入 `TaskSpec`。论文控制周期为 0.02 s；不同物理积分器或执行器会影响可比性。
2. **独立适配器。** 在 Isaac Lab 的 `ManagerBasedRLEnv` 或 `DirectRLEnv` 外构建 `IsaacLabEnvAdapter`，暴露完整状态、无噪观测、真实执行动作、DR context、参考帧、终止前状态。Isaac Lab 官方说明其 RL wrapper 是框架接口适配层，现成 [`RslRlVecEnvWrapper`](https://isaac-sim.github.io/IsaacLab/develop/source/api/lab_rl/isaaclab_rl.html) 不保证提供本算法的所有字段。
3. **一致性对照。** 对同一个仿真状态，比较适配器的奖励、终止、无噪观测与想象头的解析实现；比较 `state_codec` 编码/解码、关节顺序、接触和局部坐标变换。
4. **阶段接入。** 先用少量环境和单条短参考动作跑通完整循环，再增加并行数和运动库；检查 replay 的有效窗口比例、32 步漂移和训练稳定性。
5. **评估单列。** 固定种子、无观测噪声/DR 的测试配置与带随机化的训练配置分别运行；不能将训练集运动上的失败数解释为未见动作泛化。

Isaac Lab 的环境包装原则见[官方文档](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/source/how-to/wrap_rl_env.html)。上述适配方式是 **[建议]**，不表示论文已在 Isaac Lab 验证。

## 10. 分阶段实施与交付检查点

| 阶段 | 工作 | 完成条件 |
|---|---|---|
| 0. 规格冻结 | 写 `TaskSpec`、张量形状、时间索引、单位、坐标系、论文/改动记录 | 给定一条手工轨迹，可解释每个存储字段的时刻和来源 |
| 1. 小环境骨架 | Runner、Algorithm、ToyAdapter、Buffer、检查点 | 采集→8 步拟合→32 步 actor/critic 更新一次完整运行且无 NaN |
| 2. 数学验证 | 模型损失、软终止、GAE、梯度有限差分 | 与手算/有限差分吻合；冻结模型后 action 梯度非零 |
| 3. 单动作跟踪 | G1 状态编码、目标库、奖励/终止、历史、DR context | 仿真/解析计算一致；32 步预测误差与策略梯度稳定可监控 |
| 4. Isaac Lab 集成 | 适配器、自动 reset、并行采样、日志 | 真正执行动作与 buffer 记录一致；无跨 episode 窗口 |
| 5. 扩展与对照 | 多动作、返回率采样、论文规模配置、消融 | 固定评估脚本与配置；分别报告训练覆盖、未见动作与随机化鲁棒性 |

## 11. 必须通过的测试与验收标准

**接口与数据：**

- `sample_windows(8)` 所有同 env 样本的 `episode_id` 一致；8 个动作与 8 个下一状态逐项对齐，terminal 后不取下一 episode 的状态。
- 自动 reset 环境能返回真正的终止前 `next_state`。`context` 与仿真器实际 DR 参数一致，episode 内变化有记录或禁止。
- actor 的输入构建器没有 `state`、`ξ`、未来真实状态或 critic 特权字段；历史长度、顺序、reset 填充和参考目标索引可用断言检查。

**数学与梯度：**

- Toy dynamics 上 `∂s_hat_{t+1}/∂a_t` 与有限差分吻合；模型参数冻结时 actor 参数梯度非零，模型参数无更新。
- 关闭第 `t` 步以前的路径后，可观察后续奖励对早期动作梯度改变；正常 32 步求导无意外 `detach`。
- 常数奖励/终止概率的软存活权重与手算一致；`d_hat=0` 和趋近 1 的边界行为正确。GAE 在 `λ=0` 与 `λ=1` 时符合各自递推，目标在 critic 拟合期间固定。
- 在同一批真实状态上，解析奖励/终止与仿真任务输出逐项比对；为数值误差设置任务定义内的容差，不用“训练能跑”替代对齐。

**训练与评估：**

- 记录 1/8/32 步状态误差、终止校准、reward 误差、actor/critic 梯度范数、存活权重、有效 replay 窗口率、真实/想象 return、仿真采集与训练吞吐。
- 恢复检查点后，模型、优化器、学习率、随机数状态、replay 元数据和迭代计数保持一致；继续训练一小段可复现到约定容差。
- 固定评估使用明确的 motion split、DR/noise 开关、终止标准与帧起点；论文数字仅在相同任务、数据和评估协议下比较。

## 12. 主要风险与处理

| 风险 | 早期信号 | 处理 |
|---|---|---|
| 模型误差被 actor 利用 | 想象 return 上升、真实 return 下降 | 对同一 replay 起点做 32 步真实/想象对照；增加近期数据、检查奖励/终止头和接触状态 |
| 观测/状态混用 | actor 用特权字段才收敛；部署失败 | 强类型输入与部署输入测试；critic 与 actor 分开构建 |
| 梯度被截断或错误更新 | `action.grad` 为空、world model 参数改变 | 专门梯度测试；审计 `no_grad`/`detach`/优化器参数组 |
| 时序错一拍 | 奖励或目标对齐残差异常 | 用单步可视化轨迹核对 `s_t,a_t,r_{t+1},s_{t+1},g_t` |
| DR 不可解释 | 同一输入对应多种下一状态 | 记录真实 context；禁用未记录的随机扰动，或扩充状态/输入 |
| 32 步显存与不稳定 | OOM、梯度爆炸 | 先降批量，监控范数并按论文裁剪；优化后重新跑梯度测试 |
| 与论文结果不可比 | 指标看似接近但任务设置不同 | 版本化评估协议，报告所有偏离项，不把 Isaac Lab 迁移当作精确复现 |

## 13. 开始编码前的最小决策清单

1. 确定机器人资产、动作维度、PD/动作缩放、控制周期和单一短参考动作。
2. 给 `state`、`clean_obs`、`noisy_obs_history`、`goal`、`context` 各写一份字段表与张量形状。
3. 确定哪些奖励/观测/终止头使用解析公式，哪些需要监督学习；写明可微性和单位。
4. 固定终止、时间截断、自动 reset 的处理与回放窗口规则。
5. 先跑通 toy 环境和梯度测试，再连接 Isaac Lab 的真实仿真数据。

## 参考资料

- Liao 等，[SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model](https://arxiv.org/pdf/2610.28480)，2026，§3、附录 A–D。本文的“论文设定”均以该版本为准。
- [RSL-RL 官方概览](https://github.com/leggedrobotics/rsl_rl/blob/main/docs/guide/overview.rst)：模块职责与 Runner/Algorithm/Storage 风格。
- [Isaac Lab 官方 RSL-RL wrapper API](https://isaac-sim.github.io/IsaacLab/develop/source/api/lab_rl/isaaclab_rl.html) 与[环境包装说明](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/source/how-to/wrap_rl_env.html)：用于设计未来适配器；接入时应以锁定的 Isaac Lab 版本复核。
