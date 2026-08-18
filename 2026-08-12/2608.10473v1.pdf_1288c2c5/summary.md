---
title: "CRITIC-FREE PRETRAINING FOR EFFICIENT ONLINE REINFORCEMENT LEARNING FINE-TUNING"
source: https://arxiv.org/pdf/2608.10473v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:58:45"
---

# 论文速读：CRITIC-FREE PRETRAINING FOR EFFICIENT ONLINE REINFORCEMENT LEARNING FINE-TUNING

## 一句话总结
论文提出 **Critic-Free Pretraining (CFP)** 范式：在离线阶段完全不训练 critic，仅用行为克隆训练 flow-based actor；在线微调前从零初始化 fresh critic 并用少量离线数据 warm-up 校准，从而避免离线 critic 的分布诱导悲观偏差拖累在线策略改进。该方法可无缝集成到多种主流 O2O 算法，在稀疏奖励操作任务上实现相当或更优的在线微调性能，尤其在 Cube Triple 等挑战性任务上提升显著。

## 研究问题与动机
1. **离线 critic 在线失配问题**：传统 O2O 方法直接复用离线训练好的 critic，但离线数据集分布与在线环境存在显著偏移，导致继承的 critic 值估计与快速演化的在线策略分布失配，引发不准确的策略更新与探索低效。
2. **离线数据集的悲观偏差来源**：现代离线数据集高度由次优、随机或失败轨迹主导，高质量成功演示稀缺；TD 自举学习使低回报从这些轨迹向后传播，累积 critic 的保守性（pessimism），即使没有显式正则化也会产生系统性低估。
3. **现有缓解方法的局限**：CQL、Cal-QL 等通过保守正则化或 scale 校准间接处理值估计偏差，但仍无法根除因分布偏移导致的 value mismatch；RLPD、WSRL、OPT 等方法虽引入 replay 设计或新 critic 初始化，但结构复杂或依赖在线预训练阶段。
4. **对离线阶段角色的重新审视**：论文的核心洞察是，离线阶段的主要价值是训练一个能引导智能体在易采集有效轨迹区域内采样的 actor，而非提供精确的 value network；因此 actor 与 critic 预训练应当解耦。

## 核心贡献（创新点）
1. **提出 Critic-Free 离线训练范式**：完全摒弃离线阶段的 critic 训练，仅通过行为克隆损失更新 flow-based actor；与已有 O2O 工作（如 FQL、QCFQL 等）同时更新 actor/critic 的设定本质不同，本文证明离线 critic 训练反而会损害后续在线微调，该方法同时降低计算开销与内存需求。
2. **设计 fresh critic 初始化与离线 warm-up 校准机制**：在线微调前从零初始化 critic，仅在离线数据上进行 $N_{\text{warmup}}$ 步（默认 10K）的联合更新；区别于 WSRL（仅用 online 数据校准）与 OPT（双 critic 调度组合），CFP 在校准阶段完全保留离线 actor 的分布先验，实现更简单的结构。
3. **理论分析与玩具 MDP 验证 value mismatch 机制**：通过构造离线数据集偏向负终端、在线策略偏向正终端的 16 状态 MDP，量化证明 offline-trained critic 的 Q 值结构偏离在线真实 Q 值（RMSE 更高），而 fresh critic 在在线微调过程中更快收敛到更接近真实值的分布；与 CQL/Cal-QL 等事后约束方法不同，CFP 从源头避免偏差继承。
4. **在多种主流 flow-based O2O 算法上验证通用性**：将 CFP 范式集成到 FQL、QC、QCFQL、QCFQL-nstep 四个基线，在 OGBench（5 个稀疏奖励操作域）与 Robomimic（3 个任务）上取得相当或更优性能，尤其在 Cube Triple、Puzzle 4×4 等挑战性任务上表现突出，证明 CFP 是一种独立于底层算法的通用优化思路。

## 方法详解
- **离线阶段（Critic-Free Pretraining）**：仅训练行为克隆 actor，loss 为流式策略蒸馏损失（Eq. 7）：
  $$L_\pi(\omega) = \alpha \mathbb{E}_{s \sim \mathcal{D},\, z \sim \mathcal{N}(0, I^d)} \left[ \| \mu_\omega(s, z) - \mu_\theta(s, z) \|_2^2 \right]$$
  不包含任何 critic 参数更新；actor 的更新完全不依赖 critic 梯度，达到标准 BC 效果。
- **在线微调前的初始化**：丢弃所有离线 critic 权重，从零（randomly initialized）创建新的 critic $Q_\phi$，避免继承任何与离线分布绑定的价值偏差。
- **Warm-up 阶段**：在离线数据集 $\mathcal{D}$ 上进行 $N_{\text{warmup}}$ 步联合更新，actor loss 包含 BC 项与 Q-loss（Eq. 20）：
  $$L_\pi(\omega) = \alpha \mathbb{E}_{s \sim \mathcal{D},\, z \sim \mathcal{N}(0, I^d)} \left[ \| \mu_\omega(s, z) - \mu_\theta(s, z) \|_2^2 \right] - \mathbb{E}_{s \sim \mathcal{D},\, a \sim \pi_\omega(\cdot|s)} [Q_\phi(s, a)]$$
  critic loss 为标准 TD 误差（Eq. 21）：
  $$L_Q(\phi) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( Q_\phi(s,a) - [r + \gamma Q_{\bar{\phi}}(s', a')] \right)^2 \right]$$
  该阶段虽短（仅占总训练步数的 ~0.5%），但使 fresh critic 快速获得初步 Q 值估计能力。
- **在线阶段**：在真实环境中交互收集 $(s,a,r,s') \sim \pi_\theta$ 并追加到 batch $\mathcal{B}$，actor/critic loss 形式与 warm-up 相同，仅数据来源切换为在线经验（Eq. 22–23）。
- **关键设计选择**：消融实验（Fig. 6）表明 warm-up 期间保留 Q-loss 对 actor 的梯度回传仍然有益，即使 fresh critic 初始不准确；warm-up 步数对性能相对鲁棒（Fig. 7），0 步时在线微调可能出现训练不稳定，故取 10K 步平衡稳定性与计算效率（A800 GPU 约 30 秒）。
- **算法兼容性**：CFP 可直接替换任意 flow-based O2O 算法的离线阶段（仅保留 actor BC 训练），warm-up 与在线阶段沿用原算法的 actor/critic 更新公式，无需修改网络结构或调度逻辑。

## 实验与结果
- **数据集与环境**：OGBench（Cube-Double/Triple/Quadruple、Puzzle-4×4、Scene，共 5 个稀疏奖励操作域，使用 default play 数据集；Cube Quadruple 使用 100M 大版本）与 Robomimic（Lift、Can、Square，使用 Multi-Human 数据集）。
- **评估基线**：FQL、QC、QCFQL、QCFQL-nstep 四种 flow-based 算法的原始 O2O 实现。
- **主要结果**：
  - **OGBench**：所有 CFP 变体在各任务上均达到相当或更优性能；Cube Triple 任务提升最显著，如 QCFQL-nstep-CFP 在 Task 1 从 ~33% 提升至 100%，Task 4 从 75.6% 提升至 91.6%；FQL-CFP 在 Puzzle 4×4 上从 <0.4 跃升至 ~0.7 成功率终点，超越所有 QCFQL/QC 变体。
  - **Robomimic**：CFP 变体在 Lift、Can 上与 O2O 相当；Square 任务 QCFQL-nstep-CFP 相对 O2O 提升显著（表 5：最终成功率 80.4% vs 82.8% 起始基线）。
  - **最强结果**：Cube Triple Task 1 上 QCFQL-nstep-CFP 达到 100% 成功率（O2O 基线 100%，但 CFP 更早收敛）；Cube Quadruple 多任务稳定达到 100%；Cube Triple Task 4 上 QCFQL-nstep-CFP 达 91.6%（O2O 75.6%）。
- **消融结论**：保留 warm-up Q-loss 对 actor 梯度整体提升性能（Fig. 6）；warm-up 步数敏感性较低，10K 步为效率与稳定性的平衡点（Fig. 7）；无 Q-loss 时 CFP 对 warm-up 长度更敏感，性能随步数增加下降。
- **玩具 MDP 验证**：16 状态 4×4 网格 MDP，离线数据集偏向负终端（state 0），在线策略偏向正终端（state 15）；CFP critic 的 RMSE 始终低于 O2O，且 Q 值结构更接近在线真实 Q 值分布（Fig. 3、Fig. 12）。

## 相关工作脉络
1. **CQL (Kumar et al., 2020)**：通过保守正则化惩罚离线数据不支持的动作以降低 overestimation；CFP 不依赖事后保守约束，而是从源头避免离线 critic 训练带来的偏差继承。
2. **Cal-QL (Nakamoto et al., 2023)**：约束保守值不低于参考策略值以建立合理 scale；CFP 认为问题根源在于值估计与在线分布的结构性失配，绕过根源而非校准 scale。
3. **BCQ / TD3+BC / AWAC**：通过显式约束策略不偏离数据集分布（condition generative model、BC 项、advantage-weighted ML）；CFP 保留离线 actor 的分布先验，但不引入额外保守惩罚，结构更轻量。
4. **RLPD (Ball et al., 2023)**：联合采样固定离线数据与在线经验并配合 critic 正则化；CFP 同样使用混合数据但在 warm-up 阶段完全基于离线数据，不依赖在线回放buffer。
5. **WSRL (Zhou et al., 2025)**：仅在 online 数据上 warm-up 校准预训练 critic；CFP 的 warm-up 在离线数据上进行，保护 actor 的离线分布先验不被在线数据污染。
6. **OPT (Shin et al., 2025)**：引入新初始化 critic + 在线预训练阶段，fine-tuning 时调度组合离线/在线 pretrained critic；CFP 仅依赖单个 fresh critic 加离线 warm-up，无调度复杂度。

## 局限性与未来方向
- CFP 在部分 Robomimic 任务上并未显著优于传统 O2O，说明丢弃离线 critic 的收益依赖于继承值估计是否存在显著分布诱导失配；未来需开发诊断方法预测何时此类失配会发生。
- Warm-up 阶段完全依赖离线数据，若离线数据质量极差或覆盖度极低，fresh critic 可能难以快速校准；可探索结合少量 online 数据或自适应 warm-up 调度策略。
- 玩具 MDP 实验控制了分布偏移的极端情形，实际复杂连续控制任务中的 value mismatch 机制仍需更深入的理论分析（如 Q-value drift 速率建模）。
- 当前实验集中在 flow-based policy 设置，未来可扩展至基于 value-based 或 policy gradient 的主流 O2O 算法（如 SAC、TD3+、Dreamer 系列）验证普适性。

## 研究启发与可借鉴点
1. **方法论迁移**：Critic-Free 思想可迁移至其他 offline-to-online 迁移场景（如 robotics、game-playing、LLM alignment），只要存在显著分布偏移风险，跳过 critic 预训练并用短暂 warm-up 替代可能是更稳健的选择。
2. **实验论证策略**：用 toy MDP 显式构造离线/在线 Q 分布不匹配来验证假设，比仅在复杂 benchmark 上报告数字更具说服力；可借鉴此“构造控制实验 + 可视化对比 + RMSE 量化”的论证链路。
3. **超参设计启发**：warm-up 阶段保留 Q-loss 对 actor 的梯度被证明有效，提示在 actor-critic 解耦训练时仍需谨慎处理梯度回传路径，不可简单移除；同时 warm-up 步数鲁棒性降低了对精细调参的依赖。
4. **创新机会**：将 CFP 范式与 Diffusion Policy 等更 expressive 的 policy class 结合，或在 large-scale embodied RL pipeline 中替换标准 offline critic pretraining stage，有望进一步提升 sample efficiency。
5. **诊断工具方向**：开发 offline critic quality proxy metric（如 OOD 覆盖率估计、value variance 分析、Q-value bootstrap error 监测），可在训练前预测是否适用 CFP，降低 trial-and-error cost。

## 关键术语表
**Offline-to-Online (O2O) RL**：先在静态离线数据集上预训练策略，再利用在线环境交互进行微调的强化学习范式，旨在缓解纯在线学习样本效率低下的问题。
**Critic-Free Pretraining (CFP)**：本文提出的离线训练范式，完全跳过 critic 训练，仅通过行为克隆
