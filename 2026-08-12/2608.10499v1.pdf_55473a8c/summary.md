---
title: "Exploration-Driven Personalized Federated Reinforcement Learning via Intrinsic Motivation"
source: https://arxiv.org/pdf/2608.10499v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:01:33"
field: "联邦强化学习与个性化学习"
keywords: ["federated reinforcement learning", "personalized FL", "intrinsic motivation", "exploration", "random network distillation", "sparse reward", "privacy-preserving"]
innovations: ["将 RND 内在动机与全局新颖性先验聚合结合，实现隐私保护的跨客户端探索协调", "设计低维探索统计摘要与偏差经验采样机制，引导客户端学习全局未探索状态"]
benchmarks: ["MountainCar-v0", "CartPole-sparse"]
---

# 论文速读：Exploration-Driven Personalized Federated Reinforcement Learning via Intrinsic Motivation

## 一句话总结
本文提出 EDPFRL-IM 框架，将内在动机（Random Network Distillation）引入个人化联邦强化学习（PFRL），通过服务器聚合客户端探索统计信息生成全局新颖性先验，引导各客户端在稀疏/延迟奖励环境下进行协调但隐私保护的探索，显著提升样本效率与个性化策略质量。

## 研究问题与动机
- **现有 PFRL 方法忽视探索**：主流 PFRL（如 FedRL、FedAvg-RL、pFedMe）聚焦策略优化与参数聚合，但假设奖励密集或环境同质，在稀疏奖励、非平稳或冷启动场景下容易陷入次优策略。
- **传统探索策略不适于联邦设置**：ϵ-greedy、熵正则化等标准 RL 探索方法依赖全局经验共享或协同调度，而 PFRL 中客户端数据隐私隔离、环境异构（non-IID），难以直接迁移。
- **内在动机在联邦场景下尚未探索**：RND、ICM、count-based 等内在动机方法已在集中式 RL 中验证有效，但在去中心化联邦强化学习中如何实现"既探索又保护隐私"仍是空白。
- **缺乏跨客户端探索协调机制**：若各客户端完全独立探索，会导致冗余试错；但直接共享原始经验或梯度又会破坏隐私，亟需一种低通信开销的协调方案。

## 核心贡献（创新点）
- **提出 EDPFRL-IM 框架**：首次在 PFRL 中集成内在动机驱动的探索机制，使各客户端在稀疏奖励下仍能高效发现有益状态，而非仅依赖外在奖励信号。
- **设计隐私保护的探索统计压缩与全局先验广播协议**：客户端仅上传低维探索摘要（访问计数、状态簇直方图、top-k 新颖状态哈希），服务器聚合生成全局新颖性先验 $\mathcal{P}_{\text{novel}}$ 并广播，实现协调探索而不泄露原始轨迹或策略参数。
- **基于全局先验的偏差经验采样策略**：客户端按 $p_i(s) \propto (1 + \beta \cdot \mathcal{P}_{\text{novel}}(s)) \cdot p_i^{\text{uniform}}(s)$ 重新加权经验缓冲区中的状态采样概率，使各客户端的探索方向与全局未探索区域对齐。
- **系统性实验验证**：在 MountainCar-v0 与 CartPole-sparse（均配备稀疏奖励与异构动力学）上，相比 Local RL、FedRL、FedRL+RND 及多种单智能体/个性化 FL 基线，EDPFRL-IM 取得显著更高的平均回报与更宽的探索覆盖范围。

## 方法详解
### 问题设定
- N 个客户端，每个客户端 $i$ 拥有本地 MDP $\mathcal{M}_i = (S_i, \mathcal{A}, \mathcal{P}_i, R_i, \gamma)$，其中状态空间 $S_i$ 与奖励函数 $R_i$ 因客户端而异，动作空间 $\mathcal{A}$ 和折扣因子 $\gamma$ 全局共享。
- 目标：学习个性化策略 $\pi_i(a|s)$，最大化 $J_i(\pi_i) = \mathbb{E}_{\pi_i}[\sum_{t=0}^{\infty} \gamma^t r_{i,t}]$。
- 策略更新采用梯度上升：$\theta_i \leftarrow \theta_i + \eta \nabla_{\theta_i} J_i(\pi_{\theta_i})$。

### 本地内在动机探索（RND）
- 每个客户端维护一个固定的随机目标网络 $f_{\text{tgt}}$ 和一个可训练预测网络 $f_{\text{pred}}$，均为单层隐藏层（128 维度）的全连接网络。
- 内在奖励定义为下一状态的预测误差：
  $$r_t^{\text{int}} = \| f_{\text{tgt}}(s_{t+1}) - f_{\text{pred}}(s_{t+1}) \|_2^2$$
- 总奖励合并外在与内在信号：
  $$r_t^{\text{total}} = r_t^{\text{ext}} + \alpha_i \cdot r_t^{\text{int}}$$
  其中 $\alpha_i = 0.1$ 控制探索-利用权衡，各客户端可独立设置。
- 策略通过本地 PPO 更新（每轮 E=10 步）优化上述总奖励。

### 联邦探索协调机制
- **探索统计摘要**：每个客户端生成压缩的探索统计量 $\mathcal{E}_i$，如状态访问计数、状态嵌入的频率直方图、或基于内在奖励的 top-k 新颖状态哈希。
- **全局聚合**：服务器汇总所有客户端的摘要：
  $$v_{\text{global}}[k] = \sum_{i=1}^{N} v_i[k]$$
  进而计算全局新颖性先验：
  $$\mathcal{P}_{\text{novel}}(s) = \frac{1}{1 + v_{\text{global}}[\text{cluster}(s)]}$$
  其中 $\text{cluster}(s)$ 将状态映射到离散簇索引。
- **偏差采样引导探索**：客户端 $i$ 按其经验缓冲区 $\mathcal{D}_i$ 中状态的概率分布调整为：
  $$p_i(s) \propto (1 + \beta \cdot \mathcal{P}_{\text{novel}}(s)) \cdot p_i^{\text{uniform}}(s), \quad p_i^{\text{uniform}}(s) = 1/|\mathcal{D}_i|$$
  其中 $\beta = 0.5$ 控制全局先验的引导强度。
- **通信开销**：每轮仅上传/下载低维统计量与先验向量，无原始轨迹、无梯度，严格保隐私。

### 算法流程（Algorithm 1）
1. 每全局轮次 $t$，所有客户端并行：
   - 初始化本地缓冲区 $\mathcal{D}_i$
   - 与环境交互收集 $(s_t, a_t, r_t^{\text{ext}}, s_{t+1})$
   - 计算 RND 内在奖励，构建 $r_t^{\text{total}}$
   - 存储转移至 $\mathcal{D}_i$，用本地 PPO 更新策略
   - 计算并上传探索摘要 $\mathcal{E}_i$
2. 服务器聚合生成 $\mathcal{P}_{\text{novel}}$ 并广播
3. 各客户端按公式 (9) 调整采样分布，进入下一轮

## 实验与结果
### 实验设置
- **环境**：MountainCar-v0（稀疏奖励）与 CartPole-sparse（仅在连续平衡足够长时间后给予奖励），各客户端通过调整重力（0.0025–0.006）、摩擦力、奖励塑形等参数制造异构性，构成极端 non-IID 场景。
- **联邦配置**：N=10 客户端，T=100 全局通信轮次，每轮 E=10 次本地 PPO 更新。
- **网络结构**：策略网络为双层全连接（64 隐层，ReLU）；RND 目标/预测网络为单层 128 维。
- **优化器**：Adam，学习率 $3 \times 10^{-4}$，折扣因子 $\gamma = 0.99$。
- **基线方法**：
  - 联邦类：Local RL、FedRL、FedRL+RND、FedAvg-RL、pFedMe、FedPer++
  - 单智能体内在动机：RND、ICM、Count-Based Exploration（CBE）

### 主要结果
- **平均回报（Table III）**：EDPFRL-IM 在 MountainCar-v0 达 **0.76**，CartPole-sparse 达 **0.74**；对比 FedRL+RND（0.48 / 0.52）提升约 **58%/42%**，对比最优单智能体 CBE（0.39 / 0.49）提升约 **95%/51%**。
- **逐客户端个性化表现（Table I）**：10 个客户端在两个环境中均稳定优于 FedRL，且方差更低，表明个性化与探索协调同时获益。
- **消融实验（Table II）**：移除全局协调（β=0）时 MountainCar 回报从 0.80 降至 0.63；完全去掉 RND 时仅 0.40，证明内在动机与联邦协调各自贡献显著。
- **探索覆盖范围（Fig. 3）**：EDPFRL-IM 访问的状态空间比例显著高于 FedRL 与 FedRL+RND。
- **冷启动鲁棒性（Fig. 4）**：第 5 轮引入新客户端后，EDPFRL-IM 凭借全局先验快速适应，回报曲线明显优于基线。

### 最强结果
MountainCar-v0 最终平均回报 **0.76**，较次优基线 FedRL+RND 提升 **58%**；CartPole-sparse 为 **0.74**，较次优基线提升 **42%**。

## 相关工作脉络
- **Federated RL（FedRL, FedAvg-RL, pFedMe, FedPer++）**：这些工作聚焦参数聚合或个性化正则化，但未引入探索引导机制，在稀疏奖励场景下性能明显落后于 EDPFRL-IM。
- **Intrinsic Motivation in Centralized RL（RND, ICM, CBE）**：RND、ICM、CBE 在集中式 RL 中证明了好奇心驱动探索的有效性，但均为单智能体方法，无法直接迁移至联邦隐私场景；本文首次将其与联邦聚合框架结合。
- **FedRL with RND（FedRL+RND）**：将 RND 内在地用于本地奖励但无跨客户端协调，相当于"各自的 curiosity"，性能低于引入全局新颖性先验的 EDPFRL-IM。
- **Personalized FL（pFedMe, FedPer++）**：解决异构数据分布下的个性化问题，但未考虑强化学习中探索-利用的核心挑战，不适用于稀疏奖励的 sequential decision-making。
- **定位差异**：本文填补了 PFRL 在探索机制层面的空白，核心创新是"用压缩统计量实现隐私保护的跨客户端探索协调"，而非简单的内在奖励叠加或参数个性化。

## 局限性与未来方向
- **实验环境有限**：仅在 MountainCar-v0 与 CartPole-sparse 两个经典连续控制任务上验证，缺乏高维视觉输入或复杂离散动作空间的测试。
- **统计压缩形式未充分展开**：论文提到访问计数、直方图、top-k 哈希等多种摘要形式，但未系统比较哪种压缩方式效果最佳或通信开销最低。
- **缺乏理论保证**：未提供收敛性分析、样本复杂度上界或对非平稳环境的适应性理论保证。
- **超参数敏感**：$\alpha_i$（内在奖励权重）与 $\beta$（全局先验偏差强度）需人工设定，缺乏自适应机制。
- **未来方向**：可扩展至多模态/视觉环境、设计自适应超参数调节策略、开展理论收敛分析、以及探索更高效的压缩协议（如差分隐私下的统计摘要）。

## 研究启发与可借鉴点
- **联邦探索协调的设计思路**：将"状态访问统计"作为隐私友好的中间表示进行聚合，既避免了原始数据共享，又实现了跨客户端的探索知识传递，该范式可迁移至其他联邦学习场景（如联邦多臂老虎机、联邦离线 RL）。
- **RND 内在奖励与联邦聚合的结合方式**：$r^{\text{total}} = r^{\text{ext}} + \alpha \cdot r^{\text{int}}$ 的奖励重塑结构简洁且模块化，可无缝嵌入任意策略梯度方法（PPO、SAC 等）。
- **偏差采样（biased replay）机制**：公式 (9) 的采样重加权是一种轻量的探索引导手段，无需修改策略网络结构，仅改变数据使用顺序即可实现"全局指导的本地探索"，实现成本低。
- **冷启动适应性评估视角**：论文通过中途注入新客户端评估方法适应性，这一实验设计对实际联邦系统的动态扩缩容场景具有参考价值。
- **可结合本团队方向的机会**：若团队研究涉及联邦离线强化学习或多智能体协作，可将全局新颖性先验与行为克隆/离线策略优化结合，解决"无在线探索条件下如何利用历史数据发现未知状态"的问题。

## 关键术语表
**Personalized Federated Reinforcement Learning (PFRL)**：在联邦设置中，各客户端学习针对自身环境异质性个性化的强化学习策略，同时通过聚合通信共享探索知识。

**Intrinsic Motivation（内在动机）**：RL 中为鼓励探索而设计的外部奖励辅助信号，不依赖环境提供的稀疏外在奖励，常见形式包括 RND 预测误差、ICM 逆动力学不一致性等。

**Random Network Distillation (RND)**：一种内在动机方法，维护固定随机目标网络与可训练预测网络，用两者对下一状态嵌入的预测误差作为 novelty 奖励。

**Global Novelty Prior（全局新颖性先验）**：由服务器聚合所有客户端探索统计量后生成的状态级先验分布，值越低表示该状态越未被探索，用于引导客户端偏向采样的概率分布。

**Exploration Summary（探索统计摘要）**：客户端上报给服务器的低维压缩信息，如状态访问计数、频率直方图或 top-k 新颖状态哈希，用于在保护隐私的前提下传递探索知识。

**Biased Experience Sampling（偏差经验采样）**：按全局新颖性先验调整经验缓冲区中状态的重采样概率，使客户端更频繁地回放未探索状态对应的转移。

**Cold-start Client（冷启动客户端）**：在训练中途新加入的客户端，尚无本地经验积累，依赖全局先验快速适配所在环境。

**Non-IID MDP（非独立同分布 MDP）**：各客户端环境动力学与奖励函数存在显著差异的设定，是 PFRL 区别于标准联邦学习的核心特征。

## 可复现要素
- **数据集/环境**：MountainCar-v0、CartPole-sparse（OpenAI Gym 标准环境，已公开）
- **代码是否开源**：论文未提及代码仓库链接
- **权重是否开源**：论文未提及
- **关键超参**：N=10 客户端，T=100 全局轮次，E=10 本地 PPO 步数，$\alpha_i = 0.1$，$\beta = 0.5$，$\gamma = 0.99$，学习率 $3 \times 10^{-4}$，策略网络 2 层×64 隐层，RND 网络单层×128 隐层
- **实现细节**：Adam 优化器，ReLU 激活，RND 目标网络固定不更新

---
