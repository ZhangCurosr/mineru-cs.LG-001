---
title: "Exploration-Driven Personalized Federated Reinforcement Learning via Intrinsic Motivation"
source: https://arxiv.org/pdf/2608.10499v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:01:43"
field: "联邦强化学习与个性化学习"
keywords: ["federated reinforcement learning", "personalized FL", "intrinsic motivation", "random network distillation", "exploration", "sparse reward"]
innovations: ["提出EDPFRL-IM框架，将RND内在好奇心与联邦新颖性先验协调机制结合", "设计隐私保护的探索统计摘要协议，以低维压缩信息替代原始经验/梯度交换", "引入全局新颖性先验引导的偏差经验采样，实现跨客户端协调探索"]
benchmarks: ["MountainCar-v0", "CartPole-sparse"]
---

# 论文速读：Exploration-Driven Personalized Federated Reinforcement Learning via Intrinsic Motivation

## 一句话总结
本文提出了 EDPFRL-IM 框架，将 RND 内在好奇心驱动与联邦新颖性先验协调机制相结合，使各客户端在隐私保护前提下实现高效协同探索，显著提升稀疏奖励环境下个性化策略的学习效率与样本效率。

## 研究问题与动机
- PFRL 现有方法过度依赖外部奖励信号进行策略优化，忽视了非平稳/稀疏奖励环境中的探索机制，导致冷启动或动态环境下的次优策略。
- 标准 RL 探索策略（如 ε-greedy、熵正则化）在联邦设置下效果有限，因为客户端独立运行且受隐私约束，缺乏跨客户端的探索协调。
- 集中式 RL 中有效的内在动机方法（RND、ICM、count-based）在联邦场景下尚未被充分探索，缺乏隐私保护且通信高效的跨客户端探索机制。
- 现有联邦 RL 基线（FedRL、FedAvg-RL、pFedMe 等）假设同构或监督式设置，无法有效应对异构客户端的非 IID 分布。

## 核心贡献（创新点）
1. **提出 EDPFRL-IM 框架**：将 RND 内在好奇心集成到个性化联邦 RL 中，使各客户端在本地通过内在奖励自主探索，同时保持完全隐私。与 FedRL 等仅依赖外部奖励的方法本质不同。
2. **设计通信高效的探索协调协议**：客户端仅上传低维探索统计摘要（访问计数、状态嵌入直方图或 top-k 新颖状态哈希），服务器聚合后广播全局新颖性先验，而非交换原始经验或本地梯度。区别于 FedRL+RND 的无协调 intrinsic reward 方案。
3. **引入基于全局新颖性先验的偏差经验采样**：客户端利用广播的全局先验调整经验回放采样概率，引导各自关注全局未充分探索的状态空间，与单智能体 RND/ICM/CBE 的纯本地探索形成对比。
4. **系统性实验验证**：在 MountainCar-v0 和 CartPole-sparse 稀疏奖励 benchmark 上，EDPFRL-IM 在最终平均回报上分别达 0.76 和 0.74，相比最强基线 FedRL+RND（0.48/0.52）提升约 58%/42%，并验证了对冷启动客户端的快速适应能力。

## 方法详解
- **问题设定**：N 个客户端，每个本地 MDP $\mathcal{M}_i = (S_i, \mathcal{A}, \mathcal{P}_i, R_i, \gamma)$，策略 $\pi_i(a|s)$ 最大化 $J_i(\pi_i) = \mathbb{E}[\sum \gamma^t r_{i,t}]$，参数通过梯度上升更新。
- **本地内在奖励（RND）**：每个客户端维护固定随机目标网络 $f_{tgt}$ 和可训练预测网络 $f_{pred}$，对下一状态 $s_{t+1}$ 计算预测误差作为内在奖励：$r_t^{int} = \|f_{tgt}(s_{t+1}) - f_{pred}(s_{t+1})\|_2^2$，总奖励 $r_t^{total} = r_t^{ext} + \alpha_i \cdot r_t^{int}$，其中 $\alpha_i = 0.1$。
- **探索统计摘要**：客户端生成低维压缩统计量 $\mathcal{E}_i$（如访问计数直方图、top-k 新颖状态哈希），发送至服务器。
- **全局新颖性先验**：服务器聚合得 $v_{global}[k] = \sum_i v_i[k]$，计算 $\mathcal{P}_{novel}(s) = \frac{1}{1 + v_{global}[\text{cluster}(s)]}$ 并广播给所有客户端（$\beta = 0.5$）。
- **偏差经验采样**：客户端按 $p_i(s) \propto (1 + \beta \cdot \mathcal{P}_{novel}(s)) \cdot p_i^{uniform}(s)$ 重采样经验缓冲区，引导学习聚焦于全局新颖状态。
- **流程**：每轮客户端本地交互→计算内在奖励→PPO 策略更新→发送探索摘要→服务器聚合广播先验→客户端更新采样策略，全程 100 轮通信，每轮 10 次本地 PPO 更新。

## 实验与结果
- **环境**：MountainCar-v0（重力 0.0025–0.006 异构）、CartPole-sparse（极杆质量/摩擦各异，仅持续平衡后给奖励），N=10 客户端，T=100 轮，E=10 本地 PPO 步，隐藏层 64×2，γ=0.99，lr=3×10⁻⁴。
- **主要结果**：EDPFRL-IM 在 MountainCar-v0 获 0.76、CartPole-sparse 获 0.74，均位列第一；相对最强基线 FedRL+RND（0.48/0.52）分别提升 **58.3%** 和 **42.3%**；相对无协调的 FedRL（0.40/0.35）提升 **90%** / **117%**。
- **消融**：移除全局先验（β=0）时 MountainCar 仅 0.63/CartPole 0.58；完全去掉 RND 时跌至 0.40/0.35，证实协调探索与内在激励缺一不可。
- **探索覆盖**：EDPFRL-IM 状态空间覆盖率显著高于 FedRL 和 FedRL+RND（Fig.3）。
- **冷启动鲁棒性**：第 5 轮加入冷启动客户端，EDPFRL-IM 利用全局先验实现快速平滑适应，优于仅依赖本地经验的 FedRL/FedRL+RND（Fig.4）。
- **跨方法对比**（Table III）：超越 RND(0.31/0.32)、ICM(0.34/0.36)、CBE(0.39/0.49)、FedAvg-RL(0.38/0.42)、pFedMe(0.40/0.45)、FedPer++(0.41/0.44)。

## 相关工作脉络
1. **FedRL [2]**：联邦 RL 全局策略聚合基线，无个性化、无内在激励探索；本文在其基础上引入客户端个性化与 RND 内在奖励。
2. **FedAvg-RL [4] / pFedMe [5] / FedPer++ [12]**：个性化联邦学习算法，但面向监督/同构设置，缺乏对稀疏奖励 RL 中探索问题的处理；本文定位为其在 FRL 稀疏奖励场景的延伸。
3. **RND [6]**：集中式内在激励方法，通过预测误差驱动探索；本文将其适配至联邦设置并加入跨客户端协调机制，突破其纯本地探索的局限。
4. **ICM [7] / CBE [8]**：其他集中式内在动机方法；本文指出其在联邦隐私约束下未充分探索，并以压缩摘要+全局先验方式替代其直接应用。
5. **FedRL+RND**：本文设计的直接对比基线，仅本地加 RND 无协调；本文框架通过服务器聚合 novelty prior 实现了 FedRL+RND 所缺失的跨客户端探索一致性。

## 局限性与未来方向
- 实验仅在两个经典 2D 连续控制 benchmark（MountainCar、CartPole）上验证，未见更复杂高维环境（如 MuJoCo、Atari）或真实世界部署实验。
- 探索统计摘要的具体形式（直方图 vs. top-k 哈希）和聚合函数的设计较为简化，未系统探讨不同压缩策略对隐私-效用 trade-off 的影响。
- β 和 α_i 等超参依赖人工设置，未讨论自适应调节机制或敏感性分析。
- 客户端数量 N=10 规模较小，未评估大规模分布式场景下的通信开销与收敛性质。
- 未来可扩展至部分可观察 MDP（POMDP）、多任务联邦 RL 或结合 ICM 等其它内在动机机制。

## 研究启发与可借鉴点
1. **隐私保护型探索协调范式**：用"压缩探索统计→服务器聚合→广播先验→客户端偏差采样"的协议替代梯度/原始数据交换，为联邦好奇驱动学习提供了通用模板，可迁移至其他基于内在动机的多智能体协作场景。
2. **全局新颖性先验引导经验重采样**：公式 $p_i(s) \propto (1+\beta\cdot\mathcal{P}_{novel}(s))\cdot p_i^{uniform}(s)$ 巧妙地将服务器端聚合的全局探索缺口反馈至本地训练循环，思路可直接复用于任何基于经验回放的离线/在线 RL 算法。
3. **冷启动客户端快速适应评估**：论文在训练中途注入冷启动客户的实验设计，为评估联邦系统的动态扩展性提供了可借鉴的 benchmark 方案。
4. **与团队方向结合机会**：若团队研究涉及机器人/医疗设备的联邦 RL，EDPFRL-IM 的内在激励+隐私协调框架可直接复用；可进一步探索将其与 pFedMe 式元优化或个人化层结合，构建更强的个性化 FRL 系统。

## 关键术语表
- **Personalized Federated Reinforcement Learning (PFRL)**：在联邦学习框架下为每个客户端学习个性化策略的强化学习方法，兼顾隐私与个性化。
- **Intrinsic Motivation**：源自心理学/RL 的内在驱动机制，通过自我生成的好奇心奖励（如预测误差）替代仅依赖外部稀疏奖励进行探索。
- **Random Network Distillation (RND)**：一种内在动机方法，训练预测网络拟合固定随机目标网络的输出，预测误差即为内在奖励信号。
- **Global Novelty Prior**：服务器聚合各客户端探索统计后计算的全局状态新颖度分布，用于指导各客户端优先采样未充分探索的状态。
- **Exploration Summary**：客户端发送给服务器的低维压缩统计（如访问计数直方图、top-k 状态哈希），用于跨客户端协调探索而无需暴露原始数据。
- **Sparse-Reward Environment**：环境仅在特定条件下提供非零外部奖励的强化学习任务，探索效率成为学习成败的关键。
- **Cold-Start Client**：训练中途加入的新客户端，缺乏历史经验，依赖全局知识快速适应本地环境。

## 可复现要素
- **数据集/环境**：MountainCar-v0、CartPole-sparse（OpenAI Gym 标准环境），**论文未明确说明是否公开发布代码**。
- **代码开源情况**：论文未提及 GitHub/代码仓库链接；若需复现需自行实现 EDPFRL-IM 框架。
- **关键超参**：N=10 客户端，T=100 通信轮，E=10 本地 PPO 步；策略网络 2 层×64 ReLU；RND 网络 1 层×128；α_i=0.1，β=0.5，γ=0.99，lr=3×10⁻⁴，Adam 优化器。
