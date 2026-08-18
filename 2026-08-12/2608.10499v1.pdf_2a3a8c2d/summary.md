---
title: "Exploration-Driven Personalized Federated Reinforcement Learning via Intrinsic Motivation"
source: https://arxiv.org/pdf/2608.10499v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:01:53"
field: "联邦强化学习"
keywords: ["federated reinforcement learning", "personalized federated learning", "intrinsic motivation", "random network distillation", "exploration", "sparse reward", "cold-start adaptation"]
innovations: ["将RND内在动机引入个性化联邦强化学习，设计客户端本地好奇探索+服务器全局新奇先验的双层协调架构", "提出压缩探索摘要的隐私保护通信协议，服务器仅聚合低维统计量即可广播全局探索导向", "基于全局新奇先验的有偏经验采样策略，使客户端在保持个性化策略的同时协同覆盖更广状态空间"]
benchmarks: ["MountainCar-v0", "CartPole-sparse"]
---

# 论文速读：Exploration-Driven Personalized Federated Reinforcement Learning via Intrinsic Motivation

## 一句话总结
提出 EDPFRL-IM 框架，将内在动机（Random Network Distillation, RND）引入个性化联邦强化学习，通过客户端本地好奇心探索 + 服务器聚合全局新奇先验的双层机制，解决稀疏奖励与非平稳环境下联邦客户端探索效率低、冷启动适应慢的问题，同时保持隐私与低通信开销。

## 研究问题与动机
- **核心问题**：现有 PFRL 方法过度依赖外部奖励信号优化策略，忽视探索；在稀疏/延迟奖励和非平稳环境中，客户端易陷入次优策略。
- **现有方法不足 1**：标准 RL 探索策略（ε-greedy、熵正则化）在联邦场景下效果差——客户端独立运行，隐私约束限制了跨客户端协调探索。
- **现有方法不足 2**：内在动机方法（RND、ICM、count-based）已在中心化 RL 中验证有效，但其在联邦场景下的探索协调机制几乎未被研究。
- **现有方法不足 3**：冷启动客户端缺乏全局探索知识的引导，导致适应速度慢；现有 FedRL 基线无法有效支持早期学习者。

## 核心贡献（创新点）
- **提出 EDPFRL-IM 框架**：首次将内在好奇心驱动的探索机制系统性地集成到个性化联邦强化学习架构中，兼顾局部探索效率与全局协调。
- **设计通信高效的隐私保护协调协议**：客户端仅上传压缩的探索统计摘要（如访问计数/频率直方图/top-k 新奇状态哈希），服务器聚合后广播全局新奇先验；无需传输原始经验或梯度，显著降低通信与隐私风险。
- **基于全局新奇先验的有偏经验采样**：在客户端策略更新时引入优先级采样 $p_i(s) \propto (1 + \beta \cdot \mathcal{P}_{\mathrm{novel}}(s)) \cdot p_i^{\mathrm{uniform}}(s)$，使各客户端在保持个性化策略的同时，协同探索全局未充分覆盖的状态空间。
- **系统性实验验证**：在 MountainCar-v0 与 CartPole-sparse 两个稀疏奖励基准上，EDPFRL-IM 在平均回报、样本效率、冷启动适应和探索覆盖面上均显著优于全部基线。

## 方法详解
### 问题设定
- N 个客户端，每个客户端 $C_i$ 与本地 MDP $\mathcal{M}_i = (S_i, \mathcal{A}, \mathcal{P}_i, R_i, \gamma)$ 交互，其中 $S_i, R_i$ 因客户端而异（非 IID），$\mathcal{A}, \gamma$ 共享。
- 客户端学习目标：$J_i(\pi_i) = \mathbb{E}_{\pi_i}\left[\sum_{t=0}^{\infty} \gamma^t r_{i,t}\right]$，参数更新 $\theta_i \leftarrow \theta_i + \eta \nabla_{\theta_i} J_i(\pi_{\theta_i})$。

### 客户端本地内在动机探索（RND）
- 每个客户端维护一对固定目标网络 $f_{\mathrm{tgt}}$ 和可训练预测器 $f_{\mathrm{pred}}$（均为单隐藏层，128 维）。
- 内在奖励：$r_t^{\mathrm{int}} = \|f_{\mathrm{tgt}}(s_{t+1}) - f_{\mathrm{pred}}(s_{t+1})\|_2^2$，鼓励探索难以预测的新奇状态。
- 总奖励：$r_t^{\mathrm{total}} = r_t^{\mathrm{ext}} + \alpha_i \cdot r_t^{\mathrm{int}}$，$\alpha_i = 0.1$ 控制探索-利用权衡。

### 联邦协调：探索摘要与全局新奇先验
- 客户端生成压缩探索统计 $\mathcal{E}_i$（如状态聚类访问计数直方图），上传至服务器。
- 服务器聚合：$v_{\mathrm{global}}[k] = \sum_{i=1}^N v_i[k]$，计算全局新奇先验：$\mathcal{P}_{\mathrm{novel}}(s) = \frac{1}{1 + v_{\mathrm{global}}[\mathrm{cluster}(s)]}$。
- 服务器广播 $\mathcal{P}_{\mathrm{novel}}$ 给所有客户端。

### 有偏经验采样策略更新
- 客户端从本地缓冲区 $\mathcal{D}_i$ 中按 $p_i(s) \propto (1 + \beta \cdot \mathcal{P}_{\mathrm{novel}}(s)) \cdot p_i^{\mathrm{uniform}}(s)$ 概率采样经验，$\beta = 0.5$ 控制全局先验影响强度。
- 基于采样经验执行本地 PPO 更新（每轮 E=10 步），再进入下一联邦通信轮。

### 算法流程（Algorithm 1）
1. 每轮每客户端初始化本地缓冲区 $\mathcal{D}_i$；
2. 与环境交互收集 $(s_t, a_t, r_t^{\mathrm{ext}}, s_{t+1})$，计算 $r_t^{\mathrm{int}}$，存储总奖励经验；
3. 本地 PPO 更新策略 $\pi_i$；
4. 生成探索摘要 → 上传服务器；
5. 服务器聚合 → 广播全局新奇先验；
6. 客户端更新采样策略 → 进入下一轮。

## 实验与结果
### 实验设置
- **环境**：Modified MountainCar-v0（重力 0.0025~0.006、摩擦力、距离/速度奖励塑形）、CartPole-sparse（杆质量、摩擦力异质，需维持平衡最低时长才获奖励）。
- **联邦配置**：N=10 客户端，T=100 轮全局通信，每轮 E=10 次本地 PPO 更新；策略网络为 2 层全连接（64 hidden, ReLU）；RND 网络单隐藏层 128 维；$\gamma=0.99$，learning rate=$3\times10^{-4}$，Adam 优化。

### 主要结果
| 方法 | MountainCar-v0 | CartPole-sparse |
|------|---------------|-----------------|
| RND [6] | 0.31 | 0.32 |
| ICM [7] | 0.34 | 0.36 |
| CBE [8] | 0.39 | 0.49 |
| FedAvg-RL [4] | 0.38 | 0.42 |
| pFedMe [5] | 0.40 | 0.45 |
| FedPer++ [12] | 0.41 | 0.44 |
| FedRL+RND | 0.48 | 0.52 |
| **EDPFRL-IM（Ours）** | **0.76** | **0.74** |

- **最强提升**：相比最强基线 FedRL+RND，EDPFRL-IM 在 MountainCar-v0 上提升 **58.3%**（0.48→0.76），在 CartPole-sparse 上提升 **42.3%**（0.52→0.74）。
- **Table I 逐客户端**：EDPFRL-IM 在所有 10 个客户端上均高于 FedRL（MountainCar: 0.66~0.75 vs 0.38~0.44；CartPole: 0.65~0.72 vs 0.33~0.38），且方差更低，体现个性化一致性优势。
- **消融（Table II）**：去掉 RND（FedRL）→ 0.40/0.35；仅加 RND 无协调（FedRL+RND）→ 0.60/0.54；加全局先验但 β=0 → 0.63/0.58；完整方法（β=0.5）→ **0.80/0.76**，证明协调探索是关键增益来源。
- **探索覆盖率（Fig. 3）**：EDPFRL-IM 的跨客户端状态空间覆盖显著更广，验证了协作新奇度的有效性。
- **冷启动鲁棒性（Fig. 4）**：第 5 轮插入新客户端，EDPFRL-IM 利用全局先验快速适应，明显快于 FedRL 和 FedRL+RND。

## 相关工作脉络
- **RND [6]**：中心化 RL 的随机网络蒸馏内在激励方法，关注新奇状态探索；本文将其引入联邦场景，并通过服务器聚合实现跨客户端协调，弥补 RND 无联邦协作的局限。
- **ICM [7]**：逆动力模型驱动的内在动机；本文方法基于 RND 而非 ICM，优势在于无需学习逆动力学模型，计算更轻量且适合非结构化状态空间。
- **CBE [8]**：基于 successor representation 的计数探索；本文指其适合小规模离散空间，在复杂高维连续空间中扩展性存疑，而 RND+全局先验更具通用性。
- **FedRL [2]**：容忍故障的联邦强化学习基线，关注全局策略聚合；本文聚焦个性化而非全局共享策略，解决非 IID 场景下的异质性问题。
- **FedAvg-RL [4] / pFedMe [5] / FedPer++ [12]**：个性化联邦学习基线；三者均缺乏系统性探索机制设计，本文填补了"个性化 + 内在探索 + 联邦协调"三位一体的空白。
- **FedRL+RND**：将 RND 引入联邦强化学习但无跨客户端协调；本文通过全局新奇先验和有偏采样，解决了独立 RND 探索方向不一致的问题。

## 局限性与未来方向
- **状态空间假设**：当前方法依赖状态聚类（cluster(s)），适用于离散/低维空间；在连续高维状态空间（如图像输入）中，聚类效率与新奇度量有待改进。
- **单一内在动机**：仅采用 RND，未与 ICM、count-based 等其他内在动机方法对比或融合，可能错过互补优势。
- **实验环境局限**：仅在 2D 经典控制任务（MountainCar、CartPole）上验证；需在更复杂环境（如 Atari、机器人控制、多智能体场景）中检验可扩展性。
- **RND 网络开销**：每客户端额外维护目标网络+预测器，对资源受限设备（如边缘 IoT）的计算/内存压力需进一步优化。
- **联邦聚合策略**：当前 Aggregate 仅做简单求和；未来可探索加权聚合（按客户端数据量、探索质量动态加权）以提升公平性与效率。
- **理论分析**：论文缺乏收敛性证明或探索-协调的理论 bound，可作为后续理论工作的方向。

## 研究启发与可借鉴点
- **内在动机 + 联邦学习的结合范式**：将 RND 内在奖励与联邦架构结合的思路，可直接迁移至任何需要"个性化 + 稀疏奖励 + 隐私保护"的场景（如个性化推荐、边缘设备控制、医疗决策辅助）。
- **压缩探索摘要的通信协议设计**：以访问计数/直方图/top-k 哈希等低维统计量替代原始经验或梯度，为联邦强化学习的通信效率优化提供了可复用的设计模式。
- **全局新奇先验引导的有偏采样**：$p(s) \propto (1+\beta\cdot\mathcal{P}_{\mathrm{novel}}(s))$ 的经验重采样策略，可推广至其他需要"集体探索引导个体学习"的分布式 RL 场景（如多机器人协同、自动驾驶仿真集群）。
- **冷启动客户端的快速适应机制**：利用全局探索知识辅助新加入客户端的框架，对联邦学习系统中的动态客户端加入/退出场景具有直接参考价值。
- **实验设计借鉴**：冷启动鲁棒性测试、探索覆盖率量化分析、多维消融（无RND/无协调/无全局先验）的实验设计严谨，值得在后续工作中复用。

## 关键术语表
- **Personalized Federated Reinforcement Learning (PFRL)**：个性化联邦强化学习，在联邦学习架构下为每个客户端学习个性化策略，而非全局共享策略，适应非 IID 数据分布。
- **Intrinsic Motivation**：内在动机，指不依赖外部奖励信号、由代理自身好奇心或探索欲望驱动的辅助奖励，用于解决稀疏奖励环境下的探索难题。
- **Random Network Distillation (RND)**：随机网络蒸馏，一种内在动机方法，通过预测固定随机目标网络输出的误差（预测不确定性）来衡量状态新奇度，误差越大越新奇。
- **Global Novelty Prior**：全局新奇先验，由服务器聚合所有客户端探索统计后计算得到的状态新奇度分布，用于指导各客户端的探索方向。
- **Biased Experience Sampling**：有偏经验采样，在客户端本地经验缓冲池中，按照全局新奇先验调整的采样概率进行经验回放，以提升对新奇状态的利用效率。
- **Exploration Summary**：探索摘要，客户端向服务器上传的压缩探索统计信息（如访问计数直方图、top-k 新奇状态哈希），用于隐私保护下的跨客户端协调。
- **Cold-Start Client**：冷启动客户端，在联邦训练中途新加入、无历史经验的客户端，需要快速适应并利用全局知识进行高效学习。
- **Sample Efficiency**：样本效率，指代理在获得一定性能水平前所需与环境交互的样本数量，是评估强化学习算法实用性的关键指标。

## 可复现要素
- **数据集/环境**：Modified MountainCar-v0、Modified CartPole-sparse（基于 OpenAI Gym 的标准环境，论文未提供公开数据集链接，但环境描述详细可复现）。
- **代码/权重开源**：论文未提及代码开源状态，论文未提供 GitHub 链接。
- **关键超参**：N=10 客户端；T=100 全局轮数；E=10 本地 PPO 步数；RND 隐藏层 128 维；$\alpha_i=0.1$；$\beta=0.5$；$\gamma=0.99$；learning rate=$3\times10^{-4}$；Adam 优化器；策略网络 2 层全连接 64 隐藏 ReLU。
