---
title: "Redistribution-based-Cost-Inference-Improves-Sparse-Safe-Off"
source: https://arxiv.org/pdf/2608.12306v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:35:42"
---

# 论文速读：Redistribution-based-Cost-Inference-Improves-Sparse-Safe-Off

## 一句话总结
本文提出 RCI（Redistribution-based Cost Inference）框架，将真实部署中常见的稀疏轨迹级“停止反馈”通过回报分解转化为稠密逐跳成本标签，使标准安全离线强化学习算法能在无需逐跳人工标注的条件下有效训练；理论证明该重分配保持 CMDP 可行策略集与最优拉格朗日鞍点不变，实验在 HighwayEnv 与 Safe-FetchReach 上实现约五倍的违规率下降且无显著任务性能损失。

## 研究问题与动机
- **核心问题**：现有安全离线 RL 方法（如 COPO、CPQ、BCQ-Lagrangian）高度依赖稠密的逐跳成本标注，但实际安全监控（人工或自动化系统）通常仅能提供轨迹级二值停止反馈，即仅在首次检测到不安全转换时发出终止信号，缺乏对早期动作的责任归属。
- **现有方法不足1（强假设局限）**：主流 CMDP 与离线安全 RL 文献均假设数据集自带完整成本轨迹，而现实中的历史采集往往因安全仪器缺失或成本过高而无法提供逐跳标注，后验人工标注在规模上亦不可行。
- **现有方法不足2（在线交互依赖）**：近期处理稀疏安全反馈的工作（如 TraCeS、RLSF）依赖在线标注器查询进行迭代优化，无法直接应用于固定离线数据集场景。
- **动机**：将单次终止信号还原为逐跳成本是一个典型的时间信用分配问题；在分布偏移与一次性数据收集约束下，需设计一种信息无损且易于优化的成本重构机制，以兼容现有安全离线 RL 训练管线。

## 核心贡献（创新点）
1. **提出模块化 RCI 框架**：将安全离线学习解耦为“停止反馈收集→回报分解成本推断→约束离线策略学习”三阶段，支持任意回报等价分解算法与任意安全离线 RL 求解器自由替换。与既有耦合式设计相比，RCI 明确分离信号重构与策略优化，提升工程可扩展性。
2. **建立回报等价重分配的严格理论保证**：通过差分重建加终端补偿项 $\delta_T$，证明重分配后的逐跳成本严格满足 $\sum_t \tilde{c}_t = C(\tau)$，进而推导命题1（约束等价）与定理1（策略不变性），确认可行策略集与最优拉格朗日鞍点 $(\pi^*, \lambda^*)$ 在稀疏与稠密表述下完全一致。与 TraCeS 等经验性传播方法不同，该保证不依赖序列模型的预测精度。
3. **揭示稠密监督对学习条件的改善机制**：理论等价性之外，作者指出稀疏终端信号会使 cost critic 面临极端稀疏的监督信号，而重分配后的稠密成本提供条件更优的回归目标，显著提升 Q 值估计收敛质量。这一区分避免了“仅理论等价”的过度宣称，为后续稀疏监督方法提供清晰的理论-实践桥梁。
4. **双基准鲁棒性实证**：在 HighwayEnv 与 Safe-FetchReach 上，RCI 在混合行为策略数据集、含噪标签（终止点偏移 ±15 步）与对抗标签（20% 翻转）设置下均保持违规率大幅下降且任务返回稳定，证明机制对数据异质性与标注误差具有强容错性。

## 方法详解
- **Stage 1：专家反馈收集**。对每条轨迹 $\tau_i$，专家/自动评估器输出首个不安全转换索引 $t^*$（安全轨迹为空集），轨迹在 $t^*$ 处截断，生成稀疏成本 $c^{\mathrm{sparse}}(s_t, a_t) = \mathbf{1}\{t = t^*\}$。
- **Stage 2：回报分解与成本推断**。训练序列模型 $\hat{C}(s_{0:t}, a_{0:t})$ 预测 episodic cost，损失为 $\mathcal{L} = \mathbb{E}_{\tau \sim \mathcal{D}}[(\hat{C}(\tau) - C(\tau))^2]$。稠密逐跳成本通过连续预测差分获得：
  $$\tilde{c}_t = \hat{C}(s_{0:t}, a_{0:t}) - \hat{C}(s_{0:t-1}, a_{0:t-1}) + \delta_t, \quad \hat{C}(s_{0:-1}, a_{0:-1}) = 0$$
  终端补偿项 $\delta_T = C(\tau) - \hat{C}(\tau)$ 保证裂项求和闭合，其余 $\delta_t = 0$。本文实例化 $\hat{C}$ 为 LSTM，亦可替换为 RUDDER 或 GRD。
- **理论保障**。Lemma 1 证明 telescoping sum 严格满足 Definition 1（Return-Equivalent Cost Redistribution）；Proposition 1 与 Theorem 1 推导 $\mathbb{E}_\pi[\sum \tilde{c}_t] = \mathbb{E}_\pi[C(\tau)]$，进而证明可行集相同且拉格朗日 $\mathcal{L}(\pi, \lambda; \tilde{c}) = \mathcal{L}(\pi, \lambda; c^{\mathrm{sparse}})$，鞍点完全一致。Remark 1 强调补偿项使理论保证与 $\hat{C}$ 预测误差无关；Remark 2 指出实践差异源于 cost critic 的回归条件改善。
- **Stage 3：约束离线策略优化**。将增强数据集 $\mathcal{D}_{\mathrm{dense}} = \{(s_t, a_t, r_t, \tilde{c}_t)\}$ 输入标准安全离线 RL 算法。本文选用 BCQ-Lagrangian，Bellman 目标为 $y = r(s,a) + \gamma \max_{a'}[Q_{\theta'}(s',a') - \lambda \tilde{c}(s',a')]$，拉格朗日乘子按 $\lambda \leftarrow \max(0, \lambda + \alpha_\lambda(C_{\mathrm{batch}} - d))$ 自适应更新，$d$ 为安全预算。

## 实验与结果
- **环境与数据**：HighwayEnv（多车道避撞）与 Safe-FetchReach（7-DOF 机械臂避球形障碍）。每个环境使用 PPO 行为策略生成 5,000 条离线轨迹（任务导向但忽略安全），由自动化评估器标注首个不安全步骤并截断。额外生成 PPO、Random、Mixed（等比例混合）行为策略数据集；噪声实验包括 Noisy labels（终止点均匀偏移 $\delta \in [-15,15]$）与 Adversarial labels（20% 轨迹标签翻转）。
- **评估基线**：Reward-Only（无视安全，$d \to \infty$）、Sparse（直接使用终端稀疏标签）、Hazard（双头二元分类器，$P_1$ 预测是否出现在不安全轨迹，$P_2$ 预测是否为终止点，Focal Loss 处理类别不平衡）。
- **主要结果**：在 HighwayEnv 混合数据集上，RCI 将违规率较 Sparse 与 Hazard 基线大幅下降（结论称约五倍提升）；与无约束 BCQ-Vanilla 的两样本 t 检验显示 $t = 0.9962, p = 0.3483$，任务返回无统计学显著损失。Safe-FetchReach 上同样实现低违规率与竞争性返回。
- **鲁棒性**：跨 PPO/Random/Mixed 三种数据组成均保持违规率下降；在标签噪声下，Sparse/Hazard 违规率对误标极为敏感，而 RCI 因回报分解的平滑效应表现更稳定，任务返回维持不变。
- **定性分析**：空间成本景观显示，Sparse 仅在终端状态非零，Hazard 仅在危险邻域衰减，RCI 能生成沿接近走廊逐渐升高的连贯风险分布，恢复时间与空间的因果结构。

## 相关工作脉络
- **Offline Safe RL（BCQ, COPO, CPQ）**：依赖密集逐跳成本假设；RCI 仅用轨迹级停止反馈即可无缝接入同类架构，放松了数据标注前提。
- **稀疏安全反馈
