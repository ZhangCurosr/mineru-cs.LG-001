---
title: "TradingMoE-Routing-the-Right-Experts-in-Evolving-Markets"
source: https://arxiv.org/pdf/2608.11785v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-08-18 19:04:09"
---

# 论文速读：TradingMoE: Routing-the-Right-Experts-in-Evolving-Markets

## 一句话总结
本文提出 TradingMoE，一种构建于冻结稠密 LLM 主干之上的稀疏内部 MoE 路由框架，通过 Query–Key Router 与稀疏专家选择更新机制直接生成 Token 级交易决策；在股票与加密货币市场大幅超越 22 个基线模型，且性能不依赖预训练时间泄露。

## 研究问题与动机
- **LLM 直接交易的预测异质性**：大语言模型在金融分析中展现潜力，但其预测能力随资产、决策领域与市场条件剧烈波动，单一稠密模型难以稳定胜任动态交易。
- **现有路由机制与真实贡献脱节**：传统 MoE 路由器仅基于 hidden state 打分，原生路由得分与对照实验测量的收益增益 Pearson 相关系数仅 −0.015，且 66.76% 的决策 token 至少存在一个未被选中的更优专家。
- **外部专家协调粒度粗糙**：现有 LLM 交易系统多依赖人类定义的外部专家模块（如新闻分析、价格预测分支），路由粒度停留在粗粒度的模块级，无法适配跨资产、跨 Token 的细粒度需求。
- **Token–Expert 信用具备可压缩的低秩结构**：反事实信用分析显示，专家适用性可在紧凑潜空间中表示（秩 16 即可保留 74%–78% 信用能量），为轻量级路由设计提供理论依据。

## 核心贡献（创新点）
1. **提出 TradingMoE 稀疏内部 MoE 框架**。在冻结的稠密 LLM 主干每个 MLP 块中插入共享 down-LoRA experts，直接生成结构化交易决策。与依赖人工外部专家或固定 agent 流程的方法本质不同，本文实现了 token 级细粒度、端到端的动态专家分工。
2. **设计 Query–Key Router 匹配机制**。将 token 后注意力表示映射为低维交易需求 query，与可学习专家 key 做内积 Top-k 匹配。区别于传统 MoE 仅用 hidden state 投影打分，该设计显式构建低维潜空间，使路由决策与 token 特定交易意图强对齐。
3. **提出稀疏专家选择更新机制（Sparse Expert Selection Update）**。每步采样少量不活跃专家作为 challenger，与当前最低分活跃专家对比任务损失，利用一阶近似在同一反向传播中更新路由边际。与现有方法缺乏未选中专家监督信号不同，该方法直接优化反事实信用差值，显著缓解原生路由器评分失真问题。
4. **提供无偏采样理论保证与边界相对信用分析**。证明均匀无放回采样 m 个不活跃专家的梯度是全部 E−k 个专家梯度的无偏估计，并将计算开销从 O(E−k) 降至 O(m)。这从理论上保证了轻量采样更新的科学性，填补了 MoE 路由器在交易任务中缺乏严格信用分配的空白。
5. **严格的泄露控制与滚动纸面交易验证**。在同等提示下对比冻结主干基线的负收益，证明 TradingMoE 的性能提升并非源于预训练时间泄露，且在 22 个跨类别基线（树模型、时序网络、RL、金融/通用 LLM、外部路由）上取得最优。与竞品相比，本文更强调前向部署的稳健性与泛化能力。

## 方法详解
- **基座与架构插入**：冻结 Qwen3.5-9B 主干，在每个 Transformer MLP 块中插入共享 down-LoRA experts，构成面向交易的稀疏内部 MoE。训练时仅更新共享 down-projections、expert up-projections、query heads 与 expert codes。
- **Query–Key Router**：query head 结构为 `Linear(d, d_q)–GELU–
