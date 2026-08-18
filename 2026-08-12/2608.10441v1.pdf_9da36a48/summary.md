---
title: "Detecting an Effect Is Not Learning to Act on It: A Reward–SNR Floor for LLM Acquisition Agents"
source: https://arxiv.org/pdf/2608.10441v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:57:03"
---

# 论文速读：Detecting an Effect Is Not Learning to Act on It: A Reward–SNR Floor for LLM Acquisition Agents

## 一句话总结
本文严格区分“检测昂贵信号的平均收益”与“学习按实例获取信号的策略”，提出奖励信噪比（Reward–SNR）可检测性阈值 ρ* ≈ 2.8/√N；以结构化假设嵌入（SHE）为实例证明：当数据集低于该阈值时，离线学习的逐例获取路由会坍缩为噪声阶序统计，真正可部署的单位是设计时基于廉价切片特征的 regime gate。

## 研究问题与动机
- 现有 LLM-as-feature 管线默认可从离线奖励数据中学习“何时调用昂贵 LLM 信号”的逐例策略，但未验证该策略是否具备统计可估计性。
- 在低 SNR 下，按实现奖励 Δ_i 排序的样本内 oracle 看似有显著增益，实则是噪声的阶序统计（order statistics），并非可迁移的可学习结构。
- 不同下游骨干网络强度与用户历史 regime（稀疏冷启动 vs 长多意图）会显著调制信号价值，但全局冗余往往不可区分于零，导致结论易被误读。
- 缺乏一条清晰的必要条件来界定“哪些场景下成本敏感型语义观测值得投入，哪些场景下仅适合固定门控”。

## 核心贡献（创新点）
- **诊断结论**：表观的获取可学性多为噪声阶序统计；匹配矩 i.i.d. 噪声替代变量可重现 ≥100% 的 oracle 表观增益，证明其非真实结构。
- **理论阈值**：提出奖励–SNR 可检测性阈值 ρ*(N) ≈ 2.8/√N，严格区分平均效应检测与逐例策略可学性，并通过正对照验证其为真实低 SNR 极限而非管线缺陷。
- **方法实例化**：结构化假设嵌入（SHE），用冻结 LLM 生成 K=3 个带置信度与证据索引的意图假设，嵌入后以 max-over-facets 方式与候选匹配并晚期融合至推荐骨干；提供可检验的 grounded faithfulness 指标。
- **可操作处方**：当 SNR 低于阈值时放弃学习逐例路由，改用基于廉价切片特征的设计时 regime gate，并在池化 regime 层面验证。
- **完整工件**：58 条声明账本映射至脚本/CSV/图表，支持一键离线复现。

## 方法详解
- **SHE 构建**：冻结 LLM（GPT-5.4/5.5）将用户交互历史 H 解码为 K=3 条排好序的意图假设 h_k，每条附带可校准置信度 γ_k ∈ [0,1] 与证据索引 E_k ⊆ {1,…,n}。
- **嵌入与特征向量**：假设与候选物品 c 分别通过固定文本编码器 φ（ada-002 或 TF-IDF+TruncatedSVD LSA）映射至 ℓ2 归一化空间；SHE 分支输出四维特征 [f_max, f_max^γ, f_mean, γ_{k*}]，核心坐标 f_max^γ = max_k γ_k
