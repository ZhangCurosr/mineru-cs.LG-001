---
title: "Multi-Granular Rationale-Guided Molecular LLM for Property Prediction"
source: https://arxiv.org/pdf/2608.10480v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:00:53"
---

# 论文速读：Multi-Granular Rationale-Guided Molecular LLM for Property Prediction

## 一句话总结
本文提出 MR-MoL，一种多粒度 rationale 引导的分子大语言模型。该方法利用微调后的 GNN 通过子结构掩码计算归因，将最具影响力的子结构序列化为带方向标签的排名文本证据，与 SMILES 和分子图嵌入共同输入 LLM，在 MoleculeNet 八项性质预测任务上显著缩小了通用分子 LLM 与专家模型的性能差距。

## 研究问题与动机
1. **子结构贡献隐式化**：现有分子 LLM 依赖 1D SMILES 或 2D 分子图编码，分子内各个局部片段对最终预测的贡献被淹没在连续表征中，缺乏可读的结构级线索。
2. **外部增强无法替代内部证据**：RAG 或知识图谱检索虽然能补充上下文，但引入的是外部相似分子或全局化学知识，而非当前分子自身亚结构对目标性质的直接驱动作用。
3. **与化学推理范式脱节**：化学家在判断
