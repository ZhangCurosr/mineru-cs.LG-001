---
title: "TOWARDS-UNDERSTANDING-ON-POLICY-DISTILLA-TION-THROUGH-THE-LE"
source: https://arxiv.org/pdf/2608.11829v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:01:03"
---

# 论文速读：TOWARDS UNDERSTANDING ON-POLICY DISTILLATION THROUGH THE LENS OF TEST-TIME SCALING

## 一句话总结
本文通过测试时缩放视角系统评估了在线策略蒸馏（OPD），发现OPD主要提升了小采样预算下的推理访问效率，但并未真正扩展学生的能力边界；相反，大采样预算下性能反而下降，且遗忘的问题多于新学会的问题，呈现出一种“虚假蒸馏”效应。

## 研究问题与动机
- 工业界与学界普遍假设OPD能让学生模型从更强教师处蒸馏出新的推理能力，从而突破预训练基座的能力上限（如DeepSeek-V4、Qwen3、Nemotron-Cascade 2的实践）。
- 然而近期研究（Li et al., 2026; Zhu et al., 2026）报告OPD后模型未必稳定超越，甚至可能低于预训练基座，引发核心疑问：OPD究竟在扩展能力边界，还是仅通过重塑分布提升了已有能力的访问效率？
- 传统离线蒸馏已被证明能有效转移教师能力，但OPD依赖学生自生成轨迹与反向KL目标，其效果边界尚缺乏系统性量化分析。
- 现有评估多聚焦单一K值或最终验证指标，缺乏能够同时刻画“采样效率”与“能力边界”的细粒度诊断框架，难以客观归因OPD的真实增益来源。

## 核心贡献（创新点）
- **构建基于测试时缩放的OPD诊断框架**：通过连续变化采样预算K并联合绘制pass@K与avg@K曲线，首次系统分离了OPD对采样效率与能力边界的差异化影响，为后训练方法评估提供了标准化分析范式。
- **揭示OPD的“虚假蒸馏”效应**：实证表明OPD主要在有限采样预算内更易触达正确推理路径，并未像传统蒸馏那样从教师处获得真正新的推理能力，其表层增益本质是分布重加权。
- **提出问题级可解性迁移分析**：以pass@1024为能力边界代理，定量统计“保留/新学/遗忘”三类问题比例，发现OPD导致更多原本可解问题变为不可解，呈现显著的学习-遗忘不对称性。
- **验证改进型OPD的共性行为并对比离线蒸馏**：ExOPD、Direct-OPD、EOPD及纯前向KL变体均复现相同的效率-边界权衡趋势，说明该现象具有方法鲁棒性；同步证明离线蒸馏可在全K范围内稳定超越基座，具备真实能力扩展性。
- **轨迹困惑度与微观案例交叉验证**：通过Base/OPD/Teacher三方交叉perplexity分析与配对轨迹对比，证实OPD仅将概率质量偏移至学生自身分布已支持、且更受教师青睐的路径上，未引入学生分布外支撑的全新推理模式。

## 方法详解
- **标准OPD目标函数**：采用反向KL散度，在学生自生成轨迹上对齐学生与教师的下一词token分布：
  $\mathcal{L}_{\mathrm{OPD}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta}[\sum_{t=1}^L D_{\mathrm{KL}}(\pi_\theta(\cdot|x,y_{<t}) \| \pi_T(\cdot|x,y_{<t}))]$。
  实际实现采用top-k token KL（student top-16），相比full-vocabulary KL计算更轻量，相比sampled-token KL方差更稳定。
- **测试时缩放评估指标**：
