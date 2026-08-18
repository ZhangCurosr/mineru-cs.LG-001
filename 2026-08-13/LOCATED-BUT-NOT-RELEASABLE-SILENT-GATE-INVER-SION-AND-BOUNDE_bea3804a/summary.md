---
title: "LOCATED-BUT-NOT-RELEASABLE-SILENT-GATE-INVER-SION-AND-BOUNDE"
source: https://arxiv.org/pdf/2608.11822v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:29:37"
---

# 论文速读：LOCATED-BUT-NOT-RELEASABLE-SILENT-GATE-INVER-SION-AND-BOUNDE

## 一句话总结
本文在完全预注册（preregistered）的端到端压力测试中验证：当小规模因果推理Transformer中存在“表征-行为鸿沟”时，即使精确定位到抑制结构并验证其充分性，行为级释放仍会因门控外分布倒置与线性方向剂量饱和而双重失败，证明“找到结构”不等于“能释放行为”。

## 研究问题与动机
- **核心问题**：已有大量工作表明LLM内部表征了任务相关结构却未使用（representation–behavior gap），但若将该结构精确找到，能否在预注册标准下完整转化为可观测行为？
- **现有方法不足**：探测（probing）与因果定位（causal mediation）研究多止步于表征级指标，隐含“定位成功即控制可达”的流水线假设；激活导向（steering）方法依赖平均效应报告，缺乏对分布偏移、触发机制与剂量边界的严格检验。
- **现象基础**：Xun (2026) 已在25.7M Transformer上确立“抑制现象”——因果结构在残差流中线性可解码，但模型行为仍追随观测关联而非因果效应，为端到端测试提供了明确的ground truth。
- **动机**：补全检测-定位-释放全链路验证，并通过冻结标准与哈希归档排除事后归因偏差，给出可审计的失败分解。

## 核心贡献（创新点）
- **端到端预注册压力测试范式**：构建并严格执行四组件CCG流水线，首次将“表征-行为鸿沟”置于冻结标准与哈希归档的端到端验证下，区别于以往仅报告平均效应或局部定位的文献。
- **揭示表示级定位与行为级发布的可解耦性**：中段层观测证据通道的激活移植确能恢复目标行为（paired release advantage 0.563/0.8
