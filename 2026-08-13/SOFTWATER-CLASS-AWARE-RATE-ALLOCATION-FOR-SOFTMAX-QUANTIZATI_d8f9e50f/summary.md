---
title: "SOFTWATER-CLASS-AWARE-RATE-ALLOCATION-FOR-SOFTMAX-QUANTIZATI"
source: https://arxiv.org/pdf/2608.12026v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:37:49"
---

# 论文速读：SOFTWATER: CLASS-AWARE RATE ALLOCATION FOR SOFTMAX QUANTIZATION

## 一句话总结
本文针对大语言模型中常被跳过的 Softmax 输出头，将其量化建模为基于输出 KL 散度的率失真问题（SoftWater），通过二阶展开与可分离近似实现按类别频率与方差自适应分配量化网格，在 1B-32B 多模型上显著优于 WaterSIC 基线，可在仅增加约 3% 困惑度的前提下将模型体积压缩 4
