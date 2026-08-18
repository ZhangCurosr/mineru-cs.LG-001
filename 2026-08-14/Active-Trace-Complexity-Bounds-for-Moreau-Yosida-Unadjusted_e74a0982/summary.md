---
title: "Active-Trace-Complexity-Bounds-for-Moreau-Yosida-Unadjusted"
source: https://arxiv.org/pdf/2608.13467v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 19:05:57"
---

# 论文速读：Active-Trace-Complexity-Bounds-for-Moreau-Yosida-Unadjusted

## 一句话总结
本文针对非光滑复合目标 $\pi(x)\propto \exp\{-f(x)-g(x)\}$ 的 MYULA 采样算法，引入“参考活性迹”$B_{\text{ref}}$ 替代全局曲率界 $d/\lambda$ 控制离散化误差；对 Lasso、Group Lasso、总变差等结构化惩罚，证明了端到端迭代复杂度可从传统 $\tilde{O}(\varepsilon^{-3})$ 提升至 $\tilde{O}(\varepsilon^{-2})$，且无需修改经典 MYULA 迭代核。

## 研究问题
