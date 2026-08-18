---
title: "Coordinating the Unknown Lipschitz Constant in Multiplayer Bandits"
source: https://arxiv.org/pdf/2608.10526v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 07:16:10"
---

# 论文速读：Coordinating the Unknown Lipschitz Constant in Multiplayer Bandits

## 一句话总结
本文针对未知 Lipschitz 常数 $L$ 的连续动作空间多智能体合作博弈，设计元算法 mECAB，通过三种信息对齐机制（共享奖励自然一致、动作隐式广播聚合、抖动量化解耦实例）使无法通信的玩家在各自估计 $L$ 后仍能达成相同的离散化网格，并在三种信息结构下均给出 $O(T^{(Md+1)/(Md+2)})$ 量级的次线性遗憾界。

## 研究问题与动机
- 无线频谱接入、分布式传感雷达等实际分布式系统中，多个决策者需同时在连续动作空间中协作，但通常无法在学习阶段进行显式通信。
- 经典 Lipschitz Bandit 离散化方法依赖已知平滑常数 $L$ 确定网格粒度；若 $L$ 未知，各玩家基于自身有限观测独立估计会得到不同数值，导致离散网格不一致，无法视为同一离散博弈问题。
- 现有合作多玩家 Bandit 工作（如 [4], [10], [11]）多假设离散动作空间或已知 $L$，尚未
