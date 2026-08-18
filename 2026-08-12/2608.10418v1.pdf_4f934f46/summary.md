---
title: "A lower bound for stepsize-based acceleration of gradient descent"
source: https://arxiv.org/pdf/2608.10418v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 03:49:27"
---

# 论文速读：A lower bound for stepsize-based acceleration of gradient descent

## 一句话总结
本文针对仅通过预设非负步长调度加速无动量朴素梯度下降（GD）的问题，建立了末次迭代收敛速率的新下界 $\Omega(T^{-p})$（其中 $p > \sqrt{2+\sqrt{3}} \approx 1.9319$），从理论上证明了纯步长调节无法将 GD 加速至最优的 $O(T^{-2})$ 速率。

## 研究问题与动机
- **核心问题**：在光滑凸优化中，不引入动量或其他算法修改，仅依赖预先设定（且已知总迭代次数 $T$）的非负步长调度，朴素梯度下降能否达到最优的 $O(T^{-2})$ 收敛速率？
- **现有方法不足**：经典一阶方法下界 $\Omega(T^{-2})$ 适用于所有使用 $\leq T$ 次 oracle 的方法，无法区分朴素 GD 与 Nesterov 等加速方法；已有针对 GD 的更紧下界仅适用于基本可组合调度或 "anytime" 无限调度场景，且不能排除朴素 GD 达到 $O(T^{-2})$ 的可能性。
- **最新进展**：近期研究（Altschuler & Parrilo, 2025; Grimmer et al., 2025）基于银比 $\rho_{\text{sil}} = 1+\sqrt{2}$ 构造了包含偶发长步的调度，实现了 $O(T^{-\log_2(1+\sqrt{2})}) \approx O(T^{-1.2715})$ 的上界，但是否最优仍为开放猜想。
- **动机**：建立专门针对预设步长调度 GD 的下界，明确此类纯步长加速方法的理论极限，填补现有理论空白。

## 核心贡献（创新点）
- **确立新下界**：证明对任意 $p > p_\star = \sqrt{2+\sqrt{3}} \approx 1.9319$，存在 $c_p>0$，使得对任意预设计划为 $T$ 步的非负步长调度 $\eta$，均存在维度 $d\leq T+1$ 的光滑凸实例满足 $f(x_T)-f(x_\star) \geq c_p L R^2 (T+1)^{-p}$。
- **首次否定纯步长 $O(T^{-2})$ 加速**：提供了严谨证据，表明仅调节预设步长不足以将朴素 GD 加速至最优多项式速率，打破了相关猜想。
- **提出无时序依赖的分析框架**：通过 Moreau 包络几何构造硬实例，结合图匹配技巧消除长步长期 temporal 顺序的影响，再用 Lyapunov 势函数控制剩余调度质量，实现从实例构造到下界推导的完整闭环。
- **区分 prescribed-horizon 与 anytime 设定**：明确本文固定 $T$ 的预设调度下界强于 Tsai et al. (2026) 的 anytime 下界 $o(T^{-4/3})$，两者不可互推。

## 方法详解
- **归一化与步长分解**：设 $L=R=1$，将归一化步长 $h_t = L\eta_t$ 分解为截断部分 $\min\{h_t, 1\}$ 与超额部分 $y_t = (h_t-1)_+$，定义
