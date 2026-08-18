---
title: "InSight-doc: Agentic Visual Perception for Long-Document Understanding"
source: https://arxiv.org/pdf/2608.10628v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:52:17"
---

# 论文速读：InSight-doc: Agentic Visual Perception for Long-Document Understanding

## 一句话总结
本文提出 InSight-doc，一种端到端、无需外部检索器的智能体视觉感知框架，将视觉分辨率视为推理时可动态分配的自适应资源；通过从低分辨率全景出发、多轮 zoom-in 截取高分辨率区域证据，显著提升了长文档 VQA 准确率，同时将幻觉率降低超 40%、推理延迟削减 41%–68%。

## 研究问题与动机
1. **长上下文计算瓶颈与 Context Rot**：Transformer 的注意力机制带来 O(N²) 时间与 O(N) 空间开销，且随提示变长易出现 context rot（注意力稀释导致性能骤降），直接喂入全分辨率长文档性价比极低。
2. **现有长文档方法的结构性缺陷**：端到端高分输入法冗余 token 多、难扩展；视觉 RAG 依赖外部检索器，存在 k 值敏感、检索误差难回退、额外索引延迟等问题；粗到细方法（如 Doc-V*、CogDoc）仍停留在页面级切换，无法实现子页面区域的精细定位。
3. **缺乏主动感知训练信号**：现有视觉搜索基准多集中于单张自然图像或单页文本图，缺少涵盖多页、多跳、跨区域证据整合的主动 zoom-in 轨迹数据，模型难以内生学会“何时放大、放哪里、如何整合”。

## 核心贡献（创新点）
1. **提出 InSight-doc 端到端智能体视觉感知框架**：放弃固定高分输入与外部检索器，允许模型在 interleaved CoT 中动态发起 zoom-in 工具调用，按需获取子页面高分辨率证据。
2. **构建大规模主动感知训练语料（17.9K SFT + 19.2K RL）**：设计三阶段级联过滤与 InSight-o3 双智能体轨迹生成管线，覆盖多页多跳可答题与人工构造的 hard unanswerable 样本，补齐长文档主动感知数据空白。
3. **SFT+RL 联合训练实现精度-效率帕累托优化**：在 Qwen3-VL-8B 基座上，平均准确率提升 4.3–16.4 点，同时序列长度与延迟大幅下降；理论推导给出序列长度与推理延迟的严格上界，验证了降采样策略的收益边界。
4. **开放完整可复现资产**：代码、训练数据集与 InSight-doc-8B 权重全部开源，并给出详尽的 judge 校准流程与加权采样策略，推动该方向的工程落地。

## 方法详解
1. **初始视觉上下文构建**：将原始高分文档页面集合 $\{I_k\}$ 以缩放因子 $r \in \{0.25, 0.35, 0.5, 0.7\}$（对应 50–140 DPI）下采样为 $\{\tilde{I}_k^{(0)}\}$，构成初始视觉上下文 $\mathcal{I}_{\mathrm{ctx}}^{(0)}$，大幅压缩首步 token 预算。
2. **多轮 Zoom-in 工具调用机制**：在第 $t$ 步，模型可发出 `zoom_in(k, d, b)`，其中 $k$ 为上下文图片索引，$d$ 为自由文本区域描述，$b$ 为边界框。调用成功后从原始高分源 $I_{s(k)}$ 裁剪，并按放大因子 $c>1$ 重采样为 $\tilde{I}_{\mathrm{crop}}^{(t)} = \mathsf{resize}(I_{\mathrm{crop}}^{(t)}, c \cdot r \cdot \mathsf{size}(I_{\mathrm{crop}}^{(t)}))$，追加至 $\mathcal{I}_{\mathrm{ctx}}^{(t)}$ 供后续推理。
3. **SFT 训练阶段**：利用 InSight-o3 的 vReasoner（GPT-5-mini）与 vSearcher（Qwen3-VL-8B-Instruct）生成多轮 trajectories，将双智能体交互合并为扁平的单智能体 multimodal CoT 序列；最终回答经 GPT-5-nano 规范化，去除 think 噪声，保留纯净答案格式。
4. **RL 训练阶段**：以 SFT 模型为起点，采用 GRPO 算法，reward 仅设为二元准确率（1.0）。通过 weighted refill sampler 控制每 batch 可答/不可答比例（86%/14%），并引入多选题与结构化文档 add-on 防止过度拒答；rollout temperature 0.7、top-p 0.8、tool-use limit 10。
5. **效率理论分析**：推导相对序列长度上界 $S_r/S_0 \leq x(r) + \kappa^{-1}y(r)$ 与相对延迟上界 $T_r/T_0 \leq w_p x(r)^2 + w_c x(r)y(r) + w_g y(r)^2$，证明在长文档 VQA 典型参数下，InSight-doc 可将序列长度压缩至基线的 7.8%–45.0%，延迟压缩至 32.5% 左右。

## 实验与结果
1. **评测设置**：覆盖标准文档 VQA（DUDE, MP-DocVQA）、长文档 VQA（MMLongBench-Doc, LongDocURL）与通用高分 VQA（MME-RealWorld-Lite, O3-Bench）；所有 PDF 统一光栅化为 200 DPI，测试时按 $r$ 下采样；答案判分使用校准后的 GPT-5-nano judge（legacy-v2 prompt）。
2. **主结果**：在 $r=0.25$（50 DPI）下，InSight-doc-8B (SFT+RL) 平均准确率 66.9%，较 Qwen3-VL-8B 提升 **+16.4 点**；在 $r=0.5$（100 DPI）下平均 72.6%，提升 **+4.3 点**。全面超越同规模开源模型，并在 $r=0.25$ 下超越所有 GPT 变体。
3. **效率与幻觉控制**：70 DPI 下以 **减少 66% token** 取得优于 140 DPI 基线的精度；端到端延迟降低 **41%–68%**（最长文档子集降低 71%）。在不可答子集上 F1 提升 15–26 点，幻觉率下降超 40%。
4. **轨迹质量**：RL 使 LongDocURL evidence-box coverage 从 SFT 的 68.1%/70.2% 提升至 82.3%/77.0%；冗余率降至 5.8%/2
