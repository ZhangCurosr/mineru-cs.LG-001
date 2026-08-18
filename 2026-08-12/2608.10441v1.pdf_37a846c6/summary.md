---
title: "Detecting an Effect Is Not Learning to Act on It: A Reward–SNR Floor for LLM Acquisition Agents"
source: https://arxiv.org/pdf/2608.10441v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:55:33"
---

# 论文速读：Detecting an Effect Is Not Learning to Act on It: A Reward–SNR Floor for LLM Acquisition Agents

## 一句话总结
本文指出“检测辅助信号在平均意义上有效”与“学习每个样本层面的获取策略”是两件本质不同的事，并提出 reward–SNR 可检测性下界 $\rho^\star(N) \approx 2.8/\sqrt{N}$；通过 Structured Hypothesis Embeddings (SHE) 在 MIND、Amazon-Beauty、REES46 上的实证表明，尽管 LLM 衍生的结构化意图信号本身忠实可靠，但由于下游奖励 SNR 低于该下界，任何粒度的离线学习获取策略均无法战胜随机基线，真正可部署的单位是设计时的 regime 门控。

## 研究问题与动机
- **核心问题**：在推荐/决策管线中，系统可支付延迟、算力或金钱成本获取一个辅助信号（如 LLM 结构化推理、慢速专家、人工标注），离线数据中究竟能否学会“对哪些样本付费获取该信号”？
- **认知盲区**：现有 LLM-as-feature、active learning、learning-to-defer 等工作普遍默认“信号均值有效 ⇒ 可学 per-instance 路由”，忽略了异质性策略估计受 reward 噪声支配的统计极限。
- **in-sample oracle 陷阱**：按 realized $\Delta_i$ 排序选 top-b 的“显然有效”证据，往往是噪声的次序统计量（order statistics of noise），并非可泛化的结构。
- **现实动机**：生产经验表明结构化意图在冷启动/低确定性子场景价值最高，但工程上广泛尝试用 learned routing 自动分配预算，缺乏先验的统计可识别性检验。

## 核心贡献（创新点）
- **诊断发现**：在三种粒度（per-impression、K=4−64 聚类、hand-defined/uplift-tree regimes）及三个数据集上，任何可部署获取策略均无法显著优于随机；matched-moment i.i.d. noise placebo 可复现 ≥100% 的 oracle 表观增益，证明其为噪声次序统计量。
- **理论下界**：提出 reward–SNR 可检测性下界 $\rho^\star(N) \approx 2.8/\sqrt{N}$，严格区分“均值检测”与“策略可学性”；辅以正控制实验证明这是真实低 SNR 极限而非管道失效。
- **方法实例 SHE**：设计 Structured Hypothesis Embeddings，用冻结 LLM 将用户历史分解为 K 个带置信度与证据引用的排名假设，作为推荐器 input-embedding 分支；提供可测试的 grounded faithfulness 与可校准置信度。
- **落地处方**：证明 per-instance 路由在 SNR 低于下界时不可学，主张改用 design-time regime gate（基于历史长度/稀疏度等廉价特征预划定子系统），并在 pooled-regime 层面完成统计验证。
- **可复现 artifact**：发布 58-claim ledger 与一命令离线复现脚本，将每项结论透明映射到代码/CSV/图表，排除“黑箱显著性”争议。

## 方法详解
- **SHE 生成（Scheme B）**：对长度为 $n$ 的用户历史 $H=(e_1,\dots,e_n)$，调用冻结 LLM（GPT-5.5，high reasoning effort）生成恰好 $K=3$ 个按优先级排序的意图假设 $h_k$，每个附带校准置信度 $\gamma_k \in [0,1]$ 与支持证据索引集 $E_k \subseteq \{1,\dots,n\}$；强制边界规则保证弱信号假设置信度 <0.5 且含固定 fallback 文本。
- **编码与特征构造**：每个假设经固定文本编码器 $\phi$（MIND/REES46 用 OpenAI text-embedding-ada-002，Amazon-Beauty 用本地 TF-IDF+TruncatedSVD LSA，d=256）映射为 $\ell_2$ 归一化向量 $e_k$；候选 item $c$ 在该分支上的主特征为置信加权最大面匹配：
  $$f_B^{\max}(c) = \max_{k \in \{1,\dots,K\}} \gamma_k \cos(c, e_k)$$
  拼接 mean/max 变体与最佳面置信度构成 4 维固定宽特征块。
- **Late Fusion 与下游头**：SHE 特征块与廉价 base backbone（mean-pool、GRU 或 SASRec）输出在输入嵌入层晚期拼接，由 $\ell$ 正则化 logistic 分类/重排头处理；分支完全冻结，无反向传播进入 LLM。
- **忠实度与校准度量**：grounded faithfulness 定义为“假设与引用证据的余弦相似度减去未引用证据的差值”；distinctiveness 用假设间 $1-\cos$ 衡量；ECE 通过 cross-fit isotonic regression 校准。
- **Reward–SNR 下界**：设每样本奖励变化 $\Delta_i = R_i(\text{with } o_i) - R_i(\text{without})$，均值 $\mu$、标准差 $\sigma$，定义 $\rho = \mu/\sigma$。在 $\alpha=0.05, 1-\beta=0.8$ 下，可检测性必要条件为：
  $$\rho \ge \rho^\star(N) \approx \frac{2.8}{\sqrt{N}},
