---
title: "Beyond Detection Accuracy: Measuring Explanation Cost, Stability, and Utility for Resource-Aware IoT Intrusion Detection"
source: https://arxiv.org/pdf/2608.10349v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 03:17:07"
---

# 论文速读：Beyond Detection Accuracy: Measuring Explanation Cost, Stability, and Utility for Resource-Aware IoT Intrusion Detection

## 一句话总结
本文在构建的无数据泄露CICIoT2023二元物联网入侵检测语料上，联合评估了预测性能、TreeSHAP解释成本、局部解释稳定性与选择性解释策略，证明仅凭检测精度无法决定资源受限场景下的部署选择，必须统筹计算开销、架构依赖的解释效率与安全优先的调用预算。

## 研究问题与动机
1. **预测精度不足以指导边缘部署**：现有IoT IDS研究过度聚焦Accuracy，但在高度不平衡的攻击分布下，高精度往往伴随大量Benign误报（如LR的FPR高达0.308），而边缘网关同时受内存、延迟与算力约束，需综合权衡Recall、FPR与资源开销。
2. **解释生成并非免费的后处理步骤**：TreeSHAP等事后归因的计算成本与模型架构强相关，同一批次下Random Forest的TreeSHAP耗时约为XGBoost的476倍，预测速度不能代理解释速度。
3. **解释可用≠解释可靠**：局部特征归因在小扰动下可能发生显著结构性变化（即使预测类别保持不变），且攻击流量的解释稳定性普遍弱于正常流量，单一稳定性指标容易产生误判。
4. **全量解释在重度攻击分布下不经济**：基于验证集校准的风险触发策略可在保留安全关键覆盖面的同时削减解释算力，但节省幅度高度依赖测试集的先验分布，需建立成本-覆盖权衡的量化基准。

## 核心贡献（创新点）
1. **构建哈希级防泄露的二元CICIoT2023语料**：通过非有限值清洗、54.8176%重复率精确去重、保守的跨原始标签碰撞剔除与确定性哈希划分，实现训练/验证/测试集零精确39维特征向量重叠，填补现有IoT基准未经审计直接使用的评估漏洞。
2. **首次联合度量预测性能、TreeSHAP成本与多维权稳定性**：在同一任务上同步输出预测指标、绝对/相对解释延迟、吞吐率与四维度稳定性分数，揭示“预测效率高≠解释效率高”的架构依赖现象。
3. **验证集校准的选择性解释策略与成本-覆盖前沿**：提出基于假阴性目标覆盖率（90%/95%）的风险触发阈值，在均衡测试集上实现15%–32%算力节省，并量化攻击主导分布下节省空间的急剧收缩。
4. **Native Importance可控消融规范**：针对异常高权重的`Number`特征进行受控移除重训，证明高原生重要性与强类别分离度并不等价于模型不可替代性，为归因可信度提供实证检验流程。

## 方法详解
- **语料审计与分区**：清洗CICIoT2023原始309个CSV（移除含缺失/无穷值的1,040行），对剩余46,775,660行计算完整39维特征精确哈希，剔除含交叉原始标签/攻击族碰撞的哈希后保留20,600,823个安全哈希，按70/15/15 deterministic hash-level划分，彻底消除精确特征向量泄露。
- **预测模型配置**：LR使用`SGDClassifier(loss="log_loss")`训练全部14,421,214样本（5 epochs，类权重9.423/0.528）；DT、RF、XGBoost共用确定性5,000,000训练子集。RF含150棵estimators（max_depth=20, sqrt特征划分）；XGBoost含300棵（max_depth=8, lr=0.1, subsample=0.8, min_child_weight=5, histogram）。全局seed=2026。
- **TreeSHAP成本度量**：在5,000样本平衡主样本上，以batch size 1/10/100/1,000/5,000分别测量预测时间$T^{\text{predict}}$与解释时间$T^{\text{explain}}$，计算绝对增量$C_{\text{XAI}}=T^{\text{explain}}-T^{\text{predict}}$与开销比率$O_{\text{XAI}}=T^{\text{explain}}/T^{\text{predict}}$，并报告微秒/样本延迟与吞吐率。
- **预测保持扰动稳定性协议**：选取100个基样本（50 benign/50 attack），在9个连续流量特征上施加$\sigma_j=0.01\cdot\text{IQR}_j$扰动，候选仅保留满足$f(x')=f(x)$且$|p(x')-p(x)|\leq0.05$的样本。使用Top-5 Jaccard、Spearman相关性、有向余弦相似度、归一化$L_1$漂移四维度量，以基样本为统计单元进行10,000次percentile-bootstrap 95% CI。
- **选择性解释策略**：P0全量解释；P1仅解释预测攻击；P2仅解释不确定性区间$[0.4,0.6]$；P3解释攻击或不确定性；P4基于验证集假阴性概率分位数校准阈值（DT=0.0046, RF=0.0106, XGBoost=0.0065），目标逼近指定FN覆盖率，测试集严格不参与阈值选择。

## 实验与结果
- **数据集与基线**：自然测试集3,090,099样本（攻击占比94.7%
