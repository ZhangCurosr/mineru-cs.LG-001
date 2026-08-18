---
title: "Beyond Detection Accuracy: Measuring Explanation Cost, Stability, and Utility for Resource-Aware IoT Intrusion Detection"
source: https://arxiv.org/pdf/2608.10349v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 03:17:05"
field: "可解释AI与网络安全交叉"
keywords: ["IoT Intrusion Detection", "Explainable AI", "TreeSHAP", "Explanation Cost", "Explanation Stability", "Selective XAI", "CICIoT2023", "Resource-Aware XAI"]
innovations: ["首次联合评估TreeSHAP解释成本、局部稳定性与选择性解释策略在IoT IDS中的操作权衡", "构建防泄露的二元CICIoT2023语料库，消除54.8%重复数据的精确特征向量重叠", "提出验证集校准的选择性解释策略，在平衡测试上90% FNR覆盖可实现28-32%计算节省"]
benchmarks: ["CICIoT2023"]
---

# 论文速读：Beyond Detection Accuracy: Measuring Explanation Cost, Stability, and Utility for Resource-Aware IoT Intrusion Detection

## 一句话总结
本文针对资源受限的IoT入侵检测系统，联合评估预测有效性、TreeSHAP解释的计算成本、局部稳定性及选择性解释策略，发现**检测精度不能决定部署适用性**，TreeSHAP成本因模型架构差异可达两个数量级，且成本节约高度依赖于测试分布的攻击 prevalence。

## 研究问题与动机
1. **现有IDS评估仅关注预测准确率，忽略解释成本**：TreeSHAP等事后解释方法的运行时间与内存消耗在IoT边缘网关等受限场景中可能远超预测本身，但现有文献未将解释开销与预测效率置于同一尺度衡量。
2. **解释可用≠解释可靠**：特征归因可能在预测类别保持不变的小扰动下发生显著变化（ attribution drift），单一属性图无法证明因果相关性或操作价值。
3. **CICIoT2023存在严重数据泄露风险**：原始数据集包含54.8176%的精确重复行，随机划分可能导致训练/测试间完全相同的39维特征向量重叠，产生乐观性能估计。
4. **缺乏选择性解释的实证研究**：现有工作未探索如何在保留安全相关解释覆盖的同时，通过置信度/风险评估策略减少不必要的解释计算负载。

## 核心贡献（创新点）
1. **构建防泄露的二元CICIoT2023语料库**：通过非有限值处理、精确特征去重、保守的原始标签碰撞移除及确定性哈希级别划分，消除训练/验证/测试间的39维特征向量精确重叠（论文未提及的其他方法未做此审计）。
2. **联合评估预测质量与解释成本**：在自然分布与平衡分布下系统比较LR、DT、RF、XGBoost，强调Recall、FPR、PR-AUC而非Accuracy，揭示高精度模型的虚假警报代价。
3. **首次量化TreeSHAP架构依赖成本**：测量绝对延迟、吞吐量及解释/预测比率，发现5000样本时RF需700.759s而XGBoost仅需1.471s，相差约476倍。
4. **提出验证集校准的选择性解释策略**：基于FNR覆盖目标动态调整触发阈值，在平衡测试上约90% FNR覆盖可实现28–32%计算节省，约95%覆盖实现15–23%节省。
5. **多度量解释稳定性评估协议**：使用Top-5 Jaccard、Spearman相关、余弦相似度和归一化L1漂移四个互补指标，揭示XGBoost保持高方向一致性但top-k成员更敏感的本质。

## 方法详解
1. **数据质量审计与防泄露语料构建**：
   - 扫描309个CSV文件，识别1,449个缺失值单元格和1,037个无穷值单元格（非行数），移除1,040行后剩46,775,660有效行。
   - 计算完整39维特征的精确哈希，发现25,641,309个重复实例（54.8176%重复率）。
   - 保守移除所有跨原始标签的哈希（含533,528个跨标签碰撞、451,969个跨攻击家族碰撞、905个良性-攻击碰撞），保留20,600,823个安全哈希。
   - 确定性哈希级别划分：训练14,421,214、验证3,089,510、测试3,090,099，跨分区哈希重叠为零。

2. **预测模型配置**：
   - LR：SGDClassifier(loss="log_loss")，5 epochs， balanced class weights (9.423 for benign, 0.528 for attack)，阈值0.50（敏感度分析用0.11）。
   - DT：max_depth=20, min_samples_leaf=20, balanced weights。
   - RF：150 estimators, max_depth=20, min_samples_leaf=20, n_estimators=sqrt(features), balanced subsample weights。
   - XGBoost：300 estimators, max_depth=8, learning_rate=0.10, subsample=0.80, colsample_bytree=0.80, min_child_weight=5, histogram method。
   - 三棵 trees 共用5,000,000确定性训练样本；LR使用全部14,421,214样本。

3. **解释成本测量公式**：
   - 解释开销比：$O_{XAI}(m, b) = T_{m,b}^{explain} / T_{m,b}^{predict}$
   - 绝对增量解释时间：$C_{XAI}(m, b) = T_{m,b}^{explain} - T_{m,b}^{predict}$
   - 在batch size {1, 10, 100, 1000, 5000}下测量，prediction用10次重复，TreeSHAP用自适应重复（batch 1/10用10次，batch 100用5次，batch 1000用3次，batch 5000用1次）。

4. **解释稳定性协议**：
   - 100个基础样本（50 benign + 50 attack），每个生成10个候选扰动，最多接受5个。
   - 扰动约束：仅在9个连续流量统计特征（Rate, Tot sum, Min, Max, AVG, Std, Tot size, IAT, Variance）上施加，$\sigma_j = 0.01 \times IQR_j$， clipped到验证集0.5th–99.5th百分位，非负。
   - 接受条件：$f(x') = f(x)$ 且 $|p(x') - p(x)| \leq 0.05$。
   - 四维度指标：Top-5 Jaccard（特征集重叠）、Spearman（排名一致性）、signed cosine（方向一致性）、normalized L1（幅值漂移）。
   - 统计单位：先保留扰动对级别描述，再用基础样本级别聚合（每样本内平均后再跨样本平均），bootstrap 10,000次获得95% CI。

5. **选择性解释策略**：
   - P0：解释所有预测。
   - P1：仅解释预测为attack的样本。
   - P2：仅解释不确定预测（$p(attacks) \in [0.40, 0.60]$）。
   - P3：解释attack预测或不确定预测。
   - P4：验证集校准风险触发——解释所有attack预测 + 预测benign但attack概率超过模型特定阈值的样本。
   - P4阈值从验证集FN的attack概率分位数校准，目标约95% FNR覆盖，不使用测试标签。

## 实验与结果
**数据集**：CICIoT2023（46,776,700原始行→20,600,823唯一安全哈希；原始分布：Benign 1,098,191 / DDoS 33,984,450 / DoS 7,845,120 / Mirai 2,634,054 / Recon 690,534 / Spoofing 486,458 / Web 24,829 / BruteForce 13,064）。

**预测结果（自然测试集，3,090,099样本：163,513 benign / 2,926,586 attack）**：
- XGBoost：F1=0.985597，Recall=0.971852，FPR=0.004605，PR-AUC=0.999791，ROC-AUC=0.996212 ← **最强整体预测profile**。
- RF：F1=0.984059，Recall=0.968741，FPR=0.002269 ← **最低FPR**。
- DT：F1=0.985424，Recall=0.971794，FPR=0.009718。
- LR：F1=0.897914，Recall=0.828777，FPR=0.308367 ← **严重虚假警报问题**。

**解释成本（5000样本，平衡master sample）**：
- RF：预测0.024835s，解释**700.758801s**，吞吐7.135 samples/s，overhead **28,217.04×**。
- DT：预测0.000433s，解释5.642689s，吞吐886.10 samples/s，overhead 13,025.60×。
- XGBoost：预测0.004109s，解释**1.471088s**，吞吐**3,398.84 samples/s**，overhead **357.99×**。
- **RF解释时间约为XGBoost的476倍**。

**稳定性（ε=0.01，基础样本级别，bootstrap 95% CI）**：
- RF：Jaccard=0.8854 [0.8633, 0.9067]，Spearman=0.9847 [0.9818, 0.9875]，cosine=0.9709 [0.9627, 0.9786]，L1=0.0969 [0.0816, 0.1134] ← **整体最强稳定性**。
- DT：Jaccard=0.8664 [0.8454, 0.8863]，Spearman=0.9812 [0.9786, 0.9837]，cosine=0.9538 [0.9399, 0.9665]，L1=0.1621 [0.1299, 0.1968]。
- XGBoost：Jaccard=0.7831 [0.7625, 0.8036]，Spearman=0.9609 [0.9561, 0.9653]，cosine=0.9546 [0.9457, 0.9630]，L1=0.1894 [0.1695, 0.2104]。
- 所有模型均显示攻击样本的magnitude drift大于良性样本。

**选择性解释（平衡测试，P4策略）**：
- 90% FNR覆盖目标：DT节省28.35%，RF节省31.17%，XGBoost节省**32.22%**。
- 95% FNR覆盖目标：DT节省15.24%，RF节省22.58%，XGBoost节省**23.47%**。
- 自然测试集（攻击主导）：95%覆盖仅节省1.72–2.61%，说明prevalence极度影响策略效益。

**Number特征消融**：
- XGBoost移除Number后F1变化仅-0.000023（自然）/-0.000028（平衡）← 高原生重要性≠不可替代。
- DT移除Number后F1微降-0.000450，但FPR从0.009718降至0.008183（∆=-0.001535）。

## 相关工作脉络
1. **Le et al. [14]**：将SHAP应用于IoTID20/NF-BoT-IoT-v2/NF-ToN-IoT-v2的DT和RF解释，**未显式评估解释计算成本、稳定性或选择性调用**。
2. **Patil et al. [15]**：基于CICIDS2017用LIME评估DT/RF/SVM/voting ensemble，同样**未测量解释开销或稳定性**。
3. **E-XAI [18]**：在CICIDS2017/NSL-KDD/RoEduNet-SIMARGL 2021上评估black-box方法的描述准确性、稀疏性、稳定性、效率和完整性，**但未将解释延迟与预测延迟在同一尺度比较，也未进行选择性解释策略**。
4. **Wang et al. [17]**：研究IoT IDS中SHAP/LIME导向的透明度，**未系统评估解释成本架构依赖性**。
5. **Munilla & Khammas [19]**：用DeepFool对抗扰动评估SHAP/LIME，**聚焦对抗鲁棒性而非预测保持扰动下的局部稳定性，也未测量绝对解释延迟**。
6. **本文定位**：填补"解释可用≠解释可操作"的空白，首次联合评估CICIoT2023上四种模型的预测质量+TreeSHAP绝对成本+局部稳定性+选择性解释权衡，并提供防泄露语料构建方法。

## 局限性与未来方向
1. **单一数据集**：仅在CICIoT2023二进制任务上验证，未跨数据集（如Edge-IIoTset）测试泛化性。
2. **主机计时而非边缘设备**：所有训练、推理、解释计时在Windows 11 Intel Core 7 240H主机上测量，不直接代表Raspberry Pi等IoT网关性能。
3. **记忆测量局限**：使用before/after RSS差值，未捕获瞬态峰值分配。
4. **TreeSHAP范围限制**：结论仅限TreeSHAP，不能直接推广至LIME、model-agnostic SHAP、counterfactual方法。
5. **扰动约束**：仅测试9个连续特征上的受控扰动，不构成完整的协议级或因果流量生成测试。
6. **无因果有效性验证**：稳定归因不证明模型推理是因果的或语义正确的。
7. **无统计显著性检验**：预测指标为点估计，未进行重复训练或formal significance testing。
8. **全测试集解释时间为吞吐量投影**：非直接测量，可能低估实际运行时开销。
9. **未来方向**：在真实IoT硬件上验证边缘可行性；扩展至多类别攻击家族检测；探索其他解释方法（LIME/counterfactuals）的成本-稳定性权衡；进行人机交互实验评估解释效用。

## 研究启发与可借鉴点
1. **防泄露语料构建协议可迁移**：基于精确n维特征哈希的去重+碰撞移除+确定性划分流程，可直接应用于其他存在重复/泄露风险的网络流量数据集（如NF-BoT-IoT、ToN-IoT）。
2. **多粒度稳定性评估框架**：同时报告Top-k Jaccard、Spearman、cosine、L1四个指标，并区分扰动对级别与基础样本级别聚合，为XAI稳定性评估提供标准化方法模板。
3. **选择性解释作为资源分配策略**：将selective classification的拒绝机制移植到XAI调用层面，用验证集FNR分位数校准触发阈值，无需测试标签即可实现成本-覆盖权衡，适用于任何需要事后解释的预测系统。
4. **Number特征消融启示**：高原生特征重要性（如XGBoost的Number KS=0.939）不等于预测依赖性或因果性，建议在部署前进行针对性消融验证以避免误判数据泄露。
5. **Prevalence-sensitive评估设计**：同时报告自然分布与平衡分布结果，可揭示metric对class distribution的敏感性，避免单一分布下的高Accuracy掩盖FPR缺陷。

## 关键术语表
**TreeSHAP**：针对决策树和树集成模型优化的Shapley加性解释算法，提供局部特征归因并支持全局聚合分析。

**解释开销比（Overhead Ratio）**：TreeSHAP解释时间与预测时间之比，用于衡量解释相对于预测的计算负担，需结合绝对延迟解读。

**预测保持扰动（Prediction-preserving Perturbation）**：小幅度输入扰动后模型预测类别不变且攻击置信度变化≤0.05的测试样本，用于评估局部解释稳定性。

**选择性解释（Selective Explanation）**：基于置信度/风险评估策略决定哪些预测需要生成解释，以在安全覆盖与计算成本间取得权衡。

**FNR覆盖（False Negative Explanation Coverage）**：被解释的漏报攻击占全部漏报攻击的比例，用于衡量选择性策略的安全相关性保留程度。

**哈希级别划分（Hash-level Partitioning）**：基于完整特征向量的精确哈希进行训练/验证/测试划分，确保跨分区无特征向量精确重叠。

**PR-AUC**：Precision-Recall曲线下面积，作为阈值无关的摘要指标，在严重不平衡数据集上比ROC-AUC更能反映攻击检测性能。

**归一化L1漂移（Normalized L1 Drift）**：扰动前后完整归因向量L1变化量除以原始向量L1范数，衡量解释幅值的相对稳定性。

## 可复现要素
- **数据集**：CICIoT2023公开可用（原始来源引用[9]）。
- **代码/权重**：预处理脚本、语料审计、预测评估、解释成本测量、稳定性分析、选择性解释代码及训练模型artifacts均开源，GitHub仓库：https://github.com/AbdurrahmanTolay/resource-aware-iot-xai，Zenodo归档DOI: 10.5281/zenodo.21879038。
- **关键超参**：
  - LR：SGDClassifier(loss="log_loss")，epochs=5，class_weight balanced (9.423/0.528)，threshold=0.50（sensitivity分析用0.11）。
  - DT：max_depth=20，min_samples_leaf=20，balanced weights。
  - RF：n_estimators=150，max_depth=20，min_samples_leaf=20，max_features=sqrt(39)，balanced subsample weights。
  - XGBoost：n_estimators=300，max_depth=8，learning_rate=0.10，subsample=0.80，colsample_bytree=0.80，min_child_weight=5，tree_method=histogram。
  - 全局seed=2026。
  - TreeSHAP batch sizes：1, 10, 100, 1000, 5000。
  - 稳定性扰动：ε∈{0.005, 0.010, 0.020}，9个连续特征，IQR缩放的σ_j=0.01×IQR_j。
  - 选择性解释目标：90%和95% FNR覆盖。
