---
title: "Robustness-of-AI-Art-Detectors-under-Generator-Shift"
source: https://arxiv.org/pdf/2608.11643v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:36:51"
---

# 论文速读：Robustness-of-AI-Art-Detectors-under-Generator-Shift

## 一句话总结
本文在“新生成器威胁模型”下系统评估了五种主流冻结骨干检测器从U-Net扩散模型（LDM/SD2.1）迁移到新一代Diffusion Transformer（SD3.5m）时的鲁棒性，发现所有检测器在分布外测试时性能显著下降，且失败模式高度不对称：假阴性（漏检AI图像）激增，而假阳性（误伤人类作品）维持低位。

## 研究问题与动机
1. **生成器迁移泛化空白**：现有AI艺术检测研究多在相同或相近生成器家族上训练与评估，缺乏对新兴架构（如从卷积U-Net到DiT）部署后可靠性的实证研究。
2. **现实部署风险**：社交平台、新闻核查与版权审核等场景中存在“被动规避”威胁——攻击者无需对抗扰动或模型逆向，仅需直接采用训练后新发布的公开生成器即可绕过固定阈值检测器。
3. **核心安全缺陷指向假阴性**：防御侧的主要风险并非误报人类作品，而是大量新版AI生成图像被错误放行为人类创作，从而突破 misinformation、欺诈或身份冒充筛查。
4. **现有基准的局限性**：公开数据集（如GenImage、ArtiFact）多混合多种生成器或未严格控制内容/风格分布，难以隔离“纯架构变更”带来的性能衰减，导致控场 benchmark 表现与真实部署可靠性之间存在可信度缺口。

## 核心贡献（创新点）
1. **构建提示词对齐的SD3.5m OOD数据集**：通过反向提示工程从保留的人类艺术品生成10,000张跨10种艺术风格的SD3.5m图像，确保测试集在语义与风格上与训练集可比，从而精准测量纯架构迁移导致的性能下降。
2. **统一协议下五种检测器的跨代对比评估**：在冻结骨干+线性探针设置下，系统对比ResNet系列、EfficientNet、ConvNeXt与CLIP ViT在ID与OOD上的衰减曲线，揭示强ID性能不等于跨生成器鲁棒性。
3. **定性故障归因（Grad-CAM可视化）**：首次对比检测器在ID真阳性与OOD假阴性上的空间注意力差异，证明SD3.5m生成的图像未能激活训练期学到的局部统计线索，导致预测置信度结构性塌陷。
4. **安全导向的威胁建模与纵深防御主张**：将生成器迁移形式化为无需对抗能力的“新兴合成媒体威胁”，论证单一图像检测器不可作为静态独立防线，必须与溯源元数据、水印凭证、漂移监测结合构成多层防御体系。

## 方法详解
1. **Prompt-aligned OOD数据集构建**：从AI-ArtBench人类子集中保留10,000张图像（每风格1,000张），使用CLIP Interrogator（BLIP-Large caption + CLIP ViT-L/14检索）生成反向提示词，经模板 `{art_style} style art titled {painting_name}, {CLIP_caption}` 增强后，由 `stabilityai/stable-diffusion-3.5-medium` 以28步去噪、guidance=4.5、768×768分辨率生成，每风格1,000张AI图像。
2. **冻结骨干+线性探针架构**：五个 backbone（ResNet-18/50, EfficientNet-B0, ConvNeXt-Base, CLIP ViT-L/14）权重完全冻结，移除原分类头，提取最后一层特征向量 $\mathbf{x} \in \mathbb{R}^D$，仅训练顶层线性分类器：
   $$z = \mathbf{w}^\top \mathbf{x} + b, \quad p_{\mathrm{AI}} = \sigma(z)$$
   该设计模拟零自适应部署场景，避免微调带来的过拟合与跨模型比较偏差。
3. **类别平衡与损失函数**：原始AI-ArtBench训练池存在2.6:1的AI:Human失衡，通过固定种子随机下采样使每类每风格各保留4,000张，得到120,000张平衡训练集。使用加权二值交叉熵：
   $$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} w_{y_i} \Big[ y_i \log \sigma(z_i) + (1-y_i) \log(1-\sigma(z_i)) \Big]$$
   其中 $w_c = N/(2N_c)$，计算得 Human $w_0=1.5$、AI $w_1=0.75$。
4. **阈值选择与评估协议**：在验证集上以平衡准确率为主指标、F1为 tie-breaker 进行0.01~0.99网格搜索选定最优阈值 $\tau_{\mathrm{val}}$，该阈值**固定锁定**用于ID测试集与OOD测试集，禁止后验调参。
5. **可解释性分析**：对表现最好的CNN模型ConvNeXt-Base应用Grad-CAM，以AI logit $z$ 与人类互补分数 $-z$ 分别计算目标类别的梯度加权特征图，对比ID真阳性、OOD假阴性与OOD假阳性的空间激活分布。

## 实验与结果
- **
