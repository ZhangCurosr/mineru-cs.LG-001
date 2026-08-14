# Sustaining Plasticity via Learnable Wavelet Activations in Continual

Learning

Zeyang Zhang, Tieliang Gong<sup>∗</sup>, Junyan Lu, Weizhan Zhang

Abstract—Plasticity loss has emerged as a critical challenge in continual learning that significantly hinders the acquisition of sequential tasks. While optimizing activation designs offers a potential solution, current fixed-form functions suffer from an inherent spectral bias towards low-frequency variations, whereas learnable variants permit unconstrained updates that induce catastrophic forgetting. To address these limitations, we propose a novel learnable wavelet activation that decomposes the activation function into low-frequency and high-frequency components to explicitly counter spectral bias. Furthermore, we employ dynamic wavelet injection to adaptively enhance plasticity for new tasks, alongside a regularization strategy to ensure the stability of previous learned knowledge. Theoretically, we provide rigorous mathematical guarantees for the proposed framework, proving the structural necessity of the hybrid wavelet architecture for efficient L<sup>2</sup> approximation and demonstrating that the decoupled learning rate mechanism successfully restores network plasticity for high-frequency information. Additionally, we provide a formal derivation of the loss-driven injection trigger mechanism to precisely guide the injection. Extensive empirical evaluations demonstrate that our approach maintains superior trainability and generalization throughout the learning process and achieves state-of-the-art performance across diverse continual learning benchmarks.

Index Terms—Activation function, contihual learning, plasticity loss, spectral bias, wavelet transform.

## I. INTRODUCTION

Continual learning requires neural networks to acquire new knowledge from a sequence of tasks over time without erasing previously learned information [1]. This paradigm imposes a fundamental conflict between the competing requirements of plasticity for adapting to new data and stability for preserving prior knowledge [2]. While extensive research has addressed catastrophic forgetting through replay buffers [3], [4], regularization constraints [5], [6] and parameter isolations [7], [8], recent studies have identified a critical issue known as plasticity loss, where models progressively lose the capacity to adapt to new tasks even when catastrophic forgetting is mitigated [9]–[12].

Current explanations for plasticity loss span multiple levels such as dead neurons [13], large weight norms [14], feature rank [15], the sharpness of the loss landscape [14] and noise memorization [16]. Existing works have primarily focused on resetting inactive units [17], [18], regularizing weights [19], [20], optimizing feature spaces [21], [22] and designing novel architectures [23], [24]. While existing approaches primarily address plasticity loss through optimization constraints or architectural modifications, they frequently neglect the fundamental role of the activation function.

![](images/f0e7990d708a8cfa55a89ac9286a5805fd4b26632b18ba7b9df9934ec51b6099.jpg)  
Fig. 1. Schematic illustration of the ChannelWavAct framework.

In fact, standard activation functions exhibit a structural impediment to continual learning. Although recent approaches like AID and Randomized Smooth-Leaky [25], [26] introduce stochastic mechanisms to sustain plasticity, these methods, similar to the standard ReLU [27], remain bound by fixed functional forms. Consequently, these activations exhibit an inherent spectral bias towards low-frequency functions, prioritizing the learning of global variations while struggling to capture the local details essential for complex tasks [28], [29]. Conversely, fully parameterized paradigms such as Rational approximations and KAN [30], [31] leverage learnable basis functions to enable arbitrary functional adaptation, yet they lack explicit constraints preserving prior task structure. During optimization on new task , gradient descent freely modifies activation parameters to minimize current loss, which can perturb previously learned parameters configurations. This unconstrained plasticity reintroduces catastrophic forgetting at the activation level, independent of weight-space stability mechanisms [1].

To address aforementioned challenges, we propose a novel wavelet-based activation framework that synergizes local adaptability with global stability. Our approach formulates the activation function as a learnable composition of a global base approximation and local wavelet details, incorporating a dynamic wavelet injection mechanism to actively introduce high-frequency components. This design explicitly counters the spectral bias of static functions, thereby enhancing the network’s plasticity. Simultaneously, to mitigate the catastrophic forgetting common in learnable approaches, we introduce a targeted parameter regularization strategy. By imposing constraints on the activation slopes, it maintains the stability of previously learned functional configurations. Comprehensive experiments across multiple continual learning benchmarks demonstrate that our method effectively alleviates plasticity loss and achieves superior performance compared to state-ofthe-art baselines.

## A. Related Work

Continual Learning (CL) envisions models that accumulate knowledge sequentially from dynamic data streams while preventing the erasure of previously consolidated information [1]. The central challenge in this paradigm lies in reconciling the stability-plasticity dilemma [2]. Traditionally, the literature has prioritized the stability aspect, with methodologies predominantly categorized into replay-based [3], [4], regularizationbased [5], [6] and parameter isolation-based methods [7], [8]. Recently, a growing body of literature has identified the loss of model plasticity as a critical bottleneck in long-sequence learning [14], [32]. To mitigate this degradation, current strategies have primarily focused on macroscopic interventions, such as weight re-initialization [17], [18], parameter Regularization [19], [20] and feature space optimization [21], [22]. Distinct from these approaches, we address the plasticity loss directly through the lens of activation functions.

Activation functions serve as the primary gatekeepers of gradient information, determining how much learning signal survives backpropagation. While the dominant ReLU [27] suffers from the irreversible dead-unit problem where neurons become permanently inactive [13], recent approaches have introduced stochastic interventions to mitigate this issue. Specifically, AID [25] employs an interval-wise dropout mechanism to regularize the network towards pseudo-linear behavior, while Randomized Smooth-Leaky [26] integrates function smoothness with randomized negative slopes to sustain robust gradient propagation. However, these methods essentially remain variants of static activation functions. Restricted to pre-defined functional forms, they cannot adaptively evolve to capture the evolving high-frequency details necessary for continual learning. More recently, KAN [31] and Adaptive Rational Activations [30] have introduced learnable B-splines and Pade approximants to achieve superior approximation.´ However, these approaches lack mechanisms to safeguard prior knowledge, where unrestricted adaptation inevitably leads to catastrophic forgetting. We address these limitations by proposing a wavelet activation framework that incorporates dynamic wavelet injection to ensure plasticity and parameter regularization to strictly safeguard historical stability.

## B. Main Contributions

Our main contributions in this work are as follows:

• Wavelet-based activation framework: We introduce ChannelWavAct, which decomposes activations into a fixed low-frequency base for stability and learnable highfrequency wavelets for plasticity. This explicitly achieves adaptability to local information while capturing global information.

• Adaptive capacity expansion with stability preservation: We propose (i) dynamic wavelet injection triggered by loss stagnation to expand model capacity when plasticity saturates, and (ii) slope-specific regularization that selectively penalizes wavelet magnitude parameters while permitting geometric parameters to adapt to distribution shifts.

• Rigorous theoretical foundation: We prove the necessity of our decoupled learning rate mechanism, lossstagnation trigger, and hybrid wavelet architecture. Furthermore, we provide formal guarantees for the $L ^ { 2 }$ approximation capability of ChannelWavAct and its effectiveness in restoring model plasticity for high-frequency components.

• Superior performance: Extensive experiments validate that our proposed wavelet activation exhibits optimal plasticity retention. Our approach consistently outperforms existing baseline activation functions and achieves state-of-the-art performance across a diverse range of continual learning benchmarks.

## II. PRELIMINARY

## A. Problem Setting

We consider the standard Class-Incremental Learning (CIL) setting, where a model $f _ { \theta }$ learns from a sequence of $T$ tasks, denoted as ${ \mathcal { S } } = \{ { \mathcal { T } } _ { 1 } , { \mathcal { T } } _ { 2 } , \ldots , { \mathcal { T } } _ { T } \}$ . Each task $\mathcal { T } _ { t }$ contains a dataset $\mathcal { D } _ { t } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N _ { t } }$ , where $x _ { i } \in \mathcal X _ { t }$ is the input image and $y _ { i } ~ \in ~ \mathcal { D } _ { t }$ is the corresponding label. The label spaces between tasks are disjoint, i.e., $\mathcal { V } _ { t } \cap \mathcal { V } _ { t ^ { \prime } } = \emptyset$ for $t \neq t ^ { \prime } .$ . During the training of task $\mathcal { T } _ { t } ,$ , the model only has access to $\mathcal { D } _ { t }$ . The objective is to minimize the empirical risk over all seen tasks up to the current time step t:

$$
\mathcal { L } _ { t o t a l } ( \theta ) = \mathbb { E } _ { ( x , y ) \sim \mathcal { D } _ { t } } [ \ell ( f _ { \theta } ( x ) , y ) ]\tag{1}
$$

where $\ell ( \cdot )$ is the cross-entropy loss. For simplicity, we decompose the model parameters θ into weight parameters W and bias parameters b.

## B. Kolmogorov-Arnold Networks (KAN)

The Kolmogorov-Arnold Network (KAN) has recently emerged as a promising alternative to Multi-Layer Perceptrons (MLP). While MLP are theoretically grounded in the Universal Approximation Theorem (UAT), KAN draw inspiration from the Kolmogorov-Arnold Representation Theorem (KAT) [33].The KAT asserts that any multivariate continuous function $f ( \mathbf { x } )$ defined on a bounded domain can be represented as a finite composition of continuous univariate functions and the binary operation of addition. Formally, the theorem is expressed as follows:

$$
f (  { \mathbf { x } } ) = f ( x _ { 1 } , \dots , x _ { n } ) = \sum _ { q = 0 } ^ { 2 n } \Phi _ { q } \left( \sum _ { p = 1 } ^ { n } \phi _ { q , p } ( x _ { p } ) \right)\tag{2}
$$

where $\Phi _ { q }$ and $\phi _ { q , p }$ are univariate functions for each variable. KAN parameterize these univariate functions $\phi$ using a linear

![](images/652164d126f2246b3f48e02dbe3f84de8d32138e6c528dc1f870ec4e507d2239.jpg)  
Fig. 2. Evolution of the Dormant Neuron Ratio under Input and Label Trainability settings. The left and right panels display the dormant neuron ratio across sequential tasks for Permuted MNIST and Random Label MNIST.

combination of a basis function $b ( x ) = \mathrm { s i l u } ( x ) = x / ( 1 + e ^ { - x } )$ and a B-spline function spline $\begin{array} { r } { \mathbf { \rho } ( x ) = \sum _ { i } c _ { i } B _ { i } ( x ) } \end{array}$ . The resulting learnable activation function $\phi ( x )$ is defined as:

$$
\phi ( x ) = w _ { b } \cdot b ( x ) + w _ { s } \cdot \mathrm { s p l i n e } ( x )\tag{3}
$$

where the $w _ { b }$ and $w _ { s }$ represent learnable parameters that control the overall magnitude of the activation function. Consequently, a KAN layer can be expressed as:

$$
x _ { l + 1 } = \underbrace { \left( \begin{array} { l l l } { \phi _ { l , 1 , 1 } ( \cdot ) } & { \cdot \cdot \cdot } & { \phi _ { l , 1 , n _ { l } } ( \cdot ) } \\ { \phi _ { l , 2 , 1 } ( \cdot ) } & { \cdot \cdot \cdot } & { \phi _ { l , 2 , n _ { l } } ( \cdot ) } \\ { \vdots } & { \cdot \cdot } & { \vdots } \\ { \phi _ { l , n _ { l + 1 } , 1 } ( \cdot ) } & { \cdot \cdot \cdot } & { \phi _ { l , n _ { l + 1 } , n _ { l } } ( \cdot ) } \end{array} \right) } _ { \Phi _ { l } } x _ { l }\tag{4}
$$

where $\mathbf { x } _ { l }$ and $\mathbf x _ { l + 1 }$ denote the input and output vectors of the layer, respectively, and $\Phi _ { l }$ represents the matrix of 1D univariate functions for layer l. A complete KAN network is constructed by stacking multiple such KAN layers.

## III. LEARNABLE WAVELET-BASED ACTIVATION

Prevalent continual learning architectures rely on fixed-form activation functions, such as the Rectified Linear Unit (ReLU). Despite their efficiency, these functions often suffer from the dead-unit problem: neurons that output 0 receive no gradient and thus become inactive permanently. This permanent loss of capacity progressively diminishes the model’s plasticity, severely constraining its adaptability to subsequent tasks.

Kolmogorov-Arnold Networks (KAN) attempt to solve the dead-unit problem by replacing fixed nodes with learnable activation functions. However, the original KAN relies on Bsplines, which works well for simple tabular data but struggles with high-dimensional features. In these deep learning contexts, spline-based KANs become inefficient and memoryintensive, making them difficult to use in practice. Specifically, as shown in Figure 2, current methods still suffer from neuron dormancy, where the number of inactive neurons rises rapidly to a saturation point.

To address these limitations, we introduce a novel waveletbased activation. Unlike splines which are computationally heavy, our approach leverages wavelets for their efficiency. This allows us to explicitly decompose feature signals into two parts: a stable low-frequency component to preserve existing knowledge and some flexible high-frequency components to quickly adapt to new tasks. This design naturally achieves a better balance between stability and plasticity.

## A. Channel-wise Learnable Wavelet Activation

To overcome the scalability limitations of KAN while keeping the benefits of learnable activations, we propose the Channel-wise Wavelet Activation (ChannelWavAct). This activation is inspired by the Discrete Wavelet Transform (DWT), which builds signals by combining a stable low-frequency base with high-frequency details [34]. Designed to replace standard activations in ResNet backbones, ChannelWavAct operates independently on each channel, learning a task-specific nonlinear transformation by combining a base activation with a set of wavelet bases.

Let $x _ { c }$ denote the input feature map for channel c. We define the activation function $\Phi _ { c } ( x _ { c } )$ as:

$$
\Phi _ { c } ( x _ { c } ) = \underbrace { w _ { l o w , c } \cdot \sigma ( x _ { c } ) } _ { \mathrm { A p p r o x i m a t i o n ~ ( L o w - F r e q ) } } + \underbrace { \sum _ { k = 1 } ^ { K } w _ { c , k } \cdot \psi \left( \frac { x _ { c } - \tau _ { c , k } } { s _ { k } } \right) } _ { \mathrm { D e t a i l ~ ( H i g h - F r e q ) } }\tag{5}
$$

Here, $w _ { c , k }$ controls the amplitude of the k-th wavelet in channel c. The parameter $\tau _ { c , k }$ shifts the wavelet’s position, allowing the network to focus on specific parts of the input. $s _ { k }$ denotes the scale that controls the bandwidth (frequency) of the wavelet. In our implementation, we define this as exp(log s ) to ensure the value stays positive.

Our formulation mimics the inverse DWT by combining a base component for approximation with a wavelet component for details. For the base, we use $\sigma ( x ) = S i L U ( x )$ to capture global low-frequency trends to ensure stability. For the wavelet term, we use the Mexican Hat wavelet $\psi ( u ) = ( u ^ { 2 } - 1 ) \exp ( - u ^ { 2 } / 2 )$ to handle local high-frequency variations. By optimizing the learnable weights $w _ { c , k } ,$ , shifts $\tau _ { c , k }$ , and shared scales $s _ { k } ,$ the network dynamically refines its feature extraction. We share $s _ { k }$ to keep the frequency structure consistent, allowing $w _ { c , k }$ and $\tau _ { c , k }$ to vary by channel. This flexibility ensure the model to capture complex features that fixed-form functions simply miss.

## B. Dynamic Wavelet Injection and Stability Regularization

While replacing static ReLUs with learnable activations enhances the model’s immediate representational power, utilizing a fixed configuration of learnable parameters remains insufficient for continual learning. Most current methods typically constrain the model to a static parameter space. In class-incremental settings, this rigidity creates a dilemma: optimizing the shared activation parameters to accommodate new task distributions inherently changes the feature response for previously learned tasks, causing catastrophic forgetting. Conversely, restricting parameter updates to preserve stability imposes a hard capacity limit, leading to plasticity exhaustion as the sequence of new tasks progresses.

To effectively navigate the stability-plasticity dilemma, we propose dynamic wavelet injection to actively expand model capacity when plasticity is exhausted and stability regularization to strictly preserve knowledge encoded in previous tasks.

1) Dynamic Wavelet Injection: We propose a dynamic mechanism that actively monitors the training dynamics and injects additional learnable wavelet bases when plasticity saturation is detected.

Loss-Based Trigger Mechanism. We introduce a monitoring strategy to detect plasticity loss. Let $\mathcal { L } _ { m i n } ^ { * }$ be the minimum training loss observed so far within the current epoch. At iteration i, given the current batch loss $\mathcal { L } _ { i }$ and a relative margin $\delta ,$ we formalize the update rule for the stagnation counter $C _ { b a d } ^ { \overline { { ( i ) } } }$ and the injection trigger condition as follows:

$$
C _ { b a d } ^ { ( i ) } = \left\{ \begin{array} { l l } { C _ { b a d } ^ { ( i - 1 ) } + 1 } & { \mathrm { i f ~ } \mathcal { L } _ { i } \geq \mathcal { L } _ { m i n } ^ { * } \cdot ( 1 - \delta ) } \\ { 0 } & { \mathrm { o t h e r w i s e } } \end{array} \right.\tag{6}
$$

Here, the counter $C _ { b a d }$ accumulates when the loss fails to show a sufficient drop and resets whenever a new optimum is found. The injection procedure is initiated strictly when this counter exceeds the pre-defined patience threshold $P \ ( { \mathrm { i . e . , } } \ C _ { b a d } \ >$ P), indicating that the representational capacity of the current activation basis is saturated.

Wavelet Basis Expansion. Once triggered, we expand the number of wavelets in the activation function for each channel c. Let $K _ { o l d }$ be the current number of bases. We inject $\Delta$ additional wavelet basis functions, updating the total capacity to $K _ { n e w } = K _ { o l d } + \Delta$ . To ensure these newly added components effectively capture the fine-grained features that the current model fails to resolve, we employ a targeted initialization strategy focused on high-frequency details. Specifically, the parameters for the new bases are set as follows:

• Scale Initialization $s _ { k } \colon$ To ensure positive values and a smooth frequency distribution, we optimize $s _ { k }$ in the logarithmic space. When adding new wavelets, we initialize their scales based on the average of the current ones, i.e. $s _ { n e w } \gets \operatorname* { m a x } ( \epsilon , \frac { 1 } { K } \sum s _ { i } )$ . This approach prevents abrupt shifts in frequency.

• Translation $\tau _ { c , k }$ and Weight $w _ { c , k } \mathbf { : }$ We adopt a zeroinitialization strategy where both the translation and weight parameters for the new wavelets are set to zero. This ensures new components initially contribute zero output, preventing sudden feature space perturbations and allowing the model to integrate high-frequency details gradually during training.

2) Slope-Specific Regularization: While injection provides plasticity, we must prevent the modification of existing bases from erasing old knowledge. We employ a partial regularization strategy that acts solely on the old coefficients, decoupling stability from plasticity.

When transitioning from task $\textit { t }  { t o } \textit { t } + \textit { 1 }$ , we save a snapshot of the current activation parameters: $\theta _ { o l d } \ =$ $\{ w _ { 1 : K } , \tau _ { 1 : K } , s _ { 1 : K } \}$ . During the training of task $t + 1$ (where the total number of bases increases to $K _ { t o t a l } )$ , we add a regularization term to the objective function:

$$
\mathcal { L } ( \boldsymbol { \theta } ) = \mathcal { L } _ { C E } + \lambda \mathcal { L } _ { r e g }\tag{7}
$$

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { r e g } ( \theta ) = \sum _ { c = 1 } ^ { C } { \biggl ( \lambda _ { l o w } \| w _ { l o w , c } - w _ { l o w , c } ^ { o l d } \| ^ { 2 } } } \\ { \displaystyle \qquad + \ \lambda _ { w a v } \sum _ { k = 1 } ^ { K _ { o l d } } { \| w _ { c , k } - w _ { c , k } ^ { o l d } \| ^ { 2 } } \biggr ) } \end{array}\tag{8}
$$

where $K _ { o l d }$ represents the number of bases before the current injection and the summation is limited to the range $k \_ =$ $1 \ldots K _ { o l d } ,$ covering only the components that existed before the expansion.

Here the newly injected bases $( k = K _ { o l d } + 1 , \dots , K _ { o l d } + \Delta )$ are not subject to regularization. This allows the new units to fully adapt to the new task. We choose to selectively penalize the weights $( w _ { l o w }$ and $w _ { c , k } )$ while leaving the geometric parameters (translations τ and scales s) free. The weights control the strength of feature signals. Thus constraining these parameters preserves the core features learned in previous tasks, effectively maintaining memory. Conversely, τ and s determine the position and frequency of the activation. By allowing these geometric parameters to adjust freely, we enable the existing bases to handle small shifts in the data distribution during continual learning.

3) Decoupled Optimization and Post-Activation Normalization: The use of learnable activation functions and dynamic parameter injection requires us to adjust standard optimization methods. We introduce a specialized optimization strategy and a post-activation normalization technique to ensure stable convergence.

Decoupled Learning Rates and Optimizer Reset. Since the model architecture changes during each injection phase, the previous optimizer state becomes invalid. Therefore, we rebuild the optimizer state to accommodate the new parameter space. To strictly enforce plasticity without sacrificing basic stability, we divide the model parameters into two distinct groups with specific optimization settings:

• Plasticity Group: We suggest that adaptation to new tasks requires significant adjustments in both feature extraction and frequency filtering. Therefore, we assign an amplified learning rate $\eta _ { h i g h } = \lambda _ { l r } \cdot \eta _ { b a s e }$ to both the backbone weights W and the wavelet activation parameters $\theta _ { a c t } = \{ w , \tau , s \}$ . This high learning rate ensures that the model can rapidly restructure its backbone weights and activation shapes to capture the novel distributions of the incoming task, effectively preventing under-fitting.

• Stability Group: Conversely, regarding the bias parameters b and batch normalization affine parameters, we maintain their learning rate at the standard base level $\eta _ { b a s e }$ . These parameters generally capture baseline statistics that tend to stay consistent across tasks. Preserving a lower learning rate for these components prevents the model representation from drifting uncontrollably during the aggressive adaptation of weights and activations.

Post-Activation Batch Normalization. Standard ResNet blocks usually follow a Conv-BN-ReLU sequence. However, our wavelet activation involves the summation of multiple potentially unbounded functions, which can lead to significant variance shifts in the feature map distribution, especially as K increases. To control this variance and ensure stable training, we apply an additional post-activation batch normalization step directly inside the ChannelWavAct module. Formally, the output of the activation function is defined as:

Algorithm 1 ChannelWavAct Forward Pass   
1: Input: Input x, Wavelets Numbers ${ \overline { { K } } } ,$ , Parameters $\theta =$   
$\{ w , \tau , s \}$   
2: Output: Activation Function $y = \Phi ( x )$   
3: for channel $c = 1$ to $C$ do   
4: $\begin{array} { r } { \Phi _ { c } ( x _ { c } ) = w _ { l o w , c } \sigma ( x _ { c } ) + \sum _ { j = 1 } ^ { K } w _ { c , j } \psi _ { j } ( \frac { x _ { c } - \tau _ { c , j } } { s _ { j } } ) } \end{array}$   
5: Post-Activation Batch Normalization: $y _ { c }$ $=$   
$B N \big ( \Phi _ { c } ( x _ { c } ) \big )$   
6: end for   
7: $\boldsymbol { y } = \boldsymbol \Phi ( \boldsymbol { x } ) = \mathrm { C o n c a t e n a t e } ( \{ y _ { 1 } , \dots , y _ { C } \} )$   
8: Return $y = \Phi ( x )$

$$
\mathbf { y } _ { o u t } = \mathbf { B } \mathbf { N } \left( \Phi _ { c } ( \mathbf { x } _ { i n } ) \right)\tag{9}
$$

where $\Phi _ { c }$ is the wavelet transformation defined in Equation (5). This ensures that regardless of the number of injected wavelets or their amplitude, the signal propagates to the next convolutional layer with a normalized distribution.

## C. Pseudo-code of ChannelWavAct

The forward propagation of ChannelWavAct is detailed in Algorithm 1. The process operates channel-wise. For each channel $c ,$ the activation $\Phi _ { c } ( x _ { c } )$ aggregates a base term $\sigma ( x _ { c } )$ with a weighted sum of $K$ wavelet bases, which use learnable translation $\tau _ { c , j }$ and scaling $s _ { j }$ to capture multi-scale details. To mitigate internal covariate shifts from this aggregation, a Post-Activation Batch Normalization (BN) is applied to each channel before concatenating the outputs into the final tensor y.

The comprehensive training procedure for ChannelWavAct is formalized in Algorithm 2. The procedure manages plasticity through dynamic monitoring. For each batch, a counter $C _ { b a d }$ tracks loss $( \mathcal { L } _ { C E } )$ plateaus where improvement falls below margin $\delta .$ Once $C _ { b a d }$ exceeds threshold $P ,$ the system injects $\Delta$ new wavelets per channel. To facilitate adaptation, the optimizer is then rebuilt using a decoupled learning rate strategy: a higher rate $\lambda _ { l r } \eta _ { b a s e }$ is assigned to backbone and activation parameters $( \mathbf { W } , w , \tau , s )$ to enhance plasticity, while biases retain the base rate $\eta _ { b a s e }$ . Finally, the model is updated via a composite objective $\mathcal { L } _ { t o t a l } ~ = ~ \mathcal { L } _ { C E } + \mathcal { L } _ { r e g }$ , where $\mathcal { L } _ { r e g }$ selectively penalizes deviations in pre-existing bases $( k \le K _ { o l d } )$ to mitigate forgetting without constraining newly injected components.

## IV. THEORETICAL RESULTS

We now provide rigorous theoretical foundations to establish the necessity of the proposed ChannelWavAct framework.

Theorem 1: Let $\Theta ( t )$ be the local dynamic Neural Tangent Kernel (NTK) [35] of a finite-width network. For a projection direction $v _ { i } \in \mathbb { R } ^ { N } \left( \lVert v _ { i } \rVert = 1 \right)$ representing a local task feature at frequency i, assume it satisfies the approximate eigensystem $\Theta ( t ) v _ { i } \approx \lambda _ { i } ( t ) v _ { i }$ with the instantaneous eigenvalue $\lambda _ { i } ( t ) =$ $v _ { i } ^ { T } \Theta ( t ) v _ { i }$ . Under a uniform learning rate $\eta _ { b a s e } .$ , the localized residual component $u _ { i } ( t ) = v _ { i } ^ { T } u ( t )$ evolves according to:

Algorithm 2 Training Procedure of ChannelWavAct   
1: Input: Sequential Tasks $\overline { { \mathcal { T } = \{ T _ { 1 } , . . . , T _ { N } \} } }$ , Patience $P ,$   
Threshold δ, Reg Coef $\lambda , \lambda _ { l o w }$ and $\lambda _ { w a v } ,$ Learning Rate   
Multiplier $\lambda _ { l r }$ , Base LR $\eta _ { b a s e } ,$ , Number of new wavelets   
$\Delta$ , Wavelets Numbers $K$   
2: Initialize: Model parameters $\theta = \{ \mathbf { W } , \mathbf { b } , w , \tau , s \}$ , Opti  
mizer $O p t$   
3: for task $k = 1$ to $N$ do   
4: for epoch $e = 1$ to $E$ do   
5: Reset $\mathcal { L } _ { m i n } ^ { * }  \infty , C _ { b a d }  0$   
6: for $( x , y )$ in $D _ { k }$ do   
7: Model output $f ( x )$   
8: $\mathcal { L } _ { C E } = \mathrm { C r o s s E n t r o p y } ( f ( x ) , y )$   
9: // Loss-based Trigger Mechanism   
10: if $\mathcal { L } _ { C E } \geq \mathcal { L } _ { m i n } ^ { * } \cdot ( 1 - \delta )$ then   
11: $C _ { b a d } \gets C _ { b a d } + 1$   
12: else   
13: $C _ { b a d }  0 ; \mathcal { L } _ { m i n } ^ { * }  \mathcal { L } _ { C E }$   
14: end if   
15: // Dynamic Injection & Optimizer Rebuild   
16: if $C _ { b a d } > P$ then   
17: for channel $c = 1$ to $C$ do   
18: Add new wavelet component $\psi _ { n e w }$ (Update   
$K \gets K + \Delta )$   
19: Initialize new weight $w , \tau , s$   
20: end for   
21: // Decoupled Learning Rates   
22: $O p t \gets ( \{ \mathbf { b } , \eta _ { b a s e } \} , \{ \mathbf { W } , w , \tau , s , \lambda _ { l r } \cdot \eta _ { b a s e } \} )$   
23: $C _ { b a d } \gets 0$   
24: end if   
25: // Regularization   
26: $\begin{array} { r l r } { \mathcal { L } _ { r e g } } & { { } = } & { \sum _ { c = 1 } ^ { C } \Big ( \lambda _ { l o w } \| w _ { l o w , c } ~ - ~ w _ { l o w , c } ^ { o l d } \| ^ { 2 } ~ + } \end{array}$   
$\begin{array} { r } { \lambda _ { w a v } \sum _ { j = 1 } ^ { K _ { o l d } } \| w _ { c , j } - w _ { c , j } ^ { o l d } \| ^ { 2 } \bigg ) } \end{array}$   
27: $\mathcal { L } _ { t o t a l } = \mathcal { L } _ { C E } + \lambda \mathcal { L } _ { r e g }$   
28: Update: $O p t . s t e p ( \nabla \bar { \mathcal { L } } _ { t o t a l } )$   
29: end for   
30: end for   
31: end for

$$
| { u } _ { i } ( t ) | = | { u } _ { i } ( 0 ) | \exp \left( - \eta _ { b a s e } \int _ { 0 } ^ { t } \lambda _ { i } ( \tau ) d \tau \right)\tag{10}
$$

Proof. Under a single learning rate $\begin{array} { r }  P = \eta _ { b a s e } I _ { \mathrm { : } }  \end{array}$ , the residual evolution equation degenerates into:

$$
\frac { d u ( t ) } { d t } = - \eta _ { b a s e } \Theta ( t ) u ( t )\tag{11}
$$

Multiplying both sides by the projection direction $v _ { i } ^ { T }$ from the left:

$$
v _ { i } ^ { T } \frac { d u ( t ) } { d t } = - \eta _ { b a s e } v _ { i } ^ { T } \Theta ( t ) u ( t )\tag{12}
$$

According to the assumption, $v _ { i }$ is the eigenvector corresponding to the instantaneous eigenvalue $\lambda _ { i } ( t )$ , satisfying

$v _ { i } ^ { T } \Theta ( t ) = \lambda _ { i } ( t ) v _ { i } ^ { T }$ . Substituting this into the above equation, we obtain:

$$
\frac { d ( v _ { i } ^ { T } u ( t ) ) } { d t } = - \eta _ { b a s e } \lambda _ { i } ( t ) ( v _ { i } ^ { T } u ( t ) )\tag{13}
$$

Defining $u _ { i } ( t ) = v _ { i } ^ { T } u ( t )$ , the equation can be rewritten as a first-order linear ordinary differential equation with variable coefficients:

$$
\frac { d u _ { i } ( t ) } { d t } = - \eta _ { b a s e } \lambda _ { i } ( t ) u _ { i } ( t )\tag{14}
$$

Integrating both sides from 0 to t yields the analytical solution:

$$
u _ { i } ( t ) = u _ { i } ( 0 ) \exp \left( - \eta _ { b a s e } \int _ { 0 } ^ { t } \lambda _ { i } ( \tau ) d \tau \right)\tag{15}
$$

□

According to the Dynamic Frequency Principle [36], the instantaneous eigenvalues associated with high-frequency features remain orders of magnitude smaller than those of low frequencies, i.e., $\lambda _ { \mathrm { h i g h } } ( \tau ) ~ \ll ~ \lambda _ { \mathrm { l o w } } ( \tau )$ . As a result, the integral $\textstyle \int _ { 0 } ^ { t } \lambda _ { \mathrm { h i g h } } ( \tau ) d \tau$ remains negligibly small over any finite number of training epochs, yielding a residual decay factor of $\exp ( \cdot ) \ \approx \ 1$ . It indicates that a uniform learning rate is insufficient to suppress high-frequency residuals within a practical training horizon, which in turn limits the network’s plasticity in capturing complex local features. This observation motivates a decoupled learning rate strategy, whose necessity we formally stated in Theorem 2.

Assumption 1: For a high-frequency task feature direction $v _ { \mathrm { h i g h } } .$ , it is assumed based on the Frequency Principle [36] that features at distinctly different frequencies exhibit nearorthogonality in the functional space. Consequently, projecting the global low-frequency bases onto this high-frequency direction yields only a negligible residual response $\epsilon ( t ) \ll 1$ while the localized wavelet bases dominate this subspace with an instantaneous eigenvalue $\lambda _ { \mathrm { h i g h } } ( t )$ . Formally:

$$
\begin{array} { r l } & { \Theta _ { b a s e } ( t ) v _ { h i g h } = \epsilon ( t ) v _ { h i g h } , \quad \mathrm { w h e r e ~ } \epsilon ( t ) \ll 1 } \\ & { \Theta _ { h i g h } ( t ) v _ { h i g h } = \lambda _ { h i g h } ( t ) v _ { h i g h } } \end{array}\tag{16}
$$

Theorem 2: Let the Neural Tangent Kernel (NTK) matrix be decomposed into a linear superposition of two subkernels: $\Theta ( t ) \ = \ \Theta _ { b a s e } ( t ) + \Theta _ { h i g h } ( t )$ . Under Assumption 1, by employing a decoupled learning rate matrix $P =$ dia $\mathbf { g } ( \eta _ { b a s e } I _ { b a s e } , \eta _ { h i g h } I _ { h i g h } )$ , the effective network dynamics are governed by the modified Preconditioned NTK $\tilde { \Theta } ( t )$ satisfying:

$$
\tilde { \Theta } ( t ) = \eta _ { b a s e } \Theta _ { b a s e } ( t ) + \eta _ { h i g h } \Theta _ { h i g h } ( t )\tag{17}
$$

Proof. Substituting the preconditioned gradient flow into the residual evolution equation, the dynamical system can be rewritten as:

$$
\begin{array} { c } { \displaystyle \frac { d u ( t ) } { d t } = - [ \nabla _ { \theta _ { b a s e } } f ^ { T } ( \eta _ { b a s e } I ) \nabla _ { \theta _ { b a s e } } f } \\ { \displaystyle + \nabla _ { \theta _ { h i g h } } f ^ { T } ( \eta _ { h i g h } I ) \nabla _ { \theta _ { h i g h } } f ] u ( t ) } \\ { \displaystyle = - ( \eta _ { b a s e } \Theta _ { b a s e } ( t ) + \eta _ { h i g h } \Theta _ { h i g h } ( t ) ) u ( t ) } \\ { \displaystyle = - \tilde { \Theta } ( t ) u ( t ) } \end{array}\tag{18}
$$

where $\tilde { \Theta } ( t )$ represents the effective preconditioned NTK. $\mathsf { A p - }$ plying $\tilde { \Theta } ( t )$ to the high-frequency direction $v _ { h i g h }$ and invoking the approximate orthogonality in Assumption 1, we obtain:

$$
\begin{array} { c } { { \tilde { \Theta } ( t ) v _ { h i g h } = \eta _ { b a s e } \bigl ( \epsilon ( t ) v _ { h i g h } \bigr ) + \eta _ { h i g h } \bigl ( \lambda _ { h i g h } ( t ) v _ { h i g h } \bigr ) } } \\ { { { } } } \\ { { { } = \bigl ( \eta _ { h i g h } \lambda _ { h i g h } ( t ) + \eta _ { b a s e } \epsilon ( t ) \bigr ) v _ { h i g h } } } \end{array}\tag{19}
$$

This confirms that $v _ { h i g h }$ remains an eigenvector of the preconditioned operator $\tilde { \Theta } ( i )$ with an amplified effective eigenvalue $\tilde { \lambda } _ { h i g h } ( t ) = \eta _ { h i g h } \lambda _ { h i g h } ( t ) + \eta _ { b a s e } \epsilon ( t )$ . The analytical solution for the high-frequency residual $u _ { h i g h } ( t )$ is given by:

$$
\begin{array} { l } { { u _ { h i g h } ( t ) = u _ { h i g h } ( 0 ) } } \\ { \displaystyle \cdot \exp \left( - \int _ { 0 } ^ { t } [ \eta _ { h i g h } \lambda _ { h i g h } ( \tau ) + \eta _ { b a s e } \epsilon ( \tau ) ] d \tau \right) } \end{array}\tag{20}
$$

Given that $\epsilon ( \tau ) \approx 0$ due to negligible spectral leakage, the decay rate is predominantly controlled by $\eta _ { h i g h } \lambda _ { h i g h } ( \tau )$ . By explicitly setting $\eta _ { h i g h } = \lambda _ { l r } \eta _ { b a s e }$ with $\lambda _ { l r } \gg 1$ , the inherent high-frequency spectral bias is mathematically compensated, thereby restoring the instantaneous fitting capability of the wavelet bases. □

Notably, the decomposition $\Theta ( t ) \ = \ \Theta _ { \mathrm { b a s e } } ( t ) + \Theta _ { \mathrm { h i g h } } ( t )$ originates from the linear combination of the global base and local wavelets, which splits their gradient operators and eliminates cross-terms under a decoupled optimization framework. Furthermore, the results of Theorem 2 demonstrate that our decoupled learning rate mechanism effectively compensates for the deficiency of the original spectral components. By setting $\eta _ { \mathrm { h i g h } } > \eta _ { \mathrm { b a s e } } .$ , this mechanism accelerates the convergence rate of high-frequency components, thereby enabling synchronous convergence across the entire spectrum. Next, we prove the rationality of the loss-driven monitoring mechanism in Theorem 3 to precisely determine the optimal triggering condition for the injection.

Theorem 3: Let $\mathcal { D } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ be the training set. At time t, denote the function space spanned by K wavelet bases as $\mathcal { H } _ { K } = \operatorname { s p a n } \{ \psi _ { \tau _ { 1 } } , . . . , \psi _ { \tau _ { K } } \}$ , the empirical residual vector as $u _ { K } ( t ) = f _ { K } ( X , \bar { t } ) - Y \in \bar { \mathbb { R } } ^ { N }$ and the local dynamic NTK matrix as $\Theta _ { K } ( t ) \in \mathbb { R } ^ { N \times N }$ . Under continuous-time gradient flow dynamics, the following properties hold:

1) Optimization stagnation is strictly characterized by the residual vector entering the approximate null space of $\Theta _ { K } ( t )$ . For a given tolerance $\epsilon _ { \mathrm { { t o l } } } > 0$ , this condition is equivalent to the Rayleigh quotient collapse [37]:

$$
\frac { u _ { K } ( t ) ^ { T } \Theta _ { K } ( t ) u _ { K } ( t ) } { \| u _ { K } ( t ) \| _ { 2 } ^ { 2 } } \leq \epsilon _ { t o l }\tag{21}
$$

2) Let $\Omega _ { \mathrm { r e s } } = \operatorname* { s u p p } ( \mathcal { F } ( u _ { K } ) )$ be the fourier-domain support of the residual, $\mu ( \cdot )$ be the lebesgue measure and $B _ { \psi }$ be the effective frequency bandwidth of a single wavelet. To ensure the expanded feature space fully resolves the residual, the minimum optimal incremental capacity $\Delta ^ { * }$ satisfies:

$$
\Delta ^ { * } = \left\lceil \frac { \mu ( \Omega _ { r e s } ) } { B _ { \psi } } \right\rceil\tag{22}
$$

Proof. According to the gradient flow evolution $\begin{array} { r l } { \frac { d u _ { K } ( t ) } { d t } } & { = } \end{array}$ $- \Theta _ { K } ( t ) u _ { K } ( t )$ , the time derivative of the squared residual norm $\| u _ { K } ( t ) \| _ { 2 } ^ { 2 }$ is derived as:

$$
\frac { d } { d t } \| u _ { K } ( t ) \| _ { 2 } ^ { 2 } = - 2 u _ { K } ( t ) ^ { T } \Theta _ { K } ( t ) u _ { K } ( t ) .\tag{23}
$$

Optimization stagnation occurs when the rate of residual decay falls below a threshold $\epsilon _ { t o l }$ relative to its current magnitude, i.e., $\begin{array} { r } { | \frac { d } { d t } | | u _ { K } ( t ) | | _ { 2 } ^ { 2 } | \leq 2 \epsilon _ { t o l } | | u _ { K } ( t ) | | _ { 2 } ^ { 2 } } \end{array}$ . Rearranging this inequality yields the Rayleigh Quotient bound:

$$
\frac { u _ { K } ( t ) ^ { T } \Theta _ { K } ( t ) u _ { K } ( t ) } { \| u _ { K } ( t ) \| _ { 2 } ^ { 2 } } \leq \epsilon _ { t o l } ,\tag{24}
$$

which indicates that the residual $u _ { K } ( t )$ has entered the $\mathrm { a p - }$ proximate null space of the current NTK $\Theta _ { K } ( t )$

Then, let $\Delta \mathcal { H } ~ = ~ \mathsf { s p a n } \{ \psi _ { \tau _ { n e w , i } } \} _ { i = 1 } ^ { \Delta K }$ be the incremental function space. By Parseval’s Identity [38], the projection error $\mathcal { E } _ { p r o j }$ can be expressed in the frequency domain as the energy not covered by the new wavelet filters:

$$
\mathcal { E } _ { p r o j } \propto \int _ { \omega \in \mathbb { R } } \left| \mathcal { F } ( u _ { K } ) ( \omega ) \cdot \Big ( 1 - \sum _ { i = 1 } ^ { \Delta } | \mathcal { F } ( \psi _ { \tau _ { i } } ) ( \omega ) | ^ { 2 } \Big ) \right| d \omega .\tag{25}
$$

To ensure $\mathcal { E } _ { p r o j }  0 _ { : }$ , the union of the spectral supports of the injected bases must cover the support of the residual $\Omega _ { r e s } = \mathrm { s u p p } ( \mathcal { F } ( u _ { K } ) )$ . Invoking the lebesgue measure $\mu ( \cdot )$ and the covering lemma in [39], we have:

$$
\sum _ { i = 1 } ^ { \Delta } \mu ( \operatorname { s u p p } ( { \mathcal { F } } ( \psi _ { \tau _ { i } } ) ) ) \geq \mu ( \Omega _ { r e s } ) \implies \Delta \cdot B _ { \psi } \geq \mu ( \Omega _ { r e s } ) .\tag{26}
$$

Thus, the minimum optimal number of bases is $\Delta ^ { * } =$ $\lceil \mu ( \Omega _ { r e s } ) / B _ { \psi } \rceil$ □

Theorem 3 provides a theoretical foundation for our loss detection mechanism. Since the numerator of the Rayleigh quotient governs the time derivative of the loss, i.e., $\begin{array} { r } { \frac { d } { d t } { \mathcal { L } } = } \end{array}$ $- u _ { K } ^ { T } \Theta _ { K } u _ { K }$ , its collapse manifests as a loss stagnation plateau. This justifies our monitoring strategy, validating its rationality as an effective proxy. Furthermore, the spectral coverage bound $\Delta ^ { * }$ quantitatively analyzes the minimal number of wavelet bases required to resolve the frequency components of the remaining error. To further justify the specific functional form and $L ^ { 2 }$ approximation capability of these injected components, we establish the structural necessity of our hybrid wavelet architecture in Theorem 4.

Theorem 4: Let $K \ = \ [ - M , M ] \ \subset \ \mathbb { R }$ be a compact set, and let $L ^ { 2 } ( K )$ denote the space of square-integrable target signals. We define a hybrid feature dictionary $\mathcal { D } =$ $\{ \boldsymbol { \bar { \phi ( x ) } } \} \cup \{ \psi _ { s , \tau } ( x ) \}$ , where $\phi ( x ) \ : = \ : x \cdot \sigma ( x )$ serves as the global low-frequency scaling function and $\psi _ { s , \tau } ( x ) = ( 1 -$ $\bar { ( } s ^ { - 1 } ( x - \tau ) ) ^ { 2 } ) \bar { e } ^ { - ( s ^ { - 1 } ( x - \tau ) ) ^ { 2 } / 2 }$ is the localized high-frequency mother wavelet. For any target signal $f \in L ^ { 2 } ( K )$ and any approximation tolerance $\epsilon > 0$ , there exists a finite set of parameters $\Theta = \{ w _ { l o w } \} \cup \{ w _ { k } , s _ { k } , \tau _ { k } \} _ { k = 1 } ^ { N }$ such that the hybrid operator $\begin{array} { r } { \Phi ( x ) = w _ { l o w } \phi ( x ) + \sum _ { k = 1 } ^ { N } w _ { k } \psi _ { s _ { k } , \tau _ { k } } ( x ) } \end{array}$ satisfies:

$$
\| f ( x ) - \Phi ( x ) \| _ { L ^ { 2 } ( K ) } < \epsilon\tag{27}
$$

Proof. A pure wavelet system $\begin{array} { r } { g ( x ) ~ = ~ \sum w _ { k } \psi ( \frac { x - \tau _ { k } } { s _ { k } } ) } \end{array}$ is composed of Mexican Hat mother wavelets $\psi ( x ) \ = \ ( 1 \ -$ $x ^ { 2 } ) e ^ { - x ^ { 2 } / 2 }$ , which satisfy $\begin{array} { r } { \int _ { - \infty } ^ { \infty } \psi ( x ) d x \ = \ 0 . } \end{array}$ Consequently, $\begin{array} { r } { \int _ { \mathbb { R } } g ( x ) d x = 0 } \end{array}$ . For any target signal $f ( x )$ with a non-zero mean $\begin{array} { r } { \dot { C } = \int _ { K } f ( x ) d x \neq 0 } \end{array}$ on a compact set $K ,$ , a pure wavelet system requires divergent scales $( s \to \infty )$ to approximate the global trend, leading to parameter instability in finite-width networks.

To resolve this, we introduce the SiLU-based scaling function $\phi ( x ) = x \sigma ( x )$ . Since $\phi ( x ) > 0$ for $x > 0$ and decays exponentially for $x \ < \ 0$ , its integral over $K = [ - M , M ]$ denoted as $I _ { \phi } ,$ , is strictly positive. We decompose the signal as $f ( x ) = w _ { l o w } \phi ( x ) + f _ { r e s } ( x )$ . By setting the optimal weight:

$$
w _ { l o w } ^ { * } = \frac { \int _ { K } f ( x ) d x } { I _ { \phi } } = \frac { C } { I _ { \phi } } ,\tag{28}
$$

we ensure that the remaining residual $f _ { r e s } ( x )$ has a strictly zero mean, i.e., $\textstyle \int _ { K } f _ { r e s } ( x ) d x = 0$

For the zero-mean residual $f _ { r e s } ~ \in ~ L ^ { 2 } ( K )$ , we apply the Plancherel Theorem [40] to map the $L ^ { 2 }$ norm into the frequency domain:

$$
\Vert f _ { r e s } - g \Vert _ { L ^ { 2 } } ^ { 2 } = \frac { 1 } { 2 \pi } \int _ { - \infty } ^ { \infty } | \hat { f } _ { r e s } ( \omega ) - \hat { g } ( \omega ) | ^ { 2 } d \omega .\tag{29}
$$

Since $\hat { f } _ { r e s } ( 0 ) = 0 \ :$ , the energy is concentrated in the nonzero frequency bands. Utilizing the Frame Truncation Theorem [41], there exists a finite set of wavelet parameters $\{ w _ { k } , s _ { k } , \tau _ { k } \} _ { k = 1 } ^ { N }$ such that the high-frequency approximation error is bounded by ϵ. Combining this with the low-frequency component via the triangle inequality, we conclude:

$$
\| f ( x ) - \Phi ( x ) \| _ { L ^ { 2 } ( K ) } = \| ( w _ { l o w } ^ { * } \phi + f _ { r e s } ) - ( w _ { l o w } ^ { * } \phi + g ) \| _ { L ^ { 2 } } < \epsilon ,\tag{30}
$$

□

Theorem 4 establishes the indispensability of the hybrid architecture of ChannelWavAct. Specifically, the structural necessity constraint dictates that for signals with a non-zero mean $( \dot { J } _ { K } f ( x ) d x \neq 0 )$ , a system composed solely of wavelets would require divergent scales $( s \to \infty )$ to approximate global trends, leading to a catastrophic collapse in efficiency for finite-width networks. By integrating the SiLU-based scaling function with localized wavelets, ChannelWavAct ensures optimal $L ^ { 2 }$ approximation for both global structures and local high-frequency details.

## V. EXPERIMENTS

## A. Continual Learning

To comprehensively validate the superiority of our proposed method in continual learning scenarios, we benchmark our wavelet-based activation against mainstream baselines from two critical perspectives: trainability and generalizability. These baselines can be broadly divided into static and learnable approaches. The static baselines include the standard ReLU [27], AID [25] and Randomized Smooth-Leaky activations [26]. The learnable baselines comprise Rational Activations [30] and KAN [31].

![](images/3cb604bb6a64b6e47b9452e0a5c0d38912ed111469df1dda92a0b92dd46c7d1e.jpg)  
(a) Trainability experimental results.

![](images/4e3aa4a2228bd96d661cf28b5fa57d5e80f0a5a15afb1344e19b675d39bc68e6.jpg)

![](images/67917a7918dad762e281ef0fc6169ee7b9decc27602531f4f2db4d2de142ccdd.jpg)  
(b) Evolution of Effective Rank (srank) across sequential tasks.

Fig. 3. (a) Accuracy of various activation functions on the Permuted MNIST and Random Label MNIST datasets. The vertical axis denotes Accuracy and the horizontal axis represents the Task ID. The plots show results for ReLU, AID, Randomized Smooth-Leaky (top row), Rational, B-spline and ChannelWavAct (bottom row). (b) Evaluation of Input Plasticity on Permuted MNIST (left) and Label Plasticity on Random Label MNIST (right).

1) Trainability: To assess the input and label plasticity of our wavelet-based activation against existing baselines [17], [42], we follow the evaluation protocol in AID [25] and utilize two standard benchmarks derived from the MNIST dataset [43]. Specifically, for input plasticity, we employ Permuted MNIST, which contains 10 handwritten digit classes with 60,000 training images and 10,000 testing images. We generate a sequence of 200 tasks where the $2 8 \times 2 8$ image pixels are arbitrarily shuffled with a fixed random permutation for each task [17]. Parallelly, to evaluate label plasticity, we use Random Label MNIST, which utilizes a smaller subset of 1,600 training images to simulate a rapid fitting scenario[cite: 1063, 1064]. In this setting, we construct 200 tasks where the class labels of the $2 8 \times 2 8$ images are randomly shuffled for each task while the input pixels remain fixed [42].

Figure 3a presents comparative results of trainability retention across continual learning of 200 tasks. First, ReLU exhibits substantial performance degradation across all tasks, particularly showing a rapid collapse in input trainability. Second, while other static baselines maintain stability, they fail to achieve optimal accuracy, particularly in input trainability scenarios, suggesting a limited trainability ceiling. Furthermore, learnable variants demonstrate inconsistent robustness: KAN fails to retain trainability in input trainability settings, exhibiting performance degradation comparable to ReLU, while Rational suffers from significant instability in label trainability tasks. In contrast, our proposed ChannelWavAct achieves superior and stable performance across all evaluated tasks, demonstrating exceptional robustness and sustained

trainability.

The evolution of the Effective Rank is illustrated in Figure 3b. We strictly follow the srank definition proposed by [15]. It can be observed that the standard ReLU activation suffers from a precipitous decline in effective rank as tasks progress, implying that the representation space degenerates into a low-dimensional subspace that severely limits the network’s capacity to encode new information. While the Rational activation maintains a high srank, its suboptimal downstream performance in Tables I and II suggests that these high-dimensional representations may lack semantic discriminability or contain excessive noise. In contrast, our proposed ChannelWavAct demonstrates a robust and sustained high effective rank that correlates with superior accuracy. This distinction confirms that unlike ReLU which loses capacity, and Rational which generates ineffective dimensions, ChannelWavAct successfully maintains a meaningful and rich representation space, allowing the backbone to continuously assimilate new features without plasticity loss.

2) Generalizability: Our preceding experiments have substantiated the efficacy of ChannelWavAct in maintaining the adaptability required for incoming data distributions. However, beyond mere adaptation, ensuring robust generalization to unseen data remains a paramount objective. We posit that generalization capability constitutes a critical dimension closely associated with the challenge of plasticity loss. To rigorously evaluate this attribute, we conduct assessments across two complementary continual learning paradigms: replay-based strategies and replay-free approaches.

Replay-based approach In the replay-based evaluation, we utilize the SCR framework [45], adopting the SSD method as the continual learning method [4]. This paradigm addresses catastrophic forgetting by maintaining a limited memory buffer of historical samples, which are interleaved with the current data stream during training. Utilizing a Reduced ResNet-18 backbone, our primary objective is to assess whether our proposed activation can significantly improve the model’s plasticity in acquiring new tasks. To establish a rigorous benchmark, we compare our ChannelWavAct against a comprehensive spectrum of baselines, explicitly evaluating performance based on Last Accuracy (Last), Average Accuracy (Avg), and Forgetting (Fgt).

The generalization performance of different activation functions within the replay-based SSD framework is summarized in Table I. As observed, ChannelWavAct consistently achieves superior performance across all three benchmark datasets, including Mini-ImageNet, CIFAR100, and Tiny-ImageNet. It secures the highest values in both Last and Average Accuracy metrics. In terms of Average Accuracy, ChannelWavAct achieves a gain of 2.0% on Mini-ImageNet (39.0% vs. 37.0%), 3.2% on CIFAR100 (43.5% vs. 40.3%) and 2.7% on the challenging Tiny-ImageNet (19.9% vs. 17.2%). Moreover, even with a primary focus on enhancing plasticity, our method still demonstrates a comparable capability in mitigating catastrophic forgetting. These consistent improvements validate its exceptional capability to generalize effectively in replay-based continual learning scenarios.

Replay-free approach In the replay-free evaluation, we operate within the PyCIL framework [44], implementing the EWC method [6] and WA method [46] using a standard ResNet-18 architecture. Unlike replay-based strategies, these approaches rely solely on parameter regularization or weight alignment to preserve past knowledge, thereby imposing a more rigorous test on the network’s learning capacity. Consequently, our objective is to verify whether ChannelWavAct can maintain sufficient plasticity for new tasks even when model parameters are subject to regularization constraints. We benchmark our approach against the same set of baseline activations, using Average Accuracy (Avg) and Forgetting (Fgt) as key metrics [44] to quantify the trade-off between stability and plasticity.

TABLE I  
PERFORMANCE COMPARISON ON THE SSD FRAMEWORK ACROSS MINI-IMAGENET. CIFAR100 AND TINY-IMAGENET DATASETS. ALL RESULTS ARE REPORTED AS THE AVERAGE OF 10 INDEPENDENT RUNS.
<table><tr><td rowspan="2">Method</td><td colspan="3">Mini-ImageNet</td><td colspan="3">CIFAR100</td><td colspan="3">Tiny-ImageNet</td></tr><tr><td>Last</td><td>Avg</td><td>Fgt</td><td>Last</td><td>Avg</td><td>Fgt</td><td>Last</td><td>Avg</td><td>Fgt</td></tr><tr><td>ReLU</td><td>25.1±0.5</td><td>37.0±1.4</td><td>13.4±1.2</td><td>28.3±0.5</td><td>40.3±1.2</td><td>18.5±1.2</td><td>8.6±0.3</td><td>17.2±1.0</td><td>10.3±0.5</td></tr><tr><td>AID</td><td>21.3±0.6</td><td>31.1±1.4</td><td>8.4±0.8</td><td>25.5±0.6</td><td>35.2±1.1</td><td>8.8±0.9</td><td>7.3±0.3</td><td>14.1±1.0</td><td>6.6±0.4</td></tr><tr><td>Randomized Smooth-Leaky</td><td>24.1±0.3</td><td>36.6±0.8</td><td>11.7±0.7</td><td>28.9±0.6</td><td>41.7±1.0</td><td>15.7±0.5</td><td>9.1±0.2</td><td>18.3±0.9</td><td>9.5±0.7</td></tr><tr><td>Rational</td><td>13.3±5.3</td><td>18.8±6.4</td><td>8.6±5.2</td><td>17.1±3.0</td><td>21.6±3.4</td><td>7.0±2.5</td><td>5.7±1.6</td><td>10.8±2.5</td><td>6.1±1.5</td></tr><tr><td>B-Spline (KAN)</td><td>25.0±2.1</td><td>37.5±1.5</td><td>15.8±2.9</td><td>28.9±0.4</td><td>41.7±0.7</td><td>18.4±0.5</td><td>8.0±1.2</td><td>17.6±2.8</td><td>12.1±1.3</td></tr><tr><td>ChannelWavAct</td><td>26.2±0.4</td><td>39.0±0.6</td><td>12.7±0.5</td><td>30.2±0.4</td><td>43.5±0.9</td><td>16.4±0.4</td><td>9.9±0.2</td><td>19.9±0.9</td><td>10.9±0.4</td></tr></table>

TABLE II

PERFORMANCE COMPARISON ON EWC AND WA FRAMEWORKS. ALL RESULTS WERE OBTAINED WITH FIXED RANDOM SEEDS AND FIXED TASK ORDERS [44].
<table><tr><td rowspan="2">Method</td><td colspan="4">EWC</td><td colspan="4">WA</td></tr><tr><td colspan="2">CIFAR100</td><td colspan="2">Imagenet100</td><td colspan="2">CIFAR100</td><td colspan="2">Imagenet100</td></tr><tr><td></td><td colspan="2">Avg Fgt</td><td colspan="2">Avg Fgt</td><td colspan="2">Avg Fgt</td><td colspan="2">Avg Fgt</td></tr><tr><td></td><td></td><td>w/o  $\overline { { L _ { 2 } } }$ </td><td>Regularization</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ReLU</td><td>33.4</td><td>56.4</td><td>40.0</td><td>71.2</td><td>33.0</td><td>82.3</td><td>35.8</td><td>83.2</td></tr><tr><td>AID</td><td>32.0</td><td>57.1</td><td>35.8</td><td>71.8</td><td>38.5</td><td>65.0</td><td>42.5</td><td>72.5</td></tr><tr><td>Randomized Smooth-Leaky</td><td>34.2</td><td>52.2</td><td>40.8</td><td>66.4</td><td>37.4</td><td>77.9</td><td>38.0</td><td>80.6</td></tr><tr><td>Rational</td><td>5.0</td><td>15.2</td><td>27.6</td><td>79.1</td><td>13.8</td><td>32.1</td><td>42.4</td><td>71.6</td></tr><tr><td>B-Spline</td><td>34.3</td><td>36.7</td><td></td><td></td><td>45.6</td><td>61.7</td><td></td><td></td></tr><tr><td>ChannelWavAct*</td><td>36.5</td><td>52.3</td><td>41.0</td><td>68.6</td><td>46.4</td><td>65.7</td><td>54.6</td><td>62.9</td></tr><tr><td>ChannelWavAct</td><td>41.2</td><td>45.3</td><td>40.9</td><td>68.4</td><td>55.6</td><td>48.2</td><td>64.5</td><td>40.8</td></tr><tr><td colspan="9">+ L2 Regularization</td></tr><tr><td>ReLU</td><td>31.1</td><td>75.9</td><td>28.6</td><td>87.8</td><td>63.4</td><td>35.1</td><td>67.2</td><td>35.8</td></tr><tr><td>AID</td><td>31.2</td><td>64.4</td><td>30.0</td><td>86.7</td><td>52.9</td><td>43.5</td><td>65.3</td><td>38.2</td></tr><tr><td>Randomized Smooth-Leaky</td><td>30.5</td><td>73.4</td><td>31.5</td><td>81.6</td><td>62.8</td><td>38.4</td><td>67.7</td><td>31.0</td></tr><tr><td>Rational</td><td>4.4</td><td>14.8</td><td>23.1</td><td>85.0</td><td>13.8</td><td>32.1</td><td>42.4</td><td>71.6</td></tr><tr><td>B-Spline</td><td>30.6</td><td>76.2</td><td></td><td></td><td>61.5</td><td>29.6</td><td></td><td></td></tr><tr><td>ChannelWavAct w/o decoupled lr</td><td>29.6</td><td>73.0</td><td>28.6</td><td>88.1</td><td>62.5</td><td>30.1</td><td>68.2</td><td>33.9</td></tr><tr><td>ChannelWavAct</td><td>32.5</td><td>70.3</td><td>28.0</td><td>86.3</td><td>65.7</td><td>29.9</td><td>63.9</td><td>34.9</td></tr></table>

The generalization performance of different activation functions within the replay-free framework is summarized in Table II. To more clearly isolate and compare the direct impact of activation designs on model performance, we first evaluate these functions in the absence of $L _ { 2 }$ regularization. Furthermore, to assess the compatibility of different activation functions with standard plasticity-preserving techniques, we also provide experimental results under the inclusion of $L _ { 2 }$ regularization. We exclude KAN on ImageNet100 due to severe parameter explosion. In the EWC setting, ChannelWavAct achieves 41.2% average accuracy on CIFAR-100, outperforming the ReLU baseline by 7.8%. This advantage is even more pronounced in the WA framework, where ChannelWavAct exceeds ReLU by 22.6% (CIFAR-100) and 28.7% (ImageNet-100). These results demonstrate that our learnable wavelet structure provides the critical plasticity needed to acquire new knowledge without relying on memory buffers. The results under $L _ { 2 }$ regularization further validate the robustness and compatibility of our method. While $L _ { 2 }$ regularization typically restricts the update of learnable parameters, ChannelWavAct consistently maintains its leading position or stays within the top tier. For instance, in the WA setting on CIFAR-100, our method reaches 65.7% average accuracy, proving its seamless integration with standard stability-preserving techniques. This versatility ensures that ChannelWavAct can be effectively deployed across diverse experimental configurations without compromising its intrinsic performance gains. The consistent performance of ChannelWavAct across both CIFAR and ImageNet-100 highlights its universality. Unlike other learnable activations that may suffer from parameter explosion or optimization instability on large-scale datasets, our method offers a scalable and stable solution. The synergy between the decoupled learning rate and the wavelet basis enables the model to achieve superior average accuracy while maintaining a competitive balance between plasticity and stability.

## B. Spectral Analysis

To demonstrate the effectiveness of ChannelWavAct in mitigating spectral bias, we conduct a series of spectral analysis experiments following the framework established by [28]. These experiments compare the performance of standard ReLU networks with our proposed wavelet-based activation.

![](images/16d046e357abc9abb2d272be4fd4b275ea575d0f15a1cfbffaa42dd5de7ea215.jpg)

![](images/bb441bf022b96f3018d1eb6515e89852c99184395a795a98ca33ec91926e4221.jpg)  
(a) spectral learning dynamic curves.

![](images/3f753155969ade3fd810f416ce175d28eaee059dba4bcc7c3c15a48caa78d660.jpg)

![](images/a858bb5f82a792080cf907a8f810ef0ca60ca02c0d9e52ef6892b935f0a56094.jpg)  
(b) spectral learning heatmaps.

Fig. 4. (a) Evolution of the learned amplitude for frequency components (5 Hz to 30 Hz) during the training of a 1D synthetic regression task. Results are shown for the ReLU network (left) and ChannelWavAct (right), with a target amplitude of 1.0. (b) Heatmaps of the normalized amplitude across the frequency spectrum over training iterations for ReLU (left) and ChannelWavAct (right).  
![](images/122da41ed3e2499fb0d0f767c5d788e7bcb857bbe2736078876726319ecf544d.jpg)

![](images/35df81da0cb0429370c0b21f6ef7a8c802b8f7a565e2d6314d424c47363dd1ae.jpg)  
Fig. 5. Heatmaps of normalized spectral amplitude across frequencies (x-axis) and random parameter perturbation norms ϵ (y-axis) for the ReLU network (left) and ChannelWavAct (right).

1) Experiment 1: We first examine the learning dynamics of different frequency components using a 1D regression task. The target signal is formulated as a superposition of six sinusoids with frequencies evenly spaced from 5 Hz to 30 Hz, all maintaining an equal target amplitude of 1.0. We train a four-layer Multi-Layer Perceptron (MLP) with a hidden dimension of 256 for 10,000 iterations, extracting the Fourier amplitude of each frequency component at regular intervals.

As illustrated in fig. 4, standard ReLU networks demonstrate a pronounced spectral bias. The lower frequency components converge to the target amplitude rapidly, whereas higher frequencies learn at a significantly slower rate and fail to fully reach the target amplitude even at the end of training. In contrast, ChannelWavAct utilizes its learnable Mexican Hat wavelet components to explicitly target these high-frequency variations. By injecting high-frequency wavelet bases, the model compensates for the inherent low-pass nature of standard activations, achieving an almost instantaneous convergence to the target amplitude across the entire frequency spectrum. Furthermore, while the dynamic adaptation process introduces brief, sharp perturbations during training, the network demonstrates remarkable structural robustness by recovering immediately, maintaining a uniform and accelerated learning capacity for all features regardless of their frequency.

(a) fixed k.  
![](images/a44b143d02ecdebdc9472226a3e686de8e649a42822470d42c7fdfb600818672.jpg)  
(b) fixed $\beta .$  
Fig. 6. Validation loss on the clean target T0 $\tau _ { 0 }$ for ReLU (left) and ChannelWavAct (right), evaluated as a function of (a) noise amplitude β with fixed frequencies $\vec { k } \in \{ 0 . 1 , 1 . 0 \}$ , and (b) spatial noise frequency k<sup>˜</sup> with fixed amplitudes $\beta \in \{ 0 . 5 , \dot { 1 } . 0 \}$

2) Experiment 2: To assess the structural resilience of learned frequency components against parameter drift, we evaluate the model’s robustness to random parameter perturbations. Building upon the setup of Experiment 1, we apply isotropic random noise to the network parameters with varying perturbation norms $( \epsilon \in \ [ 0 , 4 . 0 ] )$ and average the resulting spectral magnitudes over 50 independent trials.

As visually confirmed by fig. 5, standard ReLU networks reveal a severe vulnerability under global parameter perturbations. While their low-frequency components exhibit relative stability, the normalized spectral amplitude of high-frequency components collapses rapidly even under minor perturbations. This demonstrates that high-frequency representations in fixedactivation networks occupy a highly sensitive and narrow volume in the parameter space. In contrast, ChannelWavAct maintains near-perfect spectral integrity across the entire frequency spectrum, even at the maximum perturbation norm.

![](images/dd24184ab76d56bc89deb22f02f089d4a3284ba6c83ecebf33ef292791a65c6f.jpg)  
Fig. 7. Heatmaps of the normalized spectral amplitude across learned frequencies (x-axis) and 20,000 training iterations (y-axis) for target signals defined on 2D flower-shaped manifolds $\gamma _ { L }$ with $\bar { L } \in \ \{ 0 , 4 , 1 0 , \bar { 1 6 } \}$ . The visualization shows results for standard ReLU networks (top) and Channel-WavAct (bottom).

By explicitly decoupling high-frequency details into localized, multi-scale wavelet bases, ChannelWavAct structurally anchors complex features. This structural independence ensures that subsequent parameter fluctuations during the acquisition of new tasks do not catastrophically erase previously consolidated high-frequency knowledge.

3) Experiment 3: We extend our analysis to highdimensional data by training a Multi-Layer Perceptron (MLP) on a binary classification subset of MNIST (digits 3 and 8). To test the model’s ability to distinguish between relevant semantic features and frequency-dependent noise, we construct a noisy target $\tau _ { k } = \tau _ { 0 } + \beta \sin ( \dot { k } \| x \| )$ , where $\tau _ { 0 }$ is the clean label, $\beta$ is the noise amplitude, and <sup>˜</sup>k is the spatial frequency of the radial noise. We evaluate the validation loss on the clean target $\tau _ { 0 }$ across varying amplitudes $\beta \in \{ 0 . 1 , 0 . 5 , 1 . 0 , 2 . 0 \}$ and frequencies $\tilde { k } \in \{ 0 . 1 , 0 . 5 , 1 . 0 , 2 . 0 \}$

The experimental results in fig. 6 validate that standard ReLU networks exhibit a strong spectral bias, leaving them highly susceptible to low-frequency noise. As observed in the standard ReLU validation curves, low-frequency noise $\tilde { k } = 0 . 1$ is rapidly and preferentially fitted by the network, resulting in a significantly elevated and persistent validation loss that worsens as the noise amplitude $\beta$ increases. Furthermore, when exposed to high-frequency noise $\tilde { k } \geq 1 . 0$ , ReLU networks demonstrate a distinct dip in validation loss early in training before overfitting the noise, confirming their delayed capability to represent high-frequency variations. Conversely, ChannelWavAct leverages its multi-resolution wavelet bases to actively mitigate this bias. The validation loss curves for ChannelWavAct display profound resilience: the network achieves a substantially lower steady-state validation loss across all noise amplitudes and frequencies.

4) Experiment 4: Finally, we investigate the interaction between data manifold geometry and the network’s frequency learning capabilities. We construct a 2D flower-shaped manifold $\gamma _ { L } ( z )$ parameterized by a latent variable $z \in [ 0 , 1 ]$ , where the parameter $L \in \{ 0 , 4 , 1 0 , 1 6 \}$ controls the number of petals and the overall topological curvature. The network, a six-layer Multi-Layer Perceptron, is tasked with regressing a highly complex target signal comprising ten superimposed sinusoids with frequencies ranging from 20 Hz to 200 Hz defined on the latent space. The training iterations is set to be 20000.

TABLE III  
ABLATION STUDY ON THE EFFECTIVENESS OF EACH COMPONENT IN CHANNELWAVACT ON THE CIFAR100 DATASET.
<table><tr><td>Module</td><td>ReLU</td><td>Wav.</td><td>Inj.</td><td>Reg.</td><td>Last</td><td>Avg</td><td>Fgt</td></tr><tr><td>#1</td><td>√</td><td></td><td></td><td></td><td>28.3</td><td>40.3</td><td>18.5</td></tr><tr><td>#2</td><td></td><td>√</td><td></td><td></td><td>29.0</td><td>42.5</td><td>17.1</td></tr><tr><td>#3</td><td></td><td>√</td><td>√</td><td></td><td>29.6</td><td>43.3</td><td>17.7</td></tr><tr><td>#4</td><td></td><td>√</td><td></td><td>√</td><td>28.8</td><td>42.2</td><td>16.5</td></tr><tr><td>#5</td><td></td><td>√</td><td>√</td><td>√</td><td>30.2</td><td>43.5</td><td>16.4</td></tr></table>

TABLE IV

ABLATION STUDY ON THE IMPACT OF DECOUPLED LEARNING RATES ON THE CIFAR100 DATASET.
<table><tr><td>Module</td><td>ReLU</td><td>ChaWav.</td><td>Decoupled.</td><td>Avg</td><td>Fgt</td></tr><tr><td>#1</td><td>√</td><td></td><td></td><td>33.4</td><td>56.4</td></tr><tr><td>#2</td><td>√</td><td></td><td>√</td><td>38.9</td><td>43.7</td></tr><tr><td>#3</td><td></td><td>√</td><td></td><td>36.5</td><td>52.3</td></tr><tr><td>#4</td><td></td><td>√</td><td>√</td><td>41.2</td><td>45.3</td></tr></table>

The results in fig. 7 shows that standard ReLU networks are strictly constrained by the manifold’s geometric properties. On a simple circular manifold $L \ = \ 0 .$ , ReLU completely fails to capture high-frequency components. As the manifold complexity increases to $L = 1 6$ , the network’s ability to fit higher frequencies gradually improves, aligning with the theoretical premise that complex topologies allow standard networks to represent high latent frequencies using lower ambient frequencies. However, ReLU still exhibits significantly delayed convergence and struggles to fully resolve the 180–200 Hz band. Conversely, ChannelWavAct effectively decouples the model’s representational capacity from the underlying data geometry. By dynamically injecting multi-scale wavelet bases, it possesses an intrinsic capability to resolve highfrequency details regardless of the manifold’s spatial curvature. The experimental heatmaps demonstrate that ChannelWavAct achieves near-instantaneous and uniform convergence across the entire 20–200 Hz spectrum for all values of L. This structural independence from manifold complexity ensures that the model can rapidly assimilate intricate target functions across arbitrary topological structures.

## C. Ablation

1) Component analysis: To validate the efficacy of our proposed method, we conducted comprehensive ablation studies on the CIFAR-100 dataset. The results in table III demonstrate that replacing fixed ReLU with our learnable ChannelWavAct yields immediate accuracy gains (42.5% vs. 40.3%), confirming the superiority of learnable wavelet bases. While dynamic wavelet injection further enhances plasticity (43.3%), optimal performance is achieved only when synergized with stability regularization, which effectively minimizes forgetting to 16.4% and resolves the stability-plasticity dilemma. Additionally, we validate the criticality of our decoupled optimization strategy in table IV. By categorizing parameters into plasticity and stability groups, it improves average accuracy by 4.7% and reduces forgetting by 7.0%. These results confirm that differential optimization facilitates rapid restructuring to prevent under-fitting while safeguarding against uncontrolled representation drift during adaptation.

![](images/5656cad509830e29a5ed8f56e79a5f04204f12da50a0377440cef3dc45af75b1.jpg)

![](images/893d38ac66ab5148e4f834a428a04759369820e349af075efe075a5e6a3e583a.jpg)

![](images/584801db37e45cef77317cc3fa30bd41f5bf95874524a7e30cb78f4bbdf35358.jpg)

![](images/dddfb1e614f82d5880098e43a2facc6242949639ab5d8b5941d9bc70841c158a.jpg)

(a) ∆.  
![](images/0dcaaaf6a6c5bd313a80cb038575f1854363a8b38abc4937dc1988926849dad2.jpg)

![](images/5e5a3aa31c3747bcda4c0524d1f8f2a46ae4780d9dd28e99015249f8498385a2.jpg)

![](images/13e7caf2c250ad461c81ab1a4d459053b5c742e21f648cb7da5abd899555b6b0.jpg)

(b) P.  
![](images/7a68548594a813f63a42663b6a6931dd9198d24f151fb847e4ed6581bd7f1da5.jpg)

![](images/6af5ac779155482630648107397d049d51a67bf3da040b336894b004c872cec4.jpg)

![](images/ec33b703bb454b9f4606c8232ca138cc6b304c0b976db1ff20e24d062bd831fc.jpg)

(c) λ.  
![](images/95436aadcb1fccc2170292cb9f6c176b4d18ae7ba9f41d69fe474712ab88af5c.jpg)  
(d) δ.

![](images/d45b62b5cfd89a9e6721ac50bf1dad85f9b3ea0a736fc23bd73d69be0c4efca6.jpg)  
Fig. 8. Impact of key hyperparameters on continual learning performance, evaluated across: (a) the number of injected wavelet bases $\Delta \bar { \ }$ per expansion phase, (b) the patience threshold P for injection triggering, (c) the regularization coefficient λ applied to pre-existing wavelet weights and (d) the relative margin δ used for loss stagnation detection.

2) Parameter Sensitivity Analysis: To thoroughly evaluate the robustness of ChannelWavAct, we conduct a parameter sensitivity analysis on its core hyperparameters: the injection number $\Delta ,$ patience threshold $P ,$ regularization coefficient λ, and relative margin δ in fig. 8. The impact of these parameters is measured across three key continual learning metrics: Last Accuracy, Average Accuracy, and Forgetting.

Injection Number $\Delta \colon$ The hyperparameter $\Delta$ controls the number of new wavelet bases injected during each capacity expansion phase. As illustrated in fig. 8a, the model achieves its peak performance at a minimal injection size of $\Delta = 1$ Increasing the injection number leads to a noticeable drop in both Average and Last Accuracy, accompanied by an increase in forgetting. This suggests that injecting too many new parameters simultaneously disrupts the optimization landscape and may lead to redundant capacity, whereas a conservative, incremental expansion provides sufficient representational power while maintaining network stability.

Patience Threshold $P \colon$ The patience threshold determines how many iterations the model waits during loss stagnation before triggering a dynamic injection. The analysis in fig. 8b reveals an optimal balance at $P = 5 0$ . Setting the threshold too low degrades performance, likely because it triggers premature injections before the model has fully converged using its existing capacity. Conversely, a threshold that is too high delays necessary expansions, forcing the model to struggle with underfitting on complex new data distributions. $P = 5 0$ effectively ensures that capacity is expanded only when learning has genuinely plateaued.

Regularization Coefficient λ: The coefficient λ dictates the penalty strength applied to pre-existing wavelet weights to prevent catastrophic forgetting. The results in $\mathrm { f i g . }$ . 8c show a distinct performance peak at $\lambda = 1$ . When the regularization is too weak, the network suffers from severe forgetting, as it rapidly overwrites older task representations to accommodate new ones. On the other hand, excessive regularization overly constrains the parameter space, causing a steep drop in accuracy due to a loss of plasticity. Setting $\lambda = 1$ optimally navigates the stability-plasticity dilemma.

Relative Margin δ: The margin δ defines the relative loss improvement required to reset the patience counter. The model demonstrates optimal performance at a tight margin of $\delta \ : = \ : 0 . 0 0 5$ . A larger margin makes the stagnation detector overly insensitive, preventing the network from injecting capacity when it is actually needed. Conversely, an extremely small margin slightly increases forgetting, as the mechanism may overreact to standard mini-batch noise rather than true convergence. $\delta = 0 . 0 0 5$ provides the most reliable signal for actual learning plateaus.

## VI. CONCLUSION

In this paper, we address the challenge of plasticity loss in continual learning by proposing ChannelWavAct, which decomposes the activation function into a global base approximation and local wavelet details. Through dynamic wavelet injection and stability regularization, our approach effectively harmonizes the stability-plasticity trade-off. Crucially, we provide a rigorous theoretical foundation that validates the necessity of the hybrid activation structure and the decoupled optimization strategy in overcoming spectral bias. We further establish the mathematical necessity of the loss-based trigger mechanism and demonstrate the superior $L ^ { 2 }$ approximation capability of ChannelWavAct for complex target signals. Extensive experiments confirm that our method achieves stateof-the-art performance with superior trainability and generalization across diverse benchmarks.

While the current implementation incurs a relatively higher computational overhead, future work will focus on enhancing the computational efficiency of learnable wavelets and establishing more comprehensive theoretical frameworks regarding optimization and generalization dynamics.

## REFERENCES

[1] S.-A. Rebuffi, A. Kolesnikov, G. Sperl, and C. H. Lampert, “icarl: Incremental classifier and representation learning,” in Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, 2017, pp. 2001–2010.

[2] A. Lu, H. Yuan, T. Feng, and Y. Sun, “Rethinking the stability-plasticity trade-off in continual learning from an architectural perspective,” arXiv preprint arXiv:2506.03951, 2025.

[3] R. Tong, Y. Liu, J. Q. Shi, and D. Gong, “Coreset selection via reducible loss in continual learning,” in The Thirteenth International Conference on Learning Representations, 2025.

[4] J. Gu, K. Wang, W. Jiang, and Y. You, “Summarizing stream data for memory-constrained online continual learning,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 11, 2024, pp. 12217–12225.

[5] S. Hassan, N. Rasheed, and M. A. Qureshi, “A new regularizationbased continual learning framework,” in 2024 Horizons of Information Technology and Engineering (HITE). IEEE, 2024, pp. 1–5.

[6] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska et al., “Overcoming catastrophic forgetting in neural networks,” Proceedings of the national academy of sciences, vol. 114, no. 13, pp. 3521–3526, 2017.

[7] D.-W. Zhou, Q.-W. Wang, H.-J. Ye, and D.-C. Zhan, “A model or 603 exemplars: Towards memory-efficient class-incremental learning,” arXiv preprint arXiv:2205.13218, 2022.

[8] H. Jin, G.-h. Kim, C. Ahn, and E. Kim, “Growing a brain with sparsity-inducing generation for continual learning,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 18.961-18970

[9] G. Xu, R. Zheng, Y. Liang, X. Wang, Z. Yuan, T. Ji, Y. Luo, X. Liu, J. Yuan, P. Hua et al., “Drm: Mastering visual reinforcement learning through dormant ratio minimization,” arXiv preprint arXiv:2310.19668, 2023.

[10] A. Lewandowski, H. Tanaka, D. Schuurmans, and M. C. Machado, “Directions of curvature as an explanation for loss of plasticity,” arXiv preprint arXiv:2312.00246, 2023.

[11] H. Lee, H. Cho, H. Kim, D. Kim, D. Min, J. Choo, and C. Lyle, “Slow and steady wins the race: Maintaining plasticity with hare and tortoise networks,” arXiv preprint arXiv:2406.02596, 2024.

[12] E. Nikishin, J. Oh, G. Ostrovski, C. Lyle, R. Pascanu, W. Dabney, and A. Barreto, “Deep reinforcement learning with plasticity injection,” Advances in Neural Information Processing Systems, vol. 36, pp. 37 142– 37 159, 2023.

[13] G. Sokar, R. Agarwal, P. S. Castro, and U. Evci, “The dormant neuron phenomenon in deep reinforcement learning,” in International Conference on Machine Learning. PMLR, 2023, pp. 32 145–32 168.

[14] C. Lyle, Z. Zheng, E. Nikishin, B. A. Pires, R. Pascanu, and W. Dabney, “Understanding plasticity in neural networks,” in International Conference on Machine Learning. PMLR, 2023, pp. 23 190–23 211.

[15] A. Kumar, R. Agarwal, D. Ghosh, and S. Levine, “Implicit underparameterization inhibits data-efficient deep reinforcement learning,” arXiv preprint arXiv:2010.14498, 2020.

[16] B. Shin, J. Oh, H. Cho, and C. Yun, “Dash: Warm-starting neural network training in stationary settings without loss of plasticity,” Advances in Neural Information Processing Systems, vol. 37, pp. 43 300–43 340, 2024.

[17] S. Dohare, J. F. Hernandez-Garcia, Q. Lan, P. Rahman, A. R. Mahmood, and R. S. Sutton, “Loss of plasticity in deep continual learning,” Nature, vol. 632, no. 8026, pp. 768–774, 2024.

[18] M. Elsayed and A. R. Mahmood, “Addressing loss of plasticity and catastrophic forgetting in continual learning,” arXiv preprint arXiv:2404.00781, 2024.

[19] A. Lewandowski, M. Bortkiewicz, S. Kumar, A. Gyorgy, D. Schuur-¨ mans, M. Ostaszewski, and M. C. Machado, “Learning continually by spectral regularization,” arXiv preprint arXiv:2406.06811, 2024.

[20] M. Elsayed, Q. Lan, C. Lyle, and A. R. Mahmood, “Weight clipping for deep continual and reinforcement learning,” arXiv preprint arXiv:2407.01704, 2024.

[21] Q. He, T. Zhou, M. Fang, and S. Maghsudi, “Adaptive regularization of representation rank as an implicit constraint of bellman equation,” arXiv preprint arXiv:2404.12754, 2024.

[22] C. Lyle, M. Rowland, and W. Dabney, “Understanding and preventing capacity loss in reinforcement learning,” arXiv preprint arXiv:2204.09560, 2022.

[23] M. Nauman, M. Ostaszewski, K. Jankowski, P. Miłos, and M. Cygan,´ “Bigger, regularized, optimistic: scaling for compute and sample efficient continuous control,” Advances in neural information processing systems, vol. 37, pp. 113 038–113 071, 2024.

[24] J. Obando-Ceron, A. Courville, and P. S. Castro, “In value-based deep reinforcement learning, a pruned network is a good network,” arXiv preprint arXiv:2402.12479, 2024.

[25] S. Park, I. Han, S. Oh, and K.-J. Kim, “Activation by interval-wise dropout: A simple way to prevent neural networks from plasticity loss,” arXiv preprint arXiv:2502.01342, 2025.

[26] L. Lillo and N. Cheney, “Activation function design sustains plasticity in continual learning,” arXiv preprint arXiv:2509.22562, 2025.

[27] A. L. Maas, A. Y. Hannun, A. Y. Ng et al., “Rectifier nonlinearities improve neural network acoustic models,” in Proc. icml, vol. 30, no. 1. Atlanta, GA, 2013, p. 3.

[28] N. Rahaman, A. Baratin, D. Arpit, F. Draxler, M. Lin, F. Hamprecht, Y. Bengio, and A. Courville, “On the spectral bias of neural networks,” in International conference on machine learning. PMLR, 2019, pp. 5301–5310.

[29] A. Morsali, M. Vaez, M. Soltani, A. Kazerouni, B. Taati, and M. Mohammad-Noori, “Staf: Sinusoidal trainable activation functions for implicit neural representation,” arXiv preprint arXiv:2502.00869, 2025.

[30] Q. Delfosse, P. Schramowski, M. Mundt, A. Molina, and K. Kersting, “Adaptive rational activations to boost deep reinforcement learning,” arXiv preprint arXiv:2102.09407, 2021.

[31] Z. Liu, Y. Wang, S. Vaidya, F. Ruehle, J. Halverson, M. Soljaciˇ c,´ T. Y. Hou, and M. Tegmark, “Kan: Kolmogorov-arnold networks,” arXiv preprint arXiv:2404.19756, 2024.

[32] S. Dohare, R. S. Sutton, and A. R. Mahmood, “Continual backprop: Stochastic gradient descent with persistent randomness,” arXiv preprint arXiv:2108.06325, 2021.

[33] J.-N. Lin and R. Unbehauen, “On the realization of a kolmogorov network,” Neural Computation, vol. 5, no. 1, pp. 18–20, 1993.

[34] Z. Bozorgasl and H. Chen, “Wav-kan: Wavelet kolmogorov-arnold networks,” arXiv preprint arXiv:2405.12832, 2024.

[35] A. Jacot, F. Gabriel, and C. Hongler, “Neural tangent kernel: Convergence and generalization in neural networks,” Advances in neural information processing systems, vol. 31, 2018.

[36] Z.-Q. J. Xu, Y. Zhang, T. Luo, Y. Xiao, and Z. Ma, “Frequency principle: Fourier analysis sheds light on deep neural networks,” arXiv preprint arXiv:1901.06523, 2019.

[37] D. A. Robin, K. Scaman et al., “Convergence beyond the overparameterized regime using rayleigh quotients,” Advances in Neural Information Processing Systems, vol. 35, pp. 10 725–10 736, 2022.

[38] H. Pfister, “Discrete-time signal processing,” Lecture Note, pfister. ee. duke. edu/courses/ece485/dtsp. pdf, 2017.

[39] I. J. Brown, “A wavelet tour of signal processing: the sparse way.” Investigacion Operacional, vol. 30, no. 1, pp. 85–87, 2009.

[40] E. M. Stein and R. Shakarchi, Fourier analysis: an introduction. Princeton University Press, 2011, vol. 1.

[41] M. Stephane, “A wavelet tour of signal processing,” 1999.

[42] S. Kumar, H. Marklund, and B. Van Roy, “Maintaining plasticity in continual learning via regenerative regularization,” arXiv preprint arXiv:2308.11958, 2023.

[43] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to document recognition,” Proceedings of the IEEE, vol. 86, no. 11, pp. 2278–2324, 2002.

[44] D.-W. Zhou, F.-Y. Wang, H.-J. Ye, and D.-C. Zhan, “Pycil: a python toolbox for class-incremental learning,” SCIENCE CHINA Information Sciences, vol. 66, no. 9, p. 197101, 2023.

[45] Z. Mai, R. Li, H. Kim, and S. Sanner, “Supervised contrastive replay: Revisiting the nearest class mean classifier in online class-incremental continual learning,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 3589–3599.

[46] B. Zhao, X. Xiao, G. Gan, B. Zhang, and S.-T. Xia, “Maintaining discrimination and fairness in class incremental learning,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 13 208–13 217.