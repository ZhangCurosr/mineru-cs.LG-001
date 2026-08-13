# SOFTWATER: CLASS-AWARE RATE ALLOCATION FOR SOFTMAX QUANTIZATION

Joao V. Cavalcanti & Ashia C. Wilson   
Department of Electrical Engineering and Computer Science   
Massachusetts Institute of Technology   
Cambridge, MA 02139, USA   
{caval,ashia}@mit.edu

## ABSTRACT

Post-training quantization pipelines routinely leave the softmax output layer in high precision. Yet in small LLMs with modern vocabularies, the head holds 15– 30% of all parameters, so a nominal “2-bit” model with an fp16 head can store several times as many bits per weight. We pose softmax-layer quantization as a rate-distortion problem under the KL divergence between the original and quantized output distributions. A second-order analysis reveals a class-aware geometry: quantization error is weighted jointly by feature covariance and class-specific softmax curvature. A separability approximation replaces the Kn×Kn Cholesky with one n × n factorization rescaled per class, making the lattice encodable by successive interference cancellation, with both statistics from a single forward pass. The resulting method, SoftWater, gives fine grids to frequent, low-variance classes and coarse grids to rare ones, a large gap under Zipfian token distributions. Across five models from 1B to 32B, SoftWater outperforms the released WaterSIC quantizer (near-optimal under linear-layer WMSE but not output KL) at matched head rates on 59 of 60 test points, using none of that pipeline’s refinements and cutting head-induced KL by 6.5×–8.3× at 2 bits. On Llama-3.2-1B-Instruct with quantized bodies, a 2-bit head removes 45–60% of stored bytes for a 2.9–3.7% perplexity increase. Because the class-side statistic comes from calibration data, matching calibration to the deployment domain gives the lowest KL on that domain throughout. On a tied model, a 4-bit head is near-lossless and a 2-bit head costs under 4% perplexity, making head quantization of such models practical.

## 1 INTRODUCTION

Post-training quantization (PTQ) replaces trained weights with low-precision ones while keeping the output close to the original. For the linear layers that make up most of an LLM, the standard objective is a weighted mean-squared error (WMSE) on the layer output, $\mathbb { E } \| ( W - \hat { W } ) X \| ^ { 2 }$ , targeted by sequential rounding Frantar et al. (2023a;b); Zhang et al. (2026); Chen et al. (2026), lattice codebooks Chee et al. (2024); Tseng et al. (2024; 2025); Savkin et al. (2025), or learned clipping and rotations Shao et al. (2024); Liu et al. (2025b); Ashkboos et al. (2024); Shao et al. (2025b). Most recently, WaterSIC Lifar et al. (2026) brought the WMSE problem within 0.255 bits of its information-theoretic limit.

One layer does not fit this picture. When a linear map feeds a softmax, the model outputs a probability distribution, so the distortion that matters is the KL divergence between the original and quantized outputs. PTQ pipelines usually skip this layer, leaving the head in high precision and excluding it from the reported bits per weight. That is costly: in small models with modern vocabularies the head holds 15–30% of all parameters (Table 1), so a “2-bit” Llama-3.2-1B with an fp16 head stores nearly 5 bits per weight. Prior work on the output layer replaces it Grave et al. (2017); Jean et al. (2015); Liu et al. (2025a); Shao et al. (2025a); Tranheden et al. (2026); Lee et al. (2026) rather than quantizing it under the KL metric.

We treat the head as a rate-distortion problem under KL. Where WMSE uses the error weight $I { \otimes } \Sigma _ { X }$ a second-order expansion of the KL divergence gives $\mathbb { E } [ ( \mathrm { d i a g } ( p ) - p p ^ { \top } ) \otimes X X ^ { \top } ]$ : the softmax

Table 1: Share of total parameters held by the output head (vocab × hidden) in modern small LLMs.
<table><tr><td>Model</td><td>Vocab</td><td>Hidden</td><td>Head (M)</td><td>% of total</td></tr><tr><td>Gemma-3-1B</td><td>262,144</td><td>1152</td><td>302</td><td>30.2</td></tr><tr><td>Qwen3-0.6B</td><td>151,936</td><td>1024</td><td>156</td><td>26.1</td></tr><tr><td>Gemma-2-2B</td><td>256,128</td><td>2304</td><td>590</td><td>22.6</td></tr><tr><td>Llama-3.2-1B</td><td>128,256</td><td>2048</td><td>263</td><td>21.3</td></tr><tr><td>Qwen3-1.7B</td><td>151,936</td><td>2048</td><td>311</td><td>18.1</td></tr><tr><td>Gemma-3-4B</td><td>262,144</td><td>2560</td><td>671</td><td>15.6</td></tr></table>

Hessian ties every class to the input statistics. A separable surrogate, the diagonal statistic $\bar { \lambda } _ { k } =$ $\mathbb { E } [ p _ { k } ( 1 - p _ { k } ) ]$ paired with the usual feature covariance, gives class k a grid scaled by $\bar { \lambda } _ { k } ^ { - 1 / 2 }$ and column i a grid scaled by $| \ell _ { i i } | ^ { - 1 }$ . The resulting method, SoftWater, gives fine grids to frequent, lowvariance classes and coarse grids to rare ones; token distributions are Zipfian, so this gap is large. Exploiting it costs one extra diagonal accumulator in the same calibration pass, and the decode format does not change.

The raw allocation is coarsest where the expansion is least trustworthy: classes with vanishing calibration probability. We therefore smooth the calibration distribution with a prior, capping the grid spacing on those rows. The smoothing weight interpolates between two regimes: at one end the calibration distribution sets the allocation, at the other every class gets equal weight and we recover WaterSIC. SoftWater thus contains the WMSE treatment of the head as the case where the deployment distribution is assumed uniform.

## Our contributions are:

• Problem formulation. We pose softmax-layer quantization as a rate-distortion problem under output KL and derive the induced error metric. A dither argument and a separability assumption, which we validate empirically, reduce it to a class-rescaled lattice problem whose two statistics are read off one calibration pass (§3–4).

• The SoftWater algorithm. The class-side scale is class frequency discounted by crosscontext variability, so SoftWater spends bits on classes that are frequent and low-variance. A smoothing prior caps the spacing on classes the calibration set leaves uncovered, keeping the allocation inside the range where the Taylor expansion is valid (§4.1–4.4).

• Head quantization results. Across five models from 1B to 32B parameters, SoftWater outperforms the released WaterSIC quantizer at matched head rates on 59 of 60 test points while using none of that pipeline’s refinements, and transfers unchanged to models with released GuidedQuant bodies Kim et al. (2025). The class-side factor carries domain information the covariance does not, letting a head be targeted at its deployment domain, and it cuts whole-model size by up to 60% (§5).

## 2 PRIOR WORK

We review the WMSE formulation of linear layer quantization, the WaterSIC algorithm whose SIC encoder we reuse, and recent work beyond MSE objectives.

## 2.1 LINEAR LAYER QUANTIZATION

A linear layer takes an activation $X \in \mathbb { R } ^ { n }$ and a weight matrix $W \in \mathbb { R } ^ { K \times n }$ and outputs $z = W X$ Quantization looks for $\hat { W } \in \mathbb { R } ^ { K \times n }$ minimizing the distortion

$$
D = \frac { 1 } { K n } \mathbb { E } \| ( \hat { W } - W ) X \| _ { F } ^ { 2 } = \frac { 1 } { K n } \sum _ { k = 1 } ^ { K } ( \hat { W } _ { k } - W _ { k } ) \Sigma _ { X } ( \hat { W } _ { k } - W _ { k } ) ^ { \top } ,\tag{1}
$$

where $W _ { k }$ and $\hat { W } _ { k }$ are rows and $\Sigma _ { X } = \mathbb { E } _ { X } [ X X ^ { \top } ]$ . Most calibration-based PTQ methods target (1): GPTQ and its successors by sequential rounding (Frantar et al., 2023a;b; Zhang et al., 2026; Chen

et al., 2026), QuIP-style methods by incoherence processing and structured codebooks (Chee et al., 2024; Tseng et al., 2024; 2025; Savkin et al., 2025).

## 2.2 THE WATERSIC ALGORITHM

Lifar et al. (2026) solve (1) within 0.255 bits of the information-theoretic limit (Lifar et al., 2026, Theorem 3.3) by allocating rates per column from $\Sigma _ { X }$

Let L be the lower triangular Cholesky factor of $\Sigma _ { X } = L L ^ { \top }$ . WaterSIC restricts $\hat { W } _ { k }$ to a scaled integer lattice $\mathbb { Z } ^ { 1 \times n } A$ , where $A = \operatorname { d i a g } ( \alpha _ { 1 } , \cdots , \alpha _ { n } )$ with $| A | ^ { 1 / n } = { \bar { \alpha } } :$ the lattice has the same point density $\bar { \alpha } ^ { - n }$ as $\bar { \alpha } \mathbb { Z } ^ { 1 \times n }$ , but each $\alpha _ { i }$ sets the grid resolution of column i, which entropy coding turns into a per-column bit rate. The objective reads

$$
D = \frac { 1 } { K n } \sum _ { k = 1 } ^ { K } \| ( \hat { W } _ { k } A - W _ { k } ) L \| ^ { 2 } ,\tag{2}
$$

with $\hat { W } _ { k }$ restricted to $\mathbb { Z } ^ { 1 \times n }$ , an NP-hard problem. WaterSIC solves it approximately with successive interference cancellation (SIC), which exploits the triangular structure of $L \colon$ columns are encoded in sequence by scalar equations that account for previously quantized entries. All rows are processed identically, since (2) decouples them.

It remains to set the scales A. Assume the rows $W _ { k }$ are iid $\mathcal { N } ( 0 , \sigma _ { W } ^ { 2 } I )$ and $\sigma _ { W } / \bar { \alpha } \to \infty$ . Then the quantization error approaches a uniform distribution (Lifar et al., 2026, Lemma 3.2, Appendix B), and with per-column entropy coding the rate approaches, as K grows,

$$
R _ { \mathrm { S I C } } \approx R _ { \mathrm { H i g h - R a t e } } ( D , \Sigma _ { X } ) + \frac { 1 } { 2 } \log \left( \frac { 2 \pi e } { 1 2 } \right) + \frac { 1 } { 2 } \log \left( \frac { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( \alpha _ { i } \ell _ { i i } ) ^ { 2 } } { \prod _ { i = 1 } ^ { n } ( \alpha _ { i } \ell _ { i i } ) ^ { 2 / n } } \right) ,\tag{3}
$$

where $R _ { \mathrm { H i g h - R a t e } } ( D , \Sigma _ { X } )$ is the optimal rate and $\ell _ { i i }$ is the i-th diagonal entry of L. The choice

$$
\alpha _ { i } = \frac { c _ { \mathrm { W S } } } { | \ell _ { i i } | } , \quad i = 1 , \cdot \cdot , n ,\tag{4}
$$

kills the third term and attains the $( 1 / 2 ) \log ( 2 \pi e / 1 2 ) \approx 0 . 2 5 5$ bit gap; here $c _ { \mathrm { W S } } > 0$ sets the lattice density, and $c _ { \mathrm { W S } } = \bar { \alpha } | L | ^ { 1 / n }$ recovers the density $\bar { \alpha } ^ { - n }$

The released pipeline adds several modifications (Lifar et al., 2026, see §4). We keep the two that any unequal-rate scheme needs: entropy coding, which turns the integer codes into a bitstream and is what makes unequal per-column rates realizable, and rate assignment, which tunes ${ \mathit { c } } _ { \mathrm { W } \mathrm { S } }$ by the secant method since the post-coding rate cannot be set directly. The rest are quality refinements orthogonal to rate allocation, and we do not use them, so the gains in §5 are attributable to the allocation alone and are in principle additive with them. One deserves mention: dead feature erasure targets near-zero-variance coordinates in early-layer inputs, and the released quantizer applies it to the head too; because the head reads a well-conditioned final hidden state, we use nominal damping instead $( \delta = 1 0 ^ { - 6 }$ mean(diagΣ )). Activation drift and residual stream correction are vacuous for a terminal layer that runs on a fixed body and writes into nothing, so we omit them.

## 2.3 QUANTIZATION METRICS BEYOND MSE

WMSE is the standard metric in model quantization (e.g., Nagel et al., 2020; Frantar et al., 2023a), mainly because it is tractable: it decouples rows, admits tools like Cholesky decompositions, connects directly to rate-distortion theory, and needs forward passes only. But it is a proxy for output behavior, and recent work has looked for better metrics; Lifar et al. (2026) list approximating a target perplexity or KL loss as their first open direction. Lee et al. (2026) swap MSE for the cross entropy between full-precision and quantized logits in the final block before the head. Kim et al. (2025) weight the layerwise objective by end-loss guidance, made tractable by assuming Fisher coefficients coupling different output channels vanish, so the objective again decouples across rows; this is the end-to-end analogue of the cross-row decorrelation we derive in §4.3, which for the softmax layer we can justify rather than assume, and their released pipeline produces the quantized bodies of §5. Tseng et al. (2026) build YAQA on tractable Hessian sketches of end-to-end KL, also decomposing the Hessian as a Kronecker product, but target adaptive rounding for body layers, whereas we derive the layer-local KL geometry of the softmax head and its rate allocation. A separate line replaces the head itself (Grave et al., 2017; Jean et al., 2015; Liu et al., 2025a; Shao et al., 2025a; Tranheden et al., 2026); we keep the architecture and quantize the existing head.

## 3 PROBLEM SETUP

We consider a softmax layer that takes an input $X \in \mathbb { R } ^ { n }$ with n features and produces a distribution $p = \mathrm { s o f t m a x } ( z ) \in \mathbb { R } ^ { K }$ over K classes, where $z = W X$ are the logits, $W \in { \overline { { \mathbb { R } } } } ^ { K \times n }$ are the weights with rows $W _ { k } \in \mathbb { R } ^ { 1 \times n }$ , and

$$
p _ { k } = \mathrm { s o f t m a x } ( z ) _ { k } = \frac { \exp ( z _ { k } ) } { \sum _ { l = 1 } ^ { K } \exp ( z _ { l } ) } .
$$

We treat X as a random vector drawn from the distribution of layer inputs induced by the data the model is deployed on, and estimate expectations with respect to it from a calibration sample. Both z and p are therefore functions of X; we suppress the dependence except where it matters.

For LLM heads, K is the vocabulary size and W can hold a large fraction of all parameters, and is often tied to the input embedding (Table 1). To mitigate this cost, we replace W by a matrix W<sup>ˆ</sup> whose entries are described by a budget of R bits per weight. Let $\Delta = \hat { W } - W$ and $\delta = \Delta X$ denote the weight and logit perturbations, with rows $\Delta _ { k } \in \mathbb { R } ^ { 1 \times n }$ and entries $\delta _ { k } = \Delta _ { k } X$ . The class distribution of the quantized model is then $q \in \mathbb { R } ^ { K }$ , defined by $q _ { k } = \mathrm { s o f t m a x } ( z + \delta ) _ { k }$ . Our goal is to find, for a given rate R, the $\hat { W }$ that minimizes the discrepancy between p and q, which §4.2 formalizes as an expected KL divergence.

## 4 SOFTWATER: PER-COLUMN AND PER-CLASS RATE ALLOCATION

The KL geometry of a softmax layer induces rate allocation along two axes: input features and output classes. SoftWater allocates along both axes. A separability approximation makes the resulting lattice encodable by SIC at $O ( n ^ { 3 } )$ cost. We derive its objective, reduce it to a tractable problem, add a smoothing step that keeps the proxy accurate, and contrast linear layer and softmax quantization.

## 4.1 LINEAR LAYER VS SOFTMAX QUANTIZATION

In §2.2 we saw that SIC’s error geometry over the Cholesky factor L, combined with entropy coding, points to grid spacing proportional to $| \check { \ell } _ { i i } | ^ { - 1 }$ . Our goal in this section is to show that, for softmax quantization, the grid spacing should instead be proportional to $( \bar { \lambda } _ { k } ^ { 1 / 2 } | \ell _ { i i } | ) ^ { - 1 }$ , where

$$
\bar { \lambda } _ { k } = \mathbb { E } _ { X } \big [ p _ { k } ( 1 - p _ { k } ) \big ] = \mathbb { E } _ { X } \big [ p _ { k } \big ] ( 1 - \mathbb { E } _ { X } \big [ p _ { k } \big ] ) - \mathrm { V a r } \big [ p _ { k } \big ] .
$$

That ${ \mathrm { i s } } ,$ the grid spacing is inversely proportional to the square root of class frequency discounted by cross-context variance, the spread of $p _ { k }$ across contexts. Therefore, through $( \bar { \lambda } _ { k } ^ { 1 / 2 } | \ell _ { i i } | ) ^ { - 1 }$ ，

SoftWater correctsfor in-feature geometry, classfrequency and class variance.

In particular, SoftWater allocates

• more bits to frequent classes than infrequent classes;

• more bits to low-variance classes than high-variance classes.

The size of the gap between the two grid spacing choices belongs to the class distribution, not the algorithm. For a near-uniform distribution the $\bar { \lambda } _ { k }$ are nearly equal and SoftWater recovers WaterSIC. For the heavy tails of natural-language vocabularies, $\mathbb { E } _ { X } [ p _ { k } ]$ spans many orders of magnitude and the spread in $\bar { \lambda } _ { k } ^ { - 1 / 2 }$ is large, bounded only by the smoothing floor of $\ S 4 . 4$ . This is why the class side, not the feature side, is where a softmax layer’s rate budget is won or lost. Solving the softmax problem through WMSE therefore amounts to assuming identically distributed classes.

## 4.2 OBJECTIVE DERIVATION

Linear layer quantization measures distortion by the output WMSE (Frantar et al., 2023a; Lifar et al., 2026). A softmax output has more structure: it lies on the probability simplex and is a distribution. We therefore use the KL divergence, which captures changes in that distribution directly, and look for $\hat { W }$ minimizing

$$
D _ { \mathrm { K L } } \left( p \Vert q \right) = \sum _ { k } p _ { k } ( \log p _ { k } - \log q _ { k } ) .
$$

Writing

$$
\log p _ { k } = z _ { k } - \log \sum _ { l = 1 } ^ { K } \exp ( z _ { l } ) , \quad \log q _ { k } = z _ { k } + \delta _ { k } - \log \sum _ { l = 1 } ^ { K } \exp ( z _ { l } + \delta _ { l } ) ,
$$

and substituting gives

$$
D _ { \mathrm { K L } } ( { p } \| { q } ) = \log \sum _ { l = 1 } ^ { K } \exp ( z _ { l } + \delta _ { l } ) - \log \sum _ { l = 1 } ^ { K } \exp ( z _ { l } ) - p ^ { \top } \delta .
$$

Let $f ( \delta ) = D _ { \mathrm { K L } } \left( p \rVert q \right)$ , with

$$
\nabla f ( \delta ) = \operatorname { s o f t m a x } ( z + \delta ) - p \quad { \mathrm { a n d } } \quad \nabla ^ { 2 } f ( \delta ) = \operatorname { d i a g } ( q ) - q q ^ { \top } .
$$

Since $f ( 0 ) = 0$ and $\nabla f ( 0 ) = 0$ , a second-order Taylor expansion at $\delta = 0$ gives

$$
\begin{array} { l } { { \displaystyle D _ { \mathrm { K L } } \left( p \| q \right) = f ( 0 ) + \nabla f ( 0 ) ^ { \top } \delta + \frac { 1 } { 2 } \delta ^ { \top } \nabla ^ { 2 } f ( 0 ) \delta + O ( \| \delta \| ^ { 3 } ) } } \\ { { \displaystyle ~ \approx \frac { 1 } { 2 } X ^ { \top } \Delta ^ { \top } \nabla ^ { 2 } f ( 0 ) \Delta X } } \\ { { \displaystyle ~ = \frac { 1 } { 2 } \mathrm { v e c } ( \Delta ) ^ { \top } ( \nabla ^ { 2 } f ( 0 ) \otimes X X ^ { \top } ) \mathrm { v e c } ( \Delta ) } , } \end{array}\tag{5}
$$

where $\mathrm { v e c } ( \Delta ) \in \mathbb { R } ^ { K n }$ stacks the rows of $\Delta$ and

$$
\nabla ^ { 2 } f ( 0 ) = \mathrm { d i a g } ( p ) - p \boldsymbol { p } ^ { \intercal } .\tag{6}
$$

The expansion (5) holds as long as the logit perturbation stays small; §4.4 enforces this and §5.6 verifies the reduction that follows. The objective is then

$$
\mathbb { E } _ { X } \left[ \frac { 1 } { 2 } \mathrm { v e c } ( \Delta ) ^ { \top } ( \nabla ^ { 2 } f ( 0 ) \otimes X X ^ { \top } ) \mathrm { v e c } ( \Delta ) \right] = \frac { 1 } { 2 } \mathrm { v e c } ( \Delta ) ^ { \top } \mathbb { E } _ { X } [ \nabla ^ { 2 } f ( 0 ) \otimes X X ^ { \top } ] \mathrm { v e c } ( \Delta ) .\tag{7}
$$

The error weight is thus $\mathbb { E } _ { X } [ \nabla ^ { 2 } f ( 0 ) \otimes X X ^ { \top } ] , \mathrm { a } K n \times K n$ matrix coupling classes and features. Encoding against this directly would need a Cholesky factorization of that $K \bar { n } \times K n$ matrix, costing $O ( K ^ { 3 } n ^ { 3 } )$ time and $O ( K ^ { 2 } n ^ { \overline { { 2 } } } )$ memory. We need a tractable approximation, which its structure allows.

## 4.3 A TRACTABLE PROXY

The obstacle is that the expectation couples $\nabla ^ { 2 } f ( 0 )$ and $X X ^ { \top }$ , giving sample-dependent offdiagonal terms and a full matrix. If we can decouple them, then chol $( \mathsf { \bar { A } } \otimes \mathsf { \bar { B } } ) = \mathsf { \bar { c h o l } } ( \boldsymbol { \dot { A } } ) \otimes \mathsf { c h o l } ( B )$ splits the factorization into two smaller ones.

Under SIC, rows are quantized independently against the same L, with sequential correction along coordinates within each row: errors are coupled within a row, not across rows. Per-entry subtractive dither makes this precise. Under the dither ensemble the errors are zero-mean and uncorrelated across rows, and the dither is drawn independently of the data, so the expectations over X and over $\Delta$ factor. The cross-terms $- p _ { k } p _ { l } , k \ne l ,$ then contribute nothing, and the class-side diagonal is exact:

$$
\begin{array} { r } { \mathbb { E } _ { X , \Delta } [ \mathrm { v e c } ( \Delta ) ^ { \top } ( \nabla ^ { 2 } f ( 0 ) \otimes X X ^ { \top } ) \mathrm { v e c } ( \Delta ) ] = \mathbb { E } _ { X , \Delta } [ \mathrm { v e c } ( \Delta ) ^ { \top } ( \lambda \otimes X X ^ { \top } ) \mathrm { v e c } ( \Delta ) ] , } \end{array}\tag{8}
$$

where λ is the softmax curvature

$$
\lambda = \mathrm { d i a g } ( p \odot ( 1 - p ) ) .\tag{9}
$$

Replacing the error weight in (7) by its class-side diagonal gives

$$
\frac { 1 } { 2 } \mathrm { v e c } ( \Delta ) ^ { \top } \mathbb { E } _ { X } [ \lambda \otimes X X ^ { \top } ] \mathrm { v e c } ( \Delta ) ,\tag{10}
$$

again a deterministic function of the perturbation. Dithering is thus the device that identifies (10) as the right target; empirically, undithered errors already behave as if uncorrelated across rows, so we drop it from the final scheme and leave it as an optional tweak.

The matrix $\mathbb { E } _ { X } [ \lambda \otimes X X ^ { \top } ]$ is block diagonal, so K Cholesky decompositions of size n replace one of size $K n .$ . Except for very small K that is still too expensive, so we assume that $\lambda$ and $\dot { X } X ^ { \top }$ are separable.

Assumption 1 (Separability). The softmax curvature λ and $X X ^ { \top }$ are uncorrelated:

$$
\mathbb { E } _ { X } [ \lambda \otimes X X ^ { \top } ] = \mathbb { E } _ { X } [ \lambda ] \otimes \mathbb { E } _ { X } [ X X ^ { \top } ] .\tag{11}
$$

Assumption 1 holds only approximately; we bound the error below and test it in §5.6. It lets the two Kronecker factors be estimated independently: the covariance $\Sigma _ { X }$ separately from the expected curvature

$$
\bar { \lambda } = \mathbb { E } _ { X } [ \lambda ] = \mathbb { E } _ { X } [ \mathrm { d i a g } ( p \odot ( 1 - p ) ) ] .\tag{12}
$$

Moreover, $\bar { \lambda }$ is just K scalars, accumulated from the output distributions the calibration forward pass already produces: SoftWater needs no backward pass and no second pass over the data, and costs one $\dot { K } .$ -vector alongside the $n \times n$ covariance, both from the same forward pass. The Cholesky factor L of $\Sigma _ { X }$ is computed once and rescaled K times to give the factor $\mathcal { L } = \bar { \lambda } ^ { \bar { 1 } / 2 } \otimes L$ of ${ \bar { \lambda } } \otimes \Sigma _ { X }$ The distortion metric is then

$$
D = \frac { 1 } { 2 } \mathrm { { v e c } } ( \Delta ) ^ { \top } \mathcal { L } \mathcal { L } ^ { \top } \mathrm { { v e c } } ( \Delta ) ,\tag{13}
$$

a WMSE over the joint index (k, i), which SIC minimizes directly on ${ \mathcal { L } } .$

The induced grid spacing for class k and column i is

$$
\alpha _ { ( k - 1 ) n + i } = \frac { c _ { \mathrm { { S W } } } } { \bar { \lambda } _ { k } ^ { 1 / 2 } | \ell _ { i i } | } , \quad k = 1 , \cdots , K , \quad i = 1 , \cdots , n ,\tag{14}
$$

where $c _ { \mathrm { S W } } > 0$ again sets the point density. Letting

$$
B = \mathrm { d i a g } ( \bar { \lambda } _ { 1 } ^ { - 1 / 2 } , \cdot \cdot \cdot , \bar { \lambda } _ { K } ^ { - 1 / 2 } ) ,\tag{15}
$$

$$
\mathcal { A } = \mathrm { d i a g } ( | \ell _ { 1 1 } | ^ { - 1 } , \cdot \cdot \cdot , | \ell _ { n n } | ^ { - 1 } ) ,\tag{16}
$$

and choosing

$$
c _ { \mathrm { S W } } = { \bar { \alpha } } | B | ^ { - 1 / K } | A | ^ { - 1 / n } ,\tag{17}
$$

gives point density $\bar { \alpha } ^ { - K n }$ . SoftWater therefore uses $c _ { \operatorname { S W } } B \mathbb { Z } ^ { K \times n } \mathcal { A } ,$ rescaled on the class side as well as the feature side. The hyperparameter c<sub>SW</sub> is the sole rate knob: it scales the lattice uniformly, raises the post-coding rate monotonically, and is set by secant search (§2.2).

Remark. We notice that A and B are not just column and row rescalers; they change the grid spacing itself, redistributing rate across classes through the coupling with c . In contrast, plain column and row rescalers applied after coding change reconstruction but not rate.

On the separability assumption. Although $\lambda _ { k } = \lambda _ { k } ( X )$ depends on $X ,$ separability does not require independence between class probabilities and the input. It requires only that the classdependent curvature be weakly correlated with the second-order geometry of X. The error in the k-th block is

$$
{ \mathbb E } _ { X } [ \lambda _ { k } ( X ) X X ^ { \top } ] - \bar { \lambda } _ { k } \Sigma _ { X } = { \mathbb E } _ { X } \left[ ( \lambda _ { k } ( X ) - \bar { \lambda } _ { k } ) ( X X ^ { \top } - \Sigma _ { X } ) \right] ,\tag{18}
$$

so the approximation is accurate when examples with greater curvature on class k do not have systematically different feature covariance. More directly, for the k-th row $\Delta _ { k }$ of the perturbation,

$$
\Delta _ { k } \left( \mathbb { E } _ { X } [ \lambda _ { k } ( X ) X X ^ { \top } ] - \bar { \lambda } _ { k } \Sigma _ { X } \right) \Delta _ { k } ^ { \top } = \operatorname { C o v } _ { X } \left( \lambda _ { k } ( X ) , ( \Delta _ { k } X ) ^ { 2 } \right) .\tag{19}
$$

The proxy therefore need not match each block in every direction. It is enough that class uncertainty be weakly correlated with feature energy along the error directions the algorithm produces. This is plausible when representations are normalized and their geometry is stable across contexts. $\operatorname { I f } \| X \| ^ { 2 }$ is constant, for instance, the exact and separable blocks have the same trace,

$$
\operatorname { T r } \big ( \mathbb { E } _ { X } [ \lambda _ { k } ( X ) X X ^ { \top } ] \big ) = \bar { \lambda } _ { k } \operatorname { T r } ( \Sigma _ { X } ) ,
$$

so separability preserves the total curvature per class and drops only class-dependent anisotropy.

Such factorizations are common for tractable Kronecker-factored curvature. Martens & Grosse (2015) reach the same approximation from natural gradient descent, where the Fisher matrix defines the KL trust region and its output-layer block is the Hessian of our objective; they need separability to invert the preconditioner via $( A \ { \overset { \cdot } { \otimes } } \ B ) ^ { - 1 } = A ^ { - 1 } \otimes B ^ { - 1 }$ . Tseng et al. (2026) reach Kronecker approximations in weight-only PTQ, but their treatment is agnostic to the output layer and does not exploit softmax structure, so they approximate the end-to-end error Hessian and obtain their factors by power iteration, whereas ours come from an explicit analytical expression. In §5.6 we measure how far our factors sit from the Frobenius-optimal Kronecker factorization of the same error weight, and find the proxy underestimates the true distortion by at most 10% across models, rates, and datasets.

## 4.4 A TRACTABLE AND ACCURATE PROXY: SMOOTHING

The proxy (13) is accurate only if the above assumptions hold, which we verify in $\ S 5 ,$ , and if the Taylor expansion of $D _ { \mathrm { K L } } \left( p \parallel q \right)$ stays accurate, which requires a small logit perturbation δ.

Grid spacing under our metric is proportional to $( \bar { \lambda } _ { k } ^ { 1 / 2 } | \ell _ { i i } | ) ^ { - 1 }$ , so rare classes get coarser grids. If a class is absent from the calibration data, then $p _ { k } \ll 1$ and $\bar { \lambda } _ { k } ^ { 1 / 2 } \ll 1$ , the grid is far too coarse, the weights are badly perturbed, and (5) breaks down. We therefore smooth $p$ to lower bound $p _ { k }$ computing

$$
\tilde { \lambda } = \mathbb { E } _ { X } [ \mathrm { d i a g } ( \tilde { p } \odot ( 1 - \tilde { p } ) ) ] ,\tag{20}
$$

where, given $\epsilon \in ( 0 , 1 )$

$$
\tilde { p } = ( 1 - \epsilon ) p + \epsilon / K\tag{21}
$$

mixes p with a uniform distribution over the K classes. Since $\tilde { p } _ { k } \ge \epsilon / K$ pointwise, $\begin{array} { r } { \tilde { \lambda } _ { k } \ge \frac { \epsilon } { K } \left( 1 - \frac { \epsilon } { K } \right) } \end{array}$ while $\tilde { \lambda } _ { k } \le 1 / 4$ always. The ratio between the coarsest and finest class-side spacings is therefore at most $\textstyle { \frac { 1 } { 2 } } \sqrt { K / \epsilon }$ up to an $O ( \epsilon / K )$ correction, so ϵ caps the class-side rate spread at $\log _ { 2 }$ of that quantity, whatever the calibration data. Uncovered classes thus keep logit perturbations inside the regime where (5) is valid.

So ϵ interpolates the curvature. $\mathbf { A t } \epsilon = 0 , \tilde { \lambda }$ comes from the calibration data alone (12). At $\epsilon = 1$ it weighs every class equally and recovers WaterSIC. We fix $\epsilon = 0 . 1$ as default, and $\tilde { \lambda }$ replaces λ<sup>¯</sup> in the grid spacings (14), the rescalers $B ,$ and the constant c<sub>SW</sub>.

## 5 EXPERIMENTS

We evaluate SoftWater on five models from 1B to 32B parameters across the Llama and Qwen families: the head alone against an fp16 body (§5.1), then fully quantized models with released GuidedQuant bodies (§5.2), evaluated downstream in §5.4. Three further experiments probe the method: §5.3 targets calibration at a deployment domain, §5.5 asks what the class-side statistic needs to be, and §5.6 tests separability. Our arm runs the plain allocation with no post-hoc rescaler optimization, LMMSE shrinkage, or finetuning; the baseline is the full released WaterSIC pipeline minus finetuning.

Algorithm 1 SoftWater weight-only quantization of softmax   
Require: $W \in \mathbb { R } ^ { K \times n } , \Sigma _ { X } \succeq 0 , \tilde { \lambda } = \mathbb { E } _ { X } [ \mathrm { d i a g } ( \tilde { p } \odot ( 1 - \tilde { p } ) ) ] , \bar { \alpha } > 0 .$   
Ensure: $\alpha \in \mathbb { R } _ { + } ^ { n } , \beta \in \mathbb { R } _ { + } ^ { K } , B _ { 1 } , \ldots , B _ { n } \in \{ 0 , 1 \} ^ { * } \mathrm { ~ s . t . } \hat { W } = \mathrm { d i a g } ( \beta ) Z _ { \mathrm { S I C } } \mathrm { d i a g } ( \alpha ) .$   
1: Compute lower-triangular $L \in \mathbb { R } ^ { n \times n }$ s.t. $\Sigma _ { X } = L L ^ { \top }$ ▷ Cholesky   
2: $\begin{array} { r } { \beta _ { k } \gets \frac { ( \prod _ { l } \tilde { \lambda } _ { l } ^ { 1 / 2 } ) ^ { 1 / K } } { \tilde { \lambda } _ { \iota } ^ { 1 / 2 } } , \quad \forall k \in [ K ] } \end{array}$ ▷ class-side scales   
3: $\begin{array} { r } { \alpha _ { i }  \bar { \alpha } \cdot \frac { | L | ^ { 1 / n } } { | \ell _ { i i } | } , \quad \forall i \in [ n ] } \end{array}$ ▷ feature-side water filling; $\ell _ { i i }$ is the ith diagonal element of L   
4: $R \gets W L$   
5: for $i = n : 1$ do   
6: $Z _ { \mathrm { S I C } , : , i } \gets$ round $\left( R _ { : , i } \oslash \left( \alpha _ { i } \ell _ { i i } \beta \right) \right)$ ▷ ⊘: elementwise division by the K-vector $\alpha _ { i } \ell _ { i i } \beta$   
7: $R \gets R - \alpha _ { i } \big ( \beta \odot Z _ { \mathrm { S I C } , : , i } \big ) \cdot L _ { i , : }$ ▷ ⊙: elementwise product; $L _ { i , : }$ is the ith row of L   
8: end for   
9: for $i = 1 : n$ do   
10: $\begin{array} { r } { B _ { i } \gets \mathrm { E C } \left( Z _ { \mathrm { S I C } , : , i } \right) } \end{array}$ ▷ entropy coding of the ith column   
11: end for

Table 2: The body is kept in fp16 and only the head matrix is quantized. KL is measured against the all-fp16 model on identical token streams. WS: WaterSIC w/o FT. Parentheses after each model name give its vocabulary size; on PPL cells, the increase over the fp16 row of the same model; on SW KL cells, the WS/SW KL ratio at that rate. Best value per head group in bold.  
Method Head WT2 PPL (↓) WT2 KL (↓) C4 PPL (↓) C4 KL (↓)
<table><tr><td colspan="5">Llama-3.2-1B (128,256)</td></tr><tr><td>fp16</td><td>16</td><td>11.313 0</td><td>15.614</td><td>0</td></tr><tr><td>WS</td><td>2</td><td>12.049 (+6.5%)</td><td>0.06192 17.283 (+10.7%)</td><td>0.09957</td></tr><tr><td>SW (ours)</td><td>2</td><td>11.419 (+0.9%)</td><td>0.00957 (6.5×) 16.035 (+2.7%)</td><td>0.02834 (3.5×)</td></tr><tr><td>WS</td><td>3</td><td>11.469 (+1.4%) 0.01364</td><td>15.965 (+2.2%)</td><td>0.02176</td></tr><tr><td>SW (ours)</td><td>3</td><td>11.342 (+0.3%)</td><td>0.00206 (6.6×) 15.704 (+0.6%)</td><td>0.00572 (3.8×)</td></tr><tr><td>WS</td><td>4</td><td>11.356 (+0.4%)</td><td>0.00319 15.713 (+0.6%)</td><td>0.00526</td></tr><tr><td>SW (ours)</td><td>4</td><td>11.320 (+0.1%)</td><td>0.00047 (6.8×) 15.640 (+0.2%)</td><td>0.00138 (3.8×)</td></tr><tr><td colspan="5">Qwen3-32B (151,936)</td></tr><tr><td>fp16</td><td>16 8.452</td><td>0</td><td>12.269</td><td>0</td></tr><tr><td>WS</td><td>2</td><td>8.756 (+3.6%) 0.05835</td><td>13.029 (+6.2%)</td><td>0.08106</td></tr><tr><td>SW (ours)</td><td>2</td><td>8.498 (+0.5%) 0.00703 (8.3×)</td><td>12.479 (+1.7%)</td><td>0.01596 (5.1×)</td></tr><tr><td>WS</td><td>3</td><td>8.500 (+0.6%) 0.01246</td><td>12.439 (+1.4%)</td><td>0.01683</td></tr><tr><td>SW (ours)</td><td>3</td><td>8.464 (+0.1%)</td><td>0.00147 (8.5×) 12.315 (+0.4%)</td><td>0.00341 (4.9×)</td></tr><tr><td>WS</td><td>4</td><td>8.477 (+0.3%) 0.00295</td><td>12.302 (+0.3%)</td><td>0.00397</td></tr><tr><td>SW (ours)</td><td>4</td><td>8.457 (+0.1%) 0.00035 (8.4×)</td><td>12.275 (+0.1%)</td><td>0.00081 (4.9×)</td></tr></table>

## 5.1 HEAD-ONLY ISOLATION

We quantize the heads of five models to 2, 3 and 4 bits and keep their bodies in fp16; Llama-3.2-1B and Qwen3-1.7B tie head and embedding, so a separate fp16 embedding is kept. The baseline is the official WaterSIC release<sup>1</sup> without finetuning, so both arms are purely layerwise, with nominal damping $\delta = 1 0 ^ { - 6 }$ mean $( \operatorname { d i a g } \Sigma _ { X } )$ . Both are calibrated on 128 sequences of 1024 tokens from WikiText2 (WT2), and we measure perplexity (PPL) and $D _ { \mathrm { K L } } \left( p _ { \mathrm { b a s e } } \parallel p _ { \mathrm { q u a n t } } \right)$ on WT2 and C4.

Table 2 shows two of the five models; Table 7 gives the rest. Every arm lands within 0.005 bits of its target rate, and SoftWater wins 59 of 60 cells; in all five models a 2-bit SoftWater head outperforms a 3-bit WaterSIC head on $\mathbf { W T } 2 ~ \mathrm { K L }$ , so the method buys roughly one bit. The KL advantage is 6.5× to 8.3× on WT2 at 2 bits and nearly flat in head rate, but smaller out of domain<sup>2</sup> $( 3 . 5 { \times } \mathrm { - } 5 . 1 { \times }$ on C4); perplexity behaves differently, with the gap to fp16 nearly closing at 4 bits.

Table 3: Perplexity (PPL) on WikiText2 (WT2) and C4, and KL divergence from baseline: fp16- head, body quantized at 2 and 4 bits with GuidedQuant. Body and head columns give respective rates. WS: WaterSIC w/o FT. Parentheses give increase over the fp16-head row of the same body. Model size and reduction (∆size) after quantizing the tied head. Best value per head group in bold.  
Llama-3.2-1B-Instruct
<table><tr><td>Body</td><td>Method</td><td>Head</td><td>Size (MiB)</td><td>∆size</td><td>WT2 PPL (↓)</td><td>WT2 KL (↓)</td><td>C4 PPL (↓)</td><td>C4 KL (↓)</td></tr><tr><td rowspan="7">2</td><td>fp16</td><td>16</td><td>737.3</td><td>0.0%</td><td>26.81</td><td>0.497</td><td>38.54</td><td>0.511</td></tr><tr><td>WS</td><td>2</td><td>297.0</td><td>-59.7%</td><td>31.53 (+17.6%)</td><td>0.671 (+34.9%)</td><td>47.14 (+22.3%)</td><td>0.732 (+43.3%)</td></tr><tr><td>SW (ours)</td><td>2</td><td>297.0</td><td>-59.7%</td><td>27.80 (+3.7%)</td><td>0.536 (+7.9%)</td><td>42.39 (+10.0%)</td><td>0.607 (+18.8%)</td></tr><tr><td>WS</td><td>3</td><td>327.7</td><td>-55.6%</td><td>27.60 (+3.0%)</td><td>0.533 (+7.2%)</td><td>40.75 (+5.7%)</td><td>0.570 (+11.7%)</td></tr><tr><td>SW (ours)</td><td>3</td><td>327.7</td><td>-55.6%</td><td>27.03 (+0.8%)</td><td>0.506 (+1.7%)</td><td>39.49 (+2.5%)</td><td>0.536 (+4.9%)</td></tr><tr><td>WS SW (ours)</td><td>4 4</td><td>358.4</td><td>-51.4%</td><td>27.16 (+1.3%)</td><td>0.510 (+2.6%)</td><td>39.03 (+1.3%)</td><td>0.524 (+2.6%)</td></tr><tr><td></td><td></td><td>358.4</td><td>-51.4%</td><td>26.89 (+0.3%)</td><td>0.501 (+0.7%)</td><td>38.99 (+1.2%)</td><td>0.521 (+2.1%)</td></tr><tr><td rowspan="7">4</td><td>fp16</td><td>16</td><td>962.6</td><td>0.0%</td><td>16.53</td><td>0.028</td><td>24.54</td><td>0.028</td></tr><tr><td>WS</td><td></td><td>532.5</td><td>-44.7%</td><td>18.14 (+9.7%)</td><td>0.138 (+396.2%)</td><td>28.11 (+14.5%)</td><td>0.178 (+531.5%)</td></tr><tr><td>SW (ours)</td><td>22</td><td>532.5</td><td>-44.7%</td><td>17.01 (+2.9%)</td><td>0.056 (+101.9%)</td><td>26.60 (+8.4%)</td><td>0.106 (+276.9%)</td></tr><tr><td>WS</td><td>3</td><td>563.2</td><td>-41.5%</td><td>16.83 (+1.8%)</td><td>0.052 (+88.9%)</td><td>25.26 (+2.9%)</td><td>0.063 (+122.3%)</td></tr><tr><td>SW (ours)</td><td>3</td><td>563.2</td><td>-41.5%</td><td>16.64 (+0.7%)</td><td>0.034 (+24.1%)</td><td>25.12 (+2.3%)</td><td>0.052 (+84.6%)</td></tr><tr><td>WS</td><td>4</td><td>594.0</td><td>-38.3%</td><td>16.65 (+0.7%)</td><td>0.036 (+28.0%)</td><td>24.70 (+0.6%)</td><td>0.037 (+32.4%)</td></tr><tr><td>SW (ours)</td><td>4</td><td>594.0</td><td>-38.3%</td><td>16.58 (+0.3%)</td><td>0.030 (+9.3%)</td><td>24.77 (+0.9%)</td><td>0.035 (+25.3%)</td></tr></table>

## 5.2 QUANTIZED BODY, TIED HEAD

We next evaluate fully quantized Llama-3.2-1B-Instruct, whose tied head means the quantized weights are also the embedding, so head quantization hits both ends of the model at once. Since the WaterSIC repository has no quantized bodies, we use released checkpoints<sup>3</sup> (BlockLDLQ under GuidedQuant Hessians Kim et al. (2025) in the QTIP trellis format Tseng et al. (2025)), decoded to dense fp16 with a standalone reimplementation verified against the reference forward. Both quantizers calibrate on hidden states from the frozen quantized-body model, so $\Sigma _ { X }$ and λ<sup>˜</sup> reflect what the head sees at deployment, and neither activation drift nor residual compensation applies (§2.2).

Table 3 shows two of the three body rates; Table 8 gives the third. The head is 21.3% of the fp16 model’s parameters, but once the body is quantized it holds 52% to 68% of the stored bytes, so quantizing it is the only remaining way to shrink the model. A 2-bit SoftWater head removes 45– 60% of the stored bytes for a 2.9–3.7% WT2 perplexity increase, against 9.7–17.6% for WaterSIC, and SoftWater is better in 35 of 36 cells; at 4 bits the head is essentially free, costing 0.2–0.3% while still removing at least 38%. The head also dominates as bodies improve: on the 4-bit body a 2-bit head already doubles total KL under SoftWater and quintuples it under the released quantizer.

## 5.3 DOMAIN-TARGETED CALIBRATION

Rate allocation varies across domains, so we target calibration to five themes, EN Wikipedia, DE Wikipedia, Python, Math web and Case law, and evaluate in and out of domain; Table 9 in the appendix gives the corpora, splits and fields. We calibrate on 1.05M tokens rather than 131k, since at 131k most of the vocabulary is never observed and the domains would not be distinguishable through λ<sup>˜</sup>.

Table 4 shows the in-domain results from Table 10. The targeting claim holds exactly on KL: for every evaluation domain and head rate, the lowest KL among the ten arms is the SoftWater head calibrated on that domain, and at 2 bits matching calibration lowers KL by 2.0× to 5.0× over the best mismatched arm. On the diagonal SoftWater also outperforms the released quantizer everywhere at 2 bits, by 1.9× to 6.5× in KL, and wins 41 of 50 stream cells overall; seven of the nine losses are the code column under non-code calibration, the expected cost of sharpening allocation toward calibration-frequent tokens.

## 5.4 ZERO-SHOT TASKS

Perplexity and KL measure exactly what SoftWater controls, so we now check downstream accuracy. We reuse the configurations of §5.2 with head arms bit-identical to the perplexity runs, and evaluate zero-shot with lm-evaluation-harness 0.4.12 on ARC-Easy, ARC-Challenge, HellaSwag, LAMBADA, WinoGrande, PIQA and OpenBookQA, reporting normalized accuracy where defined and accuracy for LAMBADA.

Table 4: In-domain next-token top-1 accuracy of Llama-3.2-1B-Instruct, fp16 body with a tied head quantized at 2, 3 and 4 bits. Each arm is calibrated on the domain it is evaluated on. KL: divergence from the all-fp16 model on the matching stream. WS: WaterSIC w/o FT. Best value per rate in bold.
<table><tr><td colspan="2"></td><td colspan="5">Top-1 accuracy (%, ↑)</td><td colspan="5">KL (↓)</td></tr><tr><td>Head</td><td>Method</td><td>WT2</td><td>DeWiki</td><td>Code</td><td>Math</td><td>Law</td><td>WT2</td><td>DeWiki</td><td>Code</td><td>Math</td><td>Law</td></tr><tr><td>16</td><td>fp16</td><td>45.6</td><td>52.9</td><td>66.9</td><td>54.6</td><td>44.5</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td></tr><tr><td>2</td><td>WS</td><td>44.7</td><td>50.0</td><td>65.6</td><td>53.5</td><td>42.7</td><td>0.096</td><td>0.183</td><td>0.099</td><td>0.090</td><td>0.163</td></tr><tr><td rowspan="2">3</td><td>SW (ours)</td><td>45.4</td><td>52.5</td><td>66.3</td><td>54.3</td><td>44.1</td><td>0.021</td><td>0.028</td><td>0.052</td><td>0.028</td><td>0.025</td></tr><tr><td>WS</td><td>45.4</td><td>52.3</td><td>66.7</td><td>54.3</td><td>44.1</td><td>0.023</td><td>0.040</td><td>0.024</td><td>0.023</td><td>0.034</td></tr><tr><td>4</td><td>SW (ours)</td><td>45.5</td><td>52.6</td><td>66.8</td><td>54.5</td><td>44.4</td><td>0.005</td><td>0.011</td><td>0.014</td><td>0.009</td><td>0.007</td></tr><tr><td rowspan="2"></td><td>WS</td><td>45.6</td><td>52.6</td><td>66.8</td><td>54.5</td><td>44.3</td><td>0.006</td><td>0.012</td><td>0.007</td><td>0.007</td><td>0.009</td></tr><tr><td>SW (ours)</td><td>45.6</td><td>52.9</td><td>66.9</td><td>54.4</td><td>44.5</td><td>0.002</td><td>0.003</td><td>0.004</td><td>0.002</td><td>0.002</td></tr></table>

Table 5: Zero-shot accuracy (higher better) of Llama-3.2-1B-Instruct with a 2-bit tied head on GuidedQuant bodies at 2, 3 and 4 bits. WS: WaterSIC w/o FT; SW: SoftWater. CTX: context length of 131k WT2-train tokens. lm-evaluation-harness 0.4.12, 0-shot. Best value per body block in bold.
<table><tr><td>Body</td><td>Method</td><td>CTX</td><td>ARC-e</td><td>ARC-c</td><td>HellaS</td><td>LAMB</td><td>Wino</td><td>PIQA</td><td>OBQA</td><td>Mean</td></tr><tr><td>16</td><td>fp16 head</td><td></td><td>63.8</td><td>37.8</td><td>61.7</td><td>60.1</td><td>61.6</td><td>74.9</td><td>37.2</td><td>56.72</td></tr><tr><td>2</td><td>fp16 head</td><td></td><td>58.0</td><td>32.4</td><td>49.3</td><td>46.3</td><td>55.5</td><td>69.2</td><td>31.4</td><td>48.86</td></tr><tr><td rowspan="5"></td><td>WS</td><td>128</td><td>53.6</td><td>31.1</td><td>47.2</td><td>37.3</td><td>55.7</td><td>67.2</td><td>30.6</td><td>46.11</td></tr><tr><td>WS</td><td>1024</td><td>53.2</td><td>31.5</td><td>47.4</td><td>36.8</td><td>55.3</td><td>67.7</td><td>29.4</td><td>45.89</td></tr><tr><td>SW (ours)</td><td>128</td><td>56.5</td><td>31.7</td><td>48.7</td><td>45.4</td><td>56.8</td><td>67.8</td><td>32.6</td><td>48.49</td></tr><tr><td>SW (ours)</td><td>1024</td><td>54.7</td><td>31.1</td><td>47.9</td><td>43.9</td><td>55.5</td><td>68.5</td><td>31.8</td><td>47.63</td></tr><tr><td>fp16 head</td><td>一</td><td>63.0</td><td>36.3</td><td>58.3</td><td>59.5</td><td>61.7</td><td>73.1</td><td>35.6</td><td>55.35</td></tr><tr><td>3</td><td>WS</td><td>128</td><td>59.7</td><td>34.5</td><td>57.0</td><td>50.7</td><td>60.5</td><td>72.2</td><td>35.4</td><td>52.84</td></tr><tr><td></td><td>WS</td><td>1024</td><td>57.9</td><td>32.8</td><td>57.2</td><td>53.1</td><td>60.5</td><td>71.8</td><td>35.6</td><td>52.68</td></tr><tr><td></td><td>SW (ours)</td><td>128</td><td>62.8</td><td>36.6</td><td>57.2</td><td>58.5</td><td>60.2</td><td>73.0</td><td>35.2</td><td>54.78</td></tr><tr><td></td><td>SW (ours)</td><td>1024</td><td>61.0</td><td>35.4</td><td>57.5</td><td>58.4</td><td>60.2</td><td>72.3</td><td>36.2</td><td>54.43</td></tr><tr><td rowspan="5">4</td><td>fp16 head</td><td>一</td><td>62.7</td><td>37.7</td><td>60.3</td><td>60.1</td><td>62.0</td><td>73.6</td><td>37.8</td><td>56.30</td></tr><tr><td>WS</td><td>128</td><td>60.5</td><td>36.6</td><td>58.9</td><td>53.6</td><td>61.5</td><td>72.2</td><td>35.8</td><td>54.16</td></tr><tr><td>WS</td><td>1024</td><td>60.3</td><td>37.0</td><td>59.2</td><td>54.0</td><td>60.9</td><td>72.1</td><td>37.2</td><td>54.39</td></tr><tr><td>SW (ours)</td><td>128</td><td>62.2</td><td>37.4</td><td>59.6</td><td>59.4</td><td>61.8</td><td>73.2</td><td>36.4</td><td>55.72</td></tr><tr><td>SW (ours)</td><td>1024</td><td>61.2</td><td>37.5</td><td>59.3</td><td>59.2</td><td>61.6</td><td>72.7</td><td>35.0</td><td>55.23</td></tr></table>

Table 5 shows 2-bit heads on all three bodies; Tables 11–13 give every head rate. At a 2-bit head SoftWater roughly halves the mean accuracy lost to head quantization on every body, cutting the drop from 2.97, 2.67 and 1.91 points to 1.23, 0.92 and 1.07; the gap narrows with head rate and closes at 4 bits. The difference is concentrated in LAMBADA, the only task scored by next-token prediction rather than by ranking continuations, and so the one KL controls directly: at 2 bits SoftWater gains 5.2 to 7.1 points. Sharpening the grids of frequent, low-variance classes protects exactly the tokenlevel distribution KL measures.

Calibration diversity. Unlike $\Sigma _ { X }$ , the extra per-class statistic is a token frequency, so it benefits from many distinct contexts rather than many tokens. Holding the budget at 131k tokens and rechunking into 128-token contexts instead of 1024 lifts the 2-bit-head mean by 0.35 to 0.86 points, while moving WaterSIC by at most 0.23; quadrupling the budget at context length 1024 helps by a similar amount and buys nothing that re-chunking does not. Context count, not token count, is the binding resource for λ<sup>˜</sup>.

## 5.5 CLASS-SIDE STATISTIC ABLATION

For most tokens $\tilde { p } _ { k } ( 1 - \tilde { p } _ { k } ) \approx \tilde { p } _ { k }$ because $\tilde { p } _ { k }$ is small, so we ask whether the second-moment correction matters: we replace the average softmax curvature $\mathbb { E } _ { X } [ \tilde { p } _ { k } ( 1 - \tilde { p } _ { k } ) ]$ ] with the average marginal $\bar { p } _ { k } = \mathbb { E } _ { X } [ \tilde { p } _ { k } ]$ ]. We also replace it with $\tilde { \pi } _ { k } ,$ , the unigram frequency of token k smoothed the same way, which requires no forward pass and asks whether the model carries information the corpus does not.

(a) $\rho \colon$ <sub>relative</sub> <sub>proxy</sub> <sub>error</sub> <sub>(22).</sub> <sub>bits</sub> <sub>=</sub> (b) Convergence of alternating power iteration to its fixed point $\frac { 1 } { 2 } \log _ { 2 } ( 1 + \rho )$ . SW calibrated with WT2; (a<sup>⋆</sup>, B<sup>⋆</sup>) from three initializations. SW: $a _ { 0 } = \tilde { \lambda } , B _ { 0 } = \Sigma _ { X } ;$ Eval is the sample the ratio is measured on. WS: a<sub>0</sub> = I, $\boldsymbol { B } _ { 0 } = \boldsymbol { \Sigma } _ { \boldsymbol { X } } ;$ Id: $a _ { 0 } = I ,$ $B _ { 0 } = I .$
<table><tr><td rowspan="2">Eval Rate</td><td colspan="3">WT2</td><td colspan="3">C4</td><td rowspan="2">Init Round</td><td colspan="3">SW (ours) 0</td><td colspan="3">WS</td><td colspan="3">Id</td></tr><tr><td>2</td><td>3</td><td>4</td><td>2</td><td>3</td><td>4</td><td></td><td></td><td>1</td><td>2</td><td></td><td>2</td><td></td><td>1</td><td>2</td></tr><tr><td>ρ</td><td>+0.10</td><td>+0.09</td><td>+0.09</td><td>+0.08</td><td>+0.08</td><td>+0.07</td><td>cos(a, a*)</td><td>0.99</td><td>1.00</td><td>1.00</td><td>0.03</td><td>1.00</td><td>1.00</td><td>0.03</td><td>0.99</td><td>1.00</td></tr><tr><td>Bits</td><td>0.07</td><td>0.06</td><td>0.06</td><td>0.06</td><td>0.06</td><td>0.05</td><td>cos(B, B*)</td><td>0.98</td><td>1.00 1.00</td><td></td><td>0.98</td><td>1.00</td><td>1.00</td><td>0.05</td><td>1.00</td><td>1.00</td></tr></table>

Table 14 shows the results. SoftWater and the $\bar { p } _ { k }$ -variant are close in domain, but SoftWater wins every out-of-domain KL cell and 12 of 15 in-domain ones, so the second-moment correction is small but systematic and is what makes the allocation transfer off the calibration domain. Both variants beat $\tilde { \pi } _ { k }$ at every rate on every model, by roughly 2× in KL, so the model’s own marginal carries information the corpus count does not. Since $\lambda _ { k }$ and $\bar { p } _ { k }$ come from the same forward pass, SoftWater is the best of the three.

## 5.6 SEPARABILITY

To test Assumption 1, define $\begin{array} { r } { Q _ { k } = \frac { 1 } { 2 } \Delta _ { k } \mathbb { E } _ { X } [ \lambda _ { k } ( X ) X X ^ { \top } ] \Delta _ { k } ^ { \top } } \end{array}$ and $\begin{array} { r } { S _ { k } = \frac 1 2 \tilde { \lambda } _ { k } \Delta _ { k } \Sigma _ { X } \Delta _ { k } ^ { \top } } \end{array}$ , the contribution of class k to the true and approximate distortion, and the relative proxy error

$$
\rho = \frac { \sum _ { k = 1 } ^ { K } Q _ { k } } { \sum _ { k = 1 } ^ { K } S _ { k } } - 1 ,\tag{22}
$$

whose rate penalty ${ \frac { 1 } { 2 } } \log _ { 2 } ( 1 + \rho )$ is the extra bits needed to close the gap. Since $S _ { k }$ uses the smoothed $\tilde { \lambda } _ { k }$ that the deployed scheme optimizes, $\rho$ combines the separability gap with the deliberate bias from smoothing. Quantizing the head with SoftWater calibrated on WT2 and computing both sums on WT2 and C4, Table 6a shows the proxy underestimates the true distortion by at most 10%, a rate penalty of at most 0.07 bits.

Since the fit is imperfect, we ask whether better Kronecker factorizations exist. We adapt the power iteration of Tseng et al. (2026) (Appendix B) to obtain sketches $a \otimes B$ with a diagonal, run on $\mathbb { E } _ { X } [ \tilde { \lambda } ( X ) \otimes X X ^ { \top } ] \mathrm { a t } \epsilon = 0 . 1$ , so its fixed point is the Frobenius-optimal factorization of the weight SoftWater targets. Comparing three initializations, $( \tilde { \lambda } , \Sigma _ { X } )$ (SoftWater), $( I , \Sigma _ { X } )$ (WaterSIC) and (I, I), Table 6b shows the SoftWater factors are close to optimal from the start.

## 6 CONCLUSION

We posed softmax-layer quantization as a rate-distortion problem under output KL, and reduced it via a separable second-order surrogate to a class-rescaled lattice whose two statistics come from one calibration pass. SoftWater allocates rate by class frequency and variance, with a smoothing prior that caps the spread on uncovered classes and interpolates back to WaterSIC as the prior takes over. Across five models from 1B to 32B parameters it outperforms the released WaterSIC quantizer at matched head rates without any of that pipeline’s refinements, halves the zero-shot accuracy lost to a 2-bit head, targets a deployment domain through its calibration statistic, and shrinks whole-model size by up to 60%. Two limitations bound these results: the separable surrogate is accurate to within 10% on the models we measure but not guaranteed in general, and the deployed-model experiments rest on a single family of released quantized bodies. Beyond LLM heads, the analysis applies to any linear-softmax layer with a fixed class dimension, including vision and speech classifiers and MoE routers, whose skewed expert utilization mirrors the Zipfian structure we exploit.

## REFERENCES

Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan Alistarh, Torsten Hoefler, and James Hensman. QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs, October 2024. URL http://arxiv.org/abs/2404.00456. arXiv:2404.00456 [cs].

Jerry Chee, Yaohui Cai, Volodymyr Kuleshov, and Christopher De Sa. QuIP: 2-Bit Quantization of Large Language Models With Guarantees, January 2024. URL http://arxiv.org/abs/ 2307.13304. arXiv:2307.13304 [cs].

Jiale Chen, Yalda Shabanzadeh, Elvir Crnceviˇ c, Torsten Hoefler, and Dan Alistarh. The Geometry´ of LLM Quantization: GPTQ as Babai’s Nearest Plane Algorithm, May 2026. URL http: //arxiv.org/abs/2507.18553. arXiv:2507.18553 [cs.LG].

Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers, March 2023a. URL http://arxiv. org/abs/2210.17323. arXiv:2210.17323 [cs].

Elias Frantar, Sidak Pal Singh, and Dan Alistarh. Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning, January 2023b. URL http://arxiv.org/ abs/2208.11580. arXiv:2208.11580 [cs].

Edouard Grave, Armand Joulin, Moustapha Cisse, David Grangier, and Herv´ e J´ egou. Efficient soft-´ max approximation for GPUs, June 2017. URL http://arxiv.org/abs/1609.04309. arXiv:1609.04309 [cs.CL].

Sebastien Jean, Kyunghyun Cho, Roland Memisevic, and Yoshua Bengio. On Using Very Large´ Target Vocabulary for Neural Machine Translation. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 1–10, Beijing, China, 2015. Association for Computational Linguistics. doi: 10.3115/v1/P15-1001. URL http: //aclweb.org/anthology/P15-1001.

Jinuk Kim, Marwa El Halabi, Wonpyo Park, Clemens JS Schaefer, Deokjae Lee, Yeonhong Park, Jae W. Lee, and Hyun Oh Song. GuidedQuant: Large Language Model Quantization via Exploit ing End Loss Guidance, September 2025. URL http://arxiv.org/abs/2505.07004. arXiv:2505.07004 [cs.LG].

Jung Hyun Lee, June Yong Yang, Jungwook Choi, and Eunho Yang. LFQ: Logit-aware Final-block Quantization for Boosting the Generation Quality of Low-Bit Quantized LLMs, May 2026. URL http://arxiv.org/abs/2605.29756. arXiv:2605.29756 [cs.AI].

Egor Lifar, Semyon Savkin, Or Ordentlich, and Yury Polyanskiy. WaterSIC: Information-Theoretically (Near) Optimal Linear Layer Quantization, June 2026. URL http://arxiv. org/abs/2603.04956. arXiv:2603.04956 [cs.LG].

Dong Liu, Yanxuan Yu, and Ben Lengerich. CSV-Decode: Certifiable Sub-Vocabulary Decoding for Efficient Large Language Model Inference, November 2025a. URL http://arxiv.org/ abs/2511.21702. arXiv:2511.21702 [cs.CL].

Zechun Liu, Changsheng Zhao, Igor Fedorov, Bilge Soran, Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas Chandra, Yuandong Tian, and Tijmen Blankevoort. SpinQuant: LLM quantization with learned rotations, February 2025b. URL http://arxiv.org/abs/2405.16406. arXiv:2405.16406 [cs].

James Martens and Roger Grosse. Optimizing Neural Networks with Kronecker-factored Approximate Curvature, 2015. URL https://arxiv.org/abs/1503.05671. Version Number: 7.

Markus Nagel, Rana Ali Amjad, Mart van Baalen, Christos Louizos, and Tijmen Blankevoort. Up or Down? Adaptive Rounding for Post-Training Quantization, June 2020. URL http://arxiv. org/abs/2004.10568. arXiv:2004.10568 [cs].

Semyon Savkin, Eitan Porat, Or Ordentlich, and Yury Polyanskiy. NestQuant: Nested Lattice Quantization for Matrix Products and LLMs, July 2025. URL http://arxiv.org/abs/2502. 09720. arXiv:2502.09720 [cs.LG].

Jintian Shao, Hongyi Huang, Jiayi Wu, YiMing Cheng, ZhiYu Wu, You Shan, and MingKai Zheng. VQ-Logits: Compressing the Output Bottleneck of Large Language Models via Vector Quantized Logits, May 2025a. URL http://arxiv.org/abs/2505.10202. arXiv:2505.10202 [cs.CL].

Wenqi Shao, Mengzhao Chen, Zhaoyang Zhang, Peng Xu, Lirui Zhao, Zhiqian Li, Kaipeng Zhang, Peng Gao, Yu Qiao, and Ping Luo. OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models, March 2024. URL http://arxiv.org/abs/2308.13137. arXiv:2308.13137 [cs.LG].

Yuantian Shao, Yuanteng Chen, Peisong Wang, Jianlin Yu, Jing Lin, Yiwu Yao, Zhihui Wei, and Jian Cheng. DartQuant: Efficient Rotational Distribution Calibration for LLM Quantization, November 2025b. URL http://arxiv.org/abs/2511.04063. arXiv:2511.04063 [cs].

Wilhelm Tranheden, Shahnawaz Ahmed, Devdatt Dubhashi, Jonna Matthiesen, and Hannes von Essen. FlashHead: Efficient Drop-In Replacement for the Classification Head in Language Model Inference, March 2026. URL http://arxiv.org/abs/2603.14591. arXiv:2603.14591 [cs.LG].

Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, and Christopher De Sa. QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks, June 2024. URL http://arxiv.org/abs/2402.04396. arXiv:2402.04396 [cs].

Albert Tseng, Qingyao Sun, David Hou, and Christopher De Sa. QTIP: Quantization with Trellises and Incoherence Processing, June 2025. URL http://arxiv.org/abs/2406.11235. arXiv:2406.11235 [cs].

Albert Tseng, Zhaofeng Sun, and Christopher De Sa. Model-Preserving Adaptive Rounding, June 2026. URL http://arxiv.org/abs/2505.22988. arXiv:2505.22988 [cs.LG].

Charles F. Van Loan. The ubiquitous Kronecker product. Journal of Computational and Applied Mathematics, 123(1-2):85–100, November 2000. ISSN 03770427. doi: 10.1016/ S0377-0427(00)00393-9. URL https://linkinghub.elsevier.com/retrieve/ pii/S0377042700003939.

Shihao Zhang, Haoyu Zhang, Ian Colbert, and Rayan Saab. Qronos: Correcting the Past by Shaping the Future... in Post-Training Quantization, February 2026. URL http://arxiv.org/abs/ 2505.11695. arXiv:2505.11695 [cs.LG].

## A EXPERIMENT DETAILS

## A.1 HEAD-ONLY ISOLATION

Table 7: Head-only isolation: the body is kept in fp16 and only the head matrix is quantized; tied models (Llama-3.2-1B, Qwen3-1.7B) retain an fp16 copy of the embedding. KL is measured against the all-fp16 model on identical token streams. WS: WaterSIC w/o FT. Parentheses after each model name give its vocabulary size V ; on PPL cells, the increase over the fp16 row of the same model; on SW KL cells, the WS/SW KL ratio at that rate. Best value per head group in bold.
<table><tr><td>Method</td><td>Head</td><td>WT2 PPL (↓)</td><td>WT2 KL (↓)</td><td>C4 PPL (↓)</td><td>C4 KL (↓)</td></tr><tr><td colspan="6">Llama-3.2-1B (128,256)</td></tr><tr><td>fp16</td><td>16</td><td>11.313</td><td>0</td><td>15.614</td><td>0</td></tr><tr><td>WS SW (ours)</td><td>2</td><td>12.049 (+6.5%) 11.419 (+0.9%)</td><td>0.06192 0.00957 (6.5×)</td><td>17.283 (+10.7%) 16.035 (+2.7%)</td><td>0.09957 0.02834 (3.5×)</td></tr><tr><td>WS</td><td>2 3</td><td>11.469 (+1.4%)</td><td>0.01364</td><td>15.965 (+2.2%)</td><td>0.02176</td></tr><tr><td>SW (ours)</td><td>3</td><td>11.342 (+0.3%)</td><td>0.00206 (6.6×)</td><td>15.704 (+0.6%)</td><td>0.00572 (3.8×)</td></tr><tr><td>WS SW (ours)</td><td>4 4</td><td>11.356 (+0.4%) 11.320 (+0.1%)</td><td>0.00319 0.00047 (6.8×)</td><td>15.713 (+0.6%) 15.640 (+0.2%)</td><td>0.00526 0.00138 (3.8×)</td></tr><tr><td></td><td></td><td></td><td>Qwen3-1.7B (151,936)</td><td></td><td></td></tr><tr><td colspan="6"></td></tr><tr><td>fp16 WS</td><td>16 2</td><td>18.285 18.876 (+3.2%)</td><td>0 0.06788</td><td>22.621 24.129 (+6.7%)</td><td>0</td></tr><tr><td>SW (ours) WS</td><td>2</td><td>18.460 (+1.0%)</td><td>0.00900 (7.5×)</td><td>23.075 (+2.0%)</td><td>0.09054 0.01998 (4.5×)</td></tr><tr><td>SW (ours)</td><td>3 3</td><td>18.374 (+0.5%) 18.317 (+0.2%)</td><td>0.01448 0.00199 (7.3×)</td><td>22.916 (+1.3%) 22.724 (+0.5%)</td><td>0.02017 0.00443 (4.6×)</td></tr><tr><td>WS SW (ours)</td><td>4 4</td><td>18.277 (−0.0%) 18.298 (+0.1%)</td><td>0.00347 0.00045 (7.7×)</td><td>22.650 (+0.1%) 22.633 (+0.1%)</td><td>0.00487 0.00105 (4.6×)</td></tr><tr><td colspan="6">Llama-3.1-8B-Instruct (128,256)</td></tr><tr><td>fp16 WS</td><td>16 2</td><td>8.388 8.758 (+4.4%)</td><td>0 0.04583</td><td>14.189 15.462 (+9.0%)</td><td>0 0.07487</td></tr><tr><td>SW (ours) WS</td><td>2</td><td>8.437 (+0.6%) 8.458 (+0.8%)</td><td>0.00636 (7.2×) 0.00940</td><td>14.534 (+2.4%) 14.439 (+1.8%)</td><td>0.02140 (3.5×)</td></tr><tr><td>SW (ours) WS</td><td>3 3</td><td>8.398 (+0.1%)</td><td>0.00135 (7.0×)</td><td>14.258 (+0.5%)</td><td>0.01622 0.00437 (3.7×)</td></tr><tr><td>SW (ours)</td><td>4 4</td><td>8.402 (+0.2%) 8.391 (+0.0%)</td><td>0.00221 0.00032 (6.8×)</td><td>14.255 (+0.5%) 14.205 (+0.1%)</td><td>0.00383 0.00103 (3.7×)</td></tr><tr><td colspan="6">Qwen3-8B (151,936)</td></tr><tr><td>fp16 WS</td><td>16</td><td>10.832 11.081 (+2.3%)</td><td>0 0.04592</td><td>15.486 16.229 (+4.8%)</td><td>0</td></tr><tr><td>SW (ours) WS</td><td>2 2</td><td>10.898 (+0.6%)</td><td>0.00654 (7.0×)</td><td>15.689 (+1.3%)</td><td>0.06861 0.01385 (5.0×)</td></tr><tr><td>SW (ours)</td><td>3 3</td><td>10.877 (+0.4%) 10.848 (+0.1%)</td><td>0.00964 0.00132 (7.3×)</td><td>15.572 (+0.6%) 15.523 (+0.2%)</td><td>0.01319 0.00295 (4.5×)</td></tr><tr><td>WS SW (ours)</td><td>4 4</td><td>10.854 (+0.2%) 10.836 (+0.0%)</td><td>0.00230 0.00031 (7.3×)</td><td>15.511 (+0.2%) 15.501 (+0.1%)</td><td>0.00320 0.00073 (4.4×)</td></tr><tr><td colspan="6">Qwen3-32B (151,936)</td></tr><tr><td>fp16 WS</td><td>16</td><td>8.452 8.756 (+3.6%)</td><td>0 0.05835</td><td>12.269 13.029 (+6.2%)</td><td>0 0.08106</td></tr><tr><td>SW (ours)</td><td>2 2</td><td>8.498 (+0.5%)</td><td>0.00703 (8.3×)</td><td>12.479 (+1.7%)</td><td>0.01596 (5.1×)</td></tr><tr><td>WS SW (ours)</td><td>3 3</td><td>8.500 (+0.6%) 8.464 (+0.1%)</td><td>0.01246 0.00147 (8.5×)</td><td>12.439 (+1.4%) 12.315 (+0.4%)</td><td>0.01683 0.00341 (4.9×)</td></tr><tr><td>WS SW (ours)</td><td>4 4</td><td>8.477 (+0.3%) 8.457 (+0.1%)</td><td>0.00295 0.00035 (8.4×)</td><td>12.302 (+0.3%) 12.275 (+0.1%)</td><td>0.00397 0.00081 (4.9×)</td></tr></table>

## A.2 QUANTIZED BODY, TIED HEAD

Table 8: Perplexity (PPL) on WikiText2 (WT2) and C4, and KL divergence from the baseline model, consisting of the body quantized at 2, 3 and 4 bits with GuidedQuant (BlockLDLQ, QTIP format, released checkpoints) and the head kept in fp16. Body and head columns give the respective rates. The quantized head matrix is tied (also the input embedding). WS: WaterSIC w/o FT. Parentheses give the increase over the fp16-head row of the same body. Whole-model size in MiB and reduction (∆size) after quantizing the head. Best value per head group in bold.  
Llama-3.2-1B-Instruct
<table><tr><td>Body</td><td>Method</td><td>Head</td><td>Size (MiB)</td><td>∆size</td><td>WT2 PPL (↓)</td><td>WT2 KL (↓)</td><td>C4 PPL (↓)</td><td>C4 KL (↓)</td></tr><tr><td rowspan="4">2.00</td><td>fp16</td><td>16.00</td><td>737.3</td><td>0.0% -59.7%</td><td>26.81</td><td>0.497 0.671 (+34.9%)</td><td>38.54 47.14 (+22.3%)</td><td>0.511 0.732 (+43.3%)</td></tr><tr><td>WS SW (ours)</td><td>2.00 2.00</td><td>297.0 297.0</td><td>-59.7%</td><td>31.53 (+17.6%) 27.80 (+3.7%)</td><td>0.536 (+7.9%)</td><td>42.39 (+10.0%)</td><td>0.607 (+18.8%)</td></tr><tr><td>WS SW (ours)</td><td>3.00 3.00</td><td>327.7 327.7</td><td>-55.6% -55.6%</td><td>27.60 (+3.0%) 27.03 (+0.8%)</td><td>0.533 (+7.2%) 0.506 (+1.7%)</td><td>40.75 (+5.7%) 39.49 (+2.5%)</td><td>0.570 (+11.7%) 0.536 (+4.9%)</td></tr><tr><td>WS SW (ours)</td><td>4.00 4.00</td><td>358.4 358.4</td><td>-51.4% -51.4%</td><td>27.16 (+1.3%) 26.89 (+0.3%)</td><td>0.510 (+2.6%) 0.501 (+0.7%)</td><td>39.03 (+1.3%) 38.99 (+1.2%)</td><td>0.524 (+2.6%) 0.521 (+2.1%)</td></tr><tr><td rowspan="4">3.00</td><td>fp16</td><td>16.00</td><td>850.0</td><td>0.0%</td><td>17.85</td><td>0.101</td><td>26.15</td><td>0.096</td></tr><tr><td>WS SW (ours)</td><td>2.00 2.00</td><td>409.6 409.6</td><td>-51.8% -51.8%</td><td>19.99 (+12.0%) 18.44 (+3.3%)</td><td>0.230 (+126.8%) 0.136 (+34.5%)</td><td>30.84 (+17.9%) 28.52 (+9.0%)</td><td>0.269 (+180.2%) 0.180 (+87.5%)</td></tr><tr><td>WS SW (ours)</td><td>3.00 3.00</td><td>440.3 440.3</td><td>-48.2% -48.2%</td><td>18.23 (+2.1%) 17.98 (+0.7%)</td><td>0.128 (+26.0%) 0.109 (+7.1%)</td><td>26.96 (+3.1%) 26.77 (+2.4%)</td><td>0.135 (+40.4%) 0.119 (+23.8%)</td></tr><tr><td>WS SW (ours)</td><td>4.00 4.00</td><td>471.0 471.0</td><td>-44.6% -44.6%</td><td>18.08 (+1.3%) 17.88 (+0.2%)</td><td>0.113 (+11.5%) 0.104 (+2.3%)</td><td>26.45 (+1.1%) 26.32 (+0.6%)</td><td>0.107 (+11.3%) 0.102 (+6.7%)</td></tr><tr><td rowspan="4">4.00</td><td>fp16</td><td>16.00</td><td>962.6</td><td>0.0%</td><td>16.53</td><td>0.028</td><td>24.54</td><td>0.028</td></tr><tr><td>WS SW (ours)</td><td>2.00 2.00</td><td>532.5 532.5</td><td>-44.7% -44.7%</td><td>18.14 (+9.7%) 17.01 (+2.9%)</td><td>0.138 (+396.2%) 0.056 (+101.9%)</td><td>28.11 (+14.5%) 26.60 (+8.4%)</td><td>0.178 (+531.5%) 0.106 (+276.9%)</td></tr><tr><td>WS SW (ours)</td><td>3.00 3.00</td><td>563.2 563.2</td><td>-41.5% -41.5%</td><td>16.83 (+1.8%) 16.64 (+0.7%)</td><td>0.052 (+88.9%) 0.034 (+24.1%)</td><td>25.26 (+2.9%) 25.12 (+2.3%)</td><td>0.063 (+122.3%) 0.052 (+84.6%)</td></tr><tr><td>WS SW (ours)</td><td>4.00 4.00</td><td>594.0 594.0</td><td>-38.3% -38.3%</td><td>16.65 (+0.7%) 16.58 (+0.3%)</td><td>0.036 (+28.0%) 0.030 (+9.3%)</td><td>24.70 (+0.6%) 24.77(+0.9%)</td><td>0.037 (+32.4%) 0.035 (+25.3%)</td></tr></table>

## A.3 DOMAIN-TARGETED CALIBRATION

Table 9: Domains used for calibration and evaluation. Calibration and evaluation slices are disjoint: WT2 by split, the remaining domains by offsetting the calibration stream past the evaluation slice (skip 32). Every calibration slice is 1024 × 1024 tokens and every evaluation slice 32 × 1024 tokens.
<table><tr><td>Domain</td><td>Corpus</td><td>Config</td><td>Split</td><td>Field</td></tr><tr><td>WT2</td><td>wikitext</td><td>wikitext-2-raw-v1</td><td>train/test</td><td>text</td></tr><tr><td>DeWiki</td><td>wikimedia/wikipedia</td><td>20231101.de</td><td>train</td><td>text</td></tr><tr><td>Code</td><td>codeparrot/codeparrot-clean-valid</td><td></td><td>train</td><td>content</td></tr><tr><td>Math</td><td>open-web-math/open-web-math</td><td></td><td>train</td><td>text</td></tr><tr><td>Law</td><td>HFforLegal/case-law</td><td></td><td>us</td><td>document</td></tr></table>

Table 10: Calibration × evaluation domain crossover on Llama-3.2-1B-Instruct, fp16 body with a tied head quantized at 2, 3 and 4 bits. Each block gives the calibration domain used for both λ<sup>˜</sup> and $\Sigma _ { X }$ ; diagonal cells (calibration domain matches evaluation domain) are the targeting claim. Accuracy columns: next-token top-1 accuracy in percent, higher is better. KL columns: divergence from the all-fp16 model on the matching stream, lower is better. WS: WaterSIC w/o FT. The fp16- head row is repeated per rate as the reference. Best value per calibration block in bold; ties bold both.
<table><tr><td colspan="2"></td><td colspan="5">Accuracy (%, ↑)</td><td colspan="5">KL (↓)</td></tr><tr><td>Calib</td><td>Method</td><td>WT2</td><td>DeWiki</td><td>Code</td><td>Math</td><td>Law</td><td>WT2</td><td>DeWiki</td><td>Code</td><td>Math</td><td>Law</td></tr><tr><td colspan="10">Head @ 2 bits</td></tr><tr><td></td><td>fp16</td><td>45.6</td><td>52.9</td><td>66.9</td><td>54.6</td><td>44.5</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td></tr><tr><td>WT2</td><td>WS SW (ours)</td><td>44.7 45.4</td><td>50.0 50.5</td><td>64.8 64.2</td><td>52.8 53.8</td><td>42.9 43.6</td><td>0.096 0.021</td><td>0.212 0.200</td><td>0.151 0.218</td><td>0.123 0.080</td><td>0.157 0.086</td></tr><tr><td>DeWiki</td><td>WS SW (ours)</td><td>43.3 44.2</td><td>50.0 52.5</td><td>64.5 64.1</td><td>52.1 53.0</td><td>41.6 43.0</td><td>0.172 0.117</td><td>0.183 0.028</td><td>0.174 0.219</td><td>0.163 0.110</td><td>0.252 0.141</td></tr><tr><td>Code</td><td>WS SW (ours)</td><td>43.4 43.9</td><td>49.5 49.5</td><td>65.6 66.3</td><td>53.1 53.9</td><td>42.0 43.1</td><td>0.173 0.135</td><td>0.245 0.272</td><td>0.099 0.052</td><td>0.121 0.056</td><td>0.210 0.138</td></tr><tr><td>Math</td><td>WS SW (ours)</td><td>44.1 44.8</td><td>50.2 51.4</td><td>65.4 65.6</td><td>53.5 54.3</td><td>42.7 43.7</td><td>0.135 0.062</td><td>0.194 0.139</td><td>0.118 0.119</td><td>0.090 0.028</td><td>0.165 0.083</td></tr><tr><td>Law</td><td>WS SW (ours)</td><td>43.5 44.9</td><td>49.4 50.2</td><td>64.8 64.5</td><td>52.7 54.0</td><td>42.7 44.1</td><td>0.144 0.060</td><td>0.262 0.215</td><td>0.157 0.182</td><td>0.135 0.056</td><td>0.163 0.025</td></tr><tr><td colspan="10">Head @ 3 bits</td></tr><tr><td>WT2</td><td>fp16</td><td>45.6</td><td>52.9</td><td>66.9</td><td>54.6</td><td>44.5</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td></tr><tr><td></td><td>WS SW (ours)</td><td>45.4 45.5</td><td>52.2 52.2</td><td>66.5 66.4</td><td>54.1 54.4</td><td>43.9 44.2</td><td>0.023 0.005</td><td>0.053 0.052</td><td>0.034 0.049</td><td>0.028 0.015</td><td>0.040 0.022</td></tr><tr><td>DeWiki</td><td>WS SW (ours)</td><td>45.1 45.2</td><td>52.3 52.6</td><td>66.4 66.3</td><td>54.1 54.3</td><td>44.0 43.9</td><td>0.037 0.026</td><td>0.040 0.011</td><td>0.037 0.045</td><td>0.034 0.024</td><td>0.055 0.033</td></tr><tr><td>Code</td><td>WS SW (ours)</td><td>45.0 45.2</td><td>52.2 52.1</td><td>66.7 66.8</td><td>54.2 54.4</td><td>43.9 44.2</td><td>0.039 0.027</td><td>0.059 0.062</td><td>0.024 0.014</td><td>0.030 0.013</td><td>0.050 0.030</td></tr><tr><td>Math</td><td>WS SW (ours)</td><td>45.4 45.4</td><td>52.2 52.2</td><td>66.5 66.6</td><td>54.3 54.5</td><td>44.0 44.2</td><td>0.030 0.016</td><td>0.047 0.041</td><td>0.027 0.030</td><td>0.023 0.009</td><td>0.043 0.021</td></tr><tr><td>Law</td><td>WS SW (ours)</td><td>45.1 45.5</td><td>52.0 52.2</td><td>66.4 66.4</td><td>54.0 54.5</td><td>44.1 44.4</td><td>0.034 0.016</td><td>0.057 0.056</td><td>0.035 0.042</td><td>0.031 0.013</td><td>0.034 0.007</td></tr><tr><td colspan="10">Head @ 4 bits</td></tr><tr><td></td><td>fp16</td><td>45.6</td><td>52.9</td><td>66.9</td><td>54.6</td><td>44.5</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td></tr><tr><td>WT2</td><td>WS SW (ours)</td><td>45.6 45.6</td><td>52.6 52.7</td><td>66.8 66.8</td><td>54.5 54.5</td><td>44.1 44.4</td><td>0.006 0.002</td><td>0.015 0.016</td><td>0.011 0.014</td><td>0.008 0.005</td><td>0.011 0.007</td></tr><tr><td>DeWiki</td><td>WS SW (ours)</td><td>45.5 45.5</td><td>52.6 52.9</td><td>66.9 66.7</td><td>54.5 54.5</td><td>44.2 44.3</td><td>0.010 0.009</td><td>0.012 0.003</td><td>0.011 0.015</td><td>0.010 0.008</td><td>0.014 0.011</td></tr><tr><td>Code</td><td>WS SW (ours)</td><td>45.3 45.5</td><td>52.5 52.6</td><td>66.8 66.9</td><td>54.5 54.5</td><td>44.2 44.4</td><td>0.011 0.008</td><td>0.015 0.021</td><td>0.007 0.004</td><td>0.008 0.004</td><td>0.014 0.009</td></tr><tr><td>Math</td><td>WS SW (ours)</td><td>45.4 45.6</td><td>52.7 52.7</td><td>66.9 66.8</td><td>54.5 54.4</td><td>44.3 44.4</td><td>0.008 0.006</td><td>0.013 0.013</td><td>0.008 0.009</td><td>0.007 0.002</td><td>0.013 0.007</td></tr><tr><td>Law</td><td>WS SW (ours)</td><td>45.5 45.5</td><td>52.5 52.7</td><td>66.8 66.8</td><td>54.4 54.5</td><td>44.3 44.5</td><td>0.009 0.005</td><td>0.015 0.015</td><td>0.009 0.011</td><td>0.008 0.004</td><td>0.009 0.002</td></tr></table>

## A.4 ZERO-SHOT TASKS

Table 11: Zero-shot accuracy (higher better) of Llama-3.2-1B-Instruct with a 2-bit GuidedQuant body and a tied head quantized at 2, 3 and 4 bits. WS: WaterSIC w/o FT; SW: SoftWater. CTX: context length of 131k WT2-train tokens. lm-evaluation-harness 0.4.12, 0-shot. Best value per body block in bold.
<table><tr><td>Head</td><td>Method</td><td>CTX</td><td>Tokens</td><td>ARC-e</td><td>ARC-c</td><td>HellaS</td><td>LAMB</td><td>Wino</td><td>PIQA</td><td>OBQA</td><td>Mean</td></tr><tr><td>16</td><td>fp16 body</td><td></td><td></td><td>63.8</td><td>37.8</td><td>61.7</td><td>60.1</td><td>61.6</td><td>74.9</td><td>37.2</td><td>56.72</td></tr><tr><td>16</td><td>fp16 head</td><td>二</td><td></td><td>58.0</td><td>32.4</td><td>49.3</td><td>46.3</td><td>55.5</td><td>69.2</td><td>31.4</td><td>48.86</td></tr><tr><td rowspan="8">2</td><td>WS</td><td>128</td><td>131k</td><td>53.6</td><td>31.1</td><td>47.2</td><td>37.3</td><td>55.7</td><td>67.2</td><td>30.6</td><td>46.11</td></tr><tr><td>WS</td><td>256</td><td>131k</td><td>52.5</td><td>29.9</td><td>47.0</td><td>36.6</td><td>56.2</td><td>67.4</td><td>30.8</td><td>45.78</td></tr><tr><td>WS</td><td>1024</td><td>131k</td><td>53.2</td><td>31.5</td><td>47.4</td><td>36.8</td><td>55.3</td><td>67.7</td><td>29.4</td><td>45.89</td></tr><tr><td>WS</td><td>1024</td><td>1.05M</td><td>54.0</td><td>30.7</td><td>47.2</td><td>38.4</td><td>55.6</td><td>67.1</td><td>28.8</td><td>45.97</td></tr><tr><td>SW (ours)</td><td>128</td><td>131k</td><td>56.5</td><td>31.7</td><td>48.7</td><td>45.4</td><td>56.8</td><td>67.8</td><td>32.6</td><td>48.49</td></tr><tr><td>SW (ours)</td><td>256</td><td>131k</td><td>53.4</td><td>29.7</td><td>48.1</td><td>44.9</td><td>56.4</td><td>66.9</td><td>31.6</td><td>47.30</td></tr><tr><td>SW (ours)</td><td>1024</td><td>131k</td><td>54.7</td><td>31.1</td><td>47.9</td><td>43.9</td><td>55.5</td><td>68.5</td><td>31.8</td><td>47.63</td></tr><tr><td>SW (ours)</td><td>1024</td><td>1.05M</td><td>55.7</td><td>31.7</td><td>48.3</td><td>44.8</td><td>56.7</td><td>68.0</td><td>30.0</td><td>47.87</td></tr><tr><td rowspan="8">3</td><td>WS</td><td>128</td><td>131k</td><td>55.1</td><td>31.3</td><td>49.1</td><td>43.7</td><td>55.8</td><td>68.8</td><td>31.0</td><td>47.83</td></tr><tr><td>WS</td><td>256</td><td>131k</td><td>55.2</td><td>31.3</td><td>48.9</td><td>43.8</td><td>55.2</td><td>69.2</td><td>30.4</td><td>47.71</td></tr><tr><td>WS</td><td>1024</td><td>131k</td><td>56.1</td><td>31.1</td><td>48.8</td><td>43.1</td><td>56.9</td><td>67.9</td><td>30.2</td><td>47.73</td></tr><tr><td>WS</td><td>1024</td><td>1.05M</td><td>56.3</td><td>31.6</td><td>49.1</td><td>43.6</td><td>56.5</td><td>68.0</td><td>30.0</td><td>47.87</td></tr><tr><td>SW (ours)</td><td>128</td><td>131k</td><td>57.4</td><td>32.1</td><td>49.4</td><td>45.9</td><td>56.7</td><td>68.7</td><td>31.6</td><td>48.82</td></tr><tr><td>SW (ours)</td><td>256</td><td>131k</td><td>58.0</td><td>32.2</td><td>49.3</td><td>46.7</td><td>57.5</td><td>68.5</td><td>31.2</td><td>49.04</td></tr><tr><td>SW (ours)</td><td>1024</td><td>131k</td><td>57.0</td><td>32.5</td><td>49.0</td><td>45.6</td><td>56.6</td><td>68.8</td><td>31.6</td><td>48.72</td></tr><tr><td>SW (ours)</td><td>1024</td><td>1.05M</td><td>57.3</td><td>32.3</td><td>49.2</td><td>46.2</td><td>57.2</td><td>68.7</td><td>31.6</td><td>48.92</td></tr><tr><td rowspan="8">4</td><td>WS</td><td>128</td><td>131k</td><td>57.6</td><td>32.3</td><td>49.3</td><td>45.2</td><td>56.0</td><td>69.4</td><td>31.8</td><td>48.79</td></tr><tr><td>WS</td><td>256</td><td>131k</td><td>57.7</td><td>32.3</td><td>49.4</td><td>45.2</td><td>55.6</td><td>69.6</td><td>32.0</td><td>48.82</td></tr><tr><td>WS</td><td>1024</td><td>131k</td><td>57.4</td><td>31.8</td><td>49.5</td><td>45.5</td><td>55.7</td><td>69.2</td><td>32.0</td><td>48.74</td></tr><tr><td>WS</td><td>1024</td><td>1.05M</td><td>58.1</td><td>32.1</td><td>49.4</td><td>45.6</td><td>55.9</td><td>68.9</td><td>32.0</td><td>48.85</td></tr><tr><td>SW (ours)</td><td>128</td><td>131k</td><td>57.4</td><td>32.6</td><td>49.2</td><td>46.0</td><td>56.4</td><td>68.8</td><td>31.6</td><td>48.86</td></tr><tr><td>SW (ours)</td><td>256</td><td>131k</td><td>56.8</td><td>32.3</td><td>49.3</td><td>46.1</td><td>55.8</td><td>69.0</td><td>31.6</td><td>48.70</td></tr><tr><td>SW (ours)</td><td>1024</td><td>131k</td><td>57.3</td><td>31.7</td><td>49.2</td><td>46.3</td><td>56.2</td><td>69.3</td><td>31.8</td><td>48.83</td></tr><tr><td>SW (ours)</td><td>1024</td><td>1.05M</td><td>57.0</td><td>31.6</td><td>49.2</td><td>46.1</td><td>56.0</td><td>69.3</td><td>32.2</td><td>48.77</td></tr></table>

Table 12: Zero-shot accuracy (higher better) of Llama-3.2-1B-Instruct with a 3-bit GuidedQuant body and a tied head quantized at 2, 3 and 4 bits. Details as in Table 11. Best value per head group in bold.
<table><tr><td>Head</td><td>Method</td><td>ARC-e</td><td>ARC-c</td><td>HellaS</td><td>LAMB</td><td>Wino</td><td>PIQA</td><td>OBQA</td><td>Mean</td></tr><tr><td>16 16</td><td>fp16 body fp16 head</td><td>63.8 63.0</td><td>37.8 36.3</td><td>61.7 58.3</td><td>60.1 59.5</td><td>61.6 61.7</td><td>74.9 73.1</td><td>37.2 35.6</td><td>56.72 55.35</td></tr><tr><td>2</td><td>WS-c128 WS-c256 WS-c1024 WS-1M SW-c128 SW-c256 SW-c1024</td><td>59.7 59.6 57.9 59.4 62.8 62.5 61.0</td><td>34.5 34.6 32.8 34.0 36.6 36.6 35.4</td><td>57.0 56.8 57.2 56.8 57.2 57.4 57.5</td><td>50.7 51.6 53.1 52.3 58.5 58.5 58.4</td><td>60.5 59.5 60.5 59.7 60.2 60.2 60.2</td><td>72.2 71.4 71.8 71.7 73.0 72.3 72.3</td><td>35.4 37.0 35.6 36.6 35.2 36.4</td><td>52.84 52.94 52.68 52.93 54.78 54.85</td></tr><tr><td>3</td><td>SW-1M WS-c128 WS-c256 WS-c1024 WS-1M SW-c128 SW-c256 SW-c1024 SW-1M</td><td>61.3 63.0 63.1 61.8 62.1 62.1 62.5 62.1 62.2</td><td>36.7 36.4 37.0 36.8 36.5 35.2 35.8 36.2 36.2</td><td>57.3 57.7 57.9 58.0 58.0 58.3 58.3 58.0 58.1</td><td>58.7 58.3 57.1 57.9 58.5 59.3 59.3 59.0 59.3</td><td>61.1 62.4 61.3 61.1 61.1 61.4 60.9 61.2 60.6</td><td>72.6 73.4 73.3 73.0 72.9 72.7 72.7 73.1 73.1</td><td>36.2 36.4 36.0 35.8 35.4 35.6 37.2 36.8 35.8</td><td>54.43 54.87 55.31 55.08 54.84 54.95 55.18 55.20 55.07</td></tr><tr><td>4</td><td>WS-c128 WS-c256 WS-c1024 WS-1M SW-c128 SW-c256 SW-c1024 SW-1M</td><td>62.7 62.8 63.1 63.0 63.1 62.7 63.1 62.8</td><td>36.5 36.3 36.1 36.6 36.0 36.6 36.0 36.7</td><td>58.3 58.2 58.2 58.1 58.2 58.3 58.2 58.2</td><td>58.8 59.1 59.0 58.8 59.6 59.3 59.7 59.6</td><td>60.9 61.8 61.6 60.9 61.1 60.5 60.9 61.1</td><td>73.4 73.4 73.4 72.9 72.7 72.9 73.5 73.2</td><td>37.0 36.4 35.8 35.8 35.6 35.6 36.0 36.6</td><td>35.8 55.05 55.38 55.43 55.31 55.16 55.17 55.14 55.34 55.45</td></tr></table>

Table 13: Zero-shot accuracy (higher better) of Llama-3.2-1B-Instruct with a 4-bit GuidedQuant body and a tied head quantized at 2, 3 and 4 bits. Details as in Table 11. Best value per head group in bold.
<table><tr><td>Head</td><td>Method</td><td>ARC-e</td><td>ARC-c</td><td>HellaS</td><td>LAMB</td><td>Wino</td><td>PIQA</td><td>OBQA</td><td>Mean</td></tr><tr><td>16</td><td>fp16 body</td><td>63.8</td><td>37.8 37.7</td><td>61.7 60.3</td><td>60.1 60.1</td><td>61.6 62.0</td><td>74.9 73.6</td><td>37.2 37.8</td><td>56.72</td></tr><tr><td>16 2</td><td>fp16 head WS-c128 WS-c256</td><td>62.7 60.5 61.0</td><td>36.6 37.0 37.0</td><td>58.9 59.0</td><td>53.6 53.9</td><td>61.5 60.7</td><td>72.2 72.1</td><td>35.8 36.2 37.2</td><td>56.30 54.16 54.27</td></tr><tr><td>3</td><td>WS-c1024 WS-1M SW-c128 SW-c256 SW-c1024 SW-1M WS-c128 WS-c256 WS-c1024</td><td>60.3 59.5 62.2 63.2 61.2 62.5 62.2 62.7 62.5</td><td>35.5 37.4 38.7 37.5 37.8 38.5 38.6 38.2</td><td>59.2 59.5 59.6 59.5 59.3 59.7 60.4 60.2 60.2</td><td>54.0 55.0 59.4 59.2 59.2 59.9 58.7 58.6</td><td>60.9 60.1 61.8 61.3 61.6 62.0 60.9 60.3</td><td>72.1 72.5 73.2 73.3 72.7 73.6 73.8 73.4</td><td>36.8 36.4 35.4 35.0 36.2 37.4 36.4</td><td>54.39 54.12 55.72 55.81 55.23 55.96 55.98 55.74</td></tr><tr><td>4</td><td>WS-1M SW-c128 SW-c256 SW-c1024 SW-1M WS-c128 WS-c256 WS-c1024</td><td>62.7 62.2 62.5 62.8 62.8 63.2 62.6 63.1</td><td>37.9 37.9 36.9 37.5 38.1 37.0 37.5 37.8</td><td>60.3 60.3 60.2 60.2 60.1 60.3 60.4 60.3</td><td>58.2 58.4 59.5 60.0 60.2 60.1 59.5 59.4</td><td>60.5 60.5 62.0 61.8 61.1 62.0 62.0 62.0</td><td>73.1 73.8 73.5 73.4 73.3 73.6 73.7 73.6</td><td>36.4 37.0 36.2 36.6 37.6 36.6 37.8 38.0</td><td>55.60 55.79 55.94 55.93 56.10 56.18 56.23</td></tr><tr><td></td><td>WS-1M SW-c128 SW-c256 SW-c1024 SW-1M</td><td>63.2 62.5 62.5 62.9 62.4</td><td>37.7 37.8 38.0 37.5 37.8</td><td>60.3 60.2 60.2 60.3 60.3</td><td>59.7 59.5 59.8 59.9 60.3 60.2</td><td>62.0 61.5 62.5 62.4 62.2 61.9</td><td>74.2 73.6 73.4 74.0 73.6 73.7</td><td>37.8 38.4 37.6 37.4 36.8 37.8</td><td>56.22 56.41 56.30 56.27 56.35 56.22 56.29</td></tr></table>

## A.5 CLASS-SIDE STATISTIC ABLATION

Table 14: Class-side statistic ablation, head-only isolation: the body is kept in fp16 and only the head matrix is quantized; tied models (Llama-3.2-1B, Qwen3-1.7B) retain an fp16 copy of the embedding. All arms share the head, the damped Σ Cholesky, the rate-matching routine and the $\epsilon = 0 . 1$ smoothing prior, so they differ only in the class-side statistic: $\bar { p } = \mathbb { E } _ { X } [ \tilde { p } _ { k } ]$ is the model’s average marginal; π is the empirical unigram frequency of token k over the calibration window; SW uses $\tilde { \lambda } _ { k } = \mathbb { E } _ { X } [ \tilde { p } _ { k } ( 1 - \tilde { p } _ { k } ) ]$ . KL is measured against the all-fp16 model on identical token streams. Every arm lands within 0.001 bits of its nominal head rate. Parentheses after each model name give its vocabulary size $V ;$ on PPL cells, the increase over the fp16 row of the same model. Best value per column per head group in bold; ties bold both.

<table><tr><td>Statistic Head</td><td>WT2 PPL (↓)</td><td></td><td>WT2 KL (↓)</td><td>C4 PPL (↓)</td><td>C4 KL (↓)</td></tr><tr><td colspan="6">Llama-3.2-1B (128,256)</td></tr><tr><td>fp16</td><td>16 11.313</td><td></td><td>0</td><td>15.614</td><td>0</td></tr><tr><td>p</td><td>222</td><td>11.418 (+0.9%)</td><td>0.00952</td><td>16.102 (+3.1%)</td><td>0.02959</td></tr><tr><td>π</td><td></td><td>11.524 (+1.9%)</td><td>0.02146</td><td>16.549 (+6.0%)</td><td>0.05830</td></tr><tr><td>SW (ours)</td><td></td><td>11.419 (+0.9%)</td><td>0.00957</td><td>16.035 (+2.7%)</td><td>0.02834</td></tr><tr><td>p</td><td></td><td>11.344 (+0.3%)</td><td>0.00213</td><td>15.719 (+0.7%)</td><td>0.00613</td></tr><tr><td>π</td><td>333</td><td>11.356 (+0.4%)</td><td>0.00437</td><td>15.784 (+1.1%)</td><td>0.01184</td></tr><tr><td>SW (ours)</td><td></td><td>11.342 (+0.3%)</td><td>0.00206</td><td>15.704 (+0.6%)</td><td>0.00572</td></tr><tr><td>p</td><td>4</td><td>11.319 (+0.1%)</td><td>0.00049</td><td>15.624 (+0.1%)</td><td>0.00147</td></tr><tr><td>π</td><td>4</td><td>11.327 (+0.1%)</td><td>0.00105</td><td>15.658 (+0.3%)</td><td>0.00280</td></tr><tr><td>SW (ours)</td><td>4</td><td>11.320 (+0.1%)</td><td>0.00047</td><td>15.640 (+0.2%)</td><td>0.00138</td></tr><tr><td colspan="6">Qwen3-1.7B (151,936)</td></tr><tr><td>fp16</td><td>16</td><td>18.285</td><td>0</td><td>22.621</td><td>0</td></tr><tr><td></td><td></td><td>18.452 (+0.9%)</td><td>0.00951</td><td>23.138 (+2.3%)</td><td></td></tr><tr><td>p π</td><td>222</td><td>18.595 (+1.7%)</td><td>0.01912</td><td>23.494 (+3.9%)</td><td>0.02110 0.04421</td></tr><tr><td>SW (ours)</td><td></td><td>18.460 (+1.0%)</td><td>0.00900</td><td>23.075 (+2.0%)</td><td>0.01998</td></tr><tr><td>p</td><td></td><td>18.310 (+0.1%)</td><td>0.00206</td><td>22.756 (+0.6%)</td><td>0.00468</td></tr><tr><td>π</td><td>333</td><td>18.360 (+0.4%)</td><td>0.00389</td><td>22.823 (+0.9%)</td><td>0.00870</td></tr><tr><td>SW (ours)</td><td></td><td>18.317 (+0.2%)</td><td>0.00199</td><td>22.724 (+0.5%)</td><td>0.00443</td></tr><tr><td>p</td><td>4</td><td>18.303 (+0.1%)</td><td>0.00046</td><td>22.636 (+0.1%)</td><td>0.00112</td></tr><tr><td>π</td><td>4</td><td>18.304 (+0.1%)</td><td>0.00091</td><td>22.669 (+0.2%)</td><td>0.00203</td></tr><tr><td>SW (ours)</td><td>4</td><td>18.298 (+0.1%)</td><td>0.00045</td><td>22.633 (+0.1%)</td><td>0.00105</td></tr><tr><td colspan="6">Llama-3.1-8B-Instruct (128,256)</td></tr><tr><td>fp16</td><td>16</td><td>8.388</td><td>0</td><td>14.189</td><td>0</td></tr><tr><td>p</td><td></td><td>8.441 (+0.6%)</td><td>0.00658</td><td>14.530 (+2.4%)</td><td>0.02237</td></tr><tr><td>π</td><td>222</td><td>8.501 (+1.3%)</td><td>0.01356</td><td>14.822 (+4.5%)</td><td>0.04139</td></tr><tr><td>SW (ours)</td><td></td><td>8.437 (+0.6%)</td><td>0.00636</td><td>14.534 (+2.4%)</td><td>0.02140</td></tr><tr><td>p</td><td>3</td><td>8.398 (+0.1%)</td><td>0.00140</td><td>14.252 (+0.4%)</td><td>0.00465</td></tr><tr><td>π SW (ours)</td><td>3 3</td><td>8.412 (+0.3%) 8.398 (+0.1%)</td><td>0.00274</td><td>14.293 (+0.7%)</td><td>0.00852</td></tr><tr><td></td><td></td><td></td><td>0.00135</td><td>14.258 (+0.5%)</td><td>0.00437</td></tr><tr><td>p</td><td>4</td><td>8.390 (+0.0%)</td><td>0.00033</td><td>14.207 (+0.1%)</td><td>0.00113</td></tr><tr><td>π</td><td>4</td><td>8.396 (+0.1%)</td><td>0.00066</td><td>14.225 (+0.3%)</td><td>0.00200</td></tr><tr><td>SW (ours)</td><td>4</td><td>8.391 (+0.0%)</td><td>0.00032</td><td>14.205 (+0.1%)</td><td>0.00103</td></tr><tr><td colspan="6">Qwen3-8B (151,936)</td></tr><tr><td>fp16</td><td>16</td><td>10.832</td><td>0</td><td>15.486</td><td>0</td></tr><tr><td>p</td><td></td><td>10.915 (+0.8%)</td><td>0.00643</td><td>15.699 (+1.4%)</td><td>0.01484</td></tr><tr><td>π</td><td>222</td><td>10.965 (+1.2%)</td><td>0.01183</td><td>15.899 (+2.7%)</td><td>0.02919</td></tr><tr><td>SW (ours)</td><td></td><td>10.898 (+0.6%)</td><td>0.00654</td><td>15.689 (+1.3%)</td><td>0.01385</td></tr><tr><td>p</td><td>3</td><td>10.845 (+0.1%)</td><td>0.00129</td><td>15.544 (+0.4%)</td><td>0.00328</td></tr><tr><td>π</td><td>3</td><td>10.853 (+0.2%)</td><td>0.00251</td><td>15.577 (+0.6%)</td><td>0.00585</td></tr><tr><td>SW (ours)</td><td>3</td><td>10.848 (+0.1%)</td><td>0.00132</td><td>15.523 (+0.2%)</td><td>0.00295</td></tr><tr><td>p</td><td>4</td><td>10.835 (+0.0%)</td><td>0.00032</td><td>15.495 (+0.1%)</td><td>0.00075</td></tr><tr><td>π</td><td>4</td><td>10.839 (+0.1%)</td><td>0.00059</td><td>15.504 (+0.1%)</td><td>0.00132</td></tr><tr><td>SW (ours)</td><td>4</td><td>10.836 (+0.0%)</td><td>0.00031</td><td>15.501 (+0.1%)</td><td>0.00073</td></tr><tr><td colspan="6">Qwen3-32B (151,936)</td></tr><tr><td>fp16</td><td>16 8.452</td><td></td><td>0</td><td>12.269</td><td>0</td></tr><tr><td>p</td><td></td><td>8.516 (+0.8%)</td><td>0.00709</td><td>12.475 (+1.7%)</td><td>0.01663</td></tr><tr><td>π</td><td>222</td><td>8.571 (+1.4%)</td><td>0.01461</td><td>12.703 (+3.5%)</td><td>0.03348</td></tr><tr><td>SW (ours)</td><td></td><td>8.498 (+0.5%)</td><td>0.00703</td><td>12.479 (+1.7%)</td><td>0.01596</td></tr><tr><td>p</td><td>3</td><td>8.468 (+0.2%)</td><td>0.00153</td><td>12.303 (+0.3%)</td><td>0.00347</td></tr><tr><td>π</td><td>3</td><td>8.476 (+0.3%)</td><td>0.00297</td><td>12.356 (+0.7%)</td><td>0.00687</td></tr><tr><td>SW (ours)</td><td>3</td><td>8.464 (+0.1%)</td><td>0.00147</td><td>12.315 (+0.4%)</td><td>0.00341</td></tr><tr><td>p</td><td>4</td><td>8.457 (+0.1%)</td><td>0.00036</td><td>12.276 (+0.1%)</td><td>0.00087</td></tr><tr><td>π</td><td>4</td><td>8.461 (+0.1%)</td><td>0.00072</td><td>12.289 (+0.2%)</td><td>0.00167</td></tr><tr><td>SW (ours)</td><td>4</td><td>8.457 (+0.1%)</td><td>0.00035</td><td>12.275 (+0.0%)</td><td>0.00081</td></tr></table>

## B HESSIAN SKETCHING

This appendix details the power iteration scheme used in $\ S 5$ to compute optimized Kronecker factors of the head KL Hessian, adapted from the sketching subroutines of YAQA (Tseng et al., 2026).

Setup. We seek the best diagonal-Kronecker approximation

$$
a ^ { \star } , B ^ { \star } = \operatorname * { a r g m i n } _ { a \mathrm { d i a g o n a l } , B } \left\| \mathbb { E } _ { X } \left[ \tilde { \lambda } ( X ) \otimes X X ^ { \top } \right] - a \otimes B \right\| _ { F } ^ { 2 } ,\tag{23}
$$

where $\tilde { \lambda } ( X ) = \mathrm { d i a g } ( \tilde { p } \odot ( 1 - \tilde { p } ) )$ is the softmax curvature under the smoothed output distribution $\tilde { p }$ of (21) with $\epsilon = 0 . 1$ . As noted by Tseng et al. (2026), following Van Loan (2000), the best Kronecker approximation is a rank-one approximation of a rearrangement of the target matrix, so alternating least squares (ALS) on the two factors is power iteration on that rearrangement and converges to the leading singular pair, which is unique up to the scale ambiguity $( c a , B \bar { / } c )$ whenever the leading singular value is simple. Since all our comparisons (cosine similarity to the fixed point, and the quantization lattice itself, through the coupling of $c _ { \mathrm { S W } } )$ are invariant to this rescaling, we do not normalize the factors during the iteration.

Adapted updates. YAQA’s Sketch A estimates the per-token Hessian with a Monte-Carlo label sampled from the model’s predictive distribution and a modified backward pass. For the head layer both approximations are unnecessary: the layer’s KL Hessian at a hidden state X is available in closed form as $( \mathrm { d i a g } ( \tilde { p } ) - \tilde { p } \tilde { p } ^ { \top } ) \otimes X \dot { X } ^ { \top }$ , requiring only a forward pass, and the token-independence bias of Sketch A does not arise because no computation mixes tokens after the head. We therefore replace the sampled label with the exact per-token curvature, which preserves the expectation of their estimator while removing its sampling variance. Moreover, since a is constrained to be diagonal, the off-diagonal coupling $- \tilde { p } \tilde { p } ^ { \top }$ is Frobenius-orthogonal to the ansatz $a \otimes B$ , and the diagonal of the exact class-side Hessian is precisely $\tilde { \lambda } = \tilde { p } \odot ( 1 - \tilde { p } )$ ; hence each update below is the exact ALS step for (23), not an approximation of it. Writing $s ( \bar { X } ) = \tilde { p } \odot ( 1 - \bar { \tilde { p } } ) \in \mathbb { R } ^ { K }$ , one round consists of a class-side pass followed by a feature-side pass over the calibration set, mirroring the alternation order of the official YAQA release:

$$
a \gets \frac { \mathbb { E } _ { X } \left[ \left( X ^ { \top } B X \right) s ( X ) \right] } { \| B \| _ { F } ^ { 2 } } , \qquad B \gets \frac { \mathbb { E } _ { X } \left[ \left. a , s ( X ) \right. X X ^ { \top } \right] } { \| a \| _ { 2 } ^ { 2 } } ,\tag{24}
$$

where each update uses the other factor’s most recent value. Each pass costs one matrix-vector product with $B$ (or one rank-one accumulation into B) plus the head forward pass per token, and requires no backpropagation, in contrast to the general-layer setting of Tseng et al. (2026).

Protocol. We cache the calibration hidden states once (WT2-train) and run all initializations of Table 6b on the identical cached states, so that differences reflect the initialization alone. We run 8 rounds of (24); the fixed point $( a ^ { \star } , B ^ { \star } )$ is taken as the final iterate, and convergence is monitored by the cosine similarity of consecutive iterates, which exceeds 0.999999 for every initialization from round 3 onward. Note that the SoftWater factors $( \tilde { \lambda } , \Sigma _ { X } )$ are not merely a good initialization: round 0 of Table 6b shows they already agree with the fixed point to cosine similarity 0.976–0.992, quantifying how close the analytical separable factorization is to the Frobenius-optimal one.