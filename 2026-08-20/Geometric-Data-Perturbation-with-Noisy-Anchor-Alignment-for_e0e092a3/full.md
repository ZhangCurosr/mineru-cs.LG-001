# Geometric Data Perturbation with Noisy-Anchor Alignment for Privacy-Preserving Collaborative Learning

Keiyu Nosaka<sup>a</sup>, Yamato Suetake<sup>a</sup>, Yuichi Takano<sup>b,c</sup>, Yukihiko Okada<sup>b,c</sup>, Akiko Yoshise<sup>b,c,∗</sup>

<sup>a</sup>Graduate School ofScience and Technology, University ofTsukuba, 1-1-1, Tennodai, Tsukuba, 305-8573, Ibaraki, Japan <sup>b</sup>Institute of Systems and Information Engineering, University of Tsukuba, 1-1-1, Tennodai, Tsukuba, 305-8573, Ibaraki, Japan <sup>c</sup>Center for Artificial Intelligence Research, Tsukuba Institute for Advanced Research (TIAR), University of Tsukuba, 1-1-1, Tennodai, Tsukuba, 305-8577, Ibaraki, Japan

## Abstract

Geometric Data Perturbation (GDP) enables one-shot, privacy-preserving collaborative learning: each participant applies a distance-preserving transformation to its private data and uploads only the resulting representation to a central analyst. We study GDP under analyst–participant collusion, in which the analyst combines all uploaded rep resentations with the private data and transformations disclosed by colluding participants to recover a non-colluding participant’s private data. Participant-specific independent transformations resist this attack but map participants’ data into incompatible representation spaces, degrading downstream model performance. Shared-anchor alignment from Data Collaboration (DC) analysis restores compatibility and improves utility, but we show that disclosing the DC anchor matrix enables exact recovery of non-colluding participants’ private data even in the presence of collusion. Adding noise directly to the private-data representations mitigates this vulnerability but substantially reduces utility. We therefore propose adding noise to the anchor representations instead. Each participant independently transforms its private data and the shared anchor matrix, perturbs only the resulting anchor representation, and uploads both representations in a single round. Using the noisy anchor representations, the analyst aligns the private-data representations by solving a Generalized Orthogonal Procrustes Problem. We characterize alignment and recovery errors, specialize a conservative suficient condition for convergence of the alignment to our setting, and analyze three recovery attacks. Experiments on MNIST and CelebA show that, across the evaluated attacks and deployment settings, anchor noise achieves higher learning accuracy than private-data noise at comparable measured leakage, yielding a more favorable privacy–utility trade-of under the specified collusion model.

Keywords: One-Shot Representation Sharing, Geometric Data Perturbation, Data Collaboration Analysis, Anchor Alignment, Analyst-Participant Collusion, Generalized Orthogonal Procrustes Problem, Privacy-Utility Trade-of

## 1. Introduction

Pooling data across organizations can substantially improve predictive performance and robustness, especially when models must generalize across heterogeneous populations. Yet in many domains, raw data cannot simply be shared or centralized due to legal, contractual, operational, and governance constraints [1, 2, 3, 4, 5]. Privacypreserving collaborative learning (PPCL) ofers one response to this barrier: it enables organizations to participate in joint model development while constraining what other organizations, or a central analyst, can learn about their records [6, 7, 8].

We focus on a PPCL regime that we call one-shot representation sharing (OSRS). In this regime, each participant (the data-holding organization) uploads an obfuscated data matrix—a locally computed intermediate representation of its records—to a central analyst, who then trains downstream models using the uploaded representations. This upload-once, train-centrally workflow is useful when multi-round coordination is impractical, for example, because participants have limited compute resources, connectivity is unreliable, or deployment environments impose strict communication constraints [9, 10]. Whereas standard iterative federated learning algorithms repeatedly exchange and aggregate model updates, such as parameters or gradients [11, 12, 13], OSRS communicates the representation itself and requires no subsequent optimization rounds with the participants.

A central design question in one-shot representation sharing is how to obfuscate the uploaded representation so that it remains useful for downstream learning while limiting the risks of reconstruction or re-identification. A standard route to formal privacy guarantees is diferential privacy (DP), often instantiated by adding calibrated noise to the released representation [14, 15, 16]. DP provides principled worst-case protection, and additive perturbation has long been used as a practical privacy mechanism in distributed analysis settings [15, 16, 17]. However, in high-dimensional regimes, the noise required for strong DP guarantees can substantially distort the geometric and statistical structure that downstream models exploit, leading to pronounced privacy-utility trade-ofs [18, 19].

This limitation motivates the study of alternative perturbation families that can better preserve utility under explicit threat models, while recognizing that their privacy guarantees are typically heuristic and adversary-dependent. Among these, geometric data perturbation (GDP), which applies distance-preserving transformations, is particularly attractive because it can preserve statistical and geometric structure relevant to learning while remaining computationally lightweight [20, 21]. Their vulnerabilities, as well as corresponding countermeasures, have been studied extensively in privacy-preserving data mining [22, 23]. However, these analyses do not directly address the PPCL setting considered here, where multiple participants independently hold private datasets and upload perturbed representations to a central analyst for joint downstream training. We focus on the analyst-participant collusion setting: the central analyst combines all uploaded representations with the private data and transformations disclosed by some participant to infer information about another participant’s private data.

The consequences of collusion depend critically on how transformation parameters are shared across participants. We consider two regimes. In common-transform GDP (C-GDP), all participants use the same transform, whereas in independent-transform GDP (I-GDP), each participant uses a separately generated transform [24]. Under C-GDP, a colluding participant can disclose the common transform to the analyst, enabling exact inversion of the other participants’ noiseless representations. Under I-GDP, disclosure of one participant’s transform does not directly expose the transforms of the other participants.

This isolation comes at a practical cost. Although each transformation preserves within-participant geometry, representations produced under diferent transforms do not share a common coordinate system. Directly pooling them can therefore create incompatible feature-label relationships and degrade conventional downstream learning. This leads to the central design problem addressed in this paper: how can OSRS restore cross-participant comparability without disclosing alignment information that enables exact recovery of participant-specific transforms under analyst participant collusion?

Data Collaboration (DC) analysis provides a natural candidate for restoring comparability. It introduces a synthetic anchor matrix whose rows serve as common reference records. The same raw anchor matrix is shared among all participants but is withheld from the analyst. Each participant applies its own transformation to both its private data and the anchor matrix and sends the resulting representations to the analyst. Because the anchor representations correspond to the same underlying reference records, the analyst can use them to map the participant-specific private representations into a shared coordinate system [25, 26, 27, 28]. DC-style alignment, therefore, appears to combine the isolation of I-GDP with conventional pooled learning.

We show, however, that noiseless anchor alignment turns the anchor representations into an exact transformrecovery channel. If a colluding participant discloses the raw anchor matrix to the analyst, the analyst can recover the non-colluding participants’ transformations from their anchor representations, provided the required rank and identifiability conditions hold. Because GDP transformations are invertible, recovering these transformations also enables exact inversion of the corresponding noiseless private representations.

A natural approach is to add isotropic Gaussian noise to the private data representations [29]. We prove that, under matched GDP parameters and the stated alignment assumptions, this modification does not create a new final-output regime. Specifically, the aligned private-data outputs of C-GDP with private-data noise [21] and DC-style anchor aligned I-GDP with private-data noise [29] are equivalent in distribution and can be coupled to coincide exactly. Consequently, noiseless anchor alignment with private-data noise reproduces the output utility—the noise behavior of its matched C-GDP counterpart—rather than resolving the underlying dilemma of transformation sharing and

alignment.

These findings suggest perturbing the alignment signal rather than the private-data signal. We therefore add isotropic Gaussian noise to the anchor representations while leaving the private-data representations unnoised. The resulting noisy multi-participant alignment can be formulated as a generalized orthogonal Procrustes problem (GOPP), allowing us to draw on established algorithms and theoretical results for GOPP estimation and convergence [30]. Anchor noise preserves the within-participant Euclidean geometry of each private-data upload while converting exact, known-anchor transformation recovery into a noisy estimation problem whose dificulty depends on the anchor matrix, the noise level, and the available observations. The mechanism is not intended to provide record-level DP; instead, it creates tunable, attack-dependent operating points between cross-participant alignment accuracy and resistance to the evaluated reconstruction attacks. On MNIST and CelebA datasets, we observe a more favorable measured privacyutility trade-of than with C-GDP using private-data noise and, by the preceding output-equivalence result, its matched anchor-aligned I-GDP counterpart under the attacks and experimental conditions evaluated in the analyst-participant collusion setting. Participant-count experiments further suggest that anchor alignment mitigates the degradation of utility caused by independently transformed representations as the number of participants increases [24]. Overall, anchor noise restores useful cross-participant comparability, retains participant-specific transformations, and replaces exact anchor-based inversion with noise-limited transformation estimation.

Figure 1 summarizes the proposed participant–analyst workflow.

![](images/d07d19028d0c43b9a725720ceb3ba8e987865cc91a05c7c5d6f8c215b9c16c45.jpg)  
Figure 1: Overview of AA-I-GDP (anchor noise) pipeline. Each participant applies a participant-specific GDP mechanism to its private-data matrix and the shared anchor matrix, adds noise only to the resulting anchor representation, and uploads both representations. From the noisy anchor representations, the analyst estimates participant-specific alignment maps, maps the private-data representations into a shared coordinate system, and trains a downstream model.

## 1.1. Our Contributions

We propose an anchor-alignment framework with anchor noise for one-shot representation sharing based on geometric perturbations. The framework addresses the central tension between common-transform geometric data perturbation (C-GDP), which preserves cross-participant comparability but is vulnerable to disclosure of the shared transformation, and independent-transform GDP (I-GDP), which limits this direct exposure but maps participant representations into incompatible coordinate systems [21, 24]. The central innovation is to apply additive perturbation to the anchor representations used for alignment rather than to the private-data representations. This design preserves the within-participant geometry of the private-data representations while replacing exact recovery of the known-anchor transformation with noisy estimation.

The proposed framework leads to four main contributions:

1. Known-Anchor Vulnerability of Noiseless Alignment: We analyze DC-style anchor alignment in the analystparticipant collusion setting [25, 26, 27]. We show that when a colluding participant reveals the raw anchor matrix and the required rank and identifiability conditions hold, the analyst can exactly recover each noncolluding participant’s GDP transformation from its uploaded anchor representation. When the transformation is invertible, and the private-data upload contains no additional noise, this recovery enables direct inversion of the corresponding private representation.

2. Equivalence of Private-Data-Noise Constructions: We establish that adding isotropic Gaussian noise to the private-data representations does not create a distinct final-output regime. Under matched GDP parameters and the stated alignment assumptions, the final aligned private-data outputs of C-GDP with private-data noise [21] and anchor-aligned I-GDP with private-data noise [29] are equivalent in distribution and can be coupled to coincide pathwise. This result concerns the aligned collaboration outputs rather than the complete protocol transcripts.

3. Alignment with Anchor Noise: To address this limitation, we move the isotropic Gaussian perturbation from the private-data representations to the anchor representations. We formulate the resulting multi-participant alignment problem as a Generalized Orthogonal Procrustes Problem (GOPP), connecting the proposed framework to established algorithms and theoretical results for GOPP estimation and convergence [30]. The result ing method retains participant-specific transformations, restores a common coordinate system, and leaves the private-data representations free of additive noise while converting exact known-anchor recovery into a noiselimited transformation-estimation problem.

4. Empirical Privacy-Utility Characterization in the Analyst-Participant Collusion Setting: We evaluate downstream utility and reconstruction leakage on MNIST and CelebA datasets in the analyst-participant collusion setting. Across both datasets, anchor noise yields a more favorable measured privacy-utility trade-of than private-data noise under the evaluated reconstruction attacks. For the CelebA dataset, the participant-count replay with $p \in \{ 2 , 5 , 1 0 , 2 0 , 5 0 \}$ shows that low anchor noise restores the natural utility ordering induced by the growing training set, whereas the unaligned I-GDP controls remain non-monotone in $p .$

## 1.2. Organization

Section 2 introduces the notation, the OSRS protocol, the analyst-participant collusion threat model, and the GDP and anchor-aligned baselines used throughout the paper. Section 3 identifies the alignment–recovery tension in these baselines by establishing exact known-anchor recovery for noiseless alignment and the aligned-output equivalence of the private-data-noise constructions under the stated conditions. Section 4 develops Anchor-Aligned Independent Ge ometric Data Perturbation with anchor noise, formulates alignment as a Generalized Orthogonal Procrustes Problem, and analyzes alignment error, solver behavior, and reconstruction risk. Section 5 evaluates the resulting privacy-utility trade-ofs on MNIST and CelebA datasets, including sensitivity to the number of participants. Section 6 concludes the paper by summarizing the theoretical and empirical findings, discussing their limitations, and outlining directions for future work. Proofs of the formal results are provided in the supplementary material.

## 2. Preliminaries

This section fixes the notation, system model, and threat model, and restates the comparison mechanisms in a common notation. GDP and its additive-noise extension follow the geometric-perturbation literature [20, 21]; independent participant-specific transforms follow I-GDP [24]; generic anchor-based collaboration follows DC [25, 26]; the noiseless anchor-aligned I-GDP protocol and its reference-participant alignment procedure follow ODC [27]; and private-data noise in DC follows [29].

## 2.1. Notations

For a positive integer k, we write $[ k ] : = \{ 1 , 2 , \ldots , k \}$ . We denote by $\mathbb { R } ^ { p \times q }$ the set of real $p \times q$ matrices. $\pmb { M } \in \mathbb { R } ^ { p \times q }$ means M is a real p×q matrix. $( M ) _ { k , l }$ denotes the row k, column l entry of the matrix M. $\mathbf { 1 } _ { m } \in \mathbb { R } ^ { m }$ is the all-ones vector. For a matrix M, $M ^ { \top }$ denotes the transpose, $M ^ { \dagger }$ the Moore–Penrose pseudoinverse [31] (e.g., $\pmb { M } ^ { \dag } = ( \pmb { M } ^ { \top } \pmb { M } ) ^ { - 1 } \pmb { M } ^ { \top }$ when M has full column rank), $| | \pmb { M } | | _ { F }$ the Frobenius norm, and $\operatorname { t r } ( M )$ the matrix trace. We denote the orthogonal group in $\mathbb { R } ^ { \ell }$ by $O ( \ell ) : = \{ O \in \mathbb { R } ^ { \ell \times \ell } : O ^ { \top } O = I _ { \ell } \}$ . For a matrix M, its (thin) singular value decomposition (SVD) is written as $\pmb { M } = \pmb { U } \pmb { \Sigma } \pmb { V } ^ { \top }$ . We write $\big [ \pmb { M } _ { i } \big ] _ { i \in [ k ] }$ for k vertically stacked matrices. Equivalently,

$$
\left[ \pmb { M } _ { i } \right] _ { i \in [ k ] } : = \left[ \begin{array} { l } { \pmb { M } _ { 1 } } \\ { \vdots } \\ { \pmb { M } _ { k } } \end{array} \right] .
$$

## 2.2. One-Shot Representation Sharing

We study the one-shot representation sharing (OSRS) regime in the PPCL setting. Assume a total of p participants and let $\mathcal { P } : = \ [ p ]$ denote the participant set. We assume a horizontal partition: all participants use the same ddimensional feature schema, and their n<sub>i</sub>-sample record sets are disjoint. For supervised tasks, participant i also holds a label vector $\mathbf { { \cal L } } _ { i } \in \mathbb { R } ^ { n _ { i } }$ . We suppress these labels from most notations and assume that they are analyst-visible and included in the same one-shot message; the reconstruction target studied here is the feature matrix $\pmb { X } _ { i } \in \mathbb { R } ^ { n _ { i } \times d }$

Definition 2.1 (One-Shot Representation Sharing). In the one-shot representation sharing regime, the global recordby-feature data matrix $\ b { X } \in \mathbb { R } ^ { n \times d }$ is partitioned by records across p participants:

$$
X = \left[ X _ { i } \right] _ { i \in \mathcal { P } } \qquad X _ { i } \in \mathbb { R } ^ { n _ { i } \times d } , \qquad \sum _ { i \in \mathcal { P } } n _ { i } = n .\tag{1}
$$

Participant i holds the private submatrix $X _ { i \cdot }$ For any batch size m, participant i applies a local row-wise obfuscation mechanism

$$
\mathcal { M } _ { i } : \mathbb { R } ^ { m \times d }  \mathbb { R } ^ { m \times h } ,
$$

which maps records to h-dimensional representations. Applying this mechanism to its local data yields

$$
\begin{array} { r } { \pmb { Y _ { i } } = \pmb { \mathcal { M } _ { i } } ( \pmb { X _ { i } } ) \in \mathbb { R } ^ { n _ { i } \times h } . } \end{array}\tag{2}
$$

Each participant communicates with the analyst exactly once by uploading $Y _ { i }$ and the target vector $\mathbf { } L _ { i } \in \mathbb { R } ^ { n _ { i } }$ . The analyst concatenates the uploaded representations as

$$
Y = [ Y _ { i } ] _ { i \in \mathcal { P } } \in \mathbb { R } ^ { n \times h } ,\tag{3}
$$

and uses Y for downstream analysis, such as model training.

We specifically consider the analyst-participant collusion setting, defined as follows.

Definition 2.2 (Analyst-Participant Collusion Setting). Consider a one-shot representation-sharing protocol with $p \ge 2$ participants and a central analyst. Let $c \in \mathcal { P }$ denote the single colluding participant. In the analyst-participant collusion setting, the attacker comprises the central analyst and participant c.

For each participant i, let $\Theta _ { i }$ denote all participant-side state other than the private dataset $X _ { i } ,$ including private randomness, perturbation parameters, secret keys, and other quantities used to instantiate the obfuscation mechanism. We assume that participant c discloses both $X _ { c }$ and $\Theta _ { c }$ in full. The analyst observes all uploaded representations $\{ Y _ { i } : i \in \mathcal { S } \}$ . Hence, the attacker’s view is

$$
( \{ Y _ { i } : i \in \mathcal { P } \} , ( X _ { c } , \Theta _ { c } ) ) .\tag{4}
$$

Unless stated otherwise, the attacker also knows the public protocol specification, including the functional forms of the obfuscation mechanisms and any public parameters. The attacker does not directly know the private datasets or local secret states ofthe non-colluding participants, except through information inferredfrom the analyst’s transcript andfrom participant c’s private dataset and local secret state.

The attacker’s objective is to recover, or infer sensitive information about, the private dataset $\pmb { X } _ { i } \in \mathbb { R } ^ { n _ { i } \times d }$ of a non-colluding target participant $i \in \mathcal { P } \ \backslash$ {c}.

## 2.3. Geometric Data Perturbation

Geometric Data Perturbation (GDP) is a classical approach to privacy-preserving data mining in which a data owner releases a geometrically transformed version of its dataset for analysis [20, 21]. We adopt the following established orthogonal-translation variant.

Definition 2.3 (Geometric Data Perturbation). For any batch size m, a Geometric Data Perturbation (GDP) mechanism is a row-wise obfuscation map $\boldsymbol { \mathcal { M } } : \mathbb { R } ^ { m \times d }  \mathbb { R } ^ { m \times d } o f$ the form

$$
\ b { M } ( \ b { S } ) = \ b { S O } + \mathbf { 1 } _ { m } \ b { \Psi } ^ { \top } , \qquad \ b { S } \in \mathbb { R } ^ { m \times d } .\tag{5}
$$

Here, $\pmb { O } \in O ( d )$ is an orthogonal matrix, $\Psi \in \mathbb { R } ^ { d }$ is a translation vector, and $\mathbf { 1 } _ { m } \in \mathbb { R } ^ { m }$ is the all-ones vector. Thus, the same orthogonal transformation and translation are applied to every record.

The orthogonal term preserves pairwise Euclidean distances and hence the geometric structure used by downstream analyses. Although unnecessary for this preservation, the translation term obscures the fixed origin, and thus the weakly perturbed region around it, when combined with an unknown orthogonal transformation; it provides no protection on its own [21].

Classical GDP assumes a single data owner and therefore does not specify how to coordinate perturbation parameters in a multi-participant PPCL protocol. The most direct baseline is for all participants to use the same GDP parameters. We refer to this regime as Common-Transform Geometric Data Perturbation.

Definition 2.4 (Common-Transform Geometric Data Perturbation). Common-Transform Geometric Data Perturbation $( C { \cdot } G D P )$ is a one-shot representation-sharing regime in which all participants use a shared GDP mechanism $\boldsymbol { \mathcal { M } } : \mathbb { R } ^ { m \times d }  \mathbb { R } ^ { m \times d }$ with common perturbation parameters (O Ψ)for any batch size m. Thus, each participant $i \in \mathcal { P }$ holding private data $\pmb { X } _ { i } \in \mathbb { R } ^ { n _ { i } \times d }$ , uploads

$$
\begin{array} { r } { Y _ { i } = { M } ( X _ { i } ) = { X } _ { i } { O } + \mathbf { 1 } _ { n _ { i } } \Psi ^ { \top } , } \end{array}\tag{6}
$$

where $\pmb { O } \in O ( d )$ is shared across participants and $\Psi \in \mathbb { R } ^ { d }$ is the shared translation vector.

Classical analyses of GDP include known-input, known-sample, and Independent Component Analysis (ICA)- based attacks [21, 23]. We focus on the multi-participant threat model of Definition 2.2. Disclosure of shared C-GDP parameters creates an immediate exact inversion channel:

Proposition 2.5 (Secret-State Vulnerability of C-GDP in the Analyst-Participant Collusion Setting). Suppose that the uploaded representations are generated by C-GDP:

$$
\pmb { Y } _ { i } = \pmb { X } _ { i } \pmb { O } + \mathbf { 1 } _ { n _ { i } } \Psi ^ { \top } , \qquad i \in \mathcal { P } ,
$$

where $\pmb { O } \in O ( d )$ and $\Psi \in \mathbb { R } ^ { d } .$ . Let $c \in \mathcal { P }$ be the colluding participant in the analyst-participant collusion setting. Ifthe disclosed state $\Theta _ { c }$ contains $( o , \Psi )$ , then the attacker reconstructs every non-colluding participant’s private dataset exactly as

$$
X _ { i } = { \Big ( } Y _ { i } - \mathbf { 1 } _ { n _ { i } } \Psi ^ { \top } { \Big ) } O ^ { \top } , \qquad i \in \mathcal { P } \setminus \{ c \} .
$$

Proofs. Proofs of all propositions, corollaries, lemmas, and theorems stated in the main manuscript are provided in the Supplementary Material.

The analyst-participant collusion setting is closely related to known-input attacks in the classical single-data owner data-mining setting, where the attacker observes a subset of original records together with their corresponding perturbed records. A simple countermeasure studied in the GDP literature is to superimpose random noise on the geometric perturbation [21]. In our multi-participant setting, this idea yields a regime in which all participants share the same geometric transformation parameters but add independent, participant-specific noise to their transformed records before uploading the resulting representations. We refer to this regime as C-GDP (private-data noise).

Definition 2.6 (C-GDP (private-data noise)). C-GDP (private-data noise) uses the common GDP mechanism M with sharedparameters $( o , \Psi )$ and adds participant-specific additive noise to each private-data representation. Participant $i \in \mathcal { P }$ samples a noise matrix $\pmb { W } _ { i } \in \mathbb { R } ^ { n _ { i } \times d }$ from a chosen base distribution and uploads

$$
\begin{array} { r } { Y _ { i } = \mathcal { M } ( X _ { i } ) + \sigma W _ { i } = X _ { i } O + \mathbf { 1 } _ { n _ { i } } \Psi ^ { \top } + \sigma W _ { i } . } \end{array}\tag{7}
$$

The matrices $W _ { 1 } , \ldots , W _ { p }$ are sampled independently across participants and independently of the common GDP parameters, and $\sigma \geq 0$ is the private-data-noise scale. Setting $\sigma = 0$ recovers C-GDP.

Under Definition 2.2, a colluder reveals the common parameters. For any non-colluding target $i \in \mathcal { S } \setminus \{ c \}$ , direct inversion gives

$$
( { \pmb Y } _ { i } - { \pmb 1 } _ { n _ { i } } { \boldsymbol \Psi } ^ { \top } ) { \pmb O } ^ { \top } = { \pmb X } _ { i } + \sigma { \pmb W } _ { i } { \pmb O } ^ { \top } ,
$$

so the additive noise, rather than the secrecy of the common transform, is the protection in this attack channel.

Common choices for additive perturbation include Laplace and Gaussian noise [14, 18]. In this study, the entries of each $W _ { i }$ are i.i.d. $N ( 0 , 1 )$ , so $\sigma$ is the standard deviation of each added noise entry. When a formal diferential privacy (DP) baseline is required, σ can be calibrated according to the sensitivity of the released representation, the neighboring relation, and the target privacy parameters [16, 17]. Gaussian perturbation alone should not be interpreted as a formal DP guarantee: DP additionally requires a specified neighboring relation, bounded sensitivity (typically enforced by clipping), and calibration to explicit privacy parameters [18].

Formal DP provides principled worst-case protection, while additive noise has long been used as a practical privacy mechanism in distributed analysis settings [15, 16, 17]. In C-GDP (private-data noise), additive noise can obscure the paired raw- perturbed observations available to a colluding analyst, thereby reducing the accuracy of transform recovery. However, this protection is obtained by directly distorting the uploaded representations. In highdimensional regimes, the noise required for strong DP guarantees can substantially disrupt the geometric and statistical structure exploited by downstream learners, leading to pronounced privacy–utility trade-ofs [18, 19]. This motivates alternative perturbation families that more efectively preserve useful geometric structure in the analyst-participant collusion setting, while recognizing that their privacy guarantees are typically heuristic and adversary-dependent.

Motivated by the vulnerability of C-GDP in the analyst-participant collusion setting and the utility cost of addi tive noise, another natural alternative is to assign each participant its own geometric perturbation sampled indepen dently [6]. We refer to this regime as Independent-Transform Geometric Data Perturbation.

Definition 2.7 (Independent-Transform Geometric Data Perturbation). Independent-Transform Geometric Data Perturbation (I-GDP) is a one-shot representation-sharing regime in which each participant uses a participant-specific GDP mechanism $\mathcal { M } _ { i } : \mathbb { R } ^ { m \times d }  \mathbb { R } ^ { m \times d }$ with independently sampled perturbation parameters $( O _ { i } , \Psi _ { i } )$ for any batch size m. Specifically,for each participant $i \in \mathcal { P }$ holding private data $\pmb { X } _ { i } \in \mathbb { R } ^ { n _ { i } \times d }$ , the uploaded representation is

$$
\pmb { Y } _ { i } = \pmb { M } _ { i } ( \pmb { X } _ { i } ) = \pmb { X } _ { i } \pmb { O } _ { i } + \pmb { 1 } _ { n _ { i } } \pmb { \Psi } _ { i } ^ { \top } ,\tag{8}
$$

where $\mathbf { \xi } _ { O _ { i } } \in O ( d )$ is a participant-specific orthogonal transformation and $\Psi _ { i } \in \mathbb { R } ^ { d }$ is a participant-specific translation vector. The parameter pairs $( \pmb { O } _ { 1 } , \Psi _ { 1 } ) , \dots , ( \pmb { O } _ { p } , \Psi _ { p } )$ are sampled independently across participants.

I-GDP removes the shared-parameter attack channel: for $i \neq j ,$ knowledge of participant i’s transformation parameters does not directly enable inversion of participant $j ^ { \circ } \mathbf { s }$ upload. Its limitation is therefore cross-participant comparability rather than within-participant geometry. Each transformation preserves the Euclidean geometry of its participant’s data up to a rigid transformation, but the concatenated participant blocks lie in unrelated coordinate systems. Conventional learners, which assume that each feature coordinate has a consistent meaning across samples, may therefore lose accuracy. Expressive neural networks may partially adapt to this mismatch [6], but they do not establish an explicit shared representation and may require additional model capacity.

## 2.4. Data Collaboration Style Anchor-Alignment

Data Collaboration (DC) analysis is an anchor-based extension of one-shot representation sharing [10, 25, 27, 32]. In DC, participants share a synthetic anchor matrix (often randomly generated) and upload only its locally transformed representations to the analyst.

Definition 2.8 (DC anchor setting). Participant $i \in \mathcal { P }$ holds $\pmb { X } _ { i } \in \mathbb { R } ^ { n _ { i } \times d }$ , and all participants have access to a common anchor matrix $\pmb { A } \in \mathbb { R } ^ { r \times d }$ . The anchor matrix is a participant-side shared state: it is neither included in the honest protocol transcript nor revealed to the analyst. For any batch size m, participant i has a local obfuscation map

$$
\mathcal { M } _ { i } : \mathbb { R } ^ { m \times d }  \mathbb { R } ^ { m \times h } .
$$

Participant i applies the same map to its private-data matrix and the shared anchor matrix and uploads

$$
Y _ { i } = { \mathcal { M } } _ { i } ( X _ { i } ) , \qquad B _ { i } = { \mathcal { M } } _ { i } ( A ) .
$$

For anchor-based protocols, the analyst-participant collusion view in Definition 2.2 must also specify the status of the participant-side shared anchor matrix. We use the following extension throughout the analysis.

Definition 2.9 (Analyst-Participant Collusion Setting under DC). Consider a DC protocol with participant set $\mathcal { P }$ and a central analyst. Let $c \in \mathcal { P }$ denote the colluding participant. In the analyst-participant collusion setting under DC, the attacker comprises the central analyst and participant c.

Algorithm 1: Generic Data Collaboration Analysis   
Input: Private data $\overline { { \{ X _ { i } \} _ { i \in \mathcal { P } } } } ;$ participant-side shared anchor matrix A; participant-specific maps $\overline { { \{ \mathcal { M } _ { i } \} _ { i \in \mathcal { P } } } }$   
Output: Aligned collaboration matrix Z.   
foreach $i \in \mathcal { P }$ do   
$Y _ { i } = \mathcal { M } _ { i } ( X _ { i } )$ and $\pmb { { \cal B } } _ { i } = { \mathcal { M } } _ { i } ( \pmb { A } )$   
Upload $( Y _ { i } , { \pmb B } _ { i } )$ to the analyst   
Construct $\{ N _ { i } \} _ { i \in \mathcal { P } }$ from $\{ B _ { i } \} _ { i \in \mathscr { P } }$   
$\pmb { Z } = [ N _ { i } ( \pmb { Y } _ { i } ) ] _ { i \in \mathcal { P } }$ by vertical stacking   
return Z

For each participant i, let $\Theta _ { i }$ denote the participant-specific local secret state, excluding the private dataset $X _ { i }$ and the shared anchor matrix A. The state $\Theta _ { i }$ may include private parameters, randomness, or other participantside quantities used to instantiate the obfuscation mechanism $M _ { i }$ and not revealed in the protocol transcript. In the DC protocol, the analyst observes all uploaded private data and anchor representations, namely $\{ ( Y _ { i } , B _ { i } ) : i \in \mathcal { P } \}$ In addition, participant c reveals its private dataset and local secret state. Because participant c holds the shared anchor matrix A, we assume that it also reveals A to the analyst. Hence, the attacker’s view is

$$
( \{ ( Y _ { i } , B _ { i } ) : i \in \mathcal { P } \} , ( X _ { c } , \Theta _ { c } ) , A ) .\tag{9}
$$

Unless stated otherwise, the attacker also knows the public protocol specification, including the functional forms of the obfuscation mechanisms and the alignment procedure. The attacker does not directly know the private datasets or local secret states of the non-colluding participants, except through information inferred from the analyst’s transcript, participant c’s private dataset and local secret state, and the shared anchor matrix.

The attacker’s objective is to recover, or infer sensitive information about, the private dataset $\pmb { X } _ { i } \in \mathbb { R } ^ { n _ { i } \times d }$ of a non-colluding target participant $i \in \mathcal { P } \setminus \{ c \}$

Algorithm 1 summarizes the generic DC workflow. Its method-dependent step is the construction of row-sizepreserving alignment maps $N _ { i } : \mathbb { R } ^ { m \times h }  \mathbb { R } ^ { m \times h }$ , for any batch size $m ,$ from the uploaded anchor representations $\{ B _ { i } \} _ { i \in \mathscr { P } }$ . Although the analyst constructs these maps using only the anchor representations, their ability to align the private-data representations $\{ Y _ { i } \} _ { i \in \mathcal { P } }$ depends on the common participant-specific map M applied to both matrices. We call the GDP-specific instantiation considered below, which applies independent GDP transformations and aligns the resulting representations following [27], Anchor-Aligned I-GDP (AA-I-GDP).

## 2.4.1. Anchor-Aligned Independent GDP

Definition 2.10 (Obfuscation Mechanism for Anchor-Aligned Independent GDP). Anchor-Aligned Independent GDP (AA-I-GDP) uses an independently sampled GDP mechanism

$$
\ b { M } _ { i } ( \ b { S } ) = \ b { S } \pmb { O } _ { i } + \ b { 1 } _ { m } \ b { \Psi } _ { i } ^ { \top } , \qquad \ b { S } \in \mathbb { R } ^ { m \times d } .
$$

This mechanism is defined for any batch size m. Each participant samples its pair $( O _ { i } , \Psi _ { i } )$ independently, where $\mathbf { \delta } _ { O _ { i } } \in { O } \left( d \right)$ and $\Psi _ { i } \in \mathbb { R } ^ { d }$

Under AA-I-GDP, participant i applies its local GDP mechanism M<sub>i</sub> consistently to both its private data and the shared anchor matrix. Hence, participant i uploads

$$
\pmb { Y } _ { i } = \pmb { M } _ { i } ( \pmb { X } _ { i } ) = \pmb { X } _ { i } \pmb { O } _ { i } + \pmb { 1 } _ { n _ { i } } \pmb { \Psi } _ { i } ^ { \top } ,\tag{10}
$$

$$
\pmb { { \cal B } } _ { i } = \pmb { { \cal M } } _ { i } ( \pmb { { \cal A } } ) = \pmb { { \cal A } } \pmb { { \cal O } } _ { i } + \pmb { 1 } _ { r } \Psi _ { i } ^ { \top } .\tag{11}
$$

The analyst observes the uploaded matrices $\{ Y _ { i } , B _ { i } : i \in \mathcal { P } \}$ , but the raw anchor matrix A is not revealed during honest execution. The uploaded anchor representations serve as a common reference object: although each $\pmb { B } _ { i }$ is expressed in participant i’s private coordinate system, all anchor representations are generated from the same underlying matrix A with matched row correspondence.

Algorithm 2: AA-I-GDP Alignment Procedure   
Input: Private data $\pmb { X } _ { i } \in \mathbb { R } ^ { n _ { i } \times d }$ for each $i \in \mathcal { P } ;$ participant-side shared anchor matrix $\pmb { A } \in \mathbb { R } ^ { r \times d } ;$ reference   
participant $s \in { \mathcal { P } } .$   
Output: Aligned collaboration matrix $\pmb { Z } \in \mathbb { R } ^ { n \times d } ,$   
/\* Participant-side obfuscation \*/   
foreach $i \in \mathcal { P }$ do   
Privately sample GDP parameters $\mathbf { \mathscr { O } } _ { i } \in \mathcal { O } ( d )$ and $\Psi _ { i } \in \mathbb { R } ^ { d } ,$ , independently across participants   
$\mathbf { } Y _ { i } = X _ { i } \mathbf { \bar { O } } _ { i } + \bar { \mathbf { 1 } } _ { n _ { i } } \Psi _ { i } ^ { \top }$   
$\pmb { { \cal B } } _ { i } = A \pmb { { \cal O } } _ { i } + \mathbf { 1 } _ { r } \Psi _ { i } ^ { \top }$   
Upload the pair $( { Y } _ { i } , { B } _ { i } )$ to the analyst   
/\* Analyst-side alignment \*/   
foreach $i \in \mathcal { P }$ do   
$\pmb { b } _ { i } = \frac { 1 } { r } \pmb { B } _ { i } ^ { \top } \mathbf { 1 } _ { r }$   
$\overline { { \pmb { B } } } _ { i } = \dot { \pmb { B } } _ { i } - \mathbf { 1 } , \pmb { b } _ { i } ^ { \top }$   
Set $\pmb { R _ { s } } = \pmb { I _ { d } } ,$ , define $\begin{array} { r } { N _ { s } ( \pmb { T } ) = \pmb { T } , } \end{array}$ , and set $\mathbf { Z } _ { s } = \pmb { Y } _ { s }$   
foreach $i \in { \mathcal { P } } \setminus \{ s \}$ do   
Compute the SVD $\overline { { { B } } } _ { i } ^ { \top } \overline { { { B } } } _ { s } = U _ { i } \Sigma _ { i } V _ { i } ^ { \top }$   
$\pmb { R _ { i } } = \pmb { U _ { i } } \pmb { V _ { i } ^ { \top } }$   
Define $\begin{array} { r } { N _ { i } ( \pmb { T } ) = \left( \pmb { T } - \pmb { 1 } _ { m } \pmb { b } _ { i } ^ { \top } \right) \pmb { R } _ { i } + \pmb { 1 } _ { m } \pmb { b } _ { s } ^ { \top } } \end{array}$ for $\pmb { T } \in \mathbb { R } ^ { m \times d }$   
$\pmb { Z } _ { i } = \pmb { N } _ { i } ( \pmb { Y } _ { i } )$   
$\begin{array} { r } { \mathbf { Z } = \left[ \vdots \atop \vdots \atop \vdots \right] \in \mathbb { R } ^ { n \times d } } \end{array}$   
return Z

Algorithm 2 specializes the generic DC workflow to AA-I-GDP. The analyst first centers each anchor representation, eliminating its participant-specific ofset. A reference participant is then selected to fix the otherwise arbitrary common coordinate system. For every other participant, the SVD solves an orthogonal Procrustes problem [33, 27] between its centered anchor representation and that of the reference participant, yielding the relative rotation between their coordinate systems. The resulting alignment map subtracts the participant’s anchor-representation mean, applies this relative rotation, and adds the reference participant’s anchor-representation mean. Applying the same map to the corresponding private-data representation places it in the reference coordinate system, after which the aligned representations are vertically stacked.

The alignment maps $N _ { i }$ constructed from $\pmb { B } _ { i }$ as in Algorithm 2 can achieve exact DC alignment under the fullcolumn-rank condition:

Definition 2.11 (Exact DC Alignment). Suppose $h = d$ and the participant-specific maps $M _ { i }$ are deterministic. The alignment maps achieve exact DC alignment if,for every batch size m, every $i , j \in { \mathcal { S } } ,$ , and every $\pmb { S } \in \mathbb { R } ^ { m \times d }$

$$
N _ { i } ( M _ { i } ( S ) ) = N _ { j } ( M _ { j } ( S ) ) .\tag{12}
$$

Proposition 2.12 (Exact Alignment of AA-I-GDP). Suppose that, for every participant $i \in { \mathcal { P } } ,$ , the uploaded anchor representation is generated by

$$
\pmb { { \cal B } } _ { i } = A \pmb { \cal O } _ { i } + \mathbf { 1 } , \Psi _ { i } ^ { \top } .\tag{13}
$$

Let

$$
\pmb { a } = \frac { 1 } { r } \pmb { A } ^ { \top } \pmb { 1 } _ { r } , \qquad \overline { { \pmb { A } } } = \pmb { A } - \pmb { 1 } _ { r } \pmb { a } ^ { \top } .
$$

Fix a reference participant $s \in { \mathcal { P } } ,$ , and let $\{ N _ { i } \} _ { i \in { \mathscr { P } } }$ be the alignment maps constructed by Algorithm 2. Assume that A has full column rank, which necessarily requires $r \geq d + 1$ . Then the alignment maps achieve exact DC alignment in the sense ofDefinition 2.11. In particular, for every participant $i \in { \mathcal { P } } ,$ every batch size m, and every input matrix $\pmb { S } \in \mathbb { R } ^ { m \times d }$

$$
\boldsymbol { N } _ { i } ( \boldsymbol { M } _ { i } ( \boldsymbol { S } ) ) = \boldsymbol { M } _ { s } ( \boldsymbol { S } ) = \boldsymbol { S } \boldsymbol { O } _ { s } + \mathbf { 1 } _ { m } \boldsymbol { \Psi } _ { s } ^ { \top } .\tag{14}
$$

Consequently,for the private-data representations, the aligned collaboration matrix satisfies

$$
\begin{array} { r } { \pmb { Z } = \pmb { X } \pmb { O } _ { s } + \pmb { 1 } _ { n } \Psi _ { s } ^ { \top } . } \end{array}\tag{15}
$$

Equivalently, if participant s fixes the global rigid-coordinate gauge and C-GDP is coupled to AA-I-GDP by choosing its common parameters as $( \pmb { { \cal O } } , \Psi ) = ( \pmb { { \cal O } } _ { s } , \Psi _ { s } )$ , then the C-GDP and aligned AA-I-GDP collaboration matrices are equal pathwise for every private dataset X. This is an equality of analyst-side collaboration outputs; the protocols complete transcripts remain diferent.

Although AA-I-GDP exactly restores cross-participant geometric comparability, this direct anchor-based implementation introduces a significant vulnerability in the analyst-participant collusion setting. The vulnerability arises from the shared anchor matrix. We will discuss this further in Section 3.

## 2.4.2. Anchor-Aligned Independent GDP with Private-Data Noise

Prior DC work places calibrated noise on private-data intermediate representations while retaining noiseless anchor representations for alignment [29]. We instantiate that placement with isotropic Gaussian noise and call the resulting comparison baseline AA-I-GDP (private-data noise).

Definition 2.13 (AA-I-GDP (private-data noise)). AA-I-GDP (private-data noise) uses the same participant-specific GDP mechanism M for the private-data matrix and the shared anchor matrix, but adds noise only to the private-data representation. Participant i samples $\pmb { W } _ { i } \in \mathbb { R } ^ { n _ { i } \times d }$ independently across participants and independently of $( O _ { i } , \Psi _ { i } )$ with entries

$$
( W _ { i } ) _ { k \ell } \stackrel { \mathrm { i n d } } { \sim } N ( 0 , 1 ) , \qquad k \in [ n _ { i } ] , \ell \in [ d ] .
$$

It then uploads

$$
\begin{array} { r } { \pmb { Y } _ { i } = \pmb { \mathcal { M } } _ { i } ( \pmb { X } _ { i } ) + \sigma \pmb { W } _ { i } = \pmb { X } _ { i } \pmb { O } _ { i } + \pmb { 1 } _ { n _ { i } } \pmb { \Psi } _ { i } ^ { \top } + \sigma \pmb { W } _ { i } , } \end{array}\tag{16}
$$

$$
\pmb { { \cal B } } _ { i } = \pmb { { \cal M } } _ { i } ( \pmb { { \cal A } } ) = \pmb { { \cal A } } \pmb { { \cal O } } _ { i } + \pmb { 1 } _ { r } \Psi _ { i } ^ { \top } .\tag{17}
$$

The scale $\sigma \geq 0$ controls private-data noise, with $\sigma = 0$ recovering AA-I-GDP. Gaussian noise provides aformal DP guarantee only after neighboring datasets and sensitivity are defined and σ is calibrated to a target (ε δ)-DP level; proceduresfor performing this calibration in DC have been studied in [29].

## 3. The Alignment–Recovery Tension in Existing Constructions

Section 2 established the central tension among the comparison baselines. C-GDP preserves a common coordinate system for downstream learning, but disclosure of its shared perturbation parameters permits exact inversion of every participant’s upload. I-GDP removes this shared-secret channel by using participant-specific parameters, but it places participant blocks in incompatible coordinate systems. Plain AA-I-GDP appears to reconcile these properties: under the full-column-rank condition, Proposition 2.12 shows that anchor alignment restores a common analyst-side output that is pathwise equal to gauge-matched C-GDP.

This section shows why that apparent resolution is incomplete in the analyst-participant collusion setting. The anchor representations that enable exact alignment also become an exact parameter-recovery channel when a colluding participant discloses the shared raw anchor matrix. Moreover, adding isotropic Gaussian noise only to the privatedata representations leaves this channel noiseless and yields the same analyst-side collaboration-output distribution as matched C-GDP with private-data noise. These results isolate the anchor representations as the remaining point at which reconstruction risk and cross-participant alignment interact.

Algorithm 3: Known-Anchor Attack against AA-I-GDP in the Analyst-Participant Collusion Setting   
Input: Anchor matrix A revealed by the colluder; uploaded pairs $\left\{ ( { \pmb Y } _ { i } , { \pmb B } _ { i } ) \right\} _ { i = 1 } ^ { p }$ ; colluder index $c \in \mathcal { P } .$   
Output: Reconstructions $\{ \widehat { X } _ { i } : i \in \mathcal { P } \setminus \{ c \} \}$   
$\pmb { a } = \frac { 1 } { r } \pmb { A } ^ { \top } \mathbf { 1 } _ { r }$   
$\overline { { A } } = A - \mathbf { 1 } _ { r } \pmb { a } ^ { \top }$   
foreach $i \in \mathcal { P } \setminus \{ c \}$ do   
$\pmb { b } _ { i } = \frac { 1 } { r } \pmb { B } _ { i } ^ { \top } \pmb { 1 } _ { r }$   
$\overline { { \pmb { B } } } _ { i } = \pmb { B } _ { i } - \mathbf { 1 } _ { r } \pmb { b } _ { i } ^ { \top }$   
$\widehat { \pmb { O } } _ { i } = \overline { { \pmb { A } } } ^ { \dagger } \overline { { \pmb { B } } } _ { i }$   
${ \widehat { \Psi } } _ { i } = \pmb { { b } } _ { i } - { \widehat { \pmb { { O } } } } _ { i } ^ { \top } \pmb { { a } }$   
${ \widehat { \pmb { X } } } _ { i } = \left( { \pmb { Y } } _ { i } - \mathbf { 1 } _ { n _ { i } } { \widehat { \Psi } } _ { i } ^ { \top } \right) { \widehat { \pmb { O } } } _ { i } ^ { \top }$   
return $\{ \widehat { X } _ { i } : i \in \mathcal { P } \setminus \{ c \} \}$

## 3.1. Noiseless AA-I-GDP: Exact Alignment and Known-Anchor Recovery

The exact alignment established by Proposition 2.12 relies on every participant transforming the same anchor matrix. During honest execution, the analyst observes only the uploaded representations $\{ { \pmb { { \cal B } } } _ { i } \} _ { i \in \mathcal { P } }$ , not the shared ancho matrix A. In the DC analyst-participant collusion view of Definition 2.9, however, a colluding participant reveals A, so the analyst obtains the known-input pair $( A , B _ { i } )$ for every participant i. Centering removes the participant-specific translation, and full column rank of A makes $o _ { i }$ identifiable from $\overline { { B } } _ { i } ;$ the anchor means then identify $\Psi _ { i }$ . Thus, the alignment channel also becomes a transform-recovery channel.

Algorithm 3 makes the resulting attack explicit. It requires no raw records from a non-colluding target: after recovering the target’s GDP parameters, the analyst directly inverts the target’s private data upload.

The following proposition establishes that the attack reconstructs every non-colluding participant’s private dataset exactly under the stated noiseless and full-column-rank conditions.

Proposition 3.1 (Vulnerability of AA-I-GDP in the Analyst-Participant Collusion Setting). Suppose that the uploaded representations are generated exactly by AA-I-GDP. Thus,for each participant $i \in \mathcal { S }$

$$
\begin{array} { r } { Y _ { i } = X _ { i } \pmb { O } _ { i } + \mathbf { 1 } _ { n _ { i } } \Psi _ { i } ^ { \top } , } \end{array}\tag{18}
$$

$$
\pmb { { \cal B } } _ { i } = A \pmb { { \cal O } } _ { i } + \mathbf { 1 } _ { r } \Psi _ { i } ^ { \top } ,\tag{19}
$$

where $\mathbf { \mathscr { O } } _ { i } \in \mathcal { O } ( d )$ and $\Psi _ { i } \in \mathbb { R } ^ { d }$ are participant-specific GDP parameters. Let

$$
\pmb { a } = \frac { 1 } { r } \pmb { A } ^ { \top } \pmb { 1 } _ { r } , \qquad \overline { { \pmb { A } } } = \pmb { A } - \pmb { 1 } _ { r } \pmb { a } ^ { \top } .
$$

Assume that $\overline { { A } }$ hasfull column rank. Consider the analyst-participant collusion settingfor DC in Definition 2.9, with colluding participant $c \in { \mathcal { P } } .$ . Under this setting, the attacker observes the shared anchor matrix A, and Algorithm 3 outputs

$$
{ \widehat { X } } _ { i } = X _ { i }
$$

for every non-colluding participant $i \in \mathcal { P } \ \backslash$ {c}.

Taken together, Propositions 2.12 and 3.1 reveal the dual role of the noiseless anchor representations. Under the same rank condition, they are suficient to align every participant into the reference coordinate system and, once the raw anchor matrix is disclosed, to recover every participant-specific GDP transformation. With reference participant s chosen as the global rigid-coordinate gauge and C-GDP coupled through $( \pmb { { \cal O } } , \Psi ) = ( \pmb { { \cal O } } _ { s } , \Psi _ { s } )$ , the aligned AA-I-GDP and C-GDP collaboration outputs are equal pathwise. This output equality does not extend to the complete protocol transcripts, because AA-I-GDP additionally releases participant-specific anchor representations. Plain AA I-GDP therefore resolves the coordinate mismatch of I-GDP methods but relocates the exact recovery channel in the analyst-participant collusion setting from the common C-GDP parameters to the shared anchor matrix.

## 3.2. Private-Data Noise: Unchanged Alignment and Output Equivalence

A natural modification is to perturb the private-data representations while retaining noiseless anchor representations for alignment. This changes the records obtained after inversion but does not perturb the recovery of the participant-specific GDP transformations from $( A , B _ { i } )$ . Under the full-column-rank condition of Proposition 2.12, the noiseless anchor representations still determine

$$
\mathbf { } R _ { i } = \pmb { O } _ { i } ^ { \top } \pmb { O } _ { s } , \qquad i \in \mathcal { P } ,
$$

where ${ \pmb R } _ { s } = { \pmb I } _ { d }$ . Consequently, the signal component is aligned exactly as in plain AA-I-GDP, and the private-data noise enters the collaboration matrix only after an orthogonal transformation.

Proposition 3.2 (Aligned Representation under AA-I-GDP (private-data noise)). Suppose that participant $i ~ \in ~ \mathcal { S }$ uploads

$$
\pmb { Y } _ { i } = \pmb { X } _ { i } \pmb { O } _ { i } + \pmb { 1 } _ { n _ { i } } \pmb { \Psi } _ { i } ^ { \top } + \sigma \pmb { W } _ { i }
$$

and the noiseless anchor representation

$$
\pmb { { \cal B } } _ { i } = A \pmb { { \cal O } } _ { i } + \mathbf { 1 } _ { r } \Psi _ { i } ^ { \top } ,
$$

where $\pmb { O } _ { i } \in O ( d ) , \Psi _ { i } \in \mathbb { R } ^ { d } ,$ , and $\pmb { W } _ { i } \in \mathbb { R } ^ { n _ { i } \times d }$ . Assume that A has full column rank, and let $N _ { i }$ be the alignment maps constructed from the anchor representations as in Algorithm 2, with reference participant $s \in { \mathcal { P } } .$ . Define

$$
\widetilde { \pmb { W } } _ { i } = \pmb { W } _ { i } \pmb { O } _ { i } ^ { \top } \pmb { O } _ { s } , \qquad i \in \mathcal { P } .
$$

Then

$$
\begin{array} { r } { N _ { i } ( Y _ { i } ) = X _ { i } \pmb { O } _ { s } + \mathbf { 1 } _ { n _ { i } } \Psi _ { s } ^ { \top } + \sigma \widetilde { \pmb { W } } _ { i } . } \end{array}
$$

Consequently,

$$
\begin{array} { r } { \pmb { Z } = X \pmb { O } _ { s } + \mathbf { 1 } _ { n } \Psi _ { s } ^ { \top } + \sigma \widetilde { W } , \qquad \widetilde { W } = \left[ \widetilde { W } _ { i } \right] _ { i \in \mathcal { P } } . } \end{array}
$$

If the entries of $W _ { 1 } , \ldots , W _ { p }$ are mutually independent standard Gaussian variables and the collection $\{ W _ { i } \} _ { i \in { \mathscr { P } } }$ isjointly independent of all GDP parameters $\{ ( O _ { i } , \Psi _ { i } ) \} _ { i \in \mathcal { P } }$ , then the entries of $\widetilde { \pmb { W } }$ are mutually independent standard Gaussian variables. Moreover, We is independent ofthe GDP parameters.

Proposition 3.2 shows that the signal is mapped exactly into the reference participant’s coordinate system. Because right multiplication by an orthogonal matrix preserves independent isotropic Gaussian noise, the aligned noise has the same distribution as the noise added under matched C-GDP with private-data noise. The resulting output-level equivalence is stated next.

Corollary 3.3 (Collaboration-Output Equivalence under Private-Data Noise). Assume the Gaussian-noise conditions of Proposition 3.2. Fix the GDP parameters, let ${ \pmb Z } ^ { \mathrm { A A } }$ denote the aligned collaboration matrix produced by AA-I-GDP (private-data noise), and let $\bar { \boldsymbol { z } } ^ { \mathrm { c } }$ denote the collaboration matrix produced by C-GDP (private-data noise) with common parameters

$$
( { \cal O } , \Psi ) = ( { \cal O } _ { s } , \Psi _ { s } ) .
$$

Writing $U \overset { \mathrm { d } } { = }$ V to mean that random matrices U and V have the same distribution, the two collaboration matrices satisfy

$$
{ \pmb Z } ^ { \mathrm { A A } } \overset { \mathrm { d } } { = } { \pmb Z } ^ { \mathrm { C } } .
$$

Moreover,for anyfixed realization ofthe AA-I-GDP variables, consider the coupling in which the standard Gaussian noise block used by C-GDPfor participant i is

$$
\widetilde { \pmb { W } } _ { i } = \pmb { W } _ { i } \pmb { O } _ { i } ^ { \top } \pmb { O } _ { s } .
$$

Under this coupling,

$$
{ \pmb Z } ^ { \mathrm { A A } } = { \pmb Z } ^ { \mathrm { C } } .
$$

Corollary 3.3 distinguishes two forms of equivalence: equality in distribution under the stated Gaussian assumptions and realization-wise equality under the specified coupling. The same coupling also makes the compared reconstruction attacks return identical noisy records. For C-GDP, known-secret inversion gives

$$
\begin{array} { r l } & { \widehat { X } _ { i } ^ { \mathrm { C } } = \left( \mathbf { Z } _ { i } ^ { \mathrm { C } } - \mathbf { 1 } _ { n _ { i } } \Psi _ { s } ^ { \intercal } \right) { O } _ { s } ^ { \intercal } } \\ & { \qquad = X _ { i } + \sigma \widetilde { W } _ { i } { O } _ { s } ^ { \intercal } } \\ & { \qquad = X _ { i } + \sigma W _ { i } { O } _ { i } ^ { \intercal } . } \end{array}
$$

For AA-I-GDP, the full-column-rank anchor matrix permits exact recovery of $( O _ { i } , \Psi _ { i } )$ , after which known-anchor inversion gives

$$
\begin{array} { r l } & { \widehat { X } _ { i } ^ { \mathrm { A A } } = \left( { Y } _ { i } - \mathbf { 1 } _ { n _ { i } } \Psi _ { i } ^ { \top } \right) { \pmb { O } } _ { i } ^ { \top } } \\ & { ~ = { X } _ { i } + \sigma \pmb { W } _ { i } \pmb { O } _ { i } ^ { \top } . } \end{array}
$$

These identities concern the analyst-side collaboration outputs and the two stated reconstruction attacks, not the complete protocol transcripts or arbitrary attacks. Under the full-column-rank condition, the reference-participant gauge, and the stated isotropic-Gaussian coupling, AA-I-GDP with private-data noise yields neither a distinct collaborative output nor distinct reconstruction behavior against the compared attacks; it also releases participant-specific anchor representations. Adding noise only to the private-data channel, therefore, changes the recovered records through additive noise but does not weaken the noiseless anchor-recovery channel.

## 3.3. Research Question: Perturbing the Alignment Signal

The preceding results isolate one unresolved design variable: the anchor representations. Under the full-columnrank condition, they provide the information required for exact alignment but, once disclosed, also identify every participant-specific transformation. Private-data noise leaves both properties unchanged. The same signal, therefore, governs alignment utility and known-anchor recovery.

This observation suggests relocating the perturbation from the private-data representations $Y _ { i }$ to the anchor representations $\pmb { B } _ { i }$ used for alignment. We therefore ask:

Can noise added only to the anchor representations make known-anchor recovery inaccurate in the analyst-participant collusion setting while preserving cross-participant alignment accurate enough for downstream learning?

Section 4 studies this question through AA-I-GDP with anchor noise. At zero anchor noise, the protocol reduces to plain AA-I-GDP; at positive noise, both transform recovery and cross-participant alignment become approximate. The resulting problem is to characterize whether useful attack-dependent privacy-utility operating points exist between these two efects.

## 4. Proposed Noisy Anchor Alignment

Anchor-Aligned Independent-Transform GDP with Anchor Noise (AA-I-GDP (anchor noise)) is the construction proposed in this paper. It combines participant-specific GDP transformations with noisy anchor representations and Procrustes-based alignment. Relative to AA-I-GDP (private-data noise), the method moves additive noise from the private-data upload to the anchor upload: the private-data representations remain purely geometric perturbations, whereas the uploaded anchor representations are noisy. The same participant-specific GDP parameters are used for both representations, so the anchor representations still provide the necessary alignment information, although the resulting alignment is approximate rather than exact.

The optimization ingredients used below are established: the GOPP and GPM solvers are taken from [30]. Our contribution is to formulate noisy multi-participant anchor alignment as a centered GOPP, derive the protocol-specific alignment and reconstruction consequences of its solution, and specialize the cited GPM guarantees to the centered orthonormal-anchor model.

## 4.1. Protocol and Anchor-Matrix Setup

Definition 4.1 (AA-I-GDP (anchor noise)). Given a participant-side shared anchor matrix $\textbf { \em A } \in \mathbb { R } ^ { r \times d }$ , AA-I-GDP (anchor noise) uses the same participant-specific GDP mechanism $M _ { i }$ for the private-data matrix and the shared anchor matrix, but adds noise only to the anchor representation. Participant i samples $\pmb { W } _ { i } \in \mathbb { R } ^ { r \times d }$ independently across participants and independently of $( O _ { i } , \Psi _ { i } )$ , with entries

$$
( W _ { i } ) _ { k \ell } \stackrel { \mathrm { i n d } } { \sim } N ( 0 , 1 ) , \qquad k \in [ r ] , \ell \in [ d ] .
$$

It then uploads

$$
\begin{array} { r l } & { \pmb { Y } _ { i } = \pmb { M } _ { i } ( \pmb { X } _ { i } ) = \pmb { X } _ { i } \pmb { O } _ { i } + \mathbf { 1 } _ { n _ { i } } \Psi _ { i } ^ { \top } , } \\ & { \pmb { B } _ { i } = \pmb { M } _ { i } ( \pmb { A } ) + \nu \pmb { W } _ { i } = \pmb { A } \pmb { O } _ { i } + \mathbf { 1 } _ { r } \Psi _ { i } ^ { \top } + \nu \pmb { W } _ { i } . } \end{array}\tag{20}
$$

(21)

The scale $\nu \geq 0$ controls anchor noise, with $\nu = 0$ recovering AA-I-GDP.

The preceding definition applies to a participant-side shared anchor matrix A. Let $r \geq d + 1$ . Before any release, the participants generate a single shared anchor matrix $\pmb { A } \in \mathbb { R } ^ { r \times d }$ satisfying

$$
\mathbf { 1 } _ { r } ^ { \top } A = \mathbf { 0 } _ { d } ^ { \top } , \qquad A ^ { \top } A = I _ { d } .\tag{22}
$$

Thus, the anchor matrix is centered and has orthonormal columns. The first constraint fixes its centroid at the origin, while the second fixes its scale relative to the anchor-noise parameter v. One way to generate such a matrix is to center an i.i.d. standard Gaussian matrix and take the orthonormal factor of its sign-corrected thin QR factorization; the full construction is given in the supplementary material. The same realization of A is distributed to every participant and withheld from the analyst during honest execution. Neither A nor the randomness used to generate it is included in the honest protocol transcript.

The honest analyst view therefore consists of $\{ Y _ { i } , B _ { i } : i \in \mathcal { P } \}$ and excludes the anchor matrix A. The uploaded anchor representations $\pmb { B } _ { i }$ provide a noisy common reference object: each $\pmb { B } _ { i }$ is generated from the same underlying anchor matrix and the same participant-specific GDP mechanism used to generate $Y _ { i \mathrm { : } }$ , but is additionally perturbed by $\nu \pmb { W } _ { i } . \mathrm { A t } \nu = 0$ , the GOPP is equivalent to the exact-alignment construction under Proposition 2.12; at $\nu > 0$ , the anchor noise introduces alignment error.

## 4.2. Alignment via the Generalized Orthogonal Procrustes Problem

For each participant $i \in \mathcal { P }$ , the analyst observes the uploaded anchor representation $\mathbf { } B _ { i } ,$ generated as

$$
\pmb { { \cal B } } _ { i } = \pmb { { \cal A } } \pmb { { \cal O } } _ { i } + \pmb { 1 } _ { r } \Psi _ { i } ^ { \top } + \nu \pmb { W } _ { i } .
$$

Because the uploaded anchor representations $\pmb { B } _ { i }$ are noisy transformed versions of the same latent anchor configuration, the analyst estimates participant-specific alignment maps <sup>1</sup>

$$
N _ { i } : \mathbb { R } ^ { m \times d }  \mathbb { R } ^ { m \times d } , \qquad N _ { i } ( T ) = ( T - \mathbf { 1 } _ { m } \tau _ { i } ^ { \top } ) R _ { i } ,
$$

valid for any batch size m. Here, $\tau _ { i } \in \mathbb { R } ^ { d }$ and $\pmb { R } _ { i } \in O ( d )$ are the estimated translation and orthogonal alignment for participant i.

The template $\pmb { C } \in \mathbb { R } ^ { r \times d }$ , together with the alignment parameters, is estimated by solving

$$
\begin{array} { r l } { \underset { C , R _ { 1 } , \ldots , R _ { p } , \tau _ { 1 } , \ldots , \tau _ { p } } { \operatorname* { m i n } } } & { \displaystyle \sum _ { i = 1 } ^ { p } \left\| C - \left( B _ { i } - \mathbf { 1 } _ { r } \tau _ { i } ^ { \top } \right) { \pmb { R } } _ { i } \right\| _ { F } ^ { 2 } } \\ { \mathrm { s . t . } \quad R _ { i } \in O ( d ) , \qquad C \in \mathbb { R } ^ { r \times d } , \qquad \tau _ { i } \in \mathbb { R } ^ { d } , \quad } & { i = 1 , \ldots , p . } \end{array}\tag{23}
$$

We refer to this problem as the generalized orthogonal Procrustes problem (GOPP) [30]: it jointly estimates the participant-specific rigid transformations that register all observed anchor matrices into a common coordinate system.

The GOPP objective is invariant under a common change of origin and orientation in the collaboration coordinate system. The following proposition makes this ambiguity explicit.

Proposition 4.2 (Global rigid-motion invariance of the GOPP). Suppose that

$$
\left( C ^ { \star } , R _ { 1 } ^ { \star } , \ldots , R _ { p } ^ { \star } , \tau _ { 1 } ^ { \star } , \ldots , \tau _ { p } ^ { \star } \right)
$$

is a minimizer of Problem (23). For any $\pmb { s } \in \mathbb { R } ^ { d }$ and $Q \in O ( d ) ,$ , define

$$
\widetilde { \pmb { C } } = \left( \pmb { C } ^ { \star } + \mathbf { 1 } _ { r } \pmb { s } ^ { \top } \right) \pmb { Q } ,
$$

and, for every $i \in { \mathcal { P } } ,$

$$
\widetilde { \pmb { R } } _ { i } = \pmb { R } _ { i } ^ { \star } \pmb { Q } , \qquad \widetilde { \pmb { \tau } } _ { i } = \pmb { \tau } _ { i } ^ { \star } - \pmb { R } _ { i } ^ { \star } \pmb { s } .
$$

Then

$$
\left( \widetilde { C } , \widetilde { R } _ { 1 } , \ldots , \widetilde { R } _ { p } , \widetilde { \tau } _ { 1 } , \ldots , \widetilde { \tau } _ { p } \right)
$$

is also a minimizer of Problem (23) and attains the same objective value. Consequently, every minimizer belongs to an equivalence class generated by common translations and orthogonal transformations of the collaboration coordinate system. Therefore, without an external reference or a gauge-fixing constraint, the GOPP objective alone cannot yield a unique recovery ofthe participant-specific GDP parameters $( O _ { i } , \Psi _ { i } )$

Proposition 4.2 concerns the absolute origin and orientation of the collaboration coordinate system. It does not imply a loss of relative alignment: every aligned representation undergoes the same rigid transformation, which preserves pairwise Euclidean distances and the relative geometry used by downstream learning. We therefore select a convenient representative of this equivalence class by first centering the template and then eliminating the translation and template variables.

Lemma 4.3 (Centering the GOPP template). Suppose that $\pmb { { B } } _ { i } \in \mathbb { R } ^ { r \times d } , i = 1 , \ldots , p ,$ , are observed, and define

$$
\Phi ( C , R _ { 1 } , \ldots , R _ { p } , \tau _ { 1 } , \ldots , \tau _ { p } ) : = \sum _ { i = 1 } ^ { p } \left\| C - \left( B _ { i } - 1 _ { r } \tau _ { i } ^ { \top } \right) R _ { i } \right\| _ { F } ^ { 2 } .
$$

Let $\mathcal { F }$ be the feasible set defined by $\pmb { C } \in \mathbb { R } ^ { r \times d } , \pmb { R } _ { i } \in O ( d ) ,$ , and $\tau _ { i } \in \mathbb { R } ^ { d } f o r i = 1 , . . . , p ,$ and let

$$
\mathcal { F } _ { 0 } : = \left\{ ( C , R _ { 1 } , \ldots , R _ { p } , \tau _ { 1 } , \ldots , \tau _ { p } ) \in \mathcal { F } : \mathbf { 1 } _ { r } ^ { \top } C = \mathbf { 0 } _ { d } ^ { \top } \right\} .
$$

Then inf<sub>F</sub> $\Phi = \operatorname { i n f } _ { { \mathcal { F } } _ { 0 } } \Phi .$ . Hence, the GOPP may be restricted, without changing its optimal value, to templates with an average row of zero.

Thus, for purposes of the optimal value and the relative geometry of the aligned representations, it sufices to restrict the GOPP to templates whose average row is zero. Indeed, whenever an unconstrained minimizer is obtained, the centering transformation used in the proof produces a centered minimizer with the same rotations, adjusted translations, and the same objective value. The corresponding aligned anchor representations are shifted only by a common translation. Hence, we may replace Problem (23) by the following centered-template GOPP.

$$
\begin{array} { r l r l r } { \underset { C , R _ { 1 } , \ldots , R _ { p } , \tau _ { 1 } , \ldots , \tau _ { p } } { \operatorname* { m i n } } } & { \displaystyle \sum _ { i = 1 } ^ { p } \left\| C - \left( B _ { i } - 1 _ { r } \tau _ { i } ^ { \top } \right) R _ { i } \right\| _ { F } ^ { 2 } } & & { } \\ { \mathrm { s . t . } \quad R _ { i } \in O ( d ) , \quad } & { C \in \mathbb { R } ^ { r \times d } , \quad } & { \mathbf { 1 } _ { r } ^ { \top } C = \mathbf { 0 } _ { d } ^ { \top } , \quad } & { \tau _ { i } \in \mathbb { R } ^ { d } , \quad } & { i = 1 , \ldots , p . } \end{array}\tag{24}
$$

With the centered gauge fixed, the translation variables and the template can be eliminated in closed form. The following proposition collects this reduction and identifies the max-trace problem used by the GPM.

Proposition 4.4 (Reduction of the centered GOPP to max-trace form). Under the notation ofLemma 4.3, define

$$
\pmb { H } : = \pmb { I } _ { r } - \frac { 1 } { r } \pmb { 1 } _ { r } \pmb { 1 } _ { r } ^ { \top } , \qquad \overline { { \pmb { B } } } _ { i } : = \pmb { H } \pmb { B } _ { i } , \qquad i = 1 , \ldots , p .
$$

For fixed $\pmb { R } _ { 1 } , \ldots , \pmb { R } _ { p } \in O ( d )$ , the unique minimizers over the translation variables and the centered template are

$$
\pmb { \tau } _ { i } ^ { \star } = \frac { 1 } { r } \pmb { B } _ { i } ^ { \top } \pmb { 1 } _ { r } , \qquad i = 1 , \dots , p ,\tag{25}
$$

and

$$
C ^ { \star } = \frac { 1 } { p } \sum _ { i = 1 } ^ { p } \overline { { B } } _ { i } R _ { i } .\tag{26}
$$

Define the stacked rotation matrix and the block Gram matrix by

$$
\pmb { R } : = \left[ \pmb { R } _ { i } \right] _ { i \in \mathcal { P } } \in \mathbb { R } ^ { p d \times d } ,
$$

and

$$
\begin{array} { r } { \pmb { G } _ { i j } : = \overline { { \pmb { B } } } _ { i } ^ { \top } \overline { { \pmb { B } } } _ { j } , \qquad i , j = 1 , \ldots , p . } \end{array}\tag{27}
$$

Then the rotation components ofthe centered GOPP minimizers are precisely the solutions of

$$
\begin{array} { r l } { \underset { R _ { 1 } , \ldots , R _ { p } } { \operatorname* { m a x } } } & { \mathrm { t r } \big ( R ^ { \top } G R \big ) } \\ { s . t . } & { R _ { i } \in O ( d ) , \qquad i = 1 , \ldots , p . } \end{array}\tag{28}
$$

$I f ~ V _ { \mathrm { G O P P } } ^ { \star }$ denotes the minimum of Φ over the centered feasible set ${ \mathcal { F } } _ { 0 } ,$ and $V _ { \mathrm { t r } } ^ { \star }$ denotes the optimal value of Problem (28), then

$$
V _ { \mathrm { G O P P } } ^ { \star } = \sum _ { i = 1 } ^ { p } \left\| \overline { { \boldsymbol { B } } } _ { i } \right\| _ { F } ^ { 2 } - \frac { 1 } { p } V _ { \mathrm { t r } } ^ { \star } .\tag{29}
$$

Conversely, if $R _ { 1 } ^ { \star } , \ldots , R _ { p } ^ { \star }$ solve Problem (28), then Equations (25) and (26), evaluated at these rotations, produce a centered minimizer ofthe GOPP.

Problem (28) is nonconvex and is generally dificult to solve globally. We use the spectral initialization and blockwise generalized power method of [30], specialized only in notation and dimensions to the present alignment problem. For a square matrix $S \in \mathbb { R } ^ { d \times d }$ with singular value decomposition $\pmb { S } = \pmb { U } \pmb { \Sigma } \pmb { V } ^ { \top }$ , define the orthogonal pola projection

$$
\Pi ( S ) = U V ^ { \top } \in O ( d ) .\tag{30}
$$

If S is rank deficient, the nearest orthogonal factor may be nonunique; in that case, any choice obtained from a full singular value decomposition may be used. For a stacked matrix

$$
\pmb { M } = \left[ \pmb { M } _ { i } \right] _ { i \in \mathcal { P } } \in \mathbb { R } ^ { p d \times d } , \qquad \pmb { M } _ { i } \in \mathbb { R } ^ { d \times d } ,
$$

define the blockwise polar projection by

$$
\Pi _ { p } ( { \pmb M } ) : = \left[ \Pi ( { \pmb M } _ { i } ) \right] _ { i \in \mathcal { P } } .\tag{31}
$$

Because Problem (28) is nonconvex, we solve it using the spectral initialization and blockwise generalized power method of [30]. The top d eigenvectors of G are projected blockwise onto $O ( d )$ to initialize the rotations, which are then updated according to

$$
\pmb { R } ^ { ( t + 1 ) } = \Pi _ { p } \big ( \pmb { G } \pmb { R } ^ { ( t ) } \big ) .\tag{32}
$$

The iterations terminate when successive rotation estimates difer by at most ε in Frobenius norm or when $T _ { \mathrm { m a x } }$ iterations are reached. Complete pseudocode is provided in the supplementary material; satisfaction of the stopping criterion indicates numerical stabilization but does not certify global optimality.

Algorithm 4: AA-I-GDP with Anchor Noise   
Input: Private data $\pmb { X } _ { i } \in \mathbb { R } ^ { n _ { i } \times d }$ for each $i \in \mathcal { P } ;$ anchor size $r \geq d + 1 ; $ anchor-noise scale $\nu \geq 0 ;$ GPM   
parameters $T _ { \mathrm { m a x } }$ and $\varepsilon > 0 .$   
Output: Aligned collaboration matrix $\pmb { Z } \in \mathbb { R } ^ { n \times d } ,$   
/\* Participant-only Obfuscation \*/   
Generate A satisfying (22)   
Distribute the same A to every participant and withhold it from the analyst   
foreach $i \in \mathcal { P }$ do   
Privately sample GDP parameters $\mathbf { \mathscr { O } } _ { i } \in \mathcal { O } ( d )$ and $\Psi _ { i } \in \mathbb { R } ^ { d } ,$ , independently across participants   
Sample $\bar { \boldsymbol { W } _ { i } } \in \bar { \mathbb { R } } ^ { r \times d }$ with independent standard Gaussian entries   
$\pmb { Y } _ { i } = \mathbf { \bar { \mathcal { M } } } _ { i } ( \pmb { X } _ { i } ) = \pmb { X } _ { i } \pmb { O } _ { i } + \mathbf { 1 } _ { n _ { i } } \Psi _ { i } ^ { \top }$   
$\pmb { { B } } _ { i } = \pmb { M } _ { i } ( \pmb { A } ) + \nu \pmb { W } _ { i } = \pmb { A } \pmb { O } _ { i } + \pmb { 1 } _ { r } \Psi _ { i } ^ { \top } + \nu \pmb { W } _ { i }$   
Upload $( Y _ { i } , { \pmb B } _ { i } )$ to the analyst   
$^ { \prime * }$ Analyst-side Alignment \*/   
foreach $i \in \mathcal { P }$ do   
$\pmb { b } _ { i } = \frac { 1 } { r } \pmb { B } _ { i } ^ { \top } \mathbf { 1 } _ { r }$ and $\overline { { \pmb { B } } } _ { i } = \pmb { B } _ { i } - \mathbf { 1 } _ { r } \pmb { b } _ { i } ^ { \top }$   
Form $G \in \mathbb { R } ^ { p d \times p d }$ with blocks $G _ { i j } = \overline { { B } } _ { i } ^ { \top } \overline { { B } } _ { j }$   
Obtain $\widehat { \pmb { R } } = [ \widehat { \pmb { R } } _ { i } ] _ { i \in \mathcal { P } }$ using the spectral-initialized GPM in (32) with $( T _ { \operatorname* { m a x } } , \varepsilon )$   
foreach $i \in \mathcal { P }$ do   
Define ${ \cal N } _ { i } ( { \cal T } ) = \left( { \cal T } - \mathbf { 1 } _ { m } { \cal b } _ { i } ^ { \top } \right) \widehat { R } _ { i }$ for $\pmb { T } \in \mathbb { R } ^ { m \times d }$   
$\pmb { Z } _ { i } = \pmb { N } _ { i } ( \pmb { Y } _ { i } )$   
$\pmb { Z } = \left[ \pmb { Z } _ { i } \right] _ { i \in \mathcal { P } }$   
return Z

For each participant, define the anchor mean

$$
\pmb { b } _ { i } : = \frac { 1 } { r } \pmb { B } _ { i } ^ { \top } \pmb { 1 } _ { r } \in \mathbb { R } ^ { d } .\tag{33}
$$

Using the resulting rotation estimates, AA-I-GDP (anchor noise) defines the anchor-derived alignment map

$$
\begin{array} { r } { { \cal N } _ { i } ( T ) = \left( T - { \bf 1 } _ { m } { \cal b } _ { i } ^ { \top } \right) \widehat { \pmb { R } } _ { i } , \qquad { \pmb { T } } \in \mathbb { R } ^ { m \times d } . } \end{array}\tag{34}
$$

The analyst applies these maps to the uploaded private-data representations and forms the AA-I-GDP (anchor noise) collaboration matrix

$$
\pmb { Z } = \left[ N _ { i } ( \pmb { Y } _ { i } ) \right] _ { i \in \mathcal { P } } \in \mathbb { R } ^ { n \times d } .\tag{35}
$$

Algorithm 4 summarizes the complete honest-execution pipeline, combining the participant-only anchor-matrix setup and uploads with the analyst-side noisy-anchor alignment described above.

## 4.3. Alignment Error and Solver Guarantees

By the participant-only anchor-matrix setup in (22), the anchor matrix is centered and has orthonormal columns. When $\nu = 0 ,$ , a zero-residual GOPP solution recovers a common efective rotation: there exists $Q \in O ( d )$ such that

$$
O _ { i } \widehat { R } _ { i } = Q , \qquad i \in \mathcal { P } .\tag{36}
$$

Consequently, for every $\pmb { S } \in \mathbb { R } ^ { m \times d }$

$$
{ \cal N } _ { i } ( M _ { i } ( S ) ) = S Q ,\tag{37}
$$

which is independent of the participant and therefore satisfies exact DC alignment.

When $\nu > 0 ,$ , define the average anchor-noise vector

$$
\pmb { \eta } _ { i } : = \frac { \nu } { r } \pmb { W } _ { i } ^ { \top } \mathbf { 1 } _ { r } .\tag{38}
$$

Because the anchor matrix is centered, the mean of participant i’s noisy anchor upload is

$$
\pmb { b } _ { i } ^ { \top } = \Psi _ { i } ^ { \top } + \pmb { \eta } _ { i } ^ { \top } .\tag{39}
$$

Substitution into the alignment map gives

$$
\begin{array} { c } { { N _ { i } ( { \cal M } _ { i } ( S ) ) = \left( S O _ { i } + { \bf 1 } _ { m } \Psi _ { i } ^ { \top } - { \bf 1 } _ { m } b _ { i } ^ { \top } \right) \widehat { \pmb { R } } _ { i } } } \\ { { = S O _ { i } \widehat { \pmb { R } } _ { i } - { \bf 1 } _ { m } \eta _ { i } ^ { \top } \widehat { \pmb { R } } _ { i } . } } \end{array}\tag{40}
$$

Equation (40) shows that anchor noise afects the aligned representations in two distinct ways. The centered component $\nu H W _ { i }$ perturbs the rotations estimated by the GOPP, whereas the mean component $\pmb { \eta } _ { i }$ produces a participantspecific residual translation. Exact alignment for every possible input would require both

$$
O _ { i } \widehat { R } _ { i } = O _ { j } \widehat { R } _ { j }\tag{41}
$$

and

$$
\pmb { \eta } _ { i } ^ { \top } \widehat { \pmb { R } } _ { i } = \pmb { \eta } _ { j } ^ { \top } \widehat { \pmb { R } } _ { j }\tag{42}
$$

for every $i , j \in { \mathcal { P } } .$ The noisy GOPP controls only the first condition. Therefore, even exact recovery of the rotations would not generally restore exact alignment.

The translation component admits an exact characterization independently of the GOPP rotation error.

Proposition 4.5 (Residual Translation Under Gaussian Anchor Noise). Under the Gaussian anchor-noise model above, suppose that the matrices $W _ { i } , i \in { \mathcal { S } } ,$ , are mutually independent and that the estimated rotations are computed exclusivelyfrom the centered anchor representations. Then

$$
\pmb { \eta } _ { i } \sim { \cal N } \bigg ( \mathbf { 0 } _ { d } , \frac { \nu ^ { 2 } } { r } { \cal I } _ { d } \bigg ) , \qquad i \in \mathcal { P } ,\tag{43}
$$

and the collection $\{ \pmb { \eta } _ { i } \} _ { i \in \mathcal { P } }$ is independent of the estimated rotations. Consequently, conditionally on the estimated rotations, the vectors $\widehat { \pmb { R } } _ { i } ^ { \top } \pmb { \eta } _ { i } , i \in \mathcal { P }$ , are independent and each follows

$$
\widetilde { \pmb { R } } _ { i } ^ { \top } \pmb { \eta } _ { i } \sim { \cal N } \left( \mathbf { 0 } _ { d } , \frac { \nu ^ { 2 } } { r } { \cal I } _ { d } \right) .\tag{44}
$$

For every fixed pair i , $j ,$

$$
\widehat { \pmb { R } } _ { i } ^ { \top } \pmb { \eta } _ { i } - \widehat { \pmb { R } } _ { j } ^ { \top } \pmb { \eta } _ { j } \sim N \bigg ( \pmb { 0 } _ { d } , \frac { 2 \nu ^ { 2 } } { r } \pmb { I } _ { d } \bigg ) ,\tag{45}
$$

and hence

$$
\mathbb { E } \bigg [ \bigg \| \widehat { \pmb { R } } _ { i } ^ { \top } \pmb { \eta } _ { i } - \widehat { \pmb { R } } _ { j } ^ { \top } \pmb { \eta } _ { j } \bigg \| _ { 2 } ^ { 2 } \bigg ] = \frac { 2 d \nu ^ { 2 } } { r } .\tag{46}
$$

Moreover, for every $0 < \zeta < 1$

$$
\left\| \widehat { R } _ { i } ^ { \top } \eta _ { i } - \widehat { R } _ { j } ^ { \top } \eta _ { j } \right\| _ { 2 } \leq \frac { \sqrt { 2 } \nu } { \sqrt { r } } \left( \sqrt { d } + \sqrt { 2 \log ( 1 / \zeta ) } \right)\tag{47}
$$

with probability at least $1 - \zeta .$

Thus, the root-mean-squared participant-to-participant translation disagreement is $\nu { \sqrt { 2 d / r } } .$ . Increasing the number of anchor rows reduces this component, whereas increasing the feature dimension or anchor-noise scale increases it.

We measure the rotational component of the alignment error by

$$
\mathcal { E } : = \operatorname* { m i n } _ { Q \in O \left( d \right) } \operatorname* { m a x } _ { i \in \mathcal { P } } \left\| O _ { i } \widehat { R } _ { i } - Q \right\| _ { F } .\tag{48}
$$

After transposition and restriction to the centered row subspace, the centered anchor observations form an instance of the normalized Gaussian GOPP studied in [30]. The following statement is a protocol-specific specialization of [30, Theorems 3.2 and 3.3], not a new general GOPP convergence theorem. The displayed constants result from substituting the centered orthonormal-anchor dimensions into the suficient conditions traced in the cited proofs.

Theorem 4.6 (Specialization of the GPM Guarantee to a Centered Orthonormal Anchor Matrix). Let $\gamma > 2 ,$ , and initialize the GPM using the spectral estimator prescribed in [30]. An explicit conservative suficient condition obtained from the proofof[30, Theorem 3.2] is

$$
\nu \leq \frac { 1 } { 1 4 3 3 6 } \frac { \sqrt { p } } { \sqrt { d } \left( \sqrt { p d } + \sqrt { r - 1 } + \sqrt { 2 \gamma p \log p } \right) } .\tag{49}
$$

Under this condition, with probability at least $1 - O ( p ^ { - \gamma + 2 } )$ , the GPM converges linearly to a global maximizer of Problem (28), which is unique modulo a common orthogonal transformation. Moreover, the limiting rotations satisfy

$$
\mathcal { E } \leq \frac { 6 } { 1 - 1 / 1 0 2 4 } \frac { \nu \sqrt { d } \left( \sqrt { p d } + \sqrt { r - 1 } + \sqrt { 2 \gamma p \log p } \right) } { \sqrt { p } } .\tag{50}
$$

Proposition 4.5 quantifies the residual translation, whereas Theorem 4.6 controls GPM convergence and the rotational error E. Consequently, convergence to the global optimizer of the noisy GOPP does not imply exact DC alignment when $\nu > 0$

Empirical convergence reference. The theoretical condition in Equation (49) is suficient but conservative. For normalized anchor matrices, the numerical experiments in [30, Section 4.2] identify an empirical transition near

$$
\nu _ { \mathrm { P T } } : = \frac { 1 . 8 9 \sqrt { p } } { \sqrt { p d } + \sqrt { r - 1 } + 2 \sqrt { p \log p } } .\tag{51}
$$

Below this transition, the GPM typically reaches a certifiably global solution in the experiments of [30, Section 4.2]; above it, the probability of doing so decreases sharply. We use v as an empirical reference for interpreting the iteration counts, fixed-point residuals, and GOPP residuals reported in Section 5. Since our implementation does no compute the semidefinite dual certificate used in [30], these diagnostics do not independently certify global optimality.

## 4.4. Reconstruction Risk in the Analyst-Participant Collusion Setting

We now depart from honest execution and analyze AA-I-GDP (anchor noise) in the analyst-participant collusion setting of Definition 2.9. Let $c \in \mathcal { P }$ denote the colluding participant. The analyst observes all uploaded private data and noisy anchor representations, $\{ ( Y _ { i } , B _ { i } ) : i \in \mathcal { P } \}$ , while participant c reveals its private dataset and local secret state. Because the participant-only setup distributes the raw anchor matrix A to every participant, participant c also reveals it under Definition 2.9. Fix any non-colluding target $i \in \mathcal { P } \ \backslash$ {c}. For this target, the attacker has access to

$$
\begin{array} { r } { Y _ { i } = X _ { i } O _ { i } + \mathbf { 1 } _ { n _ { i } } \Psi _ { i } ^ { \top } , \qquad \mathbf { B } _ { i } = A \mathbf { O } _ { i } + \mathbf { 1 } _ { r } \Psi _ { i } ^ { \top } + \nu W _ { i } , } \end{array}\tag{52}
$$

together with the raw anchor matrix A, but does not observe the target participant’s private data $X _ { i }$ or local secret state $( O _ { i } , \Psi _ { i } , W _ { i } )$ .

The comparison with AA-I-GDP is immediate. In AA-I-GDP, revealing A gives the attacker a noiseless knowninput pair $( A , B _ { i } )$ for the fixed non-colluding target $i \in \mathcal { P } \backslash \{ c \}$ , and Proposition 3.1 shows that this is suficient for exact reconstruction under the full-column-rank condition on the centered anchor matrix. In AA-I-GDP (anchor noise), the same disclosure gives the attacker only a noisy known-input pair because $\pmb { { \cal B } } _ { i } = A \pmb { { \cal O } } _ { i } + \pmb { 1 } , \Psi _ { i } ^ { \top } + \nu \pmb { W } _ { i }$ . Thus, AA-I-GDP (anchor noise) does not remove the anchor-based attack surface; rather, it turns the exact transform-recovery attack against AA-I-GDP into a noisy transform-estimation problem. When $\nu = 0 ,$ the known-anchor pair becomes noiseless, and AA-I-GDP (anchor noise) inherits the exact vulnerability of AA-I-GDP in the analyst-participant col lusion setting. When $\nu > 0 ,$ , transform-recovery accuracy under the proposed protocol depends on $\nu , r ,$ and d. For a general nonnormalized anchor matrix, it additionally depends on the singular values of A, as quantified below; the normalization in (22) fixes this scale and conditioning.

Define the anchor mean and the centered anchor matrix by

$$
\pmb { a } = \frac { 1 } { r } \pmb { A } ^ { \top } \pmb { 1 } _ { r } , \qquad \overline { { \pmb { A } } } = \pmb { A } - \pmb { 1 } _ { r } \pmb { a } ^ { \top } .\tag{53}
$$

Similarly, for the non-colluding target participant $i \in \mathcal { P } \ \backslash$ {c}, define

$$
\pmb { b } _ { i } = \frac { 1 } { r } \pmb { B } _ { i } ^ { \top } \pmb { 1 } _ { r } , \qquad \ \overline { { \pmb { B } } } _ { i } = \pmb { B } _ { i } - \pmb { 1 } _ { r } \pmb { b } _ { i } ^ { \top } .\tag{54}
$$

Writing

$$
\pmb { \eta } _ { i } = \frac { \nu } { r } \pmb { W } _ { i } ^ { \top } \mathbf { 1 } _ { r } , \qquad \overline { { \pmb { W } } } _ { i } = \pmb { H } _ { r } \pmb { W } _ { i } ,\tag{55}
$$

we have

$$
\pmb { b } _ { i } = \pmb { O } _ { i } ^ { \top } \pmb { a } + \Psi _ { i } + \pmb { \eta } _ { i } , \qquad \ \overline { { \pmb { B } } } _ { i } = \overline { { \pmb { A } } } \pmb { O } _ { i } + \nu \overline { { \pmb { W } } } _ { i } .\tag{56}
$$

Equation (56) is the noisy known-anchor relation available in the analyst-participant collusion setting. The translation $\Psi _ { i }$ and anchor-noise mean $\pmb { \eta } _ { i }$ are not separately identifiable from $\pmb { B } _ { i }$ . The attacks below, therefore, center the observed anchor pair to estimate $\pmb { O } _ { i }$ and use the observed uploaded-anchor mean as a plug-in estimate for the translation.

Moore–Penrose Estimator. The simplest attack ignores the orthogonality constraint on the target GDP parameter and solves the unconstrained least-squares problem

$$
\widehat { \pmb { O } } _ { i } \in \arg \operatorname* { m i n } _ { { \pmb { O } } \in \mathbb { R } ^ { d \times d } } \left\| \overline { { \pmb { B } } } _ { i } - \overline { { \pmb { A } } } \pmb { O } \right\| _ { F } ^ { 2 } .\tag{57}
$$

Its minimum-Frobenius-norm solution is

$$
\widehat { \pmb { O } } _ { i } = \overline { { \pmb { A } } } ^ { \dagger } \overline { { \pmb { B } } } _ { i } .\tag{58}
$$

This is the same pseudoinverse-based transform-recovery step used in the exact AA-I-GDP attack in the analystparticipant collusion setting, now applied to noisy centered anchor representations. Because $\widehat { \pmb { O } } _ { i }$ is not constrained to be orthogonal, or even invertible, the corresponding reconstruction uses the Moore–Penrose pseudoinverse. The attacker estimates the translation by

$$
\widehat { \Psi } _ { i } = \pmb { b } _ { i } - \left( \widehat { \pmb { O } } _ { i } \right) ^ { \top } \pmb { a } ,\tag{59}
$$

and reconstructs

$$
\begin{array} { r } { \widehat { \pmb { X } } _ { i } = \left( \pmb { Y } _ { i } - \mathbf { 1 } _ { n _ { i } } \left( \widehat { \boldsymbol { \Psi } } _ { i } \right) ^ { \top } \right) \left( \widehat { \pmb { O } } _ { i } \right) ^ { \dagger } . } \end{array}\tag{60}
$$

Orthogonal Procrustes Estimator. A structure-aware alternative enforces the structural constraint $\pmb { O } _ { i } \in O ( d )$ . It estimates the target orthogonal parameter by solving the orthogonal Procrustes problem [34]

$$
\widehat { \pmb { O } } _ { i } \in \arg \operatorname* { m i n } _ { { \pmb { O } } \in { \cal O } ( d ) } \left\| \overline { { \pmb { B } } } _ { i } - \overline { { \pmb { A } } } \pmb { O } \right\| _ { F } ^ { 2 } .\tag{61}
$$

If

$$
\overline { { { \pmb { A } } } } ^ { \top } \overline { { { \pmb { B } } } } _ { i } = U _ { i } { \pmb { \Sigma } } _ { i } V _ { i } ^ { \top }\tag{62}
$$

is a singular value decomposition, then one minimizer is

$$
{ \widehat { \pmb { O } } } _ { i } = { \pmb U } _ { i } { \pmb V } _ { i } ^ { \top } .\tag{63}
$$

The attacker then estimates the translation by

$$
{ \widehat { \Psi } } _ { i } = b _ { i } - \left( { \widehat { \pmb { O } } } _ { i } \right) ^ { \top } { \pmb { a } }\tag{64}
$$

and reconstructs

$$
\begin{array} { r } { \widehat { X } _ { i } = \left( Y _ { i } - \mathbf { 1 } _ { n _ { i } } \left( \widehat { \Psi } _ { i } \right) ^ { \top } \right) \left( \widehat { \pmb { O } } _ { i } \right) ^ { \top } . } \end{array}\tag{65}
$$

Unlike the Moore-Penrose estimator, the orthogonal-Procrustes estimator is nonlinear in the anchor noise because it enforces the constraint ${ \cal O } \in { \cal O } ( d )$ . Thus, it does not admit the same simple, exact, unbiasedness, and variance calculation.

Alignment Map Calibration Estimator. AA-I-GDP (anchor noise) also exposes information through the alignment rotations computed from the noisy anchor representations. In the noiseless exact-alignment regime, the rotations satisfy

$$
O _ { i } \widehat { R } _ { i } = Q , \qquad i \in \mathcal { P } ,
$$

for some global orthogonal ambiguity $Q \in O ( d )$ . This ambiguity is harmless for collaborative learning, but the colluding participant can calibrate it in the analyst-participant collusion setting. Specifically, the analyst knows $\widehat { R } _ { c } ,$ and participant c reveals $\mathbf { \nabla } _ { O _ { c } . }$ The alignment-map calibration estimator, therefore, solves

$$
\widehat { \pmb { Q } } \in \arg \operatorname* { m i n } _ { { \pmb { Q } } \in O ( d ) } \left\| { \pmb { O } } _ { c } \widehat { \pmb { R } } _ { c } - { \pmb { Q } } \right\| _ { F } ^ { 2 } .\tag{66}
$$

Equivalently,

$$
\widehat { \pmb { Q } } = \Pi \left( { \pmb { O } } _ { c } \widehat { \pmb { R } } _ { c } \right) = { \pmb { O } } _ { c } \widehat { \pmb { R } } _ { c } ,\tag{67}
$$

where Π is the orthogonal polar projection defined in (30). The target transform is then estimated by using the relation $\begin{array} { r } { O _ { i } \widehat { R } _ { i } \approx Q \colon } \end{array}$

$$
\widehat { \pmb { O } } _ { i } = \widehat { \pmb { Q } } \widehat { \pmb { R } } _ { i } ^ { \top } .\tag{68}
$$

Using the observed uploaded-anchor mean, the attacker estimates the translation by

$$
{ \widehat { \Psi } } _ { i } = b _ { i } - \left( { \widehat { \pmb { O } } } _ { i } \right) ^ { \top } { \pmb { a } }\tag{69}
$$

and reconstructs

$$
\begin{array} { r } { \widehat { X } _ { i } = \left( Y _ { i } - \mathbf { 1 } _ { n _ { i } } \left( \widehat { \Psi } _ { i } \right) ^ { \top } \right) \left( \widehat { \pmb { O } } _ { i } \right) ^ { \top } . } \end{array}\tag{70}
$$

This attack is included because alignment quality and reconstruction exposure in the analyst-participant collusion setting are not independent: more accurate alignment rotations may also provide more information about the participantspecific GDP transforms.

The three estimators are feasible under the stated threat model but are not exhaustive. Their strongest-attack envelope is therefore a lower bound on leakage, not a privacy guarantee.

## 5. Experimental Evaluation

This section evaluates the theoretical and methodological claims in the order that support our argument. We ask three questions: whether the exact collaboration-output and coupled-reconstruction relations in Section 3 are observed numerically; whether moving noise from private-data representations $Y _ { i }$ to anchor representations $\pmb { B } _ { i }$ improves the measured privacy–utility frontier under the evaluated attacks; and how the proposed method behaves as the participant count changes. The first question is primarily a theory-and-implementation check, while the latter two characterize deployment-specific operating points rather than formal privacy guarantees.

We use the MNIST dataset [35] as a controlled validation setting for the equivalence results and attack implementations, and the CelebA dataset [36] as the principal higher-dimensional privacy–utility evaluation. Image datasets are useful here because reconstruction leakage is directly interpretable through visual inspection and can also be quantified using independent image-recognition models. Dataset-specific preprocessing and partitioning are described in their respective subsections.

All experiments were implemented in Python and executed on Google Colab Pro+ with a high-memory runtime and an NVIDIA A100 GPU. PyTorch and NumPy were used for numerical computation and model training; pandas for data management; scikit-learn and scikit-image for evaluation metrics; Pillow for image preprocessing; and Matplotlib for visualization.

## 5.1. Shared Experimental Scope and Dimensional Reduction

Independent Component Analysis (ICA)-based reconstruction is a well-known threat to geometric data perturbation because square multiplicative transformations yield linear mixtures without reducing their dimensionality [20, 21]. Accordingly, non-square dimension-reducing transformations are commonly used in this literature, including in privacy-preserving collaborative learning [20, 37]. A detailed analysis and empirical evaluation of ICA-based attacks are outside the scope of this paper. Nevertheless, we apply a common rank-reducing transformation in every experiment because dimensionality reduction is standard in this literature and necessary to make learning from high-dimensional images computationally feasible. The transformation is fixed within each experiment, shared by all participants, and assumed to be known to the analyst and any attacker.

Let

$$
F \in \mathbb { R } ^ { d \times h } , \qquad \mathrm { r a n k } ( F ) = h < d ,\tag{71}
$$

denote the common linear projection matrix. Each participant forms

$$
\begin{array} { r } { \widetilde { X } _ { i } = X _ { i } { F } , \qquad \widetilde { X } = X { F } \in \mathbb { R } ^ { n \times h } . } \end{array}\tag{72}
$$

The connection to the preceding theoretical analysis is given by the substitution

$$
\begin{array} { r l } { ( X _ { i } , X , d ) } & { { } \longleftarrow \quad ( \widetilde { X } _ { i } , \widetilde { X } , h ) . } \end{array}\tag{73}
$$

Thus, all GDP and alignment operations in the experiments are performed in the projected space.

Let $\dot { \widetilde { X } } _ { i } \in \mathbb { R } ^ { n _ { i } \times h }$ denote the attacker’s estimate of participant i’s projected data $\widetilde { X } _ { i } .$ . Because the common projection matrix F is assumed known, the attacker reconstructs the corresponding original-space records as

$$
\widehat { \pmb { X } } _ { i } : = \widehat { \widetilde { \pmb { X } } } _ { i } \pmb { F } ^ { \dagger } .
$$

If the projected data are recovered exactly, so that

$$
{ \widehat { \widetilde { X } } } _ { i } = { \widetilde { X } } _ { i } ,
$$

then

$$
\widehat { X } _ { i } = \widetilde { X } _ { i } F ^ { \dagger } = X _ { i } F F ^ { \dagger } .
$$

Since $h < d ,$ the matrix $F F ^ { \dagger }$ is the orthogonal projector onto the column space of $F ,$ and therefore $X _ { i } F F ^ { \dagger }$ generally difers from X . We use $X _ { i } F F ^ { \dagger }$ as the projection-only reconstruction control when evaluating image leakage.

## 5.2. MNIST Dataset: Equivalence and Controlled Validation

## 5.2.1. Settings

We use the standard MNIST split. A fixed stratified subset of 10,000 images from the oficial 60,000-image training set is reserved for validation, and the remaining 50,000 images are used for training. The oficial 10,000- image test set is not used during calibration or model selection and is reserved for final evaluation. Images are normalized to [0<sub>,</sub> 1], flattened into d = 784-dimensional row vectors, and allocated among $p = 1 0$ participants using one fixed stratified-IID partition. We use the analyst-participant collusion setting: one participant colludes with the analyst, and a distinct participant is the target of reconstruction. Every participant and compared method uses the same public, data-independent Gaussian projection [38],

$$
\mathbf { \boldsymbol { F } } \in \mathbb { R } ^ { 7 8 4 \times 1 0 0 } , \qquad ( \mathbf { \boldsymbol { F } } ) _ { k \ell } \overset { \mathrm { i i d } } { \sim } N ( 0 , 1 ) , \qquad \widetilde { \mathbf { \boldsymbol { X } } } _ { i } = \mathbf { \boldsymbol { X } } _ { i } \mathbf { \boldsymbol { F } } .
$$

The projected dimension is $h = 1 0 0 .$ . The matrix $\boldsymbol { F }$ is generated once without accessing participant data or estimating a centering vector, and the same realization is fixed throughout the experiment.

We compare the following six methods. For participant i, let ${ \pmb O } _ { i } \in { \cal O } ( h )$ and $\Psi _ { i } \in \mathbb { R } ^ { h }$ denote its independently generated GDP parameters, let $\pmb { A } \in \mathbb { R } ^ { r \times h }$ denote the common anchor matrix, and let $s \in \mathcal { S }$ denote the fixed reference participant used for alignment. In each method containing additive noise, $W _ { i }$ has independent standard Gaussian entries, $( \pmb { W } _ { i } ) _ { k \ell } \overset { \mathrm { i i d } } { \sim } N ( 0 , 1 )$ , with dimensions specified below. In the matched comparisons, C-GDP uses the common parameters $( \pmb { O } _ { s } , \Psi _ { s } )$

Table 1 summarizes the private-data and anchor representations received by the analyst. A dash indicates that no anchor representation is uploaded. The participant-specific coordinate systems therefore remain unaligned under I-GDP, whereas the analyst uses $\{ { \pmb { { \cal B } } } _ { i } \} _ { i \in \mathcal { P } }$ to align the $_ \mathrm { A A - I - G D P }$ representations to the coordinate system of the reference participant s. For the private-data-noise methods, $\pmb { W } _ { i } \in \mathbb { R } ^ { n _ { i } \times h }$ and $\sigma \geq 0 \mathrm { { . } }$ ; for $_ \mathrm { A A - I - G D P }$ (anchor noise), $\pmb { W } _ { i } \in \mathbb { R } ^ { r \times h }$ and $\nu \geq 0$

Table 1: Representations received by the analyst in the MNIST comparison.
<table><tr><td>Method</td><td>Private-data upload  $Y _ { i }$ </td><td>Anchor upload  $\pmb { { \cal B } } _ { i }$ </td></tr><tr><td>C-GDP</td><td> $\widetilde { X } _ { i } \pmb { O } _ { s } + \mathbf { 1 } _ { n _ { i } } \Psi _ { s } ^ { \top }$ </td><td></td></tr><tr><td>C-GDP (private-data noise)</td><td> $\widetilde { X } _ { i } \pmb { \mathcal { O } } _ { s } + \mathbf { 1 } _ { n _ { i } } \Psi _ { s } ^ { \top } + \sigma \pmb { W } _ { i }$ </td><td></td></tr><tr><td>I-GDP</td><td> $\widetilde { X } _ { i } \pmb { O } _ { i } + \mathbf { 1 } _ { n _ { i } } \Psi _ { i } ^ { \top }$ </td><td></td></tr><tr><td>AA-I-GDP</td><td> $\widetilde { X } _ { i } \pmb { O } _ { i } + \mathbf { 1 } _ { n _ { i } } \Psi _ { i } ^ { \top }$ </td><td> $A O _ { i } + \mathbf { 1 } _ { r } \Psi _ { i } ^ { \top }$ </td></tr><tr><td>AA-I-GDP (private-data noise)</td><td> $\widetilde { X } _ { i } \pmb { \mathcal { O } } _ { i } + \mathbf { 1 } _ { n _ { i } } \Psi _ { i } ^ { \top } + \sigma \pmb { W } _ { i }$ </td><td> $A O _ { i } + \mathbf { 1 } _ { r } \Psi _ { i } ^ { \top }$ </td></tr><tr><td>AA-I-GDP (anchor noise)</td><td> $\widetilde { X } _ { i } \pmb { O } _ { i } + \mathbf { 1 } _ { n _ { i } } \Psi _ { i } ^ { \top }$ </td><td> $A O _ { i } + \mathbf { 1 } _ { r } \Psi _ { i } ^ { \top } + \nu \mathbf { W } _ { i }$ </td></tr></table>

The common anchor matrix satisfies the participant-only setup in (22). It is generated independently of the participant data, shared among all participants but withheld from the analyst during honest execution, and fixed across all methods and noise draws. Specifically, we sample $\pmb { \Omega } \in \mathbb { R } ^ { 4 0 0 \times 1 0 0 }$ with independent standard Gaussian entries and compute the sign-corrected thin QR factorization

$$
H _ { 4 0 0 } \Omega = A R _ { A } , H _ { 4 0 0 } = I _ { 4 0 0 } - \frac { 1 } { 4 0 0 } \mathbf { 1 } _ { 4 0 0 } \mathbf { 1 } _ { 4 0 0 } ^ { \top } .
$$

This construction gives $\ b { A } \in \mathbb { R } ^ { 4 0 0 \times 1 0 0 }$ satisfying

$$
\mathbf { 1 } _ { 4 0 0 } ^ { \top } A = \mathbf { 0 } _ { 1 0 0 } ^ { \top } , \qquad A ^ { \top } A = I _ { 1 0 0 } .
$$

For each private-data-noise protocol, we evaluate 200 observations over $0 \leq \sigma \leq 5 0 ;$ for AA-I-GDP (anchor noise), we evaluate 200 observations over $0 ~ \leq ~ \nu ~ \leq ~ 0 . 5$ . The deterministic endpoint and noiseless controls are retained separately and excluded from the spline fits, bootstrap resampling, and quantitative plots.

Utility is evaluated in the honest setting, in which the analyst observes only the representations prescribed by each method: $Y _ { i }$ for methods without anchor alignment and $( B _ { i } , Y _ { i } )$ for the AA-I-GDP variants. The analyst receives neither private participant records nor attacker-side information. For the anchor-aligned methods, $\pmb { B } _ { i }$ is used only to align the uploaded data representations before training. We quantify utility by test accuracy on the standard ten-class MNIST digit-recognition task.

Following the CNN architecture used in the I-GDP study [24], each h = 100-dimensional data representation is reshaped into a single-channel $1 0 \times 1 0$ array. This reshaping only arranges the projected features for convolution; the resulting array is not interpreted as a reconstructed image. To preserve the reference architecture, the array is symmetrically zero-padded by three entries on every side, producing a 16 × 16 input without introducing additional features or trainable parameters. The resulting architecture is summarized in Table 2.

A separate classifier is trained from scratch for every method and noise realization using cross-entropy loss and Adam with learning rate $1 0 ^ { - 3 }$ , batch size 256, and a maximum of 15 epochs. The checkpoint attaining the highest validation balanced accuracy is retained for final test evaluation. All fits use identical deterministic parameter initial ization and minibatch-order seeds, but do not share learned weights. Consequently, diferences in test accuracy reflect the perturbation method and noise realization rather than diferences in initialization or training order.

Privacy is evaluated in the analyst-participant collusion setting. Let c denote the colluding participant and $i \neq c$ the reconstruction target. The analyst observes every upload; the public projection matrix $\pmb { F }$ is known; and participant c discloses its projected records and its complete local secret state. For anchor-aligned methods, the colluder also reveals A. For C-GDP and C-GDP (private-data noise), the common parameters $( O _ { s } , \Psi _ { s } )$ are therefore known exactly,

Table 2: CNN architecture used to evaluate MNIST utility.
<table><tr><td>Stage</td><td>Operation</td><td>Output size</td></tr><tr><td>Reshape</td><td>100-dimensional row to one channel</td><td> $1 \times 1 0 \times 1 0$ </td></tr><tr><td>Zero padding</td><td>Three entries on every spatial boundary</td><td> $1 \times 1 6 \times 1 6$ </td></tr><tr><td>Convolution 1</td><td>10 valid 5 × 5 filters, stride 1, ReLU</td><td> $1 0 \times 1 2 \times 1 2$ </td></tr><tr><td>Max pooling 1</td><td> $2 \times 2$  pooling, stride 2</td><td> $1 0 \times 6 \times 6$ </td></tr><tr><td>Convolution 2</td><td>20 valid  $5 \times 5$  filters, stride 1, ReLU</td><td> $2 0 \times 2 \times 2$ </td></tr><tr><td>Max pooling 2</td><td>2 × 2 pooling, stride 2</td><td> $2 0 \times 1 \times 1$ </td></tr><tr><td>Flatten</td><td>Vectorization</td><td>20</td></tr><tr><td>Fully connected</td><td>20 → 50, ReLU</td><td>50</td></tr><tr><td>Output</td><td>50 → 10</td><td>10 logits</td></tr></table>

and we use only the known-secret inversion

$$
\widehat { \widetilde { X } } _ { i } = \left( Y _ { i } - \mathbf { 1 } _ { n _ { i } } \Psi _ { s } ^ { \top } \right) \pmb { O } _ { s } ^ { \top } = \left\{ \begin{array} { l l } { \widetilde { X } _ { i } , } & { \mathrm { C - G D P , } } \\ { \widetilde { X } _ { i } + \sigma W _ { i } O _ { s } ^ { \top } , } & { \mathrm { C - G D P ~ ( p r i v a t e - d a t a ~ n o i s e ) . } } \end{array} \right.\tag{74}
$$

I-GDP is retained solely as a utility baseline; attacks against I-GDP are outside the scope of this paper, so we do not assign it a reconstruction or leakage score. For AA-I-GDP and AA-I-GDP (private-data noise), disclosure of the centered orthonormal anchor matrix gives

$$
\pmb { b } _ { i } = \frac { 1 } { r } \pmb { B } _ { i } ^ { \top } \pmb { 1 } _ { r } , \qquad \pmb { \overline { { B } } } _ { i } = \pmb { B } _ { i } - \pmb { 1 } _ { r } \pmb { b } _ { i } ^ { \top } , \qquad \pmb { \widehat { O } } _ { i } = \pmb { A } ^ { \top } \pmb { \overline { { B } } } _ { i } ,\tag{75}
$$

and hence

$$
\widehat { \widetilde { X } } _ { i } = \left( Y _ { i } - \mathbf { 1 } _ { n _ { i } } b _ { i } ^ { \top } \right) \widehat { O } _ { i } ^ { \top } = \left\{ \begin{array} { l l } { \widetilde { X } _ { i } , } & { \mathrm { A A \mathrm { - } I \mathrm { - } G D P , } } \\ { \widetilde { X } _ { i } + \sigma W _ { i } O _ { i } ^ { \top } , } & { \mathrm { A A \mathrm { - } I \mathrm { - } G D P ~ ( p r i v a t e \mathrm { - } d a t a ~ n o i s e ) . } } \end{array} \right.\tag{76}
$$

For AA-I-GDP (anchor noise), we evaluate the Moore–Penrose (MP), Orthogonal-Procrustes (OP), and alignmentmap-calibration (AM) estimators introduced in Subsection 4.4:

$$
\begin{array} { r l r } { \widehat { \widetilde { X } } _ { i } ^ { \mathrm { M P } } = \left( Y _ { i } - \mathbf { 1 } _ { n } b _ { i } ^ { \top } \right) \left( A ^ { \top } \overline { { B } } _ { i } \right) ^ { \sharp } , } & { \ \widehat { \widetilde { X } } _ { i } ^ { \mathrm { Q P } } = \left( Y _ { i } - \mathbf { 1 } _ { n } b _ { i } ^ { \top } \right) \left( \Pi \big ( A ^ { \top } \overline { { B } } _ { i } \big ) \right) ^ { \top } , } & { \ \widehat { \widetilde { X } } _ { i } ^ { \mathrm { A M } } = \left( Y _ { i } - \mathbf { 1 } _ { n } b _ { i } ^ { \top } \right) \left( \Pi \big ( { \cal O } _ { \mathcal { C } } \widehat { R } _ { \mathcal { C } } \big ) \widehat { R } _ { i } ^ { \top } \right) ^ { \top } . } \end{array}\tag{77}
$$

Each applicable reduced-space estimate is decoded as

$$
\widehat { X } _ { i } = \widehat { \widetilde { X } } _ { i } F ^ { \dagger } .\tag{78}
$$

Privacy is assessed visually and via exact-source linkage by an independent auditor, trained once on clean $2 8 \times 2 8$ MNIST images and then frozen. For each reconstructed target image, a ten-way gallery contains its clean source image and nine distinct clean images with the same digit label. The auditor maps the reconstruction and all candidates to normalized 128-dimensional penultimate-layer embeddings, and selects the candidate with the largest cosine simi larity. Exact-source linkage accuracy is the proportion of the 1,000 fixed queries for which the true source is selected. Because every candidate in a gallery has the same digit label, the analyst-visible target label does not distinguish the source candidates. Random guessing yields an accuracy of 0 10; higher accuracy indicates greater source-specific leakage and therefore weaker privacy. The auditor architecture is summarized in Table 3.

The ten-class head is used only to train the auditor; privacy evaluation uses the frozen 128-dimensional embedding.

## 5.2.2. Results

Figures 2 and 3 summarize privacy in the analyst-participant collusion setting. Figure 2 reports exact-source linkage accuracy for 200 observations per noisy protocol over $0 \leq \sigma \leq 5 0$ for private-data noise and $0 \leq \nu \leq 0 . 5$ for anchor noise. Each point is the proportion of 1,000 ten-way same-digit galleries in which the frozen auditor links a reconstructed query to its true clean source. Panel (a) shows the shared C-GDP known-secret and AA-I-GDP known-anchor private-data-noise trajectory, whose paired observations coincide exactly, while panel (b) overlays the MP, OP, and AM attacks against AA-I-GDP (anchor noise). Curves are descriptive fixed cubic regression-spline fits with five equally spaced interior knots, and shaded regions are 95% pointwise percentile-bootstrap bands from 10,000 resamples. Dash-dotted horizontal lines indicate the corresponding noiseless controls, while the gray dotted 10% line indicates uniform guessing. Figure 3 provides corresponding qualitative examples; fixed-digit grids for digits 0–9 are provided in the supplementary material.

Table 3: CNN architecture used to train the frozen MNIST linkage auditor.
<table><tr><td>Stage</td><td>Operation</td><td>Output size</td></tr><tr><td>Input</td><td>Clean or reconstructed grayscale image</td><td> $1 \times 2 8 \times 2 8$ </td></tr><tr><td>Convolution 1</td><td> $3 2 3 \times 3$  filters, padding 1, ReLU</td><td> $3 2 \times 2 8 \times 2 8$ </td></tr><tr><td>Convolution 2</td><td>64 3 × 3 filters, padding 1, ReLU</td><td> $6 4 \times 2 8 \times 2 8$ </td></tr><tr><td>Max pooling 1</td><td> $2 \times 2$  pooling, stride 2</td><td> $6 4 \times 1 4 \times 1 4$ </td></tr><tr><td>Convolution 3</td><td> $6 4 3 \times 3$  filters, padding 1, ReLU</td><td> $6 4 \times 1 4 \times 1 4$ </td></tr><tr><td>Max pooling 2</td><td>2 × 2 pooling, stride 2</td><td> $6 4 \times 7 \times 7$ </td></tr><tr><td>Flatten</td><td>Vectorization</td><td>3136</td></tr><tr><td>Embedding</td><td>3136 → 128, ReLU</td><td>128</td></tr><tr><td>Training head</td><td> $1 2 8  1 0$ </td><td>10 logits</td></tr></table>

The results support the predicted collaboration-output and reconstruction agreement under the stated coupling. The noiseless known-secret and known-anchor controls both attain 36 7% exact-source linkage, the projection-limited level after decoding through $F ^ { \dagger }$ . All 200 paired private-data-noise observations also have identical noise scales and linkage values, so panel (a) displays them as one shared marker set and fitted curve. The shared fit decreases to 14 39% at $\sigma = 2 0$ and 11 11% at $\sigma = 5 0 ;$ at the latter value, its 95% pointwise band is 10 46–11 84%. It therefore approaches, but does not cross, the 10% chance level within the calibrated range. The noiseless C-GDP and AA-I-GDP rows in Figure 3 likewise recover the projection-only control defined above, as predicted by Proposition 2.12. The diferent anchor representations remain part of the AA-I-GDP transcript.

Among the three attacks against AA-I-GDP (anchor noise), OP is the strongest evaluated attack: its mean exactsource linkage is 20 44%, compared with 15 61% for MP and 16 01% for AM, and it attains the pointwise maximum in 187 of the 200 draws, including ties. $\mathrm { A t } \ \nu = 0 . 1 5$ , the fitted linkage accuracies are 9 88%, 17 17%, and 11 01% for MP, OP, and AM, respectively. The OP fit then decreases to 14 04% at $\nu = 0 . 2 0$ , 12 13% at $\nu = 0 . 3 0$ , and 11 59% at $\nu = 0 . 5 0$ . Its 95% pointwise band at $\nu = 0 . 5 0$ is 9 14–13 09%, which includes chance, although the fitted OP curve itself does not cross 10% within the evaluated range. MP and AM reach the chance region earlier, but those attacks are weaker; values below 10% are interpreted as finite-gallery variation rather than performance below random guessing.

Figure 4 reports held-out test balanced accuracy for 200 observations per noisy protocol over $0 \leq \sigma \leq 5 0$ for private-data noise and $0 \leq \nu \leq 0 . 5$ for anchor noise. The curves are descriptive fixed cubic regression-spline fits with five equally spaced interior knots, and the shaded regions are 95% pointwise percentile-bootstrap bands from 10,000 resamples. These bands describe sensitivity to noise in the observations conditional on the fixed deployment, rather than uncertainty across alternative deployments. Horizontal lines denote random guessing, the noiseless C-GDP/AA-I-GDP reference, and noiseless I-GDP. In panel (b), filled markers denote runs that reached the numerical GPM stopping criterion before 500 iterations, whereas open markers denote the returned iterate at the iteration limit. This diagnostic records numerical termination and does not, by itself, certify global optimality.

The noiseless C-GDP and AA-I-GDP controls both attain 90 9641% balanced accuracy, compared with 56 3685% for I-GDP. Under private-data noise, the paired analyst-side data matrices produced by C-GDP and AA-I-GDP agree up to a maximum relative Frobenius discrepancy of $6 . 0 \times 1 0 ^ { - 1 4 }$ . Their separately trained classifiers consequently produce fitted curves that are numerically indistinguishable at the scale of the experiment: at $\sigma = 5 .$ , the fitted accuracies are 82 0035% for C-GDP and 82 0022% for AA-I-GDP, and at $\sigma = 1 0$ they are 62 2566% and 62 2640%, respectively. The remaining fit-level diferences arise from finite-precision computation and validation-checkpoint selection, rather than a substantive diference between the protocols. $\operatorname { A s } \sigma$ increases, the additive private-data noise dominates the digit signal, and both methods approach the 10% random-guessing level. Their fits cross the I-GDP reference at approximately $\sigma = 1 1 . 6 8$ and attain 11 0668% and 10 9004%, respectively, at $\sigma = 5 0$

Panel (b) shows a diferent response to anchor noise. The fitted accuracy is 87 5476% at $\nu = 0 . 0 5 , 7 4 . 7 2 8 1 \%$ at $\nu \ = \ 0 . 1 0 .$ , 64 1547% at $\nu \ = \ 0 . 1 5 .$ and 60 4892% at $\nu ~ = ~ 0 . 2 0$ , after which it remains near 59% over most of $0 . 3 \leq \nu \leq 0 . 5$ . Anchor noise does not directly corrupt the uploaded private-data matrices; instead, it progressively removes participant-specific transformation information from the estimated alignments. Over the evaluated range, the resulting utility therefore approaches the independently transformed I-GDP regime rather than random guessing, with the observed plateau lying close to the 56 3685% I-GDP reference. This is a finite-range empirical pattern rather than an asymptotic equality, because the residual translation error also increases with v and may further reduce utility outside the calibrated range.

![](images/06eea1ff0088b52a32f293819f81da27f48f24bd17dc452f8c6c0427c03d2b70.jpg)  
Figure 2: MNIST exact-source linkage in the analyst-participant collusion setting. Panel (a) shows the exactly coincident C-GDP and AA-I-GDP private-data-noise attacks over $0 \leq \sigma \leq 5 0$ as one shared trajectory, while panel (b) overlays the MP, OP, and AM attacks against AA-I-GDP (anchor noise) over $0 \leq \nu \leq 0 . 5 .$ Each series contains 200 observations, and each observation is evaluated on the same 1,000 fixed ten-way same-digit galleries. Curves are descriptive fixed cubic regression-spline fits, and shaded regions are 95% pointwise percentile-bootstrap bands from 10,000 resamples. Deterministic endpoint controls are excluded. Horizontal lines denote the noiseless controls and 10% random guessing.

For $p = 1 0 , h = 1 0 0$ , and $r = 4 0 0$ , substituting d ← h into Equation (51) gives the empirical phase-transition reference $\nu _ { \mathrm { P T } } = 0 . 0 9 7 7$ reported in [30, Section 4.2]. Of the 200 anchor-noise observations, 50 reached the numerical stopping criterion before 500 iterations, and 150 returned capped iterates. The transition is concentrated near $\nu = 0 . 0 8 \mathrm { : }$ : the largest noise level that reached the criterion is 0 0809, the smallest capped level is 0 0801, all 50 runs that reached the criterion lie below v , and no run above v reached it. The observed transition is therefore qualitatively consistent with the scale and direction of the cited empirical reference, although it occurs slightly earlier in this deployment. The experiments in [30, Section 4.2] concern solutions subsequently certified as globally optimal, whereas our implementation records only a step-based stopping criterion and does not compute the corresponding certificate; moreover, the suficient condition in Equation (49) lies several orders of magnitude below the sampled nonzero range. The utility curve declines smoothly through both the observed stopping transition and v , attaining approximately 80 1% at $\nu = 0 . 0 8$ and 75 4% at v = 0 0975. Thus, the stopping transition characterizes solver behavior but does not appear as a discrete accuracy threshold: capped iterates remain useful, and the experiment does not isolate a causal efect of global convergence on utility because increasing v simultaneously worsens alignment and makes numerical convergence more dificult.

Figure 5 fits utility directly as a function of exact-source linkage using 200 observations per displayed method. The private-data-noise trajectory reports C-GDP only because AA-I-GDP has exactly the same linkage values and a numerically indistinguishable utility curve under the matched construction; both utility series remain visible in Figure 4(a). I-GDP is shown only as a horizontal utility reference because attacks against I-GDP are outside the scope of this paper, and no reconstruction or leakage score is assigned to it. OP is displayed for anchor noise because it is the strongest of the three evaluated reconstruction attacks over most of the privacy transition.

Relocating noise from the private-data representations to the anchor representations materially changes the lowleakage trade-of in Figure 5. At 12 4% exact-source linkage, the fitted balanced accuracy is 59 2% for anchor noise, with a 95% pointwise band of 58 4–60 0%, compared with 15 1% and 13 9–16 4% for C-GDP private-data noise. At 20% linkage, the fitted accuracies are 68 2% and 58 5%, an anchor-noise advantage of 9 7 percentage points. At 30%, the estimates are 85 5% and 83 4%, and the pointwise bands overlap. Thus, anchor noise preserves materially greater utility in the privacy-relevant low-linkage region under the evaluated fixed deployment, while the diference narrows as leakage increases. This conclusion includes all returned solver outputs: 50 of the 200 anchor-noise runs reached the stopping criterion before 500 iterations, and 150 returned capped iterates.

![](images/f937f2cedc9522078ed01fb7808ae68ddf14bf4b9a4f8af85b549ff5bfe9e075.jpg)  
Figure 3: Representative MNIST reconstructions in the analyst-participant collusion setting. Columns correspond to digits 0–9 and use distinct source records. Within every noisy row, the leftmost reconstruction is the zero-noise control, and the remaining reconstructions are selected so that full-test exact-source linkage decreases from left to right, ending at an observed chance or chance-compatible value. Linkage percentages shown beneath the images are computed over the complete 1,000-query linkage evaluation and are not per-image probabilities. The noiseless C-GDP and AA-I-GDP attacks recover the projection-only control $X _ { i } F \bar { F } ^ { \dagger }$ . Fixed-digit sweeps organized by the same exact-source criterion are provided in the supplementary material.

## 5.3. CelebA Dataset: Main Privacy–Utility Evaluation

## 5.3.1. Settings

Each CelebA image is center-cropped, bilinearly resized to 64 × 64 RGB, normalized to [0 1], and flattened into a d = 12 288-dimensional row vector. We reserve exactly 10,000 images whose identities are disjoint from those of all participants as a public dimensional-reduction pool. An uncentered rank-400 truncated SVD is fitted to this pool, and the first $h = 4 0 0$ orthonormal right singular vectors form the common matrix $\pmb { F } \in \mathbb { R } ^ { 1 2 , 2 8 8 \times 4 0 0 }$ , for which ${ \pmb F } ^ { \top } { \pmb F } = { \pmb I } _ { 4 0 0 }$ Every participant forms ${ \widetilde { \pmb { X } } } _ { i } = { \pmb { X } } _ { i } { \pmb { F } }$ , and the resulting 400-dimensional rows are reshaped into single-channel $2 0 \times 2 0$ arrays for downstream learning. The inverse used by every reconstruction attack is $F ^ { \dagger } = F ^ { \dagger }$ , so exact projected-space recovery produces the public-SVD reconstruction floor $X _ { i } F F ^ { \top }$

![](images/74b095742def3b268c064d1848a31a31a49134396de68ba8298b3b522571efc3.jpg)  
Figure 4: MNIST utility under private data and anchor noise. Panel (a) reports separate C-GDP and AA-I-GDP private-data-noise observations over $0 \leq \sigma \leq 5 0 .$ and panel (b) reports AA-I-GDP (anchor noise) over $) \leq \nu \leq 0 . 5$ . Each noisy protocol contains 200 observations. Curves are descriptive fixed cubic regression-spline fits, and shaded regions are 95% pointwise percentile-bootstrap bands from 10,000 resamples. Deterministic endpoint controls are excluded. Filled anchor-noise markers met the GPM convergence criterion, while open markers reached the 500-iteration limit. Horizontal lines indicate random guessing, noiseless C-GDP/AA-I-GDP, and noiseless I-GDP.

![](images/8ae64bbd36e5219d40a4a4d31165c2219a68182e69ee60fccb148bffa930289b.jpg)  
Figure 5: MNIST privacy–utility trade-of in the analyst-participant collusion setting. The horizontal axis shows exact-source linkage accuracy, and the vertical axis shows held-out test balanced accuracy, so the preferred region is the upper-left. Each faint marker pairs leakage and utility from one of 200 observations. The blue curve reports C-GDP (private-data noise), and the dashed orange curve reports AA-I-GDP (anchor noise) under the OP attack. Each curve is a fixed cubic regression spline of balanced accuracy on exact-source linkage with five equally spaced interior knots; shaded regions are 95% pointwise percentile-bootstrap bands from 10,000 paired resamples. The star is the noiseless C-GDP/AA-I-GDP benchmark, the dash-dotted horizontal line is I-GDP utility, and the dotted lines mark 10% random guessing. Filled OP markers met the GPM stopping criterion, while hollow markers reached the 500-iteration limit.

We use one fixed identity-disjoint allocation with $p = 1 0$ participants and 100 identities per participant. Within each participant, 70 identities are assigned to training, 15 to validation, and 15 to testing; all images of an identity remain in its assigned split, and identities are disjoint across both participants and splits. Before forming the protocol data, one image from each selected identity is reserved exclusively as a clean privacy reference and is never included in the uploaded representations or downstream training. In the analyst-participant collusion setting, participant 1 (implementation index 0) colludes with the analyst, and participant 2 (implementation index 1) is the reconstruction target. The final linkage evaluation uses one protected query from each of the target participant’s 100 identities, covering the training, validation, and test identity splits.

The MNIST results in Subsection 5.2, together with Proposition 2.12 and Corollary 3.3, show that noiseless AA-I-GDP and AA-I-GDP (private-data noise) do not provide distinct analyst-side collaboration-output baselines from their C-GDP counterparts under the stated full-rank, gauge, and coupling conditions. We therefore compare C-GDP, I-GDP, C-GDP (private-data noise), and AA-I-GDP (anchor noise). The common anchor matrix satisfies the participant-only setup in (22): a participant-side Gaussian matrix Ω ∈ R<sup>1600×400</sup> is sampled once and centered and orthonormalized to obtain A, which is shared among all participants but withheld from the analyst during honest execution. Thus, $\mathbf { 1 } _ { 1 6 0 0 } ^ { \top } \mathbf { A } = \mathbf { 0 } ^ { \top }$ and $A ^ { \top } A = I _ { 4 0 0 }$ . Each noisy method is evaluated using 200 observations: $0 \leq \sigma \leq 3$ for C-GDP (private-data noise) and $0 \leq \nu \leq 0 . 2 5$ for AA-I-GDP (anchor noise).

Utility is the held-out balanced accuracy for predicting the CelebA Smiling attribute in the honest setting, where the analyst receives only the method-appropriate representations. The 20 × 20 representation is processed by the same reference architecture used in Subsection 5.2. The checkpoint with the highest validation balanced accuracy is used once for test evaluation.

Privacy is evaluated in the analyst-participant collusion setting after decoding every recovered projected representation with F<sup>†</sup>. C-GDP is evaluated by exact known-secret inversion, C-GDP (private-data noise) by its known-secret inversion, and AA-I-GDP (anchor noise) by the MP, OP, and AM attacks defined above. I-GDP is retained solely as a utility baseline; attacks against I-GDP are outside the scope of this paper, so we do not assign it a reconstruction or identity-linkage score. A frozen Inception-ResNet-v1 model [39], pretrained on VGGFace2 [40], maps each clean or reconstructed face to a normalized 512-dimensional embedding. Cross-image identity linkage matches each reconstructed query against a ten-way gallery containing a diferent reserved image of the same identity and nine reserved images of other identities. We generate ten deterministic galleries per identity, yielding 1,000 linkage episodes per attack; random guessing is 10%, and higher accuracy indicates greater identity leakage.

## 5.3.2. Results

Figure 6 reports privacy, Figure 7 reports utility, and Figure 8 reports their joint relationship. Each noisy method contributes 200 observations over $0 \leq \sigma \leq 3$ for private-data noise or $0 \leq \nu \leq 0 . 2 5$ for anchor noise. The curves are descriptive fixed-cubic regression-spline fits with five equally spaced interior knots, and the shaded regions are 95% pointwise percentile bootstrap bands from 10,000 multinomial resamples of the 200 observations. Endpoint and equivalence controls are excluded from fitting and are not displayed as endpoint markers. Filled anchor-noise points met the recorded GPM convergence criterion, whereas open points returned the iterate at the 500-iteration limit. Figure 9 reports the solver diagnostics, and Figure 10 provides a qualitative comparison.

Figure 6 shows 81 6% cross-image identity linkage for the no-noise C-GDP and AA-I-GDP controls. The privatedata-noise fit is 18 3% at $\sigma = 1 . 5$ and 9 9% at $\sigma = 3$ , crossing the 10% level at approximately $\sigma = 2 . 9 8$ . Anchor noise reaches the same level much earlier: the MP, alignment-map, and OP fits cross 10% at approximately $\nu = 0 . 0 3 7$ 0 039, and 0 059, respectively. OP therefore remains the most conservative evaluated anchor-noise attack around the privacy transition; fitted values slightly below 10% should be interpreted as finite-gallery variation around chance.

Figure 7 shows that the noiseless C-GDP and $\nu = 0 \mathrm { \ A A \mathrm { - } I \mathrm { - } G D P }$ controls attain 86 37% and 86 22% balanced accuracy, respectively, while I-GDP attains 71 93%. The private-data-noise fit is 70 9% at $\sigma = 1 . 5$ and 59 6% at $\sigma = 3 ,$ showing the additional utility cost required to bring private-noise identity linkage to chance. The anchor-noise fit decreases sharply near the origin and then remains near 70–72% over much of the remaining range. At the OP chance crossing $\nu \approx 0 . 0 5 8 7$ , its fitted balanced accuracy is 73 6%, approximately 1 7 percentage points above I-GDP.

Figure 8 shows that relocating noise from the private-data representations to the anchor representations materially changes the fitted trade-of. At 75% fitted balanced accuracy, C-GDP (private-data noise) retains 22 5% identity linkage, whereas AA-I-GDP (anchor noise) under OP retains 11 4%. At the 71 93% I-GDP utility reference, the corresponding fitted values are 19 2% and 8 6%, respectively. Conversely, at 10% fitted linkage, private-data noise retains only 59 7% balanced accuracy, compared with 73 6% for anchor noise. Thus, under the evaluated attacks and fixed deployment, anchor noise reaches chance-level identity linkage with substantially higher downstream utility.

![](images/6e61ebf1ab7a99716c2aaa8399cf215d00c0634de8cdd6a3a678d96634808372.jpg)  
Figure 6: CelebA cross-image identity linkage in the analyst-participant collusion setting. Panel (a) reports C-GDP (private noise) under knownsecret inversion, whereas panel (b) compares AA-I-GDP (anchor noise) under the MP, OP, and AM attacks. Each noisy protocol contains 200 observations over $0 \leq \sigma \leq 3$ or $0 \leq \nu \leq 0 . 2 5$ . Curves are fixed cubic regression-spline fits; shaded regions are 95% pointwise bands from 10,000 multinomial bootstrap resamples; the dotted 10% line denotes random guessing; and the dash-dotted lines give the corresponding no-noise controls. In panel (b), filled markers indicate runs for which the GPM satisfied its convergence criterion, whereas open markers indicate iterates returned at the 500-iteration limit.

Figure 9 shows that the GPM met the $1 0 ^ { - 7 }$ stopping criterion in 84 of the 200 anchor-noise runs and reached the 500-iteration limit in the remaining 116. The largest converged observation has $\nu = 0 . 0 3 9 5$ , and the smallest capped observation has $\nu = 0 . 0 3 9 0$ , indicating a narrow overlapping transition rather than a strict threshold. Median optimization time is 6 4 seconds for converged runs and 189 1 seconds for capped runs; the capped fixed-point residual has median $3 . 9 \times 1 0 ^ { - 4 }$ . All returned iterates remain in the privacy and utility analyses because they are outputs of the implemented protocol.

Figure 10 visualizes representative reconstructions for $0 \leq \sigma \leq 1 . 5$ and $0 \leq \nu \leq 0 . 2 5$ . Comparisons are made vertically within each column because each column uses a diferent source face; the left-to-right ordering reflects decreasing identity leakage rather than equal increments on a common noise scale. Within the displayed private-datanoise range, C-GDP retains recognizable facial structure and remains above chance, whereas the anchor-noise attacks reach chance-level linkage. Over the complete quantitative range $0 \leq \sigma \leq 3$ , private-data noise reaches approximately 10% linkage near $\sigma = 3$ , but with fitted balanced accuracy reduced to about 59 7%, compared with 73 6% for anchor noise at its OP chance crossing. The supplementary material provides a separate reconstruction grid for each identity shown in Figure 10.

## 5.3.3. Descriptive Sensitivity to the Number of Participants

We repeat the CelebA experiment with p ∈ {2 5 10 20 50} using nested prefixes of the same frozen 50-participant identity allocation, GDP-parameter bank, public SVD, and anchor matrix. Every participant retains 100 identities with the same 70/15/15 identity-level allocation for training, validation, and testing, while participant 1 remains the sole colluder and participant 2 remains the reconstruction target. For each participant count, AA-I-GDP (anchor noise) is evaluated using 100 draws with v ∼ Uniform(0 0 1); the noise amplitudes and random seeds are paired across participant counts by draw index. Because the participant-level identity budget is fixed, the total training set size scales with $p .$ This is therefore a growing-deployment sensitivity experiment rather than a fixed-sample estimate of the isolated causal efect of $p .$ Moreover, the same value of v need not yield the same realized privacy level across diferent participant counts, so we compare utility based on the observed identity-linkage accuracy rather than only against v.

![](images/806b778f20ba8dd3d7f3ec7c376fac0ecf2e61b2ef33ab8f0262bdbd67559e44.jpg)  
Figure 7: CelebA utility over the evaluated noise ranges. Panel (a) reports C-GDP (private-data noise) for $0 \leq \sigma \leq 3 ,$ and panel (b) reports AA-I-GDP (anchor noise) for $0 \leq \nu \leq 0 . 2 5 .$ Each panel contains 200 observations. Curves are fixed cubic regression-spline fits, and shaded regions are 95% pointwise bands from 10,000 multinomial bootstrap resamples. Horizontal lines indicate balanced random guessing, noiseless C-GDP or AA-I-GDP, and noiseless I-GDP. Filled anchor-noise markers met the GPM convergence criterion; open markers reached the 500-iteration limit.

Figure 11 uses identity-linkage attack accuracy on the horizontal axis and balanced accuracy on the vertical axis, so the preferred region is the upper left. The protected query identities, episode count, and ten-way task are fixed across $p ,$ but the distractor identities are drawn from each active participant pool, and the total training-set size grows with $p .$ The comparison is consequently descriptive rather than an exactly controlled estimate of the isolated efect of participant count. C-GDP is shown at its observed participant-specific leakage–utility coordinate, whereas I-GDP remains a horizontal utility reference because attacks against I-GDP are outside the scope of this paper, and no linkage score is assigned to it.

In the low-noise regime, AA-I-GDP (anchor noise) follows the natural order of growing data. $\mathrm { A t } \ \nu = 0 . 0 1$ , its fitted balanced accuracies for $p = 2 , 5 , 1 0 , 2 0 , 5 0$ are 73 67%, 80 04%, 83 40%, 85 40%, and 86 93%, respectively, while fitted linkage remains within the narrow range 52 20%–54 61%. The same ordering appears at matched privacy levels: at 50% fitted linkage, the corresponding utilities are 73 47%, 79 94%, 83 32%, 85 23%, and 86 70%, and at 20% linkage they are 69 61%, 76 64%, 80 14%, 81 03%, and 82 05%. Thus, when the anchor representations remain suficiently informative for alignment, the additional participant data translates into higher downstream utility in the expected order.

The I-GDP controls do not follow this data-size ordering: their balanced accuracies rank as $p \ = \ 5 \ ( 6 9 . 8 5 \% )$ p = 10 (68 92%), p = 20 (65 93%), p = 2 (64 89%), and $p = 5 0 ( 6 1 . 8 5 \% )$ . This non-monotonicity is consistent with the fact that increasing p under I-GDP adds both observations and independently transformed participant blocks, so a conventional shared-coordinate learner need not benefit monotonically from the larger dataset. Low anchor noise restores enough comparability for the growing-data benefit to dominate. Near $\nu = 0 . 1$ , however, every anchor-noise fit lies within 1 56 percentage points of its corresponding I-GDP reference, and the natural participant-count ordering disappears. This finite-range behavior supports the interpretation that increasing anchor noise progressively removes the utility benefit of alignment rather than directly corrupting every private-data representation. Because identity linkage is already near its 10% chance level in this regime, the trajectories compress near the left edge; observed or fitted values below 10% are not interpreted as privacy better than random guessing.

For h = 400 and r = 1600, Equation (51) gives the empirical references $\nu _ { \mathrm { P T } } ~ = ~ 0 . 0 3 7 8 5 , ~ 0 . 0 4 6 7 6 , ~ 0 . 0 5 2 9 7 ,$

![](images/449cd453a2c8e0088fc27e789c3c5174be1a16199f44845f26780afe3012c06e.jpg)  
Figure 8: CelebA privacy–utility trade-of in the analyst-participant collusion setting. The horizontal axis shows identity-linkage attack accuracy, and the vertical axis shows held-out Smiling balanced accuracy; the preferred region is the upper left. The blue trajectory reports C-GDP (privatedata noise) under known-secret inversion, and the dashed orange trajectory reports AA-I-GDP (anchor noise) under the OP attack. Each trajectory is fitted to 200 paired leakage-utility observations. The trajectories pair the noise-conditioned cubic-spline estimates, and the shaded envelopes combine the corresponding 95% pointwise bands from 10,000 multinomial bootstrap resamples. The star gives noiseless C-GDP, the dash-dotted horizontal line gives I-GDP utility, and the dotted vertical and horizontal lines indicate random guessing.

![](images/b578bfe248bcbbe693f030027695b8001565ae1f5c84c4357618b85f405b3d16.jpg)

![](images/56d631d16acab869fd7cccb88ebb60aeb020f327854aae0f01ada1c10bdcd29a.jpg)

![](images/b8e6d04a5ac5349c9a73df42001ea269db85e090dd02e1ef774e9d8dda83a733.jpg)  
Figure 9: GPM diagnostics for the 200 AA-I-GDP anchor-noise observations over $0 < \nu < 0 . 2 5 .$ From left to right, the panels report the iteration count, the fixed-point residual, and the optimization time. Filled points are the 84 runs that met the convergence criterion; open points are the 116 runs that reached the 500-iteration limit.

0 05833, and 0 06383 for $\begin{array} { r } { p = 2 , 5 , 1 0 , 2 0 , 5 0 . } \end{array}$ , respectively. Of the 100 draws at each participant count, 23, 26, 32, 34, and 40 meet the numerical GPM stopping criterion before the 500-iteration limit. The upward movement of the observed transition with $p$ is consistent with the participant-count scaling of the empirical reference in [30, Section 4.2]. Nevertheless, every stopped run lies below its corresponding v , while 7, 12, 12, 19, and 15 sub-reference runs still reach the iteration cap. The empirical reference, therefore, tracks the transition scale but is optimistic for this deployment and is not a run-level convergence certificate. The utility trajectories remain smooth across the filled and open markers, so reaching the numerical stopping criterion does not appear as a discrete accuracy threshold.

![](images/304c38f0d201b096ec6e39df3a12fc81a88bc980090e9e706781c258cfba92b3.jpg)  
Figure 10: CelebA reconstructions in the analyst-participant collusion setting. Each column contains a distinct source face, paired vertically across all rows. Within each noisy-attack row, the first column is the zero-noise control, and the remaining reconstructions are ordered so that identitylinkage accuracy decreases from left to right. The displayed C-GDP private-data-noise row covers $0 \leq \sigma \leq 1 . 5 ,$ while the anchor-noise rows cover $0 \leq \nu \leq 0 . 2 5$ . The rightmost anchor-noise reconstructions are at the 10% chance level; the C-GDP private-noise row ends at the lowest leakage attained within its displayed range. The equations give the decoded reconstructions after application of $F ^ { \dagger } .$

![](images/85535e9499618b3959afee213e069a186fa73c85e751262bd8a715c1efce3efa.jpg)  
Figure 11: CelebA participant-count sensitivity of the AA-I-GDP anchor-noise privacy–utility relationship. Curves for $p ~ \in ~ \{ 2 , 5 , 1 0 , 2 0 , 5 0 \}$ trace held-out Smiling balanced accuracy against OP cross-image identity-linkage accuracy. Each condition contains 100 draws with v ∼ Uniform(0 0 1), paired across participant counts by draw index. Solid curves are fixed cubic regression-spline fits to linkage and utility as functions of v, and the shaded ribbons connect their 95% pointwise envelopes from 10,000 paired-draw bootstrap resamples. Low-opacity filled and open symbols denote draws for which the GPM met its stopping criterion and reached the 500-iteration cap, respectively. Stars denote C-GDP, participant-colored dashed horizontal lines denote I-GDP utility, and dotted lines mark the 10% linkage and 50% balanced-accuracy chance levels. Deterministic controls are excluded from fitting; noiseless AA-I-GDP is omitted.

## 6. Conclusion

This paper studied geometric data perturbation for one-shot representation sharing in the analyst-participant col lusion setting. C-GDP preserves comparability across participants but allows a colluding participant to expose every non-colluding upload by revealing the common perturbation parameters. I-GDP removes this shared secret but places participants in mutually misaligned coordinate systems, impairing centralized learning. Under the stated full-columnrank condition and reference-participant gauge, we showed that noiseless AA-I-GDP and the corresponding coupled C-GDP construction yield the same aligned outputs available to the analyst, even though their protocol transcripts difer. Their private-data-noise outputs are likewise equal in distribution and become pathwise equal under the stated noise coupling. Anchor alignment, therefore, restores cross-participant comparability but does not, by itself, resolve the vulnerability in the analyst-participant collusion setting.

We consequently introduced anchor noise for Anchor-Aligned Independent Geometric Data Perturbation. The method perturbs the synthetic anchor representations used to estimate the alignment maps while leaving the uploaded private-data representations free of additive noise. We formulated the resulting alignment problem as a generalized orthogonal Procrustes problem, characterized its rotational and residual-translation errors, adapted existing GPM convergence results to the centered orthonormal-anchor setting, and analyzed the Moore–Penrose, orthogonal-Procrustes, and alignment-map reconstruction attacks. Under the normalized anchor construction, these results identify the anchor-noise scale $\nu ,$ anchor size $r ,$ and representation dimension d as the principal quantities governing both alignment accuracy and known-anchor reconstruction.

The experiments support the theoretical equivalence of the baseline constructions and show that relocating noise from the private-data representations to the anchor representations can materially improve the observed privacy- utility trade-of. At 12 4% exact-source linkage on the MNIST dataset, AA-I-GDP with anchor noise retains 59 2% balanced accuracy under the orthogonal-Procrustes attack, compared with 15 1% for C-GDP with private-data noise. On the CelebA dataset, anchor noise retains 73 6% balanced accuracy when fitted identity linkage reaches the 10% chance level, whereas private-data noise retains 59 7%. Across $p \in \{ 2 , 5 , 1 0 , 2 0 , 5 0 \}$ , AA-I-GDP with low anchor noise recovers the increasing-utility ordering associated with the growing training set, whereas I-GDP remains non-monotone in p; as v approaches 0 1, the anchor-noise fits move toward their corresponding I-GDP references and the natural ordering disappears. Within the evaluated ranges, increasing private-data noise drives utility toward random guessing, whereas increasing anchor noise progressively removes the utility benefit of alignment and moves performance toward the corresponding I-GDP reference.

These conclusions remain specific to the evaluated attacks and deployments. Anchor noise does not eliminate the anchor attack surface, and the three evaluated reconstruction attacks do not exhaust the set of learned or adap tive attacks. Multi-colluder and malicious-participant strategies fall outside the analyst-participant collusion setting studied here. I-GDP is included as a utility reference but is not assigned a reconstruction-based leakage score. The empirical results use fixed data partitions, projections, anchor matrices, and training seeds and are limited to image tasks. Moreover, the participant-count experiment fixes the per-participant identity budget, so the total training-set size grows with $p$ and the comparison does not isolate the causal efect of participant count. Some GPM runs also reach the iteration limit without a certificate of global optimality. The reported results should therefore be interpreted as an attack-dependent empirical privacy-utility advantage rather than a universal privacy guarantee.

Several directions merit further study. First, the proposed framework can benefit from a formal privacy calibration, including record-level diferential privacy formulations where appropriate, evaluation against stronger reconstruction attacks, and validation on non-image data and across independent deployments. Second, tighter convergence and estimation bounds, together with certified or more eficient GOPP solvers, would help distinguish the statistical efects of anchor noise from numerical optimization error. Third, the linear structure of GDP leaves a residual known-anchor recovery channel because participant-specific transformations can be estimated from anchor correspondences using tractable linear-algebraic methods, including the pseudoinverse and Orthogonal Procrustes estimators considered here Extending the noisy-anchor framework to nonlinear obfuscation and alignment mappings could reduce the efective ness of these linear estimators, although appropriately adapted nonlinear attacks would also need to be investigated. Kernel-based Target-Normalized Integration (KTI), which integrates intermediate representations produced through nonlinear dimensionality reduction, provides one possible foundation for such an extension [41]. Characterizing the resulting trade-of between known-anchor reconstruction risk and cross-participant alignment accuracy would clarify the settings in which noisy-anchor alignment can provide useful resistance to reconstruction while retaining the utility

[4] W. G. van Panhuis, P. Paul, C. Emerson, J. Grefenstette, R. Wilder, A. J. Herbst, D. Heymann, D. S. Burke, A systematic review of barriers to data sharing in public health, BMC Public Health 14 (2014) 1144. doi:10.1186/1471-2458-14-1144.

URL https://fpf.org/wp-content/uploads/2022/12/FPF-Playbook-singles.pdf

of centrally trained models.

## CRediT authorship contribution statement

Keiyu Nosaka: Conceptualization, Methodology, Software, Validation, Investigation, Data curation, Writing – original draft, Visualization, Funding acquisition, Project administration.

Yamato Suetake: Writing – review & editing, Software.

Yuichi Takano: Writing – review & editing, Supervision.

Yukihiko Okada: Writing – review & editing, Supervision.

Akiko Yoshise: Writing – review & editing, Supervision, Funding acquisition.

## Declaration of Competing Interest

The authors declare that they have no known competing financial interests or personal relationships that could have influenced the work reported in this paper.

## Declaration of Generative AI and AI-Assisted Technologies in the Writing Process

During the preparation of this work, the authors used ChatGPT and Grammarly to enhance the manuscript’s readability and language. Following the use of these services, the authors thoroughly reviewed and edited the content and assume full responsibility for the final published article.

## Data Availability

The experiment code, Colab execution notebooks, processed result records, provenance information, and figuregeneration scripts supporting this study are publicly available in the project repository. The version used for this manuscript is archived at commit 20383a6. The original MNIST and CelebA datasets and pretrained model weights are not redistributed; they can be obtained from their respective providers and are downloaded or prepared by the supplied notebooks subject to the applicable access conditions and licenses.

## Funding Sources

This work was partially supported by the Japan Society for the Promotion of Science (JSPS) KAKENHI under Grant Numbers JP22K18866, JP23K26327, JP25KJ0701, and JP26K22555.

## References

[1] European Parliament and Council of the European Union, Regulation (EU) 2016/679 (General Data Protection Regulation), Oficial Journal of the European Union, see Article 5: principles including purpose limitation, data minimisation, and integrity/confidentiality (2016). URL https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679

[2] OECD, OECD recommendation of the council on enhancing access to and sharing of data, OECD Legal Instruments (OECD/LEGAL/0463) (2021).

URL https://legalinstruments.oecd.org/api/print?ids=668&lang=en

[3] UK Government, Data sharing governance framework, GOV.UK (2022). URL https://www.gov.uk/government/publications/data-sharing-governance-framework/ data-sharing-governance-framework

[6] L. Jiang, R. Tan, X. Lou, G. Lin, On lightweight privacy-preserving collaborative learning for internet-of-things objects, in: Proceedings of the ACM/IEEE International Conference on Internet of Things Design and Implementation (IoTDI), 2019, pp. 70–81. doi:10.1145/ 3302505.3310070.

[7] B. Liu, Y. Jiang, F. Sha, R. Govindan, Cloud-enabled privacy-preserving collaborative learning for mobile sensing, in: Proceedings of the 10th ACM Conference on Embedded Network Sensor Systems (SenSys), ACM, 2012, pp. 57–70. doi:10.1145/2426656.2426663.

[8] J. Shao, F. Wu, J. Zhang, Selective knowledge sharing for privacy-preserving federated distillation without a good teacher, Nature Communications 15 (2024) 349. doi:10.1038/s41467-023-44383-9.

[9] K. Bonawitz, H. Eichner, W. Grieskamp, D. Huba, A. Ingerman, V. Ivanov, C. Kiddon, J. Konecný, S. Mazzocchi, H. B. McMahan,ˇ T. Van Overveldt, D. Petrou, D. Ramage, J. Roselander, Towards federated learning at scale: System design, in: Proceedings of Machine Learning and Systems, Vol. 1, 2019, pp. 374–388.

URL https://proceedings.mlsys.org/paper\_files/paper/2019/hash/7b770da633baf74895be22a8807f1a8f-Abstract. html

[10] S. Sugawara, Y. Kawamata, A. Toyoda, T. Nakayama, Y. Okada, Single-round clustered federated learning via data collaboration analysis for non-IID data (2026). arXiv:2601.09304.

[11] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, B. Agüera y Arcas, Communication-eficient learning of deep networks from decentralized data, in: Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS), Vol. 54 of Proceedings of Machine Learning Research, PMLR, 2017, pp. 1273–1282. URL https://proceedings mlr press/v54/mcmahan17a html URL https://proceedings.mlr.press/v54/mcmahan17a.html

[12] T. Li, A. K. Sahu, A. Talwalkar, V. Smith, Federated learning: Challenges, methods, and future directions, IEEE Signal Processing Magazine 37 (3) (2020) 50–60. doi:10.1109/MSP.2020.2975749.

[13] P. Kairouz, H. B. McMahan, B. Avent, A. Bellet, M. Bennis, A. N. Bhagoji, K. Bonawitz, Z. Charles, G. Cormode, R. Cummings, et al., Advances and open problems in federated learning, Foundations and Trends in Machine Learning 14 (1–2) (2021) 1–210. doi:10.1561/ 2200000083.

[14] C. Dwork, F. McSherry, K. Nissim, A. D. Smith, Calibrating noise to sensitivity in private data analysis, in: S. Halevi, T. Rabin (Eds.), Theory of Cryptography (TCC 2006), Vol. 3876 of Lecture Notes in Computer Science, Springer, 2006, pp. 265–284. doi:10.1007/11681878\_14.

[15] C. Dwork, Diferential privacy: A survey of results, in: Theory and Applications of Models of Computation (TAMC), Vol. 4978 of Lecture Notes in Computer Science, Springer, 2008, pp. 1–19. doi:10.1007/978-3-540-79228-4\_1.

[16] C. Dwork, A. Roth, The algorithmic foundations of diferential privacy, Foundations and Trends in Theoretical Computer Science 9 (3–4) (2014) 211–407. doi:10.1561/0400000042.

[17] A. P. Sanil, A. F. Karr, X. Lin, J. P. Reiter, Privacy preserving regression modelling via distributed computation, in: Proceedings of the Tenth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), ACM, 2004, pp. 677–682. doi:10.1145/ 1014052.1014139.

[18] B. Balle, Y.-X. Wang, Improving the gaussian mechanism for diferential privacy: Analytical calibration and optimal denoising, in: Pro ceedings of the 35th International Conference on Machine Learning (ICML), Vol. 80 of Proceedings of Machine Learning Research, PMLR, 2018, pp. 394–403.

URL https://proceedings.mlr.press/v80/balle18a.html

[19] R. Xu, N. Baracaldo, J. Joshi, Privacy-preserving machine learning: Methods, challenges and directions (2021). arXiv:2108.04417.

[21] K. Chen, L. Liu, Geometric data perturbation for privacy preserving outsourced data mining, Knowledge and Information Systems 29 (3) (2011) 657–695. doi:10.1007/s10115-010-0362-4.

[22] K. Liu, C. Giannella, H. Kargupta, An attacker’s view of distance preserving maps for privacy preserving data mining, in: Proceedings of the 10th European Conference on Principles and Practice of Knowledge Discovery in Databases (PKDD), Vol. 4213 of Lecture Notes in Computer Science, Springer, 2006, pp. 297–308.

[23] C. R. Giannella, K. Liu, H. Kargupta, Breaching euclidean distance-preserving data perturbation using few known inputs, Data & Knowledge Engineering 83 (2013) 93–110. doi:10.1016/j.datak.2012.10.004.

[24] L. Jiang, R. Tan, X. Lou, G. Lin, On lightweight privacy-preserving collaborative learning for internet of things by independent random projections (2020). arXiv:2012.07626.

[25] A. Imakura, T. Sakurai, Data collaboration analysis framework using centralization of individual intermediate representations for distributed data sets, ASCE-ASME Journal of Risk and Uncertainty in Engineering Systems, Part A: Civil Engineering 6 (2) (2020) 04020018.

[26] A. Imakura, A. Bogdanova, T. Yamazoe, K. Omote, T. Sakurai, Accuracy and privacy evaluations of collaborative data analysis, The Second AAAI Workshop on Privacy-Preserving Artificial Intelligence (PPAI-21) (2021). URL https://ppai21.github.io/files/12-paper.pdf

[27] K. Nosaka, Y. Suetake, Y. Takano, A. Yoshise, Data collaboration analysis with orthonormal basis selection and alignment, Computers and Electrical Engineering 135 (2026) 111192. doi:10.1016/j.compeleceng.2026.111192.

[28] Y. Kawakami, Y. Takano, A. Imakura, New solutions based on the generalized eigenvalue problem for the data collaboration analysis, Information Sciences 723 (2026) 122642. doi:https://doi.org/10.1016/j.ins.2025.122642.

[29] H. Yamashiro, K. Omote, A. Imakura, T. Sakurai, Toward the application of diferential privacy to data collaboration, IEEE Access 12 (2024) 63292–63301. doi:10.1109/ACCESS.2024.3396146.

[30] S. Ling, Near-optimal bounds for the generalized orthogonal procrustes problem via generalized power method, Applied and Computationa Harmonic Analysis 66 (2023) 62–100. doi:10.1016/j.acha.2023.04.008.

[31] R. Penrose, A generalized inverse for matrices, Proceedings of the Cambridge Philosophical Society 51 (3) (1955) 406–413. doi:10.1017/ S0305004100030401.

[32] A. Imakura, T. Sakurai, Y. Okada, T. Fujii, T. Sakamoto, H. Abe, Non-readily identifiable data collaboration analysis for multiple datasets including personal information, Information Fusion 98 (2023) 101826. doi:10.1016/j.inffus.2023.101826.

[33] J. M. F. Ten Berge, Orthogonal procrustes rotation for two or more matrices, Psychometrika 42 (2) (1977) 267–276. doi:10.1007/ BF02294053.

[34] P. H. Schönemann, A generalized solution of the orthogonal procrustes problem, Psychometrika 31 (1) (1966) 1–10. doi:10.1007/ BF02289451.

[35] Y. LeCun, L. Bottou, Y. Bengio, P. Hafner, Gradient-based learning applied to document recognition, Proceedings of the IEEE 86 (11) (1998) 2278–2324. doi:10.1109/5.726791.

[36] Z. Liu, P. Luo, X. Wang, X. Tang, Deep learning face attributes in the wild, in: Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2015, pp. 3730–3738. doi:10.1109/ICCV.2015.425.

[37] L. Lyu, X. He, Y. W. Law, M. Palaniswami, Privacy-preserving collaborative deep learning with application to human activity recognition, in: Proceedings of the 2017 ACM on Conference on Information and Knowledge Management, ACM, 2017, pp. 1219–1228. doi:10.1145/ 3132847.3132990

[38] S. S. Vempala, The Random Projection Method, Vol. 65 of DIMACS Series in Discrete Mathematics and Theoretical Computer Science, American Mathematical Society, 2005.

[39] C. Szegedy, S. Iofe, V. Vanhoucke, A. A. Alemi, Inception-v4, inception-resnet and the impact of residual connections on learning, in: Proceedings of the Thirty-First AAAI Conference on Artificial Intelligence, 2017, pp. 4278–4284. doi:10.1609/aaai.v31i1.11231.

[40] Q. Cao, L. Shen, W. Xie, O. M. Parkhi, A. Zisserman, VGGFace2: A dataset for recognising faces across pose and age, in: 2018 13th IEEE International Conference on Automatic Face & Gesture Recognition, 2018, pp. 67–74. doi:10.1109/FG.2018.00020.

[41] Y. Suetake, Y. Kawakami, S. Ikeda, Y. Takano, Nonlinear data integration via kernel methods for data collaboration analysis (2026). arXiv: 2605.27219. URL https://arxiv.org/abs/2605.27219

# Supplementary Material for Geometric Data Perturbation with Noisy-Anchor Alignment for Privacy-Preserving Collaborative Learning

Keiyu Nosaka<sup>a</sup>, Yamato Suetake<sup>a</sup>, Yuichi Takano<sup>b,c</sup>, Yukihiko Okada<sup>b,c</sup>, Akiko Yoshise<sup>b,c,∗</sup>

<sup>a</sup>Graduate School ofScience and Technology, University ofTsukuba, 1-1-1, Tennodai, Tsukuba, 305-8573, Ibaraki, Japan <sup>b</sup>Institute of Systems and Information Engineering, University of Tsukuba, 1-1-1, Tennodai, Tsukuba, 305-8573, Ibaraki, Japan <sup>c</sup>Centerfor Artificial Intelligence Research, Tsukuba Institutefor Advanced Research (TIAR), University ofTsukuba, 1-1-1, Tennodai, Tsukuba, 305-8577, Ibaraki, Japan

## Abstract

This supplementary material provides the complete proofs of the results stated in the main manuscript, together with additional solver diagnostics and reconstruction examples.

## Notation

Throughout this supplement, $\mathcal { S } : = \{ 1 , . . . , p \}$ denotes the participant index set, and $i ~ \in ~ \mathcal { S }$ denotes a generic participant. In an analyst-participant collusion argument, an arbitrary non-colluding reconstruction target is explicitly restricted as $i \in \mathcal { P } \setminus \{ c \}$ , while $c \in \mathcal { P }$ denotes the colluding participant. When used, $s \in \mathcal S$ denotes the reference participant selected to fix the alignment coordinate system; this role does not designate the colluder.

For a positive integer k, we write $[ k ] : = \{ 1 , 2 , \ldots , k \}$ . We denote by $\mathbb { R } ^ { p \times q }$ the set of real $p \times q$ matrices. $\pmb { M } \in \mathbb { R } ^ { p \times q }$ means M is a real p×q matrix. $( M ) _ { k , l }$ denotes the row k, column l entry of the matrix M. $\mathbf { 1 } _ { m } \in \mathbb { R } ^ { m }$ is the all-ones vector. For a matrix M, M<sup>⊤</sup> denotes the transpose, $M ^ { \dagger }$ the Moore–Penrose pseudoinverse $\begin{array} { r } { [ 1 ] (  { \mathbf { e . g . } } , \pmb { M } ^ { \dagger } = ( \pmb { M } ^ { \top } \pmb { M } ) ^ { - 1 } \pmb { M } ^ { \top } } \end{array}$ when M has full column rank), $| | \pmb { M } | | _ { F }$ the Frobenius norm, and tr(M) the matrix trace. We denote the orthogonal group in $\mathbb { R } ^ { \ell }$ by $O ( \ell ) : = \{ O \in \mathbb { R } ^ { \ell \times \ell } : O ^ { \top } O = I _ { \ell } \}$ . For a matrix M, its (thin) SVD is written as $\pmb { M } = \pmb { U } \pmb { \Sigma } \pmb { V } ^ { \top }$ . We write $\big [ M _ { i } \big ] _ { i \in [ k ] }$ for k vertically stacked matrices. Equivalently,

$$
\left[ \pmb { M } _ { i } \right] _ { i \in [ k ] } : = \left[ \begin{array} { l } { \pmb { M } _ { 1 } } \\ { \vdots } \\ { \pmb { M } _ { k } } \end{array} \right] .
$$

Definition 2.1 (One-Shot Representation Sharing). In the one-shot representation sharing regime, the global recordby-feature data matrix $\ b { X } \in \mathbb { R } ^ { n \times d }$ is partitioned $b y$ records across p participants:

$$
X = \left[ X _ { i } \right] _ { i \in \mathcal { P } } \qquad X _ { i } \in \mathbb { R } ^ { n _ { i } \times d } , \qquad \sum _ { i \in \mathcal { P } } n _ { i } = n .\tag{S.1}
$$

Participant i holds the private submatrix $X _ { i } .$ . For any batch size m, participant i applies a local row-wise obfuscation mechanism

$$
\mathcal { M } _ { i } : \mathbb { R } ^ { m \times d }  \mathbb { R } ^ { m \times h } ,
$$

which maps records to h-dimensional representations. Applying this mechanism to its local data yields

$$
\begin{array} { r } { \pmb { Y _ { i } } = \pmb { \mathcal { M } _ { i } } ( \pmb { X _ { i } } ) \in \mathbb { R } ^ { n _ { i } \times h } . } \end{array}\tag{S.2}
$$

Each participant communicates with the analyst exactly once by uploading $Y _ { i }$ and the target vector $\mathbf { } L _ { i } \in \mathbb { R } ^ { n _ { i } }$ . The analyst concatenates the uploaded representations as

$$
Y = [ Y _ { i } ] _ { i \in \mathcal { P } } \in \mathbb { R } ^ { n \times h } ,\tag{S.3}
$$

and uses Y for downstream analysis, such as model training.

Definition 2.2 (Analyst-Participant Collusion Setting). Consider a one-shot representation-sharing protocol with $p \ge 2$ participants and a central analyst. Let $c \in \mathcal { P }$ denote the single colluding participant. In the analyst-participant collusion setting, the attacker comprises the central analyst and participant c.

For each participant i, let $\Theta _ { i }$ denote all participant-side state other than the private dataset $X _ { i } ,$ including private randomness, perturbation parameters, secret keys, and other quantities used to instantiate the obfuscation mechanism. We assume that participant c discloses both $X _ { c }$ and $\Theta _ { c }$ in full. The analyst observes all uploaded representations $\{ Y _ { i } : i \in \mathcal { S } \}$ . Hence, the attacker’s view is

$$
( \{ Y _ { i } : i \in \mathcal { P } \} , ( X _ { c } , \Theta _ { c } ) ) .\tag{S.4}
$$

Unless stated otherwise, the attacker also knows the public protocol specification, including the functional forms of the obfuscation mechanisms and any public parameters. The attacker does not directly know the private datasets or local secret states ofthe non-colluding participants, except through information inferredfrom the analyst’s transcript andfrom participant c’s private dataset and local secret state.

The attacker’s objective is to recover, or infer sensitive information about, the private dataset $\pmb { X } _ { i } \in \mathbb { R } ^ { n _ { i } \times d }$ of a non-colluding target participant $i \in \mathcal { P } \ \backslash$ {c}.

Definition 2.3 (Geometric Data Perturbation). For any batch size $m ,$ a Geometric Data Perturbation (GDP) mechanism is a row-wise obfuscation map $\boldsymbol { \mathcal { M } } : \mathbb { R } ^ { m \times d }  \mathbb { R } ^ { m \times d }$ of the form

$$
\ b { M } ( \ b { S } ) = \ b { S O } + \mathbf { 1 } _ { m } \ b { \Psi } ^ { \top } , \qquad \ b { S } \in \mathbb { R } ^ { m \times d } .\tag{S.5}
$$

Here, $\pmb { O } \in O ( d )$ is an orthogonal matrix, $\Psi \in \mathbb { R } ^ { d }$ is a translation vector, and $\mathbf { 1 } _ { m } \in \mathbb { R } ^ { m }$ is the all-ones vector. Thus, the same orthogonal transformation and translation are applied to every record.

Definition 2.4 (Common-Transform Geometric Data Perturbation). Common-Transform Geometric Data Perturba tion (C-GDP) is a one-shot representation-sharing regime in which all participants use a shared GDP mechanism $\boldsymbol { \mathcal { M } } : \mathbb { R } ^ { m \times d }  \mathbb { R } ^ { m \times d }$ with common perturbation parameters (O Ψ)for any batch size m. Thus, each participant $i \in { \mathcal { P } } ,$ holding private data $\pmb { X } _ { i } \in \mathbb { R } ^ { n _ { i } \times d }$ , uploads

$$
\begin{array} { r } { Y _ { i } = \boldsymbol { \mathcal { M } } ( \boldsymbol { X } _ { i } ) = \boldsymbol { X } _ { i } \boldsymbol { O } + \mathbf { 1 } _ { n _ { i } } \boldsymbol { \Psi } ^ { \top } , } \end{array}\tag{S.6}
$$

where $\pmb { O } \in O ( d )$ is shared across participants and $\Psi \in \mathbb { R } ^ { d }$ is the shared translation vector.

## S.1. Proposition 2.5: Secret-State Vulnerability of C-GDP in the Analyst-Participant Collusion Setting

Proposition 2.5 (Secret-State Vulnerability of C-GDP in the Analyst-Participant Collusion Setting). Suppose that the uploaded representations are generated by C-GDP:

$$
\pmb { Y } _ { i } = \pmb { X } _ { i } \pmb { O } + \mathbf { 1 } _ { n _ { i } } \Psi ^ { \top } , \qquad i \in \mathcal { P } ,
$$

where $\pmb { O } \in O ( d )$ and $\Psi \in \mathbb { R } ^ { d }$ are common to all participants. Let $c \in \mathcal { P }$ be the colluding participant in the analyst participant collusion setting. If the disclosed state 0 $\Theta _ { c }$ contains (O Ψ), then the attacker reconstructs every noncolluding participant’s private dataset exactly as

$$
X _ { i } = { \Big ( } Y _ { i } - \mathbf { 1 } _ { n _ { i } } \Psi ^ { \top } { \Big ) } O ^ { \top } , \qquad i \in \mathcal { P } \setminus \{ c \} .
$$

Proof. Participant c discloses $( o , \Psi )$ , so the attacker knows the common perturbation parameters and observes every uploaded representation. Fix any non-colluding participant $i \in \mathcal { P } \setminus \{ c \}$ . Then

$$
\begin{array} { r l } { \left( Y _ { i } - \mathbf { 1 } _ { n _ { i } } \Psi ^ { \top } \right) \pmb { O } ^ { \top } = \left( X _ { i } \pmb { O } + \mathbf { 1 } _ { n _ { i } } \Psi ^ { \top } - \mathbf { 1 } _ { n _ { i } } \Psi ^ { \top } \right) \pmb { O } ^ { \top } } & { } \\ { = X _ { i } \pmb { O O } ^ { \top } } & { } \\ { = X _ { i } , } \end{array}
$$

where the final equality follows from $o o ^ { \tau } = I _ { d } .$ . Since i was arbitrary, the attacker reconstructs every non-colluding participant’s private dataset exactly. □

Definition 2.6 (C-GDP (private-data noise)). C-GDP (private-data noise) uses the common GDP mechanism M with sharedparameters (O Ψ) and adds participant-specific additive noise to each private-data representation. Participant $i \in \mathcal { P }$ samples a noise matrix $W _ { i } \in \mathbb { R } ^ { n _ { i } \times d }$ from a chosen base distribution and uploads

$$
\begin{array} { r } { Y _ { i } = \mathcal { M } ( X _ { i } ) + \sigma W _ { i } = X _ { i } O + \mathbf { 1 } _ { n _ { i } } \Psi ^ { \top } + \sigma W _ { i } . } \end{array}\tag{S.7}
$$

The matrices $W _ { 1 } , \ldots , W _ { p }$ are sampled independently across participants and independently of the common GDP parameters, and $\sigma \geq 0$ is the private-data-noise scale. Setting $\sigma = 0$ recovers C-GDP.

In the analyst-participant collusion setting of Definition 2.2, participant c reveals the common parameters. For each non-colluding participant $i \in \mathcal { P } \ \backslash$ {c}, direct inversion gives

$$
( { \pmb Y } _ { i } - { \pmb 1 } _ { n _ { i } } { \boldsymbol \Psi } ^ { \top } ) { \pmb O } ^ { \top } = { \pmb X } _ { i } + \sigma { \pmb W } _ { i } { \pmb O } ^ { \top } ,
$$

so the additive noise, rather than the secrecy of the common transform, is the protection in this attack channel.

Common choices for additive perturbation include Laplace and Gaussian noise [2, 3]. In this study, the entries of each $W _ { i }$ are i.i.d. N(0 1), so σ is the standard deviation of each added noise entry. When a formal diferential privacy (DP) baseline is required, $\sigma$ can be calibrated according to the sensitivity of the released representation, the neighboring relation, and the target privacy parameters [4, 5]. Gaussian perturbation alone should not be interpreted as a formal DP guarantee: DP additionally requires a specified neighboring relation, bounded sensitivity (typically enforced by clipping), and calibration to explicit privacy parameters [3].

Definition 2.8 (DC anchor setting). Participant i ∈ P holds $\pmb { X } _ { i } \in \mathbb { R } ^ { n _ { i } \times d }$ , and all participants have access to a common anchor matrix $\pmb { A } \in \mathbb { R } ^ { r \times d }$ . The anchor matrix is a participant-side shared state: it is not included in the honest protocol transcript and is not revealed to the analyst. For any batch size m, participant i has a local obfuscation map

$$
\mathcal { M } _ { i } : \mathbb { R } ^ { m \times d }  \mathbb { R } ^ { m \times h } .
$$

Participant i applies the same map to its private-data matrix and the shared anchor matrix and uploads

$$
Y _ { i } = { \mathcal { M } } _ { i } ( X _ { i } ) , \qquad B _ { i } = { \mathcal { M } } _ { i } ( A ) .
$$

Definition 2.9 (Analyst-Participant Collusion Setting under DC). Consider a DC protocol with participant set P and a central analyst. Let $c \in { \mathcal { P } }$ denote the single colluding participant. In the analyst-participant collusion setting under DC, the attacker comprises the central analyst and participant c.

For each participant i, let Θ denote the participant-specific local secret state, excluding the private dataset $X _ { i }$ and the shared anchor matrix A. The anchor matrix is listed separately below because it is a common participantside state rather than a participant-specific state. The state Θ may include private parameters, randomness, or other participant-side quantities used to instantiate the obfuscation mechanism $M _ { i }$ and not revealed in the protocol transcript. In the DC protocol, the analyst observes all uploaded private-data and anchor representations, namely $\{ ( Y _ { i } , B _ { i } ) : i \in \mathcal { P } \}$ . In addition, participant c reveals its private dataset and local secret state. Because participant c holds the shared anchor matrix A, we assume that it also reveals A to the analyst. Hence, the attacker’s view is

$$
( \{ ( Y _ { i } , B _ { i } ) : i \in \mathcal { P } \} , ( X _ { c } , \Theta _ { c } ) , A ) .\tag{S.8}
$$

Algorithm 2: AA-I-GDP Alignment Procedure   
Input: Private data $\pmb { X } _ { i } \in \mathbb { R } ^ { n _ { i } \times d }$ for each $i \in \mathcal { P } ;$ participant-side shared anchor matrix $\pmb { A } \in \mathbb { R } ^ { r \times d } ;$ reference   
participant $s \in { \mathcal { P } } .$   
Output: Aligned collaboration matrix $\pmb { Z } \in \mathbb { R } ^ { n \times d } ,$   
/\* Participant-side obfuscation \*/   
foreach $i \in \mathcal { P }$ do   
Privately sample GDP parameters $\mathbf { \mathscr { O } } _ { i } \in \mathcal { O } ( d )$ and $\Psi _ { i } \in \mathbb { R } ^ { d } ,$ , independently across participants   
$\mathbf { } Y _ { i } = X _ { i } \mathbf { \bar { O } } _ { i } + \bar { \mathbf { 1 } } _ { n _ { i } } \Psi _ { i } ^ { \top }$   
$\pmb { { \cal B } } _ { i } = A \pmb { { \cal O } } _ { i } + \mathbf { 1 } _ { r } \Psi _ { i } ^ { \top }$   
Upload the pair $( { Y } _ { i } , { B } _ { i } )$ to the analyst   
/\* Analyst-side alignment \*/   
foreach $i \in \mathcal { P }$ do   
$\pmb { b } _ { i } = \frac { 1 } { r } \pmb { B } _ { i } ^ { \top } \mathbf { 1 } _ { r }$   
$\overline { { \pmb { B } } } _ { i } = \pmb { B } _ { i } - \mathbf { 1 } _ { r } \pmb { b } _ { i } ^ { \top }$   
Set $\pmb { R _ { s } } = \pmb { I _ { d } } ,$ , define $\begin{array} { r } { N _ { s } ( \pmb { T } ) = \pmb { T } , } \end{array}$ , and set $\mathbf { Z } _ { s } = \pmb { Y } _ { s }$   
foreach $i \in { \mathcal { P } } \setminus \{ s \}$ do   
Compute the SVD $\overline { { { B } } } _ { i } ^ { \top } \overline { { { B } } } _ { s } = U _ { i } \Sigma _ { i } V _ { i } ^ { \top }$   
$\pmb { R _ { i } } = \pmb { U _ { i } } \pmb { V _ { i } ^ { \top } }$   
Define $\begin{array} { r } { N _ { i } ( \pmb { T } ) = \left( \pmb { T } - \pmb { 1 } _ { m } \pmb { b } _ { i } ^ { \top } \right) \pmb { R } _ { i } + \pmb { 1 } _ { m } \pmb { b } _ { s } ^ { \top } } \end{array}$ for $\pmb { T } \in \mathbb { R } ^ { m \times d }$   
$\pmb { Z } _ { i } = \pmb { N } _ { i } ( \pmb { Y } _ { i } )$   
$\mathbf { Z } = \left[ \vdots \atop \vdots \atop \mathbf { Z } _ { p } \right] \in \mathbb { R } ^ { n \times d }$   
return Z

Unless stated otherwise, the attacker also knows the public protocol specification, including the functional forms of the obfuscation mechanisms and the alignment procedure. The attacker does not directly know the private datasets or local secret states ofthe non-colluding participants, except through information inferredfrom the analyst’s transcript, participant c’s private dataset and local secret state, and the shared anchor matrix.

The attacker’s objective is to recover, or infer sensitive information about, the private dataset $\pmb { X } _ { i } \in \mathbb { R } ^ { n _ { i } \times d }$ of a non-colluding target participant $i \in \mathcal { P } \ \backslash$ {c}.

Definition 2.10 (Obfuscation Mechanism for Anchor-Aligned Independent GDP). Anchor-Aligned Independent GDP (AA-I-GDP) uses an independently sampled GDP mechanism

$$
\ b { M } _ { i } ( \ b { S } ) = \ b { S } \pmb { O } _ { i } + \ b { 1 } _ { m } \ b { \Psi } _ { i } ^ { \top } , \qquad \ b { S } \in \mathbb { R } ^ { m \times d } .
$$

This mechanism is defined for any batch size m. Each participant samples its pair $( O _ { i } , \Psi _ { i } )$ independently, where $\mathbf { \delta } _ { O _ { i } } \in { O } \left( d \right)$ and $\Psi _ { i } \in \mathbb { R } ^ { d }$

## S.2. Proposition 2.12: Exact Alignment of AA-I-GDP

Definition 2.11 (Exact DC Alignment). Suppose $h = d$ and the participant-specific maps M are deterministic. The alignment maps achieve exact DC alignment if,for every batch size m, every $i , j \in { \mathcal { S } } ,$ , and every $S \in \mathbb { R } ^ { m \times d } ;$

$$
N _ { i } ( M _ { i } ( S ) ) = N _ { j } ( M _ { j } ( S ) ) .\tag{S.9}
$$

Proposition 2.12 (Exact Alignment of AA-I-GDP). Suppose that, for every participant $i \in \mathcal { P } _ { : }$ , the uploaded anchor representation is generated by

$$
\pmb { { \cal B } } _ { i } = A \pmb { \cal O } _ { i } + \mathbf { 1 } , \Psi _ { i } ^ { \top } .\tag{S.10}
$$

Let

$$
\pmb { a } = \frac { 1 } { r } \pmb { A } ^ { \top } \pmb { 1 } _ { r } , \qquad \overline { { \pmb { A } } } = \pmb { A } - \pmb { 1 } _ { r } \pmb { a } ^ { \top } .
$$

Fix a reference participant $s \in { \mathcal { P } } ,$ , and let $\{ N _ { i } \} _ { i \in { \mathscr { P } } }$ be the alignment maps constructed by Algorithm 2. Assume that $\overline { { A } }$ has full column rank, which necessarily requires $r \geq d + 1$ . Then the alignment maps achieve exact DC alignment in the sense ofDefinition 2.11. In particular, for every participant $i \in { \mathcal { S } } ,$ , every batch size m, and every input matrix $\pmb { S } \in \mathbb { R } ^ { m \times d }$

$$
\boldsymbol { N } _ { i } ( \boldsymbol { M } _ { i } ( \boldsymbol { S } ) ) = \boldsymbol { M } _ { s } ( \boldsymbol { S } ) = \boldsymbol { S } \boldsymbol { O } _ { s } + \mathbf { 1 } _ { m } \boldsymbol { \Psi } _ { s } ^ { \top } .\tag{S.11}
$$

Consequently,for the private-data uploads, the aligned collaboration matrix satisfies

$$
\begin{array} { r } { \pmb { Z } = \pmb { X } \pmb { O } _ { s } + \pmb { 1 } _ { n } \Psi _ { s } ^ { \top } . } \end{array}\tag{S.12}
$$

Equivalently, if participant s fixes the global rigid-coordinate gauge and C-GDP is coupled to AA-I-GDP by choosing its common parameters as $( \pmb { { \cal O } } , \Psi ) = ( \pmb { { \cal O } } _ { s } , \Psi _ { s } )$ , then the C-GDP and aligned AA-I-GDP collaboration matrices are equal pathwise for every private dataset X. This is an equality of analyst-side collaboration outputs; the protocols complete transcripts remain diferent.

The proof uses the following two auxiliary lemmas.

Lemma S.1. Suppose that the uploaded anchor representation of participant i is generated by

$$
\pmb { { \cal B } } _ { i } = A \pmb { \cal O } _ { i } + \mathbf { 1 } _ { r } \Psi _ { i } ^ { \top } .\tag{S.13}
$$

Let

$$
\pmb { a } = \frac { 1 } { r } \pmb { A } ^ { \top } \pmb { 1 } _ { r } , \qquad \overline { { \pmb { A } } } = \pmb { A } - \pmb { 1 } _ { r } \pmb { a } ^ { \top } ,
$$

and let $\pmb { b } _ { i }$ and $\overline { { B } } _ { i }$ be defined as in Algorithm 2. Then

$$
{ \pmb b } _ { i } = { \pmb O } _ { i } ^ { \top } { \pmb a } + \Psi _ { i } , \qquad { \pmb \overline { { \pmb { B } } } } _ { i } = \overline { { \pmb { A } } } \pmb { O } _ { i } .\tag{S.14}
$$

Proof. By the definition of $\pmb { b } _ { i }$ and the expression for $\pmb { B } _ { i }$

$$
\begin{array} { l } { { \displaystyle b _ { i } = \frac { 1 } { r } B _ { i } ^ { \top } { \bf 1 } _ { r } } } \\ { ~ } \\ { { \displaystyle ~ = \frac { 1 } { r } \left( { \cal O } _ { i } ^ { \top } { \cal A } ^ { \top } + \Psi _ { i } { \bf 1 } _ { r } ^ { \top } \right) { \bf 1 } _ { r } } } \\ { { \displaystyle ~ = { \cal O } _ { i } ^ { \top } \left( \frac { 1 } { r } { \cal A } ^ { \top } { \bf 1 } _ { r } \right) + \Psi _ { i } } } \\ { { \displaystyle ~ = { \cal O } _ { i } ^ { \top } { \cal a } + \Psi _ { i } . } } \end{array}\tag{S.15}
$$

Equivalently, $\pmb { b } _ { i } ^ { \top } = \pmb { a } ^ { \top } \pmb { O } _ { i } + \Psi _ { i } ^ { \top }$ . Therefore,

$$
\begin{array} { r l } & { \overline { { B } } _ { i } = B _ { i } - \mathbf { 1 } _ { r } b _ { i } ^ { \top } } \\ & { \quad = A O _ { i } + \mathbf { 1 } _ { r } \Psi _ { i } ^ { \top } - \mathbf { 1 } _ { r } \left( a ^ { \top } O _ { i } + \Psi _ { i } ^ { \top } \right) } \\ & { \quad = \left( A - \mathbf { 1 } _ { r } a ^ { \top } \right) O _ { i } } \\ & { \quad = \overline { { A } } O _ { i } . } \end{array}\tag{S.16}
$$

Lemma S.2. Suppose that the uploaded anchor representation ofparticipant i is generated by

$$
\pmb { { \cal B } } _ { i } = A \pmb { \cal O } _ { i } + \mathbf { 1 } _ { r } \Psi _ { i } ^ { \top } .\tag{S.17}
$$

Let

$$
\pmb { a } = \frac { 1 } { r } \pmb { A } ^ { \top } \pmb { 1 } _ { r } , \qquad \overline { { \pmb { A } } } = \pmb { A } - \pmb { 1 } _ { r } \pmb { a } ^ { \top } ,
$$

and let $\pmb { b } _ { i }$ and $\overline { { B } } _ { i }$ be defined as in Algorithm 2. Assume $\overline { { A } }$ hasfull column rank. Since rank $( \overline { { A } } ) \leq r - 1$ , this assumption requires $r \geq d + 1$ . Fix a participant $i \in \mathcal { P }$ and a reference participant $s \in { \mathcal { P } } .$ . Let

$$
\overline { { { B } } } _ { i } ^ { \top } \overline { { { B } } } _ { s } = U _ { i } \Sigma _ { i } V _ { i } ^ { \top }\tag{S.18}
$$

be afull singular value decomposition as used in Algorithm 2, and define $\pmb { R _ { i } } = \pmb { U _ { i } } \pmb { V _ { i } ^ { \top } }$ . Then

$$
\pmb { R } _ { i } = \pmb { O } _ { i } ^ { \top } \pmb { O } _ { s } .\tag{S.19}
$$

Proof. By Lemma S.1, the centered uploaded anchor representations satisfy

$$
\begin{array} { c c } { \overline { { { \pmb { B } } } } _ { i } = \overline { { { \pmb { A } } } } \pmb { O } _ { i } , \qquad \overline { { { \pmb { B } } } } _ { s } = \overline { { { \pmb { A } } } } \pmb { O } _ { s } . } \end{array}\tag{S.20}
$$

Then

$$
\overline { { { B } } } _ { i } ^ { \top } \overline { { { B } } } _ { s } = { O } _ { i } ^ { \top } \overline { { { A } } } ^ { \top } \overline { { { A } } } \overline { { { O } } } _ { s } .\tag{S.21}
$$

Since $\overline { { A } }$ has full column rank, $\overline { { A } } ^ { \top } \overline { { A } }$ is symmetric positive definite. Hence $\partial _ { s } ^ { \top } \overline { { A } } ^ { \top } \overline { { A } } \partial _ { s }$ is also symmetric positive definite. Therefore,

$$
\pmb { H } _ { i } = \pmb { O } _ { i } ^ { \top } \overline { { \pmb { A } } } ^ { \top } \overline { { \pmb { A } } } \pmb { O } _ { s } = \big ( \pmb { O } _ { i } ^ { \top } \pmb { O } _ { s } \big ) \Big ( \pmb { O } _ { s } ^ { \top } \overline { { \pmb { A } } } ^ { \top } \overline { { \pmb { A } } } \pmb { O } _ { s } \Big )\tag{S.22}
$$

is the polar decomposition of $\overline { { \pmb { B } } } _ { i } ^ { \top } \overline { { \pmb { B } } } _ { s }$ , because ${ \pmb O } _ { i } ^ { \top } { \pmb O } _ { s }$ is orthogonal and $\pmb { O } _ { s } ^ { \top } \overline { { \pmb { A } } } ^ { \top } \overline { { \pmb { A } } } \pmb { O } _ { s }$ is symmetric positive definite. Since $\overline { { B } } _ { i } ^ { \top } \overline { { B } } _ { s }$ is nonsingular, its polar factor is unique. On the other hand, if $\overline { { { B } } } _ { i } ^ { \top } \overline { { { B } } } _ { s } = U _ { i } \Sigma _ { i } V _ { i } ^ { \top }$ is the SVD, then its polar factor is $U _ { i } V _ { i } ^ { \top }$ . Thus,

$$
\begin{array} { r } { U _ { i } V _ { i } ^ { \top } = \pmb { O } _ { i } ^ { \top } \pmb { O } _ { s } . } \end{array}\tag{S.23}
$$

By the definition $\pmb { R } _ { i } = \pmb { U } _ { i } \pmb { V } _ { i } ^ { \top }$ , we obtain

$$
\pmb { R } _ { i } = \pmb { O } _ { i } ^ { \top } \pmb { O } _ { s } .\tag{S.24}
$$

Proof. For the reference participant s, Algorithm 2 defines $N _ { s }$ as the identity map, so $N _ { s } ( M _ { s } ( S ) ) = M _ { s } ( S )$ . Now fix any participant $i \in { \mathcal { P } } \setminus \{ s \}$ . By Lemma S.1, we have $\pmb { b } _ { i } ^ { \top } = \pmb { a } ^ { \top } \pmb { O } _ { i } + \Psi _ { i } ^ { \top }$ and $\pmb { b } _ { s } ^ { \top } = \pmb { a } ^ { \top } \pmb { O } _ { s } + \Psi _ { s } ^ { \top }$ . By Lemma S.2, the SVD-based alignment matrix satisfies ${ \pmb R } _ { i } = { \pmb O } _ { i } ^ { \top } { \pmb O } _ { s }$ . Therefore, for any $\dot { \mathbf { S } } \in \mathbb { R } ^ { m \times d }$

$$
\begin{array} { r l } & { N _ { i } ( \mathcal { M } _ { i } ( S ) ) = \left( S O _ { i } + \mathbf { 1 } _ { m } \Psi _ { i } ^ { \top } - \mathbf { 1 } _ { m } b _ { i } ^ { \top } \right) R _ { i } + \mathbf { 1 } _ { m } b _ { s } ^ { \top } } \\ & { \qquad = \left( S O _ { i } - \mathbf { 1 } _ { m } a ^ { \top } O _ { i } \right) O _ { i } ^ { \top } O _ { s } + \mathbf { 1 } _ { m } \left( a ^ { \top } O _ { s } + \Psi _ { s } ^ { \top } \right) } \\ & { \qquad = \left( S - \mathbf { 1 } _ { m } a ^ { \top } \right) O _ { s } + \mathbf { 1 } _ { m } a ^ { \top } O _ { s } + \mathbf { 1 } _ { m } \Psi _ { s } ^ { \top } } \\ & { \qquad = S O _ { s } + \mathbf { 1 } _ { m } \Psi _ { s } ^ { \top } } \\ & { \qquad = \mathcal { M } _ { s } ( S ) . } \end{array}\tag{S.25}
$$

Thus, for every $i \in \mathcal { P }$ , the aligned representation $N _ { i } ( M _ { i } ( S ) )$ equals the same reference representation $M _ { s } ( \pmb { S } )$ . Hence, for all $i , j \in { \mathcal { P } } _ { : }$

$$
N _ { i } ( M _ { i } ( S ) ) = N _ { j } ( M _ { j } ( S ) ) ,\tag{S.26}
$$

which is exact DC alignment. Finally, applying the identity above to $S = X _ { i }$ for each participant and concatenating the aligned private-data representations gives

$$
\begin{array} { r } { \mathbf { Z } = \left[ N _ { i } ( Y _ { i } ) \right] _ { i \in \mathcal { P } } = \left[ X _ { i } \pmb { O } _ { s } + \mathbf { 1 } _ { n _ { i } } \Psi _ { s } ^ { \top } \right] _ { i \in \mathcal { P } } = X \pmb { O } _ { s } + \mathbf { 1 } _ { n } \Psi _ { s } ^ { \top } . } \end{array}\tag{S.27}
$$

Under the coupling $( \pmb { { \cal O } } , \Psi ) = ( \pmb { { \cal O } } _ { s } , \Psi _ { s } )$ , C-GDP produces the same collaboration matrix for every X, establishing the stated pathwise equivalence. This completes the proof. □

Algorithm 3: Known-Anchor Attack against AA-I-GDP in the Analyst-Participant Collusion Setting   
Input: Anchor matrix A revealed by the colluder; uploaded pairs $\left\{ ( { \pmb Y } _ { i } , { \pmb B } _ { i } ) \right\} _ { i = 1 } ^ { p }$ ; colluder index $c \in { \mathcal { P } } .$   
Output: Reconstructions $\{ \widehat { X } _ { i } : i \in \mathcal { P } \setminus \{ c \} \}$   
$\pmb { a } = \frac { 1 } { r } \pmb { A } ^ { \top } \mathbf { 1 } _ { r }$   
$\overline { { A } } = A - \mathbf { 1 } _ { r } \pmb { a } ^ { \top }$   
foreach $i \in \mathcal { P } \ \backslash$ {c} do   
$\pmb { b } _ { i } = \frac { 1 } { r } \pmb { B } _ { i } ^ { \top } \mathbf { 1 } _ { r }$   
$\overline { { \pmb { B } } } _ { i } = \pmb { B } _ { i } - \mathbf { 1 } _ { r } \pmb { b } _ { i } ^ { \top }$   
$\widehat { \pmb { O } } _ { i } = \overline { { \pmb { A } } } ^ { \dagger } \overline { { \pmb { B } } } _ { i }$   
${ \widehat { \Psi } } _ { i } = \pmb { { b } } _ { i } - { \widehat { \pmb { { O } } } } _ { i } ^ { \top } \pmb { { a } }$   
${ \widehat { \pmb { X } } } _ { i } = \left( { \pmb { Y } } _ { i } - \mathbf { 1 } _ { n _ { i } } { \widehat { \Psi } } _ { i } ^ { \top } \right) { \widehat { \pmb { O } } } _ { i } ^ { \top }$   
return $\{ \widehat { X } _ { i } : i \in \mathcal { P } \setminus \{ c \} \}$

Definition 2.13 (AA-I-GDP (private-data noise)). AA-I-GDP (private-data noise) uses the same participant-specific GDP mechanism M<sub>i</sub> for the private-data matrix and the shared anchor matrix, but adds noise only to the private-data representation. Participant i samples $\pmb { W } _ { i } \in \mathbb { R } ^ { n _ { i } \times d }$ independently across participants and independently of $( O _ { i } , \Psi _ { i } )$ with entries

$$
( W _ { i } ) _ { k \ell } \stackrel { \mathrm { i n d } } { \sim } N ( 0 , 1 ) , \qquad k \in [ n _ { i } ] , \ell \in [ d ] .
$$

It then uploads

$$
\begin{array} { r } { \pmb { Y } _ { i } = \pmb { \mathcal { M } } _ { i } ( \pmb { X } _ { i } ) + \sigma \pmb { W } _ { i } = \pmb { X } _ { i } \pmb { O } _ { i } + \pmb { 1 } _ { n _ { i } } \pmb { \Psi } _ { i } ^ { \top } + \sigma \pmb { W } _ { i } , } \end{array}\tag{S.28}
$$

$$
\pmb { { \cal B } } _ { i } = \pmb { { \cal M } } _ { i } ( \pmb { { \cal A } } ) = \pmb { { \cal A } } \pmb { { \cal O } } _ { i } + \pmb { 1 } _ { r } \Psi _ { i } ^ { \top } .\tag{S.29}
$$

The scale $\sigma \geq 0$ controls private-data noise, with $\sigma = 0$ recovering AA-I-GDP. Gaussian noise provides aformal DP guarantee only after neighboring datasets and sensitivity are defined and σ is calibrated to a target $( \varepsilon , \delta ) – D P$ level; proceduresfor performing this calibration in DC have been studied in $I 6 J .$

## S.3. Proposition 3.1: Vulnerability of AA-I-GDP in the Analyst-Participant Collusion Setting

Proposition 3.1 (Vulnerability of AA-I-GDP in the Analyst-Participant Collusion Setting). Suppose that the uploaded representations are generated exactly by AA-I-GDP. Thus,for each participant $i \in \mathcal { P } _ { ; }$

$$
\begin{array} { r } { { \pmb { Y } } _ { i } = { \pmb { X } } _ { i } { \pmb { O } } _ { i } + \mathbf { 1 } _ { n _ { i } } \Psi _ { i } ^ { \top } , } \\ { { \pmb { B } } _ { i } = { \pmb { A } } { \pmb { O } } _ { i } + \mathbf { 1 } _ { r } \Psi _ { i } ^ { \top } , } \end{array}\tag{S.30}
$$

(S.31)

where $\mathbf { \mathscr { O } } _ { i } \in \mathcal { O } ( d )$ and $\Psi _ { i } \in \mathbb { R } ^ { d }$ are participant-specific GDP parameters. Let

$$
\pmb { a } = \frac { 1 } { r } \pmb { A } ^ { \top } \pmb { 1 } _ { r } , \qquad \overline { { \pmb { A } } } = \pmb { A } - \pmb { 1 } _ { r } \pmb { a } ^ { \top } .
$$

Assume that $\overline { { A } }$ hasfull column rank. Consider the analyst-participant collusion settingfor DC in Definition 2.9, with colluding participant $c \in { \mathcal { P } } .$ . Under this setting, the attacker observes the shared anchor matrix A, and Algorithm 3 outputs

$$
{ \widehat { X } } _ { i } = X _ { i }
$$

for every non-colluding participant $i \in \mathcal { P } \ \backslash$ {c}.

Proof. Fix an arbitrary non-colluding participant $i \in \mathcal { P } \setminus \{ c \}$ . From the anchor-upload relation,

$$
\begin{array} { l } { { \displaystyle b _ { i } = \frac { 1 } { r } B _ { i } ^ { \top } { \bf 1 } _ { r } } } \\ { ~ } \\ { { \displaystyle ~ = \frac { 1 } { r } \left( { \cal O } _ { i } ^ { \top } { \cal A } ^ { \top } + \Psi _ { i } { \bf 1 } _ { r } ^ { \top } \right) { \bf 1 } _ { r } } } \\ { ~ } \\ { { \displaystyle ~ = { \cal O } _ { i } ^ { \top } { \cal a } + \Psi _ { i } } . } \end{array}
$$

Consequently,

$$
\begin{array} { r l } & { \pmb { \overline { { B } } } _ { i } = \pmb { B } _ { i } - \mathbf { 1 } _ { r } \pmb { b } _ { i } ^ { \top } } \\ & { \quad = \pmb { A } \pmb { O } _ { i } + \mathbf { 1 } _ { r } \Psi _ { i } ^ { \top } - \mathbf { 1 } _ { r } \left( \pmb { a } ^ { \top } \pmb { O } _ { i } + \Psi _ { i } ^ { \top } \right) } \\ & { \quad = \overline { { \pmb { A } } } \pmb { O } _ { i } . } \end{array}
$$

Since $\overline { { A } }$ has full column rank, $\overline { { A } } ^ { \dagger } \overline { { A } } = I _ { d } .$ , and hence

$$
\widehat { \pmb { O } } _ { i } = \overline { { \pmb { A } } } ^ { \dagger } \overline { { \pmb { B } } } _ { i } = \overline { { \pmb { A } } } ^ { \dagger } \overline { { \pmb { A } } } \pmb { O } _ { i } = \pmb { O } _ { i } .
$$

It follows that

$$
\widehat { \Psi } _ { i } = \pmb { b } _ { i } - \widehat { \pmb { O } } _ { i } ^ { \top } \pmb { a } = \pmb { O } _ { i } ^ { \top } \pmb { a } + \Psi _ { i } - \pmb { O } _ { i } ^ { \top } \pmb { a } = \Psi _ { i } .
$$

Therefore,

$$
\begin{array} { r l } & { \widehat { X } _ { i } = \left( { Y } _ { i } - \mathbf { 1 } _ { n _ { i } } \widehat { \Psi } _ { i } ^ { \top } \right) \widehat { \pmb { O } } _ { i } ^ { \top } } \\ & { \quad \quad = \left( { X } _ { i } \pmb { O } _ { i } + \mathbf { 1 } _ { n _ { i } } \Psi _ { i } ^ { \top } - \mathbf { 1 } _ { n _ { i } } \Psi _ { i } ^ { \top } \right) \pmb { O } _ { i } ^ { \top } } \\ & { \quad \quad = { X } _ { i } \pmb { O } _ { i } \pmb { O } _ { i } ^ { \top } = { X } _ { i } . } \end{array}
$$

Since i was arbitrary, the reconstruction is exact for every non-colluding participant.

## S.4. Proposition 3.2: Aligned Representation under AA-I-GDP (private-data noise)

Proposition 3.2 (Aligned Representation under AA-I-GDP (private-data noise)). Suppose that participant $i ~ \in ~ \mathcal { S }$ uploads

$$
\pmb { Y } _ { i } = \pmb { X } _ { i } \pmb { O } _ { i } + \pmb { 1 } _ { n _ { i } } \pmb { \Psi } _ { i } ^ { \top } + \sigma \pmb { W } _ { i }
$$

and the noiseless anchor representation

$$
\pmb { { \cal B } } _ { i } = A \pmb { { \cal O } } _ { i } + \mathbf { 1 } _ { r } \Psi _ { i } ^ { \top } ,
$$

where $\pmb { O } _ { i } \in O ( d ) , \Psi _ { i } \in \mathbb { R } ^ { d } ,$ , and $\pmb { W } _ { i } \in \mathbb { R } ^ { n _ { i } \times d }$ . Assume that $\overline { { A } }$ has full column rank, and let $N _ { i }$ be the alignment maps constructed from the anchor representations as in Algorithm 2, with reference participant $s \in \mathcal { P }$ . Define

$$
\widetilde { \pmb { W } } _ { i } = \pmb { W } _ { i } \pmb { O } _ { i } ^ { \top } \pmb { O } _ { s } , \qquad i \in \mathcal { P } .
$$

Then

$$
\begin{array} { r } { N _ { i } ( Y _ { i } ) = X _ { i } \pmb { O } _ { s } + \mathbf { 1 } _ { n _ { i } } \Psi _ { s } ^ { \top } + \sigma \widetilde { \pmb { W } } _ { i } . } \end{array}
$$

Consequently,

$$
\begin{array} { r } { \pmb { Z } = X \pmb { O } _ { s } + \mathbf { 1 } _ { n } \Psi _ { s } ^ { \top } + \sigma \widetilde { W } , \qquad \widetilde { W } = \left[ \widetilde { W } _ { i } \right] _ { i \in \mathcal { P } } . } \end{array}
$$

If the entries of $W _ { 1 } , \ldots , W _ { p }$ are mutually independent standard Gaussian variables and the collection $\{ W _ { i } \} _ { i \in \mathcal { P } } i s j o i n t l y$ independent of all participant-specific GDP parameters, then the entries of $\widetilde { W }$ are mutually independent standard Gaussian variables. Moreover, $\widetilde { W }$ is independent ofthe GDP parameters.

Proof. Let

$$
\pmb { a } = \frac { 1 } { r } \pmb { A } ^ { \top } \pmb { 1 } _ { r } , \qquad \overline { { \pmb { A } } } = \pmb { A } - \pmb { 1 } _ { r } \pmb { a } ^ { \top } .
$$

Use $\pmb { b } _ { i } , \overline { { \pmb { B } } } _ { i } , \pmb { R } _ { i } ,$ and $N _ { i }$ as constructed in Algorithm 2. Centering the anchor representations gives

$$
\pmb { b } _ { i } ^ { \top } = \pmb { a } ^ { \top } \pmb { O } _ { i } + \Psi _ { i } ^ { \top } , \qquad \ \overline { { \pmb { B } } } _ { i } = \overline { { \pmb { A } } } \pmb { O } _ { i } .
$$

For each $i \neq s ,$

$$
\overline { { { B } } } _ { i } ^ { \top } \overline { { { B } } } _ { s } = { O } _ { i } ^ { \top } \overline { { { A } } } ^ { \top } \overline { { { A } } } \overline { { { O } } } _ { s } .
$$

Since $\overline { { A } }$ has full column rank, $\overline { { A } } ^ { \top } \overline { { A } }$ is positive definite. Hence,

$$
\pmb { O } _ { i } ^ { \top } \overline { { \pmb { A } } } ^ { \top } \overline { { \pmb { A } } } \pmb { O } _ { s } = \big ( \pmb { O } _ { i } ^ { \top } \pmb { O } _ { s } \big ) \Big ( \pmb { O } _ { s } ^ { \top } \overline { { \pmb { A } } } ^ { \top } \overline { { \pmb { A } } } \pmb { O } _ { s } \Big )
$$

is the polar decomposition of $\overline { { \pmb { B } } } _ { i } ^ { \top } \overline { { \pmb { B } } } _ { s }$ . Its orthogonal polar factor is therefore unique and satisfies

$$
\pmb { R } _ { i } = \pmb { O } _ { i } ^ { \top } \pmb { O } _ { s } .
$$

For $i \neq s ,$

$$
\begin{array} { r l } & { N _ { i } ( Y _ { i } ) = \left( X _ { i } { O _ { i } } + \mathbf { 1 } _ { n _ { i } } \Psi _ { i } ^ { \top } + \sigma W _ { i } - \mathbf { 1 } _ { n _ { i } } b _ { i } ^ { \top } \right) R _ { i } + \mathbf { 1 } _ { n _ { i } } b _ { s } ^ { \top } } \\ & { \qquad = \left( X _ { i } { O _ { i } } - \mathbf { 1 } _ { n _ { i } } a ^ { \top } { O _ { i } } + \sigma W _ { i } \right) { O _ { i } ^ { \top } } { O _ { s } } + \mathbf { 1 } _ { n _ { i } } \left( a ^ { \top } { O _ { s } } + \Psi _ { s } ^ { \top } \right) } \\ & { \qquad = X _ { i } { O _ { s } } + \mathbf { 1 } _ { n _ { i } } \Psi _ { s } ^ { \top } + \sigma W _ { i } { O _ { i } ^ { \top } } { O _ { s } } } \\ & { \qquad = X _ { i } { O _ { s } } + \mathbf { 1 } _ { n _ { i } } \Psi _ { s } ^ { \top } + \sigma \widetilde { W } _ { i } . } \end{array}
$$

For the reference participant, $N _ { s }$ is the identity and

$$
\widetilde { \pmb { W } } _ { s } = { \pmb W } _ { s } \pmb { O } _ { s } ^ { \top } \pmb { O } _ { s } = { \pmb W } _ { s } .
$$

Therefore,

$$
\begin{array} { r } { N _ { s } ( \pmb { Y } _ { s } ) = \pmb { X } _ { s } \pmb { O } _ { s } + \pmb { 1 } _ { n _ { s } } \Psi _ { s } ^ { \top } + \sigma \widetilde { \pmb { W } } _ { s } , } \end{array}
$$

so the claimed formula holds for every $i \in \mathcal { P } .$ Vertical stacking gives

$$
\pmb { Z } = \pmb { X } \pmb { O } _ { s } + \pmb { 1 } _ { n } \Psi _ { s } ^ { \top } + \sigma \widetilde { \pmb { W } } .
$$

It remains to establish the Gaussian claim. Condition on all GDP parameters. For each participant, ${ \pmb { O } } _ { i } ^ { \top } { \pmb { O } } _ { i }$ is then a fixed orthogonal matrix. Each row of $W _ { i }$ has distribution ${ \cal N } ( \mathbf { 0 } , { \cal I } _ { d } ) .$ , so rotational invariance gives

$$
\pmb { w O } _ { i } ^ { \top } \pmb { O } _ { s } \sim N ( \pmb { 0 } , \pmb { I } _ { d } )
$$

for every row w of $W _ { i } .$ Conditional independence is preserved across rows and participants because the original noise entries are mutually independent. Thus, conditional on the GDP parameters, the entries of $\widetilde { W }$ are mutually independent standard Gaussian variables. This conditional joint distribution is the same for every realization of the GDP parameters. It follows that the same product-Gaussian law holds unconditionally and that $\check { \overline { W } }$ is independent of the GDP parameters. □

## S.5. Corollary 3.3: Collaboration-Output Equivalence under Private-Data Noise

Corollary 3.3 (Collaboration-Output Equivalence under Private-Data Noise). Assume the Gaussian-noise conditions of Proposition 3.2. Fix the GDP parameters, let ${ \pmb Z } ^ { \mathrm { A A } }$ denote the aligned collaboration matrix produced by AA-I-GDP (private-data noise), and let $Z ^ { \mathrm { C } }$ denote the collaboration matrix produced by C-GDP (private-data noise) with common parameters

$$
( { \cal O } , \Psi ) = ( { \cal O } _ { s } , \Psi _ { s } ) .
$$

Writing $U \overset { \mathrm { d } } { = } V$ to mean that random matrices U and V have the same distribution, the two collaboration matrices satisfy

$$
{ \pmb Z } ^ { \mathrm { A A } } \overset { \mathrm { d } } { = } { \pmb Z } ^ { \mathrm { C } } .
$$

Moreover,for anyfixed realization ofthe AA-I-GDP variables, consider the coupling in which the standard Gaussian noise block used by C-GDPfor participant i is

$$
\widetilde { \pmb { W } } _ { i } = \pmb { W } _ { i } \pmb { O } _ { i } ^ { \top } \pmb { O } _ { s } .
$$

Under this coupling,

$$
{ \pmb Z } ^ { \mathrm { A A } } = { \pmb Z } ^ { \mathrm { C } } .
$$

Proof. Let

$$
\boldsymbol { W } ^ { \mathrm { C } } = \left[ W _ { i } ^ { \mathrm { C } } \right] _ { i \in \mathcal { P } }
$$

denote the stacked standard Gaussian matrix used by C-GDP, independent of the common GDP parameters. For the matched parameters $( \pmb { { \cal O } } , \Psi ) = ( \pmb { { \cal O } } _ { s } , \Psi _ { s } )$ ,

$$
\begin{array} { r } { \pmb { Z } ^ { \mathrm { C } } = \pmb { X } \pmb { O } _ { s } + \pmb { 1 } _ { n } \Psi _ { s } ^ { \top } + \sigma \pmb { W } ^ { \mathrm { C } } . } \end{array}
$$

Proposition 3.2 gives

$$
\begin{array} { r } { \pmb { Z } ^ { \mathrm { A A } } = \pmb { X } \pmb { O } _ { s } + \pmb { 1 } _ { n } \pmb { \Psi } _ { s } ^ { \top } + \sigma \widetilde { \pmb { W } } , } \end{array}
$$

where $\widetilde { W }$ has the same product-standard-Gaussian distribution as $W ^ { \mathrm { C } }$ and is independent of the GDP parameters. Therefore, conditional on the fixed GDP parameters,

$$
{ \pmb Z } ^ { \mathrm { A A } } \overset { \mathrm { d } } { = } { \pmb Z } ^ { \mathrm { C } } .
$$

For the realization-wise statement, couple the protocols by setting

$$
\begin{array} { r } { W _ { i } ^ { \mathrm { C } } = \widetilde { W } _ { i } = W _ { i } { O } _ { i } ^ { \top } { O } _ { s } , \qquad i \in \mathcal { P } . } \end{array}
$$

Then, for every participant,

$$
\begin{array} { r l } & { \pmb { Z } _ { i } ^ { \mathrm { C } } = X _ { i } \pmb { O } _ { s } + \pmb { 1 } _ { n _ { i } } \Psi _ { s } ^ { \top } + \sigma \pmb { W } _ { i } ^ { \mathrm { C } } } \\ & { \quad \quad = X _ { i } \pmb { O } _ { s } + \pmb { 1 } _ { n _ { i } } \Psi _ { s } ^ { \top } + \sigma \widetilde { \pmb { W } } _ { i } } \\ & { \quad = N _ { i } ( Y _ { i } ) = \pmb { Z } _ { i } ^ { \mathrm { A A } } . } \end{array}
$$

Vertical stacking yields

$$
\begin{array} { r } { Z ^ { \mathrm { A A } } = Z ^ { \mathrm { C } } , } \end{array}
$$

as claimed.

Definition 4.1 (AA-I-GDP (anchor noise)). Given a participant-side shared anchor matrix $\pmb { A } \in \mathbb { R } ^ { r \times d } ,$ , AA-I-GDP (anchor noise) uses the same participant-specific GDP mechanism $M _ { i }$ for the private-data matrix and the shared anchor matrix, but adds noise only to the anchor representation. Participant i samples $\pmb { W } _ { i } \in \mathbb { R } ^ { r \times d }$ independently across participants and independently of $( O _ { i } , \Psi _ { i } )$ , with entries

$$
( W _ { i } ) _ { k \ell } \stackrel { \mathrm { i n d } } { \sim } N ( 0 , 1 ) , \qquad k \in [ r ] , \ell \in [ d ] .
$$

It then uploads

$$
\pmb { Y } _ { i } = \pmb { M } _ { i } ( \pmb { X } _ { i } ) = \pmb { X } _ { i } \pmb { O } _ { i } + \pmb { 1 } _ { n _ { i } } \pmb { \Psi } _ { i } ^ { \top } ,\tag{S.32}
$$

$$
\pmb { { B } } _ { i } = \pmb { M } _ { i } ( \pmb { A } ) + \nu \pmb { W } _ { i } = \pmb { A } \pmb { O } _ { i } + \pmb { 1 } _ { r } \Psi _ { i } ^ { \top } + \nu \pmb { W } _ { i } .\tag{S.33}
$$

The scale $\nu \geq 0$ controls anchor noise, with $\nu = 0$ recovering AA-I-GDP.

The preceding definition applies to a participant-side shared anchor matrix A. Let $r \geq d + 1$ . Before any release, the participants generate a single shared anchor matrix $\pmb { A } \in \mathbb { R } ^ { r \times d }$ satisfying

$$
\mathbf { 1 } _ { r } ^ { \top } A = \mathbf { 0 } _ { d } ^ { \top } , \qquad A ^ { \top } A = I _ { d } .\tag{S.34}
$$

Thus, the anchor matrix is centered and has orthonormal columns. The first constraint fixes its centroid at the origin, while the second fixes its scale relative to the anchor-noise parameter v. To construct such a matrix, define

$$
\pmb { H } _ { r } = \pmb { I } _ { r } - \frac { 1 } { r } \pmb { 1 } _ { r } \pmb { 1 } _ { r } ^ { \top } ,
$$

sample a participant-side Gaussian matrix $\mathbf { \Omega } ^ { \textbf { \textit { \textbf { Q } } } \in \textbf { \textit { \textbf { R } } } ^ { r \times d } }$ with independent standard normal entries, and compute the sign-corrected thin QR factorization

$$
H _ { r } \pmb { \Omega } = \pmb { A } \pmb { R } _ { A } ,
$$

where $\pmb { R } _ { A } \in \mathbb { R } ^ { d \times d }$ is upper triangular with positive diagonal. The factorization is well defined almost surely and produces a matrix satisfying (S.34). The same realization of A is distributed to every participant and withheld from the analyst during honest execution. Neither A nor the randomness used to generate it is included in the honest protocol transcript.

## S.6. Proposition 4.2: Global rigid-motion invariance of the GOPP

The generalized orthogonal Procrustes problem used in this result is

$$
\begin{array} { r l } { \underset { C , R _ { 1 } , \ldots , R _ { p } , \tau _ { 1 } , \ldots , \tau _ { p } } { \operatorname* { m i n } } } & { \displaystyle \sum _ { i = 1 } ^ { p } \left\| C - \left( B _ { i } - \mathbf { 1 } _ { r } \tau _ { i } ^ { \top } \right) { \pmb { R } } _ { i } \right\| _ { F } ^ { 2 } } \\ { \mathrm { s . t . } \quad R _ { i } \in O ( d ) , \qquad C \in \mathbb { R } ^ { r \times d } , \qquad \tau _ { i } \in \mathbb { R } ^ { d } , \quad } & { i = 1 , \ldots , p . } \end{array}\tag{S.35}
$$

Proposition 4.2 (Global rigid-motion invariance of the GOPP). Suppose that

$$
\left( C ^ { \star } , R _ { 1 } ^ { \star } , \ldots , R _ { p } ^ { \star } , \tau _ { 1 } ^ { \star } , \ldots , \tau _ { p } ^ { \star } \right)
$$

is a minimizer of Problem (S.35). For any $\pmb { s } \in \mathbb { R } ^ { d }$ and $Q \in O ( d )$ , define

$$
\widetilde { \pmb { C } } = \left( \pmb { C } ^ { \star } + \mathbf { 1 } _ { r } \pmb { s } ^ { \top } \right) \pmb { Q } ,
$$

and, for every $i \in { \mathcal { P } } ,$

$$
\widetilde { \pmb { R } } _ { i } = \pmb { R } _ { i } ^ { \star } \pmb { Q } , \qquad \widetilde { \pmb { \tau } } _ { i } = \pmb { \tau } _ { i } ^ { \star } - \pmb { R } _ { i } ^ { \star } \pmb { s } .
$$

Then

$$
\left( \widetilde { C } , \widetilde { R } _ { 1 } , \ldots , \widetilde { R } _ { p } , \widetilde { \tau } _ { 1 } , \ldots , \widetilde { \tau } _ { p } \right)
$$

is also a minimizer ofProblem (S.35) and attains the same objective value. Consequently, every minimizer belongs to an equivalence class generated by common translations and orthogonal transformations of the collaboration coordinate system. Therefore, without an external reference or a gauge-fixing constraint, the GOPP objective alone cannot yield a unique recovery ofthe participant-specific GDP parameters $( O _ { i } , \Psi _ { i } )$

Proof. Fix $\pmb { s } \in \mathbb { R } ^ { d }$ and $Q \in O ( d )$ , and define $\widetilde { C } , \widetilde { R } _ { i }$ , and $\widetilde { \tau } _ { i }$ as in the proposition. Since $R _ { i } ^ { \star } \in O ( d )$ and $Q \in O ( d )$

$$
\widetilde { \pmb { R } } _ { i } = \pmb { R } _ { i } ^ { \star } \pmb { Q } \in O ( d ) ,
$$

so the transformed tuple is feasible. For every $i \in { \mathcal { P } } .$

$$
\begin{array} { r l } & { \widetilde { C } - \left( { \cal B } _ { i } - \mathbf { 1 } _ { r } \widetilde { \tau } _ { i } ^ { \intercal } \right) \widetilde { R } _ { i } } \\ & { \quad = \left( C ^ { \star } + \mathbf { 1 } _ { r } s ^ { \intercal } \right) Q - \left[ { \cal B } _ { i } - \mathbf { 1 } _ { r } \left( \tau _ { i } ^ { \star } - R _ { i } ^ { \star } s \right) ^ { \intercal } \right] R _ { i } ^ { \star } Q } \\ & { \quad = \left[ C ^ { \star } - \left( { \cal B } _ { i } - \mathbf { 1 } _ { r } \tau _ { i } ^ { \star \intercal } \right) R _ { i } ^ { \star } \right] Q . } \end{array}
$$

Because $\varrho$ is orthogonal, right multiplication by $\varrho$ preserves the Frobenius norm. Hence,

$$
\left\| \widetilde { C } - \left( B _ { i } - \mathbf { 1 } _ { r } \widetilde { \tau } _ { i } ^ { \intercal } \right) \widetilde { R } _ { i } \right\| _ { F } = \left\| C ^ { \star } - \left( B _ { i } - \mathbf { 1 } _ { r } \tau _ { i } ^ { \star \intercal } \right) R _ { i } ^ { \star } \right\| _ { F } .
$$

Summing the squared identities over $i \in \mathcal { P }$ shows that the transformed tuple has the same objective value as the original minimizer. Since it is feasible, it is also a minimizer of Problem (S.35). Because s and $\varrho$ are arbitrary, the GOPP objective cannot distinguish collaboration coordinate systems related by these common rigid transformations, proving the stated non-identifiability. □

## S.7. Lemma 4.3: Centering the GOPP template

Lemma 4.3 (Centering the GOPP template). Suppose that $\pmb { { B } } _ { i } \in \mathbb { R } ^ { r \times d } , i = 1 , \ldots , p ,$ are observed, and define

$$
\Phi ( C , R _ { 1 } , \ldots , R _ { p } , \tau _ { 1 } , \ldots , \tau _ { p } ) : = \sum _ { i = 1 } ^ { p } \left\| C - \left( B _ { i } - 1 _ { r } \tau _ { i } ^ { \intercal } \right) R _ { i } \right\| _ { F } ^ { 2 } .
$$

Let $\mathcal { F }$ be the feasible set defined by $\pmb { C } \in \mathbb { R } ^ { r \times d } , \pmb { R } _ { i } \in O ( d ) ,$ , and $\tau _ { i } \in \mathbb { R } ^ { d } f o r i = 1 , . . . , p ,$ and let

$$
\mathcal { F } _ { 0 } : = \left\{ ( C , R _ { 1 } , \ldots , R _ { p } , \tau _ { 1 } , \ldots , \tau _ { p } ) \in \mathcal { F } : \mathbf { 1 } _ { r } ^ { \top } C = \mathbf { 0 } _ { d } ^ { \top } \right\} .
$$

Then inf<sub>F</sub> $\Phi = \operatorname { i n f } _ { \mathcal { F } _ { 0 } }$ Φ. Hence, the GOPP may be restricted, without changing its optimal value, to templates whose average row is zero.

Proof. Since ${ \mathcal { F } } _ { 0 } \subseteq { \mathcal { F } }$ , we have

$$
\operatorname* { i n f } _ { \mathcal { F } } \Phi \le \operatorname* { i n f } _ { \mathcal { F } _ { 0 } } \Phi .
$$

It remains to prove the reverse inequality. Fix

$$
( C , R _ { 1 } , \ldots , R _ { p } , \tau _ { 1 } , \ldots , \tau _ { p } ) \in \mathcal { F } .
$$

Let $\pmb { c } \in \mathbb { R } ^ { d }$ be arbitrary and define

$$
\overline { { C } } : = C - \mathbf { 1 } _ { r } c ^ { \top } , \qquad \overline { { \tau } } _ { i } : = \tau _ { i } + R _ { i } c , \qquad i = 1 , \dots , p .
$$

For each i, the corresponding residual satisfies

$$
\begin{array} { r l } & { \overline { { C } } - \left( B _ { i } - \mathbf { 1 } _ { r } \overline { { \tau } } _ { i } ^ { \top } \right) R _ { i } = C - \mathbf { 1 } _ { r } c ^ { \top } - \left( B _ { i } - \mathbf { 1 } _ { r } ( \tau _ { i } + R _ { i } c ) ^ { \top } \right) R _ { i } } \\ & { \qquad = C - \mathbf { 1 } _ { r } c ^ { \top } - \left( B _ { i } - \mathbf { 1 } _ { r } \tau _ { i } ^ { \top } - \mathbf { 1 } _ { r } c ^ { \top } R _ { i } ^ { \top } \right) R _ { i } } \\ & { \qquad = C - \mathbf { 1 } _ { r } c ^ { \top } - B _ { i } R _ { i } + \mathbf { 1 } _ { r } \tau _ { i } ^ { \top } R _ { i } + \mathbf { 1 } _ { r } c ^ { \top } R _ { i } ^ { \top } R _ { i } } \\ & { \qquad = C - B _ { i } R _ { i } + \mathbf { 1 } _ { r } \tau _ { i } ^ { \top } R _ { i } } \\ & { \qquad = C - \left( B _ { i } - \mathbf { 1 } _ { r } \tau _ { i } ^ { \top } \right) R _ { i } , } \end{array}
$$

where we used $\pmb { R } _ { i } ^ { \top } \pmb { R } _ { i } = \pmb { I } _ { d } .$ . Hence the objective value is unchanged by the transformation

$$
C \mapsto C - \mathbf { 1 } _ { r } \pmb { c } ^ { \top } , \qquad \pmb { \tau } _ { i } \mapsto \pmb { \tau } _ { i } + \pmb { R } _ { i } \pmb { c } .
$$

Now choose

$$
\pmb { c } = \frac { 1 } { r } { C } ^ { \top } \mathbf { 1 } _ { r } .
$$

Then

$$
\mathbf { 1 } _ { r } ^ { \top } \overline { { \boldsymbol { C } } } = \mathbf { 1 } _ { r } ^ { \top } \left( \boldsymbol { C } - \mathbf { 1 } _ { r } \boldsymbol { c } ^ { \top } \right) = \mathbf { 1 } _ { r } ^ { \top } \boldsymbol { C } - r \boldsymbol { c } ^ { \top } = \mathbf { 0 } _ { d } ^ { \top } .
$$

Thus, the transformed point belongs to $\mathcal { F } _ { 0 }$ and has the same objective value as the original point. Therefore, every value attained over $\mathcal { F }$ is also attained over $\mathcal { F } _ { 0 }$ , and so

$$
\operatorname* { i n f } _ { \mathcal { F } _ { 0 } } \Phi \le \operatorname* { i n f } _ { \mathcal { F } } \Phi .
$$

Combining the two inequalities gives

$$
\operatorname* { i n f } _ { \mathcal { F } } \Phi = \operatorname* { i n f } _ { \mathcal { F } _ { 0 } } \Phi .
$$

## S.8. Proposition 4.4: Reduction of the centered GOPP to max-trace form

Proposition 4.4 (Reduction of the centered GOPP to max-trace form). Under the notation of Lemma 4.3, define

$$
\pmb { H } : = \pmb { I } _ { r } - \frac { 1 } { r } \pmb { 1 } _ { r } \pmb { 1 } _ { r } ^ { \top } , \qquad \overline { { \pmb { B } } } _ { i } : = \pmb { H } \pmb { B } _ { i } , \qquad i = 1 , \ldots , p .
$$

For fixed $\pmb { R } _ { 1 } , \ldots , \pmb { R } _ { p } \in O ( d )$ , the unique minimizers over the translation variables and the centered template are

$$
\pmb { \tau } _ { i } ^ { \star } = \frac { 1 } { r } \pmb { B } _ { i } ^ { \top } \pmb { 1 } _ { r } , \qquad i = 1 , \dots , p ,\tag{S.36}
$$

and

$$
C ^ { \star } = \frac { 1 } { p } \sum _ { i = 1 } ^ { p } \overline { { B } } _ { i } R _ { i } .\tag{S.37}
$$

Define the stacked rotation matrix and the block Gram matrix $b y$

$$
\pmb { R } : = \left[ \pmb { R } _ { i } \right] _ { i \in \mathcal { P } } \in \mathbb { R } ^ { p d \times d } ,
$$

and

$$
\begin{array} { r } { \pmb { G } _ { i j } : = \overline { { \pmb { B } } } _ { i } ^ { \top } \overline { { \pmb { B } } } _ { j } , \qquad i , j = 1 , \ldots , p . } \end{array}\tag{S.38}
$$

Then the rotation components of the centered GOPP minimizers are precisely the solutions of

$$
\begin{array} { r l } { \underset { R _ { 1 } , \ldots , R _ { p } } { \operatorname* { m a x } } } & { \mathrm { t r } \big ( R ^ { \top } G R \big ) } \\ { s . t . } & { R _ { i } \in O ( d ) , \qquad i = 1 , \ldots , p . } \end{array}\tag{S.39}
$$

$I f ~ V _ { \mathrm { G O P P } } ^ { \star }$ denotes the minimum of Φ over the centered feasible set ${ \mathcal { F } } _ { 0 } ,$ , and $V _ { \mathrm { t r } } ^ { \star }$ denotes the optimal value of Prob lem (S.39), then

$$
V _ { \mathrm { G O P P } } ^ { \star } = \sum _ { i = 1 } ^ { p } \left\| \overline { { \boldsymbol { B } } } _ { i } \right\| _ { F } ^ { 2 } - \frac { 1 } { p } V _ { \mathrm { t r } } ^ { \star } .\tag{S.40}
$$

Conversely, $i f R _ { 1 } ^ { \star } , \ldots , R _ { p } ^ { \star }$ solve Problem (S.39), then Equations (S.36) and (S.37), evaluated at these rotations, produce a centered minimizer ofthe GOPP.

The proposition follows from the next three auxiliary lemmas.

Lemma S.3 (Eliminating translations). For any $\pmb { C } \in \mathbb { R } ^ { r \times d }$ satisfying 1<sup>⊤</sup><sub>r</sub> $\mathbf { \hat { C } } = \mathbf { 0 } _ { d } ^ { \top }$ and any $\pmb { R } _ { 1 } , \ldots , \pmb { R } _ { p } \in O ( d )$

$$
\begin{array} { l } { \displaystyle \operatorname* { m i n } _ { \tau _ { 1 } , \ldots , \tau _ { p } } \Phi ( \boldsymbol { C } , \boldsymbol { R } _ { 1 } , \ldots , \boldsymbol { R } _ { p } , \tau _ { 1 } , \ldots , \tau _ { p } ) } \\ { \displaystyle \qquad = \sum _ { i = 1 } ^ { p } \left\| \boldsymbol { C } - \overline { { \boldsymbol { B } } } _ { i } \boldsymbol { R } _ { i } \right\| _ { F } ^ { 2 } . } \end{array}
$$

The minimizers are unique and are given by

$$
\pmb { \tau } _ { i } ^ { \star } = \frac { 1 } { r } \pmb { B } _ { i } ^ { \top } \pmb { 1 } _ { r } , \qquad i = 1 , \dots , p .
$$

Proof. We first record a simple columnwise least-squares calculation. For any $\pmb { M } \in \mathbb { R } ^ { r \times d }$ , the unique minimizer of

$$
\operatorname* { m i n } _ { \tau \in \mathbb { R } ^ { d } } \left\| M - \mathbf { 1 } _ { r } \tau ^ { \top } \right\| _ { F } ^ { 2 }
$$

is

$$
( { \pmb { \tau } } ^ { \star } ) ^ { \top } = ( \mathbf { 1 } _ { r } ^ { \top } \mathbf { 1 } _ { r } ) ^ { - 1 } \mathbf { 1 } _ { r } ^ { \top } { \pmb { M } } = \frac { 1 } { r } \mathbf { 1 } _ { r } ^ { \top } { \pmb { M } } ,
$$

or equivalently

$$
\boldsymbol { \tau } ^ { \star } = \frac { 1 } { r } \boldsymbol { M } ^ { \intercal } \mathbf { 1 } _ { r } .
$$

The corresponding minimized residual is

$$
M - \mathbf { 1 } _ { r } ( \pmb { \tau } ^ { \star } ) ^ { \top } = \left( \pmb { I } _ { r } - \frac { 1 } { r } \mathbf { 1 } _ { r } \mathbf { 1 } _ { r } ^ { \top } \right) \pmb { M } = \pmb { H } \pmb { M } .
$$

Now fix any

$$
\pmb { C } \in \mathbb { R } ^ { r \times d } , \qquad \mathbf { 1 } _ { r } ^ { \top } \pmb { C } = \mathbf { 0 } _ { d } ^ { \top } , \qquad \pmb { R } _ { i } \in O ( d ) , \qquad i = 1 , \dots , p .
$$

For each i, right multiplication by $\pmb { R } _ { i } ^ { \top }$ preserves the Frobenius norm. Hence

$$
\begin{array} { r l } & { \left\| { C - \left( { { B } _ { i } } - { { \bf { 1 } } _ { r } } { \tau _ { i } ^ { \top } } \right) { { \bf { \cal R } } _ { i } } } \right\| _ { F } ^ { 2 } = \left\| { \left[ { C - \left( { { B } _ { i } } - { { \bf { 1 } } _ { r } } { \tau _ { i } ^ { \top } } \right) { { \cal R } } _ { i }  { \cal R } _ { i } ^ { \top } } } \righ\right]t\| _ { F } ^ { 2 } } \\ & { \qquad = { \left\| { C { \cal R } _ { i } ^ { \top } - { { \cal B } _ { i } } + { { \bf { 1 } } _ { r } } { \tau _ { i } ^ { \top } } } \right\| _ { F } ^ { 2 } } } \\ & { \qquad = { \left\| { { { \cal B } _ { i } } - C { \cal R } _ { i } ^ { \top } - { { \bf { 1 } } _ { r } } { \tau _ { i } ^ { \top } } } \right\| _ { F } ^ { 2 } } . } \end{array}
$$

Applying the preceding least-squares fact with

$$
\pmb { M } _ { i } : = \pmb { B } _ { i } - C \pmb { R } _ { i } ^ { \top }
$$

gives

$$
\pmb { \tau } _ { i } ^ { \star } = \frac { 1 } { r } \pmb { M } _ { i } ^ { \top } \pmb { 1 } _ { r } = \frac { 1 } { r } \left( \pmb { B } _ { i } - \pmb { C } \pmb { R } _ { i } ^ { \top } \right) ^ { \top } \pmb { 1 } _ { r } .
$$

Since $\mathbf { 1 } _ { r } ^ { \top } C = \mathbf { 0 } _ { d } ^ { \top }$ , we have $C ^ { \top } \mathbf { 1 } _ { r } = \mathbf { 0 } _ { d } .$ . Therefore

$$
\pmb { \tau } _ { i } ^ { \star } = \frac { 1 } { r } \pmb { B } _ { i } ^ { \top } \mathbf { 1 } _ { r } .
$$

The minimized value of the i-th term is

$$
\begin{array} { l } { \displaystyle \operatorname* { m i n } _ { \tau _ { i } \in \mathbb { R } ^ { d } } \left\| C - \left( B _ { i } - \mathbf { I } _ { r } \tau _ { i } ^ { \top } \right) R _ { i } \right\| _ { F } ^ { 2 } = \left\| H \left( B _ { i } - C R _ { i } ^ { \top } \right) \right\| _ { F } ^ { 2 } } \\ { \displaystyle = \left\| H B _ { i } - H C R _ { i } ^ { \top } \right\| _ { F } ^ { 2 } } \\ { \displaystyle = \left\| H B _ { i } - C R _ { i } ^ { \top } \right\| _ { F } ^ { 2 } } \\ { \displaystyle = \left\| \left( H B _ { i } - C R _ { i } ^ { \top } \right) R _ { i } \right\| _ { F } ^ { 2 } } \\ { \displaystyle = \left\| \left( H B _ { i } - C R _ { i } ^ { \top } \right) R _ { i } \right\| _ { F } ^ { 2 } } \\ { \displaystyle = \left\| H B _ { i } R _ { i } - C \right\| _ { F } ^ { 2 } } \\ { \displaystyle = \left\| C - H B _ { i } R _ { i } \right\| _ { F } ^ { 2 } . } \end{array}
$$

Here we used $H C = C ,$ , which follows from $\mathbf { 1 } _ { r } ^ { \top } C = \mathbf { 0 } _ { d } ^ { \top }$ , and $\pmb { R } _ { i } ^ { \top } \pmb { R } _ { i } = \pmb { I } _ { d }$

Since $\tau _ { 1 } , \ldots , \tau _ { p }$ appear separately in the objective, the minimization over translations separates across i. Thus, for every feasible $( C , R _ { 1 } , \ldots , R _ { p } )$

$$
\begin{array} { r l } & { \underset { \tau _ { 1 } , \ldots , \tau _ { p } } { \operatorname* { m i n } } \sum _ { i = 1 } ^ { p } { \left\| \pmb { C } - \left( \pmb { B } _ { i } - \mathbf { 1 } _ { r } \pmb { \tau } _ { i } ^ { \top } \right) \pmb { R } _ { i } \right\| _ { F } ^ { 2 } } } \\ & { \qquad = \displaystyle \sum _ { i = 1 } ^ { p } \| \pmb { C } - \pmb { H } \pmb { B } _ { i } \pmb { R } _ { i } \| _ { F } ^ { 2 } . } \end{array}
$$

This establishes both the reduced objective and the uniqueness of the translations.

Lemma S.4 (Eliminating the template). Forfixed $\pmb { R } _ { 1 } , \ldots , \pmb { R } _ { p } \in O ( d )$ , the unique minimizer of

$$
\sum _ { i = 1 } ^ { p } \left\| C - \overline { { B } } _ { i } R _ { i } \right\| _ { F } ^ { 2 }
$$

over $\pmb { C } \in \mathbb { R } ^ { r \times d }$ satisfying $\mathbf { 1 } _ { r } ^ { \top } C = \mathbf { 0 } _ { d } ^ { \top }$ is

$$
C ^ { \star } = \frac { 1 } { p } \sum _ { i = 1 } ^ { p } \overline { { B } } _ { i } R _ { i } .
$$

The corresponding minimized value is

$$
\sum _ { i = 1 } ^ { p } \left\| \overline { { \boldsymbol B } } _ { i } \right\| _ { F } ^ { 2 } - \frac { 1 } { p } \left\| \sum _ { i = 1 } ^ { p } \overline { { \boldsymbol B } } _ { i } \boldsymbol B _ { i } \right\| _ { F } ^ { 2 } .
$$

Proof. Fix $\pmb { R } _ { 1 } , \ldots , \pmb { R } _ { p } \in O ( d )$ . From the definition of H,

$$
\mathbf { 1 } _ { r } ^ { \top } \pmb { H } = \mathbf { 1 } _ { r } ^ { \top } - \frac { 1 } { r } \mathbf { 1 } _ { r } ^ { \top } \mathbf { 1 } _ { r } \mathbf { 1 } _ { r } ^ { \top } = \mathbf { 1 } _ { r } ^ { \top } - \mathbf { 1 } _ { r } ^ { \top } = \mathbf { 0 } _ { r } ^ { \top } .
$$

Thus, for each i,

$$
\mathbf { 1 } _ { r } ^ { \top } \overline { { \mathbf { B } } } _ { i } R _ { i } = \mathbf { 1 } _ { r } ^ { \top } H B _ { i } R _ { i } = \mathbf { 0 } _ { r } ^ { \top } B _ { i } R _ { i } = \mathbf { 0 } _ { d } ^ { \top } .
$$

Hence each $\overline { { B } } _ { i } R _ { i }$ has zero average row. Since the centering constraint is linear, the average of these matrices is also centered across rows. Define

$$
\widehat { \boldsymbol { c } } : = \frac { 1 } { p } \sum _ { i = 1 } ^ { p } \overline { { \boldsymbol { B } } } _ { i } \boldsymbol { R } _ { i } .
$$

Then

$$
\mathbf { 1 } _ { r } ^ { \top } \widehat { \pmb { C } } = \frac { 1 } { p } \sum _ { i = 1 } ^ { p } \mathbf { 1 } _ { r } ^ { \top } \overline { { \pmb { B } } } _ { i } \pmb { R } _ { i } = \mathbf { 0 } _ { d } ^ { \top } .
$$

Thus $\widehat { c }$ is feasible for the constraint $\mathbf { 1 } _ { r } ^ { \top } C = \mathbf { 0 } _ { d } ^ { \top }$ . For any feasible C, we have

$$
\begin{array} { r } { \displaystyle \sum _ { i = 1 } ^ { p } \Big \| C - \overline { { B } } _ { i } R _ { i } \Big \| _ { F } ^ { 2 } = \displaystyle \sum _ { i = 1 } ^ { p } \Big \| ( C - \widehat { C } ) + ( \widehat { C } - \overline { { B } } _ { i } R _ { i } ) \Big \| _ { F } ^ { 2 } } \\ { = p \| C - \widehat { C } \| _ { F } ^ { 2 } + \displaystyle \sum _ { i = 1 } ^ { p } \Big \| \widehat { C } - \overline { { B } } _ { i } R _ { i } \Big \| _ { F } ^ { 2 } , } \end{array}
$$

because the cross term vanishes:

$$
2 \left. C - { \widehat C } , \sum _ { i = 1 } ^ { p } \left( { \widehat C } - { \overline { { B } } } _ { i } R _ { i } \right) \right. _ { F } = 0 ,
$$

since

$$
\sum _ { i = 1 } ^ { p } { \left( \widehat { C } - \widehat { B } _ { i } R _ { i } \right) } = p \widehat { C } - \sum _ { i = 1 } ^ { p } \overline { { B } } _ { i } R _ { i } = 0 .
$$

Hence the objective is minimized uniquely at $C = { \widehat { C } } .$ which proves (S.37). It remains to compute the reduced objective. Using $\begin{array} { r } { \sum _ { i = 1 } ^ { p } \overline { { B } } _ { i } R _ { i } = p C ^ { \star } } \end{array}$ , expanding the squared norms yields

$$
\begin{array} { l } { \displaystyle \operatorname* { m i n } _ { { t \in \mathbb { R } ^ { n \times d } } } \displaystyle \sum _ { i = 1 } ^ { p } \left\| C - \overline { { B } } _ { i } R _ { i } \right\| _ { F } ^ { 2 } = \displaystyle \sum _ { i = 1 } ^ { p } \left\| \overline { { B } } _ { i } R _ { i } - C ^ { \star } \right\| _ { F } ^ { 2 } } \\ { \displaystyle \quad \mathrm { ~ } = \sum _ { i = 1 } ^ { p } \left\| \overline { { B } } _ { i } R _ { i } \right\| _ { F } ^ { 2 } - p \| C ^ { \star } \| _ { F } ^ { 2 } } \\ { \displaystyle \quad = \sum _ { i = 1 } ^ { p } \left\| \overline { { B } } _ { i } \right\| _ { F } ^ { 2 } - \displaystyle \frac { 1 } { p } \left\| \sum _ { i = 1 } ^ { p } \overline { { B } } _ { i } R _ { i } \right\| _ { F } ^ { 2 } . } \end{array}
$$

This proves the stated template update and reduced objective.

Lemma S.5 (Max-trace identity). For any $\pmb { R } _ { 1 } , \ldots , \pmb { R } _ { p } \in O ( d )$ , let

$$
\pmb { R } = \left[ \pmb { R } _ { i } \right] _ { i \in \mathcal { P } } .
$$

With G defined by Equation (S.38),

$$
\left\| \sum _ { i = 1 } ^ { p } \overline { { B } } _ { i } R _ { i } \right\| _ { F } ^ { 2 } = \operatorname { t r } \big ( R ^ { \top } G R \big ) .
$$

Proof. Let

$$
\pmb { R } = \left[ \pmb { R } _ { i } \right] _ { i \in \mathcal { P } } .
$$

Then

$$
\begin{array} { r l } { \displaystyle \left\| \displaystyle \sum _ { i \in \mathcal { P } } \overline { { B } } _ { i } R _ { i } \right\| _ { F } ^ { 2 } = \mathrm { t r } \left[ \left( \displaystyle \sum _ { i \neq \mathcal { P } } \overline { { B } } _ { i } R _ { i } \right) ^ { \top } \left( \displaystyle \sum _ { j \neq \mathcal { P } } \overline { { B } } _ { j } R _ { j } \right) \right] } & { } \\ { = \displaystyle \sum _ { i \in \mathcal { P } } \displaystyle \sum _ { j \neq \mathcal { P } } \mathrm { t r } \left( R _ { i } ^ { \top } \overline { { B } } _ { i } ^ { \top } \overline { { B } } _ { j } R _ { j } \right) } & { } \\ { = \displaystyle \sum _ { i \in \mathcal { P } } \displaystyle \sum _ { j \neq \mathcal { P } } \mathrm { t r } \left( R _ { i } ^ { \top } G _ { i j } R _ { j } \right) } & { } \\ { = \mathrm { t r } \left( R ^ { \top } G R \right) . } \end{array}
$$

This proves the identity.

ProofofProposition 4.4. Lemma S.3 gives the unique translation updates in Equation (S.36) and reduces the centered GOPP objective to an optimization over the centered template and rotations. Lemma S.4 then gives the unique template update in Equation (S.37) and shows that, for fixed rotations, the minimized objective equals

$$
\sum _ { i = 1 } ^ { p } \left\| \overline { { \boldsymbol B } } _ { i } \right\| _ { F } ^ { 2 } - \frac { 1 } { p } \left\| \sum _ { i = 1 } ^ { p } \overline { { \boldsymbol B } } _ { i } \boldsymbol B _ { i } \right\| _ { F } ^ { 2 } .
$$

By Lemma S.5, the squared norm in the second term is $\operatorname { t r } ( R ^ { \intercal } G R )$ . The first term is independent of the rotations, and $- 1 / p < 0 ;$ ; hence minimizing the centered GOPP is equivalent, for the rotation variables, to solving Problem (S.39). The same identity gives Equation (S.40).

Every centered GOPP minimizer must therefore have rotations that solve Problem (S.39) and must use the unique translation and template updates above. Conversely, any maximizer $R _ { 1 } ^ { \star } , \ldots , R _ { p } ^ { \star }$ of Problem (S.39), augmented with those updates, attains $V _ { \mathrm { G O P P } } ^ { \star }$ and is consequently a centered GOPP minimizer. □

Problem (S.39) is nonconvex and is generally dificult to solve globally. We use the spectral initialization and blockwise generalized power method of [7], specialized only in notation and dimensions to the present alignment problem. For a square matrix $S \in \mathbb { R } ^ { d \times d }$ with singular value decomposition $\pmb { S } = \pmb { U } \pmb { \Sigma } \pmb { V } ^ { \top }$ , define the orthogonal polar projection

$$
\Pi ( S ) = U V ^ { \top } \in O ( d ) .\tag{S.41}
$$

If S is rank deficient, the nearest orthogonal factor may be nonunique; in that case, any choice obtained from a ful singular value decomposition may be used. For a stacked matrix

$$
\pmb { M } = \left[ \pmb { M } _ { i } \right] _ { i \in \mathcal { P } } \in \mathbb { R } ^ { p d \times d } , \qquad \pmb { M } _ { i } \in \mathbb { R } ^ { d \times d } ,
$$

define the blockwise polar projection by

$$
\Pi _ { p } ( { \pmb M } ) : = \left[ \Pi ( { \pmb M } _ { i } ) \right] _ { i \in \mathcal { P } } .\tag{S.42}
$$

Algorithm 4: GPM alignment routine specialized from [7]   
Input: Block Gram matrix $G \in \mathbb { R } ^ { p d \times p d } ;$ maximum iterations $T _ { \mathrm { m a x } } ;$ tolerance $\varepsilon > 0 .$   
Output: Estimated stacked rotation $\widehat { \pmb { R } } \in \mathbb { R } ^ { p d \times d }$ , with block rows $\widehat { \pmb { R } } _ { i } \in O ( d ) , i = 1 , \ldots , p .$   
Compute the top d eigenvectors of G and stack them as $U \in \mathbb { R } ^ { p d \times d }$   
Initialize   
$\pmb { R } ^ { ( 0 ) } = \Pi _ { p } ( \pmb { U } )$   
Set $T _ { \mathrm { o u t } } = T _ { \mathrm { m a x } }$   
for $t = 0 , \ldots , T _ { \mathrm { m a x } } - 1$ do   
Compute   
$\pmb { M } ^ { ( t ) } = \pmb { G } \pmb { R } ^ { ( t ) }$   
Update   
$\pmb { R } ^ { ( t + 1 ) } = \Pi _ { p } ( \pmb { M } ^ { ( t ) } )$   
if $\| \pmb { R } ^ { ( t + 1 ) } - \pmb { R } ^ { ( t ) } \| _ { F } \leq \varepsilon$ then   
$T _ { \mathrm { o u t } } = t + 1$   
break   
Set $\widehat { \pmb { R } } = \pmb { R } ^ { ( T _ { \mathrm { o u t } } ) }$ , where   
$\widehat { \pmb { R } } = \left[ \widehat { \pmb { R } } _ { i } \right] _ { i \in \mathcal { P } }$   
return $\widehat { R }$

## S.9. Proposition 4.5: Residual Translation Under Gaussian Anchor Noise

Under the anchor-noise model in Definition 4.1, define the average anchor-noise vector

$$
\pmb { \eta } _ { i } : = \frac { \nu } { r } \pmb { W } _ { i } ^ { \top } \mathbf { 1 } _ { r } .\tag{S.43}
$$

For this result, let

$$
\pmb { H } = \pmb { I } _ { r } - \frac { 1 } { r } \pmb { 1 } _ { r } \pmb { 1 } _ { r } ^ { \top } .
$$

Proposition 4.5 (Residual Translation Under Gaussian Anchor Noise). Under the Gaussian anchor-noise model above, suppose that the matrices $W _ { i } , i \in { \mathcal { S } } ,$ , are mutually independent and that the estimated rotations are computed exclusivelyfrom the centered anchor uploads. Then

$$
\pmb { \eta } _ { i } \sim { \cal N } \bigg ( \mathbf { 0 } _ { d } , \frac { \nu ^ { 2 } } { r } { \cal I } _ { d } \bigg ) , \qquad i \in \mathcal { P } ,\tag{S.44}
$$

and the collection $\{ \pmb { \eta } _ { i } \} _ { i \in \mathcal { P } }$ is independent of the estimated rotations. Consequently, conditionally on the estimated rotations, the vectors $\widehat { R } _ { i } ^ { \top } \eta _ { i } , i \in { \mathcal P } ,$ , are independent and eachfollows

$$
\widehat { \pmb { R } } _ { i } ^ { \top } \pmb { \eta } _ { i } \sim { \cal N } \bigg ( \pmb { 0 } _ { d } , \frac { \nu ^ { 2 } } { r } { \cal I } _ { d } \bigg ) .\tag{S.45}
$$

For every fixed pair $i \neq j ,$

$$
\widehat { \pmb { R } } _ { i } ^ { \top } \pmb { \eta } _ { i } - \widehat { \pmb { R } } _ { j } ^ { \top } \pmb { \eta } _ { j } \sim N \bigg ( \pmb { 0 } _ { d } , \frac { 2 \nu ^ { 2 } } { r } \pmb { I } _ { d } \bigg ) ,\tag{S.46}
$$

and hence

$$
\mathbb { E } \bigg [ \bigg \| \widehat { \pmb { R } } _ { i } ^ { \top } \pmb { \eta } _ { i } - \widehat { \pmb { R } } _ { j } ^ { \top } \pmb { \eta } _ { j } \bigg \| _ { 2 } ^ { 2 } \bigg ] = \frac { 2 d \nu ^ { 2 } } { r } .\tag{S.47}
$$

Moreover, for every $0 < \zeta < 1$

$$
\left\| \widehat { R } _ { i } ^ { \top } \eta _ { i } - \widehat { R } _ { j } ^ { \top } \eta _ { j } \right\| _ { 2 } \leq \frac { \sqrt { 2 } \nu } { \sqrt { r } } \left( \sqrt { d } + \sqrt { 2 \log ( 1 / \zeta ) } \right)\tag{S.48}
$$

with probability at least $1 - \zeta .$

Proof. For each coordinate $k \in \{ 1 , . . . , d \} , \eta _ { i , k }$ is v times the average of $r$ independent standard Gaussian variables. Distinct coordinates are formed from distinct columns of $W _ { i }$ and are therefore independent. Hence,

$$
\pmb { \eta } _ { i } \sim \mathcal { N } \bigg ( \pmb { 0 } _ { d } , \frac { \nu ^ { 2 } } { r } \pmb { I } _ { d } \bigg ) .
$$

The vector $\pmb { \eta } _ { i }$ and the centered matrix ${ H W _ { i } }$ are jointly Gaussian. For coordinates $k , \ell \in \{ 1 , \dots , d \}$ and a row index $a \in \{ 1 , \ldots , r \}$

$$
\begin{array} { r } { \cos \left( \eta _ { i , k } , ( H W _ { i } ) _ { a \ell } \right) = \frac { \nu } { r } \delta _ { k \ell } \left( \mathbf { 1 } _ { r } ^ { \top } H \right) _ { a } } \\ { = 0 , \qquad } \end{array}
$$

because $\mathbf { 1 } _ { r } ^ { \top } \pmb { H } = \mathbf { 0 } _ { r } ^ { \top }$ . Thus $\pmb { \eta } _ { i }$ and ${ H W _ { i } }$ are independent. Mutual independence of the participant noise matrices implies that the collection $\{ \pmb { \eta } _ { i } \} _ { i \in \mathcal { P } }$ is independent of $\{ H W _ { i } \} _ { i \in \mathcal { P } }$ , and hence of the estimated rotations.

Conditionally on the estimated rotations, orthogonal invariance gives

$$
\widehat { \pmb { R } } _ { i } ^ { \top } \pmb { \eta } _ { i } \sim \mathcal { N } \bigg ( \mathbf { 0 } _ { d } , \frac { \nu ^ { 2 } } { r } \widehat { \pmb { R } } _ { i } ^ { \top } \widehat { \pmb { R } } _ { i } \bigg ) = \mathcal { N } \bigg ( \mathbf { 0 } _ { d } , \frac { \nu ^ { 2 } } { r } \pmb { I } _ { d } \bigg ) .
$$

The transformed vectors remain conditionally independent. Hence, for every fixed $i \neq j ,$ their diference is Gaussian with covariance $2 \nu ^ { 2 } I _ { d } / r ,$ proving Equation (S.46). Taking the trace of this covariance gives Equation (S.47).

Finally, write

$$
\widehat { \pmb { R } } _ { i } ^ { \top } \pmb { \eta } _ { i } - \widehat { \pmb { R } } _ { j } ^ { \top } \pmb { \eta } _ { j } = \frac { \sqrt { 2 } \nu } { \sqrt { r } } \pmb { g } , \qquad \pmb { g } \sim { \cal N } ( \pmb { 0 } _ { d } , \pmb { I } _ { d } ) .
$$

The standard Gaussian norm concentration inequality gives

$$
\operatorname* { P r } \bigl ( \| g \| _ { 2 } > \sqrt { d } + \sqrt { 2 \log ( 1 / \zeta ) } \bigr ) \leq \zeta ,
$$

which proves the stated tail bound.

□

We measure the rotational component of the alignment error by

$$
\mathcal { E } : = \operatorname* { m i n } _ { Q \in O \left( d \right) } \operatorname* { m a x } _ { i \in \mathcal { P } } \left\| O _ { i } \widehat { R } _ { i } - Q \right\| _ { F } .\tag{S.49}
$$

After transposition and restriction to the centered row subspace, the centered anchor observations form an instance of the normalized Gaussian GOPP studied in [7]. The following statement is a protocol-specific specialization of $[ 7 ,$ Theorems 3.2 and 3.3], not a new general GOPP convergence theorem. The displayed constants result from substituting the centered orthonormal-anchor dimensions into the suficient conditions traced in the cited proofs.

## S.10. Theorem 4.6: Specialization of the GPM Guarantee to a Centered Orthonormal Anchor Matrix

Theorem 4.6 (Specialization of the GPM Guarantee to a Centered Orthonormal Anchor Matrix). Let $\gamma > 2 ,$ , and apply the spectral initializer and GPM prescribed in $I 7 J$ to Problem (S.39). An explicit conservative suficient condition obtainedfrom the proofof[7, Theorem 3.2] is

$$
\nu \leq \frac { 1 } { 1 4 3 3 6 } \frac { \sqrt { p } } { \sqrt { d } \left( \sqrt { p d } + \sqrt { r - 1 } + \sqrt { 2 \gamma p \log p } \right) } .\tag{S.50}
$$

Under this condition, with probability at least $1 - O ( p ^ { - \gamma + 2 } )$ , the GPM converges linearly to a global maximizer

$$
\widehat { \pmb { R } } = \left[ \widehat { \pmb { R } } _ { i } \right] _ { i \in \mathcal { P } } \in \mathbb { R } ^ { p d \times d }
$$

of Problem (S.39). Every global maximizer is of the form $\widehat { R } Q$ for some $Q \in O ( d )$ , and $\widehat { R R } ^ { \top }$ is the unique optimizer of the associated semidefinite relaxation. Moreover, the limiting rotations satisfy

$$
\mathcal { E } \leq \frac { 6 } { 1 - 1 / 1 0 2 4 } \frac { \nu \sqrt { d } \left( \sqrt { p d } + \sqrt { r - 1 } + \sqrt { 2 \gamma p \log p } \right) } { \sqrt { p } } .\tag{S.51}
$$

Proof of the specialization. Let $U _ { \perp } \in \mathbb { R } ^ { r \times ( r - 1 ) }$ have orthonormal columns spanning ${ \mathbf { 1 } } _ { r } ^ { \perp }$ . Then

$$
\begin{array} { r } { \pmb { U } _ { \bot } ^ { \top } \pmb { U } _ { \bot } = \pmb { I } _ { r - 1 } , \qquad \pmb { H } = \pmb { U } _ { \bot } \pmb { U } _ { \bot } ^ { \top } . } \end{array}
$$

For each $i \in \mathcal { P }$ , define the restricted centered observation, signal matrix, and noise matrix by

$$
\begin{array} { r } { D _ { \mathrm { L } , i } : = \overline { { B } } _ { i } ^ { \top } U _ { \bot } , \qquad A _ { \mathrm { L } } : = A ^ { \top } U _ { \bot } , \qquad W _ { \mathrm { L } , i } : = W _ { i } ^ { \top } U _ { \bot } . } \end{array}
$$

Centering removes the translation component of the anchor upload, and hence

$$
\pmb { D } _ { \mathrm { L } , i } = \pmb { O } _ { i } ^ { \top } A _ { \mathrm { L } } + \nu \pmb { W } _ { \mathrm { L } , i } .\tag{S.52}
$$

Because the entries of $W _ { i }$ are jointly Gaussian and $U _ { \perp }$ has orthonormal columns, $W _ { \mathrm { L } , i }$ has independent standard Gaussian entries. Independence of the noise matrices across participants is also preserved. Equation (S.52) is therefore the Gaussian observation model studied in [7], with signal matrix $A _ { \mathrm { I } }$ and planted rotations $\pmb { O } _ { i } ^ { \top }$

Vertically stack the restricted observations as

$$
\begin{array} { r } { D _ { \mathrm { L } } : = \left[ D _ { \mathrm { L } , i } \right] _ { i \in \mathcal { P } } \in \mathbb { R } ^ { p d \times ( r - 1 ) } . } \end{array}
$$

The $( i , j )$ block of its Gram matrix satisfies

$$
\begin{array} { r l } & { \left( \boldsymbol { D } _ { \mathrm { L } } \boldsymbol { D } _ { \mathrm { L } } ^ { \top } \right) _ { i j } = \boldsymbol { D } _ { \mathrm { L } , i } \boldsymbol { D } _ { \mathrm { L } , j } ^ { \top } } \\ & { \quad \quad \quad = \overline { { \boldsymbol B } } _ { i } ^ { \top } \boldsymbol U _ { \perp } \boldsymbol U _ { \perp } ^ { \top } \overline { { \boldsymbol B } } _ { j } } \\ & { \quad \quad \quad = \overline { { \boldsymbol B } } _ { i } ^ { \top } \boldsymbol H \overline { { \boldsymbol B } } _ { j } } \\ & { \quad \quad \quad = \overline { { \boldsymbol B } } _ { i } ^ { \top } \overline { { \boldsymbol B } } _ { j } } \\ & { \quad \quad \quad = G _ { i j } . } \end{array}\tag{S.53}
$$

The fourth equality follows because $\begin{array} { r } { { \cal H } \overline { { \cal B } } _ { j } = \overline { { \cal B } } _ { j } } \end{array}$ . Consequently,

$$
\pmb { D } _ { \mathrm { L } } \pmb { D } _ { \mathrm { L } } ^ { \top } = \pmb { G } .\tag{S.54}
$$

For the stacked rotation matrix R, Equation (S.54) gives

$$
\begin{array} { r l } & { \mathrm { t r } \big ( R ^ { \top } G R \big ) = \big \langle G , R R ^ { \top } \big \rangle } \\ & { \qquad = \big \langle D _ { \mathrm { L } } D _ { \mathrm { L } } ^ { \top } , R R ^ { \top } \big \rangle } \\ & { \qquad = \big \| D _ { \mathrm { L } } ^ { \top } R \big \| _ { F } ^ { 2 } . } \end{array}\tag{S.55}
$$

Together with the identical block constraint $\pmb { R } _ { i } \in O ( d )$ , Equation (S.55) shows that Problem (S.39) is exactly the nonconvex program in [7, Equation (2.4)]. In particular, the matrix denoted by D in [7] corresponds to $\pmb { D } _ { \mathrm { L } }$ , while its block Gram matrix $\pmb { C } = \pmb { D } \pmb { D } ^ { \top }$ corresponds exactly to G. No rescaling of the objective is required.

The remaining correspondence with the notation of [7] is

$$
n _ { \mathrm { L } } = p , \qquad m _ { \mathrm { L } } = r - 1 , \qquad d _ { \mathrm { L } } = d , \qquad \sigma _ { \mathrm { L } } = \nu .
$$

Moreover, the centered orthonormal-anchor conditions imply

$$
\begin{array} { r l } & { A _ { \mathrm { L } } A _ { \mathrm { L } } ^ { \top } = A ^ { \top } U _ { \perp } U _ { \perp } ^ { \top } A } \\ & { \qquad = A ^ { \top } H A } \\ & { \qquad = A ^ { \top } A } \\ & { \qquad = I _ { d } . } \end{array}
$$

Thus,

$$
\sigma _ { \mathrm { m i n } } ( A _ { \mathrm { L } } ) = \| A _ { \mathrm { L } } \| = 1 , \qquad \kappa = 1 .
$$

In the spectral-initialization argument of [7, Section 5.5], ϵ is the prescribed tolerance that places the spectral initializer and its leave-one-out counterparts inside the local convergence neighborhood. It is fixed by

$$
\epsilon = \frac { 1 } { 3 2 \kappa ^ { 2 } \sqrt { d } } , ~ \mathrm { e q u i v a l e n t l y } ~ \epsilon \sqrt { d } = \frac { 1 } { 3 2 \kappa ^ { 2 } } .
$$

The proof first controls the distance from the spectral initializer to the planted solution. The constant $1 1 2 \kappa ^ { 2 }$ results from applying the polar-projection perturbation bound to the blockwise singular-vector estimate, and Equation (5.40) of [7] gives the suficient condition

$$
\sigma _ { \mathrm { L } } \leq \frac { \epsilon \sqrt { d } \sigma _ { \operatorname* { m i n } } ( A _ { \mathrm { L } } ) } { 1 1 2 \kappa ^ { 2 } } \frac { \sqrt { n _ { \mathrm { L } } } } { \sqrt { d } \left( \sqrt { n _ { \mathrm { L } } d } + \sqrt { m _ { \mathrm { L } } } + \sqrt { 2 \gamma n _ { \mathrm { L } } \log n _ { \mathrm { L } } } \right) } .
$$

The subsequent leave-one-out argument must also control the distance between the full spectral initializer and each auxiliary initializer. In [7, Section 5.5], Lemma 5.14 contributes the factor 56κ in the leave-one-out singular-vector bound, while Lemma 5.5 contributes 8κ when that bound is transferred through the polar projection. Their product gives $4 4 8 \kappa ^ { 2 }$ , and requiring the resulting distance to be at most $\varepsilon { \sqrt { d } }$ yields the stronger condition

$$
\sigma _ { \mathrm { L } } \leq \frac { \epsilon \sqrt { d } \sigma _ { \operatorname* { m i n } } ( A _ { \mathrm { L } } ) } { 4 4 8 \kappa ^ { 2 } } \frac { \sqrt { n _ { \mathrm { L } } } } { \sqrt { d } \left( \sqrt { n _ { \mathrm { L } } d } + \sqrt { m _ { \mathrm { L } } } + \sqrt { 2 \gamma n _ { \mathrm { L } } \log n _ { \mathrm { L } } } \right) } .
$$

Because $4 4 8 > 1 1 2$ , the leave-one-out condition implies the first condition and is therefore the active restriction. Substituting ϵ $\sqrt { d } = 1 / ( 3 2 \kappa ^ { 2 } )$ gives the prefactor

$$
\frac { \sigma _ { \mathrm { m i n } } ( A _ { \mathrm { L } } ) } { 1 4 3 3 6 \kappa ^ { 4 } } .
$$

In the present specialization, $\sigma _ { \mathrm { m i n } } ( A _ { \mathrm { L } } ) = \kappa = 1$ . Substituting $n _ { \mathrm { L } } = p , m _ { \mathrm { L } } = r - 1$ , and $\sigma _ { \mathrm { { L } } } = \nu$ therefore yields Equation (S.50).

The blockwise estimate established in the proof of [7, Theorem 3.3] has the explicit leading factor

$$
\frac { 6 \kappa ^ { 2 } } { ( 1 - \epsilon ^ { 2 } d ) | | A _ { \mathrm { L } } | | } .
$$

Here,

$$
\kappa = \| A _ { \mathrm { L } } \| = 1 , \qquad \epsilon ^ { 2 } d = \frac { 1 } { 1 0 2 4 } ,
$$

which gives the factor $6 / ( 1 - 1 / 1 0 2 4 )$ . Since the planted rotations in the specialized instance are $\pmb { O } _ { i } ^ { \top }$ , orthogonal invariance of the Frobenius norm gives

$$
\begin{array} { r l } & { \underset { Q \in O ( d ) } { \operatorname* { m i n } } \underset { i \in \mathcal { P } } { \operatorname* { m a x } } \left\| \widehat { \pmb { R } } _ { i } - \pmb { O } _ { i } ^ { \top } \pmb { Q } \right\| _ { F } = \underset { Q \in O ( d ) } { \operatorname* { m i n } } \underset { i \in \mathcal { P } } { \operatorname* { m a x } } \left\| \pmb { O } _ { i } \widehat { \pmb { R } } _ { i } - \pmb { Q } \right\| _ { F } } \\ & { \qquad = \mathcal { E } . } \end{array}
$$

The specialized blockwise estimate is therefore Equation (S.51). The convergence, global-optimality, global-maximizerorbit, semidefinite-relaxation, and probability conclusions follow from [7, Theorem 3.2], while the blockwise estimation bound follows from [7, Theorem 3.3]. □

![](images/b5100c9a86f9ead6bc1361b504f9187021a866f54678486d448385c35c1ae9bd.jpg)  
Figure S.1: MNIST digit 0: original images and reconstructions ordered from left to right by decreasing full-test exact-source linkage.

## S.11. Additional MNIST Reconstruction Examples

The following panels complement the mixed-digit reconstruction panel in the main manuscript by holding the digit class fixed. Each column uses a distinct source record of the stated digit and is paired vertically across all rows. Within every noisy row, the leftmost reconstruction is the zero-noise control, and the remaining saved reconstructions are ordered by decreasing full-test exact-source linkage, ending at an observed chance or chance-compatible value. Percentages beneath the reconstructions are computed over the complete 1,000-query linkage evaluation; they are not per-digit or per-image probabilities. The panels are qualitative, while all reported privacy values are computed over the complete target test set.

## S.12. Identity-Specific CelebA Reconstruction Examples

The following figures complement the mixed-identity CelebA reconstruction figure in the main manuscript by holding the source identity fixed. Figure S.11 uses the identity displayed in the first column of the mixed figure, Figure S.12 uses the identity in its second column, and so forth. All ten columns within each supplementary figure therefore show the same source face. The original and noiseless C-GDP rows repeat their fixed reference images, whereas each noisy row follows the same leakage-ordered sequence of completed protocol realizations used in the main figure: the first column is the zero-noise control, and cross-image identity-linkage accuracy decreases from left to right. The percentages printed beneath the reconstructions are the saved draw-level linkage accuracies evaluated over the complete privacy test set, rather than scores computed from the single displayed face. The rightmost AA-I-GDP (anchor noise) reconstructions reach the approximately 10% chance level; the rightmost C-GDP (private-data noise) reconstruction is the lowest-leakage realization attained within its calibrated range.

![](images/281e6f3b4164924d6a82a89c2e1de3fb4eed2a22961026a64e7fb748945ba89b.jpg)  
Figure S.2: MNIST digit 1: original images and reconstructions ordered from left to right by decreasing full-test exact-source linkage.

![](images/51bcac405baa57377c53a9e622c3122e90b13ac98d2088ad251209556e812052.jpg)  
Figure S.3: MNIST digit 2: original images and reconstructions ordered from left to right by decreasing full-test exact-source linkage.

![](images/2335b68f891074418dd29be14281889ffa77cd14e6ba2218af7901926b8afc67.jpg)  
Figure S.4: MNIST digit 3: original images and reconstructions ordered from left to right by decreasing full-test exact-source linkage.

![](images/4396403f448a780737d8fe108f02153dd84b79a6077bdfe1263ea8941ea26e68.jpg)  
Figure S.5: MNIST digit 4: original images and reconstructions ordered from left to right by decreasing full-test exact-source linkage.

![](images/3f283d0b497017f7e17506a146b276cd468a3d3180dbf246b72dd85e23d8fa46.jpg)  
Figure S.6: MNIST digit 5: original images and reconstructions ordered from left to right by decreasing full-test exact-source linkage.

![](images/4cffbc4fcd87110568a88fab25bc6b7cb854f333fcfe3e81eda453bb93d5e565.jpg)  
Figure S.7: MNIST digit 6: original images and reconstructions ordered from left to right by decreasing full-test exact-source linkage.

![](images/5a06af922139bb4c8e86a1ce17c4d73f169c3aca767e9c3855563ab2cfd4ebd9.jpg)  
Figure S.8: MNIST digit 7: original images and reconstructions ordered from left to right by decreasing full-test exact-source linkage.

![](images/bc72261899471773ed862ea22bec177a9b43cf61d322f85a2cc8ed73158e1448.jpg)  
Figure S.9: MNIST digit 8: original images and reconstructions ordered from left to right by decreasing full-test exact-source linkage.

![](images/39c7f8e2423de11c0a867cbc7d300bfed7b9367e20fdb0fe8cccd4147b37f64a.jpg)  
Figure S.10: MNIST digit 9: original images and reconstructions ordered from left to right by decreasing full-test exact-source linkage.

![](images/5a565834bd78ad8c48173030d1ac3c026e2f44b128eb4046dd43fee82b0a6af3.jpg)  
Figure S.11: CelebA displayed identity 1: original image and reconstructions across the leakage-ordered attack settings.

## References

[1] R. Penrose, A generalized inverse for matrices, Proceedings of the Cambridge Philosophical Society 51 (3) (1955) 406–413. doi:10.1017/S0305004100030401.

[2] C. Dwork, F. McSherry, K. Nissim, A. D. Smith, Calibrating noise to sensitivity in private data analysis, in: S. Halevi, T. Rabin (Eds.), Theory of Cryptography (TCC 2006), Vol. 3876 of Lecture Notes in Computer Science, Springer, 2006, pp. 265–284. doi:10.1007/11681878\_14.

[3] B. Balle, Y.-X. Wang, Improving the gaussian mechanism for diferential privacy: Analytical calibration and optimal denoising, in: Proceedings of the 35th International Conference on Machine Learning (ICML), Vol. 80 of Proceedings of Machine Learning Research, PMLR, 2018, pp. 394–403. URL https://proceedings.mlr.press/v80/balle18a.html

[4] C. Dwork, A. Roth, The algorithmic foundations of diferential privacy, Foundations and Trends in Theoretical Computer Science 9 (3–4) (2014) 211–407. doi:10.1561/0400000042.

[5] A. P. Sanil, A. F. Karr, X. Lin, J. P. Reiter, Privacy preserving regression modelling via distributed computation, in: Proceedings of the Tenth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), ACM, 2004, pp. 677–682. doi:10.1145/1014052.1014139.

[6] H. Yamashiro, K. Omote, A. Imakura, T. Sakurai, Toward the application of diferential privacy to data collabo ration, IEEE Access 12 (2024) 63292–63301. doi:10.1109/ACCESS.2024.3396146.

![](images/9fbba5965b94d48101e54f6c5293766df90ebc5872be7385a804ace0d23cdd61.jpg)  
Figure S.12: CelebA displayed identity 2: original image and reconstructions across the leakage-ordered attack settings.

![](images/eae196320bd14d503c58b76c9ff458923bd0b52a3ea6a1a32ba7ce0a1b52b640.jpg)  
Figure S.13: CelebA displayed identity 3: original image and reconstructions across the leakage-ordered attack settings.

![](images/62d81ed7e540fbc26b5dee224e600a3af7412570bcf1300af2aafee0ae116b7b.jpg)  
Figure S.14: CelebA displayed identity 4: original image and reconstructions across the leakage-ordered attack settings.

![](images/5a08c0da7fe49824e5fa46407f94974c3e18ce13eedd30de5e7007241ec36f23.jpg)  
Figure S.15: CelebA displayed identity 5: original image and reconstructions across the leakage-ordered attack settings.

![](images/7f6b6684ea6fd8e4fe1b787c8e92bfadd64acb8ec7fa5d80852f5367cad3ab33.jpg)  
Figure S.16: CelebA displayed identity 6: original image and reconstructions across the leakage-ordered attack settings.

![](images/9c87c1d0090ec3e0cbc405a10202ef8cd8746d573967297f14666e2f1daadbed.jpg)  
Figure S.17: CelebA displayed identity 7: original image and reconstructions across the leakage-ordered attack settings.

![](images/7d37a4289890a6e6a4557c9d065e28d15818c7037da5756b4c9503fb3eaa89a1.jpg)  
Figure S.18: CelebA displayed identity 8: original image and reconstructions across the leakage-ordered attack settings.

![](images/7a950f072b30eb938d624f39b739e9bd77652465b63fc637b81edc5dc65d090c.jpg)  
Figure S.19: CelebA displayed identity 9: original image and reconstructions across the leakage-ordered attack settings.

![](images/8b0dade917c7c27ac945d1a44bd35eb9a7a4fecb0f0d9c769d325c0d22280cb8.jpg)  
Figure S.20: CelebA displayed identity 10: original image and reconstructions across the leakage-ordered attack settings.  
[7] S. Ling, Near-optimal bounds for the generalized orthogonal procrustes problem via generalized power method, Applied and Computational Harmonic Analysis 66 (2023) 62–100. doi:10.1016/j.acha.2023.04.008.