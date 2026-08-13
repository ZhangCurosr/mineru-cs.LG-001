# FLARE++: LOW-RANK ATTENTION WITH DYNAMIC ATTENTION ROUTING

Vedant Puri, Yongjie Jessica Zhang & Levent Burak Kara   
Department of Mechanical Engineering   
Carnegie Mellon University   
Pittsburgh, Pennsylvania, USA

## ABSTRACT

Full self-attention (Vaswani et al., 2017) is a strong token mixer for PDE surrogates on irregular domains, but its quadratic cost limits its use on high-resolution problems. Efficient latent-attention models such as the Fast Low-rank Attention Routing Engine (FLARE) (Puri et al., 2026) avoid that cost by routing all N tokens through M N learned latent queries, but those queries are parameters: once trained, the same learned query templates serve every input. We remove this restriction with FLARE++, a low-rank attention architecture with dynamic token routing. FLARE++ reuses FLARE’s own encoder to build its routing queries: learned latent seeds drive one extra encode call that gathers the N input tokens into M input-conditioned queries, and those queries then determine how the same tokens are compressed and redistributed. This preserves FLARE’s explicit lowrank factorization and linear (NM) complexity, and expresses the complete routing operation with standard scaled dot-product attention (SDPA) calls alone. We also provide a multi-GPU context-parallel implementation that shards input tokens across devices without ever gathering the full token sequence on one of them. FLARE++ is competitive across a set of standard PDE surrogate benchmarks, improving on fixed-query FLARE by 24% on average, and it gains 2.3 points of average accuracy on Long Range Arena.

## 1 INTRODUCTION

Self-attention has become a dominant architecture for PDE surrogates because it lets every discretization point communicate with every other one. In every surrogate we consider, each discretization point is embedded as its own token, so a mesh of N points is a sequence of N tokens; we use the latter term throughout, since it is the entity the mixer acts on. The cost of that generality is an N N communication matrix and therefore (N<sup>2</sup>) work in the number of tokens (Vaswani et al., 2017), which is out of reach on the meshes engineering problems actually produce.

A range of efficient token mixers communicate instead through M N latent tokens. We build on FLARE (Puri et al., 2026), which uses latent tokens only for routing: for each attention head, M learned latent queries gather the N input tokens into M latent tokens through one scaled dot-product attention (SDPA) call, and a second call reverses the direction and dispatches those values back to the N input positions. The two calls form an encode–decode factorization inducing, for each fixed input, an explicit input-to-input attention matrix of rank at most M that can be implemented entirely with fused SDPA. Latent-workspace models such as Transolver (Wu et al., 2024) instead process a latent sequence with its own self-attention stage. We build on FLARE because it isolates routing as the token-mixing operation itself, which lets us make that routing input-dependent without introducing a separate latent-processing stage.

Every mixer of this kind, including PerceiverIO (Jaegle et al., 2021a), LNO (Wang & Wang, 2024), the Transolver family (Wu et al., 2024; Luo et al., 2025; Zhou et al., 2026), and FLARE, separates into two objects: a compression template, the M-slot structure that determines how information is gathered and redistributed, and the field that template is applied to. The compressed representation depends on the field in all of these methods, trivially so, and that is not what distinguishes them. What distinguishes them is where the template comes from: in FLARE, the learned queries that define it are parameters, whereas other models use a learned pointwise map or a fixed field-dependent rule to construct their templates. In FLARE, training learns M query templates per layer-head once, and those same templates then serve every geometry and every boundary condition: the routing weights respond to the current keys, but the queries defining the template do not. This is the one part of the operator that never sees the field it is compressing, and it is what fixes how the rank-M bottleneck is spent.

![](images/563cb6120649c4a676dd28ad5f0bc5be8c74d5f79085b5362268649699444a77.jpg)  
Figure 1: The FLARE++ mixer. Learned seeds ${ \widetilde { Q } } _ { h }$ synthesize M input-conditioned routing queries $Q _ { h } ( X )$ (left), which gather the N tokens into M latent values and redistribute them (right). The arrow inside each score matrix marks the axis its softmax normalizes: $W _ { \mathrm { e n c } , h }$ over the N tokens, $W _ { \mathrm { d e c } , h }$ over the M routes. Below: the mixer over all H heads with its cost, and the rank-M operator its two routing calls compose to. SDPA is fused, so $S _ { h }$ never reaches memory and space stays (NC).

We present FLARE++, which constructs the compression template from the input tokens instead of fixing it. This dynamic query construction reuses FLARE’s own encode mechanism: learned latent seeds act as the queries of one extra encode call, which gathers the N input tokens into M vectors, and those M vectors are then used as the routing queries of the encode–decode pair that compresses and redistributes the field. Because the queries are produced by the same encode call FLARE already performs, their construction inherits its efficient fused SDPA implementation and needs no new kernel. The change is confined to the mixer, and within the mixer to the construction of the template: the induced routing matrix remains rank at most M for each fixed input, and the complexity remains (NM). The residual stream is untouched, carrying the same number of blocks at the same width with the same residual updates, so nothing reported below is bought by making the network deeper or wider. As Figure 1 shows, FLARE++ replaces a static query parameter with one additional SDPA call and changes nothing else.

We evaluate FLARE++ against FLARE and the Transolver family under a matched backbone on standard PDE surrogate benchmarks, and find that FLARE++ attains the lowest relative $L ^ { 2 }$ error on all five (Table 1), reducing fixed-template FLARE’s error by 24% on average and Transolver-3’s by 31%. Joint ablations against FLARE on latent budget M and residual depth B find that dynamic routing improves on fixed-template FLARE in every configuration we measured (Section 5.2). Additionally, FLARE++ continues improving over the measured latent-budget range, whereas FLARE saturates in that range. Furthermore, dynamic routing substitutes for depth, with FLARE++ reaching a lower error than FLARE at a shallower residual depth.

![](images/562f4e71ade184e79e8f1e8b72c73c6a25c2572442fd679bde5140d76699ce9e.jpg)  
Figure 2: The two-dimensional benchmarks. Top: the input each surrogate receives, namely a point cloud on Elasticity, the permeability field on Darcy, and body-fitted meshes on Airfoil and Pipe. Bottom: the target field. One test case each, drawn to true aspect ratio; Airfoil is cropped to the body, whose mesh extends to the far field.

Outside PDE surrogates, the same substitution improves every Long Range Arena task and lifts the average of FLARE by 2.3 points.

Dynamic routing is not free, costing 1.3–1.5 FLARE’s step time at matched depth and latent budget (Section C.3), but it recovers part of that by reaching a given accuracy with fewer blocks. To alleviate that cost, we provide an exact token-sharded implementation that shards input tokens across devices without ever gathering the full token sequence on one of them, and find that parallel efficiency stays at or near unity in both time and memory. We summarize our contributions below.

• Low-rank self-attention via a synthesized compression template. FLARE++ uses SDPA to construct M routing queries from the input, so the input tokens determine how they are themselves gathered into and redistributed from a compact latent representation. It preserves FLARE’s explicit rank-M encode–decode operator, independent per-head pathways, (NM) complexity, and fused-SDPA implementation, and it adds no depth or width to the residual stream.

• Multi-GPU context parallelism. An exact token-sharded implementation distributes pointwise activations and attention work across accelerators, communicating only latent outputs and softmax statistics, so the collective payload is independent of the number of input tokens and decoding needs no all-gather. Over four ranks, parallel efficiency stays at or near unity in both time and memory.

• Evaluation on accuracy and on cost. We compare dynamic routing against fixed-query FLARE, Transolver, and full self-attention under a matched backbone on standard PDE benchmarks and on Long Range Arena, and measure what each mixer costs in single-GPU time and memory over three orders of magnitude in the number of tokens, and in multi-GPU parallel efficiency. We report both where the mechanism pays and where it does not.

## 2 RELATED WORK

Neural operators on irregular and complex domains. Neural operators learn maps between function spaces and have been developed from regular-grid formulations to models that accept irregular point sets and complex geometries (Li et al., 2020; Lu et al., 2021; Kovachki et al., 2023; Li et al., 2023). Fourier neural operators provide efficient global mixing on regular grids (Li et al., 2020), whereas graph and point-cloud operators extend learned PDE maps to unstructured discretizations (Pfaff et al., 2020; Li et al., 2023). GNOT applies transformer-style operator learning to irregular meshes and multiple input functions (Hao et al., 2023); GINO and GINOT encode geometry before evaluating fields at query points (Li et al., 2023; Liu et al., 2025); and regional graph operators build multiscale communication graphs (Mousavi et al., 2025). Our setting follows this line of work but focuses on the token mixer inside a global operator.

Table 1: Test relative $L _ { 2 }$ error (in %) on standard PDE benchmarks; bold and underline mark best and second-best. Full self-attention sits above the rule and is excluded from the ranking only as reference for unrestricted routing, not a candidate surrogate at these resolutions.  marks where it is prohibitively slow to train under our budget.
<table><tr><td>Model</td><td>Elasticity (1K points)</td><td>Darcy (7K points)</td><td>Airfoil (11K points)</td><td>Pipe (16K points)</td><td>DrivAerML-40K (40K points)</td></tr><tr><td>Full self-attention</td><td>0.41</td><td>0.43</td><td>0.58</td><td>~</td><td>2</td></tr><tr><td>PerceiverIO</td><td>2.80</td><td>2.06</td><td>0.77</td><td>0.69</td><td>24.80</td></tr><tr><td>Set Transformer</td><td>0.52</td><td>1.25</td><td>0.80</td><td>0.39</td><td>6.20</td></tr><tr><td>GNOT</td><td>1.33</td><td>1.69</td><td>10.30</td><td>0.59</td><td>11.50</td></tr><tr><td>LNO</td><td>0.93</td><td>0.76</td><td>1.78</td><td>0.81</td><td>14.60</td></tr><tr><td>Transolver</td><td>0.72</td><td>1.04</td><td>0.60</td><td>0.38</td><td>7.39</td></tr><tr><td>Transolver++</td><td>1.02</td><td>2.76</td><td>1.27</td><td>0.76</td><td>9.39</td></tr><tr><td>Transolver-3</td><td>0.68</td><td>1.03</td><td>0.73</td><td>0.44</td><td>7.36</td></tr><tr><td>FLARE</td><td>0.64</td><td>0.76</td><td>0.57</td><td>0.51</td><td>7.22</td></tr><tr><td>FLARE++ (ours)</td><td>0.38</td><td>0.59</td><td>0.52</td><td>0.34</td><td>6.04</td></tr></table>

Full and efficient attention for PDE surrogates. Full self-attention offers unrestricted global communication and has quadratic time complexity in the number of tokens, while fused implementations can keep peak memory linear by avoiding materialization of the N N score matrix (Vaswani et al., 2017; Dao, 2024). Perceiver established the latent-space processing paradigm used by this line of efficient-attention models: a fixed-size latent array cross-attends to a variable-length input and is then processed using latent self-attention (Jaegle et al., 2021b;a). LNO adapted this latent-workspace paradigm to PDE surrogate modeling by projecting discretized fields into a fixed-length representation, processing them in latent space, and decoding them back to the physical domain (Wang & Wang, 2024). Transolver introduced physics-aware slice tokens and repeats projection, latent self-attention, and unprojection within each block; Transolver++ extends this construction to larger geometries (Wu et al., 2024; Luo et al., 2025). Transolver-3 extends this latent-workspace family to industrial-scale geometries (Zhou et al., 2026). FLARE instead uses latent queries only to define an explicit encode–decode routing matrix of rank at most M for each fixed input, implemented by two SDPA calls without latent self-attention (Puri et al., 2026). FLARE++ preserves this factorization while synthesizing its routing queries from the current input.

What determines the compression template. What separates the mixers of this line of work is not whether the compressed representation depends on the field, which it always does, but where the compression template itself comes from. In one family it is fixed once training ends. Perceiver and Set Transformer attend through learned latent arrays and inducing points that are identical for every input (Jaegle et al., 2021b; Lee et al., 2019), Linformer projects the sequence through learned matrices of fixed shape (Wang et al., 2020), sparse and windowed attention impose a connectivity pattern chosen in advance (Child et al., 2019; Beltagy et al., 2020; Zaheer et al., 2020), and FLARE’s routing queries are parameters (Puri et al., 2026). Transolver sits in this family as well, in a way worth stating precisely: its slice weights are computed from each token’s own features, so the assignment of tokens to slices varies across inputs, but the learned map defining what the slices are is fixed and applied pointwise, with no aggregation over the field (Wu et al., 2024). A second family derives the template from the field, using a fixed rule to do so: Nyströmformer pools queries and keys into landmarks (Xiong et al., 2021), Reformer groups tokens by hashing their content (Kitaev et al., 2020), Sinkhorn attention learns a differentiable sorting of blocks (Tay et al., 2020), and Funnel-Transformer pools the sequence as it deepens (Dai et al., 2020). FLARE++ belongs to this second family and differs from it in construction and in what it preserves: the template is synthesized by attention over the very field it will compress, rather than by pooling, hashing, or sorting, and the mixer remains an explicit rank-M encode–decode operator with no latent self-attention stage, no sequence-projection matrix, and no added residual depth. This isolation is what lets Section 4 attribute a difference to the template alone. A third line leaves the routing structure untouched and replaces the softmax with a kernel feature map (Katharopoulos et al., 2020; Choromanski et al., 2020; Zhang et al., 2024), with Synthesizer at the complementary extreme, generating attention weights without comparing tokens at all (Tay et al., 2021a). These are orthogonal to the question studied here, and we compare against several of them on Long Range Arena (Tay et al., 2021b) in Section C.5.

![](images/6ad0c98399b8f662b7917ec1d5ec91f3922d245e3c1001129f46fe76029d0363.jpg)

![](images/9d5c3817d59a0d05468e4bc570c9e256edf44467c827b42758b1fc314669ff4e.jpg)

<table><tr><td>Blocks B</td><td>FLARE</td><td>FLARE++</td><td> $\Delta$ </td></tr><tr><td>2</td><td>1.76</td><td>0.96</td><td>-45%</td></tr><tr><td>4</td><td>0.93</td><td>0.52</td><td>-44%</td></tr><tr><td>8</td><td>0.68</td><td>0.38</td><td>-44%</td></tr></table>

<table><tr><td>Blocks B</td><td>FLARE</td><td>FLARE++</td><td> $\Delta$ </td></tr><tr><td>2</td><td>1.66</td><td>1.04</td><td>-38%</td></tr><tr><td>4</td><td>1.04</td><td>0.75</td><td>-28%</td></tr><tr><td>8</td><td>0.76</td><td>0.59</td><td>-22%</td></tr></table>

Figure 3: Fixed versus dynamic routing over the joint (M, B) grid; FLARE dashed with open markers, FLARE++ solid with filled. Top: Elasticity against depth (colour per latent budget M) and Darcy against latent budget (colour per depth B). Read horizontally at fixed error, FLARE++ matches FLARE at half the depth; unlike Elasticity, Darcy keeps converting a larger budget into accuracy, the rank-limited behavior Puri et al. (2026) report. Bottom: the same runs against depth, with $\Delta$ the relative error reduction, averaged over M on Elasticity and taken at $M = 1 \bar { 2 } 8$ on Darcy.

Conditioning and generated parameters. Making one part of a network a function of its input is a general technique: feature-wise modulation conditions activations on an auxiliary signal (Perez et al., 2018), and hypernetworks generate weights from a context, including for low-rank PDE models (Cho et al., 2023). Template synthesis in FLARE++ is an instance of this pattern, with the generated object being the M routing queries and the generator being an attention call over the same tokens that are about to be routed.

## 3 METHOD

## 3.1 PRELIMINARIES

We consider PDE surrogate models that operate on fields discretized at $N$ spatial points. Each point carries problem-dependent input quantities, such as coordinates, boundary conditions, material parameters, forcing terms, or an initial state. A pointwise input projection embeds these quantities into a sequence

$$
X = [ x _ { 1 } , \ldots , x _ { N } ] ^ { \top } \in \mathbb { R } ^ { N \times C } ,\tag{1}
$$

where C is the hidden width. Every discretization point thus becomes exactly one token, and we refer to the rows of X as tokens from here on; nothing in the method depends on a token carrying mesh coordinates. A stack of residual token-mixing and feedforward (FFN) blocks communicates information between the N tokens, and a pointwise output projection decodes the requested solution field. The architectural question studied here is how to perform this global token mixing efficiently. We first review full self-attention and the fixed-query low-rank operator used by FLARE, analyzing each attention head independently as in the FLARE formulation (Puri et al., 2026).

Multi-head self-attention. Learned projections construct $Q = X W _ { Q } , K = X W _ { K } , V = X W _ { V }$ with $W _ { Q } , W _ { K } , W _ { V } \in \mathbb { R } ^ { C \times C }$ , split into H heads of width $D = C / H$ . For head h, scaled dot-product attention computes

$$
S _ { h } = \frac { Q _ { h } K _ { h } ^ { \top } } { \sqrt { D } } \in \mathbb { R } ^ { N \times N } , \qquad Y _ { h } = \mathrm { s o f t m a x } ( S _ { h } ) V _ { h } ,\tag{2}
$$

and the head outputs are concatenated and projected, $Y = [ Y _ { 1 } , \dots , Y _ { H } ] W _ { O }$ . Both sides of $S _ { h }$ are functions of the current input, so no fixed latent bottleneck is imposed on the token–token routing matrix; the cost is $\mathcal { O } ( N ^ { 2 } \bar { D ) }$ work per head.

We write SDPA for this primitive viewed as a function of its three arguments,

$$
\mathrm { S D P A } ( Q , K , V ) = \mathrm { s o f t m a x } \left( \frac { Q K ^ { \top } } { \sqrt { D } } \right) V \in \mathbb { R } ^ { N _ { q } \times D } , \qquad Q \in \mathbb { R } ^ { N _ { q } \times D } , \quad K , V \in \mathbb { R } ^ { N _ { k } \times D } ,\tag{3}
$$

where the softmax normalizes over the $N _ { k }$ keys, so a call returns one D-dimensional output per query. The query count and the key count need not agree: self-attention is the case $N _ { q } = N _ { k } \stackrel { \textstyle = } { = } N$ , whereas the operators below take $N _ { q } \neq N _ { k }$ to move information between the N tokens and the M latent routes.

FLARE: fixed-query latent routing. FLARE (Puri et al., 2026) replaces the dense matrix with an encode–decode factorization through $M \ll N$ latent routes. It projects only the keys and values from the current input, while each head owns an independent learned query set:

$$
K _ { h } = X W _ { K , h } \in \mathbb { R } ^ { N \times D } , \qquad V _ { h } = X W _ { V , h } \in \mathbb { R } ^ { N \times D } , \qquad Q _ { h } \in \mathbb { R } ^ { M \times D } { \mathrm { ~ i s ~ l e a r n e d } } .\tag{4}
$$

When channel widths match before head splitting, a residual key parameterization $K = X + X W _ { K }$ initializes the addressing space near the current physical representation; this is an initialization prior rather than additional capacity. The encoder and decoder are standard SDPA calls,

$$
Z _ { h } = \mathrm { S D P A } ( Q _ { h } , K _ { h } , V _ { h } ) , \qquad Y _ { h } = \mathrm { S D P A } ( K _ { h } , Q _ { h } , Z _ { h } ) ,\tag{5}
$$

which, writing $\cdot S _ { h } = Q _ { h } K _ { h } ^ { \top } / \sqrt { D } \in \mathbb { R } ^ { M \times N } , W _ { \mathrm { e n c } , h } = \mathrm { s o f t m a x } ( S _ { h } ) \mathrm { ~ a n d ~ } W _ { \mathrm { d e c } , h } = \mathrm { s o f t m a x } ( S _ { h } ^ { \top } )$ gather the N input values into M latent values and scatter them back:

$$
Y _ { h } = W _ { \mathrm { e f f } , h } V _ { h } , \qquad W _ { \mathrm { e f f } , h } = W _ { \mathrm { d e c } , h } W _ { \mathrm { e n c } , h } \in \mathbb { R } ^ { N \times N } .\tag{6}
$$

Because $W _ { \mathrm { e f f } , h }$ factors through M latent routes its rank is at most M, and it is never materialized: the two SDPA calls apply its factors sequentially at $\mathcal { O } ( N M D )$ cost. The learned queries $Q _ { h }$ that define those routes, however, remain fixed across samples.

## 3.2 FLARE++: DYNAMIC TOKEN ROUTING

The queries $Q _ { h }$ are the compression templates which decide, for a given head, which parts of the input token set each of the M latent slots draws from and returns to. FLARE++ preserves FLARE’s M routing slots but changes where that template comes from. Rather than using a learned set directly as the routing queries, FLARE++ applies FLARE’s own encoder a second time, with learned seeds $\widetilde { Q } _ { h } \in \mathbb { R } ^ { M \times \bar { D } }$ as its queries, and takes its M outputs as the sample-specific routing queries $Q _ { h } ( X )$ . Separate projections of X give

$$
\begin{array} { r } { \widetilde { K } _ { h } = X \widetilde { W } _ { K , h } , \qquad \widetilde { V } _ { h } = X \widetilde { W } _ { V , h } . } \end{array}\tag{7}
$$

One SDPA call synthesizes the routing queries,

$$
Q _ { h } ( X ) = \mathrm { S D P A } ( \widetilde { Q } _ { h } , \widetilde { K } _ { h } , \widetilde { V } _ { h } ) = \mathrm { s o f t m a x } \left( \frac { \widetilde { Q } _ { h } \widetilde { K } _ { h } ^ { \top } } { \sqrt { D } } \right) \widetilde { V } _ { h } \in \mathbb { R } ^ { M \times D } .\tag{8}
$$

This is exactly the FLARE encoder of equation 5, with the learned seeds ${ \widetilde { Q } } _ { h }$ in place of $Q _ { h }$ and its own key and value projections; the only difference is what its output is used for. Instead of being the transported latent values $Z _ { h }$ , the M gathered vectors become the routing queries of the encode–decode pair that follows. Because they are recomputed from the current block input X, the routing-query set changes with the sample and at every layer. The synthesized queries are then used in both factors of the FLARE operator:

$$
K _ { h } = X W _ { K , h } , \quad \quad V _ { h } = X W _ { V , h } ,\tag{9}
$$

$$
Z _ { h } = \mathrm { S D P A } ( Q _ { h } ( X ) , K _ { h } , V _ { h } ) ,\tag{10}
$$

$$
Y _ { h } = \mathrm { S D P A } ( K _ { h } , Q _ { h } ( X ) , Z _ { h } ) .\tag{11}
$$

Writing the resulting routing factors as $W _ { \mathrm { e n c } , h } ( X )$ and $W _ { \mathrm { d e c } , h } ( X )$ gives

$$
Y _ { h } = W _ { \mathrm { d e c } , h } W _ { \mathrm { e n c } , h } V _ { h } , \qquad \mathrm { r a n k } ( W _ { \mathrm { d e c } , h } W _ { \mathrm { e n c } , h } ) \le M .\tag{12}
$$

The synthesized queries $Q _ { h } ( X )$ define both factors of a conditional routing matrix whose rank is at most ${ \mathrm { ~ \dot { M } ~ } } ,$ so the field participates in constructing the template by which it is itself compressed. Query synthesis, gathering, and dispatch all use standard SDPA, with no custom attention kernel or explicit sequence-projection matrix. The complete mixer is three fused SDPA calls, listed in Figure 1.

What dynamic routing does not change. Every modification above is confined to the construction of the routing queries. The residual stream is untouched: a token passes through the same B blocks at the same width $C ,$ , with the same normalization, residual additions, and pointwise input and output projections as in FLARE. The query-synthesis branch of equation 8 sits outside that stream, since its output is consumed as queries and never added back into the token representation, so FLARE++ adds neither residual depth nor a latent-processing stage of the kind a latent-workspace model introduces. The two extra $C \times C$ projections in equation 7 do add parameters, so the mixers are not parametermatched (Section C.2); what is matched is the depth and width of the representation being mixed, and any accuracy difference is therefore attributable to how the M routes are chosen.

Computational complexity. Both mixers are built from the same two primitives: a pointwise projection, costing $\mathcal { O } ( N \bar { C } ^ { 2 } )$ time and $\mathcal { O } ( N C )$ space, and a token–latent SDPA call, costing $\mathcal { O } ( N \bar { M } C )$ time and $\mathcal { O } ( N C )$ space. The latter is linear rather than quadratic in space because a fused kernel never materializes the $N \times M$ score matrix, and the latent tensors are $\bar { \mathcal { O } } ( M C )$ with $M \ll N$ . The mixers differ only in how many of each they use: FLARE performs three projections (K, V , and the output) and two SDPA calls, giving $\mathcal { O } ( N ( 3 C ^ { 2 } + 2 M C ) ) ,$ ) per block, whereas query synthesis adds ${ \tilde { K } } , { \tilde { V } }$ , and one more call, giving $\mathrm { F L A R E + + } \mathcal { O } ( N ( 5 C ^ { 2 } + 3 M C ) )$ ). Both are linear in N in time and $\mathcal { O } ( N C )$ in space, and depth multiplies each by $B ;$ full self-attention instead needs $\mathcal { O } ( N ^ { 2 } C )$ time.

FLARE++ is therefore not a free improvement: at matched $( M , B , C )$ it performs roughly $1 . 6 \times$ the mixer arithmetic of FLARE, so dynamic routing is preferable on cost only if it reaches a given accuracy at a smaller latent budget or depth. These are operation counts and not running times, and the two primitives carry very different constants: a dense projection is a large matrix multiplication near the arithmetic peak of the device, whereas a token–latent SDPA call at $M \ll N$ is a short memory-bound reduction far below it. We therefore treat the counts as a scaling argument and measure the wall-clock consequence in Section C.3.

## 3.3 MULTI-GPU CONTEXT PARALLELISM

Linear complexity does not by itself make a high-resolution mesh fit on one accelerator, because pointwise activations still grow with N. We therefore shard the token dimension across R ranks, so that rank r holds $X _ { r } \in \mathbb { R } ^ { \breve { B } \times N _ { r } \times C }$ with $\textstyle \sum _ { r } N _ { r } = N$ , writing $\boldsymbol { B }$ for the batch size to keep B for the number of blocks. Pointwise projections, normalization, feed-forward layers, and residual updates are then local.

The only globally coupled primitive is the encoder that gathers the sharded input tokens into M replicated latent tokens. Each rank runs a fused SDPA primitive on its local keys and values $K _ { r } , V _ { r }$ exposing a local latent output $O _ { r }$ and its rowwise log-normalizer $L _ { r }$ , and the exact global result is recovered by

$$
L = \mathrm { l o g s u m e x p } _ { r = 0 } ^ { R - 1 } ( L _ { r } ) , \qquad Z = \sum _ { r = 0 } ^ { R - 1 } \exp ( L _ { r } - L ) O _ { r } .\tag{13}
$$

This is algebraically identical to applying the encoder to the concatenated token sequence, but communicates only latent outputs and softmax statistics. Decoding, $Y _ { r } = \mathrm { S D P A } ( \bar { K _ { r } } , Q , Z )$ , is entirely local once the routing queries $Q$ and the latent values Z are replicated, so no all-gather over tokens is ever required.

Each encoder therefore reduces a per-rank payload of $\mathcal { O } ( B H M ( D + 1 ) ) ,$ ) values, independent of N, while token-dependent storage and attention work divide across ranks. FLARE invokes the encoder once per mixer; FLARE++ invokes it twice, once to synthesize $Q ( X )$ and once to gather physical values. Appendix A gives the stable reductions, the two communication schedules, and the backward pass, which must differentiate the globally normalized attention rather than R independently normalized local ones.

Measured scaling. On meshes of $5 \times 1 0 ^ { 5 }$ and $1 0 ^ { 6 }$ points, sharded over up to four ranks, parallel efficiency stays at or near unity on both axes: it never falls below 0.92 in time and 0.95 in memory, where unity means the step time and the per-rank peak memory both divide by the rank count. The collective therefore costs little of the time it saves, and the largest mesh a given machine can train grows almost linearly with the rank count, since no stage of the forward or backward pass reconstructs an N-token tensor. Both effects are insensitive to the latent budget, as the N-independent payload predicts. Section A.7 gives the measurements and conditions.

## 4 EXPERIMENTS

All experiments reported in this paper are conducted on NVIDIA H100 GPUs. Figure 2 shows the two-dimensional benchmarks themselves, the discretization each problem is posed on and the field to be predicted, covering both structured and unstructured meshes.

## 4.1 STANDARD PDE SURROGATE BENCHMARKS

Benchmark problems. We evaluate on the Elasticity, Darcy, Airfoil, Pipe, and DrivAerML-40K benchmarks studied by FLARE (Puri et al., 2026). These problems span structured and unstructured discretizations with approximately 1K–40K points per sample; we refer readers to the FLARE paper for complete dataset definitions. Point counts and splits are reproduced in Table 3.

Baselines. We compare FLARE++ with full self-attention (Vaswani et al., 2017), the Transolver family (Transolver (Wu et al., 2024), Transolver++ (Luo et al., 2025), and Transolver-3 (Zhou et al., 2026)), and FLARE (Puri et al., 2026) under a shared backbone, with matched channel width, head dimension, depth, and latent count wherever applicable. Full self-attention is not a practical PDE surrogate architecture at these resolutions and serves only as a reference for how much a token mixer gives up by imposing a low-rank bottleneck. We therefore separate it from the other entries in every table and exclude it from best-result rankings. We include our Transolver++ runs for completeness, but could not reliably reproduce its reported improvements; related concerns are documented by FLARE, AB-UPT, and NVIDIA PhysicsNeMo (Puri et al., 2026; Alkin et al., 2025; NVIDIA PhysicsNeMo Team, 2026). We also include PerceiverIO (Jaegle et al., 2021a), Set Transformer (Lee et al., 2019), GNOT (Hao et al., 2023), and LNO (Wang & Wang, 2024). These four are evaluated as complete architectures rather than as token mixers in a shared backbone, because an identical-backbone mixer swap is not well defined for them (Section B.2). All models are trained in FP32 precision. Section B.1 gives the complete settings.

Results and discussion. FLARE++ records the lowest error in Table 1 on all five benchmarks. It reduces FLARE’s error by 9–41%, averaging 24%, and Transolver-3’s by 18–44%, averaging 31%. Full self-attention is affordable only on the three smallest benchmarks, where FLARE++ is more accurate on Elasticity and Airfoil, and less accurate on Darcy. The Transolver variants do not separate under this backbone: Transolver-3 is ahead of Transolver on three benchmarks by at most 6% and behind on two by 16–22%, so its published gains do not reproduce at matched width, depth, and latent budget. Model family does not determine the ranking either, since Set Transformer is the strongest non-FLARE model on two benchmarks and leads every Transolver variant on the largest one. The latent-workspace models PerceiverIO and LNO, which process a latent sequence with their own self-attention stage, trail FLARE++ on every benchmark, by up to 4.1 .

## 4.2 LONG RANGE ARENA BENCHMARK

PDE surrogate modeling is the empirical focus of this paper, but the routing mechanism is not specific to it. We therefore also evaluate FLARE and FLARE++ on Long Range Arena (Tay et al., 2021b) under an identical-backbone protocol, against full self-attention, Transolver, and a broad set of established efficient-attention methods. Because FLARE and FLARE++ assume no canonical token order, we compare against efficient-attention architectures rather than fixed-order sequence models such as S4 or Mamba (Gu et al., 2021; Gu & Dao, 2024).

FLARE++ attains the strongest average accuracy in this comparison, raising the FLARE average from 58.08 to 60.36 while preserving an (NM) token mixer. The gain is not carried by one task: dynamic routing improves on fixed-query FLARE on all five, by 0.2 points on Retrieval and by 5.2 and 3.5 points on Image and Pathfinder-32. It also places the low-rank mixer above the full self-attention row (60.36 against 57.51), which fixed-query FLARE does not manage. We note that LRA is a secondary benchmark here, and recent work documents strong locality and positional biases in several of its tasks (Miralles-González et al., 2025). Section C.5 gives the full table, the baseline list, and the configurations used. That the same backbone gains 2.3 points of average accuracy from dynamic routing alone is evidence that the construction is not tuned to PDE discretizations.

## 5 MODEL ANALYSIS AND ABLATIONS

The benchmarks above compare token mixers at a fixed architecture and report accuracy alone. This section supplies the two axes they leave out: what each mixer costs, and how each converts a larger latent budget or depth into accuracy.

## 5.1 EFFICIENCY AND SCALING

Section C.3 reports wall-clock time and peak memory on a single NVIDIA H100 for complete models that differ only in the token mixer, swept over N from $1 0 ^ { 3 }$ to 10<sup>6</sup>. Full self-attention separates from the low-rank mixers in time rather than in memory, and FLARE++ costs a constant factor of 1.3–1.5 FLARE, flat in N. Beyond one device, sharding the token dimension divides activation storage and attention work across ranks while leaving the collective payload independent of N; Table 2 reports the measured efficiency and per-rank memory.

## 5.2 FIXED VERSUS DYNAMIC ROUTING ACROSS THE LATENT BUDGET

We sweep the latent budget M and the depth B jointly for FLARE and FLARE++, holding everything else at the values of Section B.1 for the elasticity and darcy benchmarks. The two mixers differ only in how the M routing queries are obtained, so any difference in the grid is attributable to that choice. Figure 3 plots the sweep and tabulates every cell.

Dynamic routing wins in every cell of the grid. FLARE++ is more accurate than FLARE in all 21 matched (M, B) cells, nine on Elasticity and twelve on Darcy, by between 21% and 53% relative (Figure 3). The two benchmarks differ in how that margin behaves with depth. On Elasticity it is flat, at 45%, 44%, and 44% for B = 2, 4, 8 averaged over the latent budget, whereas on Darcy it decays, from 38% to 28% to 22% at M = 128. Depth substitutes for dynamic routing on the rank-limited benchmark and does not on the low-rank one, which is the first sign that the two mixers use additional capacity differently.

Fixed queries saturate in M; input-conditioned queries do not. Plotting the same runs against the latent budget separates the two mechanisms, and the two benchmarks respond differently because they demand different routing ranks. Puri et al. (2026) report that global communication on Elasticity is fundamentally low-rank, so accuracy there stops improving with M almost immediately, whereas Darcy is rank-limited and keeps benefiting from additional latents over most of the range. On Elasticity, enlarging the latent budget from M = 32 to $M = 1 2 8$ makes FLARE monotonically worse at $B = 2$ and $B = 4 ( 1 . 6 3 \stackrel { - } {  } 1 . 9 1$ and 0.90 0.95) and leaves it unchanged at $B = 8 ,$ , while FLARE++ improves over the identical grid $( 1 . 0 2  0 . 9 0 $ and $0 . 4 0  0 . 3 5 )$ . On Darcy the effect is milder but the same in kind: both mixers convert additional routes into accuracy over most of the range, and both flatten past $M = 1 2 8$ at the largest depth, where doubling the budget changes FLARE by $+ 0 . 5 \%$ and FLARE++ by 1.2%. The difference between the two mechanisms is therefore where saturation sets in and how much has been extracted by then, not whether it happens at all. Enlarging a fixed template adds routes that are largely redundant across inputs, and past some budget they cost more than they contribute, whereas a template built from the current field keeps using the routes it is given.

Dynamic routing substitutes for depth. FLARE++ cannot undercut FLARE by shrinking M, because its two extra projections floor the per-block cost independently of the latent budget (Section C.2). The grid shows the trade running the other way, and without exception: at $B = 4 ,$ , FLARE++ is more accurate than FLARE at $B = 8$ (half the residual depth) at every one of the seven latent budgets measured across the two benchmarks. The same substitution one level down, $B = 2$ against $B = 4 ,$ holds in only two of those seven, so halving the depth is supported at the depths we swept and not below them. Dynamic routing buys back its per-block cost by needing fewer blocks, which is the opposite of the mechanism we had anticipated. The trade is stated in operation counts rather than in measured training time, but the measured per-block penalty of 1.3–1.5 (Section C.3) is smaller than the arithmetic 1.6 , so halving the depth would be expected to reduce wall-clock cost as well as FLOPs; end-to-end training-time savings are not measured here. We conclude that dynamic routing matters most where a small number of latent routes must serve a heterogeneous input, and least where fixed queries already saturate the achievable accuracy.

## 6 CONCLUSION

We introduced FLARE++, an efficient token mixer that synthesizes its routing queries from the current input tokens before gathering and redistributing information. This makes FLARE’s low-rank communication scaffold adaptive to each sample and layer while preserving linear complexity in the number of tokens at a fixed latent budget. Under a matched backbone, dynamic routing gives the lowest error on a set of standard PDE benchmarks and raises the Long Range Arena average of the same architecture. None of this is bought with depth or width: the residual stream carries the same number of blocks at the same channel count as fixed-query FLARE, and only the construction of the compression template differs.

Synthesizing the template is not free, and the measurements say what it costs. A FLARE++ block runs at $1 . 3 – 1 . 5 × \mathrm { F L A R E " s }$ step time and 1.18 its peak memory, flat in the number of tokens and below the 1.6 its operation count predicts, because the added work is dense projection rather than attention. That cost is recovered through depth rather than through a smaller latent budget, since FLARE++ reaches a lower error than FLARE at a shallower residual depth. Both models remain linear in the number of tokens where full self-attention does not, and they separate from it in time rather than in memory: at $5 \times 1 0 ^ { 5 }$ tokens the unrestricted operator is two orders of magnitude slower while using less storage. An exact token-sharded implementation extends both beyond a single accelerator with a collective payload that does not grow with the number of tokens, at parallel efficiency at or near unity in both time and memory over four ranks.

## A MULTI-GPU CONTEXT PARALLELISM

## A.1 TOKEN-SHARDED REPRESENTATION

Let the input to a token-mixing block be

$$
\boldsymbol { X } \in \mathbb { R } ^ { \boldsymbol { B \times N \times C } } ,\tag{14}
$$

where is the batch size, N is the number of tokens, and $C = H D$ is the channel width for H heads of dimension D; the batch dimension is written throughout this appendix because B denotes the number of blocks elsewhere in the paper. The context-parallel group contains R ranks, and rank r owns

$$
X _ { r } \in \mathbb { R } ^ {  { \mathcal { B } } \times N _ { r } \times C } , \qquad X = [ X _ { 0 } ; \ldots ; X _ { R - 1 } ] , \qquad \sum _ { r = 0 } ^ { R - 1 } N _ { r } = N .\tag{15}
$$

The shards used by the token mixer need not form spatially contiguous mesh regions because FLARE does not assign meaning to token ordering. Unequal $N _ { r }$ are allowed as long as local masks exclude padding from the attention normalization.

All operations that act independently on tokens remain local. In particular, rank r computes

$$
K _ { r } = f _ { K } ( X _ { r } ) , \qquad V _ { r } = f _ { V } ( X _ { r } ) , \qquad K _ { r } , V _ { r } \in \mathbb { R } ^ { \mathcal { B } \times H \times N _ { r } \times D } .\tag{16}
$$

The pointwise input and output projections, normalization layers, feed-forward networks, residual connections, and prediction head use the same sharding. The learned latent query or seed tensors have shape $\mathbb { R } ^ { H \times \dot { M } \times D }$ and are sufficiently small to replicate across the context-parallel group.

## A.2 EXACT DISTRIBUTED FLARE ENCODER

Consider a FLARE encoder with a replicated query tensor

$$
Q \in \mathbb { R } ^ { B \times H \times M \times D }\tag{17}
$$

and token-sharded keys and values. This tensor stacks the per-head queries $Q _ { h }$ along the H dimension. On the concatenated token sequence, the desired result is

$$
Z = \mathrm { s o f t m a x } \big ( s Q K ^ { \top } \big ) V , \qquad K = [ K _ { 0 } ; \ldots ; K _ { R - 1 } ] , \qquad V = [ V _ { 0 } ; \ldots ; V _ { R - 1 } ] ,\tag{18}
$$

where $s = D ^ { - 1 / 2 }$ and $Z \in \mathbb { R } ^ { B \times H \times M \times D }$ . The $\mathrm { g l }$ lobal K and V are notation only and are never assembled.

Rank r evaluates the encoder on its local token shard,

$$
S _ { r } = s Q K _ { r } ^ { \top } \in \mathbb { R } ^ { \mathcal { B } \times H \times M \times N _ { r } } ,\tag{19}
$$

$$
L _ { r } = \log \sum _ { j = 1 } ^ { N _ { r } } \exp ( S _ { r , j } ) \in \mathbb { R } ^ { \mathcal { B } \times H \times M } ,\tag{20}
$$

$$
O _ { r } = \operatorname { s o f t m a x } ( S _ { r } ) V _ { r } \in \mathbb { R } ^ { \mathcal { B } \times H \times M \times D } .\tag{21}
$$

Because $O _ { r }$ is normalized only over shard $r ,$ the local outputs cannot be averaged uniformly. The global rowwise log-normalizer and attention mass assigned to rank r are

$$
L = \mathrm { l o g s u m e x p } _ { r = 0 } ^ { R - 1 } ( L _ { r } ) , \qquad \alpha _ { r } = \exp ( L _ { r } - L ) .\tag{22}
$$

The exact global output is therefore

$$
Z = \sum _ { r = 0 } ^ { R - 1 } \alpha _ { r } O _ { r } .\tag{23}
$$

Equations equation 18 and equation 23 are identical because $\alpha _ { r }$ restores the fraction of each globally normalized attention row assigned to shard r.

## A.3 STABLE REDUCTIONS AND FUSED LOCAL ATTENTION

The cross-rank log-sum-exp is evaluated using an elementwise maximum,

$$
L _ { \operatorname* { m a x } } = \operatorname* { m a x } _ { r } L _ { r } ,\tag{24}
$$

$$
a _ { r } = \exp ( L _ { r } - L _ { \operatorname* { m a x } } ) ,\tag{25}
$$

$$
\boldsymbol { a } = \sum _ { r } \boldsymbol { a } _ { r } ,\tag{26}
$$

$$
U = \sum _ { r } a _ { r } O _ { r } ,\tag{27}
$$

$$
L = L _ { \mathrm { m a x } } + \log a , \qquad Z = { \frac { U } { a } } .\tag{28}
$$

The implementation uses an all-reduce maximum for $L _ { \mathrm { m a x } } ,$ an all-reduce sum for $^ { a , }$ and an all-reduce sum for U. All reductions are elementwise over the replicated tensors, and $a _ { r }$ is broadcast over the value dimension D of $O _ { r }$ . For a locally empty attention row, the local primitive must return $L _ { r } = - \infty$ and $O _ { r } = 0$ rather than a NaN. The global merge is valid provided that at least one shard contains a valid key for every attention row.

The score tensor $S _ { r }$ is not materialized. Instead, a fused FlashAttention-backed SDPA primitive (Dao et al., 2022; Dao, 2024) streams the local keys and values through an online softmax and returns both $O _ { r }$ and $L _ { r }$ . Exposing the local log-normalizer is essential, because the public output of an independently normalized local attention call is insufficient to reconstruct the global result.

## A.4 FLARE AND FLARE++ COMMUNICATION SCHEDULES

In FLARE, the fixed queries $Q$ are replicated and one distributed encoder constructs the physical latent values:

$$
Z = \operatorname { E n c } _ { \mathrm { d i s t } } ( Q , \{ K _ { r } \} , \{ V _ { r } \} ) .\tag{29}
$$

The resulting $Z$ is replicated by the reductions in equation 23. Rank r then performs the decode locally,

$$
Y _ { r } = \mathrm { S D P A } ( K _ { r } , Q , Z ) .\tag{30}
$$

No decode communication is required because the query for each output token is local and both latent tensors are replicated.

FLARE++ invokes the same distributed encoder twice. The first invocation constructs the dynamic routing queries,

$$
Q ( X ) = \operatorname { E n c } _ { \mathrm { d i s t } } \left( \widetilde Q , \{ \widetilde K _ { r } \} , \{ \widetilde V _ { r } \} \right) ,\tag{31}
$$

where the learned latent seeds $\widetilde { Q }$ are replicated and the projected input tensors $\widetilde { K } _ { r } , \widetilde { V } _ { r }$ remain sharded. The second invocation gathers the physical values,

$$
Z = \operatorname { E n c } _ { \mathrm { d i s t } } \left( Q ( X ) , \{ K _ { r } \} , \{ V _ { r } \} \right) .\tag{32}
$$

Each rank finally applies the local decode

$$
Y _ { r } = \mathrm { S D P A } ( K _ { r } , Q ( X ) , Z ) .\tag{33}
$$

Thus FLARE uses one globally communicating encoder and one local decoder, whereas FLARE++ uses two globally communicating encoders and one local decoder. The output stays token-sharded and can enter the next block without reconstructing an N-token tensor.

## A.5 BACKWARD PASS

The backward pass must differentiate the globally normalized attention rather than R independently normalized local attentions. Because the replicated latent output $Z$ feeds a local decoder and subsequent local operations on every rank, the first step sums their contributions:

$$
\overline { { \nabla _ { Z } \mathcal { L } } } = \sum _ { r = 0 } ^ { R - 1 } \nabla _ { Z } \mathcal { L } _ { r } .\tag{34}
$$

For the encoder logits on rank r, the globally normalized local probability block is

$$
P _ { r } = \exp ( S _ { r } - L ) ,\tag{35}
$$

where the global L from the forward pass is broadcast along the local key dimension. Let

$$
\boldsymbol { c } = \left. \overline { { \nabla _ { Z } \mathcal { L } } } , Z \right. _ { D } \in \mathbb { R } ^ { B \times H \times M }\tag{36}
$$

denote the inner product over the value dimension for each attention row. The local gradients are

$$
\nabla _ { V _ { r } } \mathcal { L } = P _ { r } ^ { \top } \overline { { \nabla _ { Z } \mathcal { L } } } ,\tag{37}
$$

$$
\nabla _ { S _ { r } } \mathcal { L } = P _ { r } \odot \left( \overline { { \nabla _ { Z } \mathcal { L } } } V _ { r } ^ { \top } - c \right) ,\tag{38}
$$

$$
\nabla _ { K _ { r } } \mathcal { L } = s \nabla _ { S _ { r } } ^ { \top } \mathcal { L } Q ,\tag{39}
$$

$$
\nabla _ { Q } \mathcal { L } = s \sum _ { r = 0 } ^ { R - 1 } \nabla _ { S _ { r } } \mathcal { L } K _ { r } .\tag{40}
$$

The scalar c is broadcast over the $N _ { r }$ local keys. Its subtraction accounts for redistribution of probability mass both within and between shards. Using a conventional local SDPA backward on each rank would omit the between-shard term and would therefore not reproduce global attention.

An implementation may reuse a fused SDPA backward primitive if it accepts the global forward output $Z$ and global log-normalizer L together with the local $Q , K _ { r } , V _ { r }$ . This reconstructs $P _ { r }$ without materializing the score matrix. The query-gradient contributions are summed across the context-parallel group, while $K _ { r }$ and $V _ { r }$ gradients stay local. Gradients of replicated parameters are synchronized across the appropriate data- and context-parallel process groups.

For FLARE++, each rank’s local decoder produces a contribution to the replicated dynamic queries $Q ( X )$ , and these decoder contributions are summed across the context-parallel group. The resulting decoder gradient is added to the globally reduced query gradient from the physical-value encoder. The accumulated gradient is then propagated through the first distributed encoder using the same global backward construction, producing local gradients for ${ \widetilde { K } } _ { \iota }$ <sub>r</sub> and $\widetilde { V } _ { r }$ and a synchronized gradient for the learned seeds $\widetilde { Q } .$ . The saved output and log-normalizer must correspond to the relevant encoder invocation; the query-synthesis and physical-value encoders cannot share these forward statistics.

## A.6 COMPLEXITY AND COMMUNICATION

For balanced shards with $N _ { r } \approx N / R$ , each rank stores approximately $N / R$ token features and performs

$$
\mathcal { O } \bigg ( \frac { N M H D } { R } \bigg )\tag{41}
$$

token-dependent attention work per distributed encoder. Pointwise projections and feed-forward computation are divided in the same manner. One encoder reduces latent outputs of size ( HMD) and normalization statistics of size $\mathcal { O } ( B H M )$ , giving the per-rank collective payload

$$
\mathcal { O } ( B H M ( D + 1 ) ) ,\tag{42}
$$

independent of N. Aggregate network traffic and latency additionally depend on R and the collective implementation. FLARE incurs this encoder communication once per mixer, whereas FLARE++ incurs it twice. Neither model all-gathers X, K, V, or Y, and the decoder adds no context-parallel collective.

## A.7 MEASURED SCALING

Table 2 reports the measurements summarized in Section 3.3, taken on a single node of four NVIDIA H100 GPUs; we did not have access to a larger rank count, so the table does not speak to cross-node interconnects.

Table 2: Context-parallel strong scaling on NVIDIA H100 GPUs, FP16, at $C = 1 2 8 , H = 8 , B = 8$ Both columns are efficiencies normalized within a row against that row’s single-rank measurement, so ideal scaling is 1.00 in each: E is the usual parallel efficiency in step time, and $E _ { \mathrm { m e m } }$ is the per-rank peak-memory reduction divided by the rank count. $M = 6 4$ and $M = 1 2 8$ agree to within 1.2% on efficiency and 0.1% on memory, so only $M = 1 2 8$ is shown. This is not a comparison between FLARE and FLARE++.
<table><tr><td rowspan="2">Model</td><td rowspan="2"> $N$ </td><td colspan="2">R = 1</td><td colspan="2"> $R = 2$ </td><td colspan="2">R = 4</td></tr><tr><td> $E$ </td><td> $E _ { \mathrm { m e m } }$ </td><td> $E$ </td><td> $E _ { \mathrm { m e m } }$ </td><td>E</td><td> $E _ { \mathrm { m e m } }$ </td></tr><tr><td>FLARE</td><td>500K</td><td>Ref.</td><td>Ref.</td><td>0.98</td><td>0.97</td><td>0.94</td><td>0.97</td></tr><tr><td rowspan="2">FLARE++</td><td>1M</td><td>Ref.</td><td>Ref.</td><td>1.01</td><td>0.98</td><td>0.98</td><td>0.97</td></tr><tr><td>500K</td><td>Ref.</td><td>Ref.</td><td>0.97</td><td>0.96</td><td>0.92</td><td>0.95</td></tr><tr><td></td><td>1M</td><td>Ref.</td><td>Ref.</td><td>0.96</td><td>0.97</td><td>0.97</td><td>0.96</td></tr></table>

Efficiency is measured over 30 timed steps per configuration and normalized within each model against its own single-rank step time, which is why the $R = 1$ column reports no efficiency of its own. Two entries exceed unity by a percent, which is within the run-to-run spread rather than evidence of superlinear scaling.

The memory column is the one with practical consequences. Dividing the token dimension divides pointwise activation storage with it, near-ideally at every rank count measured: memory efficiency stays between 0.95 and 0.98 throughout, so a mesh that does not fit on one device fits on R of them, and time efficiency near unity says that this costs almost nothing in throughput. We report that division as a ratio rather than in gigabytes because the claim is about how storage divides, not about how much of it there is, and because absolute peak memory is not comparable across the two measurement harnesses used in this paper: at both mesh sizes the context-parallel harness reports a single-rank FLARE peak 1.89 that of the single-GPU harness of Section C.3, while agreeing with it on FLARE++ to 3% and on step time to 5%. A within-row ratio is unchanged by a multiplicative offset of this kind, and Section C.3 is the sole source of absolute memory in this paper.

## B DATASET AND PROTOCOL DETAILS

For reference, we reproduce the PDE benchmark summary from Table 4 of the FLARE arXiv paper (Puri et al., 2026).

Table 3: PDE benchmark summary reproduced from Table 4 of the FLARE arXiv paper (Puri et al., 2026).
<table><tr><td colspan="4"></td><td rowspan="2">Input/Output features</td><td rowspan="2">Train/Test cases</td></tr><tr><td>Benchmark</td><td>Dimension</td><td>Grid type</td><td>Points</td></tr><tr><td>Elasticity</td><td>2D</td><td>Unstructured</td><td>972</td><td>2/1</td><td>1000 / 200</td></tr><tr><td>Plasticity</td><td>2D+Time</td><td>Structured</td><td>3,131</td><td>3/4</td><td>900 / 80</td></tr><tr><td>Darcy</td><td>2D</td><td>Structured</td><td>7,225</td><td>1/1</td><td>1000 / 200</td></tr><tr><td>Airfoil</td><td>2D</td><td>Structured</td><td>11,271</td><td>2/1</td><td>1000 / 200</td></tr><tr><td>Pipe</td><td>2D</td><td>Structured</td><td>16,641</td><td>2/1</td><td>1000 / 200</td></tr><tr><td>DrivAerML-40K</td><td>3D</td><td>Unstructured</td><td>40,000</td><td>3/1</td><td>387 / 97</td></tr></table>

## B.1 STANDARD PDE BENCHMARK CONFIGURATIONS

Table 4 gives the settings held fixed across all datasets and mixers, and Table 5 gives the per-dataset settings for the models we train under the matched backbone: full self-attention, Transolver, Transolver++, Transolver-3, FLARE, and FLARE++. No hyperparameter is tuned per mixer. PerceiverIO, Set Transformer, GNOT, and LNO are configured as described in the FLARE study (Puri et al., 2026), since they are evaluated as complete architectures rather than as mixers in this backbone. Normalization follows precision rather than model: FP32 runs use LayerNorm and mixed-precision runs use RMSNorm (Zhang & Sennrich, 2019), which is better behaved in half precision (Micikevicius et al., 2018).

## B.2 COMPARISON-MODEL CONSTRUCTION

An identical-backbone token-mixer swap is not well-defined for ISAB or PerceiverIO because neither exposes a standalone mixer with the same feed-forward boundary. ISAB’s encode and decode stages are full cross-attention blocks, each with its own projections, residual path, normalization, and feed-forward network, while PerceiverIO encodes into a latent sequence once, mixes tokens through latent-space self-attention, and decodes only at the output. Extracting only their attention operations would create new hybrid models rather than evaluate ISAB or PerceiverIO, so we compare their complete architectures in Table 1.

For PerceiverIO and LNO, we match the number of transformer blocks to the number of latent-space blocks specified by their respective configurations. For Set Transformer, we replace a Transformer block consisting of attention and feed-forward sublayers with an ISAB block consisting of encode, feed-forward, decode, and feed-forward sublayers. These choices keep the depth of the latent processing comparable.

Table 4: Settings held fixed across all datasets and all matched-backbone mixers. The input and output projections follow the FLARE study (Puri et al., 2026).
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Architecture</td><td></td></tr><tr><td>Channel width C</td><td>128</td></tr><tr><td>Attention heads H</td><td>8</td></tr><tr><td>Head width  $D = C / H$ </td><td>16</td></tr><tr><td>Input/output projection depth</td><td>2 layers, following Puri et al. (2026)</td></tr><tr><td>Output projection normalization Block normalization</td><td>enabled</td></tr><tr><td></td><td>pre-norm LayerNorm</td></tr><tr><td>Block feed-forward network</td><td>GELU MLP</td></tr><tr><td>Feed-forward MLP ratio</td><td>2.0</td></tr><tr><td>Optimization</td><td> $\mathrm { A d a m W } , \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9 , \varepsilon = 1 0 ^ { - 8 }$ </td></tr><tr><td>Optimizer Peak learning rate</td><td></td></tr><tr><td>Schedule</td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>Exponential moving average</td><td>one-cycle enabled</td></tr><tr><td></td><td></td></tr><tr><td>Execution</td><td></td></tr><tr><td>Precision</td><td>FP32</td></tr><tr><td>Hardware</td><td>NVIDIA H100 GPU</td></tr></table>

## C EXTENDED RESULTS

## C.1 MIXED-PRECISION SENSITIVITY

The standard-benchmark results of Section 4.1 are trained in FP32, whereas mixed precision is the usual choice at larger mesh sizes, so it matters whether the two precisions rank the mixers differently. Figure 4 repeats the depth sweep in both precisions for full self-attention, FLARE, and FLARE++ at the latent budget each dataset uses in Table 1, with the two panels of each benchmark on a shared axis.

Precision changes the errors materially but does not reorder the two latent mixers. Most cells sit within 10–30% of their FP32 error under FP16; the two exceptions are both full self-attention, which degrades by 82% on Darcy at B = 2 and improves by 30% on Airfoil at $B = 8 .$ FLARE++ is below FLARE at every depth on Elasticity and Darcy in both precisions, and on Airfoil in FP16 the two coincide at B = 4 (0.756 against 0.753) while FLARE++ leads at $B = 2$ and $B = 8$ . The row that does move is the reference: in FP32, FLARE++ is more accurate than full self-attention on Airfoil at $B = 8 ( 0 . 5 1 5$ against 0.683), and in FP16 the order reverses (0.491 against 0.480). FP16 is therefore not a confound for the comparisons drawn between latent mixers in the main text, but comparisons against the unrestricted reference are precision-dependent even where those are not. This is a milder statement than the mechanism would allow: FLARE++ places an additional softmax on the path producing the routing queries, so a low-precision perturbation there moves the queries defining both rank-M factors rather than only the values being transported, whereas in FLARE the queries are exact parameters and cannot drift this way. On the three benchmarks plotted here that exposure does not materialize. Every number in this comparison is a single seed, and the three benchmarks plotted are those with a full self-attention run in both precisions.

Table 5: Per-dataset settings. The depth B is chosen to suit the size of each benchmark and is shared by every mixer on that dataset, as is the latent budget M. Full self-attention has no latent budget. Batch size is the global batch; all runs here use a single device.
<table><tr><td>Dataset</td><td>Blocks B</td><td>Latents M</td><td>Batch size</td><td>Weight decay</td><td>Epochs</td></tr><tr><td>Elasticity</td><td>8</td><td>64</td><td>2</td><td> $1 0 ^ { - 5 }$ </td><td>500</td></tr><tr><td>Darcy</td><td>8</td><td>128</td><td>2</td><td> $1 0 ^ { - 5 }$ </td><td>500</td></tr><tr><td>Airfoil</td><td>8</td><td>64</td><td>2</td><td> $1 0 ^ { - 5 }$ </td><td>500</td></tr><tr><td>Pipe</td><td>2</td><td>64</td><td>2</td><td> $1 0 ^ { - 5 }$ </td><td>500</td></tr><tr><td>DrivAerML-40K</td><td>4</td><td>64</td><td>1</td><td> $1 0 ^ { - 2 }$ </td><td>500</td></tr></table>

![](images/20362e188d13663eb0dd8a202e0f5583023603ec98470c87d536df7ce3258d10.jpg)  
Figure 4: Test error against depth in FP32 (top) and FP16 (bottom), one column per benchmark and one line per token mixer, at each benchmark’s latent budget from Table 1 $( M = 1 2 8$ on Darcy, $M = 6 4$ otherwise). Panels within a column share a vertical axis, so precision sensitivity appears as a change in the shape or ordering of the curves rather than their position; axes are not shared across columns.

## C.2 COST MODEL

FLARE++ adds two $C \times C$ projections and one token–latent SDPA call per block relative to FLARE, so the two are not parameter- or FLOP-matched by construction. The consequence that matters is a floor: the extra projections contribute $\mathcal { O } ( N C ^ { 2 } )$ work that does not depend on M, so no FLARE++ configuration can undercut FLARE by reducing its latent budget alone, and the route to a better trade-off cannot run through a smaller M. The measurements in Section 5.2 show that it runs through depth instead: because fixed-query routing saturates in M, and on Elasticity degrades with it, the comparison is decided by how many blocks each model needs rather than by per-block cost. A second route remains open and is not yet measured: a shared-projection variant that ties $\widetilde { K } _ { h } , \widetilde { V } _ { h }$ to $K _ { h } , V _ { h }$ would remove the extra projections entirely, at an unknown cost in accuracy.

These are operation counts, and two effects invisible in them push the measured ratio below the arithmetic one: the projections are large matrix multiplications that run near the arithmetic peak of the device whereas a token–latent SDPA call at $M \ll N$ is a short memory-bound reduction, and a complete model also carries normalization and feed-forward layers that are identical across mixers.

## C.3 SINGLE-GPU TIME AND MEMORY

To connect the asymptotic cost model to observed hardware behavior, we measure forward-plusbackward step time and peak memory for complete models that differ only in their token mixer. All models use the same $\mathbf { \bar { \boldsymbol { B } } } = 8 , C = \mathbf { \bar { \boldsymbol { 1 } } } 2 8$ , and $H = 8$ backbone, FP16 fused SDPA, and a single NVIDIA H100 80 GB GPU, while the token count N ranges from $\mathrm { i 0 ^ { 3 } }$ to $1 0 ^ { 6 }$ . This controlled sweep isolates the token-mixing contribution and separates scaling with N from constant-factor overheads that operation counts alone do not predict. The resulting measurements are plotted in Figure 5.

![](images/1ed272537213f878b59516c80888cd6c4cd6f42edc062668ca7f45bd08bb70d4.jpg)  
Figure 5: Forward-plus-backward time and peak memory against input size, for complete models differing only in the token mixer, at $B = 8 , C = 1 2 8 , H = 8 , \mathrm { F P 1 6 } ,$ , on one NVIDIA H100 80 GB GPU. All models use fused SDPA, so every memory curve is linear in N: full self-attention separates on time, not storage, and at $5 \times 1 0 ^ { 5 }$ tokens is two orders of magnitude slower than FLARE++ while using less memory. FLARE++ costs 1.3–1.5 FLARE’s time and 1.18 its memory, flat in N. Transolver-3 matches FLARE on time but needs 1.7–2.5 its memory, and exceeds the device at $1 0 ^ { 6 }$ tokens with 128 slices.

Three readings follow, none of them implied by the operation counts alone.

First, full self-attention separates from every low-rank mixer in time and not in memory: the fused kernel tiles the score matrix and never stores it, so its peak memory sits below FLARE++’s and every model is linear in N. This is why Section 5.1 frames the reference-only argument around compute rather than around an out-of-memory point: there is no such point.

Second, FLARE++ is consistently more expensive than FLARE, and by less than the arithmetic predicts, because the shared feed-forward network dilutes the mixer overhead. The ratio is nearly flat in N beyond $1 0 ^ { 5 }$ , confirming a constant factor rather than a difference in scaling. It is largest at the smallest latent budget, where FLARE++’s cost is set by its extra projections rather than by the extra attention call while FLARE still gets cheaper as M falls.

Third, Transolver-3 matches FLARE on time but not on memory, and the gap widens with the latent budget: at $5 \times 1 0 ^ { 5 }$ tokens it needs 1.65 FLARE’s memory at 64 slices and 2.47 at 128, exceeding the device at $1 0 ^ { 6 }$ tokens in the latter case. FLARE’s peak memory is identical at 64, 128, and 256 latents, because the latent tensors are $\mathcal { O } ( M C )$ and the fused kernel never materializes the $N \times M$ scores, whereas Transolver-3’s grows with the slice count. The same insensitivity holds for FLARE++, at a constant 1.18 offset.

## C.4 QUALITATIVE FIELD PREDICTIONS

Aggregate relative $L ^ { 2 }$ errors do not reveal where a surrogate fails: two models with the same error can distribute it smoothly across the domain or concentrate it in the boundary layers, shocks, and geometric features a designer cares about. Figures 6–9 therefore show, for one benchmark each, the reference solution and both predictions above the two pointwise error maps. Every figure uses the test case at the median error of FLARE++, never the best case, and both models are always shown on the same case. Within a figure the three field panels share one colour scale and the two error panels share another, so a smaller error appears as an emptier panel rather than as a rescaled one. These figures are diagnostic and are not evidence of a ranking.

Where the error sits. On every benchmark the residual error is concentrated on a small part of the domain, and it is the part the discretization was refined for. On Elasticity it lies in a thin band along the void boundary, where the stress concentrations are; on Darcy it follows the interfaces of the piecewise-constant permeability field and appears as filaments rather than as a smooth background; on Airfoil it collects at the suction peak above the leading edge and along the wake line behind the trailing edge; on Pipe it is confined to the near-wall region.

The two models fail in the same places. Across all four benchmarks the FLARE and FLARE++ error maps are structurally alike and differ in amplitude rather than in location. We read this as evidence that input-conditioned routing changes how much of a fixed error structure a rank-M operator removes, not which features it can represent at all; a mechanism that changed the latter would move the error somewhere else, and it does not. Elasticity is where the amplitude difference is most visible, consistent with it being the benchmark with the largest gain in Table 1, and Airfoil is where the two are hardest to tell apart.

Rendering conventions. The panels show the signed pointwise error on a diverging scale centred at zero. Error colour limits are the 99.5th percentile of the larger model’s error, so at most 0.5% of points in any panel fall outside them. Grey marks regions with no data, including the Elasticity void.

![](images/2ebed53279b5642b4d82482132b31e5f31c0da9a60a16bc480ae929499ce0322.jpg)  
Figure 6: Elasticity: the von Mises stress σ on the unstructured unit cell. Top row, the reference solution and the two predictions; bottom row, each model’s pointwise error.

![](images/7b832ce0d816c6ab77f8e5953bd8ee8d7b85adf727d64d0a78bb5e4a61338f6c.jpg)

![](images/2a5699140aa07a81f32951be89180183660e15855638ebc8f5756d11a974239c.jpg)  
Figure 7: Darcy: the pressure u. Top row, the reference solution and the two predictions; bottom row, each model’s pointwise error.  
Figure 8: Airfoil: the Mach number on the body-fitted mesh, cropped to the aerofoil. Top row, the reference solution and the two predictions; bottom row, each model’s pointwise error.

![](images/b7f45392ccefd8074eb7b69e6c8b74946c1d665b3d28ac16d103f3a1e124dc9e.jpg)  
Figure 9: Pipe: the streamwise velocity $u _ { x }$ . Top row, the reference solution and the two predictions; bottom row, each model’s pointwise error.

## C.5 LONG RANGE ARENA

All models use the same Transformer-block backbone, including the same linear input and output projections and feed-forward network; only the token mixer varies. The comparison includes full self-attention (Vaswani et al., 2017); local attention and the original LRA baselines (Tay et al., 2021b); Reformer (Kitaev et al., 2020); Sparse Transformer (Child et al., 2019); Sinkhorn Transformer (Tay et al., 2020); Linformer (Wang et al., 2020); Performer and FAVOR++ (Choromanski et al., 2020); Funnel-Transformer (Dai et al., 2020); Synthesizer (Tay et al., 2021a); linear attention (Katharopoulos et al., 2020); Longformer (Beltagy et al., 2020); BigBird (Zaheer et al., 2020); Norm attention (Qin et al., 2022a); cosFormer (Qin et al., 2022b); Nyströmformer (Xiong et al., 2021); Skyformer (Chen et al., 2021); Hedgehog (Zhang et al., 2024); and Transolver (Wu et al., 2024). FLARE uses the lightweight configuration reported in the FLARE study, with linear key/value projections, a feedforward network with GELU activation, and query/key normalization; FLARE++ uses the matched configuration so that its only substantive change is dynamic token routing.

Table 6: Accuracy (%) on Long Range Arena (LRA) tasks (Tay et al., 2021b). Rows marked with <sup>†</sup> use the reported result from Zhang et al. (2024); the other baseline results and the FLARE result reproduce the reruns reported alongside the FLARE arXiv paper (Puri et al., 2026). The best result in each column is bold and the second best is underlined.
<table><tr><td>Model</td><td>ListOps</td><td>Text</td><td>Retrieval</td><td>Image</td><td>Pathfinder-32</td><td>Avg</td></tr><tr><td>Full self-attention</td><td>36.70</td><td>64.93</td><td>77.18</td><td>38.22</td><td>70.52</td><td>57.51</td></tr><tr><td>Local attention†</td><td>15.82</td><td>52.98</td><td>53.39</td><td>41.46</td><td>66.63</td><td>46.06</td></tr><tr><td>Reformer†</td><td>37.27</td><td>56.10</td><td>53.40</td><td>38.07</td><td>68.50</td><td>50.67</td></tr><tr><td>Sparse Transformer† Sinkhorn Transformer†</td><td>17.07 33.67</td><td>63.58</td><td>59.59</td><td>44.24</td><td>71.71</td><td>51.24</td></tr><tr><td>Linformer (KV sharing)</td><td>36.95</td><td>61.20 51.74</td><td>53.83</td><td>41.23</td><td>67.45</td><td>51.29</td></tr><tr><td></td><td>35.90</td><td></td><td>77.86</td><td>42.82</td><td>50.02</td><td>51.88</td></tr><tr><td>Performer Funnel-Transformer</td><td></td><td>64.21</td><td>68.42</td><td>37.60</td><td>53.83</td><td>51.99</td></tr><tr><td>Synthesizer†</td><td>38.50</td><td>61.17</td><td>61.55</td><td>53.10</td><td>49.98</td><td>52.86</td></tr><tr><td>Linear attention</td><td>36.99 17.95</td><td>61.68</td><td>54.67</td><td>41.61</td><td>69.45</td><td>52.88</td></tr><tr><td>Longformer†</td><td>35.63</td><td>66.00 62.85</td><td>71.84</td><td>34.66</td><td>75.00</td><td>53.09</td></tr><tr><td>Linformer (headwise sharing)</td><td>36.70</td><td>53.00</td><td>56.89 64.72</td><td>42.22 43.42</td><td>69.71 70.09</td><td>53.46</td></tr><tr><td>BigBird†</td><td>36.05</td><td>64.02</td><td>59.29</td><td>40.83</td><td>74.87</td><td>53.59</td></tr><tr><td>Norm attention</td><td>18.30</td><td>63.08</td><td>76.07</td><td>48.22</td><td>70.15</td><td>55.01 55.16</td></tr><tr><td>Performer (FAVOR++)</td><td>36.00</td><td>64.26</td><td>76.74</td><td>35.60</td><td>69.67</td><td>56.45</td></tr><tr><td>cosFormer</td><td>36.20</td><td>64.59</td><td>76.78</td><td>41.52</td><td>75.38</td><td>58.89</td></tr><tr><td>Nyströmformer†</td><td>37.15</td><td>65.52</td><td>79.56</td><td>41.58</td><td>70.94</td><td></td></tr><tr><td>Skyformer†</td><td>39.25</td><td>64.70</td><td></td><td>40.77</td><td></td><td>58.95</td></tr><tr><td>Hedgehog†</td><td></td><td></td><td>82.06</td><td></td><td>70.73</td><td>59.50</td></tr><tr><td>Transolver</td><td>37.15</td><td>64.60</td><td>82.24</td><td>40.15</td><td>74.16</td><td>59.66</td></tr><tr><td></td><td>17.65</td><td>60.95</td><td>76.56</td><td>34.50</td><td>66.94</td><td>51.32</td></tr><tr><td>FLARE (Puri et al., 2026)</td><td>36.85</td><td>65.23</td><td>78.07</td><td>36.86</td><td>73.39</td><td>58.08</td></tr><tr><td>FLARE++ (ours)</td><td>38.05</td><td>66.56</td><td>78.31</td><td>42.04</td><td>76.85</td><td>60.36</td></tr></table>

## REFERENCES

Benedikt Alkin, Maurits Bleeker, Richard Kurle, Tobias Kronlachner, Reinhard Sonnleitner, Matthias Dorfer, and Johannes Brandstetter. Ab-upt: Scaling neural cfd surrogates for high-fidelity automotive aerodynamics simulations via anchored-branched universal physics transformers, 2025. URL https://arxiv.org/abs/2502.09692.

Iz Beltagy, Matthew E. Peters, and Arman Cohan. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150, 2020.

Yifan Chen, Qi Zeng, Heng Ji, and Yun Yang. Skyformer: Remodel self-attention with gaussian kernel and nyström method. Advances in Neural Information Processing Systems Workshop on Efficient Natural Language and Speech Processing, 2021.

Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509, 2019.

Woojin Cho, Kookjin Lee, Donsub Rim, and Noseong Park. Hypernetwork-based meta-learning for low-rank physics-informed neural networks. In Advances in Neural Information Processing Systems, volume 36, pp. 11219–11231, 2023. URL https://proceedings.neurips.cc/paper\_files/paper/2023/file/ 24f8dd1b8f154f1ee0d7a59e368eccf3-Paper-Conference.pdf.

Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, et al. Rethinking attention with performers. arXiv preprint arXiv:2009.14794, 2020.

Zihang Dai, Guokun Lai, Yiming Yang, and Quoc V. Le. Funnel-transformer: Filtering out sequential redundancy for efficient language processing. arXiv preprint arXiv:2006.03236, 2020.

Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. In International Conference on Learning Representations, 2024.

Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memoryefficient exact attention with io-awareness. Advances in Neural Information Processing Systems, 35:16344–16359, 2022.

Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. In First Conference on Language Modeling, 2024.

Albert Gu, Karan Goel, and Christopher Ré. Efficiently modeling long sequences with structured state spaces. arXiv preprint arXiv:2111.00396, 2021.

Zhongkai Hao, Zhengyi Wang, Hang Su, Chengyang Ying, Yinpeng Dong, Songming Liu, Ze Cheng, Jian Song, and Jun Zhu. GNOT: A general neural operator transformer for operator learning. In International Conference on Machine Learning, pp. 12556–12569. PMLR, 2023.

Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch, Catalin Ionescu, David Ding, Skanda Koppula, Daniel Zoran, Andrew Brock, Evan Shelhamer, et al. Perceiver IO: A general architecture for structured inputs & outputs. arXiv preprint arXiv:2107.14795, 2021a.

Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, and Joao Carreira. Perceiver: General perception with iterative attention. In International Conference on Machine Learning, pp. 4651–4664. PMLR, 2021b.

Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In International conference on machine learning, pp. 5156–5165. PMLR, 2020.

Nikita Kitaev, Lukasz Kaiser, and Anselm Levskaya. Reformer: The efficient transformer. arXiv preprint arXiv:2001.04451, 2020.

Nikola Kovachki, Zongyi Li, Burigede Liu, Kamyar Azizzadenesheli, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Neural operator: Learning maps between function spaces with applications to PDEs. Journal ofMachine Learning Research, 24(89):1–97, 2023.

Juho Lee, Yoonho Lee, Jungtaek Kim, Adam Kosiorek, Seungjin Choi, and Yee Whye Teh. Set transformer: A framework for attention-based permutation-invariant neural networks. In Proceedings ofthe 36th International Conference on Machine Learning, pp. 3744–3753, 2019.

Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Fourier neural operator for parametric partial differential equations. arXiv preprint arXiv:2010.08895, 2020.

Zongyi Li, Nikola Kovachki, Chris Choy, Boyi Li, Jean Kossaifi, Shourya Otta, Mohammad Amin Nabian, Maximilian Stadler, Christian Hundt, Kamyar Azizzadenesheli, et al. Geometry-informed neural operator for large-scale 3D PDEs. Advances in Neural Information Processing Systems, 36: 35836–35854, 2023.

Qibang Liu, Weiheng Zhong, Hadi Meidani, Diab Abueidda, Seid Koric, and Philippe Geubelle. Geometry-informed neural operator transformer, 2025. URL https://arxiv.org/abs/ 2504.19452.

Lu Lu, Pengzhan Jin, Guofei Pang, Zhongqiang Zhang, and George Em Karniadakis. Learning nonlinear operators via deeponet based on the universal approximation theorem of operators. Nature Machine Intelligence, 3(3):218–229, 2021.

Huakun Luo, Haixu Wu, Hang Zhou, Lanxiang Xing, Yichen Di, Jianmin Wang, and Mingsheng Long. Transolver++: An accurate neural solver for pdes on million-scale geometries. In Forty-second International Conference on Machine Learning, 2025.

Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, and Hao Wu. Mixed precision training. In International Conference on Learning Representations, 2018. URL https: //arxiv.org/abs/1710.03740.

Pablo Miralles-González, Javier Huertas-Tato, Alejandro Martín, and David Camacho. On the locality bias and results in the long range arena. arXiv preprint arXiv:2501.14850, 2025.

Sepehr Mousavi, Shizheng Wen, Levi Lingsch, Maximilian Herde, Bogdan Raonic, and Siddhartha´ Mishra. RIGNO: A graph-based framework for robust and accurate operator learning for PDEs on arbitrary domains, 2025. URL https://arxiv.org/abs/2501.19205.

NVIDIA PhysicsNeMo Team. Transformer models for external aerodynamics on irregular meshes, 2026. URL https://docs.nvidia.com/physicsnemo/26.03/physicsnemo/ examples/cfd/external\_aerodynamics/transformer\_models/README. html. NVIDIA PhysicsNeMo Framework documentation. Accessed April 30, 2026.

Ethan Perez, Florian Strub, Harm De Vries, Vincent Dumoulin, and Aaron Courville. Film: Visual reasoning with a general conditioning layer. In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018.

Tobias Pfaff, Meire Fortunato, Alvaro Sanchez-Gonzalez, and Peter Battaglia. Learning mesh-based simulation with graph networks. In International Conference on Learning Representations, 2020.

Vedant Puri, Yongjie Jessica Zhang, and Levent Burak Kara. FLARE: Fast low-rank attention routing engine, 2026. URL https://arxiv.org/abs/2508.12594.

Zhen Qin, Xiaodong Han, Weixuan Sun, Dongxu Li, Lingpeng Kong, Nick Barnes, and Yiran Zhong. The devil in linear transformer. arXiv preprint arXiv:2210.10340, 2022a.

Zhen Qin, Weixuan Sun, Hui Deng, Dongxu Li, Yunshen Wei, Baohong Lv, Junjie Yan, Lingpeng Kong, and Yiran Zhong. cosformer: Rethinking softmax in attention. arXiv preprint arXiv:2202.08791, 2022b.

Yi Tay, Dara Bahri, Liu Yang, Donald Metzler, and Da-Cheng Juan. Sparse sinkhorn attention. arXiv preprint arXiv:2002.11296, 2020.

Yi Tay, Dara Bahri, Donald Metzler, Da-Cheng Juan, Zhe Zhao, and Che Zheng. Synthesizer: Rethinking self-attention in transformer models. In International Conference on Machine Learning, pp. 10183–10192. PMLR, 2021a.

Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, Dara Bahri, Philip Pham, Jinfeng Rao, Liu Yang, Sebastian Ruder, and Donald Metzler. Long range arena: A benchmark for efficient transformers. In International Conference on Learning Representations, 2021b.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in Neural Information Processing Systems, 30, 2017.

Sinong Wang, Belinda Z Li, Madian Khabsa, Han Fang, and Hao Ma. Linformer: Self-attention with linear complexity. arXiv preprint arXiv:2006.04768, 2020.

Tian Wang and Chuang Wang. Latent neural operator for solving forward and inverse pde problems. arXiv preprint arXiv:2406.03923, 2024.

Haixu Wu, Huakun Luo, Haowen Wang, Jianmin Wang, and Mingsheng Long. Transolver: A fast transformer solver for pdes on general geometries. arXiv preprint arXiv:2402.02366, 2024.

Yunyang Xiong, Zhanpeng Zeng, Rudrasis Chakraborty, Mingxing Tan, Glenn Fung, Yin Li, and Vikas Singh. Nyströmformer: A nyström-based algorithm for approximating self-attention. In Proceedings ofthe AAAI conference on artificial intelligence, volume 35, pp. 14138–14148, 2021.

Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontañón, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, and Amr Ahmed. Big bird: Transformers for longer sequences. Advances in Neural Information Processing Systems, 33:17283–17297, 2020.

Biao Zhang and Rico Sennrich. Root mean square layer normalization. In Advances in Neural Information Processing Systems, volume 32, 2019. URL https://arxiv.org/abs/1910. 07467.

Michael Zhang, Kush Bhatia, Hermann Kumbong, and Christopher Ré. The hedgehog & the porcupine: Expressive linear attentions with softmax mimicry. arXiv preprint arXiv:2402.04347, 2024.

Hang Zhou, Haixu Wu, Haonan Shangguan, Yuezhou Ma, Huikun Weng, Jianmin Wang, and Mingsheng Long. Transolver-3: Scaling up transformer solvers to industrial-scale geometries. arXiv preprint arXiv:2602.04940, 2026.