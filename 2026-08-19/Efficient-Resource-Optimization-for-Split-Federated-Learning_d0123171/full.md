# Efficient Resource Optimization for Split Federated Learning

Wei Wei, Graduate Student Member, IEEE and Xianhao Chen, Member, IEEE

Abstract—Split federated learning (SFL) has emerged as a powerful paradigm for model training at the edge. However, SFL inherently involves discrete decision variables for model splitting and resource allocation, resulting in a challenging mixed-integer problem. Consequently, prior optimization schemes for SFL are either heuristic or computationally inefficient, which cannot handle large-scale user populations. To address this limitation, this work establishes an efficient optimization framework for SFL under resource-constrained networks. Our framework jointly optimizes model splitting and resource allocation to minimize training cost, which is defined as the weighted sum of latency and energy costs. We first study the model splitting problem and develop a polynomial-time algorithm that achieves the global optimum. Then, we extend the approach to the joint model splitting and resource allocation problem. In this case, we formulate it as a two-dimensional master problem and develop an efficient approximation method with a (1 + ϵ)-approximation guarantee. Extensive experiments show that the proposed approach provides efficient solutions to strike the optimal energy–latency tradeoff.

Index Terms—Mobile edge computing (MEC), split federated learning, model splitting, resource allocation.

## I. INTRODUCTION

Edge learning enables artificial intelligence (AI) computation close to the data source, enhancing privacy, saving bandwidth, and enabling personalization of edge devices [1]–[5]. However, the rapid expansion of AI models presents unprecedented computational challenges. State-of-the-art learning is inherently incompatible with the limited hardware resources of edge devices. For instance, even powerful edge devices such as the NVIDIA Jetson AGX Xavier struggle to train models with more than 50 million parameters [6]. Given these challenges, split federated learning (SFL) [7]–[9], which utilizes a clientserver architecture, presents a promising solution. In SFL, the AI model is divided into two segments: the initial layers are trained on clients over local data, whereas the subsequent layers are trained on a server that provides more powerful computing resources. Moreover, a federated server (or simply Fed server) periodically aggregates client-side sub-models, akin to the federated learning (FL) paradigm [10]–[12], to synchronize models across multiple clients in parallel. SFL alleviates the hardware constraints of clients by offloading the most demanding computations to more powerful servers, thereby recognized as a promising framework to enable largescale model training on edge devices.

![](images/baaf448aa9c9de61b1281b80da9de01a94459858b80c91278d11d80e3a95c417.jpg)

![](images/9f1d4c0c594dda6a9182b8103d14159c005eb66f83495427ca3960f0821bbb33.jpg)  
(a) Illustration of SFL framework.  
(b) Illustration of client-side idle time.  
Fig. 1. Illustration of SFL workflow. (a) SFL scenario includes uploading, downloading and aggregation processes. (b) Since the server starts only after all clients finish local FP and activation uploading, and the next round begins only after all clients complete BP and gradient downloading, moderately reducing the GPU frequency and transmission power of faster clients does not affect the overall training time but can save energy.

As with other edge learning paradigms, resource optimization is essential for SFL, since its performance critically depends on how computation, communication, and modelsplitting decisions are orchestrated under limited network and computing resources [13]–[16]. On the client side, heterogeneous communication and computing capabilities, combined with model-splitting choices, often create a straggler bottleneck where the slowest clients dominate the overall training time (see Fig. 1(b)). Mitigating this requires strategic resource management. First, optimizing power and computing resource allocation for heterogeneous clients can save energy while prioritizing resources for slower clients to accelerate training. Second, the model splitting decision determines how computation and communication loads are distributed between the client and server, thereby affecting both energy consumption and end-to-end latency, as shown in Fig. 2. Thus, realizing efficient SFL at the edge requires the joint optimization of computation, communication, and model-splitting decisions.

In recent years, the SFL literature has addressed resource allocation for improving training efficiency, covering client/server workload coordination [17]–[20], model splitting [13], [14], [21], [22], wireless resource allocation [22]– [26], and aggregation control [13], [14]. While these advances have significantly improved SFL performance, two fundamental challenges in SFL remain unresolved: First, although the model-splitting decision is a core component of SFL, existing studies still lack a scalable polynomial-time solution. It remains an open question whether cut-layer optimization in SFL, due to its combinatorial structure, can admit an exact polynomial-time algorithm. Resolving this problem is crucial for enabling scalable SFL optimization with large user populations (e.g., on the order of hundreds of users or more). Second, beyond model partitioning, the performance of SFL is strongly influenced by GPU operating frequencies and transmission powers, which jointly determine the energy–latency trade-off in heterogeneous edge environments. Nevertheless, the joint optimization of model partitioning and resource allocation substantially increases the complexity of the problem, rendering the overall optimization task more challenging.

![](images/1144e7ba869b300352705439d9550cce2873da2fe42e4aeb3f8a815e2d3f1dee.jpg)

![](images/805d02f0f18beb5d7e13390136366932f27a5ec577df8b483ef32067ecb1d588.jpg)  
Fig. 2. Impact of cut layer on energy and latency. The experiments are conducted with ResNet-50 on the CIFAR-10 dataset under non-IID settings. Other configurations are identical to experimental settings in Section VII.

Motivated by these challenges, this paper aims to answer two research questions:

Q1: Can model-splitting optimization in SFL be solved optimally in polynomial time?

Q2: Can model-splitting decisions, GPU frequency scaling, and transmission power control be jointly optimized within a unified framework?

This paper firmly answers both questions through the following contributions.

• We establish a unified resource optimization framework for SFL by jointly considering model splitting, GPU frequency scaling, and transmission power control.

• For the model-splitting problem, we derive an optimal polynomial-time solution approach.

• For the joint model-splitting and resource optimization problem, we formulate it as a two-dimensional master problem and develop a polynomial-time approximation scheme with a (1 + ϵ) - approximation guarantee.

• Extensive experiments demonstrate that the proposed method consistently outperforms baseline schemes in terms of both latency–energy tradeoff and running-time efficiency.

The remainder of this paper is organized as follows. Section II reviews related work. Section III introduces the system model and the proposed resource optimization framework for SFL. Section IV presents the problem formulation. Section V considers a special case with model-splitting optimization and derives a polynomial-time optimal solution. Section VI investigates the general joint optimization problem, reformulates it as a two-dimensional master problem, and develops an approximation algorithm. Section VII presents the numerical results. Finally, Section VIII concludes this paper.

## II. RELATED WORK

Resource management in distributed learning and edge AI. Recent studies have shown that runtime resource control, especially GPU frequency scaling and power management, can substantially affect the time and performance of deep learning. For example, Drechsler et al. [27] show that lowering GPU frequency can significantly reduce power consumption for CNN inference with only marginal impact on computation time. More recent studies extend this observation to training tasks. Maliakel et al. [28] systematically analyze the impact of dynamic voltage and frequency scaling across different models and tasks, while throttLL’eM [29] exploits GPU frequency scaling to improve training efficiency under latency constraints. Beyond frequency control, several works have explored other forms of runtime resource control for improving the time–resource tradeoff of deep learning systems. Zeus [30] jointly controls batch size and GPU power limits for recurring DNN training jobs to better balance training time and energy consumption. EnvPipe [31] improves training efficiency by leveraging idle pipeline intervals and dynamically adjusting streaming multiprocessor frequencies without sacrificing performance. PCCL [32] focuses on distributed training and identifies more resource-efficient GPU frequency settings for collective communication calls, thereby reducing communication overhead with negligible throughput degradation. Perseus [33] further improves the time–resource tradeoff of large-model training by optimizing computation scheduling along the performance frontier.

These studies demonstrate that resource control is an effective mechanism for improving the efficiency of deep learning workloads in both distributed learning and edge AI systems. However, they are primarily designed for conventional training, inference, or distributed GPU platforms. Thus, they either rely on fixed model partitioning or fail to address partitioning across heterogeneous edge devices.

Resource management in SFL. Recent studies on SFL have increasingly focused on resource allocation for improving training efficiency in heterogeneous edge environments. For example, ESFL [17] optimizes client-side workload distribution and server-side resource allocation over heterogeneous wireless devices, while Wu et al. [23] and HMSFL [21] further incorporate model splitting, communication/computing resource allocation, and hierarchical split designs to improve training efficiency. In addition, AdaptSFL and HSFL [13], [14] show that split decisions and aggregation intervals can be adjusted to better balance communication and computation during training, thereby affecting communication overhead, local computation burden, and convergence speed. More recently, ASFL [22] combines adaptive model splitting with wireless resource allocation, and Wang et al. [24] further consider resource-constrained edge environments with unreliable wireless links. Nevertheless, most existing approaches focus on latency reduction, without considering a unified latencyenergy tradeoff. More importantly, the joint optimization of model splitting and resource allocation remains largely open, where polynomial-time algorithms with provable approximation guarantees are still lacking.

TABLE I SUMMARY OF IMPORTANT NOTATIONS.
<table><tr><td>Notation</td><td>Description</td></tr><tr><td> $\overline { { \mathcal { N } } }$ </td><td>The set of all clients</td></tr><tr><td> $N$ </td><td>The number of clients</td></tr><tr><td> $L$ </td><td>The number of layers in the global model</td></tr><tr><td> $L _ { c , i }$ </td><td>The cut layer of client i</td></tr><tr><td> $n _ { i }$ </td><td>The number of GPU FLOPs per cycle of client i</td></tr><tr><td> $n _ { \mathrm { S } }$ </td><td>The number of GPU FLOPs per cycle of the edge server</td></tr><tr><td> $f _ { i , j }$ </td><td>The clock frequency of client i&#x27;s GPU</td></tr><tr><td> $f _ { \mathrm { S } , j }$ </td><td>The clock frequency of the edge server&#x27;s GPU</td></tr><tr><td> $a _ { i } ( L _ { c , i } )$ </td><td>The activations of client i at the  $L _ { c , i } – \mathrm { t h }$  cut layer</td></tr><tr><td> $\rho _ { l }$ </td><td>The FP computational cost of the l-th DNN layer</td></tr><tr><td> $\varpi _ { l }$ </td><td>The BP computational cost of the l-th DNN layer</td></tr><tr><td> $B _ { i } ^ { \mathrm { U } }$ </td><td>The uplink bandwidth allocated to client i</td></tr><tr><td> $B _ { i } ^ { \mathrm { D } }$ </td><td>The downlink bandwidth allocated to client i</td></tr><tr><td> $P _ { i , j } ^ { \mathrm { T X } }$ </td><td>The transmit power of client i at the  $j \cdot$  th stage</td></tr><tr><td> $P _ { \mathrm { S } , j } ^ { \tilde { \mathrm { T X } } }$ </td><td>The transmit power of the edge server at the j-th stage</td></tr><tr><td> $h _ { i }$ </td><td>The uplink channel fading coefficient</td></tr><tr><td> $h _ { \mathrm { S } }$ </td><td>The downlink channel fading coefficient</td></tr><tr><td> $N _ { 0 }$ </td><td>The noise power spectral density</td></tr><tr><td> $g _ { i } ( L _ { c , i } )$ </td><td>The gradients of client i at the  $\boldsymbol { L _ { c , i } }$  -th cut layer</td></tr><tr><td> $G _ { i }$ </td><td>The coefficient depends on the chip of client ¿</td></tr><tr><td> $G _ { \mathrm { { S } } }$ </td><td>The coefficient depends on the chip of the edge server</td></tr><tr><td> $u _ { i , j }$ </td><td>The control input of client i at the j-th stage</td></tr><tr><td> $\lambda$ </td><td>Balances energy efficiency and training latency</td></tr></table>

## III. SYSTEM MODEL

## A. SFL Architecture

We consider an SFL system with a set of clients $\mathcal { N } =$ $\{ 1 , \ldots , N \}$ , an edge server that executes the server-side submodel, and a federated server that periodically aggregates model parameters. For each client $i ,$ the global DNN model is partitioned into a client-side sub-model, which comprises the initial DNN layers up to the cut layer indexed by $L _ { c , i } ,$ , and a server-side sub-model, which includes the remaining DNN layers from $\boldsymbol { L } _ { c , i } + 1$ to the final output layer. Client i performs local forward pass (FP) up to the cut layer and transmits the corresponding intermediate activations to the edge server, which continues the remaining computation and returns the cut-layer gradients to the client for completing backward pass (BP).

## B. Stages of SFL

Within one SFL round, each client i performs a five-stage pipeline<sup>1</sup> that captures both computation and communication operations. Specifically, the stages include: 1) client-side FP, 2) uplink transmissions of activations, 3) server-side FP and BP, 4) downlink transmissions of gradients, and 5) client-side BP. We denote these stages as $\breve { \mathcal { T } _ { i } ^ { \mathrm { F P } } } , \mathcal { T } _ { i } ^ { \mathrm { U } } , \mathcal { T } _ { i } ^ { \mathrm { F P + B P } } , \mathcal { T } _ { i } ^ { \mathrm { D } }$ , and $\mathcal { I } _ { i } ^ { \mathrm { B P } }$ , respectively.

1) Client-side FP. Client i executes the first $\boldsymbol { L _ { c , i } }$ DNN layers on its local data, producing activations $a _ { i } ( L _ { c , i } )$ at the cut layer. The FP time of client i is modeled as

$$
s _ { i , j } ( f _ { i , j } , L _ { c , i } ) = \sum _ { l = 1 } ^ { L _ { c , i } } \frac { \rho _ { l } } { f _ { i , j } n _ { i } } , \quad j = 1 ,\tag{1}
$$

where $\rho _ { l }$ denotes the FP computational cost (in FLOPs) of the l-th DNN layer, $n _ { i }$ denotes the number of GPU FLOPs per cycle and $f _ { i , j }$ represents the clock frequency of client i’s GPU.

2) Uplink transmissions of activations. Client i transmits the activations $a _ { i } ( L _ { c , i } )$ to the edge server. The data size of $a _ { i } ( L _ { c , i } )$ is denoted by $d _ { i } ^ { \mathrm { U } } ( L _ { c , i } ) = { \tt b i t s } ( a _ { i } ( L _ { c , i } ) )$ , and the corresponding transmission time is given by

$$
s _ { i , j } ( P _ { i , j } ^ { \mathrm { T X } } , L _ { c , i } ) = \frac { d _ { i } ^ { \mathrm { U } } ( L _ { c , i } ) } { B _ { i } ^ { \mathrm { U } } \log \left( 1 + \frac { P _ { i , j } ^ { \mathrm { T X } } | h _ { i } | ^ { 2 } } { B _ { i } ^ { \mathrm { U } } N _ { 0 } } \right) } , \quad j = 2 .\tag{2}
$$

Here, $B _ { i } ^ { \mathrm { U } }$ denotes the uplink bandwidth allocated to client $i , \bar { P } _ { i , j } ^ { \mathrm { T X } }$ is the transmit power of client i at the j-th stage, $h _ { i }$ denotes the uplink channel fading coefficient, $h _ { i } ^ { 2 }$ is the corresponding channel gain, and $N _ { 0 }$ is the noise power spectral density.

3) Server-side FP and BP. Upon receiving $a _ { i } ( L _ { c , i } )$ , the edge server performs the remaining forward and backward passes from layer $L _ { c , i } + 1$ to $L ,$ and outputs the corresponding gradient $g _ { i } ( L _ { c , i } )$ . The data size of $g _ { i } ( L _ { c , i } )$ is denoted by $d _ { i } ^ { \mathrm { D } } ( L _ { c , i } ) = \mathfrak { b i t s } ( g _ { i } ( L _ { c , i } ) )$ . The server-side computation time is

$$
s _ { i , j } ( f _ { \mathrm { S } , j } , L _ { c , i } ) = \sum _ { l = L _ { c , i } + 1 } ^ { L } \frac { \rho _ { l } + \varpi _ { l } } { f _ { \mathrm { S } , j } n _ { \mathrm { S } } } , \quad j = 3 ,\tag{3}
$$

where $\varpi _ { l }$ denotes the BP computational cost (in FLOPs) of the l-th DNN layer, $f _ { \mathrm { S } , j }$ is the clock frequency of the edge server’s GPU, and n<sub>S</sub> denotes the number of GPU FLOPs per cycle of edge server.

4) Downlink transmissions of gradients. The edge server transmits $g _ { i } ( L _ { c , i } )$ back to client i through the downlink channel. The data size of $g _ { i } ( L _ { c , i } )$ is denoted by $d _ { i } ^ { \mathrm { D } } ( L _ { c , i } )$ , and the corresponding transmission time is modeled as

$$
s _ { i , j } ( P _ { \mathrm { S } , j } ^ { \mathrm { T X } } , L _ { c , i } ) = \frac { d _ { i } ^ { \mathrm { D } } ( L _ { c , i } ) } { B _ { i } ^ { \mathrm { D } } \log \left( 1 + \frac { P _ { \mathrm { S } , j } ^ { \mathrm { T X } } | h _ { \mathrm { S } } | ^ { 2 } } { B _ { i } ^ { \mathrm { D } } N _ { 0 } } \right) } , \quad j = 4 ,\tag{4}
$$

Here, $B _ { i } ^ { \mathrm { D } }$ denotes the downlink bandwidth, $P _ { \mathrm { S } , j } ^ { \mathrm { T X } }$ is the transmit power of the edge server at the j-th stage, and $h _ { \mathrm { S } }$ represents the channel fading coefficient.

5) Client-side BP. After receiving the gradients from the edge server, client i performs BP on its local DNN layers, and subsequently updates the parameters of its

client-side sub-model. The service time for this stage is modeled as

$$
s _ { i , j } ( f _ { i , j } , L _ { c , i } ) = \sum _ { l = 1 } ^ { L _ { c , i } } \frac { \varpi _ { l } } { f _ { i , j } n _ { i } } , \quad j = 5 .\tag{5}
$$

## C. Per-round Latency

Based on the defined five-stage SFL, we consider a parallel execution model across clients within each SFL round. Under this model, the per-round training latency of client-side FP and uplink is dominated by the slowest client. Moreover, the edge server maintains a shared server-side sub-model and must process the server-side FP and BP for all clients sequentially. Thus, server-side workloads contribute additively, yielding an aggregate latency term. Similarly, the per-round training latency of client-side BP and downlink is dominated by the slowest client. Accordingly, we define the per-round latency as

$$
\begin{array} { r l r } {  { T _ { \mathrm { r } } ( f _ { i , j } , P _ { i , j } ^ { \mathrm { T X } } , f _ { \mathrm { S } , j } , P _ { \mathrm { S } , j } ^ { \mathrm { T X } } , L _ { c , i } ) = } } \\ & { \underset { i \in \mathcal { N } } { \operatorname* { m a x } } \Big \{ s _ { i , 1 } ( f _ { i , 1 } , L _ { c , i } ) + s _ { i , 2 } ( P _ { i , 2 } ^ { \mathrm { T X } } , L _ { c , i } ) \Big \} + \sum _ { i \in \mathcal { N } } s _ { i , 3 } ( f _ { \mathrm { S } , 3 } , L _ { c , i } ) } \\ & { + \underset { i \in \mathcal { N } } { \operatorname* { m a x } } \Big \{ s _ { i , 4 } ( P _ { \mathrm { S } , 4 } ^ { \mathrm { T X } } , L _ { c , i } ) + s _ { i , 5 } ( f _ { i , 5 } , L _ { c , i } ) \Big \} . } \end{array}
$$

## D. Energy Model

We next quantify the per-round energy consumption by incorporating both computing and communication energy in the five-stage SFL pipeline. For each client $i \in \mathcal { N } .$ , we define the energy consumed at the j-th stage $( j \in \{ 1 , \ldots , 5 \} )$ ) under cut layer $\mathit { L _ { c , i } }$ as

$$
\begin{array} { r l } &  \mathbb { E } _ { \rho , j } ( \rho , j _ { \infty } ^ { \infty } , \rho , \phi _ { j _ { \infty } , j _ { \infty } ^ { \infty } , j _ { \infty } ^ { \infty } , j _ { \infty } ^ { \infty } , \rho , \phi _ { j _ { \infty } , j _ { \infty } ^ { \infty } , j _ { \infty } ^ { \infty } } ) \stackrel { \leq } - } \\ &  ( \begin{array} { l l l } { \underset { j _ { \infty } ^ { \infty } } { \sum _ { i = 1 } ^ { \infty } } } & & \\ { \frac { \sum _ { i = 1 } ^ { \infty } } { \rho _ { i j } \sum _ { j _ { \infty } , j _ { \infty } , j _ { \infty } ^ { \infty } , j _ { \infty } ^ { \infty } } } ] ^ { \frac { j _ { \infty } ^ { \infty } } { \rho _ { i j } } } , } & { - 1 , } & \\  \frac { \log ^ { \frac { j _ { \infty } ^ { \infty } } { \rho _ { i j } } ( \rho , j _ { \infty } ^ { \infty } , I _ { \infty } ) j _ { \infty } ^ { \infty } } } { \rho _ { i j } \rho _ { j } ( j _ { \infty } , j _ { \infty } ^ { \infty } , I _ { \infty } ^ { \infty } ) } \Biggr ) ^ { \frac { j _ { \infty } ^ { \infty } } { \rho _ { i j } } } \binom { 2 ( \rho - \rho _ { i j _ { \infty } , j _ { \infty } ^ { \infty } , I _ { \infty } ^ { \infty } , I _ { \infty } ^ { \infty } , I _ { \infty } ^ { \infty } ) } { I _ { \infty } ^ { \infty } , I _ { \infty } ^ { \infty } , I _ { \infty } ^ { \infty } } - 1 \Biggr ) , \ j = 2 , } \\ &  ( \begin{array} { l }  \frac { \sum _ { i = 1 } ^ { \infty } \binom { j _ { \infty } } { \rho _ { i j } } + ( \rho - \rho _ { i j _ { \infty } , j _ { \infty } ^ { \infty } , I _ { \infty } ^ { \infty } ) } }  \rho _ { i j } \rho _ { j } \sum _ \end{array} \end{array} \end{array}\tag{7}
$$

where the coefficients $G _ { i }$ and $G _ { \mathrm { { S } } }$ [in $\mathrm { W a t t / ( c y c l e / s ) ^ { 3 } ] }$ depend on the chip architecture of client i and the edge server [34]. Accordingly, the total energy consumption of one SFL round is given by

$$
\begin{array} { r } { E _ { \mathrm { r } } ( f _ { i , j } , P _ { i , j } ^ { \mathrm { T X } } , f _ { \mathrm { S } , j } , P _ { \mathrm { S } , j } ^ { \mathrm { T X } } , L _ { c , i } ) = } \end{array}
$$

$$
\sum _ { i \in \mathcal { N } } \sum _ { j = 1 } ^ { 5 } E _ { i , j } ( f _ { i , j } , P _ { i , j } ^ { \mathrm { T X } } , f _ { \mathrm { S } , j } , P _ { \mathrm { S } , j } ^ { \mathrm { T X } } , L _ { c , i } ) .\tag{8}
$$

## E. Control Input of Each SFL Stage

The control input $u _ { i , j } ( t )$ represents the adjustable resource variables that govern the evolution rate $f _ { i , j } \big ( z _ { i , j } ( t ) , u _ { i , j } ( t ) \big )$ and thus determine the stage service time $s _ { i , j } \left( u _ { i , j } ( t ) , L _ { c , i } \right)$ Formally,

$$
u _ { i , j } ( t ) = \left\{ \begin{array} { l l } { f _ { i , j } ( t ) , } & { j \in \{ 1 , 5 \} , } \\ { f _ { \mathrm { S } , j } ( t ) , } & { j = 3 , } \\ { P _ { i , j } ^ { \mathrm { T X } } ( t ) , } & { j = 2 , } \\ { P _ { \mathrm { S } , j } ^ { \mathrm { T X } } ( t ) , } & { j = 4 . } \end{array} \right.\tag{9}
$$

Here, $f _ { i , j } ( t )$ and $f _ { \mathrm { S } , j } ( t )$ denote the instantaneous GPU fre quencies of client i and the edge server, while $P _ { i , j } ^ { \mathrm { T X } } ( t )$ and $P _ { \mathrm { S } , j } ^ { \mathrm { T X } } ( t )$ represent their respective uplink and downlink transmit powers. The cut-layer index $\boldsymbol { L _ { c , i } }$ specifies which subset of DNN layers is executed locally and which subset is offloaded to the edge server, thereby influencing the mapping from control inputs to the service time $s _ { i , j } ( \cdot )$ . Moreover, each client and the edge server are subject to bounded GPU frequencies and transmit powers, $f _ { i } ^ { \mathrm { m i n } } ~ \leq ~ f _ { i , j } ~ \leq \_ f _ { i } ^ { \mathrm { m a x } }$ $f _ { \mathrm { S } } ^ { \mathrm { m i n } } ~ \leq ~ f _ { \mathrm { S } , j } ~ \leq ~ f _ { \mathrm { S } } ^ { \mathrm { m a x } }$ TX,min $\begin{array} { r l r } { \mathrm { ~  ~ \underline { ~ } { ~ \leq ~ } ~ } } & { { } } & { P _ { i , j } ^ { \mathrm { T X } } \mathrm { ~  ~ \leq ~ } P _ { i } ^ { \mathrm { T X , m a x } } } \end{array}$ $\tilde { P _ { \mathrm { S } } } ^ { \mathrm { T X , m i n } } ~ \leq ~ \tilde { P _ { \mathrm { S } , j } } ^ { \mathrm { T X } } ~ \leq ~ \tilde { P _ { \mathrm { S } } } ^ { \mathrm { T X , m a x } }$ , which jointly constrain the feasible processing and transmission rates, thereby implying both lower and upper bounds on the resulting service times.

## IV. PROBLEM FORMULATION

Our goal is to jointly optimize the cut-layer decisions $\{ L _ { c , i } \}$ and the control variables $u _ { i , j } ~ = ~ \{ f _ { i , j } , \mathrm { \ ' { { P } } _ { { i , j } } ^ { T X } , \it { f _ { S , j } , P _ { S , j } ^ { T X } } \} }$ to balance per-round energy consumption and training latency. Thus, we consider the optimal hybrid control problem with energy-latency tradeoff objective:

$$
\begin{array} { r l } { \mathbf { P 1 } : \underset { \{ u _ { i , j } , L _ { c , i } \} } { \operatorname* { m i n } } } & { E _ { \mathrm { r } } ( u _ { i , j } , s _ { i , j } , L _ { c , i } ) } \\ & { \qquad + \lambda T _ { \mathrm { r } } ( u _ { i , j } , s _ { i , j } , L _ { c , i } ) } \end{array}\tag{10a}
$$

$$
\mathrm { s . t . } f _ { i } ^ { \mathrm { m i n } } \leq f _ { i , j } \leq f _ { i } ^ { \mathrm { m a x } } ,\tag{10b}
$$

$$
f _ { \mathrm { S } } ^ { \mathrm { m i n } } \le f _ { \mathrm { S } , j } \le f _ { \mathrm { S } } ^ { \mathrm { m a x } } ,\tag{10c}
$$

$$
P _ { i } ^ { \mathrm { T X , m i n } } \leq P _ { i , j } ^ { \mathrm { T X } } \leq P _ { i } ^ { \mathrm { T X , m a x } } ,\tag{10d}
$$

$$
P _ { \mathrm { S } } ^ { \mathrm { T X , m i n } } \leq P _ { \mathrm { S } , j } ^ { \mathrm { T X } } \leq P _ { \mathrm { S } } ^ { \mathrm { T X , m a x } } .\tag{10e}
$$

where $\lambda > 0$ controls the tradeoff between energy costs and training latency.

## A. Problem Transformation

By lifting $\{ s _ { i , j } \}$ as high-level decision variables, we can rewrite $T _ { \mathrm { r } } \left( z _ { i , j } , u _ { i , j } , s _ { i , j } , L _ { c , i } \right)$ as $\widehat { T } ( \{ s _ { i , j } \} )$ ). Specifically, the per-round latency is given by

$$
\begin{array} { r l } & { \quad T _ { \mathrm { r } } ( u _ { i , j } , s _ { i , j } , L _ { c , i } ) } \\ & { = \underset { i \in \mathcal { N } } { \operatorname* { m a x } } \left\{ s _ { i , 1 } ( f _ { i , j } , L _ { c , i } ) + s _ { i , 2 } ( P _ { i , j } ^ { \mathrm { T X } } , L _ { c , i } ) \right\} + \underset { i \in \mathcal { N } } { \sum } s _ { i , 3 } ( f _ { \mathrm { S } , j } , } \end{array}
$$

$$
\begin{array} { r l } & { \quad L _ { c , i } ) + \displaystyle \operatorname* { m a x } _ { i \in \mathcal { N } } \Big \{ s _ { i , 4 } \big ( P _ { \mathrm { S } , j } ^ { \mathrm { T X } } , L _ { c , i } \big ) + s _ { i , 5 } \big ( f _ { i , j } , L _ { c , i } \big ) \Big \} } \\ & { \triangleq \widehat { T } \big ( \{ s _ { i , j } \big ( u _ { i , j } , L _ { c , i } \big ) \} \big ) , } \end{array}\tag{11}
$$

where each $s _ { i , j }$ is determined by $( u _ { i , j } , L _ { c , i } )$ via the stage models. To handle the max(·) terms, we introduce epigraph variables $T _ { 1 }$ and $T _ { 2 }$ such that

$$
s _ { i , 1 } + s _ { i , 2 } \leq T _ { 1 } , \forall i \in N ,\tag{12}
$$

$$
s _ { i , 4 } + s _ { i , 5 } \leq T _ { 2 } , \forall i \in N .\tag{13}
$$

Then, minimizing $\lambda \widehat { T } ( \{ s _ { i , j } \} )$ is equivalent to minimizing $\begin{array} { r } { \lambda ( T _ { 1 } + T _ { 2 } ) + \lambda \sum _ { i \in \mathcal { N } } s _ { i , 3 } } \end{array}$ subject to Eq. (12) and Eq. (13), because at optimality $T _ { 1 } ~ = ~ \mathrm { m a x } _ { i } ( s _ { i , 1 } + s _ { i , 2 } )$ and $T _ { 2 } ~ =$ max<sub>i</sub> $( s _ { i , 4 } + s _ { i , 5 } )$

Minimizing the energy cost ${ \psi } _ { i , j } ( u _ { i , j } , s _ { i , j } , L _ { c , i } )$ of client i at stage j for a given duration $s _ { i , j }$ constitutes a classical optimal control problem with state specified at a fixed terminal time [35], [36]. Let the optimal control input be denoted as $\{ \mu _ { i , j } ^ { * } ( s _ { i , j } ) , L _ { c , i } ^ { * } \}$ } and the optimal stage cost can be denoted as

$$
\begin{array} { l } { \Psi _ { i , j } \big ( s _ { i , j } ) = \underset { u _ { i , j } ( t ) , L _ { c , i } } { \operatorname* { m i n } } E _ { i , j } \big ( u _ { i , j } , s _ { i , j } , L _ { c , i } \big ) } \\ { = \underset { \ell \in \{ 1 , 2 , \ldots , L _ { \operatorname* { m a x } } \} } { \operatorname* { m i n } } \underset { u _ { i , j } ( t ) } { \operatorname* { m i n } } E _ { i , j } \big ( u _ { i , j } , s _ { i , j } , L _ { c , i } = \ell \big ) } \\ { = \underset { \ell \in \{ 1 , 2 , \ldots , L _ { \operatorname* { m a x } } \} } { \operatorname* { m i n } } \underset { u _ { i , j } ( t ) } { \operatorname* { m i n } } E _ { i , j } ^ { ( \ell ) } \big ( u _ { i , j } , s _ { i , j } \big ) } \\ { = \underset { \ell \in \{ 1 , 2 , \ldots , L _ { \operatorname* { m a x } } \} } { \operatorname* { m i n } } \underset { \bar { \Psi } _ { i , j } ^ { ( \ell ) } } { \operatorname* { m i n } } \big ( s _ { i , j } \big ) } \end{array}\tag{14}
$$

where ℓ denotes each candidate cut layer and $L _ { \mathrm { m a x } } ~ \le ~ L$ denotes the maximum admissible cut layer. Thus, the optimal control problem is denoted by

$$
\begin{array} { r l r } {  { \mathbf { P 2 } : \operatorname* { m i n } _ { \{ s _ { i , j } , y _ { i , \ell } \} , \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { 5 } \sum _ { \ell = 1 } ^ { L _ { \operatorname* { m a x } } } y _ { i , \ell } \widehat { \Psi } _ { i , j } ^ { ( \ell ) } ( s _ { i , j } ) + \lambda ( T _ { 1 } + T _ { 2 } ) } } } \\ & { } & { \ + \lambda \sum _ { i \in \mathcal { N } } s _ { i , 3 } \qquad ( 1 } \end{array}\tag{5a}
$$

$$
\mathrm { s . t . } ~ s _ { i , 1 } + s _ { i , 2 } \leq T _ { 1 } , \quad \forall i \in \mathcal { N } ,\tag{15b}
$$

$$
s _ { i , 4 } + s _ { i , 5 } \leq T _ { 2 } , \quad \forall i \in N ,\tag{15c}
$$

$$
\sum _ { \ell = 1 } ^ { L _ { \operatorname* { m a x } } } S _ { i , j } ^ { ( \ell ) , \operatorname* { m i n } } y _ { i , \ell } \leq s _ { i , j } \leq \sum _ { \ell = 1 } ^ { L _ { \operatorname* { m a x } } } S _ { i , j } ^ { ( \ell ) , \operatorname* { m a x } } y _ { i , \ell } , \ \forall i , j ,\tag{15d}
$$

$$
\sum _ { \ell = 1 } ^ { L _ { \mathrm { m a x } } } y _ { i , \ell } = 1 , \quad y _ { i , \ell } \in \{ 0 , 1 \} , \quad \forall i .\tag{15e}
$$

where $y _ { i , \ell }$ is a binary cut-layer selection indicator for client i, with $y _ { i , \ell } = 1$ indicating that client i chooses cut layer ℓ, and $y _ { i , \ell } ~ = ~ 0$ otherwise. The service-time bounds in (15d) couple this discrete choice with the stage durations $\{ s _ { i , j } \}$ by restricting $s _ { i , j }$ to the range corresponding to the selected cut layer, and the constraint in (15e) enforces that each client selects exactly one cut layer. Moreover, $S _ { i , j } ^ { ( \ell ) , \mathrm { m i n } }$ and $S _ { i , j } ^ { ( \ell ) }$ ,max are denoted by

$$
\left\{ \begin{array} { l l } { \sum _ { l = 1 } ^ { \ell } \rho _ { l } / \big ( n _ { i } f _ { i } ^ { \mathrm { m a x } } ) , ~ j = 1 , ~ i \in \mathcal { N } , } \\ { \begin{array} { l } { d _ { i } ^ { \mathrm { U } } ( \ell ) / \bigg ( B ^ { \mathrm { U } } \log _ { 2 } \\\\\\\\\big ( 1 + \frac { P _ { i } ^ { \mathrm { T X } , \mathrm { m a x } } h _ { i } ^ { 2 } } { N _ { 0 } } \big ) \bigg ) , ~ j = 2 , ~ i \in \mathcal { N } , } \\ { \sum _ { l = \ell + 1 } ^ { L } \big ( \rho _ { l } + \varpi _ { l } \big ) / \big ( n _ { \mathrm { S } } f _ { \mathrm { S } } ^ { \mathrm { m a x } } \big ) , ~ j = 3 , ~ i \in \mathcal { N } . } \end{array} } \\ { d _ { i } ^ { \mathrm { D } } ( \ell ) / \bigg ( B ^ { \mathrm { D } } \log _ { 2 } \\\\\\\big ( 1 + \frac { P _ { \mathrm { S } } ^ { \mathrm { T X } , \mathrm { m a x } } h _ { \mathrm { S } } ^ { 2 } } { N _ { 0 } } \big ) \bigg ) , ~ j = 4 , ~ i \in \mathcal { N } , } \\ { \sum _ { l = 1 } ^ { \ell } \varpi _ { l } / \big ( n _ { i } f _ { i } ^ { \mathrm { m a x } } \big ) , ~ j = 5 , ~ i \in \mathcal { N } , } \end{array} \right.\tag{16}
$$

$$
\begin{array} { r l } & { S _ { i , j } ^ { ( \ell ) , \mathrm { m a x } } = } \\ & { \left\{ \begin{array} { l l } { \sum _ { l = 1 } ^ { \ell } { \rho _ { l } / ( n _ { i } f _ { i } ^ { \mathrm { m i n } } ) } , \ j = 1 , \ i \in \mathcal { N } , } \\ { \begin{array} { l } { \mathcal { A } _ { i } ^ { \mathrm { U } } ( \ell ) / \left( B ^ { \mathrm { U } } \log _ { 2 } \left( 1 + \frac { P _ { i } ^ { \mathrm { T X } , \mathrm { m i n } } h _ { i } ^ { 2 } } { N _ { 0 } } \right) \right) , \ j = 2 , \ i \in \mathcal { N } , } \\ { \sum _ { l = \ell + 1 } ^ { L } ( \rho _ { l } + \varpi _ { l } ) / ( n _ { \mathrm { S } } f _ { \mathrm { S } } ^ { \mathrm { m i n } } ) , \ j = 3 , \ i \in \mathcal { N } . } \end{array} } \\ & { \begin{array} { r } { d _ { i } ^ { \mathrm { D } } ( \ell ) / \left( B ^ { \mathrm { D } } \log _ { 2 } \left( 1 + \frac { P _ { \mathrm { S } } ^ { \mathrm { T X } , \mathrm { m i n } } h _ { i } ^ { 2 } } { N _ { 0 } } \right) \right) , \ j = 4 , \ i \in \mathcal { N } , } \\ { \sum _ { l = 1 } ^ { \ell } \varpi _ { l } / ( n _ { i } f _ { i } ^ { \mathrm { m i n } } ) , \ j = 5 , \ i \in \mathcal { N } , } \end{array} } \end{array} \right. } \end{array}\tag{17}
$$

We note that this paper does not explicitly optimize test accuracy. In our prior work [13]–[15], we showed that shallower split points generally lead to faster convergence, whereas overly deep split points may degrade learning performance. Motivated by this observation, we restrict the cut-layer design space to

$$
\mathcal { L } _ { \operatorname* { m a x } } \triangleq \{ 1 , 2 , \dots , L _ { \operatorname* { m a x } } \} .
$$

Accordingly, the subsequent optimization focuses on the system-level energy–latency tradeoff over the restricted cutlayer set.

To analyze the energy-latency tradeoff, we consider the Pareto efficiency associated with Problem P2. For any feasible decision vector $\xi ,$ define

$$
E ( \xi ) \triangleq \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { 5 } \Psi _ { i , j } ( s _ { i , j } ) ,\tag{18}
$$

and

$$
T ( \xi ) \triangleq \left( T _ { 1 } + T _ { 2 } \right) + \sum _ { i \in \mathcal { N } } s _ { i , 3 } ,\tag{19}
$$

where $\{ s _ { i , j } \} , T _ { 1 }$ , and $T _ { 2 }$ are components of $\xi .$

A feasible solution $\xi ^ { * }$ is Pareto efficient with respect to $( E , T )$ if there exists no feasible $\hat { \xi }$ such that $E ( \hat { \xi } ) \leq E ( \xi ^ { * } )$ and ${ \tilde { T } } ( \hat { \xi } ) \leq T ( \xi ^ { * } )$ , with at least one strict inequality. Consider the weighted-sum scalarization

$$
\operatorname* { m i n } _ { \xi \in \mathcal { X } } F _ { \lambda } ( \xi ) \triangleq E ( \xi ) + \lambda T ( \xi ) , \qquad \lambda > 0 ,\tag{20}
$$

where X is the feasible set of P2. For any fixed $\lambda > 0$ , any global minimizer of (20) is Pareto efficient. However, varying λ does not necessarily recover all Pareto-efficient solutions, because the attainable objective set $\{ ( E ( \xi ) , T ( \xi ) ) : \xi \in \mathcal { X } \}$ is generally non-convex in our setting due to the binary variables $y _ { i , \ell } \in \{ 0 , 1 \}$ and the generally non-convex stage-energy term $\begin{array} { r } { \Psi _ { i , j } ( s _ { i , j } ) = \operatorname* { m i n } _ { \ell } \widehat { \Psi } _ { i , j } ^ { ( \overline { { \ell } } ) } ( s _ { i , j } ) } \end{array}$ . Hence, (20) may recover only supported Pareto-efficient points.

## V. SPECIAL CASE: MODEL-SPLITTING OPTIMIZATION

In this section, we consider model-splitting optimization for SFL, which is an important special case of P1. While the solution space of the integer problem is exponential, we introduce auxiliary variables to divide the joint problem into client subproblems to solve it exactly in polynomial time.

## A. Model-splitting Problem Formulation

We first study a model-splitting optimization of the original energy-latency problem in Eq. (10). Specifically, we fix the per-stage resource controls at prescribed feasible constants, $\mathrm { ~ \bar { e } . g . , ~ } \check { P } _ { i , j } ^ { \mathrm { T X } } = \hat { P } _ { i , j } ^ { \mathrm { T X } } , \ : P _ { \mathrm { S } , j } ^ { \mathrm { T X } } = \hat { P } _ { \mathrm { S } , j } ^ { \mathrm { \bar { T } X } } , \ : f _ { i , j } = \hat { f } _ { i , j } , \ : f _ { \mathrm { S } , j } = \hat { f } _ { \mathrm { S } , j } ,$ where $\tilde { \hat { P } } _ { i , j } ^ { \mathrm { T X } } , \ \hat { P } _ { \mathrm { S } , j } ^ { \mathrm { T X } } , \ \hat { f } _ { i , j } ,$ and $\hat { f } _ { \mathrm { S } , j }$ denote prescribed feasible constants for the client transmit power, server transmit power, client GPU frequency, and server GPU frequency, respectively. In this case, for each candidate cut layer $\ell \in \{ 1 , \dots , L _ { \mathrm { m a x } } \}$ the corresponding stage time and energy become deterministic constants, denoted by

$$
\begin{array} { r } { S _ { i , j } ^ { [ \ell ] } = s _ { i , j } ( \hat { f } _ { i , j } , \hat { P } _ { i , j } ^ { \mathrm { T X } } , \hat { f } _ { \mathrm { S } , j } , \hat { P } _ { \mathrm { S } , j } ^ { \mathrm { T X } } , \ell ) , } \end{array}\tag{21}
$$

$$
\begin{array} { r } { E _ { i , j } ^ { [ \ell ] } = E _ { i , j } ( \hat { f } _ { i , j } , \hat { P } _ { i , j } ^ { \mathrm { T X } } , \hat { f } _ { \mathrm { S } , j } , \hat { P } _ { \mathrm { S } , j } ^ { \mathrm { T X } } , \ell ) . } \end{array}\tag{22}
$$

Furthermore, we introduce binary decision variables $y _ { i , \ell } \in$ {0, 1} indicating whether client i selects cut layer $\ell ,$ i.e.,

$$
\sum _ { \ell = 1 } ^ { L _ { \operatorname* { m a x } } } y _ { i , \ell } = 1 , \ y _ { i , \ell } \in \{ 0 , 1 \} , \quad \forall i \in \mathcal { N } .\tag{23}
$$

Then the stage time of client i can be rewritten as affine selections:

$$
s _ { i , j } = \sum _ { \ell = 1 } ^ { L _ { \operatorname* { m a x } } } S _ { i , j } ^ { [ \ell ] } y _ { i , \ell } , \quad \forall i , j ,\tag{24}
$$

Substituting (24) into (6)–(10), we obtain

$$
\begin{array} { r l r } & { } & { \mathbf { P 3 } : \displaystyle \operatorname* { m i n } _ { \{ y _ { i } , \ell \} } \sum _ { i \in { \mathcal { N } } } \sum _ { \ell = 1 } ^ { L _ { \operatorname* { m a x } } } E _ { i } ^ { [ \ell ] } y _ { i , \ell } + \lambda \Big ( \displaystyle \operatorname* { m a x } _ { i \in { \mathcal { N } } } \sum _ { \ell = 1 } ^ { L _ { \operatorname* { m a x } } } \big ( S _ { i , 1 } ^ { [ \ell ] } + S _ { i , 2 } ^ { [ \ell ] } \big ) y _ { i , \ell } } \\ & { } & { \quad \quad \quad + \displaystyle \sum _ { i \in { \mathcal { N } } } \sum _ { \ell = 1 } ^ { L _ { \operatorname* { m a x } } } S _ { i , 3 } ^ { [ \ell ] } y _ { i , \ell } + \displaystyle \operatorname* { m a x } _ { i \in { \mathcal { N } } } \sum _ { \ell = 1 } ^ { L _ { \operatorname* { m a x } } } \big ( S _ { i , 4 } ^ { [ \ell ] } + S _ { i , 5 } ^ { [ \ell ] } \big ) y _ { i , \ell } \Big ) } \end{array}\tag{25a}
$$

$$
\mathrm { s . t . } \sum _ { \ell = 1 } ^ { L _ { \mathrm { m a x } } } y _ { i , \ell } = 1 , \quad y _ { i , \ell } \in \{ 0 , 1 \} , \quad \forall i \in \mathcal { N } .\tag{25b}
$$

where $\begin{array} { r } { E _ { i } ^ { [ \ell ] } \triangleq \sum _ { j = 1 } ^ { 5 } E _ { i , j } ^ { [ \ell ] } \left( \forall i , \ell \right) } \end{array}$ . Problem P3 in (25a) is a 0−1 discrete optimization that searches for the best cut layer across clients with exponentially cut-layer configurations in general. Despite its complexity, we will develop a polynomial-time algorithm to solve it exactly in the subsequent development.

## B. Solution Approach

We introduce two variables $T _ { 1 }$ and $T _ { 2 }$ satisfying:

$$
\sum _ { \ell = 1 } ^ { L _ { \mathrm { m a x } } } \big ( S _ { i , 1 } ^ { [ \ell ] } + S _ { i , 2 } ^ { [ \ell ] } \big ) y _ { i , \ell } \leq T _ { 1 } , \quad \forall i \in \mathcal { N } ,\tag{26}
$$

$$
\sum _ { \ell = 1 } ^ { L _ { \mathrm { m a x } } } \big ( S _ { i , 4 } ^ { [ \ell ] } + S _ { i , 5 } ^ { [ \ell ] } \big ) y _ { i , \ell } \leq T _ { 2 } , \quad \forall i \in \mathcal { N } .\tag{27}
$$

Then, Problem P3 can be equivalently transformed into

$$
\mathbf { P 3 } ^ { \prime } : \operatorname* { m i n } _ { \{ y _ { i } , \ell \} , T _ { 1 } , T _ { 2 } } \sum _ { i \in { \cal N } } \sum _ { \ell = 1 } ^ { L _ { \operatorname* { m a x } } } \Big ( E _ { i } ^ { [ \ell ] } + \lambda S _ { i , 3 } ^ { [ \ell ] } \Big ) y _ { i , \ell } + \lambda ( T _ { 1 } + T _ { 2 } )\tag{28a}
$$

$$
\mathrm { s . t . } \quad \sum _ { \ell = 1 } ^ { L _ { \operatorname* { m a x } } } y _ { i , \ell } = 1 , \quad y _ { i , \ell } \in \{ 0 , 1 \} , \quad \forall i \in \mathcal { N } ,\tag{28b}
$$

$$
\sum _ { \ell = 1 } ^ { L _ { \mathrm { m a x } } } \big ( S _ { i , 1 } ^ { [ \ell ] } + S _ { i , 2 } ^ { [ \ell ] } \big ) y _ { i , \ell } \leq T _ { 1 } , \quad \forall i \in \mathcal { N } ,\tag{28c}
$$

$$
\sum _ { \ell = 1 } ^ { L _ { \mathrm { m a x } } } \big ( S _ { i , 4 } ^ { [ \ell ] } + S _ { i , 5 } ^ { [ \ell ] } \big ) y _ { i , \ell } \leq T _ { 2 } , \quad \forall i \in \mathcal { N } .\tag{28d}
$$

To obtain the optimal solution, we enumerate the candidate thresholds according to

$$
T _ { 1 } \in \Bigl \{ S _ { i , 1 } ^ { [ \ell ] } + S _ { i , 2 } ^ { [ \ell ] } \Bigr \} ,\tag{29}
$$

$$
T _ { 2 } \in \Big \{ S _ { i , 4 } ^ { [ \ell ] } + S _ { i , 5 } ^ { [ \ell ] } \Big \} ,\tag{30}
$$

which are finite sets of cardinality at most $N ( L - 1 )$ . For any fixed $( T _ { 1 } , T _ { 2 } )$ , client i selects the cut layer with minimum cost:

$$
\ell _ { i } ^ { * } ( T _ { 1 } , T _ { 2 } ) = \arg \operatorname* { m i n } _ { \ell \in \mathcal { F } _ { i } ( T _ { 1 } , T _ { 2 } ) } \Big ( E _ { i } ^ { [ \ell ] } + \lambda S _ { i , 3 } ^ { [ \ell ] } \Big ) ,\tag{31}
$$

where the feasible cut-layer set $\mathcal { F } _ { i } ( T _ { 1 } , T _ { 2 } )$ is defined as

$$
\begin{array} { r } { \mathcal { F } _ { i } ( T _ { 1 } , T _ { 2 } ) \triangleq \Big \{ \ell : \ S _ { i , 1 } ^ { [ \ell ] } + S _ { i , 2 } ^ { [ \ell ] } \leq T _ { 1 } , \ S _ { i , 4 } ^ { [ \ell ] } + S _ { i , 5 } ^ { [ \ell ] } \leq T _ { 2 } \Big \} . } \end{array}\tag{32}
$$

The resulting objective value is

$$
J ( T _ { 1 } , T _ { 2 } ) = \underbrace { \sum _ { i \in \mathcal { N } } \underset { \ell \in \mathcal { F } _ { i } ( T _ { 1 } , T _ { 2 } ) } { \operatorname* { m i n } } \left( E _ { i } ^ { [ \ell ] } + \lambda S _ { i , 3 } ^ { [ \ell ] } \right) } _ { G ( T _ { 1 } , T _ { 2 } ) \colon \mathrm { N o n - i n c r e a s i n g } } + \underbrace { \lambda ( T _ { 1 } + T _ { 2 } ) } _ { H ( T _ { 1 } , T _ { 2 } ) \colon \mathrm { S t r i c t l y } } .\tag{33}
$$

Finally, the globally optimal thresholds are obtained by

$$
( T _ { 1 } ^ { * } , T _ { 2 } ^ { * } ) = \arg \operatorname* { m i n } _ { T _ { 1 } , T _ { 2 } } J ( T _ { 1 } , T _ { 2 } ) ,\tag{34}
$$

which yields a globally optimal cut-layer decision $\{ y _ { i , \ell } ^ { * } \}$ for (25a).

To solve Problem P3, Algorithm 1 first constructs the candidate threshold sets

$$
\begin{array} { r l } & { \mathcal { T } _ { 1 } \triangleq \Big \{ S _ { i , 1 } ^ { [ \ell ] } + S _ { i , 2 } ^ { [ \ell ] } \ \big | \ i \in \mathcal { N } , \ \ell \in \{ 1 , \dots , L _ { \operatorname* { m a x } } \} \ \Big \} , } \\ & { \mathcal { T } _ { 2 } \triangleq \Big \{ S _ { i , 4 } ^ { [ \ell ] } + S _ { i , 5 } ^ { [ \ell ] } \ \big | \ i \in \mathcal { N } , \ \ell \in \{ 1 , \dots , L _ { \operatorname* { m a x } } \} \ \Big \} . } \end{array}
$$

The two sets are sorted in ascending order and enumerated as candidate threshold values. For each fixed $T _ { 2 } ~ \in ~ \mathcal { T } _ { 2 }$ , the algorithm sequentially sweeps $T _ { 1 } ~ \in ~ \mathcal { T } _ { 1 }$ , updates the best feasible cut-layer choice for each client, and applies a boundbased pruning rule to terminate the $T _ { 1 }$ scan early whenever no better objective value can be attained. By enumerating all candidate threshold pairs together with this pruning rule, Algorithm 1 solves Problem P3 to global optimality.

Algorithm 1: Optimal Model-splitting Optimization   
via Plane Sweep and Pruning   
Input: $\{ S _ { i , j } ^ { [ \ell ] } \} , \{ E _ { i } ^ { [ \ell ] } \} , \lambda . V _ { i } ^ { [ \ell ] } \triangleq E _ { i } ^ { [ \ell ] } + \lambda S _ { i , 3 } ^ { [ \ell ] } .$   
Output: $J ^ { * } , \{ \ell _ { i } ^ { * } \}$   
$\mathcal { T } _ { 1 }  \mathrm { s o r t } ( \{ S _ { i , 1 } ^ { [ \ell ] } + S _ { i , 2 } ^ { [ \ell ] } \} ) , \mathcal { T } _ { 2 }  \mathrm { s o r t } ( \{ S _ { i , 4 } ^ { [ \ell ] } + S _ { i , 5 } ^ { [ \ell ] } \} ) ;$   
$\begin{array} { r } { G _ { \operatorname* { m i n } } \gets \sum _ { i \in \mathcal { N } } \operatorname* { m i n } _ { \ell \in \{ 1 , \dots , L _ { \operatorname* { m a x } } \} } ( E _ { i } ^ { [ \ell ] } + \lambda S _ { i , 3 } ^ { [ \ell ] } ) ; } \end{array}$   
$\mathcal { B } [ t ]  \{ ( i , \ell ) \mid S _ { i , 1 } ^ { [ \ell ] } + S _ { i , 2 } ^ { [ \ell ] } = t \} ; J ^ { \ast }  \infty ;$   
foreach $T _ { 2 } \in \mathcal { T } _ { 2 }$ do   
// outer loop: fix $T _ { 2 }$   
$c _ { i }  \infty , \ \ell _ { i }  - 1 \ ( \forall i ) ; G  0 ;$ count $ 0 ;$   
for k ← 1 to $\lvert \mathcal { T } _ { 1 } \rvert$ do   
$/ /$ inner loop: sweep $T _ { 1 }$   
$T _ { 1 } \gets \mathcal { T } _ { 1 } [ k ] ;$   
foreach $( i , \ell ) \in B [ T _ { 1 } ]$ do   
$/ /$ process pairs introduced by   
the current $T _ { 1 }$   
if $S _ { i , 4 } ^ { [ \ell ] } + S _ { i , 5 } ^ { [ \ell ] } \le T _ { 2 }$ and $( E _ { i } ^ { [ \ell ] } + \lambda S _ { i , 3 } ^ { [ \ell ] } ) < c _ { i }$   
then   
$/ /$ passes $T _ { 2 }$ check and   
reduces client-i cost   
if $c _ { i } = \infty$ then   
count ← count $_ { \mathrm { ~ \scriptsize ~ + ~ } 1 ; }$   
$G  G + ( E _ { i } ^ { [ \ell ] } + \lambda S _ { i , 3 } ^ { [ \ell ] } ) ;$   
else   
$/ /$ Replace the previous   
best cut of client $i$   
$G  G - c _ { i } + ( E _ { i } ^ { [ \ell ] } + \lambda S _ { i , 3 } ^ { [ \ell ] } ) ;$   
end   
$c _ { i }  ( E _ { i } ^ { [ \ell ] } + \lambda S _ { i , 3 } ^ { [ \ell ] } ) ; \ell _ { i }  \ell ;$   
end   
end   
if count = |N| then   
$J  G + \lambda ( T _ { 1 } + T _ { 2 } ) \colon$   
if $J < J ^ { * }$ then   
$\begin{array} { r l } { \mid } & { { } J ^ { * }  J ; \ \{ \ell _ { i } ^ { * } \}  \{ \ell _ { i } \} } \end{array}$   
end   
$/ /$ Prune the remaining $T _ { 1 }$   
candidates for this fixed   
$T _ { 2 }$ (Theorem 1)   
if k $< | \mathcal { T } _ { 1 } |$ and $\lambda ( \mathcal { T } _ { 1 } [ k + 1 ] - \mathcal { T } _ { 1 } [ k ] ) >$   
$G - G _ { \mathrm { m i n } }$ then   
break   
end   
end   
end   
end   
return $J ^ { * } , \{ \ell _ { i } ^ { * } \}$ ;

Theorem 1 (Optimality of Algorithm 1). Algorithm 1 obtains the globally optimal solution to Problem P3.

Proof. Fix any $T _ { 2 } \in \mathcal { T } _ { 2 }$ . Let $\{ \tau _ { 1 , k } \} _ { k = 1 } ^ { | \mathcal { T } _ { 1 } | }$ denote the elements of $\mathcal { T } _ { 1 }$ sorted in ascending order, i.e., $\tau _ { 1 , 1 } \leq \tau _ { 1 , 2 } \leq \cdot \cdot \cdot \leq \tau _ { 1 , | T _ { 1 } | } .$ Recall $J ( T _ { 1 } , T _ { 2 } ) = G ( T _ { 1 } , T _ { 2 } ) + \lambda ( T _ { 1 } + T _ { 2 } )$ , where $G ( \cdot , T _ { 2 } )$ is non-increasing in $T _ { 1 }$ and is lower bounded by $G _ { \operatorname* { m i n } } \triangleq$ where the inequality uses $G ( \tau _ { 1 , m } , T _ { 2 } ) \geq G _ { \mathrm { m i n } }$ . Since $\tau _ { 1 , m } -$ $\tau _ { 1 , k } \geq \tau _ { 1 , k + 1 } - \tau _ { 1 , k } .$ , it follows that

```latex
$\begin{array} { r } { \sum _ { i \in \mathcal { N } } \operatorname* { m i n } _ { \ell \in \{ 1 , \dots , L _ { \operatorname* { m a x } } \} } ( E _ { i } ^ { [ \ell ] } + \lambda S _ { i , 3 } ^ { [ \ell ] } ) . } \end{array}$
At step $k ,$ for any future candidate $\tau _ { 1 , m } > \tau _ { 1 , k }$ , we have
J(τ<sub>1,m</sub>, T<sub>2</sub>) − J(τ<sub>1,k</sub>, T<sub>2</sub>)
=  G(τ<sub>1,m</sub>, T<sub>2</sub>) − G(τ<sub>1,k</sub>, T<sub>2</sub>) + λ(τ<sub>1,m</sub> − τ<sub>1,k</sub>)
≥ (G<sub>min</sub> − G(τ<sub>1,k</sub>, T<sub>2</sub>)) + λ(τ<sub>1,m</sub> − τ<sub>1,k</sub>),
```

$$
J ( \tau _ { 1 , m } , T _ { 2 } ) - J ( \tau _ { 1 , k } , T _ { 2 } ) \geq - \Delta G _ { \operatorname* { m a x } } + \lambda ( \tau _ { 1 , k + 1 } - \tau _ { 1 , k } ) ,
$$

with $\Delta G _ { \mathrm { m a x } } \triangleq G ( \tau _ { 1 , k } , T _ { 2 } ) \mathrm { - } G _ { \mathrm { m i n } } .$ . Thus, if $\cdot \lambda ( \tau _ { 1 , k + 1 } - \tau _ { 1 , k } ) >$ $\Delta G _ { \mathrm { m a x } } ,$ then $J ( \tau _ { 1 , m } , T _ { 2 } ) > J ( \tau _ { 1 , k } , T _ { 2 } )$ for all $m > k ,$ so no better solution exists beyond $\tau _ { 1 , k }$ and the scan over $T _ { 1 }$ can be safely terminated for this fixed $T _ { 2 }$ . Furthermore, enumerating all $T _ { 2 } ~ \in ~ \mathcal { T } _ { 2 }$ ensures that the global minimizer $( T _ { 1 } ^ { * } , T _ { 2 } ^ { * } )$ is not missed, hence the algorithm returns a globally optimal $\{ y _ { i , \ell } ^ { * } \}$ □

Complexity. Algorithm 1 first constructs the candidate sets $\mathcal { T } _ { 1 } = \tilde { \{ S _ { i , 1 } ^ { [ \ell ] } + S _ { i , 2 } ^ { [ \ell ] } \} }$ and $\mathcal { T } _ { 2 } = \{ S _ { i , 4 } ^ { [ \ell ] } + S _ { i , 5 } ^ { [ \ell ] } \}$ and sorts them, which costs $O ( N L \log ( N L ) )$ since $| \mathcal { T } _ { 1 } | , | \mathcal { T } _ { 2 } | \le N ( L - 1 )$ . For each fixed $T _ { 2 } \in \mathcal { T } _ { 2 }$ , Algorithm 1 scans $T _ { 1 } \in \mathcal { T } _ { 1 }$ in ascending order. Because $T _ { 1 }$ is enumerated over the sorted distinct values in $\mathcal { T } _ { 1 }$ , the uplink-feasible set $\{ ( i , \ell ) | S _ { i , 1 } ^ { [ \ell ] } + S _ { i , 2 } ^ { [ \ell ] } \leq T _ { 1 } \}$ expands only when $T _ { 1 }$ advances to the next candidate; hence, at a given $T _ { 1 }$ it suffices to process only the newly activated pairs $( i , \ell )$ satisfying $S _ { i , 1 } ^ { [ \ell ] } + S _ { i , 2 } ^ { [ \ell ] } = T _ { 1 }$ . For each such pair, the algorithm performs (i) one comparison to test downlink feasibility $S _ { i , 4 } ^ { [ \ell ] } { + } S _ { i , 5 } ^ { [ \ell ] } \leq \dot { T _ { 2 } }$ , (ii) one evaluation of $E _ { i } ^ { [ \ell ] } + \lambda S _ { i , 3 } ^ { [ \ell ] }$ and (iii) a constant-time comparison and possible update of the stored best value $c _ { i }$ together with the running sum. All these operations take $O ( 1 )$ time per processed pair. Thus, over a complete scan of $T _ { 1 }$ for a fixed $T _ { 2 }$ , each $( i , \ell )$ is processed at most once, and the total update cost per $T _ { 2 }$ is $O ( N L )$ Consequently, the overall running time of Algorithm 1 is

$$
\begin{array} { r l } & { \quad O \big ( | T _ { 2 } | \cdot N L \big ) + O \big ( N L \log ( N L ) \big ) } \\ & { { \bf \delta } = O \big ( N ^ { 2 } L ^ { 2 } + N L \log ( N L ) \big ) } \\ & { { \bf \delta } = O \big ( N ^ { 2 } L ^ { 2 } \big ) . } \end{array}\tag{35}
$$

## VI. JOINT OPTIMIZATION OF MODEL SPLITTING AND RESOURCE ALLOCATION

We now present a polynomial-time approximation method for the general joint optimization of model splitting and resource allocation. Specifically, we first reformulate Problem P2 exactly as a two-dimensional master problem with respect to the synchronization variables $( T _ { 1 } , T _ { 2 } )$ . Since the resulting master problem is generally nonconvex, we then approximate it via a uniform grid search over the $( T _ { 1 } , T _ { 2 } )$ domain. This leads to a polynomial-time approximation algorithm with an explicit performance guarantee.

## A. Two-Dimensional Bottleneck Projection

Recall that the latency term in Problem P2 is characterized by the two epigraph variables $T _ { 1 }$ and $T _ { 2 }$ , where

$$
s _ { i , 1 } + s _ { i , 2 } \leq T _ { 1 } , \qquad \forall i \in \mathcal { N } ,\tag{36}
$$

$$
s _ { i , 4 } + s _ { i , 5 } \leq T _ { 2 } , \qquad \forall i \in \mathcal { N } .\tag{37}
$$

Once $( T _ { 1 } , T _ { 2 } )$ is fixed, the optimization variables across clients become decoupled. This key observation enables us to project the original mixed-integer problem onto the two synchronization variables. Specifically, for each client $i ,$ stage $j ,$ and cut layer $\ell \in \{ 1 , 2 , \dots , L _ { \mathrm { m a x } } \}$ , let $\widehat { \Psi } _ { i , j } ^ { ( \ell ) } \left( s _ { i , j } \right)$ denote the minimum stage energy when the service time is $s _ { i , j }$ and the cut layer is fixed at ℓ, i.e.,

$$
\widehat { \Psi } _ { i , j } ^ { ( \ell ) } \left( s _ { i , j } \right) = \operatorname* { m i n } _ { u _ { i , j } ( t ) } E _ { i , j } \left( u _ { i , j } , s _ { i , j } , L _ { c , i } { = } \ell \right) .\tag{38}
$$

Moreover, the feasible service-time is given by

$$
S _ { i , j } ^ { ( \ell ) , \operatorname* { m i n } } \leq s _ { i , j } \leq S _ { i , j } ^ { ( \ell ) , \operatorname* { m a x } } .\tag{39}
$$

For fixed $( T _ { 1 } , T _ { 2 } )$ , the per-client optimization decomposes into three independent subproblems. First, the server-side FP/BP stage contributes additively to the objective and is not directly constrained by $T _ { 1 }$ or T . Accordingly, we define

$$
\mathbf { P 4 } ^ { \prime } : U _ { i , 3 } ^ { ( \ell ) } \triangleq \operatorname* { m i n } _ { s _ { i , 3 } } \ { \widehat { \Psi } } _ { i , 3 } ^ { ( \ell ) } { \left( s _ { i , 3 } \right) } + \lambda s _ { i , 3 }\tag{40a}
$$

$$
\begin{array} { r l } { \mathrm { s . t . } } & { { } S _ { i , 3 } ^ { ( \ell ) , \mathrm { m i n } } \le s _ { i , 3 } \le S _ { i , 3 } ^ { ( \ell ) , \mathrm { m a x } } . } \end{array}\tag{40b}
$$

Second, the first synchronization variable $T _ { 1 }$ constrains the sum of client-side FP time and uplink transmission time, which is given by

$$
\mathbf { P } 4 ^ { \prime \prime } : U _ { i , 1 2 } ^ { ( \ell ) } ( T _ { 1 } ) \triangleq \operatorname* { m i n } _ { s _ { i , 1 } , s _ { i , 2 } } \widehat { \Psi } _ { i , 1 } ^ { ( \ell ) } ( s _ { i , 1 } ) + \widehat { \Psi } _ { i , 2 } ^ { ( \ell ) } ( s _ { i , 2 } )\tag{41a}
$$

$$
\mathrm { s . t . } s _ { i , 1 } + s _ { i , 2 } \leq T _ { 1 } ,\tag{41b}
$$

$$
S _ { i , 1 } ^ { ( \ell ) , \operatorname* { m i n } } \leq s _ { i , 1 } \leq S _ { i , 1 } ^ { ( \ell ) , \operatorname* { m a x } } ,\tag{41c}
$$

$$
S _ { i , 2 } ^ { ( \ell ) , \mathrm { { m i n } } } \leq s _ { i , 2 } \leq S _ { i , 2 } ^ { ( \ell ) , \mathrm { { m a x } } } .\tag{41d}
$$

If $T _ { 1 } < S _ { i , 1 } ^ { ( \ell ) , \mathrm { m i n } } + S _ { i , 2 } ^ { ( \ell ) , \mathrm { m i n } }$ , then cut layer ℓ is infeasible for client i under $T _ { 1 }$ , and we set $U _ { i , 1 2 } ^ { ( \ell ) } ( T _ { 1 } ) = + \infty$

Third, considering $T _ { 2 }$ constrains the sum of the downlink transmission time and client-side BP time, we define

$$
\mathbf { P } 4 ^ { \prime \prime \prime } : U _ { i , 4 5 } ^ { ( \ell ) } ( T _ { 2 } ) \triangleq \operatorname* { m i n } _ { s _ { i , 4 } , s _ { i , 5 } } \widehat { \Psi } _ { i , 4 } ^ { ( \ell ) } ( s _ { i , 4 } ) + \widehat { \Psi } _ { i , 5 } ^ { ( \ell ) } ( s _ { i , 5 } )\tag{42a}
$$

$$
\mathrm { s . t . } s _ { i , 4 } + s _ { i , 5 } \leq T _ { 2 } ,\tag{42b}
$$

$$
S _ { i , 4 } ^ { ( \ell ) , \operatorname* { m i n } } \leq s _ { i , 4 } \leq S _ { i , 4 } ^ { ( \ell ) , \operatorname* { m a x } } ,\tag{42c}
$$

$$
S _ { i , 5 } ^ { ( \ell ) , \operatorname* { m i n } } \leq s _ { i , 5 } \leq S _ { i , 5 } ^ { ( \ell ) , \operatorname* { m a x } } .\tag{42d}
$$

If $T _ { 2 } < S _ { i , 4 } ^ { ( \ell ) , \mathrm { m i n } } + S _ { i , 5 } ^ { ( \ell ) , \mathrm { m i n } }$ , then cut layer ℓ is infeasible for client i under $T _ { 2 } ,$ , and we set $U _ { i , 4 5 } ^ { ( \ell ) } ( T _ { 2 } ) = + \infty$

For fixed $( T _ { 1 } , T _ { 2 } )$ , client i evaluates all candidate cut layers and chooses the least-cost feasible one. This yields the client value function

$$
V _ { i } ( T _ { 1 } , T _ { 2 } ) \triangleq \operatorname* { m i n } _ { \ell \in \{ 1 , 2 , \ldots , L _ { \operatorname* { m a x } } \} } \Big ( U _ { i , 1 2 } ^ { ( \ell ) } ( T _ { 1 } ) + U _ { i , 4 5 } ^ { ( \ell ) } ( T _ { 2 } ) + U _ { i , 3 } ^ { ( \ell ) } \Big ) .\tag{43}
$$

Hence, the original mixed-integer Problem P2 is exactly reduced to the following two-dimensional master problem:

$$
{ \bf P 5 } : \quad \operatorname* { m i n } _ { T _ { 1 } , T _ { 2 } } J ( T _ { 1 } , T _ { 2 } ) \triangleq \lambda ( T _ { 1 } + T _ { 2 } ) + \sum _ { i \in \mathcal { N } } V _ { i } ( T _ { 1 } , T _ { 2 } ) .\tag{44}
$$

where $V _ { i } ( T _ { 1 } , T _ { 2 } )$ is defined in (43). The feasibility conditions are already incorporated into the definitions of $U _ { i , 3 } ^ { ( \ell ) } , U _ { i , 1 2 } ^ { ( \ell ) } ( T _ { 1 } )$ ， and $U _ { i , 4 5 } ^ { ( \ell ) } ( T _ { 2 } )$ , with infeasible choices assigned value +∞.

Theorem 2 (Problem Equivalence). Problem P2 is equivalent to the master problem in (44). In particular, solving Problem P5 yields an optimal solution to Problem P2.

Proof. Fix any feasible pair $( T _ { 1 } , T _ { 2 } )$ . Under this pair, the only coupling among clients is removed, and the optimization variables of different clients become separable. For each client i and each cut layer $\ell ,$ the optimal contribution to the objective decomposes into the three blocks $U _ { i , 1 2 } ^ { ( \ell ) } ( T _ { 1 } ) , ~ U _ { i , 4 5 } ^ { ( \ell ) } ( T _ { 2 } )$ and $U _ { i , 3 } ^ { ( \ell ) }$ . Minimizing over $\ell \in \{ 1 , 2 , \dots , L _ { \mathrm { m a x } } \}$ gives the client value function $V _ { i } ( T _ { 1 } , T _ { 2 } )$ in (43). Summing the optimal client values over all clients and adding the synchronization penalty $\lambda ( T _ { 1 } + T _ { 2 } )$ yields (44). Thus, minimizing Problem P5 over $( T _ { 1 } , T _ { 2 } )$ is equivalent to solving Problem P2. □

## B. Grid Approximation of the Master Problem

The reformulation in (44) is equivalent to the original problem. However, the master objective $J ( T _ { 1 } , T _ { 2 } )$ is generally nonconvex because each client value function $V _ { i } ( T _ { 1 } , T _ { 2 } )$ contains a minimization over discrete cut layers. Since the master problem is only two-dimensional, we approximate it via a uniform grid search over the bounded rectangle

$$
T _ { 1 } \in [ \underline { { T } } _ { 1 } , \overline { { T } } _ { 1 } ] , \qquad T _ { 2 } \in [ \underline { { T } } _ { 2 } , \overline { { T } } _ { 2 } ] ,\tag{45}
$$

where the search box is chosen to contain at least one global optimizer of (44). For a grid step size $\Delta > 0$ , we evaluate $J ( T _ { 1 } , T _ { 2 } )$ at all grid points in (45) and return the best value, denoted by $J _ { \mathrm { g r i d } }$ . The following theorem provides an additive approximation guarantee.

Theorem 3 (Relative Error Bound). Assume that the search region $[ \underline { { T } } _ { 1 } , \overline { { T } } _ { 1 } ] \times [ \underline { { T } } _ { 2 } , \overline { { T } } _ { 2 } ]$ contains at least one global optimizer $( T _ { 1 } ^ { * } , T _ { 2 } ^ { * } ) o f \ : ( 4 4 )$ , and let $J ^ { * } \triangleq J ( T _ { 1 } ^ { * } , T _ { 2 } ^ { * } )$ . Suppose that $J ^ { * } \geq \lambda ( \underline { { T } } _ { 1 } + \underline { { T } } _ { 2 } )$ , where $\underline { { T } } _ { 1 } + \underline { { T } } _ { 2 } > 0$ . If a uniform grid with step size $\Delta$ is used in both dimensions, then

$$
\frac { J _ { \mathrm { g r i d } } - J ^ { * } } { J ^ { * } } \leq \frac { 2 \Delta } { \underline { { T } } _ { 1 } + \underline { { T } } _ { 2 } } .
$$

Consequently, in order to guarantee $J _ { \mathrm { g r i d } } ~ \le ~ ( 1 + \epsilon ) J ^ { * }$ , it suffices to choose $\begin{array} { r } { \Delta \leq \frac { \epsilon } { 2 } ( \underline { { T } } _ { 1 } + \underline { { T } } _ { 2 } ) } \end{array}$

Proof. Since the grid spacing is ∆, there exists a grid point $( \widetilde { T } _ { 1 } , \widetilde { T } _ { 2 } )$ such that

$$
\begin{array} { r } { T _ { 1 } ^ { * } \leq \widetilde { T } _ { 1 } \leq T _ { 1 } ^ { * } + \Delta , } \\ { T _ { 2 } ^ { * } \leq \widetilde { T } _ { 2 } \leq T _ { 2 } ^ { * } + \Delta . } \end{array}
$$

For each client $i \in \mathcal N ,$ increasing $T _ { 1 }$ or $T _ { 2 }$ relaxes the corresponding synchronization constraints and thus cannot increase the value function $V _ { i } ( T _ { 1 } , T _ { 2 } )$ . Hence,

$$
V _ { i } ( \widetilde { T } _ { 1 } , \widetilde { T } _ { 2 } ) \le V _ { i } ( T _ { 1 } ^ { * } , T _ { 2 } ^ { * } ) , \qquad \forall i \in \mathcal { N } .
$$

It follows that

$$
\begin{array} { r l } { J ( \widetilde { T } _ { 1 } , \widetilde { T } _ { 2 } ) = \lambda ( \widetilde { T } _ { 1 } + \widetilde { T } _ { 2 } ) + \displaystyle \sum _ { i \in \mathcal { N } } V _ { i } ( \widetilde { T } _ { 1 } , \widetilde { T } _ { 2 } ) } \\ { \displaystyle } \\ { \leq \lambda ( T _ { 1 } ^ { * } + \Delta + T _ { 2 } ^ { * } + \Delta ) + \displaystyle \sum _ { i \in \mathcal { N } } V _ { i } ( T _ { 1 } ^ { * } , T _ { 2 } ^ { * } ) } \\ { \displaystyle } \\ { \displaystyle } \end{array}
$$

Since the grid-search algorithm returns the best grid point, we further have

$$
J _ { \mathrm { g r i d } } \leq J ( \widetilde { T } _ { 1 } , \widetilde { T } _ { 2 } ) .
$$

Then, we have $J _ { \mathrm { g r i d } } - J ^ { * } \le 2 \lambda \Delta$ . Dividing both sides by $J ^ { * }$ and using $J ^ { * } \geq \bar { \lambda } ( \underline { { T } } _ { 1 } + \underline { { T } } _ { 2 } )$ , we obtain

$$
\frac { J _ { \mathrm { g r i d } } - J ^ { * } } { J ^ { * } } \leq \frac { 2 \lambda \Delta } { J ^ { * } } \leq \frac { 2 \Delta } { \underline { { T } } _ { 1 } + \underline { { T } } _ { 2 } } .
$$

The desired $( 1 + \epsilon )$ -approximation guarantee is obtained by requiring $2 \Delta / ( \underline { { T } } _ { 1 } + \underline { { T } } _ { 2 } ) \leq \epsilon .$ , which completes the proof.

Theorem 4 (Complexity of Grid Approximation). Let $\eta > 0$ denote the numerical accuracy to solve the inner subproblems. Then, the proposed grid-search algorithm has total complexity $O \big ( G _ { 1 } G _ { 2 } N L \log ( 1 / \eta ) \big )$ . Under bounded search ranges, choosing $\Delta$ according to Theorem 3 yields the total complexity $O \big ( ( N L / \epsilon ^ { 2 } ) \log ( 1 / \eta ) \big )$ . Hence, Algorithm 2 runs in polynomial time.

Proof. 1) Complexity of the inner subproblems. Fix any grid point $( T _ { 1 } , T _ { 2 } )$ , client $i ,$ and candidate cut layer ℓ. For the corresponding client i pair $( i , \ell )$ with cut layer $\ell ,$ the algorithm evaluates the three inner quantities $U _ { i , 3 } ^ { ( \ell ) } , \bar { U _ { i , 1 2 } ^ { ( \ell ) } } ( T _ { 1 } ) , U _ { i , 4 5 } ^ { ( \overline { { \ell } } ) } ( T _ { 2 } )$ We first show that these inner subproblems are convex. For the computation stages $j \in \{ 1 , 3 , 5 \}$ , under a fixed cut layer $\ell ,$ the stage-energy functions are given by

$$
\widehat { \Psi } _ { i , j } ^ { ( \ell ) } ( s _ { i , j } ) = a _ { i , j } ^ { ( \ell ) } / s _ { i , j } ^ { 2 } , \qquad j \in \{ 1 , 3 , 5 \} ,
$$

where

$$
a _ { i , j } ^ { ( \ell ) } = \left\{ \begin{array} { l l } { G _ { i } \left( \displaystyle \sum _ { k = 1 } ^ { \ell } \rho _ { k } \right) ^ { 3 } / n _ { i } ^ { 3 } , } & { j = 1 , } \\ { G _ { \mathrm { S } } \left( \displaystyle \sum _ { k = \ell + 1 } ^ { L } \left( \rho _ { k } + \varpi _ { k } \right) \right) ^ { 3 } / n _ { \mathrm { S } } ^ { 3 } , } & { j = 3 , } \\ { G _ { i } \left( \displaystyle \sum _ { k = 1 } ^ { \ell } \varpi _ { k } \right) ^ { 3 } / n _ { i } ^ { 3 } , } & { j = 5 . } \end{array} \right.
$$

Since

$$
\frac { d ^ { 2 } } { d s _ { i , j } ^ { 2 } } \left( \frac { a _ { i , j } ^ { ( \ell ) } } { s _ { i , j } ^ { 2 } } \right) = \frac { 6 a _ { i , j } ^ { ( \ell ) } } { s _ { i , j } ^ { 4 } } > 0 , \qquad \forall s _ { i , j } > 0 ,
$$

it follows that $\widehat { \Psi } _ { i , j } ^ { ( \ell ) } \left( s _ { i , j } \right)$ is convex on $s _ { i , j } > 0 .$

For the communication stages $j \in \{ 2 , 4 \}$ , under a fixed cut layer ℓ, the stage-energy functions can be written as

$$
\widehat { \Psi } _ { i , j } ^ { ( \ell ) } ( s _ { i , j } ) = b _ { i , j } ^ { ( \ell ) } s _ { i , j } \left( 2 ^ { c _ { i , j } ^ { ( \ell ) } / s _ { i , j } } - 1 \right) , \qquad j \in \{ 2 , 4 \} ,
$$

Algorithm 2: Grid Approximation for the Two-  
Dimensional Master Problem   
Input: $\mathcal { N } , \mathcal { L } , [ \underline { { T } } _ { 1 } , \overline { { T } } _ { 1 } ] , [ \underline { { T } } _ { 2 } , \overline { { T } } _ { 2 } ] , \Delta , \lambda .$   
Output: $J ^ { * } , ( T _ { 1 } ^ { * } , T _ { 2 } ^ { * } ) , \{ \ell _ { i } ^ { * } \} .$   
$\mathcal { G } _ { 1 } \stackrel { - } {  } \{ \underline { { T } } _ { 1 } , \underline { { T } } _ { 1 } + \Delta , \underline { { T } } _ { 1 } + 2 \Delta , . . . , \overline { { T } } _ { 1 } \} ;$   
$\mathcal { G } _ { 2 } \gets \{ \underline { { T } } _ { 2 } , \underline { { T } } _ { 2 } + \Delta , \underline { { T } } _ { 2 } + 2 \Delta , . . . , \overline { { T } } _ { 2 } \} ;$   
$J ^ { * } \gets \infty ;$   
foreach $T _ { 1 } \in \mathcal { G } _ { 1 }$ do   
foreach $T _ { 2 } \in \mathcal G _ { 2 }$ do   
$J \gets \lambda ( T _ { 1 } + T _ { 2 } ) ;$   
feasible $ 1 ;$   
Compute $U _ { i , 3 } ^ { ( \ell ) } , U _ { i , 1 2 } ^ { ( \ell ) } ( T _ { 1 } )$ , and $U _ { i , 4 5 } ^ { ( \ell ) } ( T _ { 2 } )$ for all   
required $( i , \ell , T _ { 1 } , T _ { 2 } )$ combinations;   
foreach $i \in \mathcal N$ do   
$V _ { i } \gets \infty , \ \ell _ { i } \gets - 1 ;$   
foreach $\ell \in { \mathcal { L } }$ do   
i $\cdot T _ { 1 } < S _ { i , 1 } ^ { ( \ell ) , \mathrm { m i n } } + S _ { i , 2 } ^ { ( \ell ) , \mathrm { m i n } }$ or   
$T _ { 2 } < S _ { i , 4 } ^ { ( \ell ) , \mathrm { m i n } } + S _ { i , 5 } ^ { ( \ell ) , \mathrm { m i n } }$ then   
continue;   
end   
$C _ { i } ^ { ( \ell ) } \gets U _ { i , 3 } ^ { ( \ell ) } + U _ { i , 1 2 } ^ { ( \ell ) } ( T _ { 1 } ) + U _ { i , 4 5 } ^ { ( \ell ) } ( T _ { 2 } ) ;$   
if $C _ { i } ^ { ( \ell ) } < V _ { i }$ then   
$V _ { i }  C _ { i } ^ { ( \ell ) } , \ell _ { i }  \ell ;$   
end   
end   
if $V _ { i } = \infty$ then   
feasible $ 0 ;$   
break;   
end   
$J \gets J + V _ { i } ;$   
end   
if feasible = 1 and $J < J ^ { * }$ then   
$J ^ { * } \gets J , ~ ( T _ { 1 } ^ { * } , T _ { 2 } ^ { * } ) \gets ( T _ { 1 } , T _ { 2 } ) , ~ \{ \ell _ { i } ^ { * } \} \gets$   
$\{ \ell _ { i } \} ;$   
end   
end   
end   
return $J ^ { * } , ( T _ { 1 } ^ { * } , T _ { 2 } ^ { * } ) , \{ \ell _ { i } ^ { * } \} ;$

where

$$
\begin{array} { r } { b _ { i , j } ^ { ( \ell ) } = \left\{ \begin{array} { l l } { \displaystyle \frac { B ^ { \mathrm { U } } N _ { 0 } } { h _ { i } ^ { 2 } } , } & { j = 2 , } \\ { \displaystyle \frac { B ^ { \mathrm { D } } N _ { 0 } } { h _ { \mathrm { S } } ^ { 2 } } , } & { j = 4 , } \end{array} \right. \qquad c _ { i , j } ^ { ( \ell ) } = \left\{ \begin{array} { l l } { \displaystyle \frac { d _ { i } ^ { \mathrm { U } } \left( \ell \right) } { B ^ { \mathrm { U } } } , } & { j = 2 , } \\ { \displaystyle \frac { d _ { i } ^ { \mathrm { D } } \left( \ell \right) } { B ^ { \mathrm { D } } } , } & { j = 4 . } \end{array} \right. } \end{array}
$$

Define

$$
\phi _ { i , j } ^ { ( \ell ) } ( s _ { i , j } ) \triangleq b _ { i , j } ^ { ( \ell ) } s _ { i , j } \left( 2 ^ { c _ { i , j } ^ { ( \ell ) } / s _ { i , j } } - 1 \right) , \qquad s _ { i , j } > 0 .
$$

Then

$$
\frac { d ^ { 2 } } { d s _ { i , j } ^ { 2 } } \phi _ { i , j } ^ { ( \ell ) } ( s _ { i , j } ) = \frac { b _ { i , j } ^ { ( \ell ) } 2 ^ { c _ { i , j } ^ { ( \ell ) } / s _ { i , j } } \big ( c _ { i , j } ^ { ( \ell ) } \big ) ^ { 2 } ( \ln 2 ) ^ { 2 } } { s _ { i , j } ^ { 3 } } > 0 , \ \forall s _ { i , j } > 0 .
$$

$$
\widehat { \Psi } _ { i , j } ^ { ( \ell ) } \left( s _ { i , j } \right)
$$

$$
U _ { i , 3 } ^ { ( \ell ) }
$$

$$
j \in \{ 2 , 4 \}
$$

the interval $[ S _ { i , 3 } ^ { ( \ell ) , \mathrm { m i n } } , S _ { i , 3 } ^ { ( \ell ) , \mathrm { m a x } } ]$ . Moreover, $U _ { i , 1 2 } ^ { ( \ell ) } ( T _ { 1 } )$ and $U _ { i , 4 5 } ^ { ( \ell ) } ( T _ { 2 } )$ are fixed-dimensional convex programs with box constraints and one affine coupling constraint. Hence, these inner subproblems can be solved efficiently. In particular, $U _ { i , 3 } ^ { ( \ell ) }$ admits a closed-form solution after projecting the unconstrained minimizer onto its feasible interval, while $U _ { i , 1 2 } ^ { ( \ell ) } ( T _ { 1 } )$ and $U _ { i , 4 5 } ^ { ( \ell ) } ( T _ { 2 } )$ can be solved to accuracy η in time polynomial in log(1/η) by standard convex optimization methods. Thus, the total cost per client–cut-layer pair is ${ \cal O } ( \log ( 1 / \eta ) )$

2) Complexity of the outer grid search. Let

$$
{ \cal G } _ { 1 } \triangleq \left\lceil \frac { \overline { T } _ { 1 } - \underline { T } _ { 1 } } { \Delta } \right\rceil + 1 , \qquad { \cal G } _ { 2 } \triangleq \left\lceil \frac { \overline { T } _ { 2 } - \underline { T } _ { 2 } } { \Delta } \right\rceil + 1
$$

denote the numbers of grid points along the two dimensions. Then, the grid contains $G _ { 1 } G _ { 2 }$ points in total. At each grid point, the algorithm evaluates all N clients and at most $L - 1$ candidate cut layers. Combining the complexity of the inner subproblems, the total complexity is

$$
O \big ( G _ { 1 } G _ { 2 } N ( L - 1 ) \log ( 1 / \eta ) \big ) = O \big ( G _ { 1 } G _ { 2 } N L \log ( 1 / \eta ) \big ) .
$$

Finally, under bounded search ranges, Theorem 3 implies that $\Delta = O ( \epsilon )$ . Hence, $G _ { 1 } = { \cal O } ( 1 / \epsilon ) , G _ { 2 } = { \cal O } ( 1 / \epsilon )$ , and thus $G _ { 1 } G _ { 2 } = O ( 1 / \epsilon ^ { 2 } )$ . Substituting this into the above bound yields $O \big ( ( N L / \epsilon ^ { 2 } ) \log ( 1 / \eta ) \big )$ , which completes the proof.

## C. Recovering the Optimal Control Inputs

Once the optimal solution of the high-level problem P is obtained, the optimal solution to the original hybrid control problem can be recovered by solving the lower-level control subproblems stage by stage. In particular, the high-level optimizer determines the optimal cut-layer decisions $\{ L _ { c , i } ^ { * } \}$ and the optimal service-time variables $\left\{ \boldsymbol { s } _ { i , j } ^ { * } \right\}$ . With these decisions solved, the corresponding optimal control input is recovered as

$$
u _ { i , j } ^ { * } ( s _ { i , j } ^ { * } ) = \arg \operatorname* { m i n } _ { u _ { i , j } } \psi _ { i , j } \left( z _ { i , j } , u _ { i , j } , s _ { i , j } ^ { * } , L _ { c , i } ^ { * } \right) ,\tag{46}
$$

subject to the associated system dynamics and boundary conditions. Thus, the optimal policy of the original hybrid control problem is obtained by combining the high-level optimal resource-allocation decisions with the recovered lowlevel control inputs.

## VII. SIMULATIONS

## A. Simulation Setup

We use the datasets MNIST [37] and CIFAR-10 [38] to evaluate the performance of the proposed method for the models ResNet-50 [39] and VGG-16 [40] under both IID and non-IID data distributions. Notably, for the non-IID setting, each client is assigned a primary class from which approximately 70% of its samples are drawn, while the remaining samples are sourced from other classes. We consider a network of $N = 2 0$ clients, unless specified otherwise. Each client uses a mini-batch size of 32 with a learning rate $\gamma = 0 . 0 0 1$ . The algorithms are run on a workstation equipped with an AMD Ryzen Threadripper PRO 5975WX and NVIDIA GeForce RTX 4090.

In the experiments, the tradeoff coefficient is set to $\lambda =$ 100 to balance training latency and energy consumption. The client-side computation capability is heterogeneous, with $n _ { i }$ uniformly selected from $[ 0 . 2 , 3 ] \ \times \ 1 0 ^ { 4 } \mathrm { F L O P s / c y c l e } .$ while the edge server is configured with $n _ { \mathrm { { S } } } ~ = ~ 3 ~ \times ~ 1 0 ^ { 4 }$ FLOPs/cycle. The computation-energy coefficients are set to $G _ { i } \ \stackrel { \cdot } { = } \ 1 \times 1 0 ^ { - 2 6 } \mathrm { W } / ( \mathrm { c y c l e / s } ) ^ { 3 }$ for clients and $\ : G _ { \mathrm { S } } ^ { } \ : = \ :$ $2 \times 1 0 ^ { - 2 6 } \mathrm { W } / ( \mathrm { c y c l e / s } ) ^ { 3 }$ for the server. For the NVIDIA Jetson Orin Nano client platform, the computing frequency is configured within the measured graphics-clock range, namely from $f _ { i } ^ { \mathrm { m i n } } ~ = ~ 5 0$ MHz to $f _ { i } ^ { \mathrm { m a x } } ~ = ~ 6 2 5 ~ \mathrm { { \ : M H z } }$ . For the NVIDIA GeForce RTX 4090 server platform, the computing frequency is configured within the supported graphics-clock range measured on our testbed, namely from $f _ { \mathrm { S } } ^ { \mathrm { m i n } } = 2 1 0 \ : \mathrm { M H z }$ to $f _ { \mathrm { S } } ^ { \mathrm { m a x } } = 3 . 1 3 5 \ : \mathrm { G H z }$ [41], [42]. For wireless communication, the uplink and downlink bandwidths are set to $B _ { i } ^ { \mathrm { U } } = 1 0 0 \ : \mathrm { M H z }$ and $B _ { i } ^ { \mathrm { D } } = 2 0 0 \ : \mathrm { M H z } ,$ , respectively. The transmit power of each client and the edge server are bounded within [20, 33] dBm. The uplink channels are modeled as independent Rayleigh fading channels, where the average channel power gain of client i is denoted by $| h _ { i } | ^ { 2 }$ and generated within $[ 0 . 1 , 1 ] \times 1 0 ^ { - 6 }$ Similarly, the average downlink channel power gain is denoted by $| h _ { \mathrm { S } } | ^ { 2 }$ and fixed at $1 \times 1 0 ^ { - 6 }$ [34]. The noise power spectral density is set to $N _ { 0 } = 3 . 9 8 \times 1 0 ^ { - 2 1 } \mathrm { W / H z }$ . For the reader’s convenience, the detailed simulation parameters are summarized in Table II.

To demonstrate the advantages of our proposed method “Optimal Cut and Optimal GPU Frequency Scaling and Power Control” (OC + OGP), we compare it with five benchmarks: 1) “Optimal Cut and Optimal GPU Frequency Scaling” (OC $\mathbf { \nabla } + \mathbf { \nabla } \mathbf { O } \mathbf { G } )$ , which optimizes the model partition point and GPU frequency; 2) “Optimal Cut and Optimal Power Control” $( \mathbf { O C \theta } + \mathbf { \theta } \mathbf { O P } )$ which jointly optimizes the model partition point and transmit power; 3) “Optimal Cut” (OC), which optimizes only the model partition; 4) efficient split federated learning (ESFL) [17], which aims to reduce training latency by optimizing client-side workload and server-side workload; and 5) dynamic-step Q-Learning method (DSQL) [23], which employs a reinforcement learning method to reduce the weighted sum of training time and energy consumption. In our experiments, the total cost is defined as the cumulative cost incurred until the model reaches 70% test accuracy on CIFAR-10 under both IID and non-IID data distributions.

## B. Performance Comparison

Fig. 3 evaluates the training performance by showing the cost function against test accuracy using ResNet-50 and VGG-16 models on both MNIST and CIFAR-10 datasets. The proposed OC+OGP method consistently achieves the lowest cost for attaining a given test accuracy, demonstrating that the proposed method remains effective across different datasets, data distributions, and model architectures. It is also observed that the cost generally increases with the test accuracy, and the increase becomes faster as the test accuracy approaches convergence.

Fig. 4 illustrates the impact of varying uplink and downlink bandwidths on the total cost function. As the uplink or downlink bandwidth increases, the overall cost gradually decreases for all compared baselines, since larger bandwidth improves transmission efficiency and shortens the communication time. Moreover, the proposed OC+OGP consistently achieves the lowest cost under all evaluated bandwidth settings, indicating its effectiveness under different communication resource conditions. Fig. 5 illustrates the impact of the maximum allowable GPU frequency on the cost function. In Fig. 5(a), increasing the maximum allowable client-side GPU frequency continuously reduces the cost for all compared methods, which indicates that, within the tested range, enlarging the feasible client-side frequency set is beneficial for cost reduction. Fig. 5(b) shows that enlarging the maximum allowable server-side GPU frequency does not necessarily reduce the cost. For the proposed OC+OGP and OC+OG, the cost remains unchanged once the feasible range already includes the server-side GPU frequency that gives the lowest cost, since the proposed method chooses the cost-minimizing frequency instead of simply using the maximum available frequency. By contrast, the baseline schemes incur higher cost at large serverside frequency values because they operate at the maximum allowable GPU frequency.

![](images/f63d6822518752f605f728aeaf357bc6979751afe66ee8d88915600c60f84a5b.jpg)

![](images/c6f583d0ad1d9a5983e84d7210f505eb444e843128349ccd64184dc03f3bce77.jpg)  
(a) Cost function with test accuracy (MNIST, IID, ResNet-50).  
(b) Cost function with test accuracy (MNIST, non-IID, ResNet-50).

![](images/20137363964c5685d695359dec80dbb2e939eb1b600644d657d58e3fd9f311d9.jpg)

![](images/6c158773fc3f5eaa806879690609ec4c171f756a520d3e4dedfe5df443a80206.jpg)

![](images/4fd8281072c02ad9562f050aefc7acb263ba6b055ffc6ec19d2c4c8ca26d1a80.jpg)  
(e) Cost function with test accuracy (MNIST, IID, VGG-16).

(c) Cost function with test accuracy (CIFAR-10, IID, ResNet-50).  
(d) Cost function with test accuracy (CIFAR-10, non-IID, ResNet-50).  
![](images/36f8e79a611683f11c48cb6ae9e05e99b9f5a79be62faa4273dde8cf31fa4d0c.jpg)  
(f) Cost function with test accuracy (MNIST, non-IID, VGG-16).

![](images/28e9fb5cdd061b7bcf8411c2e4e490cdaa3e762771bb3be6209deb0288e53df2.jpg)  
(g) Cost function with test accuracy (CIFAR-10, IID, VGG-16).

![](images/5fb27ff6e556ce3ff5f722f3e08fbcd236a0c9085affcfaa5cda4e8b0f84554d.jpg)  
(h) Cost function with test accuracy (CIFAR-10, non-IID, VGG-16).  
Fig. 3. Evaluating training performance on CIFAR-10 and MNIST using ResNet-50 and VGG-16 (λ = 100).

TABLE II SIMULATION PARAMETERS
<table><tr><td rowspan=1 colspan=1>Parameter</td><td rowspan=1 colspan=1>Value</td><td rowspan=1 colspan=1>Parameter</td><td rowspan=1 colspan=1>Value</td></tr><tr><td rowspan=1 colspan=1> $C$ </td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1> $N _ { 0 }$ </td><td rowspan=1 colspan=1> $\overline { { 3 . 9 8 \times 1 0 ^ { - 2 1 } } }$  $\mathrm { W / H z }$ </td></tr><tr><td rowspan=1 colspan=1>J</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>λ</td><td rowspan=1 colspan=1>100</td></tr><tr><td rowspan=1 colspan=1> $n _ { i }$ </td><td rowspan=1 colspan=1> $\overline { { [ 0 . 2 , 3 ] \times 1 0 ^ { 4 } } }$  $\mathrm { \dot { F } L O P s / c y c l e }$ </td><td rowspan=1 colspan=1> $n _ { \mathrm { S } }$ </td><td rowspan=1 colspan=1> $\overline { { 3 \times 1 0 ^ { 4 } } }$  $\mathrm { F L O P s / c y c l e }$ </td></tr><tr><td rowspan=1 colspan=1> $G _ { i }$ </td><td rowspan=1 colspan=1> $\overline { { 1 \times 1 0 ^ { - 2 6 } } }$  $\mathrm { W / ( c y c l e / s ) ^ { 3 } }$ </td><td rowspan=1 colspan=1> $G _ { \mathrm { { S } } }$ </td><td rowspan=1 colspan=1> $\overline { { 2 \times 1 0 ^ { - 2 6 } } }$  $\mathrm { W / ( c y c l e / s ) ^ { 3 } }$ </td></tr><tr><td rowspan=1 colspan=1> $\overline { { f _ { i } ^ { \mathrm { m a x } } } }$ </td><td rowspan=1 colspan=1>625 MHz</td><td rowspan=1 colspan=1> $\overline { { f _ { i } ^ { \mathrm { m i n } } } }$ </td><td rowspan=1 colspan=1>50 MHz</td></tr><tr><td rowspan=1 colspan=1> $f _ { \mathrm { S } } ^ { \mathrm { m a x } }$ </td><td rowspan=1 colspan=1>3.135 GHz</td><td rowspan=1 colspan=1> $\overline { { f _ { \mathrm { S } } ^ { \mathrm { m i n } } } }$ </td><td rowspan=1 colspan=1>210 MHz</td></tr><tr><td rowspan=1 colspan=1> $\overline { { B _ { i } ^ { \mathrm { U } } } }$ </td><td rowspan=1 colspan=1>100 MHz</td><td rowspan=1 colspan=1> $\overline { { B _ { i } ^ { \mathrm { D } } } }$ </td><td rowspan=1 colspan=1>200 MHz</td></tr><tr><td rowspan=1 colspan=1> $\overline { { P _ { i } ^ { \mathrm { m a x } } } }$ </td><td rowspan=1 colspan=1>33 dBm</td><td rowspan=1 colspan=1> $\overline { { P _ { i } ^ { \mathrm { m i n } } } }$ </td><td rowspan=1 colspan=1>20 dBm</td></tr><tr><td rowspan=1 colspan=1> $\overline { { P _ { \mathrm { S } } ^ { \mathrm { m a x } } } }$ </td><td rowspan=1 colspan=1>33 dBm</td><td rowspan=1 colspan=1> $\overline { { P _ { \mathrm { S } } ^ { \mathrm { m i n } } } }$ </td><td rowspan=1 colspan=1>20 dBm</td></tr><tr><td rowspan=1 colspan=1> $\overline { { { | h _ { i } | } ^ { 2 } } }$ </td><td rowspan=1 colspan=1> $\overline { { [ 0 . 1 , 1 ] \times 1 0 ^ { - 6 } } }$ </td><td rowspan=1 colspan=1> $\overline { { { | h _ { \mathrm { S } } | } ^ { 2 } } }$ </td><td rowspan=1 colspan=1> $\overline { { 1 \times 1 0 ^ { - 6 } } }$ </td></tr></table>

![](images/347527487aee6754e8c4e69cba160138067a24288ba5f9b6eb8e4d23a6d8a0e0.jpg)  
(a) Cost function vs. uplink bandwidth.

![](images/f1598519b68d676896bbe5ef63a8b5a2cadced8e44a58586cb9274b3f9d9ec47.jpg)  
(b) Cost function vs. downlink bandwidth.  
Fig. 4. The impact of uplink bandwidth and downlink bandwidth on cost function.

![](images/79e1d986e43e853bbb0036d69074fc8bfc0517257fa056dee1bd1dff1a887acb.jpg)

![](images/be73b1e3768909c682aace5ea76aa6869481f980d72a25d4e82baced3e9a46a8.jpg)  
(a) Cost function vs. client-side GPU frequency.  
(b) Cost function vs. server-side GPU frequency.  
Fig. 5. The impact of GPU frequency on cost function.

In edge networks, computing resources and channel conditions may fluctuate during training, resulting in a mismatch between the measured parameters used for optimization and the actual operating conditions. To evaluate the impact of such uncertainty, we model the computing capabilities and average channel power gains as Gaussian random perturbations around their nominal values, with different coefficients of variation (CV) [43], [44]. Specifically, we consider uncertainty in four key parameters: n<sub>i</sub>, n<sub>S</sub>, $| h _ { i } | ^ { 2 } ,$ , and $| h _ { \mathrm { S } } | ^ { 2 }$ , where $n _ { i }$ and n<sub>S</sub> denote the numbers of GPU FLOPs per cycle of client i and the edge server, respectively, and $| h _ { i } | ^ { 2 }$ and $| h _ { \mathrm { S } } | ^ { 2 }$ denote the average uplink and downlink channel power gains, respectively. In our experiments, the CV is set to 10% and 20% to represent moderate and high uncertainty levels, respectively. For each CV level, we repeat the simulation over 200 random realizations. Fig. 6 shows that the objective value increases as the CV grows, due to the mismatch between the optimized parameters and the actual conditions. Nevertheless, the proposed method consistently maintains competitive performance, demonstrating good robustness under practical resource uncertainty.

![](images/283b0ddc1f9e7f64ccb9b04fe8444b27cedb32dd585e85104930283a31aef240.jpg)

![](images/e921ab190944b5a01de7de6d51234c7417187e30368a903f95b0d8d2343dd128.jpg)  
(a) Client-side computing uncer- (b) Server-side computing uncertainty. tainty.

![](images/10fb8f2cfd3d15baf481b4cacf7acd05b7b5f928097c9a6fe95efa6ddcae0af5.jpg)

![](images/9a83d69caea2d82621d698481469f832d061bf4f2eb2d48057f5614a3eb87789.jpg)  
(c) Uplink transmission uncertainty. (d) Downlink transmission uncertainty.

Fig. 6. The impact of network resource fluctuation on cost function.  
![](images/a1e3696aed473a7fbccbf3fb6f78d8a1e1a69645bf3236fbff687986ac602704.jpg)

![](images/3ac17509e2d73149525bddac1c53fb63d3e3fb2aad0ab27a3bb6f1a5ee95f110.jpg)  
(a) The impact of number of clients (b) The impact of number of clients on cost function. on running time.  
Fig. 7. The impact of number of clients on cost function and running time.

## C. Running Time Comparison

Fig. 7 illustrates the impact of the number of clients on both the cost function and the running time. As shown in Fig. 7(a), the cost function generally increases as the number of clients grows for all methods compared. The proposed method OC+OGP achieves a lower cost than baselines across all the numbers of clients tested. Moreover, the proposed OC method achieves the same cost as ESFL, while consistently outperforming DSQL. Fig. 7(b) further shows that the proposed efficient OC+OGP and OC methods require lower running time compared with ESFL and DSQL. To sum up, the results verify the theoretical analysis that the proposed OC attains the optimal solution with polynomial-time complexity. In addition, it shows that OC+OGP remains polynomial-time as the number of clients increases, which indicates that the proposed method is able to obtain high-quality approximate solutions efficiently.

![](images/2873562de9a287aaaf93961c4e98d790324ce6a2b24e98aac5b1be8c881efdc9.jpg)  
(a) Energy minimization $( \lambda = 0 )$

![](images/aa37bb22553209c42280fb0f5828a6901563f9b2a8fe95f9efd46a4f9aac39ed.jpg)  
(b) Latency minimization $( \lambda = \infty )$  
Fig. 8. Performance comparison under λ = 0 and $\lambda = \infty$

## D. Impact of Hyperparameter λ

Fig. 8 further examines the two extreme cases. When $\lambda \ = \ 0 ,$ , i.e., under energy-only optimization, the proposed OC+OGP achieves the lowest energy among all compared methods. When $\lambda = \infty$ , i.e., under latency-only optimization, OC+OGP achieves nearly the same latency as OC+OG, OC+OP, OC, and ESFL, while remaining lower than DSQL. This is because the latency-only objective drives the optimized GPU frequencies and transmit powers to their maximum feasible values, so the additional resource-control degrees of freedom in OC+OGP, OC+OG, and OC+OP do not further reduce latency beyond the cut-layer optimization. These results demonstrate that the proposed method can adapt to different optimization preferences and maintain favorable performance under both energy-oriented and latency-oriented settings.

## VIII. CONCLUSION

In this paper, we investigated the joint optimization of model splitting and resource allocation for SFL in resourceconstrained edge systems. We formulated the resulting design as a hybrid optimization problem that captures the coupled effects of computation, communication, and model-splitting decisions. For the special model-splitting case, we developed an optimal polynomial-time solution. For the general joint optimization problem, we reformulated it as a two-dimensional master problem and developed a computationally efficient approximation method with explicit theoretical guarantees. Numerical results demonstrated the effectiveness of the proposed approach and its advantage over existing baselines. Overall, the proposed framework provides an efficient method for resource optimization in SFL under the energy–latency tradeoff.

## REFERENCES

[1] W. Wu, M. Li, K. Qu, C. Zhou, X. Shen, W. Zhuang, X. Li, and W. Shi, “Split learning over wireless networks: Parallel design and resource management,” IEEE Journal on Selected Areas in Communications, vol. 41, no. 4, pp. 1051–1066, Feb. 2023.

[2] W. Fan, P. Chen, X. Chun, and Y. Liu, “MADRL-based model partitioning, aggregation control, and resource allocation for cloud-edge-device collaborative split federated learning,” IEEE Transactions on Mobile Computing, vol. 24, no. 6, pp. 5324–5341, May 2025.

[3] W. Wei, Z. Lin, T. Li, X. Li, and X. Chen, “Pipelining split learning in multi-hop edge networks,” IEEE Transactions on Mobile Computing, pp. 1–14, 2026.

[4] Z. Wang, K. Huang, and Y. C. Eldar, “Spectrum breathing: Protecting over-the-air federated learning against interference,” IEEE Trans. Wireless Commun., vol. 23, no. 8, pp. 10 058–10 071, 2024.

[5] G. Qu, Q. Chen, W. Wei, Z. Lin, X. Chen, and K. Huang, “Mobile edge intelligence for large language models: A contemporary survey,” IEEE Communications Surveys & Tutorials, vol. 27, no. 6, pp. 3820–3860, Dec. 2025.

[6] NVIDIA, “NVIDIA Jetson Xavier,” 2019. [Online]. Available: https://www.nvidia.com/en-us/autonomous-machines/ embedded-systems/jetson-xavier-series/

[7] P. Vepakomma, O. Gupta, T. Swedish, and R. Raskar, “Split Learning for Health: Distributed Deep Learning Without Sharing Raw Patient Data,” in ICLR workshop on AIfor social good, New Orleans, LA, USA, Apr. 2019, pp. 1–7.

[8] Z. Lin, G. Qu, X. Chen, and K. Huang, “Split Learning in 6G Edge Networks,” IEEE Wireless Communications, vol. 31, no. 4, pp. 170– 176, Aug. 2024.

[9] Z. Yao, J. Qi, Y. Xu, Y. Liao, H. Xu, and L. Wang, “Pairingfl: Efficient federated learning with model splitting and client pairing,” IEEE Transactions on Networking, vol. 33, no. 4, pp. 1811–1825, May 2025.

[10] T. Li, A. K. Sahu, A. Talwalkar, and V. Smith, “Federated learning: Challenges, methods, and future directions,” IEEE Signal Processing Magazine, vol. 37, no. 3, pp. 50–60, May 2020.

[11] J. Konecnˇ y, “Federated learning: Strategies for improving communica-\` tion efficiency,” arXiv preprint arXiv:1610.05492, 2016.

[12] G. Lan, X.-Y. Liu, Y. Zhang, and X. Wang, “Communication-efficient federated learning for resource-constrained edge devices,” IEEE Transactions on Machine Learning in Communications and Networking, vol. 1, pp. 210–224, Aug. 2023.

[13] Z. Lin, G. Qu, W. Wei, X. Chen, and K. K. Leung, “AdaptSFL: Adaptive split federated learning in resource-constrained edge networks,” IEEE Transactions on Networking, vol. 33, no. 6, pp. 2993–3008, Jun. 2025.

[14] Z. Lin, W. Wei, Z. Chen, C.-T. Lam, X. Chen, Y. Gao, and J. Luo, “Hierarchical split federated learning: Convergence analysis and system optimization,” IEEE Transactions on Mobile Computing, vol. 24, no. 10, pp. 9352–9367, Apr. 2025.

[15] W. Wei, Z. Lin, X. Liu, H. Du, D. Niyato, and X. Chen, “Optimizing split federated learning with unstable client participation,” IEEE Transactions on Mobile Computing, pp. 1–15, Mar. 2026.

[16] G. Qu, T. Li, Q. Chen, X. Chen, and S. Zhou, “Slide: Simultaneous model downloading and inference at the wireless network edge,” IEEE Transactions on Mobile Computing, pp. 1–12, 2026.

[17] G. Zhu, Y. Deng, X. Chen, H. Zhang, Y. Fang, and T. F. Wong, “Esfl: Efficient split federated learning over resource-constrained heterogeneous wireless devices,” IEEE Internet of Things Journal, vol. 11, no. 16, pp. 27 153–27 166, Aug. 2024.

[18] W. Wei, J. Wang, J. Du, Z. Fang, C. Jiang, and Y. Ren, “Underwater differential game: Finite-time target hunting task with communication delay,” in IEEE International Conference on Communications (ICC), Seoul, Korea, May, 2022, pp. 3989–3994.

[19] W. Wei, J. Wang, J. Du, Z. Fang, Y. Ren, and C. L. P. Chen, “Differential game-based deep reinforcement learning in underwater target hunting task,” IEEE Transactions on Neural Networks and Learning Systems, vol. 36, no. 1, pp. 462–474, Jan., 2025.

[20] W. Wei, J. Wang, Z. Fang, J. Chen, Y. Ren, and Y. Dong, “3U: Joint design of UAV-USV-UUV networks for cooperative target hunting,” IEEE Transactions on Vehicular Technology, vol. 72, no. 3, pp. 4085– 4090, Mar. 2023.

[21] C. Guo, F. Zhou, L. Feng, and W. Li, “Hierarchical multiple split federated learning for low-carbon resource-constrained user equipment,” IEEE Internet ofThings Journal, vol. 11, no. 24, pp. 39 127–39 141, Dec. 2024.

[22] C. Meng, M. Tang, and V. W. S. Wong, “ASFL: An adaptive model splitting and resource allocation framework for split federated learning,” arXiv preprint arXiv:2603.04437, 2026.

[23] M. Wu, R. Yang, X. Huang, Y. Wu, J. Kang, and S. Xie, “Joint optimization of model partition and resource allocation for split federated learning over vehicular edge networks,” IEEE Transactions on Vehicular Technology, vol. 73, no. 10, pp. 15 860–15 865, Oct. 2024.

[24] X. Wang, S. Song, Z. Zhang, X. Hou, Z. Li, T. Xing, and X.-P. Zhang, “Split federated learning for resource-constrained edge computing networks,” IEEE Transactions on Consumer Electronics, vol. 71, no. 4, pp. 11.001–11.013, Nov, 2025.

[25] Z. Wang, Z. Zhang, J. Wang, C. Jiang, W. Wei, and Y. Ren, “AUVassisted node repair for iout relying on multiagent reinforcement learning,” IEEE Internet of Things Journal, vol. 11, no. 3, pp. 4139–4151, 2024.

[26] Q. Chen, X. Chen, and K. Huang, “SiftMoE: Similarity-aware energyefficient expert selection for wireless distributed moe inference,” arXiv preprint arXiv:2603.23888, 2026.

[27] R. Drechsler, C. A. Metz, and C. Plump, “Energy-efficient CNN inferencing on GPUs with dynamic frequency scaling,” in International Conference on Innovations in Data Analytics, Kolkata, India, Nov. 2023, pp. 375–389.

[28] P. J. Maliakel, S. Ilager, and I. Brandic, “Characterizing LLM inference energy-performance tradeoffs across workloads and GPU scaling,” arXiv preprint arXiv:2501.08219, Feb. 2026.

[29] A. K. Kakolyris, D. Masouros, P. Vavaroutsos, S. Xydis, and D. Soudris, “throttLL’eM: Predictive GPU throttling for energy efficient LLM inference serving,” in 2025 IEEE International Symposium on High Performance Computer Architecture (HPCA), Las Vegas, NV, USA, Mar. 2025, pp. 1363–1378.

[30] J. You, J.-W. Chung, and M. Chowdhury, “Zeus: Understanding and optimizing GPU energy consumption of DNN training,” in Proceedings of the USENIX Symposium on Networked Systems Design and Implementation, Boston, MA, USA, Apr. 2023, p. 119 – 139.

[31] S. Choi, I. Koo, J. Ahn, M. Jeon, and Y. Kwon, “ENVPIPE: Performance-preserving DNN training framework for saving energy,” in Proceedings of the USENIX Annual Technical Conference, Boston, MA, USA, Jul. 2023, p. 851 – 864.

[32] Z. Jia, L. N. Bhuyan, and D. Wong, “PCCL: Energy-efficient LLM training with power-aware collective communication,” in 2024 IEEE 42nd International Conference on Computer Design (ICCD), Milan, Italy, Nov. 2024, pp. 84–91.

[33] J.-W. Chung, Y. Gu, I. Jang, L. Meng, N. Bansal, and M. Chowdhury, “Reducing energy bloat in large model training,” in Proceedings of the ACM Symposium on Operating Systems Principles, New York, NY, USA, Nov. 2024, p. 144–159.

[34] Q. Zeng, Y. Du, K. Huang, and K. K. Leung, “Energy-efficient resource management for federated edge learning with CPU-GPU heterogeneous computing,” IEEE Transactions on Wireless Communications, vol. 20, no. 12, pp. 7947–7962, Dec. 2021.

[35] A. E. Bryson, Applied optimal control: optimization, estimation and control. Routledge, 2018.

[36] K. Gokbayrak and C. Cassandras, “A hierarchical decomposition method for optimal control of hybrid systems,” in Proceedings of IEEE Conference on Decision and Control, Sydney, NSW, Australia, Dec. 2000, pp. 1816–1821.

[37] L. Deng, “The MNIST database of handwritten digit images for machine learning research,” IEEE Signal Processing Magazine, vol. 29, no. 6, pp. 141–142, Nov. 2012.

[38] Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to document recognition,” Proc. IEEE, vol. 86, no. 11, pp. 2278– 2324, Nov. 1998.

[39] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” arXiv preprint arXiv:1512.03385, 2015.

[40] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” arXiv preprint arXiv:1409.1556, Apr. 2015.

[41] NVIDIA, “Solving entry-level edge AI challenges with NVIDIA Jetson Orin Nano,” https://developer.nvidia.com/blog/ solving-entry-level-edge-ai-challenges-with-nvidia-jetson-orin-nano/, accessed: 2026-05-07.

[42] ——, “NVIDIA system management interface (nvidia-smi),” https:// docs.nvidia.com/deploy/nvidia-smi/, accessed: 2026-05-07.

[43] W. J. Robinson M., F. Esposito, and M. A. Zuluaga, “DTS: A simulator to estimate the training time of distributed deep neural networks,” in International Symposium on Modeling, Analysis, and Simulation of

Computer and Telecommunication Systems, Nice, France, Mar. 2023, pp. 17–24.

[44] Y. Yoo and S. Jung, “Modeling forecast errors for microgrid operation

using Gaussian process regression,” Scientific Reports, vol. 14, no. 1, p. 2166, Jan. 2024.