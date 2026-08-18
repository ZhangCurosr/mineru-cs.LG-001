# Pallas: A Proactive KV Cache Migration Framework for LLM Inference in AI-RAN

Tianhang Ding, Jianchun Liu, Hongli Xu University of Science and Technology of China

## Abstract

AI-RAN brings large language model (LLM) serving close to mobile users, but cellular handover can separate an active request from its inference state: the user attaches to a target base station (gNB) while the large and growing key-value (KV) cache remains at the source. Retaining inference at the source preserves service continuity but persistently increases inter-token latency (ITL), whereas recovering the state at the target restores serving locality but requires KV-cache transfer, recomputation, or a combination of both only after handover, directly prolonging service interruption time (SIT).

This work presents Pallas, a proactive KV-cache migration framework that prepares the inference state at the predicted target before handover, in parallel with ongoing source-side inference and token delivery. At the preparation trigger, Pallas partitions the token sequence into a stable historical prefix and an evolving sufix. The target reconstructs the prefix through local prefill, while the source streams the KV blocks generated for the sufix. At handover, the target assembles both portions into an up-to-date KV cache and resumes decoding locally, leaving only unfinished preparation to contribute to SIT. An online scheduler selects the prefetching window, which determines how early preparation begins before handover, based on mobility predictions and runtime telemetry. Across three LLMs and 100–500 Mbps inter-gNB links, our vLLM-based prototype reduces average SIT by factors of 2.28–89.68 over target-side recovery approaches and lowers average ITL by 16.0%–50.0% compared with source-side forwarding.

## Keywords

AI-RAN, LLM Serving, Handover, Low-Latency, Proactive KV-Cache Migration.

## 1 Introduction

Radio Access Networks (RANs) are evolving from connectivity infrastructure into distributed computing platforms, with AI-RAN colocating accelerators and RAN functions at or near base stations (gNBs) [1–3]. This proximity shortens the communication path to mobile users, making AI-RAN attractive for latency-sensitive large language model (LLM) services that continuously stream generated tokens [4, 5]. However, user mobility disrupts this serving model: after handover, the user equipment (UE) attaches to a target gNB while its service state remains at the source gNB [6, 7]. For LLM inference, this state includes an ever-growing KV-cache required to generate subsequent tokens [8]. Mobility can therefore separate an ongoing request from the state needed to continue it, undermining the latency advantage of AI-RAN.

Restoring LLM serving locality after this separation is challenging because the inference state is both large and continuously evolving. During autoregressive decoding, the KV cache stores the key and value tensors of every processed token across all transformer layers and grows linearly with context length [8, 9]. For Qwen3-32B [10], an FP16 or BF16 KV cache grows by approximately 256 KiB per token<sup>1</sup>, reaching approximately 1 GiB at 4,000 tokens. At a nominal inter-gNB bandwidth of 300 Mbps, transmitting the payload alone takes approximately 28 seconds. Thus, handover creates a pronounced timescale mismatch, i.e., the UE’s network attachment can change rapidly, whereas relocating the inference state required to continue its inference session may take orders of magnitude longer.

This timescale mismatch creates a fundamental tension between service continuity and serving locality. To avoid relocating the KV cache, Detour [11] retains inference at the source gNB and forwards the generated tokens to the UE through the target gNB after handover. Although this approach keeps the accumulated inference state available at the source and therefore avoids LLM-state recovery, it introduces additional routing hops and persistently degrades inter-token latency (ITL). As shown in Figure 1, additional forwarding hops progressively shift the ITL distribution toward higher latency and substantially worsen its tail. Under the three-hop setting with Qwen3-32B, the P90 and P99 ITLs reach 126.68 ms and 263.54 ms, respectively, exceeding the 100 ms latency objective [12]. Detour therefore preserves service continuity only by sacrificing serving locality and sustained token-delivery latency.

In contrast, restoring serving locality requires making the inference state available at the target gNB. A straightforward approach, Full-Copy [13], transfers the complete KV cache, following the basic mechanism of live KV-cache migration in data-center LLM serving systems. However, over bandwidth-constrained inter-gNB links, the target cannot resume decoding until the entire, potentially multi-gigabyte KV cache has been transferred, placing the full transfer delay on the post-handover critical path and preventing the UE from receiving any newly generated tokens during this interval. Recomputation [14] avoids transferring the cache itself by sending the much smaller token sequence and rebuilding the cache through target-side prefill. Nevertheless, the target must replay the entire accumulated context before it can generate even the first post-handover token, causing the service interruption to grow with context length and consuming substantial target-side GPU computation precisely when the request needs to resume.

The most closely related approach, ctHO [15], combines target-side token prefill with direct KV-cache transfer. Specifically, the target reconstructs a prefix of the accumulated KV cache from historical tokens, while the source transfers the remaining KV-cache sufix. By jointly selecting this prefix–sufix partition and allocating backhaul bandwidth across multiple UEs undergoing handover, ctHO balances the completion times of target-side prefix reconstruction and source-to-target sufix transfer, thereby minimizing the worst-user handover delay. However, ctHO assumes that token streaming is suspended at handover and places both target-side prefill and sufix transfer after that point. Consequently, even with an optimized prefix–sufix partition, target-side decoding must wait for both recovery paths to finish, and the slower path directly determines the resulting service interruption time (SIT). In conclusion, Full-Copy, Recomputation, and ctHO explore diferent combinations of communication and computation, but share the same temporal structure, i.e., state recovery begins only after handover.

This shared limitation motivates treating handover as a deadline for target-side state preparation rather than as the trigger for recovery. Prior studies have shown that radio measurements, mobility trajectories, and association histories can predict the likely target gNB and approximate handover time [6, 7, 16]. Such advance estimates allow the target to prepare the inference state before handover while the UE remains connected to the source and continues receiving generated tokens. However, exploiting this opportunity is not as simple as prefetching a static object: the KV cache grows with every token generated at the source, so the target must prepare the historical state while continuously incorporating newly generated state.

To this end, we propose Pallas, a proactive KV-cache migration framework for low-latency LLM serving in AI-RAN. At the preparation trigger, Pallas marks the block-aligned split position as a split point: all tokens processed by that time form a stable prefix, while tokens generated afterward form an evolving sufix. The target gNB receives the compact token sequence of the stable prefix and reconstructs its KV cache through local prefill. In parallel, the source gNB continues generating and delivering tokens to the UE while streaming the KV blocks of the evolving sufix to the target. The target then combines the reconstructed prefix with the received sufix to obtain a complete, up-to-date KV cache. After handover, local decoding can resume as soon as both portions are available, so only unfinished preparation contributes to SIT. In essence, Pallas reconstructs what is already stable and transfers only what continues to change.

We define the interval between the preparation trigger and the predicted handover as the prefetching window. A short window leaves prefix reconstruction or sufix synchronization unfinished, causing the remaining work to contribute directly to SIT. Conversely, an excessively long window establishes an earlier split point, producing a larger incremental sufix, significantly increasing KV-cache transfer volume. Importantly, finding a proper window is nontrivial because the trigger time jointly determines the sizes of the historical prefix and evolving sufix, whose completion times depend on the context length, target-side prefill throughput, sourceside decode throughput, and inter-gNB goodput. Because these factors difer across requests and change over time, no fixed window can consistently balance service interruption against preparation overhead. Mobility-prediction updates further complicate this decision, i.e., an updated handover time changes the remaining preparation time, while a new target prediction can invalidate the state already prepared.

The main contributions of this paper are as follows.

• We reveal the continuity–locality trade-of in mobile LLM handover, showing that existing strategies incur either persistent detour latency or post-handover staterecovery delay, which motivates treating handover as a preparation deadline rather than a recovery trigger.

• We design Pallas, a proactive KV-cache migration framework that reconstructs the stable historical prefix at the target in parallel with source-side decoding and incremental sufix streaming, thereby moving most state-recovery work before handover.

• We formulate prefetching-window selection based on prefill/decoding rates and inter-gNB goodput, and develop an online scheduler that adapts the trigger to changing system conditions and mobility predictions while bounding stale-preparation duration.

• We implement Pallas on vLLM 0.8.5 and deploy it across two physical servers, each equipped with a single NVIDIA RTX A6000 GPU and connected by 1-Gbps Ethernet. Prototype experiments and tracedriven repeated-handover simulations across three

LLMs and diverse inter-gNB conditions show that Pallas substantially reduces SIT relative to baselines while avoiding the persistent ITL overhead.

## 2 Background and Motivation

## 2.1 LLM Serving under AI-RAN Handover

We consider an AI-RAN system in which GPU-equipped edge servers are colocated with gNBs to serve mobile UEs. During an active LLM session, the UE is connected to a serving gNB over the radio access link, while the colocated inference engine executes the request and returns generated tokens directly to the UE. This proximity shortens the service path and reduces communication latency [2, 3, 17]. When the UE moves across a cell boundary, the RAN performs a handover that redirects its network attachment from the current serving gNB (or source) to a new serving gNB (or target) [6]. Handover updates the UE’s access path and maintains network connectivity, but does not automatically relocate the application-layer state of an ongoing request [7]. We assume that the source and target can execute the same LLM configuration, and our focus is the per-request inference state that must be preserved across handover.

An LLM request proceeds through two inference phases: prefilling and decoding [4]. During prefilling, the inference engine processes the input prompt and constructs the corresponding key-value (KV) cache. It then enters autoregressive decoding, where each new token is generated using the cached key and value tensors of all preceding tokens. The KV cache avoids repeatedly processing the complete token history, but grows whenever a new token is generated [8, 9]. Consequently, it is both indispensable for continuing the request and continuously evolving throughout the decoding process. For an LLM with � transformer layers, �<sub>KV</sub> KV heads, head dimension $d _ { h }$ , and � bytes per cache element, the per-token KV-cache footprint is

$$
s _ { \mathrm { K V } } = 2 \cdot L \cdot H _ { \mathrm { K V } } \cdot d _ { h } \cdot b , \qquad S _ { \mathrm { K V } } ( N ) = N \cdot s _ { \mathrm { K V } } ,\tag{1}
$$

where the factor of two accounts for the key and value tensors, and $S _ { \mathrm { K V } } ( N )$ denotes the cache size after � tokens.

During decoding, this request-specific state resides in the GPU memory of the source gNB. After handover, however, the UE communicates through the target gNB while its accumulated KV cache remains at the source. The target therefore lacks the state required to immediately continue decoding, even though the UE’s network attachment has already been updated. Continuing the session requires either retaining inference at the source and forwarding subsequent tokens through the target, or making the inference state available at the target through KV-cache transfer, recomputation, or a combination of both. The next subsection characterizes the limitations of these alternatives.

![](images/32e1ce73ba2245825c5e3e6ca4c933b1a9f36015c44b1d9e1ceea7c018c40847.jpg)

![](images/3052be1533eac8b73f4b779705f387375487c220dce38354b56d21fa263f314b.jpg)  
Figure 1: CDF of ITL under the Detour strategy. Increased routing hops lead to severe ITL degradation. The vertical red dashed line at 100 ms marks the SLO.

## 2.2 Limitations of Existing Strategies

To quantify the limitations of existing methods, we conduct a set of preliminary experiments that examine the latency overhead of continuing inference at the source gNB and the service interruption caused by KV-cache recovery at the target gNB.

Source-side retention. Detour [11] retains inference at the source gNB after handover and uses the target gNB as a relay for every subsequently generated token. Although this strategy avoids recovering the KV cache at the target, it lengthens the serving path throughout the remainder of the request. To quantify this persistent cost, we measure ITL while varying the number offorwarding hops between the inference engine and the UE. As shown in Figure 1, the ITL distributions of both Llama-3-8B and Qwen3-32B progressively shift toward higher latency and develop substantially longer tails as the routing path grows. Under the three-hop setting, their P90 ITLs reach 130.54 ms and 126.68 ms, respectively; the P99 ITL of Qwen3-32B further reaches 263.54 ms. Following the conversational TPOT bound adopted by MLPerf [12], we use 100 ms as the token-latency service level objective (SLO) throughout the evaluation. The fraction of individual ITL samples exceeding this threshold increases from 1.74% and 1.93% under two hops to 12.61% and 12.94% under three hops for Llama-3-8B and Qwen3-32B, respectively. Unlike a one-time migration cost, this path-induced penalty afects every token generated after handover.

Observation 1. Detour avoids target-side state recovery by converting it into a persistent token-delivery penalty that worsens with routing distance.

Target-side recovery. Restoring serving locality requires making the accumulated KV cache available at the target before local decoding can resume. We next characterize the communication and computation costs of representative target-side recovery strategies. Full-Copy [13] transfers the complete KV cache from the source to the target. Its recovery time therefore grows with cache size and is inversely proportional to the inter-gNB goodput available to the migration. As shown in Table 1, at 300 Mbps inter-gNB goodput, the

Table 1: Target-side recovery time (ms) of representative recovery strategies with Qwen3-32B at 300 Mbps inter-gNB goodput.
<table><tr><td>Strategy</td><td>1K</td><td>2K</td><td>4K</td></tr><tr><td>Full-Copy</td><td>7165</td><td>14310</td><td>29066</td></tr><tr><td>Recomputation</td><td>581</td><td>1208</td><td>2103</td></tr><tr><td>ctHO</td><td>554</td><td>1119</td><td>2058</td></tr></table>

SIT of Full-Copy increases from 7.17 s at a 1K-token context to 29.07 s at 4K tokens. Because transfer starts only after handover, the target cannot resume decoding throughout this interval.

Recomputation [14] instead sends the compact token sequence and reconstructs the KV cache through target-side prefill. This avoids transferring the cache itself but places the entire prefill computation on the interruption path. Table 1 shows that the SIT of Recomputation increases from 581 ms at a 1K-token context to 2,103 ms at 4K tokens, corresponding to an approximately 3.6× increase as the context grows. Thus, although Recomputation replaces KV-cache transfer with local computation, its increasing recomputation delay remains entirely on the post-handover interruption path before the UE can receive the first token from the target.

The most closely related strategy, ctHO [15], combines the two recovery operations: the target reconstructs a KV-cache prefix through token prefill, while the source transfers the remaining sufix. By optimizing the prefix–sufix partition, ctHO balances the completion times of recomputation and transfer and reduces handover delay relative to using either operation alone. Nevertheless, Table 1 shows that the SIT of ctHO still reaches 2,058 ms at a 4K-token context, despite its optimized prefix–sufix partition. Both operations are placed after handover, and decoding cannot resume until they have both completed. Consequently, the entire recovery interval remains exposed as SIT.

Observation 2. Full-Copy, Recomputation, and ctHO explore diferent combinations of communication and computation, but share the same reactive timing, i.e., their staterecovery work begins only after handover and therefore directly contributes to SIT.

## 2.3 Opportunities for System Design

The reactive timing identified in § 2.2 suggests a temporal opportunity: instead of treating handover as the point at which recovery begins, the system can use it as a deadline by which state preparation should be largely completed. Prior studies have shown that radio measurements, mobility trajectories, and association histories can be used to predict both the next serving cell and the handover time [6, 7, 16]. We denote the predicted target gNB and the predicted handover time by $\hat { g } _ { \mathrm { t a r } }$ and �ˆ<sub>HO</sub>, respectively. If preparation is triggered at time $t _ { \mathrm { t r i g } } ,$ the resulting prefetching window is

![](images/1f5b0fdaab28b6d7c5bf0c5d47b03c7c4ef7822527a182a7d280f6551337c91e.jpg)  
Figure 2: Prefetching-window trade-of under a representative operating condition (300 Mbps inter-gNB goodput, Qwen3-14B, and a 3,000-token context).

$$
T _ { w } = \hat { t } _ { \mathrm { H O } } - t _ { \mathrm { t r i g } } .\tag{2}
$$

During this interval, the UE remains connected to the source, which continues generating and delivering tokens, while the predicted target prepares the inference state in parallel.

Preparing an evolving inference state. Proactive preparation is more dificult than prefetching a static object because the KV cache continues to grow while the source serves the UE. At the trigger time, the accumulated token sequence forms a stable prefix whose KV cache can be reconstructed at the target through prefill. Tokens generated afterward form an evolving sufix whose KV blocks must be continuously incorporated into the target state. Let $r _ { \mathrm { d } }$ denote the per-request decode throughput at the source gNB. The incremental KV-cache volume produced over a window of length $T _ { w }$ is approximately

$$
S _ { \mathrm { i n c } } ( T _ { w } ) \approx r _ { \mathrm { d } } \cdot T _ { w } \cdot s _ { \mathrm { K V } } ,\tag{3}
$$

A one-time snapshot becomes stale as decoding proceeds, whereas pausing the source to preserve a static state would itself interrupt token delivery. A viable proactive design thus reconstructs the stable history while tracking newly generated state without suspending source-side inference.

Selecting the prefetching window. The trigger time determines both how much preparation is completed before handover and how early the target-side state is prepared. To illustrate this trade-of, we vary $T _ { w }$ under a representative setting of 300 Mbps inter-gNB goodput, Qwen3-14B, and a 3,000-token context. As shown in Figure 2, increasing $T _ { w }$ from 0 to about 1 s reduces SIT by approximately 77%, from 1.1 s to 0.25 s, while the target-side idle time remains negligible. Within an intermediate window range, both SIT and idle time remain low. Beyond this range, further increasing $T _ { w }$ provides little additional SIT reduction but causes the idle time to grow rapidly, reaching approximately 2.4 s at $T _ { w } = 4 \ : s ,$ equivalent to 60% of the entire prefetching window. This illustrates the two-sided trade-of: a short window leaves residual preparation on the handover critical path, whereas a long window triggers preparation unnecessarily early. Hence, efective proactive migration requires selecting a proper prefetching window that balances residual service interruption against early target-side idle time.

![](images/3cf11c39ffff171316bb6e0f57e26e826ea698f9c85aae40eb5e537d4484540c.jpg)  
Figure 3: The workflow of Pallas.

## 3 System Overview of Pallas

To exploit the aforementioned opportunities, we propose Pallas, which proactively moves inference-state preparation into the prefetching window and executes it in parallel with ongoing source-side inference. Figure 3 provides an overview ofits online decision and parallel state-preparation workflow. At each control cycle, the Scheduler Engine of Pallas combines mobility predictions with the current context length and runtime estimates of target-side prefill throughput, source-side per-request decode throughput, and inter-gNB goodput. It first verifies that the predicted target can execute a compatible LLM instance and has suficient available VRAM. If the predicted target is infeasible, Pallas invokes a fallback policy that temporarily retains source-side service or redirects preparation to a feasible serving tier. If the target is feasible, the Scheduler Engine evaluates candidate prefetching windows and selects the window with the lowest estimated cost. If the predicted remaining time to handover is still longer than the selected window, migration is deferred until the next control cycle. Otherwise, Pallas triggers parallel state preparation at the predicted target. Once triggered, the source records the block-aligned split position as the prefix– sufix split and sends the exact historical-prefix token IDs to the target, which reconstructs the corresponding KV cache through local prefill. Meanwhile, the source remains the only active decoder, continues generating and delivering tokens to the UE, and asynchronously streams newly completed sufix KV-cache blocks to the target. Thus, historical-prefix reconstruction, source-side decoding, and sufix transfer proceed in parallel during the prefetching window. At handover, the source records the final sequence position and sends the final token-sequence snapshot together with any unsynchronized KV-cache blocks. Once prefix reconstruction is complete and the state for all sequence positions is available, the target combines the reconstructed prefix with the received sufix and activates the request for local decoding. Before preparation begins, updated mobility predictions and runtime telemetry are incorporated into subsequent window decisions. After preparation begins, a revised handover time changes the preparation deadline without invalidating the existing state, whereas a change in the predicted target cancels the obsolete preparation and triggers a new admission and window decision.

![](images/7152f729594559e39a395ab16d9f1069ee6c9a70a98f5f9dc2db00904210e682.jpg)  
Figure 4: KV-cache recovery timelines of Full-Copy, Recomputation, ctHO, and Pallas during handover.

Figure 4 further illustrates how Pallas difers from existing KV-cache recovery strategies in terms of execution timing. As shown in Figure 4(a), Full-Copy postpones the entire KV-cache transfer until after handover, placing the full transfer delay on the post-handover critical path. Recomputation replaces KV-cache transfer with target-side historical KV-cache reconstruction, but Figure 4(b) shows that the complete recomputation still remains on the posthandover path. ctHO further overlaps prefix reconstruction with sufix transfer, as illustrated in Figure 4(c), thereby reducing the recovery delay to the completion time of the slower operation. However, both operations begin only after handover, leaving target-side decoding blocked by state recovery. In contrast, Pallas moves state preparation ahead of handover, as shown in Figure 4(d). Historical KV-cache reconstruction at the target and incremental KV-cache transfer from the source proceed within the prefetching window in parallel with ongoing source-side inference. As a result, most state-preparation work can be completed before handover.

## 4 System Design of Pallas

## 4.1 Prefetching-Window Trade-of Model

As discussed in § 2.3, the prefetching-window length $T _ { w }$ controls two opposing efects. A short window may leave state-recovery work unfinished at handover, whereas an excessively long window initiates preparation earlier than necessary and keeps the reconstructed state at the target for longer. We make this trade-of explicit by modeling the expected computation and communication progress within a candidate window.

Historical-prefix reconstruction. At the preparation trigger, Pallas splits the token sequence at the block-aligned split position. All tokens processed by that time form a stable historical prefix, while subsequently generated tokens form an evolving sufix. Let $L _ { \mathrm { h i s t } }$ denote the number of tokens in this prefix, and let $r _ { p }$ denote the expected prefill throughput available to the migration session at the target gNB. Unlike the standalone peak throughput, $r _ { p }$ accounts for the computation capacity expected to be available under the current target-side load. For an isolated migration, it corresponds to the target’s measured prefill throughput, whereas under resource contention it reflects the efective throughput expected to be available to the migration session. Let $T _ { \mathrm { l o a d } }$ denote the latency of loading the model weights from host memory to the target GPU [18]. The target-side time required to reconstruct the historical KV cache is:

$$
T _ { \mathrm { h i s t } } = T _ { \mathrm { l o a d } } + \frac { L _ { \mathrm { h i s t } } } { r _ { \mathrm { p } } } .\tag{4}
$$

If the required model is already resident on the target GPU, as in our prototype evaluation, then $T _ { \mathrm { l o a d } } = 0$

Incremental-sufix synchronization. While the target reconstructs the historical prefix, the source continues decoding and streams newly completed KV-cache blocks to the target. As defined in Eq. (3), the resulting incremental KV-cache volume over a window of length $T _ { w }$ is $S _ { \mathrm { i n c } } ( T _ { w } )$ Let � denote the expected inter-gNB goodput available to the migration session. Because the incremental KV cache is streamed while it is generated, only the portion that cannot be delivered within the prefetching window remains on the handover critical path. The corresponding residual transmission delay is

$$
T _ { \mathrm { r e s } } ( T _ { w } ) = \operatorname* { m a x } \left\{ 0 , \frac { S _ { \mathrm { i n c } } ( T _ { w } ) } { B } - T _ { w } \right\} .\tag{5}
$$

Here, � is the efective goodput observed by the migration session and therefore accounts for its available link capacity and protocol overhead.

Service interruption. Target-side decoding can resume only after both the historical KV cache and the incremental sufix have become available. Because historical-prefix reconstruction and sufix transmission proceed in parallel, the slower unfinished operation determines the migrationinduced delay after handover. Let $T _ { 0 }$ denote the windowindependent activation latency after the required state becomes available, including final state assembly, request activation, and the first target-side decoding step. The resulting service interruption is

$$
T _ { \mathrm { S I T } } ( T _ { w } ) = T _ { 0 } + \operatorname* { m a x } \left\{ 0 , T _ { \mathrm { h i s t } } - T _ { w } , T _ { \mathrm { r e s } } ( T _ { w } ) \right\} .\tag{6}
$$

Thus, if both preparation paths complete before handover, the resulting service interruption is reduced to $T _ { 0 }$

Early-preparation exposure. A longer window may allow historical recomputation to finish well before handover. During this interval, the reconstructed historical KV cache remains resident at the target even though the UE is still served by the source. We quantify this early-preparation exposure as

$$
T _ { \mathrm { e a r l y } } ( T _ { w } ) = \operatorname* { m a x } \left\{ 0 , T _ { w } - T _ { \mathrm { h i s t } } \right\} .\tag{7}
$$

$T _ { \mathrm { e a r l y } }$ is not GPU idle time: the target GPU may continue serving other requests after completing the prefill. Instead, it measures how long migration-specific state is committed before it is needed and serves as a proxy for premature resource commitment and exposure to wasted preparation if the predicted target changes.

Pallas balances post-handover interruption against early preparation using the following objective:

$$
\begin{array} { r } { \mathcal { I } ( T _ { w } ) = \alpha { \cdot } T _ { \mathrm { S I T } } ( T _ { w } ) + ( 1 - \alpha ) { \cdot } T _ { \mathrm { e a r l y } } ( T _ { w } ) , \qquad 0 \leq \alpha \leq 1 , } \end{array}\tag{8}
$$

where a larger � prioritizes reducing service interruption, whereas a smaller � discourages unnecessarily early preparation. The objective captures the latency consequences of incremental KV-cache transfer rather than directly penalizing its total volume. Although a longer window produces more incremental KV-cache blocks, these blocks are transmitted concurrently with source-side decoding. Only the backlog remaining at handover contributes to �<sub>SIT</sub> through $T _ { \mathrm { r e s } } ( T _ { w } )$ . When evaluating a candidate window before handover, $L _ { \mathrm { { h i s t } } }$ is not yet observed because the source continues decoding until the candidate trigger time. The online procedure in § 4.2 therefore predicts the candidate-specific historical-prefix length and evaluates Eq. (8) using runtime estimates of $r _ { p } , r _ { d } ,$ and �.

## 4.2 Online Prefetching-Window Selection

The model in §4.1 quantifies the cost of a given prefetching window, but selecting the window online introduces two practical challenges. First, the historical-prefix length at a candidate trigger time is not yet known because decoding continues until preparation begins. Second, the resources available to migration vary with target-side load and network conditions. Pallas addresses these challenges with a lightweight Scheduler Engine that periodically combines mobility predictions with runtime telemetry and determines whether preparation should be triggered.

Mobility and telemetry inputs. Pallas treats mobility prediction as an external input rather than requiring a particular prediction algorithm. At each control cycle, the predictor provides the predicted target $\mathrm { g N B } \hat { g } _ { \mathrm { t a r } }$ and handover time �ˆ<sub>HO</sub>. Given the current time $t _ { \mathrm { c u r r } }$ , the predicted remaining time is $\widehat { T } _ { \mathrm { r e m a i n } } = \widehat { t } _ { \mathrm { H O } } - t _ { \mathrm { c u r r } }$ . The Scheduler Engine additionally collects the current context length $K _ { \mathrm { c u r r } } ,$ target-side prefill throughput $r _ { \mathrm { p } } ,$ source-side per-request decode throughput $r _ { \mathrm { { d } } } ,$ and inter-gNB goodput �. The Scheduler Engine uses session-level efective rates rather than peak hardware or link capacities. For example, $r _ { p }$ reflects the prefill throughput available to the migrating request under the current target-side load, while � reflects the goodput available after contention and protocol overhead. To avoid reacting to short-term fluctuations, Pallas smooths the instantaneous measurements using an exponentially weighted moving average. The resulting estimates allow the window decision to account for both handover–handover and handover–inference contention without assuming exclusive GPU or network resources.

Admission decision. Before evaluating candidate windows, the Scheduler Engine verifies that the predicted target can execute a compatible LLM instance and has suficient available VRAM for the request state expected at handover. If either condition is not satisfied, Pallas does not initiate direct preparation at that gNB and invokes the fallback policy described in §3. Model compatibility and VRAM availability are therefore treated as admission conditions, and window optimization is performed only for a feasible target.

Candidate-window evaluation. Pallas limits the candidate window to $T _ { \mathrm { l i m i t } } = \mathrm { m i n } ( T _ { \mathrm { m a x } } , \widehat { T } _ { \mathrm { r e m a i n } } )$ , where $T _ { \mathrm { m a x } }$ prevents preparation from starting excessively early. The Scheduler Engine enumerates candidate windows over $[ 0 , T _ { \mathrm { l i m i t } } ]$ with granularity Δ� . For a candidate $T _ { w } ,$ preparation would begin after $\widehat { T } _ { \mathrm { r e m a i n } } - T _ { w }$ seconds. Because the source continues decoding during this interval, Pallas first predicts the historical-prefix length at the corresponding trigger time and then evaluates the cost model from §4.1:

$$
\begin{array} { r } { \widehat { L } _ { \mathrm { h i s t } } ( T _ { w } ) = K _ { \mathrm { c u r r } } + r _ { d } \cdot \big ( \widehat { T } _ { \mathrm { r e m a i n } } - T _ { w } \big ) , \ } \\ { T _ { w } ^ { * } = \mathop { \arg \operatorname* { m i n } } _ { T _ { w } \in \mathcal { W } } \mathcal { T } \big ( T _ { w } ; \widehat { L } _ { \mathrm { h i s t } } ( T _ { w } ) \big ) , \ } \end{array}\tag{9}
$$

where $\mathcal { W }$ is the set of candidate windows and $\mathcal { T } ( \cdot )$ is the objective in Eq. (8). This evaluation explicitly captures that a later trigger produces a longer historical prefix, whereas an earlier trigger produces a longer incremental sufix.

Trigger decision. If $\widehat { T } _ { \mathrm { r e m a i n } } > T _ { w } ^ { * } .$ , the Scheduler Engine defers migration and reevaluates the decision in the next control cycle. Once $\widehat { T } _ { \mathrm { r e m a i n } } \ \leq \ T _ { w } ^ { * }$ , it triggers parallel state preparation at the predicted target. The source records its block-aligned split position as the prefix–sufix split and starts parallel state preparation. Triggering is a one-time transition for the current target prediction; subsequent changes in the predicted handover time or target are handled through replanning or cancellation rather than by reversing the completed trigger.

Algorithm 1: Online prefetching-window selection   
Input: Mobility prediction $( \widehat { g } _ { \mathrm { t a r } } , \widehat { t } _ { \mathrm { H O } } )$ , current   
request state, and runtime telemetry   
Output: Selected window $T _ { w } ^ { \ast }$ if the predicted target   
is feasible   
1 Update $\widehat { T } _ { \mathrm { r e m a i n } }$ and smooth $r _ { p } , r _ { d } ,$ and �   
2 if the predicted target lacks the required model or   
$\underline { { \mathrm { V R A M } } }$ then   
3 Invoke the fallback policy and return   
4 Set $T _ { \mathrm { l i m i t } }$ ← min $( T _ { \mathrm { m a x } } , \widehat { T } _ { \mathrm { r e m a i n } } )$   
5 Construct candidate set ${ \mathcal { W } } \gets \{ 0 , \Delta T , . . . , T _ { \mathrm { l i m i t } } \}$   
6 foreach $T _ { w } \in \mathcal { W }$ do   
7 Predict $\widehat { L } _ { \mathrm { h i s t } } ( T _ { w } )$   
8 Evaluate $\mathcal { T } ( T _ { w } ; \widehat { L } _ { \mathrm { h i s t } } ( T _ { w } ) )$   
9 $T _ { w } ^ { \ast }$ ← arg min $\boldsymbol { 1 } _ { T _ { w } \in \mathcal { W } } \mathcal { T } \Big ( \boldsymbol { T } _ { w } ; \widehat { L } _ { \mathrm { h i s t } } \big ( \boldsymbol { T } _ { w } \big ) \Big )$   
10 if $\widehat { T } _ { \mathrm { r e m a i n } } \leq T _ { w } ^ { * }$ then   
11 Trigger parallel state preparation at $\hat { g } _ { \mathrm { t a r } } ;$   
12 else   
13 Defer migration until the next control cycle

Algorithm 1 summarizes the online procedure. With a bounded grid search, each control cycle evaluates ${ \cal O } ( T _ { \mathrm { l i m i t } } / \Delta T )$ candidates. The cost evaluation involves only scalar arithmetic and completes within microseconds in our implementation, making its overhead negligible relative to the control period and LLM inference timescale. Pallas uses the observed resource availability to decide when migration should begin, but does not jointly allocate GPU cycles or inter-gNB bandwidth among concurrent migrations. Such contention is reflected through the efective rates supplied to the Scheduler Engine, and general multi-user resource allocation remains outside the scope of this work.

## 4.3 Parallel State Preparation

Once preparation is triggered, Pallas reconstructs the targetside inference state without suspending source-side token generation. The protocol consists of initialization, parallel preparation, and final synchronization.

Initialization and parallel preparation. At the trigger time, the source remains the sole owner of the request and records its block-aligned split position �. Tokens in [0, �) form an immutable historical prefix, whereas subsequently generated tokens form an evolving sufix. The source sends the exact prefix token IDs, split position, request identifier, and migration epoch to the target, which reconstructs the prefix KV cache through local prefill. Meanwhile, the source continues generating and delivering tokens to the UE. Newly completed sufix KV-cache blocks are streamed asynchronously with their logical token positions and bufered at the target. Only completed blocks are transferred during decoding; any completed but unsent blocks remain in a sender-side queue, whose backlog at handover corresponds to the residual transfer delay in Eq. (5). Thus, prefix reconstruction, source-side decoding, and sufix transfer proceed in parallel.

Final synchronization and activation. At handover, the source finishes its current decoding step and records the final sequence position �. It then sends the final token-sequence snapshot and any unsynchronized KV-cache blocks, including the residual boundary block. The target waits until prefix reconstruction is complete and state for all positions in $[ k , n )$ is available. It then combines the reconstructed prefix and received sufix and activates the request in the decoding stage, generating token � without repeating prefill over the complete context. Only preparation unfinished at handover contributes to SIT.

State consistency. Before activation, Pallas verifies three conditions: the prefix is reconstructed from the exact token IDs supplied by the source; the assembled KV cache covers all logical positions in [0, �) without gaps or conflicts; and the source remains the only active decoder until ownership is transferred to the target. The migration epoch prevents blocks from a cancelled preparation attempt from entering the active request state.

## 4.4 Prediction-Aware Migration Replanning

Mobility predictions may change as the UE approaches a cell boundary. Before preparation begins, Pallas simply updates $( \widehat { g } _ { \mathrm { t a r } } , \widehat { t } _ { \mathrm { H O } } )$ and reruns Algorithm 1. Because no state has been prepared, these updates incur only Schedule Engine overhead. After preparation begins, a revised handover time does not invalidate existing state. If handover is predicted earlier, Pallas continues preparation against the shortened deadline, and unfinished work contributes to SIT. If handover is delayed, the source continues serving the UE and streaming incremental KV blocks; the reconstructed prefix remains valid, although it may remain resident at the target for longer.

A change in the predicted target invalidates preparation at the previous gNB. Pallas increments the migration epoch, cancels the obsolete attempt, and releases its reconstructed and bufered state. It then performs admission control and reruns the window decision for the new target. Because

Pallas prepares at only one predicted target at a time, obsolete messages can be discarded using their migration epoch. If the actual handover target difers from the prepared target, Pallas falls back to post-handover recovery at the actual target rather than activating an inconsistent state. If the new target is infeasible, Pallas invokes the fallback policy described in §3.

We bound the duration of preparation discarded because of incorrect target predictions. Let $T _ { \mathrm { a v a i l } }$ denote the interval between the time when the target prediction becomes stably correct and the actual handover, with $T _ { \mathrm { a v a i l } } = 0$ if this never occurs. Let $E _ { \mathrm { e a r l y } }$ denote the maximum amount by which the predicted handover time precedes the actual handover before target stabilization. Because Pallas caps the prefetching window at $T _ { \mathrm { m a x } }$ , the discarded preparation duration satisfies

$$
T _ { \mathrm { w a s t e } } \leq \mathrm { m a x } \big \{ T _ { \mathrm { m a x } } + E _ { \mathrm { e a r l y } } - T _ { \mathrm { a v a i l } } , 0 \big \} .\tag{10}
$$

The bound follows because preparation can begin at mos $T _ { \mathrm { m a x } } +$ $E _ { \mathrm { e a r l y } }$ before the actual handover, whereas preparation performed during the final $T _ { \mathrm { a v a i l } }$ interval is useful. Equation (10) bounds stale-preparation duration, rather than GPU cycles, transferred bytes, or occupied VRAM.

## 4.5 Contention-Aware Migration Operation

Pallas does not assume exclusive access to the target GPU or inter-gNB link. Instead, the Scheduler Engine uses the effective prefill throughput $r _ { p }$ and goodput � available to each migration session, allowing the selected window to reflect current resource contention. Under handover–handover contention, concurrent prefix-prefill tasks share the target GPU and their sufix streams share the inter-gNB link. The resulting efective rates are incorporated into the periodic window decision, while target-side admission prevents the aggregate inference and migration state from exceeding available VRAM. If contention increases after preparation begins, migration continues at the reduced rates and any remaining work contributes to SIT. Under handover–inference con tention, migration prefill competes with resident decoding requests. Pallas uses chunked prefill to interleave prefix reconstruction with decoding and limit long GPU-blocking operations [19]. Incremental KV-cache extraction and transfer proceed asynchronously, while their memory and interconnect overhead remains reflected in the measured efective rates. Pallas adapts when migration begins but does not jointly allocate GPU cycles or link bandwidth among competing requests. Such allocation remains the responsibility of the underlying serving and network runtimes.

## 5 System Implementation

We implement Pallas on vLLM 0.8.5 [8] with approximately 1.7K lines of Python code. Each gNB runs an independent vLLM AsyncLLMEngine, while a node daemon coordinates migration through asynchronous control requests and gRPCbased KV-cache transfer. Rather than modifying model kernels, Pallas directly interfaces with vLLM’s scheduler, PagedAttention block manager, and GPU KV cache to access and reconstruct request state. We use a PagedAttention block size of 16 tokens and the same implementation for the single-host and cross-host experiments in § 6.1.

Parallel state preparation. We align the prefix–sufix split to PagedAttention block boundaries so that only completed KV blocks are transferred during decoding. At the trigger, the source identifies the last complete block, snapshots the prefix token IDs, and sends them to the target for normal vLLM prefill. The source continues decoding without waiting for the target. As complete blocks are produced, Pallas obtains their physical block IDs from vLLM’s block manager, extracts the corresponding per-layer KV tensors from the GPU cache, and asynchronously streams them through gRPC. The tensors are serialized in memory using PyTorch and io.BytesIO, while the target bufers received sufix blocks for each request and independently reconstructs the prefix.

Final synchronization and activation. At handover, the source takes a final token-sequence snapshot and transfers any remaining unsynchronized KV blocks before stopping source-side decoding. Migrated KV blocks cannot be directly attached to an arbitrary active vLLM request because its scheduler state, sequence metadata, block table, and physical KV allocation must remain consistent. Pallas therefore reconstructs a target-side request for the final token sequence, allocates a new physical block table, and copies the reconstructed prefix and bufered sufix blocks into the corresponding GPU locations. It then updates the computed-token state and decoding stage and inserts the request into vLLM’s running queue, allowing decoding to resume from position � without repeating prefill over [0, �).

Cancellation and contention. Pallas maintains migration state separately for each request, including the selected target, transfer progress, and bufered blocks, so cancelled preparation can be discarded without afecting other requests. KV extraction and transfer execute asynchronously with sourceside generation, while target-side prefix reconstruction and resource sharing remain managed by the underlying vLLM and communication runtimes.

## 6 Performance Evaluation

## 6.1 Experimental Methodology

Hardware and Workloads. For the single-host experiments, the source and target gNBs are deployed on a server with an Intel(R) Xeon(R) Platinum 8358P CPU (2.60 GHz), 503 GiB of memory, and NVIDIA RTX A6000 GPUs (48 GB each), with each gNB exclusively using one GPU. For the crosshost experiment, the two gNBs run on two physical servers connected by 1-Gbps full-duplex Ethernet, each using one RTX A6000 GPU. The native inter-host RTT is 0.21 ms and the unshaped TCP goodput is approximately 940 Mbps. All experiments use our vLLM 0.8.5 prototype.

We evaluate Meta-Llama-3-8B, Qwen3-14B, and Qwen3- 32B-Instruct-AWQ using ShareGPT conversational requests. The AWQ-quantized checkpoint allows Qwen3-32B to fit within a single RTX A6000. For the single-handover and ablation experiments, we divide the 500–4,500-token context range into four equal intervals and randomly select 15 samples from each interval. Repeated-handover sessions start with a 1,000-token prompt and generate tokens throughout trajectory replay. The concurrency experiment uses a fixed 2,000-token context, while the window-decision and prediction-stress experiments use 1,000-, 2,000-, or 3,000- token contexts.

Network and Mobility Settings. The controlled experiments use inter-gNB goodputs of100, 300, and 500 Mbps, covering bandwidth-constrained operating regimes [20]. Within each setting, all compared schemes use the same goodput. We additionally replay time-varying goodput traces from the SNOB-5G dataset [21], collected from an outdoor 60-GHz WiGig testbed under normal and blockage conditions, over the native cross-host path. We use vehicle trajectories from the nuScenes v1.0-trainval split [22] and integrate a constantvelocity-and-heading (CVH) predictor [23]. Using the UE’s latest position, velocity, and heading, it predicts the future trajectory, target gNB, and handover time. We evaluate its prediction availability and impact on Pallas in § 6.3.

Trajectories are translated and rotated, without scaling, into square-grid topologies with cell side lengths of 50, 100, and 150 m. We linearly interpolate between consecutive samples that cross a cell boundary to obtain the actual handover time and derive $T _ { \mathrm { a v a i l } }$ as defined in § 4.4. For the singlehandover, window-decision, ablation, and prediction-stress experiments, we extract individual handover events and drive the prototype using their predicted targets and handover times. Each paired comparison uses the same event, predictor outputs, workload, and network setting. The prediction stress experiment varies $T _ { \mathrm { a v a i l } }$ by delaying the stable correct target prediction. The repeated-handover experiment instead replays complete trajectory segments with the predictor and Scheduler Engine operating online.

Baselines. We compare Pallas with Detour, Full-Copy, Recomputation, and ctHO under identical workloads, network conditions, and handover events. For ctHO, the prefix– sufix split is selected at KV-block granularity to minimize post-handover recovery delay. We use its single-user formulation for the single- and repeated-handover experiments and its multi-user formulation when multiple UEs hand over concurrently to the same target, under the same aggregate inter-gNB goodput constraint.

![](images/b20b8802b792bbc84379d6a54d88e58cffb580933b18b14b47518602e168e7fd.jpg)  
(a) 100 Mbps

![](images/5ee3f50126733330dc36b4412e946ba066202e3a6ae7d71447aa991da1e80ab4.jpg)  
(b) 300 Mbps

![](images/37be047264f1d1cd5f2fb72cd5e4c453659c168f91bb095f83566ef0efe5ac6e.jpg)  
(c) 500 Mbps

![](images/26c2d3402562c633c68a3c4aed12fd8501f4fffa18c1f306bb4e7e1c52a85eaa.jpg)  
(d) All Fixed-goodput Settings

![](images/54ef4738df83629596af55de597ad70d51f61995415d75736980e24242cf3d15.jpg)  
(e) Measured Goodputs  
Figure 5: Single-handover performance of five methods across three models: (a)–(c) SIT under fixed goodputs of 100, 300, and 500 Mbps; (d) ITL across all fixed-goodput settings; (e) SIT under measured goodput traces.

<table><tr><td>Model</td><td>Metric</td><td>Detour</td><td>Full-Copy</td><td>Recomputation</td><td>ctHO</td><td>Pallas</td></tr><tr><td rowspan="8">Llama-3-8B</td><td>SIT Avg.</td><td>287.7</td><td>19645.7</td><td>837.3</td><td>812.7</td><td>153.8</td></tr><tr><td>SIT P50</td><td>279.1</td><td>18018.7</td><td>828.8</td><td>812.3</td><td>152.8</td></tr><tr><td>SIT P90</td><td>481.7</td><td>26931.4</td><td>1082.8</td><td>1040.2</td><td>190.9</td></tr><tr><td>SIT P99</td><td>652.2</td><td>36163.7</td><td>1252.2</td><td>1230.3</td><td>216.3</td></tr><tr><td>ITL Avg.</td><td>241.3</td><td>25.0</td><td>25.1</td><td>25.1</td><td>25.1</td></tr><tr><td>ITL P90</td><td>433.7</td><td>26.4</td><td>26.4</td><td>26.4</td><td>26.4</td></tr><tr><td>ITL P99</td><td>642.9</td><td>27.3</td><td>27.3</td><td>27.3</td><td>27.2</td></tr><tr><td>SIT Avg.</td><td>307.2</td><td>20983.8</td><td>1360.8</td><td>1305.9</td><td>232.3</td></tr><tr><td rowspan="6">Qwen3-14B</td><td>SIT P50</td><td>298.1</td><td>20163.3</td><td>1327.7</td><td>1273.5</td><td>227.7</td></tr><tr><td>SIT P90</td><td>501.4</td><td>26898.7</td><td>1676.2</td><td>1604.2</td><td>279.1</td></tr><tr><td>SIT P99</td><td>682.8</td><td>36087.5</td><td>2176.0</td><td>2079.5</td><td>343.4</td></tr><tr><td>ITL Avg.</td><td>261.7</td><td>45.4</td><td>45.5</td><td>45.5</td><td>45.6</td></tr><tr><td>ITL P90</td><td>454.2</td><td>46.9</td><td>46.8</td><td>46.8</td><td>46.9</td></tr><tr><td>ITL P99</td><td>663.4</td><td>47.6</td><td>47.5</td><td>47.6</td><td>47.7</td></tr><tr><td rowspan="7">Qwen3-32B</td><td>SIT Avg.</td><td>311.9</td><td>19746.9</td><td>4432.8</td><td>3678.6</td><td>323.8</td></tr><tr><td>SIT P50</td><td>306.3</td><td>19056.7</td><td>4248.1</td><td>3621.7</td><td>317.9</td></tr><tr><td>SIT P90</td><td>523.2</td><td>24856.9</td><td>4891.0</td><td>4163.2</td><td>387.4</td></tr><tr><td>SIT P99</td><td>699.3</td><td>32802.9</td><td>5533.8</td><td>4830.8</td><td>488.1</td></tr><tr><td>ITL Avg.</td><td>283.0</td><td>66.7</td><td>66.8</td><td>66.7</td><td>66.7</td></tr><tr><td>ITL P90</td><td>475.4</td><td>68.0</td><td>68.0</td><td>68.0</td><td>68.0</td></tr><tr><td>ITL P99</td><td>684.6</td><td>68.8</td><td>69.1</td><td>68.9</td><td>68.8</td></tr></table>

Metrics. We report service interruption time (SIT) and inter-token latency (ITL) as defined in § 2.2. The windowdecision experiment additionally reports the early-preparation exposure $T _ { \mathrm { e a r l y } }$ defined in Eq. (7). We use $T _ { \mathrm { a v a i l } }$ to characterize the prediction time available before handover.

## 6.2 Efectiveness of Pallas

Single-Handover Performance. We first compare the five schemes under fixed inter-gNB goodputs. As shown in Figures 5(a)–(c), Pallas consistently achieves the lowest aver age SIT among the target-side recovery schemes. Even at 100 Mbps with Qwen3-32B, its average SIT is only 321 ms. Across all settings, Pallas reduces average SIT by factors of 70.13–89.68, 2.35–7.28, and 2.28–6.76 relative to Full-Copy, Recomputation, and ctHO, respectively. These gains arise because the baselines begin state recovery after handover, whereas Pallas completes most prefix reconstruction and suffix synchronization beforehand. Figure 5(d) shows that Pallas maintains an average ITL of 28–48 ms, reducing it by 16.0%– 50.0% relative to Detour by restoring serving locality after handover. Under the measured time-varying goodput traces in Figure 5(e), Pallas achieves an average SIT of 204–316 ms and reduces SIT by factors of 71.73–82.99, 2.34–7.12, and 2.31–6.53 relative to Full-Copy, Recomputation, and ctHO, respectively. These results confirm that its benefits persist across physical hosts and time-varying network conditions.

Repeated-Handover Performance. We replay nuScenes trajectory segments over an 8×8 grid of 100-m cells, with the CVH predictor and Scheduler Engine operating throughout each session. We conduct ten rounds, each containing eight trajectories for every handover count from one to eight. This produces 64 sessions and 288 handovers per round, or 2,880 handovers per model. For each handover, goodput is sampled from a normal distribution with a mean of 300 Mbps and a standard deviation of 50 Mbps, truncated below at 150 Mbps. All schemes use identical trajectories, predictions, goodput samples, and workloads. As shown in Table 2, Pallas achieves an average SIT of 153.8–323.8 ms and a P99 SIT of 216.3–488.1 ms across the three models, outperforming all target-side recovery baselines. Its average and P99 ITLs remain at 25.1–66.7 ms and 27.2–68.8 ms, respectively, while Detour incurs 4.24–9.60× its average ITL because repeated handovers progressively lengthen the forwarding path. These results demonstrate that Pallas preserves low SIT and stable token delivery across long-running sessions with repeated mobility events.

## 6.3 Window Decision and Prediction Robustness

Prefetching-Window Decision. We evaluate the Scheduler Engine by sweeping fixed windows from 0 to 5 s in 0.2-s increments and comparing them with the window selected by Pallas. We use nine configurations from a three-level orthogonal array spanning inter-gNB goodput, model size, and context length. Within each configuration, the handover event, predictor outputs, and other settings remain unchanged. Figure 6 confirms the trade-of modeled in § 4.1: short windows leave more preparation unfinished at handover, whereas long windows increase early-preparation exposure without further reducing SIT. For example, at 300 Mbps with Qwen3-14B and a 3,000-token context, Pallas selects a 1.64-s window, resulting in a SIT of 227.57 ms and only 0.017 s of early preparation exposure. Across all nine configurations, the selected windows keep both metrics near their low-value regions. We set � = 0.8 in Eq. (8), prioritizing SIT reduction over early preparation.

![](images/1e10b3fd501ae6d56c5638b8b3bc323b788f844b6a01911d5c594af07db4777e.jpg)  
Figure 6: Prefetching-window selection across diferent goodputs, models, and context lengths.

Prediction robustness. Table 3 characterizes the CVH predictor used in our evaluation. As the cell size increases from 50 to 150 m, the fraction of handovers with at least 2 s of $T _ { \mathrm { a v a i l } }$ increases from 51.9% to 62.8%, while the fraction experiencing an incorrect target prediction decreases from 37.3% to 11.2%. The $\mathrm { P 9 0 ~ } E _ { \mathrm { e a r l y } }$ also decreases from 4.19 to 0.94 s. Correspondingly, the mean $T _ { \mathrm { w a s t e } }$ decreases from 1.46 to 0.58 s and remains below the mean analytical bound for every cell size. We further stress Pallas by varying $T _ { \mathrm { a v a i l } }$ from 0 to 6 s in 0.1-s increments while keeping each handover event, workload, and network setting unchanged. As shown in Figure 7, SIT remains low until the available preparation time becomes insuficient for the selected model and context. For example, at 300 Mbps with Qwen3-14B and a 2,000-token context, noticeable degradation appears only below $T _ { \mathrm { a v a i l } } = 1 ~ s$ . Reducing $T _ { \mathrm { a v a i l } }$ leaves more preparation on the post-handover path and therefore increases SIT. Together, Figure 7 and Table 3 show that prediction availability determines the achievable benefit, while stale-preparation duration remains consistent with the bound in § 4.4.

Table 3: Prediction availability and failure overhead.
<table><tr><td>Cell (m)</td><td> $T _ { \mathrm { a v a i l } } \geq 2 ~ s$  (%)</td><td> $T _ { \mathrm { a v a i l } } \geq 3 ~ s$  (%)</td><td>Îtar  $\ t \ g _ { \mathrm { t a r } } ^ { * }$  (%)</td><td> $E _ { \mathrm { e a r l y } }$  P90 (s)</td><td> $T _ { \mathrm { w a s t e } }$  Mean (s)</td><td>Upper Bound Mean (s)</td></tr><tr><td>50</td><td>51.9</td><td>34.6</td><td>37.3</td><td>4.19</td><td>1.46</td><td>3.79</td></tr><tr><td>100</td><td>57.9</td><td>42.2</td><td>19.3</td><td>2.16</td><td>1.02</td><td>2.93</td></tr><tr><td>150</td><td>62.8</td><td>48.0</td><td>11.2</td><td>0.94</td><td>0.58</td><td>2.57</td></tr></table>

![](images/849ff266bb4827f3561e6174ba93b215389f90354cd661398978daa811b13416.jpg)  
Figure 7: Performance of Pallas under controlled reductions in $T _ { \mathrm { a v a i l } } .$

## 6.4 Concurrency Evaluation

We evaluate Pallas under handover–handover (HH) and handover–inference (HI) contention, as well as a joint HH– HI case. HH represents concurrent migrations to the same target, whereas HI represents migration preparation sharing the target GPU with resident inference. All experiments use Qwen3-14B, a 2,000-token migration context, and an aggregate inter-gNB goodput of 1 Gbps. Each configuration is repeated ten times; error bars in Figure 8 show the corresponding standard deviations.

Handover–handover contention. We simultaneously migrate � ∈ 1, 2, 3, 4 UEs from one source to the same target, with � = 1 serving as the non-concurrent reference. Their prefill tasks share the target GPU in first-come, first-served order, while their KV-cache transfers equally share the aggregate goodput. We compare Pallas with multi-user ctHO under the same resource constraints and report average and worst-user SIT. As shown in Figure 8(a), the average SIT of Pallas remains at 236–240 ms for $K \leq 3$ and increases to 292 ms at � = 4. Its worst-user SIT increases from 237 ms at � = 1 to 283 ms at � = 3 and 359 ms at � = 4. In comparison, the worst-user SIT of ctHO is 442–507 ms, with Pallas reducing it by 29–51%. Proactive preparation absorbs most of the contention before handover; only at higher concurrency does the increased prefill and transfer backlog noticeably raise SIT. In contrast, ctHO exposes its concurrent recovery operations entirely after handover.

![](images/e4db9d34897e8639d6e5b732604b48fb02d0b72daf757fd91acf383004b162a7.jpg)  
(a) HH Concurrency

![](images/a4f21a25dbbf708e74fcb58aa5b110e4e86c5273c6358050764aa9f5d19c86c9.jpg)  
(b) HI Concurrency

![](images/4c74b9f77897901d526ed24e7f1d0c66c838ae4f4b120db5979bdca9ae169914.jpg)  
(c) Joint HH & HI  
Figure 8: Concurrency performance under (a) HH, (b) HI, and (c) joint HH–HI contention.

Handover–inference contention. We next migrate one UE to a target serving � ∈ 0, 1, 2, 3 resident sessions, where � = 0 is the no-load reference. Migration prefill and resident decoding share the GPU, and a 256-token prefill chunk is used to interleave their execution. Because only one UE migrates, its transfer uses the full configured inter-gNB goodput. Figure 8(b) shows that the migrating UE’s average SIT increases only from 232 ms at � = 0 to 244 ms at � = 3. Across � = 1–3, the resident UEs’ average ITL increases from 53.24 to 61.38 ms, while their P99 ITL rises from 84.67 to 104.15 ms. Chunked prefill therefore limits the efect on average decoding latency, although the P99 result at � = 3 reveals measurable tail interference under heavier target-side load.

Joint HH–HI contention. Finally, we evaluate � = 2 migrating UEs sharing the target with � = 2 resident sessions. As shown in Figure 8(c), Pallas achieves an average SIT of 281 ms and a worst-user SIT of 322 ms. The resident UEs maintain an average ITL of63 ms, while their P99 ITL reaches 104 ms. The simultaneous GPU and link contention increases both migration and inference latency relative to the isolated cases, but average migration SIT remains below 300 ms.

Overall, Pallas maintains an average SIT below 300 ms across all evaluated HH, HI, and joint configurations, while contention primarily afects worst-user SIT and resident tail ITL. Joint multi-user resource allocation remains outside the scope of this work.

## 6.5 Ablation Study

We isolate the benefits of historical KV-cache recomputation and incremental KV-cache streaming using two variants of Pallas. To control for proactive timing, all variants use the same handover events, predictor outputs, prefetchingwindow decisions, workloads, and network conditions. Pallas w/o RC disables historical-prefix reconstruction and proactively transfers the complete KV cache from the source. Pallas w/o SM disables incremental streaming; after handover, the target receives the tokens generated during the prefetching window and recomputes their KV cache before resuming decoding.

![](images/42b42be2df1b5ef99409f44b6b2d5fcf75de02793f5805aff8dd4b955e150750.jpg)  
(a) 100 Mbps

![](images/6718a0ff56880be9ec0b0728fd7cc21944de3b7108ffd36282d99a504042fb3f.jpg)  
(b) 300 Mbps

![](images/33ebc7c3b72e5a84cf1e0b41924ab478763291a6c55489ddfea12dc11f0a7394.jpg)  
(c) 500 Mbps  
Figure 9: Ablation study of single-handover SIT across three goodputs and three models.

As shown in Figure 9, removing either component increases SIT. At 300 Mbps with Qwen3-14B, Pallas achieves an SIT of 235 ms, compared with 12,486 ms for Pallas w/o RC and 655 ms for Pallas w/o SM. Across all evaluated settings, Pallas reduces SIT by an average of 97.93% and 33.3% relative to the two variants, respectively. Without recomputation, transferring the complete KV cache remains communicationintensive; without incremental streaming, the target must recompute the state generated during preparation after handover. Combining the two mechanisms therefore minimizes the residual work on the post-handover interruption path.

## 7 Related Work

Service migration has been extensively studied in mobile edge computing (MEC). Prior systems use Markov decision processes, deep reinforcement learning, mobility prediction, and collaborative optimization to determine when and where services should migrate [24–27]. These approaches generally model service state as a relatively static transferable object, whereas an active LLM request maintains a KV cache that grows throughout decoding. Data-center LLM serving systems address this state through direct KV-cache migration, as in Llumnix [13], or token-based recomputation, as in ServerlessLLM [14]. KV-cache compression techniques, including CacheGen, GEAR, and KIVI, can further reduce storage or transfer volume [9, 28, 29]. However, these systems do not consider mobility-induced handover or the need to prepare an evolving inference state while source-side decoding continues.

AI-RAN brings inference into radio access networks, motivating mobility support for edge AI workloads. AoRA demonstrates AI inference colocated with RAN functions [3], while the AI-RAN working group highlights proactive mobility management as an important requirement [11]. The most closely related work, ctHO [15], combines target-side token prefill with direct KV-cache transfer and jointly optimizes their partition and backhaul allocation to reduce worst-user handover delay. However, both recovery operations begin after handover. In contrast, Pallas prepares the target beforehand by overlapping historical-prefix reconstruction and incremental KV-cache streaming with ongoing source-side inference, and adaptively selects the prefetching window to reduce the work remaining on the post-handover interruption path.

## 8 Conclusion

In this work, we present Pallas, a proactive parallel KV-cache migration framework for mobile LLM serving in AI-RAN.By overlapping target-side recomputation and incremental KVcache streaming with ongoing source-side inference, Pallas moves most state-preparation work before handover.Our vLLM-based prototype and trace-driven evaluation show that Pallas substantially reduces SIT compared with targetside recovery baselines while avoiding the persistent ITL penalty of Detour.Future work will explore joint resource allocation and window replanning under dynamic multi-user contention.

## References

[1] Michele Polese, Leonardo Bonati, Salvatore D’Oro, Stefano Basagni, and Tommaso Melodia. Understanding O-RAN: Architecture, interfaces, algorithms, security, and research challenges. IEEE Communications Surveys & Tutorials, 25(2):1376–1411, 2023.

[2] Lopamudra Kundu, Xingqin Lin, Rajesh Gadiyar, Jean-Francois Lacasse, and Shuvo Chowdhury. AI-RAN: Transforming RAN with AI-driven computing infrastructure. arXiv preprint arXiv:2501.09007, 2025.

[3] Siyavushkhon Kholmatov, Seongsik Cho, Song Chong, and Kyunghan Lee. AoRA: AI-on-RAN for backhaul-free edge inference. In Proceedings of the ACM SIGCOMM 2025 Conference, pages 1263– 1265, 2025.

[4] Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, and Hao Zhang. DistServe: Disaggregating prefill and decoding for goodput-optimized large language model serv ing. In 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24), pages 193–210, 2024.

[5] Yuxing Xiang, Xue Li, Kun Qian, Yan Zhang, Wenyuan Yu, Ennan Zhai, Xin Jin, and Jingren Zhou. ServeGen: Workload characteri zation and generation of large language model serving in produc tion. In 23rd USENIX Symposium on Networked Systems Design and Implementation (NSDI 26), pages 1845–1859, Renton, WA, May 2026. USENIX Association.

[6] Hadeel Abdah, João Paulo Barraca, and Rui L. Aguiar. Handover prediction integrated with service migration in 5G systems. In ICC 2020 - 2020 IEEE International Conference on Communications (ICC), pages 1–7, 2020.

[7] Mukhtiar Ahmad, Faaiq Bilal, Mutahar Ali, Syed Muhammad Ali Nawazish, Amir Salman, Shazer Ali, Fawad Ahmad, and Zafar Ayyub Qazi. Warping the Edge: Enabling instant mobility for stateful appli cations over 5G and beyond. In Proceedings of the Tenth ACM/IEEE Symposium on Edge Computing, pages 1–18. Association for Computing Machinery, 2025.

[8] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Eficient memory management for large language model serving with

PagedAttention. In Proceedings of the 29th symposium on operating systems principles, pages 611–626, 2023.

[9] Yuhan Liu, Hanchen Li, Yihua Cheng, Siddhant Ray, Yuyang Huang, Qizheng Zhang, Kuntai Du, Jiayi Yao, Shan Lu, Ganesh Ananthanarayanan, Michael Maire, Henry Hofmann, Ari Holtzman, and Junchen Jiang. CacheGen: KV cache compression and streaming for fast large language model serving. In Proceedings of the ACM SIGCOMM 2024 Conference, ACM SIGCOMM ’24, pages 38–56, New York, NY, USA, 2024. Association for Computing Machinery.

[10] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[11] AI-RAN Alliance. AI-on-RAN: Enabling monetizable diferentiated connectivity for AI. Working group 3 white paper, AI-RAN Alliance, 2026.

[12] MLCommons. MLPerf Inference Rules. https://github.com/ mlcommons/inference\_policies/blob/master/inference\_rules.adoc, 2025. Llama3.1-8B conversational serving uses a 100 ms TPOT constraint.

[13] Biao Sun, Ziming Huang, Hanyu Zhao, Wencong Xiao, Xinyi Zhang, Yong Li, and Wei Lin. Llumnix: Dynamic scheduling for large language model serving. In 18th USENIX symposium on operating systems design and implementation (OSDI 24), pages 173–191, 2024.

[14] Yao Fu, Leyang Xue, Yeqi Huang, Andrei-Octavian Brabete, Dmitrii Ustiugov, Yuvraj Patel, and Luo Mai. ServerlessLLM: Low-Latency serverless inference for large language models. In 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24), pages 135–153, 2024.

[15] Seunghun Lee, Jihong Park, Ce Zheng, and Hyuncheol Park. Lowlatency edge LLM handover via joint KV cache transfer and token prefill, 2026.

[16] Lifan Mei, Jinrui Gou, Yujin Cai, Houwei Cao, and Yong Liu. Realtime mobile bandwidth and handof predictions in 4G/5G networks. Computer Networks, 204:108736, 2022.

[17] 3GPP. Enabling edge computing applications in 3gpp, February 2021. Accessed: July 28, 2026.

[18] Shan Yu, Yifan Qiao, Mingyuan Ma, Yangmin Li, Shuo Yang, Xinyuan Tong, Yang Wang, Zhiqiang Xie, Yuwei An, Shiyi Cao, Ke Bao, Deepak Vij, Xiaoning Ding, Yichen Wang, Qingda Lu, Zhong Wang, Gao Gao, Harry Xu, Junyi Shu, Jiarong Xing, and Ying Sheng. Prism: Cost-Eficient Multi-LLM serving via GPU memory balloon ing. In 20th USENIX Symposium on Operating Systems Design and Implementation (OSDI 26), pages 75–93, Seattle, WA, July 2026. USENIX Association.

[19] Amey Agrawal, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S Gulavani, and Ramachandran Ramjee. Sarathi: Eficient LLM inference by piggybacking decodes with chunked prefills. arXiv preprint arXiv:2308.16369, 2023.

[20] NGMN Alliance. Small cell backhaul requirements. White Paper Version 1.0 Final, Next Generation Mobile Networks Alliance, June 2012. Final deliverable, approved 4 June 2012.

[21] Tânia Ferreira, Duarte Raposo, Alexandre Figueiredo, Eurico Dias, Pedro Rito, Miguel Luís, and Susana Sargento. Experimental mmwave wigig-based backhaul network dataset. Data in Brief, 52:109954, 2024.

[22] Holger Caesar, Varun Bankiti, Alex H. Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, and Oscar Beijbom. nuscenes: A multimodal dataset for autonomous driving. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2020.

[23] Phillip Karle, Lukas Furtner, and Markus Lienkamp. Self-evaluation of trajectory predictors for autonomous driving. Electronics, 13(5):946, 2024.

[24] Tarik Taleb, Adlen Ksentini, and Pantelis A Frangoudis. Follow-me cloud: When cloud services follow mobile users. IEEE Transactions on Cloud Computing, 7(2):369–382, 2016.

[25] Lusungu Josh Mwasinga, Duc-Tai Le, Syed M Raza, Rajesh Challa, Moonseong Kim, and Hyunseung Choo. RASM: Resource-aware service migration in edge computing based on deep reinforcement learning. Journal of Parallel and Distributed Computing, 182:104745, 2023.

[26] Xuhui Zhao, Yan Shi, Shanzhi Chen, Jianghui Liu, Baofeng Ji, and Shahid Mumtaz. MAPSM: Mobility-aware proactive service migration framework for mobile edge computing in consumer internet of vehicles. IEEE Transactions on Consumer Electronics, 2025.

[27] Luchuan Zeng, Chen Zhang, Zichen Wang, Hongwei Du, and Xiaohua Jia. Towards collaborative and latency-aware microservice migration in mobile edge computing. IEEE Internet of Things Journal, 2025.

[28] Hao Kang, Qingru Zhang, Souvik Kundu, Geonhwa Jeong, Zaoxing Liu, Tushar Krishna, and Tuo Zhao. GEAR: An eficient error reduction framework for KV cache compression in LLM inference. In Proceedings of The 4th NeurIPS Eficient Natural Language and Speech Processing Workshop, volume 262, pages 305–321, 2024.

[29] Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and Xia Hu. KIVI: A tuningfree asymmetric 2-bit quantization for KV cache. arXiv preprint arXiv:2402.02750, 2024.