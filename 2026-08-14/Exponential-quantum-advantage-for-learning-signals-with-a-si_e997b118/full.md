# Exponential quantum advantage for learning signals with a single qubit

Ishaan Kannan,<sup>1,</sup> <sup>2,</sup> <sup>∗</sup> Sridhar Prabhu,<sup>3,</sup> <sup>4,</sup> <sup>∗</sup> Saeed A. Khan,<sup>3</sup> Mandar M. Sohoni,<sup>3</sup> Xingrui Song,<sup>3</sup> Saswata Roy,<sup>3,</sup> <sup>4</sup> Alen Senanian,<sup>3,</sup> <sup>4,</sup> <sup>†</sup> Valla Fatemi,<sup>3</sup> Peter L. McMahon,<sup>3,</sup> <sup>5,</sup> <sup>‡</sup> and Jordan Cotler<sup>1,</sup> <sup>2,</sup> <sup>‡</sup>

<sup>1</sup>Harvard Quantum Initiative, Harvard University, Cambridge, MA 02138, USA

<sup>2</sup>Department of Physics, Harvard University, Cambridge, MA 02138, USA

<sup>3</sup>School of Applied and Engineering Physics, Cornell University, Ithaca, NY 14853, USA

<sup>4</sup>Department of Physics, Cornell University, Ithaca, NY 14853, USA

<sup>5</sup>Kavli Institute at Cornell for Nanoscale Science, Cornell University, Ithaca, NY 14853, USA

Quantum technology has the potential to transform scientific discovery, but quantum advantages often require processing capabilities well beyond the reach of experimental platforms. We show that coupling a single controllable qubit to an otherwise conventional sensor can exponentially reduce the number of measurements required to learn classical signals. These rigorous quantum advantages apply to fundamental sensing tasks, including learning Fourier coeficients, extracting temporal correlations from time-varying signals, and estimating transformations of physical observables. Using a superconducting cavity–qubit architecture, we experimentally demonstrate 10<sup>7</sup>-fold reductions in the number of measurements required for Fourier-amplitude and time-varying signal learning. Our quantum feature sensing algorithms further enable orders-of-magnitude improvements in simulations of weak-signal dark matter detection and wireless communication applications. These quantum advantages are derived from Quantum Phase-Space Inference (QΨ), a unifying theory of quantum-enhanced experiments that simultaneously converts a set of experimental objectives and constraints into tight lower bounds and optimal quantum-enhanced learning algorithms while producing a certificate of quantum advantage. QΨ extends beyond the regimes captured by quantum Fisher information and provides a framework for systematically identifying rigorous quantum advantages in practical experimental tasks. Together, our results establish that near-term quantum technology can exponentially enhance our ability to learn from classical signals.

## I. Introduction

Despite decades of progress since quantum computing was first proposed, examples of quantum advantage with clear practical utility remain scarce. A growing body of work has suggested a new route to quantum speedups: using experiments enhanced by quantum information processing to learn features of the natural world [1–7]. These foundational results largely establish worst-case separations or existence theorems rather than near-term practical advantages, and many rely on extensive entanglement together with high fidelity, fast local control. Translating such learning advantages into experimental applications requires augmenting quantum sensors with quantum information processing, despite the more limited coherence and control available on most sensing platforms [8–11]. The resulting experimental demands place technologically meaningful implementations of many existing protocols beyond current capabilities.

Meanwhile, quantum metrology has traditionally focused on achieving Heisenberg-limited precision, which provides a quadratic improvement over the classical scaling for estimating a well-specified parameter [12–15]. This framework is well suited to settings in which the signal model is known and estimation precision is the dominant asymptotic quantity. Many experimentally relevant sensing tasks, however, involve learning poorly characterized signals, resolving structured properties, or operating under platform-specific constraints that fall outside this standard parameterestimation setting [16–28]. A central open question is whether these broader regimes admit a rigorous framework in which near-term quantum information processing, integrated with modern quantum sensors, enables superpolynomial improvements over classical methods for learning physical signals.

In this paper, we show that a single controllable ancilla qubit can exponentially reduce the measurement complexity of learning classical signals with a conventional single-mode sensor. We establish this advantage for several basic tasks, including estimating Fourier coeficients, extracting temporal correlations from time-varying signals, and evaluating nonlinear functionals of a signal distribution (Figure 1(a)). The required architecture consists only of the sensor and one controllable qubit, making the protocols compatible with existing experimental platforms [29–31]. We demonstrate these quantum advantages in proof-ofprinciple superconducting circuit experiments and illustrate their potential broader utility by numerically simulating applications to axionic dark matter search and wireless communication. Our protocols are designed using Quantum Phase-Space Inference (QΨ), a rigorous mathematical framework for experimentally constrained learning tasks. QΨ identifies broad classes of sensing problems for which integrating near-term quantum information processing with modern quantum sensors yields superpolynomial advantages over both classical and conventional quantum methods.

QΨ unifies learning, metrology, and estimation theory by mapping experimentally motivated tasks to precise complexity bounds and optimal quantumenhanced strategies subject to the constraints of the experimental platform. Its central quantity is the accessible feature information (AFI), a phase-space statistical overlap which determines both a tight lower bound on the resources required for a given task and an algorithm that attains this bound under the specified experimental constraints. When QΨ is applied to learn properties of classical signals with quantum control, we refer to the resulting family of sensing strategies as quantum feature sensing (QFS). By extending beyond the local parameter-estimation regime described by quantum Fisher information, the AFI provides a systematic way to identify quadratic, beyondquadratic, and superpolynomial quantum advantages in realistic sensing experiments. QΨ thereby provides an operational theory of quantum experimental advantage in which sensing architectures and learning protocols can be co-designed, encompassing sensing tasks that previously lay beyond the purview of canonical quantum metrology.

![](images/6e1fa829d634988169d87c2f6e628441838b45c6040772d01e5db632acd57244.jpg)  
FIG. 1. Overview of exponential advantages in quantum-enhanced sensing. (a) Architectures for sensing the classical world. Data across experimental science and modern technology is transmitted through classical signals which may have superposed frequencies, time-varying drift, and stochasticity. In these realistic settings we consider sensing with near-term quantum architectures and predicting physical observables such as Fourier components, temporal correlations, and nonlinear transformation of the measured signals. Our QΨ framework encapsulates all of these settings. (b) Hierarchy of quantum-enhanced sensing architectures. As the sensing tasks become richer, sensor architectures with increasing computational capabilities enable a corresponding hierarchy of provable exponential quantum advantages.

## II. Exponential quantum advantage

Many sensing tasks can be formulated as learning a classical signal from the quantum channel that it induces on a quantum sensor [32–36]. Within this framework, Fourier coeficients characterize the coupling of the sensor to distinct phase-space modes of the channel, with higher-order coeficients resolving progressively finer signal structure. We show that coupling a conventional quantum sensor to a single ancilla qubit allows any phase-space Fourier coeficient to be learned exponentially more eficiently than is possible with sensing protocols that do not use quantum information processing.

To formulate this task in a model that applies to contemporary sensing platforms, we consider systems such as precision interferometers that control continuous electromagnetic degrees of freedom. The sensor is described by a quantum continuous-variable mode, and the classical signal acts on a probe state ρ through a displacement channel, $\begin{array} { r } { \rho \mapsto \int \bar { d ^ { 2 } } \alpha P ( \alpha ) D ( \dot { \alpha } ) \rho D ^ { \dagger } ( \dot { \alpha } ) } \end{array}$ Here $\alpha = x + i p$ labels the canonical quadrature coordinates, $D ( \alpha )$ is the displacement operator, and the probability density $P ( \alpha )$ completely characterizes the signal. For example, the kth Fourier coeficient of the signal along the x quadrature is $\mathbb { E } _ { x + i p \sim P } [ e ^ { i k x } ]$

Because a physical signal is available for only a finite duration, the relevant resource is the number of times a sensing protocol must query the displacement channel to estimate properties of $P ( \alpha )$ . For conventional quantum sensors with energy $O ( k )$ , we prove that estimating the $k ^ { \mathrm { t h } }$ Fourier coeficient requires a number of signal queries that grows exponentially with k. Gaussian resources such as squeezing, despite reducing measurement noise, leave this asymptotic scaling unchanged. We show that coupling a sensor that is otherwise capable only of classical operation to a single controllable ancilla qubit reduces the query complexity to O(k):

Theorem 1 (Exponential quantum advantage with a single qubit). A sensing protocol with probe energy O(k), one ancilla qubit, and one control operation can learn the $k ^ { \mathrm { t h } }$ Fourier coeficient of a signal using O(k) signal queries. Any conventional protocol with the same energy scaling requires exp(Ω(k)) queries.

This result is directly relevant to modern quantum sensors with Gaussian probes, including squeezed states that suppress readout variance below the standard quantum limit imposed by the uncertainty principle on unsqueezed probes. Theorem 1 shows that coupling a single controllable qubit to sensors provides a qualitatively new resource, reducing the measurement complexity exponentially rather than further improving the precision of a Gaussian probe. A Gaussian

(c)

![](images/e2351079518a6902f7f2fe32dd80b42453c9e5d92a8ff2e478225a7ff4e8ac6a.jpg)

![](images/0d3618d36af769a39c2011af3bcbb74422ec5747e9748b091025f85d8f8eb3e6.jpg)

![](images/69ee89c0d13efa14c1e14b36899fb75d739722a22e5b659f967af829bce1cd6b.jpg)  
FIG. 2. Exponential quantum advantage of single-qubit control for learning Fourier amplitudes. (a) Visualization of quantum advantage. The signal contains a high-frequency phase-space Fourier component. The pictured distribution of displacements $P ( \alpha )$ , for $k = 2 0$ , is implemented in our experiment. The response of single-qubit quantum feature sensing (top) acts like a frequency comb, probing fine oscillatory structure in each measurement. The response of a conventional Gaussian protocol (bottom) sees the signal severely blurred by Gaussian convolution noise. Plotted is the same signal as seen by $\bar { 1 } 0 ^ { 4 }$ shots from QFS vs. an entangled two-mode squeezed sensor with the same energy. (b) Exponentially improved accuracy in learning Fourier amplitudes using single-qubit QFS. We sense a family of classical signals with $\overset { \circ } { k } ^ { \mathrm { t h } }$ Fourier amplitude of 1.0 using $1 0 ^ { 4 }$ shots. Our experimental implementation of QFS accurately learns the amplitude for all frequencies k, whereas an idealized, noiseless simulation of squeezed homodyne sensing produces erratic estimates. Shaded regions depict one standard deviation of uncertainty in the learned amplitude (pink: QFS, experiment; gray: ideal squeezed homodyne, simulation). $( c )$ Experimental demonstration of exponential advantage. We experimentally realize Theorem 1 using the protocol given in Appendix F 2 and the rigorous lower bound of Appendix E 2. For the signals depicted in (a), single-qubit QFS distinguishes the signals with prediction accuracy 70% using only ∼ 100 shots, a factor of $1 0 ^ { 7 }$ improvement over any conventional Gaussian approach with equal energy. We use a 70% convention throughout, because success probabili $\mathrm { t y } \geq 1 - \delta$ only incurs logarithmic shot overhead $\overset { \vartriangle } { \sim } \log \left( \delta ^ { - 1 } \right)$ ; other values of δ are shown in Appendix B.

protocol can match this query complexity only by increasing the probe energy by a polynomial factor in k. We demonstrate this advantage with a device comprising a transmon qubit coupled to the fundamental electromagnetic mode of a high-purity aluminum cavity for up to $k = 2 0$ in Figure 2. This device is used in all subsequent demonstrations. Even in the presence of decoherence, our quantum-enhanced sensor requires seven orders of magnitude fewer signal queries than a decoherence-free conventional Gaussian protocol.

Having established that a single control operation can provide an exponential advantage for learning classical signals, we next ask whether other quantum processing resources enable comparable gains for different sensing primitives. We organize these possibilities into a hierarchy of quantum sensing architectures, with each level obtained from the previous one by adding a single quantum resource, and examine whether ascending the hierarchy makes progressively richer sensing tasks amenable to quantum advantage. Figure 1(b) summarizes this structure.

Our first result showed how a single controllable qubit can enhance the capabilities of modern quantum sensors. We next ask whether conventional quantum sensors themselves already ofer exponential advantages over fully classical sensing strategies, in which the probe is restricted to classical coherent states of light. We show that they do: sensing with simple nonclassical Gaussian probe states and conventional measurements can yield exponential improvements over all classical sensing strategies.

Theorem 2 (Exponential advantage of quantum probes). A conventional Gaussian sensing protocol with energy O(k) can learn the $k ^ { \mathrm { t h } }$ angular Fourier coeficient of a signal using O(1) queries. Any classical sensing protocol, regardless of available energy, requires exp(Ω(k)) queries to do so.

This separation is proved in Appendix E 1 for angular Fourier coeficients, which are equivalent to moments of the density P. Our experimental demonstration of this result is presented in Figure 3(b), where we implemented a modified quantum protocol with an ideal query complexity of ${ \bar { O ( k ^ { 2 / 3 } ) } }$

Although we have shown that conventional quantum sensors that use quantum oscillators, such as optical or microwave cavities prepared in squeezed states, can achieve dramatic improvements beyond classical limits, they are unprotected quantum degrees of freedom generally limited to performing short-time destructive measurements due to decoherence. While the setting of Theorem 1 considers coupling a sensor to a single physical qubit, future quantum information processing-enabled sensors may have access to encoded logical ancillas that can function both as controls and quantum memories, enabling coherent accumulation of signal over time. This capability aligns with sensing of timedependent processes that are governed by displacement channels over correlated sequences of displacement variables $\alpha _ { \vec { t } } = ( \alpha _ { t _ { 1 } } , \alpha _ { t _ { 2 } } , . . . , \alpha _ { t _ { m } } )$ for times $t _ { 1 } <$ $t _ { 2 } < \cdots < t _ { m }$ The corresponding channel acts as $\begin{array} { r } { \rho \mapsto \int d ^ { 2 m } \alpha _ { \vec { t } } P _ { \vec { t } } ( \alpha _ { \vec { t } } ) D ( \alpha _ { \vec { t } } ) \rho D ^ { \dagger } ( \alpha _ { \vec { t } } ) } \end{array}$ , where $D ( \alpha _ { \vec { t } } )$ is the (time-ordered) product $\textstyle \prod _ { j = 1 } ^ { m } { \dot { D } } ( \alpha _ { t _ { j } } )$ . If $\chi _ { t _ { j } }$ represents the characteristic function of the signal at time $t _ { j } .$ such a time-varying signal is characterized by multi-time correlations of the form $\begin{array} { r } { \int d ^ { 2 m } \alpha _ { \vec { t } } P _ { \vec { t } } ( \alpha _ { \vec { t } } ) \prod _ { j = 1 } ^ { m } \chi _ { t _ { j } } \big ( \alpha _ { t _ { j } } \big ) } \end{array}$ We establish the following:

![](images/9ce0840985f18dc9d4714c783105725008d8401ce5a66addb5582e2d4754f199.jpg)

![](images/c10455d815728071dad21e0b4abc6b59af4f0d20475d26896fd8215c28fe45c7.jpg)

![](images/eaa01639804457369d39bf8ad61285beff4143d33e2bab877fdcf7e6ceb7a5a6.jpg)  
FIG. 3. Exponential quantum advantages with quantum probes and quantum memory. (a) Top: schematic of sensing with classical vs. conventional Gaussian quantum probes. Classical sensors are modeled as preparing coherent probe states, whereas conventional Gaussian sensors can utilize squeezed probes and generaldyne readout. Bottom: schematic of quantum sensing with vs. without quantum memory. A sensor equipped with quantum memory coherently accumulates temporal correlations, whereas a memoryless sensor must measure destructively after each interaction and carry forward only the resulting classical information. (b) Experimental demonstration $o f$ separation between conven tional Gaussian sensor and classical probes. A conventional Gaussian sensor estimates phase-space Fourier coeficients exponentially faster than any strategy using classical coherent-state probes and arbitrarily powerful measurements. An exponential advantage is also achieved by a protocol using a control qubit, detailed in Appendix F 1 b, which we implement in experiment. (c) Experimental demonstration of exponential separation between quantum-enhanced, but memoryless, sensing and sensing with quantum memory. A sensor coupled to a single memory qubit eficiently estimates temporal correlations of a signal which varies across time, while any memoryless strategy of comparable energy and arbitrary processing is exponentially costly. The simulated benchmarks include the Helstrom measurement, which is the optimal binary non-Gaussian measurement at each time step, and squeezed homodyne sensing. Both benchmarks are given five times the energy available to the memory-enabled protocol.

Theorem 3 (Exponential advantage from a single qubit of quantum memory). Let $t _ { 1 } < t _ { 2 } < \cdots < t _ { m }$ be a list of times separated by at most $\Delta$ . A sensor that decoheres on a timescale much shorter than $\Delta$ , when coupled to a single qubit that remains coherent up to time $t _ { m }$ , can estimate the corresponding m-point temporal correlator using O(1) queries to the time-varying signal. By contrast, any protocol with the same energy and no quantum degree of freedom coherent for times much longer than $\Delta$ requires exp(Ω(m)) queries, even given access to arbitrarily many qubits or modes.

This result identifies long-lived quantum memory as a resource that enables eficient characterization of time-dependent signals, with potential applications to radar, noise spectroscopy, and biomedical sensing [37– 39]. In prominent previous works [4, 6], the size of the quantum advantage grows with the number of memory qubits or modes, and an exponential separation requires a memory register whose size increases with the problem. Here, the sensor itself may decohere on a much shorter timescale; repeated interactions with a single coherent qubit are suficient to preserve the accumulated quantum information. To our knowledge, Theorem 3 provides the first exponential quantum advantage obtained solely by adding one qubit of quantum memory. Figure 3(c) presents a proof-of-principle implementation within the coherence times available in our experimental system.

We extend the hierarchy of quantum sensing architectures in the Appendices, and show that modest increases in quantum circuit depth can exponentially improve the estimation of polynomial functionals of the signal distribution $P ,$ the standard objective in matched filtering and estimation of signal moments. We also construct single-mode and multimode sensing problems for which control using a single qubit yields quantum advantages over all Gaussian strategies, even when those strategies have access to arbitrarily large energy and entanglement, and demonstrate that the entanglement-enabled advantages of Refs. [41, 42] can be simplified to entanglement-free quantum feature sensing advantages in many cases. Whereas the demonstration of Ref. [42] focused on full tomography of a particular worst-case multimode displacement channel, the exponential speedups we demonstrate apply to elementary signal-learning primitives, such as Fourier coeficients and temporal correlations, of common single-mode displacement signals.

![](images/52d965aa01adda7da53ed01d1606996ee18ae79af9b99d07e285476dc91bb3cf.jpg)

![](images/3b3564b105c20f65925cfd92a01d22a66fa2564587efdd92275f580b675375a3.jpg)  
FIG. 4. Demonstrating speedups in practical applications through numerical simulations. (a) Characterizing cold axion streams. Daily periodicity in the angle of an incident field would constitute smoking-gun evidence of axionic dark matter. This angle manifests as Gaussian covariance between spatially separated receivers. Using estimated velocity parameters of the Sagittarius stream [40], we simulate QFS vs. the leading practical approaches that do not use longrange, long-lived entangled sensors. The QFS protocol only requires a single qubit that is separately coupled to each detector at diferent times; a realization could use state teleportation of the ancilla, but no long-range-entangled probe is used for sensing. QFS accesses the covariance and resolves the stream angle to within 1<sup>◦</sup> quadratically faster than even highly non-Gaussian photon counting or entangled two-mode squeezing. (b) A quantum-enhanced receiver for wireless communication. Wireless communications commonly encode information by modulating the amplitude and phase of a carrier wavepacket of known frequency. We simulate identification of an unknown transmitted symbol from 8-QAM and 64-QAM communication schemes, common in digital cable television, cellular networks, and satellite communications. We simulate a QFS receiver equipped with three qubits, so that each readout can perform at least 8-class classification. To identify one of 64 candidate symbols from a weak signal, QFS requires ∼ 100× fewer shots than a two-mode-entangled quantum receiver, which requires ∼ 100× fewer shots than a classical heterodyne receiver. For single-shot decoding, QFS achieves large success probabilities at far smaller field strengths.

In all of our results and demonstrations, the signal is a purely classical process inducing Gaussian displacements of the sensor; no exotic non-Gaussianity or nonclassical interaction is engineered. Moreover, our approach allows noise in the signal to be treated as part of the distribution, so that our distribution-agnostic protocols retain exponential advantage in the presence of Markovian, white, or thermal noise backgrounds.

Our quantum advantages and QFS protocols bear on many tasks in experimental science and technology. In Figure 4(a), we apply, in numerical simulations, QFS to the sensing of particle streams from primordial dark matter sources, for which accurate characterization of the incident angle would constitute smoking gun evidence of axionic dark matter [43]. We find that QFS, requiring only a single qubit and no long-range-entangled sensing resource, quadratically outperforms the longstanding leading approaches of squeezed photon-number-resolving measurements and two-mode entangled squeezing, at stream parameters estimated directly from astronomical measurements and calculations [40, 43]. In Figure 4(b), we turn to a modern technological application in wireless communications, where the task is to decode transmitted signals. We simulate transmission of signals from standard 3-bit and 6-bit communication schemes used across digital streaming, cellular networks, and satellite communication and observe that in the weaksignal regime, a shallow-depth QFS receiver identifies a transmitted symbol using a factor of ∼ 100 fewer shots than a receiver based on a two-mode-squeezedvacuum interferometer [44] of equal energy, and ∼ 10<sup>4</sup> fewer shots than a classical receiver. Details of our simulations are given in Appendix B. These examples illustrate that the applicability of QFS is not limited to Theorems 1, 2, and 3; the quantum advantages from our approach may become realizable in a wide range of frontier scientific and technological applications.

## III. Methods

## A. Discovery of advantages with Quantum Phase-Space Inference

Proving an exponential quantum advantage requires both an eficient quantum-enhanced protocol and a lower bound that applies to every strategy in the specified class of conventional sensing protocols, including adaptive measurements with unrestricted classical postprocessing. Such lower bounds have so far been tractable in settings where each conventional measurement reveals only a microscopic amount of information relevant to the task, even when the protocol has access to large energy, substantial quantum depth, or arbitrarily sophisticated postprocessing [4, 6, 11, 41, 45]. These limitations on proving general lower bounds have confined rigorous separations with practical relevance to a narrow range of problems. A general method that derives the conventional lower bound and an eficient quantum-enhanced strategy from the same experimental objective has remained unavailable.

We introduce Quantum Phase-Space Inference (QΨ) to produce optimal lower bounds and eficient quantum-enhanced protocols within a common framework. A QΨ instance consists of a conventional experimental family $\mathcal { M } _ { \mathrm { c o n v } }$ defined by constraints on resources such as energy, entanglement, or non-Gaussianity; a quantum information processingenabled family $\mathcal { M } _ { \mathrm { Q I P } }$ with access to additional resources; and a target physical observable O. From these ingredients, QΨ constructs phase-space representations of the observable and of the response functions generated by protocols in each family.

The central quantity in QΨ is the accessible feature information (AFI), which measures how strongly the response functions available to an experimental family overlap with the phase-space representation of the target observable. For a protocol family M and observable O, the accessible feature information is denoted $\mathsf { A F I } ( \mathcal { M } ; \mathcal { O } )$ ; its formal definition is given in Appendix D 1 b. We prove that any strategy restricted to M, including arbitrary adaptive measurements and classical postprocessing, requires $\Omega ( 1 / \mathsf { A F I } ( \mathcal { M } ; \mathcal { O } ) )$ measurements to learn O. The same phase-space construction identifies a protocol within M that attains this scaling, so the AFI characterizes the optimal measurement complexity of the experimental family.

For a sequence of observables ${ \mathcal { O } } ^ { ( k ) }$ , an exponential advantage of the quantum information processingenabled family over the conventional family occurs precisely when $\mathsf { A F I } ( \mathcal { M } _ { \mathrm { Q I P } } ; \mathcal { O } ^ { ( k ) } ) / \mathsf { A F I } ( \mathcal { M } _ { \mathrm { c o n v } } ; \mathcal { O } ^ { ( k ) } ) \ge$ exp(Ω(k)). When this condition holds, QΨ produces both a tight lower bound for every conventional strategy and an optimal quantum information processingenabled protocol achieving the corresponding upper bound. Once $\mathcal { M } _ { \mathrm { c o n v } } , \mathcal { M } _ { \mathrm { Q I P } }$ , and O have been specified, the quantum learning problem is thereby reduced to a classical statistical comparison of overlaps between real-valued phase-space functions.

QΨ extends the range of settings in which quantum advantages can be identified and proved. For example, quantum learning theory has ofered few general tools for characterizing the learning power of boundeddepth quantum circuits outside full tomography [46]; in Appendix $\mathrm { E 6 } ,$ we use QΨ to derive a fine-grained family of exponential separations between sensing architectures whose circuit depths increase incrementally. Several known separations also follow as special cases of the framework, often with substantially simpler proofs, as illustrated in Appendix E 5 a. QΨ thus provides a systematic method for discovering rigorous quantum advantages from the structure of experimentally motivated learning tasks and the physical constraints imposed on the available platforms.

## B. Experimental demonstration of quantum advantage

The modest control requirements of our protocols allow their experimental realization with existing superconducting-circuit technology. Our device is an ancilla qubit capacitively coupled to a continuousvariable microwave sensor. The qubit is measured via an on-chip readout resonator (see Appendix B for a description of the experiment setup and system Hamiltonian, along with the experimental results). The signal displacement is implemented by applying a calibrated resonant microwave pulse to the continuous-variable mode, while entangling operations are realized by exploiting the cross-Kerr interaction between the qubit and cavity. The control unitaries are composed of single-qubit rotations and echoed conditional displacements, $\mathrm { E C D } ( \beta ) = D ( \beta / 2 ) | g \rangle \langle e | + D ( - \beta / 2 ) | e \rangle \langle g |$ . Together, these operations provide programmable non-Gaussian control and readout of the qubit–cavity system, the key resource underlying the exponential quantum sensing advantages described above.

## IV. Discussion

We have demonstrated that a single qubit can exponentially enhance our ability to sense the classical world. Our results further show that, across a broad range of fundamental sensing tasks, modest amounts of coherent control, quantum memory, or circuit depth are suficient to produce additional exponential advantages. Useful quantum advantages need not rely on large entangled systems or local control across many quantum degrees of freedom, and can instead arise in architectures built from a conventional oscillator and a small number of quantum information-processing resources.

Our experimental demonstrations illustrate that strong single-photon nonlinearities, such as those readily achievable in superconducting circuits [29], enable non-Gaussian sensing protocols that ofer an exponential advantage over Gaussian protocols at equivalent energy, highlighting single-photon nonlinearities as an important resource for quantum sensing alongside their established roles in quantum computing and control [47, 48].

The quantum advantages presented here are produced by the QΨ framework, which provides a unified theory for designing quantum-enhanced experiments from explicit operational constraints. Its central quantity, the AFI, connects conventional hardness with quantum achievability by simultaneously lower-bounding the measurement complexity of protocols within a specified experimental class and identifying strategies that achieve the corresponding scaling. Whereas quantum Fisher information characterizes local sensitivity within a chosen parameterization, the AFI applies to general learning objectives while incorporating restrictions on the available experimental architecture. This broader scope makes it natural to adapt tools from Bayesian experimental design [49], multiparameter optimization [50, 51], and semidefinite programming [15, 52] to the systematic search for experiments with provable advantages beyond the regimes described by quantum Fisher information [53]. Since QΨ recasts the search over sensing architectures and learning protocols as an optimization of classical statistical overlaps, it may provide the foundation for a broader program centered on the computational design of quantum-enhanced experiments with rigorous, certifiable advantages. Appendix H develops these directions in greater detail.

By identifying sensing tasks for which conventional architectures require exponentially more measurements, and by supplying experimentally accessible protocols with improved scaling, our framework opens a path toward quantum-enhanced scientific instruments with provable advantages. When these gains translate into an overall practical advantage depends on the full set of resources relevant to a given application, including achievable state preparation and control as well as experimental overhead. In particular, identifying settings in which QFS protocols outperform conventional sensors requires understanding when the non-Gaussian control or quantum memory available on a given platform provides gains beyond those attainable with comparatively mature Gaussian architectures capable of preparing highly squeezed states. The proof-of-principle experiments reported here already demonstrate reductions of several orders of magnitude in sample complexity under equal-resource comparisons. We further identify several promising domains of application, including fundamental particle detection, noise spectroscopy, and polyspectrum learning, in Appendix H, while a broader range of scientific applications remains to be explored.

Relatively simple quantum control can therefore provide classical learners a quantum-mechanical interface to the natural world that is exponentially more expressive than any classical instrument. Whereas existing exponential separations have relied upon structured quantum features of tailored, worst-case quantum processes [4, 6, 41], our work gives the first examples of provable exponential quantum advantages for learning practical classical signals and stochastic processes. Through the QΨ framework, modest quantum resources allow existing devices to realize these advantages while retaining rigorous performance guarantees. These results provide a path toward quantumenhanced measurements in radar, astronomy, communication, chemistry and other fields in which the features of interest are encoded in classical signals.

[1] S. Aaronson, Shadow Tomography of Quantum States, in Proceedings of the 50th Annual ACM SIGACT Symposium on Theory ofComputing (STOC 2018) (2018).

## Data and code availability

All code and data produced during the experiments and numerical simulations in this work are available at https://doi.org/10.5281/zenodo.21896087.

## Acknowledgements

We thank Senrui Chen and Luke Cofman for helpful comments and are grateful to the organizers of the EPFL workshop on quantum learning and sensing for stimulating discussions. IK is supported in part by the Nobile Research Initiative. JC is supported by a fellowship from the Alfred P. Sloan Foundation. The authors would also like to thank Bradley Cole, Clayton Larson, Britton Plourde, Eric Yelton, and Luojia Zhang for the fabrication of the transmon and on-chip resonator, Chris Wang for the design of the transmon, the on-chip resonator and the 3D superconducting cavity, and Nord Quantique for the fabrication of the 3D superconducting cavity. We gratefully acknowledge the Army Research Ofice for support under award number W911NF-25-1-0261; MIT Lincoln Laboratory for supplying the Josephson travelingwave parametric amplifier (TWPA) used in our experiments; and the Air Force Ofice of Scientific Research for equipment purchased using a DURIP award with award number FA9550-22-1-0080.

## Author contributions

I.K. conceived the main theoretical ideas underlying this work and conducted the theoretical investigation. S.P. and I.K. designed the experiments, with contributions from S.A.K. and X.S.. S.P. conducted the experiments. M.M.S. suggested the protocol in Appendix F 2 a. V.F. oversaw the design and fabrication of the superconducting devices by S.R. and others, which were characterized by S.R., A.S. and S.P.. I.K., S.P., J.C., P.L.M. wrote the manuscript with input from all authors. J.C. and P.L.M. supervised the work.

[2] J. Cotler and F. Wilczek, Quantum Overlapping Tomography, Physical Review Letters 124, 100401 (2020).

[3] H.-Y. Huang, R. Kueng, and J. Preskill, Predicting many properties of a quantum system from very few

measurements, Nature Physics 16, 1050–1057 (2020).

[4] S. Chen, J. Cotler, H.-Y. Huang, and J. Li, Exponential separations between learning with and without quantum memory, in 2021 IEEE 62nd Annual Symposium on Foundations of Computer Science (FOCS) (IEEE, 2022) pp. 574–585.

[5] D. Aharonov, J. Cotler, and X.-L. Qi, Quantum algorithmic measurement, Nature Communications 13 (2022).

[6] H.-Y. Huang, M. Broughton, J. Cotler, S. Chen, J. Li, M. Mohseni, H. Neven, R. Babbush, R. Kueng, J. Preskill, and others, Quantum advantage in learning from experiments, Science 376, 1182–1186 (2022).

[7] S. Chen, J. Cotler, H.-Y. Huang, and J. Li, The Complexity of NISQ, Nature Communications 14, 6001 (2023).

[8] S. F. Huelga, C. Macchiavello, T. Pellizzari, A. K. Ekert, M. B. Plenio, and J. I. Cirac, Improvement of Frequency Standards with Quantum Entanglement, Physical Review Letters 79, 3865–3868 (1997).

[9] V. Giovannetti, S. Lloyd, and L. Maccone, Advances in quantum metrology, Nature Photonics 5, 222–229 (2011).

[10] C. Degen, F. Reinhard, and P. Cappellaro, Quantum sensing, Reviews of Modern Physics 89 (2017).

[11] J. Cotler, W. Gong, and I. Kannan, Noisy quantum learning theory, Nature Communications (2026).

[12] V. Giovannetti, S. Lloyd, and L. Maccone, Quantum-Enhanced Measurements: Beating the Standard Quantum Limit, Science 306, 1330–1336 (2004).

[13] V. Giovannetti, S. Lloyd, and L. Maccone, Quantum Metrology, Physical Review Letters 96 (2006).

[14] R. Demkowicz-Dobrzański, J. Kołodyński, and M. Guţă, The elusive Heisenberg limit in quantumenhanced metrology, Nature Communications 3 (2012).

[15] S. Zhou, M. Zhang, J. Preskill, and L. Jiang, Achieving the Heisenberg limit in quantum metrology using quantum error correction, Nature Communications 9 (2018).

[16] B. Allen, W. G. Anderson, P. R. Brady, D. A. Brown, and J. D. E. Creighton, FINDCHIRP: An algorithm for detection of gravitational waves from inspiraling compact binaries, Physical Review D 85 (2012).

[17] M. Tsang, H. M. Wiseman, and C. M. Caves, Fundamental Quantum Limit to Waveform Estimation, Physical Review Letters 106 (2011).

[18] G. A. Álvarez and D. Suter, Measuring the Spectrum of Colored Noise by Dynamical Decoupling, Phys. Rev. Lett. 107, 230501 (2011).

[19] H. J. Mamin, M. Kim, M. H. Sherwood, C. T. Rettner, K. Ohno, D. D. Awschalom, and D. Rugar, Nanoscale Nuclear Magnetic Resonance with a Nitrogen-Vacancy Spin Sensor, Science 339, 557 (2013).

[20] K. M. Backes, D. A. Palken, S. A. Kenany, B. M. Brubaker, S. B. Cahn, A. Droster, G. C. Hilton, S. Ghosh, H. Jackson, S. K. Lamoreaux, and others, A quantum enhanced search for dark matter axions, Nature 590, 238–242 (2021).

[21] H.-M. Chin, N. Jain, D. Zibar, U. L. Andersen, and T. Gehring, Machine learning aided carrier recovery in continuous-variable quantum key distribution, npj Quantum Information 7 (2021).

[22] S. Chen, J. Cotler, and H.-Y. Huang, Quantum Probe Tomography (2025), arXiv:2510.08499.

[23] J. Cotler, D. L. Danielson, and I. Kannan, Quantum Advantage for Sensing Properties of Classical Fields (2026), arXiv:2602.17591.

[24] S. Prabhu, S. A. Khan, X. Song, M. Ouellet, R. Yanagimoto, S. Roy, A. Senanian, L. G. Wright, V. Fatemi, and P. L. McMahon, Quantum computational displacement sensing (2026), arXiv:2604.13177.

[25] S. A. Khan, S. Prabhu, L. G. Wright, and P. L. McMahon, Quantum Computational-Sensing Advantage (2025).

[26] S. A. Khan, S. Prabhu, L. G. Wright, and P. L. McMahon, Quantum computational sensing using quantum signal processing, quantum neural networks, and Hamiltonian engineering, npj Quantum Information (2026).

[27] S. Prabhu, V. Kremenetski, S. A. Khan, R. Yanagimoto, and P. L. McMahon, Exponential advantage in quantum sensing of correlated parameters, Quantum 10, 1963 (2026).

[28] C. C. V. de Pradenne, I. Kannan, H. Putterman, and J. Cotler, Restrictions on non-Cliford fault tolerance and ruling out beyond-SQL quantum metrology (2026), arXiv:2607.27342.

[29] P. Krantz, M. Kjaergaard, F. Yan, T. P. Orlando, S. Gustavsson, and W. D. Oliver, A quantum engineer’s guide to superconducting qubits, Applied Physics Reviews 6 (2019).

[30] C. D. Bruzewicz, J. Chiaverini, R. McConnell, and J. M. Sage, Trapped-ion quantum computing: Progress and challenges, Applied Physics Reviews 6 (2019).

[31] M. Aspelmeyer, T. J. Kippenberg, and F. Marquardt, Cavity optomechanics, Reviews of Modern Physics 86, 1391 (2014).

[32] D. Thomson, Spectrum estimation and harmonic analysis, Proceedings of the IEEE 70, 1055 (1982).

[33] B. Allen and J. D. Romano, Detecting a stochastic background of gravitational radiation: Signal processing strategies and sensitivities, Physical Review D 59 (1999).

[34] S. Haykin, Cognitive radio: brain-empowered wireless communications, IEEE Journal on Selected Areas in Communications 23, 201 (2005).

[35] A. A. Clerk, M. H. Devoret, S. M. Girvin, F. Marquardt, and R. J. Schoelkopf, Introduction to quantum noise, measurement, and amplification, Reviews of Modern Physics 82, 1155–1208 (2010).

[36] J. Bylander, S. Gustavsson, F. Yan, F. Yoshihara, K. Harrabi, G. Fitch, D. G. Cory, Y. Nakamura, J.-S. Tsai, and W. D. Oliver, Noise spectroscopy through dynamical decoupling with a superconducting flux qubit, Nature Physics 7, 565 (2011).

[37] W. Melvin, A STAP overview, IEEE Aerospace and Electronic Systems Magazine 19, 19 (2004).

[38] E. M. Buckley, A. B. Parthasarathy, P. E. Grant, A. G. Yodh, and M. A. Franceschini, Difuse correlation spectroscopy for measurement of cerebral blood flow: future prospects, Neurophotonics 1, 011009 (2014).

[39] L. M. Norris, G. A. Paz-Silva, and L. Viola, Qubit Noise Spectroscopy for Non-Gaussian Dephasing Environments, Physical Review Letters 116 (2016).

[40] D. Lynden-Bell and R. M. Lynden-Bell, Ghostly streams from the formation of the Galaxy’s halo, Mon. Not. Roy. Astron. Soc. 275, 429 (1995).

[41] C. Oh, S. Chen, Y. Wong, S. Zhou, H.-Y. Huang, J. A. Nielsen, Z.-H. Liu, J. S. Neergaard-Nielsen, U. L. Andersen, L. Jiang, and others, Entanglement-Enabled Advantage for Learning a Bosonic Random Displacement Channel, Physical Review Letters 133 (2024).

[42] Z.-H. Liu, R. Brunel, E. E. B. Østergaard, O. Cordero, S. Chen, Y. Wong, J. A. H. Nielsen, A. B. Bregnsbo, S. Zhou, H.-Y. Huang, and others, Quantum learning advantage on a scalable photonic platform, Science 389, 1332 (2025).

[43] J. W. Foster, Y. Kahn, R. Nguyen, N. L. Rodd, and B. R. Safdi, Dark matter interferometry, Physical Review D 103 (2021).

[44] S. L. Braunstein and H. J. Kimble, Dense coding for continuous variables, Physical Review A 61, 042302 (2000).

[45] I. Kannan, H. Putterman, and J. Cotler, Exponential speedups in fault-tolerant processing of quantum experiments (2026), arXiv:2605.02057.

[46] H. Zhao, L. Lewis, I. Kannan, Y. Quek, H.-Y. Huang, and M. C. Caro, Learning Quantum States and Unitaries of Bounded Gate Complexity, PRX Quantum 5 (2024).

[47] R. Yanagimoto, E. Ng, M. Jankowski, R. Nehra, T. P. McKenna, T. Onodera, L. G. Wright, R. Hamerly, A. Marandi, M. Fejer, and others, Mesoscopic ultrafast nonlinear optics—the emergence of multimode quantum non-Gaussian physics, Optica 11, 896 (2024).

[48] H. J. Kimble, Strong interactions of single atoms and photons in cavity QED, Physica Scripta 1998, 127 (1998).

[49] K. Macieszczak, M. Fraas, and R. Demkowicz-Dobrzański, Bayesian quantum frequency estimation in presence of collective dephasing, New Journal of Physics 16, 113002 (2014).

[50] P. C. Humphreys, M. Barbieri, A. Datta, and I. A. Walmsley, Quantum Enhanced Multiple Phase Estimation, Physical Review Letters 111 (2013).

[51] S. Ragy, M. Jarzyna, and R. Demkowicz-Dobrzański, Compatibility in multiparameter quantum metrology, Physical Review A 94 (2016).

[52] F. Albarelli, J. F. Friel, and A. Datta, Evaluating the Holevo Cramér-Rao Bound for Multiparameter Quantum Metrology, Physical Review Letters 123 (2019).

[53] R. Demkowicz-Dobrzański, W. Górecki, and M. Guţă, Multi-parameter estimation beyond quantum Fisher information, Journal of Physics A: Mathematical and Theoretical 53, 363001 (2020).

## Appendices

## Contents & Roadmap

A. Overview and related works 12   
1. Quantum learning theory 12   
2. Quantum sensing 13   
B. Experiment and simulation details and results 15   
1. Experimental setup, system Hamiltonian and parameters 15   
2. Demonstration of separation between classical and quantum probes 17   
3. Demonstration of separation between conventional quantum experiments and quantum control 19   
4. Demonstration of quantum memory advantage 21   
5. Simulation of cold axion stream characterization 23   
6. Simulation of wireless communication receiver 26   
C. Preliminaries and definitions 30   
1. Continuous-variable quantum information 30   
2. Gaussian learning and estimation theory 34   
3. Formalizing sensing of classical signals through Quantum Signal Learning 39   
4. A hierarchy of conventional and quantum-enhanced sensing 41   
D. Quantum Phase-Space Inference (QΨ) 44   
1. Building the language of QΨ 44   
a. Experiments, observables, and uncharacterized systems 44   
b. Defining the Accessible Feature Information (AFI) 48   
c. Information-theoretic lower bound tools 50   
2. QΨ lower bounds 53   
3. QΨ hypothesis testing guarantee 56   
4. AFI as certificate of advantage and global learning 58   
5. Using QΨ to discover quantum advantages 62   
E. Hardness of learning from constrained experiments 64   
1. Hardness of learning with classical probes 64   
2. Hardness of learning with conventional quantum sensing 67   
3. Hardness of learning time-dependent signals without memory 71   
a. Time-varying hypothesis testing for frequency-k, two-point correlation 71   
b. Constant-frequency, m-point correlation estimation 73   
4. Lower bound for infinite-energy conventional quantum sensing 77   
5. Multimode hardness of conventional characteristic function learning 79   
a. Simplified entanglement-free lower bound on multimode channel learning through QΨ 81   
6. A hierarchy of exponential sensing lower bounds with increased control depth 82   
a. Fourier response of depth-r measurements and hypothesis testing 83   
b. Proof of Theorem E.13 92   
F. Exponential advantage in learning signals with Quantum Feature Sensing 93   
1. Learning with quantum probes without quantum control 93   
a. Fixed-amplitude learning with squeezed probes 93   
b. Experimental proxy protocol 99   
2. Learning with quantum control 100   
a. Alternative approach using SU(1, 1) interferometry or Gaussian boson sampling 105   
3. Learning with quantum control and memory 107   
4. A quadratic advantage over infinite-energy conventional quantum sensing 110   
5. Entanglement-free multimode advantage for learning characteristic functions with one qubit 112   
a. Displacement sensing with quantum phase estimation and GKP states 112   
b. Characteristic-function learning from phase estimation 117   
6. Learning with depth-k quantum control 120   
G. Self-contained list of results 122   
H. Open directions 125   
1. Applications of Quantum Feature Sensing 125   
a. Fundamental particle searches 125   
b. Stochastic and spatiotemporal noise learning 126   
c. Estimating polyspectra 126   
2. QΨ and future quantum advantages in experimental science 127   
a. Multiparameter QΨ and optimization 127   
b. Automating the discovery of quantum advantages 128

## Roadmap

Our main takeaways can be understood from just a few short sections of these appendices. We first provide a roadmap, then recommend a guided reading path through selected sections. A high-level understanding of each section can be obtained by reading the gray boxes, and readers can return to this guide by clicking on the page number at the top-right of each page.

In Section A, we place this paper in the broader context of quantum learning theory and metrology. In Section B, we provide experimental details. In Section C, we provide preliminaries in quantum information and statistical learning theory, then lay out the mathematical models of sensing architectures used throughout our work. In Section D, we introduce the central QΨ framework. In Sections E and F we apply our QΨ machinery to prove separations between architectures at increasing levels of the sensing hierarchy. Section G is a short, self-contained reference of all of our results. In Section H, we give concrete statements of research programs that emerge from this work.

Reader’s guide. The QΨ framework introduced in this work, and its applications to sensing classical signals, are the central theoretical contributions of our work. We believe that this framework may motivate new research directions in quantum metrology and learning theory, and we recommend the following reading path for understanding the core concepts without delving into the full technical details.

1. Contextualize this work: Begin with Section A. This section contextualizes the timeliness of our work in the history of quantum learning and metrology, elucidates how QΨ unifies these fields, and lays out the natural research programs that arise from this unification.

2. Understand the scope: Skim Section C 3 to understand how we model general sensing tasks and observables in a common framework. Then read Section C 4 to understand the hierarchy of sensing architectures, which frames the scope of our separations.

3. Understand QΨ: Read Section C 1, “Wigner functions and Weyl symbols”, for important intuition on QΨ. Then jump to Section D, and read the gray boxes. These are the workhorse definitions and theorems that drive QΨ, including the definition of the AFI, the upper and lower bound guarantees, the general QΨ learning algorithm, and a template for using QΨ to design quantum advantages. Readers interested in applying QΨ to their own research can then read Section D in further detail.

4. Work through an example: Readers interested in additional detail can skim one of our several exponential separations as a worked example of QΨ. Appendices E and F contain the exponential lower bounds and eficient upper bounds respectively; each subsection is paired by number across appendices. We suggest reading the paired Sections E 2 and F 2, which together prove Theorem 1.

5. Recap: A short list of the ten main results of this work, including the QΨ toolkit, is provided in Section G.

6. Explore future directions: Finally, jump to Section H, where we lay out concrete statements of open questions at the intersection of quantum metrology, learning theory, and experimental physics that arise from our work. These include applications of our sensing protocols to disciplines including fundamental physics, cosmology, signal detection, and device characterization, as well as a broader program aimed at automating the discovery of practical quantum advantages.

Readers with more specific interests may also consider the following sections. For experimental details, refer to Section B. Readers with background in bosonic quantum learning theory can refer to Sections E 5 and F 5, where we show that the well-known entanglement-enabled learning advantage presented in Ref. [1] can in fact be achieved without entanglement for most instances, and instead reduces to a separation between Gaussian and non-Gaussian strategies. We give a single-qubit-enabled algorithm which realizes this separation. Moreover in Section E 5 a, we show how QΨ simplifies the lower bound proof of Ref. [1].

# Appendix A: Overview and related works

## 1. Quantum learning theory

A motivating thesis of modern quantum learning theory (QLT) is that quantum information processing can enable scientific discoveries that are otherwise impossible. Early work in the field, inspired by rapid development of experimental quantum platforms, was often aimed at state-tomographic tasks that would enable eficient readout of quantum data from a quantum device or future quantum simulation [2–10]. Over the last decade, this operational goal has evolved into an organizing principle: using quantum information processing in unison with statistical learning theory to accelerate scientific discovery while obtaining provable guarantees of quantum advantage [11, 12]. Thus far, however, the thesis has been established more as an epistemological claim than a practically actionable one.

Seminal works have shown that there exist quantum states and processes which encode data that is provably more dificult to learn without quantum information-processing resources. The relevant data ranged from a few specific observables, such as fidelity or purity, up to tomographically complete descriptions of a state or channel, with separations most often enabled by large quantum memories and entanglement [12–19]. Intended as proofs of concept rather than practical use cases of QLT, these results often assumed worst-case classes of states or channels that seldom align with experimental reality, alongside noiseless, arbitrarily fast quantum processing and unbounded classical computation [18].

The natural, and thus far unestablished, refinement of the core thesis is therefore the question: can quantum information processing enable large speedups in experimentally useful applications? Recent work has clarified that answering it is not a matter of addressing minor details in the existing literature: in most cases, the idealized assumptions that enable an exponential separation lie at its core, and the advantage does not persist once they are relaxed [18, 19]. The hard instances are frequently ensembles of highly entangled states or processes that do not arise in the experiments one wishes to enhance, and relaxing this structure causes the separation to disappear, while many separations scale only in the number of qubits and apply only to observables, such as purity, whose hardness derives from worst-case structure rather than from physically occurring signals.

Moreover, instantiating quantum information processing in real-world experiments requires interfacing it with Nature through a quantum sensor: the bridge between the natural system under study and the processor. Practical quantum learning advantages must therefore operate within the constraints of sensing platforms, whose coherence and control are far more limited than the capabilities assumed in QLT [18, 20–22]. When the necessary capabilities include deep multi-qubit control operations between sensors and processors, any realization is placed far in the future.

More recent lines of work have extended QLT from the qubit setting to bosonic and fermionic systems, which describe many of the platforms and signals encountered outside of quantum simulation [1, 23–28]. These extensions constitute important progress: they demonstrate that learning separations are not artifacts of qubit structure, develop the phase-space and Gaussian machinery on which parts of the present work directly build, and sharpen the understanding of entanglement and quantum memory as learning resources in continuous variable settings. Their objectives, however, have naturally mirrored those of the founding qubit results, ful tomography of multimode Gaussian states [28], for instance, or worst-case learning of displacement channels [1], which are rarely the quantities most relevant to experimental practice and often presume control beyond what sensing platforms currently provide except for proof-of-concept demonstrations [27, 29]. The gap between provable learning advantage and experimental utility thus persists across qubit, bosonic, and fermionic settings, and closing it requires rethinking how the learning tasks themselves are formulated.

Part of why this gap has persisted is structural. A provable advantage requires two ingredients that work against one another: an eficient quantum-enhanced protocol, which favors instances with clean, exploitable structure, and a lower bound applying to every conventional strategy, including adaptive measurements with unrestricted classical postprocessing, which favors instances that conceal information from an entire measurement class. Satisfying both simultaneously has historically forced constructions organized around the available proof techniques (such as the learning tree formalism [15] or Fano’s inequality [30, 31]) rather than around any experimental setting, so that quantum learning advantages have been discovered through manual worst-case construction, with experimental relevance assessed only after the fact.

What is needed, then, is a single framework into which realistic experimental constraints and objectives can be inserted, and out of which tight separations — matching upper and lower bounds — emerge simultaneously, so that finding quantum learning advantages becomes experimentally guided discovery rather than manual worst-case construction. Such a framework must moreover interface naturally with quantum sensing, the point of contact between quantum information processing and the natural world, while accounting for a range of experimental constraints more amenably than the learning tree formalism. QΨ accomplishes all of this at once: an instance encodes the learning objective, the admissible conventional and quantum-enhanced experiments, and their operational constraints, and its central quantity, the accessible feature information (AFI), simultaneously lower-bounds every conventional protocol and certifies a matched strategy attaining the same scaling. Realistic constraints appear as natural restrictions on the Fourier support of real-valued phase-space functions, making the obstructions they pose quantitative and transparent. The results of the present work are correspondingly diferent in character from the separations surveyed above: to our knowledge, they are the first exponential quantum advantages in learning that use only a single qubit and modest quantum control, and they apply to basic experimental tasks such as estimating Fourier coeficients and temporal correlations of classical signals. The separations also apply to realistic signal instances: our hard instances merely isolate particular Fourier frequencies and temporal correlations, and the lower bounds continue to apply in practice because realistic signals generically contain these hard components; their hardness relies only on standard properties such as high-frequency and multi-time structure, not on highly entangled constructions that do not appear in nature.

## 2. Quantum sensing

Modern quantum metrology developed alongside advances in quantum optics and atomic interferometry, together with a quantum information-theoretic formulation of sensing in terms of parameterized unitary dynamics [32–37]. As the study of quantum computation with qubit systems grew more prominent, these metrological concepts whose roots lay in continuous-variable systems transferred naturally to the qubit DC and AC phase-sensing settings often studied today [37–39]. This quantum metrological reformulation enabled a unified asymptotic theory of quantum sensing, and at its core lay the canonical quadratic separation between the standard quantum limit (SQL) achieved by entanglement-free sensors and the Heisenberg limit (HL) achieved by ideal, maximally-entangled quantum sensors. These seminal results spurred a long line of work delineating when quantum mechanics could enable a sensing speedup, largely centered around the mathematical formalism of quantum Fisher information and quantum Cramer-Rao bounds [21, 40–42]. Bosonic sensing also continued to develop, often in task-specific manners motivated by major experimental objectives like gravitational-wave detection [43–45].

However, an asymptotic obstruction to the canonical quadratic speedup was clear: when Markovian noise present during sensing is supported by the same generators as the signal, the Heisenberg speedup generally reduces to a constant factor [20, 46–48]. Resolutions and even ideas for beyond-Heisenberg limited metrology were proposed, drawing upon special physical phenomena such as quantum criticality and nonlinear phaseshift interactions, but these apparent workarounds often broke down once all physical resources were counted carefully [49–52], hence why our careful accounting of mean photon number is essential in this work. Quantum error correction then emerged as a natural route through which to understand the limits of canonical quantum metrology in realistic noise environments, and enabled a study of when noise can be corrected without also erasing the signal [53, 54].

Recent work, however, has clarified two important limitations of the canonical quantum metrology framework [48]. Firstly, the Cramer-Rao framework cannot comprehensively be used to certify or rule out quantum metrological speedups, because it assumes it operates within a diferentiable statistical model and is concerned with unbiased estimation. In machine learning and statistics, biased estimators can surpass many of the limitations of unbiased estimators; moreover, while the mathematical formalism of metrology utilizes unbiasedness to simplify analysis, statistical learning theory has developed many techniques to understand the resource cost of biased estimators. This observation suggests a natural unification between quantum sensing and quantum learning theory. The second observation of Ref. [48] is that for the canonical DC and AC sensing problems which have become the centerpiece of theoretical quantum metrology, asymptotic beyond-SQL sensing can be ruled out whenever signal and noise are aligned, agnostic to the assumptions of Cramer-Rao.

Taken together, these developments show that quantum metrology has progressed through several distinct paradigms, each organized around a sharper understanding of the physical resources responsible for quantum advantage. The limitations of the canonical phase-sensing problem are now suficiently well understood to motivate a new question: which experimentally natural sensing problems admit asymptotic quantum advantages once the learning objective and the available experimental architecture are treated as part of the problem itself? Rather than seeking further ways to recover Heisenberg scaling within one metrological model, one may ask whether diferent sensing objectives support advantages whose mechanisms and resource scalings are qualitatively distinct from coherent phase accumulation and which scale beyond-quadratically.

Realistic sensing experiments motivate a new, broader formulation. The signal generator may be only partially known, the interrogation time fixed by the source or sensor, and the objective a Fourier coeficient, temporal correlation, or another functional of a stochastic or multiparameter signal. Any practical advantage must also respect constraints on energy, coherence, quantum control, and accessible non-Gaussian operations. These considerations arise across bosonic, qubit, and distributed sensing platforms, but have largely been treated case by case because no encompassing theoretical framework captures both the learning task and the experimental architecture in a unified way. Quantum Fisher information remains efective for quantifying local sensitivity within a specified diferentiable model, but it does not by itself formulate global learning objectives or lowerbound every protocol obeying a collection of operational constraints. Such questions generally require separate analyses of the signal model, probe family, and accessible measurements, making it dificult to compare resources such as squeezing, non-Gaussian control, quantum memory, entanglement, and coherent communication within one theory.

The QΨ framework is designed to fill this role and unify a wide range of problems across quantum learning, metrology, and estimation theory, including their practical experimental constraints, in a single operational picture. A QΨ instance specifies the signal family, learning objective, admissible conventional and quantumenhanced experiments, and the physical resource to be optimized. Once the learning task and its admissible experiments can be represented mathematically, QΨ absorbs their complexity into a purely classical inference problem. The AFI then simultaneously lower-bounds every protocol in the specified class and identifies a matched strategy attaining the same scaling, with constraints on energy, memory, control, and measurement entering directly through the optimization domain. This construction provides a common language for quantum learning and quantum metrology. The present work is one application of QΨ to the problem of bosonic sensing of classical fields, where the QSL phase-space representation introduced in Ref. [55] provides the mathematical interface to convert arbitrary signals, objectives, and architectural constraints into a form that can be processed by QΨ.

Several open problems in quantum sensing of classical fields admit natural formulations within our framework. In non-Gaussian noise spectroscopy, one may ask for the measurement complexity of estimating a selected higher-order correlation or polyspectral coeficient under constraints on pulse bandwidth, sensor coherence, and quantum memory [56–58]. In broadband searches for weak stochastic signals such as in axion or gravitational wave detection, one may compare conventional, squeezed, entanglement-assisted, and non-Gaussian receivers while measuring performance through discovery probability, localization error, scan bandwidth, and total integration time [59–62]. Our work illustrates that even rudimentary quantum control is a qualitatively new resource that can bring new insights to these well-studied problems.

Although bosonic phase-space structure makes the present application especially direct, QΨ is not restricted to continuous-variable systems. For a qubit sensor or sensor network, the unknown instance may specify a family of Hamiltonians or quantum channels, and the objective may be a global property rather than the full parameter vector. A natural next step is to develop a QSL formalism for qubit sensors and apply QΨ to quantum sensing with qubits. This can enable the study of several tasks beyond phase sensing, such as identifying a weak AC signal from a bank of temporal templates, estimating a spatial Fourier component or nonlinear feature of a distributed field, learning a higher-order cross-spectrum, and directly learning a selected property of an unknown many-body Hamiltonian rather than first learning all of its coeficients [63–66]. The corresponding protocol classes may restrict entanglement, coherent control, quantum communication, memory, or accessible measurements. An AFI analysis could then determine whether these resources change the scaling with the number of candidate signals, correlation order, network size, or desired accuracy, rather than only the constant or quadratic precision improvement associated with canonical phase estimation.

Appendix B: Experiment and simulation details and results

## 1. Experimental setup, system Hamiltonian and parameters

![](images/c7162eabdfbb69897d1eab40e7712d709e817601dd43c38b55612ad489e1a17f.jpg)  
FIG. 1. Wiring Diagram of the experiment. Experiment setup for microwave hardware and cabling for the device used in this work.

In this work, we use a transmon qubit coupled to the fundamental mode of a 3D stub-post cavity made of high-purity 4N Aluminum. The transmon qubit (the same device used in Ref. [67]) is fabricated on resistive silicon chip, and is made of Niobium, with an Aluminum - Aluminum Oxide Josephson junction. The chip also contains a readout resonator also made out of Niobium. This chip is inserted in the Aluminum cavity and is held by a copper clamp thermalized to the cold finger made of Oxygen-free high conductivity (OFHC) copper (see Fig. 2). The cold finger is enclosed in a Berkeley black coated copper shield and an Aluminum shield. The outermost can of the dilution fridge is lined with a cryoperm shield on the inside. The device is mounted to the cold finger at the base plate of a dilution fridge and is cooled to 10 mK. Details of the cryogenic setup are shown in Fig. 1.

Microwave pulses are generated using Zurich Instruments (ZI) HDAWG at baseband. Acquisition of microwave pulses from the devices are received by ZI UHFQA. All up and down conversions are performed using Rohde & Schwarz SGS100A, with in-built IQ mixers. The readout pulse generation and acquisition use the same SGS100A local oscillator (using a splitter). This is done with external Marki IQ mixers (MMIQ-0416LSM-2). The pump for the traveling-wave Parametric Amplifier (TWPA) is gated with a marker line from the ZI

![](images/0df41c8e63f93a26cc1029458cc540ef8231181515145130782f9a7b83f4059b.jpg)  
FIG. 2. Schematic illustration and photographs. (a) Schematic illustration of the superconducting device, showing the 3D oscillator, the transmon qubit, and the transmon’s readout resonator (along with their corresponding drive ports). (b) Photographs of the device and the cold finger it is mounted on.

HDAWG which is sent to the corresponding SGS100A which produces the pump signal. The readout signal, after amplification by the TWPA, is further amplified by a high-electron mobility transistor (HEMT) at the 4K stage (LNF-LNC0.3\_14B from Low Noise Factory), and finally at room temperature outside the fridge with a high gain amplifier (ZVA-1W-10 3+ from Mini Circuits). The microwave lines into and out of the fridge are filtered with a series of DC blocks and Mini Circuits bandpass filters centered around the frequency of the tones. For qubit readout, a constant pulse of 0.5 µs duration is sent, and the signal is measured for 1 µs. The TWPA is gated for 2 µs. We wait for 1 ms between runs of the protocol to provide plenty of time for the qubit to thermalize to the ground state, and the oscillator to the vacuum state.

<table><tr><td rowspan=1 colspan=1>Parameter</td><td rowspan=1 colspan=1>Value</td></tr><tr><td rowspan=1 colspan=1>Transmon g-e transition frequency</td><td rowspan=1 colspan=1> $\omega _ { q } = 2 \pi \times 7 . 3 3 2 ~ \mathrm { G H z }$ </td></tr><tr><td rowspan=1 colspan=1>Transmon self-Kerr</td><td rowspan=1 colspan=1> $K = 2 K _ { q } = 2 \pi \times 3 4 0$ MHz</td></tr><tr><td rowspan=1 colspan=1>Transmon relaxation time</td><td rowspan=1 colspan=1> $T _ { 1 } = 3 0 ~ \mu \mathrm { s }$ </td></tr><tr><td rowspan=1 colspan=1>Transmon Ramsey coherence time</td><td rowspan=1 colspan=1> $T _ { 2 R } = 3 0 ~ \mu \mathrm { s }$ </td></tr><tr><td rowspan=1 colspan=1>Transmon Echo coherence time</td><td rowspan=1 colspan=1> $T _ { 2 E } = 4 0 ~ \mu \mathrm { s }$ </td></tr><tr><td rowspan=1 colspan=1>Transmon readout fidelity for |g〉</td><td rowspan=1 colspan=1>95%</td></tr><tr><td rowspan=1 colspan=1>Transmon readout fidelity for |e〉</td><td rowspan=1 colspan=1>95%</td></tr><tr><td rowspan=1 colspan=1>Readout frequency</td><td rowspan=1 colspan=1> $\omega _ { r }$ = 2π × 8.920 GHz</td></tr><tr><td rowspan=1 colspan=1>Readout-Transmon dispersive shift</td><td rowspan=1 colspan=1> $\chi _ { r q } \sim 2 \pi \times 1$ MHz</td></tr><tr><td rowspan=1 colspan=1>TWPA pump frequency</td><td rowspan=1 colspan=1> $\omega _ { \mathrm { p u m p } } = 2 \pi \times 8 . 5 5 0 ~ \mathrm { G H z }$ </td></tr><tr><td rowspan=1 colspan=1>Oscillator frequency</td><td rowspan=1 colspan=1> $\omega _ { a } = 2 \pi \times 6 . 0 7 2$ GHz</td></tr><tr><td rowspan=1 colspan=1>Oscillator-Transmon dispersive shift</td><td rowspan=1 colspan=1> $\chi = 2 \pi \times ( 1 5 . 5 \pm 0 . 3 )$ kHz</td></tr><tr><td rowspan=1 colspan=1>Oscillator relaxation time</td><td rowspan=1 colspan=1> $T _ { 1 c } = 2 5 0 ~ \mu \mathrm { s }$ </td></tr></table>

TABLE I. Measured experiment parameters and readout fidelity. These parameters are obtained using standard spectroscopic and time-resolved measurements following [68, 69].

For the experiments demonstrated in this work, the Hamiltonian of our device is well-described by:

$$
\hat { H } / \hbar = \omega _ { q } \hat { q } ^ { \dagger } \hat { q } + \omega _ { a } \hat { a } ^ { \dagger } \hat { a } - K _ { q } \hat { q } ^ { \dagger 2 } \hat { q } ^ { 2 } - K _ { a } \hat { a } ^ { \dagger 2 } \hat { a } ^ { 2 } - \chi \hat { q } ^ { \dagger } \hat { q } \hat { a } ^ { \dagger } \hat { a } - \chi ^ { \prime } \hat { q } ^ { \dagger } \hat { q } \hat { a } ^ { \dagger 2 } \hat { a } ^ { 2 } + \Omega ( t ) \hat { q } ^ { \dagger } + \varepsilon ( t ) \hat { a } ^ { \dagger } + \mathrm { h . c . } ,\tag{B1}
$$

where $\hat { q }$ is the annihilation operator in the transmon Hilbert space, while aˆ is the annihilation operator in the oscillator Hilbert space, describing a transmon and an oscillator mode with frequencies $\omega _ { q }$ and $\omega _ { a }$ respectively. The transmon anharmonicity $K _ { q }$ is large relative to its drive bandwidth, allowing the transmon mode to be approximated as a two-level qubit system, with spin-Z operator $\hat { \sigma } _ { z } = 1 - 2 \hat { q } ^ { \dagger } \hat { q }$ . The interaction between the transmon and the oscillator is described by the cross-Kerr interaction $\chi ,$ which has strength of O(10) kHz. In this situation, the higher-order terms of oscillator anharmonicity $K _ { a }$ and second-order cross-Kerr interaction strength $\chi ^ { \prime }$ are even smaller (of O(10) Hz), and their efects can be ignored for the regime of experiments demonstrated in this work. The term Ω(t) describes a generally time-dependent pulse on the transmon, while ε(t) describes a generally time-dependent pulse on the oscillator. In Table I, we list the measured values of the relevant terms of the Hamiltonian, alongside other parameters.

In this work, the sensing displacement $D ( \alpha )$ is implemented by setting the amplitude and phase of a calibrated Gaussian pulse (truncated at 2σ on either side) with a total duration of 50ns. The pulse is calibrated such that the magnitude of the displacement $| \alpha | = 1 1 . 9 2$ when the A.W.G. amplitude of the pulse is 1, and the displacement is linear in amplitude (see Ref. [67] for further details of this characterization).

In Figure 3, we show how we realize the echoed conditional displacement gate $\begin{array} { r } { \mathrm { E C D } ( \beta ) = D ( \beta / 2 ) | g \rangle \langle e | + } \end{array}$ $D ( - \beta / 2 ) | e \rangle \langle g |$ . Our protocol, based on that introduced in Ref. [69], involves large “out-and-back” displacements on the oscillator. We optimize the pulse shapes to generate large values of $\beta$ as eficiently as possible (that is, with the highest sensitivity to sensing displacements). The pulses for the oscillator are square-wave, with a duration of t. The out and back pulses are spaced by a delay of duration $d .$ The qubit pulses are in the shape of a Gaussian, truncated at 2σ at either side (and a total duration of 100ns). For each experiment, we set the values of t and d (according to the requirements of the task and the limit of the performance of the device). This sets the largest value of $| \beta | = \beta _ { \mathrm { m a x } }$ (achieved when $s = 1 )$ . We then can tune the value of $| \beta | \leq \beta _ { \mathrm { m a x } }$ by setting the value of s. The phase of $\beta$ is determined by the phase of the pulses.

(a)  
![](images/709fe9dffc4246096b22dc03abd37acc0fc4994a9864f6252936f8dac8308954.jpg)

(b)  
![](images/5dd291ccfba43abc423f903ffc7360fa7239b77338cdd6fcce458eb406a862d1.jpg)  
FIG. 3. Realization of the echoed conditional displacement gate. (a) Circuit diagram of the ECD(β) gate, which acts on both the qubit with the oscillator. (b) Pulse-level diagram. The entire gate involves 4 pulses on the oscillator, and a single π pulse on the qubit. The magnitude of $\beta$ depends on the choice of the timescales t and $d ,$ and the amplitude s.

## 2. Demonstration of separation between classical and quantum probes

We discuss the experimental implementation of the single-qubit ancilla quantum protocol for the task and results presented in Figure $\mathrm { 3 ( a ) }$ , for which the protocol obtains an exponential advantage over classical sensors. We begin with a discussion of the characterization of the experiment, before discussing the demonstration of the task.

## Characterization

The circuit diagram we implement for this experiment is illustrated in Figure 4(a). The protocol can be divided into three stages. In the first stage, a single-qubit rotation and an echoed conditional displacement prepare an entangled cat-qubit state. Then, in the second stage, the oscillator senses a displacement. Finally, in the third stage, an echoed conditional displacement with the same parameter is implemented, which disentangles the qubit from the oscillator. The information of the displacement is imparted onto the phase of the qubit, which is inferred from a qubit measurement at the end of the protocol. Figure 4(b) represents the Wigner function of the oscillator conditioned on the state of the qubit. The trajectories enclose an area in phase-space proportional to the size of the cat $\beta$ and the displacement orthogonal to the cat axis $\alpha .$ . The phase imparted onto the qubit is proportional to this area. In Figure 4(c), we plot the measured qubit probability $P _ { e }$ as a function of the sensed displacement, for $t = 1 1 0 \mathrm { n s }$ and d = 800ns. The frequency of the response determines the size of the cat. We vary the value of $\beta$ by changing the A.W.G. amplitude on the pulses to the oscillator (which form the conditional displacement gate) [69]. We fit the data to $P _ { e } = A _ { 0 } + A _ { 1 } \cos \left( 2 \beta \alpha + \phi \right)$ , and extract the best-fit parameters $A _ { 0 } , A _ { 1 } , \beta , \phi$ . In Figure 4(d), we plot the dependence of $\beta$ (left) and $A _ { 1 } ~ \mathrm { ( r i g h t ) }$ on the A.W.G. amplitude. The value of $\beta$ (which corresponds to the size of the cat) scales linearly with the amplitude. We use this relationship to set the size of the cat required for a corresponding experiment. However, due to experimental non-idealities, such as decoherence, the contrast $A _ { 1 }$ reduces with increasing $\beta .$ We notice $\phi$ has a quadratic dependence on $\beta .$ This corresponds to a shift in the the displacement of most sensitivity. We correct this shift by applying an appropriate phase roll in the final qubit rotation. The data presented in Figure 4 represents the results after this correction. This correction helps reduce the simulation-experiment gap, resulting in a better sample performance of the experiment

![](images/2f6ca05f9a26dc9ef6d0428b4fabc54b6c96d78a026b0465daa9ea8593428927.jpg)

![](images/e82998e0fadf648566b0249445c740807e2084dd88694025ec9d29f6c413c4d6.jpg)

![](images/f17043966cc79a4439fcf04c8f95e830bb39f1f7ecf68d85b4700e083ee705fd.jpg)

![](images/a31c9d6610ab6f3efe93f723e177843ecc01d9d7daf4540b0667c4bc92ecc054.jpg)  
FIG. 4. Experimental characterization of the cat-state sensing protocol. (a) Circuit diagram. (b) Schematic illustration of the trajectories of Wigner function of the oscillator conditioned on the qubit state along the corresponding stages of the protocol. (c) Measured qubit probability as a function of the sensed displacement α (orthogonal to the axis of the cat), for diferent values of the cat. The cat size is varied by changing the amplitude of the pulse generated by the arbitrary waveform generator (A.W.G.). (d) Best-fit parameters as a function of the A.W.G. amplitude.

## Experiment

We consider a binary classification task, where the goal is to distinguish between two distributions of displacement (see Figure 5(a)). The magnitude of the displacements is fixed to be $A _ { k } = \sqrt { k } / 2$ , where k is the integer-valued Fourier coeficient. All the information which distinguishes the two distributions is in the phase of the displacement. We illustrate this in Figure 5(b) where we plot the probability of the displacement to originate from Class A (where the diferent rings correspond to diferent values of k). When the probability is close to 1, that particular displacement is more likely to originate from Class A (and vice versa). We plot the distributions for even values of k ranging between 2 and 40. The experiment proceeds as follows:

1. For each Fourier coeficient k, construct a dataset of phases θ size D (in this experiment $D = 1 0 ^ { 4 } )$ for Class A and B via rejection sampling.

2. Stream the dataset to the experiment, where we run the experiment once per dataset (therefore generating D measurements in total for each class). The sensed displacement $D ( \alpha )$ has a value $\alpha = A _ { k } e ^ { i \theta }$ for the particular value θ of the dataset.

3. The optimal value of the conditional displacement is $\beta _ { k } = z _ { k } / ( 2 A _ { k } )$ , where $z _ { k }$ is the first positive zero of the derivative $J _ { k } ( z )$ (which is the kth Bessel function of the first kind). We realize this value of $\beta _ { k }$ by setting the A.W.G. amplitude for the $\mathrm { E C D } ( \beta _ { k } )$ gate to be $s _ { k } = \beta / \beta _ { \mathrm { m a x } }$ , where $\beta _ { \mathrm { m a x } } = 7 . 5$ is the maximum value obtained when $s = 1$ , obtained from the calibration results presented in Figure 4(d).

4. Run the protocol cat-state protocol with this parameter (illustrated in Figure 4(a)), and record the qubit measurement outcome (that is, whether it was measured to be in the ground or excited state).

## Results

A protocol which has a response which matches this distribution will eficiently distinguish the two distributions. This is indeed the case with the quantum-enhanced sensor. In Figure 5(c), we plot the measured qubit probability for the same range of displacements as the dataset. The probability matches the periodicity of the dataset for a significant range of phase. On the other hand, as illustrated in Figure 5(d), the contrast of the classical sensor’s response exponentially decays with k towards 0.5. In Figure 5(e), we plot the theoretical and experimental probability of the qubit in the excited state for each of the class. The experimental probability is obtained by simply averaging over the entire dataset of the corresponding class (the errorbars are too small to

(a)

Given displacements $D ( A _ { k } e ^ { i \theta } )$ where θ is sampled from eithe $P _ { A } ( \theta ) = \frac { 1 + 0 . 9 5 \cos { k \theta } } { 2 \pi } ~ \mathrm { o r } ~ P _ { B } ( \theta ) = \frac { 1 - 0 . 9 5 \cos { k \theta } } { 2 \pi }$ how many samples do you need to distinguish the two probability distributions with success probability at least $1 \ : - \ : \delta ?$

![](images/106aa3cfe153870560ebc3a2ebd67d28135b3d3d0704a77e605987083824a2ca.jpg)

![](images/7a6f27ab57697433e2f3a351db06fef538392c4dcc5bb63cb1563e17a161c052.jpg)

(d)  
![](images/54a413e5f2c8cdd10ef080541d24af0a95f2fada3fff81f6a6347509dd9b9121.jpg)

(e)  
![](images/12fe4303e49df82e9401321a0a3bec4e2a1008e14107d28eb377673f4cb8bd42.jpg)

(f)  
![](images/f9f1f34b6836fa69b3a0fc95ebfcd8780b1b66e27b080d20245a7aa0b31126d8.jpg)

![](images/a892f6764dcfdbd203c0208cc174d95e3fdd67e64e313052c627dd9a65575a7a.jpg)  
FIG. 5. Experimental demonstration of separation between classical and quantum probes. (a) We consider the sample requirement for correctly classifying the distribution with a success probability within δ of 1. (b) Illustration of the probability distribution of a signal given to originate from Class A. If the probability is 1, then if a displacement with the corresponding value is sensed, it can only originate from Class A. (c) Response of the quantum-enhanced sensor in experiment. (d) Response of the classical sensor in simulation. The response of the quantum-enhanced sensor visually shows how it can eficiently distinguish the two classes. (e) Qubit excitation probability as a function of the Fourier coeficient k in theory and experiment for the two classes. (f) (Left) Corresponding sample complexity of the quantumenhanced sensor in experiment and the classical baseline in simulation to achieve classification accuracies 70% , 80% and 90%. (Right) Zoom in of the sample complexity of the quantum-enhanced sensor in experiment, along with the expected performance in the ideal limit.

be visible on the plot). For a large range of k, the experimental probability is close to the theoretical value. The significant source of discrepancy is the readout fidelity, whose efect is to push the experimental value closer to 0.5. From this dataset, we can then estimate how many samples would be required to correctly identify the class label 70%, 80% and 90% of the time (see Figure 5(f)). Due to the lower separation of probabilities in experiment compared to the ideal value, the corresponding sample requirement is larger. However, even for $k = 4 0$ , the sample requirement for achieving an accuracy of at least 70% is around 100, several orders of magnitude below that of a classical heterodyne or homodyne protocols.

## 3. Demonstration of separation between conventional quantum experiments and quantum control

We discuss the experimental implementation for the task and results presented in Figure 2. In this scenario, the experiment (single-qubit ancilla entangled with the sensing oscillator) achieves an exponential advantage over conventional quantum sensing protocols with access to only Gaussian resources with the same energy.

## Characterization

The protocol, illustrated in Figure 6(a), is essentially identical to that in Figure 4(a). The main diference is the amplitude of the initial qubit rotation, which depends on the Fourier coeficient k. We choose this value such that the number of photons in the displaced frame scales linearly with k (see Appendix C 4 for an explanation for the choice of this resource metric). This in general therefore realizes an “unbalanced” cat, where the coeficient of wavefunction conditioned on the ground and excited states of the qubit are diferent. This is schematically illustrated in Figure 6(b), where the faded color scale of the Wigner function conditioned on |g⟩ is due to the smaller weight of the wavefunction overlap with this state. This protocol results in a Ramsey response for the qubit excitation probability, but with a smaller amplitude, as shown in Figure 6(c), where we set t = 110ns and $d = 1 1 0 0 \mathrm { n s }$ . Like before, we plot the best-fit parameters to a sinusoidal response, shown in Figure 6(d). We also plot the value of the amplitude expected as a function of k in the idealized scenario without experimental imperfections.

(c)  
(a)  
![](images/59bd94f4745fe5190bdfc57deffb375bc4e88b1d4e5d2a103085926b7ab24153.jpg)

![](images/6e2bf5fbe2201394d569c4fe395b6ada44a9006df8150a120f1087bddbd811d0.jpg)

![](images/2ce25ebf518d1f790c6285f6f36b92765daa456ab4b348343bbb8f1035eb9b29.jpg)

![](images/44d9853f0e1bfa5e760ad2d7345d354f6e43673dcdbf43539389b7f220814286.jpg)  
FIG. 6. Experimental characterization of the unbalanced cat-state sensing protocol. (a) Circuit diagram. (b) Schematic illustration of the trajectories of Wigner function of the oscillator conditioned on the qubit state along the corresponding stage of the protocol. (c) Measured qubit probability as a function of the sensed displacement α (orthogonal to the axis of the cat), for diferent values of the cat.. (d) Best-fit parameters as a function of the A.W.G. amplitude, along the theory-expected value under ideal conditions of no decoherence.

## Experiment

For this task, the two classes are described by the distribution written in Figure $\mathrm { 7 ( a ) }$ at the parameters $\theta = 0$ and $\theta = \pi / 2$ . In this case, the distributions are spread out over both the position and momentum components of the displacement. Figure 7(b) are scatter plots of the dataset of each task, for select values of the Fourier coeficient k. Here k is a proxy for the number of “fringes" in the dataset. Our protocol can eficiently distinguish the two distributions but matching the periodicity of these fringes. The experiment proceeds as follows:

1. For each coeficient $k ,$ construct a dataset for Class A and B of size D (in this experiment $D = 1 0 ^ { 4 } )$ via rejection sampling.

2. Stream the dataset to the experiment, where we run the experiment once per dataset (therefore generating D measurements in total for each class).

3. The value of the conditional displacement is $\beta _ { k } = k / 2$ , while the qubit-rotation amplitude is chosen such that the corresponding rotation operator $R _ { X } ( \theta _ { k } )$ has a value $\theta _ { k } \overset { \cdot } { = } 2 \sin ^ { - 1 } ( \sqrt { 0 . 4 / k } )$ We realize $\beta _ { k }$ by setting the A.W.G. amplitude for the ECD(β ) gate to be $s _ { k } = \beta / \beta _ { \mathrm { m a x } } .$ where $\beta _ { \mathrm { m a x } } = 1 0$ is the maximum value obtained when $s = 1$ , obtained from the calibration results presented in Figure 6(d).

4. Run the protocol imbalanced cat-state protocol with this parameter (illustrated in Figure 6(a)), and record the qubit measurement outcome (that is, whether is was measured to be in the ground or excited state).

## Results

Figure 7(b) visually shows how the two classes can be distinguished. The orientation and size of the cat is such that the red dots (corresponding to the qubit being measured in the ground state) are more frequent for Class A. On the other hand, for Class B, there are similar frequency of both outcomes. This results in the qubit excitation probability in experiment (see Figure 7(c)) to be close to 0.5. On the other hand, the qubit

(a)

Task

Given displacements $D ( \alpha )$ where $\alpha = \alpha _ { x } + i \alpha _ { p }$ is sampled from either $P _ { \theta = 0 } ( \alpha )$ or $P _ { \theta = \pi / 2 } ( \alpha )$

$$
P _ { \theta } ( \alpha ) = \frac { e ^ { - | \alpha | ^ { 2 } / 2 } } { 2 \pi } \left( 1 + \frac { \cos ( k \alpha _ { x } \cos ( \theta ) + k \alpha _ { p } \sin ( \theta ) ) - e ^ { - k ^ { 2 } / 2 } } { 1 + e ^ { - k ^ { 2 } / 2 } } \right)
$$

how many samples do you need to distinguish the two probability distributions with success probability at least $1 \ : - \ : \delta ?$

Quantum-enhanced sensor (experiment)

(b)  
(c)  
![](images/876bfbfe2b9be8b2a92a0b8beadf08ca5d471121fe6e3757364befc9df492755.jpg)

Probability of the qubit in the excited state averaged over the dataset

![](images/62769351a13486130b5caacb7384628e398bea4846993596e898465661364c6d.jpg)

![](images/10ee5012011fba529ce274e20647da6d15d253dca17e02032b9e20541c27419f.jpg)  
(d)

![](images/e231fee7b2ec31194331d36533998a512b74fae8c6cca73596447c01c39017b4.jpg)  
FIG. 7. Experimental demonstration of separation between conventional quantum experiments and quantum control. (a) We consider the sample requirement for correctly classifying the distribution with a success probability within δ of 1. (b) Illustration of the datasets $o f$ each class for $k = 2 , 6 , 1 0$ . The data points are colored based on the qubit measurement outcomes of that particular experiment. (c) Qubit excitation probability as a function of the Fourier coeficient k in theory and experiment for the two classes. (d) (Left) Corresponding sample complexity of the quantumenhanced sensor in experiment and the Gaussian baseline in simulation to achieve classification accuracies 70%, 80%, and 90%. (Right) Zoom in of the sample complexity of the quantum-enhanced sensor in experiment, along with the expected performance in the ideal limit.

excitation probability for Class A is smaller. Similar to before, the errorbars are too small to be visible on the plot. From this dataset, we estimate the sample requirement for a classification accuracy of 70%, 80% and 90%. In Figure 7(d) we plot this as a function of $k \in [ 2 , 2 0 ]$ for both the experimental and the theoretically idea results. Similar to before, we observe a sample advantage of several orders of magnitude, persisting for $k = 2 0$

## 4. Demonstration of quantum memory advantage

We discuss the experiment results of the task presented in Figure $3 ( \mathrm { c } )$ , for the scenario where the ancilla qubit has coherent quantum memory across the duration of the sensing events. Our protocol achieves an exponentia advantage over protocols without access to quantum memory during sensing, with the same photon number.

## Experiment

We consider a binary classification task where the two distributions are distinguished in terms of what sequence of displacements are allowed. $\boldsymbol { b } = \left( b _ { 1 } , b _ { 2 } , \ldots , b _ { N } \right)$ represents a bitstring which denotes the sequence of displacements on the oscillator. The displacement on the oscillator is $D ( \gamma b _ { i } )$ for bit $b _ { i }$ . Therefore, the two possibilities are the oscillator senses a displacement or not. We consider a binary classification task in distinguishing two distributions, represented by Class A and Class B. Each class is a probability distribution over

(a)

Given a sequence of displacements $D ( \gamma b _ { 1 } ) , D ( \gamma b _ { 2 } ) , \dots D ( \gamma b _ { N } )$ where the bitstring $\boldsymbol { b } = \left( b _ { 1 } , b _ { 2 } , \dots b _ { N } \right)$ is sampled from either $P _ { A } ( b ) = { \frac { 1 } { 2 ^ { N } } } ( 1 + ( - 1 ) ^ { b _ { 1 } + b _ { 2 } + \cdots + b _ { N } } ) \mathsf { o r } P _ { B } ( b ) = { \frac { 1 } { 2 ^ { N } } } ( 1 - ( - 1 ) ^ { b _ { 1 } + b _ { 2 } + \cdots + b _ { N } } )$

how many samples do you need to distinguish the two probability distributions with success probability at least $1 \ : - \ : \delta ?$

(b)  
Quantum-enhanced sensor: continuous-sensing (experiment)  
![](images/67fe8db0114b0852e1f06350b94c67bbaac84228d4d90aca41d9f34a58066100.jpg)  
Quantum-enhanced sensor: interleaved-sensing (experiment)

![](images/254ade78fea96364518a0be79735149dcbb3c2682aca748050bd64e6d5625014.jpg)

![](images/46d6a2d95f42e1bbee3a88592b57b2aa73ee29ee8e33187210d69b55bb672981.jpg)

![](images/f872503626eb4f602cc6636090d0582a444868ea6b26fcde122dc50f67b8facc.jpg)  
FIG. 8. Experimental demonstration of quantum memory advantage. (a) We consider the sample requirement for correctly classifying the distribution with a success probability within δ of 1. (b) Circuit diagrams of the two protocols we consider. The two protocols consider the scenarios where the sensor is or is not coherent across multiple sensing rounds. (c) Qubit excitation probability as a function of the depth N in theory and experiment (for both choices of protocols) for the two classes. (d) (Left) Corresponding sample complexity of the quantum-enhanced sensor in experiment and the baseline without quantum memory in simulation to achieve classification accuracies 70%, 80%, and 90%. (Right) Zoom in of the sample complexity of the quantum-enhanced sensor in experiment, along with the expected performance in the ideal limit.

all possible $2 ^ { N }$ bitstrings for depth N. As defined in Figure 8(a), Class A consists of all bitstrings with even parity, while Class B consists of all bitstrings with odd parity. We consider two variations of quantum-enhanced protocols with memory: continuous-time and interleaved-time. In the first version, we allow the quantum sensor to continuously accrue displacements over time. Such a protocol relies on both the oscillator and qubit to remain coherent on the time scale of the sensing. On the other hand, one can consider the situation where the oscillator cannot stay coherent over the entire sensing duration (only the ancilla qubit can). This corresponds to interleaved-sensing, which is designed to disentangle the qubit from the oscillator after a sensing round. Therefore, the sensor can be reset without destroying the information stored in the qubit. For both protocols, we set the amplitude $s = 1$ , time scales $t = 1 0 0$ ns and $d = 4 0 0$ ns for the parameters of the ECD gate. This corresponds to $\beta = 4 . 1 5$ . The experiment proceed as follows:

1. For each depth N, construct a dataset for Class A and B of size D (in this experiment $D = 1 0 ^ { 4 } )$ via rejection sampling.

2. Stream the dataset to the experiment, where we run the experiment once per dataset (therefore generating D measurements in total for each class).

3. The value of the qubit rotation angle is $\theta = 2 \sin ^ { - 1 } \big ( \sqrt { E / \beta ^ { 2 } } \big )$ , with $E = ( ( 2 \nu \beta / \pi ) ^ { 2 } - 1 ) / 2$ and $\nu = 0 . 4 1$ This is not unique. Instead these were chosen to maximize the quantum advantage over the baseline.

4. Run each of the two protocols (continuous-sensing and interleaved-sensing). For each protocol, record the qubit measurement outcome (that is, whether is was measured to be in the ground or excited state).

## Results

In Figure 8, we plot the qubit excitation probability, averaged over the dataset, for each class, as a function of the depth N. The theoretical value for both protocols are identical. As the depth N increases, the probabilities in experiment tend towards 0.5, with the separation reducing. This is due to decoherence, since the lifetime of the experiment is finite. This is more pronounced for the interleaved-sensing experiment, since the duration of the protocol is much longer due to the inclusion of additional ECD gates during the sensing stage. For our experimental regime, the oscillator has a longer coherence time than the qubit – therefore, the device does not benefit from the motivation of this protocol to disentangle the ancilla qubit from the sensor. However, the results of this protocol provide a proof-of-principle demonstration for devices which operate in the regime of the lifetime of the ancilla mode much longer than the sensing mode. The efect of decoherence is marginal for the continuous-sensing protocol. Since each sensing duration takes place in 50ns, there is only a slight enhancement in the overall decoherence for the range of depths considered. This is reflected in Figure 8, where we plot the sample requirement of the experiment, and theoretical baseline, for achieving 70%, 80%, and 90% accuracy. The sample requirement of both the continuous-sensing and theoretical baseline are O(1) for achieving an accuracy at least 70%. On the other hand, a sensor without access to quantum memory will have a sample complexity scaling exponential in the depth N.

## 5. Simulation of cold axion stream characterization

Here we describe the numerical experiment depicted in Figure 4(a). We will provide a high-level description of the axion-physics models used in our numerics, which are derived from the results of Ref. [70]. We refer readers to this work for a detailed understanding of how the formulae and physical parameters are obtained.

## Derivation of task from axion physics

A nonrelativistic axion propagating through free space is a real scalar field of mass $m _ { a }$ with plane-wave solutions

$$
a _ { \mathbf { v } } ( \mathbf { x } , t ) = a _ { \mathbf { v } } ( \cos ( \omega _ { \mathbf { v } } t - \mathbf { k _ { v } } \cdot \mathbf { x } - \phi _ { \mathbf { v } } ) ) \ ,\tag{B2}
$$

where v is the 3-velocity, $\mathbf { k _ { v } } = \mathbf { m _ { a } } \mathbf { v } .$ , and $\omega _ { \mathbf { v } } = \sqrt { m _ { a } ^ { 2 } + | \mathbf { k } _ { \mathbf { v } } | ^ { 2 } }$ . The oscillation induced in a detector, then, is of the form

$$
\Phi ( { \bf x } , t ) = m _ { a } \cdot \boldsymbol { \kappa } \cdot a _ { \bf v } ( { \bf x } , t ) \ ,\tag{B3}
$$

where κ captures the physics of the coupling of the detector to the field. A realistic axion field is not simply one plane wave, but a mixture of many incoming waves at diferent velocities (frequencies) and phases. Ref. [70] shows that under the assumption of a uniform phase distribution, the field is well-described by

$$
a _ { \mathbf { v } } ( \mathbf { x } , t ) = \frac { \sqrt { \rho _ { \mathrm { D M } } } } { m _ { a } } \sum _ { d } \alpha _ { d } \sqrt { f ( \mathbf { v } _ { d } ) ( \Delta v ) ^ { 3 } } ( \cos ( \omega _ { d } t - \mathbf { k } _ { d } \cdot \mathbf { x } - \phi _ { d } ) ) \ ,\tag{B4}
$$

where velocity space is discretized into intervals of $\Delta v$ in each direction and each speed is labeled by d. Moreover, ρ<sub>DM</sub> is the local dark matter energy density, f(v) is the normalized three-dimensional lab-frame velocity density, $\alpha _ { d }$ is Rayleigh distributed as $p ( \alpha _ { d } ) = \alpha _ { d } e ^ { - \alpha _ { d } ^ { 2 } / 2 }$ , every $\phi _ { d }$ is uniformly distributed and we assume that diferent velocity increments have independent $\alpha _ { d } , \phi _ { d } .$

This model gives quadrature amplitudes which are exactly Gaussian, which is evident upon setting $X _ { d } =$ $\alpha _ { d }$ cos $\phi _ { d } , Y _ { d } = \alpha _ { d }$ sin ϕ<sub>d</sub> and computing the joint distribution. Upon combining equations (B3) and (B4), the response of a detector i is

$$
\Phi _ { i } ( { \bf x } , t ) = \sqrt { \rho \kappa _ { i } ^ { 2 } } \sum _ { d } \alpha _ { d } \sqrt { f ( { \bf v } _ { d } ) ( \Delta v ) ^ { 3 } } ( \cos \omega _ { d } t - { \bf k } _ { d } \cdot { \bf x } - \phi _ { d } ) ~ .\tag{B5}
$$

One can then discretize interrogation time into intervals of $\Delta t$ such that $t _ { n } = n \Delta t$ for $n = 0 , 1 , . . . , M - 1$ , and write $\Phi _ { i , n } = \Phi _ { i } ( t _ { n } )$ , where we omit the position coordinate and take the detector’s position to be fixed in the laboratory frame. Then one can compute a discrete Fourier transform of $\Phi _ { i , n } \mathrm { { : } }$

$$
\widetilde { \Phi } _ { i , \ell } = \sum _ { n } \Phi _ { i , n } e ^ { - i 2 \pi \ell n / M } \ ,\tag{B6}
$$

and define the frequency-domain quadrature coordinates

$$
R _ { i , \ell } = \frac { \Delta t } { \sqrt { M \Delta t } } \mathrm { R e } ( \tilde { \Phi } _ { i , \ell } ) , \qquad I _ { i , \ell } = \frac { \Delta t } { \sqrt { M \Delta t } } \mathrm { I m } ( \tilde { \Phi } _ { i , \ell } ) \ .\tag{B7}
$$

Note that both expressions are functions of the angular frequency $\omega _ { \ell } = 2 \pi \ell / ( M \Delta t )$ . The normalization is chosen precisely so that $\begin{array} { r } { R _ { i , \ell } ^ { 2 } + I _ { i , \ell } = \frac { ( \Delta t ) ^ { 2 } } { T } | \widetilde { \Phi } _ { i , \ell } | ^ { 2 } } \end{array}$ , which is exactly the power spectral density in frequency bin ℓ. Ref.[70] then shows that the random variables $R _ { i , \ell } , I _ { i , \ell }$ have exactly zero mean, and moreover that

$$
\mathrm { V a r } ( R _ { i , \ell } ) = \mathrm { V a r } ( I _ { i , \ell } ) = \frac { \pi A _ { i } } { 2 m _ { a } v _ { \omega } } f _ { \mathrm { s p } } ( v _ { \omega } ) \ ,\tag{B8}
$$

where $A _ { i } = \rho _ { \mathrm { D M } } \kappa _ { i } ^ { 2 } , v _ { \omega }$ is the velocity which satisfies $\omega = m _ { a } ( 1 + v _ { \omega } ^ { 2 } / 2 )$ , and

$$
f _ { \mathrm { s p } } ( v ) = \int d ^ { 3 } { \bf u } \ f ( u ) \delta ( | { \bf u } | - v ) \ .\tag{B9}
$$

Moreover, the covariance is zero. Any additive background does not correlate the quadratures, and only contributes a possibly frequency-dependent variance $\lambda _ { i } ( \omega ) / 2$ . Therefore, the detector response is a mean-0 random variable with covariance matrix, in these quadrature coordinates, of

$$
\Sigma _ { i } ( \omega ) = { \frac { 1 } { 2 } } \left( { \frac { \pi A _ { i } } { m _ { a } v _ { \omega } } } f _ { \mathrm { s p } } ( v _ { \omega } ) + \lambda _ { i } ( \omega ) \right) I _ { 2 } ~ .\tag{B10}
$$

Now, we can describe this physics using bosonic phase space. We introduce the complex detector amplitude coordinate $z _ { i } ( \omega ) = R _ { i } ( \omega ) + i I _ { i } ( \omega )$ . Then, it immediately follows that

$$
z _ { i } ( \omega ) \sim \mathcal { N } ( 0 , P _ { i } ( \omega ) + \lambda _ { i } ( \omega ) )\tag{B11}
$$

with $\begin{array} { r } { P _ { i } ( \omega ) = \frac { \pi A _ { i } } { m _ { a } v _ { \omega } } f _ { \mathrm { s p } } ( v _ { \omega } ) } \end{array}$ . Therefore, the detector maps a classical field amplitude to a displacement of a single bosonic mode in the appropriate quadrature coordinates. Characterizing the axion field then amounts to learning properties of the Gaussian displacement channel that describes these physics.

With this context, we now understand the main motivating question of Ref. [70]. Notice that $\Sigma _ { i } ( \omega )$ depends on the velocity distribution $v _ { \omega }$ only through $f _ { \mathrm { s p } }$ , but this distribution integrates over all velocities with fixed magnitude |v| and therefore contains no directional information. As such, a single detector cannot characterize the direction of an anisotropic axion stream. This is essential, because as described in Ref. [70], the daily modulation that would be observed in the incident angle of a detected field would imply the existence of a fixed-orientation stream and would constitute smoking-gun evidence of an axion field.

The key observation of Ref. [70] is that while a single detector is insensitive to the stream’s direction, two spatially-separated detectors can ascertain the angle of incidence. We now omit the derivation and state the conclusion: letting $z _ { i } , z _ { j }$ denote the complex displacement coordinate of two spatially separated detectors located at $\mathbf { x } _ { i } , \mathbf { x } _ { j }$ and separated by vector $\mathbf { d } = \mathbf { x } _ { i } - \mathbf { x } _ { j }$ , the covariance matrix of the two coordinates has the form

$$
\Sigma _ { \theta } ( v ) = P ( v ) \left( \begin{array} { c c } { { 1 + \beta ( v ) } } & { { \gamma _ { \theta } ( v ) } } \\ { { \gamma _ { \theta } ( v ) ^ { * } } } & { { 1 + \beta ( v ) } } \end{array} \right) ,\tag{B12}
$$

where $P ( v ) , \beta ( v )$ are independent of the stream direction, but $\gamma _ { \theta } ( v )$ is sensitive to the angle of incidence θ between the stream and the detector plane:

$$
\gamma _ { \theta } ( v ) = \frac { F _ { 1 2 } ^ { c } ( v ; \theta ) - i F _ { 1 2 } ^ { s } ( v ; \theta ) } { f _ { \mathrm { s p } } ( v ) } , \quad \mathrm { ~ w h e r e ~ }\tag{B13}
$$

$$
F _ { 1 2 } ^ { c } ( v ) = \int d ^ { 3 } { \bf v } ^ { \prime } f ( { \bf v } ^ { \prime } ) \cos ( m _ { a } { \bf v } ^ { \prime } \cdot { \bf d } ) \delta ( | { \bf v } ^ { \prime } | - v ) ,\tag{B14}
$$

$$
F _ { 1 2 } ^ { s } ( v ) = \int d ^ { 3 } { \bf v } ^ { \prime } f ( { \bf v } ^ { \prime } ) \sin ( m _ { a } { \bf v } ^ { \prime } \cdot { \bf d } ) \delta ( | { \bf v } ^ { \prime } | - v ) .\tag{B15}
$$

Here the functions $F _ { 1 2 }$ depend on θ because of the term $\mathbf { v } ^ { \prime } \cdot \mathbf { d }$ . The entire task of learning the angle of incidence therefore reduces to sensing the of-diagonal covariance in a displacement channel’s distribution.

These expressions can be simplified to numerical values given real physical parameters. Ref. [70] models the

Sagittarius stream [71] using a Maxwellian speed distribution with boost vector (0, 93.2, −388) km $/ \mathrm { s } ,$ dispersion $v _ { 0 } = 1 0 ~ \mathrm { k m / s } .$ and velocity $u = 4 0 0$ km/s. Substituting a Maxwellian speed distribution and simplifying gives us

$$
\gamma _ { \theta } ( v ) = \frac { s ( v ) } { \chi _ { \theta } ( v ) } \frac { \sinh \chi _ { \theta } ( v ) } { \sinh s ( v ) }\tag{B16}
$$

$$
s ( v ) = \frac { 2 v u } { v _ { 0 } ^ { 2 } } \ ,\tag{B17}
$$

$$
q ( v ) = m _ { a } d v\tag{B18}
$$

$$
\chi _ { \theta } ( v ) = \sqrt { a ( v ) ^ { 2 } - q ( v ) ^ { 2 } - 2 i a ( v ) q ( v ) \cos \theta } \ .\tag{B19}
$$

Using an axion mass estimate of 25.2 $\mu \mathrm { e V }$ alongside the estimated Sagittarius velocity parameters, we can compute numerical values for $\gamma _ { \theta } .$ . In line with Ref. [70], we consider learning the incident angle $\theta \ : = \ : 4 5 ^ { \circ }$ Substituting $\theta \ : = \ : 4 5 ^ { \circ }$ and $4 6 ^ { \circ }$ into our covariance matrix expression and evaluating numerically, we are left with the task of distinguishing two two-mode Gaussian displacement channels with equal mean and difering covariance matrices

$$
\Sigma _ { 4 5 } = \Sigma _ { 0 } + \epsilon \left( \begin{array} { c c } { { 1 } } & { { 0 . 6 0 6 5 6 4 8 - 0 . 0 0 6 6 9 1 6 i } } \\ { { 0 . 6 0 6 5 6 4 8 + 0 . 0 0 6 6 9 1 6 i } } & { { 1 } } \end{array} \right) ~ ,\tag{B20}
$$

$$
\Sigma _ { 4 6 } = \Sigma _ { 0 } + \epsilon \left( \begin{array} { c c } { { 1 } } & { { 0 . 3 2 9 8 7 5 6 + 0 . 4 9 6 5 2 1 0 i } } \\ { { 0 . 3 2 9 8 7 5 6 - 0 . 4 9 6 5 2 1 0 i } } & { { 1 } } \end{array} \right) ~ .\tag{B21}
$$

Here, $\epsilon$ is the normalized value corresponding to the mapping of the physical scale parameters onto the bosonic phase space. ϵ therefore captures the total strength of interaction, which depends broadly on the axion field’s power and its coupling constant to the detector. This constant cannot be evaluated without a particular detector specification, but in phase-space coordinates it is generally $\ll 1$ because the of-diagonal covariance is much weaker than the diagonal variance components due to background and power broadening. ϵ controls the hardness of the sensing task, since weaker fields are naturally more dificult to characterize.

This task, derived directly from the physical model of axion detection of Ref. [70] and evaluated with estimated parameters of the Sagittarius stream, therefore captures the challenge of resolving the angle of the incident axion stream to within one degree.

## Numerical implementation

We compare QFS against two receivers that act locally at the detector sites, fixing the total mean photon number in the two sensing modes to $E = 1$ . The first prepares an independent single-mode squeezed vacuum at each detector, applies the inverse local squeezing after sensing, and performs local photon-number-resolving measurements. This combines phase-sensitive Gaussian probes with a non-Gaussian readout. The second prepares an entangled two-mode squeezed vacuum (TMSV) state between each sensing mode and a local idler, and performs a local continuous-variable Bell measurement after sensing. This converts each displacement into a noisy complex-amplitude sample and is the canonical entanglement-assisted Gaussian receiver. The two signal arms again contain total energy $E = 1$ , while the stored idlers contain one additional photon on average in total. Together, these protocols use the principal Gaussian and non-Gaussian resources available locally, and therefore provide stringent benchmarks for strategies which do not perform coherent operations between detector sites. Squeezing, photon-counting receivers, and TMSV-based entanglement assistance are among the principal quantum-enhanced architectures studied for stochastic sensing and axion detection and are established as near-optimal in many cases [60–62, 72, 73].

One could instead distribute entanglement between the probe states used at two detector sites or perform a joint quantum measurement across them. Such strategies may access the cross-covariance more directly, but require the generation, distribution, and stabilization of entangled bosonic modes over the detector separation, which is far more demanding than the significant challenges already posed by the above benchmarks. QFS does not require an entangled bosonic probe shared across this baseline. Instead, a single ancilla qubit coherently controls conditional displacements of the two sensing modes, mapping the selected collective quadrature onto a single-qubit measurement after one layer of control. While ancillary preshared entanglement could facilitate execution by quantum teleportation of the control qubit (rather than physically shuttling the qubit between detector sites), these resources are never exposed to the signal and can therefore be protected by $\mathrm { e . g . }$ quantum error correction or error mitigation, unlike entangled sensor probes that must remain sensitive to their environment.

In our simulation, one shot consists of one independent draw of the two-mode Gaussian displacement under either angular hypothesis. For QFS, we prepare the depth-one conditional coherent-state probe, apply the displacement, invert the conditional displacement, and measure the ancilla in the X basis. A shot of the squeezed photon-counting receiver returns the pair of local photon numbers obtained after inverse squeezing, while a shot of the TMSV receiver returns the pair of complex outcomes from the two local Bell measurements, which corresponds to sampling a Gaussian convolution of the displacement channel. For an episode of length $N _ { : }$ , we repeat the corresponding shot independently N times and distinguish the 45<sup>◦</sup> and 46<sup>◦</sup> hypotheses using the exact likelihood ratio of the complete measurement record. At each field strength and candidate value of N, we estimate the classification accuracy using 5,000 balanced Monte Carlo episodes and report the smallest tested value for which the empirical accuracy is at least 70%.

## Origin of quadratic separation

The origin of the quadratic separation in Fig. 4 is most transparent in the weak-field limit. In our numerica experiment, we write the stream-induced covariance under hypothesis $j$ as $\Gamma _ { j } ( \epsilon ) = \epsilon S _ { j }$ . For a QFS filter with conditional displacement $\beta ,$ let $c _ { j } = \beta ^ { \dagger } S _ { j } \beta$ . The probability of the negative qubit outcome is

$$
1 - p _ { j } = \frac { 1 - \exp ( - \epsilon c _ { j } ) } { 2 } = \frac { \epsilon c _ { j } } { 2 } + O ( \epsilon ^ { 2 } ) .\tag{B22}
$$

Provided that $c _ { 0 } \neq c _ { 1 }$ , the hypotheses therefore produce diferent rare-event rates at first order in ϵ. After N shots, both the mean number and variance of these events scale as $N \epsilon$ , so constant discrimination accuracy requires $N \epsilon = \Theta ( 1 )$

The baseline receivers behave diferently. For the TMSV receiver, the measured covariance has the form $V _ { j } = \nu I _ { 2 } + \epsilon S _ { j }$ with $\nu > 0$ . The two outcome distributions coincide at $\epsilon = 0$ in the presence of this fixed measurement noise, so their statistical divergence begins at second order in the covariance perturbation. For squeezed photon counting, the local photon-number marginals are identical under the two hypotheses, and the first hypothesis-dependent information appears in joint photon-number events whose probabilities scale as $\epsilon ^ { 2 }$ Consequently, the one-shot Chernof informations $C$ and the corresponding sample requirements satisfy

$$
\begin{array} { r } { C _ { \mathrm { Q F S } } ( \epsilon ) = \Theta ( \epsilon ) , } \\ { C _ { \mathrm { S M S V } } ( \epsilon ) = C _ { \mathrm { T M S V } } ( \epsilon ) = \Theta ( \epsilon ^ { 2 } ) , } \end{array}
$$

$$
N _ { \mathrm { Q F S } } = \Theta ( \epsilon ^ { - 1 } ) ,\tag{B23}
$$

$$
N _ { \mathrm { S M S V } } , N _ { \mathrm { T M S V } } = \Theta ( \epsilon ^ { - 2 } ) = \Theta ( N _ { \mathrm { Q F S } } ^ { 2 } ) .\tag{B24}
$$

Thus QFS converts the covariance diference into a first-order rare-event rate and achieves Heisenberg scaling in the field strength, while the local receivers, despite using diferent resources even beyond QFS (more non-Gaussinity for the photon-counting receiver, and an entangled idler mode for the TMSV receiver), they only see the signal through second-order changes in their measurement statistics. This produces the quadratic separation observed in Figure 4(a). The core message is that programmable single-qubit control, even if only a depth-1 quantum circuit, can engineer filter functions which are asymptotically more sensitive to a signal of interest than even significantly sophisticated conventional strategies that may still use non-Gaussianity or entanglement. For axion detection in particular, this suggests that superconducting-circuit receivers, enabled by QFS design principles, may enable significant improvements to the detection of axionic dark matter, even when compared to longstanding and well-established quantum sensing strategies.

## 6. Simulation of wireless communication receiver

## Task motivation: quadrature amplitude modulation

Quadrature amplitude modulation (QAM) is a standard method for encoding digital information in wireless signals. A passband waveform associated with symbol m may be written as

$$
s _ { m } ( t ) = I _ { m } \cos ( \omega _ { c } t ) - Q _ { m } \sin ( \omega _ { c } t ) ,\tag{B25}
$$

where $I _ { m }$ and $Q _ { m }$ are its in-phase and quadrature amplitudes. In a frame rotating at the carrier frequency, these amplitudes are precisely the two quadratures of a single electromagnetic mode, and the complex envelope $I _ { m } + i Q _ { m }$ may be identified with the bosonic phase-space coordinate $\alpha _ { m }$ . An M-QAM scheme encodes each symbol as one of M predetermined points in the I-Q plane. In square $6 4 \mathrm { - Q A M }$ , each quadrature takes one of eight possible values, giving 64 symbols and therefore six encoded bits per symbol. Channel noise ordinarily broadens the received location, and the receiver must determine which constellation point was transmitted. We model this task directly as discrimination among the displacement channels associated with the $\mathrm { Q A M }$ constellation.

## Numerical implementation

We consider the square 64-QAM constellation and a rectangular $8 – \mathrm { Q A M }$ constellation,

$$
\mathcal { A } _ { 6 4 } ( A ) = \left\{ \alpha _ { a b } = \frac { A } { \sqrt { 4 2 } } ( a + i b ) ~ \middle | ~ a , b \in \{ - 7 , - 5 , - 3 , - 1 , 1 , 3 , 5 , 7 \} \right\} ,\tag{B26}
$$

$$
\mathcal { A } _ { 8 } ( A ) = \left\{ \alpha _ { a b } = \frac { A } { \sqrt { 6 } } ( a + i b ) ~ \middle | ~ a \in \{ - 3 , - 1 , 1 , 3 \} , ~ b \in \{ - 1 , 1 \} \right\} .\tag{B27}
$$

Both constellations are normalized such that

$$
{ \frac { 1 } { | { \mathcal { A } } _ { M } | } } \sum _ { \alpha \in { \mathcal { A } } _ { M } } | \alpha | ^ { 2 } = A ^ { 2 } .\tag{B28}
$$

The 8-QAM constellation is therefore a $4 \times 2$ rectangular constellation containing three encoded bits. Its unnormalized coordinates form a subset of the $6 4 \mathrm { - Q A M }$ lattice, but the two constellations are normalized independently. Consequently, $A _ { 8 } ( A )$ coincides with the corresponding physical subset of $\mathcal { A } _ { 6 4 } ( \sqrt { 7 } A )$ rather than $\mathcal { A } _ { 6 4 } ( A )$

Symbol m implements the deterministic displacement channel

$$
\mathcal { E } _ { m } ( \rho ) = D ( \alpha _ { m } ) \rho D ( \alpha _ { m } ) ^ { \dagger } .\tag{B29}
$$

In each episode, the transmitted symbol is sampled uniformly from the relevant constellation and held fixed over all channel interactions. The receiver is given the constellation but not the transmitted symbol. The random-guessing probabilities are therefore $1 / 6 4$ and $1 / 8$ for the two tasks.

The shallow QFS receiver uses three ancilla qubits coupled to the received bosonic mode. Each qubit undergoes a depth-ten sequence of equatorial rotations and signal-interleaved controlled displacements before all three qubits are measured. Let $\mathbf { q } _ { j s }$ be the phase-space wavevector applied to qubit j in layer $s ,$ and let $R ( \theta _ { j s } , \phi _ { j s } )$ denote the corresponding equatorial qubit rotation. After the unknown displacement, the efective action of one layer on candidate m can be represented as

$$
\left| \psi _ { m j } ^ { ( s ) } \right. = Z ( \mathbf { q } _ { j s } \cdot \pmb { \alpha } _ { m } ) R ( \theta _ { j s } , \phi _ { j s } ) \left| \psi _ { m j } ^ { ( s - 1 ) } \right. ,\tag{B30}
$$

where $\pmb { \alpha } _ { m } = \left( I _ { m } , Q _ { m } \right)$ and $Z ( \varphi ) = \left| 0 \right. \left. 0 \right| + e ^ { i \varphi } \left| 1 \right. \left. 1 \right|$ . In our displacement convention, a wavevector $\mathbf { q } = \left( q _ { I } , q _ { Q } \right)$ corresponds to the controlled-displacement amplitude $\beta = ( q _ { Q } - i q _ { I } ) / 2$ . We constrain the mean oscillator energy immediately before every signal interaction according to

$$
E _ { m j s } = P _ { e , m j s } \frac { \| \mathbf { q } _ { j s } \| ^ { 2 } } { 4 } \leq E _ { \mathrm { c a p } } ,\tag{B31}
$$

where $P _ { e , m j s }$ is the excited-state population of qubit $j$ before layer s. We take $E _ { \mathrm { c a p } } = 1 0$ throughout.

If $p _ { m j }$ denotes the terminal probability of outcome one on qubit $j ,$ one terminal measurement produces a three-bit outcome $ { \mathbf { b } } \in \{ 0 , 1 \} ^ { 3 }$ with conditional probability

$$
P ( \mathbf { b } | m ) = \prod _ { j = 1 } ^ { 3 } p _ { m j } ^ { b _ { j } } ( 1 - p _ { m j } ) ^ { 1 - b _ { j } } .\tag{B32}
$$

We refer to one joint terminal measurement of the three qubits as one shot. A depth-ten $\mathrm { Q F S }$ shot contains at most ten signal interactions per qubit and therefore thirty signal interactions in total.

For the 64-QAM sample-complexity calculation, we optimize a symmetry-restricted but explicitly realizable subclass of these circuits. Each qubit is prepared in an equal superposition, the same phase gradient $\mathbf { q } _ { j }$ is applied in all ten layers, and the intermediate rotations are set to the identity. Choosing the final analysis phase appropriately gives

$$
p _ { m j } = \frac { 1 } { 2 } \left[ 1 + \sin ( 1 0 { \bf q } _ { j } \cdot { \pmb \alpha } _ { m } ) \right] ,\tag{B33}
$$

with $\| \mathbf { q } _ { j } \| ^ { 2 } / 8 \le E _ { \mathrm { c a p } }$ . We optimize the three accumulated phase gradients over four orientation families,

$$
\left\{ \left( 0 , \frac { \pi } { 2 } , \frac { \pi } { 4 } \right) , \left( 0 , \frac { \pi } { 3 } , \frac { 2 \pi } { 3 } \right) , \left( \frac { \pi } { 6 } , \frac { \pi } { 2 } , \frac { 5 \pi } { 6 } \right) , \left( \frac { \pi } { 4 } , \frac { 3 \pi } { 4 } , 0 \right) \right\} .\tag{B34}
$$

For each family, the three gradient magnitudes are chosen to maximize the worst-case pairwise Bhattacharyya information,

$$
\operatorname* { m a x } _ { \{ { \bf q } _ { j } \} } \operatorname* { m i n } _ { m < n } B _ { m n } , \qquad B _ { m n } = - \log \sum _ { \bf b } \sqrt { P ( { \bf b } | m ) P ( { \bf b } | n ) } .\tag{B35}
$$

For a total of N terminal shots, let $n _ { \mathbf { b } }$ denote the number of occurrences of joint outcome b. Candidate m is assigned the exact multinomial log likelihood

$$
\mathcal { L } _ { m } ^ { \mathrm { Q F S } } = \sum _ { \mathbf { b } } n _ { \mathbf { b } } \log P ( \mathbf { b } | m ) ,\tag{B36}
$$

and the transmitted symbol is decoded as the candidate with the largest likelihood. We estimate the 64-QAM decoding probability using 100,000 uniformly sampled truth and measurement episodes at each queried value of N, and report the smallest integer shot count attaining 70% success. At larger field amplitudes, we also include the achievable response obtained by attenuating the signal to a previously optimized amplitude. This preserves the conditional response of the earlier circuit and prevents an artificial degradation in the strong-signal regime.

For the one-shot calculation, we use the rectangular 8-QAM constellation. Its three Gray-coded binary partitions specify the sign of the I quadrature, whether the I quadrature is on an inner or outer level, and the sign of the Q quadrature. We train one depth-ten circuit for each binary partition at seven logarithmically spaced amplitudes between $1 0 ^ { - 2 }$ and 10. At every plotted amplitude, we evaluate the resulting response bank and choose one circuit for each qubit to maximize the exact one-shot decoding probability,

$$
P _ { \mathrm { Q F S } } ^ { ( 1 ) } = \frac { 1 } { 8 } \sum _ { \mathbf { b } } \operatorname* { m a x } _ { m } P ( { \mathbf b } | m ) .\tag{B37}
$$

The plotted curve again includes the response-preserving attenuation envelope. Since all eight possible threequbit outcomes are retained, a single terminal measurement can encode the three bits required to distinguish the eight candidate symbols.

We compare QFS against ideal heterodyne detection and an energy-matched two-mode squeezed vacuum receiver. A Gaussian receiver observation is modeled as

$$
\begin{array} { r } { { \bf y } _ { t } = { \pmb { \alpha } } _ { m } + { \bf z } _ { t } , \qquad { \bf z } _ { t } \sim { \mathcal N } ( 0 , \nu I _ { 2 } ) . } \end{array}\tag{B38}
$$

For TMSV, one mode probes the displacement while the second mode is retained as an idler and a joint Gaussian measurement is performed at the receiver. We take sinh $\lfloor ^ { 2 } r = E _ { \mathrm { { c a p } } } = 1 0$ in the signal arm, with the retained idler energy not included in the quoted constraint. Its efective noise variance is $\nu _ { \mathrm { T M S V } } = e ^ { - 2 r } \nu _ { \mathrm { h e t } }$

The benchmark Gaussian success probabilities for TMSV and heterodyne are evaluated exactly from the nearest-neighbor decision regions. Defining

$$
S _ { L } ( u ) = \frac { 2 ( L - 1 ) \Phi ( u ) - ( L - 2 ) } { L } ,\tag{B39}
$$

the success probability for square 64-QAM after N observations is

$$
P _ { 6 4 } ( N ; \nu ) = S _ { 8 } \left( A \sqrt { \frac { N } { 4 2 \nu } } \right) ^ { 2 } .\tag{B40}
$$

For rectangular $8 – \mathrm { Q A M }$ , the one-shot probability is

$$
P _ { 8 } ( 1 ; \nu ) = S _ { 4 } \bigg ( { \frac { A } { \sqrt { 6 \nu } } } \bigg ) S _ { 2 } \bigg ( { \frac { A } { \sqrt { 6 \nu } } } \bigg ) .\tag{B41}
$$

We use the conventional complex-amplitude normalization $\nu _ { \mathrm { h e t } } = 1 / 2$ . The TMSV variance is reduced by the same factor $e ^ { - 2 r }$ relative to heterodyne in each calculation.

The upper panel of Figure 4 reports the number of terminal shots required to identify a uniformly sampled 64-QAM symbol with 70% probability. At A = 0.1, QFS requires 12 terminal shots, compared with 175 TMSV shots and 7324 heterodyne shots. At $A = 0 . 0 1$ , the respective requirements are 64, 17,447, and 732,343. The lower panel reports the exact one-shot decoding probability for rectangular 8-QAM. At $A = 0 . 1$ , QFS identifies the transmitted symbol with probability 0.903, compared with 0.303 for TMSV and 0.149 for heterodyne.

## Origin of speedup

For two symbols separated by distance $d ,$ Gaussian observations with covariance $\nu I _ { 2 }$ have one-shot Bhattacharyya information

$$
B _ { \mathrm { h e t } } ( d ) = { \frac { d ^ { 2 } } { 8 \nu } } .\tag{B42}
$$

Heterodyne detection must resolve this displacement through a noisy phase-space estimate whose variance decreases as $1 / N$ . QFS instead converts the displacement diference into an accumulated relative phase,

$$
\Delta \varphi _ { m n } = \sum _ { s } \mathbf { q } _ { s } \cdot \left( \pmb { \alpha } _ { m } - \pmb { \alpha } _ { n } \right) .\tag{B43}
$$

Quantum control allows this phase to accumulate coherently over several interactions before the ancilla is measured. The three-qubit receiver additionally produces eight possible terminal outcomes rather than one binary outcome. For 8-QAM, the three qubits can be matched directly to the three binary partitions of the constellation, allowing one joint outcome to resolve all three encoded bits. For 64-QAM, repeated three-bit outcomes provide a likelihood signature that distinguishes the complete set of 64 symbols. Therefore, the highlevel intuition for the observed advantage is that QFS can trade locally-resolved measurement of any point in phase space for amplification of signal in demarcated, noncontiguous regions of tunable geometry, at the cost of a loss of resolution within the regions themselves. Thus, whereas Gaussian strategies must resolve every point equally well, a QFS receiver can combine various settings to amplify the detected response on each QAM symbol and achieve a trainable task-specific advantage at equal energy.

## Appendix C: Preliminaries and definitions

In this section we begin by reviewing concepts in quantum information theory useful for understanding our work. We next prove a number of results in Gaussian statistics relevant to Theorem 1. Then we lay out the core operational definitions that underlie our separations: we recall the Quantum Signal Learning framework from Ref. [55], understand how QSL allows us to mathematically study the sensing of classical observables, and finally explain the mathematical formulation of the sensing hierarchy in Figure 1.

## 1. Continuous-variable quantum information

Here we collect the standard facts in continuous-variable quantum mechanics used throughout this work. We set $\hbar = 1$ and work with a single bosonic mode unless stated otherwise; all statements extend to n modes by promoting phase-space coordinates to $\mathbb { R } ^ { 2 n }$ and taking direct sums.

## Quantum measurements

The most general measurement permitted by quantum mechanics is described by a positive operator-valued measure.

Definition 1 (POVM). A positive operator-valued measure $( P O V M )$ on a Hilbert space H with outcome set $\mathcal { V }$ is a collection of positive semidefinite operators $\{ M _ { y } \} _ { y \in \mathcal { Y } }$ satisfying $\begin{array} { r } { \sum _ { y } M _ { y } = I ; } \end{array}$ for continuous outcomes one works with an operator-valued measure $M ( d y )$ with $\textstyle \int _ { y } M ( d y ) = I$ . Measuring a state ρ yields outcome y with probability $\operatorname* { P r } [ y ] = \operatorname { T r } [ M _ { y } \rho ]$

While a POVM specifies the statistics a measurement produces, it does not specify how the measurement is physically enacted. That information is carried by its dilation.

Fact 1 (Naimark dilation). For every $P O V M \left\{ M _ { y } \right\}$ on H there exist an ancilla Hilbert space $\mathcal { H } _ { A }$ with a pure state $\sigma _ { A }$ , a unitary U on $\mathcal { H } \otimes \mathcal { H } _ { A }$ , and a projective measurement $\{ \Pi _ { y } \}$ such that

$$
\mathrm { T r } [ M _ { y } \rho ] = \mathrm { T r } \big [ \Pi _ { y } U ( \rho \otimes \sigma _ { A } ) U ^ { \dagger } \big ]\tag{C1}
$$

for every state ρ and outcome $y .$

Fact 1 gives POVMs an operational reading that will recur throughout this work: the dilation is an accounting of the ancillary resources used to make the measurement. Constraining a family of sensing protocols is then, mathematically, the act of constraining the allowed POVM family through its dilations: which ancilla states may be prepared, which joint unitaries may be applied, and which projective readouts are available. As a concrete example, the class of Gaussian (generaldyne) measurements reviewed below consists, in full generality, of exactly those POVMs admitting a dilation in which the ancilla state is Gaussian, the joint unitary is Gaussian, and the final readout is homodyne, followed by classical postprocessing; the energy constraint of our conventional class then amounts to bounding the total mean photon number of probe and ancilla in the dilation. Even non-continuous-variable resources fit into this picture: a protocol that couples the sensor mode to an ancilla qubit and reads out the qubit is, from the perspective of the mode alone, simply another POVM on the mode one obtained from a dilation with a two-dimensional ancilla, and, as we will see in Appendix $\mathrm { E } ,$ one that no Gaussian dilation can reproduce. The sensing hierarchy of Figure 1(b) is in this sense a nested family of dilation constraints, and quantum-enhanced sensing is the enlargement of the accessible POVM family by new dilation resources. Finally, a multi-round protocol is a sequence of such measurements in which the POVM applied in each round may be chosen adaptively from the classical transcript of earlier outcomes; this is the form in which adaptivity enters all of our lower bounds.

## Displacement, squeezing, and Gaussian states

A bosonic mode is described by canonical quadrature operators $\hat { x } , \hat { p }$ obeying $[ \hat { x } , \hat { p } ] = i ,$ related to the annihilation operator by $\hat { x } = ( \hat { a } ^ { \dagger } + \hat { a } ) / \sqrt { 2 }$ and $\hat { p } = i ( \hat { a } ^ { \dag } - \hat { a } ) / \sqrt { 2 }$ , and collected into the quadrature vector $\hat { z } = ( \hat { x } , \hat { p } )$ . This convention fixes the vacuum noise floor: the vacuum state |0⟩ satisfies $\Delta x ^ { 2 } = \Delta p ^ { 2 } = 1 / 2$

We use the single-mode phase-space convention $\alpha = ( q _ { \alpha } , p _ { \alpha } ) \in \mathbb { R } ^ { 2 }$ . The standard symplectic form is

$$
\Omega ( \zeta , \alpha ) = q _ { \zeta } p _ { \alpha } - p _ { \zeta } q _ { \alpha } .\tag{C2}
$$

If J is the standard symplectic matrix, then $\Omega ( \zeta , \alpha ) = \zeta ^ { T } J \alpha$ . For each $\zeta \in \mathbb { R } ^ { 2 }$ , define the Weyl character

$$
\begin{array} { r } { \chi _ { \zeta } ( \alpha ) = e ^ { i \Omega ( \zeta , \alpha ) } . } \end{array}\tag{C3}
$$

For each phase-space point $\alpha ,$ the displacement operator is

$$
D ( \alpha ) = e ^ { - i \Omega ( \alpha , \hat { z } ) } = e ^ { i ( p _ { \alpha } \hat { x } - q _ { \alpha } \hat { p } ) } ,\tag{C4}
$$

which shifts the quadratures as $D ( \alpha ) ^ { \dag } \hat { z } D ( \alpha ) = \hat { z } + \alpha$ and generates the coherent states $\left| \alpha \right. = D ( \alpha ) \left| 0 \right.$ from the vacuum. These characters are the natural Fourier modes of phase-space displacement sensing. The reason is the Weyl relation

$$
D ( a ) D ( b ) = e ^ { - i \Omega ( a , b ) / 2 } D ( a + b ) ,\tag{C5}
$$

which says that the noncommutativity of phase-space displacements is exactly measured by the symplectic phase $\Omega ( a , b )$ . Thus whenever a known control displacement is coherently compared with an unknown signal displacement, the qubit response is built from phases of the form $e ^ { i \Omega ( \zeta , \alpha ) }$

The second elementary Gaussian resource is squeezing. The single-mode squeezer $S ( r )$ is the unitary defined by its action on the quadratures,

$$
S ( r ) ^ { \dagger } \hat { x } S ( r ) = e ^ { - r } \hat { x } , \qquad S ( r ) ^ { \dagger } \hat { p } S ( r ) = e ^ { r } \hat { p } ,\tag{C6}
$$

so that the squeezed vacuum $S ( r ) \left| 0 \right.$ has quadrature variances $\Delta x ^ { 2 } = e ^ { - 2 r } / 2$ and $\Delta p ^ { 2 } = e ^ { 2 r } / 2$ , purchasing sub-vacuum resolution along one axis at a mean photon cost of sinh<sup>2</sup> r.

A Gaussian state is one whose Wigner function, reviewed below, is a Gaussian density on phase space. Such states are fully specified by their mean vector and covariance matrix,

$$
\begin{array} { r } { m _ { j } = \mathrm { T r } [ \rho \hat { z } _ { j } ] , \qquad V _ { j k } = \frac 1 2 \mathrm { T r } [ \rho \left\{ \hat { z } _ { j } - m _ { j } , \hat { z } _ { k } - m _ { k } \right\} ] , } \end{array}\tag{C7}
$$

with the vacuum corresponding to $m = 0$ and $V = I _ { 2 } / 2$ . Gaussian states are exactly those generated from the vacuum by Gaussian unitaries — unitaries generated by Hamiltonians at most quadratic in the quadratures — which for a single mode are compositions of displacements, phase-space rotations, and squeezers, supplemented by beamsplitters in the multimode setting. The following elementary fact governs all of the energy accounting in this work.

Fact 2 (Energy of a Gaussian state). A single-mode Gaussian state with mean vector m and covariance matrix V has mean photon number

$$
\begin{array} { r } { \bar { n } = \mathrm { T r } \left[ \rho \hat { a } ^ { \dagger } \hat { a } \right] = \frac { 1 } { 2 } \left( \mathrm { T r } V + \| m \| ^ { 2 } - 1 \right) , } \end{array}\tag{C8}
$$

which follows directly from $\hat { a } ^ { \dagger } \hat { a } = ( \hat { x } ^ { 2 } + \hat { p } ^ { 2 } - 1 ) / 2$ . In particular, an energy budget $\bar { n } \leq E$ enforces Tr $V + \| m \| ^ { 2 } \leq$ $2 E + 1$

Fact 2 cleanly separates the two ways a Gaussian protocol can spend energy. The term $\scriptstyle { \frac { 1 } { 2 } } \| m \| ^ { 2 }$ is the classical share of the budget, spent on coherent displacement, while $\textstyle { \frac { 1 } { 2 } } ( \operatorname { T r } V - 1 ) \geq 0$ counts the photons invested in squeezing (or absorbed as thermal noise); the uncertainty relation det $V \geq 1 / 4$ forces $\operatorname { T r } V \geq 1$ , with equality precisely for coherent states. It is the latter, nonclassical share of the budget that powers conventional quantum sensing, and holding the total budget fixed across protocol classes is what makes our separations statements about quantum information processing rather than about energy. We remark that a few proofs in the appendices rescale phase-space coordinates so that a convenient reference Gaussian has identity covariance; wherever this is done, the convention is stated locally.

## Homodyne, heterodyne, and generaldyne measurements

The two ubiquitous measurement primitives of quantum optics sit at opposite extremes of a single Gaussian family. Homodyne measurement is the projective measurement of a single rotated quadrature $\hat { x } _ { \varphi } = \cos { \varphi } \hat { x } +$ sin $\varphi \hat { p } ,$ with POVM elements $\vert x _ { \varphi } \rangle \langle x _ { \varphi } \vert$ and outcome density $P _ { \rho } ^ { \varphi } ( x ) = \langle { x _ { \varphi } } | { \rho } | { x _ { \varphi } } \rangle ;$ equivalently, as recalled below, $P _ { \rho } ^ { \varphi }$ is the marginal of the Wigner function of ρ along the direction conjugate to $\hat { x } _ { \varphi }$ . Because the outcome variance is exactly the intrinsic quadrature variance of the state, homodyne detection resolves squeezed fluctuations at the level $e ^ { - 2 r } / 2$ when the squeezing axis is aligned with the measured quadrature; but it reveals only a onedimensional marginal per shot. Physically, it is implemented by interfering the signal with a strong local oscillator at phase $\varphi$ on a balanced beamsplitter and subtracting photocurrents; in microwave platforms, IQ mixing and phase-sensitive amplification play the same role.

Heterodyne measurement is the coherent-state POVM

$$
M ( d ^ { 2 } \alpha ) = \frac { 1 } { 2 \pi } \left| \alpha \right. \langle \alpha | d ^ { 2 } \alpha ,\tag{C9}
$$

which measures both quadratures at once. Its outcome density is the Wigner function of $\rho$ convolved with the vacuum Gaussian, so every heterodyne record carries an irreducible unit of vacuum noise that is independent of the probe state and cannot be removed by squeezing. In exchange, the record is informationally complete. Physically, heterodyne is realized as double homodyne: the signal is split with vacuum on a balanced beamsplitter and the two outputs are homodyned along conjugate axes.

The full Gaussian measurement family interpolating between these extremes is the generaldyne class, which is most usefully defined through its dilation.

Definition 2 (Generaldyne measurement). A generaldyne measurement is any POVM realized by appending ancilla modes prepared in Gaussian states, applying a Gaussian unitary to the joint system, and performing homodyne measurements on a commuting set of output quadratures.

Definition 2 makes the resource content of conventional bosonic sensing explicit, and it has a structural consequence that drives our conventional lower bounds: because Gaussian dilations act afinely on quadratures, the outcome of any generaldyne experiment performed on a Gaussian probe subject to a deterministic displacement z is itself a Gaussian random variable, $Y \sim \mathcal { N } ( A z + b , C )$ , with $( A , b , C )$ determined by the probe, dilation, and readout (this is restated and proven in Appendix E, where it is first used). Together with classical adaptivity between shots and the energy constraint of Fact 2, generaldyne measurements on Gaussian probes constitute the class of conventional finite-energy protocols against which our quantum-enhanced separations are proven.

## Wigner functions and Weyl symbols

Phase space furnishes a faithful representation of bosonic states and observables, and it is the representation in which all of our arguments take place. For states, the standard object is the Wigner function.

Definition 3 (Wigner function). The Wigner function of a single-mode state $\rho$ is

$$
W _ { \rho } ( \alpha ) = \frac { 1 } { 2 \pi } \int _ { \mathbb { R } } d y e ^ { i p _ { \alpha } y } \langle q _ { \alpha } - y / 2 | \rho | q _ { \alpha } + y / 2 \rangle .\tag{C10}
$$

Fact 3 (Properties of the Wigner function). The Wigner function is real and normalized, $\textstyle \int d ^ { 2 } \alpha W _ { \rho } ( \alpha ) = 1$ though it may take negative values; by Hudson’s theorem, a pure state has a nonnegative Wigner function if and only if it is Gaussian. Its marginals reproduce homodyne statistics: integrating $W _ { \rho }$ along the direction conjugate to $\hat { x } _ { \varphi }$ yields $P _ { \rho } ^ { \varphi }$ . It is covariant under displacements, ${ W _ { D ( \beta ) \rho D ( \beta ) ^ { \dagger } } ( \alpha ) = W _ { \rho } ( \dot { \alpha } - \beta ) }$ , and $a$ Gaussian state with moments $( m , V )$ has Wigner function $\mathcal { N } ( \alpha ; m , V )$ . Finally, overlaps are computed by the rule $\operatorname { T r } [ \rho \sigma ] =$ $\begin{array} { r } { 2 \pi \int d ^ { 2 } \alpha W _ { \rho } ( \alpha ) \dot { W } _ { \sigma } ( \dot { \alpha } ) } \end{array}$

The Wigner function extends from states to arbitrary operators. This extension, the Weyl symbol, provides the phase-space intuition underlying the $\mathrm { Q } \Psi$ framework of Appendix D.

Definition 4 (Weyl symbol). The Weyl symbol of an operator O is

$$
W _ { O } ( \alpha ) = \int _ { \mathbb { R } } d y e ^ { i p _ { \alpha } y } \langle q _ { \alpha } - y / 2 | O | q _ { \alpha } + y / 2 \rangle ,\tag{C11}
$$

so that the Wigner function is $1 / 2 \pi$ times the Weyl symbol of the density operator. The mismatched normalization is chosen so that states integrate to one while expectation values obey the clean trace rule

$$
\mathrm { T r } [ O \rho ] = \int d ^ { 2 } \alpha W _ { O } ( \alpha ) W _ { \rho } ( \alpha ) .\tag{C12}
$$

Fact 4 (Properties of Weyl symbols). The Weyl symbol is linear in $O ,$ real for Hermitian $O ,$ and satisfies $W _ { I } = 1$ . Consequently, the symbols of a POVM form a partition of unity on phase space, $\begin{array} { r } { \sum _ { y } W _ { M _ { y } } ( \alpha ) = 1 } \end{array}$ pointwise — but positivity of $M _ { y }$ does not imply positivity of its symbol, and this signed structure is precisely

the non-Gaussian resource exploited in this work. The displacement operators are the operator representation of phase-space Fourier modes:

$$
W _ { D ( \zeta ) } ( \alpha ) = e ^ { - i \Omega ( \zeta , \alpha ) } = \overline { { { \chi _ { \zeta } ( \alpha ) } } } ,\tag{C13}
$$

so that expanding an operator in the displacement basis, $\begin{array} { r } { O = \frac { 1 } { 2 \pi } \int d ^ { 2 } \zeta \operatorname { T r } \bigl [ O D ( \zeta ) ^ { \dagger } \bigr ] D ( \zeta ) } \end{array}$ , is the same as symplectically Fourier-expanding its Weyl symbol. For a state, this Fourier data is the characteristic function $\begin{array} { r } { \chi _ { \rho } ( \zeta ) = \mathrm { T r } [ \rho D ( \zeta ) ] = \int d ^ { 2 } \alpha \dot { W } _ { \rho } ( \alpha ) \overline { { \chi _ { \zeta } ( \alpha ) } } } \end{array}$ , the symplectic Fourier transform of the Wigner function.

Gaussian states and generaldyne POVM elements have Gaussian symbols; by contrast, the photon-number parity operator Π<sup>ˆ</sup> satisfies $\mathrm { T r } \left\lceil \hat { \Pi } D ( \zeta ) \right\rceil = 1 / 2$ for every $\zeta ,$ so its Weyl symbol is the distribution $\pi \delta ^ { 2 } ( \alpha )$ . It therefore has uniform weight across all symplectic frequencies, in sharp contrast to a Gaussian symbol. This is why photon-counting measurements can solve qualitatively diferent problems than conventional Gaussian sensing.

We say that a measurement is Wigner-positive if its Weyl symbol is nonnegative everywhere in phase space; all Gaussian measurements are Wigner-positive. Moreover, Wigner-positivity has an important learning-theoretic consequence.

Fact 5. Given N copies of any state ρ, the outcome of any strategy making Wigner-positive measurements of $\rho ,$ interleaved with classical adaptivity, is exactly equivalent to classical postprocessing of the Wigner function of $\rho ^ { \otimes N }$

This fact arises because positive Weyl symbols exactly form a Markov kernel, so any distribution $P ^ { ( N ) }$ over measurement transcripts can be written as

$$
P ^ { ( N ) } ( y _ { 1 . . . N } ) = \int \kappa ^ { ( N ) } ( y _ { 1 . . . N } | z _ { 1 . . . N } ) W _ { \rho } ^ { \otimes N } ( z _ { 1 . . . N } ) d z _ { 1 . . . N } \ .\tag{C14}
$$

Weyl symbols and the measurement kernels at the heart of $\mathrm { Q } \Psi$ go hand in hand. Consider any protocol that prepares a probe $\rho ,$ exposes it to a deterministic signal displacement $D ( \alpha )$ , and measures a POVM element $M _ { y }$ Its response function is, by the trace rule (C12) and displacement covariance,

$$
f _ { M , y } ( \alpha ) = \mathrm { T r } \big [ M _ { y } D ( \alpha ) \rho D ( \alpha ) ^ { \dagger } \big ] = \int d ^ { 2 } z W _ { M _ { y } } ( z ) W _ { \rho } ( z - \alpha ) ~ .\tag{C15}
$$

The classical statistical response of any experiment is the Weyl symbol of its measurement, smoothed by the Wigner function of its probe. Taking symplectic Fourier transforms, the convolution theorem factorizes the response frequency by frequency,

$$
\hat { f } _ { M , y } ( \zeta ) = 2 \pi ~ \mathrm { T r } [ M _ { y } D ( \zeta ) ] \overline { { { \chi _ { \rho } ( \zeta ) } } } , \qquad \hat { f } ( \zeta ) : = \int d ^ { 2 } \alpha f ( \alpha ) e ^ { - i \Omega ( \zeta , \alpha ) } .\tag{C16}
$$

When the probe is entangled with ancillas, or the measurement is qubit-assisted, the product on the right is replaced by a joint coeficient, but the conclusion is unchanged: every response function is a superposition of Weyl characters, and its amplitude at frequency $\zeta$ must be supplied jointly by the probe and the measurement. This identity is the intuition underlying the entire QΨ formalism, and it holds on any phase space – not only on bosonic ones. The functions $f _ { M , y }$ are exactly the response kernels $K _ { M } ( d y | \alpha )$ of Appendix D, and the accessible feature information measures their overlap with the phase-space representation of the target observable.

QΨ is, at heart, harmonic analysis of Weyl symbols under the dilation constraints of a physical experiment – or in this paper, constraints of the sensing hierarchy. It also renders our separations intuitive before any calculation. A Gaussian state with covariance V has $\begin{array} { r } { | \chi _ { \rho } ( \zeta ) | = \exp \left( - \frac { 1 } { 2 } \zeta ^ { T } ( J V J ^ { T } ) \zeta \right) } \end{array}$ , and Fact 2 bounds $\operatorname { T r } V \leq$ $2 E + 1$ , so an energy-E Gaussian probe (and, by Definition $2 ,$ an energy-E generaldyne measurement) carries appreciable Fourier weight only within an ellipse whose longest semi-axis is $O ( { \sqrt { E } } )$ ; its response to a frequency-k character is consequently suppressed as $\exp \bar { \left( - \Omega ( k ^ { 2 } / E ) \right) }$ even when the squeezing is optimally aligned, which is the mechanism behind the lower bound of Theorem 1. A single echoed conditional displacement with gate parameter $\beta ,$ by contrast, imprints the character $\chi _ { \beta }$ on the qubit coherence directly, so that frequency k is reached with response amplitude of order $\sqrt { E } / k$ , which is polynomially, rather than exponentially, small. The exponential separations of this work are precisely the formalization of this gap in a variety of settings.

## 2. Gaussian learning and estimation theory

Theorem 1 establishes an exponential lower bound against conventional bosonic quantum sensors using Gaussian operations. To prove this bound, we will prove a number of preliminary lemmas in Gaussian statistical learning theory and linear algebra. The role of these results is to control the measurement distribution of energy-constrained Gaussian protocols.

A general conventional Gaussian sensing experiment allows the preparation of an arbitrary Gaussian probe state, possibly entangled with many Gaussian ancillas, followed by interrogation of the signal channel and arbitrary Gaussian measurement. For full generality, we allow the final measurement to be any generaldyne measurement using ancilla systems, and we allow the collected classical data to be used to adaptively inform subsequent signal queries. The formal definition is given in Definition 2. The following lemma allows us to characterize the output of any such quantum experiment using classical Gaussian statistics.

Lemma 5. Let Y be the random variable associated with the outcome of any single-mode Gaussian experiment in which the signal acts as a displacement $D _ { S } ( z )$ , where the state before measurement is

$$
D _ { S } ( z ) \rho _ { S A } D _ { S } ^ { \dagger } ( z ) ~ .\tag{C17}
$$

Then there exists a real matrix $A ,$ real vector $b ,$ and covariance matrix $C$ such that $Y$ is distributed as $Y \sim$ $\mathcal { N } ( A z + b , C )$

Proof. Fix a single experiment, including the choice made from any previous transcript. A generaldyne measurement, by definition, admits a Gaussian dilation: one appends Gaussian ancillas, applies a Gaussian unitary, and homodynes a collection of commuting quadratures. Let R be the quadrature vector of all modes in this dilation before the final Gaussian unitary. Since the probe and measurement ancillas are Gaussian, the joint state is Gaussian with some mean vector m and covariance matrix $V .$ The signal displacement translates only the signal quadratures, so there is a fixed real matrix L such that after applying $D _ { S } ( z )$ the mean is

$$
m ( z ) = m + L z\tag{C18}
$$

and the covariance remains V . The Gaussian unitary in the measurement dilation acts on quadratures as

$$
R \longmapsto S R + d\tag{C19}
$$

for a real symplectic matrix $S$ and a real vector d. The homodyne readout then selects a fixed real linear projection of these quadratures, so for some real matrix M, the measurement outcome is distributed as the linear margina

$$
\begin{array} { r } { Y = M ( S R + d ) ~ . } \end{array}\tag{C20}
$$

Linear marginals of Gaussian states are Gaussian. Hence, conditional on the displacement z, the outcome has mean and covariance

$$
\mathbb { E } [ Y \mid z ] = M S ( m + L z ) + M d ,\tag{C21}
$$

$$
\operatorname { C o v } ( Y \mid z ) = M S V S ^ { T } M ^ { T } ~ .\tag{C22}
$$

Setting

$$
A = M S L , \qquad b = M S m + M d , \qquad C = M S V S ^ { T } M ^ { T } ,\tag{C23}
$$

gives $Y \sim \mathcal { N } ( A z + b , C )$

This lemma tells us that regardless of the adaptivity we allow between measurements, any single experiment in a conventional Gaussian protocol with a deterministic displacement has an outcome distribution that follows some Gaussian distribution; we can then use linearity to compute the output distribution of a displacement channel. This fact heavily constrains the kinds of integrals we need to bound when controlling the total variation. However, we need a way to quantify the efect of an experimental energy constraint on this measurement distribution. This is done with the following lemmas.

First, we recall the definition of the classical and quantum Fisher information, which are used in the next lemma.

Definition 6 (Classical and quantum Fisher information). Let $\{ \rho _ { z } \} _ { z \in \mathbb { R } ^ { d } }$ be a diferentiable family of quantum states, and let a fixed measurement produce an outcome distribution with density $p _ { z } ( y )$ . The classical Fisher information matrix of the outcome distribution is

$$
[ F _ { c } ( z ) ] _ { j k } = \mathbb { E } _ { Y \sim p _ { z } } \left[ \frac { \partial \log p _ { z } ( Y ) } { \partial z _ { j } } \frac { \partial \log p _ { z } ( Y ) } { \partial z _ { k } } \right] .\tag{C24}
$$

For each $j ,$ define the symmetric logarithmic derivative $( S L D )  L _ { j }$ to be a Hermitian operator satisfying

$$
\frac { \partial \rho _ { z } } { \partial z _ { j } } = \frac { 1 } { 2 } \left( L _ { j } \rho _ { z } + \rho _ { z } L _ { j } \right) .\tag{C25}
$$

The quantum Fisher information matrix is then

$$
[ F _ { Q } ( z ) ] _ { j k } = \frac { 1 } { 2 } \operatorname { T r } \left( \rho _ { z } \{ L _ { j } , L _ { k } \} \right) .\tag{C26}
$$

For any family of quantum states and any fixed measurement, the classical Fisher information is bounded by the quantum Fisher information, $F _ { c } ( z ) \preceq F _ { Q } ( z )$ . Moreover, the following specialization to single-parameter unitary families is a well-known fact in quantum metrology.

Lemma 7 (Quantum Fisher information of a unitary family). Let $\rho _ { \theta } = e ^ { - i \theta G } \rho e ^ { i \theta G }$ . Then the quantum Fisher information $o f \left\{ \rho _ { \theta } \right\} _ { \theta }$ satisfies

$$
F _ { Q } ( \rho , G ) \leq 4 \operatorname { V a r } _ { \rho } ( G ) .\tag{C27}
$$

Proof. Write $\begin{array} { r } { \rho = \sum _ { j } p _ { j } | j  \langle j | } \end{array}$ . The quantum Fisher information of the unitary family is

$$
F _ { Q } ( \rho , G ) = 2 \sum _ { j , k : p _ { j } + p _ { k } > 0 } \frac { ( p _ { j } - p _ { k } ) ^ { 2 } } { p _ { j } + p _ { k } } | \langle j | G | k \rangle | ^ { 2 } .\tag{C28}
$$

Since $( p _ { j } - p _ { k } ) ^ { 2 } \leq ( p _ { j } + p _ { k } ) ^ { 2 }$ 2

$$
F _ { Q } ( \rho , G ) \leq 2 \sum _ { j , k } ( p _ { j } + p _ { k } ) | \langle j | G | k \rangle | ^ { 2 } = 4 \operatorname { T r } \left( \rho G ^ { 2 } \right) .\tag{C29}
$$

Adding a scalar multiple of the identity to $G$ does not change $\rho _ { \theta }$ . Applying the preceding bound to ${ \widetilde { G } } =$ $G - \operatorname { T r } ( \rho G ) I$ therefore gives

$$
F _ { Q } ( \rho , G ) = F _ { Q } ( \rho , { \widetilde G } ) \leq 4 \operatorname { T r } \left( \rho { \widetilde G } ^ { 2 } \right) = 4 \operatorname { V a r } _ { \rho } ( G ) .\tag{C30}
$$

□

Finally, we record a special feature of classical Fisher information of Gaussian random variables.

Lemma 8. Let $Y \sim \mathcal { N } ( A z + b , C )$ be a Gaussian random variable where C is full rank and where $z \in \mathbb { R } ^ { m }$ for any $m \in \mathbb { Z } _ { + }$ . Then the classical Fisher information of $\cdot Y$ with respect to z is

$$
F _ { c } ( z ) = A ^ { T } C ^ { + } A \ ,\tag{C31}
$$

where $C ^ { + }$ is the Moore-Penrose pseudoinverse of the covariance matrix $C .$

Proof. Let $X = Y - b ;$ ; the Fisher information of $Y$ is the same as the Fisher information of X since they exactly determine each other. Then:

$$
F _ { c } ( z ) = \mathbb { E } _ { x \sim p _ { z } ( x ) } \left[ \nabla \log p _ { z } ( x ) ( \nabla \log p _ { z } ( x ) ) ^ { T } \right] .\tag{C32}
$$

Directly evaluating the derivative of the log-likelihood using the PDF of a multivariate Gaussian,

$$
\nabla \log p _ { z } ( x ) = \nabla \left[ - \frac { 1 } { 2 } ( x - A z ) ^ { T } C ^ { + } ( x - A z ) + \mathrm { c o n s t . } \right]\tag{C33}
$$

$$
= \frac { 1 } { 2 } \left[ A ^ { T } C ^ { + } ( x - A z ) + ( x - A z ) ^ { T } C ^ { + } A \right]\tag{C34}
$$

$$
= A ^ { T } C ^ { + } ( x - A z )\tag{C35}
$$

where in the final equality we use that $C ^ { + }$ is symmetric to rearrange terms. Substituting back into the Fisher information,

$$
F _ { c } ( z ) = \mathbb { E } _ { x \sim p _ { z } ( x ) } \left[ A ^ { T } C ^ { + } ( x - A z ) ( x - A z ) ^ { T } C ^ { + } A \right]\tag{C36}
$$

$$
= A ^ { T } C ^ { + } C C ^ { + } A = A ^ { T } C ^ { + } A\tag{C37}
$$

where the second equality uses the definition of the covariance matrix. This completes the proof.

We require a lemma which allows us to relate the obtained classical distribution to the energy utilized in the quantum experiment. This is the conceptual core of the separation.

Lemma 9. Let $Y \sim \mathcal { N } ( A z + b , C )$ be the random variable distributed according to the outcome of any conventional single-mode protocol limited to energy $E ,$ for a single displacement. Then the covariance matrix C necessarily satisfies

$$
C \succeq ( 8 E + 4 ) ^ { - 1 } A A ^ { T } \enspace .\tag{C38}
$$

Equivalently,

$$
A ^ { T } C ^ { + } A \preceq 8 E + 4 I _ { 2 } \ ,\tag{C39}
$$

where $C ^ { + }$ is the Moore-Penrose pseudoinverse of $C .$

Proof. In this proof, we use the quantum Fisher information and some of its known properties as an intermediate step to obtain the bound. Recall that the state before measurement is

$$
\rho _ { z } = D _ { S } ( z ) \rho _ { S A } D _ { S } ^ { \dagger } ( z ) ~ .\tag{C40}
$$

Let the direction of $z$ be an unknown parameter. That is, let $u = ( u _ { x } , u _ { p } ) = z / \| z \| _ { 2 }$ . The generator of this normalized displacement is the operator

$$
G _ { u } = u _ { x } \hat { p } _ { S } - u _ { p } \hat { x } _ { S } .\tag{C41}
$$

$\mathrm { B y }$ Lemma 7, $F _ { Q } ( u ) \leq 4 \mathrm { V a r } _ { \rho _ { S A } } ( G _ { u } )$ . Now notice that $\operatorname { V a r } _ { \rho _ { S A } } ( G _ { u } ) \le \langle \hat { x } _ { S } ^ { 2 } + \hat { p } _ { S } ^ { 2 } \rangle = \operatorname { T r } ( V )$ , where V is the covariance matrix of $\rho _ { S }$ . By Fact 10, we have $\begin{array} { r } { \operatorname { V a r } _ { \rho _ { S A } } ( G _ { u } ) \leq 2 E + 1 } \end{array}$ , and therefore,

$$
F _ { Q } \preceq ( 8 E + 4 ) I _ { 2 } ~ .\tag{C42}
$$

Since $C$ results from a Gaussian measurement, it is full-rank, so im $( A ) \subseteq \operatorname { i m } ( C )$ , so by Lemma 8, $F _ { c }$ of the outcome $Y$ satisfies $F _ { c } = A ^ { T } C ^ { + } A$ . Because $F _ { c } \preceq F _ { Q }$ , it follows that

$$
A ^ { T } C ^ { + } A \preceq ( 8 E + 4 ) I _ { 2 } ~ .\tag{C43}
$$

Lemma 10. A one-mode Gaussian state with mean vector m and covariance V has mean photon number

$$
\bar { n } = \frac { 1 } { 2 } ( \mathrm { T r } V + \| m \| ^ { 2 } - 1 ) \ .\tag{C44}
$$

For an energy constraint $\bar { n } \leq E$ (working in $\hbar = 1$ units), this implies

$$
\mathrm { T r } V \leq 2 E + 1\tag{C45}
$$

Proof. Let $R = ( \hat { x } , \hat { p } ) ^ { T }$ , with mean vector $m = \langle R \rangle$ and covariance matrix

$$
V _ { i j } = \frac { 1 } { 2 } \langle \{ R _ { i } - m _ { i } , R _ { j } - m _ { j } \} \rangle ~ .\tag{C46}
$$

In the convention $a = ( { \hat { x } } + i { \hat { p } } ) / { \sqrt { 2 } } ,$ the number operator is

$$
\hat { n } = a ^ { \dagger } a = \frac { 1 } { 2 } ( \hat { x } ^ { 2 } + \hat { p } ^ { 2 } - 1 ) \ .\tag{C47}
$$

Therefore

$$
\bar { n } = \frac { 1 } { 2 } \left( \langle \hat { x } ^ { 2 } \rangle + \langle \hat { p } ^ { 2 } \rangle - 1 \right)\tag{C48}
$$

$$
= { \frac { 1 } { 2 } } \left( V _ { 1 1 } + V _ { 2 2 } + m _ { 1 } ^ { 2 } + m _ { 2 } ^ { 2 } - 1 \right)\tag{C49}
$$

$$
= { \frac { 1 } { 2 } } \left( \operatorname { T r } V + \| m \| ^ { 2 } - 1 \right) ~ .\tag{C50}
$$

If $\bar { n } \leq E$ , then

$$
\mathrm { T r } V + \| m \| ^ { 2 } \leq 2 E + 1 .\tag{C51}
$$

Since $\| m \| ^ { 2 } \geq 0$ , this implies Tr $V \leq 2 E + 1$

Our bounds will also require the following results in classical Gaussian estimation theory and linear algebra.

Lemma 11 (Gaussian conditioning formula). Let $( A , B )$ be random variables that are jointly Gaussian (each may itself be multivariate). Let (A,<sup>¯</sup> B<sup>¯</sup>) denote the mean of the joint distribution, and let

$$
\Sigma = \left( { \begin{array} { l l } { \Sigma _ { A A } } & { \Sigma _ { A B } } \\ { \Sigma _ { B A } } & { \Sigma _ { B B } } \end{array} } \right)\tag{C52}
$$

be the joint covariance matrix in block-matrix form. Then the conditional variable $A | B = b$ is distributed as $\mathcal { N } ( \bar { \mu } , \bar { \Sigma } )$ , where

$$
\bar { \mu } = \mu _ { A } + \Sigma _ { A B } \Sigma _ { B B } ^ { - 1 } ( b - \mu _ { B } ) ~ ,\tag{C53}
$$

$$
\bar { \Sigma } = \Sigma _ { A A } - \Sigma _ { A B } \Sigma _ { B B } ^ { - 1 } \Sigma _ { B A } ~ .\tag{C54}
$$

Proof. Assume first that $\Sigma _ { B B }$ is invertible. Define

$$
K = \Sigma _ { A B } \Sigma _ { B B } ^ { - 1 }\tag{C55}
$$

and write

$$
A = \mu _ { A } + K ( B - \mu _ { B } ) + R\tag{C56}
$$

where

$$
R = A - \mu _ { A } - K ( B - \mu _ { B } ) ~ .\tag{C57}
$$

Since $( A , B )$ is jointly Gaussian, the pair $( R , B )$ is jointly Gaussian. We now compute the mean and covariance of R. First,

$$
\begin{array} { r l } & { \mathbb { E } [ R ] = \mathbb { E } [ A ] - \mu _ { A } - K ( \mathbb { E } [ B ] - \mu _ { B } ) } \\ & { \quad \quad = 0 ~ . } \end{array}\tag{C58}
$$

(C59)

Next,

$$
\operatorname { C o v } ( R , B ) = \operatorname { C o v } ( A - \mu _ { A } - K ( B - \mu _ { B } ) , B )\tag{C60}
$$

$$
= \Sigma _ { A B } - K \Sigma _ { B B }\tag{C61}
$$

$$
= \Sigma _ { A B } - \Sigma _ { A B } \Sigma _ { B B } ^ { - 1 } \Sigma _ { B B }\tag{C62}
$$

(C63)

Thus R and B are uncorrelated jointly Gaussian random variables, so they are independent. Conditioning on $B = b$ therefore makes the term $\mu _ { A } + K ( B - \mu _ { B } )$ deterministic, but leaves the distribution over R unchanged,

allowing us to reduce the randomness in $A | B$ to randomness in R. Hence

$$
A \mid ( B = b ) = \mu _ { A } + K ( b - \mu _ { B } ) + R\tag{C64}
$$

in distribution. The conditional mean is therefore

$$
\mathbb { E } [ A \mid B = b ] = \mu _ { A } + K ( b - \mu _ { B } ) + \mathbb { E } [ R ]\tag{C65}
$$

$$
\begin{array} { r } { = \mu _ { A } + \Sigma _ { A B } \Sigma _ { B B } ^ { - 1 } ( b - \mu _ { B } ) . } \end{array}\tag{C66}
$$

The conditional covariance is

$$
\operatorname { C o v } ( A \mid B = b ) = \operatorname { C o v } ( R )
$$

$$
= \operatorname { C o v } ( A - K B )\tag{C67}
$$

(C68)

$$
= \Sigma _ { A A } - K \Sigma _ { B A } - \Sigma _ { A B } K ^ { T } + K \Sigma _ { B B } K ^ { T }\tag{C69}
$$

$$
= \Sigma _ { A A } - \Sigma _ { A B } \Sigma _ { B B } ^ { - 1 } \Sigma _ { B A } ~ .\tag{C70}
$$

Since an afine transformation of a jointly Gaussian random variable is Gaussian, this proves

$$
A \mid B = b \sim { \cal N } \left( \mu _ { A } + \Sigma _ { A B } \Sigma _ { B B } ^ { - 1 } ( b - \mu _ { B } ) , \Sigma _ { A A } - \Sigma _ { A B } \Sigma _ { B B } ^ { - 1 } \Sigma _ { B A } \right) \ .\tag{C71}
$$

Lemma 12 (Woodbury identity). Let $C \succ 0$ be a real covariance matrix and let A be a real matrix with compatible dimensions. Then

$$
I _ { 2 } - A ^ { T } ( A A ^ { T } + C ) ^ { - 1 } A = ( I _ { 2 } + A ^ { T } C ^ { - 1 } A ) ^ { - 1 } \ .\tag{C72}
$$

Proof. Define

$$
M = I _ { 2 } + A ^ { T } C ^ { - 1 } A \ .\tag{C73}
$$

We first claim that

$$
( A A ^ { T } + C ) ^ { - 1 } A = C ^ { - 1 } A M ^ { - 1 } \ .\tag{C74}
$$

Indeed,

$$
( A A ^ { T } + C ) C ^ { - 1 } A M ^ { - 1 } = ( A + A A ^ { T } C ^ { - 1 } A ) M ^ { - 1 }\tag{C75}
$$

$$
= A ( I _ { 2 } + A ^ { T } C ^ { - 1 } A ) M ^ { - 1 }\tag{C76}
$$

$$
= A M M ^ { - 1 }\tag{C77}
$$

$$
= A .\tag{C78}
$$

Since $A A ^ { T } + C$ is invertible, the claim follows. Therefore,

$$
A ^ { T } ( A A ^ { T } + C ) ^ { - 1 } A = A ^ { T } C ^ { - 1 } A M ^ { - 1 }\tag{C79}
$$

$$
= ( M - I _ { 2 } ) M ^ { - 1 }\tag{C80}
$$

$$
\begin{array} { r } { { \bf \Psi } = I _ { 2 } - M ^ { - 1 } . } \end{array}\tag{C81}
$$

Rearranging gives

$$
I _ { 2 } - A ^ { T } ( A A ^ { T } + C ) ^ { - 1 } A = M ^ { - 1 } = ( I _ { 2 } + A ^ { T } C ^ { - 1 } A ) ^ { - 1 } .\tag{C82}
$$

The above lemmas will be used in Section E 2. For Section E 4, we further require the following results.

Lemma 13. Let $P = \mathcal { N } ( 0 , \Sigma _ { 0 } ) , Q = \mathcal { N } ( 0 , \Sigma _ { 1 } )$ be two d-dimensional centered Gaussians with positive definite

covariance matrices. Then,

$$
D _ { \mathrm { K L } } ( P \Vert Q ) = \frac { 1 } { 2 } \left[ \mathrm { T r } \left( \Sigma _ { 1 } ^ { - 1 } \Sigma _ { 0 } \right) - \log \operatorname * { d e t } \left( \Sigma _ { 1 } ^ { - 1 } \Sigma _ { 0 } \right) - d \right] ~ .\tag{C83}
$$

Proof. Evaluating the log likelihood ratio:

$$
\log \frac { p ( x ) } { q ( x ) } = \frac { 1 } { 2 } \left[ - \log \operatorname* { d e t } \Sigma _ { 0 } - x ^ { T } \Sigma _ { 0 } ^ { - 1 } x + \log \operatorname* { d e t } \Sigma _ { 1 } + x ^ { T } \Sigma _ { 1 } ^ { - 1 } x \right]\tag{C84}
$$

$$
= \frac { 1 } { 2 } \left( \log \frac { \operatorname * { d e t } \Sigma _ { 1 } } { \operatorname * { d e t } \Sigma _ { 0 } } + x ^ { T } \bigl ( \Sigma _ { 1 } ^ { - 1 } - \Sigma _ { 0 } ^ { - 1 } \bigl ) x \right) \ .\tag{C85}
$$

The KL divergence is

$$
D _ { \mathrm { K L } } ( P \Vert Q ) = \mathbb { E } _ { X \sim P } \left[ \log \frac { p ( X ) } { q ( X ) } \right] = \frac { 1 } { 2 } \left( \log \frac { \operatorname* { d e t } \Sigma _ { 1 } } { \operatorname* { d e t } \Sigma _ { 0 } } + \mathbb { E } _ { X \sim P } [ X ^ { T } \Sigma _ { 1 } ^ { - 1 } X - X ^ { T } \Sigma _ { 0 } ^ { - 1 } X ] \right) \ .\tag{C86}
$$

Using $x ^ { T } A x = \operatorname { T r } \left( A x x ^ { T } \right)$ , log $\begin{array} { r } { \frac { \operatorname* { d e t } \Sigma _ { 1 } } { \operatorname* { d e t } \Sigma _ { 0 } } = - \log \operatorname* { d e t } \left( \Sigma _ { 1 } ^ { - 1 } \Sigma _ { 0 } \right) } \end{array}$ , and $\mathbb { E } [ X X ^ { T } ] = \Sigma _ { 0 }$ for centered Gaussians gives us the desired conclusion. □

## 3. Formalizing sensing of classical signals through Quantum Signal Learning

This work considers the setting of learning from classical signals using continuous-variable quantum modes. Such quantum systems are utilized in ubiquitous sensing platforms because they naturally arise as quantized electromagnetic, mechanical, and vibrational degrees of freedom that coherently accumulate weak signals. Consequently, they underpin a wide range of quantum sensors. Optical cavity modes enable displacement and gravitational-wave detection in interferometers such as LIGO [45]. Microwave cavity modes form the basis of superconducting-circuit sensors and quantum-limited microwave measurements [74, 75] and mechanical resonators are used for force and mass sensing [22]. These sensing protocols can be realized across a variety of hardware platforms, including superconducting circuits, optical and optomechanical cavities, trapped ions and nanomechanical resonators [22, 76, 77].

In all such instances, the interaction between the probe, which is a single bosonic mode, and a classical signal is governed by a Hamiltonian of the following form:

$$
H ( t ) = f _ { x } ( t ) \hat { x } + f _ { p } ( t ) \hat { p } \ .\tag{C87}
$$

where the c-number coeficients $f _ { x } ( t ) , f _ { p } ( t )$ may be time-dependent and are usually stochastic, due to noise and uncertain priors. Importantly, Hamiltonians for classical-field sensing are linear in the canonical quadrature operators, or equivalently in raising-lowering operators. This interaction persists up to some timescale T after which point the signal has passed. Ref. [55] shows that this composite evolution is always described by a displacement channel.

Fact 6 (Classical-field sensing is displacement channel learning). Let $\rho$ be the initial state of a bosonic mode, and let it evolve under any Hamiltonian of the form in (C87) for time T. Then the final state can always be written in the form

$$
{ \mathcal E } _ { H } ^ { ( T ) } ( \rho ) = \int d ^ { 2 } \alpha ~ P _ { H } ^ { ( T ) } ( \alpha ) D ( \alpha ) \rho D ( \alpha ) ^ { \dagger }\tag{C88}
$$

where $P _ { H } ^ { ( T ) }$ is a positive, real probability density over $\mathbb { C }$ or $\mathbb { R } ^ { 2 }$ , depending on the phase-space convention.

A proof of this standard fact and its multimode generalization is shown in [55]. In this work, we largely study the single-mode $( N = 1 )$ setting (aside from Sections E 5 and F 5 which address a canonical multimode learning task), but a future direction is to extend the speedups demonstrated in this work to the multimode setting and understand whether further advantages in mode number can be discovered.

Importantly, Fact 6 tells us that classical sensing with continuous-variable quantum sensors equates to learning observables from query access to a displacement channel, and that all information about the signal accessible at interrogation time T is contained in $P _ { H } ^ { ( T ) }$ . This is even the case classically: a classical sensing protocol can be described as probing the channel $\mathcal { E } _ { H } ^ { ( T ) }$ with a coherent state, a classical state of light without any quantum nonlinearity.

To formulate the most general possible sensing task, Ref. [55] then introduces the following definition of a signal property.

Definition 14 (Bell smoothing operator). Let $\varphi _ { r }$ denote the density $\mathcal { N } _ { \mathbb { C } } ( 0 , e ^ { - 2 r } \mathbb { 1 } _ { n } )$ over $\mathbb { C } ^ { n }$ , and let p∗q denote the convolution between two densities over $\mathbb { C } ^ { n }$

$$
( p * q ) ( \xi ) = \int d ^ { 2 n } \alpha \ p ( \alpha ) q ( \xi - \alpha ) \ .\tag{C89}
$$

Then the Bell smoothing operator $T _ { r }$ is the Gaussian convolution acting on functions $f : \mathbb { C } ^ { n } \to \mathbb { C }$ as

$$
( T _ { r } f ) ( \alpha ) = ( f * \varphi _ { r } ) ( \alpha ) \ .\tag{C90}
$$

Definition 15 (Property of a classical signal). Fix an N-mode linear Hamiltonian H generated by an unknown classical signal and an evolution time $T .$ Moreover, fix a model class $\mathcal { P }$ of distributions, such that any $P _ { H } ^ { ( T ) } \in \mathcal { P }$ A property of the signal is the linear functional

$$
\Psi \Big ( P _ { H } ^ { ( T ) } \Big ) = \mathbb { E } _ { \alpha \sim P _ { H } ^ { ( T ) } } [ \psi ( \alpha ) ]\tag{C91}
$$

where we refer to the function $\psi ( \alpha )$ as the property kernel. For any valid property kernel, there must exist some $r \geq 0$ such that

1. (ψ can be recovered from the measurement outcomes) There exists a measurable function $g _ { \psi }$ such that

$$
T _ { r } g _ { \psi } = \psi\tag{C92}
$$

2. (The recovery map has bounded variance for all relevant signals) For every $P \in { \mathcal { P } }$

$$
\begin{array} { r } { \mathbb { E } _ { \xi \sim ( P * \varphi _ { r } ) } \left[ | g _ { \psi } ( \xi ) | ^ { 2 } \right] < \infty \ . } \end{array}\tag{C93}
$$

The operationally meaningful part of Definition 15 is simply that a physical observable corresponds to a linear functional of the displacement distributions. The two qualifiers are basic regularity conditions to ensure that the learning task is physically well defined. That is, one introduces the Bell smoothing operator in Definition 14 to model the measurement response of the continuous-variable Bell sampling protocol used in Ref. [55] assuming that input states have been squeezed with a squeezing strength $e ^ { - \bar { 2 } r }$ , because this measurement is informationally complete. As a result, if a particular property is to be estimable at all, there should exist some bounded, measurable function which recovers the property’s kernel under smoothing by the Bell distribution. This is only a technical detail that will not appear in any of our subsequent discussion as all physically welldefined properties will satisfy these regularity conditions.

To gain some basic intuition for this definition, we look at some special cases. If one chooses the kernel $\psi _ { \mathrm { F o u r i e r } } ^ { ( k ) } ( x + i p ) = \exp ( i k x )$ , note that the corresponding property is exactly the k-th Fourier coeficient of the signal along the x quadrature. Moments, polyspectra, and correlations all appear as kernels which are polynomials of the components of α. Alternatively, by choosing the kernel to be an appropriate Dirac delta function, one can extract quantities such as the amplitude or phase of the underlying signal. Upon imposing additional structure on H as is done in conventional AC or DC metrology, a delta-function kernel recovers the canonical metrology problem. More generally, matched filtering tasks considered in e.g. gravitational wave sensing or fundamental particle detection correspond to more complex choices of $\psi$ which may be polynomials or Lorentzian functions of α.

We can also extend the notion of a property to nonlinear functionals of a signal. This is done by allowing properties of the form

$$
\Psi ( P ) = \int \psi ( \alpha _ { 1 } , \alpha _ { 2 } , . . . , \alpha _ { m } ) \prod _ { j = 1 } ^ { m } P ( d ^ { 2 n } \alpha _ { j } ) ~ .\tag{C94}
$$

Such a definition includes properties that cannot be extracted from any experiment making only a single query to the signal, including features like the quantum channel purity of the signal. More practically, properties of time-varying signals such as multi-time correlations fall under this more general notion via joint distributions $P ( d \alpha _ { 1 } \cdot \cdot \cdot d \alpha _ { m } )$ , which we explore in this work. Now, to package this definition into a learning task, we introduce the full QSL problem.

Definition 16 (Quantum Signal Learning). Fix a linear Hamiltonian H and an evolution time T, and a property Ψ. The Quantum Signal Learning problem $\mathrm { Q S L } ( H , T , \Psi , \epsilon , \delta )$ is to estimate $\Psi ( P _ { H } ^ { ( T ) } )$ to within absolute error ϵ, with success probability at least $1 - \delta _ { i }$ given query access to $\mathcal { E } _ { H } ^ { ( T ) }$

Ref. [55] studied QSL in the post-hoc setting, where one only receives the list of desired properties after making all measurements. However, the more experimentally motivated setting considered in this work considers learning a few predetermined properties that are set before deciding the sensing procedure. This also opens up the potential for exponential quantum advantages that do not rely on entangled measurements. The main results of this work are concerned with several practically meaningful instances of QSL – namely, the properties of interest include directional and angular Fourier coeficients, temporal correlation functions, and other polynomial transformations of the displacement channel.

We will study the resource complexity of learning these fundamental observables when sensors possess practically-motivated, near-term quantum processing resources, vs. when they do not. Our separations parameterize the properties of interest by a positive number k, and show that any protocol, regardless of adaptive control, that lacks a particular quantum resource requires $S \geq \exp ( \Omega ( k ) )$ signal shots to solve QSL. Meanwhile, appending a single, modest resource enables a protocol to solve the same QSL task using $S = \mathrm { p o l y } ( k )$ shots. This hierarchy of resources is depicted in Figure 1(b) and detailed in the next section.

To obtain these quantum advantages, we introduce Quantum Phase-Space Inference (QΨ) in Section D. However, we remark that $Q \Psi$ applies more broadly beyond the results of this paper: it is a flexible framework for discovering quantum speedups across a broad range of learning tasks. To apply QΨ to a particular experimentally-grounded setting, such as sensing classical fields with single-mode sensors and additional QIP resources, one requires a mathematical model for the tasks of interest; upon defining the appropriate model, QΨ handles the rest of the analysis. QSL, with property kernels chosen to produce the basic sensing primitives above, is the appropriate QΨ model for this work.

## 4. A hierarchy of conventional and quantum-enhanced sensing

This work rigorously demonstrates that even rudimentary quantum control, coupled to conventional quantum sensors, can confer dramatic advantages in our ability to learn a wide array of signals. To formalize this claim, it will be important to clearly understand the distinction between conventional and quantum-enhanced sensing. Moreover, as quantum-enhanced sensing technology continues to develop, more sophisticated experiments wil have access to more powerful forms of quantum information processing, such as increased control depth or long-lived quantum memory. This motivates us to define the hierarchy of quantum-enhanced sensing in Figure 1(b).

To prove an exponential separation between each level in Figure 1(b), our strategy is to define classes of sensing frameworks which include each lower level as a strict subset, while adding a single quantum processing capability.

## Classical vs. conventional quantum sensing.

The first level of the sensing hierarchy contains all classical sensing protocols, which can only perform classical probe state preparation and measurement. For our lower bound, however, the following definition is convenient.

Definition 17 (Coherent-probe protocol.). A coherent-probe protocol for quantum sensing of a channel $\mathcal { E } _ { P }$ is one in which any query to $\mathcal { E } _ { F }$ can only be enacted upon a coherent state $\rho = D ( \alpha ) | 0 \rangle \langle 0 | D ( \alpha ) ^ { \dagger }$ . Arbitrary quantum information processing and measurement (i.e. any $P O V M )$ is allowed on the state $\mathcal { E } _ { P } ( \rho )$

Coherent-probe protocols simply require that the state used to interrogate the signal be classical. Beyond this requirement, all forms of adaptivity and quantum measurement are allowed. Of course, this circumscribes all purely classical sensing protocols in which not only the probe state, but the measurements, must be made classically.

When we discuss quantum sensing with conventional quantum probes, we are concerned with the class of Gaussian measurements, defined as follows.

Definition 18 (Gaussian sensing). A Gaussian sensing protocol is one in which ρ may be any Gaussian state, and any generaldyne measurement can be made on the state $\mathcal { E } _ { P } ( \rho )$ . Moreover, classical adaptivity is permitted between measurements.

Experimentally, these operations can be implemented using highly developed technologies in the optical domain such as optical parametric oscillators for generating squeezed light, beam splitters and phase shifters for Gaussian interferometry, and balanced homodyne or heterodyne detection for readout. Fast electronic processing further enables measurement outcomes to be used in real time to modify subsequent operations, making adaptive Gaussian protocols experimentally practical. Gaussian sensing therefore encompasses the realistic protocols which can be implemented in mature technologies (such as optics) but which cannot currently access the single-photon nonlinear regime [78].

To establish the first separation, we show that a simple protocol using squeezed input states and homodyne measurements requires exponentially fewer measurements than any coherent-probe protocol to estimate any angular Fourier coeficient. We demonstrate that other simple quantum-enhanced probes also sufice to instantiate this separation by proposing a simple protocol using a cat probe state. We realize this quantum advantage experimentally, depicted in Fig. 3(b); due to the limitations on performing homodyne measurements on our superconducting-circuit platform, we instead implement the protocol given in Section F 1 b.

## Conventional quantum sensing vs. constant-depth single-qubit control.

Up to this point we have discussed classes of sensing strategies which can be readily realized in e.g. conventional optical cavity platforms. Before we introduce our first quantum processing-enhanced class, it is important to understand resource accounting. Namely, the performance of conventional bosonic sensing platforms is usually constrained by the amount of non-classicality they can generate. In the Gaussian setting, this is encapsulated by the amount of squeezing that can be prepared in probe and ancilla states, since passive linear optics and displacements are classical operations that, in conjunction with squeezing, realize the full class of Gaussian protocols.

The quantity that operationally constrains conventional protocols is therefore the maximum accessible photon number ${ \bar { n } } ,$ which is equivalently the maximum energy E used by the protocol upon multiplying by Planck’s constant ℏ. To discuss a quantum enhancement, it will be important to ensure that E is held constant in all comparisons so that the advantage is truly due to a quantum information processing capability and not an energy increase.

The constraint of constant E is also apparent when considering the experimental implementation of this protocol with superconducting circuits. Here, the dominant source of decoherence of the bosonic mode is photon loss, whose efect scales with E. Therefore, the coherence timescale of the mode sets an efective maximum value of E which can be obtained. For our experimental implementation, the relevant definition of E depends on the displaced frame where this number is minimized. With photon loss rate κ, the Lindblad master equation describing the evolution of the density matrix is:

$$
\begin{array} { r } { \partial _ { t } \rho = - i [ H , \rho ] + \kappa \mathcal { D } [ a ] \rho , } \end{array}\tag{C95}
$$

where $\mathcal { D } [ L ] = L \rho L ^ { \dagger } - ( 1 / 2 ) \{ L ^ { \dagger } L , \rho \}$ is the disspator and H is the Hamiltonian. Here we have set $\hbar = 1$ . When moving into a time-dependent displaced frame $\tilde { \rho } = D ^ { \dagger } ( \alpha ) \rho D ( \alpha )$ , the the dissipator term transforms as:

$$
\kappa \mathcal { D } [ a + \alpha ] \tilde { \rho } = \kappa \mathcal { D } [ a ] \tilde { \rho } - i \left[ i \frac { \kappa } { 2 } ( \alpha ^ { * } a - \alpha a ^ { \dagger } ) , \tilde { \rho } \right] ,\tag{C96}
$$

which results in two contributions: a deterministic re-centering force at a rate $\kappa | \alpha | / 2$ (which can be corrected for), and a dissipator whose efect is strength is unafected. Therefore, changing the displaced frame (or equivalently, displacing the oscillator state) does not enhance the efective decoherence rate. Therefore, the true resource, for this description of decoherence, is the value of E in the displaced frame.

Now, keeping all sensing resources constant, our next level in the hierarchy is to add a single qubit supported in the two-dimensional linear subspace $\{ | g \rangle , | e \rangle \}$ . For concreteness, we allow two unitary operations of the following form:

$$
{ \mathrm { A n y ~ } } U ( \theta , \phi ) \in { \mathrm { S U } } ( 2 ) , \qquad { \mathrm { J C } } ( \beta _ { 1 } , \beta _ { 2 } ) = D ( \beta _ { 1 } ) \otimes | e \rangle \langle g | + D ( \beta _ { 2 } ) \otimes | g \rangle \langle e | ~ .\tag{C97}
$$

The first is simply any rotation of the qubit, which can be easily realized in most qubit platforms. The second is a generalized Jaynes-Cummings-like interaction which can be realized in constant depth by combining a symmetric Echoed Conditional Displacement (ECD) gate

$$
\mathrm { E C D } ( \beta ) = D ( \beta ) \otimes | e \rangle \langle g | + D ( - \beta ) \otimes | g \rangle \langle e |\tag{C98}
$$

with a displacement of the bosonic mode. Our results are not unique to this choice of gateset, as will become clear in the following sections; we simply choose to work with this model theoretically for consistency, as it is the

gate model implemented in our experiments (see Figure 3). At the first quantum-enhanced level of the hierarchy, we use only a single application of each of the two gates to realize an exponential advantage in learning Fourier amplitudes.

## A single memory qubit vs. quantum processing-enhanced, memoryless sensing

Another limiting feature of conventional experiments is that they perform destructive measurements on the joint system of qubit and mode. Beyond basic control capabilities, adding a single qubit also gives us the ability to retain a small quantum memory that stores a quantum state as the mode interacts with the signal multiple times. Mathematically, the distinction is as follows. In all previous levels of the hierarchy, a single signal query is made, followed by POVMs defined by the hierarchy class; the resulting classical outcome is stored and can be used to inform the next POVM. However when memory is allowed, we are permitted to perform measurements of (or trace out) only the bosonic mode, while the qubit retains its state across queries to the signal and can be measured at any time. We show that a simple protocol which only requires the qubit to remain coherent over time even while the sensor mode decoheres quickly can achieve an exponential advantage over any protocol which measures conventional resources destructively, even given arbitrarily powerful but short-lived quantum processing. We demonstrate this experimentally in Fig. 3(c).

## Beyond shallow depth

The separations at previous levels of the hierarchy establish that with rudimentary quantum information processing already accessible with modern technology, exponential advantages can be realized for learning classical signals. However, future sensing platforms are likely to have more sophisticated quantum processing capabilities, which can enable deep, coherent quantum circuits.

Ref. [67] experimentally demonstrated that trainable, coherent ECD circuits could realize displacement-sensing protocols that outperform conventional baselines by performing sophisticated QIP before readout. For concreteness, we work within our current circuit model, and consider sensing protocols which interleave control operations of the form

$$
U _ { r } = \prod _ { j = 1 } ^ { r } \mathrm { E C D } ( \beta _ { j } ) ( I \otimes U ( \theta _ { j } , \phi _ { j } ) ) ~ ,\tag{C99}
$$

with queries to a signal and measure the qubit in any basis. This constitutes a powerful class of non-Gaussian sensing protocols that include, for instance, any strategy in the Quantum Computational Displacement Sensing (QCDS) framework [67] that builds on Quantum Signal Processing (QSP) [79]. Here, we establish that protocols with access to coherent depth r can estimate properties of classical signals using ex $) ( \Omega ( \gamma r ) )$ fewer signal queries than any any protocol with depth $\leq \gamma r$ , for a constant γ less than 1.

## Appendix D: Quantum Phase-Space Inference (QΨ)

In this section, we introduce Quantum Phase-Space Inference (QΨ), a framework for analyzing and designing quantum learning experiments under explicit physical constraints. QΨ starts from three ingredients: a family of possible physical systems, a class of experiments allowed by the available architecture, and a property to be learned. It represents these ingredients on a common phase space and quantifies how strongly the allowed experiments can respond to the property of interest. This produces lower bounds on the resource complexity of any experiment in the specified architecture and identifies quantum-enhanced experiments that saturate the optimal scaling. Comparing the resulting bounds across architectures then certifies the advantage provided by additional quantum resources. This perspective allows us to derive exponential quantum advantages for basic sensing tasks using only a single qubit coupled to a conventional sensor.

We develop QΨ here beyond its specific application to sensing with a single-mode oscillator. First, we build a phase-space formalism for understanding the outcomes of general quantum experiments and define the accessible feature information (AFI), a flexible information quantity which controls downstream QΨ bounds. We then review information-theoretic learning lower bound techniques, which allow us to prove the first of three main tools in the QΨ toolkit: the semiclassical measurement lemma. This lemma turns a sensing or learning task into a phase-space description and outputs a tight query-complexity lower bound based on the AFI. We next build the upper-bound machinery, and show the QΨ saturability guarantee; this theorem proves that our QΨ lower bounds are saturable and produces a hypothesis-testing algorithm which saturates them. Finally, we extend from hypothesis testing to global learning tasks, and show that QΨ systematically translates quantum experimental design into classical statistical optimization of Fourier-domain overlaps between phase-space functions. We conclude with a template for QΨ quantum advantage design that combines these tools.

## 1. Building the language of QΨ

The basic idea of QΨ can be stated before introducing the formalism. A point x in phase space labels a possible realization of the physical system. An experiment converts that realization into a classical outcome $Y ,$ described by a conditional distribution $K _ { \mathsf { E } } ( d y | x )$ , while the property to be learned is represented by a function of the same realization x. The learning power of the experiment is therefore controlled by how much information about the target function survives the map from x to the observed outcome $Y .$ . The purpose of this subsection is to formalize these three objects: the physical realization, the experimental response, and the property of interest.

## a. Experiments, observables, and uncharacterized systems

To formalize how general quantum experiments, observables, and uncharacterized systems interact on a single phase-space, we introduce tools from probability theory and harmonic analysis. Note that these definitions let us work with generic phase-spaces, not only single-mode bosonic ones.

Definition 1 (Probability space). A probability space is a triple $( \mathcal { X } , \mathcal { F } , P _ { 0 } )$ , where X is the sample space, $\mathcal { F }$ is the σ-algebra of measurable subsets of $x ,$ and $P _ { 0 }$ is a probability measure on $( \mathcal { X } , \mathcal { F } )$

The reference measure $P _ { 0 }$ will play two roles in QΨ. It defines the inner product used to compare functions on phase space, and it provides the reference law around which we formulate local learning problems.

In QΨ, we work with phase-space equipped with a normalized probability measure, thus treating phase-space as a probability space. In the single-mode bosonic setting, X is $\mathbb { R } ^ { 2 }$ , and $P _ { 0 }$ can, for instance, be a Gaussian distribution over points in phase-space corresponding to the vacuum noise distribution. When we represent bosonic density matrices as functions on phase-space, we are generally working with representations on the Hilbert space

$$
L ^ { 2 } ( \mathbb { C } ) = \left\{ f : \mathbb { C } \to \mathbb { C } : \int | f ( \alpha ) | ^ { 2 } d ^ { 2 } \alpha < \infty \right\} \ ,\tag{D1}
$$

i.e. normalized functions relative to the Lebesgue measure. However, in QΨ it will be important to work on the Hilbert space

$$
L ^ { 2 } ( P _ { 0 } ) = \left\{ f : \mathbb { C } \to \mathbb { C } : \int | f ( \alpha ) | ^ { 2 } d P _ { 0 } < \infty \right\}\tag{D2}
$$

equipped with the inner product

$$
\langle f , g \rangle _ { P _ { 0 } } = \int \overline { { { f ( x ) } } } g ( x ) d P _ { 0 } ( x ) \ .\tag{D3}
$$

Here we identify the sample space X with $\mathbb { R } ^ { 2 }$ or $\mathbb { C }$ and the σ-algebra with the corresponding Borel σ-algebra. When working with other quantum systems such as qubits, the sample space may be a finite field like $\mathbb { F } _ { 2 } ^ { 2 n }$ , or even a nonabelian group like SU(2). Next we define a Fourier character family, which generalizes the notion of a complete basis of functions supporting a Fourier transform on an arbitrary sample space.

Definition 2 (Fourier character family). A Fourier character family on a sample space $\mathcal { X }$ is a family

$$
{ \mathcal { D } } = \{ \chi _ { \omega } : \mathcal { X }  \mathbb { C } \} _ { \omega \in \Omega }\tag{D4}
$$

where Ω is an abelian label group such that $\chi _ { 0 } ( x ) = 1 , | \chi _ { \omega } ( x ) | = 1 , \chi _ { \omega + \nu } ( x ) = \chi _ { \omega } ( x ) \chi _ { \nu } ( x )$ , and $\chi _ { - \omega } ( x ) = \overline { { \chi _ { \omega } ( x ) } }$ Given a measure $P _ { 0 }$ on X, the mean of a character is

$$
\mu _ { \omega } = \int \chi _ { \omega } ( x ) d P _ { 0 } ( x )\tag{D5}
$$

and the centered character is $\widetilde \chi _ { \omega } ( x ) = \chi _ { \omega } ( x ) - \mu _ { \omega }$ . The centered character family $\widetilde { \mathcal { D } } _ { P _ { 0 } }$ is the set of all centered characters on X with respect to measure $P _ { 0 }$

For the single-mode bosonic phase-space, a canonical Fourier character family is the Weyl family $\chi _ { \zeta } ( \alpha ) =$ $e ^ { i \Omega _ { \mathrm { s p } } ( \zeta , \alpha ) }$ , where $\Omega _ { \mathrm { s p } }$ is the symplectic form. For qubits, one often uses $\chi _ { s } ( z ) = ( - 1 ) ^ { s \cdot z }$ . Operationally, the character family gives $\mathrm { Q } \Psi$ a way to decompose diferent quantum operators, such as states, channels, or measurements, which live on the same phase-space into a common basis. This makes it much easier to bound overlaps.

Definition 3 (Gram matrix of centered character family). Given a probability space X equipped with measure $P _ { 0 }$ and a centered character dictionary

$$
\widetilde { D } _ { P _ { 0 } } = \{ \widetilde { \chi } _ { \omega } : \omega \in \Omega , \widetilde { \chi } _ { \omega } \neq 0 \} \ ,\tag{D6}
$$

the Gram matrix of the family is

$$
G ( \omega , \nu ) = \langle \widetilde { \chi } _ { \omega } , \widetilde { \chi } _ { \nu } \rangle _ { P _ { 0 } } ~ .\tag{D7}
$$

The Gram matrix controls overlaps between Fourier basis elements. For an orthonormal centered character family, its of-diagonal entries vanish and its diagonal entries are one.

Lemma 4 (Orthogonality of character family from compactness). Let G be a compact abelian group with normalized Haar measure $P _ { 0 } ,$ , and let $\{ \chi _ { \omega } \} _ { \omega \in \widehat { G } }$ be its character group. Then $\mu _ { \omega } ~ = ~ 0$ for every nontrivial character, and

$$
\langle \chi _ { \omega } , \chi _ { \nu } \rangle _ { P _ { 0 } } = \mathbb { 1 } [ \omega = \nu ] ~ .\tag{D8}
$$

Proof. If ω $\neq 0$ , choose $a \in G$ such that $\chi _ { \omega } ( a ) \neq 1$ . Using Haar’s theorem since we assume compactness, we have

$$
\int \chi _ { \omega } ( x ) d P _ { 0 } ( x ) = \int \chi _ { \omega } ( x + a ) d P _ { 0 } ( x ) = \chi _ { \omega } ( a ) \int \chi _ { \omega } ( x ) d P _ { 0 } ( x ) \ ,\tag{D9}
$$

and hence $\mu _ { \omega } = 0$ . Applying this to $\chi _ { \nu - \omega }$ gives the stated orthogonality relation.

The lemma provides a useful intuition: if a physical process admits a representation on a compact abelian group in phase space, it also admits an orthonormal Fourier decomposition. This is convenient because quantum measurements and the uncharacterized system of interest can then be related in the same orthonormal basis. Unfortunately, the natural Weyl basis for bosonic phase space is not orthonormal on $L ^ { 2 } ( \mathbb { R } ^ { 2 n } )$ . However, it satisfies an approximate orthonormality condition relative to a Gaussian measure, allowing us to retain many of the same advantages by equipping phase space with a Gaussian reference measure equipped with a Gaussian reference.

Lemma 5 (Approximate orthonormality of Weyl basis relative to Gaussian reference). Let $P _ { 0 } = \mathcal { N } ( 0 , \Sigma )$ on $\mathbb { R } ^ { 2 n }$ , and let $\chi _ { \zeta } ( x ) = e ^ { i \zeta ^ { T } \Omega x }$ , where Ω is the symplectic form. Then

$$
\mu _ { \zeta } = \exp \left( - \frac { 1 } { 2 } \zeta ^ { T } \Omega \Sigma \Omega ^ { T } \zeta \right)\tag{D10}
$$

and

$$
G ( \zeta , \nu ) = \exp \left( - \frac { 1 } { 2 } ( \nu - \zeta ) ^ { T } \Omega \Sigma \Omega ^ { T } ( \nu - \zeta ) \right) - \overline { { \mu _ { \zeta } } } \mu _ { \nu } \ .\tag{D11}
$$

Proof. The key point is that the Gram matrix of bosonic Weyl characters relative to a distribution $P _ { 0 }$ is exactly the Fourier transform of $P _ { 0 }$ . Indeed,

$$
\langle \chi _ { \zeta } , \chi _ { \nu } \rangle _ { P _ { 0 } } = \int e ^ { i ( \nu - \zeta ) ^ { T } \Omega x } d P _ { 0 } ( x ) ~ .\tag{D12}
$$

For a centered Gaussian, this characteristic function is the first term in the claimed expression. Centering the characters subtracts ${ \overline { { \mu _ { \zeta } } } } \mu _ { \nu }$ □

The above simple fact will allow us to argue that when measurements and channels are both decomposed in the Weyl basis, the amount of information gained from a measurement is directly quantified by a Fourier overlap.

We now connect the abstract phase space to the physical system being learned. The useful picture is that $x \in \mathcal { X }$ is a hidden classical label specifying one possible realization of the system, while $\mathcal { N } _ { x }$ describes the quantum operation associated with that realization. The family $x \mapsto \mathcal { N } _ { x }$ is known; what is unknown is the probability law $P$ governing which realizations occur. For example, in displacement sensing, $x = \alpha$ is a displacement amplitude and $\mathcal { N } _ { \alpha }$ is the corresponding deterministic displacement channel. This motivates the following definition.

Definition 6 (Phase-space oracle model). Let (X, F) be a measurable space. A phase-space oracle model is a measurable family of quantum channels

$$
x \longmapsto { \mathcal { N } } _ { x }\tag{D13}
$$

from an input system A to an output system B.

Through this definition, our choice of phase space parameterizes specific realizations of our system of interest. In an experiment, we have oracle access to a particular probability law supported the phase-space, which specifies the uncharacterized system itself.

Definition 7 (QΨ quantum oracle). Given a phase-space oracle model $x \mapsto \mathcal { N } _ { x }$ and a probability law P on (X, F), the corresponding quantum oracle is

$$
\mathcal { N } _ { P } ( \rho ) = \int _ { \mathcal { X } } \mathcal { N } _ { x } ( \rho ) d P ( x ) ~ .\tag{D14}
$$

Query access to the uncharacterized physical system means query access to $\mathcal { N } _ { P }$ . A quantum state is included as the special case in which A is one-dimensional and $\mathcal { N } _ { x } ( 1 ) = \rho _ { x }$

All “quantumness” is therefore baked into the definition of the phase space itself. This representation is useful because the phase-space X can be chosen to match the natural degrees of freedom of the physical problem. For single-mode displacement sensing, we take $\mathcal { X } = \mathbb { R } ^ { 2 } \simeq \mathbb { C }$ and associate $\alpha \in \mathbb { C }$ with

$$
\mathcal { N } _ { \alpha } ( \rho ) = D ( \alpha ) \rho D ( \alpha ) ^ { \dagger } \mathrm { ~ , ~ }\tag{D15}
$$

so that each displacement coordinate maps to its corresponding deterministic displacement channel. An unknown displacement distribution P then produces the usual random displacement channel. A deterministic displacement is contained in the same model by taking $P = \delta _ { \alpha }$

For intuition we can also consider an example with qubit systems, for which one can instead use a discrete phase-space. For example, $\mathcal { X } = \mathbb { F } _ { 2 } ^ { 2 n }$ naturally labels the n-qubit Pauli operators $P _ { x }$ , and the map

$$
\mathcal { N } _ { x } ( \rho ) = P _ { x } \rho P _ { x } ^ { \dagger }\tag{D16}
$$

gives the usual representation of a Pauli channel as a probability law over discrete phase-space. More general state-learning problems are obtained by treating states as channels with trivial input: one chooses X to parameterize the relevant family of states and sets $\mathcal { N } _ { x } ( 1 ) = \rho _ { x }$ . For instance, in a Pauli-tomography lower bound, x may label the Pauli-structured family of candidate states being distinguished, while the Pauli characters provide the natural Fourier dictionary on that family [15, 18].

We next formalize the operations available to the experimentalist. A single use of a channel difers from a single copy of a state because the experimentalist may choose which probe is supplied to the channel. More generally, if coherent quantum memory is available, the output of one query may be retained and processed coherently before a later query. Both cases are described by the same object.

Definition 8 (QΨ quantum experiment). An r-query quantum experiment E with query access to an oracle $\mathcal { N } _ { P }$ is a tuple $( M , \{ C _ { j } \} _ { j = 1 } ^ { r - 1 } , \sigma _ { R _ { 0 } A } )$ consisting of an initial state $\sigma _ { R _ { 0 } A _ { 1 } }$ , quantum channels

$$
{ \mathcal { C } } _ { j } : R _ { j - 1 } B _ { j } \longrightarrow R _ { j } A _ { j + 1 } , \qquad j = 1 , \dotsc , r - 1 ,\tag{D17}
$$

and a final POVM $M ( d y )$ on $R _ { r - 1 } B _ { r }$ . The systems $A _ { j }$ and $B _ { j }$ are respectively the input and output of the j-th oracle query, while $R _ { j }$ contains any quantum memory retained by the experiment. The experiment proceeds by alternately applying $\mathcal { N } _ { P }$ to $A _ { j }$ and the processing channel $\mathcal { C } _ { j }$ , followed after the r-th query by the POVM M.

This definition includes classical feedforward without additional notation, since intermediate measurement outcomes may be stored in classical registers contained in $R _ { j }$ . It also includes arbitrary reference systems and entangled probes.

In many near-term physical experiments, including the first three levels of the sensing hierarchy, coherent quantum memory interleaving quantum control is not available. These cases are $r = 1$ quantum experiments, which are fully specified by the input state $\sigma _ { R A }$ and a POVM M. If the oracle is a quantum state, $\sigma _ { R A }$ is no longer a degree of freedom, so a memoryless experiment is entirely specified by a choice of POVM.

A physical experiment therefore specifies how each realization of the uncharacterized system is converted into classical data. This conversion is the central object in the $\mathrm { Q } \Psi$ representation.

Definition 9 (Experimental response kernel). Let E be an r-query quantum experiment. For fixed realizations $x _ { 1 } , \ldots , x _ { r } \in \mathcal { X }$ , let $\rho _ { \mathsf { E } } ( x _ { 1 } , \ldots , x _ { r } )$ denote the final quantum state obtained by inserting $\mathcal { N } _ { x _ { j } }$ into the $j - t h$ oracle slot of E. The experimental response kernel is

$$
K _ { \mathsf { E } } ^ { ( r ) } ( d y | x _ { 1 } , \ldots , x _ { r } ) = \mathrm { T r } \left[ M ( d y ) \rho _ { \mathsf { E } } ( x _ { 1 } , \ldots , x _ { r } ) \right] \ .\tag{D18}
$$

For $r = 1$ , if E consists of the probe $\sigma _ { R A }$ and POVM $M ( d y )$ , this becomes

$$
K _ { \mathsf { E } } ( d y | x ) = \mathrm { T r } \left[ M ( d y ) ( \mathrm { i d } _ { R } \otimes { \mathsf { N } } _ { x } ) ( \sigma _ { R A } ) \right] ~ .\tag{D19}
$$

For $r = 1$ , the experimental response kernel has a simple interpretation: $K _ { \mathsf { E } } ( d y | x )$ is the conditional distribution of the classical outcome $Y$ given the physical realization $X = x$ . Thus, if the unknown realization is drawn according to $X \sim P$ , the experimentally observed distribution is simply the marginal law

$$
Q _ { \mathsf { E } , P } ( d y ) = \int _ { \mathcal { X } } K _ { \mathsf { E } } ( d y | x ) d P ( x ) \ .\tag{D20}
$$

For general r, the same interpretation applies to the tuple of realizations $( x _ { 1 } , \ldots , x _ { r } )$ . The response kernel is therefore the object that converts the underlying physical realization into experimentally accessible classical data.

This is the basic $\mathrm { Q } \Psi$ representation of an experiment. The physical system is encoded by its law $P ,$ the quantum architecture determines which kernels $K _ { \mathsf { E } }$ can be realized, and all information available to the experi mentalist is contained in the resulting classical outcome law $Q _ { \mathsf { E } , P }$

The final ingredient is the quantity the experimenter wishes to learn. At the broadest level, this is a functional of the unknown physical law. The properties most useful for $\mathrm { Q } \Psi$ admit a particularly simple representation on the same realization space.

Definition 10 (Property and property kernel). A property of an uncharacterized physical system is a functional Ψ of its law P. We say that Ψ admits an m-replica property kernel if there is a measurable function $\psi : \mathcal { X } ^ { m }  \mathbb { C }$ such that

$$
\Psi _ { \psi } ( P ) = \int _ { \chi _ { m } } \psi ( x _ { 1 } , \ldots , x _ { m } ) d P ( x _ { 1 } ) \cdot \cdot \cdot d P ( x _ { m } ) \ .\tag{D21}
$$

For $m = 1$ , we simply write

$$
\Psi _ { \psi } ( P ) = \mathbb { E } _ { X \sim P } [ \psi ( X ) ]\tag{D22}
$$

and call ψ the property kernel.

The important point is that the experiment and the learning objective are now represented on the same realization space: $K _ { \mathsf { E } } ( d y | x )$ describes how the experiment responds to $x ,$ while $\psi ( x )$ describes how strongly the property of interest depends on $x .$ This common representation is what will allow QΨ to compare experiments and properties directly.

The above definition generalizes quantum signal learning (QSL), as introduced in Section C 3, to arbitrary kinds of physical experiments through an appropriate redefinition of the phase space.

The case $m = 1$ is the most physically common setting of properties that representable as linear functionals of the quantum system; we consider some examples here for intuition. For state learning, an observable O induces

$$
\psi _ { O } ( x ) = \mathrm { T r } ( O \rho _ { x } ) ,\tag{D23}
$$

so that

$$
\Psi _ { O } ( P ) = \mathrm { T r } ( O \rho _ { P } ) , \qquad \rho _ { P } = \int \rho _ { x } d P ( x ) ~ .\tag{D24}
$$

Pauli shadow tomography is obtained by taking O from a collection of Pauli operators. For channel learning, a linear functional of the channel similarly induces a property kernel $x \mapsto \psi ( x )$ ; in finite dimensions one may equivalently view these as linear functionals of the Choi operator. Full state or channel tomography corresponds to learning a suficiently rich family of such linear properties that separates the possible physical objects.

Properties nonlinear in the averaged state or channel naturally require m $> 1$ . For example, if $\begin{array} { r } { \rho _ { P } = \int { \rho _ { x } d P ( x ) } } \end{array}$ and ${ \cal O } ^ { ( m ) }$ is an observable on m replicas, then

$$
\mathrm { T r } \Big [ { \cal O } ^ { ( m ) } \rho _ { P } ^ { \otimes m } \Big ] = \int \mathrm { T r } \Big [ { \cal O } ^ { ( m ) } \rho _ { x _ { 1 } } \otimes \cdot \cdot \cdot \otimes \rho _ { x _ { m } } \Big ] d P ( x _ { 1 } ) \cdot \cdot \cdot d P ( x _ { m } ) ,\tag{D25}
$$

which is precisely an m-replica property kernel. Purity, moments of a state, and other nonlinear quantum observables fit naturally into this form.

For the semiclassical learning problems studied below, we will primarily take $r = m = 1$ . An allowed quantum experiment E then produces a Markov kernel $K _ { \mathsf { E } } ( d y | x )$ on the realization space, while the learning objective is specified by a property or perturbation $h ( x )$ on the same space. QΨ quantifies how much of this target feature survives the map from the physical realization X to the classical measurement outcome Y. The information quantity which makes this comparison precise is the accessible feature information.

Remark 11. To summarize, QΨ identifies all ingredients of a quantum experimental task with objects on the same phase space χ, whose definition is tailored to the task at hand. The uncharacterized system is an oracle $\textstyle { \mathcal { N } } _ { P } = \int $ dx $P ( x ) \mathcal { N } _ { x }$ , where $\mathcal { N } _ { x }$ is a known quantum state or channel for each point $x \in \chi ,$ while $P$ is unknown. An experiment is specified by an initial state, a POVM, and given coherent quantum memory, additional control operations. Then, the outcomes of an experiment E are cleanly described by a classical Markov kernel $K _ { \mathsf { E } }$ , which we call the experimental response kernel. Experimental outcomes are obtained by integrating the Markov kernel against P, while physical properties of the oracle are obtained by integrating the observable’s property kernel against $P .$

We now have an experiment kernel and a property kernel. The intuitive, but crucial step is to realize that their overlap, weighted by the density of $P ,$ quantifies how well the experiment learns the property. The appropriately-defined overlap quantity is the accessible feature information.

## b. Defining the Accessible Feature Information $( A F I )$

We now have a way to relate experiments, uncharacterized physical processes, and physical features through kernels in the same phase-space picture. The following object will be the quantity central to our lower bounds.

For a one-query experiment E, let $Q _ { \mathsf { E } , 0 } = Q _ { \mathsf { E } , P _ { 0 } }$ denote its outcome distribution under the reference law:

$$
Q _ { \mathsf { E } , 0 } ( d y ) = \int K _ { \mathsf { E } } ( d y | x ) d P _ { 0 } ( x ) \ .\tag{D26}
$$

Moreover, for a property kernel h, we define

$$
A _ { \mathsf { E } , h } ( d y ) = \int h ( x ) K _ { \mathsf { E } } ( d y | x ) d P _ { 0 } ( x ) \ .\tag{D27}
$$

Because $K _ { \mathsf { E } }$ is a Markov kernel, $A _ { \mathsf { E } , h }$ is absolutely continuous with respect to $Q _ { \mathsf { E } , 0 }$ , so $d A _ { \mathsf { E } , h } / d Q _ { \mathsf { E } , 0 }$ is well defined on the support of $Q _ { \mathsf { E } , 0 }$

Definition 12 (Accessible Feature Information). For any function $h \in L _ { 0 } ^ { 2 } ( P _ { 0 } )$ , define the accessible feature information of an experiment E about h by

$$
\mathsf { A } _ { \mathsf { E } } ( h ; P _ { 0 } ) = \int \left| \frac { d A _ { \mathsf { E } , \bar { h } } } { d Q _ { \mathsf { E } , 0 } } ( y ) \right| ^ { 2 } d Q _ { \mathsf { E } , 0 } ( y ) \ ,\tag{D28}
$$

where $\bar { h } = h - \int$ h dP<sub>0</sub>. For an entire class of experiments M, define

$$
\mathsf { A } _ { \mathcal { M } } ( h ; P _ { 0 } ) = \operatorname* { s u p } _ { \mathsf { E } \in \mathcal { M } } \mathsf { A } _ { \mathsf { E } } ( h ; P _ { 0 } ) \ .\tag{D29}
$$

If E has finite or countable outcomes, write $f _ { \mathsf { E } , y } ( x ) = K _ { \mathsf { E } } ( y | x )$ and $\begin{array} { r } { q _ { \mathsf { E } , 0 } ( y ) = \int f _ { \mathsf { E } , y } ( x ) d P _ { 0 } ( x ) } \end{array}$ . Then

$$
\mathsf { A } _ { \mathsf { E } } ( h ; P _ { 0 } ) = \sum _ { y : q _ { \mathsf { E } , 0 } ( y ) > 0 } \frac { \left| \int f _ { \mathsf { E } , y } ( x ) h ( x ) d P _ { 0 } ( x ) \right| ^ { 2 } } { q _ { \mathsf { E } , 0 } ( y ) } \ .\tag{D30}
$$

The definition has a direct statistical interpretation. Draw $X \sim P _ { 0 }$ and then run the experiment conditioned on X, producing an outcome Y. For centered $h ,$ the Radon–Nikodym derivative appearing above is

$$
{ \frac { d A _ { \mathsf { E } , h } } { d Q _ { \mathsf { E } , 0 } } } ( Y ) = \mathbb { E } [ h ( X ) \mid Y ] ,\tag{D31}
$$

and therefore

$$
\mathsf { A } _ { \mathsf { E } } ( h ; P _ { 0 } ) = \mathbb { E } \left[ \left| \mathbb { E } [ h ( X ) \mid Y ] \right| ^ { 2 } \right] .\tag{D32}
$$

Since $\mathbb { E } [ h ( X ) \mid Y ]$ is the optimal mean-square predictor of $h ( X )$ from the measurement outcome, the AFI quantifies how much of the feature $h ( X )$ remains accessible after the physical realization X has been converted into the classical outcome $Y$ . In particular, $\mathsf { A } _ { \mathsf { E } } ( h ; P _ { 0 } ) = 0$ when the outcome contains no information about $h ( X )$ , while the contraction property of conditional expectation gives

$$
\mathsf { A } _ { \mathsf { E } } ( h ; P _ { 0 } ) \leq \| h \| _ { L ^ { 2 } ( P _ { 0 } ) } ^ { 2 } .\tag{D33}
$$

In the next subsection, we show that this statistical notion of accessibility directly controls the query complexity of local learning problems. Tight bounds on the AFI therefore translate directly into tight bounds on the required resources. Fourier analysis provides a useful tool for obtaining such bounds, since the overlap between response functions $f _ { E , y }$ and a property kernel h is often easiest to control in a common Fourier basis.

To this end, we introduce a concept relevant to Fourier analysis of the AFI, which we use later in our controldepth separation, Theorem E.13.

Definition 13 (Valid dictionary and sparse response class). Let M be a class of finite or countable-outcome experiments associated with probability space $( \mathcal { X } , \mathcal { F } , P _ { 0 } )$ , and let D be a Fourier character family. Then D is a valid dictionary for M if for every experiment $M \in \mathcal { M }$ and every outcome y with $q _ { M , 0 } ( y ) > 0$ , the centered response function

$$
f _ { M , y } - q _ { M , 0 } ( y )\tag{D34}
$$

lies in the closed linear span of the centered characters $\{ \widetilde { \chi } _ { \omega } \} _ { \omega \in \Omega }$ . We say that M is L-sparse in the Fourier dictionary D under $P _ { 0 }$ if, for every $M \in \mathcal { M }$ , there exists $S _ { M } \subset \Omega$ with $| S _ { M } | \le L$ such that every centered response function satisfies

$$
f _ { M , y } - q _ { M , 0 } ( y ) \in \mathrm { s p a n } \{ \widetilde { \chi } _ { \omega } : \omega \in S _ { M } \} .\tag{D35}
$$

Definition 14 (Captured mass). For a centered function $h \in L _ { 0 } ^ { 2 } ( P _ { 0 } )$ , define

$$
\mathsf C _ { L } ( h ; \mathscr D , P _ { 0 } ) = \operatorname* { s u p } _ { S \subset \Omega \atop | S | \leq L } \left\| \Pi _ { S } h \right\| _ { L ^ { 2 } ( P _ { 0 } ) } ^ { 2 } ,\tag{D36}
$$

where $\Pi _ { S }$ is the orthogonal projector onto span $\{ \widetilde { \chi } _ { \omega } : \omega \in S \}$

These definitions have a simple geometric interpretation. An L-sparse experimental class can access at most L Fourier directions at a time. The captured mass $\mathsf C _ { L } ( h ; \mathcal D , P _ { 0 } )$ measures the largest fraction of the $L ^ { 2 } ( P _ { 0 } )$ weight of the target h that can lie in any such L-dimensional Fourier subspace. Since only the component of h lying in the response subspace of a measurement can contribute to the AFI, $\mathsf { C } _ { L }$ provides a direct upper bound on the information accessible to an L-sparse architecture.

For an exactly orthonormal dictionary, this becomes especially transparent. If

$$
h = \sum _ { \omega \in S _ { h } } c _ { \omega } \chi _ { \omega } ,\tag{D37}
$$

then $\mathsf { C } _ { L }$ is simply the squared Fourier weight contained in the best choice of at most L frequencies:

$$
\mathsf C _ { L } ( \boldsymbol h ; \mathcal D , { P } _ { 0 } ) = \operatorname* { s u p } _ { S \subseteq S _ { h } \atop | S | \leq L } \sum _ { \boldsymbol \omega \in S } | c _ { \boldsymbol \omega } | ^ { 2 } .\tag{D38}
$$

Thus, if the target is spread approximately uniformly over $H \gg L$ Fourier components, an L-sparse measurement can access only an $O ( L / H )$ fraction of its total Fourier weight. In particular, $\mathsf { C } _ { L }$ is bounded by L times the squared magnitude of the largest Fourier coeficient of h.

These intuitions form the core of QΨ: the expressivity of a constrained experimental architecture appears via restrictions on the response kernels that it can realize, and these restrictions often appear naturally in an appropriate Fourier domain.

## c. Information-theoretic lower bound tools

We now understand how to represent quantum-mechanical experiments in the phase-space picture, and how the power of an experiment is controlled by its AFI. To connect these definitions to resource complexity and prove tight lower bounds, we draw on techniques from statistical learning theory.

To establish a rigorous query-complexity lower bound, we must be able to account for any arbitrary, possibly adaptive, sequence of experiments chosen from a family. The first key step in obtaining such bounds is a reduction of the full learning task to a hypothesis test, using the following observation. We state this observation for sensing for maximum operational clarity, but it trivially applies to any property-estimation task.

Fact 7 (Learning reduces to hypothesis testing). Consider two families of signals with distributions $P _ { 0 } ^ { ( k ) } , P _ { 1 } ^ { ( k ) }$ , parameterized by integer $k > 0$ , and a property of interest $\psi _ { k } ( \alpha )$ , such that for all $k _ { i }$

$$
\begin{array} { r } { \left| \mathbb { E } _ { \alpha \sim P _ { 0 } ^ { ( k ) } } \left[ \psi _ { k } ( \alpha ) \right] - \mathbb { E } _ { \alpha \sim P _ { 1 } ^ { ( k ) } } \left[ \psi _ { k } ( \alpha ) \right] \right| \ge 2 \epsilon , } \end{array}\tag{D39}
$$

where ϵ is a positive constant independent of k. Suppose there is a learning strategy L which, given access to any valid signal $P ,$ can estimate $\mathbb { E } _ { \alpha \sim P } [ \psi _ { k } ( \alpha ] )$ to accuracy ϵ with success probability at least $2 / 3 ,$ and that L uses at most $f ( k ) > 0$ samples to do so. Then, it is possible to determine with success probability at least $2 / 3$ which signal was provided with $\leq f ( k )$ queries.

This follows because a hypothesis tester can simply apply L to the provided signal, and due to the guarantee that the two distributions difer in the property $\psi _ { k }$ by at least 2ϵ, conclude based on the estimated value which signal was provided. By contrapositive, it follows that if one proves that any learning protocol in a family of experiments requires $\geq \Omega ( f ( k ) )$ queries to solve such a hypothesis testing problem, then any such protocol also requires $\geq \Omega ( f ( k ) )$ queries to learn the property $\psi _ { k }$ to constant precision. When $f ( k ) = \exp ( \Omega ( k ) )$ , this implies that exponentially many samples are required for learning.

Having established that our separations will rely on lower-bounding the sample complexity of a hypothesistesting task, we now record the mathematical tools we use to do so. First, we formally understand the most general description of a quantum experiment. We again provide the statement in the single-mode sensing context, but the extension to other experimental settings is immediate.

Definition 15 (Classical transcript for an adaptive hypothesis testing protocol). Consider any quantum sensing protocol for hypothesis testing that makes N total queries to a single-mode signal, which is promised to either be $\mathcal { E } _ { P _ { 0 } } ~ o r ~ \mathcal { E } _ { P _ { 1 } }$ . In the first round, the learner can prepare a probe state (possibly joint across modes and qubits), expose one mode to the signal, and perform a $P O V M .$ The outcome $q _ { 1 }$ is sampled from a distribution $Q _ { 1 } ^ { ( a ) }$ over all possible outcomes from this $P O V M ,$ where $a = 0 ~ o r ~ 1$ based on whether the true signal distribution is $P _ { 0 }$ or $P _ { 1 }$ respectively. In each round $j = 2 , . . . , N$ thereafter, the learner prepares a state and chooses $a \ P O V M$ given access to all outcomes $q _ { 1 } , . . . , q _ { j - 1 }$ . Finally, based on the classical transcript $q _ { 1 } , q _ { 2 } , . . . , q _ { N } $ , the learner outputs either $P _ { 0 }$ or $P _ { 1 }$ . We say that the entire transcript is sampled from the distribution

$$
Q ^ { ( a ) } ( \vec { q } ) : = Q ^ { ( a ) } ( q _ { 1 } , q _ { 2 } , . . . , q _ { N } ) = Q _ { 1 } ^ { ( a ) } ( q _ { 1 } ) Q _ { 2 } ^ { ( a ) } ( q _ { 2 } | q _ { 1 } ) Q _ { 3 } ^ { ( a ) } ( q _ { 3 } | q _ { 2 } , q _ { 1 } ) \cdot \cdot \cdot Q _ { N } ^ { ( a ) } ( q _ { N } | q _ { N - 1 } , . . . , q _ { 1 } )\tag{D40}
$$

Particular classes of sensing protocols in the hierarchy refine this definition by constraining the probe states and allowed measurements. In other experimental settings, the hypothesis test may provide query access to a quantum state, multimode or multiqubit channel, Hamiltonian evolution, etc. The allowable family of learning protocols can be tuned accordingly. The main workhorse in obtaining query-complexity lower bounds is then Le Cam’s two-point method. Before we state the theorem, we require definitions of some standard divergences between probability distributions.

Definition 16 (Total variation distance). Let $Q ^ { ( 0 ) }$ and $Q ^ { ( 1 ) }$ be two probability laws on a common measurable space (Ω, F). Their total variation distance is

$$
d _ { \mathrm { T V } } \left( Q ^ { ( 0 ) } , Q ^ { ( 1 ) } \right) : = \operatorname* { s u p } _ { A \in \mathcal { F } } \left| Q ^ { ( 0 ) } ( A ) - Q ^ { ( 1 ) } ( A ) \right| .\tag{D41}
$$

If the transcript of an N-round sensing protocol is $\vec { q } = ( q _ { 1 } , \dots , q _ { N } ) \in \mathbb { C } ^ { N }$ , as is the case for generaldyne measurements, then

$$
d _ { \mathrm { T V } } \Big ( Q ^ { ( 0 ) } , Q ^ { ( 1 ) } \Big ) = \frac { 1 } { 2 } \int _ { \mathbb { C } ^ { N } } \Big | Q ^ { ( 0 ) } ( \vec { q } ) - Q ^ { ( 1 ) } ( \vec { q } ) \Big | d ^ { 2 N } \vec { q } .\tag{D42}
$$

For homodyne or any other single-quadrature readout, one simply replaces $\mathbb { C } ^ { N }$ with $\mathbb { R } ^ { N }$

The TV distance has a useful operational meaning in the context of quantum state hypothesis testing.

Lemma 17 (Total Variation trace distance bound). Consider two hypothesis states $\rho _ { 0 } , \rho _ { 1 }$ , and any arbitrary $P O V M \left\{ F _ { s } \right\}$ which induces measurement transcripts $q _ { 0 } ( s ) = \mathrm { T r } ( F _ { s } \rho _ { 0 } )$ and $q _ { 1 } = \operatorname { T r } ( F _ { s } \rho _ { 1 } )$ . Then

$$
d _ { \mathrm { T V } } \big ( q _ { 0 } , q _ { 1 } \big ) \leq \frac { 1 } { 2 } \| \rho _ { 0 } - \rho _ { 1 } \| ~ .\tag{D43}
$$

Closely related is the Kullback-Leibler (KL) divergence, which we define only on the relevant domains:

Definition 18 (Kullback–Leibler divergence). Let $Q ^ { ( 0 ) }$ and $Q ^ { ( 1 ) }$ be two probability laws on $\mathbb { C } ^ { N }$ . If the transcript of an N-round sensing protocol is $\vec { q } = ( q _ { 1 } , \dots , q _ { N } ) \in \mathbb { C } ^ { N }$ then

$$
D _ { \mathrm { K L } } \Big ( Q ^ { ( 0 ) } \Big \| Q ^ { ( 1 ) } \Big ) = \int _ { \mathbb { C } ^ { N } } Q ^ { ( 0 ) } ( \vec { q } ) \log \Big ( \frac { Q ^ { ( 0 ) } ( \vec { q } ) } { Q ^ { ( 1 ) } ( \vec { q } ) } \Big ) \ d ^ { 2 N } \vec { q } \ .\tag{D44}
$$

For homodyne or any other single-quadrature readout, one simply replaces $\mathbb { C } ^ { N }$ with $\mathbb { R } ^ { N }$

These metrics are related by the following well-known inequality:

Lemma 19 (Pinsker’s inequality). Let $Q ^ { ( 0 ) }$ and $Q ^ { ( 1 ) }$ be two probability laws on a common measurable space. Then

$$
d _ { \mathrm { T V } } \Big ( Q ^ { ( 0 ) } , Q ^ { ( 1 ) } \Big ) \leq \sqrt { \frac { 1 } { 2 } D _ { \mathrm { K L } } \big ( Q ^ { ( 0 ) } \big \| Q ^ { ( 1 ) } \big ) } ~ .\tag{D45}
$$

Recall that our lower bounds must account for arbitrary classical adaptivity based on the collected transcript. Total variation has an important property that we will leverage to decouple the transcript distributions, so that we only need to consider bounding the TVD from a single experiment.

Lemma 20 (Adaptive TV hybrid bound). Let $Q ^ { ( 0 ) }$ and $Q ^ { ( 1 ) }$ be transcript distributions on $\mathbb { C } ^ { N }$ . If for every j and every previous transcript $\scriptstyle { { \vec { q } } < j }$

$$
d _ { \mathrm { T V } } \Big ( Q _ { j } ^ { ( 0 ) } ( \cdot \mid \vec { q } _ { < j } ) , Q _ { j } ^ { ( 1 ) } ( \cdot \mid \vec { q } _ { < j } ) \Big ) \leq \epsilon ~ ,\tag{D46}
$$

then

$$
{ d _ { \mathrm { T V } } \left( Q ^ { ( 0 ) } , Q ^ { ( 1 ) } \right) \leq N \epsilon } \ .\tag{D47}
$$

Proof. For each $j \in \{ 0 , \ldots , N \}$ , define the hybrid transcript distribution

$$
R _ { j } ( \vec { q } ) = \left( \prod _ { \ell = 1 } ^ { j } Q _ { \ell } ^ { ( 0 ) } ( q _ { \ell } \mid \vec { q } _ { < \ell } ) \right) \left( \prod _ { \ell = j + 1 } ^ { N } Q _ { \ell } ^ { ( 1 ) } ( q _ { \ell } \mid \vec { q } _ { < \ell } ) \right) .\tag{D48}
$$

Thus $R _ { 0 } = Q ^ { ( 1 ) }$ and $R _ { N } = Q ^ { ( 0 ) }$ . By the triangle inequality,

$$
d _ { \mathrm { T V } } \left( Q ^ { ( 0 ) } , Q ^ { ( 1 ) } \right) \leq \sum _ { j = 1 } ^ { N } d _ { \mathrm { T V } } \left( R _ { j } , R _ { j - 1 } \right) \ .\tag{D49}
$$

It remains to bound a single hybrid step. The distributions $R _ { j }$ and $R _ { j - 1 }$ agree on the first $j - 1$ rounds and difer only in the conditional law used for the j-th outcome. Therefore,

$$
d _ { \mathrm { T V } } \left( R _ { j } , R _ { j - 1 } \right) = \frac { 1 } { 2 } \int R _ { j - 1 } ( \vec { q } _ { < j } ) \left( \left| Q _ { j } ^ { ( 0 ) } ( q _ { j } \mid \vec { q } _ { < j } ) - Q _ { j } ^ { ( 1 ) } ( q _ { j } \mid \vec { q } _ { < j } ) \right| \right) d ^ { 2 } q _ { j } d ^ { 2 ( j - 1 ) } \vec { q } _ { < j }\tag{D50}
$$

$$
= \int R _ { j - 1 } ( \vec { q } _ { < j } ) d _ { \mathrm { T V } } \left( Q _ { j } ^ { ( 0 ) } ( \cdot \mid \vec { q } _ { < j } ) , Q _ { j } ^ { ( 1 ) } ( \cdot \mid \vec { q } _ { < j } ) \right) d ^ { 2 ( j - 1 ) } \vec { q } _ { < j }\tag{D51}
$$

(D52)

Substituting this into the telescoping bound gives

$$
{ d _ { \mathrm { T V } } } \left( { Q } ^ { ( 0 ) } , { Q } ^ { ( 1 ) } \right) \le N \epsilon ,\tag{D53}
$$

as claimed.

The KL divergence satisfies the same chain-rule inequality, which we state without a nearly identical proof.

Lemma 21 (Adaptive KL hybrid bound). Let $Q ^ { ( 0 ) }$ and $Q ^ { ( 1 ) }$ be transcript distributions on $\mathbb { C } ^ { N }$ . If for every j and every previous transcript $\scriptstyle { { \vec { q } } < j }$

$$
D _ { \mathrm { K L } } \Big ( Q _ { j } ^ { ( 0 ) } ( \cdot \mid \vec { q } _ { < j } ) , Q _ { j } ^ { ( 1 ) } ( \cdot \mid \vec { q } _ { < j } ) \Big ) \leq \epsilon ~ ,\tag{D54}
$$

then

$$
D _ { \mathrm { K L } } \Big ( Q ^ { ( 0 ) } , Q ^ { ( 1 ) } \Big ) \leq N \epsilon \ .\tag{D55}
$$

With these results, we are prepared to state Le Cam’s two-point method and understand how it lets us control query complexity in hypothesis testing.

Lemma 22 (Le Cam’s Two-Point Method). Consider a hypothesis-testing formulation of a sensing task with signals induced by distribution $P _ { 0 }$ or $P _ { 1 }$ , and an adaptive hypothesis testing protocol whose transcript is distributed according to $Q ^ { ( 0 ) }$ or $Q ^ { ( 1 ) }$ . Then, the probability that the protocol selects the correct hypothesis is upper bounded $b y$

$$
\frac { 1 } { 2 } + \frac { 1 } { 2 } d _ { \mathrm { T V } } \left( Q ^ { ( 0 ) } , Q ^ { ( 1 ) } \right)\tag{D56}
$$

Combining the TVD hybrid bound with Le Cam’s method, we obtain the core lemma used in many informationtheoretic learning lower bounds.

Lemma 23 (Single-experiment TV lower bound). Consider a hypothesis-testing formulation of a sensing task with signals induced by $P _ { 0 }$ or $P _ { 1 }$ . Suppose that $f o r$ any allowed single-query experiment, and for every previous transcript $\scriptstyle { { \vec { q } } < j }$ , the resulting outcome distributions satisfy

$$
d _ { \mathrm { T V } } \Bigl ( Q _ { j } ^ { ( 0 ) } ( \cdot \mid \vec { q } _ { < j } ) , Q _ { j } ^ { ( 1 ) } ( \cdot \mid \vec { q } _ { < j } ) \Bigr ) \leq \Delta \ .\tag{D57}
$$

Then any adaptive sensing protocol which distinguishes $P _ { 0 }$ from $P _ { 1 }$ with success probability at least $1 / 2 + \epsilon$ must use at least

$$
N \geq \frac { 2 \epsilon } { \Delta }\tag{D58}
$$

queries to the signal. In particular, any protocol which succeeds with constant success bias ϵ requires $N \geq \Omega ( 1 / \Delta )$ queries.

Proof. Let $Q ^ { ( 0 ) }$ and $Q ^ { ( 1 ) }$ be the full transcript distributions induced by an arbitrary adaptive protocol using N queries. By Lemma $2 0$

$$
d _ { \mathrm { T V } } \left( Q ^ { ( 0 ) } , Q ^ { ( 1 ) } \right) \leq N \Delta \ .\tag{D59}
$$

Lemma 22 then implies that the success probability of the protocol is at most

$$
\frac { 1 } { 2 } + \frac { 1 } { 2 } d _ { \mathrm { T V } } \Bigl ( Q ^ { ( 0 ) } , Q ^ { ( 1 ) } \Bigr ) \le \frac { 1 } { 2 } + \frac { N \Delta } { 2 } \ .\tag{D60}
$$

Therefore, if the protocol succeeds with probability at least $1 / 2 + \epsilon ,$ it must hold that

$$
\frac { 1 } { 2 } + \epsilon \leq \frac { 1 } { 2 } + \frac { N \Delta } { 2 } \ .\tag{D61}
$$

Rearranging gives

$$
N \geq \frac { 2 \epsilon } { \Delta } \ .\tag{D62}
$$

Setting $\epsilon = 1 / 6$ gives the stated lower bound for success probability at least $2 / 3 .$

Lemma 23 is the core result that we will use to interface the AFI with information-theoretic lower bound techniques.

## 2. QΨ lower bounds

We now show that the AFI directly controls the dificulty of local learning. The mechanism is simple. Starting from a reference law $P _ { 0 }$ , we perturb it in a direction h to obtain two nearby hypotheses $P _ { \pm } = ( 1 \pm \eta h ) P _ { 0 }$ . For any fixed experiment, the corresponding change in its outcome distribution is governed by the conditional expectation $\mathbb { E } [ h ( X ) ~ | ~ Y ]$ , whose squared $L ^ { 2 }$ norm is exactly the AFI. A small AFI therefore implies that each query produces only a small statistical separation between the two hypotheses. The chain rule for adaptive protocols then accumulates this separation over queries, and Le Cam’s method converts it into a query-complexity lower bound. The following lemma formalizes this argument.

Lemma 24 (Semiclassical measurement lemma). Consider a probability space $( \mathcal { X } , \mathcal { F } , P _ { 0 } )$ . Let $h : \mathcal { X }  \mathbb { R }$ satisfy $\begin{array} { r } { \int h ( x ) d P _ { 0 } ( x ) = 0 } \end{array}$ and $\| h \| _ { \infty } \leq 1$ . For $0 < \eta < 1$ , define

$$
P _ { \pm } ( d x ) = ( 1 \pm \eta h ( x ) ) P _ { 0 } ( d x ) .\tag{D63}
$$

Let M be a class of measurements with response kernels as above, and suppose

$$
\mathsf { A } _ { \mathcal { M } } ( h ; P _ { 0 } ) \le \Lambda \ .\tag{D64}
$$

Then any adaptive protocol using measurements from M requires

$$
N \geq \Omega \left( \eta ^ { - 2 } \Lambda ^ { - 1 } \right)\tag{D65}
$$

queries to distinguish $P _ { + }$ from $P _ { - }$ with success probability at least $2 / 3$ , with the convention that $\Lambda ^ { - 1 } = \infty$ $i f \Lambda = 0$

Proof. Fix a single experiment $M \in \mathcal { M }$ . Let $Q _ { 0 }$ denote the outcome distribution under $P _ { 0 } ,$ , and let $A _ { h }$ denote the signed outcome measure

$$
A _ { h } ( d y ) = \int h ( x ) K _ { M } ( d y | x ) d P _ { 0 } ( x ) ~ .\tag{D66}
$$

The outcome distributions under $P _ { + }$ and $P _ { - }$ are

$$
Q _ { \pm } ( d y ) = Q _ { 0 } ( d y ) \pm \eta A _ { h } ( d y ) .\tag{D67}
$$

Let $R _ { h } = d A _ { h } / d Q _ { 0 }$ . Equivalently, $R _ { h } ( y ) = \mathbb { E } [ h ( X ) | Y = y ]$ under $X \sim P _ { 0 }$ and the measurement response $K _ { M } ( d y | X )$ . Since $\| h \| _ { \infty } \leq 1$ , we have $| R _ { h } ( y ) | \le 1$ for $Q _ { 0 } .$ -almost every $y .$ Thus

$$
Q _ { \pm } ( d y ) = ( 1 \pm \eta R _ { h } ( y ) ) Q _ { 0 } ( d y )\tag{D68}
$$

and $| \eta R _ { h } ( y ) | \le 1 / 2$ . Moreover,

$$
\int R _ { h } ( y ) d Q _ { 0 } ( y ) = \int h ( x ) d P _ { 0 } ( x ) = 0 .\tag{D69}
$$

Using the elementary bound $( 1 + u ) \log ( 1 + u ) - u \leq u ^ { 2 }$ for all $u \in [ - 1 , 1 ]$ , we have

$$
D _ { \mathrm { K L } } ( Q _ { + } \Vert Q _ { 0 } ) \leq \int \left( 1 + \eta R _ { h } ( y ) \right) \log \left( 1 + \eta R _ { h } ( y ) \right) d Q _ { 0 } ( y )\tag{D70}
$$

$$
= \int \left[ \left( 1 + \eta R _ { h } ( y ) \right) \log \left( 1 + \eta R _ { h } ( y ) \right) - \eta R _ { h } \right] d Q _ { 0 } ( y )\tag{D71}
$$

$$
\leq \eta ^ { 2 } \int | R _ { h } ( y ) | ^ { 2 } d Q _ { 0 } ( y )\tag{D72}
$$

$$
= \eta ^ { 2 } \mathsf { A } _ { M } ( h ; P _ { 0 } )\tag{D73}
$$

$$
\leq \eta ^ { 2 } \Lambda ,\tag{D74}
$$

Now consider an adaptive N-query protocol. At each step of the transcript, the learner may choose a diferent measurement from $\mathcal { M } .$ , but the same one-query KL bound applies uniformly. As such, applying the KL chain rule for adaptive protocols gives

$$
D _ { \mathrm { K L } } ( Q _ { + } ^ { ( N ) } \lVert Q _ { 0 } ^ { ( N ) } ) \leq N \eta ^ { 2 } \Lambda ,\tag{D75}
$$

where $Q _ { + } ^ { ( N ) }$ is the transcript distributions under $P _ { + }$ . The same inequality holds when $Q _ { + }$ is replaced by $Q _ { - }$ via the above argument. If the protocol distinguishes the two hypotheses $P _ { \pm }$ with success probability at least $2 / 3 ,$ then Lemma 22 implie

$$
d _ { \mathrm { T V } } ( Q _ { + } ^ { ( N ) } , Q _ { - } ^ { ( N ) } ) \geq \frac { 1 } { 3 } ,\tag{D76}
$$

where we move from KL divergence to TVD using Pinsker’s inequality (Lemma 19) and by applying the triangle inequality. Applying Lemma 19 in the opposite direction then gives $D _ { \mathrm { K L } } ( Q _ { + } ^ { ( N ) } | | Q _ { - } ^ { ( N ) } ) \geq c$ for a universal

constant $c > 0$ . Hence

$$
N \geq \Omega \left( \eta ^ { - 2 } \Lambda ^ { - 1 } \right) .\tag{D77}
$$

The following corollary is the form of the lemma we will use when a constrained measurement class has a sparse Fourier response.

Corollary 25 (Sparse-response semiclassical measurement lemma). Consider the setting of Lemma ${ \it 2 4 } .$ . Let D be a valid dictionary for M, and suppose M is L-sparse in D under $P _ { 0 }$ . Then

$$
\mathsf { A } _ { \mathcal { M } } ( h ; P _ { 0 } ) \leq \mathsf { C } _ { L } ( h ; \mathscr { D } , P _ { 0 } ) .\tag{D78}
$$

Consequently, any adaptive protocol using measurements from M requires

$$
N \geq \Omega \left( \eta ^ { - 2 } \mathsf { C } _ { L } ( h ; \mathcal { D } , P _ { 0 } ) ^ { - 1 } \right)\tag{D79}
$$

queries to distinguish $P _ { + }$ from $P _ { - }$ with success probability at least $2 / 3$

$I f D$ is orthonormal and $\begin{array} { r } { h = \sum _ { \omega \in S _ { h } } c _ { \omega } \chi _ { \omega } } \end{array}$ , with $| S _ { h } | = H$ and $| c _ { \omega } | ^ { 2 } \leq M / H$ for every $\omega \in S _ { h }$ , then

$$
N \geq \Omega \left( \eta ^ { - 2 } \frac { H } { M L } \right) .\tag{D80}
$$

Proof. Fix $M \in { \mathcal { M } }$ . By L-sparsity, there is a set $S _ { M } \subset \Omega$ with $| S _ { M } | \le L$ such that every centered response function lies in

$$
V _ { M } = \operatorname { s p a n } \{ \widetilde { \chi } _ { \omega } : \omega \in S _ { M } \} .\tag{D81}
$$

Let $\Pi _ { M }$ denote the orthogonal projector onto $V _ { M }$ . For a finite or countable outcome measurement, write $\begin{array} { r } { q _ { M , 0 } ( y ) = \int f _ { M , y } ( x ) d P _ { 0 } ( x ) } \end{array}$ . Since h is centered,

$$
\int f _ { M , y } ( x ) h ( x ) d P _ { 0 } ( x ) = \left. f _ { M , y } - q _ { M , 0 } ( y ) , h \right. _ { P _ { 0 } } .\tag{D82}
$$

Since $f _ { M , y } - q _ { M , 0 } ( y ) \in V _ { M }$ , we can move the projection onto h:

$$
\left. f _ { M , y } - q _ { M , 0 } ( y ) , h \right. _ { P _ { 0 } } = \left. f _ { M , y } - q _ { M , 0 } ( y ) , \Pi _ { M } h \right. _ { P _ { 0 } } .\tag{D83}
$$

Therefore

$$
\mathsf { A } _ { M } ( h ; P _ { 0 } ) = \mathsf { A } _ { M } ( \Pi _ { M } h ; P _ { 0 } ) .\tag{D84}
$$

By the contraction property of accessible feature information,

$$
\mathsf { A } _ { M } ( \Pi _ { M } h ; P _ { 0 } ) \leq \| \Pi _ { M } h \| _ { L ^ { 2 } ( P _ { 0 } ) } ^ { 2 } .\tag{D85}
$$

Thus

$$
\mathsf { A } _ { M } ( h ; P _ { 0 } ) \leq \mathsf C _ { L } ( h ; \mathscr { D } , P _ { 0 } ) .\tag{D86}
$$

Taking the supremum over $M \in \mathcal { M }$ gives $\mathsf { A } _ { \mathcal { M } } ( h ; P _ { 0 } ) \leq \mathsf C _ { L } ( h ; \mathcal { D } , P _ { 0 } )$ , and Lemma 24 gives the query lower bound. The exactly orthonormal case is an illustrative example. If D is orthonormal and $\begin{array} { r } { \begin{array} { r } { h = \sum _ { \omega \in S _ { h } } c _ { \omega } \chi _ { \omega } , } \end{array} } \end{array}$ then for any set S,

$$
\big \| \Pi _ { S } h \big \| _ { L ^ { 2 } ( P _ { 0 } ) } ^ { 2 } = \sum _ { \omega \in S \cap S _ { h } } | c _ { \omega } | ^ { 2 } .\tag{D87}
$$

Therefore

$$
\mathsf C _ { L } ( h ; \mathcal D , P _ { 0 } ) = \operatorname* { m a x } _ { S \subseteq S _ { h } \atop | S | \leq L } \sum _ { \omega \in S } | c _ { \omega } | ^ { 2 } .\tag{D88}
$$

If $| c _ { \omega } | ^ { 2 } \leq M / H$ for all $\omega \in S _ { h }$ and $H = | S _ { h } |$ , then

$$
{ \mathsf C } _ { L } ( h ; { \mathcal D } , P _ { 0 } ) \le \frac { M L } { H } ,\tag{D89}
$$

which gives the final lower bound.

By Fact 7, the hypothesis-testing bound produced by the semiclassical measurement lemma applies directly to any protocol for learning the value of h to absolute precision η. We therefore see that through QΨ, a bound on the AFI directly yields a lower bound on query complexity. Next, we show that this lower bound is in fact saturable: the AFI can also certify an optimal hypothesis-testing protocol.

## 3. QΨ hypothesis testing guarantee

The lower bound above is tight because the same conditional expectation that defines the AFI also provides the optimal statistic for detecting the perturbation. For a fixed experiment M, define

$$
R _ { M , h } ( y ) = \frac { d A _ { M , h } } { d Q _ { M , 0 } } ( y ) = \mathbb { E } [ h ( X ) \mid Y = y ] .\tag{D90}
$$

Under the perturbed law $P _ { \eta } = ( 1 + \eta h ) P _ { 0 }$ , the mean of $R _ { M , h } ( Y )$ shifts by exactly $\eta \mathsf { A } _ { M } ( h ; P _ { 0 } )$ . Repeatedly measuring M and averaging this statistic therefore produces an estimator whose sample complexity is proportional to the inverse AFI. We now formalize this matching upper bound.

Lemma 26 (Variational form of accessible feature information). Let M be an experiment with response kernel $K _ { M } ( d y | x )$ , and let $h \in L ^ { 2 } ( P _ { 0 } )$ . Then

$$
\mathsf { A } _ { M } ( h ; P _ { 0 } ) = \operatorname* { s u p } _ { \substack { g : y _ { M }  \mathbb { C } } } | \int h ( x ) ( \int g ( y ) K _ { M } ( d y | x ) ) d P _ { 0 } ( x ) | ^ { 2 } .\tag{D91}
$$

If $\mathsf { A } _ { M } ( h ; P _ { 0 } ) > 0$ , the supremum is achieved by $g ( y ) = R _ { M , h } ( y ) / \sqrt { \mathsf { A } _ { M } ( h ; P _ { 0 } ) }$

Proof. By definition of $A _ { M , h }$

$$
\int h ( x ) \left( \int g ( y ) K _ { M } ( d y | x ) \right) d P _ { 0 } ( x ) = \int g ( y ) d A _ { M , h } ( y ) \ .\tag{D92}
$$

Since $d A _ { M , h } = R _ { M , h } d Q _ { M , 0 }$ , the right-hand side becomes

$$
\int g ( y ) R _ { M , h } ( y ) d Q _ { M , 0 } ( y ) \ .\tag{D93}
$$

By Cauchy–Schwarz,

$$
\left| \int g ( y ) R _ { M , h } ( y ) d Q _ { M , 0 } ( y ) \right| ^ { 2 } \leq \mathbb { E } _ { Y \sim Q _ { M , 0 } } [ | g ( Y ) | ^ { 2 } ] \mathbb { E } _ { Y \sim Q _ { M , 0 } } [ | R _ { M , h } ( Y ) | ^ { 2 } ] ~ .\tag{D94}
$$

The first factor is at most one by assumption, while the second factor is exactly $\mathsf { A } _ { M } ( h ; P _ { 0 } )$ . This proves the upper bound. If $\mathsf { A } _ { M } ( h ; P _ { 0 } ) > 0$ , choosing $g = R _ { M , h } / \sqrt { \mathsf { A } _ { M } ( h ; P _ { 0 } ) }$ saturates Cauchy–Schwarz. □

This variational form tells us that the AFI is exactly the maximum squared overlap with h that can be extracted from the measurement outcome using any classical postprocessing of that outcome. We now convert this operational statement into an explicit sample-complexity guarantee.

Theorem D.27 (QΨ sample complexity guarantee). Consider a probability space $( \mathcal { X } , \mathcal { F } , P _ { 0 } )$ and let h : X → R satisfy $\begin{array} { r } { \int h ( x ) d P _ { 0 } ( x ) = 0 } \end{array}$ and $\| h \| _ { \infty } \leq 1$ . For $| \eta | \le 1$ , define

$$
P _ { \eta } ( d x ) = ( 1 + \eta h ( x ) ) P _ { 0 } ( d x ) ~ .\tag{D95}
$$

Fix an experiment M such that $\mathsf { A } _ { M } ( h ; P _ { 0 } ) \ > \ 0$ . Run M independently N times, obtaining outcomes $Y _ { 1 } , \dots , Y _ { N }$ , and define

$$
\widehat { \eta } = \frac { 1 } { N \mathsf { A } _ { M } ( h ; P _ { 0 } ) } \sum _ { j = 1 } ^ { N } R _ { M , h } ( Y _ { j } ) \ .\tag{D96}
$$

Then $\widehat { \eta }$ is an unbiased estimator of η. Moreover, for every $0 < \epsilon \leq 1$ and $0 < \delta < 1$ ，

$$
N = O \left( \frac { 1 } { \epsilon ^ { 2 } \mathsf { A } _ { M } ( h ; P _ { 0 } ) } \log { \left( \frac { 1 } { \delta } \right) } \right)\tag{D97}
$$

samples sufice to learn η to within absolute error ϵ with probability at least $1 - \delta ,$ and thereby distinguish the hypotheses $P _ { \eta }$ and $P _ { - \eta }$ with success probability at least $1 - \delta$

Proof. Let $Q _ { \eta }$ denote the outcome distribution obtained by running experiment M when the quantum oracle is distributed according to $P _ { \eta }$ . By the definition of $R _ { M , h }$

$$
Q _ { \eta } ( d y ) = \left( 1 + \eta R _ { M , h } ( y ) \right) Q _ { M , 0 } ( d y ) \ .\tag{D98}
$$

Since $h$ is centered, $\begin{array} { r } { \int R _ { M , h } ( y ) d Q _ { M , 0 } ( y ) = 0 } \end{array}$ . Therefore

$$
\begin{array} { l } { { \displaystyle \mathbb { E } _ { Y \sim Q _ { \eta } } [ R _ { M , h } ( Y ) ] = \int R _ { M , h } ( y ) \left( 1 + \eta R _ { M , h } ( y ) \right) d Q _ { M , 0 } ( y ) } } \\ { { \displaystyle \qquad = \eta \mathsf { A } _ { M } ( h ; P _ { 0 } ) ~ . } } \end{array}\tag{D99}
$$

(D100)

This proves that $\widehat { \eta }$ is unbiased. Next we obtain the sample complexity guarantee. Since $R _ { M , h } ( y ) = \mathbb { E } [ h ( X ) | Y =$ y] and $\| h \| _ { \infty } \leq 1$ , we have $| R _ { M , h } ( y ) | \leq 1$ for $Q _ { M , 0 } .$ -almost every y. Furthermore,

$$
\mathbb { E } _ { Y \sim Q _ { \eta } } [ R _ { M , h } ( Y ) ^ { 2 } ] = \int R _ { M , h } ( y ) ^ { 2 } ( 1 + \eta R _ { M , h } ( y ) ) d Q _ { M , 0 } ( y )\tag{D101}
$$

$$
\leq ( 1 + | \eta | ) \mathsf { A } _ { M } ( h ; P _ { 0 } )\tag{D102}
$$

$$
\leq \frac 3 2 \mathsf { A } _ { M } ( h ; P _ { 0 } ) \ .\tag{D103}
$$

Thus each summand has variance at most $\textstyle { \frac { 3 } { 2 } } \mathsf { A } _ { M } ( h ; P _ { 0 } )$ and is bounded in absolute value by one. Bernstein’s inequality gives

$$
\operatorname* { P r } \left[ \left| \frac { 1 } { N } \sum _ { j = 1 } ^ { N } R _ { M , h } ( Y _ { j } ) - \eta \mathsf { A } _ { M } ( h ; P _ { 0 } ) \right| > \epsilon \mathsf { A } _ { M } ( h ; P _ { 0 } ) \right] \le 2 \exp \left( - c N \epsilon ^ { 2 } \mathsf { A } _ { M } ( h ; P _ { 0 } ) \right)\tag{D104}
$$

for a universal constant $c > 0$ , where we used $0 < \epsilon \leq 1$ and $\mathsf { A } _ { M } ( h ; P _ { 0 } ) \leq \| h \| _ { L ^ { 2 } ( P _ { 0 } ) } ^ { 2 } \leq 1$ . Rearranging proves the stated sample complexity. The hypothesis testing guarantee follows estimating η and deciding between the two hypotheses by its sign. □

Theorem D.27 shows that the accessible feature information appearing in the lower-bound lemma is exactly achievable. This is intuitive: the AFI exactly quantifies the maximum phase-space overlap achievable between a property of interest and a particular experiment. By running the experiment which maximizes the AFI within a class of interest, one saturates the lower bound.

## 4. AFI as certificate of advantage and global learning

Putting together Theorems 24 and D.27 leads us to the following fact.

Corollary 28. Let M be any class ofquantum learning protocols and consider a function h on a probability space with some measure $P _ { 0 }$ as in Theorem ${ \it 2 4 } .$ . Then,

$$
N = \Theta \left( \frac { 1 } { \mathsf { A } _ { \mathcal { M } } ( h ; P _ { 0 } ) \epsilon ^ { 2 } } \right)\tag{D105}
$$

samples from $P = P _ { \epsilon }$ or $P _ { - \epsilon }$ as in Theorem D.27 are necessary and suficient to distinguish the hypotheses with constant success probability.

There are several important points of interpretation. First, because of Corollary 28, the AFI functions operationally similarly to the QFI in estimation theory, despite having a diferent formulation and set of use cases. In particular, the AFI is an information quantity which directly controls the sample complexity of a learning protocol via our semiclassical measurement lemmas, in the same way that the QFI controls sensing time in metrology via the Cramer-Rao bounds. Like the QFI, the AFI gives a saturable lower bound. This suggests an operational design principle: upon fixing a reference distribution $P _ { 0 }$ and an observable perturbation around it $h ,$ one can design a provably optimal learning procedure by optimizing the AFI; this has been the subject of extensive work in the metrological setting, where many semidefinite programming-based approaches for maximizing the QFI have been studied across applications. Because the AFI concerns only real-valued functions, it may even lead to more classically tractable methods for quantum experimental design; these ideas are developed further in Section H. The AFI has yet another operational similarity to the QFI: it is a quantity that controls local estimation problems. In particular, it controls estimation of perturbations around a known reference distribution $P _ { 0 }$ , which is precisely the setting considered in quantum learning lower bounds.

However, the AFI has several important characteristics that greatly difer from the QFI and make it a more flexible tool for learning-theoretic analysis. On the one hand, AFI lower bounds are agnostic to estimator bias, an important blindspot for QFI-based lower bounds [48]. Moreover, the QFI can fail to produce meaningful results if the measurement response is not a continuous function of the underlying parameter, so discrete learning strategies can have arbitrarily small QFI despite succeeding with few samples or sensing time. The AFI does not face this issue and will succeed with arbitrary quantum POVMs and classical postprocessing, as it only relies on the phase-space representation of an experiment. Most importantly, the AFI is a flexible quantity for experimentally meaningful tasks. The QFI is most commonly used to study quadratic sensingtime separations between structured sensing protocols in the asymptotic setting, and the scaling that results from Cramer-Rao bounds is quite sensitive to small changes in the sensing task (for instance, the asymptotic scaling for Heisenberg-limited DC metrology tends to revert to the standard quantum limit under any noise that is more than inverse-polynomially small, losing track of important finite-size gains [48]). While QFI may be able to produce exponential lower bounds in certain engineered settings, it has not found broad use as a tool for proving superpolynomial quantum advantages. AFI, on the other hand, does not encounter these issues as evidenced by the exponential separations in the remainder of this work, and excels at producing tight learning-theoretic analyses of realistic experimental tasks.

The AFI gives a sharp characterization of local learning around a reference law $P _ { 0 } \colon$ one specifies a perturbation direction h and asks how eficiently that perturbation can be detected or estimated. Many experimental objectives are instead genuinely global, with no distinguished reference distribution or local perturbation. For these problems, QΨ uses the same phase-space idea in the Fourier domain.

The basic question is then frequency by frequency: how strongly can the experimental architecture respond to each Fourier component of the property being learned? The answer is quantified by the Fourier gain. Once the property and the experimentally achievable responses are expanded in the same Fourier dictionary, global experimental design reduces to matching the Fourier response of the experiment to the Fourier support of the target.

Definition 29 (Fourier gain of a sensing protocol family). Let A be a class of experimental protocols, including classical postprocessing, and for a protocol in ${ \mathcal { A } } ,$ let $Z$ denote the scalar output of one shot. Let

$$
f _ { Z } ( x ) = \mathbb { E } [ Z | X = x ]\tag{D106}
$$

denote the protocol’s response function, and let $\begin{array} { r } { V _ { Z } = \operatorname* { s u p } _ { x } \mathbb { E } [ | Z | ^ { 2 } | X = x ] } \end{array}$ be its maximum second moment. $_ { L e t }$

$\{ \chi _ { \omega } \}$ be a valid Fourier dictionary on the relevant sample space, such that $\begin{array} { r } { f _ { Z } = \sum _ { \omega } \widehat { f } _ { Z } ( \omega ) \chi _ { \omega } } \end{array}$ . Then, the Fourier gain of A at frequency ω is defined as

$$
G _ { \mathcal { A } } ( \omega ) = \operatorname* { s u p } _ { Z \in \mathcal { A } } \frac { | \widehat { f } _ { Z } ( \omega ) | ^ { 2 } } { V _ { Z } } \ .\tag{D107}
$$

Intuitively, $G _ { \mathcal { A } } ( \omega )$ is the largest squared Fourier coeficient that the sensing protocol can place at frequency ω per unit shot noise. By decomposing the property in the same Fourier dictionary, sample complexity will be controlled by the ratio between the coeficient of the property and the Fourier gain at that frequency. This is precisely how the phase-space-overlap picture directly controls sample complexity in true global learning problems. These observations are formalized in the following theorem.

Theorem D.30 (QΨ global learning guarantee). Let ψ be a property kernel with finite Fourier expansion

$$
\psi ( x ) = \sum _ { \omega \in S } \widehat { \psi } ( \omega ) \chi _ { \omega } ( x )\tag{D108}
$$

for some finite set S. Let P be a class of signal distributions for which $\Psi _ { \psi } ( P ) = \mathbb { E } _ { X \sim P } [ \psi ( X ) ]$ is well defined. Let A be a class of sensing protocols with Fourier gain $G _ { \mathcal { A } } ( \omega )$ on each $\omega \in S ,$ and suppose $G _ { \mathcal { A } } ( \omega ) > 0$ whenever $\widehat { \psi } ( \omega ) \neq 0 .$ . Suppose moreover that there is an integer $m \in \{ 1 , \ldots , | S | \}$ such that, for every subset $T \subseteq S$ with $| T | \leq$ m and every choice of coeficients $\{ a _ { \omega } \} _ { \omega \in T } ,$ the protocol family A can synthesize the response

$$
f _ { T } ( x ) = \sum _ { \omega \in T } a _ { \omega } \chi _ { \omega } ( x )\tag{D109}
$$

with one-shot output $Z _ { T }$ satisfying

$$
\mathbb { E } [ Z _ { T } | X = x ] = f _ { T } ( x )\tag{D110}
$$

and

$$
\operatorname* { s u p } _ { x } \mathbb { E } [ | Z _ { T } | ^ { 2 } | X = x ] \leq C _ { A } \sum _ { \omega \in T } \frac { | a _ { \omega } | ^ { 2 } } { G _ { A } ( \omega ) }\tag{D111}
$$

for a constant $C _ { A }$ independent of $S$ and $\psi .$ . Then there is a protocol in A which estimates $\Psi _ { \psi } ( P )$ to additive error ϵ with probability at least $1 - \delta$ for every $P \in { \mathcal { P } }$ using

$$
N = O \left( \frac { C _ { A } } { \epsilon ^ { 2 } } \operatorname* { m i n } \left\{ \frac { | S | } { m } \sum _ { \omega \in S } \frac { | \widehat { \psi } ( \omega ) | ^ { 2 } } { G _ { A } ( \omega ) } , \left( \sum _ { \omega \in S } \frac { | \widehat { \psi } ( \omega ) | } { \sqrt { G _ { A } ( \omega ) } } \right) ^ { 2 } \right\} \log { \left( \frac { 1 } { \delta } \right) } \right)\tag{D112}
$$

queries.

Before we give the proof, let us break down the interpretation of this result intuitively. The first important object is S, which is the Fourier support of the property of interest. As long as the chosen Fourier basis satisfies approximate orthogonality, there is no value in synthesizing a measurement whose response kernel has support outside of S, since probability amplitude spent there is uninformative and can at worst bias the outcome. Then, when we say that A has Fourier gain $G _ { \mathcal { A } } ( \omega )$ on each $\omega \in S$ , this is a statement that the best protocol in A for that single frequency ω achieves a response amplitude (per unit variance) of $G _ { \mathcal { A } } ( \omega )$ . However, the property kernel can be supported on many, possibly unknown frequencies, so we do not, a priori, know which protocol in A to choose. Ideally, we would like to synthesize a response which consists of many frequencies, with amplitudes tuned to maximize Fourier overlap with ψ. However, constraints on $\mathcal { A }$ such as energy, memory, or control depth can limit the extent to which a synthesized response can attain a large overlap. To capture this, we introduce m and say that we allow any response in A to have at most m Fourier terms. In practice, m is a byproduct of constraints.

With these definitions, we say that $Z _ { T }$ is simply the output of a chosen sensing strategy, with expectation $f _ { T } ( x )$ being the response function. However, we still need Equation (D111) for the following reason. When a protocol synthesizes a multi-frequency response, it must choose to distribute response mass in a concrete way.

This can include, for instance, classical randomization of the probe state or a non-Gaussian state preparation that creates a superposition state with many branches of difering amplitude. After querying the quantum oracle and performing a measurement, it could be the case that a poorly chosen protocol, despite having desirable Fourier overlap with the target property, has a large readout variance. A toy example is as follows.

Imagine a class of experiments $\mathcal { A }$ with exactly two allowed protocols which have response functions $f _ { 1 } ( x ) =$ $\chi _ { \omega _ { 1 } } ( x ) , f _ { 2 } ( x ) = \chi _ { \omega _ { 2 } } ( x )$ with unit variance, and suppose we wish to learn the property $\psi ( x ) = ( \chi _ { \omega _ { 1 } } + \chi _ { \omega _ { 2 } } ) ( x ) / \sqrt { 2 }$ If $f _ { 1 } + f _ { 2 }$ were included in ${ \mathcal { A } } .$ , the property could be measured directly with unit variance. However, because the responses are only separately synthesizable, the optimal protocol is to randomly measure $f _ { 1 }$ and $f _ { 2 }$ each with probability $1 / 2$ and combine with classical postprocessing, which yields a variance of 2. If one now considered a property consisting of n modes, readout variance could be amplified as many as n times. Therefore, when one places the constraint that the architecture can synthesize at most m responses simultaneously, this is only meaningful if it can do so while still retaining a bounded variance. This is the precise meaning behind Equation (D111).

With this understood, the final sample complexity bound becomes illustrative. In short, the sample complexity is controlled by the ratio

$$
\sum _ { \omega \in S } \frac { | \widehat { \psi } ( \omega ) | ^ { 2 } } { G _ { A } ( \omega ) }\tag{D113}
$$

which is exactly the overlap between the property’s Fourier amplitude at each frequency and the Fourier gain of the architecture at that frequency. This is adjusted for the number of frequencies that are simultaneously synthesizable with bounded variance. The second term in the minimum is the extreme case in which the protocol can only synthesize a single frequency at a time, such that randomizing over frequencies is the best option; this term is a tighter bound than simply sending $m \to 1$ . With this, we now see that Fourier overlap is exactly the quantity which determines sample complexity in global learning problems, giving us a meaningful optimization design principle.

Proof. We prove two upper bounds and then take the better of the two. First, we use the fact that the architecture can synthesize $m = c | S |$ Fourier components at once (letting $c = m / | S | )$ . Choose a subset $T \subseteq S$ uniformly at random among all subsets of size $m .$ Then every $\omega \in S$ is included in $T$ with probability c. Conditioned on the choice of $T ,$ use the assumed synthesis property with coeficients

$$
a _ { \omega } = { \frac { { \widehat { \psi } } ( \omega ) } { c } }\tag{D114}
$$

for $\omega \in T$ . The resulting one-shot output $Z _ { T }$ has conditional response

$$
\mathbb { E } [ Z _ { T } | X = x , T ] = \sum _ { \omega \in T } \frac { \widehat { \psi } ( \omega ) } { c } \chi _ { \omega } ( x ) \ .\tag{D115}
$$

Averaging also over the random choice of $T$ gives

$$
\mathbb { E } [ Z _ { T } | X = x ] = \mathbb { E } _ { T } \left[ \sum _ { \omega \in T } \frac { \widehat { \psi } ( \omega ) } { c } \chi _ { \omega } ( x ) \right]\tag{D116}
$$

$$
= \sum _ { \omega \in S } \operatorname* { P r } [ \omega \in T ] { \frac { { \widehat { \psi } } ( \omega ) } { c } } \chi _ { \omega } ( x )\tag{D117}
$$

$$
= \sum _ { \omega \in S } { \widehat { \psi } } ( \omega ) \chi _ { \omega } ( x )\tag{D118}
$$

$$
= \psi ( x ) ~ .\tag{D119}
$$

$$
\operatorname* { s u p } _ { x } \mathbb { E } [ | Z _ { T } | ^ { 2 } | X = x ] \leq \mathbb { E } _ { T } \left[ C _ { \cal A } \sum _ { \omega \in { \cal T } } \frac { | \widehat { \psi } ( \omega ) | ^ { 2 } } { c ^ { 2 } G _ { { \cal A } } ( \omega ) } \right]
$$

Thus this randomized one-shot protocol is an unbiased estimator of the kernel $\psi$ pointwise, and hence an unbiased estimator of $\Psi _ { \psi } ( P )$ for every $P \in \mathcal { P }$ . Its second moment is bounded by

(D120)

$$
= \frac { C _ { A } } { c } \sum _ { \omega \in S } \frac { | \widehat \psi ( \omega ) | ^ { 2 } } { G _ { A } ( \omega ) } \ .\tag{D121}
$$

Averaging independent repetitions and applying the standard median-of-means estimator to the real and imaginary parts gives

$$
N = O \left( \frac { C _ { A } } { c \epsilon ^ { 2 } } \left( \sum _ { \omega \in S } \frac { | \widehat { \psi } ( \omega ) | ^ { 2 } } { G _ { A } ( \omega ) } \right) \log { \left( \frac { 1 } { \delta } \right) } \right)\tag{D122}
$$

queries. Second, we give an estimator which estimates every frequency independently, which corresponds to the regime where one does not use a protocol that can synthesize response functions containing multiple Fourier frequencies. For each $\omega \in S ,$ define

$$
A _ { \omega } = \frac { | \widehat { \psi } ( \omega ) | } { \sqrt { G _ { \mathcal { A } } ( \omega ) } }\tag{D123}
$$

and let $\begin{array} { r } { A = \sum _ { \omega \in S } A _ { \omega } } \end{array}$ . If $A = 0 ,$ , then $\psi = 0$ and the claim is trivial, so assume $A > 0 .$ . Choose frequency ω with probability $q _ { \omega } = A _ { \omega } / A$ . Conditioned on choosing $\omega ,$ use the assumed synthesis property on the one-element set {ω} with coeficient

$$
a _ { \omega } = \frac { \widehat { \psi } ( \omega ) } { q _ { \omega } } \ .\tag{D124}
$$

The resulting one-shot output $Z _ { \omega }$ has averaged response

$$
\begin{array} { c } { { \mathbb { E } [ Z _ { \omega } | X = x ] = \displaystyle \sum _ { \omega \in S } q _ { \omega } \frac { \widehat { \psi } ( \omega ) } { q _ { \omega } } \chi _ { \omega } ( x ) } } \\ { { = \psi ( x ) ~ . } } \end{array}\tag{D125}
$$

(D126)

Its second moment is bounded by

$$
\operatorname* { s u p } _ { x } \mathbb { E } [ | Z _ { \omega } | ^ { 2 } | X = x ] \leq \sum _ { \omega \in S } q _ { \omega } C _ { \mathcal { A } } \frac { | \widehat { \psi } ( \omega ) | ^ { 2 } } { q _ { \omega } ^ { 2 } G _ { \mathcal { A } } ( \omega ) }\tag{D127}
$$

$$
= C _ { A } \sum _ { \omega \in S } \frac { | \widehat { \psi } ( \omega ) | ^ { 2 } } { q _ { \omega } G _ { A } ( \omega ) }\tag{D128}
$$

$$
= C _ { \cal { A } } \left( \sum _ { \omega \in { \cal { S } } } \frac { | \widehat \psi ( \omega ) | } { \sqrt { G _ { \cal { A } } ( \omega ) } } \right) ^ { 2 } .\tag{D129}
$$

Again applying median-of-means to the empirical average gives

$$
N = O \left( \frac { C _ { A } } { \epsilon ^ { 2 } } \left( \sum _ { \omega \in S } \frac { | \widehat { \psi } ( \omega ) | } { \sqrt { G _ { A } ( \omega ) } } \right) ^ { 2 } \log { \left( \frac { 1 } { \delta } \right) } \right)\tag{D130}
$$

queries. Taking the better of the m-frequency synthesis protocol and the independent-frequency protocol proves the theorem. □

We will see in the remainder of this work that the global learning algorithm produced by Theorem D.30 matches the sample complexity of the optimal hypothesis test produced by Theorem D.27 in many cases, enabling provably optimal global learning. Moreover, the Fourier-overlap picture suggests a natural numerical optimization route: given a continuous family of experiments, one can either find a low-dimensional parameterization or a finite-size subset of experiments, and then numerically optimize the achievable Fourier gain for a desired property. Optimal experimental design then becomes a classical statistical optimization problem, with sample complexity certified by Theorem D.30. This idea, which suggests automated design of quantum-enhanced experiments with certifiable advantage, is further developed in Section H.

To conclude, we next understand how to put the pieces of $\mathrm { Q } \Psi$ together to design and discover experimentally-

## 5. Using QΨ to discover quantum advantages

We now have all the tools to use QΨ to design quantum experiments while certifying an advantage. We have seen that the AFI tightly certifies the optimal query complexity of a hypothesis-testing task based on an observable of interest, under the constraints of an experiment class. To prove a quantum speedup, we will apply our QΨ hypothesis-testing tools to two classes of experimental protocols, with and without a particular quantum processing resource. We then prove that the added resource amplifies the AFI, establishing a hypothesis-testing speedup. We then apply our global-learning guarantee to the task of interest to produce a general-purpose learning algorithm achieving a practical quantum advantage. This procedure is detailed in the following reusable proof template. Here we assume that a phase-space $\chi$ has been fixed based on the learning task.

```latex
Template for QΨ quantum advantage. Provided an experimental task with the following inputs:
• The objective is represented as a family of functions $f ^ { ( k ) } ( x )$ , for $x \in \chi .$
• Let $P _ { 0 }$ be a probability distribution supporting approximate orthogonality of some valid Fourier
dictionary of $\chi$ (for continuous-variable systems, $P _ { 0 }$ will often be a Gaussian, or if working on a
compact phase space, a uniform distribution). Let $\begin{array} { r } { \bar { \boldsymbol { h ^ { ( k ) } } } = f ^ { ( k ) } - \int f ^ { ( k ) } ~ d P _ { 0 } } \end{array}$
• Let $\mathcal { M } _ { \mathrm { c o n v } }$ be a class of “conventional” learning protocols, and let $\mathcal { M } _ { \mathrm { Q I P } }$ be a class of stronger
protocols in which conventional experimentation is enhanced by access to a quantum processing
resource.
The QΨ procedure is as follows.
1. Certify the optimal hypothesis-testing advantage.
(a) Obtain an upper bound $\mathsf { A } _ { \mathcal { M } _ { \mathrm { c o n v } } } ( h ^ { ( k ) } ; P _ { 0 } ) \leq c ( k )$
(b) Obtain a lower bound $A _ { \mathcal { M } _ { \mathrm { Q I P } } } ( h ^ { ( k ) } ; P _ { 0 } ) \ge q ( k )$
(c) Conclude by Theorem 28 that there exists a protocol in $\mathcal { M } _ { \mathrm { Q I P } }$ achieving a quantum advantage
of at least $\Omega ( q ( k ) / c ( k ) )$ over every, possibly adaptive sequence of experiments from $\mathcal { M } _ { \mathrm { c o n v } }$
2. Produce an algorithm for global learning.
(a) Choose a parameterization or discretization of $\mathcal { M } _ { \mathrm { Q I P } }$ , and (analytically or numerically) optimize
the Fourier gain relative to $h ^ { ( k ) }$
(b) Apply Theorem D.30 to design an optimal sequence of quantum-enhanced experiments. Note
that the hypothesis-testing upper bound, Theorem D.27 may often already produce a valid
global learning algorithm.
(c) Use Fact 7 to lift the hypothesis-testing lower bound to global learning, and conclude optimal
quantum-enhanced experimental design.
```

QΨ takes as input experimentally-grounded learning tasks, architectures, and proposed quantum resources, and interfaces them using the phase-space picture we have introduced. By controlling the AFI of conventional and quantum-enhanced classes, QΨ outputs tight bounds on the achievable quantum advantage, while producing the optimal hypothesis-testing algorithm certifying this advantage under the chosen constraints. It also provides a route to systematically design optimal global-learning algorithms. Because realistic experimental constraints on resources like energy, coherence, magic, and non-Gaussianity appear as natural convex constraints on phasespace response functions, the phase-space structure leveraged by QΨ presents a uniquely tractable method for rigorous, certifiably optimal quantum-enhanced experimental design beyond the scope of existing methods in both quantum learning theory and quantum metrology.

Applied to single-mode sensing, QΨ enables the proofs of quantum advantage in the subsequent appendices, as well as the design of the experiments behind Figures 2 and 3. This new approach allows us to prove the first exponential quantum advantages in learning that use only a single qubit, as well as the first exponential separations between Gaussian and non-Gaussian strategies that are enabled by single-qubit quantum control.

We also show in Appendix E 5 a that existing quantum learning separations can emerge as special cases of the QΨ approach with substantially simplified proofs. We believe that this framework may enable the discovery of new, practical quantum advantages in experimental tasks compatible with near-term quantum hardware. These perspectives are discussed in Appendix H.

## Appendix E: Hardness of learning from constrained experiments

In this section we will establish the exponential lower bounds discussed in the main text. Our bounds reveal operational constraints on the capabilities of sensing architectures at each level of the sensing hierarchy discussed in Section C 4 and depicted in Figure 1.

First, we show that any classical sensor, irrespective of total available energy, requires $\exp ( \Omega ( k ) )$ rounds of interrogation of a signal to estimate its k-th angular Fourier coeficient or moment. Next, we move into the first level of quantum-enhanced sensing, and consider the class of conventional Gaussian sensing protocols implemented in modern optical sensing platforms. Here, the amount of energy which can be coherently utilized to create squeezed Gaussian probe states and reduce quadrature variance below the vacuum floor is the deciding resource: we show that any Gaussian protocol with energy k per experiment and arbitrary allowed adaptivity requires exp(Ω(k)) interrogations to estimate a signal’s k-th Fourier coeficient. Moving slightly deeper into the regime of quantum-enhanced sensing, we consider protocols which now couple ancilla qubits to the quantum sensor, enabling computationally universal control. Here, we prove that characterizing time-dependent Fourier spectra remains hard when the sensing architecture does not support long-lived quantum memory: estimating a Fourier coeficient across m points in time requires $\exp ( \Omega ( m ) )$ interrogations when all resources decohere much more quickly than the lifetime of the signal. Finally, we characterize the advantages gained by future quantum-enhanced sensing architectures which may support deep quantum control. We show that there exist degree-k polynomial QSL properties of the signal such that for a constant $\gamma \in ( 1 / 2 , 1 )$ , any sensing architecture which can create probe states using circuit depth no more than $\gamma k$ requires $\exp ( \Omega ( k ) )$ interrogations.

Our lower bounds make use of the novel techniques introduced in Section D, elucidating the utility of our approach for obtaining lower bounds in various constrained learning settings. Then in Section F, we will revisit the same tasks considered here and provide eficient sensing protocols which live at one higher level of the sensing hierarchy than each lower bound given. Together, these results rigorously establish a hierarchy of exponential advantages for classical sensing, using diferent quantum information processing capabilities of only a single qubit.

## 1. Hardness of learning with classical probes

First, we establish hardness for classical sensing protocols by proving the following theorem. As discussed in Section C 4, we prove a bound against the class of coherent-probe protocols from Definition 17, which circumscribe all classical sensing protocols. In fact, the following bound even allows a protocol to make arbitrarily high-energy joint measurements on many probe states at once; the only constraint is that the probes themselves must be classical.

Theorem E.1. Any adaptive coherent probe protocol that can estimate the $k ^ { \mathrm { t h } }$ angular Fourier coeficient of a signal requires least $S \geq \exp ( \Omega ( k ) )$ signal queries. More specifically, there are constants $C _ { 1 } , C _ { 2 }$ such that

$$
S \geq C _ { 1 } \exp \left( \frac { k } { 2 } \operatorname { a r s i n h } \left( C _ { 2 } k \right) - \frac { C _ { 2 } } { 2 } \left( \sqrt { 1 + C _ { 2 } ^ { 2 } k ^ { 2 } } - 1 \right) \right)\tag{E1}
$$

and such that the exponent is $\Omega ( k )$ for all $k > 0$

Proof. Our proof will be based around use of the semiclassical quantum measurement lemma 24, beginning with the instantiation of two hypothesis families that are dificult for coherent-probe protocols to distinguish. Consider the sample space ${ \mathcal { X } } = S ^ { 1 }$ to be the circle, and let $P _ { 0 } = d \theta / 2 \pi$ be the uniform distribution on the circle. Then, following the proof template D 5, consider the family of functions

$$
h _ { k } ( \alpha ) = 0 . 9 \cos ( k \theta ) \ .\tag{E2}
$$

It follows that $\begin{array} { r } { \int _ { S ^ { 1 } } h _ { k } ( \theta ) d P _ { 0 } ( \theta ) = 0 } \end{array}$ and $\| h _ { k } \| _ { \infty } \leq 1$ . This choice of $h _ { k }$ instantiates physically valid displacement distributions. Let us parameterize the phase-space C in polar coordinates $( a , \theta )$ . Consider the two families of angular probabilty distributions

$$
T _ { \pm } ^ { ( k ) } ( \theta ) = \frac { 1 } { 2 \pi } ( 1 \pm 0 . 9 \cos ( k \theta ) ) \ .\tag{E3}
$$

Then, consider the phase-space displacement distributions

$$
P _ { \pm } ^ { ( k ) } ( \theta ) = ( A e ^ { i \theta } ) _ { \# } ( T _ { \pm } ^ { ( k ) } ( \theta ) ) \ .\tag{E4}
$$

These displacement distributions are the pushforward of $T _ { \pm } ^ { ( k ) } ( \theta )$ under the map $\theta  A e ^ { i \theta }$ for a fixed $A > 0$ . We further impose that for a constant $C > 0 , A ^ { 2 } \leq C k ;$ ; in particular any constant A independent of k will already sufice for this separation. We first check that these are valid probability distributions. Since $| \cos ( k \theta ) | \le 1$ , we have

$$
T _ { \pm } ^ { ( k ) } ( \theta ) = { \frac { 1 } { 2 \pi } } ( 1 \pm 0 . 9 \cos ( k \theta ) ) \geq 0 ~ .\tag{E5}
$$

Moreover, because $k \in \mathbb { Z } ^ { + }$

$$
\int _ { 0 } ^ { 2 \pi } { T _ { \pm } ^ { ( k ) } ( \theta ) d \theta } = \frac { 1 } { 2 \pi } \left[ \theta \pm \frac { 0 . 9 } { k } \sin ( k \theta ) \right] _ { 0 } ^ { 2 \pi }\tag{E6}
$$

$$
= { \frac { 1 } { 2 \pi } } \left( 2 \pi \pm { \frac { 0 . 9 } { k } } \sin ( 2 \pi k ) \right)\tag{E7}
$$

$$
= 1 \ ,\tag{E8}
$$

Thus $T _ { \pm } ^ { ( k ) }$ is a valid probability density on the circle, and its pushforward under $\theta \mapsto A e ^ { i \theta }$ is a valid probability law $P _ { + } ^ { ( k ) }$ on phase-space. Let $P _ { + } ^ { ( k ) } ( \theta ) , P _ { - } ^ { ( k ) } ( \theta )$ induce the displacement channels $\mathcal { E } _ { + } ^ { ( k ) } , \mathcal { E } _ { - } ^ { ( k ) }$ . Physically, these channels correspond to fixed-amplitude signals whose phase distribution contains an angular harmonic of frequency k.

Now we can proceed with applying the semiclassical measurement lemma. Recall that the AFI is defined as

$$
\mathsf { A } _ { M } ( h ; P _ { 0 } ) = \int \left| \frac { d A _ { M , h } } { d Q _ { M , 0 } } ( y ) \right| ^ { 2 } d Q _ { M , 0 } ( y ) \ ,\tag{E9}
$$

where for a coherent-probe protocol with POVM $M ( d y )$ ，

$$
Q _ { M , 0 } ( d y ) = \frac { 1 } { 2 \pi } \int K _ { M } ( d y | \theta ) d \theta , \qquad A _ { M , h } = \frac { 1 } { 2 \pi } \int \cos ( k \theta ) K _ { M } ( d y | \theta ) d \theta \ ,\tag{E10}
$$

and where $K _ { M } ( d y | \theta )$ is the response kernel of $M ( d y )$ . Noting that $\begin{array} { r } { \frac { d A _ { M , h } } { d Q _ { M , 0 } } ( y ) = \mathbb { E } [ \cos ( k \theta ) | Y = y ] \leq 1 } \end{array}$ , it follows that

$$
\mathsf { A } _ { M } ( h ; P _ { 0 } ) \leq \int \left| \frac { d A _ { M , h } } { d Q _ { M , 0 } } ( y ) \right| d Q _ { M , 0 } ( y ) = \int | A _ { M , h } ( d y ) | \ .\tag{E11}
$$

Now consider a coherent-probe protocol in which the coherent state $\left| \beta \right. , \beta \in \mathbb { C }$ is exposed to the signal. Then under displacement $A e ^ { i \theta }$ , the measurement kernel is $K _ { M } ( d y | \theta ) = { \mathrm { T r } } { \big ( } M ( d y ) { \big | } \beta + A e ^ { i \theta } { \big \rangle } { \big \langle } \beta + A e ^ { i \theta } { \big | } { \big ) }$ . Using the definition of $A _ { M , h }$ , we have

$$
{ \sf A } _ { M } ( h ; P _ { 0 } ) \leq \left\| \int \frac { d \theta } { 2 \pi } 0 . 9 \cos ( k \theta ) \left| \beta + A e ^ { i \theta } \right. \left. \beta + A e ^ { i \theta } \right| \right\| _ { 1 }\tag{E12}
$$

Note that the displacements $D ( \beta ) , D ( \beta ) ^ { \dagger }$ factor out to the right and left respectively, and by unitarity do not afect the magnitude of the expression. Rewrite the expression, then as $\mathsf { A } _ { M } ( h ; P _ { 0 } ) \leq \| \Delta \| _ { 1 } / 2$ . Using the Fock basis expansion of coherent states, we have

$$
\Delta = { \frac { 0 . 9 } { \pi } } \int d \theta \ \cos ( k \theta ) ) \left| A e ^ { i \theta } \right. \left. A e ^ { i \theta } \right|\tag{E13}
$$

$$
= \frac { 0 . 9 e ^ { - A ^ { 2 } } } { \pi } \sum _ { n , m \ge 0 } \frac { A ^ { n + m } } { \sqrt { n ! m ! } } \left[ \int d \theta \cos ( k \theta ) e ^ { i ( n - m ) \theta } \right] | n \rangle \langle m |\tag{E14}
$$

$$
= 0 . 9 e ^ { - A ^ { 2 } } \sum _ { m = 0 } ^ { \infty } { \frac { A ^ { 2 m + k } } { \sqrt { m ! ( m + k ) ! } } } ( | m + k \rangle \langle m | + | m \rangle \langle m + k | )\tag{E15}
$$

where in the secondline we use the sifting property

$$
\int d \theta \cos ( k \theta ) e ^ { i ( n - m ) \theta } = { \frac { 1 } { 2 } } \int d \theta \left( e ^ { i ( n - m - k ) } + e ^ { i ( n - m + k ) } \right)\tag{E16}
$$

$$
= \pi ( \delta ( n - m - k ) + \delta ( n - m + k ) ) \ .\tag{E17}
$$

It then follows that

$$
\| \Delta \| _ { 1 } / 2 \le 0 . 9 e ^ { - A ^ { 2 } } \sum _ { m } { \frac { A ^ { 2 m + k } } { \sqrt { m ! ( m + k ) ! } } }\tag{E18}
$$

by triangle inequality. Let $p _ { m } = e ^ { - A ^ { 2 } } A ^ { 2 m } / m !$ so that

$$
\| \Delta \| _ { 1 } / 2 = 0 . 9 \sum _ { m } \sqrt { p _ { m } p _ { m + k } } \ .\tag{E19}
$$

Now note that for any constant $s > 0$ we can rewrite

$$
\sqrt { p _ { m } p _ { m + k } } = e ^ { - s k / 2 } \sqrt { e ^ { - s m } p _ { m } e ^ { s ( m + k ) } p _ { m + k } } .\tag{E20}
$$

Then

$$
\sum _ { m } \sqrt { p _ { m } p _ { m + k } } = e ^ { - s k / 2 } \sum _ { m } \sqrt { e ^ { - s m } p _ { m } e ^ { s ( m + k ) } p _ { m + k } }\tag{E21}
$$

$$
\leq e ^ { - s k / 2 } \left( \sum _ { m } e ^ { - s m } p _ { m } \right) ^ { 1 / 2 } \left( \sum _ { m } e ^ { s ( m + k ) } p _ { m + k } \right) ^ { 1 / 2 }\tag{E22}
$$

where the last step is by Cauchy-Schwarz. At this point note that upon setting $\lambda : = A ^ { 2 } , p _ { m } = e ^ { - \lambda } \lambda ^ { m } / m !$ is precisely the probability of m detections from a Poisson process of expectation λ. We can then rewrite the two multiplicative factors as:

$$
\sum _ { m } e ^ { - s m } p _ { m } = \mathbb { E } _ { m \sim \mathrm { P o i s } ( \lambda ) } [ e ^ { - s m } ]\tag{E23}
$$

$$
\sum _ { m } e ^ { s ( m + k ) } p _ { m + k } \leq \sum _ { m } e ^ { s m } p _ { m } = \mathbb { E } _ { m \sim \operatorname { P o i s } ( \lambda ) } [ e ^ { s m } ]\tag{E24}
$$

For a Poisson distribution,

$$
\mathbb { E } _ { m \sim \mathrm { P o i s } ( \lambda ) } [ e ^ { t m } ] = \sum _ { j = 0 } ^ { \infty } \frac { \exp ( t j ) \lambda ^ { j } } { j ! } \exp ( - \lambda )\tag{E25}
$$

$$
= \exp ( - \lambda ) \sum _ { j } \frac { ( e ^ { t } \lambda ) ^ { j } } { j ! }\tag{E26}
$$

$$
= \exp ( - \lambda ) \exp ( e ^ { t } \lambda )\tag{E27}
$$

$$
= \exp \left( \lambda ( e ^ { t } - 1 ) \right) .\tag{E28}
$$

We therefore have that

$$
\sum _ { m } { \sqrt { p _ { m } p _ { m + k } } } \leq \exp \left( \lambda ( \cosh ( s ) - 1 ) - { \frac { s k } { 2 } } \right) ~ .\tag{E29}
$$

which gives us

$$
\| \Delta \| _ { 1 } / 2 \le 0 . 9 \exp \left( \lambda ( \cosh ( s ) - 1 ) - { \frac { s k } { 2 } } \right) \ .\tag{E30}
$$

Now we choose s to minimize the upper bound. Let the exponent be $\mathbf { \partial } _ { - } F ( s ) ;$ ; then

$$
F ^ { \prime } ( s ) = { \frac { k } { 2 } } - \lambda \sinh ( s )\tag{E31}
$$

At this value,

$$
F ( s _ { * } ) = \frac { k } { 2 } \mathrm { a r s i n h } \left( \frac { k } { 2 \lambda } \right) - \lambda \left( \sqrt { 1 + \frac { k ^ { 2 } } { 4 \lambda ^ { 2 } } } - 1 \right) .\tag{E32}
$$

Letting $\begin{array} { r } { x : = \frac { k } { 2 \lambda } } \end{array}$ , we can rewrite this as

$$
F ( s _ { * } ) = \lambda \left( x \operatorname { a r s i n h } x - { \sqrt { 1 + x ^ { 2 } } } + 1 \right) .\tag{E33}
$$

Define h(x) := x arsinh $x - { \sqrt { 1 + x ^ { 2 } } } + 1$ . Then $h ( 0 ) = 0$ , and

$$
\begin{array} { c } { { h ^ { \prime } ( x ) = \operatorname { a r s i n h } x + \displaystyle \frac { x } { \sqrt { 1 + x ^ { 2 } } } - \displaystyle \frac { x } { \sqrt { 1 + x ^ { 2 } } } } } \\ { { = \operatorname { a r s i n h } x . } } \end{array}\tag{E34}
$$

(E35)

Hence $h ^ { \prime } ( x ) > 0$ for all $x > 0$ , so $h ( x ) > 0$ for all $x > 0$ . Therefore $F ( s _ { * } ) > 0$ for every $k > 0$ , since $x = k / ( 2 \lambda ) > 0$ . Consequently,

$$
\| \Delta \| / 2 \le 0 . 9 \exp ( - F ( s _ { * } ) ) ,\tag{E36}
$$

with a strictly positive $F ( s _ { * } )$ . With this AFI upper bound in hand, we can now apply Lemma 24 to obtain that any algorithm which successfully distinguishes the two hypotheses with probability at least $p > 1 / 2$ requires

$$
T \geq { \frac { ( 2 p - 1 ) } { 0 . 9 } } \exp ( F ( s _ { * } ) ) .\tag{E37}
$$

This is the rigorous lower bound plotted in Figure 3(b). We can see that this is exponential in k as follows. Since $A ^ { 2 } = \lambda \leq C k , { \mathrm { i f ~ } } x = k / ( 2 \lambda )$ , then $x \geq 1 / ( 2 C )$ , and

$$
F ( s _ { * } ) = k { \frac { h ( x ) } { 2 x } } \geq k { \frac { h ( 1 / ( 2 C ) ) } { 2 / ( 2 C ) } } = : c ^ { \prime } k ,\tag{E38}
$$

where $c ^ { \prime }$ is a constant depending only on C. Thus, for any constant success bias, so that $2 p - 1 = \Omega ( 1 )$ , the preceding bound implies

$$
T \geq \frac { 2 p - 1 } { 0 . 9 } \exp ( F ( s _ { * } ) ) \geq \exp ( \Omega ( k ) ) .\tag{E39}
$$

Using Fact 7 yields the desired hardness result for learning the k-th Fourier coeficient with coherent probes.

In Section F we provide quantum sensing protocols enabled by conventional squeezed probes that can learn angular Fourier coeficients using poly(k) queries, this lower bound demonstrates that quantum probes generically enable an exponential expansion of the bandwidth of angular Fourier coeficients and moments accessible to a sensing protocol.

## 2. Hardness of learning with conventional quantum sensing

In this section, we consider the fundamental limitations of conventional quantum sensing protocols using bosonic probes. As discussed in Section C, conventional cavity and interferometric systems use Gaussian probe states and measurements. The precision limits of these sensing platforms are controlled by the maximum mean photon number that can be reliably created in non-classical states. As such, we consider here the class of Gaussian protocols with a total energy budget $E ;$ in units of $\hbar = 1$ , this is equivalent to the mean photon number. The class is defined as follows.

Definition 2 (Conventional finite-energy protocol for bosonic quantum sensing). A conventional protocol for bosonic quantum sensing with access to a signal channel E and with energy constraint E allows:

1. Any Gaussian probe state $\rho _ { S }$

2. Any generaldyne measurement, performed using Gaussian ancilla state $\rho _ { A }$

3. Classical communication (adaptivity) across experiments.

4. In each experiment, it must hold that $\mathrm { T r } ( \rho _ { S A } ( \hat { n } _ { S } + \hat { n } _ { A } ) ) \leq E$ , where $\hat { n } = a ^ { \dag } a$ is the number operator.

The amount of accessible energy is a crucial resource that must be accounted for when studying separations between conventional and quantum-enhanced sensing protocols, so our separations will parameterize the total energy as the instance size k, holding this constant across our upper and lower bounds. We remark that the above class can be substantially more powerful than the kinds of single-mode sensing usually available in practice, because it permits generaldyne measurements that could require a large number of ancilla modes not present in conventional platforms. We nevertheless establish an exponential lower bound against this stronger class, thereby establishing a bound against any realistic Gaussian experimental protocol.

Theorem E.3. Any conventional bosonic sensing protocol as in Definition 2 with per-experiment energy $k \ S \geq \exp ( \Omega ( k ) )$ signal queries to estimate the $k ^ { \mathrm { t h } }$ directional Fourier coeficient of a single-mode signal to constant accuracy. Specifically, for $\eta < 1$ , at least

$$
S \geq \frac { 2 p - 1 } { \eta \left( \exp \left( - \frac { k ^ { 2 } } { 8 k + 5 } \right) + \exp \left( - k ^ { 2 } \right) \right) }\tag{E40}
$$

samples are necessary to estimate the k-th Fourier coeficient $\mathbb { E } [ e ^ { i k x } ]$ to absolute accuracy $\eta / 2$ with success probability at least $p > 1 / 2$

Proof of Theorem E.3. As before, our proof will begin with a definition of an appropriate family of functions to use the semiclassical measurement lemma (Lemma 24). Here, for a more intuitive proof, we will consider a task in which the hypothesis signals difer in their k-th Fourier coeficient along a known phase-space axis, which will be suficient to obtain an exponential lower bound. However, we remark in practice this information is not known in the harder learning setting, and one can obtain exponential lower bounds with larger exponents in that setting.

To begin, let us parameterize our phase-space in real quadrature coordinates, $z = ( x , p ) \in \mathbb { R } ^ { 2 }$ . In this proof, phase space will be our sample space, equipped with the Gaussian measure $P _ { 0 } = \mathcal { N } ( 0 , I _ { 2 } )$ . In these coordinates, the standard vacuum-noise Gaussian distribution is

$$
g ( x , p ) = \frac { 1 } { 2 \pi } \exp \left( - \frac { x ^ { 2 } + p ^ { 2 } } { 2 } \right) .\tag{E41}
$$

For any $k \in \mathbb { Z } ^ { + }$ (where we work with integers for clarity, but positive reals would sufice), define the quantity

$$
c _ { k } = \frac { 1 } { 2 \pi } \int _ { \mathbb { R } ^ { 2 } } e ^ { - ( x ^ { 2 } + p ^ { 2 } ) / 2 } \cos ( k x ) d x d p\tag{E42}
$$

$$
= \frac { 1 } { \sqrt { 2 \pi } } \int _ { \mathbb { R } } e ^ { - x ^ { 2 } / 2 } \cos ( k x ) d x\tag{E43}
$$

$$
= \operatorname { R e } \left[ { \frac { 1 } { \sqrt { 2 \pi } } } \int _ { \mathbb { R } } e ^ { - x ^ { 2 } / 2 + i k x } d x \right]\tag{E44}
$$

$$
= \mathrm { R e } \left[ e ^ { - k ^ { 2 } / 2 } \frac { 1 } { \sqrt { 2 \pi } } \int _ { \mathbb { R } } e ^ { - ( x - i k ) ^ { 2 } / 2 } d x \right]\tag{E45}
$$

$$
= e ^ { - k ^ { 2 } / 2 } \ .\tag{E46}
$$

Then define the QΨ choice of functions

$$
h _ { k } ( x , p ) = \frac { \cos ( k x ) - c _ { k } } { 1 + c _ { k } } \ .\tag{E47}
$$

It follows that $\mathbb { E } _ { P _ { 0 } } [ h _ { k } ] = 0$ and $- 1 \leq h _ { k } \leq 1$ , as required for the semiclassical measurement lemma. With these, we define the displacement distributions

$$
P _ { \pm } ^ { ( k ) } ( x , p ) = g ( x , p ) ( 1 \pm \eta h _ { k } ( x , p ) ) \ .\tag{E48}
$$

Up to the small constant $c _ { k }$ which is present for later convenience, these are highly natural distributions for a Fourier analysis problem: they correspond to raw vacuum noise superposed with a frequency-k modulation. These are also physically valid since the conditions on $h _ { k }$ are satisfied.

Next, let $\varphi _ { C } ( z ) = \mathrm { P D F } ( \mathcal { N } ( 0 , C ) )$ denote the density of a bivariate mean-0 Gaussian with covariance matrix C. By Lemma 5 it follows that conditioned on the displacement $\boldsymbol { z } = ( x , p )$ , the outcome $Y$ of any conventional Gaussian protocol, where the probe state and measurement may be chosen based on classical advice, has a response kernel of the form $K _ { M } ( d y | z ) = \varphi _ { C } ( y - A z - b ) d y$ . Here, x is the first coordinate of $z , A$ is some real 2 by 2 matrix, and b is a real 2-dimensional vector. The allowed $A , b$ will later be constrained by the accessible energy, but any conventional Gaussian protocol with access to $\mathcal { E } _ { \pm }$ must have this general form. Then

$$
Q _ { M , 0 } ( d y ) = \left( \int \varphi _ { C } ( y - A z - b ) g ( z ) d z \right) d y : = q _ { 0 } ( y ) d y ~ ,\tag{E49}
$$

and

$$
A _ { M , h } ( d y ) = \left( \int \frac { \cos ( k x ) - c _ { k } } { 1 + c _ { k } } \varphi _ { C } ( y - A z - b ) g ( z ) d z \right) d y \ .\tag{E50}
$$

Defining

$$
q _ { c } ( w ) = \int d z \ \varphi _ { C } ( w - A z - b ) g ( z ) \cos ( k x ) \ ,\tag{E51}
$$

we have

$$
A _ { M , h } ( d y ) = \frac { q _ { c } ( y ) - c _ { k } q _ { 0 } ( y ) } { 1 + c _ { k } } d y \Longrightarrow \frac { d A _ { M , h } } { d Q _ { M , 0 } } ( y ) = \frac { 1 } { 1 + c _ { k } } \left( \frac { q _ { c } ( y ) } { q _ { 0 } ( y ) } - c _ { k } \right)\tag{E52}
$$

To control this expression, we proceed by obtaining a Bayesian interpretation of $q _ { c }$ . Let Z denote the random variable corresponding to draws from $g ( z )$ . Let W denote the random variable distributed according to $q _ { c } ( w )$ . It follows that

$$
W | ( Z = z ) \sim \mathcal { N } ( A z + b , C ) ~ .\tag{E53}
$$

Hence the conditional density of $W = w$ is

$$
p _ { W | Z } ( w | z ) = \varphi _ { C } ( w - A z - b ) ~ .\tag{E54}
$$

By the definition of conditional probability,

$$
p _ { W , Z } ( w , z ) = p _ { Z } ( z ) p _ { W | Z } ( w | z ) = g ( z ) \varphi _ { C } ( w - A z - b ) ~ .\tag{E55}
$$

By Bayes’ theorem,

$$
p _ { Z | W = w } ( z ) = \frac { p _ { W , Z } ( w , z ) } { p _ { W } ( w ) }\tag{E56}
$$

The denominator is simply obtained by integrating over z in the joint density:

$$
p _ { W } ( w ) = \int g ( z ) \varphi _ { C } ( w - A z - b ) ~ d z = q _ { 0 } ( w ) ~ .\tag{E57}
$$

We thus have a clean expression for the density $p _ { Z | W = w } ( z )$ . Plugging this in to compute a conditional expectation, we have the expression

$$
\mathbb { E } _ { Z } [ \cos ( k x ) | W = w ] = { \frac { 1 } { q _ { 0 } ( w ) } } \int d z \ \varphi _ { C } ( w - A z - b ) g ( z ) \cos ( k x ) = { \frac { q _ { c } ( w ) } { q _ { 0 } ( w ) } } \ .\tag{E58}
$$

Rearranging, we find

$$
q _ { c } ( w ) = q _ { 0 } ( w ) \mathbb { E } _ { Z } [ \cos ( k x ) | W = w ] \mathrm { ~ . ~ }\tag{E59}
$$

This tells us that the AFI is precisely

$$
\mathsf { A } _ { M } ( h ; P _ { 0 } ) = \mathbb { E } _ { Y \sim Q _ { 0 } } \left[ \left( \frac { \mathbb { E } [ \cos ( k X ) | Y ] - c _ { k } } { 1 + c _ { k } } \right) ^ { 2 } \right] \leq \mathbb { E } _ { Y \sim Q _ { 0 } } \left[ \mathbb { E } [ \cos ( k X ) | Y ] ^ { 2 } \right] + c _ { k } ^ { 2 }\tag{E60}
$$

where the inequality uses that $1 + c _ { k } \ge 1$ . Thus to obtain our AFI bound, we need to control the expectation of $\cos ( k x )$ conditioned on the value of $W$ (where X is simply the first coordinate of the random variable $Z )$ . To do this, we must characterize the distribution $Z | W = w$ . Lemma 11 enables precisely this computation; we use its block-matrix notation here.

Note that $( Z , W )$ are jointly Gaussian. We know all the moments:

$$
\mathbb { E } [ Z ] = 0 , \qquad \mathbb { E } [ W ] = b\tag{E61}
$$

and

$$
\Sigma _ { Z Z } = I _ { 2 } , \qquad \Sigma _ { W W } = A A ^ { T } + C\tag{E62}
$$

Then for $\zeta \sim \mathcal { N } ( 0 , C )$ , we have $\Sigma _ { Z W } = \operatorname { C o v } ( Z , A Z + b + \zeta ) = \operatorname { C o v } ( Z , Z ) A ^ { T } = A ^ { T }$ . Using Lemma 11, we then find that the covariance of $Z | W = w$ is

$$
\operatorname { C o v } ( Z \mid W = w ) = I _ { 2 } - A ^ { T } ( A A ^ { T } + C ) ^ { - 1 } A ~ .\tag{E63}
$$

Applying the Woodbury identity,

$$
\Sigma _ { Z | W } = ( I _ { 2 } + A ^ { T } C ^ { - 1 } A ) ^ { - 1 } ~ .\tag{E64}
$$

We’re interested characterizing the first coordinate, which has variance $s ^ { 2 }$ given by

$$
s ^ { 2 } = \left( 1 0 \right) \Sigma _ { Z | W } \left( \ O _ { 0 } ^ { 1 } \right)\tag{E65}
$$

Crucially, $s ^ { 2 }$ is entirely independent of w. Now note that

$$
| \mathbb { E } _ { Z } [ \cos ( k x ) | W = w ] | \leq | \mathbb { E } _ { Z } [ e ^ { i k X } | W = w ] | = \left| \int _ { \mathbb { R } } { \frac { e ^ { i k x } } { \sqrt { 2 \pi s ^ { 2 } } } } \exp \left( - { \frac { ( x - \mu ( w ) ) ^ { 2 } } { 2 s ^ { 2 } } } \right) \right| .\tag{E66}
$$

Shift and rescale the integral to a standard normal with the change of variables $x = \mu ( w ) + s t$ . Then

$$
\mathbb { E } _ { Z } [ e ^ { i k X } | W = w ] = e ^ { i k \mu ( w ) } \mathbb { E } _ { t \sim \mathcal { N } ( 0 , 1 ) } [ e ^ { i k s t } ] = e ^ { i k \mu ( w ) } e ^ { - k ^ { 2 } s ^ { 2 } / 2 }\tag{E67}
$$

Taking absolute magnitudes,

$$
| \mathbb { E } _ { Z } [ \cos ( k x ) | W = w ] | \le e ^ { - k ^ { 2 } s ^ { 2 } / 2 }\tag{E68}
$$

Returning to the inequality (E60), we thus have

$$
\mathsf { A } _ { M } ( h ; P _ { 0 } ) \le e ^ { - k ^ { 2 } s ^ { 2 } } + e ^ { - k ^ { 2 } } \ .\tag{E69}
$$

Now recall Lemma 9, which tells us that

$$
( I _ { 2 } + A ^ { T } C ^ { - 1 } A ) ^ { - 1 } \preceq ( 8 k + 5 ) I _ { 2 } ~ .\tag{E70}
$$

Therefore, $\Sigma _ { Z | W } \succeq ( 8 k + 5 ) ^ { - 1 } I _ { 2 } .$ , so $s ^ { 2 } \geq ( 8 k + 5 ) ^ { - 1 }$ . For $k \geq 1$ , this makes the first term in the TV bound dominate. Together, we have

$$
\mathsf { A } _ { M } ( h ; P _ { 0 } ) \le \exp \left( - \frac { k ^ { 2 } } { 8 k + 5 } \right) + \exp \left( - k ^ { 2 } \right) = \exp ( - O ( k ) ) \ .\tag{E71}
$$

Finally applying Lemma 24, any strategy using

$$
S \leq \exp ( O ( k ) )\tag{E72}
$$

experiments has success probability $1 / 2 + o ( 1 )$ . This establishes that at least

$$
S \geq { \frac { 2 p - 1 } { \eta \left( \exp \left( - { \frac { k ^ { 2 } } { 8 k + 5 } } \right) + \exp \left( - k ^ { 2 } \right) \right) } } = \exp ( \Omega ( k ) )\tag{E73}
$$

samples are necessary to solve the distinguishing task with constant success probability $p > 1 / 2 .$ . Applying Fact 7, we lift to the desired sample lower bound for Fourier learning. □

## 3. Hardness of learning time-dependent signals without memory

The previous section established an exponential lower bound for static Fourier coeficients with conventional finite-energy sensing. As we will see in Section $\mathrm { F } ,$ the single-qubit-enhanced protocol which estimates these coeficients eficiently given the same energy uses the control capability of the qubit to generate non-Gaussian probe states. However, coupling a qubit to the sensor mode adds another meaningful resource: quantum memory. A qubit can be retained as a memory state while the sensor mode undergoes multiple evolutions under a classical signal. Here, we focus on a separation which explicitly uses the ability of the qubit to act as a memory. The core point is that this capability can add exponential sensing power to a conventional bosonic sensing device when signals are time-dependent.

A time-dependent signal is obtained when the generating Hamiltonian has time-varying coeficients $f _ { x } ( t ) , f _ { p } ( t )$ Physically, these are stochastic processes that are continuous functions of time. However, any physical experiment that performs destructive quantum measurements accesses the induced displacement channel at a discrete set of times, which we call $\tau .$ As such, a time-dependent signal is operationally characterized by a probability distribution $P$ over the stochastic process $\{ Z _ { t } \} _ { t \in \mathcal { T } }$ , where $Z _ { t } = ( X _ { t } , P _ { t } ) \in \mathbb { R } ^ { 2 }$ is the random variable corresponding to the induced displacement at time t. In the static setting, each $Z _ { t }$ is independent and identical, whereas the time-varying case allows for diferent, temporally-correlated displacement distributions. Keeping with the Fourier analysis setting, we work with the following object in the time-varying case; Ω denotes the usual symplectic form.

Definition 4 (Temporal Fourier coeficient). For times $\mathbf { t } ~ = ~ ( t _ { 1 } , \ldots , t _ { m } )$ and phase-space frequencies $\zeta =$ $\left( \zeta _ { 1 } , \ldots , \zeta _ { m } \right)$ , define the $( \mathbf { t } , \pmb { \zeta } )$ -temporal Fourier coeficient is

$$
\widehat { P } ( \mathbf { t } , \zeta ) = \mathbb { E } _ { P } \left[ \exp \left( i \sum _ { j = 1 } ^ { m } \Omega ( \zeta _ { j } , Z _ { t _ { j } } ) \right) \right] \ .\tag{E74}
$$

This definition reduces to the usual static Fourier coeficient when $m = 1$ . Moreover, as we detail in Section $\mathrm { ~ F 3 , }$ a wide array of time-dependent properties can be extracted from temporal Fourier coeficients by taking derivatives at the origin. First, we will consider estimation of the two-point temporal Fourier correlation. In particular, we restrict to coeficients of the form

$$
\Theta _ { k } ( t _ { 0 } , t _ { 1 } ) = \mathbb { E } _ { P } \left[ e ^ { i k ( X _ { t _ { 0 } } - X _ { t _ { 1 } } ) } \right] ~ .\tag{E75}
$$

In symplectic notation this is $\widehat { P } ( ( t _ { 0 } , t _ { 1 } ) , ( \zeta _ { k } , - \zeta _ { k } ) )$ , where $\zeta _ { k }$ is chosen so that $\Omega ( \zeta _ { k } , Z _ { t } ) = k X _ { t }$

## a. Time-varying hypothesis testing for frequency-k, two-point correlation

We now describe the hypothesis-testing instance which realizes a memory-based advantage in sensing with a single qubit. Once again let

$$
g ( x , p ) = { \frac { 1 } { 2 \pi } } \exp \left( - { \frac { x ^ { 2 } + p ^ { 2 } } { 2 } } \right)\tag{E76}
$$

and define $c _ { k } = e ^ { - k ^ { 2 } / 2 }$ . For any $\theta \in [ 0 , 2 \pi )$ , define

$$
h _ { k , \theta } ( x , p ) = \cos ( k x + \theta ) - c _ { k } \cos ( \theta ) \ .\tag{E77}
$$

The subtraction term simply ensures that $\mathbb { E } _ { q } [ h _ { k , \theta } ( X , P ) ] = 0$ . Fix a constant $\eta \leq 1 / 4$ . In each realization of the time-dependent signal, draw a latent phase $\Phi \sim \mathrm { U n i f } [ 0 , 2 \pi )$ . Let $\sigma = \pm 1$ be a fixed sign, and let the time-t displacement distribution conditioned on Φ correspond to

$$
P _ { \sigma , \Phi , t } ^ { ( k ) } ( x , p ) = g ( x , p ) \left( 1 + \eta h _ { k , \Phi + \sigma \Omega _ { 0 } t } ( x , p ) \right) \ .\tag{E78}
$$

Here $\Omega _ { 0 }$ is a known drift rate. The two hypotheses correspond to $\sigma = \pm 1$ . Physically, this is a simple model: a k-th order Fourier feature drifts in phase-space over time, and the direction of the drift is unknown. Upon querying the signal, one does not know the phase-space direction along which the feature resides, akin to how the signal phase is generally unknown in AC sensing, so it is assumed to be drawn from a uniform prior, after which point subsequent queries see a constant-rate drift from the initial point.

Since $| h _ { k , \theta } | \leq 2$ , these are valid densities for $\eta \leq 1 / 4$ . Moreover, for every fixed time $t ,$

$$
\int _ { 0 } ^ { 2 \pi } { \frac { d \Phi } { 2 \pi } } { \cal P } _ { \sigma , \Phi , t } ^ { ( k ) } ( x , p ) = g ( x , p ) \ .\tag{E79}
$$

Thus, every one-time displacement channel is identical under $\sigma = + 1$ and $\sigma = - 1$ . The class label is only present in multi-time Fourier coeficients.

Choose two times $t _ { 0 } , t _ { 1 }$ such that $\Omega _ { 0 } ( t _ { 1 } - t _ { 0 } ) = \pi / 2$ . We now compute the separation in the two-point coeficient. For any θ, define

$$
m _ { k } ( \theta ) = \mathbb { E } _ { ( X , P ) \sim g ( 1 + \eta h _ { k , \theta } ) } \left[ e ^ { i k X } \right] \ .\tag{E80}
$$

A direct Gaussian integral gives

$$
m _ { k } ( \theta ) = c _ { k } + \eta \left( a _ { k } e ^ { - i \theta } + b _ { k } e ^ { i \theta } \right)\tag{E81}
$$

where

$$
a _ { k } = { \frac { 1 - e ^ { - k ^ { 2 } } } { 2 } } , \qquad b _ { k } = { \frac { e ^ { - 2 k ^ { 2 } } - e ^ { - k ^ { 2 } } } { 2 } } \ .\tag{E82}
$$

Therefore

$$
\begin{array} { r l } & { \Theta _ { k } ^ { ( \sigma ) } ( t _ { 0 } , t _ { 1 } ) = \mathbb { E } _ { \Phi } \left[ m _ { k } ( \Phi + \sigma \Omega _ { 0 } t _ { 0 } ) \overline { { m _ { k } ( \Phi + \sigma \Omega _ { 0 } t _ { 1 } ) } } \right] } \\ & { \qquad = c _ { k } ^ { 2 } + \eta ^ { 2 } \left( a _ { k } ^ { 2 } e ^ { i \sigma \Omega _ { 0 } ( t _ { 1 } - t _ { 0 } ) } + b _ { k } ^ { 2 } e ^ { - i \sigma \Omega _ { 0 } ( t _ { 1 } - t _ { 0 } ) } \right) \ . } \end{array}\tag{E83}
$$

(E84)

Taking imaginary parts and using $\Omega _ { 0 } ( t _ { 1 } - t _ { 0 } ) = \pi / 2$ , we obtain

$$
\mathrm { I m } \Theta _ { k } ^ { ( \sigma ) } ( t _ { 0 } , t _ { 1 } ) = \sigma \eta ^ { 2 } ( a _ { k } ^ { 2 } - b _ { k } ^ { 2 } ) \ .\tag{E85}
$$

Finally,

$$
a _ { k } ^ { 2 } - b _ { k } ^ { 2 } = \frac { ( 1 - e ^ { - 2 k ^ { 2 } } ) ( 1 - e ^ { - k ^ { 2 } } ) ^ { 2 } } { 4 } \ .\tag{E86}
$$

Thus, for $\textstyle k \geq 1 , a _ { k } ^ { 2 } - b _ { k } ^ { 2 } \geq { \frac { 1 } { 1 6 } }$ . The two hypotheses therefore difer by a constant amount in the temporal Fourier coeficient Im $\Theta _ { k } \big ( t _ { 0 } , t _ { 1 } \big )$ . As such, any protocol which can learn $\Theta _ { k } \big ( t _ { 0 } , t _ { 1 } \big )$ for a time-varying signal to acccuracy at least $\epsilon = \eta ^ { 2 } / 3 2$ can be used to distinguish the two hypotheses, so the temporal Fourier analysis truly reduces.

Theorem E.5 (Hardness of temporal Fourier learning without memory). For any integer $k \geq 1$ , any conventional Gaussian protocol with energy k per experiment requires at least

$$
S \geq \frac { c _ { p } } { \eta \left( \exp \left( - \frac { k ^ { 2 } } { 1 6 k + 1 0 } \right) + \exp \left( - \frac { k ^ { 2 } } { 2 } \right) \right) } = \exp ( \Omega ( k ) )\tag{E87}
$$

signal queries to successfully distinguish the two drift directions with success probability at least $p > 1 / 2$ , where

$c _ { p } > 0$ depends only on p.

Proof. We first recall the one-experiment response bound from the proof of Theorem E.3. Fix any destructive Gaussian experiment with energy at most $k ,$ chosen using arbitrary previous classical advice. By Lemma $5 ,$ the outcome distribution conditioned on a deterministic displacement $z = \left( x , p \right)$ has the form $W \sim \mathcal { N } ( A z + b , C )$ Define

$$
q _ { 0 } ( w ) = \int d z ~ \varphi _ { C } ( w - A z - b ) g ( z )\tag{E88}
$$

and

$$
r _ { \theta } ( w ) = \int d z \ \varphi _ { C } ( w - A z - b ) g ( z ) h _ { k , \theta } ( z ) \ .\tag{E89}
$$

The same conditional-Gaussian calculation used in Theorem E.3, with $\cos ( k x + \theta )$ in place of cos $( k x )$ , gives the uniform bound $\| r _ { \theta } \| _ { 1 } \le \varepsilon _ { k }$ for every θ, where

$$
\varepsilon _ { k } = \exp \left( - \frac { k ^ { 2 } } { 1 6 k + 1 0 } \right) + \exp \left( - \frac { k ^ { 2 } } { 2 } \right) .\tag{E90}
$$

Now consider an arbitrary adaptive conventional protocol making S signal queries to the time-dependent signal. Before query $\ell ,$ the protocol has a classical transcript $w _ { < \ell } .$ , chooses a time $t _ { \ell } ,$ and chooses a Gaussian experiment with energy at most $k .$ Conditional on the phase Φ and hypothesis $\sigma _ { : }$ , the conditional density of the next outcome w can be written as

$$
q _ { \ell , 0 } ( w _ { \ell } ) + \eta r _ { \ell , \theta _ { \sigma } ( t ) } ( w _ { \ell } ) ~ ,\tag{E91}
$$

where $\theta _ { \sigma } ( t ) = \Phi + \sigma \Omega _ { 0 } t _ { \ell }$ . Hence, the full transcript density under hypothesis $\sigma$ is

$$
Q _ { \sigma } ( w _ { 1 } , \dots , w _ { S } ) = \int _ { 0 } ^ { 2 \pi } \frac { d \Phi } { 2 \pi } \prod _ { \ell = 1 } ^ { S } \left( q _ { \ell , 0 } ( w _ { \ell } ) + \eta r _ { \ell , \theta _ { \sigma } ( t ) } ( w _ { \ell } ) ( w _ { \ell } ) \right) ~ .\tag{E92}
$$

Expanding this product, the $q _ { \ell , 0 } ^ { S }$ term is independent of σ. All terms with exactly one factor of $r _ { \ell , \theta _ { \sigma } ( t ) }$ vanish after averaging over $\Phi _ { i }$ , since $\begin{array} { r } { \int _ { 0 } ^ { 2 \pi } h _ { k , \Phi + \theta } ( x , p ) d \Phi = 0 } \end{array}$ pointwise. Thus the diference $Q _ { + } - Q _ { - }$ only contains terms with at least two response factors.

Taking the $L _ { 1 }$ norm and integrating sequentially over the transcript variables $w _ { 1 } . . . w _ { S } ,$ , each response factor contributes at most $\varepsilon _ { k } .$ , while each $q _ { \ell , 0 }$ density integrates to one. Therefore, whenever $S \eta \varepsilon _ { k } \le 1$

$$
\begin{array} { l } { \displaystyle \| Q _ { + } - Q _ { - } \| _ { 1 } \leq 2 \sum _ { r = 2 } ^ { S } \binom { S } { r } ( \eta \varepsilon _ { k } ) ^ { r } } \\ { \leq 4 S ^ { 2 } \eta ^ { 2 } \varepsilon _ { k } ^ { 2 } . } \end{array}\tag{E93}
$$

(E94)

As such we obtain $d _ { \mathrm { T V } } ( Q _ { + } , Q _ { - } ) \leq 2 S ^ { 2 } \eta ^ { 2 } \varepsilon _ { k } ^ { 2 }$ . If $S \eta \varepsilon _ { k }$ is not bounded by a suficiently small constant, then the desired exponential lower bound is already true. Thus, Le Cam’s two-point method implies that any protocol succeeding with probability at least $p > 1 / 2$ must satisfy $\begin{array} { r } { S \ge \frac { c _ { p } } { \eta \varepsilon _ { k } } } \end{array}$ for a constant $c _ { p } > 0$ depending only on $p .$ Substituting the expression for $\varepsilon _ { k }$ gives the stated bound. □

This result establishes an exponential lower bound for two-point temporal Fourier coeficients, where the bound is exponential in the Fourier frequency. One can also instantiate a separation that is exponential in the temporal order of the correlation itself; we consider this next.

## b. Constant-frequency, m-point correlation estimation

We now give a second time-dependent instance which realizes a diferent memory-based separation. The previous example was exponential in the Fourier frequency of a two-point temporal coeficient. Here, the local Fourier frequency is fixed once and for all, and the exponential hardness comes from the order of the temporal correlation itself. To avoid overloading notation, let m denote the number of time points in this construction.

The lower bound below applies to a broader memoryless class than the Gaussian class used above. We allow arbitrary probe states, arbitrary ancillas, and arbitrary destructive measurements, but we impose a constant energy constraint on the signal-coupled mode in each destructive experiment. The only forbidden operation is the retention of a quantum system from one time point to the next. Thus the protocol may use arbitrary classical adaptivity, but it has no coherent memory.

Before proceeding with the bound, we make an important conceptual remark. Our proof here will utilize a parity-based construction inspired by Ref. [55]; however, the context in which we do so is substantially diferent. In Ref. [55], the parity-based construction was presented as an example that without careful accounting of energetic budgets, one could prove quantum separations that are not physically meaningful to real experiments, in part because the parity-based hypothesis test hid an unphysical correlation in a high moment of a classical signal that could be accessed with higher-energy conventional probes. Our coherent-probe bounds should be viewed as a more physically meaningful form of the idea that quantum probes can outperform classical ones. Moreover, our next lower bound will use the parity construction to realize a lower bound in a much more physically meaningful context: by bounding against protocols that can even use arbitrary quantum control resources in each measurement while carefully tracking energy budgets, this result isolates that quantum memory can be a truly indispensable sensing resource for time-varying signals. As such, in this case, realizations of an exponential lower bound that are more physically well-motivated than hidden parity correlation are likely possible, but our bound is a proof-of-concept of the operating principle that constant-sized quantum memory can yield exponential quantum advantages in learning classical signals.

Let M denote any protocol for temporal Fourier analysis. We use the convention that a single query corresponds to a pass of the signal for all times in t.

Theorem E.6 (Hardness of constant-frequency m-point temporal correlations without memory). Fix constants $0 < \epsilon , \delta < 1 / 2$ and $E _ { 0 } > 0$ . For every m $\geq 1$ and any $E _ { 0 }$ there is a constant $C ( E _ { 0 } )$ such that any memoryless protocol M with energy $E _ { 0 }$ per experiment requires at least

$$
S \geq 2 ^ { m } \frac { 1 - 2 \delta } { \epsilon }\tag{E95}
$$

queries to the signal to learn an m-point temporal Fourier coeficient of frequency $C$ at each point in tim $^ { , e , }$ to absolute precision ϵ and with success probability at least $1 - \delta$ . In particular, this is $\exp ( \Omega ( m ) )$ for any constant $\epsilon , \delta$

Proof. We begin by defining the appropriate hypothesis testing instance. Fix a constant energy bound $E _ { 0 }$ and let $\nu = 1 / 2$ , and define

$$
x _ { \star } = \frac { \nu } { \sqrt { 2 E _ { 0 } + 1 } } \ , \qquad \lambda _ { \star } = \frac { \pi } { x _ { \star } } .\tag{E96}
$$

Then let $\zeta _ { * }$ be the constant phase-space frequency such that $\Omega ( \zeta _ { \star } , ( x , p ) ) = \lambda _ { \star } x .$ , independent of $m$ . Let $t _ { 1 } , \ldots , t _ { m }$ be m designated sensing times. The m-point temporal Fourier coeficient we will consider is

$$
\Xi _ { m } ( P ) = { \widehat { P } } ( ( t _ { 1 } , \dots , t _ { m } ) , ( \zeta _ { \star } , \dots , \zeta _ { \star } ) ) = \mathbb { E } _ { P } \left[ \exp \left( i \lambda _ { \star } \sum _ { j = 1 } ^ { m } X _ { t _ { j } } \right) \right] ~ .\tag{E97}
$$

The hypothesis testing problem is as follows. In each realization of the time-dependent signal, draw a bit string $B = ( B _ { 1 } , \ldots , B _ { m } ) \in \{ 0 , 1 \} ^ { m }$ . Under the hypothesis $\sigma \in \{ + 1 , - 1 \}$ , the bit string has distribution

$$
{ \underset { \sigma } { \operatorname* { P r } } } ( B = b ) = 2 ^ { - m } \left( 1 + \sigma \epsilon ( - 1 ) ^ { b _ { 1 } + \cdots + b _ { m } } \right)\tag{E98}
$$

where $\gamma \in \mathsf { \Gamma } ( 0 , 1 ]$ is a fixed constant. Conditioned on the bit string, the signal displacement at time $t _ { j }$ is $Z _ { t _ { j } } = B _ { j } ( x _ { \star } , 0 )$ . The distributions above are valid since

$$
\sum _ { b \in \{ 0 , 1 \} ^ { m } } ( - 1 ) ^ { b _ { 1 } + \cdots + b _ { m } } = 0\tag{E99}
$$

and $0 \leq \epsilon \leq 1$ . For this instance,

$$
\Xi _ { m } ( P _ { \sigma } ) = \mathbb { E } _ { \sigma } \left[ \exp \left( i \lambda _ { \star } x _ { \star } \sum _ { j = 1 } ^ { m } B _ { j } \right) \right]\tag{E100}
$$

$$
= \mathbb { E } _ { \sigma } \left[ ( - 1 ) ^ { B _ { 1 } + \cdots + B _ { m } } \right]\tag{E101}
$$

(E102)

Therefore the two hypotheses difer by $2 \epsilon$ in an m-point temporal Fourier coeficient; any protocol which can learn an m-point temporal Fourier coeficient (with precision parameters $\epsilon , \delta )$ can solve the hypothesis testing problem with success probability at least $1 - \delta .$ As such a lower bound on the query complexity of this hypothesis testing instance immediately lifts to the main claim. We say that under $P _ { + }$ , the above m-point temporal Fourier coeficient has expectation $+ \epsilon .$ , while under $P _ { - } .$ it has expectation −ϵ. We now prove that any memoryless finite-energy protocol obtains only exponentially small total variation distance from one realization of the m-time process. Consider one destructive signal query at a fixed time $t _ { j }$ , and condition on an arbitrary previous classical transcript. The protocol may choose any joint probe-ancilla state $\rho ,$ with mean photon number at most $E _ { 0 }$ in the sensor mode, and may perform any POVM after the signal displacement. If $B _ { j } = 0 _ { \mathrm { : } }$ , the post-signal state is $\rho .$ . If $B _ { j } = 1$ , the post-signal state is

$$
\rho _ { 1 } = D ( ( x _ { \star } , 0 ) ) \rho D ( ( x _ { \star } , 0 ) ) ^ { \dagger } \ .\tag{E103}
$$

Let $K _ { 0 }$ and $K _ { 1 }$ denote the resulting classical outcome distributions under $B _ { j } = 0$ and $B _ { j } = 1 . \mathrm { \ B y }$ Lemma 17,

$$
d _ { \mathrm { T V } } ( K _ { 0 } , K _ { 1 } ) \leq \frac { 1 } { 2 } \| \rho - D ( ( x _ { \star } , 0 ) ) \rho D ( ( x _ { \star } , 0 ) ) ^ { \dagger } \| _ { 1 } \ .\tag{E104}
$$

We now bound the right-hand side using the energy constraint. Since $D ( ( x _ { \star } , 0 ) ) = \exp ( - i x _ { \star } \hat { p } )$ , for any pure state $| \psi \rangle$

$$
\frac { 1 } { 2 } \left\| | \psi \rangle \langle \psi | - e ^ { - i x _ { \star } \hat { p } } | \psi \rangle \langle \psi | e ^ { i x _ { \star } \hat { p } } \right\| _ { 1 } = \sqrt { 1 - | \langle \psi | e ^ { - i x _ { \star } \hat { p } } | \psi \rangle | ^ { 2 } }
$$

$$
\leq \left. ( I - e ^ { - i x _ { \star } \hat { p } } ) \left. \psi \right. \right.\tag{E105}
$$

(E106)

$$
\leq x _ { \star } \sqrt { \left. \psi \right| \hat { p } ^ { 2 } \left| \psi \right. } \mathrm { ~ . ~ }\tag{E107}
$$

The last inequality follows from the spectral theorem applied to ${ \hat { p } } .$ For a mixed state, we can apply this bound to any pure-state decomposition of $\rho$ and use convexity of the trace norm. Thus

$$
\frac { 1 } { 2 } \| \rho - D ( ( x _ { \star } , 0 ) ) \rho D ( ( x _ { \star } , 0 ) ) ^ { \dagger } \| _ { 1 } \leq x _ { \star } \sqrt { \mathrm { T r } ( \rho \hat { p } ^ { 2 } ) } \ .\tag{E108}
$$

The energy constraint is $\mathrm { T r } ( \rho \hat { n } ) \leq E _ { 0 }$ , and

$$
\mathrm { T r } ( \rho \hat { n } ) = \frac { 1 } { 2 } \mathrm { T r } \big ( \rho ( \hat { x } ^ { 2 } + \hat { p } ^ { 2 } - 1 ) \big ) \ .\tag{E109}
$$

As such we have

$$
\mathrm { T r } \left( \rho \hat { p } ^ { 2 } \right) \leq 2 \mathrm { T r } ( \rho \hat { n } ) + 1 \leq 2 E _ { 0 } + 1 \ .\tag{E110}
$$

Therefore

$$
d _ { \mathrm { T V } } ( K _ { 0 } , K _ { 1 } ) \leq x _ { \star } \sqrt { 2 E _ { 0 } + 1 } = \nu ~ .\tag{E111}
$$

This bound holds for every possible previous transcript and every adaptively chosen state and measurement. Now consider one full realization of the m-time signal. Let $Y _ { j }$ denote the classical outcome obtained at time $t _ { j } ,$ , and let $Y _ { < j } = ( Y _ { 1 } , \dots , Y _ { j - 1 } )$ . Conditioned on a history $y _ { < j } ,$ the protocol chooses a particular measurement protocol which results in outcome $Y _ { j }$ . Since the only input parameter of measurement $j$ is the sign $B _ { j }$ , an adaptive experiment is a classical map from $B _ { j }$ to the distribution of $Y _ { j }$ conditioned on $y _ { < j } \colon$

$$
K _ { b } ^ { y _ { < j } } ( d y _ { j } ) = \operatorname* { P r } ( Y _ { j } \in d y _ { j } | B _ { j } = b , { \mathrm { p r e v i o u s ~ o u t c o m e s ~ } } y _ { < j } )\tag{E112}
$$

Said another way, $K _ { b } ^ { y < j }$ is the density of outcome $Y _ { j }$ conditioned on the last $j - 1$ classical outcomes and the unknown bit $B _ { j }$ being b. Next, define the average and diference signed kernels

$$
A _ { j } ^ { y < j } = \frac { 1 } { 2 } \left( K _ { 0 } ^ { y < j } + K _ { 1 } ^ { y < j } \right) , \qquad \Delta _ { j } ^ { y < j } = \frac { 1 } { 2 } \left( K _ { 0 } ^ { y < j } - K _ { 1 } ^ { y < j } \right) \ .\tag{E113}
$$

These allow us to rewrite $K _ { b } ^ { y _ { < j } } = A _ { j } ^ { y _ { < j } } + ( - 1 ) ^ { b } \Delta _ { j } ^ { y _ { < j } } , A _ { j } ^ { y _ { < j } }$ , so that the diference in outcome densities from any step in an adaptive experiment under $B _ { j } = + 1 ~ \mathrm { v s . } ~ - 1$ is entirely encoded in $\Delta _ { j } ^ { y < j }$ . Then we already have the following $L _ { \mathrm { 1 } } \mathrm { - b o u n d }$ , because the $L _ { 1 }$ norm of $\Delta _ { j } ^ { y _ { < j } }$ is exactly the total variation between $K _ { 0 }$ and $K _ { 1 }$ conditioned on a history:

$$
\| \Delta _ { j } ^ { y _ { < j } } \| _ { 1 } = d _ { \mathrm { T V } } ( K _ { 0 } ^ { y _ { < j } } , K _ { 1 } ^ { y _ { < j } } ) \le \nu\tag{E114}
$$

for every $j$ and every history $y _ { < j }$ . Now let $Q _ { \sigma }$ be the distribution of the full transcript $Y = ( Y _ { 1 } , \dots , Y _ { m } )$ obtained from one realization of the signal under hypothesis $\sigma .$ For a fixed bitstring $b _ { 1 } b _ { 2 } . . . b _ { m }$ , the outcome density is $\Pi _ { j = 1 } ^ { m } K _ { b _ { j } } ^ { y < j }$ . Averaging over randomness in the bitstring by the law of total probability,

$$
Q _ { \sigma } ( d y _ { 1 } , \dots , d y _ { m } ) = 2 ^ { - m } \sum _ { b \in \{ 0 , 1 \} ^ { m } } \bigl ( 1 + \sigma \epsilon ( - 1 ) ^ { b _ { 1 } + \dots + b _ { m } } \bigr ) \prod _ { j = 1 } ^ { m } K _ { j , b _ { j } } ^ { y < j } ( d y _ { j } ) \ .\tag{E115}
$$

Now we consider the composite densities under hypothesis $\sigma = + 1 \mathrm { { \ v s . \ - 1 } }$ . Then we have

$$
Q _ { + } - Q _ { - } = 2 \epsilon \cdot 2 ^ { - m } \sum _ { b \in \{ 0 , 1 \} ^ { m } } ( - 1 ) ^ { b _ { 1 } + \cdots + b _ { m } } \prod _ { j = 1 } ^ { m } K _ { b _ { j } } ^ { y _ { < j } } ~ .\tag{E116}
$$

Substituting $K _ { b _ { j } } ^ { y _ { < j } } = A _ { j } ^ { y _ { < j } } + ( - 1 ) ^ { b _ { j } } \Delta _ { j } ^ { y _ { < j } }$ ，

$$
\sum _ { b \in \{ 0 , 1 \} ^ { m } } ( - 1 ) ^ { b _ { 1 } + \cdots + b _ { m } } \prod _ { j = 1 } ^ { m } K _ { b _ { j } } ^ { y _ { < j } } = \sum _ { b \in \{ 0 , 1 \} ^ { m - 1 } } ( - 1 ) ^ { b _ { 1 } + \cdots + b _ { m - 1 } } [ ( A _ { m } ^ { y _ { < m } } + \Delta _ { m } ^ { y _ { < m } } ) - ( A _ { m } ^ { y _ { < m } } - \Delta _ { m } ^ { y _ { < m } } ) ] \prod _ { j = 1 } ^ { m - 1 } K _ { b _ { j } } ^ { y _ { < j } }\tag{E117}
$$

$$
= 2 \Delta _ { m } ^ { < y _ { m } } \sum _ { b \in \{ 0 , 1 \} ^ { m - 1 } } ( - 1 ) ^ { b _ { 1 } + \dots + b _ { m - 1 } } \prod _ { j = 1 } ^ { m - 1 } K _ { b _ { j } } ^ { y _ { < j } }\tag{E118}
$$

$$
\mathrm { r e p e a t i n g , } \ = 2 ^ { m } \Delta _ { m } ^ { < y _ { m } } \Delta _ { m - 1 } ^ { < y _ { m = 1 } } \cdot \cdot \cdot \Delta _ { 1 }\tag{E119}
$$

This leaves us with

$$
Q _ { + } - Q _ { - } = 2 \epsilon \Delta _ { 1 } ( d y _ { 1 } ) \Delta _ { 2 } ^ { y _ { 1 } } ( d y _ { 2 } ) \cdot \cdot \cdot \Delta _ { m } ^ { y _ { < m } } ( d y _ { m } ) \ .\tag{E120}
$$

Taking total variation and integrating sequentially,

$$
\begin{array} { l } { \displaystyle \| Q _ { + } - Q _ { - } \| _ { 1 } \leq 2 \epsilon \int | \Delta _ { 1 } | ( d y _ { 1 } ) \int | \Delta _ { 2 } ^ { y _ { 1 } } | ( d y _ { 2 } ) \cdots \int | \Delta _ { m } ^ { y _ { < m } } | ( d y _ { m } ) } \\ { \leq 2 \epsilon \nu ^ { m } ~ . } \end{array}\tag{E121}
$$

(E122)

Therefore one realization of the m-time signal yields $d _ { \mathrm { T V } } ( Q _ { + } , Q _ { - } ) \leq \epsilon \nu ^ { m }$

The same bound holds conditionally on any previous realization transcripts, since the one-time response bound was uniform over all classical advice. Applying the adaptive hybrid argument of Lemma 20 over $S$ independent realizations gives

$$
d _ { \mathrm { T V } } ( Q _ { + } ^ { ( S ) } , Q _ { - } ^ { ( S ) } ) \leq S \epsilon \nu ^ { m } ~ .\tag{E123}
$$

By Lemma 22, any memoryless protocol using $S$ realizations succeeds in distinguishing the two hypotheses with probability at most $\begin{array} { r } { \frac { 1 } { 2 } + \frac { S \gamma \nu ^ { m } } { 2 } } \end{array}$ . Therefore any protocol with success probability at least $p > 1 / 2$ must satisfy

$$
S \geq \frac { 2 p - 1 } { \epsilon \nu ^ { m } } .\tag{E124}
$$

Substituting $\nu = 1 / 2$ and $\delta = 1 - p$ and lifting to a learning lower bound proves the claim.

This theorem emphasizes that coherent memory-enabled quantum processing of quantum data can exponentially outperform measurement and classical postprocessing when the target is estimating temporal correlations, even if the frequency of those correlations is small. The corresponding upper bound using single-qubit memory is simple and will be given in Section F.

Remark 7. The two mechanisms for memory-based separations are independent. Conventional protocols can struggle to estimate high-frequency temporal Fourier coeficients for the same reason that they struggle to estimate static Fourier coeficients: high-frequency response is exponentially small with Gaussian protocols, whether it lives in a single or multiple time-slices. Memoryless protocols, on the other hand, struggle to measure multipoint temporal correlations because classical postprocessing can exponentially underperform coherent quantum processing before destructive readout, independent of the frequency. In principle, estimating a frequency-k, mpoint temporal Fourier coeficient results in the above bounds stacking, yielding an asymptotic lower bound of exp(Ω(mk)) for conventional sensing protocols.

## 4. Lower bound for infinite-energy conventional quantum sensing

The previous sections established exponential energy-aware lower bounds for static and time-varying Fourier analysis tasks. A natural question is whether provable separations for single-mode sensing persist even if we remove all energy constraints and consider arbitrarily powerful Gaussian measurements. Conceptually, this would illustrate that the non-Gaussian sensing capabilities that become cheap upon obtaining single-qubit control can confer quantum advantages that cannot be replicated by any, arbitrarily high-quality conventiona experiment.

In this section, we describe a task for which such a separation is possible. In particular, this task is concerned with estimating the variance of a single-mode quantum state. This is physically natural, as the broader problem of learning observables of Gaussian states is precisely what one is faced with upon sensing classical fields with Gaussian probes. We prove a quadratic lower bound for variance estimation with unconstrained Gaussian measurements. In Section F 4 we will then demonstrate a non-Gaussian strategy using a constant-depth qubitenabled Ramsey protocol to estimate the same variance using linear samples, exhibiting a quadratic advantage.

This lower bound applies directly to the task of estimating the phase-space variance of a quantum state.

Definition 8 (Variance estimation). Let $\hat { R } = ( \hat { x } , \hat { p } ) ^ { T }$ denote the quadrature vector of a single-mode quantum state $\rho ,$ and let

$$
V _ { i j } ( \rho ) = \frac 1 2 \operatorname { T r } \Bigl ( \rho \{ \hat { R } _ { i } - \langle \hat { R } _ { i } \rangle _ { \rho } , \hat { R } _ { j } - \langle \hat { R } _ { j } \rangle _ { \rho } \} \Bigr )\tag{E125}
$$

denote its covariance matrix. The variance of $\rho$ is

$$
v ( \rho ) = \frac { 1 } { 2 } \operatorname { T r } V ( \rho ) = \frac { 1 } { 2 } \left( \operatorname { V a r } _ { \rho } ( \hat { x } ) + \operatorname { V a r } _ { \rho } ( \hat { p } ) \right) ~ .\tag{E126}
$$

The k, δ-variance estimation task is to estimate $v ( \rho )$ for any single-mode state to within absolute error $1 / ( 2 k )$ with success probability at least $1 - \delta$

Variance estimation is often a core task in sensing stochastic signals; for instance, an important task for the detection of axionic dark matter is to estimate small, narrowband power increases in a broad frequency band, and the signal encountered in each frequency bin is very well-modeled by a Gaussian displacement channel [72]. A signal detection amounts to accurate certification of a slight broadening in one particular bin. As such, the quadratic separation we present here applies directly to this setting and provides rigorous evidence for the large empirical speedups observed in [80, 81] using non-Gaussian photon-counting measurements.

We prove the following.

Theorem E.9. Any adaptive Gaussian protocol which solves the variance estimation task for constant $0 < \delta <$ 1/2 requires $\Omega ( k ^ { 2 } )$ samples.

Proof. We again proceed by reduction to hypothesis testing. Let $\tau _ { t }$ be the single-mode thermal state with mean photon number t given by

$$
\tau _ { t } = \sum _ { n = 0 } ^ { \infty } \frac { t ^ { n } } { ( 1 + t ) ^ { n + 1 } } \left| n \middle > \middle < n \right| \ .\tag{E127}
$$

Recall that this has covariance matrix $( 1 + 2 t ) I _ { 2 }$ . We will consider the following two hypotheses:

$$
\rho _ { 0 } ^ { ( k ) } = | 0 \rangle \langle 0 | , \qquad \rho _ { 1 } ^ { ( k ) } = \tau _ { 1 / k } .\tag{E128}
$$

Thus one hypothesis is simply the vacuum state, while the other is a thermal state whose mean photon number is $1 / k .$ Both states are Gaussian and have nonnegative Wigner functions. In the convention where the vacuum covariance is $I _ { 2 }$ , their Wigner functions are

$$
W _ { 0 } ( z ) = \frac { 1 } { 2 \pi } \exp \biggl ( - \frac { | z | ^ { 2 } } { 2 } \biggr )\tag{E129}
$$

and

$$
W _ { 1 } ( z ) = \frac { 1 } { 2 \pi ( 1 + 2 / k ) } \exp \left( - \frac { | z | ^ { 2 } } { 2 ( 1 + 2 / k ) } \right) \ .\tag{E130}
$$

The corresponding phase-space variances are

$$
v ( \rho _ { 0 } ^ { ( k ) } ) = 1 , \qquad v ( \rho _ { 1 } ^ { ( k ) } ) = 1 + { \frac { 2 } { k } } \ .\tag{E131}
$$

As such, any algorithm which can estimate the variance to within accuracy $1 / ( 2 k )$ can solve this distinguishing task by thresholding its estimate at $1 + 1 / k$

We now proceed with the sample lower bound on the hypothesis testing problem. We will prove the stronger statement that the lower bound holds for any adaptive protocol whose measurements are Wigner-positive. This includes adaptive Gaussian measurement protocols. The reason is that a Wigner-positive measurement maps the Wigner function of the measured state to a classical outcome distribution by a stochastic map on phase-space (see Equation (C14)). Thus, even if the measurement at time t is chosen adaptively from all previous outcomes, the entire transcript distribution obtained from N copies is a classical stochastic postprocessing of the product Wigner distribution $W _ { i } ^ { \otimes N }$ , where $i \in \{ 0 , 1 \}$ labels the hypothesis.

Let $P _ { 0 } ^ { ( N ) }$ and $P _ { 1 } ^ { ( N ) }$ denote the transcript distributions induced by an arbitrary adaptive Wigner-positive protocol using $N$ copies. By data processing for total variation distance,

$$
d _ { \mathrm { T V } } \left( P _ { 0 } ^ { ( N ) } , P _ { 1 } ^ { ( N ) } \right) \leq \frac { 1 } { 2 } \left. W _ { 0 } ^ { \otimes N } - W _ { 1 } ^ { \otimes N } \right. _ { 1 } .\tag{E132}
$$

We now bound this $L ^ { 1 }$ distance using the usual KL step. Pinsker’s inequality and additivity of KL divergence give

$$
d _ { \mathrm { T V } } \left( P _ { 0 } ^ { ( N ) } , P _ { 1 } ^ { ( N ) } \right) \leq \sqrt { \frac { 1 } { 2 } D _ { K L } \left( W _ { 0 } ^ { \otimes N } \| W _ { 1 } ^ { \otimes N } \right) }\tag{E133}
$$

$$
= \sqrt { \frac { N } { 2 } D _ { K L } \left( W _ { 0 } \| W _ { 1 } \right) } .\tag{E134}
$$

Let $\begin{array} { r } { a = 1 + \frac { 2 } { k } } \end{array}$ . Since $W _ { 0 }$ and $W _ { 1 }$ are centered two-dimensional Gaussians with covariance matrices $I _ { 2 }$ and $a I _ { 2 }$ respectively, we have by Lemma 13 that

$$
D _ { K L } \left( W _ { 0 } \| W _ { 1 } \right) = \log a + \frac { 1 } { a } - 1 \ .\tag{E135}
$$

Setting $x = 2 / k$ , this is

$$
D _ { K L } \left( W _ { 0 } \| W _ { 1 } \right) = \log ( 1 + x ) + \frac { 1 } { 1 + x } - 1 \ .\tag{E136}
$$

For $k \geq 2 , x \leq 1$ , and we can use

$$
\log ( 1 + x ) + \frac { 1 } { 1 + x } - 1 \leq x ^ { 2 } \ .\tag{E137}
$$

Therefore

$$
D _ { K L } \left( W _ { 0 } \| W _ { 1 } \right) \leq \frac { 4 } { k ^ { 2 } } \ .\tag{E138}
$$

It follows that

$$
d _ { \mathrm { T V } } \left( P _ { 0 } ^ { ( N ) } , P _ { 1 } ^ { ( N ) } \right) \leq { \cal O } \left( \frac { \sqrt { N } } { k } \right) \ .\tag{E139}
$$

Hence if $N = o ( k ^ { 2 } )$ , the total variation distance between the two transcript distributions is $o ( 1 ) ;$ ; by Lemma 22, no such protocol can distinguish the two hypotheses with nonvanishing success bias. Thus any adaptive Gaussian protocol requires

$$
N \geq \Omega ( k ^ { 2 } ) ~ .\tag{E140}
$$

□

## 5. Multimode hardness of conventional characteristic function learning

We have just established a single-mode learning task for which arbitrarily high-energy Gaussian protocols face a quadratic lower bound, while we later show a non-Gaussian strategy, enabled by single-qubit control, which succeeds using linear samples. We next discuss how such energy-independent separations between Gaussian and non-Gaussian can appear in multimode learning tasks in more dramatic ways.

In particular, our discussion here concerns results obtained by Ref. [1], which studied an entanglementenabled quantum advantage for learning the characteristic function of a displacement channel. This task is rather diferent (and more dificult) than our Fourier analysis setting despite its cosmetic similarity, because it requires collecting a dataset that allows the learner to post-hoc evaluate the channel’s characteristic function at any point within a phase-space radius. Rather than the physically-motivated property estimation tasks we have considered thus far, characteristic function learning is equivalent to learning a complete description of the channel itself, as one might do in full quantum tomography of states or processes. The core observation of Ref. [1] is that due to the special structure of displacement channels as the continuous-variable analog of qubit Pauli channels, measurement with entangled EPR pairs enables an eficient learning protocol. Ref. [1] then demonstrates that all entanglement-free strategies (i.e. those without ancillary quantum memory) must, in the worst case, utilize $\exp ( \Omega ( n ) )$ ) queries to learn an n-mode characteristic function of a displacement channel.

Here, we establish that an exponential separation enabled by single-qubit control extends to multimode sensing problems by showing that, surprisingly, the characteristic function learning problem can be eficiently solved by a polynomial-energy, entanglement-free strategy that utilizes single-qubit control of an n-mode sensor. The protocol that enables this advantage will be presented in Section F 5 by realizing another class of non-Gaussian measurements; in this section, we will restate the exponential lower bounds proven in Ref. [1] and understand why, in light of their entanglement-free lower bounds, an exponential advantage is even possible without entangled bosonic probes.

We now restate the relevant setting and lower bounds in a form that will be useful for comparison with our single-qubit non-Gaussian protocol in Section F 5. For an n-mode displacement channel

$$
{ \mathcal E } _ { P } ( \rho ) = \int d ^ { 2 n } \alpha ~ P ( \alpha ) D ( \alpha ) \rho D ( \alpha ) ^ { \dagger } ~ ,\tag{E141}
$$

we write its characteristic function as

$$
\lambda _ { P } \left( \beta \right) = \mathbb { E } _ { \alpha \sim P } \left[ e ^ { i \Omega \left( \beta , \alpha \right) } \right] ~ .\tag{E142}
$$

The post-hoc characteristic-function learning task is the following.

Definition 10 (Post-hoc characteristic-function learning). Fix $\kappa , \epsilon , \delta > 0$ . A protocol learns the n-mode displacement channel $\mathcal { E } _ { P }$ on the radius-κn ball if, after making N channel queries and storing only its classical transcript, it receives an arbitrary query $\beta \in \mathbb { R } ^ { 2 n }$ satisfying $| \beta | ^ { 2 } \leq \kappa n$ and outputs an estimate $\tilde { \lambda } ( \beta )$ such that

$$
\begin{array} { r } { \operatorname* { P r } \left[ \left| \tilde { \lambda } ( \beta ) - \lambda _ { \cal P } ( \beta ) \right| \le \epsilon \right] \ge 1 - \delta ~ . } \end{array}\tag{E143}
$$

The hard family used by Ref. [1] to instantiate a hypothesis-testing lower bound is a three-peak family in characteristic-function space. In our phase-space convention, a representative form is

$$
\lambda _ { s , \gamma } ( \beta ) = e ^ { - | \beta | ^ { 2 } / ( 2 \sigma ^ { 2 } ) } + 2 i s \eta _ { 0 } e ^ { - | \beta - \gamma | ^ { 2 } / ( 2 \sigma ^ { 2 } ) } - 2 i s \eta _ { 0 } e ^ { - | \beta + \gamma | ^ { 2 } / ( 2 \sigma ^ { 2 } ) } \ ,\tag{E144}
$$

where $s \in \{ \pm 1 \} , \gamma \in \mathbb { R } ^ { 2 n }$ , and $\eta _ { 0 }$ is a suficiently small constant. The corresponding displacement density (i.e. the symplectic Fourier transform of the above characteristic function) has the form

$$
P _ { s , \gamma } ( \alpha ) = \left( \frac { 2 \sigma ^ { 2 } } { \pi } \right) ^ { n } e ^ { - 2 \sigma ^ { 2 } | \alpha | ^ { 2 } } \left[ 1 + 4 s \eta _ { 0 } \sin { \left( 2 \Omega ( \gamma , \alpha ) \right) } \right] .\tag{E145}
$$

The real parameter σ controls the width of the Gaussian envelope in the displacement distribution. Smaller σ means broader displacement noise, while larger σ means the channel usually applies smaller displacements.

The main lower bound of Ref. [1] applies to all entanglement-free strategies, including arbitrary non-Gaussian input states and arbitrary non-Gaussian destructive measurements after each channel use. However, for the finite-energy three-peak family, this lower bound requires an additional small-σ condition. Defining

$$
M _ { \kappa } = \operatorname* { m a x } \left\{ 1 - 1 . 9 8 \kappa , 0 . 9 9 \kappa \left( \sqrt { 1 + ( 0 . 9 9 \kappa ) ^ { - 2 } } - 1 \right) \right\} ,\tag{E146}
$$

Ref. [1] proves that if $n \geq 8 , \epsilon \leq 0 . 2 4$ , and importantly, $2 \sigma ^ { 2 } \le M _ { \kappa }$ , then any entanglement-free protocol that learns the 3-peak hard family (with unknown $s , \gamma )$ on the ball $| \beta | ^ { 2 } \leq$ κn must use

$$
N \geq 0 . 0 1 \epsilon ^ { - 2 } \left( 1 + \frac { 1 . 9 8 \kappa } { 1 + 2 \sigma ^ { 2 } } \right) ^ { n }\tag{E147}
$$

queries. This should be contrasted with the lower bound Ref. [1] obtains for Gaussian entanglement-free strategies. If the input states and destructive measurements are Gaussian, then for the same three-peak family the following lower bound holds for every $\sigma > 0$ , with no small-σ assumption:

$$
N \geq 0 . 0 1 \epsilon ^ { - 2 } \operatorname* { m i n } \left\{ \left( 1 + \frac { 0 . 9 9 \kappa } { \sigma ^ { 2 } } \right) ^ { n / 2 } , \left( 1 + \frac { 1 . 9 8 \kappa } { 1 + 2 \sigma ^ { 2 } } \right) ^ { n } \right\} \ .\tag{E148}
$$

In particular, for any fixed $\kappa > 0$ and any constant $\sigma ^ { 2 } = O ( 1 )$ , Gaussian entanglement-free characteristicfunction learning remains exponentially hard in $n ,$ even allowing arbitrarily high-energy Gaussian states and measurements. Substituting $\sigma ^ { 2 } = C$ log n for a fixed constant C into the Gaussian lower bound, we get

$$
N _ { \mathrm { G a u s s } } \geq \epsilon ^ { - 2 } \exp \left( \Omega _ { \kappa } \left( { \frac { n } { \log n } } \right) \right) \ .\tag{E149}
$$

Namely, for $\sigma$ of order log n, a superpolynomial lower bound still persists in the Gaussian setting, whereas no entanglement-free lower bound is known to exist. It is not a priori clear whether this missing condition is merely an artifact of the proof method or whether it marks a real transition in the power of non-Gaussian entanglement-free measurements. Surprisingly, we show in Section F 5 that the latter is the correct interpretation: once the displacement distribution is narrow enough (i.e. σ is of order log n), a non-Gaussian probe prepared and measured using only single-qubit control of an n-mode probe state can convert each channel use into an approximate classical sample of the displacement. This gives an eficient entanglement-free learner in a regime where the general non-Gaussian lower bound no longer applies, while the Gaussian lower bound still gives a superpolynomial lower bound. This result marks a superpolynomial Gaussian vs. non-Gaussian learning separation in the multimode setting, enabled by only single-qubit control.

## a. Simplified entanglement-free lower bound on multimode channel learning through QΨ

At the conclusion of Appendix D, we noted that QΨ is a broad framework that goes beyond existing techniques in quantum learning theory in its ability to incorporate realistic constraints and yield simultaneous upper and lower bounds. In fact, many of the existing separations in quantum learning theory can be shown to arise as special cases of QΨ, and often in a far simpler manner. Here, we provide a reproof of one such canonical result, both as a worked example of QΨ-style proofs and a demonstration of how QΨ can substantially simplify otherwise dificult separations.

We consider the lower bound of Ref. [1], which proved that any entanglement-free strategy sufers an exponential query complexity to learn the full characteristic function of a worst-case instance of a multimode displacement channel under post-hoc queries. The original proof is complex and involves two overarching steps: first, relating the success probability of an arbitrary adaptive N-query protocol to a problem-specific information quantity via involved bosonic analysis calculations, and second, controlling the information quantity itself. The QΨ approach greatly simplifies the former half of the proof, and it follows that the particular information quantity arises naturally as a bound on the AFI.

We recall some notation from Ref. [1]. Let $\gamma \in \mathbb { R } ^ { 2 n }$ be drawn from

$$
q _ { \sigma _ { \gamma } } ( \gamma ) = \left( \frac { 1 } { 2 \pi \sigma _ { \gamma } ^ { 2 } } \right) ^ { n } \exp \left( - \frac { | \gamma | ^ { 2 } } { 2 \sigma _ { \gamma } ^ { 2 } } \right) .\tag{E150}
$$

For a normalized n-mode state |A⟩ and a nonzero vector |B⟩, define

$$
G ^ { A , B } ( \beta ) = \frac { \langle B | D ( \beta ) ^ { \dagger } | B \rangle } { \langle B | B \rangle } \langle A | D ( \beta ) | A \rangle ,\tag{E151}
$$

$$
G _ { \sigma } ^ { A , B } ( \gamma ) = \frac { \int d ^ { 2 n } \beta \ e ^ { - | \beta - \gamma | ^ { 2 } / ( 2 \sigma ^ { 2 } ) } G ^ { A , B } ( \beta ) } { \int d ^ { 2 n } \beta \ e ^ { - | \beta | ^ { 2 } / ( 2 \sigma ^ { 2 } ) } G ^ { A , B } ( \beta ) } .\tag{E152}
$$

We then take the following lemma which controls a problem-specific information quantity as given. While QΨ does not simplify the proof of this lemma, we will show here that it simplifies the rest of the proof substantially.

Lemma 11 (Lemma S1 of Ref. [1]). If

$$
\sigma ^ { 2 } \leq \operatorname* { m a x } \left\{ \frac { 1 } { 2 } - 2 \sigma _ { \gamma } ^ { 2 } , \sigma _ { \gamma } ^ { 2 } \left( \sqrt { 1 + \frac { 1 } { 4 \sigma _ { \gamma } ^ { 4 } } } - 1 \right) \right\} ,\tag{E153}
$$

then for every normalized |A⟩ and nonzero $| B \rangle$

$$
\mathbb { E } _ { \gamma } \left[ \left| G _ { \sigma } ^ { A , B } ( \gamma ) \right| ^ { 2 } \right] \leq \left( \frac { 1 + 2 \sigma ^ { 2 } } { 1 + 2 \sigma ^ { 2 } + 4 \sigma _ { \gamma } ^ { 2 } } \right) ^ { n } .\tag{E154}
$$

Theorem E.12 (Entanglement-free lower bound of Ref. [1]). Let $n \geq 8 , \kappa > 0$ , and $0 < \epsilon \le 0 . 2 4$ . Suppose an adaptive entanglement-free protocol makes N queries to an arbitrary n-mode displacement channel, then receives any β satisfying $| \beta | ^ { 2 } \leq \kappa n$ , and estimates the channel characteristic function at $\beta$ to error ϵ with success probability at least $2 / 3$ . Then

$$
N \geq 0 . 0 1 \epsilon ^ { - 2 } ( 1 + 1 . 9 8 \kappa ) ^ { n } .\tag{E155}
$$

Proof. Set $2 \sigma _ { \gamma } ^ { 2 } = 0 . 9 9 \kappa .$ , and fix $\sigma > 0$ small enough that Lemma 11 applies and $\sigma ^ { 2 } \le \sigma _ { \gamma } ^ { 2 }$ . In QΨ language, consider the Gaussian reference background and Fourier perturbation

$$
P _ { 0 } ( d \alpha ) = \left( \frac { 2 \sigma ^ { 2 } } { \pi } \right) ^ { n } e ^ { - 2 \sigma ^ { 2 } | \alpha | ^ { 2 } } d ^ { 2 n } \alpha , \qquad h _ { \gamma } ( \alpha ) = \sin \left( 2 \Omega ( \gamma , \alpha ) \right) .\tag{E156}
$$

Let $\epsilon _ { 0 } = \epsilon / 0 . 9 8$ and $\eta = 4 \epsilon _ { 0 } < 1$ . Since $h _ { \gamma }$ is centered under $P _ { 0 }$ and $\| h _ { \gamma } \| _ { \infty } \leq 1$ , the laws

$$
P _ { s , \gamma } ( d \alpha ) = ( 1 + s \eta h _ { \gamma } ( \alpha ) ) P _ { 0 } ( d \alpha ) , \qquad s \in \{ \pm 1 \} ,\tag{E157}
$$

are valid probability distributions. Their characteristic functions are

$$
\lambda _ { 0 } ( \beta ) = e ^ { - | \beta | ^ { 2 } / ( 2 \sigma ^ { 2 } ) } ,\tag{E158}
$$

$$
\lambda _ { s , \gamma } ( \beta ) = \lambda _ { 0 } ( \beta ) + 2 i s \epsilon _ { 0 } e ^ { - | \beta - \gamma | ^ { 2 } / ( 2 \sigma ^ { 2 } ) } - 2 i s \epsilon _ { 0 } e ^ { - | \beta + \gamma | ^ { 2 } / ( 2 \sigma ^ { 2 } ) } ,\tag{E159}
$$

illustrating that the QΨ hypothesis-testing formulation yields the same characteristic function distinguishing task considered in [1]. Now consider one entanglement-free query to the hypothesis channel. Revealing a purestate decomposition of the input and refining the final POVM to rank-one outcomes can only increase the available information [15], so it sufices to use a pure probe $| A \rangle$ and an outcome proportional to a projector $| B _ { y } \rangle \langle B _ { y } |$ . Substituting the characteristic functions above into the channel Fourier representation gives

$$
Q _ { s , \gamma } ( d y ) = ( 1 + s \eta R _ { M , \gamma } ( y ) ) Q _ { 0 } ( d y ) , \qquad R _ { M , \gamma } ( y ) = - \mathrm { I m } G _ { \sigma } ^ { A , B _ { y } } ( \gamma ) ,\tag{E160}
$$

where $Q _ { 0 }$ is the outcome law under $P _ { 0 }$ . Thus $R _ { M , \gamma }$ is the $\mathrm { Q } \Psi$ score for the direction $h _ { \gamma }$ , and

$$
\mathbb { E } _ { \gamma } \left[ \mathsf { A } _ { M } ( h _ { \gamma } ; P _ { 0 } ) \right] = \int \mathbb { E } _ { \gamma } \left[ | R _ { M , \gamma } ( y ) | ^ { 2 } \right] d Q _ { 0 } ( y )\tag{E161}
$$

$$
\leq \Lambda _ { n } , \quad \quad \Lambda _ { n } : = \left( \frac { 1 + 2 \sigma ^ { 2 } } { 1 + 2 \sigma ^ { 2 } + 4 \sigma _ { \gamma } ^ { 2 } } \right) ^ { n } ,\tag{E162}
$$

where the inequality is Lemma 11. We now use Lemma 24 and obtain that any entanglement-free protocol which achieves constant success bias for this hypothesis test requires

$$
N \ge \Omega ( 1 / \epsilon _ { 0 } ^ { 2 } \Lambda _ { n } ) \ .\tag{E163}
$$

It remains to reduce characteristic-function learning to this test. Sample $\gamma \sim q _ { \sigma }$ and a uniform sign s, give the learner either $\mathcal { E } _ { P _ { 0 } }$ or $\mathcal { E } _ { P _ { s , \gamma } }$ with equal prior probability, and reveal $\gamma$ only after all channel queries are complete. On the event

$$
\mathcal { G } = \left\{ 2 \sigma ^ { 2 } < | \gamma | ^ { 2 } \leq \kappa n \right\} ,\tag{E164}
$$

querying the learned characteristic function at $\beta = \gamma$ separates the null value from either signed alternative because

$$
| \lambda _ { 0 } ( \gamma ) - \lambda _ { s , \gamma } ( \gamma ) | = 2 \epsilon _ { 0 } \left( 1 - e ^ { - 2 | \gamma | ^ { 2 } / \sigma ^ { 2 } } \right) > 2 \epsilon .\tag{E165}
$$

The Gaussian tail estimate of Ref. [1] gives $\operatorname* { P r } ( \mathcal G ) \ge 0 . 4 9 9 8 7$ for $n \geq 8$ and $\sigma ^ { 2 } \le \sigma _ { \gamma } ^ { 2 }$ . The binary testing identity therefore gives

$$
{ \mathbb E } _ { \gamma } d _ { \mathrm { T V } } \left( Q _ { 0 } ^ { ( N ) } , Q _ { \operatorname* { m i x } , \gamma } ^ { ( N ) } \right) \geq \frac { 1 } { 3 } \operatorname* { P r } ( \mathcal { G } ) \geq 0 . 1 6 6 6 .\tag{E166}
$$

Combining the two total-variation bounds and using $\epsilon = 0 . 9 8 \epsilon _ { 0 }$ gives

$$
N \geq 0 . 0 1 \epsilon ^ { - 2 } \left( 1 + \frac { 4 \sigma _ { \gamma } ^ { 2 } } { 1 + 2 \sigma ^ { 2 } } \right) ^ { n } = 0 . 0 1 \epsilon ^ { - 2 } \left( 1 + \frac { 1 . 9 8 \kappa } { 1 + 2 \sigma ^ { 2 } } \right) ^ { n } .\tag{E167}
$$

Taking $\sigma \to 0$ proves the theorem.

This gives a greatly simplified proof relative to the original result in S3 of [1].

## 6. A hierarchy of exponential sensing lower bounds with increased control depth

The separations above show that adding a single controllable qubit to an otherwise conventional bosonic sensor can dramatically increase the Fourier bandwidth of accessible sensing tasks. A natural next question is whether increasing the coherent processing capability of that qubit leads to a genuine hierarchy of sensing power.

Proving such a separation presents an important technical challenge. Previously, most query-complexity lower bounds in quantum learning and sensing have operated in highly unconstrained settings such as allowing arbitrary POVMs; in some cases, additional constraints such as quantum memory [15] or noise [18] have been accounted for with substantially more complex proofs. Circuit depth lower bounds on learning have been similarly limited beyond studies of full state tomography [31]. This is because unconstrained lower bounds only require the use of simple properties of the measurement class, such as positivity and normalization of arbitrary POVMs. When additional constraints are present, it often becomes intractable to control their efect on quantities like one-sided likelihood ratios or statistical divergence bounds, which form the core of known information-theoretic lower bound techniques.

In this section, we prove precisely this kind of hierarchy lower bound.

Theorem E.13 (Hardness of learning with slightly shallower depths). For any $k \in \mathbb { Z } ^ { + }$ and $\gamma < \log _ { 3 } 2$ ≈ 0.63, there exist functions $h _ { k } ( \alpha )$ such that any protocol of control depth $\leq \gamma k$ (as in Definition $1 \text{‰}$ requires $\exp ( \Omega ( k ) )$ queries to estimate $\mathbb { E } [ h _ { k } ( \alpha ) ]$ for any signal to constant accuracy.

Our proof elucidates the utility of the QΨ formalism for characterizing the power of constrained quantum measurements. This first part of this section is devoted to instantiating the appropriate family of polynomials $h _ { k }$ that will appear in the Lemma 24 hypothesis test. This will require introducing harmonic-analytic techniques that connect the control-depth hierarchy to the Fourier support picture. Then, we use these results to obtain bounds on the support size of depth-r response kernels so that we may use Corollary 25 to obtain the main result of this section.

## a. Fourier response of depth-r measurements and hypothesis testing

Our aim is to use the semiclassical measurement lemma to control the hardness of hypothesis testing with bounded-depth circuits. To do so, we must first characterize the Fourier decomposition of response functions in this class. To instantiate a control depth-dependent bound, we choose a concrete circuit model. As described in Section C 4, we work with circuits that interleave qubit rotations with echoed conditional displacement (ECD) gates, because these are precisely the class of circuits we use in our experiments. These circuits are universal and can eficiently implement powerful primitives such as quantum signal processing, as realized in Ref. [67]. We further work with binary measurements, which is without loss of generality when classical postprocessing is allowed [15]. These choices are not limiting: the QΨ proof technique enables lower bounds for other circuit models as long as one can characterize the Fourier support of their response kernels.

Definition 14 (Depth-r measurement class). Consider a joint system of a bosonic mode and a qubit supported on $\left\{ \left| g \right. , \left| e \right. \right\}$ . A signal-interleaved depth-r measurement is any binary measurement on this system implemented as follows. The protocol starts from a fixed oscillator-qubit state $\rho _ { 0 . }$ , for instance $| 0 , g \rangle \langle 0 , g |$ for concreteness. It then applies a circuit of the form

$$
V _ { r } ( \alpha ) = S _ { r } ( \alpha ) G _ { r } S _ { r - 1 } ( \alpha ) G _ { r - 1 } \cdot \cdot \cdot S _ { 1 } ( \alpha ) G _ { 1 } S _ { 0 } ( \alpha ) ,\tag{E168}
$$

where each signal window is

$$
S _ { \ell } ( \alpha ) = D ( t _ { \ell } \alpha ) \otimes I _ { q }\tag{E169}
$$

for a known $t _ { \ell } \in \{ 0 , 1 \}$ , and each control layer is

$$
G _ { j } = \operatorname { E C D } ( \beta _ { j } ) ( I \otimes R _ { j } ) .\tag{E170}
$$

Here $R _ { j }$ is an arbitrary single-qubit SU(2) rotation and

$$
\mathrm { E C D } ( \beta _ { j } ) = D ( \beta _ { j } / 2 ) \otimes | e \rangle \langle g | + D ( - \beta _ { j } / 2 ) \otimes | g \rangle \langle e | .\tag{E171}
$$

After the final signal or control window, a binary POVM is performed on the qubit. We denote by $\mathcal { M } _ { r }$ the family of all binary measurements obtainable in this way, allowing arbitrary ECD displacement and qubit rotation parameters at every layer and arbitrary choices of the known interleaving variables $t _ { 0 } , \ldots , t _ { r }$

This is a physically-motivated model in which a signal imparts a displacement α sampled from its underlying displacement during each sensing window, and we can apply fast interleaved control to our sensor using the operations available on our circuit-QED platform $\ ( \mathrm { F i g . \ 3 . }$ Setting a particular $t _ { \ell }$ to 0 simply denotes applying multiple consecutive controls. The displacement parameter α from the signal is, of course, refreshed between measurements, as we take each sensing experiment to be an independent query of the signal. We remark that all subsequent arguments go through identically for the two-parameter gate $\mathrm { J C } ( \beta _ { 1 } , \beta _ { 2 } )$ from Equation (C97); in that case the role of $\beta _ { j }$ below is played by the branch diference $\beta _ { j , 2 } - \beta _ { j , 1 }$ . We work with the symmetric ECD gate for clarity. The following lemmas connect bounded control depth to support size in the Weyl basis.

Lemma 15 (Weyl expansion of depth-r sensing circuits). Let $V _ { r } ( \alpha )$ be a signal-interleaved depth-r circuit as in Definition $1 \% .$ Then there are branch labels indexed $\textit { b y s } = ( s _ { 1 } , \ldots , s _ { r } ) \in \{ \pm 1 \} ^ { r }$ , qubit operators $A _ { s }$ signal-independent phases $\theta _ { s } ,$ and phase-space labels $\Gamma _ { s } , \Lambda _ { s } \in \mathbb { R } ^ { 2 }$ such that

$$
V _ { r } ( \alpha ) = \sum _ { s \in \{ \pm 1 \} ^ { r } } e ^ { i \theta _ { s } } e ^ { i \Omega ( \Lambda _ { s } , \alpha ) } D ( T \alpha + \Gamma _ { s } ) \otimes A _ { s } ,\tag{E172}
$$

where $\begin{array} { r } { T = \sum _ { \ell = 0 } ^ { r } t _ { \ell } , \Gamma _ { s } = \frac { 1 } { 2 } \sum _ { j = 1 } ^ { r } s _ { j } \beta _ { j } } \end{array}$ , and

$$
\Lambda _ { s } = \frac { 1 } { 4 } \sum _ { j = 1 } ^ { r } s _ { j } \left( T _ { \geq j } - T _ { < j } \right) \beta _ { j } .\tag{E173}
$$

Here

$$
T _ { < j } = \sum _ { \ell = 0 } ^ { j - 1 } t _ { \ell } , \qquad T _ { \geq j } = \sum _ { \ell = j } ^ { r } t _ { \ell } .\tag{E174}
$$

In particular, the signal-interleaved circuit has at most $2 ^ { r }$ coherent displacement branches.

Proof. Recall that an ECD gate can be written as

$$
\mathrm { E C D } ( \beta ) = \sum _ { s \in \{ \pm 1 \} } D ( s \beta / 2 ) \otimes F _ { s } ,\tag{E175}
$$

where $F _ { + } = | e \rangle \langle g |$ and $F _ { - } = | g \rangle \langle e |$ . A single-qubit rotation has the form $I \otimes R$ , so it changes only the qubit-side operator. Expanding each ECD gate in the circuit gives a sum over branch strings $s = ( s _ { 1 } , \ldots , s _ { r } ) \in \{ \pm 1 \} ^ { r }$ On the branch s, the oscillator displacement sequence appearing in $V _ { r } ( \alpha )$ is

$$
D ( t _ { r } \alpha ) D ( s _ { r } \beta _ { r } / 2 ) D ( t _ { r - 1 } \alpha ) \cdot \cdot \cdot D ( t _ { 1 } \alpha ) D ( s _ { 1 } \beta _ { 1 } / 2 ) D ( t _ { 0 } \alpha ) .\tag{E176}
$$

The corresponding qubit-side operator is a product of the $F _ { s _ { j } } \mathrm { \Delta } ^ { \prime } \mathrm { s }$ and the single-qubit rotations. We denote this operator by $A _ { s }$

We now compute the oscillator product along a fixed branch. We repeatedly use the Weyl relation

$$
D ( a ) D ( b ) = e ^ { - i \Omega ( a , b ) / 2 } D ( a + b ) .\tag{E177}
$$

For any ordered sequence of displacement labels $a _ { 0 } , \ldots , a _ { m }$ , this relation implies

$$
D ( a _ { m } ) \cdot \cdot \cdot D ( a _ { 0 } ) = \exp \left( - \frac { i } { 2 } \sum _ { u > v } \Omega ( a _ { u } , a _ { v } ) \right) D \left( \sum _ { u = 0 } ^ { m } a _ { u } \right) .\tag{E178}
$$

In our branch, the total displacement label is

$$
\sum _ { \ell = 0 } ^ { r } t _ { \ell } \alpha + \frac { 1 } { 2 } \sum _ { j = 1 } ^ { r } s _ { j } \beta _ { j } = T \alpha + \Gamma _ { s } .\tag{E179}
$$

The Weyl phase in the product has three types of terms. The signal-signal terms vanish because $\Omega ( \alpha , \alpha ) = 0$ . The control-control terms are independent of α and can be collected into a signal-independent phase $e ^ { i \theta _ { s } }$ . The only signal-dependent terms come from pairs containing one signal displacement $t _ { \ell } \alpha$ and one branch displacement $s _ { j } \beta _ { j } / 2$

Fix $j .$ If the signal window $t _ { \ell } \alpha$ occurs after the j-th ECD layer, meaning $\ell \geq j ,$ then the pair contributes

$$
- \frac { i } { 2 } \Omega ( t _ { \ell } \alpha , s _ { j } \beta _ { j } / 2 ) = \frac { i } { 4 } s _ { j } t _ { \ell } \Omega ( \beta _ { j } , \alpha ) .\tag{E180}
$$

If the signal window occurs before the $j \mathrm { - t h }$ ECD layer, meaning $\ell < j$ , then the pair contributes

$$
- \frac { i } { 2 } \Omega ( s _ { j } \beta _ { j } / 2 , t _ { \ell } \alpha ) = - \frac { i } { 4 } s _ { j } t _ { \ell } \Omega ( \beta _ { j } , \alpha ) .\tag{E181}
$$

Summing over all signal windows, the total signal-dependent phase associated with the j-th control displacement is

$$
\frac { i } { 4 } s _ { j } \ : ( T _ { \geq j } - T _ { < j } ) \ : \Omega ( \beta _ { j } , \alpha ) .\tag{E182}
$$

Therefore the full signal-dependent phase on branch s is

$$
\exp \left( i \Omega \left( \frac { 1 } { 4 } \sum _ { j = 1 } ^ { r } s _ { j } \left( T _ { \geq j } - T _ { < j } \right) \beta _ { j } , \alpha \right) \right) .\tag{E183}
$$

This is exactly $e ^ { i \Omega ( \Lambda _ { s } , \alpha ) }$ . Hence the branch contribution has the form

$$
e ^ { i \theta _ { s } } e ^ { i \Omega ( \Lambda _ { s } , \alpha ) } D ( T \alpha + \Gamma _ { s } ) \otimes A _ { s } .\tag{E184}
$$

Summing over all $2 ^ { r }$ branch strings proves the claim.

Lemma 16 (Weyl support of depth-r measurements). Let $M \in { \mathcal { M } } _ { \tau }$ be any signal-interleaved depth-r measurement. Then there exists a set $S _ { M } \subset \mathbb { R } ^ { 2 }$ with $| S _ { M } | \le 3 ^ { r }$ such that

$$
f _ { M } ( \alpha ) = a _ { 0 } + \sum _ { \zeta \in S _ { M } } a _ { \zeta } \chi _ { \zeta } ( \alpha ) .\tag{E185}
$$

Consequently, for any reference distribution $P _ { 0 }$ , the centered response satisfies

$$
f _ { M } - \int f _ { M } \ : d P _ { 0 } \in \mathrm { s p a n } \{ \widetilde { \chi } _ { \zeta } : \zeta \in S _ { M } \} ,\tag{E186}
$$

where $\{ \widetilde { \chi } _ { \zeta } \} _ { \zeta \in \mathbb { R } ^ { 2 } }$ is the centered Weyl family.

Proof. Let $E _ { q }$ denote the accepting element of the final binary qubit POVM, so that $0 \leq E _ { q } \leq I _ { q }$ . By Lemma 15, the signal-dependent circuit has the expansion

$$
V _ { r } ( \alpha ) = \sum _ { s \in \{ \pm 1 \} ^ { r } } e ^ { i \theta _ { s } } e ^ { i \Omega ( \Lambda _ { s } , \alpha ) } D ( T \alpha + \Gamma _ { s } ) \otimes A _ { s } .\tag{E187}
$$

The response function is

$$
f _ { M } ( \alpha ) = \mathrm { T r } \left[ ( I \otimes E _ { q } ) V _ { r } ( \alpha ) \rho _ { 0 } V _ { r } ( \alpha ) ^ { \dagger } \right] .\tag{E188}
$$

Equivalently, by cyclicity of trace,

$$
f _ { M } ( \alpha ) = \mathrm { T r } \left[ \rho _ { 0 } V _ { r } ( \alpha ) ^ { \dagger } ( I \otimes E _ { q } ) V _ { r } ( \alpha ) \right] .\tag{E189}
$$

Substituting the expansion of $V _ { r } ( \alpha )$ gives

$$
f _ { M } ( \alpha ) = \sum _ { s , s ^ { \prime } \in \{ \pm 1 \} ^ { r } } e ^ { - i \theta _ { s } } e ^ { i \theta _ { s ^ { \prime } } } e ^ { - i \Omega ( \Lambda _ { s } , \alpha ) } e ^ { i \Omega ( \Lambda _ { s ^ { \prime } } , \alpha ) }\tag{E190}
$$

$$
\times \mathrm { T r } \left[ \rho _ { 0 } \left( D ( T \alpha + \Gamma _ { s } ) ^ { \dagger } D ( T \alpha + \Gamma _ { s ^ { \prime } } ) \otimes A _ { s } ^ { \dagger } E _ { q } A _ { s ^ { \prime } } \right) \right] .\tag{E191}
$$

We now compute the remaining oscillator displacement product. Since $D ( x ) ^ { \dagger } = D ( - x )$ ,

$$
D ( T \alpha + \Gamma _ { s } ) ^ { \dagger } D ( T \alpha + \Gamma _ { s ^ { \prime } } ) = D ( - T \alpha - \Gamma _ { s } ) D ( T \alpha + \Gamma _ { s ^ { \prime } } ) .\tag{E192}
$$

Using the Weyl relation,

$$
D ( - T \alpha - \Gamma _ { s } ) D ( T \alpha + \Gamma _ { s ^ { \prime } } ) = e ^ { - i \Omega ( - T \alpha - \Gamma _ { s } , T \alpha + \Gamma _ { s ^ { \prime } } ) / 2 } D ( \Gamma _ { s ^ { \prime } } - \Gamma _ { s } ) .\tag{E193}
$$

The exponent simplifies as follows:

$$
\Omega ( - T \alpha - \Gamma _ { s } , T \alpha + \Gamma _ { s ^ { \prime } } ) = \Omega ( - T \alpha , T \alpha ) + \Omega ( - T \alpha , \Gamma _ { s ^ { \prime } } ) + \Omega ( - \Gamma _ { s } , T \alpha ) + \Omega ( - \Gamma _ { s } , \Gamma _ { s ^ { \prime } } )
$$

$$
- T \Omega ( \alpha , \Gamma _ { s ^ { \prime } } ) - T \Omega ( \Gamma _ { s } , \alpha ) - \Omega ( \Gamma _ { s } , \Gamma _ { s ^ { \prime } } )\tag{E194}
$$

(E195)

$$
= T \Omega ( \Gamma _ { s ^ { \prime } } , \alpha ) - T \Omega ( \Gamma _ { s } , \alpha ) - \Omega ( \Gamma _ { s } , \Gamma _ { s ^ { \prime } } )\tag{E196}
$$

$$
= T \Omega ( \Gamma _ { s ^ { \prime } } - \Gamma _ { s } , \alpha ) - \Omega ( \Gamma _ { s } , \Gamma _ { s ^ { \prime } } ) .\tag{E197}
$$

Therefore

$$
D ( T \alpha + \Gamma _ { s } ) ^ { \dagger } D ( T \alpha + \Gamma _ { s ^ { \prime } } ) = e ^ { - i T \Omega ( \Gamma _ { s ^ { \prime } } - \Gamma _ { s } , \alpha ) / 2 } e ^ { i \Omega ( \Gamma _ { s } , \Gamma _ { s ^ { \prime } } ) / 2 } D ( \Gamma _ { s ^ { \prime } } - \Gamma _ { s } ) .\tag{E198}
$$

Equivalently, the α-dependent part of this product is $e ^ { i \Omega \left( T \left( \Gamma _ { s } - \Gamma _ { s ^ { \prime } } \right) / 2 , \alpha \right) }$

Now combine all α-dependent phases in the $( s , s ^ { \prime } )$ summand. The branch phases contribute $e ^ { i \Omega ( \Lambda _ { s ^ { \prime } } - \Lambda _ { s } , \alpha ) }$ ， and the displacement product contributes $e ^ { i \Omega ( T ( \Gamma _ { s } - \hat { \Gamma } _ { s ^ { \prime } } ) / 2 , \alpha ) }$ . Thus the total α-dependent phase is $e ^ { i \Omega ( \zeta _ { s , s ^ { \prime } } , \alpha ) }$ , where

$$
\zeta _ { s , s ^ { \prime } } = \Lambda _ { s ^ { \prime } } - \Lambda _ { s } + \frac { T } { 2 } ( \Gamma _ { s } - \Gamma _ { s ^ { \prime } } ) .\tag{E199}
$$

All remaining factors in the $( s , s ^ { \prime } )$ summand are independent of $\alpha$

We now show that the possible labels $\zeta _ { s , s ^ { \prime } }$ form a set of size at most $3 ^ { r }$ . Using

$$
\Gamma _ { s } = \frac { 1 } { 2 } \sum _ { j = 1 } ^ { r } s _ { j } \beta _ { j }\tag{E200}
$$

and

$$
\Lambda _ { s } = \frac { 1 } { 4 } \sum _ { j = 1 } ^ { r } s _ { j } \left( T _ { \geq j } - T _ { < j } \right) \beta _ { j } ,\tag{E201}
$$

we obtain

$$
\begin{array} { l } { { \zeta _ { s , s ^ { \prime } } = \displaystyle \frac { 1 } { 4 } \sum _ { j = 1 } ^ { r } ( s _ { j } ^ { \prime } - s _ { j } ) ( T _ { \ge j } - T _ { < j } ) \beta _ { j } + \frac { T } { 4 } \sum _ { j = 1 } ^ { r } ( s _ { j } - s _ { j } ^ { \prime } ) \beta _ { j } } } \\ { { \mathrm { ~ } = \displaystyle \frac { 1 } { 4 } \sum _ { j = 1 } ^ { r } ( s _ { j } ^ { \prime } - s _ { j } ) ( T _ { \ge j } - T _ { < j } - T ) \beta _ { j } . } } \end{array}\tag{E202}
$$

(E203)

Since $T = T _ { \geq j } + T _ { < j }$ , this becomes

$$
\zeta _ { s , s ^ { \prime } } = - \frac { 1 } { 2 } \sum _ { j = 1 } ^ { r } ( s _ { j } ^ { \prime } - s _ { j } ) T _ { < j } \beta _ { j } .\tag{E204}
$$

Equivalently,

$$
\zeta _ { s , s ^ { \prime } } = \sum _ { j = 1 } ^ { r } m _ { j } T _ { < j } \beta _ { j } ,\tag{E205}
$$

where $m _ { j } = ( s _ { j } - s _ { j } ^ { \prime } ) / 2$ lies in $\{ - 1 , 0 , 1 \}$ . Therefore every response frequency lies in the set

$$
\left\{ \sum _ { j = 1 } ^ { r } m _ { j } T _ { < j } \beta _ { j } : m _ { j } \in \{ - 1 , 0 , 1 \} \right\} .\tag{E206}
$$

This set has cardinality at most $3 ^ { r }$ . Grouping all summands with the same value of $\zeta _ { s , s ^ { \prime } }$ gives

$$
f _ { M } ( \alpha ) = \sum _ { \zeta \in S _ { M } \cup \{ 0 \} } a _ { \zeta } \chi _ { \zeta } ( \alpha ) ,\tag{E207}
$$

where $| S _ { M } | \le 3 ^ { r }$ after absorbing the $\zeta = 0$ contribution into the constant term $a _ { 0 }$ . Hence

$$
f _ { M } ( \alpha ) = a _ { 0 } + \sum _ { \zeta \in S _ { M } } a _ { \zeta } \chi _ { \zeta } ( \alpha )\tag{E208}
$$

with $| S _ { M } | \le 3 ^ { r }$ . It remains only to prove the centered statement. Let $P _ { 0 }$ be any reference distribution and define

$$
\mu _ { \zeta } = \int \chi _ { \zeta } ( \alpha ) d P _ { 0 } ( \alpha ) , \qquad \widetilde { \chi } _ { \zeta } = \chi _ { \zeta } - \mu _ { \zeta } .\tag{E209}
$$

Taking the $P _ { 0 }$ -mean of the expansion gives

$$
\int f _ { M } d P _ { 0 } = a _ { 0 } + \sum _ { \zeta \in S _ { M } } a _ { \zeta } \mu _ { \zeta } .\tag{E210}
$$

Therefore

$$
f _ { M } - \int f _ { M } d P _ { 0 } = a _ { 0 } + \sum _ { \zeta \in S _ { M } } a _ { \zeta } \chi _ { \zeta } - a _ { 0 } - \sum _ { \zeta \in S _ { M } } a _ { \zeta } \mu _ { \zeta }\tag{E211}
$$

$$
= \sum _ { \zeta \in S _ { M } } a _ { \zeta } ( \chi _ { \zeta } - \mu _ { \zeta } )\tag{E212}
$$

$$
= \sum _ { \zeta \in S _ { M } } a _ { \zeta } \widetilde { \chi } _ { \zeta } .\tag{E213}
$$

Thus the same support set $S _ { M }$ works after centering.

The second ingredient in our semiclassical measurement lemma is a perturbation h which has a sparse Fourier support in the same Weyl basis. We wish to construct a family of such functions $h _ { k }$ , indexed by positive integers $k \in \mathbb { Z } ^ { + }$ , such that the following requirements are satisfied. Let $P _ { 0 }$ denote the centered Gaussian distribution on $\mathbb { R } ^ { 2 }$ with covariance $\sigma ^ { 2 } I$ , where σ is a positive constant.

1. (Centering and normalization) For all $k ,$ we have $\begin{array} { r } { \int h ( \alpha ) \ d P _ { 0 } ( \alpha ) = 0 } \end{array}$ and $\| h _ { k } \| _ { \infty } \leq 1$

2. (Flatness and sparsity) They have exp(k) many nonzero Fourier coeficients, all of order $\exp ( - k )$ magnitude

3. (Exact realizability) A depth-k measurement can distinguish the hypothesis signals $P _ { \pm } = P _ { 0 } ( 1 { \pm } \eta h _ { k } )$ with constant success probability, when $\eta < 1$ is a constant.

To construct such a perturbation, we modify a sequence of polynomials known as Golay-Rudin-Shapiro polynomials in harmonic analysis. They were introduced as explicit examples of sign polynomials with unusually flat magnitude on the unit circle. This flatness is exactly the feature we need: no single Fourier coeficient of the resulting response should carry too much of the signal. Because of this property, we can fall back to simply counting the number of terms in the Fourier support, rather than having to bound their amplitudes.

We use the following definitions. Let $e \in \mathbb { R } ^ { 2 }$ be any fixed unit vector (for concreteness, one can take e.g. (0, 1)) and a constant spacing parameter $B > 0$ . Define

$$
\beta _ { j } = B 3 ^ { j - 1 } e , \qquad j = 1 , \ldots , k .\tag{E214}
$$

For $n = ( n _ { 1 } , \dots , n _ { k } ) \in \{ - 1 , 0 , 1 \} ^ { k }$ , write

$$
\xi _ { n } = \sum _ { j = 1 } ^ { k } n _ { j } \beta _ { j } .\tag{E215}
$$

It follows that if $n \neq m$ , then $\xi _ { n } \neq \xi _ { m }$

Lemma 17 (Golay-Rudin-Shapiro flatness). For every integer $k > 0 _ { : }$ consider the functions $P _ { k } ^ { \mathrm { G R S } }$ and $Q _ { k } ^ { \mathrm { G R S } }$ generated by the recurrence

$$
P _ { 0 } ^ { \mathrm { G R S } } = Q _ { 0 } ^ { \mathrm { G R S } } = \frac { 1 } { \sqrt { 2 } } ,\tag{E216}
$$

$$
P _ { j } ^ { \mathrm { G R S } } = \frac { P _ { j - 1 } ^ { \mathrm { G R S } } + z _ { j } Q _ { j - 1 } ^ { \mathrm { G R S } } } { \sqrt { 2 } } , \qquad Q _ { j } ^ { \mathrm { G R S } } = \frac { P _ { j - 1 } ^ { \mathrm { G R S } } - z _ { j } Q _ { j - 1 } ^ { \mathrm { G R S } } } { \sqrt { 2 } } ,\tag{E217}
$$

where

$$
z _ { j } ( \alpha ) = \chi _ { \beta _ { j } } ( \alpha ) = e ^ { i \Omega ( \beta _ { j } , \alpha ) } .\tag{E218}
$$

Then the function

$$
h _ { k } ( \alpha ) = | P _ { k } ^ { \mathrm { G R S } } ( \alpha ) | ^ { 2 } - | Q _ { k } ^ { \mathrm { G R S } } ( \alpha ) | ^ { 2 }\tag{E219}
$$

satisfies $| h _ { k } ( \alpha ) | \le 1$ for every $\alpha ,$ and has a Weyl expansion

$$
h _ { k } ( \alpha ) = \sum _ { n \in \{ - 1 , 0 , 1 \} ^ { k } \backslash \{ 0 \} } c _ { n } \chi _ { \xi _ { n } } ( \alpha )\tag{E220}
$$

with coeficient flatness $| c _ { n } | ^ { 2 } \leq C 2 ^ { - k }$ for a universal constant C.

Proof. First, note that at each level of the recurrence, the next level is obtained by a unitary transformation of the current polynomials. Therefore,

$$
| P _ { j } ^ { \mathrm { G R S } } ( \alpha ) | ^ { 2 } + | Q _ { j } ^ { \mathrm { G R S } } ( \alpha ) | ^ { 2 } = | P _ { j - 1 } ^ { \mathrm { G R S } } ( \alpha ) | ^ { 2 } + | Q _ { j - 1 } ^ { \mathrm { G R S } } ( \alpha ) | ^ { 2 }\tag{E221}
$$

for every $j$ and every α. Since the initial sum is one, we have

$$
| P _ { k } ^ { \mathrm { G R S } } ( \alpha ) | ^ { 2 } + | Q _ { k } ^ { \mathrm { G R S } } ( \alpha ) | ^ { 2 } = 1 ,\tag{E222}
$$

which implies $| h _ { k } ( \alpha ) | \le 1$

Next we prove the coeficient bound. Define $G _ { j } = P _ { j } ^ { \mathrm { G R S } } \overline { { Q _ { j } ^ { \mathrm { G R S } } } }$ . A direct expansion of the recurrence gives

$$
h _ { j } = z _ { j } { \overline { { G _ { j - 1 } } } } + z _ { j } ^ { - 1 } G _ { j - 1 }\tag{E223}
$$

and

$$
G _ { j } = \frac { 1 } { 2 } \left( h _ { j - 1 } + z _ { j } \overline { { { G _ { j - 1 } } } } - z _ { j } ^ { - 1 } G _ { j - 1 } \right) .\tag{E224}
$$

For any finite Weyl polynomial F, write

$$
F ( \alpha ) = \sum _ { \xi } { \widehat { F } } ( \xi ) \chi _ { \xi } ( \alpha )\tag{E225}
$$

and define $\operatorname { s u p p } ( F ) = \{ \xi : { \widehat { F } } ( \xi ) \neq 0 \}$ . It will be useful to track supports explicitly. Let

$$
T _ { j } = \left\{ \sum _ { \ell = 1 } ^ { j } n _ { \ell } \beta _ { \ell } : n _ { \ell } \in \{ - 1 , 0 , 1 \} \right\} .\tag{E226}
$$

We claim inductively that

$$
\operatorname { s u p p } ( h _ { j } ) \subseteq T _ { j } , \qquad \operatorname { s u p p } ( G _ { j } ) \subseteq T _ { j } .\tag{E227}
$$

This is immediate for $j = 0$ , since $h _ { 0 } = 0$ and $G _ { 0 } = P _ { 0 } ^ { \mathrm { G R S } } \overline { { Q _ { 0 } ^ { \mathrm { G R S } } } } = 1 / 2$ has support contained in $T _ { 0 } = \{ 0 \}$ . If the claim holds at step $j - 1$ , then multiplication by $z _ { j } = \chi _ { \beta _ { j } }$ shifts Weyl labels $\mathrm { b y } + \beta _ { j }$ , multiplication by $z _ { i } ^ { - 1 }$ shifts them by $- \beta _ { j }$ , and complex conjugation maps a label $\dot { \xi } _ { \mathrm { ~ \scriptsize ~ t o ~ } - \xi }$ . Since the set of points $T _ { j - 1 }$ is preserved under sign flip $\xi \mapsto - \xi .$ the recurrence relations for $h _ { j } , G _ { j }$ imply that the supports at step $j$ are contained in $T _ { j }$

Since all $\beta _ { j }$ are scalar multiples of the fixed unit vector $e ,$ every label in $T _ { j }$ lies on the line spanned by e. Write the scalar coordinate of a label $\xi$ on this line as $t ( \xi )$ , so that $\xi = t ( \xi ) e$ . For labels in $T _ { j - 1 }$

$$
| t ( \xi ) | \leq B \sum _ { \ell = 1 } ^ { j - 1 } 3 ^ { \ell - 1 } = \frac { B ( 3 ^ { j - 1 } - 1 ) } { 2 } .\tag{E228}
$$

Let $R _ { j - 1 } = B ( 3 ^ { j - 1 } - 1 ) / 2$ . Then $T _ { j - 1 }$ is contained in the interval $[ - R _ { j - 1 } , R _ { j - 1 } ]$ along the $e$ direction. The shifted set $\beta _ { j } + T _ { j - 1 }$ is contained in the interval

$$
[ B 3 ^ { j - 1 } - R _ { j - 1 } , B 3 ^ { j - 1 } + R _ { j - 1 } ] ,\tag{E229}
$$

whose left endpoint is

$$
B 3 ^ { j - 1 } - R _ { j - 1 } = { \frac { B ( 3 ^ { j - 1 } + 1 ) } { 2 } } > R _ { j - 1 } .\tag{E230}
$$

Thus $\beta _ { j } + T _ { j - 1 }$ is disjoint from $T _ { j - 1 }$ . Similarly, $, - \beta _ { j } + T _ { j - 1 }$ is contained in an interval whose right endpoint is

$$
- B 3 ^ { j - 1 } + R _ { j - 1 } = - \frac { B ( 3 ^ { j - 1 } + 1 ) } { 2 } < - R _ { j - 1 } ,\tag{E231}
$$

so $- \beta _ { j } + T _ { j - 1 }$ is also disjoint from $T _ { j - 1 }$ . Finally, $\beta _ { j } + T _ { j - 1 }$ and $- \beta _ { j } + T _ { j - 1 }$ are disjoint because the first lies entirely to the right of $R _ { j - 1 }$ and the second lies entirely to the left of $- R _ { j - 1 }$ . Therefore the three sets

$$
T _ { j - 1 } , \qquad \beta _ { j } + T _ { j - 1 } , \qquad - \beta _ { j } + T _ { j - 1 }\tag{E232}
$$

are pairwise disjoint. This disjointness lets us read of coeficient magnitudes. Define

$$
a _ { j } = \operatorname* { m a x } _ { \xi } | \widehat { h } _ { j } ( \xi ) | , \qquad b _ { j } = \operatorname* { m a x } _ { \xi } | \widehat { G } _ { j } ( \xi ) | .\tag{E233}
$$

Multiplication by phases $z _ { j }$ or $z _ { j } ^ { - 1 }$ only shifts the support, not the magnitude, and complex conjugation only changes ${ \widehat { G } } _ { j - 1 } ( \xi )$ to the conjugate of a coeficient at the label −ξ. Thus these operations preserve absolute values of coeficients. Since the two supports in

$$
h _ { j } = z _ { j } { \overline { { G _ { j - 1 } } } } + z _ { j } ^ { - 1 } G _ { j - 1 }\tag{E234}
$$

are disjoint, no two coeficients add at the same Weyl label. Hence $a _ { j } = b _ { j - 1 }$ . Likewise, the three supports in

$$
G _ { j } = \frac { 1 } { 2 } \left( h _ { j - 1 } + z _ { j } \overline { { { G _ { j - 1 } } } } - z _ { j } ^ { - 1 } G _ { j - 1 } \right)\tag{E235}
$$

are disjoint, so every coeficient of $G _ { j }$ is exactly one half of a coeficient coming either from $h _ { j - 1 }$ , from $G _ { j - 1 }$ , or from $\overline { { G _ { j - 1 } } }$ . Therefore

$$
b _ { j } = \frac { 1 } { 2 } \operatorname* { m a x } ( a _ { j - 1 } , b _ { j - 1 } ) .\tag{E236}
$$

The initial values are $\begin{array} { r } { a _ { 0 } = 0 , b _ { 0 } = \frac { 1 } { 2 } } \end{array}$ . We now solve the recurrence on the coeficient magnitudes. We prove by induction that for all $j \geq 0$

$$
a _ { j } \leq 2 ^ { - \lfloor j / 2 \rfloor - 1 } , \qquad b _ { j } \leq 2 ^ { - \lceil j / 2 \rceil - 1 } ,\tag{E237}
$$

where the bound on $a _ { 0 }$ is trivial. Suppose the bounds hold at step $j - 1$ . Then

$$
a _ { j } = b _ { j - 1 } \leq 2 ^ { - \lceil ( j - 1 ) / 2 \rceil - 1 } = 2 ^ { - \lfloor j / 2 \rfloor - 1 } .\tag{E238}
$$

For $b _ { j }$ , we use Equation (E236). If $j = 2 t$ is even, then

$$
a _ { j - 1 } \leq 2 ^ { - t } , \qquad b _ { j - 1 } \leq 2 ^ { - t - 1 } ,\tag{E239}
$$

and hence $b _ { j } \leq 2 ^ { - t - 1 } = 2 ^ { - \lceil j / 2 \rceil - 1 } . \mathrm { ~ I f ~ } j = 2 t + 1$ , then

$$
a _ { j - 1 } \leq 2 ^ { - t - 1 } , \qquad b _ { j - 1 } \leq 2 ^ { - t - 1 } ,\tag{E240}
$$

and hence $b _ { j } \leq 2 ^ { - t - 2 } = 2 ^ { - \lceil j / 2 \rceil - 1 }$ . This completes the induction.

Therefore every Weyl coeficient $c _ { n }$ of $h _ { k }$ satisfies $| c _ { n } | \leq a _ { k } \leq 2 ^ { - \lfloor k / 2 \rfloor - 1 }$ . Squaring gives

$$
| c _ { n } | ^ { 2 } \leq 2 ^ { - 2 \lfloor k / 2 \rfloor - 2 } \leq 2 ^ { - k } .\tag{E241}
$$

This proves the desired flatness bound.

Finally, we record that the function just constructed is exactly realized by a depth-k measurement circuit. This will be used later in Section F 6 to give an algorithm for the hypothesis testing problem, completing the exponential depth-based separation.

Lemma 18 (Depth-k realization of the Golay-Rudin-Shapiro kernel). There is a depth-k circuit whose final binary qubit measurement produces an outcome $Y \in \{ - 1 , 1 \}$ satisfying

$$
\mathbb { E } [ Y \mid \alpha ] = h _ { k } ( \alpha ) .\tag{E242}
$$

Proof. We now show that the same recurrence is exactly realized by a depth-k phase-kick circuit built from controlled displacements and single-qubit rotations. We first analyze one controlled displacement, built from ECD gates compatible with our hardware $\left( \mathrm { F i g . ~ 3 } \right)$ , carefully. Let

$$
C ( \beta ) = D ( \beta / 2 ) \otimes | g \rangle \langle g | + D ( - \beta / 2 ) \otimes | e \rangle \langle e | .\tag{E243}
$$

With our ECD convention,

$$
C ( \beta ) = \operatorname { E C D } ( \beta ) ( I \otimes X _ { q } ) ,\tag{E244}
$$

where $X _ { q } = | g \rangle \langle e | + | e \rangle \langle g |$ . Thus $C ( \beta )$ is obtained from one ECD gate and a fixed qubit bit flip, and we count it as one control layer.

Fix an arbitrary oscillator state |ψ⟩ and an arbitrary qubit state $P \left| g \right. + Q \left| e \right.$ . Before the signal displacement, applying $C ( \beta )$ gives

$$
C ( \beta ) \left| \psi \right. ( P \left| g \right. + Q \left| e \right. ) = P D ( \beta / 2 ) \left| \psi \right. \left| g \right. + Q D ( - \beta / 2 ) \left| \psi \right. \left| e \right. .\tag{E245}
$$

After the signal displacement D(α) acts on the oscillator, the state becomes

$$
P D ( \alpha ) D ( \beta / 2 ) \left| \psi \right. \left| g \right. + Q D ( \alpha ) D ( - \beta / 2 ) \left| \psi \right. \left| e \right. .\tag{E246}
$$

Finally we uncompute the controlled displacement by applying $C ( \beta ) ^ { \dagger }$ . Since

$$
C ( \beta ) ^ { \dagger } = D ( - \beta / 2 ) \otimes | g \rangle \langle g | + D ( \beta / 2 ) \otimes | e \rangle \langle e | ,\tag{E247}
$$

the final state is

$$
P D ( - \beta / 2 ) D ( \alpha ) D ( \beta / 2 ) \left| \psi \right. \left| g \right. + Q D ( \beta / 2 ) D ( \alpha ) D ( - \beta / 2 ) \left| \psi \right. \left| e \right. .\tag{E248}
$$

Using the Weyl relation $D ( a ) D ( b ) = e ^ { - i \Omega ( a , b ) / 2 } D ( a + b )$ , we have

$$
D ( - \beta / 2 ) D ( \alpha ) D ( \beta / 2 ) = e ^ { i \Omega ( \beta , \alpha ) / 2 } D ( \alpha ) ,\tag{E249}
$$

$$
D ( \beta / 2 ) D ( \alpha ) D ( - \beta / 2 ) = e ^ { - i \Omega ( \beta , \alpha ) / 2 } D ( \alpha ) .\tag{E250}
$$

Therefore the final state is

$$
D ( \alpha ) \left| \psi \right. \left( P e ^ { i \Omega ( \beta , \alpha ) / 2 } \left| g \right. + Q e ^ { - i \Omega ( \beta , \alpha ) / 2 } \left| e \right. \right) .\tag{E251}
$$

After factoring out the global phase $e ^ { i \Omega ( \beta , \alpha ) / 2 }$ , which has no efect on measurement probabilities, the qubit state is

$$
P \left| g \right. + Q e ^ { - i \Omega ( \beta , \alpha ) } \left| e \right. .\tag{E252}
$$

Thus the echoed controlled displacement $C ( \beta ) ^ { \dagger } D ( \alpha ) C ( \beta )$ implements a qubit phase kick with relative phase $e ^ { - i \Omega \left( \beta , \alpha \right) }$ , while leaving only a common oscillator displacement $D ( \alpha )$ . Equivalently, using $C ( - \beta )$ implements the phase kick with relative phase $e ^ { i \Omega ( \beta , \alpha ) }$ . In the recurrence below we use $C ( - \beta _ { j } )$ so that the phase variable is

$$
z _ { j } ( \alpha ) = e ^ { i \Omega ( \beta _ { j } , \alpha ) } .\tag{E253}
$$

We now iterate this primitive. Let $| \eta _ { j } ( \alpha ) \rangle$ denote the oscillator state after the first $j$ echoed phase-kick primitives. Its exact form is irrelevant, because at every step the oscillator factor is common to both qubit branches and therefore does not afect the final qubit probabilities. We prove by induction that after $j$ rounds the joint state has the product form

$$
\left| \eta _ { j } ( \alpha ) \right. \left( P _ { j } ^ { \mathrm { G R S } } ( \alpha ) \left| g \right. + Q _ { j } ^ { \mathrm { G R S } } ( \alpha ) \left| e \right. \right) ,\tag{E254}
$$

where $P _ { i } ^ { \mathrm { G R S } }$ and $Q _ { i } ^ { \mathrm { G R S } }$ obey the Golay-Rudin-Shapiro recurrence.

For $j \doteq 0$ , take $| \dot { \eta _ { 0 } } \rangle = | \psi \rangle$ and prepare the qubit state

$$
\frac { 1 } { \sqrt { 2 } } \left| g \right. + \frac { 1 } { \sqrt { 2 } } \left| e \right. .\tag{E255}
$$

Thus $\begin{array} { r } { P _ { 0 } ^ { \mathrm { G R S } } = Q _ { 0 } ^ { \mathrm { G R S } } = \frac { 1 } { \sqrt { 2 } } } \end{array}$ . Assume the claim holds after $j - 1$ rounds, so the state is

$$
\left| \eta _ { j - 1 } ( \alpha ) \right. \left( P _ { j - 1 } ^ { \mathrm { G R S } } ( \alpha ) \left| g \right. + Q _ { j - 1 } ^ { \mathrm { G R S } } ( \alpha ) \left| e \right. \right) .\tag{E256}
$$

Apply the echoed controlled displacement using $C ( - \beta _ { j } )$ . By the single-layer calculation above, this maps the state to

$$
\left| \eta _ { j } ^ { \prime } ( \alpha ) \right. \left( P _ { j - 1 } ^ { \mathrm { G R S } } ( \alpha ) \left| g \right. + z _ { j } ( \alpha ) Q _ { j - 1 } ^ { \mathrm { G R S } } ( \alpha ) \left| e \right. \right) ,\tag{E257}
$$

where $\left| \eta _ { j } ^ { \prime } ( \alpha ) \right.$ is a common oscillator state and $z _ { j } ( \alpha ) = e ^ { i \Omega ( \beta _ { j } , \alpha ) }$ . Now apply the Hadamard rotation

$$
H = \frac { 1 } { \sqrt { 2 } } \left( \begin{array} { c c } { 1 } & { 1 } \\ { 1 } & { - 1 } \end{array} \right)\tag{E258}
$$

in the ordered basis $\left( \left| g \right. , \left| e \right. \right)$ . Since $H \left| g \right. = ( \left| g \right. + \left| e \right. ) / { \sqrt { 2 } }$ and $H \left| e \right. = ( \left| g \right. - \left| e \right. ) / { \sqrt { 2 } } .$ , the state becomes

$$
| \eta _ { j } ( \alpha ) \rangle \left( \frac { P _ { j - 1 } ^ { \mathrm { G R S } } ( \alpha ) + z _ { j } ( \alpha ) Q _ { j - 1 } ^ { \mathrm { G R S } } ( \alpha ) } { \sqrt { 2 } } | g \rangle + \frac { P _ { j - 1 } ^ { \mathrm { G R S } } ( \alpha ) - z _ { j } ( \alpha ) Q _ { j - 1 } ^ { \mathrm { G R S } } ( \alpha ) } { \sqrt { 2 } } | e \rangle \right) .\tag{E259}
$$

Thus the new qubit amplitudes satisfy

$$
P _ { j } ^ { \mathrm { G R S } } = \frac { P _ { j - 1 } ^ { \mathrm { G R S } } + z _ { j } Q _ { j - 1 } ^ { \mathrm { G R S } } } { \sqrt { 2 } } , \qquad Q _ { j } ^ { \mathrm { G R S } } = \frac { P _ { j - 1 } ^ { \mathrm { G R S } } - z _ { j } Q _ { j - 1 } ^ { \mathrm { G R S } } } { \sqrt { 2 } } .\tag{E260}
$$

This is exactly the Golay-Rudin-Shapiro recurrence. The induction is complete. After k rounds, the state has the form

$$
\left| \eta _ { k } ( \alpha ) \right. \left( P _ { k } ^ { \mathrm { G R S } } ( \alpha ) \left| g \right. + Q _ { k } ^ { \mathrm { G R S } } ( \alpha ) \left| e \right. \right) .\tag{E261}
$$

Measure the qubit in the computational basis and define $Y = 1$ for outcome $| g \rangle$ and $Y = - 1$ for outcome $| e \rangle$ Since the oscillator factor is common to both branches, tracing it out does not change the qubit probabilities. Hence

$$
\operatorname* { P r } [ Y = 1 \mid \alpha ] = | P _ { k } ^ { \mathrm { G R S } } ( \alpha ) | ^ { 2 }\tag{E262}
$$

and

$$
\mathrm { P r } [ Y = - 1 \mid \alpha ] = \vert Q _ { k } ^ { \mathrm { G R S } } ( \alpha ) \vert ^ { 2 } .\tag{E263}
$$

Therefore

$$
\operatorname { \mathbb { E } } [ Y \mid \alpha ] = \operatorname* { P r } [ Y = 1 \mid \alpha ] - \operatorname* { P r } [ Y = - 1 \mid \alpha ]\tag{E264}
$$

$$
= | P _ { k } ^ { \mathrm { G R S } } ( \alpha ) | ^ { 2 } - | Q _ { k } ^ { \mathrm { G R S } } ( \alpha ) | ^ { 2 }\tag{E265}
$$

$$
h _ { k } ( \alpha ) .\tag{E266}
$$

Thus the depth-k phase-kick circuit realizes exactly the desired measurement response.

## b. Proof of Theorem E.13

Now we have all the ingredients to prove the control-depth lower bound.

Proof. As usual, we reduce the learning task to hypothesis testing. Any algorithm for estimation of $\mathbb { E } [ f ( \alpha ) ]$ for f in a fixed class of functions F and for general signals can distinguish two signals $P _ { 0 } , P _ { 1 }$ for which for some $f \in \mathcal { F } , | \mathbb { E } _ { P _ { 0 } } [ f ( \alpha ) ] - \mathbb { E } _ { P _ { 1 } } [ f ( \alpha ) ] | = \Theta ( 1 )$ . By demonstrating functions which no depth $< \gamma k$ algorithm can distinguish with subexponential signal queries, we establish a learning lower bound.

Let $P _ { 0 }$ be the centered Gaussian distribution on R<sup>2</sup> with covariance $\sigma ^ { 2 } I$ , and let W be the centered Weyl dictionary associated to $P _ { 0 }$ . Fix a constant $0 < \eta \leq 1 / 4$ and let $h _ { k }$ be the Golay-Rudin-Shapiro kernel given in Lemma 17. Define

$$
g _ { k } ( \alpha ) = h _ { k } ( \alpha ) - \int h _ { k } ( \alpha ) d P _ { 0 } ( \alpha ) .\tag{E267}
$$

Then define two signals induced by the distributions

$$
P _ { \pm } ^ { ( k ) } ( \alpha ) = ( 1 \pm \eta g _ { k } ( \alpha ) ) P _ { 0 } ( \alpha ) .\tag{E268}
$$

These two hypotheses are valid probability distributions. By Lemma 17, $| h _ { k } | \leq 1$ , and so $\| g _ { k } \| _ { \infty } \leq 2$ . Because $\eta \leq 1 / 4$

$$
1 \pm \eta g _ { k } ( \alpha ) \geq \frac { 1 } { 2 }\tag{E269}
$$

for every α. Also, by definition of $g _ { k }$

$$
\int g _ { k } ( \alpha ) d P _ { 0 } ( \alpha ) = 0 .\tag{E270}
$$

Therefore $P _ { \pm } ^ { ( k ) }$ are valid probability distributions. $\mathrm { N e x t }$ , for $r \in \mathbb { Z } ^ { + }$ , let M be any single-query measurement in $\mathcal { M } _ { r } ,$ and let $f _ { M }$ be its acceptance response. By Lemma 16, the centered response satisfies

$$
f _ { M } - \int f _ { M } d P _ { 0 } \in \mathrm { s p a n } \{ \widetilde { \chi } _ { \zeta } : \zeta \in S _ { M } \}\tag{E271}
$$

for some set $S _ { M }$ with $| S _ { M } | \le 3 ^ { r }$ . Thus every measurement in $\mathcal { M } _ { r }$ is $3 ^ { r }$ -sparse in the centered $\mathrm { W e y l }$ dictionary W. Moreover, from Lemma 17 we have that each Fourier coeficient $c _ { n }$ of $g _ { k }$ in the Weyl dictionary satisfies $| c _ { n } | ^ { 2 } \leq C 2 ^ { - k }$ for some universal constant $C > 0$ . Therefore we have that

$$
{ \mathsf { C } } _ { 3 ^ { r } } ( g _ { k } ; \mathcal { W } , P _ { 0 } ) \le C _ { 0 } 3 ^ { r } 2 ^ { - k }\tag{E272}
$$

By the semiclassical support-counting Corollary 25 with

$$
h = g _ { k } , \qquad L = 3 ^ { r } , \qquad \mathcal { D } = \mathcal { W } .\tag{E273}
$$

we obtain that any adaptive protocol using measurements from $\mathcal { M } _ { r }$ requires

$$
N \geq \Omega \left( \eta ^ { - 2 } \left( C 3 ^ { r } 2 ^ { - k } \right) ^ { - 1 } \right) = \Omega ( 2 ^ { k } / 3 ^ { r } ) .\tag{E274}
$$

For $r < k \log _ { 3 } 2$ , this is $\exp ( \Omega ( k ) )$ ).

## Appendix F: Exponential advantage in learning signals with Quantum Feature Sensing

Here we present the quantum-enhanced protocols and their sample complexity analyses that complete our exponential separations. We discuss our results through the Quantum Phase-Space Inference framework from Section D, illustrating how each sensing protocol operationally maximizes the relevant Fourier overlap quantity under constraints, and how the upper bounds emerge from the same considerations as their respective lower bounds.

Our results imply an important operational message. Modern continuous-variable quantum sensors are extremely mature technologies that operate at the forefront of modern engineering, and can already achieve measurement precision orders of magnitude below the classical floor, making further improvements ever more challenging and increasingly incremental. Each of our separations, however, considers taking a conventional sensor and adding an auxiliary resource in the form of quantum control. That is, our exponential sensing gains are enabled entirely by quantum information processing technology already present today, without requiring any precision improvement to the sensor itself. As such, these developments suggest a new paradigm in which development of sensing technologies can be accelerated by focusing on architectures which couple to rudimentary quantum processors.

## 1. Learning with quantum probes without quantum control

We begin with the first separation in the hierarchy, where the learner may use nonclassical probe states but performs no coherent quantum control during the sensing interaction. This puts us in the class of conventional quantum sensors which can prepare Gaussian probe states. We will first utilize the estimator from Theorem D.27 and a lower bound of the AFI (Definition 12) to immediately establish an exponential advantage for the simpler hypothesis testing task. Then, we will give an estimator for the broader angular Fourier learning task and bound its sample complexity.

## a. Fixed-amplitude learning with squeezed probes

First we recall the QΨ hypothesis testing problem used to establish an exponential lower bound against coherent probe protocols in Section E 1. Consider a displacement distribution P supported on the circle $\bar { \{ A e ^ { i \theta } \} }$ $\theta \in [ 0 , 2 \pi ) \}$ , for fixed $A = \Theta ( { \sqrt { k } } )$ . We let $\Lambda _ { k } ( P ) = \bar { \mathbb { E } } _ { \theta \sim F } \left[ e ^ { i k \theta } \right]$ denote the k-th angular Fourier coeficient. Then the two hypothesis distributions used in the coherent-probe lower bound,

$$
T _ { \pm } ^ { ( k ) } ( \theta ) = \frac { 1 } { 2 \pi } \left( 1 \pm 0 . 9 \cos ( k \theta ) \right) ,\tag{F1}
$$

we have $\Lambda _ { k } ( P _ { \pm } ^ { ( k ) } ) = \pm 0 . 4 5$ , where $P _ { \pm }$ is the pushforward of $T _ { \pm }$ under the map $\theta  A e ^ { i \theta }$

The class of response functions at hand now corresponds to the set of Gaussian sensing protocols. To utilize Theorem D.27, we first compute the Gaussian protocol response kernels. For simplicity, consider the response upon performing standard squeezed homodyne detection. Let r be the squeezing parameter and set $\nu = e ^ { - 2 r } / 2$ If the signal applies the displacement $D ( A e ^ { i \theta } )$ and we perform a squeezed homodyne measurement along angle $\varphi ,$ the outcome has the form $Y _ { \varphi } = R \cos ( \theta - \varphi ) + G$ , where $G \sim { \mathcal { N } } ( 0 , \nu )$ . Let $\begin{array} { r } { g _ { \nu } ( y ) = \frac { 1 } { \sqrt { 2 \pi \nu } } \exp { \left( - \frac { y ^ { 2 } } { 2 \nu } \right) } } \end{array}$ ; under the reference distribution $P _ { 0 } ( d \theta ) = d \theta / 2 \pi$ and perturbation $h _ { k } ( \theta ) = 0 . 9 \cos ( k \theta )$ , we have

$$
q _ { 0 , k } ( y ) = \frac { 1 } { 2 \pi } \int _ { 0 } ^ { 2 \pi } g _ { \nu } ( y - A \sqrt { 2 } \cos \theta ) d \theta\tag{F2}
$$

$$
a _ { h , k } ( y ) = \frac { 0 . 9 } { 2 \pi } \int _ { 0 } ^ { 2 \pi } \cos ( k \theta ) g _ { \nu } ( y - A \sqrt { 2 } \cos \theta ) d \theta\tag{F3}
$$

$$
R _ { h , k } ( y ) = \frac { a _ { h , k } ( y ) } { q _ { 0 , k } ( y ) } \ .\tag{F4}
$$

These are the quantities used in the AFI-optimal estimator for Theorem D.27. Now we show that the AFI of this protocol saturates the achievable AFI for the given hypothesis test among any conventional Gaussian protocol, up to constants.

Lemma 1. Let $M _ { \nu , \phi }$ denote the squeezed homodyne measurement strategy with squeezing strength ν and homodyne angle ϕ. For the task of distinguishing displacement distributions $P _ { \pm }$ above with $A = C { \sqrt { k } }$ and C a constant, there exist constants $C _ { 1 } , C _ { 2 }$ such that

$$
\mathsf { A } _ { M _ { \nu , 0 } } ( h _ { k } ; P _ { 0 } ) \ge C _ { 1 }\tag{F5}
$$

for any $\nu \leq C _ { 2 } / k$ . This is optimal up to constants for any measurement strategy.

Proof. By definition of AFI for the measurement $M _ { \nu , 0 }$

$$
\mathsf { A } _ { M _ { \nu , 0 } } ( h _ { k } ; P _ { 0 } ) = \int _ { \mathbb { R } } \frac { a _ { h , k } ( y ) ^ { 2 } } { q _ { 0 , k } ( y ) } d y .\tag{F6}
$$

Let $R = A { \sqrt { 2 } } ,$ , let Θ be uniformly distributed on $[ 0 , 2 \pi )$ , let $G \sim { \mathcal { N } } ( 0 , \nu )$ be independent of Θ, and set

$$
X = R \cos \Theta , \qquad Y = X + G .\tag{F7}
$$

Then $q _ { 0 , k }$ is the density of Y , and Bayes’ rule gives

$$
\frac { a _ { k } ( y ) } { q _ { 0 , k } ( y ) } = \mathbb { E } [ \cos ( k \Theta ) | Y = y ] .\tag{F8}
$$

Consequently,

$$
\int _ { \mathbb { R } } \frac { a _ { k } ( y ) ^ { 2 } } { q _ { 0 , k } ( y ) } d y = \mathbb { E } \left[ \mathbb { E } [ \cos ( k \Theta ) | Y ] ^ { 2 } \right] .\tag{F9}
$$

By Cauchy-Schwarz, for any $f : \mathbb { R } \to \mathbb { R }$ with bounded variance,

$$
\mathbb { E } \left[ \mathbb { E } [ \cos ( k \Theta ) | Y ] ^ { 2 } \right] \geq \frac { \left| \mathbb { E } [ f ( Y ) \cos ( k \Theta ) ] \right| ^ { 2 } } { \mathbb { E } [ f ( Y ) ^ { 2 } ] } .\tag{F10}
$$

We now choose a test function which reconstructs cos(kΘ) from the noiseless homodyne outcome. Let $T _ { k }$ be the k-th Chebyshev polynomial, and define

$$
\tau _ { k } ( y ) = T _ { k } \left( \operatorname* { m a x } \left\{ - 1 , \operatorname* { m i n } \left\{ 1 , \frac { y } { R } \right\} \right\} \right) .\tag{F11}
$$

Since $| T _ { k } ( x ) | \leq 1$ for $x \in [ - 1 , 1 ]$ , we have $| \tau _ { k } ( y ) | \le 1$ for all y. Moreover, because $X = R \cos \Theta$

$$
\tau _ { k } ( X ) = T _ { k } ( \cos \Theta ) = \cos ( k \Theta ) .\tag{F12}
$$

Therefore,

$$
\mathbb { E } [ \tau _ { k } ( X ) \cos ( k \Theta ) ] = \mathbb { E } [ \cos ^ { 2 } ( k \Theta ) ] = \frac { 1 } { 2 } .\tag{F13}
$$

This is not yet what we want, as we have to show that upon replacing X with Y by adding Gaussian noise, this expected value remains large. Fix a small universal constant $\Delta _ { 0 } \in ( 0 , 1 / 1 0 )$ , to be chosen below. Define the two bad events

$$
\begin{array} { r } { \mathcal { B } = \{ | \sin \Theta | \le \Delta _ { 0 } \} , \qquad \mathcal { C } = \{ | G | > R \Delta _ { 0 } ^ { 2 } / 4 \} . } \end{array}\tag{F14}
$$

The event B is the region where the map $\theta \mapsto \cos \theta$ approaches a turning point, while C is the event where the homodyne noise is large. For appropriately chosen $\Delta _ { 0 }$ , these bad events contain the cases in $\tau _ { k } ( Y ) \cos ( k \Theta )$ becomes too small compared to $\tau _ { k } ( X )$ cos(kΘ). Since Θ is uniform and sin $. ( x ) \leq x , \mathrm { P r } ( B ) \leq 2 \Delta _ { 0 }$ . Also, by a standard Gaussian tail bound,

$$
\mathrm { P r } ( \mathcal { C } ) \leq 2 \exp \left( - \frac { R ^ { 2 } \Delta _ { 0 } ^ { 4 } } { 3 2 \nu } \right) .\tag{F15}
$$

We next control the Lipschitz constant of $\tau _ { k }$ on the complement of these events. On $\boldsymbol { B ^ { c } }$ , note that | sin $\Theta | > \Delta _ { 0 }$

implies | cos $\Theta | \le \sqrt { 1 - \Delta _ { 0 } ^ { 2 } }$ . As such on $B ^ { c }$ we have

$$
\left| \frac { X } { R } \right| = | \cos \Theta | \leq \sqrt { 1 - \Delta _ { 0 } ^ { 2 } } \leq 1 - \frac { \Delta _ { 0 } ^ { 2 } } { 2 } .\tag{F16}
$$

Moreover on $\mathcal { C } ^ { c } , \vert G \vert / R \le \Delta _ { 0 } ^ { 2 } / 4$ , so we also have $| Y / R - X / R | \le \Delta _ { 0 } ^ { 2 } / 4$ . Hence every point z between $X / R$ and $Y / R$ on this complement region satisfies $| z | \leq 1 - \Delta _ { 0 } ^ { 2 } / 4$ . Within the complement region, due to our choice of normalization, $\tau _ { k } = T _ { k }$ . Moreover, the Chebyshev derivative satisfies $T _ { k } ^ { \prime } ( z ) = k U _ { k - 1 } ( z )$ , where $U _ { k - 1 }$ is the Chebyshev polynomial of the second kind. Writing $z = \cos v ,$ it holds that $U _ { k - 1 } = \sin ( k v ) / \sin ( v )$ , so we have the bound

$$
| U _ { k - 1 } ( z ) | \leq \frac { 1 } { \sqrt { 1 - z ^ { 2 } } } , \qquad z \in ( - 1 , 1 ) .\tag{F17}
$$

This then gives us $\begin{array} { r } { | T _ { k } ^ { \prime } ( z ) | \le \frac { k } { \sqrt { 1 - z ^ { 2 } } } } \end{array}$ . Since $\left. z \right. \leq 1 - \Delta _ { 0 } ^ { 2 } / 4$ , we have

$$
1 - z ^ { 2 } \geq 1 - \left( 1 - \frac { \Delta _ { 0 } ^ { 2 } } { 4 } \right) ^ { 2 }\tag{F18}
$$

$$
= \frac { \Delta _ { 0 } ^ { 2 } } { 2 } - \frac { \Delta _ { 0 } ^ { 4 } } { 1 6 }\tag{F19}
$$

$$
\geq { \frac { \Delta _ { 0 } ^ { 2 } } { 4 } } ,\tag{F20}
$$

where the last inequality uses $\Delta _ { 0 } \leq 1$ . Therefore

$$
| T _ { k } ^ { \prime } ( z ) | \leq \frac { 2 k } { \Delta _ { 0 } } .\tag{F21}
$$

Combining this with $| Y / R - X / R | = | G | / R$ , we find that on $( B \cup { \mathcal { C } } ) ^ { c }$ ，

$$
| \tau _ { k } ( Y ) - \tau _ { k } ( X ) | \leq \frac { 2 k } { R \Delta _ { 0 } } | G | ,\tag{F22}
$$

so the function $\tau _ { k }$ is $2 k / R \Delta _ { 0 ^ { - } }$ Lipschitz on the complement region. On the bad event $B \cup { \mathcal { C } } ,$ we use the trivial bound $| \tau _ { k } ( Y ) - \tau _ { k } ( X ) | \leq 2$ . Therefore,

$$
\mathbb { E } [ | \tau _ { k } ( Y ) - \tau _ { k } ( X ) | ] \leq 2 \operatorname* { P r } ( \mathcal { B } ) + 2 \operatorname* { P r } ( \mathcal { C } ) + \frac { 2 k } { R \Delta _ { 0 } } \mathbb { E } [ | G | ]\tag{F23}
$$

$$
\leq 4 \Delta _ { 0 } + 4 \exp \left( - \frac { R ^ { 2 } \Delta _ { 0 } ^ { 4 } } { 3 2 \nu } \right) + \frac { 2 k \sqrt { \nu } } { R \Delta _ { 0 } } .\tag{F24}
$$

Now choose $\Delta _ { 0 } > 0$ small enough so that the first term is at most $1 / 1 2$ . Then choose a universal constant $c _ { 0 } > 0$ small enough so that whenever

$$
\nu \leq c _ { 0 } \frac { R ^ { 2 } } { k ^ { 2 } } ,\tag{F25}
$$

the last two terms together are at most $1 / 6$ . With this choice, we have $\mathbb { E } [ | \tau _ { k } ( Y ) - \tau _ { k } ( X ) | ] \le 1 / 4$ . It follows that

$$
| \mathbb { E } [ \tau _ { k } ( Y ) \cos ( k \Theta ) ] | \geq | \mathbb { E } [ \tau _ { k } ( X ) \cos ( k \Theta ) ] | - \mathbb { E } [ | \tau _ { k } ( Y ) - \tau _ { k } ( X ) | ]\tag{F26}
$$

$$
\geq { \frac { 1 } { 2 } } - { \frac { 1 } { 4 } } = { \frac { 1 } { 4 } } .\tag{F27}
$$

Since $\mathbb { E } [ \tau _ { k } ( Y ) ^ { 2 } ] \leq 1$ , plugging this into Equation F10 gives us

$$
\int _ { \mathbb { R } } \frac { a _ { k } ( y ) ^ { 2 } } { q _ { 0 , k } ( y ) } d y \geq \frac { 1 } { 1 6 } .\tag{F28}
$$

With constants included from $h _ { k } ( \theta ) = 0 . 9 \cos ( k \theta )$ , we obtain

$$
\mathsf { A } _ { M _ { \nu , 0 } } ( h _ { k } ; P _ { 0 } ) \ge \frac { 0 . 9 ^ { 2 } } { 1 6 } .\tag{F29}
$$

Thus we may take $C _ { 1 } = 0 . 9 ^ { 2 } / 1 6$ . Finally, since $A = C { \sqrt { k } }$ and $R = A { \sqrt { 2 } }$ , the condition $\nu \leq c _ { 0 } R ^ { 2 } / k ^ { 2 }$ becomes

$$
\nu \leq \frac { 2 c _ { 0 } C ^ { 2 } } { k } .\tag{F30}
$$

Taking $C _ { 2 } = 2 c _ { 0 } C ^ { 2 }$ proves the claimed statement. We also note that this constant lower bound is optimal up to universal factors, since AFI satisfies

$$
\mathsf { A } _ { M } ( h _ { k } ; P _ { 0 } ) \leq \| h _ { k } \| _ { L ^ { 2 } ( P _ { 0 } ) } ^ { 2 } = \frac { 0 . 9 ^ { 2 } } { 2 }\tag{F31}
$$

for any measurement M.

Through the QΨ framework, this result gives us our first exponential separation.

Corollary 2 (Exponential advantage of conventional quantum sensing over coherent classical probes). There is a squeezed-probe protocol with mean photon number O(k) per query which distinguishes $P _ { + } ^ { ( k ) }$ from $P _ { - } ^ { ( k ) }$ with O(1) queries and constant success probability. In contrast, any coherent-probe protocol, regardless of energy, requires exp(Ω(k)) queries to distinguish the same two hypotheses with constant success probability.

Proof. The squeezed homodyne protocol $M _ { \nu , 0 }$ , using energy O(k) (due to $\nu \leq O ( 1 / k ) )$ , has AFI Ω(1). By applying the estimator from Theorem D.27, the QΨ guarantee than gives a constant success probability using O(1) samples. The coherent-probe lower bound is Theorem E.1. □

This result already enables a rigorous exponential separation. However, the given protocol is specialized to the stated hypotheses testing problem. More broadly, it is practically useful to be able to estimate general k-th angular Fourier coeficients. We now give an optimal Gaussian protocol for this more general task.

In Section D we introduced the Fourier gain and Theorem D.30 to handle the setting of global learning; we now switch from the local AFI language to this framework. On the fixed-amplitude circle, the relevant Fourier dictionary is

$$
\chi _ { \ell } ( \theta ) = e ^ { i \ell \theta } , \qquad \ell \in \mathbb { Z } .\tag{F32}
$$

Let $\mathcal { A } _ { \mathrm { s q } } ( A , \nu )$ denote the class of randomized squeezed-homodyne protocols at fixed amplitude A and homodyne noise variance $\nu ,$ together with arbitrary classical postprocessing. Its Fourier gain (introduced before Theorem D.30) at angular frequency ℓ is $G ^ { A , \nu } ( \ell ) \stackrel { \cdot } { = } G _ { \mathcal { A } _ { \mathrm { s q } } ( A , \nu ) } ( \ell )$ . The goal is to show that $\dot { G } _ { \mathrm { s q } } ^ { A , \nu } ( k )$ is constant once the squeezed quadrature has enough resolution to see one period of the k-th angular fringe.

For the remainder of this subsection, we work with the angular response

$$
a _ { \ell } ( y ) = \frac { 1 } { 2 \pi } \int _ { 0 } ^ { 2 \pi } e ^ { - i \ell u } g _ { \nu } ( y - R \cos u ) d u ,\tag{F33}
$$

defined for each frequency $\ell \geq 1$ . Note that the replacement of the cosine with an exponential does not change the response because the sine term vanishes by the symmetry $u \mapsto - u$ . Finally define

$$
I ( \ell ; A , \nu ) = \int _ { \mathbb { R } } \frac { a _ { \ell } ( y ) ^ { 2 } } { q _ { 0 } ( y ) } d y .\tag{F34}
$$

If we treated $e ^ { - i \ell u }$ as a perturbation on top of a uniform background and asked about hypothesis testing, this would be exactly the AFI. Here we use a diferent symbol to conceptually diferentiate from hypothesis testing, as we are concerned with global learning of angular Fourier coeficients.

Lemma 3 (Squeezed homodyne certifies angular Fourier gain). For every $\ell \geq 1$

$$
G ^ { A , \nu } ( \ell ) \geq I ( \ell ; A , \nu ) .\tag{F35}
$$

Moreover, there exist universal constants $c _ { 0 } , c _ { 1 } > 0$ such that $I ( \ell ; A , \nu ) \ge c _ { 1 }$ whenever $\nu \leq c _ { 0 } \frac { A ^ { 2 } } { \ell ^ { 2 } }$

Proof. To show a Fourier gain lower bound in the broader Gaussian sensing class, we explicitly provide a protocol with Fourier gain exceeding this bound. The protocol draws $\varphi$ uniformly from [0, 2π), performs squeezed homodyne along angle $\varphi ,$ obtains $Y = R \cos ( \theta - \varphi ) + G$ , and outputs the unnormalized score

$$
W _ { \ell } = e ^ { i \ell \varphi } { \frac { a _ { \ell } ( Y ) } { q _ { 0 } ( Y ) } } .\tag{F36}
$$

We now compute its response function and show that it is peaked at the Fourier character labeled by ℓ. Conditioning on a deterministic angle $\theta ,$

$$
\mathbb { E } [ W _ { \ell } | \theta ] = \int _ { 0 } ^ { 2 \pi } \frac { d \varphi } { 2 \pi } \int _ { \mathbb { R } } d y \ e ^ { i \ell \varphi } \frac { a _ { \ell } ( y ) } { q _ { 0 } ( y ) } g _ { \nu } ( y - R \cos ( \theta - \varphi ) ) .\tag{F37}
$$

Make the change of variables $u = \theta - \varphi$ . Then $e ^ { i \ell \varphi } = e ^ { i \ell \theta } e ^ { - i \ell u }$ , so

$$
\mathbb { E } [ W _ { \ell } | \theta ] = e ^ { i \ell \theta } \int _ { \mathbb { R } } d y \frac { a _ { \ell } ( y ) } { q _ { 0 } ( y ) } \frac { 1 } { 2 \pi } \int _ { 0 } ^ { 2 \pi } d u e ^ { - i \ell u } g _ { \nu } ( y - R \cos u )\tag{F38}
$$

$$
= e ^ { i \ell \theta } \int _ { \mathbb { R } } \frac { a _ { \ell } ( y ) ^ { 2 } } { q _ { 0 } ( y ) } d y\tag{F39}
$$

$$
= I ( \ell ; A , \nu ) e ^ { i \ell \theta } .\tag{F40}
$$

Thus the response function of this one-shot protocol is

$$
f _ { W _ { \ell } } ( \theta ) = I ( \ell ; A , \nu ) \chi _ { \ell } ( \theta ) ,\tag{F41}
$$

so its ℓ-th Fourier coeficient is precisely ${ \widehat { f } } _ { W _ { \ell } } ( \ell ) = I ( \ell ; A , \nu )$ . In words, the Fourier support of the measurement class on the desired coeficient is precisely the AFI quantity $I ( \ell ; A , \nu )$ . To obtain the Fourier gain and use Theorem D.30, we require one more piece: the variance of the protocol’s readout. This is essential because the Fourier gain is defined as the Fourier mass per unit readout variance. Again conditioning on $\theta ,$

$$
\mathbb { E } [ | W _ { \ell } | ^ { 2 } | \theta ] = \int _ { 0 } ^ { 2 \pi } \frac { d \varphi } { 2 \pi } \int _ { \mathbb { R } } d y \ \frac { a _ { \ell } ( y ) ^ { 2 } } { q _ { 0 } ( y ) ^ { 2 } } g _ { \nu } ( y - R \cos ( \theta - \varphi ) ) .\tag{F42}
$$

Averaging the conditional density over the uniformly random homodyne angle gives $q _ { 0 } ( y )$

$$
\int _ { 0 } ^ { 2 \pi } \frac { d \varphi } { 2 \pi } g _ { \nu } ( y - R \cos ( \theta - \varphi ) ) = q _ { 0 } ( y ) .\tag{F43}
$$

Therefore

$$
\begin{array} { r l } & { \mathbb { E } [ | W _ { \ell } | ^ { 2 } | \theta ] = \displaystyle \int _ { \mathbb { R } } \frac { a _ { \ell } ( y ) ^ { 2 } } { q _ { 0 } ( y ) } d y } \\ & { \quad \quad \quad = I ( \ell ; A , \nu ) . } \end{array}\tag{F44}
$$

(F45)

Hence $V _ { W _ { \ell } } = \gamma _ { \mathrm { s q } } ( \ell ; A , \nu )$ , and the definition of Fourier gain gives

$$
G ^ { A , \nu } ( \ell ) \geq \frac { | \widehat { f } _ { W _ { \ell } } ( \ell ) | ^ { 2 } } { V _ { W _ { \ell } } } = I ( \ell ; A , \nu ) .\tag{F46}
$$

It remains only to lower bound this certified gain. This is exactly the Chebyshev Lipschitz-continuity in the preceding lemma, with the factor 0.9 removed and with k replaced by ℓ. Drawing on that result, we have

$$
\int _ { \mathbb { R } } \frac { a _ { \ell } ( y ) ^ { 2 } } { q _ { 0 } ( y ) } d y \geq c _ { 1 }\tag{F47}
$$

whenever $\nu \leq c _ { 0 } \frac { R ^ { 2 } } { \ell ^ { 2 } }$

By Lemma 3, constant Fourier gain at frequency ℓ is achieved whenever

$$
r \ge \log \left( \frac { \ell } { A } \right) + O ( 1 ) ,\tag{F48}
$$

which gives us that a Gaussian protocol with energy $O ( \ell ^ { 2 } / A ^ { 2 } )$ sufices to learn the frequency-ℓ angular Fourier coeficient of a distribution with amplitude A. To formalize this and conclude, we state the full algorithm and rigorously combine Theorem D.30 and Lemma 3.

Algorithm 1 Randomized squeezed-homodyne estimator for fixed-amplitude angular Fourier learning lg

Input: Frequency $k \geq 1$ , fixed amplitude $A > 0 ,$ , squeezing parameter $r ,$ accuracy parameters $\epsilon , \delta ,$ and query access to   
$\mathcal { E } _ { P } .$   
Output: An estimate $\widehat { \Lambda } _ { k }$ of $\Lambda _ { k } ( P ) = \mathbb { E } _ { \theta \sim F } [ e ^ { i k \theta } ]$   
1: Set $\nu = e ^ { - 2 r } / 2$ and $R = A { \sqrt { 2 } } .$   
2: Compute   
1 2π   
q<sub>0</sub>(y) = 2π J0 g<sub>ν</sub>(y − R cos u)du, (F49)   
1 2π   
a<sub>k</sub>(y) = cos(ku)g<sub>ν</sub>(y − R cos u)du, (F50)   
2π J0   
and   
a<sub>k</sub>(y)<sup>2</sup>   
γ<sub>sq</sub>(k; A, ν) = dy. (F51)   
<sub>R</sub> q<sub>0</sub>(y)   
3: Take   
1 4   
<sup>N</sup> <sup>=</sup> <sup>C</sup> γ (k; A, ν)ϵ<sup>2 log</sup> δ (F52)   
shots for a suficiently large universal constant $C .$   
4: for $j = 1 , \ldots , N$ do   
5: Draw $\varphi _ { j }$ uniformly from [0, 2π).   
6: Prepare a squeezed vacuum whose squeezed quadrature is the homodyne quadrature at angle $\varphi _ { j }$   
7: Send the probe through $\mathcal { E } _ { P }$ and homodyne the same quadrature, obtaining $Y _ { j }$   
8: Set   
$Z _ { j } = e ^ { i k \varphi _ { j } } \frac { a _ { k } ( Y _ { j } ) } { q _ { 0 } ( Y _ { j } ) \gamma _ { \mathrm { s q } } ( k ; A , \nu ) } .$ (F53)   
9: end for   
10: Output   
N   
Λb<sub>k</sub> = <sup>1</sup> X Z<sub>j</sub> . (F54)   
j=1

Theorem F.4 (Fixed-amplitude angular Fourier learning from squeezed Fourier gain). Let $P$ be any   
displacement distribution with amplitude A. Then Algorithm $^ { 1 , }$ with $r \geq \Omega ( \log { ( k / A ) } )$ , outputs an estimate   
$\widehat { \Lambda } _ { k }$ satisfying $| \widehat { \Lambda } _ { k } - \Lambda _ { k } ( P ) | \leq \epsilon$ with probability at least $1 - \delta$ given   
N = O <sup>−2</sup> log δ (F55)   
samples. This requires total energy or mean photon number $O ( k ^ { 2 } / A ^ { 2 } )$

Proof. For the single-coeficient learning problem, take the property kernel $\psi ( \theta ) = \chi _ { k } ( \theta ) = e ^ { i k \theta }$ . Its Fourier support is the singleton $S = \{ k \}$ and $\widehat { \psi } ( k ) = 1$ . The Fourier-overlap guarantee in Theorem D.30 therefore gives a protocol with sample complexity

$$
N = O \left( \frac { 1 } { G ^ { A , \nu } ( k ) \epsilon ^ { 2 } } \log \left( \frac { 1 } { \delta } \right) \right) .\tag{F56}
$$

Algorithm 1 is exactly this protocol, specialized to the randomized-homodyne measurement. By Lemma $^ { 3 , }$ $G ^ { A , \nu } ( k ) \geq I ( k ; A , \nu )$ , and if $\nu \leq c _ { 0 } A ^ { 2 } / k ^ { 2 }$ , then $I ( k ; A , \nu ) \ge c _ { 1 }$ for a universal constant $c _ { 1 } > 0$ . Since $\nu = e ^ { - 2 r } / 2 .$ this condition is equivalent to choosing $\begin{array} { r } { r \ge \log \left( \frac { k } { A } \right) + \ ' { O } ( 1 ) } \end{array}$ . Substituting the constant gain lower bound into the Fourier-overlap sample complexity gives

$$
N = O \left( \epsilon ^ { - 2 } \log \left( \frac { 1 } { \delta } \right) \right) .\tag{F57}
$$

Finally, choosing $r = \log ( k / A ) + O ( 1 )$ gives mean photon number sinh $\mathrm { \Large ~ l ~ } ^ { 2 } r = O ( e ^ { 2 r } ) = O ( k ^ { 2 } / A ^ { 2 } )$

## b. Experimental proxy protocol

Our experimental results in Figure 3(b) aim to illustrate the Corollary 2 separation in Fourier amplitude learning. However due to the incompatibility of squeezed homodyne measurements with our microwave cavity superconducting circuit apparatus, we implement a diferent protocol than the one above. Here we describe this protocol and rigorously demonstrate that it is strictly weaker than the squeezed homodyne strategy, so that any separation present in our experiment is no larger than the one that would be observed upon performing squeezed homodyne sensing with similar fidelity.

We use the following fact about Bessel functions of the first kind:

Fact 8. Let $J _ { k } ( z )$ denote the k-th Bessel function of the first kind. There exist universal constants $c _ { B } , C _ { B } > 0$ such that, for every $k \geq 1$

$$
c _ { B } k ^ { - 1 / 3 } \leq \operatorname* { s u p } _ { z \in \mathbb { R } } | J _ { k } ( z ) | \leq C _ { B } k ^ { - 1 / 3 } .\tag{F58}
$$

In the subsequent discussion we let $z _ { k }$ be a point satisfying $| J _ { k } ( z _ { k } ) | \geq c _ { B } k ^ { - 1 / 3 }$ and $| z _ { k } | = O ( k )$

```latex
Algorithm 2 Experimental proxy estimator for angular Fourier learning
Input: Frequency $k \geq 1 ,$ fixed amplitude $A > 0 ,$ accuracy parameters $\epsilon , \delta ,$ and query access to $\mathcal { E } _ { P }$
Output: An estimate $\widehat { \Lambda } _ { k }$ of $\Lambda _ { k } ( P ) = \mathbb { E } [ e ^ { i k \theta } ]$
1: Set $R = A { \sqrt { 2 } }$ and $t = z _ { k } / R .$
2: Take
$N = C k ^ { 2 / 3 } \epsilon ^ { - 2 } \log \left( \frac { 4 } { \delta } \right)$ (F59)
shots for a suficiently large universal constant C.
3: for $j = 1 , \ldots , N$ do
4: Draw $\varphi _ { j }$ uniformly from [0, 2π).
5: Set
$\beta _ { t , \varphi _ { j } } = - \frac { i t } { \sqrt { 2 } } e ^ { i \varphi _ { j } } .$ (F60)
6: Prepare
$\left| \Psi _ { t , \varphi _ { j } } \right. = \frac { 1 } { \sqrt { 2 } } \left( \left| g \right. \left| 0 \right. + \left| e \right. D ( \beta _ { t , \varphi _ { j } } ) \left| 0 \right. \right)$ (F61)
7: Send the sensing mode through $\mathcal { E } _ { P } .$
8: Apply $D ( - \beta _ { t , \varphi _ { j } } )$ on the |e⟩ arm.
9: With probability $1 / 2 ,$ measure X on the qubit, obtain $S _ { j } \in \{ \pm 1 \}$ , and set $W _ { j } = 2 S _ { j }$
10: With probability $1 / 2 ,$ measure $Y$ on the qubit, obtain $\bar { S _ { j } } \in \{ \pm 1 \}$ , and set ${ \cal W } _ { j } ^ { \prime } = 2 i \check { S } _ { j }$
11: Set
$Z _ { j } = \frac { i ^ { - k } e ^ { i k \varphi _ { j } } } { J _ { k } ( z _ { k } ) } W _ { j } .$ (F62)
12: end for
13: Output
$\widehat { \Lambda } _ { k } = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } Z _ { j } .$ (F63)
```

Lemma 5 (Sample complexity of the experimental proxy). Algorithm $\mathcal { Q }$ estimates $\Lambda _ { k } ( P )$ to additive error ϵ

with probability at least $1 - \delta$ for every fixed-amplitude distribution $P ,$ using

$$
N = O \left( k ^ { 2 / 3 } \epsilon ^ { - 2 } \log \left( \frac { 1 } { \delta } \right) \right)\tag{F64}
$$

queries. The mean photon number in the sensing mode before the signal is $O ( k ^ { 2 } / A ^ { 2 } )$ , and hence is $O ( k )$ when $A = \Theta ( { \sqrt { k } } )$

Proof. Fix a deterministic signal displacement $D ( A e ^ { i \theta } )$ . The Weyl relation gives

$$
D ( - \beta _ { t , \varphi } ) D ( A e ^ { i \theta } ) D ( \beta _ { t , \varphi } ) = \exp \left( 2 i \operatorname { I m } ( A e ^ { i \theta } { \overline { { \beta } } } _ { t , \varphi } ) \right) D ( A e ^ { i \theta } ) .\tag{F65}
$$

By the choice of $\beta _ { t , \varphi }$

$$
2 \operatorname { I m } ( A e ^ { i \theta } { \overline { { \beta } } } _ { t , \varphi } ) = t R \cos ( \theta - \varphi ) .\tag{F66}
$$

Thus after the inverse conditional displacement $D ( - \beta _ { t , \varphi _ { j } } )$ , the sensing mode factors out and the qubit carries the relative phase $e ^ { i t R \cos ( \theta - \varphi ) }$ . The $X / Y$ then gives

$$
\mathbb { E } [ W _ { j } | \theta , \varphi ] = e ^ { i t R \cos ( \theta - \varphi ) } .\tag{F67}
$$

Since $t R = z _ { k }$ , the Jacobi-Anger expansion gives

$$
e ^ { i z _ { k } \cos ( \theta - \varphi ) } = \sum _ { m \in \mathbb { Z } } i ^ { m } J _ { m } ( z _ { k } ) e ^ { i m ( \theta - \varphi ) } .\tag{F68}
$$

Averaging over the random angle $\varphi ,$

$$
\begin{array} { c } { { \mathbb { E } [ Z _ { j } | \theta ] = \displaystyle \frac { i ^ { - k } } { J _ { k } \left( z _ { k } \right) } \int _ { 0 } ^ { 2 \pi } \frac { d \varphi } { 2 \pi } e ^ { i k \varphi } e ^ { i z _ { k } \cos \left( \theta - \varphi \right) } } } \\ { { = e ^ { i k \theta } . } } \end{array}\tag{F69}
$$

(F70)

Averaging over $\theta$ proves ${ \mathbb E } [ Z _ { j } ] = \Lambda _ { k } ( { P } )$ . Moreover, $| W _ { j } | = 2$ and $| J _ { k } ( z _ { k } ) | \ge c _ { B } k ^ { - 1 / 3 }$ , so $\begin{array} { r } { | Z _ { j } | \le \frac { 2 } { c _ { B } } k ^ { 1 / 3 } } \end{array}$ Hoefding’s inequality applied to the real and imaginary parts gives

$$
\operatorname* { P r } \left[ \left| \frac { 1 } { N } \sum _ { j = 1 } ^ { N } Z _ { j } - \Lambda _ { k } ( P ) \right| > \epsilon \right] \leq 4 \exp \left( - c \frac { N \epsilon ^ { 2 } } { k ^ { 2 / 3 } } \right)\tag{F71}
$$

for a universal constant $c > 0$ , which proves the stated sample complexity.

The sensing mode is in vacuum on one arm and in the coherent state $D ( \beta _ { t , \varphi } ) \left| 0 \right.$ on the other arm, so its mean photon number is at most a universal constant times

$$
| \beta _ { t , \varphi } | ^ { 2 } = { \frac { t ^ { 2 } } { 2 } } = { \frac { z _ { k } ^ { 2 } } { 2 R ^ { 2 } } } = O \left( { \frac { k ^ { 2 } } { A ^ { 2 } } } \right) .\tag{F72}
$$

For $A = \Theta ( { \sqrt { k } } )$ , this is $O ( k )$

For the hypothesis testing distributions $P _ { \pm } ^ { ( k ) }$ , the proxy therefore distinguishes the two cases with $O ( k ^ { 2 / 3 } )$ samples and $O ( k )$ energy when $A = \Theta ( { \sqrt { k } } )$ . This is polynomially worse than the squeezed homodyne protocol, which achieves $O ( 1 )$ samples at the same energy scaling. Thus the experimental proxy is strictly less informative for this task than the squeezed homodyne strategy. It is nevertheless still exponentially more sample-eficient than coherent-probe sensing by Theorem E.1, so observing the separation with this weaker protocol gives a conservative experimental demonstration of the Fourier analysis advantage in Corollary 2.

## 2. Learning with quantum control

We next move one level higher in the sensing hierarchy. The lower bound in Section E 2 shows that conventional Gaussian sensing protocols with bounded energy have exponentially small overlap with high-frequency Fourier modes. Here, we show that a single control qubit and one controlled displacement convert this exponentially small overlap into an inverse-polynomial one, enabling eficient Quantum Feature Sensing. As in the previous subsection, we begin with the hypothesis testing task from Section E 2, and use the QΨ hypothesis testing guarantee based on an AFI lower bound to demonstrate the exponential advantage. We then use the global Fourier-gain theorem (Theorem D.30) to give a broader learning algorithm for eficiently learning high-frequency Fourier coeficients.

First we recall the hypothesis testing task. We work in quadrature coordinates $z = ( x , p ) \in \mathbb { R } ^ { 2 }$ , and let

$$
g ( x , p ) = { \frac { 1 } { 2 \pi } } \exp \left( - { \frac { x ^ { 2 } + p ^ { 2 } } { 2 } } \right)\tag{F73}
$$

be the standard Gaussian reference distribution. For $k \geq 1$ , recall that the hypothesis families were given by

$$
c _ { k } = \mathbb { E } _ { g } [ \cos ( k X ) ] = e ^ { - k ^ { 2 } / 2 }\tag{F74}
$$

$$
h _ { k } ( x , p ) = { \frac { \cos ( k x ) - c _ { k } } { 1 + c _ { k } } }\tag{F75}
$$

$$
P _ { \pm } ^ { k } ( x , p ) = g ( x , p ) \left( 1 \pm h _ { k } ( x , p ) \right) .\tag{F76}
$$

These signals difer in the observable

$$
\Psi _ { R } ( P ) = \mathbb { E } _ { ( X , P ^ { \prime } ) \sim P } [ \cos ( k X ) ] .\tag{F77}
$$

by a constant margin. Naturally, estimating the Fourier coeficient $\mathbb { E } [ e ^ { i k X } ]$ and taking the real part would reveal this quantity and solve the hypothesis testing problem. Previously we demonstrated that any adaptive protocol using Gaussian measurements and energy budget $\leq c k$ , for any constant c independent of k requires $\exp ( \Omega ( k ) )$ queries to the signal to solve this task with constant bias.

Now consider the following protocol for the hypothesis testing task. For a constant $c > 0$ , fix an energy budget E and assume $k ^ { 2 } \geq 8 E$ . Set

$$
p _ { E } = \frac { 4 E } { k ^ { 2 } } , \qquad \beta _ { k } = - \frac { i k } { 2 } .\tag{F78}
$$

Prepare the state

$$
\sqrt { 1 - p _ { E } } \left. g \right. \left. 0 \right. + \sqrt { p _ { E } } \left. e \right. D ( \beta _ { R } ) \left. 0 \right.\tag{F79}
$$

by initializing an ancilla qubit in the state $\sqrt { 1 - p _ { E } } | 0 \rangle + \sqrt { p _ { E } } | 1 \rangle$ and applying $\mathrm { E C D } ( \beta _ { k } )$ , send the sensing mode through the displacement channel, apply $\mathrm { E C D } ( \beta _ { k } ) ^ { \dagger }$ on the |e⟩ arm, and then measure X on the qubit. This straightforwardly defines a Ramsey sequence using single-qubit ancilla control that requires only a single controlled gate to prepare the probe state and a single inverse gate to perform the readout. We denote the resulting binary measurement by $M ^ { E , k }$ and its outcome by $W \in \{ \pm 1 \}$ . Importantly, at no point during the protocol does the sensor state have a mean photon number $> E$

Lemma 6 (AFI of the depth-one controlled sensing protocol). For the hypothesis testing direction $h _ { R }$ above,

$$
\mathsf { A } _ { M ^ { E , k } } ( h _ { k } ; g ) \ge C \frac { E } { k ^ { 2 } }\tag{F80}
$$

for a universal constant $C > 0 ,$ , whenever $k \geq 1$ and $k ^ { 2 } \geq 8 E$ . Furthermore, this lower bound is tight up to constants for any depth-one sensing protocol with $e n e r g y \le E$

Proof. We first compute the response of the protocol. Fix a deterministic signal displacement $D ( x + i p ^ { \prime } )$ . The Weyl relation gives

$$
D ( - \beta _ { k } ) D ( x + i p ^ { \prime } ) D ( \beta _ { k } ) = \exp \left( 2 i \mathrm { I m } ( ( x + i p ^ { \prime } ) \overline { { { \beta _ { k } } } } ) \right) D ( x + i p ^ { \prime } ) .\tag{F81}
$$

Since $\beta _ { k } = - i k / 2 , 2 \mathrm { I m } ( ( x + i p ^ { \prime } ) \overline { { \beta _ { k } } } ) = k x$ . Thus after the inverse controlled displacement, the sensing mode factors out and the qubit is in the state

$$
\sqrt { 1 - p _ { E } } \left. g \right. + \sqrt { p _ { E } } e ^ { i k x } \left. e \right. .\tag{F82}
$$

Measuring X gives

$$
\mathbb { E } [ W | x , p ^ { \prime } ] = 2 \sqrt { p _ { E } ( 1 - p _ { E } ) } \cos ( k x ) .\tag{F83}
$$

Let $s _ { E } = 2 \sqrt { p _ { E } ( 1 - p _ { E } ) }$ . We now compute the AFI, noting that because we are now measuring the qubit, the response functions will simply be two-outcome discrete probabilities; nevertheless, the QΨ formalism applies directly. The response under reference distribution g is

$$
Q _ { 0 } ( W = w ) = \frac 1 2 ( 1 + w s _ { E } c _ { k } ) , \qquad w \in \{ \pm 1 \} .\tag{F84}
$$

The response function under the perturbed hypothesis is

$$
A _ { h _ { k } } ( W = w ) = \int h _ { k } ( x , p ^ { \prime } ) \frac { 1 } { 2 } \left( 1 + w s _ { E } \cos ( k x ) \right) g ( x , p ^ { \prime } ) d x d p ^ { \prime }\tag{F85}
$$

$$
= \frac { w s _ { E } } { 2 } \mathbb { E } _ { g } [ h _ { k } ( X , P ^ { \prime } ) \cos ( k X ) ]\tag{F86}
$$

$$
\frac { w s _ { E } } { 2 } \frac { v _ { k } } { 1 + c _ { k } } ~ ,\tag{F87}
$$

where $v _ { k }$ is given by

$$
v _ { k } = \mathbb { E } _ { g } [ \cos ^ { 2 } ( R X ) ] - \mathbb { E } _ { g } [ \cos ( R X ) ] ^ { 2 } ~ .\tag{F88}
$$

Therefore

$$
\mathsf { A } _ { M ^ { E , R } } ( h _ { R } ; g ) = \sum _ { w = \pm 1 } \frac { A _ { h _ { R } } ( W = w ) ^ { 2 } } { Q _ { 0 } ( W = w ) }\tag{F89}
$$

$$
= \displaystyle \frac { s _ { E } ^ { 2 } v _ { R } ^ { 2 } } { ( 1 + c _ { R } ) ^ { 2 } ( 1 - s _ { E } ^ { 2 } c _ { R } ^ { 2 } ) } .\tag{F90}
$$

Since $k ^ { 2 } \geq 8 E .$ , we have $p _ { E } \leq 1 / 2$ , and hence $s _ { E } ^ { 2 } = 4 p _ { E } ( 1 - p _ { E } ) \geq 2 p _ { E }$ . Also, for $R \geq 1$

$$
v _ { k } = { \frac { 1 + e ^ { - 2 k ^ { 2 } } } { 2 } } - e ^ { - k ^ { 2 } } \geq { \frac { 1 } { 2 } } - { \frac { 1 } { e } } > { \frac { 1 } { 8 } } ,\tag{F91}
$$

so $v _ { k } > 1 / 8$ and $1 + c _ { k } \le 2$ . Thus

$$
\mathsf { A } _ { M ^ { E , R } } ( h _ { R } ; g ) \geq C p _ { E } = C \frac { E } { k ^ { 2 } }\tag{F92}
$$

for a universal constant $C > 0$ . Next we show that this is optimal for depth-one protocols. Given a single qubit and a single controlled displacement, the state of the sensor must always be of the form

$$
\left| \Psi _ { 0 } \right. = \sqrt { a } \left| 0 \right. \left| \beta _ { 1 } \right. + e ^ { i \chi } \sqrt { 1 - a } \left| 1 \right. \left| \beta _ { 2 } \right.\tag{F93}
$$

for some $a \in [ 0 , 1 ] , \chi \in \mathbb { R }$ , and $\beta _ { 1 } , \beta _ { 2 } \in \mathbb { C }$ . The mean photon number in the sensing mode is

$$
E _ { \Psi } = a | \beta _ { 1 } | ^ { 2 } + ( 1 - a ) | \beta _ { 2 } | ^ { 2 }\tag{F94}
$$

$$
= | a \beta _ { 1 } + ( 1 - a ) \beta _ { 2 } | ^ { 2 } + a ( 1 - a ) | \beta _ { 2 } - \beta _ { 1 } | ^ { 2 } .\tag{F95}
$$

Thus, if $\Delta = \beta _ { 2 } - \beta _ { 1 }$ , the energy constraint implies

$$
a ( 1 - a ) | \Delta | ^ { 2 } \leq E .\tag{F96}
$$

The first term in the energy is a common displacement of both branches; as is evident from the rest of this work, the more refined resource is not ordinary energy but nonclassical energy, so energy allocated towards the first term will not asymptotically change performance. It can only contribute the ordinary coherent-probe response, whereas the controlled part of the protocol is governed by the branch separation $\Delta .$ This is made concrete by the following argument.

Let $\boldsymbol { z } = ( x , p )$ , let $\sigma _ { z } = D ( z ) \left| 0 \right. \left. 0 \right| D ( z ) ^ { \dagger }$ , and write

$$
\xi _ { \Delta } \cdot z = 2 \mathrm { I m } ( z \overline { { { \Delta } } } ) , \qquad \Delta = \beta _ { 2 } - \beta _ { 1 } .\tag{F97}
$$

Conditioning on a displacement $D ( z )$ , the sensor state after the inverse controlled displacement becomes

$$
\left. \Psi _ { z } \right. = \left( \sqrt { a } \left. 0 \right. + e ^ { i \chi } \sqrt { 1 - a } e ^ { i \xi _ { \Delta } \cdot z } \left. 1 \right. \right) D ( z ) \left. 0 \right. ,\tag{F98}
$$

Averaging over the signal distribution $g ( z ) ( 1 + \eta h _ { k } ( z ) )$ , the pre-measurement state is

$$
\rho _ { \eta } = \int d z \ g ( z ) ( 1 + \eta h _ { k } ( z ) ) \left| \Psi _ { z } \right. \langle \Psi _ { z } \vert .\tag{F99}
$$

We introduced η so that we can let $\begin{array} { r } { \dot { \rho } = \left. \frac { d } { d \eta } \rho _ { \eta } \right. _ { \eta = 0 } } \end{array}$ . This will give us a clean way to write the AFI, in a QFIreminiscent manner. The diagonal blocks of $\dot { \rho }$ are proportional to $\begin{array} { r } { \int d z \ g ( z ) h _ { k } ( z ) \sigma _ { z } } \end{array}$ , independent of the relative separation in the initial state and have exponentially small operator norm in k by the conventional Gaussian lower bound. The of-diagonal block is

$$
\dot { \rho } _ { \mathrm { o f f } } = \sqrt { a ( 1 - a ) } \left( e ^ { - i \chi } | 0 \rangle \langle 1 | \otimes T _ { \Delta } + e ^ { i \chi } | 1 \rangle \langle 0 | \otimes T _ { \Delta } ^ { \dagger } \right) ,\tag{F100}
$$

where

$$
T _ { \Delta } = \int d z g ( z ) h _ { R } ( z ) e ^ { i \xi \Delta \cdot z } \sigma _ { z } .\tag{F101}
$$

Now fix an arbitrary POVM $\{ M _ { s } \}$ <sub>s</sub> applied to the final qubit-sensor state. If $p _ { s } ( \eta ) = \mathrm { T r } ( M _ { s } \rho _ { \eta } )$ , then the classical AFI of this POVM in the direction $\dot { \rho } _ { \mathrm { o f f } }$ is

$$
\sum _ { s } \frac { \dot { p } _ { s } ^ { 2 } } { p _ { s } ( 0 ) } , \qquad \dot { p } _ { s } = \mathrm { T r } ( M _ { s } \dot { \rho } _ { \mathrm { o f f } } ) .\tag{F102}
$$

Using the block form above and Cauchy-Schwarz in the Hilbert-Schmidt inner product, one obtains

$$
\sum _ { s } \frac { \dot { p } _ { s } ^ { 2 } } { p _ { s } ( 0 ) } \le C a ( 1 - a ) \operatorname { T r } \left( T _ { \Delta } ^ { \dagger } \sigma _ { 0 } ^ { - 1 } T _ { \Delta } \right) ,\tag{F103}
$$

where $\begin{array} { r } { \sigma _ { 0 } = \int d z \ g ( z ) \sigma _ { z } } \end{array}$ and $C > 0$ is a universal constant. Now we control the operator norm in (F103). Since

$$
h _ { k } ( z ) = { \frac { \cos ( R x ) - c _ { k } } { 1 + c _ { k } } } ,\tag{F104}
$$

the function $h _ { k } ( z ) e ^ { i \xi _ { \Delta } \cdot z }$ is a linear combination of Gaussian Fourier modes at frequencies $\xi _ { \Delta } + ( R , 0 ) , \xi _ { \Delta } - ( R , 0 )$ and $\xi _ { \Delta }$ . The Gaussian coherent-state channel damps a Fourier mode of phase-space frequency κ by a factor exponentially small in $| \kappa | ^ { 2 }$ . Omitting the Gaussian integration algebra, we find

$$
\operatorname { T r } \left( T _ { \Delta } ^ { \dagger } \sigma _ { 0 } ^ { - 1 } T _ { \Delta } \right) \leq C \left( e ^ { - c | \xi _ { \Delta } - ( k , 0 ) | ^ { 2 } } + e ^ { - c | \xi _ { \Delta } + ( k , 0 ) | ^ { 2 } } + e ^ { - c k ^ { 2 } } e ^ { - c | \xi _ { \Delta } | ^ { 2 } } \right)\tag{F105}
$$

for universal constants $c , C > 0$ . We have that if $\mathrm { e . g . } | \xi _ { \Delta } | < k / 2$ , the controlled contribution to the AFI is exponentially small in $k ^ { 2 }$ . Any response that is significant has $| \xi _ { \Delta } | \ge k / 2$ (and must indeed be $\sim k$ , but the cruder bound will sufice). Since $| \xi _ { \Delta } | = 2 | \Delta |$ |, this implies $| \Delta | \geq k / 4$ . Combining this with the energy constraint $a ( 1 - a ) | \Delta | ^ { 2 } \leq E$ gives

$$
a ( 1 - a ) \leq { \frac { 1 6 E } { k ^ { 2 } } } .\tag{F106}
$$

Substituting into (F103), the contribution to the AFI is at most $O ( E / k ^ { 2 } )$ , up to terms exponentially small in $k ^ { 2 }$ . The diagonal block contributes only exponentially small AFI as discussed earlier. This proves that no final POVM can extract more than $O ( E / k ^ { 2 } )$ AFI for this hypothesis testing problem once an energy-E probe state is created using a single controlled displacement. □

This Lemma gives us the QΨ hypothesis testing separation.

Corollary 7 (Exponential advantage from one ancilla qubit and constant control depth). There is an optimal depth-one single-qubit control protocol using energy $k$ per query which distinguishes $P _ { + } ^ { k }$ from $P _ { - } ^ { k }$ in Equation (F74) with constant bias using $O ( k )$ queries. In contrast, any conventional Gaussian sensing protocol with the same energy budget requires exp(Ω(k)) queries to do so.

Proof. Lemma 6 and Theorem D.27 give sample complexity

$$
O \left( \frac { 1 } { \mathsf { A } _ { M ^ { E = k , k } } \left( h _ { k } ; g \right) } \log \left( \frac { 1 } { \delta } \right) \right) = O \left( k \right) ,\tag{F107}
$$

since $\delta$ is a fixed constant. The protocol uses energy $E = k$ by construction. The lower bound is Theorem E.3. This is exactly the statement that conventional Gaussian sensing with energy k has exponentially small access to the frequency-k phase-space Fourier mode, while the one-qubit controlled-displacement experiment has AFI $\Omega ( 1 / k )$ and therefore succeeds with only $O ( k )$ queries. □

We now turn to the genuinely global version of this learning problem. The previous hypothesis test fixed the Fourier fringe along the x axis. In practice, however, the relevant Fourier direction may not be known when the data are collected, and will then need to be ascertained from the collected data. For this we consider a directional analog of the characteristic function,

$$
\lambda _ { P } ( k , \theta ) = \mathbb { E } _ { z \sim P } \left[ e ^ { i k u _ { \theta } \cdot z } \right] , \qquad u _ { \theta } = ( \cos \theta , \sin \theta ) .\tag{F108}
$$

The controlled-displacement protocol can learn this object post-hoc on an angular grid by randomizing the controlled-displacement angle and storing the angle used in each shot. Let $\Theta _ { m } = \{ 2 \pi r / m : r = 0 , \ldots , m - 1 \}$ For $\theta \in [ 0 , 2 \pi )$ , let $\Pi _ { m } ( \theta )$ denote a nearest point in $\Theta _ { m }$

Algorithm 3 Post-hoc directional Fourier learning with one control qubit   
Input: Frequency $k \geq 1$ , energy $E$ with $k ^ { 2 } \geq 8 E .$ angular net size m, query access to $\mathcal { E } _ { P } ,$ and number of shots $N$   
Output: A classical transcript which can estimate $\lambda _ { P } ( k , \theta )$ for post-hoc directions $\theta .$   
1: Set $p _ { E } = 4 E / k ^ { 2 }$ and $s _ { E } = 2 \sqrt { p _ { E } ( 1 - p _ { E } ) } ,$   
2: for $\mathbf { \bar { \rho } } _ { j } = 1 , \dots , N$ do   
3: Draw φ uniformly from $\Theta _ { m }$ and set $\beta _ { j } = - i k e ^ { i \varphi _ { j } } / 2 .$   
4: Prepare $\sqrt { 1 - p _ { E } } \left. g \right. \left. 0 \right. + \sqrt { p _ { E } } \left. e \right. D ( \beta _ { j } ) \left. 0 \right.$   
5: Send the sensing mode through $\mathcal { E } _ { P }$ and apply $D ( - \beta _ { j } )$ on the $| e \rangle$ arm.   
6: Measure X with probability $\bar { 1 } / 2$ and set $W _ { j } = 2 \dot { S } _ { j }$ , or measure $\dot { Y }$ with probability $1 / 2$ and set $W _ { j } = 2 i S _ { j }$   
7: Store $( \varphi _ { j } , W _ { j } )$   
8: end for   
9: After receiving a query $\theta ,$ set $\theta ^ { * } = \Pi _ { m } ( \theta )$ and output   
$\widehat { \lambda } _ { P } ( k , \theta ) = \frac { m } { N s _ { E } } \sum _ { j = 1 } ^ { N } \mathbb { 1 } \{ \varphi _ { j } = \theta ^ { * } \} W _ { j } .$ (F109)

Theorem F.8 (Post-hoc directional Fourier learning). For every displacement distribution $P$ and every   
$\theta \in \Theta _ { m }$ , Algorithm $\mathcal { B }$ estimates $\lambda _ { P } ( k , \theta )$ to additive error ϵ with probability at least $1 - \delta$ using   
$N = O \left( \frac { m k ^ { 2 } } { E \epsilon ^ { 2 } } \log \left( \frac { m } { \delta } \right) \right)$ (F110)   
queries, simultaneously for all $\theta \in \Theta _ { m }$ $H ,$ moreover, $P$ satisfies $\mathbb { E } _ { z \sim P } [ \lVert z \rVert ] \le B$ , then choosing m =   
$O ( k B / \epsilon )$ gives a post-hoc estimator for every $\theta \in [ 0 , 2 \pi )$ with sample complexity   
$N = O \left( \frac { k ^ { 3 } B } { E \epsilon ^ { 3 } } \log \left( \frac { k B } { \epsilon \delta } \right) \right)$ (F111)

Proof. Fix $z = \left( x , p \right)$ and a chosen angle $\varphi .$ By the Weyl relation and the choice $\beta = - i k e ^ { i \varphi } / 2$ , the echo imprints the relative phase $e ^ { i k u _ { \varphi } \cdot z }$ on the qubit. Hence the randomized $X / Y$ readout satisfies

$$
\begin{array} { r } { \mathbb { E } [ W _ { j } | z , \varphi _ { j } ] = s _ { E } e ^ { i k { u _ { \varphi _ { j } } } \cdot z } . } \end{array}\tag{F112}
$$

For a fixed net point $\theta \in \Theta _ { m }$ , the postprocessed one-shot variable

$$
Z _ { \theta , j } = { \frac { m } { s _ { E } } } \mathbb { 1 } \{ \varphi _ { j } = \theta \} W _ { j }\tag{F113}
$$

therefore satisfies $\mathbb { E } [ Z _ { \theta , j } \vert z ] = e ^ { i k u _ { \theta } \cdot z }$ . Averaging over $z \sim P$ gives $\mathbb { E } [ Z _ { \theta , j } ] = \lambda _ { P } ( k , \theta )$

The second moment obeys $\mathbb { E } [ | Z _ { \theta , j } | ^ { 2 } ] \le 4 m / s _ { E } ^ { 2 }$ . Since $k ^ { 2 } \geq 8 E$ , we have $p _ { E } \leq 1 / 2$ , so $s _ { E } ^ { 2 } = 4 p _ { E } ( 1 - p _ { E } ) =$ $\Omega ( E / k ^ { 2 } )$ . Bernstein’s inequality, applied to real and imaginary parts and union bounded over the m net points, $\mathrm { g i v e s }$

$$
N = O \left( { \frac { m k ^ { 2 } } { E \epsilon ^ { 2 } } } \log \left( { \frac { m } { \delta } } \right) \right) .\tag{F114}
$$

This is exactly the Fourier-overlap theorem with gain $\Omega ( E / ( m k ^ { 2 } ) )$ per net direction, because the randomized protocol spends a $1 / m$ fraction of its shots on each Fourier response.

For the continuum statement, let $\theta ^ { \sharp } = \Pi _ { m } ( \theta )$ . Since $| e ^ { i a } - e ^ { i b } | \leq | a - b |$

$$
| \lambda _ { P } ( k , \theta ) - \lambda _ { P } ( k , \theta ^ { \sharp } ) | \leq k \mathbb { E } _ { z \sim P } \left[ | ( u _ { \theta } - u _ { \theta ^ { \sharp } } ) \cdot z | \right]\tag{F115}
$$

$$
\leq k B | \theta - \theta ^ { \sharp } |\tag{F116}
$$

$$
\leq { \frac { 2 \pi k B } { m } } .\tag{F117}
$$

Choosing $m = O ( k B / \epsilon )$ makes this discretization error at most $\epsilon / 2$ , and applying the net guarantee with accuracy $\epsilon / 2$ proves the claim. □

This is the global Fourier-overlap picture for one-qubit control. A known direction requires gain $\Omega ( E / k ^ { 2 } )$ and hence ${ \cal O } ( k ^ { 2 } / ( E \epsilon ^ { 2 } ) )$ samples. If the direction is not known when data are collected, randomized cat angles spread this gain over the angular net, producing gain $\Omega ( E / ( m k ^ { 2 } ) )$ per post-hoc direction. In the main-text scaling $E = k$ , this remains polynomial in $k ,$ while the conventional Gaussian lower bound is exponential for the same high-frequency features.

## a. Alternative approach using SU(1, 1) interferometry or Gaussian boson sampling

Gaussian boson samplers are most often studied in settings where sampling from the circuit output distribution is itself the objective. However, here we demonstrate that the same hardware can instead be used to learn a property of an external signal more eficiently than any conventional Gaussian sensor. We consider the same angular Fourier coeficient task above and show that the echo implemented by a canonical SU(1, 1) interferometry protocol [33], followed by photon-number parity measurement, estimates $\Lambda _ { k } ( P )$ with polynomially many signal queries and mean photon number $O ( k )$ .

Let $S ( r , \phi )$ denote a single-mode squeezing unitary oriented so that

$$
S ( r , \phi ) ^ { \dagger } D ( A e ^ { i \theta } ) S ( r , \phi ) = D \left( e ^ { i \phi } A \left( e ^ { - r } \cos ( \theta - \phi ) + i e ^ { r } \sin ( \theta - \phi ) \right) \right) .\tag{F118}
$$

We take $A = { \sqrt { k } }$ and $r = \log A$ . We also define the following normalization constant:

$$
\gamma _ { k } = \frac { 1 } { 2 \pi } \int _ { 0 } ^ { 2 \pi } d u \exp \left( - 2 ( 1 + \cos u ) ^ { 2 } - 2 k ^ { 2 } \sin ^ { 2 } u \right) e ^ { - i k u } .\tag{F119}
$$

This coeficient depends only on k and is used in the following estimator.

Algorithm 4 Randomized SU(1, 1) interferometry for angular Fourier learning   
Input: Frequency $k \in \mathbb { Z } ^ { + }$ , accuracy parameters $\epsilon , \delta ,$ and query access to $\mathcal { E } _ { P }$ , where $P$ has fixed amplitude $A = { \sqrt { k } }$   
Output: An estimate $\widehat { \Lambda } _ { k }$ of $\Lambda _ { k } ( P ) = \mathbb { E } _ { \theta \sim F } [ e ^ { i k \theta } ]$   
1: Set r = log A and   
$N = \Biggl \lceil \frac { C } { | \gamma _ { k } | ^ { 2 } \epsilon ^ { 2 } } \log \left( \frac { 4 } { \delta } \right) \Biggr \rceil$ (F120)   
for a suficiently large universal constant $C .$   
2: for $j = 1 , \ldots , N$ do   
3: Draw $\phi _ { j }$ uniformly from $[ 0 , 2 \pi )$   
4: Prepare $D ( e ^ { i \phi _ { j } } ) \mathinner { | { 0 } \rangle }$ and apply $\lbrack { \cal { S } } ( r , \phi _ { j } ) .$   
5: Send the mode through $\mathcal { E } _ { P }$ and apply $S ( r , \phi _ { j } ) ^ { \dagger }$   
6: Measure $\hat { \Pi } = ( - 1 ) ^ { \hat { n } }$ , obtaining $Y _ { j } \in \{ - 1 , 1 \}$   
7: Set $Z _ { j } = e ^ { i k \phi _ { j } } \dot { Y _ { j } } / \gamma _ { k } ,$   
8: end for   
9: Output   
$\widehat { \Lambda } _ { k } = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } Z _ { j } .$ (F121)

Theorem F.9 (Angular Fourier learning with ${ \mathsf { S U } } ( 1 , 1 )$ interferometry). $_ { L e t \ P }$ be any displacement distribution supported on $\{ A e ^ { i \theta } : \theta \in [ 0 , 2 \pi ) \}$ with $A = { \sqrt { k } }$ . Algorithm $F \mathcal { Q } a$ outputs $\widehat { \Lambda } _ { k }$ satisfying

$$
\left| \widehat { \Lambda } _ { k } - \Lambda _ { k } ( P ) \right| \leq \epsilon\tag{F122}
$$

with probability at least $1 - \delta$ using

$$
N = O \left( \frac { k ^ { 2 } } { \epsilon ^ { 2 } } \log \left( \frac { 1 } { \delta } \right) \right)\tag{F123}
$$

signal queries. The mean photon number used is $O ( k )$

Proof. Fix a deterministic displacement $A e ^ { i \theta }$ and write $u = \theta - \phi$ . After applying the inverse squeezer, the output is, up to a global phase, the coherent state

$$
\left| { \psi _ { \theta , \phi } } \right. = \left| { e ^ { i \phi } \left( { 1 + \cos u + i k \sin u } \right) } \right.\tag{F124}
$$

which follows from the commutation relation between squeezing and displacement operators. Since $\langle \alpha | \hat { \Pi } | \alpha \rangle =$ $e ^ { - 2 | \alpha | ^ { 2 } }$ , the measurement response is given by

$$
\mathbb { E } [ Y | \theta , \phi ] = q _ { k } ( \theta - \phi ) ,\tag{F125}
$$

where

$$
q _ { k } ( u ) = \exp \left( - 2 ( 1 + \cos u ) ^ { 2 } - 2 k ^ { 2 } \sin ^ { 2 } u \right) .\tag{F126}
$$

Averaging over the random squeezing angle and applying the normalization factor gives

$$
\begin{array} { l } { { \displaystyle \mathbb { E } [ Z | \theta ] = \frac { 1 } { \gamma _ { k } } \frac { 1 } { 2 \pi } \int _ { 0 } ^ { 2 \pi } d \phi e ^ { i k \phi } q _ { k } ( \theta - \phi ) } } \\ { { \displaystyle ~ = e ^ { i k \theta } . } } \end{array}\tag{F127}
$$

(F128)

It follows that $\mathbb { E } [ Z ] = \Lambda _ { k } ( P )$ for every angular distribution $F .$ Next we control the sample complexity. The response $q _ { k }$ is concentrated in neighborhoods of width $\Theta ( 1 / k )$ around $u = 0$ and $u \ = \ \pi$ . Rescaling these neighborhoods by $u = s / k$ and $u = \pi + s / k$ gives

$$
\eta _ { k } ^ { \mathrm { p a r } } = \frac { e ^ { - 1 / 8 } } { 2 \sqrt { 2 \pi } k } \left( e ^ { - 8 } + ( - 1 ) ^ { k } \right) + o ( k ^ { - 1 } ) .\tag{F129}
$$

Consequently, there are universal constants $c , C > 0$ such that

$$
{ \frac { c } { k } } \leq | \eta _ { k } ^ { \mathrm { p a r } } | \leq { \frac { C } { k } } .\tag{F130}
$$

The normalization is nonzero for every k: indeed,

$$
q _ { k } ( u ) = e ^ { - ( k ^ { 2 } + 3 ) } e ^ { - 4 \cos u } e ^ { ( k ^ { 2 } - 1 ) \cos ( 2 u ) } ,\tag{F131}
$$

and its modified-Bessel expansion shows that the sign of its k-th Fourier coeficient is $( - 1 ) ^ { k }$ . Since $| Y | = 1$ each postprocessed outcome satisfies $| Z | = O ( k )$ . Hoefding’s inequality applied to the real and imaginary parts therefore gives

$$
\mathrm { P r } \left[ \left| \widehat { \Lambda } _ { k } - \Lambda _ { k } ( P ) \right| > \epsilon \right] \leq 4 \exp \left( - c \frac { N \epsilon ^ { 2 } } { k ^ { 2 } } \right) ,\tag{F132}
$$

which proves the sample-complexity claim.

Finally, the state incident on the signal is $S ( r , \phi ) D ( e ^ { i \phi } ) \left| 0 \right.$ . Its mean photon number is bounded by

$$
\overline { { { n } } } _ { \mathrm { p r o b e } } \leq e ^ { 2 r } + \sinh ^ { 2 } r = O ( k ) .\tag{F133}
$$

The signal displacement has amplitude $A = { \sqrt { k } } ,$ so the occupation remains $O ( k )$ throughout the sensing window. Moreover, squeezing preserves photon-number parity, so $[ S ( r , \phi ) , \hat { \Pi } ] = 0$ and the inverse squeezer can equivalently be absorbed into the parity readout. Thus it does not require a larger probe-energy budget. □

The non-Gaussian resource in this protocol is the photon-counting readout itself. At the level of non-Gaussian resource counting, the bosonic phase supplied by the ECD echo may be represented by a Kerr unitary of the form

$$
K _ { \pi / 2 } = \exp \left( \frac { i \pi } { 2 } \hat { n } ^ { 2 } \right) ,\tag{F134}
$$

where $\hat { n } = \hat { a } ^ { \dag } \hat { a }$ and $\hat { n } ^ { 2 }$ is fourth order in the ladder operators. At this special phase,

$$
K _ { \pi / 2 } = \frac { 1 + i } { 2 } I + \frac { 1 - i } { 2 } \hat { \Pi } ,\tag{F135}
$$

because its eigenvalue is 1 on even photon numbers and i on odd photon numbers. A single photon-number measurement therefore supplies exactly the spectral information needed to evaluate this quartic phase. Under this accounting, the squeezed-parity protocol uses the same one layer of non-Gaussian resource as the ECD protocol.

## 3. Learning with quantum control and memory

Next, we illustrate the power of a single qubit of long-lived quantum memory for sensing. In Section E 3 we established that energy-constrained protocols without long-lived degrees of freedom, even with arbitrary ancillas, modes, and measurements, incur an exponential cost to estimate temporal Fourier coeficients. Here we show that a single qubit of memory coherent over the lifetime of a time-varying signal enables eficient estimation of these temporal correlations, even if the sensor itself decoheres quickly.

First let us understand how temporal Fourier coeficients capture properties of time-varying signals. Let P denote the probability law of the time-dependent displacement signal whose induced displacements at diferent points in time are described by a stochastic process $\{ Z _ { t } \} _ { t \in \mathcal { T } } .$ , where $Z _ { t } = ( X _ { t } , P _ { t } ) \in \mathbb { R } ^ { 2 }$ . For a finite list of times $\mathbf { t } = \left( t _ { 1 } , \ldots , t _ { m } \right)$ , the resulting marginal distribution is $P _ { \mathbf { t } }$ on $( \mathbb { R } ^ { 2 } ) ^ { m }$ defined by

$$
P _ { \mathbf { t } } ( A _ { 1 } \times \cdots \times A _ { m } ) = P [ Z _ { t _ { 1 } } \in A _ { 1 } , \ldots , Z _ { t _ { m } } \in A _ { m } ] .\tag{F136}
$$

In Quantum Signal Learning language, physical observables of such time-varying signals are defined by a kernel

function $F : ( \mathbb { R } ^ { 2 } ) ^ { m } \to \mathbb { C }$ , such that a property is given by

$$
\mathbb { E } _ { P _ { \mathbf { t } } } [ F ] = \int _ { ( \mathbb { R } ^ { 2 } ) ^ { m } } F ( z _ { 1 } , \ldots , z _ { m } ) p _ { \mathbf { t } } ( z _ { 1 } , \ldots , z _ { m } ) \prod _ { j = 1 } ^ { m } d ^ { 2 } z _ { j } .\tag{F137}
$$

where $P _ { \mathbf { t } }$ has a density $p _ { \mathbf { t } }$ . The temporal Fourier coeficient is the special case in which the kernel is a product of Weyl characters:

$$
\widehat { P } ( { \bf t } , \zeta ) = \int _ { ( \mathbb { R } ^ { 2 } ) ^ { m } } \prod _ { j = 1 } ^ { m } e ^ { i \Omega ( \zeta _ { j } , z _ { j } ) } P _ { \bf t } ( d z _ { 1 } \cdot \cdot \cdot d z _ { m } ) .\tag{F138}
$$

This is the characteristic function of the joint distribution $P _ { \mathbf { t } }$ . Note, importantly, that $P _ { \mathbf { t } }$ is generally not the product $P _ { t _ { 1 } } \otimes \cdots \cdot \otimes P _ { t _ { m } }$ . The temporal correlations are exactly the information contained in the joint distribution that is not captured by single-time marginals.

This picture shows us that other temporal observables can be recovered from measuring temporal Fourier coeficients, because these objects can be used to identify the characteristic function of the signal in an informationally-complete manner. Let $\psi : ( \mathbb { R } ^ { 2 } ) ^ { m } \to \mathbb { C }$ be any property kernel with symplectic Fourier transform

$$
\widehat { \psi } ( \zeta ) = \frac { 1 } { ( 2 \pi ) ^ { 2 m } } \int _ { ( \mathbb { R } ^ { 2 } ) ^ { m } } \psi ( \mathbf { z } ) \exp \left( - i \sum _ { j = 1 } ^ { m } \Omega ( \zeta _ { j } , z _ { j } ) \right) \prod _ { j = 1 } ^ { m } d ^ { 2 } z _ { j } ,\tag{F139}
$$

and assume $\widehat \psi \in L ^ { 1 }$ . The inverse symplectic Fourier transform gives

$$
\psi ( z _ { 1 } , \ldots , z _ { m } ) = \int _ { ( \mathbb { R } ^ { 2 } ) ^ { m } } \widehat { \psi } ( \zeta ) \exp \left( i \sum _ { j = 1 } ^ { m } \Omega ( \zeta _ { j } , z _ { j } ) \right) \prod _ { j = 1 } ^ { m } d ^ { 2 } \zeta _ { j } .\tag{F140}
$$

Consequently, the corresponding temporal observable $\Psi _ { \psi } ( P _ { \mathbf { t } } ) = \mathbb { E } _ { P _ { \mathbf { t } } } [ \psi ( Z _ { t _ { 1 } } , \dots , Z _ { t _ { m } } ) ]$ can be written as

$$
\Psi _ { \psi } ( P _ { \mathbf { t } } ) = \int _ { ( \mathbb { R } ^ { 2 } ) ^ { m } } \widehat { \psi } ( \zeta ) \widehat { P } ( \mathbf { t } , \zeta ) \prod _ { j = 1 } ^ { m } d ^ { 2 } \zeta _ { j } .\tag{F141}
$$

We therefore see that estimation of $\widehat { P } ( \mathbf { t } , \zeta )$ , the appropriate temporal Fourier coeficients, enables downstream estimation of $\Psi _ { \psi } ( P _ { \mathbf { t } } )$ using classical postprocessing (with a corresponding conditioning-factor overhead depending on the complexity of $\psi )$ . For many observables with controlled Fourier support the additional measurement overhead required for this postprocessing can be modest. A practical example of physical observables that can be extracted from temporal Fourier coeficients are ordinary temporal moments, which emerge from taking derivatives at the origin. For a quadrature direction u $\in \mathbb { R } ^ { 2 }$ , let $\zeta ( u )$ denote the phase-space vector satisfying $\Omega ( \zeta ( u ) , z ) = u ^ { T } z$ for all $z \in \mathbb { R } ^ { 2 }$ . If the moment exists and the characteristic function is diferentiable at the origin, then

$$
\mathbb { E } _ { P _ { \star } } \left[ \prod _ { j = 1 } ^ { m } u _ { j } ^ { T } Z _ { t _ { j } } \right] = ( - i ) ^ { m } \left. \frac { \partial ^ { m } } { \partial \lambda _ { 1 } \cdots \partial \lambda _ { m } } \widehat { P } \left( \mathbf { t } , ( \lambda _ { 1 } \zeta ( u _ { 1 } ) , \ldots , \lambda _ { m } \zeta ( u _ { m } ) ) \right) \right| _ { \lambda _ { 1 } = \cdots = \lambda _ { m } = 0 } .\tag{F142}
$$

Thus we have seen that the temporal Fourier coeficients are a general and highly expressive object for characterizing time-varying signals. Now we give a quantum feature sensing protocol using a long-lived memory qubit that estimates these coeficients directly.

Algorithm 5 Temporal Fourier coeficient estimation with one memory qubit   
Input: Times $\mathbf { t } = \left( t _ { 1 } , \ldots , t _ { m } \right)$ , phase-space frequencies ${ \boldsymbol { \zeta } } = ( \zeta _ { 1 } , \ldots , \zeta _ { m } ) .$ , number of trajectories $N ,$ and query access to   
independent realizations of the time-dependent displacement process.   
Output: An estimate $\widehat { G }$ of $\widehat { P } ( \mathbf { t } , \zeta )$   
1: for $\ell = 1 , \ldots , N$ do   
2: Prepare the memory qubit in $| + \rangle = ( | g \rangle + | e \rangle ) / { \sqrt { 2 } } .$   
3: for $j = 1 , \ldots , m$ do   
4: Prepare or retain the sensor mode in any state independent of the qubit.   
5: Apply $C _ { \zeta _ { j } } .$   
6: Let the signal act on the sensor mode at time $t _ { j }$ , applying the unknown displacement $D ( Z _ { t _ { j } } )$ for the current   
trajectory.   
7: Apply $\boldsymbol { C } _ { \zeta _ { j } } ^ { \dagger }$   
8: Discard, reset, or leave the sensor mode idle before the next time point.   
9: end for   
10: With probability $1 / 2 ,$ , measure $X _ { q }$ on the memory qubit, obtain $S _ { \ell } \in \{ \pm 1 \}$ , and set $W _ { \ell } = 2 S _ { \ell }$   
11: With probability $1 / 2$ , measure $Y _ { q }$ on the memory qubit, obtain $S _ { \ell } \in \{ \pm 1 \}$ , and set $W _ { \ell } = 2 i S _ { \ell }$   
12: end for   
13: Output   
$\widehat { G } = \frac { 1 } { N } \sum _ { \ell = 1 } ^ { N } W _ { \ell } .$ (F143)

Theorem F.10 (Temporal Fourier learning with one memory qubit). For any time-dependent displacement process $P ,$ Algorithm 5 estimates $\widehat { P } ( \mathbf { t } , \zeta )$ to additive error ϵ with probability at least $1 - \delta$ using

$$
N = O \left( \epsilon ^ { - 2 } \log \left( \frac { 1 } { \delta } \right) \right)\tag{F144}
$$

independent signal trajectories. Each trajectory uses m controlled-displacement echoes and one qubit that remains coherent over the time interval containing t. The sensor mode need not remain coherent between time points after each echo is closed.

Proof. Fix a deterministic trajectory $z _ { 1 } , \ldots , z _ { m }$ , where $z _ { j }$ is the displacement at time $t _ { j }$ . We first compute one echo. By the Weyl relation,

$$
C _ { \zeta } ^ { \dagger } ( D ( z ) \otimes I _ { q } ) C _ { \zeta } = D ( z ) \otimes \left( e ^ { - i \Omega ( \zeta , z ) / 2 } | g \rangle \langle g | + e ^ { i \Omega ( \zeta , z ) / 2 } | e \rangle \langle e | \right) .\tag{F145}
$$

The oscillator displacement $D ( z )$ is common to both qubit branches. Therefore, up to an irrelevant global phase, the qubit transformation is

$$
\left| g \right. + \xi \left| e \right. \mapsto \left| g \right. + \xi e ^ { i \Omega ( \zeta , z ) } \left| e \right. .\tag{F146}
$$

Applying this identity at the m requested time points gives, for the fixed trajectory,

$$
| + \rangle \mapsto \frac 1 { \sqrt { 2 } } \left( | g \rangle + \Phi _ { \zeta } ( z _ { 1 } , \dots , z _ { m } ) | e \rangle \right) ,\tag{F147}
$$

where the accumulated phase is

$$
\Phi _ { \zeta } ( z _ { 1 } , \dots , z _ { m } ) = \prod _ { j = 1 } ^ { m } e ^ { i \Omega ( \zeta _ { j } , z _ { j } ) } = \exp \left( i \sum _ { j = 1 } ^ { m } \Omega ( \zeta _ { j } , z _ { j } ) \right) .\tag{F148}
$$

Since the oscillator factor is common to the two qubit branches after every echo, tracing out or resetting the oscillator after a time point does not change the qubit coherence.

Averaging over the trajectory law $P _ { \mathbf { t } } .$ , the final qubit satisfies

$$
\mathbb { E } [ X _ { q } ] = \mathrm { R e } \widehat { P } ( { \mathbf t } , \zeta ) , \qquad \mathbb { E } [ Y _ { q } ] = \mathrm { I m } \widehat { P } ( { \mathbf t } , \zeta ) .\tag{F149}
$$

The randomized $X _ { q } / Y _ { q }$ readout used in the algorithm therefore has expectation

$$
\mathbb { E } [ W _ { \ell } ] = \widehat { P } ( \mathbf { t } , \zeta ) .\tag{F150}
$$

Moreover $| W _ { \ell } | = 2$ for every shot. Hoefding’s inequality applied to the real and imaginary parts gives

$$
\operatorname* { P r } \left[ \left| \frac { 1 } { N } \sum _ { \ell = 1 } ^ { N } W _ { \ell } - \widehat { P } ( \mathbf { t } , \zeta ) \right| > \epsilon \right] \leq 4 \exp \left( - c N \epsilon ^ { 2 } \right)\tag{F151}
$$

for a universal constant $c > 0$ . Taking $N = O ( \epsilon ^ { - 2 } \log ( 1 / \delta ) )$ proves the stated query complexity.

Finally, the parity instance from Theorem E.6 can be directly solved with this protocol. In that construction, the signal alphabet at time $t _ { j }$ is

$$
Z _ { t _ { j } } = B _ { j } ( x _ { \star } , 0 ) , \qquad B _ { j } \in \{ 0 , 1 \} .\tag{F152}
$$

Choose $\zeta _ { \star }$ so that $\Omega ( \zeta _ { \star } , ( x , p ) ) = \lambda ,$ x and $\lambda _ { \star } x _ { \star } = \pi$ . Then the local Weyl character is

$$
e ^ { i \Omega ( \zeta _ { \star } , Z _ { t _ { j } } ) } = e ^ { i \pi B _ { j } } = ( - 1 ) ^ { B _ { j } } = 1 - \frac { 2 X _ { t _ { j } } } { x _ { \star } } .\tag{F153}
$$

Applying Algorithm 5 with $\boldsymbol { \zeta } = ( \zeta _ { \star } , \ldots , \zeta _ { \star } )$ therefore estimates

$$
{ \widehat { P } } ( ( t _ { 1 } , \dots , t _ { m } ) , ( \zeta _ { \star } , \dots , \zeta _ { \star } ) ) = \mathbb { E } \left[ \prod _ { j = 1 } ^ { m } ( - 1 ) ^ { B _ { j } } \right]\tag{F154}
$$

$$
= \mathbb { E } \left[ \prod _ { j = 1 } ^ { m } \left( 1 - \frac { 2 X _ { t _ { j } } } { x _ { \star } } \right) \right] .\tag{F155}
$$

Under the two distributions $P _ { \sigma }$ used in the lower bound, this expectation is $\sigma \epsilon _ { \mathrm { p a r } } .$ giving us the desired separation.

Corollary 11. Given a list of times $\mathbf { t } = t _ { 1 } , t _ { 2 } , . . . , t _ { m }$ separated by at most $\Delta$ , a single quantum oscillator that may be coherent for time $\ll \Delta$ , and a single qubit coherent over the time span of t, there is an algorithm that can learn any temporal Fourier coeficient indexed by t using $O _ { m } ( 1 )$ signal queries using an energy $E _ { 0 }$ independent of m. Any strategy with energy $E _ { 0 }$ including arbitrary ancilla qubits or modes, and any non-Gaussian control, but without resources coherent for time $> \Delta$ requires $\exp ( \Omega ( m ) )$ signal queries.

## 4. A quadratic advantage over infinite-energy conventional quantum sensing

In Section E 4, we demonstrated a quadratic lower bound for estimating the variance of a single-mode quantum state. We considered the physically-motivated task of learning the strength of a weak thermal background by bounding the sample complexity of discriminating the two single-mode states

$$
\rho _ { 0 } ^ { ( k ) } = | 0 \rangle \langle 0 | \ , \qquad \rho _ { 1 } ^ { ( k ) } = \tau _ { 1 / k } ,\tag{F156}
$$

where $\tau _ { 1 / k }$ is the thermal state with mean occupation number $1 / k$

$$
\tau _ { t } = \sum _ { n = 0 } ^ { \infty } \frac { t ^ { n } } { ( 1 + t ) ^ { n + 1 } } \left| n \middle > \middle < n \right| \ ,\tag{F157}
$$

and any conventional Gaussian sensing protocol, even with access to arbitrarily high energy, requires at least $\Omega ( k ^ { 2 } )$ samples to estimate their variance to the required accuracy. Here we show that there is a simple non Gaussian measurement that estimates this variance using $O ( k )$ samples.

Lemma 12. There is a non-Gaussian measurement which estimates the variance of the above states to the required accuracy using $O ( k )$ samples.

Proof. Measure the photon number nˆ on each copy. For an arbitrary single-mode state $\rho ,$ the photon-counting outcome X satisfies

$$
\mathbb { E } [ X ] = \operatorname { T r } ( \hat { n } \rho ) \ ,\tag{F158}
$$

so the empirical photon number directly estimates the phase-space variance $v ( \rho ) = 1 \mathrm { + 2 T r } ( \hat { n } \rho )$ . For the thermal state $\tau _ { t }$ , the photon-counting distribution is geometric, with

$$
\mathbb { E } [ X ] = t , \qquad \operatorname { V a r } ( X ) = t ( 1 + t ) ~ .\tag{F159}
$$

Let $\widehat { t }$ be the empirical mean of N photon-counting outcomes and define $\widehat { v } = 1 + 2 \widehat { t } .$ For the states above, $t \in \{ 0 , 1 / k \}$ , and hence

$$
\operatorname { V a r } ( X ) \leq { \frac { 1 } { k } } \left( 1 + { \frac { 1 } { k } } \right) \ .\tag{F160}
$$

Standard concentration for geometric random variables therefore gives, for a universal constant $c > 0 .$

$$
\mathrm { P r } \left( \left| \widehat { t } - t \right| \geq \frac { 1 } { 4 k } \right) \leq 2 \exp \left( - c \frac { N } { k } \right) .\tag{F161}
$$

On the complementary event,

$$
\left| \widehat { v } - v ( \rho ) \right| = 2 \left| \widehat { t } - t \right| \leq \frac { 1 } { 2 k } .\tag{F162}
$$

Taking $N \ge C k \log ( 2 / \delta )$ for a suficiently large universal constant $C$ makes the failure probability at most $\delta .$ Thus $O ( k \log ( 1 / \delta ) )$ samples sufice, and in particular $O ( k )$ samples sufice for constant success probability.

For solving only the two-hypothesis discrimination task, one can also obtain a quadratic speedup over Gaussian protocols by using a parity measurement with POVM $\{ \Pi _ { \mathrm { e v e n } } , \Pi _ { \mathrm { o d d } } \}$ . Let $\Pi _ { \mathrm { o d d } }$ be the projector onto the odd Fock subspace. For a thermal state $\tau _ { t } ,$

$$
\mathrm { P r } ( \mathrm { o d d } ) = \sum _ { l \geq 0 } \frac { t ^ { 2 l + 1 } } { ( 1 + t ) ^ { 2 l + 2 } } = \frac { t } { 1 + 2 t } \ .\tag{F163}
$$

Thus

$$
\mathrm { P r } ( \mathrm { o d d } ) _ { \rho _ { 0 } } = 0 \qquad \mathrm { P r } ( \mathrm { o d d } ) _ { \rho _ { 1 } } = { \frac { 1 } { k + 2 } } .\tag{F164}
$$

More generally, if $p _ { \mathrm { o d d } } = \mathrm { P r } ( \mathrm { o d d } )$ for a thermal state, then

$$
t = { \frac { p _ { \mathrm { o d d } } } { 1 - 2 p _ { \mathrm { o d d } } } } \ .\tag{F165}
$$

Estimating $p _ { \mathrm { o d d } }$ by the empirical odd-outcome frequency and applying the same concentration argument therefore estimates $v ( \tau _ { t } ) = 1 + 2 t$ to accuracy $1 / ( 2 k )$ using ${ \cal O } ( k \log ( 1 / \delta ) )$ samples. This parity measurement can be realized using the Ramsey sequence

$$
\left| + \right. \longrightarrow \mathrm { e x p } ( ( - i \pi / 2 ) \sigma _ { z } \hat { n } ) \longrightarrow R _ { y } ( - \pi / 2 ) \longrightarrow Z \mathrm { - m e a s u r e m e n t } \ .\tag{F166}
$$

Together, Lemma 12 and Theorem E.9 illustrate a simple but illuminating result.

Corollary 13. There exists a family of single-mode Gaussian states whose variance can be estimated by a non-Gaussian protocol in quadratically fewer samples than by any arbitrarily high-energy adaptive Gaussian protocol.

This result illustrates that non-Gaussianity as is aforded by single-qubit control is a resource that cannot be replicated by any high-energy Gaussian processing. While this is well-established in the case of computation [82], where non-Gaussian operations are analogous to the non-Cliford gates that enable universal quantum computation, the above result establishes that non-Gaussianity is also an irreplaceable resource in learning and sensing. This is operationally crucial given that conventional cutting-edge sensing platforms such as optica cavities utilize high-performance Gaussian modes that by design struggle to prepare non-Gaussian states and perform non-Gaussian measurements.

## 5. Entanglement-free multimode advantage for learning characteristic functions with one qubit

In Section E 5, we identified a gap in the entanglement-free lower bounds of Ref. [1]. Here we show that the gap is operationally meaningful. When the displacement distribution is suficiently concentrated in phase-space, single-qubit control of the bosonic sensor enables a non-Gaussian readout that approximately samples the actual displacement drawn by the channel. Once such samples are obtained, post-hoc characteristic-function learning is just classical empirical Fourier estimation.

## a. Displacement sensing with quantum phase estimation and GKP states

We first provide a basic overview Gottesman-Kitaev-Preskill (GKP) states as they apply to the characteristicfunction learning protocol. Here we do not review their more well-known role in bosonic quantum error correction. For a single bosonic mode let $\hat { x } , \hat { p }$ denote canonical quadratures satisfying $[ \hat { x } , \hat { p } ] = i$ . For $( x , p ) \in \mathbb { R } ^ { 2 }$ we write the standard displacement operator as $D ( x , p ) = \exp \left( i ( p { \hat { x } } - x { \hat { p } } ) \right)$ . In these coordinates, the Weyl commutation relations, Equation (C5), imply

$$
D ( x , p ) D ( x ^ { \prime } , p ^ { \prime } ) = \exp \left( - \frac { i } { 2 } ( x p ^ { \prime } - p x ^ { \prime } ) \right) D ( x + x ^ { \prime } , p + p ^ { \prime } ) ,\tag{F167}
$$

$$
D ( x , p ) D ( x ^ { \prime } , p ^ { \prime } ) = \exp \left( - i ( x p ^ { \prime } - p x ^ { \prime } ) \right) D ( x ^ { \prime } , p ^ { \prime } ) D ( x , p ) \ .\tag{F168}
$$

For n modes, we write $\alpha = ( ( x _ { 1 } , p _ { 1 } ) , \ldots , ( x _ { n } , p _ { n } ) ) \in \mathbb { R } ^ { 2 n }$ and use

$$
D ( \alpha ) = \bigotimes _ { m = 1 } ^ { n } D ( x _ { m } , p _ { m } ) \ .\tag{F169}
$$

We also write the symplectic form as

$$
\Omega ( \beta , \alpha ) = \sum _ { m = 1 } ^ { n } ( x _ { m } p _ { m } ^ { \prime } - p _ { m } x _ { m } ^ { \prime } )\tag{F170}
$$

when $\beta = ( ( x _ { 1 } , p _ { 1 } ) , \dotsc , ( x _ { n } , p _ { n } ) )$ and $\alpha = ( ( x _ { 1 } ^ { \prime } , p _ { 1 } ^ { \prime } ) , \ldots , ( x _ { n } ^ { \prime } , p _ { n } ^ { \prime } ) )$ . With this convention, the characteristic function of a displacement distribution $P$ is

$$
\begin{array} { r } { \lambda _ { P } \big ( \beta \big ) = \mathbb { E } _ { \alpha \sim P } \left[ e ^ { i \Omega \left( \beta , \alpha \right) } \right] \ . } \end{array}\tag{F171}
$$

We now define the square GKP grid state. Let $L = { \sqrt { 2 \pi } }$ and consider the two displacement operators

$$
S _ { x } = D ( L , 0 ) , \qquad S _ { p } = D ( 0 , L ) \ .\tag{F172}
$$

These commute because the their commutator, from the Weyl relation, is the phase exp $\left( - i L ^ { 2 } \right) = \exp ( - i 2 \pi ) = 1$ The ideal square GKP grid state is the simultaneous +1 eigenstate of $S _ { x }$ and $S _ { p }$ . In the x basis, it may be written formally as the Dirac comb

$$
| \mathrm { G K P } \rangle \propto \sum _ { s \in \mathbb { Z } } | x = s L \rangle .\tag{F173}
$$

However, this is not a normalizable physical state, and we will shortly discuss a more physical realization; nevertheless, the Dirac comb representation is the simplest way to understand the properties of a GKP state. Importantly, it has peaks separated by L in x, and because it is also a +1 eigenstate of $D ( 0 , L )$ , it has the corresponding periodic structure in $p .$ Now suppose an unknown signal acts on a GKP state by applying a

displacement $D ( a )$ , where $a = \left( x _ { a } , p _ { a } \right)$ . Then

$$
S _ { x } D ( a ) \left| \mathrm { G K P } \right. = e ^ { - i L p _ { a } } D ( a ) \left| \mathrm { G K P } \right. ,\tag{F174}
$$

$$
S _ { p } D ( a ) \left| \mathrm { G K P } \right. = e ^ { i L x _ { a } } D ( a ) \left| \mathrm { G K P } \right. .\tag{F175}
$$

So after the displacement, the phases of the two stabilizers contain $p _ { a }$ and $x _ { a }$ modulo L (more precisely, modulo $2 \pi / L = L )$ . If one could then measure the GKP stabilizer phases, the resulting two quantities form a mod $L \mathbb { Z } ^ { 2 }$ Importantly, simultaneous measurements of these phases are possible precisely because $S _ { x }$ and $S _ { p }$ commute. This idea extends to the multimode setting. For n modes, we consider the product grid state $\left| \operatorname { G K P } \right. ^ { \otimes n }$ . The corresponding measurement of all $2 n$ phases, via the corresponding commuting stabilizers, returns α mod $L \mathbb { Z } ^ { 2 n }$ We will call the centered fundamental cell

$$
\mathcal { V } _ { L } = \left[ - \frac { L } { 2 } , \frac { L } { 2 } \right) ^ { 2 n }\tag{F176}
$$

the GKP decoding cell. From the above, we see that the true displacement α lies in $\mathcal { V } _ { L }$ , then the modular outcome can be used to recover α itself. However if $\alpha$ lies outside $\mathcal { V } _ { L }$ , then the measurement wraps it back into the cell and we will misidentify the true displacement. We refer to such an error as an aliasing error.

However, as we discussed, the ideal GKP state is not physically realizable, and it remains to explain how to extract the stabilizer phases. We will resolve these issues simultaneously by using Quantum Phase Estimation (QPE) to demonstrate a quantum circuit which uses ancilla qubit ECD control of bosonic sensing modes to prepare and read out displacement phases from GKP probe states. First, we recall the well-known QPE primitive. Let $M = 2 ^ { r }$ and let $\mathbb { Z } _ { M } = \{ 0 , \ldots , M - 1 \}$ . The phase-estimation circuit takes a unitary U and tries to learn an eigenphase $\theta \in [ 0 , 1 )$ , where $U \left| \theta \right. = e ^ { 2 \bar { \pi } i \theta } \left| \theta \right.$ . The procedure, described in Algorithm 6, works by coherently applying many powers of $U$ and then Fourier transforming the accumulated phases.

```latex
Algorithm 6 M-bin quantum phase estimation for a unitary U
Input: A system state $\rho ,$ controlled access to the powers $U ^ { 2 ^ { j } }$ for $j = 0 , \ldots , r - 1 ,$ , and $M = 2 ^ { r }$
Output: A classical outcome $b \in \mathbb { Z } _ { M } .$
1: Prepare an r-qubit register in $| 0 \rangle ^ { \otimes r } .$
2: Apply Hadamards to the register, obtaining $\begin{array} { r } { \frac { 1 } { \sqrt { M } } \sum _ { t = 0 } ^ { M - 1 } \left| t \right. } \end{array}$
3: for $j = 0 , \ldots , r - 1$ do
4: Apply controlled- $\cdot U ^ { 2 ^ { j } }$ from the j-th register qubit to the system.
5: end for
6: The joint operation has mapped |t⟩ |ψ⟩ to $| t \rangle U ^ { t } | \psi \rangle$ . Apply the inverse Fourier transform $F _ { M } ^ { \dagger }$ on the register, where
$\begin{array} { r } { F _ { M } ^ { \dagger } \left| t \right. = \frac { 1 } { \sqrt { M } } \sum _ { b = 0 } ^ { M - 1 } e ^ { - 2 \pi i b t / M } \left| b \right. } \end{array}$
7: Measure the register in the computational basis and output the resulting bitstring $b .$
```

When the given state is an eigenstate of $U _ { ; }$ , the algorithmic guarantee is as follows. If the system is in eigenstate $| \theta \rangle$ , then after the controlled powers of U the joint state is

$$
\frac { 1 } { \sqrt { M } } \sum _ { t = 0 } ^ { M - 1 } e ^ { 2 \pi i t \theta } \left| t \right. \left| \theta \right. .\tag{F177}
$$

Thus the ancilla register contains a length-M complex phase with frequency θ. The inverse Fourier transform is exactly the operation that converts this phase into a computational-basis label. Assuming $\theta = b _ { 0 } / M$ lies exactly on an equally spaced M-point grid of phases,

$$
F _ { M } ^ { \dagger } \left( \frac { 1 } { \sqrt { M } } \sum _ { t = 0 } ^ { M - 1 } e ^ { 2 \pi i t b _ { 0 } / M } \left| t \right. \right) = \frac { 1 } { M } \sum _ { b = 0 } ^ { M - 1 } \sum _ { t = 0 } ^ { M - 1 } e ^ { 2 \pi i t b _ { 0 } / M } e ^ { - 2 \pi i b t / M } \left| b \right.\tag{F178}
$$

$$
= \sum _ { b = 0 } ^ { M - 1 } \left( \frac { 1 } { M } \sum _ { t = 0 } ^ { M - 1 } e ^ { 2 \pi i t ( b _ { 0 } - b ) / M } \right) \left| b \right.\tag{F179}
$$

$$
\quad = \left. b _ { 0 } \right. .\tag{F180}
$$

For a general continuous phase, a similar calculation shows that the output is distributed according to

$$
Q _ { M } ( b | \theta ) = \frac { 1 } { M ^ { 2 } } \left| \sum _ { t = 0 } ^ { M - 1 } e ^ { 2 \pi i t ( \theta - b / M ) } \right| ^ { 2 } = \frac { 1 } { M ^ { 2 } } \frac { \sin ^ { 2 } ( \pi M ( \theta - b / M ) ) } { \sin ^ { 2 } ( \pi ( \theta - b / M ) ) } \ ,\tag{F181}
$$

The second equality is defined away from any grid point $\theta = b / M ,$ but the exact distribution is continuous and well-defined everywhere. The distribution $Q _ { M } ( b | \theta )$ is sharply peaked near the closest grid point to θ. For a general input state, Algorithm 6 defines a measurement with Kraus operators

$$
A _ { b } ^ { ( M ) } ( U ) = \frac { 1 } { M } \sum _ { t = 0 } ^ { M - 1 } e ^ { - 2 \pi i b t / M } U ^ { t } \ .\tag{F182}
$$

These operators satisfy $\begin{array} { l r } { \sum _ { b = 0 } ^ { M - 1 } A _ { b } ^ { ( M ) } ( U ) ^ { \dagger } A _ { b } ^ { ( M ) } ( U ) } & { = } & { I , } \end{array}$ which follows from the finite Fourier identity $\begin{array} { r } { \sum _ { b = 0 } ^ { M - 1 } e ^ { 2 \pi i b ( t - t ^ { \prime } ) / M } = M \bar { \delta _ { t , t ^ { \prime } } } } \end{array}$ . QPE then measures the expected value of the phase of U, with finite resolution $\mathrm { i } / M$

We can now see that QPE gives us a natural way to extract displacement-induced phases from GKP states. To align notation, we define

$$
U _ { x } = S _ { p } = D ( 0 , L ) , \qquad U _ { p } = S _ { x } ^ { \dagger } = D ( - L , 0 ) \ .\tag{F183}
$$

This choice simply fixes a sign convention. The unitary $U _ { x }$ is used to estimate an x displacement, and $U _ { p }$ used to estimate a p displacement. Indeed, if the signal applies $D ( x _ { a } , p _ { a } )$ , the Weyl relation gives

$$
U _ { x } D ( x _ { a } , p _ { a } ) = e ^ { i L x _ { a } } D ( x _ { a } , p _ { a } ) U _ { x } \ ,\tag{F184}
$$

$$
U _ { p } D ( x _ { a } , p _ { a } ) = e ^ { i { \cal L } p _ { a } } D ( x _ { a } , p _ { a } ) U _ { p } ~ .\tag{F185}
$$

Since $L ^ { 2 } = 2 \pi$ , the phases $e ^ { i L x _ { a } }$ and $e ^ { i L p _ { a } }$ are equal to $e ^ { 2 \pi i x _ { a } / L }$ and $e ^ { 2 \pi i p _ { a } / L }$ . Therefore, if we estimate the phases of $U _ { x }$ and $U _ { p }$ with respect to the probe state before and after the unknown displacement, their diferences estimate $x _ { a } / L$ and $p _ { a } / L$ modulo 1. Multiplying by $L$ gives the displacement modulo $L \mathbb { Z } ^ { 2 }$ . For $k \in \mathbb { Z } _ { M }$ , let cent ${ } _ { M } ( k )$ denote the unique integer representative of k in $[ - M / 2 , M / 2 )$ . Thus L cent $_ M ( k ) / M$ lies in $[ - L / 2 , L / 2 )$ and will encode the displacement coordinates which one obtains from their corresponding phase. We now give the actual finite-GKP sensing primitive in Algorithm 7.

Algorithm 7 Finite-GKP displacement sensing by phase estimation   
Input: One query to an n-mode displacement channel, $L = { \sqrt { 2 \pi } } ,$ and $M = 2 ^ { r }$   
Output: A decoded modular displacement estimate $\widehat { \alpha } \in \mathcal { V } _ { L }$   
1: Prepare the n-mode vacuum state. For each mode $m ,$ draw $\left( \xi _ { x , m } , \xi _ { p , m } \right)$ uniformly from $[ - L / 2 , L / 2 ) ^ { 2 }$ and apply the   
known displacement $D ( \xi _ { x , m } , \xi _ { p , m } )$   
2: for $m = 1 , \ldots , n$ do   
3: Run Algorithm 6 on $U _ { x , m } = D _ { m } ( 0 , L )$ and record $b _ { x , m } ^ { ( 0 ) } .$   
4: Run Algorithm 6 on $U _ { p , m } = D _ { m } ( - L , 0 )$ and record $b _ { p , m } ^ { ( 0 ) } .$   
5: end for   
6: Send the n modes through the unknown displacement channel.   
7: for $m = 1 , \ldots , n$ do   
Run Algorithm 6 on $U _ { x , m } = D _ { m } ( 0 , L )$ and record $b _ { x , m } ^ { ( 1 ) }$   
9: Run Algorithm 6 on $U _ { p , m } = D _ { m } ( - L , 0 )$ and record $b _ { p , m } ^ { ( 1 ) }$   
10: end for   
11: for $m = 1 , \ldots , n$ do   
12: Set $k _ { x , m } = \hat { b _ { x , m } } - b _ { x , m } ^ { ( 0 ) }$ mod M and $k _ { p , m } = b _ { p , m } ^ { ( 1 ) } - b _ { p , m } ^ { ( 0 ) }$ mod M.   
13: Set $\widehat { x } _ { m } = L$ cent<sub>M</sub>(k<sub>x,m</sub>)/M and $\widehat { p } _ { m } = L$ cent<sub>M</sub> $( k _ { p , m } ) / M .$   
14: end for   
15: Output ${ \widehat { \alpha } } = { \big ( } { \big ( } { \widehat { x } } _ { 1 } , { \widehat { p } } _ { 1 } { \big ) } , \dots , { \big ( } { \widehat { x } } _ { n } , { \widehat { p } } _ { n } { \big ) } { \big ) } ,$

The first line of Algorithm 7 is a known randomization performed by the learner. Its role is to randomize the initial stabilizer phases, so that the diference between the post-channel and pre-channel phase estimates has an exact shift-covariant distribution. After state preparation, the structure of the algorithm is simple: perform QPE on the initial state, evolve by the unknown channel, and perform QPE once again. After the first two phase-estimation measurements on a mode, the state sent through the unknown channel is precisely the finiteresolution physical instantiation of a GKP probe state. For example, conditional on preparation outcomes $b _ { x } ^ { ( 0 ) }$ and $b _ { p } ^ { ( 0 ) }$ , and on the known initial displacement ξ, the unnormalized one-mode probe is

$$
A _ { b _ { p } ^ { ( 0 ) } } ^ { ( M ) } ( U _ { p } ) A _ { b _ { x } ^ { ( 0 ) } } ^ { ( M ) } ( U _ { x } ) D ( \xi ) \left| 0 \right. = \frac 1 { M ^ { 2 } } \sum _ { s , t = 0 } ^ { M - 1 } e ^ { - 2 \pi i ( s b _ { p } ^ { ( 0 ) } + t b _ { x } ^ { ( 0 ) } ) / M } D ( - s L , t L ) D ( \xi ) \left| 0 \right. \ .\tag{F186}
$$

The probe is a finite superposition of coherent states, and the parameter M controls how sharply it approximates a stabilizer-phase eigenstate.

We will next demonstrate that Algorithm 7 is precisely the sensing primitive we want. Define the onecoordinate diference kernel

$$
H _ { M } ( k | s ) = \sum _ { b = 0 } ^ { M - 1 } \int _ { 0 } ^ { 1 } Q _ { M } ( b | \theta ) Q _ { M } ( b + k | \theta + s ) d \theta\tag{F187}
$$

for $k \in \mathbb { Z } _ { M }$ and $s \in [ 0 , 1 )$ , where $b + k$ is interpreted modulo M. Here θ is the initial stabilizer phase, s is the phase shift caused by the displacement, b is the pre-channel QPE outcome, and $b + k$ is the post-channel QPE outcome. Expanding the finite-QPE kernel in Fourier modes gives the equivalent expression

$$
H _ { M } ( k | s ) = \frac { 1 } { M } \sum _ { j = - ( M { - } 1 ) } ^ { M - 1 } \left( 1 - \frac { | j | } { M } \right) ^ { 2 } e ^ { 2 \pi i j ( s - k / M ) } .\tag{F188}
$$

The definition as an average of products of probabilities makes clear that $H _ { M } ( k | s ) \geq 0$ , and summing over $k \in \mathbb { Z } _ { M }$ gives 1. Once the diference bitstring k has been obtained, the algorithm outputs the corresponding displacement phase by centering and normalizing k. For notational convenience, define the one-dimensional decoded grid

$$
\mathcal { G } _ { M , L } = \left\{ \frac { L j } { M } : j \in \mathbb { Z } , \ - \frac { M } { 2 } \leq j < \frac { M } { 2 } \right\} \subset \left[ - \frac { L } { 2 } , \frac { L } { 2 } \right) \ .\tag{F189}
$$

For $y = L j / M \in \mathcal { G } _ { M , L }$ , define the one-coordinate readout kernel

$$
{ \sf H } _ { M , L } ( y | a ) = H _ { M } \left( j \mathrm { m o d } M \Big | \frac { a } { L } \right) .\tag{F190}
$$

Thus $\mathsf { H } _ { M , L } ( y | a )$ is the probability that the finite-QPE readout of a single coordinate returns the decoded value y, when the true coordinate displacement is a. Mathematically these probabilities are one-to-one with $H _ { M } ( k | s )$ , but here y and a correspond to estimates and actual phase-space coordinates of the sensed displacement respectively. With these definitions, we can exactly characterize the output distribution of Algorithm 7.

Lemma 14 (Algorithmic guarantee for GKP-based sensing). Suppose the channel applies a deterministic displacement $\alpha = ( \alpha _ { 1 } , \ldots , \alpha _ { 2 n } ) \in \mathbb { R } ^ { 2 n }$ , where the coordinates are ordered as $( x _ { 1 } , p _ { 1 } , \ldots , x _ { n } , p _ { n } )$ . Then Algorithm 7 outputs a grid point $\widehat { \alpha } \in \mathcal { G } _ { M , L } ^ { 2 n } \subset \mathcal { V } _ { L }$ with probability law

$$
\operatorname* { P r } \left[ { \widehat { \alpha } } = y | \alpha \right] = \prod _ { j = 1 } ^ { 2 n } \mathsf { H } _ { M , L } ( y _ { j } | \alpha _ { j } )\tag{F191}
$$

for every $y = ( y _ { 1 } , \dots , y _ { 2 n } ) \in \mathcal { G } _ { M , L } ^ { 2 n }$ . Furthermore, for every $0 < h \leq L / 2$

$$
\operatorname* { P r } [ \operatorname* { m a x } _ { j \in \{ 1 , \dots , 2 n \} } \operatorname* { m i n } _ { q \in \mathbb { Z } } | \widehat { \alpha } _ { j } - \alpha _ { j } + q L | > h | \alpha ] \leq \frac { 4 n L } { M h } \ .\tag{F192}
$$

The minimax expression simply quantifies the error in the estimate after taking the true displacement α modulo the GKP period L and centering.

Proof. We first prove the one-coordinate statement. Consider the x coordinate of one mode, so the relevant unitary is $U _ { x } = D ( 0 , L )$ . The known random displacement applied at the beginning shifts the initial eigenphase of $U _ { x }$ by a uniform random element of [0, 1). Thus the initial phase θ may be treated as uniform. If the true displacement has x coordinate $x _ { a }$ , then the phase of $U _ { x }$ is shifted by $s = x _ { a } / L$ modulo 1. By Algorithm 6, the first phase-estimation outcome is distributed as $Q _ { M } ( b _ { 0 } | \theta )$ , and the second is distributed as $Q _ { M } ( b _ { 1 } | \theta + s )$

Therefore the diference $k = b _ { 1 } - b _ { 0 }$ mod M has probability

$$
\sum _ { b = 0 } ^ { M - 1 } \int _ { 0 } ^ { 1 } Q _ { M } ( b | \theta ) Q _ { M } ( b + k | \theta + s ) d \theta = H _ { M } ( k | s ) ~ .\tag{F193}
$$

The same argument applies to the $p$ coordinate using $U _ { p } = D ( - L , 0 )$ , for which the phase shift is $p _ { a } / L$ . Since all the stabilizers used in the procedure commute, the calculation applies jointly to all 2n coordinates, giving the product formula. It remains to prove the concentration bound. For one phase-estimation measurement on an eigenphase θ, the outcome kernel satisfies

$$
Q _ { M } ( b | \theta ) = \frac { 1 } { M ^ { 2 } } \frac { \sin ^ { 2 } ( \pi M ( \theta - b / M ) ) } { \sin ^ { 2 } ( \pi ( \theta - b / M ) ) } \ .\tag{F194}
$$

away from any exact grid points. Let $\lVert u \rVert _ { \mathbb { T } }$ denote distance to the nearest integer. Since $| \sin ( \pi u ) | \geq 2 \| u \| _ { \mathbb { T } }$ for $\| u \| _ { \mathbb { T } } \leq 1 / 2$ , we have

$$
Q _ { M } ( b | \theta ) \leq \frac { 1 } { 4 M ^ { 2 } \| \theta - b / M \| _ { \mathbb { T } } ^ { 2 } }\tag{F195}
$$

whenever $b / M \neq \theta$ modulo 1. Summing this bound over all grid points with $\| \theta - b / M \| _ { \mathbb { T } } > \tau$ gives

$$
\operatorname* { P r } \left[ \lVert b / M - \theta \rVert _ { \mathbb { T } } > \tau \right] \leq \frac { 1 } { 2 M \tau }\tag{F196}
$$

for every $0 < \tau \le 1 / 2$ . The decoded coordinate error is the diference of two phase-estimation errors, multiplied by L. If this error exceeds h on the circle of circumference $L ,$ then at least one of the two phase-estimation errors must exceed $h / ( 2 L )$ . Therefore, for one coordinate,

$$
\operatorname* { P r } \left[ \mathrm { c o o r d i n a t e ~ e r r o r } > h \right] \leq \frac { 2 L } { M h } ~ .\tag{F197}
$$

A union bound over the 2n coordinates gives 4nL/(Mh).

Let us now understand the resource requirements of this sensing protocol. One M-bin QPE measurement uses $r = \log _ { 2 } M$ controlled powers of the relevant stabilizer. In our setting these powers are controlled displacements of sizes $L , 2 L , \ldots , 2 ^ { r - 1 } L ,$ , so the largest controlled displacement has magnitude $M L / 2$ . Algorithm 7 performs two stabilizer phase estimations per mode before the channel and two after the channel, for a total of 4n $\log _ { 2 } M$ controlled displacements per channel query.

The probe sent through the unknown channel has finite energy. To see the scaling, consider one mode and average over the preparation outcomes. For a single phase-estimation measurement with Kraus operators

$$
A _ { b } ^ { ( M ) } ( U ) = \frac { 1 } { M } \sum _ { t = 0 } ^ { M - 1 } e ^ { - 2 \pi i b t / M } U ^ { t } \ ,\tag{F198}
$$

summing over b gives the map

$$
\sum _ { b = 0 } ^ { M - 1 } A _ { b } ^ { ( M ) } ( U ) \rho A _ { b } ^ { ( M ) } ( U ) ^ { \dagger } = \frac { 1 } { M } \sum _ { t = 0 } ^ { M - 1 } U ^ { t } \rho U ^ { - t } ~ .\tag{F199}
$$

Thus the two pre-channel stabilizer phase-estimation measurements act, after averaging over their outcomes, like a uniform mixture over the displacements ${ \cal D } ( - s L , t L )$ with $s , t \in \{ 0 , \ldots , M - 1 \}$ . Starting from vacuum, the average photon number is therefore

$$
{ \frac { 1 } { M ^ { 2 } } } \sum _ { s , t = 0 } ^ { M - 1 } { \frac { L ^ { 2 } s ^ { 2 } + L ^ { 2 } t ^ { 2 } } { 2 } } = { \frac { L ^ { 2 } ( M - 1 ) ( 2 M - 1 ) } { 6 } } \ .\tag{F200}
$$

The initial random displacement inside the GKP cell contributes only $O ( L ^ { 2 } )$ additional energy on average. Hence one mode has average energy $O ( L ^ { 2 } M ^ { 2 } )$ , and the n-mode product probe has average energy $O ( n L ^ { 2 } M ^ { 2 } ) =$ $O ( n M ^ { 2 } )$ for fixed $L = { \sqrt { 2 \pi } }$ . To summarize, the total energy used in the protocol is $O ( n M ^ { 2 } + n M \log M ) =$

O(nM<sup>2</sup>).

The presentation above used an r-qubit phase-estimation register because it is the cleanest way to see the measurement. However, the same measurement can be implemented with a single controllable qubit by semiclassical iterative phase estimation. For the j-th bit, one prepares the qubit in |+⟩, applies the controlled stabilizer power $U ^ { 2 ^ { j } }$ , applies a qubit phase correction determined by the previously measured bits, measures the qubit, and resets it. Repeating this for $j = 0 , \ldots , r - 1$ produces exactly the same outcome distribution as Algorithm 6. Each phase-estimation unitary is a standard controlled-displacement gate that can be implemented with a single ECD, qubit rotation, and mode displacement. As such, the GKP sensing protocol is implementable within the same single-qubit control model used throughout this work.

## b. Characteristic-function learning from phase estimation

The GKP sensing protocol turns each channel use into an approximate displacement sample from the underlying distribution, up to potential modular errors. We next show that this enables eficient learning of characteristic functions without entangled resources. Fix a query radius $R > 0$ . For $0 < h < L / 2$ , define the shrunken GKP cell

$$
\mathcal { V } _ { L , h } = \left[ - \frac { L } { 2 } + h , \frac { L } { 2 } - h \right] ^ { 2 n }\tag{F201}
$$

and the corresponding functional

$$
{ \mathfrak { a } } _ { P } ( h ) = \operatorname* { P r } _ { \alpha \sim P } \left[ \alpha \notin \mathcal { V } _ { L , h } \right] \ .\tag{F202}
$$

This quantity controls the likelihood of an aliasing error, and is small exactly when the random displacement drawn by the channel usually lies inside the GKP cell, and not too close to its boundary. The role of the boundary margin is only to ensure that a small modular readout error can be interpreted as an ordinary displacement error.

Algorithm 8 Post-hoc characteristic-function learning from GKP sensing   
Input: Query access to an n-mode displacement channel $\mathcal { E } _ { P } .$ , phase-estimation resolution $M = 2 ^ { r }$ , and number of channe   
uses N.   
Output: After a query $\beta$ is revealed, an estimate $\widehat { \lambda } ( \beta )$ of $\lambda _ { P } ( \beta )$   
1: for $t = 1 , \ldots , N$ do   
2: Run Algorithm 7 on one use of $\mathcal { E } _ { P }$ and store the decoded outcome $\widehat { \alpha } ^ { ( t ) } \in \mathcal { V } _ { L }$   
3: end for   
4: After receiving a query $\beta \in \mathbb { R } ^ { 2 n }$ , output   
$\widehat { \lambda } ( \beta ) = \frac { 1 } { N } \sum _ { t = 1 } ^ { N } e ^ { i \Omega ( \beta , \widehat { \alpha } ^ { ( t ) } ) }$ (F203)

This algorithm is geared towards the dificult post-hoc characteristic function learning task, as the data $\widehat { \alpha } ^ { ( 1 ) } , \ldots , \widetilde { \alpha } ^ { ( N ) }$ are collected before a query $\beta$ is provided. The postprocessing is just empirical Fourier estimation as done in Refs. [1, 55]. We next study the performance guarantees of this learning algorithm.

Theorem F.15. Let P be any distribution on $\mathbb { R } ^ { 2 n }$ , let $R > 0 _ { : }$ , and let $0 < h < L / 2$ . Run Algorithm $\boldsymbol { \delta }$ with phase-estimation resolution $M .$ . Then for every post-hoc query $\beta$ satisfying $| \beta | \leq R$ , with probability at least $1 - \delta$

$$
\left| \widehat { \lambda } ( \beta ) - \lambda _ { P } ( \beta ) \right| \le 2 \sqrt { \frac { \log ( 4 / \delta ) } { N } } + \Delta _ { P } ( R , M , h )\tag{F204}
$$

where

$$
\Delta _ { P } ( R , M , h ) = 4 R \sqrt { 2 n } { \frac { L ( 1 + \log M ) } { M } } + 2 { \mathfrak a } _ { P } ( h ) + { \frac { 8 n L } { M h } } \ .\tag{F205}
$$

Consequently, if M and h are chosen so that $\Delta _ { P } ( R , M , h ) \le \epsilon / 2$ , then

$$
N \geq 1 6 \epsilon ^ { - 2 } \log \left( \frac { 4 } { \delta } \right)\tag{F206}
$$

channel uses sufice to estimate $\lambda _ { P } ( \beta )$ to additive error ϵ with success probability at least $1 - \delta$

Proof. Let $\widehat { \alpha }$ denote the output of one run of Algorithm 7, and let $\alpha \sim P$ denote the true displacement drawn by the channel in that run. We first bound the bias

$$
\left| \mathbb { E } \left[ e ^ { i \Omega ( \beta , \widehat { \alpha } ) } \right] - \mathbb { E } \left[ e ^ { i \Omega ( \beta , \alpha ) } \right] \right| \ .\tag{F207}
$$

Let $B _ { h }$ be the bad event that either $\alpha \notin \mathcal { V } _ { L , h }$ or the finite-QPE modular error exceeds h in at least one coordinate. By Lemma 14,

$$
\mathrm { P r } [ B _ { h } ] \leq { \mathfrak { a } } _ { P } ( h ) + { \frac { 4 n L } { M h } } \ .\tag{F208}
$$

On the complement of $B _ { h }$ , the modular readout is also an ordinary readout in $\mathbb { R } ^ { 2 n }$ , so $\widehat { \alpha } - \alpha$ is an ordinary coordinate-wise error vector. We next bound the average size of this QPE error. From the proof of Lemma 14, the error in one decoded coordinate has tail at most $2 L / ( M u )$ at error scale u. Therefore

$$
\mathbb { E } \left[ | \widehat { \alpha } _ { j } - \alpha _ { j } | _ { \mathrm { m o d } } \right] \leq \frac { 2 L } { M } + \int _ { 2 L / M } ^ { L / 2 } \frac { 2 L } { M u } d u\tag{F209}
$$

$$
\leq \frac { 4 L ( 1 + \log M ) } { M } \ .\tag{F210}
$$

Here $| \cdot | _ { \mathrm { m o d } }$ denotes the coordinate-wise error after reducing modulo L and choosing the centered representative. On $B _ { h } ^ { c }$ this is the same as the ordinary coordinate error. Using $| e ^ { i u } - e ^ { i v } | \leq | u - v |$ , we have on $B _ { h } ^ { c }$

$$
\left. e ^ { i \Omega ( \beta , \widehat { \alpha } ) } - e ^ { i \Omega ( \beta , \alpha ) } \right. \leq \left. \Omega ( \beta , \widehat { \alpha } - \alpha ) \right. .\tag{F211}
$$

The symplectic form is a signed sum of coordinate products, so

$$
\vert \Omega ( \beta , \widehat { \alpha } - \alpha ) \vert \leq \sum _ { j = 1 } ^ { 2 n } \vert \beta _ { j } \vert \vert \widehat { \alpha } _ { j } - \alpha _ { j } \vert ~ .\tag{F212}
$$

Taking expectation and using $| \beta | _ { 1 } \leq \sqrt { 2 n } | \beta | \leq R \sqrt { 2 n }$ gives

$$
\mathbb { E } \left[ \left| e ^ { i \Omega ( \beta , \widehat { \alpha } ) } - e ^ { i \Omega ( \beta , \alpha ) } \right| \mathbf { 1 } _ { B _ { h } ^ { c } } \right] \leq 4 R \sqrt { 2 n } \frac { L ( 1 + \log M ) } { M } \ .\tag{F213}
$$

On the bad event, the two phases can difer by at most 2. Hence

$$
\left| \mathbb { E } \left[ e ^ { i \Omega ( \beta , \widehat { \alpha } ) } \right] - \lambda _ { P } ( \beta ) \right| \leq 4 R \sqrt { 2 n } \frac { L ( 1 + \log M ) } { M } + 2 \mathfrak { a } _ { P } ( h ) + \frac { 8 n L } { M h } ~ .\tag{F214}
$$

This is exactly $\Delta _ { P } ( R , M , h )$ . It remains to bound sampling error. Each random variable $e ^ { i \Omega ( \beta , \widehat { \alpha } ^ { ( t ) } ) }$ has magnitude 1. Applying Hoefding’s inequality to the real and imaginary parts and taking a union bound gives

$$
\operatorname* { P r } \left[ \left| { \widehat { \lambda } } ( \beta ) - \mathbb { E } { \widehat { \lambda } } ( \beta ) \right| > 2 { \sqrt { \frac { \log ( 4 / \delta ) } { N } } } \right] \leq \delta ~ .\tag{F215}
$$

Combining the sampling bound with the bias bound proves the theorem.

The theorem has a useful interpretation. For any distribution $P ,$ the finite-GKP learner succeeds to the extent that P is contained in the GKP dynamic range. If P has substantial mass outside $\gamma _ { L }$ , then this particular product-GKP sensor sees only the displacement modulo the GKP lattice, and full characteristic-function learning is impossible without additional prior information. If, however, $P$ is concentrated well inside the cell, the learner produces genuine approximate displacement samples, and characteristic-function learning becomes ordinary empirical Fourier estimation. To understand how restrictive this condition actually is, we show that any family which (in displacement space) is contained within a Gaussian envelope of width no broader than $O ( \log n )$ satisfies the constraint. In Fourier space, this corresponds to characteristic functions with width at least $\Omega ( \log n )$ . The condition itself is simply a bound on the tails of P, such that with high probability most of P fits inside the GKP decoding cell. Let

$$
g _ { \sigma } ( \alpha ) = \left( \frac { 2 \sigma ^ { 2 } } { \pi } \right) ^ { n } e ^ { - 2 \sigma ^ { 2 } | \alpha | ^ { 2 } }\tag{F216}
$$

be the centered Gaussian density whose real phase-space coordinates have variance $1 / ( 4 \sigma ^ { 2 } )$ . The following result formalizes the above.

Corollary 16 (Eficient learning for Gaussian-enveloped displacement channels). Suppose P is dominated by a Gaussian envelope in the sense that $P ( \alpha ) \leq C _ { 0 } g _ { \sigma } ( \alpha )$ for all $\alpha \in \mathbb { R } ^ { 2 n }$ , where $C _ { 0 }$ is a constant. Fix $R ^ { 2 } = \kappa n$ and $\epsilon , \delta \in ( 0 , 1 )$ . If

$$
\sigma ^ { 2 } \geq \frac { 4 } { \pi } \log \left( \frac { 6 4 C _ { 0 } n } { \epsilon } \right) ,\tag{F217}
$$

then Algorithm 8, with

$$
{ \cal M } = { \cal O } \biggl ( \frac { ( \sqrt { \kappa } + 1 ) n } { \epsilon } \log \biggl ( \frac { ( \sqrt { \kappa } + 1 ) n } { \epsilon } \biggr ) \biggr ) ,\tag{F218}
$$

estimates $\lambda _ { P } ( \beta )$ for any post-hoc query $| \beta | ^ { 2 } \leq \kappa n$ to additive error ϵ with success probability at least $1 - \delta$ using

$$
N = O \left( \epsilon ^ { - 2 } \log \left( \frac { 1 } { \delta } \right) \right)\tag{F219}
$$

channel queries.

Proof. Take $h = L / 4$ . Then $\mathcal { V } _ { L , h } = [ - L / 4 , L / 4 ] ^ { 2 n }$ . Since $P \leq C _ { 0 } g _ { \sigma }$ , a union bound over the 2n coordinates and the Gaussian tail bound give

$$
\mathfrak { a } _ { P } ( L / 4 ) \le 4 C _ { 0 } n \exp \left( - 2 \sigma ^ { 2 } ( L / 4 ) ^ { 2 } \right)\tag{F220}
$$

$$
= 4 C _ { 0 } n \exp \left( - \frac { \pi \sigma ^ { 2 } } { 4 } \right) \ .\tag{F221}
$$

The specified lower bound on $\sigma ^ { 2 }$ makes $2 { \tt a } _ { P } ( L / 4 ) \leq \epsilon / 8$ . The two finite-QPE resolution terms in Theorem F.15 are

$$
4 \sqrt { 2 \kappa } n \frac { L ( 1 + \log M ) } { M } + \frac { 3 2 n } { M } ~ ,\tag{F222}
$$

where we used $R ^ { 2 } \ = \ \kappa n$ and $h = L / 4$ . The stated choice of M makes these terms at most 3ϵ/8, up to increasing the universal constant. Thus $\Delta _ { P } ( R , M , h ) \le \epsilon / 2$ , and Theorem F.15 gives the claimed channel-query complexity. □

Using this result, we can evaluate how well the entanglement-free GKP sensing protocol performs for learning the three-peak hard family introduced in Ref. [1]. This three-peak displacement density can be written as

$$
P _ { s , \gamma } ^ { \eta _ { 0 } , \sigma } ( \alpha ) = \left( \frac { 2 \sigma ^ { 2 } } { \pi } \right) ^ { n } e ^ { - 2 \sigma ^ { 2 } | \alpha | ^ { 2 } } \left[ 1 + 4 s \eta _ { 0 } \sin \left( 2 \Omega ( \gamma , \alpha ) \right) \right] .\tag{F223}
$$

In Ref. [1], this is precisely the family used to show an exponential lower bound against (separately) Gaussian and all entanglement-free strategies. However, for the latter, their proof required that σ be upper bounded by a constant dependent on the radius parameter κ. While it may have seemed that this constraint is only an artifact of the proof strategy, and that an entanglement-enabled protocol was the only way to achieve a superpolynomial advantage, our next corollary shows that this is not the case: for modestly broad distributions, non-Gaussian sensing enabled by single-qubit control is already suficient to realize a superpolynomial speedup over Gaussian strategies, and to match the query-complexity scaling of the entanglement-enabled Bell sampling strategy.

□

Corollary 17 (Phase-estimation learning of the three-peak family, formal version of Theorem G.5). $F o r$ the three-peak family of Ref. [1], if $R ^ { 2 } = \kappa n$ and

$$
\sigma ^ { 2 } \geq C \log \left( \frac { n } { \epsilon } \right)\tag{F224}
$$

for a suficiently large universal constant $C ,$ then the finite-GKP learner estimates $\lambda _ { s , \gamma } ^ { \eta _ { 0 } , \sigma } ( \beta )$ for any post-hoc query $| \beta | ^ { 2 } \leq$ κn using

$$
N = O \left( \epsilon ^ { - 2 } \log \left( \frac { 1 } { \delta } \right) \right)\tag{F225}
$$

channel queries. Any Gaussian strategy requires exp $\left( \Omega ( n / \log n ) \right)$ samples to do so.

Proof. For $\eta _ { 0 } \leq 1 / 4$ , the three-peak density obeys $P _ { s , \gamma } ^ { \eta _ { 0 } , \sigma } ( \alpha ) \leq 2 g _ { \sigma } ( \alpha )$ uniformly in s and $\gamma .$ Therefore Corollary 16 applies with $C _ { 0 } = 2$ . The lower bound follows from Equation (E149):

$$
N _ { \mathrm { G a u s s } } \geq 0 . 0 1 \epsilon ^ { - 2 } \operatorname* { m i n } \left\{ \left( 1 + \frac { 0 . 9 9 \kappa } { C \log n } \right) ^ { n / 2 } , \left( 1 + \frac { 1 . 9 8 \kappa } { 1 + 2 C \log n } \right) ^ { n } \right\} = \epsilon ^ { - 2 } \exp \left( \Omega _ { \kappa } \left( \frac { n } { \log n } \right) \right)\tag{F226}
$$

The resource scaling of this approach is modest. Each channel query uses $O ( n \log M )$ controlled displacements, and the largest controlled displacement has size $O ( M L )$ . The average photon number in the n-mode probe is

$$
E _ { \mathrm { G K P } } = O ( n M ^ { 2 } ) = O \bigg ( \frac { \kappa n ^ { 3 } } { \epsilon ^ { 2 } } \log ^ { 2 } \bigg ( \frac { ( \sqrt { \kappa } + 1 ) n } { \epsilon } \bigg ) \bigg )\tag{F227}
$$

for fixed $L = { \sqrt { 2 \pi } }$ . It is useful to compare this to the entanglement-assisted Bell sampling protocol. A twomode squeezed Bell measurement produces data of the form $Z = \alpha + G _ { \mathrm { B e l l } }$ , where $G _ { \mathrm { B e l l } }$ is Gaussian noise with variance set by the inverse EPR squeezing. To keep characteristic-function deconvolution stable for all $| \beta | ^ { 2 } \leq \kappa n .$ , one needs Gaussian noise variance $O ( 1 / ( \kappa n ) )$ . This corresponds to total two-mode-squeezed energy ${ \dot { E } } _ { \mathrm { B e l l } } = O ( \kappa n ^ { 2 } )$ . By using only an $O ( n )$ larger amount of energy, entanglement-free GKP based sensing matches the query-complexity scaling of the Bell sampling algorithm while eliminating the need for n ancillary memory modes. Moreover, our construction is neither energy-optimized nor tailored precisely for characteristic function estimation. We use product GKP probe states and vanilla phase-estimation, but it may be possible to fully close the gap in the parameter σ to a constant by designing non-Gaussian lattice probe states with more nontrivial geometries that widen the “good" section of the sensing hypercube. This would rigorously demonstrate that an entanglement-enabled advantage only persists when features of the displacement channel are simultaneously high-frequency and extremely sharp, and that single-qubit control is already suficient to solve this dificult learning task when either requirement is relaxed.

## 6. Learning with depth-k quantum control

Here we complete the exponential depth-based separation by proving that sensing protocols with control depth k can eficiently distinguish the hypotheses instantiated in Section E 6.

Lemma 18. There is a depth-k protocol, as in Definition $^ { 1 \downarrow , }$ that can distinguish the signals induced by the distributions in Equation (E268) with constant bias using $O ( 1 )$ sensing rounds.

Proof. Consider the QSL property

$$
\Psi _ { k } ( P ) = \int g _ { k } ( \alpha ) P ( d \alpha ) .\tag{F228}
$$

Under the two hypotheses from Equation (E268),

$$
\Psi _ { k } ( P _ { + } ^ { ( k ) } ) - \Psi _ { k } ( P _ { - } ^ { ( k ) } ) = 2 \eta \int g _ { k } ( \alpha ) ^ { 2 } d P _ { 0 } ( \alpha )\tag{F229}
$$

$$
= 2 \eta \| g _ { k } \| _ { L ^ { 2 } ( P _ { 0 } ) } ^ { 2 } .\tag{F230}
$$

From the construction of $g _ { k }$ , we know that there exists a constant $c _ { 0 }$ independent of k such that $\| g _ { k } \| _ { L ^ { 2 } ( P _ { 0 } ) } ^ { 2 } \geq c _ { 0 }$ As such,

$$
\Psi _ { k } ( P _ { + } ^ { ( k ) } ) - \Psi _ { k } ( P _ { - } ^ { ( k ) } ) \geq 2 c _ { 0 } \eta .\tag{F231}
$$

By Lemma 18, there is a depth-k QCDS circuit producing a random variable $Y \in \{ - 1 , 1 \}$ with $\mathbb { E } [ Y \mid \alpha ] = h _ { k } ( \alpha )$ Let $\begin{array} { r } { m _ { k } = \int h _ { k } ( \alpha ) d P _ { 0 } ( \alpha ) } \end{array}$ . Under $P _ { \pm } ^ { ( k ) }$ )

$$
\mathbb { E } _ { P _ { \pm } ^ { ( k ) } } [ Y ] = \int h _ { k } ( \alpha ) \left( 1 \pm \eta g _ { k } ( \alpha ) \right) d P _ { 0 } ( \alpha )\tag{F232}
$$

$$
= m _ { k } \pm \eta \int h _ { k } ( \alpha ) g _ { k } ( \alpha ) d P _ { 0 } ( \alpha ) .\tag{F233}
$$

Since $g _ { k } = h _ { k } - m _ { k }$

$$
\int h _ { k } ( \alpha ) g _ { k } ( \alpha ) d P _ { 0 } ( \alpha ) = \int ( g _ { k } ( \alpha ) + m _ { k } ) g _ { k } ( \alpha ) d P _ { 0 } ( \alpha )\tag{F234}
$$

$$
= \int g _ { k } ( \alpha ) ^ { 2 } d P _ { 0 } ( \alpha ) + { m _ { k } } \int g _ { k } ( \alpha ) d P _ { 0 } ( \alpha )\tag{F235}
$$

$$
= \| g _ { k } \| _ { L ^ { 2 } ( P _ { 0 } ) } ^ { 2 } .\tag{F236}
$$

Therefore

$$
\mathbb { E } _ { P _ { \pm } ^ { ( k ) } } [ Y ] = m _ { k } \pm \eta \| g _ { k } \| _ { L ^ { 2 } ( P _ { 0 } ) } ^ { 2 } .\tag{F237}
$$

The two means are separated by at least $2 c _ { 0 } \eta$ . Run the depth-k circuit independently for N queries and let

$$
\overline { { Y } } = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } Y _ { j } .\tag{F238}
$$

Then outputs (+) if ${ \overline { { Y } } } \geq m _ { k }$ and output (−) otherwise. Since each $Y _ { j } \in [ - 1 , 1 ]$ , Hoefding’s inequality gives

$$
\mathrm { P r } [ \mathrm { e r r o r } ] \leq \exp \left( - \frac { N \eta ^ { 2 } \| g _ { k } \| _ { L ^ { 2 } ( P _ { 0 } ) } ^ { 4 } } { 2 } \right) .\tag{F239}
$$

Using $\| g _ { k } \| _ { L ^ { 2 } ( P _ { 0 } ) } ^ { 2 } \geq c _ { 0 }$ , it sufices to take $N = O ( \eta ^ { - 2 } \log ( 1 / \delta ) )$ to achieve failure probability at most δ. Using constant η, δ completes the proof. □

Finally, by combining Lemma 18 with Theorem E.13, we obtain the full exponential control-depth separation.

Corollary 19. There exist QSL properties with kernels that are polynomials of degree k such that a protocol of control depth k can estimate the property to constant precision ϵ using O(1) queries to the signal, whereas $f o r \gamma < \log _ { 3 } 2 $ , any protocol using control depth at most γk per measurement requires exp(Ω(k)) queries to do so.

# Appendix G: Self-contained list of results

Our appendices contain several contributions not detailed in the main text. For the reader’s convenience, we provide a self-contained reference list of the main results in our work.

## Main separations

We begin with informal statements of all main hierarchy separations in ascending order.

Theorem G.1 (Theorem 2, Exponential advantage in Fourier analysis with nonclassical Gaussian probes). A conventional quantum protocol using squeezed homodyne measurements of energy O(k) estimates the $\bar { k } ^ { \mathrm { t h } }$ angular Fourier coeficient of a signal using O(1) queries. Any protocol exposing coherent-state probes to the signal, regardless of available energy in ancillas or measurements, requires exp(Ω(k)) queries to do so. A non-Gaussian quantum protocol with O(k) energy also achieves an exponential advantage over all coherent-probe protocols.

Theorem G.2 (Theorem 1, Exponential advantage in Fourier analysis with a single qubit). Using a sensor probe with energy O(k), one ancilla qubit, and one control operation, any $k ^ { \mathrm { t h } }$ directional Fourier coeficient of a signal can be estimated with O(k) signal queries. Any adaptive Gaussian protocol with the same energy scaling requires exp(Ω(k)) queries.

Theorem G.3 (Theorem 3, Exponential advantage in learning temporal correlations with one memory qubit). Let $t _ { 1 } < t _ { 2 } < \cdots < t _ { m }$ be a list of times separated by at most ∆. A sensor that decoheres on a timescale much shorter than ∆, when coupled to a single qubit that remains coherent up to time $t _ { m }$ , can estimate the corresponding m-point temporal correlator using O(1) queries to the time-dependent signal. By contrast, any protocol with the same energy and no quantum degree of freedom coherent for times much longer than ∆ requires exp(Ω(m)) queries, even if it has access to arbitrarily many short-lived ancilla qubits or modes.

Theorem G.4 (Theorem 19, Exponential advantage of increasing circuit depth for learning polynomial functionals of signals). There exist QSL properties with kernels that are polynomials of degree k such that a protocol of control depth k can estimate the property to constant precision ϵ using O(1) queries to the signal, whereas for $\gamma < \log _ { 3 } 2 _ { \mathrm { { ; } } }$ , any protocol using control depth at most γk per measurement requires exp(Ω(k)) queries to do so.

These separations establish that each level of the sensing hierarchy in Figure 1 is exponentially separated from the last, and that adding a single quantum resource to an otherwise identical sensor can enable exponential resource savings. We also prove other separations that operate in the multimode sensing setting or provide speedups against infinite-energy Gaussian sensing.

Theorem G.5 (Theorem 17, Superpolynomial entanglement-free advantage in learning a displacement channel). Let $\kappa > 0 , n \in \mathbb { Z } ^ { + }$ . For any displacement channel with frequency-domain peaks of width Ω(log n), an entanglement-free protocol using a single ancilla qubit, poly(n) energy, and no ancilla modes can learn its characteristic function within a phase-space radius κn using a number of measurements independent of n (as eficiently as the best-known entanglement-enabled strategy). Meanwhile, any Gaussian protocol with unconstrained energy requires exp(Ω(n/ log n)) measurements to do so.

The Gaussian lower bound for this task is due to Ref. [1]; surprisingly, we showed that a superpolynomial advantage can be achieved without entangled sensors, and by using only a single ancilla qubit. Our protocol used one qubit to implement polynomial-energy approximate GKP state quantum phase estimation. Therefore, the entanglement-enabled advantage of Ref. [1], achieved by performing two-mode squeezed vacuum Bell measurements, is only necessary in edge cases when the displacement channel has extremely fine peaks in unknown locations.

Theorem G.6 (Theorem 13, Quadratic single-mode advantage over infinite-energy Gaussian sensing). Any Gaussian protocol of arbitrary energy which can estimate the variance of a (Gaussian) thermal state to accuracy 1/k requires Ω(k<sup>2</sup>) samples. There exist non-Gaussian measurements using O(1) energy, including parity and photon-counting measurement, that can do so using O(k) samples.

This theorem establishes the existence of single-mode Gaussian states which require non-Gaussian measurements to be distinguished optimally. Moreover, it provides a rigorous backing for why non-Gaussian photon-counting measurements provide a large practical speedup for stochastic sensing tasks which involve estimating smal perturbations to white background power spectra. The measurements which enable this speedup are easily implemented in superconducting-circuit platforms with a transmon qubit coupled to a microwave cavity, as we use in this work.

## QΨ toolkit

We also provide a high-level, intuitive recap of the QΨ toolkit. Here, for clarity, we present QΨ restricted to single-mode quantum sensing with bosonic oscillators; the most general formulation, which encapsulates much more general quantum-enhanced experiments, is given in Appendix D.

QΨ inputs. We let $P _ { 0 }$ denote a reference signal, whose representation on phase space is compact or approximately compact (in the sense of Definition 3 and Lemma 4). An example of such a reference is a Gaussian signal. With the reference fixed, QΨ requires three inputs: a family of weaker "conventional" experiments, ${ \mathcal { M } } _ { \mathrm { c o n v } } ,$ a family of quantum-enhanced experiments, $\mathcal { M } _ { \mathrm { Q I P } }$ , and a physical observable specified by its phase-space representation as a function h on $L ^ { 2 } ( P _ { 0 } )$ . Mathematically, a family of experiments is represented by a collection of response functions which record classical probability distributions over potential outcomes from each allowed experiment. The function h can represent an arbitrary physical property such as a Fourier amplitude, phase, moment, or temporal correlation.

AFI. The accessible feature information (AFI), denoted by $\mathsf { A } _ { M } ( h ; P _ { 0 } )$ for any experiment M in ${ \mathcal { M } } _ { \mathrm { c o n v } }$ or M<sub>QIP</sub> is a real-valued function of these inputs (Definition 12). For the entire family of experiments $\mathcal { M } = \mathcal { M } _ { \mathrm { c o n v } }$ or ${ \mathcal { M } } _ { \mathrm { Q I P } }$ , the AFI $\mathsf { A } _ { \mathcal { M } } ( h ; P _ { 0 } )$ is simply $\mathrm { s u p } _ { M \in \mathcal { M } } \mathsf { A } _ { M } ( h ; P _ { 0 } )$ .

Semiclassical quantum measurement lemma and tightness certificate. The primary lower-bound tool of $\mathrm { Q } \Psi$ is the semiclassical quantum measurement lemma:

Theorem G.7 (Informal statement of Theorem 24, Semiclassical Quantum Measurement Lemma). Consider an arbitrary experimental protocol for learning property h to absolute precision ϵ, in which each experiment is in the class $\mathcal { M }$ and arbitrary classical adaptivity is allowed between experiments. Any such protocol then requires $N \ge \Omega ( \epsilon ^ { - 2 } \mathsf { A } _ { \mathcal { M } } ( h ; P _ { 0 } ) ^ { - 1 } )$ queries to the unknown signal (or more generally, an uncharacterized physical system).

This theorem is obtained by combining information-theoretic lower bound techniques with harmonic analysis of phase-space response functions, and the learning lower bound is established by lifting discrimination of the signals $( 1 \pm \epsilon h ) P _ { 0 }$ to the harder task of learning h to absolute precision ϵ. The next element of the QΨ toolkit establishes that the semiclassical measurement lemma is tight:

Theorem G.8 (Informal statement of Theorem D.27, QΨ saturability guarantee). There exists an adaptive experimental protocol, with each round in the class of experiments ${ \mathcal { M } } ,$ which discriminates the signals $P _ { \pm } =$ $( 1 \pm \epsilon h ) P _ { 0 }$ with constant bias using $N = O ( \epsilon ^ { - 2 } \mathsf { A } _ { \mathcal { M } } ( h ; P _ { 0 } ) ^ { - 1 } )$ queries.

This upper bound both certifies the existence of a lower bound-saturating protocol and also produces the protocol directly, as detailed in the proof of Theorem D.27. Together, they produce the following implication:

Corollary 9 (Certifiably optimal quantum advantage, informal). Let $h ^ { ( k ) }$ be a family of observables parameterized by values of $k > 0$ $I f A _ { \mathcal { M } _ { \mathrm { c o n v } } } ( h ^ { ( k ) } ; P _ { 0 } ) \leq c ( k )$ and $A _ { \mathcal { M } _ { \mathrm { Q I P } } } ( h ^ { ( k ) } ; P _ { 0 } ) \ge q ( k )$ , then the quantum-enhanced family of experiments achieves an $\Omega ( q ( k ) / c ( k ) )$ quantum advantage for learning the properties $h ^ { ( k ) }$ , and this advantage is the largest possible.

In the present paper, we have shown that several physically natural properties such as frequency-k Fourier coeficients, order-k temporal correlations, and degree-k polynomial transformations, all force $c ( k ) \leq \exp ( - \Omega ( k ) )$ for the respective lower rungs of the hierarchy, while adding a single resource like squeezing, single-qubit control, memory, or circuit depth, enables $q ( k ) \geq \Omega ( 1 )$ . QΨ then immediately outputs a certificate of quantum advantage.

Global learning. QΨ makes new quantum separations transparent by recasting the design of optimal quantum experiments as a classical problem of statistical overlap. In particular, the optimal physical experiment is the one whose phase-space response functions have the largest overlap with the phase-space representation of the property of interest. In the frequency domain, this leads to a direct optimization: compute the Fourier representations of the experimental responses and the target property, and choose the measurement with the largest response on the Fourier characters supporting the property. Importantly, this optimization is entirely classical and commutative, suggesting the automated design principles discussed in Section H.

This phase-space overlap formulation also allows $\mathrm { Q } \Psi$ to produce optimal global-learning algorithms beyond the hypothesis-testing setting. We define the Fourier gain of an experimental family as follows. We choose a Fourier dictionary $\chi _ { \omega }$ that forms a complete basis for the physically relevant signals. Given a protocol with response function $f _ { j } ( x )$ , where x denotes the unobserved signal realization, we write its Fourier expansion as $\begin{array} { r } { f _ { j } ( x ) = \sum _ { \omega } \widehat { f } _ { j } ( \omega ) \chi _ { \omega } ( x ) } \end{array}$ . The Fourier gain of the family M at frequency ω is $\begin{array} { r } { G _ { \mathcal { M } } ( \omega ) = \operatorname* { s u p } _ { j \in \mathcal { M } } | \widehat { f } _ { j } ( \omega ) | ^ { 2 } / V _ { j } } \end{array}$ Here $V _ { j }$ is a protocol-dependent normalization. Thus, $G _ { \mathcal { M } } ( \omega )$ quantifies the largest phase-space response that the experimental class can achieve at frequency ω.

Theorem G.10 (Informal statement of Theorem D.30, QΨ global learning guarantee). Let h be a property of interest, and define $\widehat { h }$ by the Fourier transform

$$
h ( x ) = \sum _ { \omega \in S } \widehat { h } ( x ) \chi _ { \omega } ( x ) \ ,\tag{G1}
$$

where S is the Fourier-domain support of h. Then there is a learning algorithm in the class M which learns h to absolute accuracy ϵ using

$$
N \propto \frac { 1 } { \epsilon ^ { 2 } } \sum _ { \omega \in S } \frac { | \widehat { h } ( \omega ) | ^ { 2 } } { G _ { \mathcal { M } } ( \omega ) }\tag{G2}
$$

experiments.

A more detailed sample complexity statement is given in Theorem D.30. In short, the resource complexity is directly controlled by the ratio of the observable’s Fourier support at each frequency and the best response synthesizable by any measurement in the class at that frequency. The algorithm which achieves this scaling allocates experiments based on their relative overlap with the desired observable and is given in Section D.

Summary. QΨ is therefore a unifying framework of quantum learning, metrology, and estimation theory useful for understanding the power of quantum information processing in experimental science. It reduces a quantum learning problem into phase-space overlap of classical statistical response functions, produces provably optima lower bounds against conventional experiments and upper bounds achieved by quantum-enhanced ones, and certifies a rigorous quantum advantage while designing an asymptotically optimal quantum experiment. The central quantities are the AFI and Fourier gain, which characterize the achievable measurement complexity and guide the design of optimal quantum-enhanced experiments.

# Appendix H: Open directions

## 1. Applications of Quantum Feature Sensing

In this work, we have established that Quantum Feature Sensing, the family of quantum processing-enhanced bosonic sensing protocols that emerge from QΨ, can enable quantum advantages for basic sensing primitives. Here, we have covered estimation of Fourier coeficients, moments, temporal correlations, and polynomial functionals. Our point of connection between these physical tasks and the QΨ framework was Quantum Signal Learning, which translates arbitrary physical observables of classical signals incident on a quantum oscillator into mathematical descriptions that can be utilized by QΨ. Importantly, this QSL formalism circumscribes many of the sensing tasks studied in e.g. fundamental particle searches, device calibration, classical noise spectroscopy, and receivers, making these broad disciplines amenable to QFS speedups that have thus far been dificult to systematically discover. Here, we provide context for these applications and present concrete open questions of when and how the techniques introduced here can enable speedups for a wide array of classical sensing applications.

## a. Fundamental particle searches

Cavity haloscopes search for wave-like dark matter candidates, such as axions and dark photons, by monitoring a high-quality microwave cavity for a feeble drive at an unknown frequency set by the particle mass. The galactic velocity dispersion endows this drive with a narrow stochastic lineshape of quality factor ∼ 10<sup>6</sup>, so the hypothetical signal is, in our language, a displacement channel whose density P is the thermal background perturbed by a small excess occupation $n _ { a } \ll 1$ concentrated near one frequency in a multi-gigahertz band. Detection is a hypothesis test between background and background-plus-excess, and the figure of merit is the scan rate: the bandwidth that can be excluded or discovered per unit integration time at fixed statistical significance.

The historical development of dark-matter sensing traces the first rungs of our sensing hierarchy. Initially, squeezed-vacuum receivers were shown to roughly double the scan rate [59, 60]. Then, replacing quadrature readout with qubit-based photon counting was shown to improve search speed by up to three orders of magnitude. In practice, these measurements were implementable by a transmon dispersively coupled to a microwave cavity, which is precisely the architecture we have used in this work [80, 81]. Moreover, proposals of entanglementassisted receivers are prominent [61, 62]. This progression through quantum-enhanced architectures has been assembled empirically, once the constraints of each level of the hierarchy became clear. In light of the present work, QΨ provides a unifying understanding of why each additional quantum resource enabled speedups for these tasks, as well as a systematic strategy to discover future quantum enhancements.

Open Problem (Quantum information processing for fundamental particle search). Can we formulate the detection and characterization of wave-like dark matter, and weak stochastic signals more broadly, as Quantum Signal Learning problems, and use QΨ to chart which quantum informationprocessing resources provably accelerate them? In particular: do the higher rungs of the hierarchy in Figure 1(b) confer advantages that are qualitatively beyond single-photon detection, and how large can these advantages be?

Our results supply both rigor for the enhancements this field has already found and reason to believe it has explored only the first rung of what quantum control ofers. The vacuum-versus-thermal separation of Sections E 4 and F 4 proves that photon-counting-style readout achieves a sample-complexity scaling that no Gaussian receiver of any energy can match — a precise version of the folklore underlying Refs. [80, 81], and an explanation for why the leap from squeezing to counting was a change in kind rather than in degree. Moreover, section F 2 a shows that the single echoed conditional displacement powering our exponential advantage is interchangeable with photon-counting readout as a non-Gaussian resource. In Figure 4(a) and Section B 5 we give concrete evidence that a superconducting circuit platform with a single qubit may produce asymptotic improvements for important tasks in the characterization of axionic dark matter.

Single-photon detection is therefore not an endpoint but the first unit of quantum control, and each additional resource in our hierarchy, including even a single additional non-Gaussian control, corresponds to a capability that no photon counter possesses. A single memory qubit can coherently accumulate the dark-matter field’s phase across many of its coherence windows (Theorem 3), suggesting a route to estimating the signal lineshape (equivalently, the local dark-matter velocity distribution [70, 83]) as temporal Fourier data rather than by prolonged power integration; deeper coherent control synthesizes matched filters directly as polynomial functionals of the signal (Sections E 6 and F 6).

## b. Stochastic and spatiotemporal noise learning

Characterizing environmental noise is essential for calibrating quantum platforms. A common technique is dynamical-decoupling noise spectroscopy, in which one applies pulse sequences whose filter functions concentrate the sensor’s response near a chosen frequency and thereby reconstructs the noise spectral density $S ( \omega )$ from measured decay rates [22, 84, 85]. In our language, the fluctuating environment is a stochastic process $\{ Z _ { t } \}$ , a pulse sequence followed by destructive readout is a memoryless response kernel, and $S ( \omega )$ is two-point temporal Fourier data. Every such protocol inherits a structural limitation: a sensor with coherence time $T _ { 2 }$ produces response kernels of spectral linewidth $\sim 1 / T _ { 2 } .$ , and cannot resolve finer structure no matter how many measurements are taken. Experiments have breached this limit by storing phase in a long-lived nuclear-spin memory [86], but a general account of which resolution–measurement tradeofs are possible with bounded quantum resources is not present.

Open Problem (Quantum information processing for noise spectroscopy). Can we use $\mathrm { Q } \Psi$ to chart which quantum information-processing resources provably extend the reach of noise spectroscopy? What resolution and measurement-complexity tradeofs become achievable with quantum memory, deeper coherent control, or entanglement across sensors, and which of these are provably impossible for existing filter-function protocols?

Our results suggest a number of applications to measurement of spectral densities, higher-resolution spectral features, and their spatiotemporal counterparts across sensor arrays. Theorem 3 is a proof that the mechanism observed in memory-assisted spectrometers [86] constitutes an exponential quantum advantage; Section E 3 shows the linewidth limit of memoryless sensing is an information-theoretic barrier rather than a practical inconvenience, while a single long-lived qubit crosses it. Sub-linewidth estimation of a feature of width $\delta \omega \ll 1 / T _ { 2 }$ spans $m \sim ( \delta \omega T _ { 2 } ) ^ { - 1 }$ sensor coherence windows, and certifying an exponential-versus-polynomial separation there, under realistic constraints on pulse bandwidth and total wall-clock time, would elucidate the role of memory as a resource for noise spectroscopy. In the spatiotemporal setting, arrays of sensors probing a correlated noise field target cross-spectra and spatial Fourier modes, and in the spirit of Section F 5 one may ask which correlation structures single-qubit control resolves without any entanglement, and where entanglement across sensors is provably necessary. In the bosonic noise spectroscopy setting, QSL already provides the relevant language, whereas the spin-sensor paradigm requires a qubit-native formalism. The multiqubit correlationspectroscopy protocols of Refs. [58, 87, 88] provide conventional baselines for this direction.

## c. Estimating polyspectra

A Gaussian stochastic process is fully characterized by its spectral density which is a function of a single frequency, but non-Gaussian noise is not: its complete frequency-domain description requires polyspectra, the Fourier transforms of higher-order cumulants, beginning with the bispectrum $S _ { 2 } ( \omega _ { 1 } , \omega _ { 2 } )$ . Polyspectra are standard objects in classical signal processing, where they detect phase coupling and nonlinearity invisible to the power spectrum, and they have entered the quantum domain through dynamical-decoupling protocols for reconstructing the dephasing bispectrum [56], realized experimentally on a superconducting qubit [57]. A classica estimator of an order-m spectrum averages products of m noisy samples, so its variance compounds multiplicatively with the order, precisely the mechanism depicted in Figure 3(a), often rendering orders beyond the bispectrum inaccessible. This higher-order data has broad utility: polyspectra discriminate targets from non-Gaussian clutter in radar and sonar [89], constrain primordial non-Gaussianity in cosmology [90], and identify the microscopic origin of a qubit’s noise environment.

Open Problem (Quantum information processing for higher-order spectra). Formulate the estimation of polyspectra of non-Gaussian noise as Quantum Signal Learning, and determine through QΨ whether quantum memory and coherent control enable provably eficient access to higher-order polyspectra relevant to the above applications. How do the achievable advantages scale jointly in the frequency and the order of the targeted coeficient?

Theorem 3 and Algorithm 5 show that full m-point temporal Fourier coeficients, the moments of the process, accumulate on a single qubit coherence with a trajectory count independent of m, while any memoryless protocol provably pays the exponential compounding cost. Polyspectra are formed from the full order-m coeficient minus all products of lower-order coeficients. As such, one must determine whether varying the control parameters of each gate in a memory-enabled algorithm can isolate the polyspectrum coeficient within the qubit coherence itself, and whether a memoryless lower bound holds against a strengthened adversary granted all correlations of order < m for free. More broadly, the remark concluding Section E 3 suggests that a valuable direction is to investigate separations scaling both with the polyspectrum frequency and temporal order. A numerical starting point for this direction is to benchmark the frequency-dependent and order-dependent advantages of memory-enabled QFS against the canonical comb-based reconstruction of Ref. [56].

## 2. QΨ and future quantum advantages in experimental science

## a. Multiparameter QΨ and optimization

Multiparameter estimation is a rich subject in canonical quantum metrology, and central to many realistic sensing tasks. The usual approach promotes the quantum Fisher information to a matrix and the quantum Cramer-Rao bound to a matrix-valued inequality. Upon doing so, the definition of an optimal protocol is dependent on a choice of cost function which encodes which parameters are more important to the estimation task at hand. Due to incompatibility of diferent measurement bases for estimating noncommuting observables, the matrix-valued Cramer-Rao bound is generally unattainable, so the central goal moves towards designing algorithms that output measurement configurations that approach the lower bound.

Importantly, the multiparameter extension of the QFI/QCRB picture inherits many of the challenges associated with its single-parameter counterpart. In particular, the model still assumes unbiased estimation of all parameters in a diferentiable statistical model; moreover, entries of the QFI matrix are often dificult to estimate due to the noncommutative structure of the SLD and observables at hand.

Given this context, a natural and high-value direction is to extend the QΨ formalism to the multiparameter setting. The basic extensions are immediate, and similar to the QFI formalism. Let $P _ { 0 }$ be a reference law on the signal space and let $h = ( h _ { 1 } , \ldots , h _ { d } )$ be a vector of features, with $h _ { a } \in L ^ { 2 } ( P _ { 0 } )$ . For an experiment M with response kernel $K _ { M } ( d y | x )$ , define

$$
Q _ { M , 0 } ( d y ) = \int K _ { M } ( d y | x ) d P _ { 0 } ( x ) , \qquad A _ { M , a } ( d y ) = \int h _ { a } ( x ) K _ { M } ( d y | x ) d P _ { 0 } ( x ) .\tag{H1}
$$

The corresponding score vector is

$$
R _ { M } ( y ) = \frac { d A _ { M } } { d Q _ { M , 0 } } ( y ) = \mathbb { E } [ h ( X ) \mid Y = y ] ,\tag{H2}
$$

where $X \sim P _ { 0 }$ and Y is the measurement outcome. The natural $\mathrm { Q } \Psi$ signal family is

$$
\begin{array} { r } { P _ { \theta } ( d x ) = \left( 1 + \sum _ { a = 1 } ^ { d } \theta _ { a } h _ { a } ( x ) \right) P _ { 0 } ( d x ) , \qquad \theta \in \mathbb { R } ^ { d } . } \end{array}\tag{H3}
$$

For a measurement M with scores $R _ { M , h _ { a } } ( y ) = \mathbb { E } [ h _ { a } ( X ) \vert Y = y ]$ as in Definition 12, we can define the matrixvalued AFI

$$
\left[ \mathsf { A } _ { M } ( { \bf h } ; P _ { 0 } ) \right] _ { a b } = \int R _ { M , h _ { a } } ( y ) R _ { M , h _ { b } } ( y ) d Q _ { M , 0 } ( y ) .\tag{H4}
$$

Upon fixing a particular linear combination of the features via a vector $\theta ,$ so that we obtain the local statistica model $P _ { \theta } ( d x ) = ( 1 + \theta ^ { T } h ( x ) ) P _ { 0 } ( d x )$ , we find that $\mathsf { A } _ { M } ( \mathbf { h } ; P _ { 0 } )$ is exactly the classical Fisher information matrix

of the outcome distribution induced by M at $\theta = 0$ . Thus the matrix AFI is the feature-learning analogue of classical Fisher information after the restrictions of the sensing architecture have been imposed.

Then, for a matrix of weights $W \succeq 0$ encoding the cost function, the optimal weighted error of N adaptive queries from M should satisfy

$$
\operatorname* { i n f } _ { \widehat { \theta } } \ \mathbb { E } \big [ ( \widehat { \theta } - \theta ) ^ { T } W ( \widehat { \theta } - \theta ) \big ] \ = \ \Theta \bigg ( \frac { 1 } { N } \mathcal { E } _ { W } ( \mathcal { M } ) \bigg ) , \qquad \mathcal { E } _ { W } ( \mathcal { M } ) = \operatorname* { i n f } _ { F \in \mathcal { K } ( \mathcal { M } ) } \mathrm { T r } \big [ W F ^ { - 1 } \big ] ,\tag{H5}
$$

with achievability by allocating shots among finitely many measurements and applying the matched-score estimator of Theorem D.27 componentwise.

Open Problem (Multiparameter QΨ). Prove the matrix-valued semiclassical measurement lemma establishing Equation (H5), including full classical adaptivity, and protocols with persistent quantum memory? When is the information-region bound $\mathcal { E } _ { W } \mathrm { \ t i g h t }$ , and which multiparameter sensing problems exhibit large gaps between the rungs of the sensing hierarchy under simultaneous estimation?

The proof of the multiparameter bound does not directly emerge from the transcript factorization underlying Lemma 24, because Le Cam’s two-point method does not address incompatibility between diferent features. Natural approaches include Assouad’s hypercube method, or first developing a Bayesian analog of AFI theory along the lines of van Trees’ inequality and applying it to the proof.

This direction would further clarify the advantages of an AFI approach to metrology. We expect that many problems in multiparameter quantum metrology may become more numerically tractable due to the phase-space reduction of noncommutative quantum measurements to scalar statistical response functions, enabling eficient classical search. This also suggests a natural direction for convex optimization theory and numerical search.

## b. Automating the discovery of quantum advantages

By a Schur complement, the quantity ${ \mathcal { E } } _ { W }$ of Equation (H5) is the value of the convex program

$$
{ \mathcal { E } } _ { W } ( { \mathcal { M } } ) = \operatorname* { m i n } _ { V , F } \ { \mathrm { T r } } [ W V ] \qquad { \mathrm { s . t . } } \qquad { \binom { V } { I } } \ F \ F \ G \ , \qquad F \in K ( { \mathcal { M } } ) .\tag{H6}
$$

Here, K(M) := conv $\{ \mathsf { A } _ { M } ( h ; P _ { 0 } ) : M \in \mathcal { M } \} \subseteq \mathbb { S } _ { + } ^ { d }$ is the convex hull of AFI matrices obtained by randomizing over experiments. Moreover, the phase-space structure of QΨ suggests that eficient descriptions of the information region $\kappa ( \mathcal { M } )$ exist. An immediate finite-dimensional optimization can be performed by fixing a candidate family of experimental protocols $M _ { 1 } , \dots , M _ { r } \in { \mathcal { M } }$ , with AFI matrices $A _ { i } = A _ { M _ { i } } ( h ; P _ { 0 } )$ . Given this family, the problem reduces to optimizing over convex combinations $\begin{array} { r } { F = \sum _ { i } p _ { i } A _ { i } , } \end{array}$ where $p _ { i } \geq 0$ and $\textstyle \sum _ { i } p _ { i } = 1$ . However, a more interesting question is how to parameterize the entire architecture class $\kappa ( \mathcal { M } )$ , and whether such a description is amenable to numerical optimization. We expect that it is in many cases, because positivity and energy constraints often admit convex descriptions, while the achievable information region $K ( \mathcal { M } )$ is convex by construction under classical randomization. Constraints such as Gaussianity or bounded circuit depth are generally nonconvex at the level of the underlying protocols, and may instead require restricted parameterizations, semidefinite relaxations, or heuristic optimization.

These observations suggest a broad and fruitful intersection between quantum experiments, convex optimization, and deep learning.

Open Problem (QΨ-guided automated discovery of quantum advantage). Can we build a broad, closed-form formulation of the multiparameter QΨ design problem (H6), and thereby enable numerical design of optimal quantum experiments via:

1. Convex optimization, when experimental constraints are convex, or

2. QΨ-guided machine learning, more generally?

Can this lead to automated discovery of quantum advantages, where quantum experimental problems and realistic constraints are inserted into classical machine learning, and optimal experimental designs emerge alongside rigorous certificates of quantum advantage?

[1] C. Oh, S. Chen, Y. Wong, S. Zhou, H.-Y. Huang, J. A. Nielsen, Z.-H. Liu, J. S. Neergaard-Nielsen, U. L. Andersen, L. Jiang, and others, Entanglement-Enabled Advantage for Learning a Bosonic Random Displacement Channel, Physical Review Letters 133 (2024).

[2] K. Vogel and H. Risken, Determination of quasiprobability distributions in terms of probability distributions for the rotated quadrature phase, Phys. Rev. A 40, 2847(R) (1989).

[3] D. T. Smithey, M. Beck, M. G. Raymer, and A. Faridani, Measurement of the Wigner distribution and the density matrix of a light mode using optical homodyne tomography: Application to squeezed states and the vacuum, Phys. Rev. Lett. 70, 1244 (1993).

[4] Z. Hradil, Quantum state estimation, Physical Review A 55, R1561 (1997), arXiv:quant-ph/9609012.

[5] I. L. Chuang and M. A. Nielsen, Prescription for experimental determination of the dynamics of a quantum black box, Journal of Modern Optics 44, 2455–2467 (1997).

[6] A. Luis and L. L. Sánchez-Soto, Complete Characterization of Arbitrary Quantum Measurement Processes, Physical Review Letters 83, 3573 (1999).

[7] M. Cramer, M. B. Plenio, S. T. Flammia, D. Gross, S. D. Bartlett, R. Somma, O. Landon-Cardinal, Y.-K. Liu, and D. Poulin, Eficient quantum state tomography, Nature Communications 1, 149 (2010), arXiv:1101.4366 [quant-ph].

[8] D. Gross, Y.-K. Liu, S. T. Flammia, S. Becker, and J. Eisert, Quantum State Tomography via Compressed Sensing, Physical Review Letters 105, 150401 (2010).

[9] S. T. Merkel, J. M. Gambetta, J. A. Smolin, S. Poletto, A. D. Córcoles, B. R. Johnson, C. A. Ryan, and M. Stefen, Self-consistent quantum process tomography, Physical Review A 87, 062119 (2013).

[10] S. Aaronson, Shadow Tomography of Quantum States, in Proceedings of the 50th Annual ACM SIGACT Symposium on Theory of Computing (STOC 2018) (2018).

[11] H.-Y. Huang, R. Kueng, and J. Preskill, Predicting many properties of a quantum system from very few measurements, Nature Physics 16, 1050–1057 (2020).

[12] H.-Y. Huang, M. Broughton, J. Cotler, S. Chen, J. Li, M. Mohseni, H. Neven, R. Babbush, R. Kueng, J. Preskill, and others, Quantum advantage in learning from experiments, Science 376, 1182–1186 (2022).

[13] A. Montanaro, Learning stabilizer states by Bell sampling (2017), arXiv:1707.04012 [quant-ph].

[14] D. Aharonov, J. Cotler, and X.-L. Qi, Quantum algorithmic measurement, Nature Communications 13 (2022).

[15] S. Chen, J. Cotler, H.-Y. Huang, and J. Li, Exponential separations between learning with and without quantum memory, in 2021 IEEE 62nd Annual Symposium on Foundations of Computer Science (FOCS) (IEEE, 2022) pp. 574–585.

[16] S. Chen, J. Cotler, H.-Y. Huang, and J. Li, The Complexity of NISQ, Nature Communications 14, 6001 (2023).

[17] S. Chen, C. Oh, S. Zhou, H.-Y. Huang, and L. Jiang, Tight Bounds on Pauli Channel Learning without Entanglement, Physical Review Letters 132, 10.1103/physrevlett.132.180805 (2024).

[18] J. Cotler, W. Gong, and I. Kannan, Noisy quantum learning theory, Nature Communications (2026).

[19] I. Kannan, H. Putterman, and J. Cotler, Exponential speedups in fault-tolerant processing of quantum experiments (2026), arXiv:2605.02057.

[20] S. F. Huelga, C. Macchiavello, T. Pellizzari, A. K. Ekert, M. B. Plenio, and J. I. Cirac, Improvement of Frequency Standards with Quantum Entanglement, Physical Review Letters 79, 3865–3868 (1997).

[21] V. Giovannetti, S. Lloyd, and L. Maccone, Advances in quantum metrology, Nature Photonics 5, 222–229 (2011).

[22] C. Degen, F. Reinhard, and P. Cappellaro, Quantum sensing, Reviews of Modern Physics 89 (2017).

[23] Y.-D. Wu, Y. Zhu, G. Chiribella, and N. Liu, Eficient learning of continuous-variable quantum states, Physical Review Research 6, 10.1103/physrevresearch.6.033280 (2024).

[24] L. Bittel, F. A. Mele, J. Eisert, and A. A. Mele, Energy-independent tomography of Gaussian states (2025), arXiv:2508.14979 [quant-ph].

[25] F. A. Mele, A. A. Mele, L. Bittel, J. Eisert, V. Giovannetti, L. Lami, L. Leone, and S. F. E. Oliviero, Learning quantum states of continuous-variable systems, Nature Physics 21, 2002–2008 (2025).

[26] A. A. Mele and Y. Herasymenko, Eficient Learning of Quantum States Prepared With Few Fermionic Non-Gaussian Gates, PRX Quantum 6, 10.1103/prxquantum.6.010319 (2025).

[27] Z.-H. Liu, R. Brunel, E. E. B. Østergaard, O. Cordero, S. Chen, Y. Wong, J. A. H. Nielsen, A. B. Bregnsbo, S. Zhou, H.-Y. Huang, and others, Quantum learning advantage on a scalable photonic platform, Science 389, 1332 (2025).

[28] S. Chen, M. Fanizza, F. Girardi, L. Lami, F. A. Mele, M. Walter, and F. Witteveen, Optimal tomography of bosonic and fermionic Gaussian states (2026), arXiv:2607.11847 [quant-ph].

[29] A. Seif, S. Chen, S. Majumder, H. Liao, D. S. Wang, M. Malekakhlagh, A. Javadi-Abhari, L. Jiang, and Z. K. Minev, Entanglement-enhanced learning of quantum processes at scale (2024), arXiv:2408.03376.

[30] B. Yu, Assouad, Fano, and Le Cam, in Festschrift for Lucien Le Cam: Research Papers in Probability and Statistics, edited by D. Pollard, E. Torgersen, and G. L. Yang (Springer, New York, NY, 1997) pp. 423–435.

[31] H. Zhao, L. Lewis, I. Kannan, Y. Quek, H.-Y. Huang, and M. C. Caro, Learning Quantum States and Unitaries of Bounded Gate Complexity, PRX Quantum 5 (2024).

[32] C. M. Caves, Quantum-mechanical noise in an interferometer, Physical Review D 23, 1693 (1981).

[33] B. Yurke, S. L. McCall, and J. R. Klauder, SU(2) and SU(1,1) interferometers, Physical Review A 33, 4033 (1986).

[34] D. J. Wineland, J. J. Bollinger, W. M. Itano, F. L. Moore, and D. J. Heinzen, Spin squeezing and reduced quantum noise in spectroscopy, Physical Review A 46, R6797 (1992).

[35] M. J. Holland and K. Burnett, Interferometric detection of optical phase shifts at the Heisenberg limit, Physical

Review Letters 71, 1355 (1993).

[36] J. J. . Bollinger, W. M. Itano, D. J. Wineland, and D. J. Heinzen, Optimal frequency measurements with maximally correlated states, Physical Review A 54, R4649 (1996).

[37] V. Giovannetti, S. Lloyd, and L. Maccone, Quantum-Enhanced Measurements: Beating the Standard Quantum Limit, Science 306, 1330–1336 (2004).

[38] D. Leibfried, M. D. Barrett, T. Schaetz, J. Britton, J. Chiaverini, W. M. Itano, J. D. Jost, C. Langer, and D. J. Wineland, Toward Heisenberg-Limited Spectroscopy with Multiparticle Entangled States, Science 304, 1476 (2004).

[39] V. Giovannetti, S. Lloyd, and L. Maccone, Quantum Metrology, Physical Review Letters 96 (2006).

[40] S. L. Braunstein and C. M. Caves, Statistical distance and the geometry of quantum states, Physical Review Letters 72, 3439 (1994).

[41] M. G. A. Paris, Quantum estimation for quantum technology (2008).

[42] L. Pezzè, A. Smerzi, M. K. Oberthaler, R. Schmied, and P. Treutlein, Quantum metrology with nonclassical states of atomic ensembles, Reviews of Modern Physics 90, 035005 (2018).

[43] H. J. Kimble, Y. Levin, A. B. Matsko, K. S. Thorne, and S. P. Vyatchanin, Conversion of conventional gravitationalwave interferometers into quantum nondemolition interferometers by modifying their input and/or output optics, Physical Review D 65, 022002 (2001).

[44] R. Schnabel, N. Mavalvala, D. E. McClelland, and P. K. Lam, Quantum metrology for gravitational wave astronomy, Nature Communications 1, 121 (2010).

[45] J. Aasi, J. Abadie, B. P. Abbott, R. Abbott, T. D. Abbott, M. R. Abernathy, C. Adams, T. Adams, P. Addesso, and R. X. Adhikari, Enhanced sensitivity of the LIGO gravitational wave detector by using squeezed states of light, Nature Photonics 7, 613–619 (2013).

[46] B. M. Escher, R. L. de Matos Filho, and L. Davidovich, General framework for estimating the ultimate precision limit in noisy quantum-enhanced metrology, Nature Physics 7, 406–411 (2011).

[47] R. Demkowicz-Dobrzański, J. Kołodyński, and M. Guţă, The elusive Heisenberg limit in quantum-enhanced metrol ogy, Nature Communications 3 (2012).

[48] C. C. V. de Pradenne, I. Kannan, H. Putterman, and J. Cotler, Restrictions on non-Cliford fault tolerance and ruling out beyond-SQL quantum metrology (2026), arXiv:2607.27342.

[49] M. Zwierz, C. A. Pérez-Delgado, and P. Kok, General Optimality of the Heisenberg Limit for Quantum Metrology, Physical Review Letters 105, 180402 (2010).

[50] V. Giovannetti and L. Maccone, Sub-Heisenberg Estimation Strategies Are Inefective, Physical Review Letters 108, 210404 (2012).

[51] M. J. W. Hall and H. M. Wiseman, Does Nonlinear Metrology Ofer Improved Resolution? Answers from Quantum Information Theory, Physical Review X 2, 041006 (2012).

[52] M. M. Rams, P. Sierant, O. Dutta, P. Horodecki, and J. Zakrzewski, At the Limits of Criticality-Based Quantum Metrology: Apparent Super-Heisenberg Scaling Revisited, Physical Review X 8, 021022 (2018).

[53] E. Kessler, I. Lovchinsky, A. Sushkov, and M. Lukin, Quantum Error Correction for Metrology, Physical Review Letters 112, 10.1103/physrevlett.112.150802 (2014).

[54] S. Zhou, M. Zhang, J. Preskill, and L. Jiang, Achieving the Heisenberg limit in quantum metrology using quantum error correction, Nature Communications 9 (2018).

[55] J. Cotler, D. L. Danielson, and I. Kannan, Quantum Advantage for Sensing Properties of Classical Fields (2026), arXiv:2602.17591.

[56] L. M. Norris, G. A. Paz-Silva, and L. Viola, Qubit Noise Spectroscopy for Non-Gaussian Dephasing Environments, Physical Review Letters 116 (2016).

[57] Y. Sung, F. Beaudoin, L. M. Norris, F. Yan, D. K. Kim, J. Y. Qiu, U. von Lüpke, J. L. Yoder, T. P. Orlando, S. Gustavsson, and others, Non-Gaussian noise spectroscopy with a superconducting qubit sensor, Nature Communications 10, 3715 (2019).

[58] von Lüpke, Uwe and Beaudoin, Félix and Norris, Leigh M. and Sung, Youngkyu and Winik, Roni and Qiu, Jack Y. and Kjaergaard, Morten and Kim, David and Yoder, Jonilyn and Gustavsson, Simon and Viola, Lorenza and Oliver, William D., Two-Qubit Spectroscopy of Spatiotemporally Correlated Quantum Noise in Superconducting Qubits, PRX Quantum 1, 10.1103/prxquantum.1.010305 (2020).

[59] M. Malnou, D. A. Palken, B. M. Brubaker, L. R. Vale, G. C. Hilton, and K. W. Lehnert, Squeezed vacuum used to accelerate the search for a weak classical signal (2018).

[60] K. M. Backes, D. A. Palken, S. A. Kenany, B. M. Brubaker, S. B. Cahn, A. Droster, G. C. Hilton, S. Ghosh, H. Jackson, S. K. Lamoreaux, and others, A quantum enhanced search for dark matter axions, Nature 590, 238–242 (2021).

[61] A. J. Brady, C. Gao, R. Harnik, Z. Liu, Z. Zhang, and Q. Zhuang, Entangled sensor-networks for dark-matter searches, PRX Quantum 3, 030333 (2022), arXiv:2203.05375 [quant-ph].

[62] H. Shi and Q. Zhuang, Ultimate precision limit of noise sensing and dark matter search, npj Quantum Information 9, 10.1038/s41534-023-00693-w (2023).

[63] T. J. Proctor, P. A. Knott, and J. A. Dunningham, Multiparameter Estimation in Networked Quantum Sensors, Physical Review Letters 120, 080501 (2018).

[64] Z. Eldredge, M. Foss-Feig, J. A. Gross, S. L. Rolston, and A. V. Gorshkov, Optimal and secure measurement protocols for quantum sensor networks, Physical Review A 97, 042337 (2018).

[65] T. Qian, J. Bringewatt, I. Boettcher, P. Bienias, and A. V. Gorshkov, Optimal Measurement of Field Properties with Quantum Sensor Networks, Physical Review A 103, L030601 (2021), arXiv:2011.01259 [quant-ph].

[66] N. Wiebe, C. Granade, C. Ferrie, and D. Cory, Hamiltonian Learning and Certification Using Quantum Resources,

Physical Review Letters 112, 10.1103/physrevlett.112.190501 (2014).

[67] S. Prabhu, S. A. Khan, X. Song, M. Ouellet, R. Yanagimoto, S. Roy, A. Senanian, L. G. Wright, V. Fatemi, and P. L. McMahon, Quantum computational displacement sensing (2026), arXiv:2604.13177.

[68] K. S. Chou, Teleported operations between logical qubits in circuit quantum electrodynamics (Ph.D. Thesis) (Yale University, 2018).

[69] A. Eickbusch, V. Sivak, A. Z. Ding, S. S. Elder, S. R. Jha, J. Venkatraman, B. Royer, S. M. Girvin, R. J. Schoelkopf, and M. H. Devoret, Fast universal control of an oscillator with weak dispersive coupling to a qubit, Nature Physics 18, 1464 (2022).

[70] J. W. Foster, Y. Kahn, R. Nguyen, N. L. Rodd, and B. R. Safdi, Dark matter interferometry, Physical Review D 103 (2021).

[71] D. Lynden-Bell and R. M. Lynden-Bell, Ghostly streams from the formation of the Galaxy’s halo, Mon. Not. Roy. Astron. Soc. 275, 429 (1995).

[72] J. W. Gardner, T. Gefen, S. A. Haine, J. J. Hope, J. Preskill, Y. Chen, and L. McCuller, Stochastic Waveform Estimation at the Fundamental Quantum Limit, PRX Quantum 6, 10.1103/h91r-4ws9 (2025).

[73] A. O. Sushkov, Quantum Science and the Search for Axion Dark Matter, PRX Quantum 4, 10.1103/prxquantum.4.020101 (2023).

[74] A. A. Clerk, M. H. Devoret, S. M. Girvin, F. Marquardt, and R. J. Schoelkopf, Introduction to quantum noise, measurement, and amplification, Reviews of Modern Physics 82, 1155–1208 (2010).

[75] P. Krantz, M. Kjaergaard, F. Yan, T. P. Orlando, S. Gustavsson, and W. D. Oliver, A quantum engineer’s guide to superconducting qubits, Applied Physics Reviews 6 (2019).

[76] M. Aspelmeyer, T. J. Kippenberg, and F. Marquardt, Cavity optomechanics, Reviews of Modern Physics 86, 1391 (2014).

[77] C. D. Bruzewicz, J. Chiaverini, R. McConnell, and J. M. Sage, Trapped-ion quantum computing: Progress and challenges, Applied Physics Reviews 6 (2019).

[78] C. C. Gerry, P. L. Knight, and M. Beck, Introductory Quantum Optics, Vol. 1 (Cambridge university press Cambridge, 2005).

[79] G. H. Low and I. L. Chuang, Optimal Hamiltonian Simulation by Quantum Signal Processing, Physical Review Letters 118, 10.1103/physrevlett.118.010501 (2017).

[80] S. K. Lamoreaux, K. A. van Bibber, K. W. Lehnert, and G. Carosi, Analysis of single-photon and linear amplifier detectors for microwave cavity dark matter axion searches, Physical Review D 88, 10.1103/physrevd.88.035020 (2013).

[81] A. V. Dixit, S. Chakram, K. He, A. Agrawal, R. K. Naik, D. I. Schuster, and A. Chou, Searching for Dark Matter with a Superconducting Qubit, Physical Review Letters 126, 10.1103/physrevlett.126.141302 (2021).

[82] S. Lloyd and S. L. Braunstein, Quantum Computation over Continuous Variables, Physical Review Letters 82, 1784–1787 (1999).

[83] J. W. Foster, N. L. Rodd, and B. R. Safdi, Revealing the dark matter halo with axion direct detection, Physical Review D 97, 10.1103/physrevd.97.123006 (2018).

[84] G. A. Álvarez and D. Suter, Measuring the Spectrum of Colored Noise by Dynamical Decoupling, Phys. Rev. Lett. 107, 230501 (2011).

[85] J. Bylander, S. Gustavsson, F. Yan, F. Yoshihara, K. Harrabi, G. Fitch, D. G. Cory, Y. Nakamura, J.-S. Tsai, and W. D. Oliver, Noise spectroscopy through dynamical decoupling with a superconducting flux qubit, Nature Physics 7, 565 (2011).

[86] T. Rosskopf, J. Zopes, J. M. Boss, and C. L. Degen, A quantum spectrum analyzer enhanced by a nuclear spin memory, npj Quantum Information 3, 33 (2017).

[87] G. A. Paz-Silva, L. M. Norris, and L. Viola, Multiqubit spectroscopy of Gaussian quantum noise, Physical Review A 95, 10.1103/physreva.95.022121 (2017).

[88] Szańkowski, P and Ramon, G and Krzywda, J and Kwiatkowski, D and Cywiński, Ł, Environmental noise spectroscopy with qubits subjected to dynamical decoupling, Journal of Physics: Condensed Matter 29, 333001 (2017).

[89] M. Shao and C. L. Nikias, Signal Processing with Fractional Lower-Order Moments: Stable Processes and Their Applications, Proceedings of the IEEE 81, 986 (1993).

[90] P. Collaboration, Y. Akrami, F. Arroja, M. Ashdown, J. Aumont, C. Baccigalupi, M. Ballardini, A. J. Banday, R. B. Barreiro, N. Bartolo, and others, Planck 2018 results. IX. Constraints on primordial non-Gaussianity (2019), arXiv:1905.05697 [astro-ph.CO].