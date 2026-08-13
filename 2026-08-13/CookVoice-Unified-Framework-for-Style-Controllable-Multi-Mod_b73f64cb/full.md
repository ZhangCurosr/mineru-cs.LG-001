# CookVoice: Unified Framework for Style Controllable Multi-Modal Human Voice Generation

Haowei Lou <sup>ID</sup> <sup>1</sup>, Hye-Young Paik <sup>ID</sup> <sup>1</sup>, Dai Jia <sup>ID</sup> <sup>2</sup>, Kai Li <sup>ID</sup> <sup>2</sup>, Lina Yao <sup>ID</sup> <sup>1</sup>

<sup>1</sup> UNSW Sydney

<sup>2</sup> Dolby Laboratories

haowei.lou@unsw.edu.au, h.paik@unsw.edu.au, Jia.Dai@dolby.com, Kai.Li@dolby.com, lina.yao@unsw.edu.au

## Abstract

Human voice generation has made rapid progress in speech generation, singing voice generation, voice cloning, and voice editing. However, most existing systems are designed for specific tasks and often rely on task-dependent architectures, control signals, or autoregressive decoding, limiting fine-grained controllability and inference efficiency. In this paper, we propose CookVoice, a unified framework for multimodal, multi-style, and multi-task human voice generation. CookVoice decomposes the human voice into three key factors: content, prosody, and style, enabling both speech and singing voice generation within a unified model. To achieve precise and flexible controllability, we design a flexible alignment strategy that maps text, style, and prosody control signals onto the frame-level of spectrogram. This design allows CookVoice to support a wide range of tasks, including text-to-speech, text-to-singing voice, style-controllable generation, voice mimicry, voice conversion, and voice editing. Experimental results show that CookVoice achieves generation quality comparable to existing Text-to-Speech and text-to-singing voice baselines, while providing stronger style and prosody controllability. Moreover, CookVoice achieves comparable performance to large-scale baselines with only 43.51 million parameters and efficient inference using as few as 4 ODE steps, making it a practical solution for real-world human voice generation applications. Demo page is available at https://haoweilou.github.io/CookVoice/.

Index Terms: voice generation, generative artificial intelligence, speech generation, singing voice generation

## 1. Introduction

Recent years have witnessed an emergence of human voice generative AI, with models capable of high-quality text-tospeech (TTS) [1, 2, 3, 4], singing voice generation [5, 6, 7], zero-shot voice cloning [2, 4, 1], and voice editing [8]. Driven by advances in acoustic representation learning [9, 10], and generative artificial intelligence [11, 12] modern systems can generate increasingly natural and expressive voices from multi-modal input. However, despite this progress, most existing voice generation systems are still designed around isolated tasks. TTS models focus on generating speech from text [13], singing voice synthesis models typically require musical scores and note input [14, 5], style-controllable TTS models rely on text style prompts or reference speech [3, 1], and voice cloning models mainly emphasize speaker similarity [4]. Different forms of human voice generation tasks are commonly treated as separate problems, requiring task-specific architectures, datasets, training objectives, and control signals.

This fragmentation makes human voice generation unnecessarily complex. Theoretically, speech and singing share the same fundamental linguistic basis, yet they differ substantially in temporal structure and prosodic behavior. In speech, duration and prosody are often implicit and context-dependent, while in singing, they are explicitly constrained by musical notes, beats, and melody. Moreover, style-controllable voice generation focuses on cloning the rich paralinguistic styles within reference speech, such as emotion, gender, age, timbre, and speaking or singing style. These styles may come from different modalities, including natural language descriptions, reference speech, reference singing voices, lexical tones, stresses, notes, or F0 contours. Existing methods usually handle only a subset of these controls, making it difficult to flexibly decide what to generate, how it should sound, and where its prosody should come from.

A unified model capable of generating different forms of human voice under a single controllable interface is highly desirable for both academia and the industrial community. Ideally, such a model should support both speech and singing voice generation, accept multimodal style control, and allow explicit control over prosody. This requires a framework that can represent human voice generation through shared factors, including content, prosody, and style, while preserving the distinct characteristics of different tasks. More importantly, the framework should remain simple to train, straightforward to use, and efficient at inference time.

However, achieving such a unified framework remains challenging. Many recent voice generation models rely on autoregressive (AR) decoding or implicit alignment mechanisms [1, 2], while recent unified voice generation systems further attempt to cover multiple tasks within a single model [15, 8]. Although these models can achieve strong generation quality, their controllability is often constrained by the underlying generation mechanism. AR models generate acoustic tokens sequentially and therefore model duration and prosody implicitly through token prediction. This design is flexible at the utterance level, but it makes fine-grained temporal control difficult, since users cannot directly specify how each phoneme, note, or frame should evolve over time. In contrast, many NAR systems [13, 3, 16, 4] improve temporal stability by imposing explicit or restricted alignments, such as phoneme-level durations [13, 16, 3], noteto-phoneme mapping [14, 6, 7], or sentence-level prosody references [15, 8]. While these designs improve controllability over AR decoding, they often rely on predefined assumptions between content and prosody, which limits their flexibility when multiple control signals are jointly involved.

These limitations become more severe for tasks requiring precise temporal control, such as singing voice generation and prosody mimicry. For singing, phonemes must follow scoredefined note durations and melodies; for voice mimicry, the generated voice should preserve detailed frame-level prosodic patterns from a reference signal. When content, style, and prosody are combined, implicit AR alignment or restricted NAR alignment makes it difficult to achieve both flexible control and precise temporal consistency. To address this issue, CookVoice adopts a flexible frame-level alignment strategy. Instead of assuming a fixed dependency between content and prosody, CookVoice explicitly expands different control signals to the acoustic frame level and allows them to be combined freely. This design enables more fine-grained and flexible controllability across speech, singing voice generation, and prosody mimicry.

Table 1: Alignment method and controllability comparison.
<table><tr><td>Model Class</td><td>Alignment Method</td><td>Controllability</td></tr><tr><td>AR</td><td>Token prediction</td><td>Voice-level</td></tr><tr><td>NAR</td><td>Restricted alignment</td><td>Phoneme-level</td></tr><tr><td>CookVoice</td><td>Flexible alignment</td><td>Frame-level</td></tr></table>

In this work, we propose CookVoice, a unified nonautoregressive (NAR) framework for multimodal, multi-style, and multi-task human voice generation. CookVoice formulates human voice generation as conditional latent acoustic generation, where the target voice is represented in a compact latent space and generated from three shared factors: content, prosody, and style. Content is represented by phoneme sequences derived from text or lyrics. Prosody can be controlled through discrete signals, such as lexical tones, stresses, and MIDI notes, or continuous frame-level F0 contours extracted from a reference voice. Style can be provided by either textbased descriptions or reference voice prompts, enabling both natural language style control and reference-based style transfer.

Instead of relying on implicit AR token-prediction, CookVoice adopts temporally aligned NAR generation pipeline which explicitly align content, prosody, and style tokens to acoustic tokens at frame level. For speech, phoneme durations are learned or predicted from alignment information. For singing voice, durations are deterministically derived from musical score timing. This design allows speech and singing voices to be modeled under the same temporal framework, while preserving their different duration and prosody structures. The temporally aligned representations are then integrated through a multimodal adaptive fusion module and passed to a flowmatching Diffusion Transformer for high-quality latent acoustic generation.

CookVoice supports a broad range of human voice generation tasks within a single framework, including TTS, text-stylecontrollable TTS, voice-style-controllable TTS, singing voice generation, style-controllable singing voice generation, prosody mimicry and many others. These tasks are handled by activating different combinations of content, prosody, and style conditions, rather than training separate task-specific models. As a result, CookVoice is a simple, interpretable, and highly controllable framework for the generation of human voice in both speech and singing domains. The main contributions are summarized as follows:

1. We propose CookVoice, a unified framework for multi-task human voice generation. By decomposing human voice into style, content, and prosody, CookVoice successfully unifying multiple voice generation task within a unified model.

2. We design a flexible alignment mechanism for fine-grained controllable human voice generation. By aligning different control signals at frame level, CookVoice can flexibly support multimodal style and prosody control, enabling more precise control over paralinguistic style, melody, duration, and prosodic expression.

3. Our experiment demonstrate CookVoice achieves higher style and prosody controllability than existing baselines, including models trained with larger-scale data or larger model sizes. Despite using a lightweight architecture and a much smaller training data, CookVoice achieves superior style similarity and F0 controllability while maintaining efficient inference and competitive generation quality.

## 2. Preliminary

## 2.1. Problem Formulation

Human Voice Generation (HVG) is to generate waveform from different modality of control signal. In this study, we define the human voice as a combination of human speech and the singing voice. Control signals include text, lyric, note, and reference voice. Full details covered tasks are presented in Appendix D. For readers unfamiliar with linguistic and musical concepts, we provide terminology definitions in Appendix E.

Formally, we formulate HVG tasks as follow. Let $Y \in$ $\mathbb { R } ^ { N \times T }$ denotes the latent embedding extracted from a spectrogram via HiFi-GAN [9], with N and T denoting the latent dimension and the number of frames. To systematically model the generation process, we assume the human voice can be disentangled into five independent key components:

Style $( S \in \mathbb { R } ^ { D } )$ : Encodes a global paralinguistic styles embedding of voice, indicating how the voice is expressed (e.g., gender, age, and emotion). This style embedding can be conditioned on different sources: either a text description $S _ { T }$ (e.g., ”A female speaking happily”), or a reference voice $S _ { \nu }$ to clone the speaking style of target voice V;

Content $( X \in \mathbb { R } ^ { L _ { 1 } } ) \colon$ Represents the linguistic content to be voiced. Content in Text-to-Speech (TTS) can be text, while in Text-To-Singing Voice (TTSV), it refers to the lyrics. In this study, X specifically denotes the sequence of phonemes;

Prosody $( { \mathcal { P } } ) { \mathrm { : } }$ contains all forms of prosody variations that shape the expressiveness and melody of the voice. We categorize prosody signals into discrete and continuous signals. The discrete signal $( \mathcal { P } _ { d i s c } )$ contains lexical prosody tokens $\mathcal { P } _ { l e x } \in \mathbb { R } ^ { L _ { 1 } }$ (e.g., tones in Mandarin Chinese or word stress in English) for speech, and musical notes $\mathcal { P } _ { n o t e } \in \mathbb { R } ^ { L _ { 2 } }$ for singing voices. While musical notes govern prosody variation similar to lexical tones, their sequence length $( L _ { 2 } )$ can differ from the phoneme length $( L _ { 1 } )$ , as a single musical note may span multiple phonemes, or a single phoneme may stretch across multiple notes. Alternatively, the continuous prosody $( \mathcal { P } _ { c o n t } )$ represents the frame-level fundamental frequency $( F _ { 0 } \in \mathbb { R } ^ { T } )$ contour extracted from a voice. It aligns temporally with the latent embedding, sharing the same frame length T.

Our objective is to train a generative model G(·) that can generate latent embedding $\hat { Y }$ conditioned on different source of control signal $( \hat { Y } = \mathcal G ( S , X , \mathcal P ) )$ ). Depending on the requirements of different tasks, S and P can be selectively conditioned on different source. Detailed task formulations are provided in Table 5.

## 2.2. Latent Acoustic Encoding

We employ an Autoencoder (AE) with similar architecture with the HiFi-GAN [9] to efficiently encode audio signals into latent embedding. The encoder takes a linear spectrogram spec $\in$ $\mathbb { R } ^ { 5 1 3 \times T }$ as input and compresses it into a continuous latent embedding $\boldsymbol { Y } \in \mathbb { R } ^ { \boldsymbol { N } \times \boldsymbol { T } }$ , where N denotes the latent channel dimension and $T$ represents the temporal frame length. The decoder is constructed using a series of transposed 1D convolutional layers, which progressively upsample the latent embedding $Y$ to reconstruct the original audio waveform. To ensure high-fidelity audio generation, the AE is jointly optimized using an adversarial loss from a discriminator alongside a reconstruction loss. Once the AE is fully trained, its weights are frozen. Latent embedding Y serves as the ground-truth target for CookVoice during voice generation.

## 3. CookVoice

The overall architecture of CookVoice is presented in Figure 1.   
This section will present the architecture of CookVoice in detail.

## 3.1. Style Encoder

Style Encoder is designed to generate a style embedding $S ,$ which dictates the stylistic characteristics of the generated voice. In CookVoice, style conditioning can be derived from two modalities.

The first modality is text, coming from the text-based style prompt $S _ { T }$ . In this scenario, the style is controlled by natural language descriptions (e.g., a female is speaking with a happy emotion). Given a text-based style description $\tau$ , we first extract its semantic embedding using a pre-trained, frozen, MP-Net [17] sentence encoder. The resulting embedding is then projected through a trainable linear layer to obtain the text style embedding $S _ { T } \in \mathbb { R } ^ { D }$

The second modality is a reference voice, V. This addresses scenarios requiring zero-shot voice cloning or timbre transfer, where the generated voice must mimic a reference voice. We encode the acoustic signal into a latent space using the encoding method described in Section 2.2. The resulting latent embedding $\boldsymbol { \nu } \in \mathbb { R } ^ { N \times T }$ is then processed by a trainable Transformer [18] encoder with an attentive pooling layer to obtain the voice style embedding $S _ { \nu } \in \mathbb { R } ^ { D }$

During model training, the overall style condition S is uniformly sampled from either the text modality $S _ { T }$ or the voice modality $S _ { \nu }$ with a 50% probability for each item within the batch. To align with the temporal resolution of the latent embedding $\boldsymbol { \nu } \in \mathbb { R } ^ { \boldsymbol { \breve { N } \times T } }$ , the style embedding $S$ is expanded by repeating it T times along the temporal axis, producing the sequencelength style embedding $S _ { e } \in \mathbb { R } ^ { D \times T }$

## 3.2. Content Encoder

We employ a LanStyleTTS multilingual Grapheme-to-phenome (G2P) module [19] to convert text into phonemes $X$ and prosody tokens $P ,$ , where $X , P \in \mathbb { R } ^ { L _ { 1 } }$ . The phoneme sequence X captures the linguistic content, while the prosody sequence $P$ explicitly models phoneme-level prosodic variations (e.g., lexical tones in Mandarin or phoneme stress in English) across different linguistic contexts. In CookVoice, X strictly denotes the linguistic content, while $P$ serves as a discrete prosodyconditioning signal (also detailed in Section 3.3.1).

Phonemes are first mapped into a sequence of embeddings $X \in \mathbb { R } ^ { D \times L _ { 1 } }$ and then processed through Feed-Forward Transformer (FFT) blocks [13]. A critical factor for natural and expressive voice generation is the duration of the phoneme $\bar { \mathcal { D } _ { X } } \in \mathbb { Z } ^ { L _ { 1 } }$ , which exhibits vastly different behaviors in the speech versus singing voice. In conventional TTS tasks, the absence of explicit duration constraints necessitates the use of a duration predictor [13, 3] or AR architectures [1, 20] for implicit inference. Conversely, in TTSV tasks, note durations are explicitly dictated by musical scores. To unify both tasks within CookVoice, we adopt a flexible duration alignment and expansion strategy.

During the training stage, we handle speech and singing voice differently. For speech, we use pre-trained ParaStyleTTS [3] to find optimal alignments between the speech audio and the phoneme sequence, yielding ground-truth durations $\mathcal { D } _ { X }$ For singing voice, where score durations are denoted in relative beats rather than absolute time (e.g., seconds), we compute the relative temporal proportion of each phoneme based on its assigned beats. This relative proportion is then multiplied by the total number of acoustic frames T to ascertain the absolute frame-level duration. Both duration sequences satisfy the constraints $\mathcal { D } _ { X } \in \mathbb { Z } ^ { L _ { 1 } }$ and $\begin{array} { r } { \sum _ { i = 1 } ^ { L _ { 1 } } { D _ { X } ^ { i } } = \dot { T } } \end{array}$ . Each phoneme embedding $X _ { i }$ is then repeated $\mathcal { D } _ { X } ^ { i }$ times to produce the durationexpanded content embedding $\mathbf { \bar { \Psi } } _ { X _ { e } } ^ { \mathbf { \Psi } } \in \mathbf { \Psi } \mathbb { R } ^ { D \times T }$ , which perfectly matches the temporal dimension of the latent acoustic embedding.

During the inference stage, we use a trained duration predictor from ParaStyleTTS [3] to predict phoneme durations for bilingual speech. For singing voice generation, the duration is deterministically computed from the provided musical score using the relative proportion methodology applied during training.

## 3.3. Prosody Encoder

Prosody affects how the content is expressed differently for speech and singing. In CookVoice, we categorize control signals of prosody, denoted as $\mathcal { P } _ { \cdot }$ , into discrete and continuous signals. The expanded prosody token is denoted as $\mathcal { P } _ { e } \in \mathbb { R } ^ { D \times T }$

## 3.3.1. Discrete Signals

Discrete prosody signals come from either the G2P module (for speech) or musical scores (for singing). For speech, these consist of the aforementioned lexical prosody tokens $P _ { l e x }$ (tones and stresses). Since the sequence length of $P _ { l e x }$ matches the phonemes $( L _ { 1 } )$ , each token has the same duration $\mathcal { D } ^ { i }$ of its corresponding phoneme. Consequently, the mapped token embeddings are repeated $\mathcal { D } ^ { i }$ times, giving the expanded discrete speech prosody embedding $\bar { P _ { d i s c , e } } \in \bar { \mathbb { R } } ^ { D \times { T } }$

For singing voice, the discrete signals are musical note tokens $\overline { { P _ { n o t e } } } ^ { \mathbf { \bar { \alpha } } } \in \mathbb { R } ^ { L _ { 2 } }$ . Similarly to the duration expansion in the Phoneme Encoder, we compute the exact frame duration for each note based on the score’s beat allocation and the total frames $T .$ The note embeddings are then expanded accordingly, serving as the singing counterpart for $P _ { d i s c , e }$

## 3.3.2. Continuous Signals

Continuous prosody signals are extracted directly from the reference voice waveform to explicitly represent fundamental frequency variations over time. We extract a continuous $F _ { 0 }$ contour $\dot { \in \mathbb { R } ^ { T } }$ , which is passed through a linear projection layer to generate the continuous prosody embedding $\mathbf { \bar { \mathit { P } } } _ { c o n t } \in \mathbb { R } ^ { \mathbf { \bar { \mathit { D } } } \times T }$

During model training, the prosody embedding $\mathcal { P } _ { e } \in$ $\mathbb { R } ^ { D \times T }$ is constructed by randomly selecting the discrete representation $( P _ { d i s c , e } )$ or the continuous representation $( P _ { c o n t } )$ with equal probability. This strategy ensures that the model learns robust prosody control from both discrete tokens and continuous signals.

![](images/2054c35d757e0ef2d6c0b9c91bc90026595d7b8564e46c7addc4c78e265d9a88.jpg)  
Figure 1: Architecture of the CookVoice

## 3.4. Flow-Matching DiT

Given the style $S _ { e } ,$ content $X _ { e } ,$ , and prosody $\mathcal { P } _ { e }$ satisfy $S _ { e } , X _ { e } , \mathcal { P } _ { e } \in \mathbb { R } ^ { D \times T }$ , we concatenate them along the feature dimension to construct a conditioning embedding $\breve { C } \in \mathbb { R } ^ { 3 D \times T } ;$

$$
C = [ S _ { e } ; X _ { e } ; \mathcal { P } _ { e } ]
$$

In the final stage of CookVoice, our objective is to translate the condition embedding $C \in \mathbb { R } ^ { 3 \acute { D } \times T }$ into high-fidelity latent embedding. To achieve this, we use a Diffusion Transformer (DiT) [11] as the core acoustic model, where $C$ serves as the essential cross-attention conditioning signal to guide the voice generation process.

We frame the generative process within the Optimal Transport Flow-Matching [12] (OT-FM) paradigm. Assuming a standard Gaussian prior $Y _ { 0 } \ \sim \ { \mathcal { N } } ( 0 , I )$ and a target voice latent embedding $\boldsymbol { Y } _ { 1 } ~ \in ~ \mathbb { R } ^ { \boldsymbol { N } \times \boldsymbol { T } }$ , our DiT network, parameterized as $v _ { \theta } ( Y _ { t } , t , C )$ , is trained to regress the target vector field $Y _ { 1 } - Y _ { 0 }$ that drives the linear probability path between the prior and the empirical data distribution. The model is optimized using the standard Flow-Matching Mean Squared Error (MSE) objective.

$$
\mathcal { L } _ { F M } = \mathbb { E } _ { t , Y _ { 0 } , Y _ { 1 } } \left[ | | v _ { \theta } ( Y _ { t } , t , C ) - ( Y _ { 1 } - Y _ { 0 } ) | | ^ { 2 } \right]\tag{1}
$$

During inference, we sample the noise $Y _ { 0 }$ and simulate the empirical probability flow Ordinary Differential Equation (ODE), given by $d Y _ { t } = v _ { \theta } ( Y _ { t } , t , C ) d t$ . To solve this ODE and map the unstructured noise to the final acoustic representation $Y _ { 1 }$ , we employ a standard first-order Euler numerical solver. Starting from $t = 0 ,$ , the trajectory is discretely integrated with a fixed step size ∆t:

$$
Y _ { t + \Delta t } = Y _ { t } + \Delta t \cdot v _ { \theta } ( Y _ { t } , t , C )\tag{2}
$$

To transform noise $Y _ { 0 }$ into latent embedding condition on $C .$

## 3.5. Multi-Task Training

CookVoice supports multiple generation tasks through a simple condition-switching training strategy. Instead of designing taskspecific objectives or separate task heads, we randomly replace the conditioning embeddings during training. For each sample in the training batch, the style condition is randomly sampled from either the text-based style embedding $S _ { T }$ or the voicebased style embedding $S _ { \nu }$ . Similarly, the prosody condition is randomly sampled from either the discrete prosody embedding $\mathcal { P } _ { d i s c , e }$ or the continuous $F _ { 0 }$ -based prosody embedding $\mathcal { P } _ { c o n t } \colon$

$$
S = \left\{ { \begin{array} { l l } { S _ { T } , } & { p = 0 . 5 , } \\ { S _ { \nu } , } & { p = 0 . 5 , } \end{array} } \right. ~ \mathcal { P } _ { e } = \left\{ { \begin{array} { l l } { \mathcal { P } _ { d i s c , e } , } & { p = 0 . 5 , } \\ { \mathcal { P } _ { c o n t } , } & { p = 0 . 5 . } \end{array} } \right.\tag{3}
$$

This randomization is applied at the sample level within the batch, so different samples may use different style and prosody sources in the same training step. As a result, CookVoice learns to generate voices under different combinations of control signals, such as text style with discrete prosody, voice style with discrete prosody, text style with continuous $F _ { 0 }$ , and voice style with continuous $F _ { 0 }$ . This strategy allows a single model to support multiple speech and singing voice generation tasks without changing the model architecture or training objective.

## 4. Experiments

## 4.1. Dataset

We combine multiple open-sourced bilingual (english and chinese ) speech and singing voice dataset to conduct the experiment including Baker [21], LJSpeech [22], ESD [23], CREMA-D [24], CommonPhone [25], Genshin Voice dataset [26], GTSinger [27]. The combined dataset contains approximately 123k voice samples, 168 hours of voices from 6,361 speakers.

## 4.2. Preprocessing

First, we extract the raw $F _ { 0 }$ contour (in Hz) from each voice sample. To accommodate the wide and varying pitch ranges inherent to both speech and singing voices, we project the $F _ { 0 }$ values onto a logarithmic scale and normalize them using a lower of 50 Hz.

$$
\tilde { f } = \frac { \log \mathrm { F } _ { 0 } - \log f _ { \mathrm { m i n } } } { \log f _ { \mathrm { m a x } } - \log f _ { \mathrm { m i n } } } , \quad f _ { \mathrm { m i n } } = 5 0 .\tag{4}
$$

While $\tilde { f } _ { t }$ provides a standardized absolute pitch, relying on it directly can introduce entanglement. Paralinguistic styles, such as age and gender, can effectively dictate a speaker’s absolute pitch range and are already captured by the style embedding. Using absolute $F _ { 0 }$ as an explicit condition would therefore cause interference between prosody and style. To decouple the melody from the speaker’s timbre, we convert the absolute $F _ { 0 }$ into a relative pitch contour by removing its voice-level mean. Specifically:

$$
\hat { f } = \left\{ \begin{array} { l l } { \tilde { f } _ { t } - \operatorname* { m e a n } ( \tilde { f } _ { t } ) , } & { t \in \mathcal { U } } \\ { - 2 , } & { t \not \in \mathcal { U } , } \end{array} \right.\tag{5}
$$

U denotes voiced frames. For unvoiced frames, we set a constant value of −2. This design ensures that the explicit $F _ { 0 }$ input exclusively controls relative intonation and melody, effectively avoiding conflicts with the style.

## 4.3. Implementation Details

The CookVoice model is trained on 110k samples, with a singing-to-speech ratio of 1:9. All evaluation samples are not seen by the model during training stage. For each sample, the temporally aligned style, content, and prosody embeddings are concatenated along the feature dimension to form the conditioning embedding. This conditioning embedding is then used as the condition for DiT-S backbone [11], which learns to generate the target latent acoustic embedding. The model is optimized with the Flow-Matching objective [12]. We train CookVoice on a single NVIDIA RTX 5090 GPU with 32 batch size for 800K steps.

## 4.4. Evaluation Metrics

To comprehensively evaluate the performance of CookVoice, we use both subjective and objective evaluation metrics to assess generated voice quality, intelligibility, style adherence, and prosody accuracy.

Subjective Evaluation: We evaluate CookVoice from four perspectives: perceptual quality, content intelligibility, style controllability, and prosody controllability. For subjective evaluation, we conduct Mean Opinion Score (MOS) listening tests to measure the naturalness and overall perceptual quality of generated speech and singing voices. For singing voice generation, we additionally use Melody Comparative MOS (M-CMOS) to evaluate melody consistency.

Objective Evaluation: We first assess content intelligibility using Word Error Rate (WER), Phoneme Error Rate (PhoER), and Prosody Error Rate (ProER). The generated audio is transcribed by an ASR system, Whisper [28], and compared with the ground-truth content at different levels. WER measures word-level errors, PhoER measures phonemelevel errors, and ProER measures errors in discrete prosody tokens such as tones, stresses, or note-related prosodic labels. To evaluate style controllability, we compute Style Similarity (S-SIM) between the generated voice and the target style reference. Specifically, we extract style embeddings from both audio samples using a pre-trained style encoder [29] and calculate their cosine similarity, where a higher value indicates better style preservation. To evaluate prosody controllability, we use F0 Root Mean Square Error (F0-RMSE) and F0 Pearson Correlation (F0-CORR). F0-RMSE measures the absolute pitch deviation between generated and target F0 contours, while F0- CORR measures whether the generated pitch follows the target pitch trend. Detailed metric definitions are provided in the $\mathbf { A } \mathbf { p } \mathbf { \cdot }$ pendix B.

## 5. Results & Discussion

In this section, we provide an analysis of experimental results and organize the discussion to answer the following five research questions (RQs):

• RQ1: How does CookVoice perform compared with existing TTS and TTSV baselines in term of audio?

• RQ2: How controllable is CookVoice in terms of style and prosody compared with existing baselines?

• RQ3: How do different style and prosody control signals affect the generation performance?

• RQ4: How does the number of ODE solver steps affect the quality-efficiency trade-off?

• RQ5: How efficient is CookVoice compared with existing TTS and TTSV models?

## 5.1. RQ1: Performance Comparison

We compare CookVoice with representative TTS and TTSV baselines in terms of controllability, perceptual quality, and intelligibility using both subjective and objective metrics. As shown in Tables 2 and 3, CookVoice achieves comparable generation quality while providing stronger style and prosody controllability across both TTS and TTSV tasks.

The main advantage of CookVoice lies in controllability. For TTS, CookVoice consistently improves style similarity and prosody consistency over the compared baselines. Under voicebased style control, CookVoice achieves an S-SIM of over 90% and an F0-CORR of up to 0.7102, whereas the best TTS baselines (Vevo2) achieve an SSIM of approximately 75% and an F0-CORR of approximately 0.25. A similar trend is observed for TTSV, CookVoice achieves the highest style similarity and prosody consistency, reaching an S-SIM of 95.00% and an F0- CORR of 0.8425. These results demonstrate that CookVoice more accurately preserves the style instruction from either voice or text modality and follows the prosody sourced from different types of control signals.

In terms of perceptual quality, CookVoice exhibits different performance patterns for speech and singing voice generation. For TTS, its MOS scores are lower than most TTS baselines, although its best configuration achieves a MOS of 3.98, close to the ground-truth score of 4.05. For TTSV, CookVoice achieves a best MOS of 3.40, outperforming most of the compared singing voice generation systems and remaining comparable to Vevo2, which achieves 3.42. CookVoice also obtains the highest MC-MOS score, indicating better perceived melody consistency.

For intelligibility, CookVoice achieves comparable WER results. Its TTS performance lies in the middle range of the evaluated baselines. For TTSV, CookVoice outperforms most baselines on the Chinese song and achieves the secondlowest English WER among the models that support English

Table 2: Controllability evaluation. Include style controllability, Style-Similarity S-SIM ↑ and Prosody Controllability F0-RMSE ↓ and F0-CORR ↑. Values in parentheses indicate the relative performance improvements compared to the best-performing baselines.

$$
\mathrm { T e x t } + \mathrm { D I S C }
$$

$$
\mathrm { V o i c e } + \mathrm { D I S C }
$$

$$
3 0 . 5 3 \pm 1 0 . 9 8
$$

$$
4 6 . 6 7 \pm 1 7 . 1 0
$$

$$
1 3 8 . 1 8 \pm 4 3 . 1 7
$$

$$
0 . 1 0 6 5 \pm 0 . 1 7 0 9
$$

$$
\mathrm { V o i c e } + \mathrm { D I S C }
$$

$$
6 6 . 2 9 \pm 1 3 . 7 3
$$

$$
1 2 6 . 3 5 \pm 4 5 . 2 3
$$

$$
0 . 1 9 0 2 \pm 0 . 2 4 2 3
$$

$$
\mathrm { T e x t } + \mathrm { D I S C }
$$

$$
1 3 6 . 5 6 \pm 4 6 . 2 8
$$

$$
4 2 . 9 6 \pm 1 7 . 7 3
$$

$$
0 . 1 5 0 5 \pm 0 . 1 7 0 3
$$

$$
\mathrm { V o i c e } + \mathrm { D I S C }
$$

$$
1 2 6 . 5 6 \pm 3 7 . 9 3
$$

$$
7 1 . 1 9 \pm 1 3 . 7 7
$$

$$
0 . 1 7 7 7 \pm 0 . 2 2 0 6
$$

$$
\mathrm { V o i c e } + \mathrm { D I S C }
$$

$$
1 2 3 . 6 5 \pm 4 3 . 9 9
$$

$$
7 5 . 1 1 \pm 8 . 2 0
$$

$$
0 . 2 5 4 8 \pm 0 . 2 5 6 0
$$

$$
1 3 6 . 0 0 \pm 4 5 . 0 9
$$

$$
0 . 1 2 4 2 \pm 0 . 1 8 0 6
$$

$$
S _ { T } + \mathcal { P } _ { d i s c }
$$

$$
\mathrm { T e x t } + \mathrm { D I S C }
$$

$$
6 0 . 7 8 \pm 1 4 . 8 6 _ { ( + 4 1 . 4 8 \% ) }
$$

$$
\mathrm { T e x t } + \mathrm { C O N T }
$$

$$
S \tau + \mathcal { P } _ { c o n t }
$$

$$
1 1 9 . 5 6 \pm 5 0 . 7 3 _ { ( \downarrow 5 . 5 3 \% ) }
$$

$$
0 . 3 9 2 9 \pm 0 . 2 3 9 5 _ { ( + 1 2 1 . 1 0 \% ) }
$$

$$
7 3 . 0 9 \pm 1 \mathrm { \dot { 6 } } . 3 8
$$

$$
8 8 . 8 5 \pm 3 8 . 5 6
$$

$$
\mathrm { V o i c e } + \mathrm { D I S C }
$$

$$
S _ { \mathcal { V } } + \mathcal { P } _ { d i s c }
$$

$$
8 7 . 4 7 \pm 3 . 7 7 _ { ( + 1 6 . 4 6 \% ) }
$$

$$
0 . 6 4 1 6 \pm 0 . 2 0 1 1
$$

$$
9 9 . 0 5 \pm 4 3 . 7 7 _ { ( \downarrow 1 9 . 8 9 \% ) }
$$

$$
0 . 5 3 3 0 \pm 0 . 2 3 3 8 _ { ( + 1 0 9 . 1 8 \% ) }
$$

$$
S _ { \nu } + \mathcal { P } _ { c o n t }
$$

$$
\mathrm { V o i c e } + \mathrm { C O N T }
$$

$$
9 1 . 6 5 \pm \dot { 2 } . 9 8
$$

$$
7 4 . 9 8 \pm 2 \ddot { 4 } . 5 3
$$

$$
0 . 7 1 0 2 \pm 0 . { \dot { 1 } } 7 1 4
$$

$$
\mathrm { T e x t } + \mathrm { D I S C }
$$

$$
7 5 . 8 1 \pm 5 . 8 1
$$

$$
1 0 0 . 4 8 \pm 2 6 . 2 3
$$

$$
0 . 4 5 8 0 \pm 0 . 1 8 5 7
$$

$$
\mathrm { V o i c e } + \mathrm { D I S C }
$$

$$
8 2 . 0 0 \pm 5 . 4 0
$$

$$
1 0 1 . 8 2 \pm 2 5 . 7 9
$$

$$
0 . 4 8 5 6 \pm 0 . 1 5 1 8
$$

$$
\mathrm { V o i c e } + \mathrm { D I S C }
$$

$$
8 7 . 5 3 \pm 5 . 0 2
$$

$$
9 2 . 5 0 \pm 2 2 . 0 1
$$

$$
0 . 5 6 4 3 \pm 0 . 1 7 4 9
$$

$$
\mathrm { V o i c e } + \mathrm { C O N T }
$$

$$
8 6 . 5 5 \pm 7 . 1 8
$$

$$
7 9 . 5 6 \pm 2 8 . 6 9
$$

$$
\mathrm { V o i c e } + \mathrm { C O N T }
$$

$$
8 8 . 0 9 \pm 3 . 5 6
$$

$$
0 . 7 1 2 5 \pm 0 . 1 6 1 5
$$

$$
9 0 . 0 9 \pm 2 6 . 9 4
$$

$$
0 . 6 2 4 2 \pm 0 . 1 4 7 9
$$

$$
S _ { T } + \mathcal { P } _ { d i s c }
$$

$$
\mathrm { T e x t } + \mathrm { D I S C }
$$

$$
S \tau + \mathcal { P } _ { c o n t }
$$

$$
\mathrm { T e x t } + \mathrm { C O N T }
$$

$$
8 5 . 7 5 \pm 5 . 4 1 _ { ( + 1 3 . 1 1 \% ) }
$$

$$
9 7 . 7 3 \pm 2 7 . 3 7 _ { ( \downarrow 2 . 7 4 \% ) }
$$

$$
0 . 5 3 8 3 \pm 0 . 1 8 2 5 _ { ( + 1 7 . 5 3 \% ) }
$$

$$
S _ { \mathcal { V } } + \mathcal { P } _ { d i s c }
$$

$$
7 2 . 5 1 \pm 1 8 . 0 2
$$

$$
\mathrm { V o i c e } + \mathrm { D I S C }
$$

$$
6 8 . 1 2 \pm 2 \dot { 0 } . 6 9
$$

$$
S _ { \mathcal { V } } + \mathcal { P } _ { c o n t }
$$

$$
9 3 . 7 2 \pm 2 . 0 0 _ { ( + 7 . 0 7 \% ) }
$$

$$
\mathrm { V o i c e } + \mathrm { C O N T }
$$

$$
0 . 8 4 0 1 \pm 0 . { \dot { 0 } } 6 5 5
$$

$$
9 5 . 0 1 \pm 2 8 . 4 3 _ { ( \uparrow 2 . 7 1 \% ) }
$$

$$
9 5 . 0 0 \pm 1 . 7 2 _ { ( + 7 . 8 4 \% ) }
$$

$$
5 6 . 9 6 \pm 1 4 . 7 6 _ { ( \downarrow 2 8 . 4 1 \% ) }
$$

$$
0 . 5 7 0 4 \pm 0 . 1 6 8 5 _ { ( + 1 . 0 8 \% ) }
$$

$$
0 . 8 4 2 5 \pm 0 . 0 7 1 1 _ { ( + 1 8 . 2 5 \% ) }
$$

Table 3: Quality evaluation in terms ofMean Opinion Score (MOS), Melody Comparative Mean Opinion Score (MC-MOS) and Word Error Rate (WER) for Chinese (CH) and English (EN).
<table><tr><td>Task</td><td>Model</td><td>Style + Prosody</td><td>MOS ↑</td><td>MC-MOS ↑</td><td>WER-CH↓</td><td>WER-EN↓</td></tr><tr><td rowspan="10">TTS</td><td>CosyVoice [1]</td><td> $\mathrm { T e x t } + \mathrm { D I S C }$ </td><td> $4 . 3 0 \pm 0 . 9 0$ </td><td></td><td> $5 . 9 7 \pm 7 . 5 3$ </td><td> $3 . 1 4 \pm 6 . 1 1$ </td></tr><tr><td>CosyVoice [1]</td><td> $\mathrm { V o i c e } + \mathrm { D I S C }$ </td><td> $4 . 0 5 \pm 1 . 0 7$ </td><td></td><td> $1 1 . 2 7 \pm 1 2 . 4 0$ </td><td> $5 . 7 9 \pm 9 . 8 5$ </td></tr><tr><td>F5-TTS [4]</td><td> $\mathrm { V o i c e } + \mathrm { D I S C }$ </td><td> $4 . 3 5 \pm 0 . 9 6$ </td><td></td><td> $5 . 2 3 \pm 7 . 4 5$ </td><td> $2 . 3 4 \pm 4 . 6 7$ </td></tr><tr><td>ParaStyleTTS [3]</td><td> $\mathrm { T e x t } + \mathrm { D I S C }$ </td><td> $4 . 0 3 \pm 1 . 0 1$ </td><td></td><td> $9 . 6 7 \pm 8 . 2 5$ </td><td> $5 . 2 1 \pm 6 . 9 5$ </td></tr><tr><td>IndexTTS [2]</td><td> $\mathrm { V o i c e } + \mathrm { D I S C }$ </td><td> $4 . 4 2 \pm 0 . 8 0$ </td><td></td><td> $4 . 1 6 \pm 6 . 6 3$ </td><td> $2 . 8 8 \pm 6 . 0 4$ </td></tr><tr><td>Vevo2 [8]</td><td> $\mathrm { V o i c e } + \mathrm { D I S C }$ </td><td> $4 . 3 0 \pm 0 . 7 5$ </td><td></td><td> $6 . 3 5 \pm 6 . 7 3$ </td><td> $3 . 8 2 \pm 6 . 4 7$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CookVoice</td><td> $\mathrm { T e x t } + \mathrm { D I S C }$ </td><td></td><td></td><td></td><td></td></tr><tr><td> $S _ { T } + \mathcal { P } _ { d i s c }$ </td><td> $\mathrm { T e x t } + \mathrm { C O N T }$ </td><td> $3 . 4 5 \pm 1 . 1 4$ </td><td></td><td> $9 . 4 8 \pm 1 1 . 8 8$ </td><td> $6 . 0 7 \pm 1 2 . 3 5$ </td></tr><tr><td> $S \tau + \mathcal { P } _ { c o n t }$   $S _ { \mathcal { V } } + \mathcal { P } _ { d i s c }$ </td><td> $\mathrm { V o i c e } + \mathrm { D I S C }$ </td><td> $3 . 8 0 \pm 1 . 0 8$   $3 . 6 0 \pm 1 . 0 2$ </td><td></td><td> $9 . 0 3 \pm 1 0 . 9 8$ </td><td> $5 . 2 3 \pm 8 . 7 6$ </td></tr><tr><td></td><td> $S _ { \nu } + \mathcal { P } _ { c o n t }$ </td><td> $\mathrm { V o i c e } + \mathrm { C O N T }$ </td><td></td><td></td><td> $8 . 6 6 \pm 1 0 . 1 4$   $8 . 1 3 \pm 1 0 . 0 7$ </td><td> $4 . 4 1 \pm 7 . 4 0$   $4 . 1 9 \pm 7 . 4 5$ </td></tr><tr><td></td><td>Ground Truth</td><td></td><td> $3 . 9 8 \pm 0 . 9 1$ </td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td> $4 . 0 5 \pm 0 . 9 2$ </td><td></td><td> $6 . 3 3 \pm 9 . 3 3$ </td><td> $3 . 3 9 \pm 7 . 0 7$ </td></tr><tr><td rowspan="10"></td><td>DiffSinger [5]</td><td> $\mathrm { T e x t } + \mathrm { D I S C }$ </td><td> $2 . 8 0 \pm 0 . 8 1$ </td><td> $- 0 . 8 0 \pm 1 . 6 6$ </td><td> $1 7 . 7 9 \pm 1 4 . 1 6$ </td><td></td></tr><tr><td>StyleSinger [6]</td><td> $\mathrm { V o i c e } + \mathrm { D I S C }$ </td><td> $3 . 2 0 \pm 1 . 0 8$ </td><td> $- 0 . 4 0 \pm 1 . 6 1$ </td><td> $1 1 . 3 1 \pm 7 . 6 8$ </td><td></td></tr><tr><td>TCSinger [7]</td><td> $\mathrm { V o i c e } + \mathrm { D I S C }$ </td><td> $2 . 4 0 \pm 1 . 0 7$ </td><td> $- 0 . 9 5 \pm 1 . 4 1$ </td><td> $9 . 8 8 \pm 7 . 2 3 $ </td><td> $2 9 . 0 3 \pm 1 9 . 3 1$ </td></tr><tr><td>Vevo1.5 [15]</td><td> $\mathrm { V o i c e } + \mathrm { C O N T }$ </td><td> $3 . 3 3 \pm 1 . 2 9$ </td><td> $- 0 . 7 4 \pm 1 . 6 6$ </td><td> $8 . 8 5 \pm 1 1 . 0 9$ </td><td> $2 3 . 6 2 \pm 1 4 . 6 7$ </td></tr><tr><td>Vevo2 [8]</td><td> $\mathrm { V o i c e } + \mathrm { C O N T }$ </td><td> $3 . 4 2 \pm 1 . 3 0$ </td><td> $0 . 1 1 \pm 1 . 7 6$ </td><td> $4 . 6 6 \pm 6 . 0 4$ </td><td> $1 1 . 3 7 \pm 9 . 8 0$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CookVoice</td><td> $\mathrm { T e x t } + \mathrm { D I S C }$ </td><td> $2 . 4 0 \pm 0 . 8 9$ </td><td> $- 1 . 3 9 \pm 1 . 2 4$ </td><td></td><td></td></tr><tr><td>Sτ + Pdisc</td><td> $\mathrm { T e x t } + \mathrm { C O N T }$ </td><td></td><td></td><td> $1 1 . 7 1 \pm 1 0 . 4 4$ </td><td> $1 9 . 5 6 \pm 1 4 . 4 7$ </td></tr><tr><td> $S \tau + \mathcal { P } _ { c o n t }$ </td><td></td><td> $3 . 2 0 \pm 0 . 9 3$ </td><td> $- 0 . 1 9 \pm 1 . 4 4$ </td><td> $9 . 7 9 \pm 9 . 2 6$ </td><td> $2 1 . 1 0 \pm 1 7 . 8 1$ </td></tr><tr><td> $S _ { \mathcal { V } } + \mathcal { P } _ { d i s c }$   $S _ { \mathcal { V } } + \mathcal { P } _ { c o n t }$ </td><td> $\mathrm { V o i c e } + \mathrm { D I S C }$ </td><td> $2 . 8 0 \pm 1 . 0 0$ </td><td> $- 0 . 9 4 \pm 1 . 4 1$ </td><td> $1 1 . 4 5 \pm 9 . 1 4$ </td><td> $2 2 . 5 3 \pm 1 6 . 9 2$ </td></tr><tr><td colspan="2">Ground Truth</td><td> $\mathrm { V o i c e } + \mathrm { C O N T }$ </td><td> $3 . 4 0 \pm 0 . 9 7$ </td><td> $0 . 2 8 \pm 1 . 4 4$ </td><td> $1 0 . 0 1 \pm 9 . 7 0$ </td><td> $2 4 . 3 7 \pm 1 7 . 2 3$ </td></tr></table>

Table 4: Capability–performance–efficiency comparison table with representative voice generation baselines. The task and control columns summarize the configurations evaluated in this study. T and V denote text- and voice-based style control, while D and C denote discrete and continuous prosody control, respectively. Performance values report the best observed result among the evaluated configurations of each model. S-SIM and F0-CORR are presented as TTS/TTSV.
<table><tr><td></td><td colspan="3">Functional Coverage</td><td colspan="2">Perceptual Quality</td><td colspan="2">Controllability</td><td colspan="3">Computational Efficiency</td></tr><tr><td>Model</td><td>Task</td><td>Style</td><td>Prosody</td><td>TTS MOS</td><td>TTSV MOS / MC-MOS</td><td>S-SIM (%) TTS / TTSV</td><td>F0-CORR TTS / TTSV</td><td>#Para</td><td>CUDA Mem.</td><td>RTF</td></tr><tr><td>CosyVoice [1]</td><td>TTS</td><td>T+V</td><td>D</td><td>4.30</td><td> $- / -$ </td><td>46.67 / –</td><td>0.1902 / -</td><td>436.00M</td><td>2.05G</td><td>3.74</td></tr><tr><td>F5-TTS [4]</td><td>TTS</td><td>V</td><td>D</td><td>4.35</td><td>-1-</td><td>66.29 / -</td><td>0.1505 / –</td><td>300.00M</td><td>2.15G</td><td>6.15</td></tr><tr><td>ParaStyleTTS [3]</td><td>TTS</td><td>T</td><td>D</td><td>4.03</td><td>-1-</td><td>42.96 /-</td><td>0.1777 / -</td><td>52.51M</td><td>1.22G</td><td>0.03</td></tr><tr><td>IndexTTS [2]</td><td>TTS</td><td>V</td><td>D</td><td>4.42</td><td>-1-</td><td>71.19 /-</td><td>0.2548 / -</td><td>608.00M</td><td>1.53G</td><td>0.36</td></tr><tr><td>Vevo2 [8]</td><td>TTS+TTSV</td><td>V</td><td>D+C</td><td>4.30</td><td>3.42 / 0.11</td><td>75.11 / 88.09</td><td>0.1242 / 0.6242</td><td>872.00M</td><td>6.84G</td><td>14.85</td></tr><tr><td>DiffSinger [5]</td><td>TTSV</td><td>T</td><td>D</td><td></td><td>2.80 / -0.80</td><td>-/75.81</td><td>-/0.4580</td><td>26.74M</td><td>0.60G</td><td>0.34</td></tr><tr><td>StyleSinger [6]</td><td>TTSV</td><td>V</td><td>D</td><td></td><td>3.20 / -0.40</td><td>-/82.00</td><td>-/0.4856</td><td>42.00M</td><td>0.88G</td><td>0.36</td></tr><tr><td>TCSinger [7]</td><td>TTSV</td><td>V</td><td>D</td><td></td><td>2.40 / -0.95</td><td>-/87.53</td><td>-/0.5643</td><td>329.50M</td><td>1.97G</td><td>0.33</td></tr><tr><td>Vevo1.5 [15]</td><td>TTSV</td><td>V</td><td>C</td><td></td><td>3.33 / -0.74</td><td>-/86.55</td><td>-/0.7125</td><td>1468.43M</td><td>9.44G</td><td>9.83</td></tr><tr><td>Vevo2 [8]</td><td>TTS+TTSV</td><td>V</td><td>D+C</td><td>4.30</td><td>3.42 / 0.11</td><td>75.11 / 88.09</td><td>0.1242 / 0.6242</td><td>872.00M</td><td>6.84G</td><td>14.85</td></tr><tr><td>CookVoice</td><td>TTS+TTSV</td><td>T+V</td><td>D+C</td><td>3.98</td><td>3.40 / 0.28</td><td>91.65 / 95.00</td><td>0.7102 / 0.8425</td><td>43.51M</td><td>1.37G</td><td>0.04</td></tr></table>

singing voice generation. One possible reason for the generally higher WERs in TTSV is that Whisper is primarily trained on large-scale speech data. Singing voices exhibit different prosodic patterns, including wider pitch variations and prolonged phonemes, which may increase recognition errors. CookVoice achieves comparable performance with existing baseline who is trained on substantially larger datasets. Its primary advantages lie in its stronger controllability and flexible support for different style and prosody control signals.

## 5.2. RQ2: Controllability Analysis

As shown in Table 2, existing baselines typically support only a limited combination of style and prosody conditions. In contrast, CookVoice accommodates both text- and voice-based style control together with discrete prosody and continuous $F _ { 0 }$ control. These control signals are explicitly aligned with the acoustic representation at the frame level, enabling finergrained guidance throughout the generation process.

Compared with baselines under matched style–prosody settings, CookVoice delivers clear improvements. For TTS with text-based style and discrete prosody control, CookVoice improves S-SIM by 41.48% and F0-CORR by 121.10%, while reducing F0-RMSE by 5.53%. With voice-based style and discrete prosody control, CookVoice improves S-SIM by 16.46% and F0-CORR by 109.18%, while reducing F0-RMSE by 19.89%. These results suggest that CookVoice benefits from its explicit frame-level conditioning design. Outperforming existing voice cloning methods that rely primarily on autoregressive token prediction or voice-level style conditioning.

A similar trend is observed for TTSV. Under text-based style and discrete note control, CookVoice improves SSIM by 13.11% and F0-CORR by 17.53%, while reducing F0-RMSE by 2.74%. Under voice-based style and continuous $F _ { 0 }$ control, it improves S-SIM by 7.84% and F0-CORR by 18.25%, while reducing F0-RMSE by 28.41%. The higher level of controllability of CookVoice can be attributed to the joint effect of its multimodal control interfaces and fine-grained frame-level conditioning. Together, these designs enable a broader range of controllable generation functions while improving both style preservation and prosody-following consistently across speech and singing voice generation.

![](images/f0d8d2ef10c05ae9fd47cc4d495281913d5e05e6fc6771dbdf7d0b0c799e1c18.jpg)  
(a) TTS

![](images/5c21fa37093c05e8f23943076d222374374100fd61df2bc0ad2331a78bde7610.jpg)  
(b) TTSV  
Figure 2: Marginal effects ofdifferent style and prosody control signals on CookVoice’s performance. Redder indicate stronger performance, bluer indicate weaker performance.

## 5.3. RQ3: Effects of Different Control Signals

In this section, we analyze how different control signals affect the performance of CookVoice. Figure 2 summarizes the marginal effects of different style and prosody control signals. Each value represents the average normalized performance of one control source across its matched performance, a higher value indicates better performance. The results reveal effects from the style source and the prosody source.

Effect of the style control source. For TTS in Figure 2a, voicebased style conditioning consistently outperforms text-based conditioning across all evaluated metrics. The largest difference is observed in SSIM, where Voice achieves a normalized score of 0.932, compared with 0.199 for Text. Voice conditioning also produces better prosody consistency, perceptual quality, and intelligibility, achieving substantially higher scores for F0-CORR, MOS, and both Chinese and English WER. Voices provide richer acoustic information than natural language style prompts. It allows the model to preserve paralinguistic details such as timbre, emotion and age from the reference voice more accurately.

As shown in figure 2b. Voice conditioning achieves the highest SSIM and improves both MOS and MC-MOS, confirming its advantage for accurate singing-style transfer. However, its effect on intelligibility is less consistent. Voice and Text have similar Chinese WER performance, while Text achieves a substantially better English WER score. Although voice references improve style preservation and perceptual quality, stronger acoustic style imitation does not necessarily improve lyric intelligibility, particularly for English singing.

Effect of the prosody control source. Continuous prosody signals $F _ { 0 }$ provides a clear advantage over discrete prosody signals in prosody-related metrics. For TTS, CONT achieves scores of 0.844 for F0-RMSE and 0.892 for F0-CORR, compared with 0.230 and 0.221 for DISC. It also improves MOS and WER, indicating that detailed frame-level pitch guidance benefits not only prosody following but also overall speech quality and intelligibility.

The advantage of continuous control is even more pronounced for TTSV. CONT substantially outperforms DISC in F0-RMSE, F0-CORR, MOS, MC-MOS, and Chinese WER, confirming that explicit frame-level $F _ { 0 }$ trajectories are effective for melody following and perceptual singing quality. However, unlike in TTS, DISC achieves higher S-SIM and better English WER performance in TTSV. One possible explanation is that the discrete prosody signal in TTSV consists of musical note, which already provide a structured representation of pitch, duration, and note boundaries that is closely aligned with the lyrics. This flexibility can help preserve the style and maintain clearer lyric articulation. Conversely, continuous signals contain detailed prosody trajectory from the reference voice. Which leads to an observable style-sim degradation under text-based style conditioning, where replacing discrete notes with continuous $F _ { 0 }$ reduces SSIM from 85.75% to 72.51%.

## 5.4. RQ4: Effect of ODE Steps

We further analyze how performance is affected by the number of ODE solver steps, with detailed results provided in $\mathsf { A p - }$ pendix C. Increasing the number of ODE steps generally improves style similarity and prosody fidelity, particularly under voice-based style conditioning and continuous prosody control.

Due to the high cost of collecting MOS and MC-MOS ratings, we evaluate these subjective metrics at coarser intervals, increasing the number of ODE steps by a factor of four, while the objective metrics are evaluated at each doubling of the step count. MOS continues to improve up to 16 steps before declining, whereas MC-MOS reaches its highest value at 4 steps and gradually decreases thereafter. Meanwhile, style similarity and $F _ { 0 }$ -based metrics generally stabilize between 4 and 8 steps, and intelligibility metrics such as WER, PhoER, and ProER achieve their best or near-best results at approximately 4 steps and may slightly degrade with further refinement. These results indicate that additional ODE steps can improve acoustic naturalness and prosodic details, but do not necessarily benefit melody consistency or linguistic content preservation.

Based on these observations, we suggest 4–8 ODE steps as the optimal operating range for CookVoice. Within this range, the generated voices maintain stable controllability, good perceptual quality, and intelligibility without incurring additional inference steps.

## 5.5. RQ5: Efficiency Analysis

Table 4 provides an integrated comparison of functional coverage, generation performance, controllability, and computational efficiency. The results reveal that existing systems typically optimize only a subset of these dimensions. Large-scale TTS models such as F5-TTS and IndexTTS achieve higher speech MOS scores than CookVoice, but are restricted to speech generation and require approximately 6.9× and 14.0× more parameters, respectively. Similarly, lightweight task-specific models such as ParaStyleTTS, DiffSinger, and StyleSinger provide competitive efficiency, but support only speech or singing voice generation and a limited set of control signals.

Vevo2 is the most directly comparable baseline because it supports both speech and singing voice generation. Vevo2 achieves higher TTS MOS and a marginally higher TTSV MOS than CookVoice. However, CookVoice obtains a higher MC-MOS and substantially stronger style and prosody controllability across both tasks. More importantly, CookVoice uses only 4.99% of the parameters, 20.03% of the CUDA memory, and 0.27% of the RTF required by Vevo2.

These comparisons indicate that the main advantage of CookVoice does not lie in maximizing a single task-specific quality metric. Instead, CookVoice provides a favorable tradeoff among functional coverage, fine-grained controllability, perceptual quality, and computational efficiency. With a single 43.51M-parameter model, it supports both TTS and TTSV, textand voice-based style conditioning, and discrete and continuous prosody control, while maintaining real-time generation with an RTF of 0.04.

The compact model size and fast inference of CookVoice mainly stem from its streamlined architecture. By replacing autoregressive token prediction with a lightweight DiT-S generative backbone, CookVoice limits the model size to 43.51M parameters and avoids the sequential next-token decoding required by AR models. In addition, CookVoice employs flow matching, which requires only a small number of ODE solver steps, compared with conventional diffusion sampling processes that typically involve hundreds of iterative denoising steps [30]. These design choices reduce both model complexity and inference cost.

## 6. Conclusion

In this paper, we present CookVoice, a unified framework for multimodal, multi-style, and multi-task human voice generation. By decomposing human voice into content, prosody, and style, CookVoice supports a wide range of speech and singing voice generation tasks within a single model, including TTS, TTSV, style-controllable generation, voice mimicry, and voice conversion. To enable fine-grained controllability, we design a flexible alignment strategy that expands different control signals to the level of the acoustic frame, allowing text, reference voice, discrete prosody, and continuous $F _ { 0 }$ contours to be combined in a unified generation process. Experimental results show that CookVoice achieves comparable generation quality to existing task-specific baselines while providing stronger style and prosody controllability, smaller model size and efficient inference speed.

## 7. Limitation

CookVoice shows potential for unified and controllable human voice generation, several limitations remain. First, due to resource constraints, CookVoice has not yet been scaled up. The current model only use the DiT-S version of diffusion transformer (43.51M parameters) and is trained on 168 hours of data, which is still much smaller than large-scale voice generation systems. The scaling potential of CookVoice has not been fully explored. Another limitation lies in CookVoice mainly focuses on human voice generation. Its potential applicability to broader audio generation domains, such as music, instrumental sound, and general audio generation, has not yet been investigated.

## 8. References

[1] Z. Du, Q. Chen, S. Zhang, K. Hu, H. Lu, Y. Yang, H. Hu, S. Zheng, Y. Gu, Z. Ma et al., “Cosyvoice: A scalable multilingual zero-shot text-to-speech synthesizer based on supervised semantic tokens,” arXiv preprint arXiv:2407.05407, 2024.

[2] W. Deng, S. Zhou, J. Shu, J. Wang, and L. Wang, “Indextts: An industrial-level controllable and efficient zero-shot text-to-speech system,” arXiv preprint arXiv:2502.05512, 2025.

[3] H. Lou, H.-Y. Paik, W. Hu, and L. Yao, “Parastyletts: Toward efficient and robust paralinguistic style control for expressive text-tospeech generation,” in Proceedings ofthe 34th ACM International Conference on Information and Knowledge Management, 2025, pp. 1979–1988.

[4] Y. Chen, Z. Niu, Z. Ma, K. Deng, C. Wang, J. JianZhao, K. Yu, and X. Chen, “F5-tts: A fairytaler that fakes fluent and faithful speech with flow matching,” in Proceedings of the 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), 2025, pp. 6255–6271.

[5] J. Liu, C. Li, Y. Ren, F. Chen, and Z. Zhao, “Diffsinger: Singing voice synthesis via shallow diffusion mechanism,” in Proceedings of the AAAI conference on artificial intelligence, vol. 36, no. 10, 2022, pp. 11 020–11 028.

[6] Y. Zhang, R. Huang, R. Li, J. He, Y. Xia, F. Chen, X. Duan, B. Huai, and Z. Zhao, “Stylesinger: Style transfer for out-ofdomain singing voice synthesis,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 17, 2024, pp. 19 597–19 605.

[7] Y. Zhang, Z. Jiang, R. Li, C. Pan, J. He, R. Huang, C. Wang, and Z. Zhao, “Tcsinger: Zero-shot singing voice synthesis with style transfer and multi-level style control,” in Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, 2024, pp. 1960–1975.

[8] X. Zhang, J. Zhang, Y. Wang, C. Wang, Y. Chen, D. Jia, Z. Chen, and Z. Wu, “Vevo2: A unified and controllable framework for speech and singing voice generation,” IEEE Transactions on Audio, Speech and Language Processing, 2026.

[9] J. Kong, J. Kim, and J. Bae, “Hifi-gan: Generative adversarial networks for efficient and high fidelity speech synthesis,” Advances in neural information processing systems, vol. 33, pp. 17 022– 17 033, 2020.

[10] H. Lou, H. young Paik, W. Hu, and L. Yao, “Parameta: Towards learning disentangled paralinguistic speaking styles representations from speech,” in AAAI Conference on Artificial Intelligence, 2026. [Online]. Available: https://api.semanticscholar.org/CorpusID:284910281

[11] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in Proceedings ofthe IEEE/CVF international conference on computer vision, 2023, pp. 4195–4205.

[12] Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” arXiv preprint arXiv:2210.02747, 2022.

[13] Y. Ren, Y. Ruan, X. Tan, T. Qin, S. Zhao, Z. Zhao, and T.-Y. Liu, “Fastspeech: Fast, robust and controllable text to speech,” Advances in neural information processing systems, vol. 32, 2019.

[14] L. Zhang, R. Li, S. Wang, L. Deng, J. Liu, Y. Ren, J. He, R. Huang, J. Zhu, X. Chen et al., “M4singer: A multi-style, multisinger and musical score provided mandarin singing corpus,” Advances in Neural Information Processing Systems, vol. 35, pp. 6914–6926, 2022.

[15] X. Zhang, X. Zhang, K. Peng, Z. Tang, V. Manohar, Y. Liu, J. Hwang, D. Li, Y. Wang, J. Chan, Y. Huang, Z. Wu, and M. Ma, “Vevo: Controllable zero-shot voice imitation with selfsupervised disentanglement,” in ICLR. OpenReview.net, 2025.

[16] J. Kim, J. Kong, and J. Son, “Conditional variational autoencoder with adversarial learning for end-to-end text-to-speech,” in International conference on machine learning. PMLR, 2021, pp. 5530–5540.

[17] K. Song, X. Tan, T. Qin, J. Lu, and T.-Y. Liu, “Mpnet: Masked and permuted pre-training for language understanding,” Advances in neural information processing systems, vol. 33, pp. 16 857– 16 867, 2020.

[18] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in neural information processing systems, vol. 30, 2017.

[19] H. Lou, H.-y. Paik, S. Li, W. Hu, and L. Yao, “Generalized multilingual text-to-speech generation with language-aware style adap tation,” arXiv preprint arXiv:2504.08274, 2025.

[20] Y. Wang, R. Skerry-Ryan, D. Stanton, Y. Wu, R. J. Weiss, N. Jaitly, Z. Yang, Y. Xiao, Z. Chen, S. Bengio et al., “Tacotron: Towards end-to-end speech synthesis,” arXiv preprint arXiv:1703.10135, 2017.

[21] Databaker, “Chinese mandarin female corpus,” https://en. data-baker.com/datasets/freeDatasets/, 2020, accessed: 2023-04- 20.

[22] K. Ito and L. Johnson, “The lj speech dataset,” https://keithito. com/LJ-Speech-Dataset/, 2017.

[23] K. Zhou, B. Sisman, R. Liu, and H. Li, “Seen and unseen emotional style transfer for voice conversion with a new emotional speech dataset,” in ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2021, pp. 920–924.

[24] H. Cao, D. G. Cooper, M. K. Keutmann, R. C. Gur, A. Nenkova, and R. Verma, “Crema-d: Crowd-sourced emotional multimodal actors dataset,” IEEE transactions on affective computing, vol. 5, no. 4, pp. 377–390, 2014.

[25] P. Klumpp, T. Arias, P. A. Perez-Toro, E. Noeth, and J. Orozco-´ Arroyave, “Common phone: A multilingual dataset for robust acoustic modelling,” in Proceedings of the Thirteenth Language Resources and Evaluation Conference, 2022, pp. 763–768.

[26] Simon3000, “Genshin voice: A multi-lingual voice dataset from Genshin Impact,” https://huggingface.co/datasets/simon3000/ genshin-voice, 2025, hugging Face Datasets.

[27] Y. Zhang, C. Pan, W. Guo, R. Li, Z. Zhu, J. Wang, W. Xu, J. Lu, Z. Hong, C. Wang et al., “Gtsinger: A global multi-technique singing corpus with realistic music scores for all singing tasks,” Advances in Neural Information Processing Systems, vol. 37, pp. 1117–1140, 2024.

[28] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in International conference on machine learning. PMLR, 2023, pp. 28 492–28 518.

[29] H. Lou, J. Wu, C. Huang, T. Yu, H.-y. Paik, W. Hu, and L. Yao, “Autosift: Automatic style sifting for controllable speech generation with arbitrary style infilling,” arXiv preprint arXiv:2607.12706, 2026.

[30] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” Advances in neural information processing systems, vol. 33, pp. 6840–6851, 2020.

## A. Related Work

Autoregressive voice generation. Recent advances in neural voice generation have significantly improved the naturalness and expressiveness of synthesized speech. Representative systems such as CosyVoice [1] and IndexTTS [2] adopt autoregressive (AR) generation mechanisms to model acoustic token sequences. Using token-by-token prediction and large-scale training, these models can generate expressive high-quality speech. However, AR decoding also introduces several limitations for controllability. Since duration, rhythm, and prosody are implicitly determined by the generated token sequence, the generation process is difficult to interpret and hard to control at a fine temporal granularity. This becomes especially problematic when the target task requires explicit duration or pitch control, such as singing voice generation, prosody mimicry, and frame-level voice editing. In addition, AR generation usually suffers from slower inference due to its sequential decoding process.

NAR and controllable speech generation. NAR methods improve inference efficiency and temporal stability by introducing explicit or semi-explicit alignment mechanisms. For example, FastSpeech-style models rely on duration prediction and phoneme-level expansion to generate speech in parallel, while recent flow-matching or diffusion-based systems further improve acoustic quality through continuous generative modeling [4]. These methods provide better control over phonemelevel timing than AR models. However, most existing NAR speech generation systems are still designed for specific tasks, such as TTS, expressive TTS, or voice cloning. Their control interfaces are often limited to either text prompts, speaker references, or phoneme-level durations, making it difficult to flexibly combine content, style, prosody, and singing-related controls under a unified framework.

Singing voice generation. Singing voice generation requires more explicit temporal and melodic control than ordinary speech synthesis. Traditional singing voice synthesis systems, such as DiffSinger [5], usually rely on musical scores and note-level conditions to generate singing voices. More recent work introduces style control into singing voice generation. For example, StyleSinger [6] improves controllability by incorporating singing style information. However, such systems often assume a predefined relationship between lyrics, phonemes, and notes. In particular, binding notes directly with phonemes simplifies alignment, but it may not always reflect the flexible temporal structure of real singing, where one phoneme may span multiple notes or one note may correspond to multiple phonetic units. This type of restricted alignment improves tractability but limits flexible control in more general singing voice generation scenarios.

Unified voice generation. Recent unified voice generation models attempt to cover multiple voice generation tasks using a single model. For example, Vevo-style systems [15, 8] move toward unified speech and singing voice modeling. However, many of these systems still rely on AR decoding or use global reference conditions, such as utterance-level or sentence-level prosody references. While these designs are effective for general voice generation, they provide limited direct control over fine-grained temporal details. In particular, when content, style, notes, lexical tones, stresses, and continuous $F _ { 0 }$ contours are jointly involved, existing unified models still struggle to provide flexible frame-level controllability.

## B. Evaluation Metrics

This section provides detailed definitions of the evaluation metrics used in our experiments.

Word Error Rate (WER). WER measures word-level transcription errors between the generated audio and the groundtruth text. We first transcribe the generated audio using Whisper [28], and then compute the edit distance between the predicted word sequence and the reference word sequence:

$$
\mathrm { W E R } = { \frac { S + D + I } { N } } ,\tag{6}
$$

where $S , D ,$ and I denote the number of substitutions, deletions, and insertions, respectively, and N is the number of words in the reference text. Lower WER has better intelligibility.

Phoneme Error Rate (PhoER). PhoER measures content accuracy at the phoneme level. We convert both the recognized text and the ground-truth text into phoneme sequences using the same G2P module, and then compute the normalized edit distance:

$$
\mathrm { P h o E R } = \frac { S _ { p } + D _ { p } + I _ { p } } { N _ { p } } ,\tag{7}
$$

where $S _ { p } , D _ { p } ,$ , and $I _ { p }$ denote phoneme-level substitutions, deletions, and insertions, and $N _ { p }$ is the number of phonemes in the reference sequence. A lower PhoER indicates more accurate pronunciation and content preservation.

Prosody Error Rate (ProER). ProER measures the error rate of discrete prosody tokens, such as Mandarin tones, English stresses, or note-related prosody labels. Similar to PhoER, we compute the normalized edit distance between the predicted and reference prosody-token sequences:

$$
\mathrm { P r o E R } = \frac { S _ { r } + D _ { r } + I _ { r } } { N _ { r } } ,\tag{8}
$$

where $S _ { r } , D _ { r }$ , and $I _ { r }$ denote substitution, deletion, and insertion errors in the prosody-token sequence, and $N _ { \tau }$ is the number of reference prosody tokens. A lower ProER indicates better preservation of discrete prosodic information.

Style Similarity (S-SIM). S-SIM evaluates whether the generated voice matches the target style. We extract style embeddings from the generated audio and the target reference using a pretrained style encoder. The cosine similarity between the two embeddings is computed as:

$$
\begin{array} { r } { \mathrm { S - S I M } = \frac { \mathbf { s } _ { g e n } ^ { \top } \mathbf { s } _ { r e f } } { \left\| \mathbf { s } _ { g e n } \right\| _ { 2 } \left\| \mathbf { s } _ { r e f } \right\| _ { 2 } } , } \end{array}\tag{9}
$$

where ${ \bf s } _ { g e n }$ and ${ \bf s } _ { r e f }$ are the style embeddings of the generated and reference voices, respectively. A higher S-SIM indicates better style preservation.

F0 Root Mean Square Error (F0-RMSE). F0-RMSE measures the absolute pitch difference between the generated and target F0 contours. We compute F0-RMSE over voiced frames where both generated and reference F0 values are valid:

$$
\mathrm { F 0 - R M S E } = \sqrt { \frac { 1 } { T _ { v } } \sum _ { t = 1 } ^ { T _ { v } } \left( f _ { t } ^ { g e n } - f _ { t } ^ { r e f } \right) ^ { 2 } } ,\tag{10}
$$

where $f _ { t } ^ { g e n }$ and $f _ { t } ^ { r e f }$ are the generated and reference F0 values at frame t, and $T _ { v }$ is the number of valid voiced frames. A lower F0-RMSE indicates smaller absolute pitch deviation.

F0 Pearson Correlation (F0-CORR). F0-CORR measures whether the generated pitch contour follows the same trend as the reference contour:

$$
\mathrm { F 0 - C O R R } = \frac { \sum _ { t = 1 } ^ { T _ { v } } \big ( f _ { t } ^ { g e n } - \bar { f } ^ { g e n } \big ) \big ( f _ { t } ^ { r e f } - \bar { f } ^ { r e f } \big ) } { \sqrt { \sum _ { t = 1 } ^ { T _ { v } } \big ( f _ { t } ^ { g e n } - \bar { f } ^ { g e n } \big ) ^ { 2 } } \sqrt { \sum _ { t = 1 } ^ { T _ { v } } \big ( f _ { t } ^ { r e f } - \bar { f } ^ { r e f } \big ) ^ { 2 } } } ,\tag{11}
$$

where $\bar { f } ^ { g e n }$ and $\bar { f } ^ { r e f }$ are the mean F0 values of the generated and reference contours. A higher F0-CORR indicates better preservation of pitch movement and prosodic trend.

Mean Opinion Score (MOS). MOS is used to evaluate perceptual naturalness and overall audio quality. Human listeners rate each generated sample on a scale from 1 to 5, where 1 indicates poor quality and 5 indicates excellent quality. The final MOS is the average score across all listeners and samples.

Melody Comparative MOS (M-CMOS). M-CMOS is used for singing voice evaluation. Listeners compare generated singing samples and judge which one better follows the target melody. This metric evaluates melody consistency from human perception, complementing objective F0-based metrics.

## C. Signal Impact Analysis

We analyze the impact of the number of ODE solver steps under different sources of style and prosody control. Specifically, we compare text-based and voice-based style control, as well as discrete and continuous prosody control, on both TTS and TTSV tasks. We evaluate three aspects of generation quality: style similarity, prosody fidelity, and intelligibility.

## C.1. Style similarity

As shown in Fig. 3, style similarity generally increases as the number of ODE steps increases, and gradually converges around 4–8 steps. This suggests that most style-related information is formed in the early part of the generation trajectory, while further increasing the number of steps brings diminishing returns. We also observe that reference-voice-based style control consistently outperforms text-based style control. This is expected because reference voices contain richer speaker identity and paralinguistic information than text style descriptions. In addition, continuous prosody control generally achieves higher style similarity than discrete prosody tokens, indicating that dense frame-level control signals provide more detailed guidance during generation.

## C.2. Prosody fidelity

For prosody, we evaluate F0-RMSE and F0-Correlation, as shown in Fig. 4. F0-RMSE measures the absolute pitch deviation from the ground-truth contour, while F0-Correlation measures whether the generated pitch follows the overall trend of the target F0 trajectory. For F0-RMSE, the error generally decreases as the number of ODE steps increases, with voice-based style and continuous prosody control achieving better performance. This is also reasonable because the reference voice directly provides acoustic and pitch-related cues, while continuous F0 signals preserve more fine-grained pitch information than discrete tokens.

A similar trend can be observed for F0-Correlation, where the correlation improves with more ODE steps and converges around 8 steps. However, the improvement for discrete prosody control is relatively limited, especially in the TTSV setting. One possible reason is that singing voice often follows a more constrained and stable melodic structure compared with speech. Therefore, discrete prosody tokens may already capture the coarse pitch movement, and increasing the number of ODE steps mainly improves acoustic details rather than substantially changing the global F0 correlation.

## C.3. Intelligibility

The intelligibility results are shown in Fig. 5. Different from style similarity and prosody fidelity, WER, PhoER, and ProER first decrease and reach their best performance at 4 ODE steps, but tend to increase when more ODE steps are used. This indicates that using too many ODE steps may over-refine acoustic, style, or prosodic details at the cost of linguistic clarity. In other words, a larger number of ODE steps does not always lead to better content preservation.

We also observe that TTSV generally has higher error rates than TTS. This may be partly attributed to the evaluation model, since Whisper [28] is mainly trained and optimized for speech recognition rather than singing voice recognition. Moreover, singing voice naturally contains longer phoneme durations, stronger pitch variation, and melody-driven pronunciation changes, which can make recognition more difficult.

Overall, these results suggest that different generation aspects prefer different numbers of ODE steps. Style similarity and prosody fidelity benefit from increasing ODE steps and usually converge around 4–8 steps, while intelligibility achieves the best trade-off at around 4 steps. Therefore, a small number of ODE steps provides a better balance between controllability, acoustic quality, and content preservation.

![](images/303a59e80f48ab958b1aaf359540b65aa6be0ba330096555d99a8235161088d6.jpg)  
(a) TTS

![](images/29ab550022d44e549f9a8ecec0900113858ddb6920f737aefe26850a29d96eb1.jpg)  
(b) TTSV  
Figure 3: Style similarity under different numbers of ODE steps for TTS and TTSV.

![](images/5c3fa76cce3bdccab8f26f9750f3e853401b7771bca3fc829f7b46eed11e79ce.jpg)  
(a) F0-RMSE on TTS

![](images/3077a5fbd739a2701edba8834a98ddef9c92e01eaa0d3e74d73b115318d0471a.jpg)  
(b) F0-RMSE on TTSV

![](images/af1e3fa4e477d4abd59c5f930a08eea727dacd1ca6f3fb804ec3b3653e42dea5.jpg)

![](images/aa4255a0779546a8eae7dbccdc6132f73dcd36c068793f72b359ca2adcdaf804.jpg)  
(c) F0-Correlation on TTS  
(d) F0-Correlation on TTSV  
Figure 4: Prosody fidelity under different numbers of ODE steps. F0-RMSE measures absolute pitch deviation, while F0- Correlation measures the consistency of the generated F0 contour with the target contour.

## D. Task Definition

CookVoice is designed as a unified framework for speech and singing voice generation. Instead of building separate models for different tasks, we formulate a broad range of generation, conversion, editing, and mimicry tasks under a shared conditional generation function G(·). The key difference between tasks lies in the combination of control signals, including linguistic content, style conditions, lexical or musical prosody, and frame-level pitch contours.

![](images/bb89700ca986da1052bf0d52c8f41c380393ef481e4b25cde3ff32371daffe6b.jpg)  
(a) WER on TTS

![](images/9c5211a0bb6422992052fd15eaa414aeb8edcd7908a6c6b6cf3c4e94c431ee3a.jpg)  
(b) WER on TTSV

![](images/61bebec17ec622bb37a06f3887d86504a47f9bacb42fa054af93d8e78c600f98.jpg)  
(c) PhoER on TTS

![](images/69da6e0eb872e3c70f636dd0489df81cd2c79ee7590149a2d3c6f873c5eb1970.jpg)  
(d) PhoER on TTSV

![](images/a70fcddb3966e755758671981b766674a064782c2e2940c89760b727cfc3a69c.jpg)  
(e) ProER on TTS

![](images/37505f9c92921695cff81de55bfb45b4973d4af2180d01d55d479af79adb4260.jpg)  
(f) ProER on TTSV  
Figure 5: Intelligibility under different numbers of ODE steps for TTS and TTSV. WER, PhoER, and ProER are used to evaluate word-level, phoneme-level, and prosody-token-level error rates, respectively.

![](images/85735f2b088fe435a46526449415d6878c75078b2f7640632dc92f3f35fd6a1d.jpg)  
(a) MOS on TTS

![](images/3b7a7e4f5089ee067d9d334eee5a83ff4660d0b80e00da21102fca980ca281e8.jpg)  
(b) MOS on TTSV

![](images/c5b466df1b608d4d3e318e43737399e5f86385c2bd85611c38c2dbacf1a7c9e2.jpg)  
(c) MC-MOS on TTSV  
Figure 6: Human subjective listening test evaluation under different numbers of ODE steps. MOS measures overall naturalness, MC-MOS measures the melody accuracy ofthe generated singing voice.

Specifically, X represents the phoneme derived from text or lyric to be generated, while $S _ { T }$ and $S _ { \nu }$ denote style information provided by text descriptions and reference voices, respectively. Prosodic control can be provided at different granularities: $\mathcal { P } _ { l e x }$ represents discrete lexical prosody such as tones or stresses, $\mathcal { P } _ { n o t e }$ represents musical note-level conditions for singing voice generation, and $F _ { 0 }$ provides continuous frame-level pitch control. By flexibly combining these conditions, CookVoice can support multiple tasks within a single model, as summarized in Table 5.

## E. Terminology

To make the paper accessible to readers without a background in linguistics, speech processing, or music, we summarize the key terms used in CookVoice in Table 6. These terms cover linguistic content, prosody, music-related control signals, style control, and evaluation metrics.

## F. Supplementary Results

Table 5: Summary of generation tasks and control settings supported by CookVoice. X denotes phoneme/lyric content, $S \tau$ and $S _ { \nu }$ denote text-based and voice-based style conditions, $\mathcal { P } _ { l e x }$ denotes lexical prosody such as tones or stresses, $\mathcal { P } _ { n o t e }$ denotes musical note conditions, and $F _ { 0 }$ denotes F0 contour.
<table><tr><td>Task Name</td><td>Abbr.</td><td>Formulation</td><td>Description</td></tr><tr><td>Text-to-Speech</td><td>TTS</td><td> $\mathcal { G } ( X , \mathcal { P } _ { l e x } )$ </td><td>Generate a speech waveform from text content and lexical prosody.</td></tr><tr><td>Text Style-Controllable TTS</td><td>T-SC-TTS</td><td> $\mathcal { G } ( S _ { T } , X , \mathcal { P } _ { l e x } )$ </td><td>Generate speech conditioned on a natural language style description, such as emotion, age, gender, or speaking</td></tr><tr><td>Voice Style-Controllable TTS</td><td>V-SC-TTS</td><td> $\mathcal { G } ( S \nu , X , \mathcal { P } _ { l e x } )$ </td><td>manner. Generate speech that preserves the text content while mimicking the global style or timbre of a reference voice.</td></tr><tr><td>Prosody-Controllable TTS</td><td>PC-TTS</td><td> ${ \mathcal { G } } ( S , X , F _ { 0 } )$ </td><td>Generate speech with explicit frame-level prosody control from a reference  $F _ { 0 }$  contour while keeping the target con- tent and style.</td></tr><tr><td>Text-to-Singing Voice</td><td>TTSV</td><td> $\mathcal { G } ( X , \mathcal { P } _ { n o t e } )$ </td><td>Generate a singing voice from lyrics and musical notes.</td></tr><tr><td>Text Style-Controllable TTSV</td><td>T-SC- TTSV</td><td> $\mathcal { G } ( S \tau , X , \mathcal { P } _ { n o t e } )$ </td><td>Generate singing voice from lyrics and notes while con- trolling singing style using a text prompt.</td></tr><tr><td>Voice Style-Controllable TTSV</td><td>V-SC- TTSV</td><td> $\mathcal { G } ( S \nu , X , \mathcal { P } _ { n o t e } )$ </td><td>Generate singing voice from lyrics and notes while trans- ferring the style or timbre from a reference voice.</td></tr><tr><td>Prosody-Controllable TTSV</td><td>PC-TTSV</td><td> $\mathcal { G } ( S , X , \mathcal { P } _ { F _ { 0 } } )$ </td><td>Generate singing voice by following a continuous frame- level F0 contour, enabling fine-grained melody or prosody mimicry.</td></tr><tr><td>Voice Mimicry</td><td>VM</td><td> $\mathcal { G } ( S \nu , X , \mathcal { P } _ { F _ { 0 } } )$ </td><td>Generate speech with new content while jointly mimick- ing the reference voice style and detailed prosodic trajec-</td></tr><tr><td>Voice Conversion</td><td>VC</td><td> $\mathcal { G } ( S _ { \nu } ^ { t a r } , X ^ { s r c } , \mathcal { P } _ { F _ { 0 } } ^ { s r c } )$ </td><td>tory. Convert a source speech into a target voice style while pre- serving the source linguistic content and prosody.</td></tr><tr><td>Singing Voice Conversion</td><td>SVC</td><td> $\mathcal { G } ( S _ { \nu } ^ { t a r } , X ^ { s r c } , \mathcal { P } _ { F _ { 0 } } ^ { s r c } )$ </td><td>Convert a source singing voice into a target singing style while preserving lyrics and melodic contour.</td></tr><tr><td>Speech Editing</td><td>SE</td><td> $\mathcal { G } ( S , X ^ { e d i t } , \mathcal { P } _ { F _ { 0 } } )$ </td><td>Edit the linguistic content or style of speech while main-</td></tr><tr><td>Singing Voice Editing</td><td>SVE</td><td> $\mathcal { G } ( S , X ^ { e d i t } , \mathcal { P } _ { n o t e / F _ { 0 } } )$ </td><td>taining the remaining prosodic structure. Edit lyrics, style, or melody-related controls in singing</td></tr><tr><td>Sketch-to-Voice</td><td>STV</td><td> $\mathcal { G } ( S , X , \mathcal { P } _ { F _ { 0 } } ^ { s k e t c h } )$ </td><td>voice generation. Generate speech or singing voice from a user-specified</td></tr><tr><td>Humming-To-Voice</td><td>HTV</td><td> $\mathcal { G } ( S , X , \mathcal { P } _ { F _ { 0 } } ^ { h u m m } )$ </td><td>coarse pitch sketch or manually controlled  $F _ { 0 }$  trajectory. Generate speech or singing voice from a user-specified humming  $\bar { F } _ { 0 }$  contour.</td></tr></table>

Table 6: Terminology used in CookVoice. The table provides simplified explanations of linguistic, musical, control, and evaluation terms for non-expert readers.
<table><tr><td>Term</td><td>Category</td><td>Meaning in CookVoice</td></tr><tr><td>Human Voice Genera- tion (HVG)</td><td>General task</td><td>A broad task family that generates human speech or singing voice from control signals such as text, lyrics, notes, style prompts, reference voices, or pitch</td></tr><tr><td>Text-to-Speech (TTS)</td><td>Speech task</td><td>contours. Generate a spoken voice waveform from text or phoneme content.</td></tr><tr><td>Text-to-Singing Voice Singing task (TTSV)</td><td></td><td>Generate a singing voice waveform from lyrics and musical note information.</td></tr><tr><td>Voice Conversion (VC)</td><td>Conversion task</td><td>Convert a source speech voice into a target speaker or style while preserving the source content and prosody.</td></tr><tr><td>Singing Voice Conver- Conversion task sion (SVC)</td><td></td><td>Convert a source singing voice into a target singing voice style while preserv- ing lyrics and melody.</td></tr><tr><td>Phoneme</td><td>Linguistics</td><td>The basic sound unit of speech. For example, a word is converted into a se- quence of phonemes before being generated as speech or singing voice.</td></tr><tr><td>Grapheme-to-Phoneme (G2P)</td><td>Linguistics</td><td>A text processing module that converts written text or lyrics into phoneme sequences.</td></tr><tr><td>Lyrics</td><td>Music / linguistics</td><td>The textual content of a song. In CookVoice, lyrics are converted into phonemes and used as singing content.</td></tr><tr><td>Lexical prosody</td><td>Linguistics</td><td>Discrete pronunciation-related information attached to phonemes or words, such as Mandarin tones or English word stresses.</td></tr><tr><td>Tone</td><td>Linguistics</td><td>A pitch pattern that changes word meaning in tonal languages such as Man- darin Chinese. In CookVoice, tones are used as discrete prosody tokens for</td></tr><tr><td>Stress</td><td>Linguistics</td><td>speech. The emphasis placed on a syllable or word, often affecting loudness, duration, and pitch in English speech.</td></tr><tr><td>Prosody</td><td>Speech / music</td><td>The expressive pattern of voice, including pitch, rhythm, duration, stress, and intonation. Prosody controls how the content is spoken or sung.</td></tr><tr><td>Discrete prosody</td><td>Control signal</td><td>Symbolic prosody information such as tones, stresses, or MIDI notes. It pro- vides coarse but practical control without requiring reference audio.</td></tr><tr><td>Continuous prosody</td><td>Control signal</td><td>Frame-level acoustic prosody information, mainly represented by the continu-</td></tr><tr><td> $F _ { 0 }$ </td><td>Acoustics</td><td>ous Fo contour. It provides fine-grained pitch control. Fundamental frequency, corresponding to the perceived pitch of the voice.</td></tr><tr><td> $F _ { 0 }$  contour</td><td>Acoustics</td><td>Higher  $F _ { 0 }$  generally means a higher-pitched voice. A time-varying sequence of  $F _ { 0 }$  values that describes how pitch changes over</td></tr><tr><td>Waveform</td><td>Signal processing</td><td>time. CookVoice uses it for frame-level continuous prosody control. The raw audio signal represented as amplitude values over time. It is the final</td></tr><tr><td>Spectrogram</td><td>Signal processing</td><td>audio output that listeners hear. A time-frequency representation of audio. It shows how the frequency content of a waveform changes over time, and is commonly used as an intermediate</td></tr><tr><td>Frame</td><td>Signal processing</td><td>acoustic representation for voice generation. The smallest unit of slice in the spectrogram along temporal axis. A voice waveform is converted into a spectrogram with multiple frames along the tem-</td></tr><tr><td>Latent acoustic embed- Signal processing</td><td></td><td>poral axis, and CookVoice aligns content, style, and prosody to these acoustic frames for fine-grained control. A compressed acoustic feature extracted from the spectrogram. CookVoice</td></tr><tr><td>ding Frame-level alignment Model design</td><td></td><td>generates the target latent acoustic embedding, which is then converted back into a waveform by the decoder. A mechanism that expands different control signals to match the acoustic</td></tr><tr><td>Duration</td><td>Timing</td><td>frame length, allowing content, style, and prosody to be combined at each time step. The length of time a phoneme or note lasts. Duration determines how content</td></tr></table>

Table 6: Terminology used in CookVoice. The table provides simplified explanations of linguistic, musical, control, and evaluation termsfor non-expert readers. Continued.
<table><tr><td>Term</td><td>Category</td><td>Meaning in CookVoice</td></tr><tr><td>MIDI note</td><td>Music</td><td>A symbolic musical pitch representation used in singing voice generation. It specifies which note should be sung.</td></tr><tr><td>Note duration</td><td>Music</td><td>The length of a musical note, usually derived from the score or beat informa-</td></tr><tr><td>Beat</td><td>Music</td><td>tion. It controls how long a note is sustained in singing. A unit of musical timing. In CookVoice, beat information is used to compute</td></tr><tr><td>Melody</td><td>Music</td><td>relative note and phoneme durations for singing voice generation. The ordered pitch movement of singing. It is mainly controlled by musical</td></tr><tr><td>Humming</td><td>Music / control</td><td>notes or continuous  $F _ { 0 }$  contours. A user-provided vocal melody without full lyrics. CookVoice can use its ex- tracted 1  $F _ { 0 }$  contour as a prosody control signal.</td></tr><tr><td>Pitch sketch</td><td>User control</td><td>A manually specified or coarse pitch trajectory used to guide the generated speech or singing voice.</td></tr><tr><td>Style</td><td>Voice attribute</td><td>Global paralinguistic characteristics of a voice, such as speaker identity, tim- bre, emotion, age, gender, and speaking or singing manner.</td></tr><tr><td>Timbre</td><td>Voice attribute</td><td>The tone color or acoustic identity of a voice. Two voices can sing the same note but sound different because of different timbres.</td></tr><tr><td>Text style prompt</td><td>Style control</td><td>A natural language description of the desired voice style, such as “a female</td></tr><tr><td>Reference voice</td><td>Style control</td><td>speaking happily&quot;. An example speech or singing voice used to provide target voice style, timbre,</td></tr><tr><td>Text-based style</td><td>Style control</td><td>or speaker identity. Style information extracted from a natural language prompt. It provides</td></tr><tr><td>Voice-based style</td><td>Style control</td><td>semantic-level style control. Style information extracted from a reference voice. It provides more direct acoustic style and timbre control.</td></tr><tr><td>WER</td><td>Evaluation metric</td><td>Word Error Rate. It measures word-level content errors after transcribing gen-</td></tr><tr><td>PhoER</td><td>Evaluation metric</td><td>erated audio with an ASR system. Lower is better. Phoneme Error Rate. It measures phoneme-level pronunciation or content er-</td></tr><tr><td>ProER</td><td>Evaluation metric</td><td>rors. Lower is better. Prosody Error Rate. It measures errors in discrete prosody tokens such as tones</td></tr><tr><td>S-SIM</td><td>Evaluation metric</td><td>and stresses. Lower is better. Style Similarity. It measures how similar the generated voice style is to the</td></tr><tr><td>F0-RMSE</td><td>Evaluation metric</td><td>target style. Higher is better. Root mean square error between generated and target Fo contours. It measures</td></tr><tr><td>F0-CORR</td><td>Evaluation metric</td><td>absolute pitch deviation. Lower is better. Pearson correlation between generated and target Fo contours. It measures</td></tr><tr><td>MOS</td><td>Subjective metric</td><td>whether the pitch trend is preserved. Higher is better. Mean Opinion Score. Human listeners rate the naturalness or quality of gen-</td></tr><tr><td>M-CMOS</td><td>Subjective metric</td><td>erated audio, usually on a 1–5 scale. Melody Comparative MOS. Human listeners compare whether one singing sample follows the melody better than another.</td></tr></table>

Table 7: Intelligibility and prosodic evaluation comparison on Chinese and English subsets. Word Error Rate (WER), Phoneme Error Rate (PhoER), Prosody Error Rate (ProER) ↓ indicates lower is better.
<table><tr><td rowspan="2">Task</td><td rowspan="2">Model</td><td colspan="3">Chinese</td><td colspan="3">English</td></tr><tr><td>WER</td><td>PhoER</td><td>ProER</td><td>WER</td><td>PhoER</td><td>ProER</td></tr><tr><td rowspan="10">TTS</td><td>CosyVoice [1]</td><td> $8 . 6 2 \pm 1 0 . 6 0$ </td><td> $3 . 4 8 \pm 1 2 . 1 9$ </td><td> $2 . 6 3 \pm 7 . 8 0$ </td><td> $4 . 4 6 \pm 8 . 3 0$ </td><td> $3 . 9 3 \pm 8 . 3 5$ </td><td> $2 . 6 5 \pm 6 . 7 1$ </td></tr><tr><td>F5-TTS [4]</td><td> $5 . 2 3 \pm 7 . 4 5$ </td><td> $0 . 4 7 \pm 1 . 6 7$ </td><td> $0 . 6 3 \pm 1 . 7 8$ </td><td> $2 . 3 4 \pm 4 . 6 7$ </td><td> $1 . 6 3 \pm 3 . 3 2$ </td><td> $0 . 9 8 \pm 1 . 9 6$ </td></tr><tr><td>ParaStyleTTS [3]</td><td> $9 . 6 7 \pm 8 . 2 5$ </td><td> $4 . 6 3 \pm 6 . 7 0$ </td><td> $3 . 2 7 \pm 6 . 5 4$ </td><td> $5 . 2 1 \pm 6 . 9 5$ </td><td> $5 . 4 0 \pm 7 . 4 2$ </td><td> $1 . 8 2 \pm 2 . 8 0$ </td></tr><tr><td>IndexTTS [2]</td><td> $4 . 1 6 \pm 6 . 6 3$ </td><td> $0 . 3 5 \pm 1 . 6 0$ </td><td> $0 . 3 5 \pm 1 . 6 0$ </td><td> $2 . 8 8 \pm 6 . 0 4$ </td><td> $2 . 0 4 \pm 5 . 2 6$ </td><td> $1 . 2 9 \pm 4 . 0 7$ </td></tr><tr><td>Vevo2 [8]</td><td> $6 . 3 5 \pm 6 . 7 3$ </td><td> $1 . 5 6 \pm 2 . 9 3$ </td><td> $1 . 1 7 \pm 2 . 6 2$ </td><td> $3 . 8 2 \pm 6 . 4 7$ </td><td> $2 . 7 2 \pm 6 . 0 1$ </td><td> $1 . 6 2 \pm 4 . 3 3$ </td></tr><tr><td>CookVoice</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $S \tau + \mathcal { P } _ { d i s c }$ </td><td> $9 . 4 8 \pm 1 1 . 8 8$ </td><td> $4 . 2 2 \pm 1 0 . 7 5$ </td><td> $3 . 7 9 \pm 9 . 1 7$ </td><td> $6 . 0 7 \pm 1 2 . 3 5$ </td><td> $5 . 8 0 \pm 1 2 . 4 3$ </td><td> $3 . 0 5 \pm 6 . 0 9$ </td></tr><tr><td> $S \tau + \mathcal { P } _ { c o n t }$ </td><td> $9 . 0 3 \pm 1 0 . 9 8$ </td><td> $3 . 0 7 \pm 8 . 6 5$ </td><td> $2 . 6 9 \pm 6 . 7 4$ </td><td> $5 . 2 3 \pm 8 . 7 6$ </td><td> $4 . 4 4 \pm 8 . 8 5$ </td><td> $2 . 6 4 \pm 5 . 2 0$ </td></tr><tr><td> $S _ { \mathcal { V } } + \mathcal { P } _ { d i s c }$ </td><td> $8 . 6 6 \pm 1 0 . 1 4$ </td><td> $4 . 2 8 \pm 9 . 6 6$ </td><td> $3 . 1 3 \pm 6 . 9 7$ </td><td> $4 . 4 1 \pm 7 . 4 0$ </td><td> $3 . 9 7 \pm 7 . 2 1$ </td><td> $1 . 9 0 \pm 3 . 5 0$ </td></tr><tr><td> $S _ { \nu } + \mathcal { P } _ { c o n t }$ </td><td> $8 . 1 3 \pm 1 0 . 0 7$ </td><td> $3 . 7 8 \pm 9 . 1 9$ </td><td> $3 . 1 7 \pm 6 . 6 9$ </td><td> $4 . 1 9 \pm 7 . 4 5$ </td><td> $3 . 7 7 \pm 7 . 8 1$ </td><td> $1 . 4 7 \pm 2 . 9 7$ </td></tr><tr><td rowspan="10"></td><td>Ground Truth</td><td> $6 . 3 3 \pm 9 . 3 3$ </td><td> $2 . 0 1 \pm 5 . 4 6$ </td><td> $1 . 7 8 \pm 4 . 9 9$ </td><td> $3 . 3 9 \pm 7 . 0 7$ </td><td> $2 . 8 4 \pm 7 . 1 4$ </td><td> $1 . 3 1 \pm 3 . 5 8$ </td></tr><tr><td>DiffSinger [5] StyleSinger [6]</td><td> $1 7 . 7 9 \pm 1 4 . 1 6$ </td><td> $1 6 . 4 3 \pm 1 3 . 9 9$ </td><td> $1 7 . 0 4 \pm 1 4 . 2 1$ </td><td></td><td></td><td></td></tr><tr><td>TCSinger [7]</td><td> $1 1 . 3 1 \pm 7 . 6 8$   $9 . 8 8 \pm 7 . 2 3 $ </td><td> $1 0 . 4 3 \pm 8 . 8 7$ </td><td> $1 0 . 0 5 \pm 7 . 5 4$ </td><td></td><td></td><td></td></tr><tr><td>Vevo1.5 [15]</td><td></td><td> $7 . 6 1 \pm 6 . 1 5$ </td><td> $7 . 4 0 \pm 6 . 2 6$ </td><td> $2 9 . 0 3 \pm 1 9 . 3 1$ </td><td> $3 1 . 4 5 \pm 2 1 . 8 5$ </td><td> $1 6 . 1 1 \pm 1 7 . 3 3$ </td></tr><tr><td>Vevo2 [8]</td><td> $8 . 8 5 \pm 1 1 . 0 9$ </td><td> $8 . 4 0 \pm 1 0 . 9 2$ </td><td> $9 . 3 4 \pm 9 . 4 3$ </td><td> $2 3 . 6 2 \pm 1 4 . 6 7$ </td><td> $2 4 . 9 1 \pm 1 5 . 1 7$ </td><td> $1 1 . 2 3 \pm 8 . 0 4$ </td></tr><tr><td></td><td> $4 . 6 6 \pm 6 . 0 4$ </td><td> $2 . 8 9 \pm 4 . 9 6$ </td><td> $4 . 2 1 \pm 5 . 0 0$ </td><td> $1 1 . 3 7 \pm 9 . 8 0$ </td><td> $1 0 . 8 4 \pm 9 . 6 1$ </td><td> $4 . 7 4 \pm 4 . 4 8$ </td></tr><tr><td>CookVoice</td><td> $1 1 . 7 1 \pm 1 0 . 4 4$ </td><td> $1 1 . 0 8 \pm 1 1 . 7 9$ </td><td> $1 0 . 8 0 \pm 8 . 9 6$ </td><td> $1 9 . 5 6 \pm 1 4 . 4 7$ </td><td> $2 0 . 8 1 \pm 1 5 . 7 1$ </td><td></td></tr><tr><td> $S _ { T } + \mathcal { P } _ { d i s c }$ </td><td> $9 . 7 9 \pm 9 . 2 6$ </td><td></td><td></td><td></td><td></td><td> $8 . 7 9 \pm 6 . 7 2$ </td></tr><tr><td> $S \tau + \mathcal { P } _ { c o n t }$ </td><td></td><td> $9 . 8 2 \pm 1 2 . 2 5$ </td><td> $8 . 9 2 \pm 7 . 2 7$ </td><td> $2 1 . 1 0 \pm 1 7 . 8 1$ </td><td> $2 2 . 1 7 \pm 1 9 . 7 1$ </td><td> $1 0 . 0 5 \pm 8 . 9 1$ </td></tr><tr><td> $S _ { \mathcal { V } } + \mathcal { P } _ { d i s c }$   $S _ { \mathcal { V } } + \mathcal { P } _ { c o n t }$ </td><td> $1 1 . 4 5 \pm 9 . 1 4$   $1 0 . 0 1 \pm 9 . 7 0$ </td><td> $1 1 . 1 5 \pm 1 0 . 1 2$   $8 . 6 5 \pm 1 1 . 9 9$ </td><td> $1 0 . 0 1 \pm 7 . 9 3$   $8 . 7 8 \pm 7 . 8 3$ </td><td> $2 2 . 5 3 \pm 1 6 . 9 2$   $2 4 . 3 7 \pm 1 7 . 2 3$ </td><td> $2 3 . 3 2 \pm 1 9 . 2 0$   $2 4 . 6 8 \pm 2 0 . 0 9$ </td><td> $1 0 . 5 3 \pm 7 . 8 8$ </td></tr><tr><td>Ground Truth</td><td></td><td> $5 . 0 1 \pm 4 . 7 6$ </td><td> $3 . 0 6 \pm 5 . 3 3$ </td><td> $4 . 9 9 \pm 5 . 9 8$ </td><td> $1 4 . 2 2 \pm 1 2 . 9 3$  </td><td> $1 3 . 1 6 \pm 1 0 . 9 2$ </td><td> $1 0 . 5 4 \pm 6 . 7 0$   $7 . 2 6 \pm 8 . 0 0$ </td></tr></table>