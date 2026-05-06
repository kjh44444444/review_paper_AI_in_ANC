# Artificial Intelligence and Machine Learning in Active Noise Control: Foundations, Technical Pathways, Applications, and Open Challenges

Author: [Name]  
Affiliation: [Institution]  
Corresponding author: [email]

## Abstract

Active noise control (ANC) reduces unwanted sound by generating a secondary acoustic field that destructively interferes with a primary disturbance. Although ANC is often introduced through the physical principle of superposition, practical systems are constrained by causality, acoustic propagation, secondary-path dynamics, actuator limits, sensor placement, feedback contamination, and environmental nonstationarity. Classical adaptive-filtering methods, especially filtered-reference least-mean-square (FxLMS) algorithms and their multichannel variants, remain the operational backbone of many ANC systems. However, the growing use of ANC in vehicles, ducts, headrests, structural-acoustic enclosures, distributed sensor networks, and compact consumer devices has exposed limitations of purely linear and model-dependent controllers. Artificial intelligence (AI) and machine learning (ML) methods have therefore become increasingly important for nonlinear modeling, secondary-path compensation, fixed-filter selection, acoustic-state recognition, controller generation, and transfer across operating conditions.

This review synthesizes an expanded curated body of 160 ANC-related journal and research papers from 1990 to 2026, including 50 additional recent papers collected with emphasis on current journal work. The review focuses on the transition from classical adaptive ANC to learning-assisted and deep-learning ANC. The paper first summarizes ANC principles, system architectures, and the historical role of FxLMS and secondary-path modeling. It then classifies AI/ML-enabled ANC into five technical pathways: nonlinear adaptive learning, neural and fuzzy-neural control, evolutionary and swarm optimization, deep-learning-assisted ANC, and hybrid/generative fixed-filter ANC. Mathematical formulations are included for the core feedforward ANC model, FxLMS/FxNLMS adaptation, multichannel filtered-reference control, Volterra and kernel expansions, and learning-based fixed-filter selection. Engineering applications are reviewed across transportation, HVAC and duct acoustics, headrests and personal zones, structural-acoustic systems, and distributed multichannel ANC. Finally, the review identifies persistent challenges in causality, stability, real-time implementation, benchmark design, generalization, interpretability, and safe online adaptation. The central conclusion is that the most promising ANC research is not replacing adaptive control with AI, but combining physical control constraints with learned representations and data-driven adaptation.

## Keywords

active noise control; active noise cancellation; artificial intelligence; machine learning; deep learning; FxLMS; nonlinear adaptive filtering; secondary path; generative fixed-filter ANC; multichannel ANC

## 1. Introduction

Noise control is a long-standing engineering problem because unwanted sound affects health, comfort, communication, machinery reliability, product quality, and perceived system performance. Passive treatments such as insulation, damping, absorptive materials, and barriers are effective in many settings, but they become bulky or inefficient at low frequencies where acoustic wavelengths are long. Active noise control (ANC) addresses this low-frequency limitation by driving secondary sources, usually loudspeakers or structural actuators, so that the generated field cancels the primary disturbance at target locations. The principle is simple; the implementation is not. A successful ANC system must react before the unwanted sound reaches the error location, account for the dynamics between actuator and microphone, remain stable under modeling error, avoid contaminating the reference signal, and preserve performance when the acoustic environment changes.

The modern ANC literature is built on this tension between a clean physical idea and a difficult real-time control problem. Early work established the feasibility and limits of active sound control in acoustic fields and ducts, then adaptive signal processing made ANC practical by allowing controllers to update in response to measured error signals. The development of filtered-reference adaptive algorithms, including the filtered-U and filtered-X LMS families, was decisive because it recognized that the controller output is not applied directly to the error microphone. Instead, the output passes through a secondary path containing digital-to-analog conversion, amplification, actuator dynamics, acoustic propagation, microphone response, and analog-to-digital conversion. Errors in this path model can degrade performance or destabilize adaptation, making secondary-path modeling a central theme rather than a peripheral calibration step (Eriksson, 1990; Boucher et al., 1991; Bjarnason, 1995).

For much of the field's history, ANC progress came from improving adaptive filters, multichannel implementations, secondary-path estimation, and robust update rules. These methods remain essential. Yet many modern applications violate the assumptions under which simple linear adaptive controllers perform best. Loudspeakers saturate, ducts exhibit nonlinear flow-acoustic interactions, road-noise transfer paths vary with vehicle speed and surface, structural-acoustic systems couple many modes, and personal audio zones move with the listener. In such settings, a controller designed around a fixed linear model can converge slowly, converge to an inadequate solution, or fail when the environment shifts. These practical pressures explain why AI and ML have entered ANC research.

It is useful to be precise about what "AI in ANC" means. In the ANC literature, AI is not a single method and it is not always synonymous with deep learning. Early AI-related ANC work used neural networks, recurrent radial-basis-function networks, genetic algorithms, fuzzy logic, functional-link neural networks (FLANN), Volterra expansions, kernel filters, and other nonlinear adaptive structures. These methods were often designed to preserve the adaptive-control architecture while increasing the controller's ability to represent nonlinear input-output relationships. More recent work uses convolutional neural networks (CNNs), recurrent neural networks (RNNs), long short-term memory networks (LSTMs), generative models, reinforcement learning (RL), meta-learning, and causal neural architectures to perform higher-level tasks such as filter generation, state classification, secondary-path decoupling, and multichannel control (Zhang and Wang, 2021; Luo et al., 2022; Zhang and Wang, 2022; Luo et al., 2023; Shi et al., 2023; Luo et al., 2024).

This review follows the structure of a systems-oriented review paper. Rather than listing algorithms chronologically only, it organizes the literature by control function. The guiding question is: where does learning enter the ANC loop, and what problem does it solve? Learning can be used to approximate nonlinear control filters, identify secondary or primary paths, choose among fixed filters, tune adaptive parameters, compress high-dimensional acoustic states into transferable latent variables, or synthesize controllers for a target condition. Each role has different requirements for causality, latency, stability, training data, and interpretability.

The contributions of this review are as follows:

1. It connects classical ANC theory with recent AI/ML-enabled ANC, showing why modern methods still depend on secondary-path-aware, causal, low-latency control.
2. It provides a technical classification of AI/ML ANC methods based on their function in the control architecture rather than only on algorithm names.
3. It compares nonlinear adaptive filters, neural/fuzzy systems, optimization-based controllers, deep ANC, and generative fixed-filter ANC in terms of strengths, limitations, and deployment risks.
4. It summarizes engineering application scenarios and identifies evaluation criteria beyond nominal noise reduction.
5. It outlines open research problems in benchmark design, safe learning, generalization, embedded implementation, and hybrid physics-learning control.

The rest of the paper is organized as follows. Section 2 reviews ANC principles, system architectures, and historical development. Section 3 classifies AI/ML-enabled ANC technical pathways. Section 4 discusses engineering applications. Section 5 analyzes evaluation metrics and comparative design tradeoffs. Section 6 presents challenges and future directions. Section 7 concludes the review.

## 2. Overview of Active Noise Control and Learning-Enabled ANC

### 2.1. Physical Principle and Control Objective

ANC uses destructive interference to reduce sound pressure at one or more control points. In a simple single-channel feedforward ANC system, a reference sensor measures a signal correlated with the primary disturbance. A controller transforms the reference into a control signal that drives a secondary source. An error microphone measures the residual sound after the primary and secondary fields combine. The adaptive controller updates its parameters to minimize the residual error, often by minimizing a mean-square error criterion.

This formulation hides several important constraints. First, the controller must be causal: it must generate the anti-noise early enough to arrive at the cancellation point with the right phase. This is easier for periodic or predictable low-frequency noise than for broadband random noise with short propagation delay. Second, the desired cancellation is spatially local. Reducing sound at one microphone may increase sound elsewhere, especially at higher frequencies where wavelengths are shorter. Third, the control action is filtered by the secondary path, so adaptation based on the raw reference signal gives the wrong gradient unless the reference is filtered through a secondary-path model. Fourth, actuators and sensors have amplitude, bandwidth, noise, and nonlinear distortion limits.

These constraints explain why ANC is not simply a supervised signal-separation problem. A neural network can estimate a waveform, classify an acoustic state, or generate a filter, but the generated control signal still enters a physical feedback/control loop. If a learning model improves one part of the loop while ignoring delay, phase, saturation, or robustness, the measured cancellation may not improve in a real system.

#### 2.1.1. Single-channel feedforward signal model

The standard single-channel feedforward ANC model can be written in discrete time. Let \(x(n)\) be the reference signal, \(w(n)=[w_0(n),w_1(n),...,w_{L-1}(n)]^T\) be the adaptive control filter, and

\[
\mathbf{x}(n)=[x(n),x(n-1),...,x(n-L+1)]^T .
\]

The controller output is

\[
y(n)=\mathbf{w}^T(n)\mathbf{x}(n).
\]

This signal is emitted by the secondary source and filtered by the secondary path \(s(n)\). The secondary sound arriving at the error microphone is therefore

\[
y_s(n)=s(n)*y(n),
\]

where \(*\) denotes convolution. If \(d(n)\) is the primary disturbance at the error microphone, the residual error is

\[
e(n)=d(n)+y_s(n).
\]

Some authors use \(e(n)=d(n)-y_s(n)\); the sign convention depends on whether the controller output already includes the phase inversion. The control objective is usually expressed as minimizing the expected squared residual

\[
J(n)=E[e^2(n)].
\]

This compact model shows why ANC is a control problem rather than an ordinary regression problem. The controller coefficients do not affect \(e(n)\) directly; they affect the error through \(s(n)\). Therefore, the gradient used for adaptation must account for the secondary path.

#### 2.1.2. FxLMS and FxNLMS adaptation

The filtered-X LMS algorithm approximates the negative gradient of \(J(n)\) by filtering the reference vector through an estimated secondary path \(\hat{s}(n)\). Define

\[
x_f(n)=\hat{s}(n)*x(n),
\]

and

\[
\mathbf{x}_f(n)=[x_f(n),x_f(n-1),...,x_f(n-L+1)]^T .
\]

The FxLMS coefficient update is

\[
\mathbf{w}(n+1)=\mathbf{w}(n)-\mu e(n)\mathbf{x}_f(n),
\]

where \(\mu\) is the step size. With the opposite error sign convention, the update appears with a plus sign. The normalized version, FxNLMS, scales the update by filtered-reference energy:

\[
\mathbf{w}(n+1)=\mathbf{w}(n)-\mu \frac{e(n)\mathbf{x}_f(n)}{\delta+\mathbf{x}_f^T(n)\mathbf{x}_f(n)},
\]

where \(\delta>0\) prevents division by a small number. This normalization improves robustness when reference-signal power varies.

The step size must be selected conservatively. A common qualitative bound is that \(\mu\) must decrease as filtered-reference power, filter length, and secondary-path uncertainty increase. Large \(\mu\) accelerates convergence but can increase misadjustment or instability; small \(\mu\) improves stability but may fail to track time-varying paths. Variable step-size, robust cost, momentum, affine projection, Kalman, and meta-learning approaches can be interpreted as attempts to improve this convergence-stability compromise.

### 2.2. System Architectures

The basic ANC architectures are feedforward, feedback, and hybrid control. Feedforward systems require a reference signal correlated with the disturbance before it reaches the error microphone. Feedback systems use only the residual error and are useful when a reference signal is unavailable, but stability and bandwidth constraints are more restrictive. Hybrid systems combine both.

**Table 1. ANC architectures and implications for AI/ML integration.**

| Architecture | Core signal path | Strengths | Limitations | Common AI/ML roles |
|---|---|---|---|---|
| Feedforward ANC | Reference sensor -> controller -> secondary source -> error sensor | Strong for predictable, periodic, or measurable disturbances; clear adaptive-filter formulation | Requires a clean reference; vulnerable to secondary-path feedback and causality limits | Deep fixed-filter selection, nonlinear control filters, reference feature extraction, secondary-path-aware training |
| Feedback ANC | Error sensor -> controller -> secondary source | Does not require a separate reference; compact sensor layout | Stability-bandwidth tradeoff; limited performance for broadband unpredictable noise | Neural feedback controller, model-based prediction, robust/adaptive gain tuning |
| Hybrid ANC | Feedforward and feedback loops combined | Can handle predictable and residual disturbances; improved robustness | More complex tuning and coupling between loops | Learned mode switching, parameter scheduling, adaptive filter generation |
| Multichannel ANC | Multiple references, secondary sources, and error sensors | Can enlarge quiet zones and address spatially complex fields | High computational burden; path coupling; stability and calibration difficulty | Deep multichannel mapping, decoupling, latent acoustic representations, distributed learning |
| Selective fixed-filter ANC | Classifier or selector chooses from predesigned filters | Low online adaptation burden; suitable for embedded systems | Performance depends on state coverage and classifier reliability | CNN/RNN state recognition, transfer learning, generative fixed-filter synthesis |

The architecture determines what kind of learning is appropriate. A high-capacity model may be useful offline for generating a bank of filters, but too slow for sample-by-sample control. Conversely, a compact neural expansion may be feasible online but insufficient for large environmental changes. The main design task is therefore not merely selecting an AI model; it is matching the learning role to the control architecture.

### 2.3. Classical ANC Foundations

Classical ANC research established principles that still govern learning-enabled systems. Foundational studies on active sound fields and filtered-reference algorithms showed that low-frequency active control is physically feasible but strongly dependent on propagation, modeling, and adaptation conditions (Nelson et al., 1990; Eriksson, 1990; Nelson and Elliott, 1992; Elliott and Nelson, 1993). Work on plant-model errors demonstrated that mismatch in the assumed path can significantly degrade adaptive feedforward control (Boucher et al., 1991). Analysis of the filtered-X LMS algorithm clarified convergence behavior and helped make FxLMS a standard baseline (Bjarnason, 1995).

The standard FxLMS update can be understood as a correction to ordinary LMS. Because the controller coefficients influence the error only after passing through the secondary path, the reference signal used in the gradient update must be filtered by an estimate of that path. This secondary-path estimate may be obtained offline, online, or through hybrid identification. If the estimate is inaccurate, the gradient direction becomes biased. If the acoustic path changes, a once-valid estimate may become stale.

Multichannel ANC extends this problem. With multiple loudspeakers and microphones, each actuator can influence each error microphone, so the secondary path becomes a matrix of transfer functions. Multichannel systems can create larger zones of quiet and support applications such as active headrests, vehicle cabins, enclosures, and distributed networks. However, they also multiply the computational cost and introduce path coupling. Douglas (1999) addressed fast implementations of filtered-X and LMS algorithms for multichannel ANC, a theme that remains relevant as deep models are added to the loop.

Mathematically, for \(J\) secondary sources and \(M\) error microphones, the residual at microphone \(m\) is

\[
e_m(n)=d_m(n)+\sum_{j=1}^{J}s_{mj}(n)*y_j(n),
\]

where \(s_{mj}(n)\) is the secondary path from source \(j\) to error microphone \(m\). If source \(j\) has filter \(\mathbf{w}_j(n)\), the multichannel cost is often written as

\[
J(n)=E\left[\sum_{m=1}^{M} e_m^2(n)\right]
\]

or, in vector form,

\[
J(n)=E[\mathbf{e}^T(n)\mathbf{e}(n)].
\]

The filtered-reference signal for each controller must account for every source-to-error path. This is why multichannel ANC grows rapidly in computational burden: the algorithm does not merely run \(J\) independent single-channel controllers; it must adapt through an \(M \times J\) matrix of secondary paths. Recent Kronecker-product, nearest Kronecker decomposition, circular-convolution, frequency-point selection, and distributed FxLMS methods are best understood as structured approximations to this high-dimensional filtered-reference problem (Ferrer et al., 2017; Kranthi et al., 2024; Li et al., 2025; Li et al., 2026; Xu et al., 2026).

### 2.4. The Bridge from Nonlinear ANC to AI/ML ANC

The bridge between classical ANC and modern AI/ML ANC is nonlinear adaptive filtering. Nonlinearities arise from loudspeaker saturation, amplifier distortion, turbulent flow, structural-acoustic coupling, nonlinear primary paths, and nonlinear secondary paths. Linear FxLMS controllers can fail when the dominant error component is generated by nonlinear dynamics rather than by a linear transformation of the reference.

Early nonlinear ANC methods used Volterra filters, bilinear filters, filtered-s LMS, and other structured expansions (Strauch and Mulgrew, 1998; Tan and Jiang, 2001; Das and Panda, 2004; Kuo and Wu, 2005). These methods are not always described as AI, but they share a core ML idea: expanding the representation so that a controller can learn nonlinear input-output mappings from data. Later FLANN, kernel, random Fourier, fuzzy-neural, and recurrent structures developed this theme with different complexity-accuracy tradeoffs (Sicuranza and Carini, 2011; Le et al., 2017; Liu et al., 2018; Deb et al., 2020; Zhu et al., 2021).

This historical sequence matters because it prevents a misleading narrative in which deep learning suddenly replaces adaptive control. In reality, ANC has been moving toward learning-assisted nonlinear control for decades. Deep learning is the newest and most flexible expression of that movement, but it inherits the same control constraints.

## 3. Technical Path Classification of AI/ML-Enabled ANC

AI/ML methods in ANC can be classified by where they intervene in the control problem. The categories below overlap in some papers, but they provide a useful map of the field.

### 3.1. Nonlinear Adaptive Learning: Volterra, Bilinear, Kernel, and Random-Feature Filters

Nonlinear adaptive filters are the earliest and most technically continuous path from classical ANC to ML-enabled ANC. Volterra filters model nonlinear systems as polynomial expansions with memory. Bilinear filters provide lower-complexity structures for certain nonlinear dynamics. Kernel methods implicitly map input data into high-dimensional feature spaces. Random Fourier and Nyström approximations reduce the cost of kernel-like learning.

Volterra-based ANC is attractive because it is interpretable and directly tied to nonlinear system theory. Tan and Jiang (2001) developed adaptive Volterra filters for active control of nonlinear noise processes, and later work considered second-order Volterra filtered-X RLS algorithms with sequential or partial updates. Zhao et al. (2011) proposed an adaptive extended pipelined second-order Volterra filter, addressing computational concerns in nonlinear controllers. Yu et al. (2022) used an interpolated individual weighting subband Volterra filter, showing that Volterra ideas remain relevant even after deep learning became prominent.

For a second-order Volterra controller, the output may be written as

\[
y(n)=\sum_{i=0}^{L_1-1}h_1(i)x(n-i)+\sum_{i=0}^{L_2-1}\sum_{j=0}^{L_2-1}h_2(i,j)x(n-i)x(n-j).
\]

The first term is the ordinary linear FIR controller. The second term models quadratic interactions with memory. Higher-order Volterra filters add cubic and higher terms, but the number of parameters grows rapidly. If a second-order kernel has \(L_2^2\) coefficients, then even moderate memory length becomes expensive in real time. This is why diagonal, sparse, pipelined, subband, and partial-update Volterra structures are common. The mathematical value of the Volterra form is that it directly reveals the accuracy-complexity tradeoff: nonlinear modeling capacity increases with order and memory, while adaptation burden increases combinatorially.

Bilinear and diagonal-structure filters reduce computational complexity by restricting the nonlinear representation. Kuo and Wu (2005) studied nonlinear adaptive bilinear filters for ANC, while later multichannel and saturation-focused studies used diagonal structures to control complexity (Chen et al., 2015; Guo et al., 2018). Such methods are useful when the nonlinearity is significant but a full Volterra model is too expensive.

Kernel and random-feature ANC methods provide another route. A kernel filtered-X LMS algorithm can represent nonlinear primary paths without explicitly enumerating all features (Liu et al., 2018). Random Fourier filters and cascaded random Fourier filters approximate kernel mappings with controllable complexity, which is valuable for real-time ANC (Deb et al., 2020; Zhu et al., 2021). Clustering-sparse Nyström adaptive filtering extends the same idea toward distributed nonlinear ANC (Xiao et al., 2023).

Kernel ANC can be expressed by replacing the linear inner product \(\mathbf{w}^T\mathbf{x}\) with a function \(f(\mathbf{x})\) in a reproducing kernel Hilbert space. A kernel adaptive filter represents the controller as

\[
f_n(\mathbf{x})=\sum_{i=1}^{n}\alpha_i k(\mathbf{x},\mathbf{x}_i),
\]

where \(k(\cdot,\cdot)\) is a kernel function and \(\alpha_i\) are adaptive coefficients. This gives high nonlinear flexibility, but the dictionary can grow with time. Random Fourier features approximate shift-invariant kernels by mapping the input into a finite-dimensional feature vector

\[
z(\mathbf{x})=\sqrt{\frac{2}{D}}[\cos(\omega_1^T\mathbf{x}+b_1),...,\cos(\omega_D^T\mathbf{x}+b_D)]^T,
\]

so that

\[
k(\mathbf{x},\mathbf{x}')\approx z^T(\mathbf{x})z(\mathbf{x}').
\]

The ANC controller then becomes linear in \(z(\mathbf{x})\), enabling LMS-like adaptation with nonlinear representational power. This explains the practical appeal of random Fourier and Nyström approaches: they provide a middle ground between full kernel methods and manually chosen nonlinear expansions.

The main strength of nonlinear adaptive learning is compatibility with ANC's sample-by-sample adaptation. The main limitation is scaling. As memory length, channel count, and nonlinearity order increase, the number of parameters grows quickly. Many papers therefore propose partial updates, diagonal structures, sparsity, subband processing, or random approximations. In practice, these methods are strongest when the nonlinear structure is moderate, the controller must remain causal, and interpretability matters.

### 3.2. Neural and Fuzzy-Neural Controllers

Neural-network ANC began before modern deep learning. Early work used neural feedback control and recurrent radial-basis-function networks to address nonlinear dynamics and temporal dependence (Qizhi and Yongle, 2000; Bambang, 2003). Later studies used Legendre neural networks, functional-link neural networks, fuzzy variable-step-size methods, fuzzy-neural controllers, and brain-emotional learning networks (kunchakoori et al., 2008; Das and Satapathy, 2011; Sicuranza and Carini, 2011; Thai et al., 2016; Le et al., 2019; Huynh and Chang, 2022; Nguyen et al., 2025).

Functional-link artificial neural networks are especially important in nonlinear ANC. A FLANN expands the input through nonlinear basis functions and then adapts a linear-in-parameters output layer. This structure offers more nonlinear modeling capacity than a purely linear filter while retaining simpler adaptation than a deep network. Stability analysis for recursive FLANN filters helped address one of the main concerns in nonlinear ANC: nonlinear flexibility must not come at the cost of uncontrolled adaptation (Sicuranza and Carini, 2011). Bilinear FLANN and generalized exponential FLANN methods extended the approach to more complex nonlinear and channel-reduced settings (Le et al., 2017; Le et al., 2018; Le et al., 2019).

Fuzzy and neuro-fuzzy ANC methods are useful when operating regimes vary or linguistic/control-rule structures can encode uncertainty. Fuzzy variable-step-size FxLMS modifies adaptation based on error energy or related indicators, while adaptive fuzzy feedback neural controllers target narrowband and nonlinear settings. Neuro-fuzzy hybrid systems attempt to combine interpretable fuzzy rules with neural adaptation (Huynh and Chang, 2022; Nguyen et al., 2025).

The benefit of neural and fuzzy-neural ANC is flexibility under nonlinear and time-varying conditions. The risk is that added flexibility can obscure stability properties. For deployment, these methods need clear bounds on adaptation, robust secondary-path handling, and measured latency.

### 3.3. Evolutionary, Swarm, and Heuristic Optimization

Evolutionary and swarm methods enter ANC mainly as optimization tools rather than real-time waveform controllers. Genetic algorithms, particle swarm optimization, flower pollination heuristics, and related methods can tune controller parameters, nonlinear filter structures, or decentralized control weights when gradient-based optimization is difficult or unreliable (Russo, 2006; Russo and Sicuranza, 2007; Chang and Chen, 2010; George and Panda, 2012; Khan et al., 2021).

The appeal is global or derivative-free search. In nonlinear ANC, the cost surface can be nonconvex, and controller parameters may interact with secondary-path uncertainty. Optimization heuristics can explore parameter spaces that are hard to handle analytically. For example, adaptive genetic methods have been used to avoid explicit secondary-path identification, while particle-swarm approaches have supported decentralized nonlinear ANC.

The limitation is online cost. Many heuristic optimizers are too slow for direct sample-level control, so they are better suited for offline design, initialization, filter-bank generation, parameter scheduling, or occasional retuning. Their contribution to AI-enabled ANC is therefore complementary: they can prepare or tune controllers that still run through fast adaptive-filter or fixed-filter loops.

### 3.4. Deep-Learning-Assisted ANC

Deep learning changes the scale and type of representations available to ANC. Instead of hand-designed nonlinear basis functions, deep networks can learn representations from raw or transformed acoustic signals. CNNs can learn local time-frequency patterns; RNNs, LSTMs, GRUs, and temporal convolutional networks can model temporal dependencies; attention-based models can emphasize relevant temporal or channel features; generative models can synthesize filters or control signals.

Deep ANC by Zhang and Wang (2021) is a milestone because it explicitly formulated ANC as a deep-learning control problem. Deep MCANC extended the idea to multichannel ANC, addressing the more realistic case in which multiple secondary sources and error microphones interact (Zhang and Wang, 2022). These works shifted the discussion from neural networks as nonlinear adaptive filters toward neural networks as higher-capacity controllers for acoustic cancellation.

Other studies use deep learning in more targeted roles. Chen et al. (2021) proposed a secondary-path-decoupled ANC algorithm based on deep learning, addressing one of the key barriers in classical ANC. Luo et al. (2022) combined selective fixed-filter ANC with FxNLMS adaptation, showing that deep learning can improve filter selection while classical normalized adaptation refines the result. Xiang et al. (2023) combined online adaptive estimation with offline CNN modeling for multichannel ANC. Singh et al. (2024) explored attention-based CNNs for multichannel ANC. Zhu et al. (2025) combined a hybrid adaptive self-loading filter with a GRU network for ANC in an HVAC pipe.

A deep ANC controller can be written abstractly as

\[
y(n)=F_{\theta}(\mathbf{x}_{n}),
\]

where \(F_{\theta}\) is a neural network and \(\mathbf{x}_{n}\) is a causal context window of reference samples and, in some architectures, auxiliary acoustic-state features. Training may minimize a secondary-path-aware loss such as

\[
\mathcal{L}(\theta)=\frac{1}{N}\sum_{n=1}^{N}\left[d(n)+\hat{s}(n)*F_{\theta}(\mathbf{x}_{n})\right]^2.
\]

This expression is important because it shows why ordinary waveform prediction losses are insufficient. The network output must be evaluated after propagation through the secondary path. If the training loss does not include \(\hat{s}(n)\), the model may learn a signal that looks like anti-noise in isolation but arrives at the error microphone with incorrect phase or amplitude. Recent work on real-time secondary-path estimation, path decoupling, CNN-Kalman filtering, time-domain neural path decoupling, and dynamic time-warping secondary-path interpolation directly addresses this loss-model mismatch (Chen et al., 2021; Cheng et al., 2025; Chu et al., 2026; Holzmuller and Sontacchi, 2026; Luo et al., 2024).

Deep-learning-assisted ANC is powerful but demanding. Training data must cover relevant acoustic conditions. Models must obey causality and latency constraints. Generalization must be tested across different paths, not only within one simulated environment. A deep model that performs well offline may still be unsuitable if it requires future samples, long context windows, high compute, or retraining under every new acoustic condition.

### 3.5. Hybrid and Generative Fixed-Filter ANC

One of the most promising recent directions is hybrid fixed-filter ANC. Instead of adapting a high-dimensional controller continuously online, the system selects or generates a suitable fixed filter for the current acoustic state. This approach reduces real-time adaptation burden and can be attractive for embedded ANC systems.

Selective fixed-filter ANC (SFANC) uses a predesigned filter bank and selects a filter based on estimated conditions. Deep learning improves this approach by classifying acoustic states or extracting latent representations. Luo et al. (2022) proposed a hybrid SFANC-FxNLMS method, combining deep learning-based selection with adaptive refinement. Shi et al. (2023) studied transferable latent representations for CNN-based SFANC, directly addressing the problem that a classifier trained in one environment may fail in another.

Generative fixed-filter ANC (GFANC) goes further by using deep models to generate filters rather than merely selecting them. Luo et al. (2023) proposed delayless generative fixed-filter ANC using deep learning and a Bayesian filter. Luo et al. (2024) introduced reinforcement learning-based GFANC, adding policy learning to improve filter generation or selection under changing conditions. Luo et al. (2025) examined transferability and implementation of deep-learning-based GFANC, which is critical because practical ANC systems cannot be evaluated only by simulated cancellation.

GFANC is conceptually important because it separates two timescales. A deep model can perform acoustic-state inference or filter synthesis at a slower rate, while the fixed filter runs at the audio sampling rate. This can reduce computational risk and preserve low-latency control. The remaining challenges are state coverage, safe switching, filter interpolation, robustness to misclassification, and performance when the actual environment lies between trained conditions.

The selective fixed-filter problem can be expressed as a discrete decision over a filter bank \(\{\mathbf{w}_1,\mathbf{w}_2,...,\mathbf{w}_K\}\). Given an acoustic feature vector \(\mathbf{z}(n)\), a classifier estimates

\[
k^*(n)=\arg\max_k p(k|\mathbf{z}(n)),
\]

and applies \(\mathbf{w}_{k^*}\). A soft version uses

\[
\mathbf{w}(n)=\sum_{k=1}^{K}p(k|\mathbf{z}(n))\mathbf{w}_k,
\]

which can reduce switching artifacts but may interpolate between filters that are not jointly optimal. A generative fixed-filter model replaces the discrete bank by

\[
\mathbf{w}(n)=G_{\theta}(\mathbf{z}(n)),
\]

where \(G_{\theta}\) maps acoustic-state features to filter coefficients. Reinforcement-learning GFANC can be described by a policy \(\pi_{\theta}(a|\mathbf{z})\) that chooses a filter, filter update, or latent action to maximize a reward such as negative residual energy,

\[
r(n)=-e^2(n)-\lambda C(n),
\]

where \(C(n)\) penalizes switching, computation, or excessive control effort. This formulation clarifies the design risk: a policy trained only for short-term error reduction may choose aggressive actions that create audible transients or instability. Practical GFANC therefore needs state uncertainty, safe switching rules, and fallback filters.

### 3.6. Technical Taxonomy

**Table 2. AI/ML technical pathways in ANC.**

| Pathway | Representative methods | Typical control role | Strengths | Main risks |
|---|---|---|---|---|
| Nonlinear adaptive learning | Volterra, bilinear, kernel FxLMS, random Fourier filters | Direct nonlinear controller or compensator | Causal, adaptive, interpretable compared with deep models | Parameter growth; high multichannel cost; stability analysis needed |
| Neural/fuzzy adaptive control | RBF, recurrent networks, FLANN, fuzzy-neural controllers | Nonlinear mapping, adaptive gain or step-size control, feedback control | Handles nonlinear and time-varying behavior | Harder stability guarantees; tuning complexity |
| Evolutionary/swarm optimization | Genetic algorithms, PSO, flower pollination heuristics | Offline or slow-timescale parameter optimization | Derivative-free search; useful for nonconvex design | Often too slow for real-time sample-level adaptation |
| Deep-learning-assisted ANC | CNN, LSTM, GRU, attention, deep controllers | State recognition, secondary-path decoupling, multichannel mapping, control synthesis | High representational capacity; learns from complex data | Data dependence, latency, poor extrapolation, limited interpretability |
| Hybrid/generative fixed-filter ANC | SFANC, GFANC, RL-GFANC, Bayesian filtering, transfer learning | Filter selection or generation at state timescale | Low online adaptation burden; practical embedded path | Misclassification, filter-switch artifacts, transfer failures |
| Meta-learning and causal neural models | Complex self-attention subband filters, WaveNet-Volterra, TCN-KAN | Fast adaptation, causal nonlinear modeling | Better adaptation across tasks; improved causal design | Newer evidence base; implementation and benchmark gaps |

## 4. Development Trajectory of AI/ML ANC

### 4.1. 1990s: Adaptive Control and the FxLMS Baseline

The 1990s established the ANC baseline. Nelson et al. (1990), Eriksson (1990), Nelson and Elliott (1992), and Elliott and Nelson (1993) framed ANC as a physical and adaptive signal-processing problem. Boucher et al. (1991) emphasized plant-model error, and Bjarnason (1995) analyzed FxLMS behavior. Oppenheim et al. (1994) explored single-sensor active cancellation. Douglas (1999) addressed fast multichannel adaptive implementations.

The core lesson from this period is that ANC performance is not determined only by algorithmic sophistication. It depends on whether the reference signal contains usable preview information, whether the secondary path is modeled sufficiently, and whether the controller can converge within the available delay and computational budget.

### 4.2. 2000s: Nonlinear ANC Becomes Central

The 2000s shifted attention toward nonlinear noise processes. Strauch and Mulgrew (1998), Tan and Jiang (2001), Das and Panda (2004), and Kuo and Wu (2005) developed nonlinear adaptive approaches using Volterra, filtered-s, and bilinear structures. Neural and evolutionary methods also appeared, including recurrent radial-basis-function networks and genetic optimization (Bambang, 2003; Russo, 2006; Russo and Sicuranza, 2007).

This period demonstrates that AI/ML did not enter ANC only after deep learning. The field was already using learning ideas to address nonlinearities, but the methods were usually constrained enough to remain compatible with adaptive control.

### 4.3. 2010s: Structured Nonlinearity, Stability, and Multichannel Expansion

The 2010s broadened nonlinear ANC into a family of structured adaptive methods. FLANN, fuzzy neural networks, wavelet methods, variable step-size algorithms, sparse secondary-path modeling, random features, and multichannel distributed control all became active themes. Stability and computational complexity became recurring concerns, as seen in work on recursive FLANN stability, convex combinations of nonlinear filters, GPU multichannel implementation, sparse nonlinear secondary paths, and hierarchical partial updates (Sicuranza and Carini, 2011; Ferrer et al., 2012; George and Gonzalez, 2013; Lorente et al., 2014; Guo et al., 2018; Le et al., 2019).

The technical direction was pragmatic: increase modeling capacity, but preserve real-time feasibility. Many methods focused less on replacing FxLMS and more on correcting its weak points under nonlinear, time-varying, or multichannel conditions.

### 4.4. 2020-2026: Deep, Generative, Transferable, and Causal ANC

The 2020s introduced deep learning as a major research direction. Deep ANC and Deep MCANC provided a conceptual shift toward neural controllers (Zhang and Wang, 2021; Zhang and Wang, 2022). Secondary-path-decoupled deep learning addressed a classical bottleneck (Chen et al., 2021). Hybrid SFANC-FxNLMS, delayless GFANC, transferable latent SFANC, and GFANC-RL developed a new family of systems in which deep learning supports filter selection, generation, and transfer while retaining low-latency control structures (Luo et al., 2022; Luo et al., 2023; Shi et al., 2023; Luo et al., 2024; Luo et al., 2025).

Recent papers also point toward causality and compactness. WaveNet-Volterra neural networks target fully causal ANC (Bai et al., 2025). Meta-learning-based delayless subband adaptive filtering uses complex self-attention to support fast adaptation (Feng and So, 2025). TCN-KAN proposes parameter-efficient causal neural modeling for nonlinear ANC (Li et al., 2026). Robust momentum conjugate-gradient filtering and adaptive Kalman filtering show that classical robust/adaptive estimation continues alongside deep learning (Islam et al., 2026; Yang et al., 2026).

The field is therefore not converging on one model class. It is converging on hybridization: classical adaptive filters, nonlinear structured models, deep representation learning, and robust estimation are being combined according to application constraints.

## 5. Engineering Application Scenarios

### 5.1. Vehicle and Transportation Noise

Vehicle cabins are one of the most important ANC application domains. Low-frequency road noise, powertrain noise, tire noise, and motor-related tonal components can reduce comfort and perceived quality. Electric and hybrid vehicles make some noise components more noticeable because masking from combustion engines is reduced. Rail vehicles and other transportation systems present similar issues with spatially distributed, nonstationary, and operating-condition-dependent noise.

AI/ML is useful in transportation ANC because the acoustic transfer path changes with speed, road surface, load, seat position, window state, and cabin occupancy. A fixed linear model may be valid only over a narrow range. Deep learning and fuzzy-neural prediction can help classify operating conditions or forecast disturbance components. Li et al. (2021) used a convolutional fuzzy neural network prediction approach for rail-vehicle ANC. Deep fixed-filter and transfer-learning approaches are also relevant because vehicle systems must run in real time on embedded hardware and cannot rely on expensive online training.

The main deployment issues are sensor availability, latency, robustness across road conditions, passenger movement, and integration with existing audio systems. Evaluation should include not only average dB reduction, but also tonal order reduction, spatial consistency across seats, perceptual metrics, and robustness during transitions.

### 5.2. HVAC, Duct, and Pipe Noise

Duct ANC is a classic application because one-dimensional or quasi-one-dimensional propagation at low frequencies makes active control tractable. However, real ducts can include bends, flow, turbulence, fan harmonics, and changing boundary conditions. Right-angle pipes and HVAC systems therefore provide useful testbeds for hybrid ANC.

Deep and recurrent networks can support modeling of duct noise dynamics, especially when combined with adaptive filters. Zhu et al. (2025) proposed a hybrid adaptive self-loading filter and GRU network for ANC in a right-angle bending pipe of an air conditioner. Such architectures illustrate a practical direction: use neural networks to capture operating-state or temporal features, but retain an adaptive filter to enforce real-time cancellation.

For duct applications, the key constraints are reference placement, flow noise at microphones, actuator bandwidth, and physical delay. A method that performs well in a static duct without flow may not generalize to HVAC operation.

### 5.3. Active Headrests and Personal Quiet Zones

Active headrests and personal ANC systems aim to create a quiet zone near the listener's ears without controlling the entire room or cabin. This is attractive because low-frequency control is more feasible over a small spatial region. However, head movement changes the relationship between speakers, microphones, and ears. A quiet zone measured at fixed microphones may not match the listener's actual ear positions.

Multichannel ANC and virtual sensing are central here. AI/ML can support head-position estimation, acoustic-state classification, and filter selection. Selective fixed-filter ANC is especially relevant because a headrest can store filters for different head positions or acoustic states and switch or interpolate between them. Transferable latent representations are important because retraining a filter bank for every user and seat geometry would be impractical.

Safety and comfort also matter. Switching filters must not create audible artifacts or transient amplification. A personal ANC system should be evaluated for spatial robustness and perceptual acceptability, not only microphone-level cancellation.

### 5.4. Structural-Acoustic and Enclosure Systems

Structural-acoustic ANC involves vibrations of plates, panels, enclosures, and shells that radiate sound. Examples include vibrating plates with multiple actuators, acoustic enclosures backed by clamped plates, encapsulated structures with openings, and machinery housings. These systems are challenging because vibration modes, acoustic radiation, and actuator/sensor placement interact.

Nonlinear and AI methods help when structural dynamics are nonlinear, uncertain, or difficult to model. Azadi and Ohadi (2012) applied fuzzy neural control in an enclosure backed by a clamped plate. Mazur and Pawelczyk (2013) studied ANC using a single nonlinear control filter for a vibrating plate with multiple actuators. Aboutiman et al. (2026) examined hybrid deep-learning ANC for encapsulated structures with openings.

Structural-acoustic applications require careful integration of physical modeling and learning. Purely data-driven models may overfit a single boundary condition. Physics-informed learning, modal features, and uncertainty-aware controllers are promising, but they need validation under changing structural parameters and actuator limits.

### 5.5. Distributed and Multichannel ANC

Distributed ANC uses multiple sensors and actuators across a spatial domain. It is relevant for large enclosures, vehicle cabins, machinery, and sensor networks. Multichannel systems can control more complex fields, but the path matrix grows quickly and communication constraints become important.

The literature includes fast multichannel FxLMS implementations, distributed affine projection algorithms, reduced-complexity random Fourier filters, incremental learning, and nonlinear distributed ANC (Douglas, 1999; Ferrer et al., 2017; Deb et al., 2020; Kukde et al., 2020; Kranthi et al., 2024). AI/ML can reduce complexity by learning lower-dimensional latent states, clustering acoustic conditions, or approximating nonlinear mappings compactly.

The main challenge is coordination. A controller that reduces error at one node may interfere with another. Distributed learning must handle communication delays, asynchronous updates, and partial observations. Benchmarks for this setting are still underdeveloped.

**Table 3. Application scenarios and evaluation priorities.**

| Scenario | Typical disturbances | Useful AI/ML functions | Critical evaluation criteria |
|---|---|---|---|
| Vehicle cabins | Road noise, motor orders, tire noise, broadband cabin noise | State recognition, fixed-filter selection, transfer learning, multichannel mapping | Seat-to-seat robustness, latency, speed/road transitions, perceptual quality |
| HVAC and ducts | Fan tones, broadband flow noise, bend-induced changes | Recurrent modeling, hybrid adaptive filters, nonlinear compensation | Causality, flow noise robustness, actuator limits, changing duct conditions |
| Active headrests | Local low-frequency cabin noise | Head-position/state-aware filter selection, virtual sensing | Quiet-zone size, head movement robustness, switching artifacts |
| Structural-acoustic enclosures | Plate vibration, modal radiation, machinery housing noise | Nonlinear modeling, fuzzy/neural control, physics-informed learning | Modal robustness, sensor/actuator placement, structural uncertainty |
| Distributed ANC | Spatially complex fields, coupled paths | Latent representations, sparse nonlinear filters, incremental learning | Scalability, communication delay, local/global tradeoffs |

## 6. Evaluation Metrics and Comparative Design Tradeoffs

### 6.1. Noise Reduction Is Necessary but Not Sufficient

ANC papers often report reduction in sound pressure level or mean-square error. These metrics are necessary, but insufficient for comparing AI/ML methods. A controller that gives high cancellation in one simulation may be impractical if it requires noncausal context, excessive computation, a perfectly known secondary path, or retraining under every new condition.

Recommended evaluation dimensions include:

1. Acoustic performance: broadband and narrowband reduction, residual spectrum, perceptual quality, and spatial distribution.
2. Causality and latency: algorithmic delay, frame length, lookahead requirements, and hardware processing delay.
3. Robustness: secondary-path mismatch, actuator saturation, sensor noise, reference contamination, acoustic-path drift, and nonstationary disturbances.
4. Complexity: parameter count, multiply-accumulate cost, memory, sampling rate, and embedded feasibility.
5. Adaptation behavior: convergence speed, misadjustment, transient amplification, and stability under changing paths.
6. Generalization: performance across rooms, ducts, seats, head positions, noise types, and hardware.
7. Reproducibility: dataset availability, path measurements, code, hyperparameters, and benchmark protocols.

### 6.2. Comparative Strengths of Method Families

Classical FxLMS remains hard to beat when the system is approximately linear, the reference is good, and the secondary path is well modeled. It is simple, interpretable, and computationally efficient. Nonlinear adaptive filters improve performance when distortions are moderate and structured. Neural and fuzzy-neural methods add flexibility under time-varying nonlinearities. Deep learning is most valuable when the problem requires representation learning, state recognition, multichannel mapping, or filter generation. Generative fixed-filter ANC is promising when continuous online adaptation is too expensive or unstable.

**Table 4. Design tradeoffs among major ANC method families.**

| Method family | Best fit | Less suitable when | Deployment maturity |
|---|---|---|---|
| Linear FxLMS/FxNLMS | Linear, low-frequency, measurable-reference ANC | Strong nonlinearities or rapidly changing paths dominate | High |
| Volterra/bilinear nonlinear filters | Moderate nonlinear paths with need for interpretability | High order, long memory, many channels | Medium |
| FLANN/fuzzy-neural filters | Nonlinear adaptive control with compact expansions | Rigorous stability or certification is required | Medium |
| Evolutionary/swarm optimization | Offline tuning, decentralized design, initialization | Sample-level online control is required | Low to medium |
| Deep ANC controllers | Complex nonlinear mapping and data-rich environments | Training data are sparse or latency is strict | Emerging |
| SFANC/GFANC | Embedded systems with recurring acoustic states | Unseen states are frequent or filter switching is unsafe | Emerging but promising |
| Meta-learning/causal neural ANC | Fast adaptation across tasks and causal modeling | Benchmarks and implementation evidence are limited | Early |

## 7. Technical Challenges and Future Development Trends

### 7.1. Causality and Delay-Aware Learning

Causality is the most fundamental constraint for ANC. Many deep learning models are developed for signal enhancement tasks where future context is available. ANC cannot generally use future samples because the anti-noise must arrive in time to cancel the primary sound. Future research should make delay budgets explicit and train models with causal objectives. Fully causal architectures such as WaveNet-Volterra and parameter-efficient temporal models are therefore important directions (Bai et al., 2025; Li et al., 2026).

### 7.2. Stability and Safe Online Adaptation

Classical adaptive filters have well-studied convergence and stability properties under assumptions. Deep and reinforcement-learning controllers often lack comparable guarantees. ANC systems can produce audible amplification if adaptation becomes unstable. Safe learning methods, bounded updates, Lyapunov-informed training, uncertainty estimation, and fallback controllers should become standard in AI/ML ANC.

### 7.3. Secondary-Path Uncertainty

Secondary-path modeling remains a core bottleneck. Deep learning can decouple or estimate secondary paths, but it cannot eliminate the physical path. Future work should address path drift, online recalibration, uncertainty-aware secondary-path models, and methods that remain stable under bounded mismatch. Hybrid approaches that combine robust adaptive filtering with learned path representations are likely to be stronger than purely black-box methods.

### 7.4. Generalization and Transferability

Many AI/ML ANC studies are evaluated in limited acoustic settings. Real systems vary across rooms, ducts, seats, users, hardware tolerances, and operating conditions. Transferable latent representations and GFANC implementation studies directly address this problem (Shi et al., 2023; Luo et al., 2025). Future benchmarks should include cross-domain splits, not only random train-test splits from the same environment.

### 7.5. Embedded and Edge Implementation

ANC is a real-time embedded problem. Large models may be accurate but impractical. Model compression, quantization, low-rank adaptation, efficient temporal convolutions, and two-timescale architectures will be necessary. A practical design may use a deep network at a slow update rate to choose or generate a filter, while a compact fixed or adaptive filter runs at the audio sampling rate.

### 7.6. Benchmarking and Reproducibility

The field lacks widely adopted benchmarks for AI/ML ANC. Reproducible evaluation should include measured primary and secondary paths, multiple disturbances, defined latency budgets, actuator constraints, sensor noise, and standard metrics. Simulation-only studies should report whether algorithms are causal and whether path mismatch was tested. Without such benchmarks, it is difficult to know whether one AI method is genuinely better or merely evaluated under easier assumptions.

### 7.7. Physics-Guided and Hybrid Learning-Control Systems

The future of AI/ML ANC is likely hybrid. Physics and control theory define what is possible; learning improves adaptation within those limits. Physics-guided neural networks, differentiable secondary-path models, robust adaptive filters, Bayesian uncertainty, and generative filter banks can be combined. The best systems will use learning where it reduces uncertainty or complexity, while preserving causal, stable, and interpretable control loops.

## 8. Conclusions

ANC has evolved from adaptive feedforward control into a broader learning-control field. The foundational constraints of causality, secondary-path dynamics, stability, and low-frequency spatial control remain unchanged. What has changed is the range of tools available for modeling nonlinearities, recognizing acoustic states, selecting or generating filters, and transferring controllers across environments.

The reviewed literature supports four main conclusions. First, FxLMS and related adaptive-filter methods remain essential baselines and often form the inner loop of modern systems. Second, nonlinear adaptive filters, FLANN, fuzzy-neural controllers, and kernel/random-feature methods provide an important bridge between classical ANC and deep learning. Third, deep learning is most compelling when used for representation, secondary-path handling, multichannel mapping, and fixed-filter generation rather than as an unconstrained black-box replacement for control theory. Fourth, the field needs stronger evaluation protocols that test causality, latency, robustness, generalization, and embedded feasibility.

The most defensible research direction is hybrid AI-enabled ANC: systems that use learned models to handle nonlinear and time-varying complexity while retaining the physical discipline of adaptive control. Such systems are more likely to survive the gap between simulation and engineering deployment.

## References

Aboutiman, A., Maamoun, K. S. A., Karimi, H. R., and Ripamonti, F. (2026). Hybrid deep learning-based active noise control for encapsulated structures with openings. *Expert Systems with Applications*. https://doi.org/10.1016/j.eswa.2026.131247

Aboutiman, A., Shams, R., Karimi, H. R., Ripamonti, F., and Pawelczyk, M. (2025). Active noise control in encapsulated structures with non-minimum phase characteristics using a Kalman filter approach. *Journal of Sound and Vibration*. https://doi.org/10.1016/j.jsv.2025.119187

Azadi, N., and Ohadi, A. (2012). Filtered gradient active fuzzy neural network noise control in an enclosure backed by a clamped plate. *International Journal of Adaptive Control and Signal Processing*. https://doi.org/10.1002/acs.1298

Bai, L., Lian, S., Li, M., He, Y., Rao, L., Zeng, X., Sun, R., Chen, K., and Lu, J. (2025). WaveNet-Volterra Neural Network for active noise control: A fully causal approach. *Mechanical Systems and Signal Processing*. https://doi.org/10.1016/j.ymssp.2025.113486

Bambang, R. (2003). Active noise cancellation using recurrent radial basis function neural networks. *Asia Pacific Conference on Circuits and Systems*. https://doi.org/10.1109/apccas.2002.1115201

Bjarnason, E. (1995). Analysis of the filtered-X LMS algorithm. *IEEE Transactions on Speech and Audio Processing*. https://doi.org/10.1109/89.482218

Boucher, C. C., Elliott, S. N., and Nelson, P. A. (1991). Effect of errors in the plant model on the performance of algorithms for adaptive feedforward control. *IEE Proceedings F Radar and Signal Processing*. https://doi.org/10.1049/ip-f-2.1991.0042

Chang, C., and Chen, D. (2010). Active Noise Cancellation Without Secondary Path Identification by Using an Adaptive Genetic Algorithm. *IEEE Transactions on Instrumentation and Measurement*. https://doi.org/10.1109/tim.2009.2036410

Chen, D., Cheng, L., Yao, D., Li, J., and Yan, Y. (2021). A Secondary Path-Decoupled Active Noise Control Algorithm Based on Deep Learning. *IEEE Signal Processing Letters*. https://doi.org/10.1109/lsp.2021.3130023

Chen, D., Yuan, D., Tan, L., and Du, S. (2015). Multichannel active control of nonlinear noise processes using diagonal structure bilinear FXLMS algorithm. *Proceedings of SPIE*. https://doi.org/10.1117/12.2228585

Cheng, C., Liu, Z., Chen, W., Li, X., Liao, W., and Lu, C. (2025). A multi-channel active noise control system using deep learning-based method to estimate secondary path and normalized-clustered control strategy for vehicle interior engine noise. *Applied Acoustics*. https://doi.org/10.1016/j.apacoust.2024.110263

Chu, Y., Xiang, Q., Zhao, S., Wu, M., Zhao, Y., and Yu, G. (2026). Active Noise Control Method Using Time Domain Neural Networks for Path Decoupling. *Digital Signal Processing*. https://doi.org/10.1016/j.dsp.2026.106178

Das, D. P., and Panda, G. (2004). Active Mitigation of Nonlinear Noise Processes Using a Novel Filtered-s LMS Algorithm. *IEEE Transactions on Speech and Audio Processing*. https://doi.org/10.1109/tsa.2003.822741

Das, K. K., and Satapathy, J. K. (2011). Legendre Neural Network for nonlinear Active Noise Cancellation with nonlinear secondary path. https://doi.org/10.1109/mspct.2011.6150515

Deb, T., Ray, D., and George, N. V. (2020). A Reduced Complexity Random Fourier Filter Based Nonlinear Multichannel Narrowband Active Noise Control System. *IEEE Transactions on Circuits and Systems II: Express Briefs*. https://doi.org/10.1109/tcsii.2020.3007999

Douglas, S. C. (1999). Fast implementations of the filtered-X LMS and LMS algorithms for multichannel active noise control. *IEEE Transactions on Speech and Audio Processing*. https://doi.org/10.1109/89.771315

Elliott, S. N., and Nelson, P. A. (1993). Active noise control. *IEEE Signal Processing Magazine*. https://doi.org/10.1109/79.248551

Eriksson, L. J. (1990). The development of the filtered-U algorithm for active noise control. *The Journal of the Acoustical Society of America*. https://doi.org/10.1121/1.2028384

Feng, P., and So, H. C. (2025). Meta-learning-based delayless subband adaptive filter using complex self-attention for active noise control. *Neurocomputing*. https://doi.org/10.1016/j.neucom.2025.130637

Ferrer, M., Gonzalez, A., de Diego, M., and Pinero, G. (2012). Convex Combination Filtered-X Algorithms for Active Noise Control Systems. *IEEE Transactions on Audio, Speech, and Language Processing*. https://doi.org/10.1109/tasl.2012.2215595

Ferrer, M., Gonzalez, A., de Diego, M., and Pinero, G. (2017). Distributed Affine Projection Algorithm Over Acoustically Coupled Sensor Networks. *IEEE Transactions on Signal Processing*. https://doi.org/10.1109/tsp.2017.2742987

George, N. V., and Gonzalez, A. (2013). Convex combination of nonlinear adaptive filters for active noise control. *Applied Acoustics*. https://doi.org/10.1016/j.apacoust.2013.08.005

George, N. V., and Panda, G. (2012). A Particle-Swarm-Optimization-Based Decentralized Nonlinear Active Noise Control System. *IEEE Transactions on Instrumentation and Measurement*. https://doi.org/10.1109/tim.2012.2205492

Guo, X., Li, Y., Jiang, J., Dong, C., Du, S., and Tan, L. (2018). Adaptive Function Expansion 3-D Diagonal-Structure Bilinear Filter for Active Noise Control of Saturation Nonlinearity. *IEEE Access*. https://doi.org/10.1109/access.2018.2876509

Guo, X., Li, Y., Jiang, J., Dong, C., Du, S., and Tan, L. (2018). Sparse Modeling of Nonlinear Secondary Path for Nonlinear Active Noise Control. *IEEE Transactions on Instrumentation and Measurement*. https://doi.org/10.1109/tim.2017.2781992

Holzmuller, F., and Sontacchi, A. (2026). Dynamic Time Warping for Secondary Path Interpolation in Local Active Noise Control. *IEEE Open Journal of Signal Processing*. https://doi.org/10.1109/ojsp.2026.3689448

Holzmuller, F., and Sontacchi, A. (2026). Obs-TasNet: Online Estimation of Virtual Sensing Observation Filters for Active Noise Control. *Acta Acustica*. https://doi.org/10.1051/aacus/2026027

Huynh, M., and Chang, C. (2022). Novel Adaptive Fuzzy Feedback Neural Network Controller for Narrowband Active Noise Control System. *IEEE Access*. https://doi.org/10.1109/access.2022.3167402

Islam, T., Ferrer, M., de Diego, M., and Gonzalez, A. (2026). An Error Innovation-based Adaptive Kalman Filter for multi-channel Active Noise Control. *Mechanical Systems and Signal Processing*. https://doi.org/10.1016/j.ymssp.2026.114113

Khan, W., He, Y., Raja, M. A. Z., Chaudhary, N. I., Khan, Z. A., and Shah, S. M. (2021). Flower Pollination Heuristics for Nonlinear Active Noise Control Systems. *Computers, Materials and Continua*. https://doi.org/10.32604/cmc.2021.014674

Kranthi, R., Vasundhara, Kar, A., and Christensen, M. G. (2024). Charbonnier Quasi Hyperbolic Momentum Spline Based Incremental Strategy for Nonlinear Distributed Active Noise Control. *IEEE Open Journal of Signal Processing*. https://doi.org/10.1109/ojsp.2024.3501774

Li, L., Bhattacharjee, S. S., Wang, S., Jensen, J. R., and Christensen, M. G. (2025). Nearest Kronecker product decomposition based multichannel filtered-x affine projection algorithm for active noise control. *Mechanical Systems and Signal Processing*. https://doi.org/10.1016/j.ymssp.2024.112055

Li, L., Cui, M., Liu, X., Zheng, Y., Wang, S., Jensen, J. R., and Christensen, M. G. (2026). Two-layer Kronecker product decomposition-based robust recursive adaptive filtering for multichannel active noise control. *Mechanical Systems and Signal Processing*. https://doi.org/10.1016/j.ymssp.2026.114053

Kukde, R., Manikandan, M. S., and Panda, G. (2020). Incremental Learning Based Adaptive Filter for Nonlinear Distributed Active Noise Control System. *IEEE Open Journal of Signal Processing*. https://doi.org/10.1109/ojsp.2020.2975768

Kuo, S. M., and Ji, M. (1996). Passband disturbance reduction in periodic active noise control systems. *IEEE Transactions on Speech and Audio Processing*. https://doi.org/10.1109/89.486059

Kuo, S. M., and Wu, H. (2005). Nonlinear adaptive bilinear filters for active noise control systems. *IEEE Transactions on Circuits and Systems I*. https://doi.org/10.1109/tcsi.2004.842429

Le, D. C., Zhang, J., and Pang, Y. (2017). A bilinear functional link artificial neural network filter for nonlinear active noise control and its stability condition. *Applied Acoustics*. https://doi.org/10.1016/j.apacoust.2017.10.023

Le, D. C., Zhang, J., Li, D., and Zhang, S. (2018). A generalized exponential functional link artificial neural networks filter with channel-reduced diagonal structure for nonlinear active noise control. *Applied Acoustics*. https://doi.org/10.1016/j.apacoust.2018.04.020

Le, D. C., Zhang, J., and Li, D. (2019). Hierarchical partial update generalized functional link artificial neural network filter for nonlinear active noise control. *Digital Signal Processing*. https://doi.org/10.1016/j.dsp.2019.07.006

Li, C., Zhuang, J., Du, G., and Yang, J. (2026). TCN-KAN: A Parameter-Efficient and Causal Neural Network for Nonlinear Active Noise Control. https://doi.org/10.1109/iceaai68945.2026.11442464

Li, T., He, Y., Wang, N., Feng, J., Gui, W., and Zhao, K. (2021). Active Noise Cancellation of Rail Vehicles Based on a Convolutional Fuzzy Neural Network Prediction Approach. *IEEE Vehicle Power and Propulsion Conference*. https://doi.org/10.1109/vppc53923.2021.9699325

Liu, Y., Sun, C., and Jiang, S. (2018). Kernel Filtered-x LMS Algorithm for Active Noise Control System with Nonlinear Primary Path. *Circuits, Systems, and Signal Processing*. https://doi.org/10.1007/s00034-018-0832-6

Lorente, J., Ferrer, M., de Diego, M., and Gonzalez, A. (2014). GPU Implementation of Multichannel Adaptive Algorithms for Local Active Noise Control. *IEEE/ACM Transactions on Audio, Speech, and Language Processing*. https://doi.org/10.1109/taslp.2014.2344852

Luo, Z., Shi, D., and Gan, W. S. (2022). A Hybrid SFANC-FxNLMS Algorithm for Active Noise Control Based on Deep Learning. *IEEE Signal Processing Letters*. https://doi.org/10.1109/lsp.2022.3169428

Luo, Z., Shi, D., Gan, W. S., and Huang, Q. (2023). Delayless Generative Fixed-Filter Active Noise Control Based on Deep Learning and Bayesian Filter. *IEEE/ACM Transactions on Audio, Speech, and Language Processing*. https://doi.org/10.1109/taslp.2023.3337632

Luo, Z., Shi, D., Shen, X., Ji, J., and Gan, W. S. (2024). GFANC-Kalman: Generative Fixed-Filter Active Noise Control With CNN-Kalman Filtering. *IEEE Signal Processing Letters*. https://doi.org/10.1109/lsp.2023.3334695

Luo, Z., Shi, D., Ji, J., Shen, X., and Gan, W. S. (2024). Real-time implementation and explainable AI analysis of delayless CNN-based selective fixed-filter active noise control. *Mechanical Systems and Signal Processing*. https://doi.org/10.1016/j.ymssp.2024.111364

Luo, Z., Ma, H., Shi, D., and Gan, W. S. (2024). GFANC-RL: Reinforcement Learning-based Generative Fixed-filter Active Noise Control. *Neural Networks*. https://doi.org/10.1016/j.neunet.2024.106687

Luo, Z., Shi, D., Su, X., and Gan, W. S. (2025). Frequency-Direction Aware Multichannel Selective Fixed-Filter Active Noise Control Based on Multi-Task Learning. *IEEE Transactions on Audio, Speech and Language Processing*. https://doi.org/10.1109/taslpro.2025.3590289

Luo, Z., Ji, J., Wang, B., Shi, D., Ma, H., and Gan, W. S. (2025). Deep learning-based Generative Fixed-Filter Active Noise Control: Transferability and implementation. *Mechanical Systems and Signal Processing*. https://doi.org/10.1016/j.ymssp.2025.113207

Mazur, K., and Pawelczyk, M. (2013). Active Noise Control with a Single Nonlinear Control Filter for a Vibrating Plate with Multiple Actuators. *Archives of Acoustics*. https://doi.org/10.2478/aoa-2013-0063

Nelson, P. A., Hammond, J. K., Joseph, P. F., and Elliott, S. J. (1990). Active control of stationary random sound fields. *The Journal of the Acoustical Society of America*. https://doi.org/10.1121/1.399432

Nelson, P. A., and Elliott, S. J. (1992). Active Noise Control: A Tutorial Review. *IEICE Transactions on Fundamentals of Electronics, Communications and Computer Sciences*.

Nguyen, T. T. T., Na, J., Nguyen, L. T., and Wang, X. (2025). Neuro-Fuzzy Network-Based Nonlinear Hybrid Active Noise Control Systems. *Entropy*. https://doi.org/10.3390/e27020138

Oppenheim, A. V., Weinstein, E., Zangi, K. C., Feder, M., and Gauger, D. (1994). Single-sensor active noise cancellation. *IEEE Transactions on Speech and Audio Processing*. https://doi.org/10.1109/89.279277

Qizhi, Z., and Yongle, J. (2000). Active Noise Feedback Control Using a Neural Network. *Shock and Vibration*. https://doi.org/10.1155/2001/604583

Russo, F. (2006). Genetic Optimization in Nonlinear Systems for Active Noise Control: Accuracy and Performance Evaluation. *IEEE Instrumentation and Measurement Technology Conference*. https://doi.org/10.1109/imtc.2006.236682

Russo, F., and Sicuranza, G. L. (2007). Accuracy and Performance Evaluation in the Genetic Optimization of Nonlinear Systems for Active Noise Control. *IEEE Transactions on Instrumentation and Measurement*. https://doi.org/10.1109/tim.2007.899911

Shi, D., Gan, W. S., Lam, B., Luo, Z., and Shen, X. (2023). Transferable Latent of CNN-Based Selective Fixed-Filter Active Noise Control. *IEEE/ACM Transactions on Audio, Speech, and Language Processing*. https://doi.org/10.1109/taslp.2023.3261757

Shi, D., Gan, W. S., Shen, X., Luo, Z., and Ji, J. (2024). What is behind the meta-learning initialization of adaptive filter? A naive method for accelerating convergence of adaptive multichannel active noise control. *Neural Networks*. https://doi.org/10.1016/j.neunet.2024.106145

Sicuranza, G. L., and Carini, A. (2011). On the BIBO Stability Condition of Adaptive Recursive FLANN Filters With Application to Nonlinear Active Noise Control. *IEEE Transactions on Audio, Speech, and Language Processing*. https://doi.org/10.1109/tasl.2011.2159788

Singh, D., Gupta, R., Kumar, A., and Bahl, R. (2024). Attention-based convolutional neural network for multi-channel active noise control. *NOISE-CON Proceedings*. https://doi.org/10.3397/in_2024_3971

Strauch, P., and Mulgrew, B. (1998). Active control of nonlinear noise processes in a linear duct. *IEEE Transactions on Signal Processing*. https://doi.org/10.1109/78.709529

Tan, L., and Jiang, J. (2001). Adaptive Volterra filters for active control of nonlinear noise processes. *IEEE Transactions on Signal Processing*. https://doi.org/10.1109/78.934136

Thai, N. L., Wu, X., Na, J., Guo, Y., Tin, N. T., and Le, P. X. (2016). Adaptive variable step-size neural controller for nonlinear feedback active noise control systems. *Applied Acoustics*. https://doi.org/10.1016/j.apacoust.2016.09.022

Xiang, Q., Chu, Y., Wu, M., and Yu, G. (2023). A combined method for multi-channel active noise control based on online adaptive estimation and offline convolutional neural network. *The Journal of the Acoustical Society of America*. https://doi.org/10.1121/10.0022860

Xiao, Y., Zhang, Q., Zheng, Y., Qian, J., and Wang, S. (2023). Clustering-Sparse Nystrom Adaptive Filter-Based Nonlinear Distributed Active Noise Control System. *IEEE Transactions on Circuits and Systems II: Express Briefs*. https://doi.org/10.1109/tcsii.2023.3347271

Xu, F., Han, N., and Zhang, T. (2026). A distributed frequency-constrained multichannel active noise control algorithm based on an extended penalty factor via circular convolution. *Signal Processing*. https://doi.org/10.1016/j.sigpro.2025.110240

Yang, F., Guo, J., and Yang, J. (2020). Stochastic Analysis of the Filtered-x LMS Algorithm for Active Noise Control. *IEEE/ACM Transactions on Audio, Speech, and Language Processing*. https://doi.org/10.1109/taslp.2020.3012056

Yang, W., Yin, K., Lu, L., Zhang, G., and Ji, M. (2026). Robust momentum conjugate gradient filter with M-estimation for nonlinear active noise control. *Mechanical Systems and Signal Processing*. https://doi.org/10.1016/j.ymssp.2026.114126

Yu, Y., Lu, L., Zheng, Z., and Yang, X. (2022). Interpolated Individual Weighting Subband Volterra Filter for Nonlinear Active Noise Control. *IEEE Transactions on Circuits and Systems II: Express Briefs*. https://doi.org/10.1109/tcsii.2022.3211280

Zhang, H., and Wang, D. (2021). Deep ANC: A deep learning approach to active noise control. *Neural Networks*. https://doi.org/10.1016/j.neunet.2021.03.037

Zhang, H., and Wang, D. (2022). Deep MCANC: A deep learning approach to multi-channel active noise control. *Neural Networks*. https://doi.org/10.1016/j.neunet.2022.11.029

Zhao, H., Zeng, X., Zhang, X., He, Z., Li, T., and Zhao, W. (2011). Adaptive Extended Pipelined Second-Order Volterra Filter for Nonlinear Active Noise Controller. *IEEE Transactions on Audio, Speech, and Language Processing*. https://doi.org/10.1109/tasl.2011.2175383

Zhu, W., Gu, Z., Chen, X., Xie, P., Luo, L., and Bai, Z. (2025). A New Hybrid Adaptive Self-Loading Filter and GRU-Net for Active Noise Control in a Right-Angle Bending Pipe of an Air Conditioner. *Sensors*. https://doi.org/10.3390/s25206293

Zhu, Y., Zhao, H., Bhattacharjee, S. S., and Christensen, M. G. (2024). Quantized information-theoretic learning based Laguerre functional linked neural networks for nonlinear active noise control. *Mechanical Systems and Signal Processing*. https://doi.org/10.1016/j.ymssp.2024.111348

Zhu, Y., Zhao, H., He, X., Shu, Z., and Chen, B. (2021). Cascaded Random Fourier Filter for Robust Nonlinear Active Noise Control. *IEEE/ACM Transactions on Audio, Speech, and Language Processing*. https://doi.org/10.1109/taslp.2021.3126943
