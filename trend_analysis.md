# Trend analysis: AI and machine learning in active noise control

## Scope and collection method

This folder contains 110 chronologically ordered papers on active noise control (ANC), active sound control, active noise cancellation, and AI/ML-enabled ANC. The set was built from OpenAlex metadata, targeted DOI/title searches for recent deep-learning ANC papers, and open-access PDF links where available. The bibliography emphasizes 1990s and 2000s classics first, then tracks nonlinear/adaptive/AI methods through the 2010s, and reserves 38 records for recent work from 2020-2026. Citation counts in the CSV are OpenAlex counts, so they should be treated as a screening signal rather than final bibliometric evidence.

Open-access PDF retrieval was conservative. Only files exposed through open PDF URLs were downloaded; many highly regarded ANC papers are behind IEEE, ASA, Elsevier, or society paywalls and therefore are represented by metadata rather than local PDFs. The downloaded files are listed in `pdf_inventory.txt`.

## 1. Classical ANC foundations: adaptive filtering, causality, and secondary paths

The classic period established the control architecture that still constrains modern AI systems. Early work framed ANC as an adaptive feedforward control problem where the controller must learn in the presence of acoustic propagation, actuator dynamics, and secondary-path uncertainty. Nelson et al. (1990) and Eriksson (1990) helped define the physical and algorithmic conditions for active sound control, while Nelson and Elliott's tutorial review and Elliott and Nelson's IEEE overview consolidated the filtered-reference paradigm for the wider signal-processing community (Elliott & Nelson, 1993; Nelson & Elliott, 1992).

The main technical thread was the filtered-x LMS/FxLMS family. Boucher et al. (1991) showed that plant-model errors could degrade adaptive feedforward control, which made secondary-path modeling a central concern rather than an implementation detail. Bjarnason (1995) analyzed FxLMS behavior more directly, and Douglas (1999) addressed fast multichannel implementations. These papers explain why many later neural and fuzzy systems still keep an FxLMS-like loop: the controller must remain causal, stable, and aware of the secondary path.

By the late 1990s and early 2000s, researchers were already moving beyond linear single-channel ANC. Volterra and bilinear filters targeted nonlinear primary paths, secondary paths, and loudspeaker nonlinearities (Strauch & Mulgrew, 1998; Tan & Jiang, 2001; Kuo & Wu, 2005). Neural-network ANC appeared in this period as a way to model nonlinear mappings and feedback dynamics, for example neural feedback control and recurrent radial-basis-function networks (Bambang, 2003; Qizhi & Yongle, 2000). Genetic optimization also entered as an alternative to gradient-only adaptation for nonlinear systems (Russo, 2006; Russo & Sicuranza, 2007). The key trend is that "AI" entered ANC first as nonlinear function approximation and optimization, not as end-to-end deep learning.

## 2. 2010s: nonlinear ANC, fuzzy/neural adaptive filters, and robustness

The 2010s literature is dominated by nonlinear ANC and model-structured learning. Many papers combine classical adaptive-filter insight with neural, fuzzy, Volterra, kernel, FLANN, and evolutionary components. Chang and Chen (2010) used an adaptive genetic algorithm to avoid explicit secondary-path identification, while Zhang and Ren (2010) and Krukowicz (2010) explored neural-network controllers and nonlinear input-output identification. Zhao et al. (2011) and Sicuranza and Carini (2011) focused on Volterra and recursive FLANN stability, reflecting the period's concern with keeping nonlinear controllers analyzable.

The field then diversified into hybrid nonlinear filters. George and Panda (2012) used particle swarm optimization for decentralized nonlinear ANC, Azadi and Ohadi (2012) used active fuzzy neural control in a structural-acoustic enclosure, and George and Gonzalez (2013) combined nonlinear adaptive filters. The 2014-2018 papers show a similar pattern: variable step-size FxLMS, GPU implementation, inverse nonlinear compensation, functional-link artificial neural networks, wavelets, kernels, sparse secondary-path modeling, and random-feature filters were all used to improve convergence, computation, or robustness (Chang & Chu, 2014; Guo et al., 2018; Le et al., 2018; Lorente et al., 2014; Thai et al., 2016; Zhao et al., 2016).

This period also shows a practical shift from "can a nonlinear controller reduce noise?" to "can it do so stably, cheaply, and in multichannel settings?" Multichannel headrests, vehicle powertrain/road noise, infant incubators, vibrating plates, power transformers, and distributed ANC systems appear repeatedly. In other words, AI/ML was being used mainly to handle nonlinearities and adaptation burdens inside otherwise recognizable ANC architectures.

## 3. 2020-2026: deep learning enters ANC, but hybrid control remains dominant

The recent block changes the emphasis. Deep networks are no longer only nonlinear filters inside an adaptive loop; they are used for controller synthesis, fixed-filter selection, transfer learning, secondary-path decoupling, and environmental representation. Zhang and Wang's Deep ANC paper is a clear milestone because it treats ANC as a deep-learning control problem rather than only an adaptive-filter update rule (Zhang & Wang, 2021). Their Deep MCANC extension moved this idea into multichannel ANC (Zhang & Wang, 2022).

At the same time, the most influential recent systems are still hybrid rather than purely end-to-end. Luo et al. (2022) combined selective fixed-filter ANC with FxNLMS adaptation; Luo et al. (2023) introduced delayless generative fixed-filter ANC with Bayesian filtering; Shi et al. (2023) studied transferable latent representations for CNN-based selective fixed-filter ANC. The trend is pragmatic: deep models handle classification, filter generation, latent representation, or initialization, while classical adaptive filtering keeps the real-time acoustic loop stable and interpretable.

Recent papers also target deployment constraints directly. Secondary-path decoupling and deep-learning-assisted multichannel estimation reduce the dependence on continuous error-sensor adaptation (Chen et al., 2021; Xiang et al., 2023). GFANC-RL adds reinforcement learning to generative fixed-filter ANC, showing a move toward policy-based filter selection and adaptation under changing acoustic states (Luo et al., 2024). The 2025 papers extend this line with transferability/implementation of GFANC, meta-learning with complex self-attention, WaveNet-Volterra causal modeling, GRU-based HVAC duct ANC, and neuro-fuzzy hybrid control (Bai et al., 2025; Feng & So, 2025; Luo et al., 2025; Nguyen et al., 2025; Zhu et al., 2025). Early 2026 metadata suggests continued interest in causal neural architectures, Kalman filtering, robust conjugate-gradient learning, and hybrid deep ANC for encapsulated structures (Aboutiman et al., 2026; Islam et al., 2026; Li et al., 2026; Yang et al., 2026).

## 4. Main research themes for the review paper

Four themes are useful for organizing the review.

First, the field is best understood as a progression from adaptive control to learning-assisted control. FxLMS, secondary-path modeling, and causality remain the backbone, while AI methods are layered on top to improve nonlinear modeling, filter selection, or environmental adaptation (Bjarnason, 1995; Boucher et al., 1991; Elliott & Nelson, 1993).

Second, nonlinear ANC is the bridge between classical ANC and modern ML. Volterra, bilinear, FLANN, fuzzy neural, kernel, and random-feature filters provided interpretable nonlinear expansions before deep learning became common (Sicuranza & Carini, 2011; Tan & Jiang, 2001; Zhu et al., 2021). These methods should not be treated as obsolete; they are often more causal, lighter, and easier to analyze than deep networks.

Third, recent deep-learning ANC is shifting from direct waveform cancellation toward learned representations and controller generation. Deep ANC and Deep MCANC provide the conceptual foundation, but the current frontier is generative fixed-filter ANC, transferable latent filters, reinforcement-learning selection, and meta-learned adaptive filtering (Luo et al., 2023, 2024, 2025; Shi et al., 2023; Zhang & Wang, 2021, 2022).

Fourth, evaluation is still a weak point. Many papers report simulation or controlled-lab results, but fewer establish cross-environment generalization, computational latency, actuator saturation robustness, secondary-path drift handling, and reproducible benchmarks. For a review paper, it will be important to compare methods not only by dB reduction but also by causality, real-time complexity, sensor requirements, training-data dependence, and robustness to changing acoustic paths.

## 5. Suggested review-paper structure

1. Foundations of ANC: physical constraints, FxLMS, secondary-path modeling, causality, and multichannel control.
2. Nonlinear ANC before deep learning: Volterra, bilinear, FLANN, fuzzy, kernel, and evolutionary optimization.
3. Neural and neuro-fuzzy ANC: early recurrent/RBF controllers through functional-link and fuzzy neural systems.
4. Deep-learning ANC: Deep ANC, Deep MCANC, secondary-path decoupling, CNN/LSTM/TCN/WaveNet/KAN architectures.
5. Hybrid and generative ANC: selective fixed filters, GFANC, RL-GFANC, Bayesian filtering, transfer learning, and meta-learning.
6. Deployment-oriented comparison: latency, stability, hardware cost, error sensors, robustness, and generalization.
7. Open problems: benchmark datasets, causal training objectives, uncertainty-aware control, safe online learning, and reproducible real-time implementations.

## Bottom line

The literature does not support a simple narrative that deep learning replaces FxLMS. A stronger review thesis is that ANC has become a hybrid learning-control field. Classical adaptive-filter theory supplies the real-time acoustic control constraints; AI and ML supply nonlinear modeling, environment recognition, fixed-filter generation, transferability, and robustness mechanisms. The most promising recent papers are those that respect both sides of the problem: they use learned models where they reduce uncertainty or complexity, while preserving causal, stable, low-latency control loops.
