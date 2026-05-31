# Multimodal RL

## Challenge

$$
s_t = \mathbf{x}_t = \left( x_t^{(1)}, x_t^{(2)}, \dots, x_t^{(M)} \right)
$$

### **Feature Heterogeneity**

Multimodal RL can effectively leverage heterogeneous data streams (eg., vision, audio, proprioception, and text) only when cross-modal representations are properly aligned. Due to inherent differences in temporal resolution, noise characteristics, and semantic granularity across modalities, inadequate alignment induces feature space misalignment and gradient conflicts. This inconsistency disrupts policy optimization, often destabilizing training and significantly delaying convergence [1].

### Dynamic Importance

In practice, optimal decision-making in sequential environments relies on context-dependent modality weighting rather than static strategies. The informational value of each modality fluctuates dynamically across states and task phases due to environmental shifts. 

Failure to dynamically reweight modalities can lead to noisy representations, degraded sample efficiency when irrelevant or degraded signals dominate the state representation.

## Representation Complexity

In sequential decision-making, local errors in state representation or action selection compound over time, leading to significant trajectory divergence. This compounding effect is particularly severe in multimodal RL with a high-dimensional, heterogeneous space. 

Meanwhile,unlike supervised learning with explicit ground-truth labels, RL agents must infer latent state dynamics and causal dependencies solely from scalar reward signals. This absence of direct supervision complicates representation learning. 

### Sample & Computational Inefficiency

As the number of modalities $M$ increases, the computational burden raises due to high-dimensional joint state representations and expensive cross-modal interactions. This not only increases memory consumption but also slows down policy convergence. 

Furthermore, multimodal observations often contain redundancy and modality-specific noise, forcing the agent to require excessive environment interactions to extract task-relevant signals—resulting in poor sample efficiency. 

Since $M$ is typically fixed by hardware configuration or task design, reducing the number of modalities is rarely feasible. Instead, optimization must focus on compressing the effective dimensionality of each $x_t^{(m)}$.

## Related Work

### MR-SP

Multi-modal Reinforcement Learning with Sequence Parallelism (MR-SP) [2] is a framework that reduces memory pressure by using sequence parallelism to balance workload on multiple GPUs.

### DTd

DTd (Decision Transducer) decouples the trajectory into state, action, and goal sequences, each pass through a independent encoder . It then performs hierarchical cross-modal interactions: state-action cross-attention, goal-state and goal-action cross-attention + additive fusion, and finally a causal self-attention over the most important interaction (eg., state-goal) for deep integration. The model is trained end-to-end to predict actions.

However, here are the **Limitation**: Multi-branch attention design incurs high computational overhead and relies on large pre-collected datasets, limiting scalability to online RL or resource-constrained settings.

![](C:\Users\Jiangu\AppData\Roaming\marktext\images\2026-04-21-14-09-53-image.png)

### MAIE

MAIE (Modality Alignment and Importance Enhancement) [1] introduces two mechanisms:

- **Modality alignment** uses auxiliary losses (similarity + temporal discrimination) to align features across modalities.

- **Importance enhancement** computes an importance coefficient based on how much a feature deviates from its running mean/variance. It follows a "deviation-is-informative" principle: the more a feature diverges from its historical statistics, the higher its importance weight. 

However, the paper assumes modalities share the same underlying state attributes, treating them as redundant rather than complementary. It relies solely on similarity metrics for alignment, without modelling synergistic cross-modal interactions, which limits applicability to truly heterogeneous sensor suites.

![](C:\Users\Jiangu\AppData\Roaming\marktext\images\2026-04-21-14-10-41-image.png)

### M2CURL

M2CURL[3] uses a **Transformer-based encoder** combined with **contrastive learning** to improve sample efficiency in multimodal RL. It uses self-supervised contrastive learning to align vision and touch representations from the same timestep while enforcing consistency across augmented views. Encoders process randomly augmented observations, projecting features into shared spaces via intra- and cross-modal losses. The combined multi-modal loss trains modality-invariant representations that feed into a downstream RL agent, improving sample efficiency for robotic manipulation without manual labelling or reward engineering.

However, the contrastive loss is computed over all modality pairs ( $O(M^2)$ complexity ) in the algorithm. As the number of modalities $M$ grows, this leads to higher compute cost and potential gradient conflicts. So far, results are shown only for $M=2$ .

![](C:\Users\Jiangu\AppData\Roaming\marktext\images\2026-04-21-14-11-50-image.png)

#### **Possible Direction**

Existing multimodal RL approaches like MAIE and M2CURL achieve strong performance under a restrictive assumption: modalities are predominantly redundant and should converge to a unified latent space. MAIE relies on static similarity alignment and statistical importance weighting, which cannot adapt to task-driven relevance shifts or resolve contradictory observations. M2CURL improves sample efficiency via contrastive learning, but its exhaustive pairwise objective scales as $O(M^2)$, becoming computationally prohibitive for $M \geq 3$. Crucially, both methods implicitly suppress modality-specific information and lack explicit mechanisms to preserve complementary cues or detect cross-modal inconsistencies.

To address this, the propose is to design a scalable framework for more than 2 heterogeneous modalities. And instead of assuming all modalities capture the same information, the architecture should adapts to how modalities actually relate to each other.

This design should maintains strong sample efficiency while enabling robust, adaptive decision-making—whether the incoming streams are redundant, complementary, or actively inconsistent in dynamic environments.

# Reference

###### [[1] MAIE](https://arxiv.org/pdf/2302.09318)

###### [[2] Scaling RL to Long Videos](https://arxiv.org/pdf/2507.07966)

###### [[3] M2CURL](https://arxiv.org/pdf/2401.17032)

###### [[4] DMR](https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_DMR_Decomposed_Multi-Modality_Representations_for_Frames_and_Events_Fusion_in_CVPR_2024_paper.pdf)

###### [[5] DTd](https://www.auai.org/uai2023/posters/552.pdf)
