# Target Embedding Change: Text-Only → Text + Image Prototype

## Goal

We want to change how the target embedding used by the attack pipeline is constructed. Instead of relying solely on the text-prompt ensemble (`text_emb[target_cls]`) to define the target direction, we want the target embedding to combine both the multi-template text embedding and an image prototype built from real target-class images. The resulting vector should anchor the attack toward the actual visual region occupied by genuine target-class images, rather than toward a purely linguistic direction that may float far from any real image in CLIP's shared embedding space.

## Motivation

This is motivated by CLIP's well-known modality gap, where text and image embeddings of the same class are typically only moderately aligned (cosine similarity often in the 0.6–0.8 range rather than near 1.0). A text-only target can therefore direct poisoned samples into a region that no real image of the target class actually occupies.

## Risks of Combining Image and Text

We want the agent to be fully aware of the risks involved before proceeding:

1. **ASR misalignment.** The final ASR criterion used by the pipeline is defined as argmax over the text embedding matrix. Steering the attack toward an image prototype may improve visual plausibility and gradient stability while potentially degrading the actual text-based ASR, and there is no guarantee these two objectives stay aligned.

2. **Small, unrepresentative image prototype.** The image prototype is currently built from only about 32 target-class images, which is a small and potentially unrepresentative sample — especially for classes with high visual diversity or noisy backgrounds, as is common in Caltech-101.

3. **Modality-gap cosine as the key diagnostic.** The cosine between the text embedding and the image prototype is the single most important diagnostic. If it is already high (roughly above 0.85), the combination adds little value. If it is low (below 0.6), the combination has a large effect but also a large risk of destabilizing the search.

4. **New hyperparameter.** The weighting between text and image contributions is a new hyperparameter that must be tuned empirically rather than assumed.

## Critical Inconsistency That Must Be Resolved First

Most importantly, we want the agent to understand a critical inconsistency in the current pipeline that must be resolved before any such change is made. The target embedding used for gradient-based ranking (Phase 3) is passed explicitly as `t_target`, but the greedy search phase (Phase 4) does not receive it and instead reconstructs its own target vector directly from the text embedding matrix. As long as `t_target` is text-only, these two are numerically identical and the inconsistency is harmless. However, the moment `t_target` is replaced with a mixed text-plus-image embedding, the ranking phase and the selection phase will be optimizing against different directions. This will cause the top-ranked bit-flip candidates from Phase 3 to receive near-zero or negative scores in Phase 4, driving the greedy search into an immediate plateau with no accepted flips and leaving the final ASR unchanged from its starting value.

## Hard Requirement

We therefore want the agent to treat the propagation of a single consistent target embedding across all phases — ranking, component ablation, hybrid ranking, and greedy search — as a hard requirement. At the same time, the final ASR evaluation itself must remain anchored to the standard text-based classification criterion, so that reported attack success stays comparable to previous runs.