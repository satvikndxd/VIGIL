# VIGIL Phase 4: Self-Supervised Contrastive Learning for Record-Level ECG Generalization

## Objective

Phase 4 is a **proposed research intervention**, not a reported performance result. Its purpose is to learn a morphology representation from **train-record waveforms only** before supervised five-class fine-tuning. The target failure mode is the observed cross-record N-to-V confusion: a supervised model trained on a limited set of record-specific morphologies may treat an unseen patient’s aberrantly conducted normal beat as ventricular because both can contain a wide, slurred, high-energy QRS complex.

The Phase 1–3 conclusion remains unchanged: the measured final model is the compact **RNN + RR + handcrafted morphology** baseline, with **Macro-F1 0.4012**, **Macro-AUPRC 0.5963**, and **Macro-AUROC 0.7906** on the locked test set. Phase 4 should be accepted only if it passes the same replacement gate: higher held-out Macro-F1, no major Macro-AUPRC degradation, improved N or F recall, and stability across seeds. It must not be described as a successful model before those experiments are run.

| Phase 4 design question | First-principles answer |
|---|---|
| Why pretrain without labels? | Learn morphology under nuisance variation before sparse labels force a narrow decision boundary. |
| Why record-level discipline? | A self-supervised encoder can leak just as readily as a supervised classifier if it sees test records. |
| Why ECG-specific views? | Positive pairs should vary drift, gain, and small dropout without deleting the underlying beat identity. |
| Why not claim expected gains? | Phase 1–3 showed that plausible interventions can fail under OOD evaluation. |
| What remains fixed? | Label mapping, preprocessing contract, grouped split, test set, metrics, gate, and error analysis. |

The core hypothesis is that a contrastively pretrained encoder can learn that waveform amplitude, baseline offset, and small non-central missing spans are often nuisance properties, while QRS morphology and temporal structure remain informative. This does not guarantee domain invariance: the augmentation family defines exactly what the model is asked to ignore. If that family is clinically implausible or overly destructive, the representation can become less useful than the current handcrafted baseline.

## From Absolute Scratch: What SimCLR Learns

A waveform encoder maps one beat-centered ECG window $x \in \mathbb{R}^{L}$ to an embedding $h=f_{\theta}(x) \in \mathbb{R}^{D}$. A second nonlinear **projection head** maps that representation to $z=g_{\phi}(h)$. During self-supervised pretraining, labels are not required. Instead, the system creates two independently augmented views of the same underlying beat:

$$
\tilde{x}^{(1)} = t_1(x), \qquad \tilde{x}^{(2)} = t_2(x), \qquad t_1,t_2 \sim \mathcal{T}_{ECG}.
$$

The encoder receives both views and is optimized to make their projected embeddings similar. Other waveform examples in the mini-batch act as negatives. After pretraining, the projection head is discarded; the encoder representation $h$ initializes a supervised N/S/V/F/Q classifier. This separation matters because the geometry that is optimal for a contrastive loss is not necessarily the geometry that is optimal for five-class classification.[1]

| Stage | Input | Learned object | Labels used? | Output retained for VIGIL |
|---|---|---|---|---|
| Self-supervised pretraining | Train-record waveforms only | ECG encoder + projection head | No | Encoder weights |
| Linear-probe check | Train labels | Frozen encoder + five-class head | Yes | Diagnostic transfer estimate |
| Full fine-tuning | Train labels | Encoder + five-class head | Yes | Candidate checkpoint |
| Selection | Validation records only | Hyperparameter ranking | Yes | Candidate configuration |
| Final evaluation | Locked test records once | Per-class and aggregate metrics | Yes | Gate decision |

The correct mental model is not “the network learns arrhythmia without labels.” It learns a constrained invariance relation: two transformations of the same input should be close in representation space, while two different source windows should be distinguishable. Whether that relation is useful for the five-class task is an empirical question answered only after leakage-safe fine-tuning and evaluation.

## Comic-Book Analogy: Positive and Negative Pairs

> Contrastive learning asks the encoder to recognize a superhero whether the artist draws the hero in a classic suit, a stealth suit, or under different lighting. Those are **positive pairs**: the identity is unchanged even though superficial presentation changes. A villain with a similar color palette is a **negative pair**: surface color similarity must not override the underlying identity.

For VIGIL, the “classic suit” and “stealth suit” are two views of the same ECG beat under modest baseline wander, gain scaling, or a short non-central missing segment. A different beat is a negative pair. The analogy also explains the principal risk: if an augmentation removes the QRS complex or creates a physiologically implausible morphology, it is no longer a new costume; it is a different character. The encoder would then be trained to collapse clinically meaningful variation.

## ECG-Specific Augmentations

The implementation is in `research/phase4_simclr_ecg.py`, where `ECGSimCLRAugment` composes baseline wander, amplitude scaling, and temporal masking. Each augmentation is applied independently to each positive view.

| Augmentation | Implementation | Intended invariance | Safety constraint |
|---|---|---|---|
| **Baseline wander** | Low-frequency sinusoidal offset, amplitude scaled to the beat | Slow electrode/respiratory baseline drift | Drift is modest and does not alter local depolarization shape |
| **Amplitude scaling** | Per-beat multiplicative scale sampled from a narrow range | Gain, lead-contact, and amplitude variation | Scaling range stays close to identity |
| **Temporal masking** | Short mean-filled interval | Local dropout or incomplete observation | Default sampling protects the central beat region |

### Baseline Wander Injection

For a waveform $x[t]$, the transform adds a low-frequency drift:

$$
\tilde{x}[t] = x[t] + a\sin\left(2\pi f\frac{t}{L}+\phi\right),
$$

where $a$ is a small beat-scaled amplitude, $f$ is measured in cycles per window, and $\phi$ is random phase. This models a slowly changing baseline rather than a new cardiac morphology. The implementation uses the waveform standard deviation as a robust magnitude proxy so that the perturbation remains proportionate after VIGIL’s existing normalization.

### Amplitude Scaling

The transform applies

$$
\tilde{x}[t] = \alpha x[t], \qquad \alpha \sim \mathcal{U}(0.85, 1.15).
$$

This is not intended to make the model invariant to every amplitude difference. Large amplitude changes can be clinically meaningful. The narrow range is a hypothesis about modest recording gain and contact variation; it must be ablated against the existing robustness findings, particularly the observed sensitivity to amplitude scaling.

### Protected Temporal Masking

A selected interval $[s,s+w)$ is replaced by the per-channel mean:

$$
\tilde{x}[t] =
\begin{cases}
\operatorname{mean}(x), & s \le t < s+w, \\
 x[t], & \text{otherwise.}
\end{cases}
$$

The mask typically spans a short fraction of the waveform and, where possible, avoids the central beat region. This is a design choice, not a clinical guarantee. Phase 4 should include an ablation comparing protected masking, no masking, and unrestricted masking to establish whether the positive-pair construction preserves the discriminative QRS morphology needed for N/V separation.

## Architecture Walkthrough

### Input Contract

The code accepts tensors shaped `[B, L]` or `[B, C, L]`, where $B$ is batch size, $C$ is lead/channel count, and $L$ is waveform length. VIGIL’s current beat-centered contract corresponds to a single channel and a window of 180 samples. The code does not recompute normalization; callers must pass waveforms produced by the existing VIGIL preprocessing pipeline to prevent train/test inconsistency.

### Pair Dataset

`ECGPairDataset` returns two independently transformed views of the same source waveform. It optionally stores record identifiers for auditing. That design is intentional: self-supervised learning is not exempt from leakage constraints. A checklist should assert that all pretraining `record_ids` belong to the locked training-record list before constructing the loader.

### Encoder

`ECGEncoder1D` is a lightweight **dilated residual convolutional network**. A stem convolution captures local beat morphology, then residual blocks use increasing dilation and conservative downsampling to enlarge the receptive field while preserving QRS detail. Adaptive average pooling maps the resulting feature sequence into a fixed-length embedding $h$.

| Component | Role in the signal path | Why it is appropriate for beat-centered ECG |
|---|---|---|
| Stem convolution | Local edge and slope extraction | Captures sharp QRS transitions and local waveform shape |
| Residual connections | Stable deep optimization | Preserve morphology while adding multi-scale context |
| Dilated convolutions | Larger receptive field | Relate QRS shape to nearby waveform structure without a large model |
| Adaptive pooling | Fixed-dimensional representation | Handles compatible window lengths without flattening assumptions |
| Projection MLP | Contrastive-only embedding | Lets $z$ serve NT-Xent while $h$ transfers to the classifier |

The architecture is deliberately compact. The scientific comparison should not confound “self-supervision helped” with “a much larger encoder helped.” The report must publish its parameter count, training time, and exact hyperparameters next to the existing baseline.

### Projection Head and Downstream Classifier

During pretraining, `ProjectionHead` produces $z=g_{\phi}(h)$. During fine-tuning, `ECGClassifier` attaches a layer-normalized linear N/S/V/F/Q head to $h$ and omits the projection head. The code supports a frozen-encoder linear-probe stage followed by full fine-tuning, making it possible to distinguish representation quality from classifier adaptation.

## NT-Xent / InfoNCE Loss

Given a mini-batch of $B$ source beats, SimCLR creates $2B$ transformed views. Let $z_i$ and $z_j$ be the L2-normalized projected embeddings of the two views of one source beat. Their cosine similarity is

$$
\operatorname{sim}(z_i,z_j) = z_i^\top z_j.
$$

For anchor $i$ and its positive view $j$, the **NT-Xent** loss is

$$
\ell_{i,j} = -\log
\frac{\exp\left(\operatorname{sim}(z_i,z_j)/\tau\right)}
{\sum_{k \ne i}\exp\left(\operatorname{sim}(z_i,z_k)/\tau\right)},
$$

where $\tau > 0$ is the temperature. The final objective averages the two directions of every positive pair:

$$
\mathcal{L}_{NT\text{-}Xent} = \frac{1}{2B}\sum_{i=1}^{2B}\ell_{i,p(i)},
$$

where $p(i)$ maps each view to its paired view. The denominator contains all other views in the batch, so large and diverse train-record batches provide more negatives. The implementation masks self-similarity with negative infinity and uses cross-entropy against the known positive index, which is numerically stable and exactly implements the formula above.[1]

## Heavily Commented Implementation: How to Use It

The complete implementation is provided in `research/phase4_simclr_ecg.py`. It contains the custom transform, pair dataset, 1D residual encoder, projection head, NT-Xent objective, pretraining loop, checkpoint writer, and supervised fine-tuning head. It compiles and passes a tensor-contract smoke test; the smoke test validates shapes and numerical finiteness only and does not constitute an ECG experiment.

```python
from torch.utils.data import DataLoader
from phase4_simclr_ecg import (
    ECGPairDataset,
    ECGSimCLRAugment,
    ECGSimCLR,
    SimCLRConfig,
    pretrain_simclr,
    save_pretrained_encoder,
)

# CRITICAL: X_train_waveforms and train_record_ids must come only from
# VIGIL's locked training records. Do not pretrain on validation or test data.
pair_dataset = ECGPairDataset(
    waveforms=X_train_waveforms,
    transform=ECGSimCLRAugment(),
    record_ids=train_record_ids,
)
train_loader = DataLoader(
    pair_dataset,
    batch_size=128,
    shuffle=True,
    drop_last=True,  # Keeps batch statistics and NT-Xent shape stable.
    num_workers=0,
)

simclr = ECGSimCLR(in_channels=1, embedding_dim=128, projection_dim=128)
config = SimCLRConfig(temperature=0.10, learning_rate=3e-4, epochs=100)
loss_history = pretrain_simclr(simclr, train_loader, config, device="cuda")
save_pretrained_encoder(simclr, "research_results/phase4/simclr_encoder.pt", config)
```

The subsequent fine-tuning procedure must preserve the Phase 1–3 test discipline. Use labeled training records for fitting, choose pretraining and fine-tuning hyperparameters on validation records only, and evaluate the selected candidate once on the unchanged held-out records. The current script includes `ECGClassifier`, `set_encoder_trainable`, and `fine_tune_one_epoch` so this can be implemented without accidentally carrying the contrastive projection head into the classifier.

## Proposed Phase 4 Experiment Matrix

| Experiment ID | Encoder initialization | Positive-pair transform | Fine-tuning regime | Primary question |
|---|---|---|---|---|
| P4-A | Random | None | Supervised five-class | What is the CNN-control performance? |
| P4-B | SimCLR | Wander + scale | Linear probe | Is the frozen representation linearly transferable? |
| P4-C | SimCLR | Wander + scale + protected mask | Linear probe | Does masking improve robustness without deleting QRS information? |
| P4-D | SimCLR | Best validation transform | Full fine-tuning | Does pretraining improve the final candidate? |
| P4-E | SimCLR | Best validation transform | Full fine-tuning, multiple seeds | Is any gain stable? |
| P4-F | Selected candidate | Best validated transform | Locked test analysis | Does it pass the existing replacement gate? |

The correct comparison is not merely Phase 4 versus a weak waveform-only CNN. It must include the current RNN + RR + handcrafted morphology checkpoint, a parameter-aware supervised CNN control, and the same grouped split. Any candidate that improves an in-distribution control but fails the retained baseline on locked OOD data is an informative negative result, not a replacement.

## Evaluation and Failure Criteria

| Required check | Why it matters | Failure interpretation |
|---|---|---|
| Record/subject leakage audit before SSL | Unlabeled test waveforms still leak patient style | Results cannot support OOD claims |
| Matched preprocessing | Prevents normalization from becoming a hidden intervention | Comparison is confounded |
| Five-seed evaluation | Contrastive training can be optimization-sensitive | Single-seed gain is insufficient |
| Bootstrap on selected candidate | Quantifies test-set uncertainty | Point estimate alone is incomplete |
| Per-record/per-symbol errors | Tests the N-to-V root-cause hypothesis directly | Aggregate metric can conceal record collapse |
| Calibration and corruption tests | Invariance may alter confidence and ranking | Better F1 may conceal unsafe probability behavior |

Phase 4 should be rejected if it improves only accuracy, sacrifices Macro-AUPRC materially, leaves N/F recall unchanged, or reduces performance on the hard held-out records while increasing an aggregate statistic. It should also be rejected if the augmentations cause false invariance—for example, if protected masking turns out to suppress precisely the QRS region that distinguishes aberrant conduction from ventricular ectopy.

## Thesis Defense Summary

The Phase 4 story is scientifically conservative: the prior model did not fail because it was too simple in the abstract; it failed because supervised training on a record-level split did not learn a representation sufficiently invariant to patient-specific morphological style. SimCLR-style pretraining creates an explicit testable hypothesis: learn a beat representation that is stable under realistic recording nuisance variation before exposing it to sparse class labels. The code operationalizes that hypothesis, but the VIGIL replacement gate prevents an attractive representation-learning narrative from becoming a claim without evidence.

## References

[1]: https://proceedings.mlr.press/v119/chen20j.html "Chen et al., A Simple Framework for Contrastive Learning of Visual Representations, ICML 2020"

[2]: https://physionet.org/content/mitdb/1.0.0/ "MIT-BIH Arrhythmia Database, PhysioNet"

[3]: https://github.com/satvikndxd/VIGIL "VIGIL repository and finalized Phase 1–3 research artifacts"
