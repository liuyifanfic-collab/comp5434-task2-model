# Report Draft — Task 2 Model Side (Member B)

> 团队用说明（定稿时删除）：B 负责章节——模型设计、核心算法、特征融合、结果与消融。数字与 `experiments.md` 台账一一对应。硬件规格与分布式耗时由 C 章节提供。

---

## 3.x Model design (Task 2, model side)

### 3.x.1 Problem framing

Dataset 2: predict a review rating in [1, 5] from 7.68 M training reviews.
Framed as **regression** (RMSE-optimal prediction = conditional mean), continuous outputs clipped to [1, 5]. Rating distribution is skewed (4.25 M of 7.68 M are 5-star; mean 3.842), so the constant-mean baseline already gives validation RMSE 1.5154.

### 3.x.2 Two information sources and a two-stage model evolution

1. **Text block** — what the review says (TF-IDF, later neural embedding).
2. **Structured block** — 19 dense columns of leave-one-out user / product / parent-product statistics, time and engagement fields, produced by a Spark pipeline (data-side chapter).

Our design evolved in two stages, each fully ablated: a **self-implemented linear model** (ridge over fused TF-IDF + statistics), then a **self-implemented neural text model** that became the final solution. The linear stage was not wasted work: it provides the controlled reference against which the neural upgrade and the signal structure were measured.

### 3.x.3 Stage 1 representations: TF-IDF + augmented design matrix

Self-implemented TF-IDF (no sklearn): sub-linear tf `1 + ln(count)`, smoothed idf `ln((1+N)/(1+df)) + 1`, L2-normalised rows; unigrams + bigrams, `min_df = 3`, 30 000 terms. The 30 k cap is empirical: doubling to 60 k *hurt* (val 1.4479 vs 1.4403).

Dense statistics are log1p-transformed on counts, z-standardised, and appended as a **virtual matrix** `[ X_sparse | X_dense ]` exposing the same two kernels the optimiser uses — fusion requires zero changes to the model code.

### 3.x.4 Stage 1 model: ridge by gradient descent

    L(w) = mean((Xw − ȳ)²) + λ‖w‖²,   ∇L = Xᵀ(Xw − ȳ)/n + 2λw,  λ = 1e-4

L2-normalised rows bound the Hessian by the identity, so lr ≤ 1 is unconditionally stable (we use 0.5, decay 0.995, early stopping). The gradient is exactly two sparse map-reduce reductions — the same shape Spark would parallelise.

### 3.x.5 Stage 2 model: neural text regressor (final solution)

    mean(token embeddings, 64-d) ⊕ 19 standardised dense features
        → Linear(83 → 256) → ReLU → Linear(256 → 1)

Vocabulary: 100 001 tokens (min_df = 3, full train fold), `<unk>` fallback.
**Compliance**: PyTorch is used only as a tensor/autograd engine — the architecture, loss, batching, and training loop (Adam, MSE, early stopping, per-epoch timing) are written from scratch; no ready-made estimator is called. PyTorch here plays the role numpy plays in the ridge model.

A tuning sweep fixed the capacity: doubling width/depth or doubling the learning rate all degraded validation RMSE (1.1946 / 1.1987 vs 1.1914), and a bigram vocabulary also lost (1.1959, overfitting after 5 epochs) — the unigram mean-pooling model is the sweet spot.

## 4.x Implementation details

- **Reproducibility**: single config source, team seed 5434, deterministic vocabulary, frozen canonical split (`split/val_ids.csv`), verified set-identical to the `is_val` column.
- **Leakage discipline**: TF-IDF / scaler fitted on training rows only; statistics are leave-one-out over the train fold (§5.x.4).
- **Efficiency instrumentation**: per-epoch times inside the training loop; offline and inference timings at pipeline entry points.

Final full-data run (Plan A: Spark feature generation → local training,
single-thread pinned for cross-machine bit-level reproducibility):

| Metric | Value |
|---|---|
| Training rows / validation rows | 6 916 181 / 768 464 |
| Training time per epoch (9 epochs, 1 thread) | **169.8 s avg** |
| Offline total (load + vocab + encode + dense + train) | **1800.4 s** |
| Inference, all 10 000 test records | **0.03 s** |
| Hardware | CPU-only (single machine); feature generation on Spark, see data-side chapter |

Reproducibility hardening: multi-threaded CPU reductions are not
deterministic across machines (float summation order varies), which we
observed as a 0.002 LB difference on an independent re-run. The final
pipeline pins execution to a single thread and pins library versions;
two prediction runs are then byte-identical, and the trained weights are
shipped with the code so the submission file is exactly regenerable.

## 5.x Performance evaluation and discussion

### 5.x.1 Main results (canonical split; pilot = 10 % training-fold sample)

| Model | Features | Val RMSE | Kaggle LB |
|---|---|---|---|
| Constant mean | — | 1.5154 | — |
| Structured only, closed-form ridge | 19 dense | 1.4682 | — |
| Linear ridge (stage 1, tuned) | TF-IDF + 19 | 1.4033 | 1.20087 (v4) |
| Neural (stage 2, pilot) | emb + 19 | 1.1914 | 0.79 (v5) |
| **Neural (stage 2, full data, deterministic)** | emb + 19 | **1.1705** | **0.773 (v6 0.77300; v7 = deterministic re-run, final)** |

Public-leaderboard trajectory: 1.30841 (v1, text-only linear) → 1.23869 → 1.20087 → 0.79 → 0.77300.

### 5.x.2 Ablation over feature groups (linear stage, uniform protocol)

| Group | Dense columns | Val RMSE | Δ vs text |
|---|---|---|---|
| text | — | 1.4648 | — |
| text + user | 6 | 1.4525 | −0.012 |
| text + prod/parent | 9 | 1.4418 | −0.023 |
| text + all | 19 | 1.4301 | −0.035 |

### 5.x.3 Signal absorption (stage 2 ablation)

| Model | Text only | + 19 dense | Feature contribution |
|---|---|---|---|
| Linear ridge | 1.4403 | 1.4033 | −0.035 |
| **Neural** | 1.1923 | 1.1914 | **−0.001** |

The neural embedding learns, from text alone, most of what the structured features encode — reviewer vocabulary/style and product themes are visible in the text. Feature-engineering signal being absorbed by representation learning is a direct empirical demonstration of the feature-engineering vs. representation-learning trade-off; the structured block is retained as a cheap, robust prior.

### 5.x.4 Findings

1. **Text saturates for linear models, not for data**: tripling linear training data changed nothing (1.4479 → 1.4478); the neural model gained −0.021 from full data. The linear ceiling is a *representation* limit.
2. **Leaderboard amplification**: val→LB offset grew from −0.15 (v1) to −0.20 (v4) to −0.40 (neural). Linear-era offset is explained by user-history coverage (test 95.7 % vs train 54.8 %); the neural-era offset indicates richer text signal in the test distribution (active users).
3. **Leakage control is worth real RMSE**: the inclusive historical mean equals a row's own label for single-review users (train correlation 0.778 vs val 0.115). Leave-one-out statistics + Bayesian shrinkage recovered 0.039 RMSE and made validation trustworthy.
4. **Structural shortcuts ruled out by direct measurement** (before attributing the leaderboard gap to model class): (user, product) pair lookups match 0.00 % of test rows — the dataset is deliberately constructed against memorisation; exact-text matches cover 7.8 % at lookup RMSE 1.28 (templated texts exist, one repeated 7 792 times, but their ratings vary). The 0.45–0.50 cluster is therefore attributable to pre-trained language models, which we excluded on compliance grounds.

### 5.x.5 Limitations and future work

- Mean-pooled unigram embeddings cap the text representation; attention-based or sub-word models are the natural next step.
- `votes` is post-hoc (accumulated after publication): legitimate here, unavailable in a real-time deployment.
- Random (not temporal) validation split slightly flatters the estimate.

---

*Numbers ledger: `experiments.md` (every row maps to a logged run). Efficiency/hardware for the Spark side: `features/stage3_report.json` (data-side chapter).*
