# SKILL.md — tabular-playground-series-may-2022 (TPS-May-2022)

## One-line positioning
Binary classification (`target` 0/1), evaluation metric **ROC-AUC**, submit `id,target` (target can be a probability/rank score — AUC only cares about ordering). **The score bands are extremely narrow, and you can cross tiers purely by reducing variance.** The pass line for this run is in `criterion.json` — beat it rather than any hardcoded score.

---

## Winning recipe (directly executable)

**Model: a PyTorch-equivalent reimplementation of the public AmbrosM "TPSMAY22 Keras MLP"**, with a hand-written training loop and GPU-resident tensors.

1. **Features (44, defer to the actual code `add_features`)**:
   - 33 original numeric columns `f_00..f_30` (excluding `id/target/f_27`).
   - `f_27` is a 10-character string (each position A–T): split it position-by-position into 10 ordinals `ch0..ch9 = ord(c)-ord('A')`.
   - `unique_characters = len(set(f_27))`.
   - **3 three-valued magic interactions** (±thresholded, values ∈ {-1,0,1}; the thresholds are fixed magic numbers — do not touch them):
     - `i_02_21 = (f_21+f_02>5.2) - (f_21+f_02<-5.3)`
     - `i_05_22 = (f_22+f_05>5.1) - (f_22+f_05<-5.4)`
     - `i_00_01_26`: `s=f_00+f_01+f_26`, `(s>5.0)-(s<-5.0)`
2. **Preprocessing**: per-fold/per-run `StandardScaler` fit on train, transform val/test.
3. **Network (`ExactMLP`)**: `64→128→64→32→16→1`, with the activation sequence strictly `SiLU, (PReLU or ReLU), ReLU, SiLU, SiLU`, Glorot/xavier_uniform initialization, bias=0.
4. **Training hyperparameters**: Adam, `epochs=200`, **cosine LR `0.01→0.00005`** (`cosine_epochs=150`), `batch=2048`, **L2 penalty `40e-6`** (added to the loss by hand, not weight_decay), BCEWithLogits.
5. **CV**: `KFold(n_splits=5, shuffle=False)` (note: unshuffled sequential folds, to match the reference implementation). CV mode produces the OOF self-estimated score; **the final submission uses `full` mode** (train on all data, no held-out val).
6. **Ensemble (the core of winning)**: run `full` mode with **multiple seeds**; each seed's test prediction is **rankdata/N normalized** then equal-weight averaged; persist to disk as `test_rank_seed_{i}.npy`. Deepen step by step: 10-seed → 15-seed → 25-way; each added batch of seeds moves the score a little further.
7. **Fallback**: on entry the script first writes a constant fallback `submission.csv = train.target.mean()`, guaranteeing a submission exists whenever rc=0.

---

## The tricks that actually move the metric (the most valuable part)

**The jump-off point = strictly aligning with the "specification" of the public Keras implementation** — replacing the home-grown MLP with the equivalent reimplementation of KFold/Glorot/L2=40e-6/Adam/cosine 0.01→5e-5. Before this it was stuck on a plateau for a dozen-plus moves without breaking through.

1. **Specification alignment > adding bells and whistles.** The progression was: home-grown MLP missing the magic features → features added but with a self-invented specification → copying the public implementation's specification (the jump-off). The affliction was always "the base model not reaching the reference score" — not blending, not CV, not missing features.
2. **Seed rank ensemble reduces variance to cross tiers.** Once the base model reaches the first competitive band, **multi-seed of the same model family + rank averaging** is pure variance reduction: the tiers differ by only a few 1e-5, and adding seeds crosses the line. Treating historical anchors as several "proxy seed votes" and rank-averaging them together (`pred = (5*anchor + Σ seed_ranks)/N`) also works.
3. **Rank-normalize then average** is more stable than averaging probabilities directly (AUC only cares about ordering; when cross-seed scales are inconsistent, rank flattens the scale).

### Cautionary lessons (don't repeat these)
- ❌ **Don't touch GBDT as the primary model**: several consecutive moves claimed to bring in XGBoost, but in practice it only landed a lower AUC and dragged the blend down. For this task, **NN > GBDT** is settled.
- ❌ **Don't repeatedly tear down and rebuild the pipeline**: the root cause of many consecutive moves without improvement was "execution not landing / specification inconsistent", not missing features. Force reuse of one verified reference implementation that runs end-to-end before changing anything.
- ❌ **Don't repeatedly do submission-level stack/calibration/logit-blend on the plateau**: several moves doing OOF model selection on the anchor all stayed put — when the base model is subpar, blending is useless.
- ❌ **Don't keep doing ablations after the feature matrix is confirmed correct**: several ablation moves already proved the 44-feature matrix correct; continuing to ablate wastes moves.
- ☠️ **Most fatal**: early on, five or six moves were "right direction but the code never actually landed" (compounded by an environment issue) — the magic features stayed talk and never entered the training matrix → you **must write them into the code and verify the output this very move**.

---

## Key code snippets (excerpted from the actual working code)

**Feature engineering (fixed thresholds, copy verbatim):**
```python
def add_features(df):
    df = df.copy()
    for i in range(10):
        df[f"ch{i}"] = df.f_27.str.get(i).map(lambda c: ord(c)-ord("A")).astype(np.int16)
    df["unique_characters"] = df.f_27.map(lambda s: len(set(s))).astype(np.int16)
    df["i_02_21"] = (df.f_21+df.f_02>5.2).astype(np.int8) - (df.f_21+df.f_02<-5.3).astype(np.int8)
    df["i_05_22"] = (df.f_22+df.f_05>5.1).astype(np.int8) - (df.f_22+df.f_05<-5.4).astype(np.int8)
    s = df.f_00 + df.f_01 + df.f_26
    df["i_00_01_26"] = (s>5.0).astype(np.int8) - (s<-5.0).astype(np.int8)
    features = [f for f in df.columns if f not in ("id","target","f_27")]
    return df, features
```

**Network head / forward (the activation sequence is key, don't change it):**
```python
class ExactMLP(nn.Module):
    def __init__(self, n_features, init_seed=42):
        super().__init__()
        self.d1=nn.Linear(n_features,64); self.d2=nn.Linear(64,128)
        self.d3=nn.Linear(128,64); self.d4=nn.Linear(64,32)
        self.d5=nn.Linear(32,16); self.out=nn.Linear(16,1)
        for layer in [self.d1,self.d2,self.d3,self.d4,self.d5]:
            gen=torch.Generator("cpu"); gen.manual_seed(init_seed)
            nn.init.xavier_uniform_(layer.weight, generator=gen); nn.init.zeros_(layer.bias)
    def l2_penalty(self):
        return sum((l.weight**2).sum() for l in [self.d1,self.d2,self.d3,self.d4,self.d5])
    def forward(self,x):
        x=F.silu(self.d1(x)); x=F.relu(self.d2(x))
        x=F.relu(self.d3(x)); x=F.silu(self.d4(x)); x=F.silu(self.d5(x))
        return self.out(x).squeeze(1)
```

**cosine LR + L2 into loss:**
```python
def lr_for_epoch(e, lr0, lr1, T):
    w=(1+math.cos((e%T)/(T-1)*math.pi))/2
    return w*lr0+(1-w)*lr1
# Training step:
bce = F.binary_cross_entropy_with_logits(logits, y)
loss = bce + 40e-6 * model.l2_penalty()
```

**Multi-seed rank ensemble (full mode, the decisive move):**
```python
for seed in seeds:
    test_pred = fit_one(... seed ...)        # train on all data
    ranked = rankdata(test_pred) / len(test_pred)
    pred_sum += ranked / len(seeds)
    np.save(f"test_rank_seed_{seed}.npy", ranked)   # persist for later deepening
# Treat the historical anchor as 5 proxy votes and average together:
# pred = (5*anchor_target + Σ all_seed_ranks) / (5 + n_seeds)
sub["target"] = pred_sum
```

**Submission validation (do this every time):** shape `[100000,2]`, columns `[id,target]`, `id` aligned with sample, all finite.

---

## Score milestones (self-estimated reference — relative ordering)
| Stage | Notes |
|---|---|
| Constant fallback | `target=mean`, only guards against rc≠0 |
| Home-grown MLP missing magic features | far below the competitive band |
| Features added but self-invented spec (plateau) | the dead plateau stuck for a dozen-plus moves |
| **Copy public Keras spec** | the jump-off point into the competitive band |
| Seed rank ensemble (10 → 15 → 25 seeds) | each batch of seeds shaves a little more variance off |

Once the base model reaches the first competitive band, **you only need to add seed depth to reduce variance to cross further** — don't switch architectures, don't touch GBDT, don't tear down and rebuild. Self-check against the `criterion.json` pass line, not a fixed threshold.
