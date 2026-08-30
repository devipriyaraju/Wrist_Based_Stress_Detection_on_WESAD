# Detecting Stress from Empatica E4 Wrist Signals (WESAD)

Building a stress classifier that actually holds up when tested on people it hasn't seen before.
An applied-ML study of wrist-based stress detection on the [WESAD](https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/)
dataset (15 subjects; Empatica E4 wrist sensors). The question is not "can we classify stress?" but the one an
applied scientist actually has to answer:

> **Given the data we have, what can we conclude with confidence, what remains uncertain, and what should we do next?**

Naive evaluation drastically overstates performance; most of the inflation is **subject-identity leakage**;
the gap can be partly closed by **per-user calibration**; and after that, a simple regularized linear model
generalizes *better* than a gradient-boosted one.

## Key results (leave-one-subject-out, wrist sensors only)

| Setting | macro-F1 |
|---|---|
| Random 5-fold CV, gradient boosting (naive) | **0.86** |
| LOSO, raw features (logreg / hgb) | 0.61 / 0.59 |
| LOSO, + idealized per-user calibration (logreg / hgb) | **0.75 / 0.71** |
| LOSO, + realistic rest-period calibration (~10 min) | **0.73** |
| Chance (majority-class macro-F1) | 0.24 |

- **Leakage is model-dependent.** Gradient boosting looks far better than logistic regression under random
  splitting (0.86 vs 0.69) but generalizes no better (0.59 vs 0.61). Capacity buys leakage, not generalization -
  the random−LOSO gap is 0.28 for boosting vs 0.08 for logistic regression.
- **Subject identity is the leak.** Baseline signals are dominated by *who the person is*: ICC 0.85 (EDA),
  0.95 (skin temperature). Random splits let the model recognise the subject; LOSO exposes it.
- **Simpson's paradox is the mechanism.** `eda_mean` vs `temp_mean` correlates **+0.18 pooled but −0.25
  within-subject** - a sign reversal from between-subject offsets. Removing that offset (per-user z-scoring)
  improves LOSO for **13/15 subjects (median +0.17, Wilcoxon p = 0.008)**.
- **...but calibration is not universally safe.** Two already-easy subjects got *worse* under normalization
  (S14 0.81→0.35), so the recommendation is population-level, not guaranteed per user.
- **Confounding is real and disclosed.** The stress condition carries a motion signature: motion-only features
  reach macro-F1 0.52 (chance 0.24); PPG quality drops under stress (0.62 vs 0.82), so HRV is missing on ~90%
  of stress windows (missing-not-at-random) and is excluded, not imputed.
- **Large per-subject spread** (0.33–0.81) is the distribution-shift story: some people are nearly
  unpredictable from others' data.

See `reports/summary.md` for the one-page narrative and `reports/model_card.md` for the full limitations.

## Repository layout

```
src/            # reusable modules (not notebook-bound)
  config.py       paths, dataset facts, feature policy
  data.py         WESAD .pkl loading + label alignment
  features.py     windowing + EDA/TEMP/HR/ACC extraction (quality-gated PPG)
  stats.py        RM-ANOVA (Greenhouse-Geisser), BH-FDR, Holm, VIF (validated vs statsmodels)
  modeling.py     LOSO, leakage audit, confounding probe, sweep, per-user calibration
notebooks/      # 01 EDA -> 02 features -> 03 leakage/confounding/LOSO -> 05 statistics -> 04 sweep
                # full_run_executed.ipynb : all stages run, with inline figures
outputs/        # tables/ and figures/ (real run outputs)
reports/        # summary.md, model_card.md
```

## Reproduce

```bash
pip install -r requirements.txt
# download WESAD, set src/config.py:DATA_DIR, then run notebooks 01..05 in order, or use the API:
python -c "from src.features import build_feature_matrix; build_feature_matrix().to_csv('outputs/tables/wrist_features_w60_s60.csv', index=False)"
```

## Design choices worth noting
- **Wrist-only** (E4): what a deployable consumer wearable actually has.
- **Non-overlapping windows + label-purity gating**: boundary-straddling windows dropped, not majority-voted.
- **Fold-fit scaling + LOSO** everywhere: no normalization or subject leakage into the numbers.
- **Label-free per-user calibration** (z-score by each user's own rest stats): no target leakage.
- **Dependency-free statistics** (RM-ANOVA/FDR/VIF), validated against statsmodels - no extra cluster installs.

*Data: WESAD (Schmidt et al., ICMI 2018). Not redistributed - download from the source.*
