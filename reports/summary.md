# Project Summary — When Wearable Stress Models Fail

**One line:** On WESAD wrist data, standard evaluation overstates stress-detection accuracy by ~0.25 macro-F1;
the inflation is subject-identity leakage; per-user calibration recovers most of the gap; and the simple
model then beats the complex one.

## The arc

1. **The baseline is much lower than the literature suggests.** With leave-one-subject-out (LOSO)
   validation — the only kind that reflects generalizing to a *new person* — wrist-based stress detection scores
   ~0.61 macro-F1, not the 0.85+ that random-split evaluation reports.

2. **The gap is subject-identity leakage.** Resting physiology is dominated by *who the subject is*
   (ICC 0.85 for EDA, 0.95 for skin temperature). Under random splitting a model can recognise the person; a
   flexible model (gradient boosting) exploits this and looks great (0.86) while generalizing no better than
   logistic regression (0.59 vs 0.61). Capacity bought leakage, not generalization.

3. **Why, precisely: Simpson's paradox.** The relationship between EDA and skin temperature is *positive* when
   data is pooled across people (r = +0.18) but *negative within* every individual (r = −0.25) — arousal raises
   EDA and lowers peripheral temperature, but between-person offsets flip the pooled sign. Pooled models are
   therefore partly learning subject identity.

4. **The fix: per-user calibration.** Z-scoring each user by their own baseline statistics (label-free, as a
   wearable auto-calibrates) improves LOSO for **13 of 15 subjects (median +0.17, Wilcoxon p = 0.008)**, raising
   macro-F1 to ~0.75. A deployment-realistic version using only a ~10-minute rest period recovers most of it
   (~0.73); a shorter calibration (3–5 min) does not — calibration length matters.

5. **The simple model wins.** After calibration, L2 logistic regression (0.75) beats gradient boosting (0.71),
   which still leaks residual subject structure. The recommendation: a regularized linear model + per-user
   calibration — interpretable, cheap, and the better generalizer.

## What I was careful about!!
- **Motion confound:** the stress protocol involves movement; motion-only features reach 0.52 macro-F1
  (chance 0.24). Part of "stress detection" is motion detection — a distribution-shift risk in the wild.
- **HRV excluded on evidence:** beat-to-beat HRV from E4 wrist PPG was physiologically implausible and missing
  on ~90% of stress windows (missing-not-at-random). It was excluded and *not* imputed.
- **Calibration is not universally safe:** two already-separable subjects got worse under normalization, so the
  gain is population-level, not guaranteed per user.
- **Scope:** 15 subjects, one session each. Supports cross-*subject* claims; does **not** test
  within-subject temporal drift over days — the harder problem for any lifelong-monitoring product.

## What I'd do next (relevance to a longitudinal product)
The unanswered question WESAD can't address is within-subject drift: does a calibrated model stay accurate for
the *same* user across days, weeks, seasons? That requires multi-session per-subject data. For a product that
monitors one person daily over a lifetime, that temporal-shift question — not cross-subject accuracy — is the
one that determines whether the model keeps working.
