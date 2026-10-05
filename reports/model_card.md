# Model Card - WESAD Wrist Stress Detection

## Intended use
Research / portfolio demonstration of rigorous evaluation for wearable affect detection. **Not** a medical
device and not validated for clinical or real-world stress monitoring.

## Data
WESAD (Schmidt et al., ICMI 2018): 15 subjects (S1, S12 excluded by dataset authors), single lab session each.
Conditions modelled: baseline, stress (Trier Social Stress Test), amusement. Wrist sensors only (Empatica E4:
BVP 64 Hz, EDA 4 Hz, TEMP 4 Hz, ACC 32 Hz). 520 non-overlapping 60 s windows after label-purity gating.
Class balance: baseline 283 / stress 155 / amusement 82.

## Features
16 trusted features: EDA (level, variability, slope, SCR rate), TEMP (level, variability, slope), HR mean,
ACC (variance/energy, not mean, which is dominated by gravity). Evaluated with macro-F1 and balanced accuracy
because of class imbalance.

## Performance (leave-one-subject-out; the estimate)
- Raw features: macro-F1 ~0.59 (hgb) / 0.61 (logreg).
- Per-user calibration (subject z-score): ~0.71 (hgb) / **0.75 (logreg, best)**, 95% CI ~[0.66, 0.83].
- Per-subject range 0.33 (S13) – 0.81 (S14): substantial cross-subject variation.
- Calibration: over-confident (multiclass Brier ~0.46); probabilities should not be taken at face value.
- Confusion: stress is most separable (recall ~0.76); baseline and amusement blur (both low-arousal).

## Known limitations & confounds (the important part)
1. **Subject-identity leakage.** Random-split evaluation overstates macro-F1 by up to ~0.28 (gradient boosting).
   Only leave-one-subject-out reflects generalization to new people. Baseline ICC 0.85 (EDA) / 0.95 (TEMP).
2. **Motion confound.** The stress protocol (TSST) involves standing/speaking; motion-only features reach
   macro-F1 0.52 vs chance 0.24. Part of "stress detection" is motion detection, which will not transfer to
   someone stressed while still, a real distribution-shift risk.
3. **HRV unreliable on E4 wrist PPG.** Beat-to-beat HRV was physiologically implausible even after
   quality-gating; excluded from the model. HRV is also missing-not-at-random (absent on ~90% of stress
   windows because motion degrades PPG), so it must not be imputed.
4. **Calibration assumption.** The 0.75 number assumes per-user calibration data at deployment. Fine for a
   longitudinal product (same user daily); a limitation for cold-start.
5. **Single session, 15 subjects.** Supports cross-*subject* generalization claims but **not** within-subject
   temporal drift over days/weeks — the harder problem for any lifelong-monitoring product.

## What to do next
- Collect multi-session per-subject data to measure temporal (within-subject) distribution shift, not just
  cross-subject.
- Evaluate stress conditions without a motion signature to isolate physiology from movement.
- Prefer the simple regularized linear model + per-user calibration: it generalizes better than the flexible
  model and is interpretable.

## Update - calibration significance & safety (paired analysis)
Per-user normalization improves LOSO macro-F1 for **13/15 subjects** (median Δ +0.17; Wilcoxon signed-rank
one-sided p = 0.008; paired t p = 0.022, weaker because it is dragged by one outlier, so Wilcoxon is primary).
**However, calibration is not universally beneficial:** two already-separable subjects degraded (S14 0.81→0.35,
S17 0.65→0.57). Per-user z-scoring can compress between-condition signal for subjects who were already easy in
raw feature space. The recommendation to calibrate is therefore population-level, and a deployment should
monitor per-user performance rather than assume calibration always helps. A realistic rest-period calibration
(~10 min) recovers most of the idealized gain (~0.73 vs 0.75); shorter calibration (3-5 min) does not.
