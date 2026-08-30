"""Validation, leakage audit, confounding probes, and the normalization sweep."""
import itertools
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import f1_score
from .config import TRUSTED_FEATURES


def subject_zscore(df, feats):
    """Label-free per-user calibration: z-score each subject by their own feature mean/std."""
    out = df.copy()
    out[feats] = df.groupby("subject")[feats].transform(lambda c: (c - c.mean()) / (c.std(ddof=0) + 1e-9))
    return out


def make_model(name="logreg", seed=0):
    if name == "logreg":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=5000, class_weight="balanced"))
    return HistGradientBoostingClassifier(random_state=seed, class_weight="balanced")


def loso_per_subject(df, feats, model="logreg", seed=0):
    """Leave-one-subject-out macro-F1, returned per held-out subject."""
    X, y, g = df[feats].values, df["condition"].values, df["subject"].values
    est = make_model(model, seed); per = {}
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        m = clone(est).fit(X[tr], y[tr])
        per[str(g[te][0])] = f1_score(y[te], m.predict(X[te]), average="macro")
    return per


def random_cv_f1(df, feats, model="logreg", seed=0):
    X, y = df[feats].values, df["condition"].values
    est = make_model(model, seed); yp = np.empty_like(y)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        yp[te] = clone(est).fit(X[tr], y[tr]).predict(X[te])
    return f1_score(y, yp, average="macro")


def boot_ci(vals, n=5000, seed=0):
    rng = np.random.default_rng(seed); vals = np.asarray(vals)
    bs = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(n)]
    return float(vals.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def confounding_probe(df, feats=TRUSTED_FEATURES, model="logreg"):
    """LOSO macro-F1 for full / motion-removed / motion-only feature sets."""
    acc = [f for f in feats if f.startswith("acc_")]
    full = np.mean(list(loso_per_subject(df, feats, model).values()))
    no_acc = np.mean(list(loso_per_subject(df, [f for f in feats if f not in acc], model).values()))
    acc_only = np.mean(list(loso_per_subject(df, acc, model).values()))
    return dict(full=full, no_acc=no_acc, acc_only=acc_only)


def run_sweep(df, feats=TRUSTED_FEATURES, splits=("random", "loso"),
              norms=("none", "subject_z"), models=("logreg", "hgb"), seeds=range(10)):
    """Full split x normalization x model x seed sweep. Returns (config_df, per_subject_df)."""
    records, per_subject = [], []
    for sp, nm, mo, sd in itertools.product(splits, norms, models, seeds):
        d = subject_zscore(df, feats) if nm == "subject_z" else df
        if sp == "random":
            f1 = random_cv_f1(d, feats, mo, sd)
        else:
            per = loso_per_subject(d, feats, mo, sd)
            f1 = float(np.mean(list(per.values())))
            per_subject += [dict(split=sp, norm=nm, model=mo, seed=sd, subject=k, macro_f1=v)
                            for k, v in per.items()]
        records.append(dict(split=sp, norm=nm, model=mo, seed=sd, macro_f1=f1))
    return pd.DataFrame(records), pd.DataFrame(per_subject)


# --- deployment-faithful per-user calibration -------------------------------------------------
def baseline_calibrate(df, feats, n_cal=5, cal_condition=1):
    """Per-user calibration from an initial REST period only (deployment-realistic).

    For each subject, estimate feature mean/std from their first `n_cal` windows of `cal_condition`
    (baseline = rest), then z-score all of that subject's windows with those stats. Returns
    (normalized_df, cal_mask) where cal_mask marks the windows spent on calibration — these must be
    excluded from evaluation because they were used to estimate the normalization.
    Uses only that subject's own rest data: no label leakage, no cross-subject leakage.
    """
    df = df.reset_index(drop=True)
    out = df.copy()
    global_sd = df[feats].std(ddof=0)
    cal_idx = []
    for subj, g in df.groupby("subject"):
        cal = g[g.condition == cal_condition].sort_values("t_start").head(n_cal)
        if len(cal) < 2:                       # not enough rest data to calibrate
            cal = g.sort_values("t_start").head(n_cal)
        cal_idx += list(cal.index)
        mu = df.loc[cal.index, feats].mean()
        sd = df.loc[cal.index, feats].std(ddof=0)
        sd = sd.where(sd > 1e-6, global_sd)    # fall back to global scale if a feature is flat in rest
        out.loc[g.index, feats] = (df.loc[g.index, feats] - mu) / (sd + 1e-9)
    return out, df.index.isin(cal_idx)


def loso_baseline_calibrated(df, feats, model="logreg", seed=0, n_cal=5, cal_condition=1):
    """LOSO where each held-out user is calibrated on their own rest period only.
    Calibration windows are excluded from that user's evaluation. Model trains on all other subjects."""
    df = df.reset_index(drop=True)
    dfn, cal_mask = baseline_calibrate(df, feats, n_cal, cal_condition)
    X, y, g = dfn[feats].values, df["condition"].values, df["subject"].values
    est = make_model(model, seed); per = {}
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        te = te[~cal_mask[te]]                 # never score windows used to calibrate
        if len(te) == 0:
            continue
        m = clone(est).fit(X[tr], y[tr])
        per[str(g[te][0])] = f1_score(y[te], m.predict(X[te]), average="macro")
    return per


def fit_deployable(df_train, feats, model="logreg", seed=0):
    """Train a deployable model on per-subject-normalized training data (each train subject z-scored
    by their own stats). At inference, a new user is calibrated separately (see calibrate_and_predict)."""
    dfn = subject_zscore(df_train, feats)
    return make_model(model, seed).fit(dfn[feats].values, df_train["condition"].values)


def calibrate_and_predict(model, feats, cal_windows, new_windows):
    """Deployment inference: estimate a new user's mean/std from their rest `cal_windows`, then
    z-score and classify `new_windows`. Both are DataFrames with the feature columns."""
    mu = cal_windows[feats].mean()
    sd = cal_windows[feats].std(ddof=0).replace(0, np.nan).fillna(1.0)
    Xz = (new_windows[feats] - mu) / (sd + 1e-9)
    return model.predict(Xz.values)
