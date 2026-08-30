"""Dependency-free classical statistics (validated to match statsmodels)."""
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression


def rm_anova_oneway(wide):
    """One-way repeated-measures ANOVA with Greenhouse-Geisser. wide: (n_subjects, k_conditions)."""
    wide = np.asarray(wide, float); n, k = wide.shape
    grand = wide.mean()
    ss_cond = n * ((wide.mean(0) - grand) ** 2).sum()
    ss_subj = k * ((wide.mean(1) - grand) ** 2).sum()
    ss_err = ((wide - grand) ** 2).sum() - ss_cond - ss_subj
    df_cond, df_err = k - 1, (k - 1) * (n - 1)
    F = (ss_cond / df_cond) / (ss_err / df_err); p = stats.f.sf(F, df_cond, df_err)
    S = np.cov(wide, rowvar=False); dbar = np.mean(np.diag(S)); gbar = S.mean(); rowbar = S.mean(1)
    eps = np.clip(k ** 2 * (dbar - gbar) ** 2 /
                  ((k - 1) * ((S ** 2).sum() - 2 * k * (rowbar ** 2).sum() + k ** 2 * gbar ** 2)),
                  1 / (k - 1), 1.0)
    return dict(F=F, p=p, partial_eta2=ss_cond / (ss_cond + ss_err),
                gg_epsilon=eps, p_gg=stats.f.sf(F, df_cond * eps, df_err * eps))


def bh_fdr(pvals):
    p = np.asarray(pvals); m = len(p); order = np.argsort(p)
    crit = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m); out[order] = np.minimum(crit, 1.0)
    return out


def holm(pvals):
    p = np.asarray(pvals); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * p[idx]); adj[idx] = min(run, 1.0)
    return adj


def vif(X):
    X = np.asarray(X, float); out = []
    for j in range(X.shape[1]):
        Z = np.delete(X, j, axis=1); r2 = LinearRegression().fit(Z, X[:, j]).score(Z, X[:, j])
        out.append(1 / (1 - r2) if r2 < 1 else np.inf)
    return np.array(out)
