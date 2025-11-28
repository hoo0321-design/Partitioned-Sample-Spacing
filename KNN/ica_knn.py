# ================================================================
# ica_nf_um_full_final.py — EEG Eye State + ICA + KL/KSG/tKL/tKSG (UM+Jacobian) + CV
# ================================================================

# ---------- 0) Compatibility ----------
import sys, os, warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    import numpy as np
    if not hasattr(np, "bool"):
        np.bool = bool  # safe alias

# Python 2 → 3 shims for legacy modules
import builtins, itertools
if not hasattr(itertools, "izip"):
    itertools.izip = zip

# ---------- 1) Imports ----------
import pandas as pd
from math import pi
from time import perf_counter
from scipy.io import arff
from scipy.stats import norm
from scipy.special import gamma as _gamma
from sklearn.decomposition import FastICA
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

# ---------- 2) Entropy estimators ----------
try:
    from ent_est.entropy import kl, ksg, tkl, tksg
    print("[ok] Imported kl, ksg, tkl, tksg from ent_est.entropy")
except Exception as e:
    print("[ERR] Could not import from ent_est.entropy:", e)
    print("      Ensure your repo is on PYTHONPATH (sys.path.insert(0, '<repo_root>'))")
    sys.exit(1)

# ================================================================
# 3) Data loading and whitening
# ================================================================
def load_eeg_eye_state(path_arff: str) -> np.ndarray:
    """Load UCI EEG Eye State ARFF → X (drop label)."""
    data, _ = arff.loadarff(path_arff)
    df = pd.DataFrame(data)
    for c in df.columns:
        if df[c].dtype == object and len(df[c]) and isinstance(df[c].iloc[0], (bytes, bytearray)):
            df[c] = df[c].str.decode('utf-8', errors='ignore')
    if df.shape[1] >= 15:
        X = df.iloc[:, :-1].to_numpy(dtype=float)
    else:
        X = df.to_numpy(dtype=float)
    return X

def whiten_data(X: np.ndarray) -> np.ndarray:
    """Zero-mean + PCA whitening."""
    X = np.asarray(X, float)
    Xc = X - X.mean(axis=0, keepdims=True)
    C  = np.cov(Xc, rowvar=False)
    lam, V = np.linalg.eigh(C)
    lam = np.clip(lam, np.finfo(float).eps, None)
    Wih = V @ np.diag(1.0/np.sqrt(lam)) @ V.T
    return Xc @ Wih

# ================================================================
# 4) Utility functions: UM(Φ), Jacobian, kNN-NLL, jitter
# ================================================================
def _unit_ball_volume_euclidean(d: int) -> float:
    return pi ** (d / 2.0) / _gamma(d / 2.0 + 1.0)

def _to_U_via_phi(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """UM transform: U = Φ(X) per-dim; clamp away from {0,1}."""
    U = norm.cdf(X)
    return np.clip(U, eps, 1 - eps)

def _jac_logdet_phi_inv_per_sample(X: np.ndarray) -> np.ndarray:
    """log|det ∂Φ^{-1}/∂U| = - ∑_j log φ(X_ij); returns (n,)"""
    return -np.sum(norm.logpdf(X), axis=1)

def _jitter_like(X: np.ndarray, scale=1e-9, seed=2025) -> np.ndarray:
    """Add tiny dequantization noise to break ties."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float)
    s = np.std(X, axis=0, ddof=1)
    s = np.where(s > 0, s, 1.0)
    J = rng.normal(0.0, 1.0, size=X.shape) * (scale * s)
    return X + J

def _knn_log_density_trainval(X_tr: np.ndarray, X_val: np.ndarray, k: int, metric: str) -> np.ndarray:
    """
    Held-out log f(x) with kNN balls.
      euclidean : log f = log k - log n_tr - log V_d - d*log r_k
      chebyshev : log f = log k - log n_tr - d*(log 2 + log r_k)
    """
    X_tr = np.asarray(X_tr, float); X_val = np.asarray(X_val, float)
    n_tr, d = X_tr.shape
    k_use = max(1, min(int(k), n_tr))
    nn = NearestNeighbors(n_neighbors=k_use, algorithm="auto", metric=metric).fit(X_tr)
    dist, _ = nn.kneighbors(X_val, n_neighbors=k_use, return_distance=True)
    rk = np.maximum(dist[:, -1], np.finfo(float).eps)
    if metric == "euclidean":
        return np.log(k_use) - np.log(n_tr) - np.log(_unit_ball_volume_euclidean(d)) - d * np.log(rk)
    elif metric == "chebyshev":
        return np.log(k_use) - np.log(n_tr) - d * (np.log(2.0) + np.log(rk))
    else:
        raise ValueError("metric must be 'euclidean' or 'chebyshev'")

# ================================================================
# 5) Cross-validation for k selection
# ================================================================
def select_k_via_cv_X(X: np.ndarray, k_grid=(1,2,3,4,5,7,10,15), n_splits=3, seed=2025, metric="euclidean") -> int:
    """CV on X-space (KL/KSG)."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for k in k_grid:
        nll_sum, cnt = 0.0, 0
        for tr_idx, te_idx in kf.split(X):
            logf = _knn_log_density_trainval(X[tr_idx], X[te_idx], k=k, metric=metric)
            ok = np.isfinite(logf)
            nll_sum += -np.sum(logf[ok]); cnt += int(np.sum(ok))
        score = np.inf if cnt == 0 else (nll_sum / cnt)
        scores.append((int(k), float(score)))
    scores.sort(key=lambda t: (t[1], t[0]))
    return scores[0][0]

def select_k_via_cv_UM(X: np.ndarray, k_grid=(1,2,3,4,5,7,10,15), n_splits=3, seed=2025, jitter_scale=1e-9, eps=1e-8) -> int:
    """CV for UM-based estimators (tKL/tKSG)."""
    U = _to_U_via_phi(X, eps=eps)
    U = _jitter_like(U, scale=jitter_scale, seed=seed)
    logphi_sum = np.sum(norm.logpdf(X), axis=1)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for k in k_grid:
        nll_sum, cnt = 0.0, 0
        for tr_idx, te_idx in kf.split(U):
            logc = _knn_log_density_trainval(U[tr_idx], U[te_idx], k=k, metric="chebyshev")
            logp = logc + logphi_sum[te_idx]
            ok = np.isfinite(logp)
            nll_sum += -np.sum(logp[ok]); cnt += int(np.sum(ok))
        score = np.inf if cnt == 0 else (nll_sum / cnt)
        scores.append((int(k), float(score)))
    scores.sort(key=lambda t: (t[1], t[0]))
    return scores[0][0]

# ================================================================
# 6) Entropy → TC (with UM correction)
# ================================================================
def _H_UM_tKL(X: np.ndarray, k: int, seed=2025, jitter_scale=1e-9, eps=1e-8) -> float:
    U = _to_U_via_phi(X, eps=eps)
    U = _jitter_like(U, scale=jitter_scale, seed=seed)
    Hc = tkl(U, k=k)
    corr = np.mean(_jac_logdet_phi_inv_per_sample(X))
    return Hc + corr

def _H_UM_tKSG(X: np.ndarray, k: int, seed=2025, jitter_scale=1e-9, eps=1e-8) -> float:
    U = _to_U_via_phi(X, eps=eps)
    U = _jitter_like(U, scale=jitter_scale, seed=seed)
    Hc = tksg(U, k=k)
    corr = np.mean(_jac_logdet_phi_inv_per_sample(X))
    return Hc + corr

def _TC_from_H(X: np.ndarray, H_func, k: int, base_seed=2025) -> float:
    """TC = Σ_j H(X_j) - H(X)."""
    d = X.shape[1]
    H_joint = H_func(X, k, seed=base_seed)
    H_marg  = 0.0
    for j in range(d):
        H_marg += H_func(X[:, [j]], k, seed=base_seed + 1 + j)
    return H_marg - H_joint

# ================================================================
# 7) Main pipeline
# ================================================================
def run_ica_tc_four_estimators_um(path_arff: str,
                                  k_grid=(1,2,3,4,5,7,10,15,20,30),
                                  n_splits=3,
                                  seed=2025,
                                  jitter_scale=1e-9,
                                  eps=1e-8):
    print("[1/5] Loading ARFF & whitening...")
    X  = load_eeg_eye_state(path_arff)
    Xw = whiten_data(X)
    n, d = Xw.shape
    print(f"    shape: n={n}, d={d}")

    print("[2/5] Selecting k* via CV ...")
    k_KL   = select_k_via_cv_X(Xw, k_grid=k_grid, n_splits=n_splits, seed=seed, metric="euclidean")
    k_KSG  = select_k_via_cv_X(Xw, k_grid=k_grid, n_splits=n_splits, seed=seed, metric="chebyshev")
    k_tKL  = select_k_via_cv_UM(Xw, k_grid=k_grid, n_splits=n_splits, seed=seed, jitter_scale=jitter_scale, eps=eps)
    k_tKSG = select_k_via_cv_UM(Xw, k_grid=k_grid, n_splits=n_splits, seed=seed, jitter_scale=jitter_scale, eps=eps)
    print(f"    k*: KL={k_KL}, KSG={k_KSG}, tKL={k_tKL}, tKSG={k_tKSG}")

    print("[3/5] TC before ICA ...")
    t0 = perf_counter(); TC_KL_before   = _TC_from_H(Xw, lambda A,k,seed: kl(A,k=k), k_KL, base_seed=seed);   tKLB  = perf_counter()-t0
    t0 = perf_counter(); TC_KSG_before  = _TC_from_H(Xw, lambda A,k,seed: ksg(A,k=k), k_KSG, base_seed=seed);  tKSGB = perf_counter()-t0
    t0 = perf_counter(); TC_tKL_before  = _TC_from_H(Xw, lambda A,k,seed: _H_UM_tKL(A,k,seed,jitter_scale,eps), k_tKL, base_seed=seed);  ttKLB  = perf_counter()-t0
    t0 = perf_counter(); TC_tKSG_before = _TC_from_H(Xw, lambda A,k,seed: _H_UM_tKSG(A,k,seed,jitter_scale,eps), k_tKSG, base_seed=seed); ttKSGB = perf_counter()-t0

    print("[4/5] Running FastICA ...")
    ica = FastICA(n_components=d, whiten=False, random_state=seed, max_iter=1000, tol=1e-6)
    t0 = perf_counter()
    Y = ica.fit_transform(Xw)
    tICA = perf_counter() - t0
    print(f"    ICA time: {tICA:.2f}s")

    print("[5/5] TC after ICA ...")
    t0 = perf_counter(); TC_KL_after   = _TC_from_H(Y, lambda A,k,seed: kl(A,k=k), k_KL, base_seed=seed);   tKLA  = perf_counter()-t0
    t0 = perf_counter(); TC_KSG_after  = _TC_from_H(Y, lambda A,k,seed: ksg(A,k=k), k_KSG, base_seed=seed);  tKSGA = perf_counter()-t0
    t0 = perf_counter(); TC_tKL_after  = _TC_from_H(Y, lambda A,k,seed: _H_UM_tKL(A,k,seed,jitter_scale,eps), k_tKL, base_seed=seed);  ttKLA  = perf_counter()-t0
    t0 = perf_counter(); TC_tKSG_after = _TC_from_H(Y, lambda A,k,seed: _H_UM_tKSG(A,k,seed,jitter_scale,eps), k_tKSG, base_seed=seed); ttKSGA = perf_counter()-t0

    results = {
        "dims": dict(n=n, d=d),
        "k_star": dict(KL=k_KL, KSG=k_KSG, tKL=k_tKL, tKSG=k_tKSG),
        "TC_before": dict(KL=TC_KL_before, KSG=TC_KSG_before, tKL=TC_tKL_before, tKSG=TC_tKSG_before),
        "TC_after":  dict(KL=TC_KL_after,  KSG=TC_KSG_after,  tKL=TC_tKL_after,  tKSG=TC_tKSG_after),
        "times_before": dict(KL=tKLB, KSG=tKSGB, tKL=ttKLB, tKSG=ttKSGB),
        "times_after":  dict(KL=tKLA, KSG=tKSGA, tKL=ttKLA, tKSG=ttKSGA),
        "time_ICA": tICA,
    }
    return results

# ================================================================
# 8) Run + Print
# ================================================================
def _warn_if_negative_tc(name: str, val: float):
    if val < 0:
        print(f"[WARN] {name}: TC < 0 (estimation artifact) → {val:.6f}")

if __name__ == "__main__":
    PATH_ARFF = "/EEG Eye State.arff"
    K_GRID    = (1,2,3,4,5,7,10,15,20,30)

    res = run_ica_tc_four_estimators_um(
        PATH_ARFF,
        k_grid=K_GRID,
        n_splits=3,
        seed=2025,
        jitter_scale=1e-9,
        eps=1e-8
    )

    print("\n== dims:", res["dims"])
    print("== k*  :", res["k_star"])

    print("\n[PREF-ICA] TC and time(s)")
    for name in ("KL","KSG","tKL","tKSG"):
        val = res["TC_before"][name]
        print(f"  {name:<4}: {val: .6f}   (time {res['times_before'][name]:.2f}s)")

    print("\n[POST-ICA] TC and time(s)")
    for name in ("KL","KSG","tKL","tKSG"):
        val = res["TC_after"][name]
        print(f"  {name:<4}: {val: .6f}   (time {res['times_after'][name]:.2f}s)")
        _warn_if_negative_tc(f"{name} (after)", val)

    print("\nΔTC (before - after)")
    for name in ("KL","KSG","tKL","tKSG"):
        delta = res["TC_before"][name] - res["TC_after"][name]
        print(f"  {name:<4}: {delta: .6f}")

    print(f"\n[ICA ] time: {res['time_ICA']:.2f}s")
