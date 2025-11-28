# --- compatibility shims (MUST be first) -------------------------------------
import sys, os
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    import numpy as np
    if not hasattr(np, "bool"):
        np.bool = bool  # safe alias

# Python 2 -> 3 shim for cPickle
import pickle as _pickle
sys.modules['cPickle'] = _pickle

# Python 2 → 3 shims
import builtins, itertools
if not hasattr(itertools, "izip"):
    itertools.izip = zip
if not hasattr(builtins, "xrange"):
    builtins.xrange = range
if not hasattr(builtins, "unicode"):
    builtins.unicode = str
if not hasattr(builtins, "long"):
    builtins.long = int

# --- locate project root (path robust) ---------------------------------------
from pathlib import Path
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent   # NFEE-main/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

print("PYTHON EXEC:", sys.executable)
print("NumPy ver  :", np.__version__)
print("Script     :", _THIS)
print("ProjectRoot:", _ROOT)

# --- output dirs (anchored to project root) ----------------------------------
FIGS_DIR = _ROOT / "figs"
DATA_DIR = _ROOT / "temp_data"
FIGS_DIR.mkdir(exist_ok=True, parents=True)
DATA_DIR.mkdir(exist_ok=True, parents=True)

# --- core imports -------------------------------------------------------------
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# your estimators
from ent_est.entropy import kl, ksg, tkl, tksg

# ==============================================================================
# Multivariate Lognormal via exp(N(mu, Sigma))
# ==============================================================================
def equicorr_cov(d, rho, sigma2=1.0):
    """Equicorrelation covariance: diag = sigma2, off-diag = rho*sigma2."""
    R = np.full((d, d), rho, dtype=float)
    np.fill_diagonal(R, 1.0)
    # scale by sigma (std) per-dim; here all sigma = sqrt(sigma2)
    Sigma = R * float(sigma2)
    return Sigma

class MVLognormal:
    """
    X ~ N_d(mu, Sigma), Y = exp(X).
    True entropy: h(Y) = 0.5 * log((2*pi*e)^d * |Sigma|) + sum(mu).
    """
    def __init__(self, mu, Sigma):
        self.mu = np.asarray(mu, dtype=float)
        self.Sigma = np.asarray(Sigma, dtype=float)
        self.d = self.mu.shape[0]
        assert self.Sigma.shape == (self.d, self.d)
        # precompute per-dim std for marginals
        self.std = np.sqrt(np.diag(self.Sigma))
        # for true entropy
        sign, logdet = np.linalg.slogdet(self.Sigma)
        if sign <= 0:
            raise ValueError("Sigma is not positive definite.")
        self.logdet = logdet

    def sim(self, n_samples):
        X = np.random.multivariate_normal(mean=self.mu, cov=self.Sigma, size=int(n_samples))
        return np.exp(X)

    def true_entropy(self):
        d = self.d
        return 0.5 * (d * np.log(2 * np.pi * np.e) + self.logdet) + float(np.sum(self.mu))

# --- experiment settings ------------------------------------------------------
ds = [1, 10, 20, 40]
n_trials = 10
all_n_samples = [5000]

# lognormal parameters (for X ~ N(mu, Sigma))
mu_scalar = 0.0
sigma2 = 1.0
rho = 0.0  # set dependence like your R code; change to 0.3, 0.6 if desired

mse1 = np.empty((len(all_n_samples), len(ds)))
mse2 = np.empty((len(all_n_samples), len(ds)))
mse3 = np.empty((len(all_n_samples), len(ds)))
mse4 = np.empty((len(all_n_samples), len(ds)))

# --- run ----------------------------------------------------------------------
for k, d in enumerate(ds):
    mu_vec = np.full(d, mu_scalar, dtype=float)
    Sigma = equicorr_cov(d, rho=rho, sigma2=sigma2)
    sim_mdl = MVLognormal(mu=mu_vec, Sigma=Sigma)

    for n_idx, n_samples in enumerate(all_n_samples):
        cal1 = np.empty(n_trials)
        cal2 = np.empty(n_trials)
        cal3 = np.empty(n_trials)
        cal4 = np.empty(n_trials)

        for i in range(n_trials):
            # KL / KSG on raw Y
            Y = sim_mdl.sim(n_samples=n_samples)
            cal1[i] = kl(Y)
            cal2[i] = ksg(Y)

            # tKL on U=F_marg(Y) and subtract E[log ∏ f_marg(Y)]
            Y = sim_mdl.sim(n_samples=n_samples)
            X = np.log(Y)                              # back to Gaussian
            # per-dim standardized for marginal CDFs
            U = (X - mu_vec) / sim_mdl.std
            U = stats.norm.cdf(U)                      # elementwise CDF to uniforms
            # log of product of lognormal marginal pdfs
            # scipy: lognorm(s=sigma, scale=exp(mu))
            logpdf_rows = np.zeros(Y.shape[0], dtype=float)
            for j in range(d):
                logpdf_rows += stats.lognorm.logpdf(
                    Y[:, j],
                    s=sim_mdl.std[j],
                    scale=np.exp(mu_vec[j])
                )
            cal3[i] = tkl(U) - np.mean(logpdf_rows)

            # tKSG (same transform, different estimator)
            Y = sim_mdl.sim(n_samples=n_samples)
            X = np.log(Y)
            U = (X - mu_vec) / sim_mdl.std
            U = stats.norm.cdf(U)
            logpdf_rows = np.zeros(Y.shape[0], dtype=float)
            for j in range(d):
                logpdf_rows += stats.lognorm.logpdf(
                    Y[:, j],
                    s=sim_mdl.std[j],
                    scale=np.exp(mu_vec[j])
                )
            cal4[i] = tksg(U) - np.mean(logpdf_rows)

        true_val = sim_mdl.true_entropy()
        mse1[n_idx, k] = np.mean((cal1 - true_val) ** 2)
        mse2[n_idx, k] = np.mean((cal2 - true_val) ** 2)
        mse3[n_idx, k] = np.mean((cal3 - true_val) ** 2)
        mse4[n_idx, k] = np.mean((cal4 - true_val) ** 2)

# --- plot RMSE vs d -----------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(6, 4.2), dpi=150)
ax.plot(ds, np.sqrt(mse3[0]), marker='o', linestyle=':', label='UM-tKL (Lognormal)', mfc='none')
ax.plot(ds, np.sqrt(mse4[0]), marker='o', linestyle='-',  label='UM-tKSG (Lognormal)', mfc='none')
ax.plot(ds, np.sqrt(mse1[0]), marker='x', linestyle=':', label='KL')
ax.plot(ds, np.sqrt(mse2[0]), marker='x', linestyle='-',  label='KSG')
ax.set_xlabel('dimension')
ax.set_ylabel('RMSE')
ax.legend()
fig.tight_layout()
plt.savefig(str(FIGS_DIR / 'RMSE_vs_d_mlognorm.png'))

# --- save raw results ---------------------------------------------------------
try:
    import util.io
    util.io.save((ds, mse1, mse2, mse3, mse4), str(DATA_DIR / "RMSE_vs_d_mlognorm"))
    print("Saved results with util.io.save ->", DATA_DIR / "RMSE_vs_d_mlognorm")
except Exception as e:
    print("[warn] util.io.save failed, falling back to pickle:", e)
    with open(DATA_DIR / "RMSE_vs_d_mlognorm.pkl", "wb") as f:
        _pickle.dump((ds, mse1, mse2, mse3, mse4), f)
    print("Saved results ->", DATA_DIR / "RMSE_vs_d_mlognorm.pkl")

print("Done.")
