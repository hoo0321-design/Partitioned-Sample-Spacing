"""Generate shared synthetic datasets for anchor-grid entropy experiments.

The generated datasets are used by all estimators so that PSS, CADEE, KL, KSG,
UM-tKL, and UM-tKSG are compared on identical Monte Carlo draws.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import special, stats


DEFAULT_FAMILIES = ["Normal", "Gamma", "Beta", "Lognormal", "Laplace"]


def parse_csv_numbers(text: str, cast=float):
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def parse_families(text: str):
    valid = {name.lower(): name for name in DEFAULT_FAMILIES}
    out = []
    for raw in text.split(","):
        key = raw.strip().lower()
        if key not in valid:
            raise ValueError(f"Unknown family: {raw}")
        out.append(valid[key])
    return out


def equicorr(d: int, rho: float):
    mat = np.full((d, d), float(rho), dtype=float)
    np.fill_diagonal(mat, 1.0)
    return mat


def entropy_gamma(shape: float, scale: float):
    return shape + np.log(scale) + special.gammaln(shape) + (1.0 - shape) * special.digamma(shape)


def entropy_beta(a: float, b: float):
    return (
        special.betaln(a, b)
        - (a - 1.0) * special.digamma(a)
        - (b - 1.0) * special.digamma(b)
        + (a + b - 2.0) * special.digamma(a + b)
    )


def qlaplace(u, scale: float):
    return np.where(u < 0.5, scale * np.log(2.0 * u), -scale * np.log(2.0 * (1.0 - u)))


def true_entropy(family: str, d: int, rho: float, params: dict[str, float]):
    sign, logdet = np.linalg.slogdet(equicorr(d, rho))
    if sign <= 0:
        raise ValueError(f"Equicorrelation matrix is not positive definite: d={d}, rho={rho}")

    if family == "Normal":
        h_margin = 0.5 * (1.0 + np.log(2.0 * np.pi))
    elif family == "Gamma":
        h_margin = entropy_gamma(params["gamma_shape"], params["gamma_scale"])
    elif family == "Beta":
        h_margin = entropy_beta(params["beta_a"], params["beta_b"])
    elif family == "Lognormal":
        h_margin = params["meanlog"] + np.log(params["sdlog"]) + 0.5 * np.log(2.0 * np.pi * np.e)
    elif family == "Laplace":
        h_margin = 1.0 + np.log(2.0 * params["laplace_scale"])
    else:
        raise ValueError(family)

    return float(d * h_margin + 0.5 * logdet)


def simulate(family: str, n: int, d: int, rho: float, params: dict[str, float]):
    z = np.random.multivariate_normal(np.zeros(d), equicorr(d, rho), size=n)
    u = np.clip(stats.norm.cdf(z), 1.0e-10, 1.0 - 1.0e-10)

    if family == "Normal":
        x = stats.norm.ppf(u)
    elif family == "Gamma":
        x = stats.gamma.ppf(u, a=params["gamma_shape"], scale=params["gamma_scale"])
    elif family == "Beta":
        x = stats.beta.ppf(u, a=params["beta_a"], b=params["beta_b"])
    elif family == "Lognormal":
        x = stats.lognorm.ppf(u, s=params["sdlog"], scale=np.exp(params["meanlog"]))
    elif family == "Laplace":
        x = qlaplace(u, params["laplace_scale"])
    else:
        raise ValueError(family)

    return np.asarray(x, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--families", default=",".join(DEFAULT_FAMILIES))
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--n-grid", default="1000,3000,10000")
    parser.add_argument("--d-grid", default="2,5,10,20")
    parser.add_argument("--rho-grid", default="0,0.1,0.3,0.5,0.8")
    parser.add_argument("--anchor-n", type=int, default=10000)
    parser.add_argument("--anchor-d", type=int, default=5)
    parser.add_argument("--anchor-rho", type=float, default=0.0)
    parser.add_argument("--seed-base", type=int, default=20260504)
    parser.add_argument("--gamma-shape", type=float, default=0.4)
    parser.add_argument("--gamma-scale", type=float, default=0.3)
    parser.add_argument("--beta-a", type=float, default=0.5)
    parser.add_argument("--beta-b", type=float, default=2.0)
    parser.add_argument("--meanlog", type=float, default=0.0)
    parser.add_argument("--sdlog", type=float, default=1.0)
    parser.add_argument("--laplace-scale", type=float, default=1.0 / np.sqrt(2.0))
    args = parser.parse_args()

    families = parse_families(args.families)
    n_grid = parse_csv_numbers(args.n_grid, int)
    d_grid = parse_csv_numbers(args.d_grid, int)
    rho_grid = parse_csv_numbers(args.rho_grid, float)
    params = {
        "gamma_shape": args.gamma_shape,
        "gamma_scale": args.gamma_scale,
        "beta_a": args.beta_a,
        "beta_b": args.beta_b,
        "meanlog": args.meanlog,
        "sdlog": args.sdlog,
        "laplace_scale": args.laplace_scale,
    }

    configs = []
    for family in families:
        for n in n_grid:
            configs.append(("N scaling", family, n, args.anchor_d, args.anchor_rho))
        for d in d_grid:
            configs.append(("d scaling", family, args.anchor_n, d, args.anchor_rho))
        for rho in rho_grid:
            configs.append(("rho scaling", family, args.anchor_n, args.anchor_d, rho))

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join(
            args.results_root,
            "anchor_grid_all_estimators_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
    datasets_dir = os.path.join(out_dir, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)

    rows = []
    for setting_id, (experiment, family, n, d, rho) in enumerate(configs, start=1):
        h_true = true_entropy(family, d, rho, params)
        for rep in range(1, args.reps + 1):
            seed = args.seed_base + setting_id * 1000 + rep
            np.random.seed(seed)
            file_name = f"setting_{setting_id:03d}_rep_{rep:02d}.csv"
            data = simulate(family, n, d, rho, params)
            pd.DataFrame(data).to_csv(os.path.join(datasets_dir, file_name), index=False)
            rows.append(
                {
                    "setting_id": setting_id,
                    "replicate": rep,
                    "experiment": experiment,
                    "distribution": family,
                    "n": n,
                    "d": d,
                    "rho": rho,
                    "true_entropy": h_true,
                    "data_file": file_name,
                    "seed": seed,
                }
            )

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "settings.csv"), index=False)
    config = dict(params)
    config.update(
        {
            "reps": args.reps,
            "families": ",".join(families),
            "n_scaling": f"{','.join(map(str, n_grid))}|d={args.anchor_d}|rho={args.anchor_rho}",
            "d_scaling": f"{','.join(map(str, d_grid))}|n={args.anchor_n}|rho={args.anchor_rho}",
            "rho_scaling": f"{','.join(map(str, rho_grid))}|n={args.anchor_n}|d={args.anchor_d}",
        }
    )
    pd.DataFrame([config]).to_csv(os.path.join(out_dir, "data_generation_config.csv"), index=False)
    print(out_dir)


if __name__ == "__main__":
    main()

