"""Run KL/KSG and trained UM-kNN baselines on anchor-grid datasets."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import special
from sklearn.neighbors import NearestNeighbors


K_GRID = [1, 3, 5, 10, 15, 20]


def set_runtime_defaults():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    os.environ.setdefault(
        "THEANO_FLAGS",
        "device=cpu,floatX=float32,optimizer=fast_compile,cxx=,blas.ldflags=",
    )


def unit_ball_volume(dim: int):
    return np.pi ** (dim / 2.0) / special.gamma(dim / 2.0 + 1.0)


def kl_entropy(y, k=1, shuffle=True, standardize=False, rng=np.random):
    y = np.asarray(y, float).copy()
    n_total, dim = y.shape

    y_std = None
    if standardize:
        y_std = np.std(y, axis=0)
        y_std[y_std == 0.0] = 1.0
        y = y / y_std

    if shuffle:
        rng.shuffle(y)

    nbrs = NearestNeighbors(
        n_neighbors=k + 1,
        algorithm="kd_tree",
        metric="minkowski",
        p=2,
    ).fit(y)
    dist, _ = nbrs.kneighbors(y)
    radius = np.maximum(dist[:, k], 1.0e-12)

    hh = dim * np.log(radius) + np.log(unit_ball_volume(dim))
    if standardize:
        hh = hh + np.sum(np.log(y_std))

    return float(-special.digamma(k) + special.digamma(n_total) + np.mean(hh))


def ksg_entropy(y, k=1, shuffle=True, standardize=True, rng=np.random):
    y = np.asarray(y, float).copy()

    if standardize:
        y_std = np.std(y, axis=0)
        y_std[y_std == 0.0] = 1.0
        y = y / y_std

    n_total, dim = y.shape

    if shuffle:
        rng.shuffle(y)

    nbrs = NearestNeighbors(
        n_neighbors=k + 1,
        algorithm="kd_tree",
        metric="chebyshev",
    ).fit(y)
    _, idx = nbrs.kneighbors(y)

    hh = np.empty(n_total)
    for row_id in range(n_total):
        radius = np.max(np.abs(y[row_id] - y[idx[row_id, 1 : k + 1]]), axis=0)
        if standardize:
            hh[row_id] = np.log(np.prod(2 * np.maximum(radius, 1.0e-12) * y_std))
        else:
            hh[row_id] = np.log(np.prod(2 * np.maximum(radius, 1.0e-12)))

    return float(-special.digamma(k) + special.digamma(n_total) + (dim - 1) / k + np.mean(hh))


def summarize(rows):
    df = pd.DataFrame(rows)
    df["Error"] = df["Estimate"] - df["True_Entropy"]
    df["Abs_Error"] = df["Error"].abs()
    keys = ["Experiment", "Distribution", "Dimensions", "N_Samples", "Correlation", "Method"]
    out = []

    for values, part in df.groupby(keys, sort=False):
        by_param = (
            part.groupby("Optimal_Param", as_index=False)
            .agg(
                RMSE=("Error", lambda x: float(np.sqrt(np.nanmean(np.square(x))))),
                Bias=("Error", "mean"),
                Abs_Error=("Abs_Error", "mean"),
                Estimate_SD=("Estimate", "std"),
                Eval_Time_s=("Eval_Time_s", "mean"),
                Train_Time_s=("Train_Time_s", "mean"),
                N_Reps=("Replicate", "nunique"),
            )
            .sort_values(["RMSE", "Optimal_Param"])
        )
        best = by_param.iloc[0].to_dict()
        squared = np.square(part[part["Optimal_Param"] == best["Optimal_Param"]]["Error"].dropna().values)
        rmse = best["RMSE"]
        rmse_se = np.nan
        if len(squared) > 1 and np.isfinite(rmse) and rmse > 0:
            rmse_se = float(np.std(squared, ddof=1) / np.sqrt(len(squared)) / (2.0 * rmse))

        rec = dict(zip(keys, values))
        rec.update(best)
        rec["RMSE_SE"] = rmse_se
        rec["Tuning"] = "Oracle-k" if rec["Method"] in ["KL", "KSG"] else "TrainedNF+Oracle-k"
        out.append(rec)

    return df, pd.DataFrame(out)


class DummySim:
    def __init__(self, samples):
        self.samples = samples

    def sim(self, n_samples):
        return self.samples[: int(n_samples)]


def refactor_python2_tree(src_dir: Path, dst_dir: Path):
    import lib2to3.refactor

    if dst_dir.exists():
        return

    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "temp_data", "figs")
    shutil.copytree(src_dir, dst_dir, ignore=ignore)

    tool = lib2to3.refactor.RefactoringTool(lib2to3.refactor.get_fixers_from_package("lib2to3.fixes"))
    for path in dst_dir.rglob("*.py"):
        text = path.read_text()
        try:
            converted = str(tool.refactor_string(text, str(path)))
        except Exception:
            converted = text
        path.write_text(converted)


def patch_converted_knn(knn_dir: Path):
    entropy_path = knn_dir / "ent_est" / "entropy.py"
    entropy_text = entropy_path.read_text()
    entropy_text = entropy_text.replace("algorithm='auto'", "algorithm='kd_tree'")
    entropy_text = entropy_text.replace('algorithm="auto"', 'algorithm="kd_tree"')
    entropy_text = entropy_text.replace(
        "def learn_density(model, xs, ws=None, regularizer=None, val_frac=0.05, step=ss.Adam(a=1.e-4), minibatch=100, patience=20, monitor_every=1, logger=sys.stdout, rng=np.random):",
        "def learn_density(model, xs, ws=None, regularizer=None, val_frac=0.05, step=ss.Adam(a=1.e-4), minibatch=100, patience=20, monitor_every=1, logger=sys.stdout, rng=np.random, maxepochs=None):",
    )
    entropy_text = entropy_text.replace(
        "trainer.train(\n"
        "            minibatch=minibatch,\n"
        "            patience=patience,\n"
        "            monitor_every=monitor_every,\n"
        "            logger=logger\n"
        "        )",
        "trainer.train(\n"
        "            minibatch=minibatch,\n"
        "            patience=patience,\n"
        "            monitor_every=monitor_every,\n"
        "            logger=logger,\n"
        "            maxepochs=maxepochs\n"
        "        )",
    )
    entropy_path.write_text(entropy_text)

    trainers_path = knn_dir / "ml" / "trainers.py"
    trainers_text = trainers_path.read_text()
    needle = "        # initialize some variables\n        iter = 0\n"
    if "tol = -float('inf') if tol is None else tol" not in trainers_text:
        trainers_text = trainers_text.replace(
            needle,
            "        # initialize some variables\n"
            "        tol = -float('inf') if tol is None else tol\n"
            "        iter = 0\n",
        )
    trainers_path.write_text(trainers_text)


def load_um_components(repo_root: Path, base_dir: Path, force_prepare=False):
    set_runtime_defaults()
    prepared_knn = base_dir / "_knn_py3"
    if force_prepare and prepared_knn.exists():
        shutil.rmtree(prepared_knn)

    refactor_python2_tree(repo_root / "KNN", prepared_knn)
    patch_converted_knn(prepared_knn)
    sys.path.insert(0, str(prepared_knn))

    from ent_est.entropy import UMestimator, learn_density
    from ml.models.mafs import MaskedAutoregressiveFlow

    return UMestimator, learn_density, MaskedAutoregressiveFlow


def add_row(rows, row, method, k, estimate, eval_time, train_time=0.0, error_message=None):
    rows.append(
        {
            "Experiment": row.experiment,
            "Distribution": row.distribution,
            "Dimensions": int(row.d),
            "N_Samples": int(row.n),
            "Correlation": float(row.rho),
            "Replicate": int(row.replicate),
            "Method": method,
            "Optimal_Param": k,
            "Estimate": estimate,
            "True_Entropy": float(row.true_entropy),
            "Eval_Time_s": eval_time,
            "Train_Time_s": train_time,
            "Error_Message": error_message,
        }
    )


def completed_keys(partial):
    key_cols = ["Experiment", "Distribution", "Dimensions", "N_Samples", "Correlation", "Replicate"]
    done = set()
    for values, part in partial.groupby(key_cols, sort=False):
        methods = set(part["Method"].dropna())
        expected = {"KL", "KSG", "UM-tKL", "UM-tKSG"}
        if expected.issubset(methods):
            done.add(values)
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--skip-um", action="store_true")
    parser.add_argument("--force-prepare-knn", action="store_true")
    args = parser.parse_args()

    repo_root = Path.cwd()
    base_dir = Path(args.base_dir)
    settings = pd.read_csv(base_dir / "settings.csv")
    datasets_dir = base_dir / "datasets"
    partial_path = base_dir / "knn_um_estimates_partial.csv"

    rows = []
    done = set()
    if partial_path.exists():
        partial = pd.read_csv(partial_path)
        rows = partial.to_dict("records")
        done = completed_keys(partial)

    UMestimator = learn_density = MaskedAutoregressiveFlow = None
    if not args.skip_um:
        UMestimator, learn_density, MaskedAutoregressiveFlow = load_um_components(
            repo_root,
            base_dir,
            force_prepare=args.force_prepare_knn,
        )

    for index, row in settings.iterrows():
        setting_key = (
            row.experiment,
            row.distribution,
            int(row.d),
            int(row.n),
            float(row.rho),
            int(row.replicate),
        )
        if setting_key in done:
            print(
                f"[{index + 1}/{len(settings)}] skip completed {row.distribution} "
                f"n={row.n} d={row.d} rho={row.rho} rep={row.replicate}",
                flush=True,
            )
            continue

        x = pd.read_csv(datasets_dir / row.data_file).values.astype(np.float32)
        d = x.shape[1]
        current_rows = []
        print(
            f"[{index + 1}/{len(settings)}] {row.distribution} n={row.n} d={row.d} "
            f"rho={row.rho} rep={row.replicate}",
            flush=True,
        )

        for k in K_GRID:
            t0 = time.perf_counter()
            estimate = kl_entropy(x, k=k)
            add_row(current_rows, row, "KL", k, estimate, time.perf_counter() - t0)

            t0 = time.perf_counter()
            estimate = ksg_entropy(x, k=k)
            add_row(current_rows, row, "KSG", k, estimate, time.perf_counter() - t0)

        if not args.skip_um:
            np.random.seed(int(row.seed) + 777)
            model = MaskedAutoregressiveFlow(
                n_inputs=d,
                n_hiddens=[args.hidden],
                act_fun="tanh",
                n_mades=1,
                batch_norm=True,
                rng=np.random,
            )
            um = UMestimator(DummySim(x), model)
            um.samples = x
            um.n_samples = x.shape[0]
            um.x_dim = x.shape[1]

            t0 = time.perf_counter()
            learn_density(
                model,
                x,
                patience=args.patience,
                monitor_every=1,
                maxepochs=args.epochs,
                logger=None,
                rng=np.random,
            )
            train_time = time.perf_counter() - t0

            for k in K_GRID:
                for method, method_arg in [("UM-tKL", "umtkl"), ("UM-tKSG", "umtksg")]:
                    t0 = time.perf_counter()
                    try:
                        estimate = float(um.calc_ent(k=k, reuse_samples=True, method=method_arg)[0])
                        error_message = None
                    except Exception as exc:
                        estimate = np.nan
                        error_message = repr(exc)
                    add_row(
                        current_rows,
                        row,
                        method,
                        k,
                        estimate,
                        time.perf_counter() - t0,
                        train_time=train_time,
                        error_message=error_message,
                    )

        pd.DataFrame(current_rows).to_csv(
            partial_path,
            mode="a",
            header=not partial_path.exists(),
            index=False,
        )
        rows.extend(current_rows)

    estimates, summary = summarize(rows)
    estimates.to_csv(base_dir / "knn_um_estimates.csv", index=False)
    summary.to_csv(base_dir / "knn_um_summary.csv", index=False)
    print("\n=== KNN + trained UM summary ===")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()

