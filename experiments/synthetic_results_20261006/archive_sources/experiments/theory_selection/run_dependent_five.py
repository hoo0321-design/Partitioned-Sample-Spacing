"""Dependent five-family PSS study; preserve the previous experiments unchanged."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import scipy
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "anchor_grid"))
import make_anchor_grid_datasets as original
from pss_theory import estimate
from reanalyze_anchor_results import bound_choice

PARAMS = dict(gamma_shape=.4, gamma_scale=.3, beta_a=.5, beta_b=2.,
              meanlog=0., sdlog=1., laplace_scale=1/math.sqrt(2))
FAMILIES = original.DEFAULT_FAMILIES
N_GRID = [1000, 3000, 10000, 30000]
D_GRID = [2, 5, 10, 20]
RHO_GRID = [0., .5, .8]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grid(d):
    return list(range(1, {2: 40, 5: 40, 10: 12, 20: 8}[d]+1))


def choices(n, d):
    return {
        "Dimension-aware rate C=1": max(1, math.floor((n/(d**6*math.log(n)**2))**(1/(d+8))+.5)),
        "Fixed-d rate C=1": max(1, math.floor((n/math.log(n)**2)**(1/(d+8))+.5)),
        "Bound delta=n^-4": bound_choice(n, d, n**(-4.0)),
        "Bound delta=0.05": bound_choice(n, d, .05),
        "Unpartitioned ell=1": 1,
    }


def sample64(family, n, d, rho, seed):
    # RandomState reproduces the original global np.random.seed stream.
    rng = np.random.RandomState(seed)
    z = rng.multivariate_normal(np.zeros(d), original.equicorr(d, rho), size=n)
    raw_u = stats.norm.cdf(z)
    clips = int(((raw_u < 1e-10) | (raw_u > 1-1e-10)).sum())
    u = np.clip(raw_u, 1e-10, 1-1e-10)
    if family == "Normal":
        x = stats.norm.ppf(u)
    elif family == "Gamma":
        x = stats.gamma.ppf(u, a=PARAMS["gamma_shape"], scale=PARAMS["gamma_scale"])
    elif family == "Beta":
        x = stats.beta.ppf(u, a=PARAMS["beta_a"], b=PARAMS["beta_b"])
    elif family == "Lognormal":
        x = stats.lognorm.ppf(u, s=PARAMS["sdlog"], scale=np.exp(PARAMS["meanlog"]))
    elif family == "Laplace":
        x = original.qlaplace(u, PARAMS["laplace_scale"])
    else:
        raise ValueError(family)
    return np.ascontiguousarray(x, dtype=np.float64), clips


def make_settings(source, reps):
    old = pd.read_csv(source / "settings.csv")
    old = old[old.experiment == "rho scaling"]
    rows = []
    config_id = 0
    for family in FAMILIES:
        specs = [(20000, 5, rho, "restored_rho") for rho in RHO_GRID]
        specs += [(n, 5, .5, "fresh_n") for n in N_GRID]
        specs += [(20000, d, .5, "fresh_d") for d in D_GRID if d != 5]
        for n, d, rho, phase in specs:
            config_id += 1
            for rep in range(1, reps+1):
                seed = 2026100700 + config_id*1000 + rep
                data_file = ""
                source_setting_id = -1
                if phase == "restored_rho":
                    match = old[(old.distribution == family) & (old.n == n) &
                                (old.d == d) & (old.rho == rho) & (old.replicate == rep)]
                    if len(match) != 1:
                        raise ValueError(f"Missing original sample: {family}, {rho}, {rep}")
                    record = match.iloc[0]
                    seed = int(record.seed)
                    data_file = record.data_file
                    source_setting_id = int(record.setting_id)
                rows.append(dict(config_id=config_id, family=family, n=n, d=d, rho=rho,
                                 phase=phase, replicate=rep, seed=seed, data_file=data_file,
                                 source_setting_id=source_setting_id,
                                 truth=original.true_entropy(family, d, rho, PARAMS)))
    result = pd.DataFrame(rows)
    if result.duplicated(["family", "n", "d", "rho", "replicate"]).any():
        raise ValueError("Duplicate samples; the d/rho anchor must be shared")
    return result


def tied_coordinate_count(x):
    return int((np.diff(np.sort(x, axis=0), axis=0) == 0).sum())


def atomic_csv(frame, path):
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def run_config(records, source_string, out_string):
    source, out = Path(source_string), Path(out_string)
    config_id = int(records[0]["config_id"])
    result_path = out / "checkpoints" / f"config_{config_id:03d}.csv"
    audit_path = out / "checkpoints" / f"audit_{config_id:03d}.csv"
    start = time.monotonic()
    expected_rows = len(records)*len(grid(int(records[0]["d"])))
    if result_path.exists() and audit_path.exists():
        previous = pd.read_csv(result_path)
        audit = pd.read_csv(audit_path)
        if len(previous) != expected_rows or len(audit) != len(records):
            raise ValueError(f"Incomplete checkpoint {config_id}")
        return dict(config_id=config_id, status="cached", seconds=0., rows=len(previous))
    results, audits = [], []
    for row in records:
        n, d, seed = int(row["n"]), int(row["d"]), int(row["seed"])
        x, clips = sample64(row["family"], n, d, row["rho"], seed)
        ties = tied_coordinate_count(x)
        if clips or ties or not np.isfinite(x).all():
            raise ValueError(f"Non-continuous numeric draw; no jitter or silent fallback: {row}, clips={clips}, ties={ties}")
        source_ties, source_hash, roundtrip = 0, "", False
        if row["phase"] == "restored_rho":
            path = source / "datasets" / row["data_file"]
            saved = pd.read_csv(path).to_numpy(dtype=np.float32)
            roundtrip = bool(np.array_equal(x.astype(np.float32), saved))
            if not roundtrip:
                raise ValueError(f"Regenerated sample does not reproduce saved float32 data: {path}")
            source_ties = tied_coordinate_count(saved)
            source_hash = sha(path)
        audits.append(dict(**row, float64_ties=ties, source_float32_ties=source_ties,
                           clipped_coordinates=clips, source_roundtrip_match=roundtrip,
                           source_sha256=source_hash,
                           float64_array_sha256=hashlib.sha256(x.tobytes()).hexdigest()))
        for ell in grid(d):
            t = time.perf_counter()
            values = estimate(x, ell)
            elapsed = time.perf_counter()-t
            if not all(np.isfinite(list(values.values()))):
                raise ValueError(f"Nonfinite estimate: {row}, ell={ell}")
            results.append(dict(**row, ell=ell, **values, elapsed_s=elapsed))
    atomic_csv(pd.DataFrame(results), result_path)
    atomic_csv(pd.DataFrame(audits), audit_path)
    return dict(config_id=config_id, status="complete", seconds=round(time.monotonic()-start, 2),
                rows=len(results), family=records[0]["family"], n=records[0]["n"],
                d=records[0]["d"], rho=records[0]["rho"])


def prepare(source, out, reps):
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    settings = make_settings(source, reps)
    code_files = [Path(__file__), HERE/"pss_theory.py", HERE/"pss_core.cpp",
                  HERE/"reanalyze_anchor_results.py", Path(original.__file__)]
    binary = HERE / ("libpss_theory.dylib" if sys.platform == "darwin" else "libpss_theory.so")
    protocol = dict(version=1, source=str(source.resolve()), reps=reps, parameters=PARAMS,
                    n_scaling=dict(n=N_GRID, d=5, rho=.5),
                    d_scaling=dict(n=20000, d=D_GRID, rho=.5),
                    rho_scaling=dict(n=20000, d=5, rho=RHO_GRID),
                    candidate_grids={str(d): grid(d) for d in D_GRID},
                    settings=50, samples=50*reps, shared_d_rho_anchor=True,
                    rng="numpy.RandomState, saved original seeds or 2026100700+config_id*1000+rep",
                    data="float64 pre-rounding counterparts; verify float32 roundtrip for every restored rho sample; no jitter",
                    degeneracy="abort visibly on float64 ties or clipping; do not select or discard draws",
                    statistic="PDF smoothed-subgrid entropy averaged over N_eff; no density normalization; legacy rank/n diagnostic",
                    theory_rules="two unit-coefficient rates and two bound minimizers, delta=n^-4 or .05; no fitted coefficients",
                    oracle="same-replicate minimum RMSE; report raw and coverage>=.95/mean cell size>=2 filtered oracles; not independent validation",
                    inference="30 independent dataset repetitions; 5000 percentile bootstrap resamples for RMSE CI; conditional on selected oracle ell",
                    assumptions="five original distributions outside bounded-support, positive bounded Lipschitz density assumptions; empirical extrapolation only",
                    source_sha256={name: sha(source/name) for name in ["settings.csv", "data_generation_config.csv"]},
                    code_sha256={str(p.relative_to(ROOT)): sha(p) for p in code_files},
                    binary_sha256=sha(binary),
                    environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                                     pandas=pd.__version__, platform=platform.platform(),
                                     OPENBLAS_NUM_THREADS=os.environ.get("OPENBLAS_NUM_THREADS", "unset")))
    protocol_path = out/"protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Frozen protocol differs; use a new output directory")
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2)+"\n")
        atomic_csv(settings, out/"settings.csv")
    return settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--phase", choices=["all", "restored_rho", "fresh"], default="all")
    args = parser.parse_args()
    source, out = args.source_dir.resolve(), args.out_dir.resolve()
    settings = prepare(source, out, args.reps)
    subset = settings
    if args.phase == "restored_rho":
        subset = settings[settings.phase == "restored_rho"]
    elif args.phase == "fresh":
        subset = settings[settings.phase != "restored_rho"]
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_config, group.to_dict("records"), str(source), str(out))
                   for _, group in subset.groupby("config_id")]
        for i, future in enumerate(as_completed(futures), 1):
            print(json.dumps(dict(progress=f"{i}/{len(futures)}", **future.result())), flush=True)
    paths = sorted((out/"checkpoints").glob("config_*.csv"))
    raw = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    atomic_csv(raw, out/"estimates.csv")
    audits = sorted((out/"checkpoints").glob("audit_*.csv"))
    atomic_csv(pd.concat([pd.read_csv(p) for p in audits], ignore_index=True), out/"data_audit.csv")
    print(json.dumps(dict(status="finished", phase=args.phase, seconds=round(time.monotonic()-start, 2),
                          completed_settings=raw.config_id.nunique(), records=len(raw), output=str(out))), flush=True)


if __name__ == "__main__":
    main()
