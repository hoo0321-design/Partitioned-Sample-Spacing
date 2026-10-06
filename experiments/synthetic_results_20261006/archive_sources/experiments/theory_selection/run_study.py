"""Checkpointed pilot -> frozen selection -> independent evaluation experiment."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import platform
import time

import numpy as np

from pss_theory import (COEFFICIENTS,FAMILIES,HERE,bound_continuous,bound_objective,
                        configurations,constants,ell_grid,estimate,lambda_n,
                        rate_ell,sample,true_entropy)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setting_key(cfg): return f'{cfg["family"]}_n{cfg["n"]}_d{cfg["d"]}'


def atomic_json(path,obj):
    tmp=path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")
    tmp.replace(path)


def run_setting(job):
    cfg,phase,reps,seed,out_dir=job
    dest=Path(out_dir)/phase/(setting_key(cfg)+".csv")
    if dest.exists(): return dict(setting=setting_key(cfg),cached=True)
    n,d=cfg["n"],cfg["d"]
    truth=true_entropy(cfg["family"],d)
    rows=[]; started=time.perf_counter()
    for rep in range(reps):
        sequence=np.random.SeedSequence([seed,0 if phase=="pilot" else 1,cfg["family_id"],n,d,rep])
        rng=np.random.default_rng(sequence)
        x=sample(rng,n,d,cfg["family"])
        for ell in ell_grid(n,d):
            tick=time.perf_counter(); metrics=estimate(x,ell)
            seconds=time.perf_counter()-tick
            rows.append(dict(family=cfg["family"],n=n,d=d,phase=phase,rep=rep,
                             ell=ell,true_entropy=truth,error=metrics["estimate"]-truth,
                             legacy_error=metrics["legacy_estimate"]-truth,
                             eval_seconds=seconds,**metrics))
    tmp=dest.with_suffix(".csv.tmp")
    with tmp.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)
    tmp.replace(dest)
    return dict(setting=setting_key(cfg),cached=False,seconds=round(time.perf_counter()-started,2),rows=len(rows))


def calibrate(out_dir,configs):
    records={}; profiles=[]
    for cfg in configs:
        if not cfg["pilot"]: continue
        path=out_dir/"pilot"/(setting_key(cfg)+".csv")
        with path.open() as f: rows=list(csv.DictReader(f))
        by_ell={}
        for row in rows: by_ell.setdefault(int(row["ell"]),[]).append(row)
        summaries={ell:dict(mse=float(np.mean([float(r["error"])**2 for r in rr])),
                            coverage_min=min(float(r["coverage"]) for r in rr)) for ell,rr in by_ell.items()}
        records[setting_key(cfg)]=summaries
    for coefficient in COEFFICIENTS:
        chosen=[records[setting_key(cfg)][rate_ell(cfg["n"],cfg["d"],coefficient)]
                for cfg in configs if cfg["pilot"]]
        profiles.append(dict(coefficient=coefficient,mean_setting_mse=float(np.mean([r["mse"] for r in chosen])),
                             min_pilot_coverage=min(r["coverage_min"] for r in chosen)))
    best=min(profiles,key=lambda p:(p["mean_setting_mse"],p["coefficient"]))
    full=[p for p in profiles if p["min_pilot_coverage"]==1]
    best_full=min(full,key=lambda p:(p["mean_setting_mse"],p["coefficient"]))
    per_setting={}
    for cfg in configs:
        key=setting_key(cfg)
        if key not in records: continue
        candidates=records[key]
        all_best=min(candidates,key=lambda l:(candidates[l]["mse"],l))
        valid=[l for l in candidates if candidates[l]["coverage_min"]==1]
        full_best=min(valid,key=lambda l:(candidates[l]["mse"],l))
        per_setting[key]=dict(pilot_oracle=all_best,pilot_oracle_full_coverage=full_best)
    return dict(frozen_at=datetime.now(timezone.utc).isoformat(),
                objective="Equal-weight mean of per-setting pilot MSE; known entropy used only in pilot.",
                coefficients=COEFFICIENTS,profiles=profiles,
                selected_coefficient=best["coefficient"],
                full_coverage_selected_coefficient=best_full["coefficient"],
                full_coverage_definition="Every pilot observation valid in every pilot setting; calibration diagnostic, not a theorem certificate.",
                per_setting=per_setting,
                pilot_file_hashes={p.name:digest(p) for p in sorted((out_dir/"pilot").glob("*.csv"))})


def phase_run(out_dir,configs,phase,reps,seed,workers):
    selected=[c for c in configs if phase=="evaluation" or c["pilot"]]
    (out_dir/phase).mkdir(exist_ok=True)
    jobs=[(c,phase,reps,seed,str(out_dir)) for c in selected]
    started=time.monotonic()
    context=multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers,mp_context=context) as pool:
        tasks=[pool.submit(run_setting,job) for job in jobs]
        for i,task in enumerate(as_completed(tasks),1):
            result=task.result()
            elapsed=time.monotonic()-started
            print(json.dumps(dict(phase=phase,complete=i,total=len(tasks),elapsed_s=round(elapsed,1),**result)),flush=True)
            atomic_json(out_dir/"progress.json",dict(phase=phase,complete=i,total=len(tasks),elapsed_s=elapsed,last=result))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out-dir",required=True)
    parser.add_argument("--phase",choices=["pilot","evaluation","both"],default="both")
    parser.add_argument("--pilot-reps",type=int,default=30)
    parser.add_argument("--eval-reps",type=int,default=100)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--seed",type=int,default=20261003)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args(); out_dir=Path(args.out_dir).resolve();out_dir.mkdir(parents=True,exist_ok=True)
    configs=configurations(args.smoke)
    design=dict(seed=args.seed,pilot_reps=args.pilot_reps,eval_reps=args.eval_reps,smoke=args.smoke,
                families=FAMILIES,coefficients=COEFFICIENTS,
                settings=[dict(**c,ells=ell_grid(c["n"],c["d"]),true_entropy=true_entropy(c["family"],c["d"]),
                               bounds=constants(c["family"],c["d"])) for c in configs],
                estimator="PDF equations (3)--(5), N_eff normalization, exact sub-grid assignment",
                confidence_delta="n^-4",
                selection_provenance="Bound and rate schedules are sample-independent. Global calibrated coefficients use pilot truth; not tuning-free.",
                theorem_scope="All densities on [0,1]^d are bounded above and below and Lipschitz. Unknown K0 prevents finite-sample certification.",
                truth_accuracy="Analytic cosine entropy, Gauss-Legendre quadrature, or a convergent moment series; not Monte Carlo truth.",
                sources={p.name:digest(p) for p in [HERE/"pss_core.cpp",HERE/"pss_theory.py",HERE/"run_study.py"]},
                python=platform.python_version(),numpy=np.__version__)
    manifest=out_dir/"design.json"
    if manifest.exists():
        if json.loads(manifest.read_text())!=design: raise RuntimeError("Design mismatch; use a new output directory.")
    else: atomic_json(manifest,design)
    frozen_path=out_dir/"frozen_selection.json"
    if args.phase in ("pilot","both"):
        phase_run(out_dir,configs,"pilot",args.pilot_reps,args.seed,args.workers)
        if not frozen_path.exists(): atomic_json(frozen_path,calibrate(out_dir,configs))
        print("FROZEN_SELECTION "+str(frozen_path),flush=True)
    if args.phase in ("evaluation","both"):
        frozen=json.loads(frozen_path.read_text())
        for name,expected in frozen["pilot_file_hashes"].items():
            if digest(out_dir/"pilot"/name)!=expected: raise RuntimeError("Pilot changed after selection freeze.")
        print("EVALUATION_START frozen_sha256="+digest(frozen_path),flush=True)
        phase_run(out_dir,configs,"evaluation",args.eval_reps,args.seed,args.workers)
    print("COMPLETE "+str(out_dir),flush=True)


if __name__=="__main__": main()
