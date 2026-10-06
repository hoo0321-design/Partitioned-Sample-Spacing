"""Extrapolate already-frozen rules to larger n without recalibration."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

from pss_theory import FAMILIES,HERE,constants,ell_grid,true_entropy
from run_study import atomic_json,digest,phase_run


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-dir",required=True)
    parser.add_argument("--out-dir",required=True)
    parser.add_argument("--reps",type=int,default=30)
    parser.add_argument("--workers",type=int,default=3)
    args=parser.parse_args(); source=Path(args.source_dir).resolve();out=Path(args.out_dir).resolve()
    out.mkdir(parents=True,exist_ok=True)
    cfgs=[dict(family=f["name"],family_id=i,n=n,d=d,pilot=f["pilot"])
          for i,f in enumerate(FAMILIES) for n in [300000,1000000] for d in [2,5]]
    design=dict(seed=20261004,pilot_reps=30,eval_reps=args.reps,
                experiment="Large-n extrapolation; no calibration on extension outcomes",
                source_dir=str(source),
                frozen_hashes={p:digest(source/p) for p in ["frozen_selection.json","additional_frozen_selection.json"]},
                source_hashes={p:digest(HERE/p) for p in ["pss_core.cpp","pss_theory.py","run_study.py","run_large_n.py"]},
                settings=[dict(**c,ells=ell_grid(c["n"],c["d"]),true_entropy=true_entropy(c["family"],c["d"]),
                               bounds=constants(c["family"],c["d"])) for c in cfgs])
    path=out/"design.json"
    if path.exists():
        if json.loads(path.read_text())!=design: raise RuntimeError("Extension design mismatch")
    else:
        atomic_json(path,design)
        for name in design["frozen_hashes"]:
            (out/name).write_bytes((source/name).read_bytes())
        atomic_json(out/"start.json",dict(started_at=datetime.now(timezone.utc).isoformat()))
    for name,sha in design["frozen_hashes"].items():
        if digest(out/name)!=sha: raise RuntimeError("Frozen selection changed")
    phase_run(out,cfgs,"evaluation",args.reps,20261004,args.workers)
    print("LARGE_N_COMPLETE "+str(out),flush=True)


if __name__=="__main__": main()
