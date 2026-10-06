"""Frozen new-seed exploratory Gaussian low-dimension / larger-n extension."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
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

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = Path('/Users/hojeongwoo/Documents/Codex/2026-05-03/d-dimensional-partitioned-sample-spacing-pss/Partitioned-Sample-Spacing')
OUT = ROOT / 'results/synthetic_gaussian_lowdim_20261006'
CONDITIONS = [(30000, 3), (100000, 3), (300000, 3), (100000, 2), (100000, 4)]
RHO = .2
ELLS = list(range(1, 9))
REPS = 30
WORKERS = 3
sys.path.insert(0, str(ARCHIVE / 'experiments/theory_selection'))
sys.path.insert(0, str(ROOT))
import run_dependent_five as generator
from PSS.pss_v2 import VERSION, estimate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def truth(d):
    return .5 * (d * math.log(2 * math.pi * math.e)
                 + (d-1) * math.log1p(-RHO) + math.log1p((d-1) * RHO))


def atomic_json(value, path):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def atomic_csv(frame, path):
    tmp = path.with_suffix('.tmp')
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def freeze():
    records = [dict(config_id=cid, family='Normal', n=n, d=d, rho=RHO,
                    replicate=rep, seed=int(np.random.SeedSequence([2026100604, n, d, rep]).generate_state(1)[0]),
                    truth=truth(d), assumption_matched=False)
               for cid, (n, d) in enumerate(CONDITIONS, 1) for rep in range(1, REPS+1)]
    assert len({r['seed'] for r in records}) == len(records)
    paths = {'PSS/pss_v2.py': ROOT / 'PSS/pss_v2.py',
             'PSS/pss_v2.cpp': ROOT / 'PSS/pss_v2.cpp',
             'run_dependent_five.py': Path(generator.__file__),
             'make_anchor_grid_datasets.py': Path(generator.original.__file__),
             'run.py': Path(__file__)}
    old = json.loads((ROOT / 'results/synthetic_gaussian_rho_20261006/protocol.json').read_text())
    for name in ['PSS/pss_v2.py', 'PSS/pss_v2.cpp']:
        assert sha(paths[name]) == old['code_sha256'][name], f'Core drift: {name}'
    protocol = dict(
        study='New-seed exploratory extension after lower-rho Gaussian results',
        estimator=VERSION, rho=RHO,
        conditions=[dict(n=n, d=d) for n, d in CONDITIONS], repetitions=REPS,
        ell_grid=ELLS, expected_datasets=len(records), expected_fits=len(records)*len(ELLS),
        seed_design='Independent fixed seeds: uint32 SeedSequence([2026100604,n,d,replicate]); no reused, rejected, or selected draws.',
        seed_schedule=records,
        sampling='Exact existing run_dependent_five.sample64 Normal generator, float64. This generator has legacy CDF clipping at 1e-10 and 1-1e-10; every clip count is audited. No additional clipping, jitter, redraw, density renormalization, or candidate retuning.',
        truth='0.5*[d*log(2*pi*e)+(d-1)*log(1-rho)+log(1+(d-1)*rho)] nats',
        failure_policy='Keep all 30 replicates, all eight candidates, failures, ties, clips and zero-valid diagnostics. Canonical zero estimate retained when n_valid=0. No coverage filter.',
        interpretation='Conditions chosen after previous rho study; new seeds do not make this a broad confirmatory study. Gaussian is outside bounded-support/positive-density-lower-bound assumptions. All specified outcomes reported.',
        theory=dict(bound_delta=.05, rate_C=1, bound='ell^-2 + sqrt(2*d*log(2*n+1)+log(96/delta))*(ell^d/n)^.25',
                    dimension_aware_rate='max(1,floor((n/(d^6*log(n)^2))^(1/(d+8))+.5))'),
        code_paths={name: str(path) for name, path in paths.items()},
        code_sha256={name: sha(path) for name, path in paths.items()},
        binary_sha256=sha(ROOT / 'PSS/libpss_v2.dylib'), workers=WORKERS,
        environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__,
                         platform=platform.platform(),
                         thread_limits={k: os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']}))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'checkpoints').mkdir(exist_ok=True)
    path = OUT / 'protocol.json'
    if path.exists():
        prior = json.loads(path.read_text())
        protocol['frozen_at_utc'] = prior['frozen_at_utc']
        assert protocol == prior, 'Frozen protocol changed.'
    else:
        protocol['frozen_at_utc'] = datetime.now(timezone.utc).isoformat()
        atomic_json(protocol, path)
    return protocol, records


def run_one(record):
    stem = f"c{record['config_id']:02d}_r{record['replicate']:02d}"
    cp = OUT / 'checkpoints' / (stem + '.csv')
    ap = OUT / 'checkpoints' / (stem + '.json')
    if cp.exists() and ap.exists():
        rows = pd.read_csv(cp)
        assert len(rows) == len(ELLS) and set(rows.ell) == set(ELLS)
        return stem
    n, d, seed = record['n'], record['d'], record['seed']
    began = time.monotonic()
    failure = ''
    try:
        x, clips = generator.sample64('Normal', n, d, RHO, seed)
        ties = int((np.diff(np.sort(x, axis=0), axis=0) == 0).sum())
        array_sha = hashlib.sha256(x.tobytes()).hexdigest()
        finite = bool(np.isfinite(x).all())
    except Exception as exc:
        x, clips, ties, array_sha, finite = None, None, None, '', False
        failure = f'{type(exc).__name__}: {exc}'
    audit = dict(**record, clips=clips, ties=ties, array_sha256=array_sha,
                 all_finite=finite, failure=failure, status='failed' if failure else 'ok')
    rows = []
    for ell in ELLS:
        started = time.perf_counter()
        try:
            if failure:
                raise RuntimeError(failure)
            result = estimate(x, ell)
            if not np.isfinite(result['estimate']):
                raise ValueError('Nonfinite estimate')
            rows.append(dict(**record, ell=ell, **result,
                             error=result['estimate']-record['truth'],
                             zero_valid=result['n_valid'] == 0, clips=clips, ties=ties,
                             status='ok', failure='', elapsed_s=time.perf_counter()-started))
        except Exception as exc:
            rows.append(dict(**record, ell=ell, estimate=np.nan, error=np.nan,
                             coverage=np.nan, n_valid=np.nan, mean_cell_size=np.nan,
                             integrated_mass=np.nan, zero_valid=False, clips=clips, ties=ties,
                             status='failed', failure=f'{type(exc).__name__}: {exc}',
                             elapsed_s=time.perf_counter()-started))
    audit['seconds'] = time.monotonic()-began
    atomic_csv(pd.DataFrame(rows), cp)
    atomic_json(audit, ap)
    return stem


def main():
    protocol, records = freeze()
    print(json.dumps(dict(status='FROZEN', path=str(OUT / 'protocol.json'),
                          frozen_at_utc=protocol['frozen_at_utc'], datasets=len(records), fits=1200)), flush=True)
    began = time.monotonic()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(run_one, row) for row in records]
        for completed, future in enumerate(as_completed(futures), 1):
            future.result()
            if completed % 15 == 0 or completed == len(records):
                print(json.dumps(dict(completed_datasets=completed, total=len(records),
                                      seconds=round(time.monotonic()-began, 2))), flush=True)
    frames, audits = [], []
    for row in records:
        stem = f"c{row['config_id']:02d}_r{row['replicate']:02d}"
        frames.append(pd.read_csv(OUT / 'checkpoints' / (stem+'.csv'), float_precision='round_trip'))
        audits.append(json.loads((OUT / 'checkpoints' / (stem+'.json')).read_text()))
    raw = pd.concat(frames, ignore_index=True).sort_values(['config_id', 'replicate', 'ell'])
    assert len(raw) == 1200 and not raw.duplicated(['n', 'd', 'replicate', 'ell']).any()
    assert raw.groupby(['n', 'd', 'ell']).size().eq(REPS).all()
    assert raw.groupby(['n', 'd', 'replicate']).size().eq(len(ELLS)).all()
    assert set(raw.ell) == set(ELLS)
    atomic_csv(raw, OUT / 'candidates.csv')
    current = {name: sha(path) for name, path in protocol['code_paths'].items()}
    assert current == protocol['code_sha256'], 'Source changed during experiment.'
    current_binary = sha(ROOT / 'PSS/libpss_v2.dylib')
    assert current_binary == protocol['binary_sha256'], 'Binary changed during experiment.'
    audit = dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
                 candidate_rows=len(raw), datasets=len(audits), conditions=len(CONDITIONS),
                 full_expected_counts=True, repetitions_per_condition_ell=REPS,
                 unique_seeds=len({a['seed'] for a in audits}),
                 failed_fits=int(raw.status.eq('failed').sum()),
                 failed_datasets=sum(a['status'] == 'failed' for a in audits),
                 zero_valid_fits=int(raw.zero_valid.sum()),
                 clips=sum(a['clips'] or 0 for a in audits), ties=sum(a['ties'] or 0 for a in audits),
                 all_generated_datasets_finite=all(a['all_finite'] for a in audits),
                 minimum_coverage=float(raw.coverage.min()),
                 maximum_coverage=float(raw.coverage.max()),
                 protocol_sha256=sha(OUT / 'protocol.json'),
                 candidates_sha256=sha(OUT / 'candidates.csv'),
                 core_source_unchanged=True, binary_unchanged=True,
                 code_sha256=current, binary_sha256=current_binary,
                 seconds=round(time.monotonic()-began, 2))
    atomic_json(audit, OUT / 'audit.json')
    print(json.dumps(dict(status='COMPLETE', **audit)), flush=True)


if __name__ == '__main__':
    main()
