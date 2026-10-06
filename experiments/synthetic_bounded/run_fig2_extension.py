"""Frozen, reproducible bounded-density Fig. 2 extension at d=3 and d=4."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = Path('/Users/hojeongwoo/Documents/Codex/2026-05-03/d-dimensional-partitioned-sample-spacing-pss/Partitioned-Sample-Spacing')
OUT = ROOT / 'results/synthetic_bounded_fig2_extension_20261006'
CONDITIONS = [(100000, 3), (100000, 4)]
FAMILY = 'ridge_medium'
ELLS = list(range(1, 8))
REPS = 100
NAMESPACE = 2026100607
WORKERS = 3
sys.path.insert(0, str(ARCHIVE / 'experiments/theory_selection'))
sys.path.insert(0, str(ROOT))
import pss_theory as generator
from PSS.pss_v2 import VERSION, estimate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(value, path):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def atomic_csv(frame, path):
    tmp = path.with_suffix('.csv.tmp')
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def freeze():
    records = [dict(config_id=cid, family=FAMILY, n=n, d=d, replicate=rep, rep=rep,
                    seed_namespace=NAMESPACE,
                    seed_sequence=[NAMESPACE, n, d, rep],
                    seed_state=np.random.SeedSequence([NAMESPACE, n, d, rep]).generate_state(4).tolist(),
                    truth=generator.true_entropy(FAMILY, d),
                    true_entropy=generator.true_entropy(FAMILY, d))
               for cid, (n, d) in enumerate(CONDITIONS, 1) for rep in range(REPS)]
    paths = {'PSS/pss_v2.py': ROOT / 'PSS/pss_v2.py',
             'PSS/pss_v2.cpp': ROOT / 'PSS/pss_v2.cpp',
             'PSS/libpss_v2.dylib': ROOT / 'PSS/libpss_v2.dylib',
             'pss_theory.py': Path(generator.__file__),
             'run_fig2_extension.py': Path(__file__)}
    previous = json.loads((ROOT / 'results/synthetic_bounded_20261006/audit.json').read_text())
    for name, expected in previous['current_implementation_hashes'].items():
        assert sha(paths[name]) == expected, f'Canonical implementation drift: {name}'
    sampler_provenance = [r for r in previous['source_provenance'] if r['file'] == 'pss_theory.py']
    assert sampler_provenance and all(sha(paths['pss_theory.py']) == r['expected_sha256']
                                      for r in sampler_provenance), 'Archived sampler drift.'
    protocol = dict(
        study='Bounded dependent-density Fig.2 extension at d=3 and d=4',
        estimator=VERSION, family=FAMILY,
        density='f(x)=1+0.7*cos(2*pi*(x1-x2)) on [0,1]^d',
        density_constants=generator.constants(FAMILY, 3),
        conditions=[dict(n=n, d=d) for n, d in CONDITIONS], repetitions=REPS,
        ell_grid=ELLS, expected_datasets=len(records), expected_fits=len(records)*len(ELLS),
        seed_design='numpy.default_rng(SeedSequence([2026100607,n,d,replicate])) with replicate=0,...,99; independent fixed new seeds.',
        seed_schedule=records,
        sampling='Unmodified archived pss_theory.sample, family ridge_medium, float64. The exact cosine rejection sampler is part of the fixed generator. No data-dependent dataset redraw, jitter, clipping, parameter or coefficient retuning.',
        truth='-(1-sqrt(1-0.7^2)+log((1+sqrt(1-0.7^2))/2)) nats; independent of d',
        failure_policy='Keep all 100 replicates and all seven candidates, failures, ties, degeneracy and zero-valid diagnostics. Canonical zero estimate retained when n_valid=0. No coverage filter.',
        interpretation='Extension chosen after existing d=2 and d=5 curves. All outcomes are retained; agreement between empirical and unit-coefficient theory-selected ell is not guaranteed.',
        theory=dict(delta=.05, coefficients=[1, 1],
                    objective='ell^-2 + sqrt(2*d*log(2*n+1)+log(96/delta))*(ell^d/n)^.25',
                    selection_grid=ELLS),
        code_paths={name: str(path) for name, path in paths.items()},
        code_sha256={name: sha(path) for name, path in paths.items()}, workers=WORKERS,
        verified_against=str(ROOT / 'results/synthetic_bounded_20261006/audit.json'),
        environment=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__,
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
    stem = f"c{record['config_id']:02d}_r{record['replicate']:03d}"
    cp = OUT / 'checkpoints' / (stem + '.csv')
    ap = OUT / 'checkpoints' / (stem + '.json')
    if cp.exists() and ap.exists():
        rows = pd.read_csv(cp)
        assert len(rows) == len(ELLS) and set(rows.ell) == set(ELLS)
        return stem
    n, d = record['n'], record['d']
    began = time.monotonic()
    failure = ''
    try:
        rng = np.random.default_rng(np.random.SeedSequence(record['seed_sequence']))
        x = generator.sample(rng, n, d, FAMILY)
        assert x.shape == (n, d) and x.dtype == np.float64 and x.flags['C_CONTIGUOUS']
        ties = int((np.diff(np.sort(x, axis=0), axis=0) == 0).sum())
        array_sha = hashlib.sha256(x.tobytes()).hexdigest()
        finite = bool(np.isfinite(x).all())
        in_support = bool(((x >= 0) & (x <= 1)).all())
        assert finite and in_support
        minimum, maximum = float(x.min()), float(x.max())
    except Exception as exc:
        x, ties, array_sha, finite, in_support = None, None, '', False, False
        minimum = maximum = None
        failure = f'{type(exc).__name__}: {exc}'
    audit = dict(**record, ties=ties, array_sha256=array_sha, minimum=minimum, maximum=maximum,
                 all_finite=finite, in_support=in_support, failure=failure,
                 status='failed' if failure else 'ok')
    csv_record = {k: v for k, v in record.items() if k not in ['seed_sequence', 'seed_state']}
    csv_record['seed_sequence'] = json.dumps(record['seed_sequence'])
    rows = []
    for ell in ELLS:
        started = time.perf_counter()
        try:
            if failure:
                raise RuntimeError(failure)
            result = estimate(x, ell)
            if not np.isfinite(result['estimate']):
                raise ValueError('Nonfinite estimate')
            rows.append(dict(**csv_record, ell=ell, **result,
                             error=result['estimate']-record['truth'],
                             squared_error=(result['estimate']-record['truth'])**2,
                             zero_valid=result['n_valid'] == 0, ties=ties,
                             status='ok', failure='', elapsed_s=time.perf_counter()-started))
        except Exception as exc:
            rows.append(dict(**csv_record, ell=ell, estimate=np.nan, error=np.nan,
                             squared_error=np.nan, coverage=np.nan, n_valid=np.nan,
                             mean_cell_size=np.nan, integrated_mass=np.nan,
                             zero_valid=False, degenerate=None, ties=ties,
                             status='failed', failure=f'{type(exc).__name__}: {exc}',
                             elapsed_s=time.perf_counter()-started))
    audit['seconds'] = time.monotonic()-began
    atomic_csv(pd.DataFrame(rows), cp)
    atomic_json(audit, ap)
    return stem


def main():
    protocol, records = freeze()
    print(json.dumps(dict(status='FROZEN', path=str(OUT / 'protocol.json'),
                          frozen_at_utc=protocol['frozen_at_utc'], datasets=len(records),
                          fits=protocol['expected_fits'])), flush=True)
    began = time.monotonic()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(run_one, row) for row in records]
        for completed, future in enumerate(as_completed(futures), 1):
            future.result()
            if completed % 20 == 0 or completed == len(records):
                print(json.dumps(dict(completed_datasets=completed, total=len(records),
                                      seconds=round(time.monotonic()-began, 2))), flush=True)
    frames, audits = [], []
    for row in records:
        stem = f"c{row['config_id']:02d}_r{row['replicate']:03d}"
        frames.append(pd.read_csv(OUT / 'checkpoints' / (stem+'.csv'), float_precision='round_trip'))
        audits.append(json.loads((OUT / 'checkpoints' / (stem+'.json')).read_text()))
    raw = pd.concat(frames, ignore_index=True).sort_values(['config_id', 'replicate', 'ell'])
    assert len(raw) == protocol['expected_fits'] and not raw.duplicated(['n', 'd', 'replicate', 'ell']).any()
    assert raw.groupby(['n', 'd', 'ell']).size().eq(REPS).all()
    assert raw.groupby(['n', 'd', 'replicate']).size().eq(len(ELLS)).all()
    assert set(raw.ell) == set(ELLS)
    atomic_csv(raw, OUT / 'candidates.csv')
    current = {name: sha(path) for name, path in protocol['code_paths'].items()}
    assert current == protocol['code_sha256'], 'Source or binary changed during experiment.'
    audit = dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
                 candidate_rows=len(raw), datasets=len(audits), conditions=len(CONDITIONS),
                 full_expected_counts=True, repetitions_per_condition_ell=REPS,
                 unique_seed_sequences=len({tuple(a['seed_sequence']) for a in audits}),
                 unique_sample_hashes=len({a['array_sha256'] for a in audits}),
                 failed_fits=int(raw.status.eq('failed').sum()),
                 failed_datasets=sum(a['status'] == 'failed' for a in audits),
                 zero_valid_fits=int(raw.zero_valid.sum()),
                 degenerate_fits=int(raw.degenerate.fillna(False).sum()),
                 ties=sum(a['ties'] or 0 for a in audits),
                 all_generated_datasets_finite=all(a['all_finite'] for a in audits),
                 all_generated_datasets_in_support=all(a['in_support'] for a in audits),
                 minimum_coverage=float(raw.coverage.min()),
                 maximum_coverage=float(raw.coverage.max()),
                 protocol_sha256=sha(OUT / 'protocol.json'),
                 candidates_sha256=sha(OUT / 'candidates.csv'),
                 archived_sampler_hash_verified=True, canonical_hashes_verified=True,
                 code_and_binary_unchanged=True, code_sha256=current,
                 seconds=round(time.monotonic()-began, 2))
    atomic_json(audit, OUT / 'audit.json')
    print(json.dumps(dict(status='COMPLETE', **audit)), flush=True)


if __name__ == '__main__':
    main()
