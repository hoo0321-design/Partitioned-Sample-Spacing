"""Frozen exploratory lower-correlation Gaussian sweep; no adaptive selection.

Reuse the archived rho=.5 seeds across the prespecified lower correlations.
Copy the rho=.5 baseline without refitting it (apart from eight spot checks).
"""
from __future__ import annotations

import argparse
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
ARCHIVE_ROOT = Path('/Users/hojeongwoo/Documents/Codex/2026-05-03/d-dimensional-partitioned-sample-spacing-pss/Partitioned-Sample-Spacing')
DEFAULT_SOURCE = ARCHIVE_ROOT / 'results/sc_cv_v2_20261005'
DEFAULT_OUTPUT = ROOT / 'results/synthetic_gaussian_rho_20261006'
RHO_GRID = [0.0, 0.1, 0.2, 0.3, 0.4]
CONDITIONS = [(20000, 2), (20000, 5), (20000, 10), (20000, 20),
              (1000, 5), (3000, 5), (10000, 5), (30000, 5)]
sys.path.insert(0, str(ARCHIVE_ROOT / 'experiments/theory_selection'))
sys.path.insert(0, str(ROOT))
import run_dependent_five as generator
from PSS.pss_v2 import VERSION, estimate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grid(d):
    return list(range(1, {2: 40, 5: 40, 10: 12, 20: 8}[d] + 1))


def truth(d, rho):
    return 0.5 * (d * math.log(2 * math.pi * math.e)
                  + (d - 1) * math.log1p(-rho) + math.log1p((d - 1) * rho))


def atomic_csv(frame, path):
    tmp = path.with_suffix('.tmp')
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_json(value, path):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read_csv(path):
    return pd.read_csv(path, float_precision='round_trip')


def get_baseline(source):
    archived = read_csv(source / 'candidates.csv')
    chosen = archived[(archived.family == 'Normal') & (archived.rho == .5)]
    groups = []
    for index, (n, d) in enumerate(CONDITIONS, 1):
        rows = chosen[(chosen.n == n) & (chosen.d == d)].copy()
        expected = 30 * len(grid(d))
        assert len(rows) == expected, (n, d, len(rows), expected)
        assert not rows.duplicated(['replicate', 'ell']).any()
        assert set(rows.replicate) == set(range(1, 31))
        assert set(rows.ell) == set(grid(d))
        assert rows.groupby('replicate').seed.nunique().eq(1).all()
        assert np.allclose(rows.truth, truth(d, .5), atol=1e-13, rtol=0)
        rows['study_config_id'] = 40 + index
        rows['origin'] = 'archived_rho_0.5_unchanged'
        rows['error'] = rows.estimate - rows.truth
        rows['zero_valid'] = rows.n_valid.eq(0)
        rows['status'] = 'ok'
        rows['failure'] = ''
        groups.append(rows.sort_values(['replicate', 'ell']))
    return groups


def freeze(source, out, baseline, workers):
    archived_protocol = json.loads((source / 'protocol.json').read_text())
    paths = {
        'PSS/pss_v2.py': ROOT / 'PSS/pss_v2.py',
        'PSS/pss_v2.cpp': ROOT / 'PSS/pss_v2.cpp',
        'experiments/theory_selection/run_dependent_five.py': Path(generator.__file__),
        'experiments/theory_selection/pss_theory.py': ARCHIVE_ROOT / 'experiments/theory_selection/pss_theory.py',
    }
    source_hashes = {name: sha(path) for name, path in paths.items()}
    for name, digest in source_hashes.items():
        assert archived_protocol['code_sha256'][name] == digest, f'Source drift: {name}'
    dependent_protocol = json.loads((Path(archived_protocol['source']) / 'protocol.json').read_text())
    original_path = Path(generator.original.__file__)
    relative_original = str(original_path.relative_to(ARCHIVE_ROOT))
    assert dependent_protocol['code_sha256'][relative_original] == sha(original_path)
    source_hashes[relative_original] = sha(original_path)
    source_hashes['experiments/synthetic_gaussian_rho/run.py'] = sha(__file__)
    settings_path = Path(archived_protocol['source']) / 'settings.csv'
    assert sha(settings_path) == archived_protocol['source_settings_sha256']
    assert archived_protocol['candidate_grids'] == {str(d): grid(d) for d in [2, 5, 10, 20]}
    seed_schedule = [dict(n=int(g.n.iloc[0]), d=int(g.d.iloc[0]),
                         source_config_id=int(g.config_id.iloc[0]),
                         replicates=[dict(replicate=int(r.replicate), seed=int(r.seed))
                                     for r in g[g.ell == 1].itertuples()]) for g in baseline]
    protocol = dict(
        study='Prespecified exploratory lower-correlation Gaussian sweep',
        estimator=VERSION, lower_rho_grid=RHO_GRID, baseline_rho=0.5,
        conditions=[dict(n=n, d=d) for n, d in CONDITIONS], repetitions=30,
        candidate_grids={str(d): grid(d) for d in [2, 5, 10, 20]},
        new_conditions=40, copied_baseline_conditions=8, new_fits=39000,
        sampling='Exact archived run_dependent_five.sample64 Normal generator; float64; no jitter.',
        seed_design='Reuse each archived rho=.5 condition/replicate seed at all lower rho; paired exploratory comparison, not independent confirmation.',
        seed_schedule=seed_schedule,
        truth='0.5*[d*log(2*pi*e)+(d-1)*log(1-rho)+log(1+(d-1)*rho)] nats',
        baseline='All archived rho=.5 estimate rows copied unchanged; only rep1 ell1 regenerated for each of eight settings as a spotcheck.',
        failure_policy='Record every candidate, failure, tie and clip; no data or candidates dropped. Preserve canonical estimate=0 when n_valid=0 and degenerate flag.',
        interpretation='All five lower correlations reported. No adaptive rho search or coefficient fitting. Normal is outside bounded-support theory assumptions; agreement is exploratory.',
        minima='Analysis must use unrestricted aggregate RMSE over the full archived candidate grid, preserving failures and zero-valid diagnostics.',
        source=str(source), source_sha256={name: sha(source / name) for name in ['protocol.json', 'candidates.csv']},
        archived_settings_sha256=sha(settings_path), code_sha256=source_hashes,
        binary_sha256=sha(ROOT / 'PSS/libpss_v2.dylib'),
        archived_binary_sha256=archived_protocol['binary_sha256'],
        binary_note='Current binary may differ from archived build; Python/C++ sources must match and eight baseline numerical spotchecks must pass.',
        workers=workers, environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__, platform=platform.platform(),
                                         thread_limits={k: os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']}),
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / 'checkpoints').mkdir(exist_ok=True)
    path = out / 'protocol.json'
    if path.exists():
        prior = json.loads(path.read_text())
        protocol['frozen_at_utc'] = prior['frozen_at_utc']
        assert prior == protocol, 'Frozen protocol changed; use a new directory.'
    else:
        protocol['frozen_at_utc'] = datetime.now(timezone.utc).isoformat()
        atomic_json(protocol, path)
    return protocol


def baseline_spotchecks(source, out, groups):
    checks = []
    for group in groups:
        row = group[(group.replicate == 1) & (group.ell == 1)].iloc[0]
        x, clips = generator.sample64('Normal', int(row.n), int(row.d), .5, int(row.seed))
        values = estimate(x, 1)
        difference = abs(values['estimate'] - row.estimate)
        source_audit = read_csv(source / 'checkpoints' / f'audit_{int(row.config_id):03d}.csv')
        saved_hash = source_audit.loc[source_audit.replicate.eq(1), 'array_sha256'].item()
        array_hash = hashlib.sha256(x.tobytes()).hexdigest()
        checks.append(dict(n=int(row.n), d=int(row.d), replicate=1, seed=int(row.seed), ell=1,
                           archived_estimate=float(row.estimate), current_estimate=values['estimate'],
                           absolute_difference=difference, clips=clips,
                           archived_array_sha256=saved_hash, current_array_sha256=array_hash,
                           array_sha256_match=array_hash == saved_hash,
                           passed=bool(difference < 1e-11 and array_hash == saved_hash and clips == 0)))
    atomic_json(checks, out / 'checkpoints/baseline_spotchecks.json')
    assert all(c['passed'] for c in checks), 'Baseline generator/estimator spotcheck failed.'
    print(json.dumps(dict(baseline_spotchecks=len(checks), max_abs_difference=max(c['absolute_difference'] for c in checks), all_data_hashes_match=True)), flush=True)


def copy_baseline(source, out, groups):
    for group in groups:
        cid = int(group.study_config_id.iloc[0])
        atomic_csv(group, out / 'checkpoints' / f'candidates_{cid:03d}.csv')
        original_cid = int(group.config_id.iloc[0])
        audit = read_csv(source / 'checkpoints' / f'audit_{original_cid:03d}.csv')
        audit['study_config_id'] = cid
        audit['n'] = int(group.n.iloc[0])
        audit['d'] = int(group.d.iloc[0])
        audit['rho'] = 0.5
        audit['origin'] = 'archived_rho_0.5_unchanged'
        atomic_csv(audit, out / 'checkpoints' / f'audit_{cid:03d}.csv')


def run_config(records, out_string):
    out = Path(out_string)
    cid = records[0]['study_config_id']
    candidate_path = out / 'checkpoints' / f'candidates_{cid:03d}.csv'
    audit_path = out / 'checkpoints' / f'audit_{cid:03d}.csv'
    expected = 30 * len(grid(records[0]['d']))
    if candidate_path.exists() and audit_path.exists():
        previous, audit = read_csv(candidate_path), read_csv(audit_path)
        assert len(previous) == expected and len(audit) == 30
        return dict(study_config_id=cid, status='cached', rows=expected)
    start = time.monotonic()
    rows, audits = [], []
    for record in records:
        n, d, rho, seed = record['n'], record['d'], record['rho'], record['seed']
        data_failure = ''
        try:
            x, clips = generator.sample64('Normal', n, d, rho, seed)
            ties = int((np.diff(np.sort(x, axis=0), axis=0) == 0).sum())
            array_hash = hashlib.sha256(x.tobytes()).hexdigest()
        except Exception as exc:
            x, clips, ties, array_hash = None, None, None, ''
            data_failure = f'{type(exc).__name__}: {exc}'
        audits.append(dict(**record, array_sha256=array_hash, clips=clips, ties=ties,
                           failure=data_failure, status='failed' if data_failure else 'ok'))
        for ell in grid(d):
            began = time.perf_counter()
            try:
                if data_failure:
                    raise RuntimeError(data_failure)
                values = estimate(x, ell)
                if not np.isfinite(values['estimate']):
                    raise ValueError('Nonfinite estimate')
                rows.append(dict(**record, ell=ell, **values, error=values['estimate'] - record['truth'],
                                 zero_valid=values['n_valid'] == 0, status='ok', failure='', elapsed_s=time.perf_counter() - began))
            except Exception as exc:
                rows.append(dict(**record, ell=ell, estimate=np.nan, error=np.nan,
                                 status='failed', failure=f'{type(exc).__name__}: {exc}', elapsed_s=time.perf_counter() - began))
    assert len(rows) == expected
    frame = pd.DataFrame(rows)
    atomic_csv(frame, candidate_path)
    atomic_csv(pd.DataFrame(audits), audit_path)
    return dict(study_config_id=cid, status='complete', rows=len(rows), n=records[0]['n'], d=records[0]['d'],
                rho=records[0]['rho'], failures=int(frame.status.eq('failed').sum()), seconds=round(time.monotonic() - start, 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    groups = get_baseline(source)
    protocol = freeze(source, out, groups, args.workers)
    print(f"FROZEN {out / 'protocol.json'} {protocol['frozen_at_utc']}", flush=True)
    baseline_spotchecks(source, out, groups)
    copy_baseline(source, out, groups)
    jobs = []
    for rho_index, rho in enumerate(RHO_GRID):
        for condition_index, group in enumerate(groups, 1):
            cid = rho_index * 8 + condition_index
            n, d = int(group.n.iloc[0]), int(group.d.iloc[0])
            records = [dict(study_config_id=cid, config_id=int(row.config_id), family='Normal', n=n, d=d, rho=rho,
                            replicate=int(row.replicate), seed=int(row.seed), truth=truth(d, rho),
                            origin='new_lower_rho_same_seed_schedule', assumption_matched=False)
                       for row in group[group.ell.eq(1)].itertuples()]
            jobs.append(records)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_config, records, str(out)) for records in jobs]
        for future in as_completed(futures):
            print(json.dumps(future.result()), flush=True)
    paths = [out / 'checkpoints' / f'candidates_{cid:03d}.csv' for cid in range(1, 49)]
    frames = [read_csv(path) for path in paths]
    total = sum(len(frame) for frame in frames)
    failures = sum(int(frame.status.eq('failed').sum()) for frame in frames)
    assert total == 46800, total
    print(json.dumps(dict(status='SWEEP_COMPLETE', candidate_rows=total, failures=failures,
                          conditions=48, output=str(out))), flush=True)


if __name__ == '__main__':
    main()
