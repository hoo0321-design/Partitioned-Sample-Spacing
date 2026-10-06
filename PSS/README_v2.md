# Manuscript-aligned PSS v2

`pss_v2.cpp` is the single density implementation for the Python
entry points. It uses the smoothed xi sub-grid, left-open/right-closed cells,
actual joint evaluation at each observation, and the `N_eff` denominator.
The density is zero outside the global empirical box and outside each cell's
marginal empirical range. It is NOT renormalized. Exact coordinate ties use
the manuscript's degenerate-zero convention; no automatic jitter is added.

## Build and test

```sh
python PSS/build_pss_v2.py
python PSS/test_density.py
```

Requires a C++17 compiler and NumPy. The included density tests compare against
an independent literal subgrid implementation.
Build on the destination machine; compiled libraries are not source artifacts.

## Python

```python
from PSS.pss_v2 import estimate, evaluate, select_sc_cv

selection = select_sc_cv(X, range(1, 6), tau=.99, n_min=10, seed=42)
result = estimate(X, selection["ell_star"])
```

`evaluate(train, query, ell)` returns log densities, training-cell sizes,
uncovered-point reasons, and integrated training density mass. Both entropy
and CV call this same implementation. Every evaluation fits from `train`
only. Out-of-range queries are not clamped to an occupied boundary cell.

## Historical implementations

The R scripts already in this repository retain their historical definitions.
They are not silently replaced by the Python/C++ v2 implementation, and their
saved results must not be relabeled as v2. The October experiment sources use
`PSS.pss_v2` explicitly.

## SC-CV contract

- Three folds by default; the grid and thresholds are chosen before seeing truth.
- A covered point has a finite positive fitted density. A stably covered
  point additionally has at least `n_min` training observations in its cell.
- `S` uses ALL validation observations as denominator. The score averages
  negative log density over COVERED observations, not only stable observations.
- Select minimum score among candidates with `S >= tau`, breaking ties by ell.
- If none is feasible, maximize `S`, then minimize score, then ell. Return
  an explicit failure if no validation point is covered at any candidate.
- No penalty, one-SE rule, truth-based cap, or silent threshold relaxation.
- This is not a full cross-entropy/KL criterion, and coverage does not certify
  entropy/MI accuracy or the theorem's occupancy/refinement conditions.

`mixed_mi` is an un-clipped entropy-difference diagnostic, not a validated
feature selector. The raw Energy data have coordinate ties, and the old
jittered feature-selection path has substantial bias even under v2.
