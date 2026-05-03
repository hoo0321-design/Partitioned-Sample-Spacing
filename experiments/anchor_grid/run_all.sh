#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="$(python experiments/anchor_grid/make_anchor_grid_datasets.py "$@")"
echo "Output directory: ${OUT_DIR}"

Rscript experiments/anchor_grid/run_pss_cadee.R "--base-dir=${OUT_DIR}"
python experiments/anchor_grid/run_knn_um.py --base-dir "${OUT_DIR}"
python experiments/anchor_grid/plot_anchor_grid_results.py --base-dir "${OUT_DIR}"

