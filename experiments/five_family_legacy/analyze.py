"""Replot original five-family results; no estimator fits or parameter changes."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
ARCHIVE=Path("/Users/hojeongwoo/Documents/Codex/2026-05-03/d-dimensional-partitioned-sample-spacing-pss/Partitioned-Sample-Spacing")
SOURCE=ARCHIVE/"results/anchor_grid_all_estimators_20260504_172951"
OUT=ROOT/"results/five_family_legacy_20261006"
FAMILIES=["Normal","Gamma","Beta","Lognormal","Laplace"]
METHODS=["PSS","CADEE","KL","KSG","UM-tKL","UM-tKSG"]
COLORS=dict(zip(METHODS,["#C23B22","#6B7280","#2563EB","#059669","#7C3AED","#D97706"]))
MARKERS=dict(zip(METHODS,["o","s","^","D","v","P"]))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    audit_script=ARCHIVE/"experiments/revision_audit/rebuild_saved_scaling.py"
    spec=importlib.util.spec_from_file_location("historical_scaling_audit",audit_script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    points,selected=module.audited_points(SOURCE)
    assert len(points)==330 and len(selected)==9900
    assert set(points.Method)==set(METHODS) and set(points.Distribution)==set(FAMILIES)
    assert points.N_Reps.eq(30).all()
    assert (points.RMSE>0).all() and (points.Mean_Time_s>0).all()
    points.to_csv(OUT/"summary.csv",index=False)
    selected.to_csv(OUT/"selected_replicates.csv",index=False)
    config=pd.read_csv(SOURCE/"data_generation_config.csv").iloc[0].to_dict()
    specs=[("N scaling","N_Samples","Sample size $n$","RMSE","RMSE_SE","(a) Sample size: $d=5$, $\\rho=0$"),
           ("N scaling","N_Samples","Sample size $n$","Mean_Time_s","Time_SE_s","(b) Runtime: selected-parameter evaluation plus flow training"),
           ("d scaling","Dimensions","Dimension $d$","RMSE","RMSE_SE","(c) Dimension: $n=20,000$, $\\rho=0$"),
           ("rho scaling","Correlation","Copula correlation $\\rho$","RMSE","RMSE_SE","(d) Dependence: $n=20,000$, $d=5$")]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.titlesize":11,
                         "axes.labelsize":10,"xtick.labelsize":9,"ytick.labelsize":9,
                         "axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"savefig.dpi":200})
    fig,axes=plt.subplots(4,5,figsize=(15.5,12.6))
    fig.subplots_adjust(left=.06,right=.99,top=.887,bottom=.115,hspace=.72,wspace=.31)
    handles=[]
    for row,(experiment,xcol,xlabel,ycol,errcol,title) in enumerate(specs):
        subset=points[points.Experiment==experiment]
        for col,family in enumerate(FAMILIES):
            ax=axes[row,col];part=subset[subset.Distribution==family]
            for method in METHODS:
                values=part[part.Method==method].sort_values(xcol)
                # Use the raw positive values: no artificial RMSE floor or axis clipping.
                assert (values[ycol]-values[errcol]>0).all(), "SE interval crosses zero on logarithmic axis"
                line=ax.errorbar(values[xcol],values[ycol],yerr=values[errcol],label=method,
                                 color=COLORS[method],marker=MARKERS[method],markersize=3.4,
                                 linewidth=1.9 if method=="PSS" else 1.2,capsize=2,elinewidth=.65,
                                 linestyle="--" if method=="UM-tKL" else "-",
                                 markerfacecolor="white" if method=="UM-tKL" else COLORS[method])
                if row==0 and col==0:handles.append(line)
            ax.set_yscale("log")
            if xcol!="Correlation":
                ax.set_xscale("log",base=10 if xcol=="N_Samples" else 2)
                ax.xaxis.set_major_formatter(plt.ScalarFormatter())
            ax.set_xticks(sorted(part[xcol].unique()))
            ax.set_title(family,pad=5);ax.set_xlabel(xlabel)
            if col==0:ax.set_ylabel("Time (seconds)" if ycol=="Mean_Time_s" else "RMSE (nats)")
            ax.grid(True,which="major",alpha=.22);ax.set_axisbelow(True)
        fig.text(.06,axes[row,0].get_position().y1+.031,title,fontsize=11,weight="bold")
    fig.suptitle("Five-family synthetic benchmark",x=.06,ha="left",y=.988,fontsize=16,weight="bold")
    fig.legend(handles,METHODS,ncol=6,frameon=False,loc="upper center",bbox_to_anchor=(.53,.967),fontsize=10)
    fig.text(.06,.069,"Saved original results; 30 repetitions per condition. Error bars: +/- 1 Monte Carlo SE, conditional on the selected oracle parameter.",fontsize=9)
    fig.text(.06,.044,"PSS uses the historical rank-spacing / n estimator. PSS and kNN-based methods use same-repetition RMSE oracle selection.",fontsize=9)
    fig.text(.06,.019,"Runtime excludes parameter search; flow methods include training plus their own evaluation. rho is latent Gaussian-copula correlation.",fontsize=9)
    pdfdir=ROOT/"output/pdf/five_family_legacy_20261006";pdfdir.mkdir(parents=True,exist_ok=True)
    fig.savefig(pdfdir/"five_family_benchmark.pdf");fig.savefig(OUT/"five_family_benchmark.png");plt.close(fig)
    protocol=dict(status="PASS",source=str(SOURCE),source_configuration=config,
                  settings=55,methods=6,summary_rows=330,selected_replicate_rows=9900,repetitions=30,
                  source_sha256={name:sha(SOURCE/name) for name in ["combined_summary.csv","r_pss_cadee_estimates.csv","knn_um_estimates.csv","data_generation_config.csv"]},
                  analysis_sha256=sha(__file__),audit_helper_sha256=sha(audit_script),
                  verification="Every saved RMSE and delta-method RMSE SE recomputed from the 30 selected replicate errors; times recomputed from saved raw records.",
                  estimator="historical rank spacing / n; not relabeled as canonical subgrid / N_eff",
                  selection="saved parameters retained; no refitting or retuning; PSS coverage-filtered oracle; kNN methods oracle; CADEE no parameter grid",
                  timing="evaluation at selected parameter plus flow training, excluding parameter search",
                  uncertainty="plus/minus one Monte Carlo SE, not 95% confidence intervals; oracle selection uncertainty excluded",
                  new_simulations=False,raw_values_clipped=False)
    (OUT/"audit.json").write_text(json.dumps(protocol,indent=2)+"\n")
    caption=("Five-family synthetic benchmark using the saved original experiment. Columns show Normal, Gamma, Beta, Lognormal, and Laplace marginals. "
             "Rows show entropy RMSE versus sample size (d=5,rho=0), runtime versus sample size, RMSE versus dimension (n=20000,rho=0), "
             "and RMSE versus Gaussian-copula correlation (n=20000,d=5). Each point summarizes 30 repetitions, with error bars of one Monte Carlo standard error. "
             "The PSS curve uses the historical rank-spacing/n implementation. PSS partition levels and kNN neighborhood sizes retain the saved same-repetition oracle choices; "
             "CADEE has no candidate parameter in this run. Runtime includes selected-parameter evaluation and normalizing-flow training where applicable, and excludes hyperparameter search.\n")
    (OUT/"caption.txt").write_text(caption)
    print(json.dumps({key:protocol[key] for key in ["status","settings","summary_rows","selected_replicate_rows","new_simulations","raw_values_clipped"]},indent=2))


if __name__=="__main__":main()
