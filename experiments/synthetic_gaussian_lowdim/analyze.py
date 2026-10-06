"""Five-setting low-dimensional Gaussian extension; preserve every outcome."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"results/synthetic_gaussian_lowdim_20261006"
PAIRS=[(30000,3),(100000,3),(300000,3),(100000,2),(100000,4)]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    raw=pd.read_csv(OUT/"candidates.csv")
    assert len(raw)==1200
    assert not raw.duplicated(["n","d","replicate","ell"]).any()
    assert set(map(tuple,raw[["n","d"]].drop_duplicates().to_numpy()))==set(PAIRS)
    assert raw.groupby(["n","d","ell"]).size().eq(30).all()
    assert set(raw.ell)==set(range(1,9)) and raw.rho.eq(.2).all()
    assert np.isfinite(raw[["estimate","truth","coverage"]]).all().all()
    truth=.5*(raw.d*np.log(2*np.pi*np.e)+(raw.d-1)*np.log(.8)+np.log1p((raw.d-1)*.2))
    np.testing.assert_allclose(raw.truth,truth,rtol=0,atol=1e-12)
    raw["error"]=raw.estimate-raw.truth
    np.testing.assert_allclose(raw.coverage,raw.n_valid/raw.n,rtol=0,atol=1e-14)
    rng=np.random.default_rng(2026100605)
    rows=[]
    for (n,d,ell),group in raw.groupby(["n","d","ell"]):
        errors=group.sort_values("replicate").error.to_numpy()
        boot=errors[rng.integers(len(errors),size=(2000,len(errors)))]
        low,high=np.quantile(np.sqrt(np.mean(boot**2,axis=1)),[.025,.975])
        rows.append(dict(n=int(n),d=int(d),ell=int(ell),reps=len(errors),rmse=float(np.sqrt(np.mean(errors**2))),
                         rmse_low=low,rmse_high=high,bias=float(errors.mean()),coverage_min=float(group.coverage.min())))
    per=pd.DataFrame(rows)
    comparisons=[]
    for n,d in PAIRS:
        group=per[(per.n==n)&(per.d==d)].sort_values("ell")
        best=group.sort_values(["rmse","ell"]).iloc[0]
        lam=2*d*np.log(2*n+1)+np.log(96/.05)
        boundell=min(group.ell,key=lambda ell:(ell**-2+np.sqrt(lam)*(float(ell)**d/n)**.25,ell))
        bound=group[group.ell==boundell].iloc[0]
        rateell=max(1,int(np.floor((n/(d**6*np.log(n)**2))**(1/(d+8))+.5)))
        rate=group[group.ell==rateell].iloc[0]
        comparisons.append(dict(n=n,d=d,empirical_ell=int(best.ell),empirical_rmse=float(best.rmse),
                                bound_ell=int(boundell),bound_rmse=float(bound.rmse),
                                bound_rmse_ratio=float(bound.rmse/best.rmse),bound_match=boundell==int(best.ell),
                                rate_C1_ell=rateell,rate_C1_rmse=float(rate.rmse),
                                empirical_coverage_min=float(best.coverage_min),
                                empirical_at_grid_upper=int(best.ell)==8))
    comparison=pd.DataFrame(comparisons)
    per.to_csv(OUT/"per_ell_summary.csv",index=False)
    comparison.to_csv(OUT/"optimal_ell_comparison.csv",index=False)
    # Paper figure at a two-column width; methods and qualifications belong in the text/caption.
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8.5,"axes.titlesize":8.8,
                         "axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"savefig.dpi":300})
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.15))
    panels=[[(100000,d) for d in [2,3,4]],[(n,3) for n in [30000,100000,300000]]]
    for i,panel in enumerate(panels):
        ax=axes[i]
        for (n,d),color in zip(panel,["#0072B2","#D55E00","#009E73"]):
            group=per[(per.n==n)&(per.d==d)].sort_values("ell")
            selected=comparison[(comparison.n==n)&(comparison.d==d)].iloc[0]
            label=f"$d={d}$" if i==0 else f"$n={n:,}$"
            ax.plot(group.ell,group.rmse,color=color,lw=1.4,marker="o",markersize=2.6,label=label)
            ax.fill_between(group.ell.to_numpy(),group.rmse_low.to_numpy(),group.rmse_high.to_numpy(),color=color,alpha=.14,lw=0)
            ax.scatter([selected.empirical_ell],[selected.empirical_rmse],color=color,marker="*",s=90,edgecolors="black",linewidths=.5,zorder=6)
            ax.scatter([selected.bound_ell],[selected.bound_rmse],facecolors="none",edgecolors=color,marker="s",s=76,linewidths=1.2,zorder=5)
        ax.set(yscale="log",xticks=range(1,9),xlabel="Partitions per coordinate $\\ell$",ylabel="Entropy RMSE (nats)")
        ax.set_title("(a) Fixed $n=100,000$; varying dimension" if i==0 else "(b) Fixed $d=3$; varying sample size")
        ax.grid(alpha=.22);ax.set_axisbelow(True)
        ax.legend(frameon=False,loc="upper left",fontsize=8,handlelength=1.8,
                  handletextpad=.55,labelspacing=.35,borderaxespad=.6)
    handles=[Line2D([],[],color="black",marker="*",linestyle="none",markersize=9,label="RMSE-minimizing $\\ell$"),
             Line2D([],[],color="black",marker="s",markerfacecolor="none",linestyle="none",markersize=7,label="Theory-selected $\\ell$ ($\\delta=0.05$)")]
    fig.legend(handles=handles,ncol=2,frameon=False,loc="upper center",bbox_to_anchor=(.5,1.0),fontsize=9)
    fig.subplots_adjust(left=.085,right=.988,top=.79,bottom=.17,wspace=.29)
    pdfdir=ROOT/"output/pdf/synthetic_gaussian_lowdim_20261006";pdfdir.mkdir(parents=True,exist_ok=True)
    fig.savefig(pdfdir/"gaussian_lowdim_ell_rmse.pdf");fig.savefig(OUT/"gaussian_lowdim_ell_rmse.png");plt.close(fig)
    notes=["Low-dimensional Gaussian extension", "",
           "rho=.2; five fixed settings, 30 new-seed repetitions each, ell1..8; all conditions reported.",
           "The bound rule minimizes ell^-2+sqrt(2*d*log(2*n+1)+log(96/.05))*(ell^d/n)^.25 over the candidate grid.",
           "Figure legend: RMSE-minimizing ell (star); Theory-selected ell (square). The latter is the unit-coefficient bound-derived criterion above.",
           "The dimension-aware rate with C=1 uses nearest-integer rounding and a lower bound of one.",
           "No coefficient is fitted and the two formulas are kept distinct.",
           "The previous rho sweep motivated this design; new seeds do not make the chosen design a general confirmation.",
           "Increasing n reduces finite-sample estimation effects. Reducing d also lowers total correlation at fixed rho.",
           "All theoretical choices minimize the stated bound shape or implement a rate, not actual finite-sample RMSE.",
           "Empirical minima and RMSE ratios are descriptive over the same 30 repetitions, with no coverage filter.",
           "The x grid is finite; an upper-boundary minimum would require separately labeled grid expansion.",
           "Normal distributions do not satisfy the stated bounded-support/positive-lower-bound density assumptions.", "",
           comparison.to_string(index=False)]
    (OUT/"notes.txt").write_text("\n".join(notes)+"\n")
    audit=dict(status="PASS",datasets=150,candidate_rows=len(raw),settings=5,bound_matches=int(comparison.bound_match.sum()),
               minimum_training_coverage=float(raw.coverage.min()),upper_boundary_minima=int(comparison.empirical_at_grid_upper.sum()),
               analysis_sha256=sha(__file__),raw_sha256=sha(OUT/"candidates.csv"),protocol_sha256=sha(OUT/"protocol.json"),
               bootstrap_seed=2026100605,bootstrap_replicates=2000)
    (OUT/"analysis_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(comparison.to_string(index=False))


if __name__=="__main__":main()
