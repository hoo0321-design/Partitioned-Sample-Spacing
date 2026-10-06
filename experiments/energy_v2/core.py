"""Training-only feature selection for the Energy downstream experiment."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"PSS"))
from pss_v2 import estimate, evaluate

METHODS=["PSS-SC", "Joint Ross MI", "Univariate Ross MI"]
K_GRID=[1,3,5,10,15,20]
TAUS=[.90,.95,.99]
N_MINS=[5,10]
NOISE=[1e-4,1e-3]
ELL_GRID=[1,2,3,4,5]
CHECKPOINTS=[5,10,20]
SELECTION_CAP=4000


def selector_configs(method):
    if method=="PSS-SC":
        return [dict(method=method,noise=noise,tau=tau,n_min=minimum)
                for noise in NOISE for tau in TAUS for minimum in N_MINS]
    return [dict(method=method,noise=noise,k=k) for noise in NOISE for k in K_GRID]


def classifier_configs():
    return [dict(scaler=scaler,C=c,gamma="scale")
            for scaler in ["standard","robust"] for c in [.1,1.,10.]]


def config_key(config):
    import json
    return json.dumps(config,sort_keys=True,separators=(",",":"))


def temporal_split(n,start,end,gap=144):
    validation_start=int(np.floor(n*start));validation_end=int(np.floor(n*end))
    training_end=validation_start-gap
    if training_end<2 or validation_end<=validation_start:
        raise ValueError("Insufficient observations for the temporal split")
    return np.arange(training_end),np.arange(validation_start,validation_end)


def labels_from_training(train_target,other_target,quantile=.5):
    threshold=float(np.quantile(train_target,quantile))
    return (np.asarray(train_target)>threshold).astype(int), (np.asarray(other_target)>threshold).astype(int), threshold


def make_selection_data(train_x,train_y,noise,seed,cap=SELECTION_CAP):
    x=np.asarray(train_x,dtype=float);y=np.asarray(train_y)
    if not np.isfinite(x).all(): raise ValueError("Missing values require a predeclared training-only imputation rule")
    std=x.std(0);active=np.flatnonzero(std>0)
    if len(active)<20: raise ValueError("Fewer than 20 nonconstant training predictors")
    center=x.mean(0)
    scale=np.where(std>0,std,1.)
    rng=np.random.default_rng(seed)
    rows=np.sort(rng.choice(len(x),min(cap,len(x)),replace=False))
    # Smoothing applies to selection scores only. The classifier sees raw inputs.
    z=(x[rows]-center)/scale
    z+=rng.uniform(-noise/2,noise/2,size=z.shape)
    if np.any(np.diff(np.sort(z[:,active],axis=0),axis=0)==0):
        raise ValueError("Unresolved coordinate ties; never silently retry a seed")
    ys=y[rows]
    if min(np.bincount(ys,minlength=2))<=max(K_GRID):
        raise ValueError("Insufficient class counts for the predeclared k grid")
    return z,ys,active,rows


def ross_mi(x,y,k):
    """Vector continuous/discrete Ross estimator with the sklearn count convention.

    Euclidean same-label kth-neighbor radii, shrunk by one floating-point step;
    full-sample counts include the query point. Raw negative scores are retained.
    """
    x=np.asarray(x,dtype=float);y=np.asarray(y)
    if x.ndim==1: x=x[:,None]
    if len(x)!=len(y) or len(x)<3 or not np.isfinite(x).all() or k<1:
        raise ValueError("Invalid Ross inputs")
    radius=np.empty(len(x));label_counts=np.empty(len(x));ks=np.empty(len(x))
    for label in np.unique(y):
        mask=y==label;count=int(mask.sum())
        if count<2: raise ValueError("Singleton labels are not silently removed")
        kval=min(int(k),count-1)
        distances=cKDTree(x[mask]).query(x[mask],k=kval+1,workers=1)[0][:,-1]
        if (distances<=0).any(): raise ValueError("Tied neighbors require explicit smoothing")
        radius[mask]=np.nextafter(distances,0.)
        label_counts[mask]=count;ks[mask]=kval
    counts=cKDTree(x).query_ball_point(x,radius,return_length=True,workers=1)
    return float(digamma(len(x))+np.mean(digamma(ks))-np.mean(digamma(label_counts))-np.mean(digamma(counts)))


def candidate_choice(rows,tau,n_min):
    key=f"stable_min_{n_min}"
    finite=[r for r in rows if np.isfinite(r["cv_score"])]
    feasible=[r for r in finite if r[key]>=tau]
    if feasible:
        row=min(feasible,key=lambda r:(r["cv_score"],r["ell"]))
        rule="all_models_stable_coverage_constrained"
    elif finite:
        row=min(finite,key=lambda r:(-r[key],r["cv_score"],r["ell"]))
        rule="fallback_max_min_component_stable_coverage"
    else:
        raise ValueError("All partition candidates have zero validation coverage")
    return dict(**row,selected_stable_coverage=row[key],fallback=not bool(feasible),
                feasible_candidates=len(feasible),selection_rule=rule)


class Selector:
    def __init__(self,x,y,active,seed):
        self.x=x;self.y=y;self.active=list(map(int,active));self.seed=seed
        self.pss_cache={};self.ross_cache={}
        # Common randomized folds, wholly within the training selection sample.
        self.fold_id=np.empty(len(y),dtype=int)
        rng=np.random.default_rng(seed+91)
        for label in np.unique(y):
            rows=np.flatnonzero(y==label)
            self.fold_id[rows]=rng.permutation(np.arange(len(rows))%3)

    def pss_subset(self,features):
        features=tuple(sorted(features))
        if features in self.pss_cache: return self.pss_cache[features]
        x=self.x[:,features];y=self.y
        components=[np.ones(len(y),bool),y==0,y==1]
        probabilities=[1.,float(np.mean(y==0)),float(np.mean(y==1))]
        table=[]
        for ell in ELL_GRID:
            full=[];stats=[]
            for mask in components:
                z=x[mask];folds=self.fold_id[mask]
                full.append(estimate(z,ell))
                covered=0;logsum=0.;stable={minimum:0 for minimum in N_MINS}
                for fold in range(3):
                    values=evaluate(z[folds!=fold],z[folds==fold],ell)
                    good=np.isfinite(values["log_density"])
                    covered+=int(good.sum());logsum+=float(values["log_density"][good].sum())
                    for minimum in N_MINS:
                        stable[minimum]+=int((good&(values["cell_size"]>=minimum)).sum())
                stats.append(dict(nll=-logsum/covered if covered else float("inf"),coverage=covered/len(z),
                                  **{f"s_{minimum}":count/len(z) for minimum,count in stable.items()}))
            score=full[0]["estimate"]-sum(p*item["estimate"] for p,item in zip(probabilities[1:],full[1:]))
            table.append(dict(ell=ell,score=score,cv_score=stats[0]["nll"],
                              pooled_cv_coverage=stats[0]["coverage"],
                              minimum_component_cv_coverage=min(s["coverage"] for s in stats),
                              training_coverage=full[0]["coverage"],
                              minimum_conditional_training_coverage=min(f["coverage"] for f in full[1:]),
                              integrated_mass=full[0]["integrated_mass"],
                              **{f"stable_min_{minimum}":min(s[f"s_{minimum}"] for s in stats) for minimum in N_MINS}))
        self.pss_cache[features]=table
        return table

    def ross_subset(self,features,k):
        key=(tuple(sorted(features)),k)
        if key not in self.ross_cache:
            self.ross_cache[key]=ross_mi(self.x[:,key[0]],self.y,k)
        return self.ross_cache[key]

    def path(self,config,max_features=20):
        start=time.perf_counter();selected=[];history=[]
        if config["method"]=="Univariate Ross MI":
            values=[(self.ross_subset([j],config["k"]),j) for j in self.active]
            order=sorted(values,key=lambda pair:(-pair[0],pair[1]))[:max_features]
            for score,j in order:
                selected.append(j);history.append(dict(step=len(selected),feature=j,features=selected.copy(),score=score))
        elif config["method"]=="PSS ell=1":
            values=[]
            for j in self.active:
                z=self.x[:,[j]]
                h=estimate(z,1)["estimate"]
                score=h-sum(np.mean(self.y==label)*estimate(z[self.y==label],1)["estimate"] for label in [0,1])
                values.append((score,j))
            total=0.
            for score,j in sorted(values,key=lambda pair:(-pair[0],pair[1]))[:max_features]:
                selected.append(j);total+=score
                history.append(dict(step=len(selected),feature=j,features=selected.copy(),score=total,ell=1))
        else:
            for step in range(1,max_features+1):
                choices=[]
                for j in self.active:
                    if j in selected: continue
                    if config["method"]=="PSS-SC":
                        result=candidate_choice(self.pss_subset(selected+[j]),config["tau"],config["n_min"])
                    else:
                        result=dict(score=self.ross_subset(selected+[j],config["k"]))
                    choices.append((result["score"],j,result))
                score,j,result=min(choices,key=lambda entry:(-entry[0],entry[1]))
                selected.append(j)
                history.append(dict(**result,step=step,feature=j,features=selected.copy()))
        return dict(history=history,selection_seconds=time.perf_counter()-start,
                    n_selection=len(self.x),definition="pss_smoothed_subgrid_neff_v2" if config["method"].startswith("PSS") else "Ross2014_raw_score")


def classify(train_x,train_y,test_x,test_y,features,config):
    features=sorted(features)
    scaler=StandardScaler() if config["scaler"]=="standard" else RobustScaler()
    start=time.perf_counter()
    z=scaler.fit_transform(train_x[:,features]);q=scaler.transform(test_x[:,features])
    model=SVC(C=config["C"],gamma=config["gamma"],cache_size=256,class_weight="balanced")
    model.fit(z,train_y);train_seconds=time.perf_counter()-start
    start=time.perf_counter();prediction=model.predict(q);decision=model.decision_function(q)
    test_seconds=time.perf_counter()-start
    return dict(accuracy=float(accuracy_score(test_y,prediction)),
                balanced_accuracy=float(balanced_accuracy_score(test_y,prediction)),
                auc=float(roc_auc_score(test_y,decision)),
                train_seconds=train_seconds,test_seconds=test_seconds,
                prediction=prediction,decision=decision)
