"""Bounded smooth densities and the exact PDF-version PSS estimator."""
from __future__ import annotations

import ctypes
import functools
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
FIELDS = ["estimate", "legacy_estimate", "coverage", "occupied_cells", "min_cell_size",
          "mean_cell_size", "singleton_cells", "boundary_fraction", "integrated_mass",
          "small_cell_point_fraction", "n_valid", "total_cells"]
FAMILIES = [
    dict(name="uniform", kind="uniform", theta=0.0, frequency=1, pilot=True),
    dict(name="linear_margin", kind="linear", theta=0.8, frequency=1, pilot=True),
    dict(name="cosine_margin", kind="margin", theta=0.8, frequency=1, pilot=True),
    dict(name="fgm", kind="fgm", theta=0.9, frequency=1, pilot=True),
    dict(name="ridge_weak", kind="ridge", theta=0.3, frequency=1, pilot=True),
    dict(name="ridge_medium", kind="ridge", theta=0.7, frequency=1, pilot=True),
    dict(name="ridge_strong", kind="ridge", theta=0.9, frequency=1, pilot=True),
    dict(name="ridge_frequency2", kind="ridge", theta=0.7, frequency=2, pilot=True),
    dict(name="ridge_frequency4", kind="ridge", theta=0.7, frequency=4, pilot=True),
    dict(name="smooth_bump", kind="bump", theta=0.8, frequency=1, pilot=False),
    dict(name="all_pairs", kind="pairs", theta=0.7, frequency=1, pilot=False),
]
FAMILY_MAP = {f["name"]: f for f in FAMILIES}
COEFFICIENTS = [0.5,0.75,1.0,1.25,1.5,2.0,2.5,3.0,4.0]


@functools.lru_cache(None)
def library():
    suffix = "dylib" if sys.platform == "darwin" else "so"
    lib = ctypes.CDLL(str(HERE / ("libpss_theory." + suffix)))
    lib.pss_estimate.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
                                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")]
    lib.pss_estimate.restype = ctypes.c_int
    return lib


def estimate(x, ell, check_ties=False):
    x = np.ascontiguousarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all() or len(x)<2:
        raise ValueError("A finite n-by-d array with n>=2 is required.")
    n,d = x.shape
    if ell < 1 or int(ell)!=ell or ell**d >= 2**63:
        raise ValueError("Invalid partition count or unsafe cell encoding.")
    if check_ties and np.any(np.diff(np.sort(x,axis=0),axis=0)==0):
        raise ValueError("This continuous-density experiment excludes null tie events.")
    out = np.empty(12, dtype=np.float64)
    rc = library().pss_estimate(x,n,d,int(ell),out)
    if rc: raise RuntimeError(f"PSS core returned {rc}")
    return dict(zip(FIELDS,out.tolist()))


def reference_estimate(x,ell):
    """Small, independent translation of equations (3)--(5), for testing."""
    x = np.asarray(x,dtype=float)
    n,d=x.shape
    bins=np.clip(np.ceil(ell*(x-x.min(0))/(x.max(0)-x.min(0))).astype(int)-1,0,ell-1)
    values=[]
    mass=0.0
    for cell in np.unique(bins,axis=0):
        z=x[np.all(bins==cell,axis=1)]
        s=len(z)
        if s<2: continue
        m=int(math.sqrt(s)+0.5)
        logs=np.full(s,math.log(s/n)); valid=np.ones(s,dtype=bool)
        cell_mass=s/n
        for j in range(d):
            t=np.sort(z[:,j])
            T=lambda r:t[max(1,min(s,r))-1]
            xi=[T(1)]+[sum(T(u) for u in range(r-m,r+m))/(2*m) for r in range(1,s+1)]+[T(s)]
            gaps=np.array([T(r+m)-T(r-m) for r in range(s+1)])
            densities=np.divide(2*m,s*gaps,out=np.zeros(s+1),where=gaps>0)
            a=np.maximum(0,np.searchsorted(xi,z[:,j],side="left")-1)
            good=densities[a]>0
            valid &= good
            logs[good]+=np.log(densities[a[good]])
            cell_mass*=np.dot(np.diff(xi),densities)
        mass+=cell_mass
        values.extend(logs[valid])
    return dict(estimate=-float(np.mean(values)) if values else 0.0,
                coverage=len(values)/n,integrated_mass=mass)


def density(x,family):
    f=FAMILY_MAP[family] if isinstance(family,str) else family
    theta=f["theta"]; kind=f["kind"]; k=f["frequency"]
    if kind=="uniform": return np.ones(len(x))
    if kind=="linear": return 1+theta*(2*x[:,0]-1)
    if kind=="margin": return 1+theta*np.cos(2*np.pi*x[:,0])
    if kind=="fgm": return 1+theta*(2*x[:,0]-1)*(2*x[:,1]-1)
    if kind=="ridge": return 1+theta*np.cos(2*np.pi*k*(x[:,0]-x[:,1]))
    if kind=="pairs":
        p=x.shape[1]//2
        return 1+theta*np.cos(2*np.pi*(x[:,:2*p:2]-x[:,1:2*p:2])).mean(axis=1)
    q=lambda t:30*t*t*(1-t)**2
    return 1-theta+theta*q(x[:,0])*q(x[:,1])


def cosine_draws(rng,n,theta,k=1):
    chunks=[]; remaining=n
    while remaining:
        batch=max(64,int(remaining*(1+theta)*1.05))
        t=rng.random(batch)
        accepted=t[rng.random(batch)*(1+theta)<1+theta*np.cos(2*np.pi*k*t)]
        accepted=accepted[:remaining]; chunks.append(accepted); remaining-=len(accepted)
    return np.concatenate(chunks)


def sample(rng,n,d,family):
    f=FAMILY_MAP[family] if isinstance(family,str) else family
    theta=f["theta"]; kind=f["kind"]
    x=rng.random((n,d))
    if kind=="linear":
        u=rng.random(n)
        x[:,0]=2*u/(1-theta+np.sqrt((1-theta)**2+4*theta*u))
    elif kind=="margin": x[:,0]=cosine_draws(rng,n,theta)
    elif kind=="ridge": x[:,1]=(x[:,0]+cosine_draws(rng,n,theta,f["frequency"]))%1
    elif kind=="pairs":
        j=2*rng.integers(d//2,size=n); i=np.arange(n)
        x[i,j+1]=(x[i,j]+cosine_draws(rng,n,theta))%1
    elif kind=="bump":
        mask=rng.random(n)<theta
        x[mask,:2]=rng.beta(3,3,size=(int(mask.sum()),2))
    elif kind=="fgm":
        remaining=n; offset=0
        while remaining:
            z=rng.random((max(64,int(remaining*(1+theta)*1.05)),2))
            z=z[rng.random(len(z))*(1+theta)<1+theta*(2*z[:,0]-1)*(2*z[:,1]-1)][:remaining]
            x[offset:offset+len(z),:2]=z
            remaining-=len(z); offset+=len(z)
    return np.ascontiguousarray(x)


def constants(family,d):
    f=FAMILY_MAP[family]; a=f["theta"]; kind=f["kind"]
    if kind=="uniform": return dict(c=1.0,C=1.0,L=0.0)
    bounds=dict(c=1-a,C=1+a)
    if kind=="linear": L=2*a
    elif kind=="margin": L=2*np.pi*a
    elif kind=="fgm": L=2*a*np.sqrt(2)
    elif kind=="ridge": L=2*np.pi*a*f["frequency"]*np.sqrt(2)
    elif kind=="pairs": L=2*np.pi*a*np.sqrt(2/(d//2))
    else:
        bounds["C"]=1-a+a*(15/8)**2
        L=a*np.sqrt(2)*(15/8)*(10/np.sqrt(3))
    return dict(**bounds,L=float(L))


@functools.lru_cache(None)
def true_entropy(family,d,order=256):
    f=FAMILY_MAP[family]; kind=f["kind"]; a=f["theta"]
    if kind=="uniform": return 0.0
    if kind in ("margin","ridge"):
        s=math.sqrt(1-a*a)
        return -(1-s+math.log((1+s)/2))
    if kind=="pairs":
        p=d//2; degree=180
        single=np.array([math.comb(k,k//2)/2**k if k%2==0 else 0.0 for k in range(degree+1)])
        moments=single.copy()
        for i in range(2,p+1):
            result=np.zeros_like(moments)
            for k in range(0,degree+1,2):
                result[k]=sum(math.comb(k,j)*moments[j]*single[k-j]*((i-1)/i)**j*(1/i)**(k-j)
                              for j in range(0,k+1,2))
            moments=result
        return -sum(a**k*moments[k]/(k*(k-1)) for k in range(2,degree+1,2))
    t,w=np.polynomial.legendre.leggauss(order); t=(t+1)/2; w=w/2
    if kind=="linear":
        v=1+a*(2*t-1)
        return -float(np.dot(w,v*np.log(v)))
    xx,yy=np.meshgrid(t,t,indexing="ij")
    v=density(np.column_stack([xx.ravel(),yy.ravel()]),f).reshape(order,order)
    return -float(w@(v*np.log(v))@w)


def ell_grid(n,d):
    # Fixed before sampling; include modest sparse-cell stress without huge grids.
    cap=min(40,max(2,int((n/8)**(1/d))))
    grid=set(range(1,cap+1))
    grid.update(rate_ell(n,d,c) for c in COEFFICIENTS)
    return sorted(grid)


def lambda_n(n,d):
    return 2*d*math.log(2*n+1)+math.log(96)+4*math.log(n)


def bound_objective(n,d,ell):
    return ell**(-2)+math.sqrt(lambda_n(n,d))*(ell**d/n)**0.25


def bound_continuous(n,d):
    return (8**4*n/(d**4*lambda_n(n,d)**2))**(1/(d+8))


def rate_ell(n,d,coefficient=1.0):
    return max(1,int(math.floor(coefficient*(n/math.log(n)**2)**(1/(d+8))+0.5)))


def configurations(smoke=False):
    settings={(n,d) for n in [1000,3000,10000,30000,100000] for d in [2,5]}
    settings.update((20000,d) for d in [2,3,5,10])
    if smoke: settings={(1000,2),(3000,5)}
    families=FAMILIES if not smoke else [FAMILIES[i] for i in [0,5,9,10]]
    return [dict(family=f["name"],family_id=FAMILIES.index(f),n=n,d=d,pilot=f["pilot"])
            for f in families for n,d in sorted(settings)]
