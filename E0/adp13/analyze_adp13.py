"""Generate the four preregistered ADP13 diagnostic figures."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np

# Work around the wm environment's native BLAS DLL failure for Matplotlib's
# tiny affine matrices. This affects plotting only, never experiment metrics.
def safe_inv(matrix):
    a=np.asarray(matrix,dtype=float);n=a.shape[0];z=np.concatenate([a.copy(),np.eye(n)],axis=1)
    for c in range(n):
        p=c+int(np.argmax(np.abs(z[c:,c])))
        if p!=c:z[[c,p]]=z[[p,c]]
        z[c]/=z[c,c]
        for r in range(n):
            if r!=c:z[r]-=z[r,c]*z[c]
    return z[:,n:]
def safe_dot(left,right):
    x,y=np.asarray(left),np.asarray(right)
    if x.ndim==2 and y.ndim==2:return (x[:,:,None]*y[None,:,:]).sum(1)
    if x.ndim==2 and y.ndim==1:return (x*y[None,:]).sum(1)
    if x.ndim==1 and y.ndim==2:return (x[:,None]*y).sum(0)
    if x.ndim==1 and y.ndim==1:return (x*y).sum()
    return np.tensordot(x,y,axes=([-1],[0]))
np.linalg.inv=safe_inv;mtransforms.inv=safe_inv;np.dot=safe_dot

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/e0_adp13_sparse_bayesian"
FIG=OUT/"aggregate_3seed/figures"
V={"S1 top-2/no evidence":"s1_top2_no_evidence","S2 top-2/evidence":"s2_top2_evidence","S3 top-3/evidence":"s3_top3_evidence","S4 top-1/evidence":"s4_top1_evidence"}

def rows(folder,seed):
    by={}
    for line in (OUT/folder/f"seed_{seed:03d}/training/round_metrics.jsonl").read_text(encoding="utf-8").splitlines():
        x=json.loads(line);by[x["round"]]=x
    return [by[k] for k in sorted(by)]

def mixing(x):
    count=np.asarray(x["candidate_family_count"],float);purity=np.asarray(x["candidate_purity_fixed"],float);ok=np.isfinite(purity)&(count>0)
    return 1-float((count[ok]*purity[ok]).sum()/count[ok].sum())

def overlap(folder,seed,r):
    z=np.load(OUT/folder/f"seed_{seed:03d}/posterior/round_{r:03d}.npz");top=z["topk"].astype(bool);vals=[]
    for j in range(5):
        for k in range(j+1,5):
            union=(top[:,j]|top[:,k]).sum()
            if union:vals.append((top[:,j]&top[:,k]).sum()/union)
    return max(vals) if vals else 0.0

def main():
    FIG.mkdir(parents=True,exist_ok=True);colors=plt.cm.tab10.colors[:5]
    fig,axes=plt.subplots(3,4,figsize=(17,10),sharex=True,sharey=True)
    for col,(title,folder) in enumerate(V.items()):
        for seed in range(3):
            data=rows(folder,seed);ax=axes[seed,col];rr=[x["round"] for x in data];inc=np.asarray([x["topk_inclusion_frequency"] for x in data])
            for j in range(5):ax.plot(rr,inc[:,j],color=colors[j],lw=1.4,label=f"C{j+1}")
            ax.set_title(f"{title}, seed {seed}");ax.grid(alpha=.2)
    axes[0,0].legend(ncol=5,fontsize=7);[ax.set_xlabel("round") for ax in axes[-1]];[ax.set_ylabel("top-K inclusion") for ax in axes[:,0]]
    fig.tight_layout();fig.savefig(FIG/"A_topk_membership_trajectory.png",dpi=180);plt.close(fig)

    fig,axes=plt.subplots(3,4,figsize=(17,10),sharex=True,sharey=True)
    for col,(title,folder) in enumerate(V.items()):
        for seed in range(3):
            data=rows(folder,seed);ax=axes[seed,col];rr=[x["round"] for x in data]
            ax.plot(rr,[x["full_entropy_mean"] for x in data],lw=1.5,label="full")
            ax.plot(rr,[x["sparse_entropy_mean"] for x in data],lw=1.5,label="sparse")
            ax.set_title(f"{title}, seed {seed}");ax.grid(alpha=.2)
    axes[0,0].legend();[ax.set_xlabel("round") for ax in axes[-1]];[ax.set_ylabel("entropy") for ax in axes[:,0]]
    fig.tight_layout();fig.savefig(FIG/"B_full_vs_sparse_entropy.png",dpi=180);plt.close(fig)

    fig,axes=plt.subplots(3,3,figsize=(13,10),sharex=True)
    for col,(title,folder) in enumerate(list(V.items())[1:]):
        for seed in range(3):
            ax=axes[seed,col]
            for j in range(5):
                paths=sorted((OUT/folder/f"seed_{seed:03d}/existence_tests").glob(f"round_*_C{j+1}.json"));xx=[json.loads(p.read_text(encoding="utf-8")) for p in paths]
                if xx:ax.plot([x["round"] for x in xx],[x["score"] for x in xx],color=colors[j],lw=1.4,label=f"C{j+1}")
            ax.axhline(.01,color="black",ls="--",lw=.8);ax.set_title(f"{title}, seed {seed}");ax.grid(alpha=.2)
    axes[0,0].legend(ncol=5,fontsize=7);[ax.set_xlabel("existence-test round") for ax in axes[-1]];[ax.set_ylabel("max bootstrap upper CI") for ax in axes[:,0]]
    fig.tight_layout();fig.savefig(FIG/"C_candidate_predictive_evidence.png",dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,6));variant_colors=["#4477aa","#228833","#cc3311","#aa4499"]
    for color,(title,folder) in zip(variant_colors,V.items()):
        for seed in range(3):
            data=rows(folder,seed);ax.plot([overlap(folder,seed,x["round"]) for x in data],[mixing(x) for x in data],color=color,lw=1.2,alpha=.65,label=title if seed==0 else None)
    ax.set_xlabel("maximum pairwise top-K Jaccard overlap");ax.set_ylabel("weighted family mixing (1-purity)");ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(FIG/"D_mixing_vs_topk_overlap.png",dpi=180);plt.close(fig)
    print(json.dumps({"pass":True,"files":sorted(p.name for p in FIG.glob("*.png"))},indent=2))
if __name__=="__main__":main()
