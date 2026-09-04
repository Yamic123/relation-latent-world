"""Generate ADP16 preregistered figures and aggregate summary."""
from __future__ import annotations
import json,sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mt
import numpy as np

def inv(a):
    a=np.asarray(a,float);n=len(a);z=np.concatenate([a.copy(),np.eye(n)],1)
    for c in range(n):
        p=c+int(np.argmax(np.abs(z[c:,c])));z[[c,p]]=z[[p,c]];z[c]/=z[c,c]
        for r in range(n):
            if r!=c:z[r]-=z[r,c]*z[c]
    return z[:,n:]
def dot(x,y):
    x,y=np.asarray(x),np.asarray(y)
    if x.ndim==2 and y.ndim==2:return (x[:,:,None]*y[None,:,:]).sum(1)
    if x.ndim==2 and y.ndim==1:return (x*y[None]).sum(1)
    if x.ndim==1 and y.ndim==2:return (x[:,None]*y).sum(0)
    return (x*y).sum()
np.linalg.inv=inv;mt.inv=inv;np.dot=dot

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import run_e0 as base
import adp16.run_adp16 as run
OUT=run.OUT;FIG=OUT/"aggregate/figures"
def read(p):return json.loads(p.read_text(encoding="utf-8"))

def main():
    FIG.mkdir(parents=True,exist_ok=True);fm=read(OUT/"stage_a/disagreement/final_metrics.json");aff=read(OUT/"stage_a/affine_bridge/final_metrics.json");state=read(OUT/"stage_a/state_bridge/final_metrics.json");b1=read(OUT/"stage_b/B1_fresh_single_coord/final_metrics.json");b2=read(OUT/"stage_b/B2_fresh_dual_coord/final_metrics.json");b3=read(OUT/"stage_b/B3_fresh_shared_trunk/final_metrics.json");wrong=read(OUT/"stage_b/wrong_pair_control/final_metrics.json");verdict=read(OUT/"aggregate/final_verdict.json")
    # A: disagreement heatmap.
    mat=np.asarray(fm["mean_disagreement_matrix"]);fig,ax=plt.subplots(figsize=(9,4));im=ax.imshow(mat,aspect="auto",cmap="magma");ax.set_xticks(np.arange(7),[f"{x:.1f}" for x in fm["alpha_quantiles"]]);ax.set_yticks(np.arange(3),fm["region_order"]);ax.set_xlabel("alpha quantile value");ax.set_title("A. C2/C3 normalized functional disagreement");fig.colorbar(im,ax=ax,label="D_raw mean");fig.tight_layout();fig.savefig(FIG/"A_functional_disagreement_heatmap.png",dpi=190);plt.close(fig)
    # B: coordinate rescue domain R2.
    domains=["validation","cross_state","cross_realization","cross_base"];fig,ax=plt.subplots(figsize=(9,5));x=np.arange(4);ax.plot(x,[aff["r2"][d] for d in domains],lw=2,label="affine bridge");ax.plot(x,[state["r2"][d] for d in domains],lw=2,label="state-dependent bridge");ax.axhline(.995,color="#dc2626",ls="--",lw=1,label="PASS gate");ax.set_xticks(x,domains,rotation=15);ax.set_ylabel("bridge R2");ax.set_title("B. Coordinate rescue");ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(FIG/"B_coordinate_rescue.png",dpi=190);plt.close(fig)
    # C: fresh union B1 learning curve.
    t=b1["telemetry"];fig,ax=plt.subplots(figsize=(9,5));ep=[r["epoch"] for r in t]
    for key,label in [("C2_val_nrmse","C2 val"),("C3_val_nrmse","C3 val"),("union_val_nrmse","global val")]:ax.plot(ep,[r[key] for r in t],lw=2,label=label)
    ax.set_xlabel("epoch");ax.set_ylabel("NRMSE");ax.set_title("C. Fresh single-coordinate union learning curve");ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(FIG/"C_fresh_union_learning_curve.png",dpi=190);plt.close(fig)
    # D: model ladder, including ADP15 warm-start L2.
    warm=read(ROOT/"outputs/e0_adp15_capacity_representation/C2_to_C3/L2_parent_head/final_metrics.json");models=[warm,b1,b2,b3];labels=["ADP15 warm L2","B1 fresh single","B2 fresh dual","B3 fresh+trunk"]
    def cmp(q):return q["comparison"]
    glob=[max(v["delta"] for v in cmp(q)["domains"].values()) for q in models];child=[cmp(q)["child_absorb_nrmse"]-cmp(q)["child_keep_nrmse"] if "child_absorb_nrmse" in cmp(q) else cmp(q)["old_family"]["child"]["delta"] for q in models];parent=[cmp(q)["parent_old_delta_nrmse"] if "parent_old_delta_nrmse" in cmp(q) else cmp(q)["old_family"]["parent"]["delta"] for q in models]
    fig,ax=plt.subplots(figsize=(10,5));x=np.arange(4);ax.plot(x,glob,lw=2,label="max global delta");ax.plot(x,child,lw=2,label="child delta");ax.plot(x,parent,lw=2,label="parent delta");ax.axhline(.01,color="#dc2626",ls="--",lw=1,label="0.01 gate");ax.set_xticks(x,labels,rotation=15);ax.set_ylabel("delta NRMSE");ax.set_title("D. C2/C3 model ladder");ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(FIG/"D_model_ladder.png",dpi=190);plt.close(fig)
    summary={**verdict,"affine_r2":aff["r2"],"state_bridge_r2":state["r2"],"B1_max_bootstrap_upper95":max(x["upper95"] for x in b1["comparison"]["domains"].values()),"B2_max_bootstrap_upper95":max(x["upper95"] for x in b2["comparison"]["domains"].values()),"B3_max_bootstrap_upper95":max(x["upper95"] for x in b3["comparison"]["domains"].values()),"B3_max_other_drift":max(b3["other_candidate_drift"].values()),"wrong_max_bootstrap_upper95":max(x["upper95"] for x in wrong["comparison"]["domains"].values()),"figures":[p.name for p in sorted(FIG.glob("*.png"))]};(OUT/"aggregate/summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8");print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
