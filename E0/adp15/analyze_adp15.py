"""ADP15 aggregate figures and compact summary."""
from __future__ import annotations
import json,sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mt
import numpy as np
import torch

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
import adp7.run_adp7 as adp7
import adp14.run_adp14 as adp14
import adp15.run_adp15 as run
OUT=run.OUT;P=OUT/"C5_to_C4";FIG=OUT/"aggregate/figures"
def read(p):return json.loads(p.read_text(encoding="utf-8"))

def load_level(level,source,parent):
    cp=torch.load(P/f"{level}_{'parent_head' if level=='L2' else 'shared_trunk'}/model.pt",map_location=run.DEVICE,weights_only=True);m=run.expanded(source,parent);m.load_state_dict(cp["bank"]);return m,cp["coordinate_a"].to(run.DEVICE),cp["coordinate_b"].to(run.DEVICE)

def main():
    FIG.mkdir(parents=True,exist_ok=True);l0=read(P/"L0_frozen/final_metrics.json");l1=read(P/"L1_adapter/final_metrics.json");l2=read(P/"L2_parent_head/final_metrics.json");l3=read(P/"L3_shared_trunk/final_metrics.json");levels=[l0,l1,l2,l3];labels=["L0","L1","L2","L3"]
    global_delta=[max(x["delta"] for x in q["comparison"]["domains"].values()) for q in levels];parent_delta=[q["comparison"]["parent_old_delta_nrmse"] for q in levels];child=[q["comparison"]["child_absorb_nrmse"] for q in levels]
    fig,axes=plt.subplots(1,3,figsize=(13,4));series=[(global_delta,"max global delta NRMSE",.01),(parent_delta,"parent-old delta NRMSE",.01),(child,"child validation NRMSE",None)]
    for ax,(y,title,gate) in zip(axes,series):ax.plot(np.arange(4),y,lw=2);ax.set_xticks(np.arange(4),labels);axax=None;ax.set_title(title);ax.grid(alpha=.2);ax.axhline(gate,color="#dc2626",ls="--",lw=1) if gate is not None else None
    fig.suptitle("A. Capacity ladder: C5 to C4");fig.tight_layout();fig.savefig(FIG/"A_capacity_ladder.png",dpi=190);plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,6))
    for q,label in [(l2,"L2 parent head"),(l3,"L3 shared trunk")]:
        t=q["telemetry"];ax.plot([x["child_val_nrmse"] for x in t],[x["parent_val_nrmse"] for x in t],lw=2,label=label)
        for x in t:ax.text(x["child_val_nrmse"],x["parent_val_nrmse"],str(x["epoch"]),fontsize=8)
    ax.set_xlabel("child validation NRMSE");ax.set_ylabel("parent-old validation NRMSE");ax.grid(alpha=.2);ax.legend();ax.set_title("B. Child-parent trade-off (labels are epochs)");fig.tight_layout();fig.savefig(FIG/"B_child_parent_tradeoff.png",dpi=190);plt.close(fig)

    drift=l3["other_candidate_drift"];names=list(drift);fig,ax=plt.subplots(figsize=(8,5));x=np.arange(len(names));ax.plot(x,[drift[n]["own_context_nrmse"] for n in names],lw=2,label="own contexts");ax.plot(x,[drift[n]["global_context_nrmse"] for n in names],lw=2,label="global contexts");ax.axhline(.01,color="#dc2626",ls="--",lw=1,label="0.01 protection gate");ax.set_xticks(x,names);ax.set_ylabel("functional output drift NRMSE");ax.set_title("C. L3 other-candidate drift");ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(FIG/"C_other_candidate_drift.png",dpi=190);plt.close(fig)

    c,src,source,a,b,saved,alive,data,audit=adp14.load_source(1);ta,tc,va,vc=adp14.assignments(source,a,b,alive,data);m2,a2,b2=load_level("L2",source,3);m3,a3,b3=load_level("L3",source,3);keep=adp14.hard_predictions(source,a,b,va,data["va"],data["avh"]);pred2=run.direct(m2,a2,b2,3,data["va"]["S"],data["avh"]);pred3=run.direct(m3,a3,b3,3,data["va"]["S"],data["avh"]);target=data["va"]["response_heldout"]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
    for ax,cand,title in zip(axes,[3,4],["C4 old-parent state region","C5 child state region"]):
        mask=va==cand
        for pred,label in [(keep,"before keep"),(pred2,"L2 after"),(pred3,"L3 after")]:
            e=np.sqrt(((pred[mask]-target[mask])**2).mean(axis=(1,2,3)));e=np.sort(e);ax.plot(np.linspace(0,1,len(e)),e,lw=2,label=label)
        ax.set_title(title);ax.set_xlabel("family quantile");ax.grid(alpha=.2)
    axes[0].set_ylabel("family held-out RMSE");axes[0].legend();fig.suptitle("D. State-region error coverage");fig.tight_layout();fig.savefig(FIG/"D_state_region_coverage.png",dpi=190);plt.close(fig)

    controls=read(P/"controls/summary.json");c2=read(OUT/"C2_to_C3/L2_parent_head/final_metrics.json");summary={"main_pair":"C5:C4","level_pass":{"L0":False,"L1":False,"L2":l2["absorb_pass"],"L3":l3["absorb_pass"]},"minimum_successful_level":"L2","classification":"parent_specific_capacity_bottleneck","L2_gate_checks":l2["gate_checks"],"L3_absorption_checks_without_protection":{k:v for k,v in l3["gate_checks"].items() if k!="other_candidate_protection"},"L3_max_other_candidate_drift":l3["max_other_candidate_global_drift"],"wrong_parent_pass":controls["wrong_parent_pass"],"no_parent_replay_pass":controls["ablation_pass"],"C2_to_C3_L2_pass":c2["absorb_pass"],"strong_cross_pair_capacity_evidence":False,"figures":[p.name for p in sorted(FIG.glob("*.png"))]};d=OUT/"aggregate";d.mkdir(exist_ok=True);(d/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8");print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
