"""Create the preregistered ADP14 Stage-A figures and compact tables."""
from __future__ import annotations
import csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np

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
OUT=ROOT/"outputs/e0_adp14_specialist_absorption"
S=OUT/"seed_001"
FIG=OUT/"aggregate/figures"

def read(p):return json.loads(p.read_text(encoding="utf-8"))

def attempts(variant="A1"):
    x=read(S/f"{variant}_final.json")
    return x["cycles"][0]["attempts"]

def main():
    FIG.mkdir(parents=True,exist_ok=True)
    scan=read(S/"pair_scan/pair_scan.json");a1=read(S/"A1_final.json");a2=read(S/"A2_final.json")
    a3=read(S/"A3_final.json");a4=read(S/"A4_final.json")
    props=scan["passing_proposals"]

    # Figure A: learner-visible directional replacement graph.
    fig,ax=plt.subplots(figsize=(8,7));ang=np.linspace(0,2*np.pi,5,endpoint=False);xy=np.c_[np.cos(ang),np.sin(ang)]
    for j,(x,y) in enumerate(xy):ax.text(x,y,f"[ C{j+1} ]",ha="center",va="center",fontsize=12,zorder=4,color="#1d4ed8")
    for p in props:
        i,j=p["child"],p["parent"];start=xy[i]*.83;end=xy[j]*.83
        vec=end-start;perp=np.array([-vec[1],vec[0]])/max(np.sqrt((vec*vec).sum()),1e-12);start=start+.045*perp;end=end+.045*perp
        # A line plus a directional endpoint marker avoids Matplotlib's native
        # Bezier polynomial path, which is broken by the local Windows BLAS DLL.
        ax.plot([start[0],end[0]],[start[1],end[1]],lw=1+3*p["P_val"],color="#dc2626",alpha=.72)
        ax.text(end[0],end[1],">",ha="center",va="center",fontsize=13,color="#dc2626",zorder=5)
        mid=(start+end)/2;ax.text(mid[0],mid[1],f"{p['P_val']:.3f}",fontsize=9,color="#7f1d1d",bbox=dict(fc="white",ec="none",alpha=.75))
    ax.set_title("A. Directional replacement graph (edge label = P_val)");ax.set_aspect("equal");ax.axis("off");fig.savefig(FIG/"A_directional_replacement_graph.png",dpi=190);plt.close(fig)

    # Figure B: all three A1 learning curves.
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    fields=[("child_nrmse","child NRMSE"),("parent_old_nrmse","parent-old NRMSE"),("anchor_mse","anchor MSE"),("global_validation_nrmse","global validation NRMSE")]
    for att in attempts("A1"):
        pair=att["proposal"]["pair"];p=S/"cycle_00/A1_constrained"/f"proposal_{att['rank']:02d}_{pair.replace('->','_to_')}"/"trial/learning_curve.json";rows=read(p);ep=[r["epoch"] for r in rows]
        for ax,(f,label) in zip(axes.flat,fields):ax.plot(ep,[r[f] for r in rows],label=pair);ax.set_ylabel(label);ax.grid(alpha=.2)
    for ax in axes[-1]:ax.set_xlabel("epoch")
    axes[0,0].legend();fig.suptitle("B. Constrained absorption learning curves (A1)");fig.tight_layout();fig.savefig(FIG/"B_absorption_learning_curves.png",dpi=190);plt.close(fig)

    # Figure C: assignment coverage before/after the closest A1 attempt (C5->C4).
    before=np.load(S/"pair_scan/assignments_costs.npz")["val"]
    after=np.load(S/"cycle_00/A1_constrained/proposal_01_C5_to_C4/trial/assignments.npz")["val"]
    order=np.lexsort((np.arange(len(before)),before));fig,axes=plt.subplots(2,1,figsize=(13,3.8),sharex=True)
    for ax,v,title in zip(axes,[before[order],after[order]],["before: M=5","trial after C5->C4: M=4"]):
        ax.imshow(v[None,:],aspect="auto",cmap="tab10",vmin=0,vmax=9);ax.set_yticks([]);ax.set_ylabel(title,rotation=0,labelpad=75,va="center")
    axes[-1].set_xlabel("validation families, sorted by pre-absorption candidate");fig.suptitle("C. Before vs after context coverage");fig.tight_layout();fig.savefig(FIG/"C_before_after_context_coverage.png",dpi=190);plt.close(fig)

    # Figure D: frozen vs adapted global domain deltas.
    domains=["iid","state","realization","base"];fig,axes=plt.subplots(1,3,figsize=(14,4),sharey=True)
    for ax,att in zip(axes,attempts("A1")):
        frozen=att["frozen_delete"]["comparison"]["domains"];adapt=att["trial"]["comparison"]["domains"];x=np.arange(4)
        ax.plot(x,[frozen[d]["delta"] for d in domains],label="frozen",color="#64748b",lw=2)
        ax.plot(x,[adapt[d]["delta"] for d in domains],label="adapted",color="#2563eb",lw=2)
        ax.axhline(.01,color="#dc2626",ls="--",lw=1,label="0.01 gate");ax.set_xticks(x,domains,rotation=25);ax.set_title(att["proposal"]["pair"]);ax.grid(axis="y",alpha=.2)
    axes[0].set_ylabel("Delta NRMSE (drop/absorb - keep)");axes[0].legend(fontsize=8);fig.suptitle("D. Frozen deletion vs constrained adaptation");fig.tight_layout();fig.savefig(FIG/"D_frozen_vs_adapted_drop.png",dpi=190);plt.close(fig)

    rows=[]
    for att in attempts("A1"):
        cmp=att["trial"]["comparison"];rows.append({"pair":att["proposal"]["pair"],"P_train":att["proposal"]["P_train"],"P_val":att["proposal"]["P_val"],"median_gap_val":att["proposal"]["val_gap"]["median"],"max_bootstrap_upper95":max(v["upper95"] for v in cmp["domains"].values()),"max_subgroup_delta":cmp["max_subgroup_delta_nrmse"],"parent_old_delta":cmp["parent_old_delta_nrmse"],"child_keep_nrmse":cmp["child_keep_nrmse"],"child_absorb_nrmse":cmp["child_absorb_nrmse"],"commit":att["commit"]})
    agg=OUT/"aggregate";agg.mkdir(exist_ok=True)
    with (agg/"A1_pair_results.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    total=sum(att["trial"]["wall_seconds"] for att in attempts("A1")+attempts("A2"))+a3["trial"]["wall_seconds"]
    summary={"stage_a_complete":True,"A1_commits":a1["commit_count"],"A2_commits":a2["commit_count"],"A3_wrong_parent_pass":a3["pass"],"A4_frozen_delete_pass":a4["pass"],"stop_A":a1["commit_count"]==0,"stage_b_executed":False,"measured_absorption_trial_seconds":total,"A1_pairs":rows,"figures":[p.name for p in sorted(FIG.glob("*.png"))]}
    (agg/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
