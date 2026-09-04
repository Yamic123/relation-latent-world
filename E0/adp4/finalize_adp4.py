"""Create ADP4-A1 compact tables and figures from completed artifacts."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/"outputs"/"e0_adp4"/"a1_group_matching"

def main():
    rows=[json.loads(x) for x in (OUT/"round_metrics.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    with (OUT/"round_metrics.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f); w.writerow(["round","train_mse","val_nrmse","unit_accuracy","base_change","base_agreement","mean_sign_split","r2_1","r2_2","r2_3","u1","u2","u3","u4","u5"])
        for x in rows: w.writerow([x["round"],x["train_unit_mse"],x["val_unit_nrmse"],x["unit_matching_accuracy"],x["base_assignment_change_rate"],x["base_exact_agreement"],x["mean_sign_split"],*x["matched_functional_r2"],*x["candidate_usage"]])
    r=np.asarray([x["round"] for x in rows]); fr=np.asarray([x["matched_functional_r2"] for x in rows])
    fig,ax=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    ax[0,0].plot(r,[x["val_unit_nrmse"] for x in rows]);ax[0,0].set_title("Validation local-unit NRMSE")
    ax[0,1].plot(r,[x["unit_matching_accuracy"] for x in rows]);ax[0,1].axhline(.8,ls="--",c="k");ax[0,1].set_title("Unit Hungarian accuracy")
    for j in range(3):ax[1,0].plot(r,fr[:,j],label=f"match {j+1}")
    ax[1,0].axhline(.7,ls="--",c="k");ax[1,0].set_title("Matched functional R2");ax[1,0].legend()
    ax[1,1].plot(r,[x["mean_sign_split"] for x in rows],label="sign split");ax[1,1].plot(r,[x["base_assignment_change_rate"] for x in rows],label="base change");ax[1,1].axhline(.25,ls="--",c="k");ax[1,1].set_title("Stability vs realization split");ax[1,1].legend()
    for a in ax.flat:a.set_xlabel("round");a.grid(alpha=.25)
    fig.savefig(OUT/"a1_dynamics.png",dpi=160);plt.close(fig)
    usage=np.asarray([x["candidate_usage"] for x in rows]);fig,ax=plt.subplots(figsize=(9,4.5),constrained_layout=True)
    for j in range(5):ax.plot(r,usage[:,j],label=f"C{j+1}")
    ax.set(xlabel="round",ylabel="usage",title="ADP4-A1 candidate usage");ax.grid(alpha=.25);ax.legend(ncol=5);fig.savefig(OUT/"candidate_usage.png",dpi=160);plt.close(fig)

if __name__=="__main__":main()
