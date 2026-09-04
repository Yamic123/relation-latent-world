"""Read-only latent-coordinate diagnostic for the final ADP5-A1 checkpoint."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import run_e0 as base
from adp5.run_adp5 import OUT,cfg,dump,family_e_step,hidden,visible

def corr(x,y):
    x=x-x.mean();y=y-y.mean();return float((x*y).sum()/max(float(np.sqrt((x*x).sum()*(y*y).sum())),1e-12))

def main():
    c=cfg();x=visible("val");h=hidden("val");bank=base.SetMechanismBank(5,c["mechanism_dim"],c["mechanism_heads"]).to(base.DEVICE)
    bank.load_state_dict(torch.load(OUT/"a1_family_discovery"/"checkpoints"/"round_039_bank.pt",map_location=base.DEVICE,weights_only=True))
    e=family_e_step(bank,x["S"],x["response_train"],c,c["optimization_seed"]+70000001+39*1000003);F=len(x["S"]);latent=e["point_v"].reshape(F,6);truth=h["train_amplitude_gt"]
    rows=[]
    for j in range(5):
        take=e["assignment"]==j
        if not take.any():continue
        labels=np.bincount(h["family_gt"][take],minlength=3);per_family=[corr(truth[i],latent[i]) for i in np.flatnonzero(take)]
        rows.append({"candidate":j,"family_count":int(take.sum()),"GT_counts":labels.tolist(),"mean_inferred_latent_by_true_amplitude":latent[take].mean(axis=0).tolist(),
                     "std_inferred_latent_by_true_amplitude":latent[take].std(axis=0).tolist(),"pooled_true_vs_inferred_correlation":corr(truth[take].reshape(-1),latent[take].reshape(-1)),
                     "mean_within_family_correlation":float(np.mean(per_family)),"mean_abs_within_family_correlation":float(np.mean(np.abs(per_family))),"negative_orientation_fraction":float(np.mean(np.asarray(per_family)<0))})
    result={"read_only_no_training":True,"true_amplitudes":truth[0].tolist(),"candidates":rows};dump(OUT/"a1_family_discovery"/"final_latent_diagnostic.json",result)
    print(result)
if __name__=="__main__":main()
