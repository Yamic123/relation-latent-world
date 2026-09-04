"""Package only already-saved ADP14 diagnostics; never performs training."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import run_e0 as base
import adp6.run_adp6 as adp6
import adp7.run_adp7 as adp7
import adp9.run_adp9 as adp9
import adp14.run_adp14 as adp14

DEVICE=base.DEVICE;SRC=adp14.OUT/"seed_001";OUT=adp14.OUT/"diagnostic_exports"

def read(p):return json.loads(p.read_text(encoding="utf-8"))

def direct(bank,a,b,parent,S,alpha):
    F,R=alpha.shape;allp=adp7.all_candidate_prediction(bank,S,alpha,a,b)
    return allp[:,:,parent]

def metadata(x,alpha,idx):
    R=alpha.shape[1];dt=np.dtype([("family_id","<i8"),("local_family_id","<i8"),("base_id","<i8"),("state","<f4",(3,2)),("alpha","<f4",(R,)),("action_vector","<f4",(R,2)),("action_magnitude","<f4",(R,)),("sign","<i1",(R,)),("realization_coordinate","<f4",(R,))]);z=np.empty(len(idx),dtype=dt)
    z["family_id"]=idx;z["local_family_id"]=x["local_family_id"][idx];z["base_id"]=x["base_id"][idx];z["state"]=x["S"][idx];z["alpha"]=alpha[idx];z["action_vector"]=x["action_train"][idx];z["action_magnitude"]=np.sqrt((x["action_train"][idx]**2).sum(-1));z["sign"]=np.sign(alpha[idx]).astype(np.int8);z["realization_coordinate"]=alpha[idx];return z

def load_trial(path,c):
    cp=torch.load(path/"model.pt",map_location=DEVICE,weights_only=True);bank=base.SetMechanismBank(5,c["mechanism_dim"],c["mechanism_heads"]).to(DEVICE);bank.load_state_dict(cp["bank"]);a=cp["coordinate_a"].to(DEVICE);b=cp["coordinate_b"].to(DEVICE);return bank,a,b

def package(child,parent,rank,variant="A1"):
    c,src,pre,pa,pb,saved,alive,data,audit=adp14.load_source(1);base_dir=SRC/"cycle_00"/("A1_constrained" if variant=="A1" else "A2_no_anchor")/f"proposal_{rank:02d}_C{child+1}_to_C{parent+1}"/"trial";tele=read(base_dir/"learning_curve.json");result=read(base_dir/"final_metrics.json");post,qa,qb=load_trial(base_dir,c)
    z=np.load(SRC/"pair_scan/assignments_costs.npz");ta=z["train"];va=z["val"];tc=z["train_cost"];cf=np.flatnonzero(ta==child);pf=np.flatnonzero(ta==parent);cv=np.flatnonzero(va==child);pv=np.flatnonzero(va==parent)
    epoch=np.asarray([0]+[r["epoch"] for r in tele],np.int64);n=len(epoch)
    def series(key,preval=np.nan):return np.asarray([preval]+[r[key] for r in tele],np.float64)
    pre_child=base.nrmse(data["tr"]["response_train"][cf],direct(pre,pa,pb,parent,data["tr"]["S"][cf],data["atr"][cf]));pre_parent=base.nrmse(data["tr"]["response_train"][pf],direct(pre,pa,pb,parent,data["tr"]["S"][pf],data["atr"][pf]))
    pre_cv=base.nrmse(data["va"]["response_heldout"][cv],direct(pre,pa,pb,parent,data["va"]["S"][cv],data["avh"][cv]));pre_pv=base.nrmse(data["va"]["response_heldout"][pv],direct(pre,pa,pb,parent,data["va"]["S"][pv],data["avh"][pv]));pre_global=base.nrmse(data["va"]["response_train"],adp14.hard_predictions(pre,pa,pb,va,data["va"],data["av"]))
    child_val=np.full(n,np.nan);parent_val=np.full(n,np.nan);child_val[0]=pre_cv;child_val[-1]=result["comparison"]["child_absorb_nrmse"];parent_val[0]=pre_pv;parent_val[-1]=pre_pv+result["comparison"]["parent_old_delta_nrmse"]
    avec=np.full(n,np.nan);bvec=np.full(n,np.nan);avec[0]=float(pa[parent]);avec[-1]=float(qa[parent]);bvec[0]=float(pb[parent]);bvec[-1]=float(qb[parent])
    _,postcost=adp6.family_e_step_structured(post,qa,qb,data["tr"]["S"],data["tr"]["response_train"],data["atr"])
    adapter_pre=pre.adapters[parent].detach().cpu().numpy();adapter_post=post.adapters[parent].detach().cpu().numpy()
    out={"epoch":epoch,"child_train_nrmse":series("child_nrmse",pre_child),"parent_train_nrmse":series("parent_old_nrmse",pre_parent),"child_val_nrmse":child_val,"parent_val_nrmse":parent_val,"global_val_nrmse":series("global_validation_nrmse",pre_global),"loss_child":series("child_loss"),"loss_parent":series("parent_replay_loss"),"loss_anchor":series("anchor_mse"),"loss_total":series("total_loss"),"a":avec,"b":bvec,
         "parent_adapter_pre":adapter_pre,"parent_adapter_post":adapter_post,"parent_a_pre":np.asarray(float(pa[parent])),"parent_a_post":np.asarray(float(qa[parent])),"parent_b_pre":np.asarray(float(pb[parent])),"parent_b_post":np.asarray(float(qb[parent])),"adapter_update_norm":np.asarray(float(np.sqrt(((adapter_post-adapter_pre)**2).sum()))),"gradient_norm":np.full(n,np.nan),
         "cost_child_pre":tc[cf,parent],"cost_child_e5":np.asarray([],np.float32),"cost_child_e10":np.asarray([],np.float32),"cost_child_e20":postcost[cf,parent],"cost_parent_pre":tc[pf,parent],"cost_parent_e5":np.asarray([],np.float32),"cost_parent_e10":np.asarray([],np.float32),"cost_parent_e20":postcost[pf,parent],
         "family_metadata_child":metadata(data["tr"],data["atr"],cf),"family_metadata_parent":metadata(data["tr"],data["atr"],pf),"child_family_indices":cf,"parent_family_indices":pf,"source_checkpoint_sha256":np.asarray(audit["source_checkpoint_sha256"]),"variant":np.asarray(variant),"child_candidate":np.asarray(child),"parent_candidate":np.asarray(parent),
         "availability_epoch_log":np.asarray("saved epochs only: 0,4,8,12,16,20; epoch 0 losses unavailable"),"availability_val_ab":np.asarray("validation subgroup NRMSE and a,b saved/recoverable only at pre and epoch20; intermediate entries are NaN"),"availability_cost":np.asarray("cost available only at pre and epoch20; e5/e10 arrays are empty"),"availability_gradient":np.asarray("gradient norms were not recorded; array is NaN")}
    return out

def main():
    OUT.mkdir(parents=True,exist_ok=True);c54=package(4,3,1,"A1");a2=package(4,3,1,"A2")
    for k,v in a2.items():
        if k not in ("family_metadata_child","family_metadata_parent","child_family_indices","parent_family_indices","source_checkpoint_sha256","child_candidate","parent_candidate"):c54["a2_"+k]=v
    c23=package(1,2,3,"A1");p54=OUT/"adp14_C5_to_C4_diagnostics.npz";p23=OUT/"adp14_C2_to_C3_diagnostics.npz";np.savez_compressed(p54,**c54);np.savez_compressed(p23,**c23)
    manifest={"training_performed":False,"uses_existing_saved_outputs_only":True,"contains_gt_metadata":False,"saved_epoch_grid":[0,4,8,12,16,20],"limitations":["No epoch5/epoch10 checkpoint or costs were saved; corresponding arrays are empty.","Intermediate validation subgroup NRMSE, a, b, and gradient norms were not saved; corresponding entries are NaN.","global_val_nrmse is the originally logged observed-realization validation NRMSE."],"files":{p54.name:{"sha256":adp14.sha256(p54),"includes_A1":True,"includes_A2_with_prefix":True},p23.name:{"sha256":adp14.sha256(p23),"includes_A1":True}}};(OUT/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8");print(json.dumps(manifest,indent=2))
if __name__=="__main__":main()
