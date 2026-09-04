"""ADP8: distinguish invalid duplicate identity from merge fine-tuning drift."""
from __future__ import annotations
import argparse,json,os,sys,time
from pathlib import Path

_DLL_HANDLE=None
_DLL_DIR=Path(sys.prefix)/"Library"/"bin"
if os.name=="nt" and _DLL_DIR.is_dir():
    os.environ["PATH"]=str(_DLL_DIR)+os.pathsep+os.environ.get("PATH","")
    _DLL_HANDLE=os.add_dll_directory(str(_DLL_DIR))
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import run_e0 as base
import adp6.run_adp6 as adp6
import adp7.run_adp7 as adp7

CONFIG=ROOT/"configs"/"e0_adp8.json"
OUT=ROOT/"outputs"/"e0_adp8"
ADP7_A2=ROOT/"outputs"/"e0_adp7"/"a2_realization_holdout"/"final_metrics.json"
DEVICE=base.DEVICE

def cfg():return json.loads(CONFIG.read_text(encoding="utf-8"))
def cv(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,torch.Tensor):return x.detach().cpu().tolist()
    raise TypeError(type(x))
def dump(path,x):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=cv),encoding="utf-8")
def meta(c):
    x=base.environment_metadata();x.update({"stage":"adp8","world_seed":base.WORLD_SEED,"dataset_seed":base.DATASET_SEED,"optimization_seed":c["optimization_seed"],"config":c,"config_sha256":base.sha256_file(CONFIG),"script_sha256":base.sha256_file(Path(__file__)),"git_commit":base.git_commit(),"adp7_joint_baseline":str(ADP7_A2)});return x
def pass_row(row,c):return row["merged_test_nrmse"]<=row["separate_test_nrmse"]+c["nrmse_tolerance_over_separate"] and row["min_functional_r2"]>c["functional_r2_threshold"] and row["min_affine_realization_r2"]>c["affine_r2_threshold"] and row["order_accuracy"]>c["order_accuracy_threshold"]

def copied_single(source_bank,source,a0,b0,c):
    one,aa,bb=adp7.make_single(source_bank,source,a0,b0,"source_copy",{"optimization_seed":c["optimization_seed"],"mechanism_dim":64,"mechanism_heads":4})
    return one,aa,bb

def common(pair,spec,c):
    bank,a0,b0,assign=adp7.load_source();x,h,alpha,alpha_h=adp7.data();assignment=assign["val"];take=np.isin(assignment,np.asarray(pair));S=x["S"][take];response=x["response_train"][take];al=alpha[take];ass=assignment[take];hh={k:v[take] for k,v in h.items()};train_mask,test_mask=adp7.split_masks(spec,x,take,al);allp=adp7.all_candidate_prediction(bank,S,al,a0,b0);sep=np.empty_like(response)
    for j in pair:sep[ass==j]=allp[ass==j,:,j]
    sep_n=base.nrmse(response[test_mask],sep[test_mask]);source,cost=adp7.best_source(bank,a0,b0,S,response,al,ass,pair,train_mask)
    return bank,a0,b0,S,response,al,hh,train_mask,test_mask,sep_n,source,cost

def evaluate_model(operation,pair,spec,source,model,a,b,S,response,al,hh,test_mask,sep_n,extra,c):
    pred=adp7.merged_predict(model,a,b,np.repeat(S[:,None],al.shape[1],axis=1),al);func=adp7.functional_scores(model,a,b,al,hh,np.ones(len(S),bool),{"functional_probe_count":c["functional_probe_count"]});v=(float(a.detach().cpu()[0])*al+float(b.detach().cpu()[0]));aff=[]
    for g in np.unique(hh["family_gt"]):
        q=hh["family_gt"]==g;aff.append(adp6.linear_r2(v[q].reshape(-1),hh["train_amplitude_gt"][q].reshape(-1)))
    row={"operation":operation,"pair":adp7.pair_name(pair),"split":spec["id"],"survivor":f"C{source+1}","separate_test_nrmse":sep_n,"merged_test_nrmse":base.nrmse(response[test_mask],pred[test_mask]),"functional_r2_by_gt":func,"min_functional_r2":min(func.values()),"coordinate_a":float(a.detach().cpu()[0]),"coordinate_b":float(b.detach().cpu()[0]),"min_affine_realization_r2":float(min(aff)),"order_accuracy":float(np.median([adp6.pair_order(al[u],v[u]) for u in range(len(S))])),**extra};row["delta_nrmse"]=row["merged_test_nrmse"]-sep_n;row["pass"]=pass_row(row,c);return row

def direct_rows(pair,spec,c):
    bank,a0,b0,S,response,al,hh,train_mask,test_mask,sep_n,best,cost=common(pair,spec,c);rows=[]
    for source in pair:
        model,a,b=copied_single(bank,source,a0,b0,c);rows.append(evaluate_model("direct_delete_no_training",pair,spec,source,model,a,b,S,response,al,hh,test_mask,sep_n,{"training_performed":False,"learner_visible_source_train_costs":cost,"is_best_train_source":source==best},c))
    return rows

def coordinate_only_row(pair,spec,c):
    bank,a0,b0,S,response,al,hh,train_mask,test_mask,sep_n,source,cost=common(pair,spec,c);model,a,b=copied_single(bank,source,a0,b0,c)
    for p in model.parameters():p.requires_grad_(False)
    idx=np.argwhere(train_mask);opt=torch.optim.Adam([a,b],lr=c["coordinate_learning_rate"]);rng=np.random.default_rng(c["optimization_seed"]+81001+sum(ord(q) for q in spec["id"])+source);losses=[];started=time.time()
    for _ in range(c["epochs"]):
        order=rng.permutation(len(idx));ep=[]
        for st in range(0,len(idx),c["response_batch_size"]):
            ij=idx[order[st:st+c["response_batch_size"]]];fi=ij[:,0];ri=ij[:,1];St=torch.from_numpy(S[fi]).to(DEVICE);at=torch.from_numpy(al[fi,ri]).to(DEVICE);dt=torch.from_numpy(response[fi,ri]).to(DEVICE);v=(a[0]*at+b[0])[:,None];pred=model.raw_effects(St,v)[:,0];loss=(pred-dt).square().mean();opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_([a,b],c["gradient_clip"]);opt.step();ep.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(ep)))
    return evaluate_model("freeze_mechanism_fit_shared_ab",pair,spec,source,model,a,b,S,response,al,hh,test_mask,sep_n,{"training_performed":True,"mechanism_frozen":True,"learner_visible_source_train_costs":cost,"train_loss_by_epoch":losses,"wall_seconds":time.time()-started},c)

def joint_rows(c):
    source=json.loads(ADP7_A2.read_text(encoding="utf-8"));out=[]
    for old in source["rows"]:
        if old["initialization"]!="source_copy":continue
        row={"operation":"joint_mechanism_and_coordinate_finetune","pair":old["pair"],"split":old["split"],"survivor":old["source_candidate"],"separate_test_nrmse":old["separate_test_nrmse"],"merged_test_nrmse":old["merged_test_nrmse"],"delta_nrmse":old["delta_nrmse"],"min_functional_r2":old["min_functional_r2"],"coordinate_a":old["coordinate_a"],"coordinate_b":old["coordinate_b"],"min_affine_realization_r2":old["min_affine_realization_r2"],"order_accuracy":old["order_accuracy"],"training_performed":True,"mechanism_frozen":False,"reused_from":str(ADP7_A2)};row["pass"]=pass_row(row,c);out.append(row)
    return out

def run(force=False):
    c=cfg();final=OUT/"final_metrics.json"
    if final.exists() and not force:return json.loads(final.read_text(encoding="utf-8"))
    specs=adp7.SPECS["adp7_a2"];direct=[];coord=[];started=time.time();torch.cuda.reset_peak_memory_stats() if DEVICE.type=="cuda" else None
    for pair0 in c["correct_pairs"]:
        pair=tuple(pair0)
        for spec in specs:
            dr=direct_rows(pair,spec,c);direct.extend(dr);cr=coordinate_only_row(pair,spec,c);coord.append(cr);print(f"[ADP8] {adp7.pair_name(pair)} {spec['id']}: direct={[round(x['delta_nrmse'],4) for x in dr]} coord={cr['delta_nrmse']:.4f}",flush=True)
    joint=joint_rows(c);direct_best=[]
    for pair0 in c["correct_pairs"]:
        pn=adp7.pair_name(tuple(pair0))
        for spec in specs:
            options=[x for x in direct if x["pair"]==pn and x["split"]==spec["id"]];direct_best.append(min(options,key=lambda x:x["merged_test_nrmse"]))
    direct_all_best=all(x["pass"] for x in direct_best);direct_both_survivors=all(x["pass"] for x in direct);coord_all=all(x["pass"] for x in coord);joint_all=all(x["pass"] for x in joint);identity_supported=direct_all_best or coord_all;catastrophic_drift=identity_supported and not joint_all
    verdict="catastrophic_drift_supported" if catastrophic_drift else ("duplicate_identity_not_supported" if not identity_supported else "joint_finetuning_also_stable")
    result={**meta(c),"direct_delete_rows":direct,"direct_delete_best_survivor_rows":direct_best,"coordinate_only_rows":coord,"joint_finetune_rows":joint,"checks":{"direct_delete_best_survivor_all_splits":direct_all_best,"direct_delete_both_survivors_all_splits":direct_both_survivors,"coordinate_only_all_splits":coord_all,"joint_finetune_all_splits":joint_all,"duplicate_identity_supported":identity_supported,"catastrophic_drift_supported":catastrophic_drift},"verdict":verdict,"pass":catastrophic_drift,"wall_seconds":time.time()-started,"cuda_peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20 if DEVICE.type=="cuda" else 0.0};dump(final,result);dump(OUT/"config.json",c);return result

def main():
    p=argparse.ArgumentParser();p.add_argument("--seed",type=int,default=0);p.add_argument("--force",action="store_true");x=p.parse_args()
    if x.seed!=0:raise ValueError("development seed fixed to 0")
    r=run(x.force);print(json.dumps({"stage":r["stage"],"verdict":r["verdict"],"checks":r["checks"]},indent=2),flush=True)
if __name__=="__main__":main()
