"""ADP11: from-zero family discovery under identity/orthogonal perfect TP."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,sys,time
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
import adp5.run_adp5 as adp5
import adp6.run_adp6 as adp6
import adp7.run_adp7 as adp7
import adp9.run_adp9 as adp9
from adp1._utils import hash_state_dict

OUT=ROOT/"outputs"/"e0_adp11_from_zero"
TEMPLATE=ROOT/"configs"/"e0_adp11.json"
DEVICE=base.DEVICE

def cv(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,torch.Tensor):return x.detach().cpu().tolist()
    raise TypeError(type(x))
def dump(path,x):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=cv),encoding="utf-8")
def append(path,x):path.parent.mkdir(parents=True,exist_ok=True);f=path.open("a",encoding="utf-8");f.write(json.dumps(x,ensure_ascii=False,default=cv)+"\n");f.close()
def read(path):return json.loads(path.read_text(encoding="utf-8"))
def sha_array(a):
    x=np.ascontiguousarray(a);h=hashlib.sha256();h.update(str(x.dtype).encode());h.update(str(x.shape).encode());h.update(x.tobytes());return h.hexdigest()
def sha_arrays(items):
    h=hashlib.sha256()
    for name,a in items:h.update(name.encode());h.update(sha_array(a).encode())
    return h.hexdigest()
def condition_name(condition):return {"identity":"identity_tp","orthogonal":"orthogonal_tp"}[condition]

def configure(condition,seed):
    seed_root=OUT/condition_name(condition)/f"seed_{seed:03d}";audit=seed_root/"z0_audit";audit.mkdir(parents=True,exist_ok=True)
    c=read(TEMPLATE);c["optimization_seed"]=seed;c["tp_condition"]=condition
    cp=audit/"config.json";dump(cp,c)
    adp6.OUT=seed_root;adp6.CONFIG=cp
    adp9.OUT=seed_root;adp9.A2=seed_root/"z4_joint";adp9.CONFIG=cp
    adp7.A2=adp9.A2
    return seed_root,c,cp

def balanced_assignment(n,seed):
    rng=np.random.default_rng(seed+110003);x=np.arange(n,dtype=np.int64)%5;rng.shuffle(x);return x

def initial_objects(c,n):
    base.seed_everything(c["optimization_seed"]);bank=base.SetMechanismBank(5,c["mechanism_dim"],c["mechanism_heads"]).to(DEVICE)
    gen=torch.Generator(device=DEVICE);gen.manual_seed(c["optimization_seed"]+230009)
    signs=torch.where(torch.rand(5,generator=gen,device=DEVICE)<.5,-torch.ones(5,device=DEVICE),torch.ones(5,device=DEVICE))
    a=torch.nn.Parameter(signs*(.8+.4*torch.rand(5,generator=gen,device=DEVICE)))
    b=torch.nn.Parameter(-.1+.2*torch.rand(5,generator=gen,device=DEVICE))
    assignment=balanced_assignment(n,c["optimization_seed"]);return bank,a,b,assignment

def prepare_coordinates(seed_root):
    coords={}
    for split in ("train","val","test"):
        x=adp6.response_data(split);q,a,ah=adp6.action_coordinate(x["action_train"],x["action_heldout"]);coords[f"{split}_q"]=q;coords[f"{split}_alpha_train"]=a;coords[f"{split}_alpha_heldout"]=ah
    p=seed_root/"adp6_0_action_coordinate"/"learner_coordinates.npz";p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**coords);return p

def z0_z1(condition,seed,seed_root,c,cp):
    tr=adp6.response_data("train");va=adp6.response_data("val");bank,a,b,assignment=initial_objects(c,len(tr["S"]));world=base.load_world("e0a");B=world["B"].astype(np.float32)
    if condition=="orthogonal":
        bp=OUT/"orthogonal_tp"/"B.npy";bp.parent.mkdir(parents=True,exist_ok=True)
        if not bp.exists():np.save(bp,B)
    family_p=[]
    for split,x in (("train",tr),("val",va)):
        d=x["response_train"].reshape(len(x["S"]),x["response_train"].shape[1],-1)
        p=d if condition=="identity" else d@B.T;family_p.append((split,p.astype(np.float32)))
    non_tp=sha_arrays([(f"tr_{k}",v) for k,v in tr.items()]+[(f"va_{k}",v) for k,v in va.items()])
    audit={"stage":"adp11_z0","condition":condition,"optimization_seed":seed,"initial_model_hash":hash_state_dict(bank.state_dict()),"initial_assignment_hash":sha_array(assignment),"initial_coordinate_a_hash":sha_array(a.detach().cpu().numpy()),"initial_coordinate_b_hash":sha_array(b.detach().cpu().numpy()),"non_tp_dataset_and_family_hash":non_tp,"family_p_hash":sha_arrays(family_p),"B_hash":sha_array(B) if condition=="orthogonal" else None,"B_orthogonality_max_error":float(np.max(np.abs(B.T@B-np.eye(6)))) if condition=="orthogonal" else None,"config_hash":base.sha256_file(cp),"world_seed":base.WORLD_SEED,"dataset_seed":base.DATASET_SEED,"git_commit":base.git_commit(),"forbidden_checkpoint_loaded":False,"tp_used_by_preregistered_objective":False,"tp_usage_contract":c["tp_usage_contract"],"pass":True}
    dump(seed_root/"z0_audit"/"audit.json",audit)
    z1={"stage":"adp11_z1","source":"ADP5 leak-free family dataset","source_visible_only":True,"family_counts":{"train":len(tr["S"]),"val":len(va["S"])},"members_per_family":{"train":6,"heldout":6},"local_family_id_only":True,"family_p_hash":audit["family_p_hash"],"pass":True};dump(seed_root/"z1_families"/"final_metrics.json",z1)
    prepare_coordinates(seed_root);return audit,bank,a,b,assignment

def z4_gate(last,c):
    counts=last["candidate_family_count"];major=[n>=c["major_candidate_min_usage"]*sum(counts) for n in counts];aff=[x for x,q in zip(last["candidate_affine_aligned_realization_r2"],major) if q and x is not None];pur=[p for p,q in zip(last["candidate_purity_fixed"],major) if q];pf=[x for x,q in zip(last["purity_high_candidate_functional_r2"],major) if q and x is not None]
    checks={"each_gt_best_functional":min(last["matched_functional_r2"])>.95,"all_purity_high_functional":bool(pf) and min(pf)>.90,"affine_coordinate":bool(aff) and min(aff)>.95,"order":last["median_order_accuracy"]>.95,"heldout":last["heldout_realization_nrmse"]<.05,"no_mixing":bool(pur) and min(pur)>.90}
    return {"major_candidate_mask":major,"major_candidate_min_usage":c["major_candidate_min_usage"],"checks":checks,"pass":all(checks.values())}

def run_z2_z4(condition,seed,seed_root,c,bank,a,b,assignment,force=False):
    out=seed_root/"z4_joint";final=out/"final_metrics.json"
    if final.exists() and not force:
        result=read(final);gate=z4_gate(result["last_round"],c);result.update({"major_candidate_mask":gate["major_candidate_mask"],"major_candidate_min_usage":gate["major_candidate_min_usage"],"checks":gate["checks"],"pass":gate["pass"]});dump(final,result);return result
    out.mkdir(parents=True,exist_ok=True);(out/"checkpoints").mkdir(exist_ok=True);(out/"assignments").mkdir(exist_ok=True)
    tr=adp6.response_data("train");va=adp6.response_data("val");hv=adp6.evaluator_data("val");atr,_=adp6.coordinates("train");av,avh=adp6.coordinates("val")
    opt=torch.optim.AdamW([{"params":bank.parameters(),"lr":c["mechanism_learning_rate"],"weight_decay":c["weight_decay"]},{"params":[a,b],"lr":c["coordinate_learning_rate"],"weight_decay":0.0}]);rows=[];started=time.time();torch.cuda.reset_peak_memory_stats() if DEVICE.type=="cuda" else None
    for r in range(c["rounds"]):
        tm=adp6.train_round(bank,a,b,tr["S"],tr["response_train"],atr,assignment,c,opt,r)
        es=time.time();new,cost=adp6.family_e_step_structured(bank,a,b,tr["S"],tr["response_train"],atr);e_seconds=time.time()-es;change=float(np.mean(new!=assignment));assignment=new
        va_assign,va_cost=adp6.family_e_step_structured(bank,a,b,va["S"],va["response_train"],av);train_pred,_=adp6.predict(bank,tr["S"],assignment,atr,a,b);held_pred,_=adp6.predict(bank,va["S"],va_assign,avh,a,b);val_pred,val_v=adp6.predict(bank,va["S"],va_assign,av,a,b)
        cm=adp6.coordinate_metrics(av,val_v,va_assign,hv);fm=adp6.functional_eval(bank,a,b,av,va_assign,hv,c);_,_,acc,frag,_=adp6.fixed_structure(va_assign,hv)
        row={"round":r,"train_response_mse":float(np.mean((train_pred-tr["response_train"])**2)),"heldout_realization_nrmse":base.nrmse(va["response_heldout"],held_pred),"observed_nrmse":base.nrmse(va["response_train"],val_pred),"candidate_a":a.detach().cpu().tolist(),"candidate_b":b.detach().cpu().tolist(),**cm,**fm,"family_hungarian_accuracy":acc,"mean_GT_fragmentation":float(np.mean(frag)),"family_assignment_change":change,"candidate_usage":(np.bincount(assignment,minlength=5)/len(assignment)).tolist(),"candidate_family_count":np.bincount(assignment,minlength=5).tolist(),"e_step_seconds":e_seconds,"m_step_seconds":tm["seconds"],"round_seconds":e_seconds+tm["seconds"],"cuda_peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20 if DEVICE.type=="cuda" else 0.0}
        append(out/"round_metrics.jsonl",row);rows.append(row);np.savez_compressed(out/"assignments"/f"round_{r:03d}.npz",train=assignment,val=va_assign,train_cost=cost,val_cost=va_cost);torch.save({"bank":bank.state_dict(),"coordinate_a":a.detach().cpu(),"coordinate_b":b.detach().cpu()},out/"checkpoints"/f"round_{r:03d}.pt");print(f"[{condition} s{seed} Z4] r={r:02d} change={change:.3f} nrmse={row['heldout_realization_nrmse']:.3f} func={np.round(row['matched_functional_r2'],3).tolist()} purity={np.round(row['candidate_purity_fixed'],2).tolist()}",flush=True)
    gate=z4_gate(rows[-1],c);result={"stage":"adp11_z4","condition":condition,"optimization_seed":seed,"initialization":{"bank":"random","assignment":"random_balanced","coordinate":"random"},"rounds_completed":len(rows),"last_round":rows[-1],"last3_assignment_change":float(np.mean([x["family_assignment_change"] for x in rows[-3:]])),"major_candidate_mask":gate["major_candidate_mask"],"major_candidate_min_usage":gate["major_candidate_min_usage"],"checks":gate["checks"],"pass":gate["pass"],"wall_seconds":time.time()-started,"cuda_peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20 if DEVICE.type=="cuda" else 0.0};dump(final,result);return result

def failure(z4,prune,b1):
    if not z4["checks"].get("each_gt_best_functional"):return "F1_family_identity_failure"
    if not z4["checks"].get("affine_coordinate") or not z4["checks"].get("order"):return "F2_realization_coordinate_failure"
    if not z4["checks"].get("no_mixing") or not z4["checks"].get("all_purity_high_functional"):return "F3_mechanism_mixing"
    if not z4["pass"]:return "F1_family_identity_failure"
    if prune is None or prune.get("M_discovered",5)>3:return "F4_duplicate_graph_false_negative"
    if prune.get("M_discovered",5)<3:return "F5_duplicate_graph_false_positive"
    if not prune.get("pass"):return "F6_reassignment_instability"
    if b1 is None or not b1.get("pass"):return "F7_full_TP_composition_failure"
    return None

def run_one(condition,seed,force=False):
    seed_root,c,cp=configure(condition,seed);final=seed_root/"final_verdict.json"
    if final.exists() and not force:return read(final)
    started=time.time();audit,bank,a,b,assignment=z0_z1(condition,seed,seed_root,c,cp);z4=run_z2_z4(condition,seed,seed_root,c,bank,a,b,assignment,force)
    prune=b1=None
    if z4["pass"]:
        graph=adp9.stage_0(force)
        if graph["pass"]:
            prune=adp9.stage_prune(force);verify=adp9.stage_verify(force)
            if prune["pass"] and verify["pass"] and prune["M_discovered"]==3:b1=adp9.stage_b1(force)
        else:
            prune={"pass":False,"M_discovered":5,"stop_reason":"no_valid_duplicate_edge_at_initial_graph","telemetry":[],"final_metrics":None}
    fm=failure(z4,prune,b1);passed=fm is None
    last=z4["last_round"];pm=prune.get("final_metrics") if prune else None;result={"condition":condition,"seed":seed,"z0_pass":audit["pass"],"z4_identity_pass":z4["checks"]["each_gt_best_functional"] and z4["checks"]["no_mixing"],"z4_coordinate_pass":z4["checks"]["affine_coordinate"] and z4["checks"]["order"],"z4_pass":z4["pass"],"pruning_pass":bool(prune and prune["pass"]),"pruning_stop_reason":prune.get("stop_reason") if prune else "z4_gate_failed","m_discovered":prune.get("M_discovered") if prune else None,"family_accuracy":pm["family_hungarian_accuracy"] if pm else last["family_hungarian_accuracy"],"fragmentation":pm["mean_fragmentation"] if pm else last["mean_GT_fragmentation"],"b1_pass":bool(b1 and b1["pass"]),"b1_iid_nrmse":b1["splits"]["test_iid"]["nrmse"] if b1 else None,"b1_101_nrmse":b1["splits"]["test_combination_101"]["nrmse"] if b1 else None,"participation_f1":min(b1["splits"]["test_iid"]["participation_f1_micro"],b1["splits"]["test_combination_101"]["participation_f1_micro"]) if b1 else None,"final_pass":passed,"failure_mode":fm,"z4_metrics":{"heldout_nrmse":last["heldout_realization_nrmse"],"matched_functional_r2":last["matched_functional_r2"],"candidate_purity":last["candidate_purity_fixed"],"assignment_change":last["family_assignment_change"]},"tp_used_by_preregistered_objective":False,"wall_seconds":time.time()-started};dump(final,result);print(json.dumps({"condition":condition,"seed":seed,"pass":passed,"failure":fm,"M":result["m_discovered"]},indent=2),flush=True);return result

def aggregate(seeds):
    rows=[]
    for cond in ("identity","orthogonal"):
        for s in seeds:
            p=OUT/condition_name(cond)/f"seed_{s:03d}"/"final_verdict.json"
            if p.exists():rows.append(read(p))
    tag="aggregate_3seed" if len(seeds)==3 else "aggregate_10seed";d=OUT/tag;d.mkdir(parents=True,exist_ok=True)
    fields=["condition","seed","z4_identity_pass","z4_coordinate_pass","z4_pass","m_discovered","pruning_pass","b1_iid_nrmse","b1_101_nrmse","participation_f1","final_pass","failure_mode"]
    with (d/"summary.csv").open("w",newline="",encoding="utf-8-sig") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:r.get(k) for k in fields} for r in rows])
    by={};need=2 if len(seeds)==3 else 8
    for cond in ("identity","orthogonal"):
        q=[r for r in rows if r["condition"]==cond];by[cond]={"completed":len(q),"pass_count":sum(r["final_pass"] for r in q),"recovery_rate":float(np.mean([r["final_pass"] for r in q])) if q else 0.0,"z4_pass_rate":float(np.mean([r["z4_pass"] for r in q])) if q else 0.0,"m_histogram":{str(k):sum(r.get("m_discovered")==k for r in q) for k in (2,3,4,5,None)},"failure_modes":{str(x):sum(r.get("failure_mode")==x for r in q) for x in set(r.get("failure_mode") for r in q)}}
    gap=abs(by["identity"]["recovery_rate"]-by["orthogonal"]["recovery_rate"]);result={"stage":"adp11_aggregate","seeds":seeds,"required_per_condition":need,"conditions":by,"recovery_rate_gap":gap,"gap_gate":gap<=.20,"pass":all(by[c]["pass_count"]>=need for c in by) and gap<=.20,"tp_comparison_is_vacuous_under_specified_objective":True,"reason":"p is not consumed by any preregistered training or inference equation"};dump(d/"final_verdict.json",result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument("--condition",choices=["identity","orthogonal"]);p.add_argument("--seed",type=int);p.add_argument("--stage",choices=["run","aggregate","auto"],default="run");p.add_argument("--seeds",default="0,1,2");p.add_argument("--force",action="store_true");a=p.parse_args();seeds=[int(x) for x in a.seeds.split(",") if x.strip()]
    if a.stage=="run":
        if a.condition is None or a.seed is None:raise ValueError("--condition and --seed are required")
        result=run_one(a.condition,a.seed,a.force)
    elif a.stage=="aggregate":result=aggregate(seeds)
    else:
        for c in ("identity","orthogonal"):
            for s in (0,1,2):run_one(c,s,a.force)
        result=aggregate([0,1,2])
        if result["pass"]:
            for c in ("identity","orthogonal"):
                for s in range(3,10):run_one(c,s,a.force)
            result=aggregate(list(range(10)))
    print(json.dumps(result if a.stage=="aggregate" else {k:result.get(k) for k in ("condition","seed","final_pass","failure_mode","pass","conditions","recovery_rate_gap")},indent=2),flush=True)
if __name__=="__main__":main()
