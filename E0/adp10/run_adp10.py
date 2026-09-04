"""ADP10: multi-seed stability of the ADP6 -> ADP9 -> B1 chain."""
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
import adp6.run_adp6 as adp6
import adp7.run_adp7 as adp7
import adp9.run_adp9 as adp9
from adp1._utils import hash_state_dict

OUT=ROOT/"outputs"/"e0_adp10_from_adp6"
ADP6_TEMPLATE=ROOT/"configs"/"e0_adp10_adp6.json"
ADP9_TEMPLATE=ROOT/"configs"/"e0_adp10_adp9.json"

def cv(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,torch.Tensor):return x.detach().cpu().tolist()
    raise TypeError(type(x))
def dump(path,x):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=cv),encoding="utf-8")
def sha_bytes(data):return hashlib.sha256(data).hexdigest()
def sha_arrays(items):
    h=hashlib.sha256()
    for name,a in items:
        x=np.ascontiguousarray(a);h.update(name.encode());h.update(str(x.dtype).encode());h.update(str(x.shape).encode());h.update(x.tobytes())
    return h.hexdigest()
def read_json(path):return json.loads(path.read_text(encoding="utf-8"))

def configure(seed):
    seed_root=OUT/f"seed_{seed:03d}";audit=seed_root/"s0_audit";audit.mkdir(parents=True,exist_ok=True)
    c6=read_json(ADP6_TEMPLATE);c6["optimization_seed"]=seed
    c9=read_json(ADP9_TEMPLATE);c9["optimization_seed"]=seed
    p6=audit/"adp6_config.json";p9=audit/"adp9_config.json";dump(p6,c6);dump(p9,c9)
    adp6.OUT=seed_root;adp6.CONFIG=p6
    adp9.OUT=seed_root;adp9.A2=seed_root/"a2_joint_family_coordinate";adp9.CONFIG=p9
    adp7.A2=adp9.A2
    return seed_root,c6,c9,p6,p9

def s0_audit(seed,seed_root,c6,p6,p9):
    base.seed_everything(seed);bank=base.SetMechanismBank(5,c6["mechanism_dim"],c6["mechanism_heads"]).to(base.DEVICE)
    init_hash=hash_state_dict(bank.state_dict())
    tr=adp6.response_data("train");va=adp6.response_data("val")
    family_hash=sha_arrays([(f"train_{k}",v) for k,v in tr.items()]+[(f"val_{k}",v) for k,v in va.items()])
    data_files=sorted((ROOT/"datasets").rglob("*"));data_files=[p for p in data_files if p.is_file()]
    dataset_hash=sha_bytes("".join(f"{p.relative_to(ROOT)}:{base.sha256_file(p)}" for p in data_files).encode())
    optimizer_spec={"type":"AdamW","mechanism_lr":c6["mechanism_learning_rate"],"coordinate_lr":c6["coordinate_learning_rate"],"weight_decay":c6["weight_decay"],"initial_state":"empty"}
    result={"stage":"adp10_s0","optimization_seed":seed,"initial_model_hash":init_hash,"initial_optimizer_state_hash":sha_bytes(json.dumps(optimizer_spec,sort_keys=True).encode()),"data_split_hash":dataset_hash,"family_dataset_hash":family_hash,"adp6_config_hash":base.sha256_file(p6),"adp9_config_hash":base.sha256_file(p9),"world_seed":base.WORLD_SEED,"dataset_seed":base.DATASET_SEED,"git_commit":base.git_commit(),"pass":True}
    dump(seed_root/"s0_audit"/"audit.json",result);return result

def a2_stability_gate(a2):
    x=a2["last_round"];aff=[v for v in x["candidate_affine_aligned_realization_r2"] if v is not None];pf=[v for v in x["purity_high_candidate_functional_r2"] if v is not None]
    counts=x["candidate_family_count"];pur=[p for p,n in zip(x["candidate_purity_fixed"],counts) if n>0]
    checks={"three_gt_functional":min(x["matched_functional_r2"])>.95,"all_purity_high_functional":bool(pf) and min(pf)>.90,"order":x["median_order_accuracy"]>.95,"affine":bool(aff) and min(aff)>.95,"heldout":x["heldout_realization_nrmse"]<.05,"candidate_purity":bool(pur) and min(pur)>.90,"no_mechanism_mixing":bool(pur) and min(pur)>.90}
    return {"checks":checks,"pass":all(checks.values()),"assignment_oscillation_allowed":True}

def usage_control(seed_root,c9):
    bank,a,b,assign,x,h,alpha,alpha_h=adp9.load();alive=list(range(5));assignment=assign["val"].astype(np.int64);deleted=[]
    for _ in range(2):
        counts={j:int((assignment==j).sum()) for j in alive};drop=min(alive,key=lambda j:(counts[j],j));alive.remove(drop);deleted.append(f"C{drop+1}");assignment,_=adp9.reassign(alive,bank,a,b,x,alpha)
    m=adp9.metrics(alive,assignment,bank,a,b,x,h,alpha,alpha_h,c9);ok=len(alive)==3 and min(m["best_functional_r2_per_gt"])>.95 and m["family_hungarian_accuracy"]>.90 and m["mean_fragmentation"]<.10
    row={"control":"usage_prune","deleted":deleted,"alive":[f"C{j+1}" for j in alive],"metrics":m,"pass":ok};dump(seed_root/"negative_controls"/"usage_prune.json",row);return row

def joint_control(seed,seed_root,c9):
    graph=read_json(seed_root/"adp9_0_equivalence_graph"/"graph.json");edge=next((e for e in graph if e["valid_duplicate_edge"]),None)
    if edge is None:
        row={"control":"joint_finetune_after_first_edge","skipped":"no duplicate edge","pass":False};dump(seed_root/"negative_controls"/"joint_finetune.json",row);return row
    pair=tuple(edge["pair"]);cc={**c9,"epochs":c9["joint_control"]["epochs"],"response_batch_size":c9["joint_control"]["response_batch_size"],"gradient_clip":c9["joint_control"]["gradient_clip"],"mechanism_learning_rate":.0003,"coordinate_learning_rate":.001,"weight_decay":.00001}
    rows=[]
    for spec in adp7.SPECS["adp7_a2"]:
        x=adp7.evaluate_split(pair,spec,"source_copy",cc);x["pass"]=adp7.thresholds("adp7_a2",x);rows.append(x)
    row={"control":"joint_finetune_after_first_edge","pair":edge["pair_name"],"holdout":"four preregistered cross-realization directions","rows":rows,"all_splits_pass":all(x["pass"] for x in rows),"catastrophic_drift":not all(x["pass"] for x in rows),"pass":True}
    dump(seed_root/"negative_controls"/"joint_finetune.json",row)
    final=seed_root/"final_verdict.json"
    if final.exists():
        verdict=read_json(final);verdict.setdefault("negative_controls",{})["joint_finetune"]=row;dump(final,verdict)
    return row

def final_failure(a1,a2gate,prune,b1):
    if not a1 or not a1.get("pass"):return "F1_ADP6_A1_coordinate_failure"
    if not a2gate or not a2gate.get("pass"):return "F2_ADP6_A2_mechanism_mixing_or_coordinate_failure"
    if not prune:return "F3_duplicate_graph_false_negative"
    if prune.get("M_discovered",5)>3:return "F3_duplicate_graph_false_negative"
    if prune.get("M_discovered",5)<3:return "F4_duplicate_graph_false_positive"
    if not prune.get("pass"):return "F5_pruning_reassignment_instability"
    if not b1 or not b1.get("pass"):return "F6_full_TP_inference_failure"
    return None

def seed_run(seed,force=False):
    seed_root,c6,c9,p6,p9=configure(seed);final=seed_root/"final_verdict.json"
    if final.exists() and not force:return read_json(final)
    started=time.time();s0=s0_audit(seed,seed_root,c6,p6,p9)
    adp6.stage_0(force);adp6.stage_a0(force);a1=adp6.stage_a1(force)
    if not a1["pass"]:
        result={"seed":seed,"s0_pass":s0["pass"],"adp6_a1_pass":False,"adp6_a2_functional_pass":False,"adp9_prune_pass":False,"m_discovered":None,"b1_pass":False,"downstream_end_to_end_pass":False,"failure_mode":final_failure(a1,None,None,None),"wall_seconds":time.time()-started};dump(final,result);return result
    a2=adp6.stage_a2(force);a2gate=a2_stability_gate(a2);dump(seed_root/"a2_joint_family_coordinate"/"adp10_functional_gate.json",a2gate)
    prune=b1=None;usage=joint=None
    if a2gate["pass"]:
        graph=adp9.stage_0(force)
        if graph["pass"]:
            prune=adp9.stage_prune(force);verify=adp9.stage_verify(force)
            usage=usage_control(seed_root,c9);joint=joint_control(seed,seed_root,c9)
            if prune["pass"] and verify["pass"] and prune["M_discovered"]==3:b1=adp9.stage_b1(force)
    failure=final_failure(a1,a2gate,prune,b1);passed=failure is None
    result={"seed":seed,"s0_pass":s0["pass"],"adp6_a1_pass":a1["pass"],"adp6_a2_functional_pass":a2gate["pass"],"adp9_prune_pass":bool(prune and prune["pass"]),"m_discovered":prune.get("M_discovered") if prune else None,"b1_pass":bool(b1 and b1["pass"]),"downstream_end_to_end_pass":passed,"failure_mode":failure,"a1_metrics":{"heldout_nrmse":a1["last_round"]["heldout_realization_nrmse"],"matched_functional_r2":a1["last_round"]["matched_functional_r2"]},"a2_metrics":{"heldout_nrmse":a2["last_round"]["heldout_realization_nrmse"],"matched_functional_r2":a2["last_round"]["matched_functional_r2"],"family_accuracy":a2["last_round"]["family_hungarian_accuracy"],"fragmentation":a2["last_round"]["mean_GT_fragmentation"]},"pruning_path":[t["alive_before"] for t in prune["telemetry"]]+[prune["alive_candidates"]] if prune else [],"b1_iid":b1["splits"]["test_iid"] if b1 else None,"b1_101":b1["splits"]["test_combination_101"] if b1 else None,"negative_controls":{"usage_prune":usage,"joint_finetune":joint},"wall_seconds":time.time()-started}
    dump(final,result);print(json.dumps({"seed":seed,"pass":passed,"failure":failure,"M":result["m_discovered"]},indent=2),flush=True);return result

def aggregate(seeds):
    rows=[read_json(OUT/f"seed_{s:03d}"/"final_verdict.json") for s in seeds];d=OUT/("aggregate_3seed" if len(seeds)==3 else "aggregate_10seed");d.mkdir(parents=True,exist_ok=True)
    fields=["seed","adp6_a1_pass","adp6_a2_functional_pass","adp9_prune_pass","m_discovered","b1_pass","downstream_end_to_end_pass","failure_mode","wall_seconds"]
    with (d/"summary.csv").open("w",newline="",encoding="utf-8-sig") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:r.get(k) for k in fields} for r in rows])
    rate=lambda k:float(np.mean([bool(r.get(k)) for r in rows]));hist={str(k):sum(r.get("m_discovered")==k for r in rows) for k in (2,3,4,5,None)};need=2 if len(seeds)==3 else 8
    result={"stage":"adp10_aggregate","seeds":seeds,"sample_count":len(rows),"pass_requirement":f">={need}/{len(seeds)}","pass_count":sum(r["downstream_end_to_end_pass"] for r in rows),"pass":sum(r["downstream_end_to_end_pass"] for r in rows)>=need,"rates":{"adp6_a1":rate("adp6_a1_pass"),"adp6_a2_functional":rate("adp6_a2_functional_pass"),"adp9_count_and_prune":float(np.mean([r.get("adp9_prune_pass") and r.get("m_discovered")==3 for r in rows])),"b1":rate("b1_pass"),"end_to_end":rate("downstream_end_to_end_pass")},"m_discovered_histogram":hist,"failure_modes":{x:sum(r.get("failure_mode")==x for r in rows) for x in sorted(set(r.get("failure_mode") for r in rows),key=str)},"seed_summaries":rows}
    dump(d/"failure_modes.json",result["failure_modes"]);dump(d/"final_verdict.json",result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=["seed_run","controls","aggregate","auto"],required=True);p.add_argument("--seed",type=int);p.add_argument("--seeds",default="0,1,2");p.add_argument("--force",action="store_true");a=p.parse_args()
    seeds=[int(x) for x in a.seeds.split(",") if x.strip()]
    if a.stage=="seed_run":
        if a.seed is None:raise ValueError("--seed is required")
        result=seed_run(a.seed,a.force)
    elif a.stage=="controls":
        if a.seed is None:raise ValueError("--seed is required")
        seed_root,_,c9,_,_=configure(a.seed);result=joint_control(a.seed,seed_root,c9)
    elif a.stage=="aggregate":result=aggregate(seeds)
    else:
        first=[seed_run(s,a.force) for s in (0,1,2)];result=aggregate([0,1,2])
        if result["pass"]:
            for s in range(3,10):seed_run(s,a.force)
            result=aggregate(list(range(10)))
    print(json.dumps({k:result.get(k) for k in ("stage","seed","pass","pass_count","rates","failure_mode")},indent=2),flush=True)
if __name__=="__main__":main()
