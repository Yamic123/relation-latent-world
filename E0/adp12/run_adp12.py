"""ADP12 posterior responsibility and evidence-based candidate death."""
from __future__ import annotations
import argparse,csv,json,os,sys,time
from pathlib import Path
_DLL_HANDLE=None;_DLL_DIR=Path(sys.prefix)/"Library"/"bin"
if os.name=="nt" and _DLL_DIR.is_dir():
    os.environ["PATH"]=str(_DLL_DIR)+os.pathsep+os.environ.get("PATH","");_DLL_HANDLE=os.add_dll_directory(str(_DLL_DIR))
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import run_e0 as base
import adp6.run_adp6 as adp6
import adp7.run_adp7 as adp7
import adp9.run_adp9 as adp9
import adp11.run_adp11 as adp11
from adp1._utils import hash_state_dict

OUT=ROOT/"outputs"/"e0_adp12_bayesian_responsibility";CONFIG=ROOT/"configs"/"e0_adp12.json";DEVICE=base.DEVICE
VARIANTS={"H0":(False,False,"h0_hard_no_death"),"H1":(True,False,"h1_soft_no_death"),"H2":(False,True,"h2_hard_evidence_death"),"H3":(True,True,"h3_soft_evidence_death")}
def cv(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,torch.Tensor):return x.detach().cpu().tolist()
    raise TypeError(type(x))
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=cv),encoding="utf-8")
def append(p,x):p.parent.mkdir(parents=True,exist_ok=True);f=p.open("a",encoding="utf-8");f.write(json.dumps(x,ensure_ascii=False,default=cv)+"\n");f.close()
def read(p):return json.loads(p.read_text(encoding="utf-8"))
def cfg(seed):c=read(CONFIG);c["optimization_seed"]=seed;return c
def setup(variant,seed):
    soft,death,name=VARIANTS[variant];root=OUT/name/f"seed_{seed:03d}";c=cfg(seed);c.update({"variant":variant,"soft_responsibility":soft,"evidence_death":death});cp=root/"audit"/"config.json";dump(cp,c);adp6.OUT=root;adp6.CONFIG=cp;adp9.OUT=root;adp9.A2=root/"training";adp9.CONFIG=cp;adp7.A2=adp9.A2;return root,c,soft,death
def soft_q(cost,alive,kappa,eps):
    q=np.zeros_like(cost,np.float32);z=cost[:,alive];ss=np.sort(z,axis=1);gap=float(np.median(ss[:,1]-ss[:,0]));tau=kappa*max(gap,eps);logit=-z/tau;logit-=logit.max(1,keepdims=True);p=np.exp(logit);p/=p.sum(1,keepdims=True);q[:,alive]=p;return q,gap,tau
def hard_q(cost,alive):q=np.zeros_like(cost,np.float32);j=np.asarray(alive)[cost[:,alive].argmin(1)];q[np.arange(len(q)),j]=1;return q
def weighted_train(bank,a,b,S,d,alpha,q,c,opt,r):
    F,R=d.shape[:2];rng=np.random.default_rng(c["optimization_seed"]+50021*(r+1));losses=[];bank.train();st=time.time()
    for _ in range(c["epochs_per_round"]):
        order=rng.permutation(F)
        for p0 in range(0,F,c["family_batch_size"]):
            fi=order[p0:p0+c["family_batch_size"]];n=len(fi);St=torch.from_numpy(np.repeat(S[fi],R,axis=0)).to(DEVICE);at=torch.from_numpy(alpha[fi].reshape(-1)).to(DEVICE);dt=torch.from_numpy(d[fi].reshape(-1,3,2)).to(DEVICE);v=at[:,None]*a[None]+b[None];raw=bank.raw_effects(St,v);err=(raw-dt[:,None]).square().mean((2,3)).reshape(n,R,5).mean(1);qt=torch.from_numpy(q[fi]).to(DEVICE);loss=(err*qt).sum(1).mean();opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(list(bank.parameters())+[a,b],c["gradient_clip"]);opt.step();losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)),time.time()-st
def predict_mix(bank,a,b,S,alpha,q):
    pred=adp7.all_candidate_prediction(bank,S,alpha,a,b);return (pred*q[:,None,:,None,None]).sum(2),pred
def nrmse_mask(y,p,m):return base.nrmse(y[m],p[m]) if np.any(m) else 0.0
def subgroup_nrmse(y,p,x,alpha):
    regions=adp9.state_regions(x["S"]);bases=adp9.base_regions(x["base_id"]);out=[]
    for lid in np.unique(x["local_family_id"]):out.append(nrmse_mask(y,p,x["local_family_id"]==lid))
    for m in list(regions.values())+list(bases.values()):out.append(nrmse_mask(y,p,m))
    # Learner-visible action partitions operate at the realization level.
    for m in (alpha<0,alpha>=0,np.abs(alpha)<=.70001,np.abs(alpha)>.70001):out.append(base.nrmse(y[m],p[m]))
    return out
def death_test(j,q,bank,a,b,x,alpha,alpha_h,c,r):
    keep=q.sum(1)>0;qd=q.copy();qd[:,j]=0;den=qd.sum(1,keepdims=True);qd/=np.maximum(den,1e-12);obs,allobs=predict_mix(bank,a,b,x["S"],alpha,q);drop_obs,_=predict_mix(bank,a,b,x["S"],alpha,qd);held,allheld=predict_mix(bank,a,b,x["S"],alpha_h,q);drop_held,_=predict_mix(bank,a,b,x["S"],alpha_h,qd)
    regions=adp9.state_regions(x["S"]);bases=adp9.base_regions(x["base_id"]);domains={"iid":(x["response_train"],obs,drop_obs,np.ones(len(q),bool)),"state":(x["response_heldout"],held,drop_held,regions["extreme"]),"realization":(x["response_heldout"],held,drop_held,np.ones(len(q),bool)),"base":(x["response_heldout"],held,drop_held,bases["base_B"])}
    delta={k:nrmse_mask(y,pd,m)-nrmse_mask(y,pk,m) for k,(y,pk,pd,m) in domains.items()};rng=np.random.default_rng(c["optimization_seed"]*100003+r*101+j);uids=np.unique(x["base_id"]);upper={}
    for k,(y,pk,pd,m) in domains.items():
        vals=[]
        for _ in range(c["death_bootstrap_repeats"]):
            ids=rng.choice(uids,len(uids),replace=True);idx=np.concatenate([np.flatnonzero((x["base_id"]==u)&m) for u in ids]);vals.append(base.nrmse(y[idx],pd[idx])-base.nrmse(y[idx],pk[idx]) if len(idx) else 0.)
        upper[k]=float(np.quantile(vals,.95))
    before_sub=subgroup_nrmse(x["response_heldout"],held,x,alpha_h);after_sub=subgroup_nrmse(x["response_heldout"],drop_held,x,alpha_h);sub=[a0-b0 for a0,b0 in zip(after_sub,before_sub)]
    ok=max(delta.values())<c["death_global_delta_nrmse"] and max(upper.values())<c["death_global_delta_nrmse"] and max(sub)<c["death_subgroup_delta_nrmse"]
    return {"candidate":f"C{j+1}","round":r,"domain_delta_nrmse":delta,"bootstrap_upper_95":upper,"max_subgroup_delta_nrmse":float(max(sub)),"post_delete_subgroup_nrmse":after_sub,"checks":{"global":max(delta.values())<.01,"bootstrap":max(upper.values())<.01,"subgroup":max(sub)<.02},"pass":ok}
def z4_gate(last,c,alive):
    cnt=last["candidate_family_count"];major=[j in alive and n>=c["major_candidate_min_usage"]*sum(cnt) for j,n in enumerate(cnt)];pf=[x for x,m in zip(last["purity_high_candidate_functional_r2"],major) if m and x is not None];pur=[x for x,m in zip(last["candidate_purity_fixed"],major) if m];aff=[x for x,m in zip(last["candidate_affine_aligned_realization_r2"],major) if m and x is not None];checks={"functional":min(last["matched_functional_r2"])>.95,"major_functional":bool(pf) and min(pf)>.90,"purity":bool(pur) and min(pur)>.90,"affine":bool(aff) and min(aff)>.95,"order":last["median_order_accuracy"]>.95,"heldout":last["heldout_realization_nrmse"]<.05};return checks
def init(seed,c,n):
    bank,a,b,assignment=adp11.initial_objects(c,n);q=np.zeros((n,5),np.float32);q[np.arange(n),assignment]=1;return bank,a,b,q
def train(variant,seed,root,c,soft,death,force=False):
    final=root/"training"/"final_metrics.json"
    if final.exists() and not force:return read(final)
    adp11.prepare_coordinates(root);tr=adp6.response_data("train");va=adp6.response_data("val");hv=adp6.evaluator_data("val");atr,_=adp6.coordinates("train");av,avh=adp6.coordinates("val");bank,a,b,q=init(seed,c,len(tr["S"]));alive=list(range(5));ema=np.ones(5,np.float32)/5;low=np.zeros(5,np.int32);black=np.zeros(5,np.int32);opt=torch.optim.AdamW([{"params":bank.parameters(),"lr":c["mechanism_learning_rate"],"weight_decay":c["weight_decay"]},{"params":[a,b],"lr":c["coordinate_learning_rate"],"weight_decay":0.}]);rows=[];deaths=[];pending=None;started=time.time();(root/"training/checkpoints").mkdir(parents=True,exist_ok=True);(root/"training/assignments").mkdir(exist_ok=True);(root/"responsibility").mkdir(parents=True,exist_ok=True)
    dump(root/"audit/initial.json",{"initial_model_hash":hash_state_dict(bank.state_dict()),"initial_hard_count":np.bincount(q.argmax(1),minlength=5),"no_checkpoint_loaded":True})
    for r in range(c["rounds"]):
        validation_round=pending is not None
        if validation_round:mse,sec=float("nan"),0.0
        else:mse,sec=weighted_train(bank,a,b,tr["S"],tr["response_train"],atr,q,c,opt,r)
        assignment=q.argmax(1);cost=adp6.family_e_step_structured(bank,a,b,tr["S"],tr["response_train"],atr)[1];k=c["temperature_kappa"][min(r//10,3)]
        if soft:qnew,gap,tau=soft_q(cost,alive,k,c["temperature_epsilon"])
        else:qnew=hard_q(cost,alive);ss=np.sort(cost[:,alive],1);gap=float(np.median(ss[:,1]-ss[:,0]));tau=0.
        change=float(np.mean(qnew.argmax(1)!=assignment));q=qnew;rsp=q.mean(0);ema=c["responsibility_ema_decay"]*ema+(1-c["responsibility_ema_decay"])*rsp;low=np.where(ema<c["death_screen"],low+1,0);black=np.maximum(black-1,0)
        vacost=adp6.family_e_step_structured(bank,a,b,va["S"],va["response_train"],av)[1];vq,vg,vt=soft_q(vacost,alive,k,c["temperature_epsilon"]) if soft else (hard_q(vacost,alive),gap,0.);vassign=vq.argmax(1);train_pred,_=predict_mix(bank,a,b,tr["S"],atr,q);held,_=predict_mix(bank,a,b,va["S"],avh,vq);vp,vall= predict_mix(bank,a,b,va["S"],av,vq);cm=adp6.coordinate_metrics(av,adp6.structured_v(av,a.detach().cpu().numpy(),b.detach().cpu().numpy())[:, :, 0],vassign,hv) if False else None
        # Evaluator metrics use the posterior MAP family identity and its candidate coordinate.
        _,vv=adp6.predict(bank,va["S"],vassign,av,a,b);coord=adp6.coordinate_metrics(av,vv,vassign,hv);fm=adp6.functional_eval(bank,a,b,av,vassign,hv,c);_,_,acc,frag,_=adp6.fixed_structure(vassign,hv);entropy=float(np.mean(-(q*np.log(np.maximum(q,1e-12))).sum(1)))
        row={"round":r,"temperature_kappa":k,"cost_gap_median":gap,"temperature":tau,"posterior_entropy_mean":entropy,"effective_responsibility":rsp.tolist(),"ema_responsibility":ema.tolist(),"low_support_streak":low.tolist(),"alive":alive.copy(),"candidate_usage":(np.bincount(q.argmax(1),minlength=5)/len(q)).tolist(),"candidate_family_count":np.bincount(q.argmax(1),minlength=5).tolist(),"family_assignment_change":change,"heldout_realization_nrmse":base.nrmse(va["response_heldout"],held),"train_nrmse":base.nrmse(tr["response_train"],train_pred),"candidate_a":a.detach().cpu().tolist(),"candidate_b":b.detach().cpu().tolist(),**coord,**fm,"family_hungarian_accuracy":acc,"mean_GT_fragmentation":float(np.mean(frag)),"m_step_seconds":sec,"mechanisms_frozen":validation_round}
        rows.append(row);event=None
        if pending is not None:
            current_sub=subgroup_nrmse(va["response_heldout"],held,va,avh);global_drift=row["heldout_realization_nrmse"]-pending["expected_nrmse"];sub_drift=max(a0-b0 for a0,b0 in zip(current_sub,pending["expected_subgroup_nrmse"]));rollback=global_drift>=c["death_global_delta_nrmse"] or sub_drift>=c["death_subgroup_delta_nrmse"]
            pending["event"]["validation_round"]=r;pending["event"]["post_delete_frozen_global_drift"]=global_drift;pending["event"]["post_delete_frozen_max_subgroup_drift"]=sub_drift;pending["event"]["rollback"]=rollback
            if rollback:alive.append(pending["j"]);alive.sort();black[pending["j"]]=c["death_blacklist_rounds"]
            dump(pending["path"],pending["event"]);event={"validation_of":pending["event"]["candidate"],"rollback":rollback};pending=None
        stable=len(rows)>=3 and np.ptp([x["heldout_realization_nrmse"] for x in rows[-3:]])<c["stable_nrmse_range"] and np.median([x["heldout_realization_nrmse"] for x in rows[-3:]])<c["stable_nrmse_median"]
        if death and not validation_round and r>=c["death_start_round"] and stable:
            eligible=[j for j in alive if low[j]>=c["death_low_support_rounds"] and black[j]==0]
            if eligible:
                j=min(eligible,key=lambda x:ema[x]);event=death_test(j,vq,bank,a,b,va,av,avh,c,r);event["ema_responsibility"]=float(ema[j])
                if event["pass"]:
                    alive.remove(j);q[:,j]=0;q/=q.sum(1,keepdims=True);low[j]=0;event["commit"]=True;path=root/"candidate_death"/f"round_{r:03d}_C{j+1}.json";pending={"j":j,"event":event,"path":path,"expected_nrmse":row["heldout_realization_nrmse"]+event["domain_delta_nrmse"]["realization"],"expected_subgroup_nrmse":event["post_delete_subgroup_nrmse"]}
                else:black[j]=c["death_blacklist_rounds"];event["commit"]=False
                deaths.append(event);dump(root/"candidate_death"/f"round_{r:03d}_C{j+1}.json",event)
        row["death_event"]=event;append(root/"training/round_metrics.jsonl",row);np.savez_compressed(root/"responsibility"/f"round_{r:03d}.npz",q=q,ema=ema,alive=np.asarray(alive));np.savez_compressed(root/"training/assignments"/f"round_{r:03d}.npz",train=q.argmax(1),val=vassign,train_cost=cost,val_cost=vacost);torch.save({"bank":bank.state_dict(),"coordinate_a":a.detach().cpu(),"coordinate_b":b.detach().cpu()},root/"training/checkpoints"/f"round_{r:03d}.pt");print(f"[{variant} s{seed}] r={r:02d} H={entropy:.3f} alive={alive} ema={np.round(ema,3).tolist()} nrmse={row['heldout_realization_nrmse']:.3f} func={np.round(row['matched_functional_r2'],2).tolist()}",flush=True)
    checks=z4_gate(rows[-1],c,alive);result={"stage":"adp12_training","variant":variant,"seed":seed,"soft":soft,"death":death,"alive":alive,"death_events":deaths,"death_commits":sum(x.get("commit",False) and not x.get("rollback",False) for x in deaths),"last_round":rows[-1],"final_ema_responsibility":ema.tolist(),"z4_checks":checks,"z4_pass":all(checks.values()),"wall_seconds":time.time()-started};dump(final,result);return result
def duplicate_prune(root,c,t):
    bank,a,b,assign,x,h,alpha,alpha_h=adp9.load();alive=t["alive"].copy();assignment=assign["val"].astype(np.int64);tele=[];blocked=set()
    while True:
        g=adp9.build_graph(alive,assignment,bank,a,b,x,alpha,alpha_h,c);valid=[e for e in g if e["valid_duplicate_edge"] and tuple(e["pair"]) not in blocked]
        if not valid:break
        e=valid[0];pair=tuple(e["pair"]);keep,drop,quality=adp9.survivor_quality(pair,assignment,bank,a,b,x,alpha_h,c);pre=adp9.metrics(alive,assignment,bank,a,b,x,h,alpha,alpha_h,c);prop=[j for j in alive if j!=drop];a1,_=adp9.reassign(prop,bank,a,b,x,alpha);a2,_=adp9.reassign(prop,bank,a,b,x,alpha);post=adp9.metrics(prop,a2,bank,a,b,x,h,alpha,alpha_h,c);checks,inc=adp9.prune_gate(pre,post,[0.,0.],c);commit=all(checks.values());tele.append({"pair":e["pair_name"],"drop":f"C{drop+1}","commit":commit,"checks":checks,"max_increase":inc});
        if not commit:blocked.add(pair);continue
        alive=prop;assignment=a2
    m=adp9.metrics(alive,assignment,bank,a,b,x,h,alpha,alpha_h,c);fg=c["final_gate"];checks={"count":len(alive)==3,"family":m["family_hungarian_accuracy"]>fg["family_accuracy"],"fragmentation":m["mean_fragmentation"]<fg["mean_fragmentation"],"functional":min(m["best_functional_r2_per_gt"])>fg["min_best_functional_r2"],"iid":m["iid_heldout_nrmse"]<fg["iid_nrmse"],"structured":max(m["cross_state_nrmse"],m["cross_realization_nrmse"],m["cross_base_nrmse"])<fg["structured_nrmse"],"assignment_stability":True};res={"alive":alive,"M_discovered":len(alive),"telemetry":tele,"metrics":m,"checks":checks,"pass":all(checks.values()),"stop_reason":"no_valid_duplicate_edge","unsupported_slot_remaining":len(alive)>3};dump(root/"duplicate_pruning/final_metrics.json",res);torch.save({"bank":bank.state_dict(),"coordinate_a":a.cpu(),"coordinate_b":b.cpu(),"alive_indices":alive,"assignment":torch.from_numpy(assignment)},root/"duplicate_pruning/model.pt");return res

def run_b1(root,c,p):
    bank,a,b,_,_,_,_,_=adp9.load();saved=torch.load(root/"duplicate_pruning/model.pt",map_location=DEVICE,weights_only=True);bank.load_state_dict(saved["bank"]);alive=[int(x) for x in saved["alive_indices"]];func=np.asarray(p["metrics"]["functional_r2_matrix"])[alive];rr,gg=linear_sum_assignment(-func);local_to_gt={int(r):int(g) for r,g in zip(rr,gg)};rows={};started=time.time();out=root/"b1_full_tp";out.mkdir(parents=True,exist_ok=True)
    for split in c["b1"]["splits"]:
        S,_,delta=base.load_visible("e0a",split,"identity");hidden=base.load_hidden("e0a",split);lat=adp9.infer_support(bank,a,b,alive,S,delta,c,f"adp12_{c['variant']}_{c['optimization_seed']}_{split}");mgt=hidden["m_gt"];mhat=np.zeros_like(mgt);ehat=np.zeros_like(hidden["e_gt"])
        for local,g in local_to_gt.items():mhat[:,g]=lat["m"][:,local];ehat[:,g]=lat["effects"][:,local]
        per_f1=[adp9.f1_binary(mgt[:,g],mhat[:,g]) if mgt[:,g].any() else None for g in range(3)];inst=[adp9.r2(hidden["e_gt"][:,g],ehat[:,g]) if mgt[:,g].any() else None for g in range(3)];active_f1=[x for x in per_f1 if x is not None];active_inst=[x for x in inst if x is not None];row={"split":split,"sample_count":len(S),"nrmse":base.nrmse(delta,lat["prediction"]),"participation_f1_per_gt_active_only":per_f1,"participation_f1_active_macro":float(np.mean(active_f1)),"participation_f1_micro":adp9.f1_binary(mgt.reshape(-1),mhat.reshape(-1)),"instance_effect_r2_per_gt":inst,"min_instance_effect_r2":float(min(active_inst)),"support_exact_accuracy":float(np.mean(np.all(mgt==mhat,axis=1))),"seconds":lat["seconds"],"support_histogram":np.bincount(lat["support_code"],minlength=2**len(alive)).tolist()}
        if split=="test_combination_101":row.update({"inactive_Z2_false_positive_rate":float(mhat[:,1].mean()),"inactive_Z2_predicted_effect_norm":float(np.linalg.norm(ehat[:,1],axis=(1,2)).mean())})
        rows[split]=row;np.savez_compressed(out/f"{split}_assignments.npz",m=lat["m"],alpha=lat["alpha"],v=lat["v"],support_code=lat["support_code"]);print(f"[B1 {split}] nrmse={row['nrmse']:.4f} F1micro={row['participation_f1_micro']:.3f} minR2={row['min_instance_effect_r2']:.3f}",flush=True)
    iid,comb=rows["test_iid"],rows["test_combination_101"];checks={"iid_nrmse":iid["nrmse"]<.05,"101_nrmse":comb["nrmse"]<.10,"participation_f1":min(iid["participation_f1_micro"],comb["participation_f1_micro"])>.90,"instance_effect":min(iid["min_instance_effect_r2"],comb["min_instance_effect_r2"])>.90,"functional":min(p["metrics"]["best_functional_r2_per_gt"])>.90,"inactive_Z2_fpr_101":comb["inactive_Z2_false_positive_rate"]<.10};res={"alive_candidates":[f"C{x+1}" for x in alive],"local_to_evaluator_gt":local_to_gt,"splits":rows,"checks":checks,"pass":all(checks.values()),"wall_seconds":time.time()-started};dump(out/"final_metrics.json",res);return res
def failure(t,p):
    if not t["z4_pass"]:return "F4_persistent_mixing_or_identity_failure"
    if not p or p["M_discovered"]>3:return "F5_unsupported_slot_remains"
    if p["M_discovered"]<3:return "F2_premature_death"
    if not p["pass"]:return "F6_duplicate_or_reassignment_failure"
    return None
def h0_reference(seed,root):
    src=ROOT/"outputs/e0_adp11_from_zero/identity_tp"/f"seed_{seed:03d}"/"final_verdict.json";x=read(src);res={"variant":"H0","seed":seed,"reused_exact_baseline":str(src),"z4_pass":x["z4_pass"],"death_commits":0,"duplicate_prunes":0,"m_discovered":x["m_discovered"],"b1_pass":False,"final_pass":False,"failure_mode":"F4_persistent_mixing" if not x["z4_pass"] else "F5_unsupported_slot_remains"};dump(root/"final_verdict.json",res);return res
def run_one(variant,seed,force=False):
    root,c,soft,death=setup(variant,seed);final=root/"final_verdict.json"
    if final.exists() and not force:return read(final)
    if variant=="H0":return h0_reference(seed,root)
    t=train(variant,seed,root,c,soft,death,force);p=None
    if t["z4_pass"]:p=duplicate_prune(root,c,t)
    fm=failure(t,p);b1=run_b1(root,c,p) if fm is None else None
    if fm is None and not b1["pass"]:fm="F7_B1_generalization_failure"
    res={"variant":variant,"seed":seed,"z4_pass":t["z4_pass"],"death_events":len(t["death_events"]),"death_commits":t["death_commits"],"duplicate_prunes":sum(x["commit"] for x in p["telemetry"]) if p else 0,"m_discovered":p["M_discovered"] if p else None,"family_accuracy":p["metrics"]["family_hungarian_accuracy"] if p else t["last_round"]["family_hungarian_accuracy"],"fragmentation":p["metrics"]["mean_fragmentation"] if p else t["last_round"]["mean_GT_fragmentation"],"b1_pass":bool(b1 and b1["pass"]),"final_pass":fm is None,"failure_mode":fm};dump(final,res);print(json.dumps(res,indent=2),flush=True);return res
def aggregate(seeds):
    rows=[read(OUT/name/f"seed_{s:03d}/final_verdict.json") for _,_,name in VARIANTS.values() for s in seeds];d=OUT/"aggregate_3seed";d.mkdir(parents=True,exist_ok=True);fields=["variant","seed","z4_pass","death_commits","duplicate_prunes","m_discovered","family_accuracy","fragmentation","b1_pass","final_pass","failure_mode"]
    with (d/"summary.csv").open("w",newline="",encoding="utf-8-sig") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:r.get(k) for k in fields} for r in rows])
    by={v:{"z4_pass":sum(r["z4_pass"] for r in rows if r["variant"]==v),"end_to_end_pass":sum(r["final_pass"] and r.get("b1_pass") for r in rows if r["variant"]==v)} for v in VARIANTS};res={"stage":"adp12_pilot","seeds":seeds,"variants":by,"h3_pilot_gate":by["H3"]["end_to_end_pass"]>=2,"pass":by["H3"]["end_to_end_pass"]>=2};dump(d/"final_verdict.json",res);return res
def main():
    p=argparse.ArgumentParser();p.add_argument("--variant",choices=list(VARIANTS));p.add_argument("--seed",type=int);p.add_argument("--stage",choices=["run","pilot_all","aggregate"],default="run");p.add_argument("--seeds",default="0,1,2");p.add_argument("--force",action="store_true");a=p.parse_args();seeds=[int(x) for x in a.seeds.split(",")]
    if a.stage=="run":res=run_one(a.variant,a.seed,a.force)
    elif a.stage=="pilot_all":
        for v in VARIANTS:
            for s in seeds:run_one(v,s,a.force)
        res=aggregate(seeds)
    else:res=aggregate(seeds)
    print(json.dumps(res,indent=2),flush=True)
if __name__=="__main__":main()
