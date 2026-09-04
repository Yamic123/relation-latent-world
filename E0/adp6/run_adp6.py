"""ADP6 stages, reusing ADP5 families, assignments and evaluators."""
from __future__ import annotations
import argparse,json,os,sys,time
from pathlib import Path
from typing import Any

# A directly-invoked Windows conda interpreter does not always inherit the
# activated environment's DLL search path.  Keep the handle alive for the
# whole process; otherwise NumPy's delayed BLAS load can fail with 0xc06d007f.
_DLL_HANDLE = None
_CONDA_DLL_DIR = Path(sys.prefix) / "Library" / "bin"
if os.name == "nt" and _CONDA_DLL_DIR.is_dir():
    os.environ["PATH"] = str(_CONDA_DLL_DIR) + os.pathsep + os.environ.get("PATH", "")
    _DLL_HANDLE = os.add_dll_directory(str(_CONDA_DLL_DIR))
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import run_e0 as base
import adp5.run_adp5 as prev
from adp3.run_adp3 import functional_matrix as probe_functional_matrix
from adp1._utils import hash_state_dict

CONFIG=ROOT/"configs"/"e0_adp6_8gb.json";OUT=ROOT/"outputs"/"e0_adp6";DEVICE=base.DEVICE
ADP5_OUT=ROOT/"outputs"/"e0_adp5";ADP5_A1=ADP5_OUT/"a1_family_discovery"
def cfg():return json.loads(CONFIG.read_text(encoding="utf-8"))
def jd(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,torch.Tensor):return x.detach().cpu().tolist()
    raise TypeError(type(x))
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=jd),encoding="utf-8")
def append(p,x):p.parent.mkdir(parents=True,exist_ok=True);f=p.open("a",encoding="utf-8");f.write(json.dumps(x,ensure_ascii=False,default=jd)+"\n");f.close()
def meta(stage,c):
    x=base.environment_metadata();x.update({"stage":stage,"world_seed":base.WORLD_SEED,"dataset_seed":base.DATASET_SEED,"optimization_seed":c["optimization_seed"],"config":c,"config_sha256":base.sha256_file(CONFIG),"script_sha256":base.sha256_file(Path(__file__)),"git_commit":base.git_commit()});return x

def response_data(split):return prev.visible(split)
def evaluator_data(split):return prev.hidden(split)

def action_coordinate(actions_train,actions_heldout):
    """Rank-1 PC1 with a learner-visible, reproducible sign convention."""
    F=len(actions_train);q=np.empty((F,2),np.float32);at=np.empty(actions_train.shape[:2],np.float32);ah=np.empty(actions_heldout.shape[:2],np.float32)
    for u in range(F):
        d=actions_train[u] # baseline action is zero
        norms=np.sqrt(np.square(d).sum(axis=1));anchor=int(np.flatnonzero(norms>1e-12)[0]);qu=d[anchor]/max(float(norms[anchor]),1e-12)
        # The anchor projection is positive by construction. This resolves PC1's sign without GT.
        alpha=(d*qu[None]).sum(axis=1);scale=max(float(np.max(np.abs(alpha))),1e-12);q[u]=qu;at[u]=alpha/scale;ah[u]=(actions_heldout[u]*qu[None]).sum(axis=1)/scale
    return q,at,ah

def pair_order(x,y):
    good=total=0
    for i in range(len(x)):
        for j in range(i+1,len(x)):
            if x[i]==x[j]:continue
            total+=1;good+=int((x[i]-x[j])*(y[i]-y[j])>0)
    acc=good/max(total,1);return max(acc,1-acc)

def pearson_manual(x,y):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64);x=x-x.mean();y=y-y.mean()
    return float((x*y).sum()/max(float(np.sqrt((x*x).sum()*(y*y).sum())),1e-12))

def ranks_no_ties(x):
    order=np.argsort(x,kind="mergesort");r=np.empty(len(x),dtype=np.float64);r[order]=np.arange(len(x),dtype=np.float64);return r

def spearman_manual(x,y):return pearson_manual(ranks_no_ties(x),ranks_no_ties(y))

def kendall_manual(x,y):
    concord=discord=0
    for i in range(len(x)):
        for j in range(i+1,len(x)):
            z=(x[i]-x[j])*(y[i]-y[j])
            if z>0:concord+=1
            elif z<0:discord+=1
    return float((concord-discord)/max(concord+discord,1))

def stage_0(force=False):
    c=cfg();path=OUT/"adp6_0_action_coordinate"/"final_metrics.json"
    if path.exists() and not force:return json.loads(path.read_text(encoding="utf-8"))
    audits={};coords={};all_abs_s=[];all_abs_k=[];all_order=[];pooled_alpha=[];pooled_gt=[]
    for split in ("train","val","test"):
        x=response_data(split);h=evaluator_data(split);q,a,ah=action_coordinate(x["action_train"],x["action_heldout"]);coords[f"{split}_q"]=q;coords[f"{split}_alpha_train"]=a;coords[f"{split}_alpha_heldout"]=ah
        ss=[];kk=[];oo=[]
        for i in range(len(a)):
            gt=h["train_amplitude_gt"][i];ss.append(spearman_manual(a[i],gt));kk.append(kendall_manual(a[i],gt));oo.append(pair_order(a[i],gt))
        all_abs_s.extend(np.abs(ss));all_abs_k.extend(np.abs(kk));all_order.extend(oo);pooled_alpha.append(a.reshape(-1));pooled_gt.append(h["train_amplitude_gt"].reshape(-1))
        audits[split]={"family_count":len(a),"median_abs_spearman":float(np.median(np.abs(ss))),"median_abs_kendall":float(np.median(np.abs(kk))),"median_pairwise_order_accuracy_allow_flip":float(np.median(oo)),"spearman_sign_positive_fraction":float(np.mean(np.asarray(ss)>0))}
    pa=np.concatenate(pooled_alpha);pg=np.concatenate(pooled_gt);same=float(np.mean(np.sign(pa)==np.sign(pg)));global_sign=max(same,1-same)
    med_s=float(np.median(all_abs_s));med_k=float(np.median(all_abs_k));med_o=float(np.median(all_order));checks={"spearman":med_s>.98,"kendall":med_k>.95,"pair_order":med_o>.98}
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path.parent/"learner_coordinates.npz",**coords)
    result={**meta("adp6_0",c),"audits":audits,"overall_median_abs_spearman":med_s,"overall_median_abs_kendall":med_k,"overall_median_pairwise_order_accuracy":med_o,"sign_agreement_after_one_global_flip":global_sign,"pc1_sign_convention":c["pc1_sign_convention"],"checks":checks,"pass":bool(all(checks.values()))};dump(path,result);dump(path.parent/"config.json",c);(path.parent/"summary.md").write_text(f"# ADP6-0\n\n- PASS: **{result['pass']}**\n- |Spearman|: {med_s}\n- |Kendall|: {med_k}\n- order: {med_o}\n- global sign agreement: {global_sign}\n",encoding="utf-8");print(json.dumps({"checks":checks,"global_sign_agreement":global_sign,"pass":result["pass"]},indent=2),flush=True);return result

def coordinates(split):
    with np.load(OUT/"adp6_0_action_coordinate"/"learner_coordinates.npz") as z:return z[f"{split}_alpha_train"],z[f"{split}_alpha_heldout"]

def derive_fixed_assignments(force=False):
    cache=OUT/"a0_free_v_baseline"/"fixed_assignments.npz"
    if cache.exists() and not force:
        with np.load(cache) as z:return {k:z[k] for k in z.files}
    c5=prev.cfg();bank=base.SetMechanismBank(5,c5["mechanism_dim"],c5["mechanism_heads"]).to(DEVICE);bank.load_state_dict(torch.load(ADP5_A1/"checkpoints"/"round_039_bank.pt",map_location=DEVICE,weights_only=True))
    train=np.load(ADP5_A1/"assignments"/"round_039.npz")["candidate"]
    vals={"train":train};hash0=hash_state_dict(bank.state_dict())
    for split in ("val","test"):
        x=response_data(split);e=prev.family_e_step(bank,x["S"],x["response_train"],c5,c5["optimization_seed"]+70000001+39*1000003+(0 if split=="val" else 800003));vals[split]=e["assignment"]
    assert hash0==hash_state_dict(bank.state_dict())
    cache.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(cache,**vals);return vals

def stage_a0(force=False):
    c=cfg();gate=stage_0(False)
    if not gate["pass"]:raise RuntimeError("STOP: ADP6-0 failed")
    source=json.loads((ADP5_A1/"final_metrics.json").read_text(encoding="utf-8"));assign=derive_fixed_assignments(force)
    path=OUT/"a0_free_v_baseline"/"final_metrics.json";result={**meta("adp6_a0",c),"reused_not_retrained":True,"source_final_metrics":str(ADP5_A1/"final_metrics.json"),"source_checkpoint":str(ADP5_A1/"checkpoints"/"round_039_bank.pt"),"source_checkpoint_sha256":base.sha256_file(ADP5_A1/"checkpoints"/"round_039_bank.pt"),"assignment_counts":{k:np.bincount(v,minlength=5).tolist() for k,v in assign.items()},"heldout_nrmse":source["last_round"]["heldout_realization_family_nrmse"],"matched_functional_r2":source["last_round"]["matched_functional_r2"],"family_accuracy":source["last_round"]["family_hungarian_accuracy"],"mean_fragmentation":source["last_round"]["mean_GT_fragmentation"],"pass_as_adp6_structured_coordinate":False};dump(path,result);dump(path.parent/"config.json",c);(path.parent/"summary.md").write_text(f"# ADP6-A0 free-v baseline\n\n- reused: true\n- heldout NRMSE: {result['heldout_nrmse']}\n- functional R2: {result['matched_functional_r2']}\n",encoding="utf-8");print(json.dumps({"reused":True,"heldout_nrmse":result["heldout_nrmse"],"functional":result["matched_functional_r2"]},indent=2),flush=True);return result

def structured_v(alpha,a,b):return alpha[:,:,None]*a[None,None,:]+b[None,None,:]

def initialize_a1_bank(c):
    """Create the ADP6-A1 bank.

    Historical ADP6 starts from the ADP5 bank.  Stability experiments can
    explicitly request an independent random ADP6 initialization while still
    reusing the same ADP5 family evidence.  The default remains unchanged.
    """
    bank=base.SetMechanismBank(5,c["mechanism_dim"],c["mechanism_heads"]).to(DEVICE)
    mode=c.get("a1_initialization","adp5_checkpoint")
    if mode=="adp5_checkpoint":
        bank.load_state_dict(torch.load(ADP5_A1/"checkpoints"/"round_039_bank.pt",map_location=DEVICE,weights_only=True))
    elif mode!="random":
        raise ValueError(f"Unknown a1_initialization: {mode}")
    return bank

@torch.no_grad()
def predict(bank,S,assignment,alpha,a,b,batch_f=128):
    F,R=alpha.shape;out=[];bank.eval();av=a.detach().cpu().numpy();bv=b.detach().cpu().numpy();v=av[assignment,None]*alpha+bv[assignment,None]
    for start in range(0,F,batch_f):
        stop=min(F,start+batch_f);n=stop-start;St=torch.from_numpy(np.repeat(S[start:stop],R,axis=0)).to(DEVICE);ct=torch.from_numpy(np.repeat(assignment[start:stop],R).astype(np.int64)).to(DEVICE);vt=torch.from_numpy(v[start:stop].reshape(-1).astype(np.float32)).to(DEVICE);va=torch.zeros((n*R,5),device=DEVICE);va.scatter_(1,ct[:,None],vt[:,None]);raw=bank.raw_effects(St,va);out.append(raw[torch.arange(n*R,device=DEVICE),ct].cpu().numpy().reshape(n,R,3,2))
    return np.concatenate(out),v

def train_round(bank,a,b,S,d,alpha,assignment,c,opt,round_id):
    F,R=d.shape[:2];rng=np.random.default_rng(c["optimization_seed"]+50021*(round_id+1));losses=[];started=time.time();bank.train()
    for _ in range(c["epochs_per_round"]):
        order=rng.permutation(F)
        for start in range(0,F,c["family_batch_size"]):
            fi=order[start:start+c["family_batch_size"]];n=len(fi);ct=torch.from_numpy(np.repeat(assignment[fi],R).astype(np.int64)).to(DEVICE);at=torch.from_numpy(alpha[fi].reshape(-1)).to(DEVICE);vt=a[ct]*at+b[ct];va=torch.zeros((n*R,5),device=DEVICE);va.scatter_(1,ct[:,None],vt[:,None]);St=torch.from_numpy(np.repeat(S[fi],R,axis=0)).to(DEVICE);dt=torch.from_numpy(d[fi].reshape(-1,3,2)).to(DEVICE)
            opt.zero_grad(set_to_none=True);raw=bank.raw_effects(St,va);pred=raw[torch.arange(n*R,device=DEVICE),ct];loss=(pred-dt).square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(list(bank.parameters())+[a,b],c["gradient_clip"]);opt.step();losses.append(float(loss.detach().cpu()))
    return {"seconds":time.time()-started,"minibatch_mse":float(np.mean(losses))}

def linear_r2(x,y):
    xm=float(x.mean());ym=float(y.mean());slope=float(((x-xm)*(y-ym)).sum()/max(float(((x-xm)**2).sum()),1e-12));pred=slope*x+(ym-slope*xm);return base.r2_score(y[:,None],pred[:,None])

def fixed_structure(assignment,h):
    cont=np.zeros((5,3),np.int64)
    for x,y in zip(assignment,h["family_gt"]):cont[int(x),int(y)]+=1
    lr,g=linear_sum_assignment(-cont);mp={int(l):int(z) for l,z in zip(lr,g)};acc=float(np.mean([mp.get(int(x),-1)==int(y) for x,y in zip(assignment,h["family_gt"])]));frag=(1-cont.max(axis=0)/cont.sum(axis=0)).tolist();purity=(cont.max(axis=1)/np.maximum(cont.sum(axis=1),1)).tolist();return cont,mp,acc,frag,purity

def coordinate_metrics(alpha,v_all,assignment,h):
    per_order=[];per_s=[];per_k=[]
    for u in range(len(alpha)):
        vv=v_all[u];per_order.append(pair_order(alpha[u],vv));per_s.append(abs(spearman_manual(alpha[u],vv)));per_k.append(abs(kendall_manual(alpha[u],vv)))
    aligned=[];samevar=[]
    gt=h["train_amplitude_gt"]
    for j in range(5):
        take=assignment==j
        if not take.any():aligned.append(None);samevar.append(None);continue
        aligned.append(float(linear_r2(v_all[take].reshape(-1),gt[take].reshape(-1))));samevar.append(np.var(v_all[take],axis=0).tolist())
    return {"median_order_accuracy":float(np.median(per_order)),"median_abs_spearman_alpha_inferred_v":float(np.median(per_s)),"median_abs_kendall_alpha_inferred_v":float(np.median(per_k)),"candidate_affine_aligned_realization_r2":aligned,"same_alpha_cross_family_variance":samevar}

def functional_eval(bank,a,b,alpha,assignment,h,c):
    F,R=alpha.shape;aa=a.detach().cpu().numpy();bb=b.detach().cpu().numpy();vall=structured_v(alpha,aa,bb).reshape(F*R,5);flat={"y_gt":np.repeat(h["family_gt"],R),"v_gt":h["train_amplitude_gt"].reshape(-1)};func=probe_functional_matrix(bank,vall,flat,{"m_max":5,"a1":{"functional_probe_count":c["functional_probe_count"]}})
    fl,fg=linear_sum_assignment(-np.nan_to_num(func,nan=-1e6));fmap={int(g):int(l) for l,g in zip(fl,fg)};matched=[float(func[fmap[g],g]) for g in range(3)];cont,mp,acc,frag,purity=fixed_structure(assignment,h);major=cont.argmax(axis=1);purity_scores=[float(func[j,major[j]]) if purity[j]>.9 and cont[j].sum()>0 else None for j in range(5)]
    return {"functional_r2_matrix":func.tolist(),"functional_gt_to_candidate":fmap,"matched_functional_r2":matched,"family_contingency":cont.tolist(),"family_accuracy_fixed":acc,"fragmentation_per_gt_fixed":frag,"candidate_purity_fixed":purity,"purity_high_candidate_functional_r2":purity_scores}

def stage_a1(force=False):
    c=cfg();gate=stage_a0(False)
    if not gate["reused_not_retrained"]:raise RuntimeError("STOP: free-v baseline missing")
    out=OUT/"a1_fixed_assignment_structured_v";final=out/"final_metrics.json"
    if final.exists() and not force:return json.loads(final.read_text(encoding="utf-8"))
    if (out/"round_metrics.jsonl").exists():raise FileExistsError(f"Refusing overwrite {out}")
    (out/"checkpoints").mkdir(parents=True,exist_ok=True);dump(out/"config.json",c);fixed=derive_fixed_assignments(False);tr=response_data("train");va=response_data("val");hv=evaluator_data("val");atr,_=coordinates("train");av,avh=coordinates("val")
    base.seed_everything(c["optimization_seed"]);bank=initialize_a1_bank(c);initial_bank_hash=hash_state_dict(bank.state_dict());a=torch.nn.Parameter(torch.ones(5,device=DEVICE));b=torch.nn.Parameter(torch.zeros(5,device=DEVICE));opt=torch.optim.AdamW([{"params":bank.parameters(),"lr":c["mechanism_learning_rate"],"weight_decay":c["weight_decay"]},{"params":[a,b],"lr":c["coordinate_learning_rate"],"weight_decay":0.0}]);rows=[];started=time.time();torch.cuda.reset_peak_memory_stats() if DEVICE.type=="cuda" else None
    for r in range(c["rounds"]):
        tm=train_round(bank,a,b,tr["S"],tr["response_train"],atr,fixed["train"],c,opt,r);train_pred,train_v=predict(bank,tr["S"],fixed["train"],atr,a,b);val_pred,val_v=predict(bank,va["S"],fixed["val"],av,a,b);held_pred,held_v=predict(bank,va["S"],fixed["val"],avh,a,b);train_mse=float(np.mean((train_pred-tr["response_train"])**2));held_nrmse=base.nrmse(va["response_heldout"],held_pred);cm=coordinate_metrics(av,val_v,fixed["val"],hv);fm=functional_eval(bank,a,b,av,fixed["val"],hv,c)
        torch.save({"bank":bank.state_dict(),"coordinate_a":a.detach().cpu(),"coordinate_b":b.detach().cpu()},out/"checkpoints"/f"round_{r:03d}.pt")
        row={"round":r,"train_response_mse":train_mse,"heldout_realization_nrmse":held_nrmse,"candidate_a":a.detach().cpu().tolist(),"candidate_b":b.detach().cpu().tolist(),**cm,**fm,"family_assignment_change":0.0,"candidate_usage":(np.bincount(fixed["train"],minlength=5)/len(fixed["train"])).tolist(),"candidate_family_count":np.bincount(fixed["train"],minlength=5).tolist(),"m_step_seconds":tm["seconds"],"round_seconds":tm["seconds"],"cuda_peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20 if DEVICE.type=="cuda" else 0.0};append(out/"round_metrics.jsonl",row);rows.append(row);print(f"[A1] r={r:02d} nrmse={held_nrmse:.4f} func={np.round(fm['matched_functional_r2'],3).tolist()} purity-func={np.round([x if x is not None else -99 for x in fm['purity_high_candidate_functional_r2']],3).tolist()} a={np.round(row['candidate_a'],3).tolist()}",flush=True)
    last=rows[-1];aff=[x for x in last["candidate_affine_aligned_realization_r2"] if x is not None];pf=[x for x in last["purity_high_candidate_functional_r2"] if x is not None];th=c["pass_thresholds"];checks={"order":last["median_order_accuracy"]>th["median_order_accuracy"],"affine_realization":min(aff)>th["candidate_affine_realization_r2"],"heldout":last["heldout_realization_nrmse"]<th["heldout_realization_nrmse"],"purity_high_functional":sum(x>th["purity_high_functional_r2"] for x in pf)>=th["min_purity_high_candidates_above_threshold"],"not_all_matched_negative":any(x>0 for x in last["matched_functional_r2"]),"improves_over_adp5":max(last["matched_functional_r2"])>max(gate["matched_functional_r2"])}
    result={**meta("adp6_a1",c),"a1_initialization":c.get("a1_initialization","adp5_checkpoint"),"initial_checkpoint":str(ADP5_A1/"checkpoints"/"round_039_bank.pt") if c.get("a1_initialization","adp5_checkpoint")=="adp5_checkpoint" else None,"initial_model_hash":initial_bank_hash,"fixed_assignment_source":str(OUT/"a0_free_v_baseline"/"fixed_assignments.npz"),"rounds_completed":len(rows),"last_round":last,"checks":checks,"pass":bool(all(checks.values())),"stop_required":not all(checks.values()),"wall_seconds":time.time()-started,"cuda_peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20 if DEVICE.type=="cuda" else 0.0};dump(final,result);(out/"summary.md").write_text(f"# ADP6-A1\n\n- PASS: **{result['pass']}**\n- heldout NRMSE: {last['heldout_realization_nrmse']}\n- functional R2: {last['matched_functional_r2']}\n- affine latent R2: {last['candidate_affine_aligned_realization_r2']}\n- order: {last['median_order_accuracy']}\n",encoding="utf-8");print(json.dumps({"pass":result["pass"],"checks":checks},indent=2),flush=True);return result

def transformed_coordinates(kind,split):
    """Learner-visible negative-control coordinates, deterministic by family."""
    at,ah=coordinates(split)
    if kind=="shuffle_alpha":
        rng=np.random.default_rng(610001+{"train":0,"val":1,"test":2}[split]);at=at.copy();ah=ah.copy()
        for u in range(len(at)):
            at[u]=at[u,rng.permutation(at.shape[1])];ah[u]=ah[u,rng.permutation(ah.shape[1])]
    elif kind=="family_flip":
        rng=np.random.default_rng(620001+{"train":0,"val":1,"test":2}[split]);sgn=rng.choice(np.array([-1.0,1.0],np.float32),size=(len(at),1));at=at*sgn;ah=ah*sgn
    return at.astype(np.float32),ah.astype(np.float32)

def run_fixed_shared_negative(kind,force=False):
    gate=stage_a1(False)
    if not gate["pass"]:raise RuntimeError("STOP: A1 failed")
    c=cfg();out=OUT/"negatives"/kind;final=out/"final_metrics.json"
    if final.exists() and not force:return json.loads(final.read_text(encoding="utf-8"))
    if (out/"round_metrics.jsonl").exists():raise FileExistsError(f"Refusing overwrite {out}")
    out.mkdir(parents=True,exist_ok=True);dump(out/"config.json",c);fixed=derive_fixed_assignments(False);tr=response_data("train");va=response_data("val");hv=evaluator_data("val");atr,_=transformed_coordinates(kind,"train");av,avh=transformed_coordinates(kind,"val")
    base.seed_everything(c["optimization_seed"]);bank=base.SetMechanismBank(5,c["mechanism_dim"],c["mechanism_heads"]).to(DEVICE);bank.load_state_dict(torch.load(ADP5_A1/"checkpoints"/"round_039_bank.pt",map_location=DEVICE,weights_only=True));a=torch.nn.Parameter(torch.ones(5,device=DEVICE));b=torch.nn.Parameter(torch.zeros(5,device=DEVICE));opt=torch.optim.AdamW([{"params":bank.parameters(),"lr":c["mechanism_learning_rate"],"weight_decay":c["weight_decay"]},{"params":[a,b],"lr":c["coordinate_learning_rate"],"weight_decay":0.0}]);rows=[];started=time.time();torch.cuda.reset_peak_memory_stats() if DEVICE.type=="cuda" else None
    for r in range(c["rounds"]):
        tm=train_round(bank,a,b,tr["S"],tr["response_train"],atr,fixed["train"],c,opt,r);train_pred,train_v=predict(bank,tr["S"],fixed["train"],atr,a,b);val_pred,val_v=predict(bank,va["S"],fixed["val"],av,a,b);held_pred,held_v=predict(bank,va["S"],fixed["val"],avh,a,b);cm=coordinate_metrics(av,val_v,fixed["val"],hv);fm=functional_eval(bank,a,b,av,fixed["val"],hv,c)
        row={"round":r,"train_response_mse":float(np.mean((train_pred-tr["response_train"])**2)),"heldout_realization_nrmse":base.nrmse(va["response_heldout"],held_pred),"candidate_a":a.detach().cpu().tolist(),"candidate_b":b.detach().cpu().tolist(),**cm,**fm,"family_assignment_change":0.0,"candidate_usage":(np.bincount(fixed["train"],minlength=5)/len(fixed["train"])).tolist(),"candidate_family_count":np.bincount(fixed["train"],minlength=5).tolist(),"m_step_seconds":tm["seconds"],"round_seconds":tm["seconds"],"cuda_peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20 if DEVICE.type=="cuda" else 0.0};append(out/"round_metrics.jsonl",row);rows.append(row);print(f"[NC {kind}] r={r:02d} nrmse={row['heldout_realization_nrmse']:.4f} func={np.round(fm['matched_functional_r2'],3).tolist()}",flush=True)
    torch.save({"bank":bank.state_dict(),"coordinate_a":a.detach().cpu(),"coordinate_b":b.detach().cpu()},out/"final.pt");result={**meta("adp6_nc_"+kind,c),"control":kind,"rounds_completed":len(rows),"last_round":rows[-1],"wall_seconds":time.time()-started,"cuda_peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20 if DEVICE.type=="cuda" else 0.0};dump(final,result);return result

def train_round_family_affine(bank,au,bu,S,d,alpha,assignment,c,opt,round_id):
    F,R=d.shape[:2];rng=np.random.default_rng(c["optimization_seed"]+50021*(round_id+1));losses=[];started=time.time();bank.train()
    for _ in range(c["epochs_per_round"]):
        order=rng.permutation(F)
        for start in range(0,F,c["family_batch_size"]):
            fi=order[start:start+c["family_batch_size"]];n=len(fi);ui=torch.from_numpy(np.repeat(fi,R).astype(np.int64)).to(DEVICE);ct=torch.from_numpy(np.repeat(assignment[fi],R).astype(np.int64)).to(DEVICE);at=torch.from_numpy(alpha[fi].reshape(-1)).to(DEVICE);vt=au[ui]*at+bu[ui];vall=torch.zeros((n*R,5),device=DEVICE);vall.scatter_(1,ct[:,None],vt[:,None]);St=torch.from_numpy(np.repeat(S[fi],R,axis=0)).to(DEVICE);dt=torch.from_numpy(d[fi].reshape(-1,3,2)).to(DEVICE);opt.zero_grad(set_to_none=True);raw=bank.raw_effects(St,vall);pred=raw[torch.arange(n*R,device=DEVICE),ct];loss=(pred-dt).square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(list(bank.parameters())+[au,bu],c["gradient_clip"]);opt.step();losses.append(float(loss.detach().cpu()))
    return {"seconds":time.time()-started,"minibatch_mse":float(np.mean(losses))}

def infer_family_affine(bank,S,d,alpha,assignment,steps=40):
    F,R=alpha.shape;au=torch.nn.Parameter(torch.ones(F,device=DEVICE));bu=torch.nn.Parameter(torch.zeros(F,device=DEVICE));requires=[p.requires_grad for p in bank.parameters()]
    for p in bank.parameters():p.requires_grad_(False)
    opt=torch.optim.Adam([au,bu],lr=.03);St=torch.from_numpy(np.repeat(S,R,axis=0)).to(DEVICE);dt=torch.from_numpy(d.reshape(-1,3,2)).to(DEVICE);ui=torch.arange(F,device=DEVICE).repeat_interleave(R);ct=torch.from_numpy(np.repeat(assignment,R).astype(np.int64)).to(DEVICE);at=torch.from_numpy(alpha.reshape(-1)).to(DEVICE)
    for _ in range(steps):
        vt=au[ui]*at+bu[ui];vall=torch.zeros((F*R,5),device=DEVICE);vall.scatter_(1,ct[:,None],vt[:,None]);raw=bank.raw_effects(St,vall);loss=(raw[torch.arange(F*R,device=DEVICE),ct]-dt).square().mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    for p,x in zip(bank.parameters(),requires):p.requires_grad_(x)
    return au.detach().cpu().numpy(),bu.detach().cpu().numpy()

def family_affine_functional(bank,alpha,au,bu,assignment,h,c):
    rng=np.random.default_rng(7777003);n=c["functional_probe_count"];S=rng.uniform(-1,1,size=(n,3,2)).astype(np.float32);m=np.ones((n,3),np.float32);vprobe=base.sample_realizations(rng,m);world=base.load_world("e0a");mat=np.full((5,3),-1e6,np.float64);vinf=au[:,None]*alpha+bu[:,None]
    for j in range(5):
        for g in range(3):
            take=(assignment==j)&(h["family_gt"]==g)
            if take.sum()<2:continue
            x=h["train_amplitude_gt"][take].reshape(-1);y=vinf[take].reshape(-1);xm=float(x.mean());ym=float(y.mean());slope=float(((x-xm)*(y-ym)).sum()/max(float(((x-xm)**2).sum()),1e-12));intercept=ym-slope*xm;parts=[]
            with torch.no_grad():
                for st in range(0,n,512):
                    en=min(n,st+512);vv=np.zeros((en-st,5),np.float32);vv[:,j]=slope*vprobe[st:en,g]+intercept;raw=bank.raw_effects(torch.from_numpy(S[st:en]).to(DEVICE),torch.from_numpy(vv).to(DEVICE));parts.append(raw[:,j].cpu().numpy())
            pred=np.concatenate(parts);mg=np.zeros((n,3),np.float32);mg[:,g]=1;vg=np.zeros((n,3),np.float32);vg[:,g]=vprobe[:,g];target=base.compute_effects(S,mg,vg,world)[:,g];mat[j,g]=base.r2_score(target,pred)
    lr,gg=linear_sum_assignment(-np.nan_to_num(mat,nan=-1e6));mp={int(g):int(l) for l,g in zip(lr,gg)};return mat,[float(mat[mp[g],g]) for g in range(3)]

def stage_nc_family_affine(force=False):
    gate=stage_a1(False)
    if not gate["pass"]:raise RuntimeError("STOP: A1 failed")
    c=cfg();out=OUT/"negatives"/"family_affine";final=out/"final_metrics.json"
    if final.exists() and not force:return json.loads(final.read_text(encoding="utf-8"))
    if (out/"round_metrics.jsonl").exists():raise FileExistsError(f"Refusing overwrite {out}")
    out.mkdir(parents=True,exist_ok=True);dump(out/"config.json",c);fixed=derive_fixed_assignments(False);tr=response_data("train");va=response_data("val");hv=evaluator_data("val");atr,_=coordinates("train");av,avh=coordinates("val");base.seed_everything(0);bank=base.SetMechanismBank(5,c["mechanism_dim"],c["mechanism_heads"]).to(DEVICE);bank.load_state_dict(torch.load(ADP5_A1/"checkpoints"/"round_039_bank.pt",map_location=DEVICE,weights_only=True));au=torch.nn.Parameter(torch.ones(len(atr),device=DEVICE));bu=torch.nn.Parameter(torch.zeros(len(atr),device=DEVICE));opt=torch.optim.AdamW([{"params":bank.parameters(),"lr":c["mechanism_learning_rate"],"weight_decay":c["weight_decay"]},{"params":[au,bu],"lr":c["coordinate_learning_rate"],"weight_decay":0.0}]);started=time.time();rows=[]
    for r in range(c["rounds"]):
        tm=train_round_family_affine(bank,au,bu,tr["S"],tr["response_train"],atr,fixed["train"],c,opt,r);row={"round":r,"train_minibatch_mse":tm["minibatch_mse"],"train_family_a_mean":float(au.mean().detach().cpu()),"train_family_a_std":float(au.std().detach().cpu()),"train_family_b_mean":float(bu.mean().detach().cpu()),"train_family_b_std":float(bu.std().detach().cpu()),"m_step_seconds":tm["seconds"]};append(out/"round_metrics.jsonl",row);rows.append(row);print(f"[NC family_affine] r={r:02d} mse={row['train_minibatch_mse']:.6f} a_std={row['train_family_a_std']:.4f}",flush=True)
    vau,vbu=infer_family_affine(bank,va["S"],va["response_train"],av,fixed["val"]);vtrain=vau[:,None]*av+vbu[:,None];vheld=vau[:,None]*avh+vbu[:,None];val_pred=[];held_pred=[]
    with torch.no_grad():
        for alpha_v,outv in ((vtrain,val_pred),(vheld,held_pred)):
            F,R=alpha_v.shape;St=torch.from_numpy(np.repeat(va["S"],R,axis=0)).to(DEVICE);ct=torch.from_numpy(np.repeat(fixed["val"],R).astype(np.int64)).to(DEVICE);vt=torch.from_numpy(alpha_v.reshape(-1).astype(np.float32)).to(DEVICE);vall=torch.zeros((F*R,5),device=DEVICE);vall.scatter_(1,ct[:,None],vt[:,None]);raw=bank.raw_effects(St,vall);outv.append(raw[torch.arange(F*R,device=DEVICE),ct].cpu().numpy().reshape(F,R,3,2))
    mat,matched=family_affine_functional(bank,av,vau,vbu,fixed["val"],hv,c);last={**rows[-1],"heldout_realization_nrmse":base.nrmse(va["response_heldout"],held_pred[0]),"observed_realization_nrmse":base.nrmse(va["response_train"],val_pred[0]),"functional_r2_matrix":mat.tolist(),"matched_functional_r2":matched,"val_family_a_mean":float(vau.mean()),"val_family_a_std":float(vau.std()),"val_family_b_mean":float(vbu.mean()),"val_family_b_std":float(vbu.std())};torch.save({"bank":bank.state_dict(),"train_family_a":au.detach().cpu(),"train_family_b":bu.detach().cpu()},out/"final.pt");result={**meta("adp6_nc_family_affine",c),"control":"family_specific_affine","validation_coordinate_inference":"40 Adam steps on observed realization responses with frozen bank","rounds_completed":len(rows),"last_round":last,"wall_seconds":time.time()-started};dump(final,result);return result

@torch.no_grad()
def family_e_step_structured(bank,a,b,S,d,alpha,batch_f=128):
    F,R=alpha.shape;costs=[];bank.eval();aa=a.detach();bb=b.detach()
    for start in range(0,F,batch_f):
        stop=min(F,start+batch_f);n=stop-start;St=torch.from_numpy(np.repeat(S[start:stop],R,axis=0)).to(DEVICE);at=torch.from_numpy(alpha[start:stop].reshape(-1)).to(DEVICE);vt=at[:,None]*aa[None,:]+bb[None,:];raw=bank.raw_effects(St,vt);dt=torch.from_numpy(d[start:stop].reshape(-1,3,2)).to(DEVICE);err=(raw-dt[:,None]).square().mean(dim=(2,3)).reshape(n,R,5).mean(dim=1);costs.append(err.cpu().numpy())
    cost=np.concatenate(costs);return cost.argmin(axis=1).astype(np.int64),cost

def stage_a2(force=False):
    gate=stage_a1(False)
    if not gate["pass"]:raise RuntimeError("STOP: A1 failed")
    c=cfg();out=OUT/"a2_joint_family_coordinate";final=out/"final_metrics.json"
    if final.exists() and not force:return json.loads(final.read_text(encoding="utf-8"))
    if (out/"round_metrics.jsonl").exists():raise FileExistsError(f"Refusing overwrite {out}")
    (out/"checkpoints").mkdir(parents=True,exist_ok=True);(out/"assignments").mkdir(parents=True,exist_ok=True);dump(out/"config.json",c);tr=response_data("train");va=response_data("val");hv=evaluator_data("val");atr,_=coordinates("train");av,avh=coordinates("val");fixed=derive_fixed_assignments(False);assignment=fixed["train"].copy()
    cp=torch.load(OUT/"a1_fixed_assignment_structured_v"/"checkpoints"/"round_039.pt",map_location=DEVICE,weights_only=True);bank=base.SetMechanismBank(5,c["mechanism_dim"],c["mechanism_heads"]).to(DEVICE);bank.load_state_dict(cp["bank"]);a=torch.nn.Parameter(cp["coordinate_a"].to(DEVICE));b=torch.nn.Parameter(cp["coordinate_b"].to(DEVICE));opt=torch.optim.AdamW([{"params":bank.parameters(),"lr":c["mechanism_learning_rate"],"weight_decay":c["weight_decay"]},{"params":[a,b],"lr":c["coordinate_learning_rate"],"weight_decay":0.0}]);rows=[];started=time.time();torch.cuda.reset_peak_memory_stats() if DEVICE.type=="cuda" else None
    for r in range(c["rounds"]):
        es=time.time();new_assignment,cost=family_e_step_structured(bank,a,b,tr["S"],tr["response_train"],atr);e_seconds=time.time()-es;change=float(np.mean(new_assignment!=assignment));assignment=new_assignment;tm=train_round(bank,a,b,tr["S"],tr["response_train"],atr,assignment,c,opt,r+1000);val_assignment,val_cost=family_e_step_structured(bank,a,b,va["S"],va["response_train"],av);train_pred,train_v=predict(bank,tr["S"],assignment,atr,a,b);held_pred,held_v=predict(bank,va["S"],val_assignment,avh,a,b);val_pred,val_v=predict(bank,va["S"],val_assignment,av,a,b);cm=coordinate_metrics(av,val_v,val_assignment,hv);fm=functional_eval(bank,a,b,av,val_assignment,hv,c);cont,mp,acc,frag,purity=fixed_structure(val_assignment,hv)
        row={"round":r,"train_response_mse":float(np.mean((train_pred-tr["response_train"])**2)),"heldout_realization_nrmse":base.nrmse(va["response_heldout"],held_pred),"candidate_a":a.detach().cpu().tolist(),"candidate_b":b.detach().cpu().tolist(),**cm,**fm,"family_hungarian_accuracy":acc,"mean_GT_fragmentation":float(np.mean(frag)),"family_assignment_change":change,"candidate_usage":(np.bincount(assignment,minlength=5)/len(assignment)).tolist(),"candidate_family_count":np.bincount(assignment,minlength=5).tolist(),"e_step_seconds":e_seconds,"m_step_seconds":tm["seconds"],"round_seconds":e_seconds+tm["seconds"],"cuda_peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20 if DEVICE.type=="cuda" else 0.0};append(out/"round_metrics.jsonl",row);rows.append(row);np.savez_compressed(out/"assignments"/f"round_{r:03d}.npz",train=assignment,val=val_assignment,train_cost=cost,val_cost=val_cost);torch.save({"bank":bank.state_dict(),"coordinate_a":a.detach().cpu(),"coordinate_b":b.detach().cpu()},out/"checkpoints"/f"round_{r:03d}.pt");print(f"[A2] r={r:02d} change={change:.3f} acc={acc:.3f} frag={np.mean(frag):.3f} nrmse={row['heldout_realization_nrmse']:.4f} func={np.round(fm['matched_functional_r2'],3).tolist()} usage={np.round(row['candidate_usage'],3).tolist()}",flush=True)
    last=rows[-1];last3=float(np.mean([x["family_assignment_change"] for x in rows[-3:]]));aff=[x for x in last["candidate_affine_aligned_realization_r2"] if x is not None];checks={"family_accuracy":last["family_hungarian_accuracy"]>.85,"fragmentation":last["mean_GT_fragmentation"]<.20,"stable":last3<.15,"functional_all":min(last["matched_functional_r2"])>.75,"functional_two":sum(x>.90 for x in last["matched_functional_r2"])>=2,"affine":min(aff)>.90,"order":last["median_order_accuracy"]>.95};result={**meta("adp6_a2",c),"initial_checkpoint":str(OUT/"a1_fixed_assignment_structured_v"/"checkpoints"/"round_039.pt"),"rounds_completed":len(rows),"last_round":last,"last3_assignment_change":last3,"checks":checks,"pass":bool(all(checks.values())),"stop_required":not all(checks.values()),"wall_seconds":time.time()-started,"cuda_peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20 if DEVICE.type=="cuda" else 0.0};dump(final,result);(out/"summary.md").write_text(f"# ADP6-A2\n\n- PASS: **{result['pass']}**\n- checks: {checks}\n- heldout NRMSE: {last['heldout_realization_nrmse']}\n- functional R2: {last['matched_functional_r2']}\n",encoding="utf-8");return result

def main():
    stages=["adp6_0","adp6_a0","adp6_a1","adp6_nc_shuffle_alpha","adp6_nc_family_flip","adp6_nc_family_affine","adp6_nc_free_v","adp6_a2"]
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=stages,required=True);p.add_argument("--seed",type=int,default=0);p.add_argument("--force",action="store_true");x=p.parse_args()
    if x.seed!=0:raise ValueError("development seed fixed to 0")
    funcs={"adp6_0":stage_0,"adp6_a0":stage_a0,"adp6_a1":stage_a1,"adp6_nc_shuffle_alpha":lambda f:run_fixed_shared_negative("shuffle_alpha",f),"adp6_nc_family_flip":lambda f:run_fixed_shared_negative("family_flip",f),"adp6_nc_family_affine":stage_nc_family_affine,"adp6_nc_free_v":stage_a0,"adp6_a2":stage_a2};r=funcs[x.stage](x.force);print(json.dumps({"stage":r["stage"],"pass":r.get("pass",r.get("pass_as_adp6_structured_coordinate"))},indent=2),flush=True)
if __name__=="__main__":main()
