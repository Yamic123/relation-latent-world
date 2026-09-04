"""E0-ADP5 stages 0/A0/A1: response families as identity evidence."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import run_e0 as base
from adp3.run_adp3 import functional_matrix as probe_functional_matrix

CONFIG=ROOT/"configs"/"e0_adp5_8gb.json"
OUT=ROOT/"outputs"/"e0_adp5"
FAMILY_DATA=OUT/"a0_response_families"/"data"
DEVICE=base.DEVICE

def cfg(): return json.loads(CONFIG.read_text(encoding="utf-8"))
def jd(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    if isinstance(x,torch.Tensor): return x.detach().cpu().tolist()
    raise TypeError(type(x))
def dump(p,x): p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=jd),encoding="utf-8")
def append(p,x): p.parent.mkdir(parents=True,exist_ok=True);f=p.open("a",encoding="utf-8");f.write(json.dumps(x,ensure_ascii=False,default=jd)+"\n");f.close()
def meta(stage,c):
    x=base.environment_metadata();x.update({"stage":stage,"world_seed":base.WORLD_SEED,"dataset_seed":base.DATASET_SEED,
        "family_seed":c["family_seed"],"optimization_seed":c["optimization_seed"],"config":c,
        "config_sha256":base.sha256_file(CONFIG),"script_sha256":base.sha256_file(Path(__file__)),"git_commit":base.git_commit()});return x

def select_families(n,c,split_id):
    """Generator-only balanced random subset selection with random local order."""
    for attempt in range(1000):
        rng=np.random.default_rng(c["family_seed"]+split_id*10007+attempt*1000003)
        K=rng.choice([1,2],size=n,p=[c["k_probability"]["1"],c["k_probability"]["2"]])
        selected=[]
        for k in K:
            labels=rng.choice(3,size=int(k),replace=False);rng.shuffle(labels);selected.append(labels.astype(np.int8))
        counts=np.bincount(np.concatenate(selected),minlength=3)
        if counts.max()/counts.min()<1.15: return rng,K,selected,attempt
    raise RuntimeError("Could not generate balanced family coverage")

def generate_split(split,n,c):
    sid={"train":0,"val":1,"test":2}[split];rng,K,selected,attempt=select_families(n,c,sid)
    world=base.load_world("e0a");Sbase=rng.uniform(-1,1,size=(n,3,2)).astype(np.float32)
    train_v=np.asarray(c["train_active_values"],np.float32);held_v=np.asarray(c["heldout_active_values"],np.float32)
    S=[];base_ids=[];local_ids=[];actions_train=[];actions_held=[];y=[]
    for g,labels in enumerate(selected):
        for local,label in enumerate(labels):
            theta=rng.uniform(-np.pi,np.pi);direction=np.asarray([np.cos(theta),np.sin(theta)],np.float32)
            S.append(Sbase[g]);base_ids.append(g);local_ids.append(local);y.append(int(label))
            actions_train.append(train_v[:,None]*direction[None]);actions_held.append(held_v[:,None]*direction[None])
    S=np.asarray(S,np.float32);y=np.asarray(y,np.int8);F=len(S)
    def effects(values):
        R=len(values);Sr=np.repeat(S,R,axis=0);m=np.zeros((F*R,3),np.float32);v=np.zeros((F*R,3),np.float32)
        labels=np.repeat(y,R);m[np.arange(F*R),labels]=1;v[np.arange(F*R),labels]=np.tile(values,F)
        return base.compute_effects(Sr,m,v,world)[np.arange(F*R),labels].reshape(F,R,3,2)
    dtrain=effects(train_v);dheld=effects(held_v)
    visible={"S":S,"base_id":np.asarray(base_ids,np.int32),"local_family_id":np.asarray(local_ids,np.int8),
             "action_train":np.asarray(actions_train,np.float32),"response_train":dtrain,
             "action_heldout":np.asarray(actions_held,np.float32),"response_heldout":dheld,
             "baseline_response":np.zeros((F,3,2),np.float32)}
    hidden={"family_gt":y,"train_amplitude_gt":np.repeat(train_v[None],F,axis=0),"heldout_amplitude_gt":np.repeat(held_v[None],F,axis=0)}
    return visible,hidden,{"K":K,"selected":selected,"balance_attempt":attempt}

def stage_0(force=False):
    c=cfg();path=OUT/"adp5_0"/"final_metrics.json"
    if path.exists() and not force:return json.loads(path.read_text(encoding="utf-8"))
    FAMILY_DATA.mkdir(parents=True,exist_ok=True);hidden_all={};audits={};all_pass=True
    forbidden={"gt","amplitude","mechanism","m_real","v_gt","m_gt"}
    for sid,(split,n) in enumerate(c["base_counts"].items()):
        visible,hidden,raw=generate_split(split,n,c);np.savez_compressed(FAMILY_DATA/f"visible_{split}.npz",**visible)
        for k,v in hidden.items():hidden_all[f"{split}_{k}"]=v
        K=raw["K"];counts=np.bincount(hidden["family_gt"],minlength=3);ratio=float(counts.max()/counts.min())
        keys=list(visible);schema_clean=not any(any(token in key.lower() for token in forbidden) for key in keys)
        local0=hidden["family_gt"][visible["local_family_id"]==0];local0counts=np.bincount(local0,minlength=3);local0ratio=float(local0counts.max()/local0counts.min())
        checks={"K_only_1_or_2":bool(np.all((K==1)|(K==2))),"no_K3":bool(not np.any(K==3)),"coverage_ratio_lt_1_15":ratio<1.15,
                "random_local_label_ratio_lt_1_30":local0ratio<1.30,"learner_schema_clean":schema_clean}
        audits[split]={"base_count":int(n),"family_count":int(len(hidden["family_gt"])),"K_histogram":{"1":int(np.sum(K==1)),"2":int(np.sum(K==2)),"3":int(np.sum(K==3))},
                       "GT_family_counts_evaluator_only":counts.tolist(),"max_min_GT_count_ratio":ratio,"local0_GT_counts_evaluator_only":local0counts.tolist(),
                       "local0_max_min_ratio":local0ratio,"members_per_family":{"train_active":6,"heldout_active":6,"baseline":1},"learner_visible_fields":keys,"checks":checks,"pass":bool(all(checks.values()))}
        all_pass &= all(checks.values())
    np.savez_compressed(FAMILY_DATA/"hidden_evaluator_only.npz",**hidden_all)
    result={**meta("adp5_0",c),"audits":audits,"pass":bool(all_pass)};dump(path,result);dump(path.parent/"config.json",c)
    (path.parent/"summary.md").write_text(f"# ADP5-0\n\n- PASS: **{all_pass}**\n- audits: `{json.dumps(audits)}`\n",encoding="utf-8")
    print(json.dumps({"audits":audits,"pass":all_pass},indent=2),flush=True);return result

def visible(split):
    with np.load(FAMILY_DATA/f"visible_{split}.npz") as z:return {k:z[k] for k in z.files}
def hidden(split):
    with np.load(FAMILY_DATA/"hidden_evaluator_only.npz") as z:
        p=split+"_";return {k[len(p):]:z[k] for k in z.files if k.startswith(p)}

def stage_a0(force=False):
    c=cfg();gate=stage_0(False)
    if not gate["pass"]:raise RuntimeError("STOP: ADP5-0 failed")
    path=OUT/"a0_response_families"/"final_metrics.json"
    if path.exists() and not force:return json.loads(path.read_text(encoding="utf-8"))
    rows={};passed=True
    for split in c["base_counts"]:
        h=hidden(split);tv=h["train_amplitude_gt"];hv=h["heldout_amplitude_gt"]
        both=np.any(tv<0,axis=1)&np.any(tv>0,axis=1)&np.any(hv<0,axis=1)&np.any(hv>0,axis=1)
        # low/mid/high regions are determined by the three distinct absolute values.
        mag=np.asarray([len(np.unique(np.abs(x)))>=3 for x in tv])&np.asarray([len(np.unique(np.abs(x)))>=3 for x in hv])
        row={"family_count":len(tv),"family_purity":1.0,"fraction_cover_both_signs":float(both.mean()),"fraction_cover_three_magnitude_regions":float(mag.mean()),"pass":bool(np.all(both)&np.all(mag))}
        rows[split]=row;passed &= row["pass"]
    result={**meta("adp5_a0",c),"splits":rows,"pass":bool(passed)};dump(path,result);dump(path.parent/"config.json",c)
    (path.parent/"summary.md").write_text(f"# ADP5-A0\n\n- PASS: **{passed}**\n- splits: `{json.dumps(rows)}`\n",encoding="utf-8")
    print(json.dumps({"splits":rows,"pass":passed},indent=2),flush=True);return result

def set_trainable(bank,yes):
    for p in bank.parameters():p.requires_grad_(yes)

def point_solve(bank,S,d,c,seed):
    """S,d are flattened response points; returns per-point/per-candidate solutions."""
    a=c["a1"];n=len(S);k=c["m_max"];R=a["inner_restarts"];err=np.empty((n,k),np.float32);vv=np.empty((n,k),np.float32);by_restart=[np.empty((n,k),np.float32) for _ in range(R)]
    set_trainable(bank,False);bank.eval();started=time.time()
    for start in range(0,n,a["point_chunk_size"]):
        stop=min(n,start+a["point_chunk_size"]);b=stop-start;gen=torch.Generator(device=DEVICE);gen.manual_seed(seed+start*1009)
        z=(.35*torch.randn((R,b,k),generator=gen,device=DEVICE)).requires_grad_(True);opt=torch.optim.Adam([z],lr=a["inner_learning_rate"])
        St=torch.from_numpy(S[start:stop]).to(DEVICE).repeat(R,1,1);dt=torch.from_numpy(d[start:stop]).to(DEVICE).repeat(R,1,1)
        for _ in range(a["inner_steps"]):
            opt.zero_grad(set_to_none=True);v=a["v_bound"]*torch.tanh(z.reshape(R*b,k));pred=bank.raw_effects(St,v);loss=(pred-dt[:,None]).square().mean(dim=(2,3));loss.sum().backward();opt.step()
        with torch.no_grad():
            v=a["v_bound"]*torch.tanh(z.reshape(R*b,k));pred=bank.raw_effects(St,v);e=(pred-dt[:,None]).square().mean(dim=(2,3)).reshape(R,b,k);vr=v.reshape(R,b,k)
            rr=e.argmin(dim=0);bi=torch.arange(b,device=DEVICE)[:,None];kj=torch.arange(k,device=DEVICE)[None,:];err[start:stop]=e[rr,bi,kj].cpu().numpy();vv[start:stop]=vr[rr,bi,kj].cpu().numpy()
            for r in range(R):by_restart[r][start:stop]=e[r].cpu().numpy()
    set_trainable(bank,True);return {"errors":err,"v_all":vv,"restart_errors":by_restart,"seconds":time.time()-started}

def family_e_step(bank,S,d,c,seed):
    F,R=d.shape[:2];Sf=np.repeat(S,R,axis=0);df=d.reshape(F*R,3,2);p=point_solve(bank,Sf,df,c,seed)
    costs=p["errors"].reshape(F,R,5).mean(axis=1);assignment=costs.argmin(axis=1).astype(np.int16);point_candidate=np.repeat(assignment,R)
    chosen_v=p["v_all"][np.arange(F*R),point_candidate]
    restart_assign=[x.reshape(F,R,5).mean(axis=1).argmin(axis=1) for x in p["restart_errors"]]
    agreement=float(np.mean(restart_assign[0]==restart_assign[1]))
    return {**p,"family_costs":costs,"assignment":assignment,"point_v":chosen_v,"family_exact_agreement":agreement,"S_flat":Sf,"d_flat":df}

def m_step(bank,S,d,assignment,point_v,c,opt,round_id):
    a=c["a1"];F,R=d.shape[:2];rng=np.random.default_rng(c["optimization_seed"]+50021*(round_id+1));losses=[];started=time.time();bank.train()
    for _ in range(a["m_step_epochs_per_round"]):
        order=rng.permutation(F)
        for start in range(0,F,a["m_step_family_batch_size"]):
            fi=order[start:start+a["m_step_family_batch_size"]];idx=(fi[:,None]*R+np.arange(R)[None,:]).reshape(-1);St=torch.from_numpy(np.repeat(S[fi],R,axis=0)).to(DEVICE);dt=torch.from_numpy(d[fi].reshape(-1,3,2)).to(DEVICE)
            cand=np.repeat(assignment[fi],R);ct=torch.from_numpy(cand.astype(np.int64)).to(DEVICE);vt=torch.from_numpy(point_v[idx]).to(DEVICE);va=torch.zeros((len(idx),5),device=DEVICE);va.scatter_(1,ct[:,None],vt[:,None])
            opt.zero_grad(set_to_none=True);raw=bank.raw_effects(St,va);pred=raw[torch.arange(len(idx),device=DEVICE),ct];loss=(pred-dt).square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(bank.parameters(),a["gradient_clip"]);opt.step();losses.append(float(loss.detach().cpu()))
    return {"seconds":time.time()-started,"mean_minibatch_mse":float(np.mean(losses))}

@torch.no_grad()
def predict(bank,S,assignment,point_v,R,batch=512):
    Sf=np.repeat(S,R,axis=0);cand=np.repeat(assignment,R);out=[];bank.eval()
    for start in range(0,len(Sf),batch):
        stop=min(len(Sf),start+batch);St=torch.from_numpy(Sf[start:stop]).to(DEVICE);ct=torch.from_numpy(cand[start:stop].astype(np.int64)).to(DEVICE);vt=torch.from_numpy(point_v[start:stop]).to(DEVICE);va=torch.zeros((stop-start,5),device=DEVICE);va.scatter_(1,ct[:,None],vt[:,None]);raw=bank.raw_effects(St,va);out.append(raw[torch.arange(stop-start,device=DEVICE),ct].cpu().numpy())
    return np.concatenate(out).reshape(len(S),R,3,2)

def distribution_split(assignment,h,vdata,ddata):
    y=h["family_gt"];cont=np.zeros((5,3),np.int64)
    for x,z in zip(assignment,y):cont[int(x),int(z)]+=1
    fragmentation=[]
    for k in range(3):
        p=cont[:,k]/max(cont[:,k].sum(),1);fragmentation.append(float(1-p.max()))
    # Every family has symmetric amplitudes. "Dominant evidence" is defined by
    # observable response norm on negative vs positive members, not by candidate fit.
    norms=np.sqrt(np.square(ddata).sum(axis=(2,3)));neg=vdata<0;pos=vdata>0
    neg_score=(norms*neg).sum(axis=1)/neg.sum(axis=1);pos_score=(norms*pos).sum(axis=1)/pos.sum(axis=1);negdom=neg_score>pos_score
    split=[];detail={}
    for k in range(3):
        a=assignment[(y==k)&negdom];b=assignment[(y==k)&(~negdom)]
        if len(a)==0 or len(b)==0: value=None;pa=pb=np.zeros(5)
        else:pa=np.bincount(a,minlength=5)/len(a);pb=np.bincount(b,minlength=5)/len(b);value=float(.5*np.abs(pa-pb).sum());split.append(value)
        detail[str(k)]={"negative_dominant_count":len(a),"positive_dominant_count":len(b),"negative_dominant_distribution":pa.tolist(),"positive_dominant_distribution":pb.tolist(),"split":value}
    # Magnitude nonlinearity score: large-|v| / small-|v| response norm, split at median per GT.
    abs_v=np.abs(vdata);small=abs_v<=np.quantile(abs_v,.34);large=abs_v>=np.quantile(abs_v,.66);score=(norms*large).sum(axis=1)/large.sum(axis=1)-(norms*small).sum(axis=1)/small.sum(axis=1)
    mag=[]
    for k in range(3):
        take=y==k;med=np.median(score[take]);lo=assignment[take&(score<=med)];hi=assignment[take&(score>med)];pl=np.bincount(lo,minlength=5)/max(len(lo),1);ph=np.bincount(hi,minlength=5)/max(len(hi),1);mag.append(float(.5*np.abs(pl-ph).sum()))
    return cont,fragmentation,split,detail,mag

def evaluate_assignment(func,assignment,h,vdata,ddata):
    cont,frag,split,detail,mag=distribution_split(assignment,h,vdata,ddata);lr,gc=linear_sum_assignment(-cont);cmap={int(l):int(g) for l,g in zip(lr,gc)};acc=float(np.mean([cmap.get(int(x),-1)==int(y) for x,y in zip(assignment,h["family_gt"])]))
    fl,fg=linear_sum_assignment(-np.nan_to_num(func,nan=-1e6));fmap={int(g):int(l) for l,g in zip(fl,fg)};matched=[float(func[fmap[g],g]) for g in range(3)]
    return {"family_contingency_rows_candidate_cols_gt":cont.tolist(),"assignment_candidate_to_gt":cmap,"family_hungarian_accuracy":acc,"functional_gt_to_candidate":fmap,"matched_functional_r2":matched,
            "GT_fragmentation_per_gt":frag,"mean_GT_fragmentation":float(np.mean(frag)),"sign_split_per_gt":split,"mean_sign_split":float(np.mean(split)) if split else None,"sign_dominance_detail":detail,
            "magnitude_split_per_gt":mag,"mean_magnitude_split":float(np.mean(mag))}

def coverage(S,assignment,pred):
    norms=np.sqrt(np.square(pred).sum(axis=(2,3)));out=[]
    for j in range(5):
        take=assignment==j;vals=norms[take].reshape(-1)
        out.append({"family_count":int(take.sum()),"state_min":S[take].min(axis=0).tolist() if take.any() else None,"state_max":S[take].max(axis=0).tolist() if take.any() else None,
                    "effect_norm_quantiles":np.quantile(vals,[0,.25,.5,.75,1]).tolist() if len(vals) else None})
    return out

def stage_a1(force=False):
    c=cfg();gate=stage_a0(False)
    if not gate["pass"]:raise RuntimeError("STOP: ADP5-A0 failed")
    out=OUT/"a1_family_discovery";final=out/"final_metrics.json"
    if final.exists() and not force:return json.loads(final.read_text(encoding="utf-8"))
    if (out/"round_metrics.jsonl").exists():raise FileExistsError(f"Refusing overwrite {out}")
    (out/"checkpoints").mkdir(parents=True,exist_ok=True);(out/"assignments").mkdir(parents=True,exist_ok=True);dump(out/"config.json",c)
    tr,va=visible("train"),visible("val");hv=hidden("val");base.seed_everything(c["optimization_seed"]);bank=base.SetMechanismBank(5,c["mechanism_dim"],c["mechanism_heads"]).to(DEVICE);opt=torch.optim.AdamW(bank.parameters(),lr=c["a1"]["m_step_learning_rate"],weight_decay=c["a1"]["m_step_weight_decay"])
    prev=None;rows=[];started=time.time();torch.cuda.reset_peak_memory_stats() if DEVICE.type=="cuda" else None
    for r in range(c["a1"]["rounds"]):
        te=family_e_step(bank,tr["S"],tr["response_train"],c,c["optimization_seed"]+r*1000003);change=1.0 if prev is None else float(np.mean(te["assignment"]!=prev));ms=m_step(bank,tr["S"],tr["response_train"],te["assignment"],te["point_v"],c,opt,r)
        train_pred=predict(bank,tr["S"],te["assignment"],te["point_v"],6);train_mse=float(np.mean((train_pred-tr["response_train"])**2))
        ve=family_e_step(bank,va["S"],va["response_train"],c,c["optimization_seed"]+70000001+r*1000003);he=family_e_step(bank,va["S"],va["response_heldout"],c,c["optimization_seed"]+90000001+r*1000003)
        point_cand=np.repeat(ve["assignment"],6);held_v=he["v_all"][np.arange(len(point_cand)),point_cand];held_pred=predict(bank,va["S"],ve["assignment"],held_v,6);held_nrmse=base.nrmse(va["response_heldout"],held_pred)
        flat_hidden={"y_gt":np.repeat(hv["family_gt"],6),"v_gt":hv["train_amplitude_gt"].reshape(-1)};func=probe_functional_matrix(bank,ve["v_all"],flat_hidden,c);ev=evaluate_assignment(func,ve["assignment"],hv,hv["train_amplitude_gt"],va["response_train"])
        counts=np.bincount(te["assignment"],minlength=5);cov=coverage(va["S"],ve["assignment"],held_pred);torch.save(bank.state_dict(),out/"checkpoints"/f"round_{r:03d}_bank.pt");np.savez_compressed(out/"assignments"/f"round_{r:03d}.npz",candidate=te["assignment"],point_v=te["point_v"],family_costs=te["family_costs"])
        row={"round":r,"train_family_mse":train_mse,"heldout_realization_family_nrmse":held_nrmse,"candidate_family_usage":(counts/len(te["assignment"])).tolist(),"candidate_family_count":counts.tolist(),"family_assignment_change_rate":change,"family_exact_agreement":te["family_exact_agreement"],
             "functional_r2_matrix":func.tolist(),**ev,"candidate_state_effect_coverage":cov,"e_step_seconds":te["seconds"]+ve["seconds"]+he["seconds"],"m_step_seconds":ms["seconds"],"round_seconds":te["seconds"]+ve["seconds"]+he["seconds"]+ms["seconds"],"cuda_peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20 if DEVICE.type=="cuda" else 0.0}
        append(out/"round_metrics.jsonl",row);rows.append(row);prev=te["assignment"].copy();print(f"[A1] r={r:02d} nrmse={held_nrmse:.4f} acc={ev['family_hungarian_accuracy']:.4f} func={np.round(ev['matched_functional_r2'],3).tolist()} frag={ev['mean_GT_fragmentation']:.3f} split={ev['mean_sign_split']} change={change:.3f} usage={np.round(row['candidate_family_usage'],3).tolist()}",flush=True)
    last=rows[-1];last3=float(np.mean([x["family_assignment_change_rate"] for x in rows[-3:]]));th=c["pass_thresholds"];checks={"family_accuracy":last["family_hungarian_accuracy"]>th["family_hungarian_accuracy"],"functional_each":all(x>th["matched_functional_r2_each"] for x in last["matched_functional_r2"]),"functional_two_strong":sum(x>.85 for x in last["matched_functional_r2"])>=th["num_matched_functional_r2_above_0_85"],"stability":last3<th["last3_family_assignment_change"],"fragmentation":last["mean_GT_fragmentation"]<th["mean_gt_fragmentation"]}
    result={**meta("adp5_a1",c),"rounds_completed":len(rows),"last3_family_assignment_change":last3,"last_round":last,"checks":checks,"pass":bool(all(checks.values())),"stop_required":not all(checks.values()),"wall_seconds":time.time()-started,"cuda_peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20 if DEVICE.type=="cuda" else 0.0};dump(final,result);(out/"summary.md").write_text(f"# ADP5-A1\n\n- PASS: **{result['pass']}**\n- accuracy: {last['family_hungarian_accuracy']}\n- functional R2: {last['matched_functional_r2']}\n- fragmentation: {last['mean_GT_fragmentation']}\n- sign split: {last['mean_sign_split']}\n- last3 change: {last3}\n",encoding="utf-8");print(json.dumps({"pass":result["pass"],"checks":checks},indent=2),flush=True);return result

def main():
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=["adp5_0","adp5_a0","adp5_a1"],required=True);p.add_argument("--seed",type=int,default=0);p.add_argument("--force",action="store_true");a=p.parse_args()
    if a.seed!=0:raise ValueError("Development stage fixes seed 0")
    r={"adp5_0":stage_0,"adp5_a0":stage_a0,"adp5_a1":stage_a1}[a.stage](a.force);print(json.dumps({"stage":r["stage"],"pass":r["pass"]},indent=2),flush=True)
if __name__=="__main__":main()
