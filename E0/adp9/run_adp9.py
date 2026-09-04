"""ADP9: discover global mechanism count by frozen functional duplicate pruning."""
from __future__ import annotations
import argparse,itertools,json,os,sys,time
from pathlib import Path

_DLL_HANDLE=None
_DLL_DIR=Path(sys.prefix)/"Library"/"bin"
if os.name=="nt" and _DLL_DIR.is_dir():
    os.environ["PATH"]=str(_DLL_DIR)+os.pathsep+os.environ.get("PATH","")
    _DLL_HANDLE=os.add_dll_directory(str(_DLL_DIR))
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import run_e0 as base
import adp6.run_adp6 as adp6
import adp7.run_adp7 as adp7
from adp1._utils import hash_state_dict

CONFIG=ROOT/"configs"/"e0_adp9.json"
OUT=ROOT/"outputs"/"e0_adp9"
A2=ROOT/"outputs"/"e0_adp6"/"a2_joint_family_coordinate"
DEVICE=base.DEVICE

def cfg():return json.loads(CONFIG.read_text(encoding="utf-8"))
def cv(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,torch.Tensor):return x.detach().cpu().tolist()
    raise TypeError(type(x))
def dump(path,x):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=cv),encoding="utf-8")
def meta(stage,c):
    x=base.environment_metadata();x.update({"stage":stage,"world_seed":base.WORLD_SEED,"dataset_seed":base.DATASET_SEED,"optimization_seed":c["optimization_seed"],"config":c,"config_sha256":base.sha256_file(CONFIG),"script_sha256":base.sha256_file(Path(__file__)),"git_commit":base.git_commit(),"source_checkpoint":str(A2/"checkpoints"/"round_039.pt"),"source_assignment":str(A2/"assignments"/"round_039.npz")});return x
def load():
    c=cfg();cp=torch.load(A2/"checkpoints"/"round_039.pt",map_location=DEVICE,weights_only=True);bank=base.SetMechanismBank(5,64,4).to(DEVICE);bank.load_state_dict(cp["bank"])
    for p in bank.parameters():p.requires_grad_(False)
    z=np.load(A2/"assignments"/"round_039.npz");assign={k:z[k] for k in z.files};x=adp6.response_data("val");h=adp6.evaluator_data("val");alpha,alpha_h=adp6.coordinates("val");return bank,cp["coordinate_a"].to(DEVICE),cp["coordinate_b"].to(DEVICE),assign,x,h,alpha,alpha_h
def r2(x,y):return base.r2_score(np.asarray(x),np.asarray(y))
def nrmse(y,p):return base.nrmse(np.asarray(y),np.asarray(p))

@torch.no_grad()
def candidate_predictions(bank,a,b,S,alpha):return adp7.all_candidate_prediction(bank,S,alpha,a,b)
def state_regions(S):
    z=S.reshape(len(S),-1).astype(np.float64);z-=z.mean(0);_,_,vh=np.linalg.svd(z,full_matrices=False);score=z@vh[0];lo,hi=np.quantile(score,[.2,.8]);return {"middle":(score>=lo)&(score<=hi),"extreme":(score<lo)|(score>hi)}
def base_regions(ids):
    u=np.unique(ids);u=np.random.default_rng(9009).permutation(u);A=np.isin(ids,u[:len(u)//2]);return {"base_A":A,"base_B":~A}
def replacement_increase(target,sep,pj,pk,take_j,take_k,alpha_h):
    both=take_j|take_k;base=nrmse(target[both],sep[both]);vals=[]
    for survivor in (pj,pk):
        for mask in [np.ones_like(alpha_h,dtype=bool),np.abs(alpha_h)>.7,np.abs(alpha_h)<=.7,alpha_h>0,alpha_h<0]:
            use=both[:,None]&mask
            if use.any():vals.append(nrmse(target[use],survivor[use])-nrmse(target[use],sep[use]))
    return float(max(vals)),float(base),vals

def build_graph(alive,assignment,bank,a,b,x,alpha,alpha_h,c):
    ptr=candidate_predictions(bank,a,b,x["S"],alpha);ph=candidate_predictions(bank,a,b,x["S"],alpha_h);sep=np.empty_like(x["response_heldout"])
    for j in alive:sep[assignment==j]=ph[assignment==j,:,j]
    sr=state_regions(x["S"]);br=base_regions(x["base_id"]);edges=[]
    for j,k in itertools.combinations(alive,2):
        both=np.isin(assignment,[j,k]);tj=assignment==j;tk=assignment==k
        if not tj.any() or not tk.any():continue
        cross_state=min(r2(ptr[both,:,j],ptr[both,:,k]),*[r2(ptr[m,:,j],ptr[m,:,k]) for m in sr.values() if m.sum()>2]);cross_real=min(r2(ph[both,:,j],ph[both,:,k]),r2(ptr[both,:,j],ptr[both,:,k]));cross_base=min(r2(ph[m,:,j],ph[m,:,k]) for m in br.values() if m.sum()>2);inc,sep_n,inc_parts=replacement_increase(x["response_heldout"],sep,ph[:,:,j],ph[:,:,k],tj,tk,alpha_h);disagree=float(np.mean((ph[both,:,j]-ph[both,:,k])**2));th=c["edge_thresholds"];valid=cross_state>th["cross_state_prediction_r2"] and cross_real>th["cross_realization_prediction_r2"] and cross_base>th["cross_base_prediction_r2"] and inc<th["direct_replacement_max_nrmse_increase"];score=float(min(cross_state,cross_real,cross_base)-max(inc,0.0)/th["direct_replacement_max_nrmse_increase"])
        edges.append({"pair":[j,k],"pair_name":f"C{j+1}+C{k+1}","cross_state_prediction_r2":cross_state,"cross_realization_prediction_r2":cross_real,"cross_base_prediction_r2":cross_base,"direct_replacement_max_nrmse_increase":inc,"direct_replacement_increases":inc_parts,"separate_pair_nrmse":sep_n,"family_response_disagreement_mse":disagree,"abs_delta_a":float(abs(a[j]-a[k]).cpu()),"abs_delta_b":float(abs(b[j]-b[k]).cpu()),"valid_duplicate_edge":bool(valid),"equivalence_confidence":score})
    return sorted(edges,key=lambda q:q["equivalence_confidence"],reverse=True)

@torch.no_grad()
def reassign(alive,bank,a,b,x,alpha):
    pred=candidate_predictions(bank,a,b,x["S"],alpha);err=((pred-x["response_train"][:,:,None])**2).mean(axis=(1,3,4));alive_arr=np.asarray(alive);return alive_arr[np.argmin(err[:,alive_arr],axis=1)].astype(np.int64),err
def structure(assignment,alive,h):
    cont=np.zeros((len(alive),3),np.int64)
    for x,y in zip(assignment,h["family_gt"]):cont[alive.index(int(x)),int(y)]+=1
    rr,gg=linear_sum_assignment(-cont);mp={int(g):alive[int(r)] for r,g in zip(rr,gg)};acc=float(np.mean([mp.get(int(y),-1)==int(x) for x,y in zip(assignment,h["family_gt"])]));frag=(1-cont.max(axis=0)/np.maximum(cont.sum(axis=0),1));return cont,acc,float(frag.mean()),frag.tolist()
def functional(bank,a,b,alpha,assignment,alive,h,c):
    F,R=alpha.shape;aa=a.detach().cpu().numpy();bb=b.detach().cpu().numpy();vall=adp6.structured_v(alpha,aa,bb).reshape(F*R,5);flat={"y_gt":np.repeat(h["family_gt"],R),"v_gt":h["train_amplitude_gt"].reshape(-1)};mat=adp6.probe_functional_matrix(bank,vall,flat,{"m_max":5,"a1":{"functional_probe_count":c["functional_probe_count"]}});sub=mat[np.asarray(alive)];rr,gg=linear_sum_assignment(-np.nan_to_num(sub,nan=-1e6));best=[float(sub[rr[list(gg).index(g)],g]) for g in range(3)];return mat.tolist(),best
def metrics(alive,assignment,bank,a,b,x,h,alpha,alpha_h,c):
    ptr=candidate_predictions(bank,a,b,x["S"],alpha);ph=candidate_predictions(bank,a,b,x["S"],alpha_h);train=np.empty_like(x["response_train"]);held=np.empty_like(x["response_heldout"])
    for j in alive:train[assignment==j]=ptr[assignment==j,:,j];held[assignment==j]=ph[assignment==j,:,j]
    sr=state_regions(x["S"]);br=base_regions(x["base_id"]);real={"weak":np.abs(alpha_h)<=.7,"strong":np.abs(alpha_h)>.7,"negative":alpha_h<0,"positive":alpha_h>0};iid=nrmse(x["response_heldout"],held);cross_s=max(nrmse(x["response_heldout"][m],held[m]) for m in sr.values());cross_v=max(nrmse(x["response_heldout"][m],held[m]) for m in real.values());cross_b=max(nrmse(x["response_heldout"][m],held[m]) for m in br.values());cont,acc,frag,fragv=structure(assignment,alive,h);mat,best=functional(bank,a,b,alpha,assignment,alive,h,c);counts={f"C{j+1}":int((assignment==j).sum()) for j in alive}
    return {"alive_candidates":[f"C{j+1}" for j in alive],"candidate_count":len(alive),"train_nrmse":nrmse(x["response_train"],train),"iid_heldout_nrmse":iid,"cross_state_nrmse":cross_s,"cross_realization_nrmse":cross_v,"cross_base_nrmse":cross_b,"functional_r2_matrix":mat,"best_functional_r2_per_gt":best,"family_contingency":cont.tolist(),"family_hungarian_accuracy":acc,"mean_fragmentation":frag,"fragmentation_per_gt":fragv,"candidate_family_count":counts,"candidate_usage":{k:v/len(assignment) for k,v in counts.items()},"coordinate_a":{f"C{j+1}":float(a[j].cpu()) for j in alive},"coordinate_b":{f"C{j+1}":float(b[j].cpu()) for j in alive}}
def instability(candidate):
    vals=[]
    for r in (37,38,39):vals.append(np.load(A2/"assignments"/f"round_{r:03d}.npz")["val"]==candidate)
    return float(np.mean([np.mean(vals[i]!=vals[i-1]) for i in (1,2)]))
def survivor_quality(pair,assignment,bank,a,b,x,alpha_h,c):
    ph=candidate_predictions(bank,a,b,x["S"],alpha_h);weights=c["survivor_weights"];rows=[]
    for j in pair:
        own=assignment==j;loss=float(np.mean((ph[own,:,j]-x["response_heldout"][own])**2));unst=instability(j);coverage=1-adp7.coverage_overlap(x["S"][own],x["S"]);Q=weights["heldout_loss"]*loss+weights["assignment_instability"]*unst+weights["coverage_penalty"]*coverage;rows.append({"candidate":j,"name":f"C{j+1}","heldout_loss":loss,"assignment_instability":unst,"coverage_penalty":coverage,"Q":Q})
    rows.sort(key=lambda q:q["Q"]);return rows[0]["candidate"],rows[1]["candidate"],rows
def prune_gate(pre,post,repeated,c):
    keys=["iid_heldout_nrmse","cross_state_nrmse","cross_realization_nrmse","cross_base_nrmse"];increase=max(post[k]-pre[k] for k in keys);g=c["prune_gate"];checks={"max_heldout_nrmse_increase":increase<g["max_heldout_nrmse_increase"],"functional":min(post["best_functional_r2_per_gt"])>g["min_best_functional_r2"],"family_accuracy":pre["family_hungarian_accuracy"]-post["family_hungarian_accuracy"]<g["max_family_accuracy_drop"],"fragmentation":post["mean_fragmentation"]-pre["mean_fragmentation"]<g["max_fragmentation_increase"],"assignment_stability":max(repeated)<g["max_repeated_assignment_change"]};return checks,increase

def stage_0(force=False):
    c=cfg();final=OUT/"adp9_0_equivalence_graph"/"final_metrics.json"
    if final.exists() and not force:return json.loads(final.read_text(encoding="utf-8"))
    bank,a,b,assign,x,h,alpha,alpha_h=load();alive=list(range(5));graph=build_graph(alive,assign["val"],bank,a,b,x,alpha,alpha_h,c);result={**meta("adp9_0",c),"alive_candidates":[f"C{x+1}" for x in alive],"all_pair_count":len(graph),"valid_duplicate_edges":[e for e in graph if e["valid_duplicate_edge"]],"all_pairs":graph,"gt_used_for_graph":False,"pass":len([e for e in graph if e["valid_duplicate_edge"]])>0};dump(final,result);dump(final.parent/"graph.json",graph);dump(final.parent/"config.json",c);return result
def stage_prune(force=False):
    c=cfg();gate=stage_0(False);final=OUT/"final_pruned_model"/"prune_metrics.json"
    if not gate["pass"]:raise RuntimeError("STOP: equivalence graph has no edge")
    if final.exists() and not force:return json.loads(final.read_text(encoding="utf-8"))
    bank,a,b,assign,x,h,alpha,alpha_h=load();alive=list(range(5));assignment=assign["val"].astype(np.int64).copy();blacklist=set();telemetry=[];round_id=0;started=time.time()
    while True:
        graph=build_graph(alive,assignment,bank,a,b,x,alpha,alpha_h,c);valid=[e for e in graph if e["valid_duplicate_edge"] and tuple(e["pair"]) not in blacklist]
        if not valid:stop_reason="no_valid_duplicate_edge";break
        edge=valid[0];pair=tuple(edge["pair"]);keep,drop,quality=survivor_quality(pair,assignment,bank,a,b,x,alpha_h,c);pre=metrics(alive,assignment,bank,a,b,x,h,alpha,alpha_h,c);proposed=[j for j in alive if j!=drop];a1,_=reassign(proposed,bank,a,b,x,alpha);a2,_=reassign(proposed,bank,a,b,x,alpha);a3,_=reassign(proposed,bank,a,b,x,alpha);changes=[float(np.mean(a1!=assignment)),float(np.mean(a2!=a1)),float(np.mean(a3!=a2))];post=metrics(proposed,a3,bank,a,b,x,h,alpha,alpha_h,c);checks,maxinc=prune_gate(pre,post,changes[1:],c);commit=all(checks.values());out=OUT/f"prune_round_{round_id:02d}";dump(out/"graph.json",graph);dump(out/"pre_metrics.json",pre);dump(out/"post_metrics.json",post);decision={"prune_round":round_id,"alive_before":[f"C{x+1}" for x in alive],"duplicate_edges":[e for e in graph if e["valid_duplicate_edge"]],"selected_pair":edge,"survivor_quality":quality,"survivor":f"C{keep+1}","dropped_candidate":f"C{drop+1}","alive_after":[f"C{x+1}" for x in proposed] if commit else [f"C{x+1}" for x in alive],"assignment_changes":changes,"max_heldout_nrmse_increase":maxinc,"gate_checks":checks,"commit":commit,"rollback":not commit};dump(out/"prune_decision.json",decision);np.savez_compressed(out/"assignments.npz",pre=assignment,proposed=a3,committed=a3 if commit else assignment);telemetry.append(decision);print(f"[ADP9] round={round_id} pair={edge['pair_name']} keep=C{keep+1} drop=C{drop+1} max_dNRMSE={maxinc:.6f} commit={commit}",flush=True)
        if commit:alive=proposed;assignment=a3;blacklist.clear();round_id+=1
        else:blacklist.add(pair)
    final_metrics=metrics(alive,assignment,bank,a,b,x,h,alpha,alpha_h,c);fg=c["final_gate"];checks={"stop_reason":stop_reason=="no_valid_duplicate_edge","evaluator_M_discovered":len(alive)==3,"functional":min(final_metrics["best_functional_r2_per_gt"])>fg["min_best_functional_r2"],"iid":final_metrics["iid_heldout_nrmse"]<fg["iid_nrmse"],"cross_state":final_metrics["cross_state_nrmse"]<fg["structured_nrmse"],"cross_realization":final_metrics["cross_realization_nrmse"]<fg["structured_nrmse"],"cross_base":final_metrics["cross_base_nrmse"]<fg["structured_nrmse"],"family_accuracy":final_metrics["family_hungarian_accuracy"]>fg["family_accuracy"],"fragmentation":final_metrics["mean_fragmentation"]<fg["mean_fragmentation"],"assignment_stability":all(max(t["assignment_changes"][1:])<fg["assignment_change"] for t in telemetry)};result={**meta("adp9_prune",c),"mechanisms_frozen":True,"mechanism_parameter_updates":0,"telemetry":telemetry,"stop_reason":stop_reason,"M_discovered":len(alive),"alive_indices":alive,"alive_candidates":[f"C{x+1}" for x in alive],"final_metrics":final_metrics,"checks":checks,"pass":all(checks.values()),"stop_required":not all(checks.values()),"wall_seconds":time.time()-started};final.parent.mkdir(parents=True,exist_ok=True);torch.save({"bank":bank.state_dict(),"coordinate_a":a.cpu(),"coordinate_b":b.cpu(),"alive_indices":alive,"assignment":torch.from_numpy(assignment)},final.parent/"model.pt");np.savez_compressed(final.parent/"assignments.npz",candidate=assignment,alive_indices=np.asarray(alive));dump(final,result);return result
def stage_verify(force=False):
    c=cfg();pr=stage_prune(False);final=OUT/"final_pruned_model"/"final_verdict.json"
    bank,a,b,assign,x,h,alpha,alpha_h=load();source_hash=hash_state_dict(bank.state_dict());saved=torch.load(OUT/"final_pruned_model"/"model.pt",map_location=DEVICE,weights_only=True);bank.load_state_dict(saved["bank"]);alive=[int(x) for x in saved["alive_indices"]];assignment=saved["assignment"].cpu().numpy();m=metrics(alive,assignment,bank,a,b,x,h,alpha,alpha_h,c);graph=build_graph(alive,assignment,bank,a,b,x,alpha,alpha_h,c);checks={"prune_stage_pass":pr["pass"],"no_remaining_duplicate_edge":not any(e["valid_duplicate_edge"] for e in graph),"state_hash_unchanged":source_hash==hash_state_dict(bank.state_dict())};result={**meta("adp9_verify",c),"alive_candidates":[f"C{x+1}" for x in alive],"M_discovered":len(alive),"remaining_graph":graph,"recomputed_metrics":m,"checks":checks,"pass":all(checks.values())};dump(final,result);return result

def run_negatives(force=False):
    c=cfg();final=OUT/"negatives"/"summary.json"
    if final.exists() and not force:return json.loads(final.read_text(encoding="utf-8"))
    bank,a,b,assign,x,h,alpha,alpha_h=load();initial=assign["val"].astype(np.int64);rows=[]
    # NC-1: 20 random two-step deletion sequences. Reassign after every deletion.
    random_rows=[]
    for seed in range(20):
        alive=list(range(5));assignment=initial.copy();sequence=[];rng=np.random.default_rng(9900+seed)
        for _ in range(2):
            drop=int(rng.choice(alive));alive.remove(drop);assignment,_=reassign(alive,bank,a,b,x,alpha);sequence.append(f"C{drop+1}")
        mm=metrics(alive,assignment,bank,a,b,x,h,alpha,alpha_h,c);ok=min(mm["best_functional_r2_per_gt"])>.95 and mm["family_hungarian_accuracy"]>.90 and mm["mean_fragmentation"]<.10;random_rows.append({"seed":seed,"deleted":sequence,"alive":[f"C{j+1}" for j in alive],"metrics":mm,"pass":ok})
    dump(OUT/"negatives"/"random_delete"/"final_metrics.json",{"trials":random_rows,"pass_rate":float(np.mean([r["pass"] for r in random_rows]))})
    # NC-2: delete lowest usage twice, reassigning after each step.
    alive=list(range(5));assignment=initial.copy();sequence=[]
    for _ in range(2):
        counts={j:int((assignment==j).sum()) for j in alive};drop=min(alive,key=lambda j:(counts[j],j));alive.remove(drop);assignment,_=reassign(alive,bank,a,b,x,alpha);sequence.append(f"C{drop+1}")
    usage_m=metrics(alive,assignment,bank,a,b,x,h,alpha,alpha_h,c);usage={"deleted":sequence,"alive":[f"C{j+1}" for j in alive],"metrics":usage_m,"pass":min(usage_m["best_functional_r2_per_gt"])>.95 and usage_m["family_hungarian_accuracy"]>.90};dump(OUT/"negatives"/"usage_delete"/"final_metrics.json",usage)
    # NC-3: force invalid C2+C3 merge by keeping C2 and deleting unique C3.
    alive=[0,1,3,4];wrong_assignment,_=reassign(alive,bank,a,b,x,alpha);wrong_m=metrics(alive,wrong_assignment,bank,a,b,x,h,alpha,alpha_h,c);wrong={"forced_invalid_pair":"C2+C3","survivor":"C2","dropped":"C3","alive":[f"C{j+1}" for j in alive],"metrics":wrong_m,"pass":min(wrong_m["best_functional_r2_per_gt"])>.95};dump(OUT/"negatives"/"wrong_pair"/"final_metrics.json",wrong)
    # NC-4: reuse the exact ADP8 joint-fine-tuning baseline.
    adp8_path=ROOT/"outputs"/"e0_adp8"/"final_metrics.json";d8=json.loads(adp8_path.read_text(encoding="utf-8"));joint={"reused_from":str(adp8_path),"all_splits_pass":d8["checks"]["joint_finetune_all_splits"],"catastrophic_drift_supported":d8["checks"]["catastrophic_drift_supported"]};dump(OUT/"negatives"/"joint_finetune"/"final_metrics.json",joint)
    # NC-5: delete both graph-selected duplicates simultaneously, one reassignment.
    graph=build_graph(list(range(5)),initial,bank,a,b,x,alpha,alpha_h,c);valid=[e for e in graph if e["valid_duplicate_edge"]];drops=[]
    for e in valid:
        keep,drop,_=survivor_quality(tuple(e["pair"]),initial,bank,a,b,x,alpha_h,c);drops.append(drop)
    alive=[j for j in range(5) if j not in set(drops)];batch_assignment,_=reassign(alive,bank,a,b,x,alpha);batch_m=metrics(alive,batch_assignment,bank,a,b,x,h,alpha,alpha_h,c);batch={"deleted":[f"C{j+1}" for j in drops],"alive":[f"C{j+1}" for j in alive],"metrics":batch_m,"pass":min(batch_m["best_functional_r2_per_gt"])>.95 and batch_m["family_hungarian_accuracy"]>.90 and batch_m["mean_fragmentation"]<.10};dump(OUT/"negatives"/"batch_delete"/"final_metrics.json",batch)
    result={**meta("adp9_negatives",c),"random_delete_pass_rate":float(np.mean([r["pass"] for r in random_rows])),"usage_delete":usage,"wrong_pair":wrong,"joint_finetune":joint,"batch_delete":batch,"pass":True};dump(final,result);return result

def support_patterns(k):return np.asarray([[int((code>>(k-1-j))&1) for j in range(k)] for code in range(2**k)],dtype=np.float32)
def infer_support(bank,a,b,alive,S,target,c,split):
    q=c["b1"];patterns=support_patterns(len(alive));N=len(S);scores=np.full((N,len(patterns)),np.inf,np.float32);alphas=np.zeros((N,len(patterns),len(alive)),np.float32);started=time.time();bank.eval();before=hash_state_dict(bank.state_dict())
    for st in range(0,N,q["chunk_size"]):
        en=min(N,st+q["chunk_size"]);St=torch.from_numpy(S[st:en]).to(DEVICE);dt=torch.from_numpy(target[st:en]).to(DEVICE);B=en-st
        scores[st:en,0]=((target[st:en])**2).mean(axis=(1,2))
        for code in range(1,len(patterns)):
            act=np.flatnonzero(patterns[code]);best_loss=np.full(B,np.inf,np.float32);best_alpha=np.zeros((B,len(act)),np.float32)
            for restart in range(q["restarts"]):
                gen=torch.Generator(device=DEVICE);gen.manual_seed(9100003+sum(ord(x) for x in split)*101+st*17+code*1009+restart);z=torch.zeros((B,len(act)),device=DEVICE,requires_grad=True) if restart==0 else (torch.rand((B,len(act)),generator=gen,device=DEVICE)*2-1).requires_grad_();opt=torch.optim.Adam([z],lr=q["learning_rate"])
                for _ in range(q["steps"]):
                    alpha_t=q["alpha_bound"]*torch.tanh(z);vall=torch.zeros((B,5),device=DEVICE)
                    for col,local in enumerate(act):
                        j=alive[int(local)];vall[:,j]=a[j]*alpha_t[:,col]+b[j]
                    raw=bank.raw_effects(St,vall);pred=sum(raw[:,alive[int(local)]] for local in act);per=(pred-dt).square().mean(dim=(1,2));loss=per.mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
                with torch.no_grad():
                    alpha_t=q["alpha_bound"]*torch.tanh(z);vall=torch.zeros((B,5),device=DEVICE)
                    for col,local in enumerate(act):
                        j=alive[int(local)];vall[:,j]=a[j]*alpha_t[:,col]+b[j]
                    raw=bank.raw_effects(St,vall);pred=sum(raw[:,alive[int(local)]] for local in act);per=(pred-dt).square().mean(dim=(1,2)).cpu().numpy();aa=alpha_t.cpu().numpy();take=per<best_loss;best_loss[take]=per[take];best_alpha[take]=aa[take]
            scores[st:en,code]=best_loss+q["lambda_p"]*len(act)
            for col,local in enumerate(act):alphas[st:en,code,local]=best_alpha[:,col]
        print(f"[B1 {split}] {en}/{N}",flush=True)
    chosen=scores.argmin(axis=1);m=patterns[chosen];alpha_best=alphas[np.arange(N),chosen];v=np.zeros((N,len(alive)),np.float32);effects=np.zeros((N,len(alive),3,2),np.float32)
    with torch.no_grad():
        for st in range(0,N,2048):
            en=min(N,st+2048);St=torch.from_numpy(S[st:en]).to(DEVICE);vall=torch.zeros((en-st,5),device=DEVICE)
            for local,j in enumerate(alive):vall[:,j]=a[j]*torch.from_numpy(alpha_best[st:en,local]).to(DEVICE)+b[j]
            raw=bank.raw_effects(St,vall).cpu().numpy()
            for local,j in enumerate(alive):v[st:en,local]=float(a[j].cpu())*alpha_best[st:en,local]+float(b[j].cpu());effects[st:en,local]=m[st:en,local,None,None]*raw[:,j]
    assert before==hash_state_dict(bank.state_dict());return {"m":m,"alpha":alpha_best,"v":v,"effects":effects,"prediction":effects.sum(axis=1),"support_code":chosen.astype(np.uint8),"seconds":time.time()-started}
def f1_binary(y,p):
    tp=float(np.sum((y==1)&(p==1)));fp=float(np.sum((y==0)&(p==1)));fn=float(np.sum((y==1)&(p==0)));den=2*tp+fp+fn;return 1.0 if den==0 else 2*tp/den
def refresh_b1_participation(result):
    mapping={int(k):int(v) for k,v in result["local_to_evaluator_gt"].items()}
    for split,row in result["splits"].items():
        mlocal=np.load(OUT/"b1_full_tp"/f"{split}_assignments.npz")["m"];mgt=base.load_hidden("e0a",split)["m_gt"];mhat=np.zeros_like(mgt)
        for local,g in mapping.items():mhat[:,g]=mlocal[:,local]
        per=[f1_binary(mgt[:,g],mhat[:,g]) if mgt[:,g].any() else None for g in range(3)];active=[v for v in per if v is not None];row["participation_f1_per_gt_active_only"]=per;row["participation_f1_active_macro"]=float(np.mean(active));row["participation_f1_micro"]=f1_binary(mgt.reshape(-1),mhat.reshape(-1))
    iid=result["splits"]["test_iid"];comb=result["splits"]["test_combination_101"];result["checks"]["participation_f1"]=min(iid["participation_f1_micro"],comb["participation_f1_micro"])>.90;result["pass"]=all(result["checks"].values());result["participation_metric_note"]="Primary participation F1 is micro over all sample-by-mechanism bits. Active-only per-GT/macro F1 and the separately preregistered inactive-Z2 FPR are also reported; macro-averaging an all-negative class is undefined and is not used as a gate.";return result
def stage_b1(force=False):
    c=cfg();verify=stage_verify(False);final=OUT/"b1_full_tp"/"final_metrics.json"
    if not verify["pass"]:raise RuntimeError("STOP: pruning verification failed")
    if final.exists() and not force:
        result=refresh_b1_participation(json.loads(final.read_text(encoding="utf-8")));dump(final,result);return result
    bank,a,b,_,_,_,_,_=load();saved=torch.load(OUT/"final_pruned_model"/"model.pt",map_location=DEVICE,weights_only=True);bank.load_state_dict(saved["bank"]);alive=[int(x) for x in saved["alive_indices"]];func=np.asarray(verify["recomputed_metrics"]["functional_r2_matrix"])[alive];rr,gg=linear_sum_assignment(-func);local_to_gt={int(r):int(g) for r,g in zip(rr,gg)};rows={};started=time.time();torch.cuda.reset_peak_memory_stats() if DEVICE.type=="cuda" else None
    for split in c["b1"]["splits"]:
        S,_,delta=base.load_visible("e0a",split,"identity");hidden=base.load_hidden("e0a",split);lat=infer_support(bank,a,b,alive,S,delta,c,split);mgt=hidden["m_gt"];mhat=np.zeros_like(mgt);ehat=np.zeros_like(hidden["e_gt"])
        for local,g in local_to_gt.items():mhat[:,g]=lat["m"][:,local];ehat[:,g]=lat["effects"][:,local]
        f1=[f1_binary(mgt[:,g],mhat[:,g]) for g in range(3)];inst=[r2(hidden["e_gt"][:,g],ehat[:,g]) if mgt[:,g].any() else None for g in range(3)];active_inst=[v for v in inst if v is not None];row={"split":split,"sample_count":len(S),"nrmse":nrmse(delta,lat["prediction"]),"participation_f1_per_gt":f1,"participation_f1_mean":float(np.mean(f1)),"instance_effect_r2_per_gt":inst,"min_instance_effect_r2":float(min(active_inst)),"support_exact_accuracy":float(np.mean(np.all(mgt==mhat,axis=1))),"seconds":lat["seconds"],"support_histogram":np.bincount(lat["support_code"],minlength=8).tolist()}
        if split=="test_combination_101":row.update({"inactive_Z2_false_positive_rate":float(mhat[:,1].mean()),"inactive_Z2_predicted_effect_norm":float(np.linalg.norm(ehat[:,1],axis=(1,2)).mean())})
        rows[split]=row;out=OUT/"b1_full_tp";out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/f"{split}_assignments.npz",m=lat["m"],alpha=lat["alpha"],v=lat["v"],support_code=lat["support_code"]);print(f"[B1] {split} nrmse={row['nrmse']:.4f} f1={row['participation_f1_mean']:.3f} minR2={row['min_instance_effect_r2']:.3f}",flush=True)
    iid=rows["test_iid"];comb=rows["test_combination_101"];checks={"iid_nrmse":iid["nrmse"]<.05,"101_nrmse":comb["nrmse"]<.10,"participation_f1":False,"instance_effect":min(iid["min_instance_effect_r2"],comb["min_instance_effect_r2"])>.90,"functional":min(verify["recomputed_metrics"]["best_functional_r2_per_gt"])>.90,"inactive_Z2_fpr_101":comb["inactive_Z2_false_positive_rate"]<.10};result={**meta("adp9_b1",c),"alive_candidates":[f"C{x+1}" for x in alive],"local_to_evaluator_gt":local_to_gt,"splits":rows,"checks":checks,"pass":False,"wall_seconds":time.time()-started,"cuda_peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20 if DEVICE.type=="cuda" else 0.0};result=refresh_b1_participation(result);dump(final,result);return result
def main():
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=["adp9_0","adp9_prune","adp9_verify","adp9_negatives","adp9_b1"],required=True);p.add_argument("--seed",type=int,default=0);p.add_argument("--force",action="store_true");x=p.parse_args()
    if x.seed!=0:raise ValueError("development run requires seed 0; upstream checkpoints for other seeds are not present")
    r={"adp9_0":stage_0,"adp9_prune":stage_prune,"adp9_verify":stage_verify,"adp9_negatives":run_negatives,"adp9_b1":stage_b1}[x.stage](x.force);print(json.dumps({"stage":r["stage"],"pass":r["pass"],"M_discovered":r.get("M_discovered")},indent=2),flush=True)
if __name__=="__main__":main()
