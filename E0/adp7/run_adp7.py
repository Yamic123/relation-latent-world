"""ADP7: single-capacity functional merge under structured holdouts."""
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

CONFIG=ROOT/"configs"/"e0_adp7_8gb.json"
OUT=ROOT/"outputs"/"e0_adp7"
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
def pair_name(pair):return f"C{pair[0]+1}_C{pair[1]+1}"
def load_source():
    c=cfg();cp=torch.load(A2/"checkpoints"/"round_039.pt",map_location=DEVICE,weights_only=True);bank=base.SetMechanismBank(5,c["mechanism_dim"],c["mechanism_heads"]).to(DEVICE);bank.load_state_dict(cp["bank"]);z=np.load(A2/"assignments"/"round_039.npz");return bank,cp["coordinate_a"].to(DEVICE),cp["coordinate_b"].to(DEVICE),{k:z[k] for k in z.files}
def data():
    x=adp6.response_data("val");h=adp6.evaluator_data("val");alpha,alpha_h=adp6.coordinates("val");return x,h,alpha,alpha_h

@torch.no_grad()
def all_candidate_prediction(bank,S,alpha,a,b,batch=128):
    F,R=alpha.shape;parts=[];aa=a.detach();bb=b.detach();bank.eval()
    for st in range(0,F,batch):
        en=min(F,st+batch);St=torch.from_numpy(np.repeat(S[st:en],R,axis=0)).to(DEVICE);at=torch.from_numpy(alpha[st:en].reshape(-1)).to(DEVICE);v=at[:,None]*aa[None]+bb[None];parts.append(bank.raw_effects(St,v).cpu().numpy().reshape(en-st,R,5,3,2))
    return np.concatenate(parts)

def r2(y,p):return base.r2_score(np.asarray(y),np.asarray(p))
def cosine_flat(x,y):
    x=x.reshape(-1).astype(np.float64);y=y.reshape(-1).astype(np.float64);return float(x@y/max(np.linalg.norm(x)*np.linalg.norm(y),1e-12))
def coverage_overlap(S1,S2):
    # Symmetric nearest-neighbour overlap, normalized by pooled state scale.
    x=S1.reshape(len(S1),-1).astype(np.float64);y=S2.reshape(len(S2),-1).astype(np.float64);scale=max(float(np.sqrt(np.mean(np.var(np.concatenate([x,y]),axis=0)))),1e-8)
    d=((x[:,None]-y[None])**2).mean(axis=2)**.5/scale
    return float(np.exp(-.5*np.mean(np.r_[d.min(axis=1),d.min(axis=0)])))

def stage_0(force=False):
    c=cfg();final=OUT/"adp7_0_pair_audit"/"final_metrics.json"
    if final.exists() and not force:return json.loads(final.read_text(encoding="utf-8"))
    bank,a,b,assign=load_source();x,h,alpha,alpha_h=data();pred=all_candidate_prediction(bank,x["S"],alpha_h,a,b);rows=[]
    pairs=[tuple(p) for p in c["correct_pairs"]+c["wrong_pairs"]]
    for pair in pairs:
        j,k=pair;take_j=assign["val"]==j;take_k=assign["val"]==k;both=take_j|take_k
        sim_r2=r2(pred[both,:,j],pred[both,:,k]);sim_cos=cosine_flat(pred[both,:,j],pred[both,:,k]);own_j=pred[take_j,:,j];cross_j=pred[take_j,:,k];own_k=pred[take_k,:,k];cross_k=pred[take_k,:,j]
        gtj=np.unique(h["family_gt"][take_j]).astype(int).tolist();gtk=np.unique(h["family_gt"][take_k]).astype(int).tolist()
        rows.append({"pair":pair_name(pair),"indices_zero_based":[j,k],"family_counts":[int(take_j.sum()),int(take_k.sum())],"cross_state_prediction_r2":sim_r2,"cross_state_prediction_cosine":sim_cos,"heldout_family_cross_prediction_r2":[r2(own_j,cross_j),r2(own_k,cross_k)],"abs_delta_a":float(abs(a[j]-a[k]).cpu()),"abs_delta_b":float(abs(b[j]-b[k]).cpu()),"state_coverage_overlap":coverage_overlap(x["S"][take_j],x["S"][take_k]),"evaluator_gt_ids":[gtj,gtk],"evaluator_gt_identity_agreement":gtj==gtk})
    result={**meta("adp7_0",c),"training_performed":False,"pairs":rows,"suspected_pairs_confirmed":all(next(z for z in rows if z["pair"]==pair_name(tuple(p)))["evaluator_gt_identity_agreement"] for p in c["correct_pairs"]),"pass":True};dump(final,result);dump(final.parent/"config.json",c);return result

def make_single(source_bank,source,a0,b0,init,c):
    base.seed_everything(c["optimization_seed"]+(100 if init=="random" else 200)+source);one=base.SetMechanismBank(1,c["mechanism_dim"],c["mechanism_heads"]).to(DEVICE)
    if init=="source_copy":
        sd=source_bank.state_dict();od=one.state_dict()
        for key in od:
            od[key]=sd[key][source:source+1].clone() if key=="adapters" else sd[key].clone()
        one.load_state_dict(od);aa=torch.nn.Parameter(a0[source:source+1].detach().clone());bb=torch.nn.Parameter(b0[source:source+1].detach().clone())
    else:aa=torch.nn.Parameter(torch.ones(1,device=DEVICE));bb=torch.nn.Parameter(torch.zeros(1,device=DEVICE))
    return one,aa,bb

@torch.no_grad()
def merged_predict(model,a,b,S,alpha,batch=1024):
    out=[];sf=S.reshape(-1,3,2);af=alpha.reshape(-1);model.eval()
    for st in range(0,len(sf),batch):
        en=min(len(sf),st+batch);St=torch.from_numpy(sf[st:en]).to(DEVICE);at=torch.from_numpy(af[st:en]).to(DEVICE);v=(a[0]*at+b[0])[:,None];out.append(model.raw_effects(St,v)[:,0].cpu().numpy())
    return np.concatenate(out).reshape(*S.shape[:-2],3,2)

def fit_merged(source_bank,a0,b0,source,S,response,alpha,mask,init,c,tag):
    model,a,b=make_single(source_bank,source,a0,b0,init,c);idx=np.argwhere(mask);opt=torch.optim.AdamW([{"params":model.parameters(),"lr":c["mechanism_learning_rate"],"weight_decay":c["weight_decay"]},{"params":[a,b],"lr":c["coordinate_learning_rate"],"weight_decay":0.0}]);rng=np.random.default_rng(c["optimization_seed"]+sum(ord(q) for q in tag));losses=[];started=time.time();torch.cuda.reset_peak_memory_stats() if DEVICE.type=="cuda" else None
    for ep in range(c["epochs"]):
        order=rng.permutation(len(idx));ep_losses=[];model.train()
        for st in range(0,len(idx),c["response_batch_size"]):
            ij=idx[order[st:st+c["response_batch_size"]]];fi=ij[:,0];ri=ij[:,1];St=torch.from_numpy(S[fi]).to(DEVICE);at=torch.from_numpy(alpha[fi,ri]).to(DEVICE);dt=torch.from_numpy(response[fi,ri]).to(DEVICE);v=(a[0]*at+b[0])[:,None];opt.zero_grad(set_to_none=True);pred=model.raw_effects(St,v)[:,0];loss=(pred-dt).square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(list(model.parameters())+[a,b],c["gradient_clip"]);opt.step();ep_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(ep_losses)))
    return model,a,b,{"train_loss_by_epoch":losses,"wall_seconds":time.time()-started,"cuda_peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20 if DEVICE.type=="cuda" else 0.0}

def functional_scores(model,a,b,alpha,h,take,c):
    gt_ids=np.unique(h["family_gt"][take]).astype(int);world=base.load_world("e0a");rng=np.random.default_rng(7777003);n=c["functional_probe_count"];S=rng.uniform(-1,1,size=(n,3,2)).astype(np.float32);m=np.ones((n,3),np.float32);vp=base.sample_realizations(rng,m);scores={}
    vinf=(float(a.detach().cpu()[0])*alpha[take]+float(b.detach().cpu()[0])).reshape(-1);gtamp=h["train_amplitude_gt"][take].reshape(-1);gtflat=np.repeat(h["family_gt"][take],alpha.shape[1])
    for g in gt_ids:
        use=gtflat==g;x=gtamp[use];y=vinf[use];xm=float(x.mean());ym=float(y.mean());slope=float(((x-xm)*(y-ym)).sum()/max(float(((x-xm)**2).sum()),1e-12));inter=ym-slope*xm;pred=[]
        with torch.no_grad():
            for st in range(0,n,512):
                en=min(n,st+512);vv=(slope*vp[st:en,g]+inter).astype(np.float32)[:,None];pred.append(model.raw_effects(torch.from_numpy(S[st:en]).to(DEVICE),torch.from_numpy(vv).to(DEVICE))[:,0].cpu().numpy())
        mg=np.zeros((n,3),np.float32);mg[:,g]=1;vg=np.zeros((n,3),np.float32);vg[:,g]=vp[:,g];target=base.compute_effects(S,mg,vg,world)[:,g];scores[str(int(g))]=r2(target,np.concatenate(pred))
    return scores

def best_source(source_bank,a,b,S,response,alpha,assignment,pair,train_mask):
    allp=all_candidate_prediction(source_bank,S,alpha,a,b);cost=[]
    for j in pair:cost.append(float(np.mean((allp[:,:,j][train_mask]-response[train_mask])**2)))
    return int(pair[int(np.argmin(cost))]),cost

def evaluate_split(pair,spec,init,c):
    source_bank,a0,b0,assign=load_source();x,h,alpha,alpha_h=data();assignment=assign["val"];pair=np.asarray(pair);take=np.isin(assignment,pair);S=x["S"][take];response=x["response_train"][take];ah_response=x["response_heldout"][take];al=alpha[take];alh=alpha_h[take];ass=assignment[take];hh={k:v[take] for k,v in h.items()};train_mask,test_mask=split_masks(spec,x,take,al);source,cost=best_source(source_bank,a0,b0,S,response,al,ass,tuple(pair),train_mask);model,a,b,train_info=fit_merged(source_bank,a0,b0,source,S,response,al,train_mask,init,c,pair_name(tuple(pair))+spec["id"]+init);sep_all=all_candidate_prediction(source_bank,S,al,a0,b0);sep=np.empty_like(response)
    for j in pair:sep[ass==j]=sep_all[ass==j,:,j]
    mp=merged_predict(model,a,b,np.repeat(S[:,None],al.shape[1],axis=1),al);sep_mse=float(np.mean((sep[test_mask]-response[test_mask])**2));merge_mse=float(np.mean((mp[test_mask]-response[test_mask])**2));sep_n=base.nrmse(response[test_mask],sep[test_mask]);merge_n=base.nrmse(response[test_mask],mp[test_mask]);func=functional_scores(model,a,b,al,hh,np.ones(len(S),bool),c);v=(float(a.detach().cpu()[0])*al+float(b.detach().cpu()[0]));aff=[]
    for g in np.unique(hh["family_gt"]):
        q=hh["family_gt"]==g;aff.append(adp6.linear_r2(v[q].reshape(-1),hh["train_amplitude_gt"][q].reshape(-1)))
    order=float(np.median([adp6.pair_order(al[u],v[u]) for u in range(len(S))]));held=merged_predict(model,a,b,np.repeat(S[:,None],alh.shape[1],axis=1),alh)
    return {"pair":pair_name(tuple(pair)),"split":spec["id"],"initialization":init,"family_count":int(len(S)),"train_observations":int(train_mask.sum()),"test_observations":int(test_mask.sum()),"source_candidate":f"C{source+1}","source_train_costs":cost,"separate_test_mse":sep_mse,"merged_test_mse":merge_mse,"relative_iid_delta":(merge_mse-sep_mse)/(sep_mse+1e-12),"separate_test_nrmse":sep_n,"merged_test_nrmse":merge_n,"delta_nrmse":merge_n-sep_n,"heldout_realization_nrmse_all_families":base.nrmse(ah_response,held),"functional_r2_by_gt":func,"min_functional_r2":min(func.values()),"coordinate_a":float(a.detach().cpu()[0]),"coordinate_b":float(b.detach().cpu()[0]),"min_affine_realization_r2":float(min(aff)),"order_accuracy":order,"single_model_parameter_count":sum(p.numel() for p in model.parameters()),**train_info}

def split_masks(spec,x,take,alpha):
    F,R=alpha.shape;allobs=np.ones((F,R),bool);baseid=x["base_id"][take];S=x["S"][take]
    if spec["kind"] in ("iid","base"):
        ids=np.unique(baseid);rng=np.random.default_rng(7007);ids=rng.permutation(ids);cut=int(round(len(ids)*spec.get("fraction",.5)));A=np.isin(baseid,ids[:cut]);B=~A
        if spec.get("reverse",False):A,B=B,A
        return A[:,None]&allobs,B[:,None]&allobs
    if spec["kind"]=="state":
        z=S.reshape(F,-1).astype(np.float64);z-=z.mean(0);_,_,vh=np.linalg.svd(z,full_matrices=False);score=z@vh[0];lo,hi=np.quantile(score,[.2,.8]);middle=(score>=lo)&(score<=hi);extreme=~middle
        if spec.get("reverse",False):middle,extreme=extreme,middle
        return middle[:,None]&allobs,extreme[:,None]&allobs
    if spec["kind"]=="realization":
        mode=spec["mode"]
        if mode=="weak_strong":return np.abs(alpha)<=.70001,np.abs(alpha)>.70001
        if mode=="strong_weak":return np.abs(alpha)>=.69999,np.abs(alpha)<.69999
        if mode=="neg_pos":return alpha<0,alpha>0
        return alpha>0,alpha<0
    if spec["kind"]=="joint":
        ids=np.unique(baseid);rng=np.random.default_rng(7007);ids=rng.permutation(ids);A=np.isin(baseid,ids[:len(ids)//2]);B=~A;z=S.reshape(F,-1).astype(np.float64);z-=z.mean(0);_,_,vh=np.linalg.svd(z,full_matrices=False);score=z@vh[0];lo,hi=np.quantile(score,[.2,.8]);middle=(score>=lo)&(score<=hi);extreme=~middle;train=A[:,None]&middle[:,None]&(alpha<0)&(np.abs(alpha)<=.70001);test=B[:,None]&extreme[:,None]&(alpha>0)&(np.abs(alpha)>.70001)
        if spec.get("reverse",False):train,test=test,train
        return train,test
    raise ValueError(spec)

SPECS={
 "adp7_a0":[{"id":"iid_70_30","kind":"iid","fraction":.7}],
 "adp7_a1":[{"id":"middle_to_extreme","kind":"state"},{"id":"extreme_to_middle","kind":"state","reverse":True}],
 "adp7_a2":[{"id":"weak_to_strong","kind":"realization","mode":"weak_strong"},{"id":"strong_to_weak","kind":"realization","mode":"strong_weak"},{"id":"negative_to_positive","kind":"realization","mode":"neg_pos"},{"id":"positive_to_negative","kind":"realization","mode":"pos_neg"}],
 "adp7_a3":[{"id":"base_A_to_B","kind":"base","fraction":.5},{"id":"base_B_to_A","kind":"base","fraction":.5,"reverse":True}],
 "adp7_a4":[{"id":"joint_forward","kind":"joint"},{"id":"joint_reverse","kind":"joint","reverse":True}]
}
STAGE_DIR={"adp7_a0":"a0_iid_merge","adp7_a1":"a1_state_holdout","adp7_a2":"a2_realization_holdout","adp7_a3":"a3_base_holdout","adp7_a4":"a4_joint_holdout"}
def thresholds(stage,row):
    if stage=="adp7_a0":return row["relative_iid_delta"]<.05 and row["min_functional_r2"]>.95 and row["min_affine_realization_r2"]>.95
    if stage=="adp7_a1":return row["delta_nrmse"]<.03 and row["min_functional_r2"]>.90 and row["min_affine_realization_r2"]>.95
    if stage=="adp7_a2":return row["merged_test_nrmse"]<=row["separate_test_nrmse"]+.03 and row["min_functional_r2"]>.90 and row["order_accuracy"]>.95 and row["min_affine_realization_r2"]>.95
    if stage=="adp7_a3":return row["delta_nrmse"]<.03 and row["min_functional_r2"]>.90 and row["heldout_realization_nrmse_all_families"]<.05
    return row["min_functional_r2"]>.85 and row["merged_test_nrmse"]<=row["separate_test_nrmse"]+.05 and row["min_affine_realization_r2"]>.90 and row["order_accuracy"]>.95
def run_stage(stage,force=False):
    c=cfg();order=["adp7_a0","adp7_a1","adp7_a2","adp7_a3","adp7_a4"];pos=order.index(stage)
    if pos:
        prev=OUT/STAGE_DIR[order[pos-1]]/"final_metrics.json"
        if not prev.exists() or not json.loads(prev.read_text(encoding="utf-8"))["pass"]:raise RuntimeError(f"STOP: previous gate {order[pos-1]} did not pass")
    out=OUT/STAGE_DIR[stage];final=out/"final_metrics.json"
    if final.exists() and not force:return json.loads(final.read_text(encoding="utf-8"))
    rows=[];started=time.time()
    for pair in map(tuple,c["correct_pairs"]):
        for spec in SPECS[stage]:
            for init in c["initializations"]:
                row=evaluate_split(pair,spec,init,c);row["threshold_pass"]=thresholds(stage,row);rows.append(row);print(f"[{stage}] {row['pair']} {spec['id']} {init}: sep={row['separate_test_nrmse']:.4f} merge={row['merged_test_nrmse']:.4f} d={row['delta_nrmse']:.4f} func={row['min_functional_r2']:.3f} pass={row['threshold_pass']}",flush=True)
    primary=[r for r in rows if r["initialization"]==c["primary_initialization"]];random_rows=[r for r in rows if r["initialization"]=="random"];gate=all(r["threshold_pass"] for r in primary);checks={"all_correct_pairs_all_splits_primary_initialization":gate,"random_initialization_robustness":all(r["threshold_pass"] for r in random_rows)}
    # Section 17 calls success from both initializations stronger evidence, but
    # it is not an explicit A0-A4 gate.  The configured primary source-copy
    # test answers functional replaceability; random is reported separately as
    # optimization robustness and is never silently discarded.
    result={**meta(stage,c),"rows":rows,"gate_policy":"source_copy is the formal stage gate; random initialization is a separately reported robustness diagnostic per section 17","checks":checks,"pass":gate,"stop_required":not gate,"wall_seconds":time.time()-started};dump(final,result);dump(out/"config.json",c);return result
def main():
    stages=["adp7_0",*SPECS]
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=stages,required=True);p.add_argument("--seed",type=int,default=0);p.add_argument("--force",action="store_true");x=p.parse_args()
    if x.seed!=0:raise ValueError("development seed fixed to 0")
    result=stage_0(x.force) if x.stage=="adp7_0" else run_stage(x.stage,x.force);print(json.dumps({"stage":result["stage"],"pass":result["pass"]},indent=2),flush=True)
if __name__=="__main__":main()
