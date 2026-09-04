"""ADP15 capacity-vs-shared-representation diagnostic."""
from __future__ import annotations
import argparse,copy,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import run_e0 as base
import adp6.run_adp6 as adp6
import adp7.run_adp7 as adp7
import adp9.run_adp9 as adp9
import adp14.run_adp14 as adp14

DEVICE=base.DEVICE;CONFIG=ROOT/"configs/e0_adp15.json";OUT=ROOT/"outputs/e0_adp15_capacity_representation"
PAIRS={"C5:C4":(4,3,"C5_to_C4"),"C2:C3":(1,2,"C2_to_C3")}

def cv(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,torch.Tensor):return x.detach().cpu().tolist()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x))
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=cv),encoding="utf-8")
def read(p):return json.loads(p.read_text(encoding="utf-8"))
def cfg():return read(CONFIG)

class ExpandedParentBank(base.SetMechanismBank):
    """Original bank plus a zero-initialized parent-only residual head."""
    def __init__(self,parent,dim=64,heads=4):
        super().__init__(5,dim,heads);self.parent=int(parent);self.parent_residual=nn.Sequential(nn.Linear(dim+1,dim),nn.GELU(),nn.Linear(dim,2));nn.init.zeros_(self.parent_residual[-1].weight);nn.init.zeros_(self.parent_residual[-1].bias)
    def raw_effects(self,S,v):
        base_h=self.s_encoder(S)+self.slot_position[None];vf=torch.stack([v,v.square()],dim=-1);condition=self.adapters[None]+self.v_encoder(vf);gamma,beta=self.film(condition).chunk(2,dim=-1);x=base_h[:,None]*(1.0+.1*gamma[:,:,None])+beta[:,:,None];B,M,T,_=x.shape
        q=self.q(x).reshape(B*M,T,self.heads,self.head_dim).transpose(1,2);k=self.k(x).reshape(B*M,T,self.heads,self.head_dim).transpose(1,2);val=self.val(x).reshape(B*M,T,self.heads,self.head_dim).transpose(1,2);score=torch.matmul(q,k.transpose(-1,-2))/math.sqrt(self.head_dim);att=torch.matmul(score.softmax(-1),val).transpose(1,2).reshape(B,M,T,self.dim);x=self.norm1(x+self.attn_out(att));x=self.norm2(x+self.ff(x));out=self.output(x)
        pv=v[:,self.parent,None,None].expand(B,T,1);out=out.clone();out[:,self.parent]=out[:,self.parent]+self.parent_residual(torch.cat([x[:,self.parent],pv],dim=-1));return out

def expanded(source,parent):
    m=ExpandedParentBank(parent,source.dim,source.heads).to(DEVICE);sd=m.state_dict();old=source.state_dict()
    for k in old:sd[k]=old[k].clone()
    m.load_state_dict(sd);return m

def norm_params(items):
    return float(math.sqrt(sum(float((p.detach().float()**2).sum()) for p in items)))
def delta_params(named,initial):
    return float(math.sqrt(sum(float(((p.detach()-initial[n].to(p.device))**2).sum()) for n,p in named)))

@torch.no_grad()
def direct(bank,a,b,parent,S,alpha):return adp7.all_candidate_prediction(bank,S,alpha,a,b)[:,:,parent]

def candidate_drift(old,oa,ob,new,a,b,parent,data,assignment):
    po=adp7.all_candidate_prediction(old,data["va"]["S"],data["avh"],oa,ob);pn=adp7.all_candidate_prediction(new,data["va"]["S"],data["avh"],a,b);rows={}
    for j in range(5):
        if j==parent:continue
        own=assignment==j
        def ratio(mask):
            x=po[mask,:,j];y=pn[mask,:,j];return float(np.sqrt(((y-x)**2).sum()/max(float((x**2).sum()),1e-12)))
        rows[f"C{j+1}"]={"own_context_nrmse":ratio(own) if own.any() else None,"global_context_nrmse":ratio(np.ones(len(assignment),bool))}
    return rows

def telemetry(bank,a,b,old,oa,ob,parent,child,alive,ta,va,data,initial,residual,epoch,loss=None):
    cf,pf=np.flatnonzero(ta==child),np.flatnonzero(ta==parent);cv,pv=np.flatnonzero(va==child),np.flatnonzero(va==parent);post_alive=[x for x in alive if x!=child];vnew,_=adp9.reassign(post_alive,bank,a,b,data["va"],data["av"]);domains,held=adp14.domain_payload(bank,a,b,post_alive,vnew,data)
    named=[(n,p) for n,p in bank.named_parameters() if not n.startswith("parent_residual") and n!="adapters"]
    row={"epoch":epoch,"child_train_nrmse":base.nrmse(data["tr"]["response_train"][cf],direct(bank,a,b,parent,data["tr"]["S"][cf],data["atr"][cf])),"parent_train_nrmse":base.nrmse(data["tr"]["response_train"][pf],direct(bank,a,b,parent,data["tr"]["S"][pf],data["atr"][pf])),"child_val_nrmse":base.nrmse(data["va"]["response_heldout"][cv],direct(bank,a,b,parent,data["va"]["S"][cv],data["avh"][cv])),"parent_val_nrmse":base.nrmse(data["va"]["response_heldout"][pv],direct(bank,a,b,parent,data["va"]["S"][pv],data["avh"][pv])),"global_val_nrmse":base.nrmse(domains["realization"][0],domains["realization"][1]),"cross_state_nrmse":base.nrmse(domains["state"][0][domains["state"][2]],domains["state"][1][domains["state"][2]]),"cross_v_nrmse":base.nrmse(domains["realization"][0],domains["realization"][1]),"cross_base_nrmse":base.nrmse(domains["base"][0][domains["base"][2]],domains["base"][1][domains["base"][2]]),"parent_adapter_norm":float(bank.adapters[parent].detach().norm()),"parent_adapter_update_norm":float((bank.adapters[parent].detach()-initial["adapters"][parent].to(DEVICE)).norm()),"parent_residual_norm":norm_params(residual),"trunk_update_norm":delta_params(named,initial),"a":float(a[parent].detach()),"b":float(b[parent].detach()),"candidate_count":len(post_alive)}
    if loss:row.update(loss)
    return row,vnew

def anchor_pool(data,assignment,candidate):
    ti=np.flatnonzero(assignment[0]==candidate);vi=np.flatnonzero(assignment[1]==candidate);R=data["atr"].shape[1];S=np.concatenate([np.repeat(data["tr"]["S"][ti],R,0),np.repeat(data["va"]["S"][vi],2*R,0)]);alpha=np.concatenate([data["atr"][ti].reshape(-1),data["av"][vi].reshape(-1),data["avh"][vi].reshape(-1)]).astype(np.float32);return S,alpha

def train_level(pair,level,force=False,lambda_parent=1.,lambda_protect=None,wrong_parent=None):
    child,parent,name=PAIRS[pair];parent=parent if wrong_parent is None else wrong_parent;c=cfg()
    if wrong_parent is not None:tag=f"{level}_wrong_C{parent+1}";rel=f"controls/{tag}"
    elif lambda_parent!=1.:tag=f"{level}_no_parent_replay";rel=f"controls/{tag}"
    elif lambda_protect is not None:tag=f"{level}_no_global_protection";rel=f"controls/{tag}"
    else:tag=level;rel=f"{level}_{'parent_head' if level=='L2' else 'shared_trunk'}"
    out=OUT/name/rel;final=out/"final_metrics.json"
    if final.exists() and not force:return read(final)
    ac,src,source,sa,sb,saved,alive,data,audit=adp14.load_source(1);ta,tc,va,vc=adp14.assignments(source,sa,sb,alive,data);base.seed_everything(15001+child*101+parent*17+(0 if level=="L2" else 1));bank=expanded(source,parent);a=torch.nn.Parameter(sa.detach().clone());b=torch.nn.Parameter(sb.detach().clone());old=copy.deepcopy(source).to(DEVICE);oa=sa.detach().clone();ob=sb.detach().clone()
    for p in old.parameters():p.requires_grad_(False)
    initial={n:p.detach().cpu().clone() for n,p in bank.named_parameters()};initial["adapters"]=bank.adapters.detach().cpu().clone()
    for p in bank.parameters():p.requires_grad_(False)
    bank.adapters.requires_grad_(True);[p.requires_grad_(True) for p in bank.parent_residual.parameters()];a.requires_grad_(True);b.requires_grad_(True)
    trunk=[]
    if level=="L3":
        trunk=[p for n,p in bank.named_parameters() if n!="adapters" and not n.startswith("parent_residual")];[p.requires_grad_(True) for p in trunk]
    groups=[{"params":[bank.adapters]+list(bank.parent_residual.parameters()),"lr":c["mechanism_learning_rate"],"weight_decay":c["weight_decay"]},{"params":[a,b],"lr":c["coordinate_learning_rate"],"weight_decay":0.}]
    if trunk:groups.append({"params":trunk,"lr":c["trunk_learning_rate"],"weight_decay":c["weight_decay"]})
    opt=torch.optim.AdamW(groups);cf,pf=np.flatnonzero(ta==child),np.flatnonzero(ta==parent);R=data["atr"].shape[1];pS,palpha=anchor_pool(data,(ta,va),parent);other={j:anchor_pool(data,(ta,va),j) for j in alive if j!=parent};rng=np.random.default_rng(1510000+child*1009+parent*9173+(2 if level=="L2" else 3));half=c["family_batch_size"]//2;steps=max(1,math.ceil(max(len(cf),len(pf))/half));protect_weight=c["lambda_protect"] if lambda_protect is None and level=="L3" else (lambda_protect or 0.);rows=[];row,vpost=telemetry(bank,a,b,old,oa,ob,parent,child,alive,ta,va,data,initial,list(bank.parent_residual.parameters()),0);rows.append(row);started=time.time();(out/"checkpoints").mkdir(parents=True,exist_ok=True)
    for ep in range(1,c["epochs"]+1):
        ls=[]
        for _ in range(steps):
            ci=rng.choice(cf,half,replace=len(cf)<half);pi=rng.choice(pf,half,replace=len(pf)<half)
            def rloss(fi):
                St=torch.from_numpy(np.repeat(data["tr"]["S"][fi],R,0)).to(DEVICE);at=torch.from_numpy(data["atr"][fi].reshape(-1)).to(DEVICE);dt=torch.from_numpy(data["tr"]["response_train"][fi].reshape(-1,3,2)).to(DEVICE);v=torch.zeros((len(fi)*R,5),device=DEVICE);v[:,parent]=a[parent]*at+b[parent];return (bank.raw_effects(St,v)[:,parent]-dt).square().mean()
            lc,lp=rloss(ci),rloss(pi);ii=rng.choice(len(palpha),c["family_batch_size"],replace=len(palpha)<c["family_batch_size"]);St=torch.from_numpy(pS[ii]).to(DEVICE);at=torch.from_numpy(palpha[ii]).to(DEVICE);vn=torch.zeros((len(ii),5),device=DEVICE);vo=torch.zeros_like(vn);vn[:,parent]=a[parent]*at+b[parent];vo[:,parent]=oa[parent]*at+ob[parent]
            with torch.no_grad():target=old.raw_effects(St,vo)[:,parent]
            la=(bank.raw_effects(St,vn)[:,parent]-target).square().mean();lpro=torch.zeros((),device=DEVICE)
            if protect_weight:
                per=[]
                for j,(oS,oalpha) in other.items():
                    oi=rng.choice(len(oalpha),max(1,c["family_batch_size"]//len(other)),replace=len(oalpha)<c["family_batch_size"]//len(other));Sj=torch.from_numpy(oS[oi]).to(DEVICE);aj=torch.from_numpy(oalpha[oi]).to(DEVICE);nv=torch.zeros((len(oi),5),device=DEVICE);ov=torch.zeros_like(nv);nv[:,j]=a[j]*aj+b[j];ov[:,j]=oa[j]*aj+ob[j]
                    with torch.no_grad():yt=old.raw_effects(Sj,ov)[:,j]
                    per.append((bank.raw_effects(Sj,nv)[:,j]-yt).square().mean())
                lpro=torch.stack(per).sum()
            total=lc+lambda_parent*lp+c["lambda_anchor"]*la+protect_weight*lpro;opt.zero_grad(set_to_none=True);total.backward()
            with torch.no_grad():
                mask=torch.zeros_like(bank.adapters.grad);mask[parent]=1;bank.adapters.grad.mul_(mask);a.grad[:parent].zero_();a.grad[parent+1:].zero_();b.grad[:parent].zero_();b.grad[parent+1:].zero_()
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]],c["gradient_clip"]);opt.step()
            with torch.no_grad():
                keep=torch.arange(5,device=DEVICE)!=parent;bank.adapters[keep]=initial["adapters"].to(DEVICE)[keep];a[keep]=sa[keep];b[keep]=sb[keep]
            ls.append([float(lc.detach()),float(lp.detach()),float(la.detach()),float(lpro.detach()),float(total.detach())])
        if ep%4==0:
            z=np.asarray(ls);loss={"loss_child":z[:,0].mean(),"loss_parent":z[:,1].mean(),"loss_anchor":z[:,2].mean(),"loss_protect":z[:,3].mean(),"loss_total":z[:,4].mean()};row,vpost=telemetry(bank,a,b,old,oa,ob,parent,child,alive,ta,va,data,initial,list(bank.parent_residual.parameters()),ep,loss);rows.append(row);torch.save({"bank":bank.state_dict(),"coordinate_a":a.detach().cpu(),"coordinate_b":b.detach().cpu(),"parent":parent},out/"checkpoints"/f"epoch_{ep:03d}.pt");print(f"[{pair} {tag}] ep={ep:02d} child={row['child_val_nrmse']:.4f} parent={row['parent_val_nrmse']:.4f} global={row['global_val_nrmse']:.4f} trunkD={row['trunk_update_norm']:.3f}",flush=True)
    post_alive=[x for x in alive if x!=child];train_post,cost_train,val_post,cost_val=adp14.assignments(bank,a,b,post_alive,data);comparison=adp14.compare_models(source,sa,sb,alive,va,bank,a,b,post_alive,val_post,child,parent,data,c,1515000+child*31+parent);drift=candidate_drift(old,oa,ob,bank,a,b,parent,data,va);maxdrift=max(v["global_context_nrmse"] for v in drift.values());checks=dict(comparison["checks"]);checks["other_candidate_protection"]=level!="L3" or maxdrift<c["other_candidate_drift_nrmse"];passed=all(checks.values());structured=adp14.structured(bank,a,b,post_alive,val_post,data,c);result={"pair":pair,"child":child,"parent":parent,"level":level,"wrong_parent":wrong_parent is not None,"lambda_parent":lambda_parent,"lambda_anchor":c["lambda_anchor"],"lambda_protect":protect_weight,"epochs":c["epochs"],"telemetry":rows,"comparison":comparison,"other_candidate_drift":drift,"max_other_candidate_global_drift":maxdrift,"gate_checks":checks,"absorb_pass":passed,"structured_evaluator_only":structured,"wall_seconds":time.time()-started,"source_checkpoint_sha256":audit["source_checkpoint_sha256"]};dump(final,result);np.savez_compressed(out/"assignments_costs.npz",train=train_post,val=val_post,train_cost=cost_train,val_cost=cost_val);torch.save({"bank":bank.state_dict(),"coordinate_a":a.detach().cpu(),"coordinate_b":b.detach().cpu(),"alive":post_alive,"parent":parent},out/"model.pt");dump(out/"telemetry.json",rows);return result

def reuse_baselines():
    dst=OUT/"C5_to_C4";a14=adp14.OUT/"seed_001/cycle_00/A1_constrained/proposal_01_C5_to_C4";l0=read(a14/"A0_frozen_delete/final_metrics.json");l1=read(a14/"trial/final_metrics.json");r0={"level":"L0","reused_from":str(a14/"A0_frozen_delete"),"training_performed":False,"comparison":l0["comparison"],"absorb_pass":l0["pass"]};r1={"level":"L1","reused_from":str(a14/"trial"),"training_performed":False,"comparison":l1["comparison"],"telemetry":read(a14/"trial/learning_curve.json"),"absorb_pass":l1["commit_gate_pass"]};dump(dst/"L0_frozen/final_metrics.json",r0);dump(dst/"L1_adapter/final_metrics.json",r1);return r0,r1

def run_main(force=False):
    l0,l1=reuse_baselines();l2=train_level("C5:C4","L2",force);l3=train_level("C5:C4","L3",force);levels={"L0":l0,"L1":l1,"L2":l2,"L3":l3};successful=[x for x in ("L1","L2","L3") if levels[x]["absorb_pass"]];minimum=successful[0] if successful else None;case="capacity_bottleneck" if l2["absorb_pass"] else ("shared_representation_bottleneck" if l3["absorb_pass"] else "deeper_mechanism_mismatch");verdict={"pair":"C5:C4","levels":{k:v["absorb_pass"] for k,v in levels.items()},"minimum_successful_level":minimum,"classification":case,"stop_1":minimum is None,"controls_required":None if minimum is None else minimum,"auxiliary_pair_required":minimum is not None};dump(OUT/"C5_to_C4/verdict.json",verdict);print(json.dumps(verdict,indent=2));return verdict

def run_controls(force=False):
    v=read(OUT/"C5_to_C4/verdict.json") if (OUT/"C5_to_C4/verdict.json").exists() else run_main(force)
    if v["stop_1"]:res={"skipped":True,"reason":"STOP-1: L1/L2/L3 all failed"};dump(OUT/"C5_to_C4/controls/summary.json",res);return res
    level=v["minimum_successful_level"];# learner-visible worst replacement parent from ADP14 was C2 (index 1)
    wrong=train_level("C5:C4",level,force,wrong_parent=1);control=train_level("C5:C4",level,force,lambda_parent=0.) if level=="L2" else train_level("C5:C4",level,force,lambda_protect=0.);res={"skipped":False,"level":level,"wrong_parent_pass":wrong["absorb_pass"],"ablation_pass":control["absorb_pass"],"stop_2":wrong["absorb_pass"]};dump(OUT/"C5_to_C4/controls/summary.json",res);return res

def replicate(force=False):
    v=read(OUT/"C5_to_C4/verdict.json") if (OUT/"C5_to_C4/verdict.json").exists() else run_main(force)
    if v["stop_1"]:res={"skipped":True,"reason":"No successful main-pair level"};dump(OUT/"C2_to_C3/verdict.json",res);return res
    x=train_level("C2:C3",v["minimum_successful_level"],force);res={"skipped":False,"level":v["minimum_successful_level"],"absorb_pass":x["absorb_pass"]};dump(OUT/"C2_to_C3/verdict.json",res);return res

def smoke():
    c,src,source,a,b,saved,alive,data,audit=adp14.load_source(1);m=expanded(source,3);v=torch.randn(2,5,device=DEVICE);S=torch.from_numpy(data["tr"]["S"][:2]).to(DEVICE)
    with torch.no_grad():err=float((m.raw_effects(S,v)-source.raw_effects(S,v)).abs().max())
    result={"device":str(DEVICE),"source_checkpoint":audit["source_checkpoint_sha256"],"zero_init_exact_max_abs_error":err,"train_families":len(data["tr"]["S"]),"val_families":len(data["va"]["S"]),"pass":err==0.0};dump(OUT/"smoke/final_metrics.json",result);print(json.dumps(result,indent=2));return result

def main():
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=["smoke","main","controls","replicate","all"],default="all");p.add_argument("--pair",default="C5:C4",choices=list(PAIRS));p.add_argument("--level",choices=["L2","L3"]);p.add_argument("--force",action="store_true");a=p.parse_args()
    if a.stage=="smoke":smoke()
    elif a.level:train_level(a.pair,a.level,a.force)
    elif a.stage=="main":run_main(a.force)
    elif a.stage=="controls":run_controls(a.force)
    elif a.stage=="replicate":replicate(a.force)
    else:
        v=run_main(a.force);run_controls(a.force);replicate(a.force)
if __name__=="__main__":main()
