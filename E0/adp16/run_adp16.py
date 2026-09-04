"""ADP16 C2/C3 coordinate mismatch vs conditional function diagnostic."""
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
import adp12.run_adp12 as adp12
import adp14.run_adp14 as adp14

DEVICE=base.DEVICE;CONFIG=ROOT/"configs/e0_adp16.json";OUT=ROOT/"outputs/e0_adp16_c2c3_diagnostic";CHILD=1;PARENT=2
def cv(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,torch.Tensor):return x.detach().cpu().tolist()
    raise TypeError(type(x))
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=cv),encoding="utf-8")
def read(p):return json.loads(p.read_text(encoding="utf-8"))
def cfg():return read(CONFIG)

@torch.no_grad()
def candidate_effect(bank,candidate,S,value,batch=512):
    out=[]
    for st in range(0,len(S),batch):
        en=min(len(S),st+batch);s=torch.from_numpy(S[st:en]).to(DEVICE);vv=torch.zeros((en-st,5),device=DEVICE);vv[:,candidate]=torch.as_tensor(value[st:en],device=DEVICE);out.append(bank.raw_effects(s,vv)[:,candidate].cpu().numpy())
    return np.concatenate(out)

def source_objects():
    c,src,bank,a,b,saved,alive,data,audit=adp14.load_source(1);ta,tc,va,vc=adp14.assignments(bank,a,b,alive,data);return bank,a,b,alive,data,audit,ta,tc,va,vc

def functional_map(force=False):
    final=OUT/"stage_a/disagreement/final_metrics.json"
    if final.exists() and not force:return read(final)
    bank,a,b,alive,data,audit,ta,tc,va,vc=source_objects();allalpha=np.concatenate([data["atr"].reshape(-1),data["av"].reshape(-1),data["avh"].reshape(-1)]);q=np.quantile(allalpha,[.05,.20,.35,.50,.65,.80,.95]).astype(np.float32);ext=adp9.state_regions(data["va"]["S"])["extreme"]
    groups={"C2_region":np.concatenate([data["tr"]["S"][ta==CHILD],data["va"]["S"][va==CHILD]]),"C3_region":np.concatenate([data["tr"]["S"][ta==PARENT],data["va"]["S"][va==PARENT]]),"cross_state":data["va"]["S"][ext]};rows=[];matrix=[]
    for name,S0 in groups.items():
        vals=[]
        for alpha in q:
            S=np.asarray(S0,np.float32);x=np.full(len(S),float(a[CHILD])*alpha+float(b[CHILD]),np.float32);y=np.full(len(S),float(a[PARENT])*alpha+float(b[PARENT]),np.float32);f2=candidate_effect(bank,CHILD,S,x);f3=candidate_effect(bank,PARENT,S,y);num=np.sqrt(((f2-f3)**2).sum((1,2)));den=np.sqrt(.5*((f2**2).sum((1,2))+(f3**2).sum((1,2))))+1e-8;d=num/den;vals.append(float(d.mean()));rows.append({"region":name,"alpha":float(alpha),"mean":float(d.mean()),"median":float(np.median(d)),"p90":float(np.quantile(d,.9))})
        matrix.append(vals)
    result={"stage":"A1_functional_map","alpha_quantiles":q,"region_order":list(groups),"mean_disagreement_matrix":matrix,"rows":rows,"learner_visible":True,"source_checkpoint_sha256":audit["source_checkpoint_sha256"]};dump(final,result);np.savez_compressed(final.parent/"disagreement_map.npz",alpha=q,regions=np.asarray(list(groups)),mean=np.asarray(matrix),rows=np.asarray([[r["mean"],r["median"],r["p90"]] for r in rows]));return result

def bridge_dataset(bank,a,b,data,assignment,split,heldout=False):
    x=data[split];alpha=(data["avh"] if heldout else data["av"]) if split=="va" else data["atr"];take=np.isin(assignment,[CHILD,PARENT]);S=np.repeat(x["S"][take],alpha.shape[1],0).astype(np.float32);al=alpha[take].reshape(-1).astype(np.float32);v2=float(a[CHILD])*al+float(b[CHILD]);target=candidate_effect(bank,CHILD,S,v2);return S,al,v2,target,np.repeat(np.flatnonzero(take),alpha.shape[1])

def eval_bridge(bank,a,b,data,va,predict):
    S,al,v2,target,idx=bridge_dataset(bank,a,b,data,va,"va",False);pred=predict(S,v2);S2,al2,v22,target2,idx2=bridge_dataset(bank,a,b,data,va,"va",True);pred2=predict(S2,v22);family_idx=idx.reshape(-1,data["av"].shape[1])[:,0];fam_mask=np.isin(np.arange(len(data["va"]["S"])),family_idx);regions=adp9.state_regions(data["va"]["S"]);bases=adp9.base_regions(data["va"]["base_id"]);state_obs=np.repeat(regions["extreme"][fam_mask],data["av"].shape[1]);base_obs=np.repeat(bases["base_B"][fam_mask],data["av"].shape[1]);scores={"validation":base.r2_score(target,pred),"cross_state":base.r2_score(target[state_obs],pred[state_obs]),"cross_realization":base.r2_score(target2,pred2),"cross_base":base.r2_score(target2[base_obs],pred2[base_obs])};return scores,{"raw_target":target,"prediction":pred,"heldout_target":target2,"heldout_prediction":pred2}

def affine_bridge(force=False):
    final=OUT/"stage_a/affine_bridge/final_metrics.json"
    if final.exists() and not force:return read(final)
    c=cfg();bank,a,b,alive,data,audit,ta,tc,va,vc=source_objects();[p.requires_grad_(False) for p in bank.parameters()];S,al,v2,target,_=bridge_dataset(bank,a,b,data,ta,"tr");gamma=torch.nn.Parameter(torch.tensor(1.,device=DEVICE));delta=torch.nn.Parameter(torch.tensor(0.,device=DEVICE));opt=torch.optim.Adam([gamma,delta],lr=c["affine_lr"]);rng=np.random.default_rng(16021);losses=[]
    for step in range(c["affine_steps"]):
        ix=rng.choice(len(S),min(1024,len(S)),replace=False);St=torch.from_numpy(S[ix]).to(DEVICE);vt=gamma*torch.from_numpy(v2[ix]).to(DEVICE)+delta;vv=torch.zeros((len(ix),5),device=DEVICE);vv[:,PARENT]=vt;yt=torch.from_numpy(target[ix]).to(DEVICE);loss=(bank.raw_effects(St,vv)[:,PARENT]-yt).square().mean();opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
    def predict(S0,v20):return candidate_effect(bank,PARENT,S0,(float(gamma.detach())*v20+float(delta.detach())).astype(np.float32))
    scores,arrays=eval_bridge(bank,a,b,data,va,predict);checks={k:v>c["bridge_r2"] for k,v in scores.items()};result={"stage":"A2_affine_bridge","gamma":float(gamma.detach()),"delta":float(delta.detach()),"train_loss_start":losses[0],"train_loss_final":losses[-1],"loss_every_10":losses[::10],"r2":scores,"checks":checks,"pass":all(checks.values())};dump(final,result);np.savez_compressed(final.parent/"bridge_predictions.npz",**arrays,loss=np.asarray(losses));return result

class StateBridge(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(6,16),nn.GELU(),nn.Linear(16,2));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias);self.net[-1].bias.data[0]=1.
    def forward(self,S,v):
        z=self.net(S.reshape(len(S),-1));return z[:,0]*v+z[:,1],z

def state_bridge(force=False):
    final=OUT/"stage_a/state_bridge/final_metrics.json";aff=affine_bridge(force)
    if aff["pass"]:res={"skipped":True,"reason":"STOP: affine bridge passed"};dump(final,res);return res
    if final.exists() and not force:return read(final)
    c=cfg();bank,a,b,alive,data,audit,ta,tc,va,vc=source_objects();[p.requires_grad_(False) for p in bank.parameters()];S,al,v2,target,_=bridge_dataset(bank,a,b,data,ta,"tr");model=StateBridge().to(DEVICE);opt=torch.optim.Adam(model.parameters(),lr=c["state_bridge_lr"]);rng=np.random.default_rng(16031);losses=[]
    for step in range(c["state_bridge_steps"]):
        ix=rng.choice(len(S),min(1024,len(S)),replace=False);St=torch.from_numpy(S[ix]).to(DEVICE);v0=torch.from_numpy(v2[ix]).to(DEVICE);vt,z=model(St,v0);vv=torch.zeros((len(ix),5),device=DEVICE);vv[:,PARENT]=vt;yt=torch.from_numpy(target[ix]).to(DEVICE);loss=(bank.raw_effects(St,vv)[:,PARENT]-yt).square().mean();opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
    def predict(S0,v20):
        with torch.no_grad():v,_=model(torch.from_numpy(S0).to(DEVICE),torch.from_numpy(v20).to(DEVICE))
        return candidate_effect(bank,PARENT,S0,v.cpu().numpy())
    scores,arrays=eval_bridge(bank,a,b,data,va,predict);checks={k:v>c["bridge_r2"] for k,v in scores.items()};result={"stage":"A3_state_bridge","train_loss_start":losses[0],"train_loss_final":losses[-1],"loss_every_20":losses[::20],"r2":scores,"checks":checks,"pass":all(checks.values())};dump(final,result);torch.save(model.state_dict(),final.parent/"model.pt");np.savez_compressed(final.parent/"bridge_predictions.npz",**arrays,loss=np.asarray(losses));return result

class UnionModel(nn.Module):
    def __init__(self,source,hidden=64):
        super().__init__();self.bank=copy.deepcopy(source);self.union_adapter=nn.Parameter(torch.randn(source.dim,device=DEVICE)*.1);self.head=nn.Sequential(nn.Linear(source.dim+1,hidden),nn.GELU(),nn.Linear(hidden,2))
    def forward(self,S,v):
        b=self.bank;base_h=b.s_encoder(S)+b.slot_position[None];vf=torch.stack([v,v.square()],-1);condition=self.union_adapter[None]+b.v_encoder(vf);gamma,beta=b.film(condition).chunk(2,-1);x=base_h*(1+.1*gamma[:,None])+beta[:,None];B,T,_=x.shape;q=b.q(x).reshape(B,T,b.heads,b.head_dim).transpose(1,2);k=b.k(x).reshape(B,T,b.heads,b.head_dim).transpose(1,2);val=b.val(x).reshape(B,T,b.heads,b.head_dim).transpose(1,2);sc=torch.matmul(q,k.transpose(-1,-2))/math.sqrt(b.head_dim);att=torch.matmul(sc.softmax(-1),val).transpose(1,2).reshape(B,T,b.dim);x=b.norm1(x+b.attn_out(att));x=b.norm2(x+b.ff(x));return self.head(torch.cat([x,v[:,None,None].expand(B,T,1)],-1))

@torch.no_grad()
def union_predict(model,S,alpha,coord,source_id=None,batch=512):
    F,R=alpha.shape;out=[]
    for st in range(0,F,batch):
        en=min(F,st+batch);St=torch.from_numpy(np.repeat(S[st:en],R,0)).to(DEVICE);at=torch.from_numpy(alpha[st:en].reshape(-1)).to(DEVICE)
        if len(coord)==2:v=coord[0]*at+coord[1]
        else:
            sid=torch.from_numpy(np.repeat(source_id[st:en],R).astype(np.int64)).to(DEVICE);v=coord[0][sid]*at+coord[1][sid]
        out.append(model(St,v).cpu().numpy().reshape(en-st,R,3,2))
    return np.concatenate(out)

def composite_metrics(source,sa,sb,model,coord,pair,ta,va,data,dual=False):
    child,parent=pair;tmask=np.isin(ta,pair);vmask=np.isin(va,pair);sid_t=(ta[tmask]==parent).astype(np.int64);sid_v=(va[vmask]==parent).astype(np.int64);keep_obs=adp14.hard_predictions(source,sa,sb,va,data["va"],data["av"]);keep_h=adp14.hard_predictions(source,sa,sb,va,data["va"],data["avh"]);post_obs=keep_obs.copy();post_h=keep_h.copy();post_obs[vmask]=union_predict(model,data["va"]["S"][vmask],data["av"][vmask],coord,sid_v if dual else None);post_h[vmask]=union_predict(model,data["va"]["S"][vmask],data["avh"][vmask],coord,sid_v if dual else None);sr=adp9.state_regions(data["va"]["S"]);br=adp9.base_regions(data["va"]["base_id"]);domains={"iid":(data["va"]["response_train"],keep_obs,np.ones(len(va),bool)),"state":(data["va"]["response_heldout"],keep_h,sr["extreme"]),"realization":(data["va"]["response_heldout"],keep_h,np.ones(len(va),bool)),"base":(data["va"]["response_heldout"],keep_h,br["base_B"])};post={k:(y,post_obs if k=="iid" else post_h,m) for k,(y,p,m) in domains.items()};boot=adp14.bootstrap_compare(domains,post,data["va"]["base_id"],cfg()["bootstrap_repeats"],16061+child*11+parent);before=adp12.subgroup_nrmse(data["va"]["response_heldout"],keep_h,data["va"],data["avh"]);after=adp12.subgroup_nrmse(data["va"]["response_heldout"],post_h,data["va"],data["avh"]);sub=np.asarray(after)-np.asarray(before);old={}
    for j,name in [(child,"child"),(parent,"parent")]:
        m=va==j;old[name]={"keep":base.nrmse(data["va"]["response_heldout"][m],keep_h[m]),"union":base.nrmse(data["va"]["response_heldout"][m],post_h[m])};old[name]["delta"]=old[name]["union"]-old[name]["keep"]
    c=cfg();checks={"bootstrap_all_domains":max(x["upper95"] for x in boot.values())<c["global_delta_nrmse"],"max_subgroup_delta":float(sub.max())<c["subgroup_delta_nrmse"],"child_old":old["child"]["delta"]<c["old_family_slack_nrmse"],"parent_old":old["parent"]["delta"]<c["old_family_slack_nrmse"],"finite":bool(np.isfinite(post_h).all() and np.isfinite(post_obs).all())};return {"domains":boot,"max_subgroup_delta_nrmse":float(sub.max()),"old_family":old,"checks":checks,"post_global":{"iid":base.nrmse(data["va"]["response_train"],post_obs),"state":base.nrmse(data["va"]["response_heldout"][sr["extreme"]],post_h[sr["extreme"]]),"realization":base.nrmse(data["va"]["response_heldout"],post_h),"base":base.nrmse(data["va"]["response_heldout"][br["base_B"]],post_h[br["base_B"]])},"keep_obs":keep_obs,"keep_held":keep_h,"post_obs":post_obs,"post_held":post_h}

def other_drift(source,model,sa,sb,exclude,data,va):
    rows={};old=source;new=model.bank
    po=adp7.all_candidate_prediction(old,data["va"]["S"],data["avh"],sa,sb);pn=adp7.all_candidate_prediction(new,data["va"]["S"],data["avh"],sa,sb)
    for j in range(5):
        if j in exclude:continue
        m=va==j;x=po[m,:,j];y=pn[m,:,j];rows[f"C{j+1}"]=float(np.sqrt(((y-x)**2).sum()/max(float((x**2).sum()),1e-12)))
    return rows

def union_run(stage,pair=(CHILD,PARENT),dual=False,trunk=False,force=False):
    folder={"B1":"B1_fresh_single_coord","B2":"B2_fresh_dual_coord","B3":"B3_fresh_shared_trunk","wrong":"wrong_pair_control"}[stage];final=OUT/"stage_b"/folder/"final_metrics.json"
    if final.exists() and not force:return read(final)
    c=cfg();source,sa,sb,alive,data,audit,ta,tc,va,vc=source_objects();child,parent=pair;base.seed_everything(16100+child*101+parent*17+(1 if dual else 0)+(2 if trunk else 0));model=UnionModel(source,c["head_hidden"]).to(DEVICE);[p.requires_grad_(False) for p in model.bank.parameters()];trunk_params=[]
    if trunk:
        trunk_params=[p for n,p in model.bank.named_parameters() if n!="adapters"];[p.requires_grad_(True) for p in trunk_params]
    if dual:a=torch.nn.Parameter(torch.tensor([float(sa[child]),float(sa[parent])],device=DEVICE));b=torch.nn.Parameter(torch.tensor([float(sb[child]),float(sb[parent])],device=DEVICE))
    else:a=torch.nn.Parameter(torch.tensor(float((sa[child]+sa[parent])/2),device=DEVICE));b=torch.nn.Parameter(torch.tensor(float((sb[child]+sb[parent])/2),device=DEVICE))
    groups=[{"params":[model.union_adapter]+list(model.head.parameters()),"lr":c["head_lr"],"weight_decay":c["weight_decay"]},{"params":[a,b],"lr":c["coordinate_lr"],"weight_decay":0.}]
    if trunk_params:groups.append({"params":trunk_params,"lr":c["trunk_lr"],"weight_decay":c["weight_decay"]})
    opt=torch.optim.AdamW(groups);cf,pf=np.flatnonzero(ta==child),np.flatnonzero(ta==parent);R=data["atr"].shape[1];rng=np.random.default_rng(16200+child*101+parent*13+(1 if dual else 0)+(2 if trunk else 0));half=c["family_batch_size"]//2;steps=max(1,math.ceil(max(len(cf),len(pf))/half));initial={n:p.detach().clone() for n,p in model.bank.named_parameters()};rows=[];started=time.time()
    def coord():return (a,b) if not dual else (a,b,"dual")
    def predict_idx(fi,alpha,split):
        sid=(ta[fi]==parent).astype(np.int64) if split=="tr" else (va[fi]==parent).astype(np.int64);return union_predict(model,data[split]["S"][fi],alpha[fi],(a,b) if not dual else (a,b,"dual"),sid if dual else None)
    def tele(ep,loss):
        cv,pv=np.flatnonzero(va==child),np.flatnonzero(va==parent);pc=predict_idx(cv,data["avh"],"va");pp=predict_idx(pv,data["avh"],"va");comp=composite_metrics(source,sa,sb,model,(a,b) if not dual else (a,b,"dual"),pair,ta,va,data,dual);delta=math.sqrt(sum(float(((p-initial[n])**2).sum()) for n,p in model.bank.named_parameters() if n!="adapters"));row={"epoch":ep,"C2_source_train_nrmse":base.nrmse(data["tr"]["response_train"][cf],predict_idx(cf,data["atr"],"tr")),"C3_source_train_nrmse":base.nrmse(data["tr"]["response_train"][pf],predict_idx(pf,data["atr"],"tr")),"C2_val_nrmse":base.nrmse(data["va"]["response_heldout"][cv],pc),"C3_val_nrmse":base.nrmse(data["va"]["response_heldout"][pv],pp),"union_val_nrmse":comp["post_global"]["realization"],"iid_nrmse":comp["post_global"]["iid"],"state_nrmse":comp["post_global"]["state"],"realization_nrmse":comp["post_global"]["realization"],"base_nrmse":comp["post_global"]["base"],"a":a.detach().cpu(),"b":b.detach().cpu(),"trunk_update_norm":delta,"loss":loss};return row
    rows.append(tele(0,None))
    for ep in range(1,c["union_epochs"]+1):
        losses=[]
        for _ in range(steps):
            ci=rng.choice(cf,half,replace=len(cf)<half);pi=rng.choice(pf,half,replace=len(pf)<half);fi=np.concatenate([ci,pi]);sid=np.concatenate([np.zeros(half,np.int64),np.ones(half,np.int64)]);order=rng.permutation(len(fi));fi=fi[order];sid=sid[order];St=torch.from_numpy(np.repeat(data["tr"]["S"][fi],R,0)).to(DEVICE);at=torch.from_numpy(data["atr"][fi].reshape(-1)).to(DEVICE);dt=torch.from_numpy(data["tr"]["response_train"][fi].reshape(-1,3,2)).to(DEVICE)
            if dual:sidt=torch.from_numpy(np.repeat(sid,R)).to(DEVICE);v=a[sidt]*at+b[sidt]
            else:v=a*at+b
            pred=model(St,v);loss=(pred-dt).square().mean();protect=torch.zeros((),device=DEVICE)
            if trunk:
                per=[]
                for j in [x for x in alive if x not in pair]:
                    ids=np.flatnonzero(ta==j);jj=rng.choice(ids,max(1,c["family_batch_size"]//3),replace=len(ids)<c["family_batch_size"]//3);Sj=torch.from_numpy(np.repeat(data["tr"]["S"][jj],R,0)).to(DEVICE);al=torch.from_numpy(data["atr"][jj].reshape(-1)).to(DEVICE);vv=torch.zeros((len(jj)*R,5),device=DEVICE);vv[:,j]=sa[j]*al+sb[j]
                    with torch.no_grad():target=source.raw_effects(Sj,vv)[:,j]
                    per.append((model.bank.raw_effects(Sj,vv)[:,j]-target).square().mean())
                protect=torch.stack(per).sum()
            total=loss+(c["lambda_protect"]*protect if trunk else 0);opt.zero_grad();total.backward();torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]],c["gradient_clip"]);opt.step();losses.append([float(loss.detach()),float(protect.detach()),float(total.detach())])
        if ep%4==0:
            z=np.asarray(losses);row=tele(ep,{"fit":z[:,0].mean(),"protect":z[:,1].mean(),"total":z[:,2].mean()});rows.append(row);print(f"[{stage} C{child+1}+C{parent+1}] ep={ep:02d} C2={row['C2_val_nrmse']:.3f} C3={row['C3_val_nrmse']:.3f} global={row['union_val_nrmse']:.3f} trunkD={row['trunk_update_norm']:.3f}",flush=True)
    comp=composite_metrics(source,sa,sb,model,(a,b) if not dual else (a,b,"dual"),pair,ta,va,data,dual);drift=other_drift(source,model,sa,sb,pair,data,va);checks=dict(comp["checks"]);checks["other_candidate_protection"]=not trunk or max(drift.values())<c["other_candidate_drift_nrmse"];passed=all(checks.values());result={"stage":stage,"pair":[f"C{child+1}",f"C{parent+1}"],"dual_coordinate":dual,"shared_trunk_trainable":trunk,"epochs":c["union_epochs"],"telemetry":rows,"comparison":{k:v for k,v in comp.items() if k not in ("keep_obs","keep_held","post_obs","post_held")},"other_candidate_drift":drift,"gate_checks":checks,"pass":passed,"wall_seconds":time.time()-started,"source_checkpoint_sha256":audit["source_checkpoint_sha256"]};dump(final,result);np.savez_compressed(final.parent/"predictions.npz",keep_obs=comp["keep_obs"],keep_held=comp["keep_held"],post_obs=comp["post_obs"],post_held=comp["post_held"],val_assignment=va);torch.save({"model":model.state_dict(),"a":a.detach().cpu(),"b":b.detach().cpu()},final.parent/"model.pt");return result

def worst_parent():
    bank,a,b,alive,data,audit,ta,tc,va,vc=source_objects();m=va==CHILD;choices=[j for j in alive if j not in (CHILD,PARENT)];rows=[{"candidate":j,"median_gap":float(np.median(vc[m,j]-vc[m,CHILD]))} for j in choices];return max(rows,key=lambda x:x["median_gap"]),rows

def state_conditional(force=False):
    final=OUT/"state_conditional/final_metrics.json"
    if final.exists() and not force:return read(final)
    source,a,b,alive,data,audit,ta,tc,va,vc=source_objects();take=np.isin(va,[CHILD,PARENT]);S=data["va"]["S"][take];z=S.reshape(len(S),-1).astype(np.float64);z-=z.mean(0);v=np.ones(z.shape[1])/math.sqrt(z.shape[1])
    for _ in range(50):
        score=(z*v[None]).sum(1);v=(z*score[:,None]).sum(0);v/=max(float(np.sqrt((v*v).sum())),1e-12)
    score=(z*v[None]).sum(1);cuts=np.quantile(score,[.25,.5,.75]);bucket=np.digitize(score,cuts);target=data["va"]["response_heldout"][take];alpha=data["avh"][take];v2=float(a[CHILD])*alpha+float(b[CHILD]);v3=float(a[PARENT])*alpha+float(b[PARENT]);p2=candidate_effect(source,CHILD,np.repeat(S,alpha.shape[1],0),v2.reshape(-1)).reshape(target.shape);p3=candidate_effect(source,PARENT,np.repeat(S,alpha.shape[1],0),v3.reshape(-1)).reshape(target.shape);b3=np.load(OUT/"stage_b/B3_fresh_shared_trunk/predictions.npz")["post_held"][take];rows=[]
    for q in range(4):
        m=bucket==q
        def nrm(p):return base.nrmse(target[m],p[m])
        rows.append({"bucket":q,"family_count":int(m.sum()),"score_min":float(score[m].min()),"score_max":float(score[m].max()),"C2_error_nrmse":nrm(p2),"C3_error_nrmse":nrm(p3),"fresh_union_B3_error_nrmse":nrm(b3),"C2_C3_disagreement_nrmse":float(np.sqrt(((p2[m]-p3[m])**2).sum()/max(float(.5*((p2[m]**2).sum()+(p3[m]**2).sum())),1e-12)))})
    best=["C2" if r["C2_error_nrmse"]<r["C3_error_nrmse"] else "C3" for r in rows];complementary=len(set(best))>1;result={"stage":"state_conditional_diagnostic","training_performed":False,"bucket_method":"learner-visible first PCA score quartiles","pca_direction":v,"cuts":cuts,"rows":rows,"best_source_by_bucket":best,"state_region_complementarity":complementary};dump(final,result);np.savez_compressed(final.parent/"bucket_diagnostics.npz",score=score,bucket=bucket,take_indices=np.flatnonzero(take),target=target,C2_prediction=p2,C3_prediction=p3,B3_union_prediction=b3);return result

def run_all(force=False):
    fm=functional_map(force);aff=affine_bridge(force);state=state_bridge(force) if not aff["pass"] else {"skipped":True};b1=union_run("B1",force=force);b2=b3=None
    if not b1["pass"]:
        b2=union_run("B2",dual=True,force=force)
        if not b2["pass"]:b3=union_run("B3",trunk=True,force=force)
    worst,table=worst_parent();wrong=union_run("wrong",pair=(CHILD,worst["candidate"]),force=force);invalid=wrong["pass"]
    terminal=(not aff["pass"] and not b1["pass"] and (b2 is not None and not b2["pass"]) and (b3 is not None and not b3["pass"]));state_diag=state_conditional(force) if terminal else None
    if invalid:case="invalid_wrong_pair_pass"
    elif aff["pass"]:case="realization_gauge_mismatch"
    elif b1["pass"]:case="warm_start_path_locking"
    elif b2 and b2["pass"]:case="coordinate_incompatibility"
    elif b3 and b3["pass"]:case="shared_representation_bottleneck"
    else:case="deeper_conditional_function_or_identity_mismatch"
    result={"functional_map_complete":True,"affine_bridge_pass":aff["pass"],"state_bridge_pass":None if state.get("skipped") else state["pass"],"B1_pass":b1["pass"],"B2_pass":None if b2 is None else b2["pass"],"B3_pass":None if b3 is None else b3["pass"],"wrong_pair_selection":worst,"wrong_pair_table":table,"wrong_pair_pass":wrong["pass"],"stop_1_invalid":invalid,"state_conditional_diagnostic":None if state_diag is None else {"state_region_complementarity":state_diag["state_region_complementarity"],"best_source_by_bucket":state_diag["best_source_by_bucket"]},"classification":case};dump(OUT/"aggregate/final_verdict.json",result);print(json.dumps(result,indent=2));return result

def smoke():
    bank,a,b,alive,data,audit,ta,tc,va,vc=source_objects();base.seed_everything(16000);m=UnionModel(bank).to(DEVICE);S=torch.from_numpy(data["tr"]["S"][:8]).to(DEVICE);v=torch.zeros(8,device=DEVICE)
    with torch.no_grad():y=m(S,v)
    result={"device":str(DEVICE),"source_checkpoint_sha256":audit["source_checkpoint_sha256"],"union_output_shape":list(y.shape),"finite":bool(torch.isfinite(y).all()),"train_family_counts":{"C2":int((ta==CHILD).sum()),"C3":int((ta==PARENT).sum())},"val_family_counts":{"C2":int((va==CHILD).sum()),"C3":int((va==PARENT).sum())},"pass":bool(torch.isfinite(y).all())};dump(OUT/"smoke/final_metrics.json",result);print(json.dumps(result,indent=2));return result

def main():
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=["smoke","functional_map","affine_bridge","state_bridge","B1","B2","B3","wrong_pair","state_conditional","all"],default="all");p.add_argument("--force",action="store_true");a=p.parse_args()
    if a.stage=="smoke":smoke()
    elif a.stage=="functional_map":functional_map(a.force)
    elif a.stage=="affine_bridge":affine_bridge(a.force)
    elif a.stage=="state_bridge":state_bridge(a.force)
    elif a.stage=="B1":union_run("B1",force=a.force)
    elif a.stage=="B2":union_run("B2",dual=True,force=a.force)
    elif a.stage=="B3":union_run("B3",trunk=True,force=a.force)
    elif a.stage=="wrong_pair":
        w,_=worst_parent();union_run("wrong",pair=(CHILD,w["candidate"]),force=a.force)
    elif a.stage=="state_conditional":state_conditional(a.force)
    else:run_all(a.force)
if __name__=="__main__":main()
