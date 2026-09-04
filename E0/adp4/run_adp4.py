"""Execute E0-ADP4 stages 0/A0/A1 with strict stage gates."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import run_e0 as base
from adp3.run_adp3 import functional_matrix as probe_functional_matrix

CONFIG_PATH = ROOT / "configs" / "e0_adp4_8gb.json"
ADP3_DATA = ROOT / "outputs" / "e0_adp3" / "contrast_data"
OUT = ROOT / "outputs" / "e0_adp4"
UNIT_DATA = OUT / "a0_local_units" / "data"
DEVICE = base.DEVICE
INJECTIVE = np.asarray(list(itertools.permutations(range(5), 3)), dtype=np.int16)
assert INJECTIVE.shape == (60, 3)


def cfg() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def default_json(x: Any) -> Any:
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, torch.Tensor): return x.detach().cpu().tolist()
    raise TypeError(type(x))


def dump(path: Path, x: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(x, indent=2, ensure_ascii=False, default=default_json), encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=default_json) + "\n")


def meta(stage: str, c: dict[str, Any]) -> dict[str, Any]:
    x = base.environment_metadata()
    x.update({"stage": stage, "world_seed": base.WORLD_SEED, "dataset_seed": base.DATASET_SEED,
              "optimization_seed": c["optimization_seed"], "config": c,
              "config_sha256": base.sha256_file(CONFIG_PATH), "script_sha256": base.sha256_file(Path(__file__)),
              "git_commit": base.git_commit()})
    return x


def adp3_visible(split: str) -> dict[str, np.ndarray]:
    with np.load(ADP3_DATA / f"visible_{split}.npz") as z:
        return {k: z[k] for k in z.files}


def adp3_hidden(split: str) -> dict[str, np.ndarray]:
    with np.load(ADP3_DATA / "hidden_gt.npz") as z:
        prefix = split + "_"
        return {k[len(prefix):]: z[k] for k in z.files if k.startswith(prefix)}


def normalized_distance(a: np.ndarray, b: np.ndarray, eps: float) -> float:
    return float(np.linalg.norm(a - b) / (0.5 * (np.linalg.norm(a) + np.linalg.norm(b)) + eps))


def components(distance: np.ndarray, tau: float) -> list[list[int]]:
    n = len(distance); seen = np.zeros(n, dtype=bool); result = []
    for root in range(n):
        if seen[root]: continue
        stack = [root]; seen[root] = True; comp = []
        while stack:
            i = stack.pop(); comp.append(i)
            for j in np.flatnonzero((distance[i] < tau) & (~seen)):
                seen[j] = True; stack.append(int(j))
        result.append(sorted(comp))
    return result


def stage_0(force: bool = False) -> dict[str, Any]:
    c = cfg(); path = OUT / "adp4_0" / "final_metrics.json"
    if path.exists() and not force: return json.loads(path.read_text(encoding="utf-8"))
    within, between, structure_ok = [], [], True
    split_rows = {}
    for split in ("train", "val", "test"):
        vis, hid = adp3_visible(split), adp3_hidden(split)
        group_stats = []
        for g in np.unique(vis["group_id"]):
            ix = np.flatnonzero(vis["group_id"] == g)
            labels = hid["y_gt"][ix]; d = vis["d"][ix]
            counts = [int(np.sum(labels == k)) for k in range(3)]
            structure_ok &= len(ix) == 12 and counts == [4, 4, 4]
            w, b = [], []
            for i in range(len(ix)):
                for j in range(i + 1, len(ix)):
                    dist = float(np.linalg.norm(d[i] - d[j]))
                    (w if labels[i] == labels[j] else b).append(dist)
            within.extend(w); between.extend(b)
            group_stats.append({"group_id": int(g), "family_counts": counts,
                                "within_family_max_distance": max(w), "between_family_min_distance": min(b)})
        split_rows[split] = {"num_groups": len(group_stats),
                             "all_3x4": all(x["family_counts"] == [4,4,4] for x in group_stats),
                             "within_family_max_distance": max(x["within_family_max_distance"] for x in group_stats),
                             "between_family_min_distance": min(x["between_family_min_distance"] for x in group_stats)}
    max_within, min_between = max(within), min(between)
    result = {**meta("adp4_0", c), "splits": split_rows, "all_bases_3x4": bool(structure_ok),
              "within_family_max_distance": max_within, "between_family_min_distance": min_between,
              "separation_ratio_within_over_between": max_within / max(min_between, 1e-30),
              "pass": bool(structure_ok and max_within < 1e-6 * min_between)}
    dump(path, result)
    (path.parent / "summary.md").write_text(
        f"# ADP4-0\n\n- PASS: **{result['pass']}**\n- within max: {max_within}\n- between min: {min_between}\n- ratio: {result['separation_ratio_within_over_between']}\n",
        encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("within_family_max_distance", "between_family_min_distance", "separation_ratio_within_over_between", "pass")}, indent=2), flush=True)
    return result


def estimate_tau(c: dict[str, Any]) -> tuple[float, np.ndarray]:
    vis = adp3_visible("train"); nearest = []
    for g in np.unique(vis["group_id"]):
        ix = np.flatnonzero(vis["group_id"] == g); d = vis["d"][ix]
        dm = np.full((len(ix), len(ix)), np.inf)
        for i in range(len(ix)):
            for j in range(i + 1, len(ix)):
                dm[i,j] = dm[j,i] = normalized_distance(d[i], d[j], c["unit_epsilon"])
        nearest.extend(dm.min(axis=1).tolist())
    nearest = np.asarray(nearest)
    tau = max(c["unit_tau_floor"], c["unit_tau_multiplier"] * float(np.quantile(nearest, .999)))
    return tau, nearest


def construct_units(split: str, tau: float, c: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    vis, hid = adp3_visible(split), adp3_hidden(split)
    S_rows=[]; d_rows=[]; var_rows=[]; group_rows=[]; member_rows=[]
    y_rows=[]; v_rows=[]; purity=[]; completeness=[]; unit_counts=[]
    unit_id=0
    for new_g, g in enumerate(np.unique(vis["group_id"])):
        ix = np.flatnonzero(vis["group_id"] == g); d = vis["d"][ix]
        dm = np.zeros((len(ix), len(ix)), dtype=np.float64)
        for i in range(len(ix)):
            for j in range(i + 1, len(ix)):
                dm[i,j] = dm[j,i] = normalized_distance(d[i], d[j], c["unit_epsilon"])
        comps = components(dm, tau); unit_counts.append(len(comps))
        for comp in comps:
            members = ix[np.asarray(comp)]
            labels = hid["y_gt"][members]
            counts = np.bincount(labels, minlength=3); label = int(np.argmax(counts))
            S_rows.append(vis["S"][members[0]]); d_rows.append(vis["d"][members].mean(axis=0)); var_rows.append(vis["d"][members].var(axis=0))
            group_rows.append(new_g); member_rows.append(vis["fragment_id"][members])
            y_rows.append(label); v_rows.append(float(hid["v_gt"][members[0]]))
            purity.append(float(counts.max()/len(members))); completeness.append(float(counts.max()/max(int(np.sum(hid["y_gt"][ix] == label)),1)))
            unit_id += 1
    max_members = max(len(x) for x in member_rows)
    member_matrix = np.full((len(member_rows), max_members), -1, dtype=np.int32)
    for i,x in enumerate(member_rows): member_matrix[i,:len(x)] = x
    learner = {"S": np.asarray(S_rows,np.float32), "d_mean": np.asarray(d_rows,np.float32), "d_variance": np.asarray(var_rows,np.float32),
               "group_id": np.asarray(group_rows,np.int32), "member_fragment_ids": member_matrix}
    hidden = {"y_gt": np.asarray(y_rows,np.int8), "v_gt": np.asarray(v_rows,np.float32)}
    metrics = {"num_units": len(S_rows), "fraction_bases_exactly_3": float(np.mean(np.asarray(unit_counts)==3)),
               "mean_unit_purity": float(np.mean(purity)), "mean_unit_completeness": float(np.mean(completeness)),
               "unit_count_distribution": {str(k): int(np.sum(np.asarray(unit_counts)==k)) for k in sorted(set(unit_counts))}}
    return learner, hidden, metrics


def stage_a0(force: bool = False) -> dict[str, Any]:
    c=cfg(); gate=stage_0(False)
    if not gate["pass"]: raise RuntimeError("STOP: ADP4-0 failed")
    path=OUT/"a0_local_units"/"final_metrics.json"
    if path.exists() and not force: return json.loads(path.read_text(encoding="utf-8"))
    tau, nearest=estimate_tau(c); metrics={}; hidden_all={}; UNIT_DATA.mkdir(parents=True,exist_ok=True)
    for split in ("train","val","test"):
        learner, hidden, sm=construct_units(split,tau,c)
        np.savez_compressed(UNIT_DATA/f"visible_{split}.npz",**learner)
        for k,v in hidden.items(): hidden_all[f"{split}_{k}"]=v
        metrics[split]=sm
    np.savez_compressed(UNIT_DATA/"hidden_gt.npz",**hidden_all)
    passed=all(x["fraction_bases_exactly_3"]>=.99 and x["mean_unit_purity"]>.99 and x["mean_unit_completeness"]>.99 for x in metrics.values())
    result={**meta("adp4_a0",c),"tau_unit":tau,"nearest_duplicate_error_q999":float(np.quantile(nearest,.999)),"splits":metrics,"pass":bool(passed)}
    dump(path,result); dump(path.parent/"config.json",c)
    (path.parent/"summary.md").write_text(f"# ADP4-A0\n\n- PASS: **{passed}**\n- tau_unit: {tau}\n- splits: `{json.dumps(metrics)}`\n",encoding="utf-8")
    print(json.dumps({"tau_unit":tau,"splits":metrics,"pass":passed},indent=2),flush=True); return result


def load_units(split: str) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    with np.load(UNIT_DATA/f"visible_{split}.npz") as z: return z["S"],z["d_mean"],z["group_id"]


def load_unit_hidden(split: str) -> dict[str,np.ndarray]:
    with np.load(UNIT_DATA/"hidden_gt.npz") as z:
        p=split+"_"; return {k[len(p):]:z[k] for k in z.files if k.startswith(p)}


def trainable(bank: torch.nn.Module, yes: bool) -> None:
    for p in bank.parameters(): p.requires_grad_(yes)


def joint_assign(errors: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    assignment=np.empty(len(errors),dtype=np.int16); chosen_perm=np.empty((int(groups.max())+1,3),dtype=np.int16)
    for g in np.unique(groups):
        ix=np.flatnonzero(groups==g)
        if len(ix)!=3: raise ValueError(f"group {g} has {len(ix)} units")
        costs=errors[ix[:,None],INJECTIVE.T[None,:,:]].sum(axis=0) if False else None
        # Explicit [60,3] gather keeps the unit axis aligned and auditable.
        costs=np.asarray([sum(errors[ix[a],perm[a]] for a in range(3)) for perm in INJECTIVE])
        best=int(np.argmin(costs)); assignment[ix]=INJECTIVE[best]; chosen_perm[int(g)]=INJECTIVE[best]
    return assignment,chosen_perm


def unit_e_step(bank:torch.nn.Module,S:np.ndarray,d:np.ndarray,groups:np.ndarray,c:dict[str,Any],seed:int)->dict[str,Any]:
    a=c["a1"]; n=len(S); k=c["m_max"]; R=a["inner_restarts"]
    errors=np.empty((n,k),np.float32); values=np.empty((n,k),np.float32); restart_errors=[np.empty((n,k),np.float32) for _ in range(R)]
    trainable(bank,False); bank.eval(); started=time.time()
    for start in range(0,n,a["e_step_chunk_size"]):
        stop=min(n,start+a["e_step_chunk_size"]); b=stop-start
        gen=torch.Generator(device=DEVICE); gen.manual_seed(seed+start*1009)
        z=(.35*torch.randn((R,b,k),generator=gen,device=DEVICE)).requires_grad_(True); opt=torch.optim.Adam([z],lr=a["inner_learning_rate"])
        St=torch.from_numpy(S[start:stop]).to(DEVICE).repeat(R,1,1); dt=torch.from_numpy(d[start:stop]).to(DEVICE).repeat(R,1,1)
        for _ in range(a["inner_steps"]):
            opt.zero_grad(set_to_none=True); v=a["v_bound"]*torch.tanh(z.reshape(R*b,k)); pred=bank.raw_effects(St,v)
            loss=(pred-dt[:,None]).square().mean(dim=(2,3)); loss.sum().backward(); opt.step()
        with torch.no_grad():
            v=a["v_bound"]*torch.tanh(z.reshape(R*b,k)); pred=bank.raw_effects(St,v)
            er=(pred-dt[:,None]).square().mean(dim=(2,3)).reshape(R,b,k); vv=v.reshape(R,b,k)
            rr=er.argmin(dim=0); bi=torch.arange(b,device=DEVICE)[:,None]; kj=torch.arange(k,device=DEVICE)[None,:]
            errors[start:stop]=er[rr,bi,kj].cpu().numpy(); values[start:stop]=vv[rr,bi,kj].cpu().numpy()
            for r in range(R): restart_errors[r][start:stop]=er[r].cpu().numpy()
    assignment,perms=joint_assign(errors,groups); chosen_v=values[np.arange(n),assignment]
    restart_perms=[joint_assign(x,groups)[1] for x in restart_errors]
    agreement=float(np.mean(np.all(restart_perms[0]==restart_perms[1],axis=1))) if R==2 else float("nan")
    trainable(bank,True)
    return {"assignment":assignment,"permutations":perms,"v":chosen_v,"v_all":values,"errors":errors,
            "chosen_error":errors[np.arange(n),assignment],"base_exact_agreement":agreement,"seconds":time.time()-started}


def m_step(bank,S,d,assignment,v,c,opt,round_id)->dict[str,float]:
    a=c["a1"]; rng=np.random.default_rng(c["optimization_seed"]+50021*(round_id+1)); losses=[]; started=time.time(); bank.train()
    for _ in range(a["m_step_epochs_per_round"]):
        order=rng.permutation(len(S))
        for start in range(0,len(S),a["m_step_batch_size"]):
            ix=order[start:start+a["m_step_batch_size"]]
            St=torch.from_numpy(S[ix]).to(DEVICE); dt=torch.from_numpy(d[ix]).to(DEVICE); ct=torch.from_numpy(assignment[ix].astype(np.int64)).to(DEVICE); vt=torch.from_numpy(v[ix]).to(DEVICE)
            va=torch.zeros((len(ix),c["m_max"]),device=DEVICE); va.scatter_(1,ct[:,None],vt[:,None]); opt.zero_grad(set_to_none=True)
            raw=bank.raw_effects(St,va); pred=raw[torch.arange(len(ix),device=DEVICE),ct]; loss=(pred-dt).square().mean(); loss.backward()
            torch.nn.utils.clip_grad_norm_(bank.parameters(),a["gradient_clip"]); opt.step(); losses.append(float(loss.detach().cpu()))
    return {"seconds":time.time()-started,"mean_minibatch_mse":float(np.mean(losses))}


@torch.no_grad()
def prediction(bank,S,assignment,v,batch=512)->np.ndarray:
    out=[]; bank.eval()
    for start in range(0,len(S),batch):
        stop=min(len(S),start+batch); St=torch.from_numpy(S[start:stop]).to(DEVICE); ct=torch.from_numpy(assignment[start:stop].astype(np.int64)).to(DEVICE); vt=torch.from_numpy(v[start:stop]).to(DEVICE)
        va=torch.zeros((stop-start,bank.m_max),device=DEVICE); va.scatter_(1,ct[:,None],vt[:,None]); raw=bank.raw_effects(St,va)
        out.append(raw[torch.arange(stop-start,device=DEVICE),ct].cpu().numpy())
    return np.concatenate(out)


def assignment_eval(func,assignment,hidden)->dict[str,Any]:
    contingency=np.zeros((5,3),dtype=np.int64)
    for x,y in zip(assignment,hidden["y_gt"]): contingency[int(x),int(y)]+=1
    lr,gc=linear_sum_assignment(-contingency); cmap={int(l):int(g) for l,g in zip(lr,gc)}
    acc=float(np.mean([cmap.get(int(x),-1)==int(y) for x,y in zip(assignment,hidden["y_gt"])]))
    fl,fg=linear_sum_assignment(-np.nan_to_num(func,nan=-1e6)); fmap={int(g):int(l) for l,g in zip(fl,fg)}
    matched=[float(func[fmap[g],g]) for g in range(3)]
    purity=[]
    for j in range(5): purity.append(float(contingency[j].max()/max(contingency[j].sum(),1)))
    sign_split=[]; sign_dist={}
    for gt in range(3):
        neg=assignment[(hidden["y_gt"]==gt)&(hidden["v_gt"]<0)]; pos=assignment[(hidden["y_gt"]==gt)&(hidden["v_gt"]>0)]
        pn=np.bincount(neg,minlength=5)/max(len(neg),1); pp=np.bincount(pos,minlength=5)/max(len(pos),1)
        sign_split.append(float(.5*np.abs(pn-pp).sum())); sign_dist[str(gt)]={"negative":pn.tolist(),"positive":pp.tolist()}
    abs_v=np.abs(hidden["v_gt"]); q=np.quantile(abs_v,[.25,.5,.75]); qbin=np.digitize(abs_v,q)
    magnitude={}
    for gt in range(3):
        magnitude[str(gt)]=[]
        for b in range(4):
            x=assignment[(hidden["y_gt"]==gt)&(qbin==b)]; magnitude[str(gt)].append((np.bincount(x,minlength=5)/max(len(x),1)).tolist())
    return {"unit_contingency_rows_candidate_cols_gt":contingency.tolist(),"assignment_candidate_to_gt":cmap,
            "unit_matching_accuracy":acc,"functional_gt_to_candidate":fmap,"matched_functional_r2":matched,
            "candidate_gt_family_purity":purity,"sign_split_per_gt":sign_split,"mean_sign_split":float(np.mean(sign_split)),
            "candidate_distribution_by_gt_and_sign":sign_dist,"candidate_distribution_by_gt_and_abs_v_quartile":magnitude,
            "abs_v_quartile_edges":q.tolist()}


def coverage_stats(S,assignment,pred)->tuple[list[Any],list[Any]]:
    rng=np.random.default_rng(20260904); flat=S.reshape(len(S),-1); weights=rng.normal(size=(6,2)); proj=(flat[:,:,None]*weights[None,:,:]).sum(axis=1); med=np.median(proj,axis=0); context=(proj[:,0]>med[0]).astype(int)*2+(proj[:,1]>med[1]).astype(int)
    coverage=[]; norms=[]
    en=np.sqrt(np.square(pred).sum(axis=(1,2)))
    for j in range(5):
        take=assignment==j; coverage.append({"contexts_present":int(len(np.unique(context[take]))),"state_min":S[take].min(axis=0).tolist() if take.any() else None,"state_max":S[take].max(axis=0).tolist() if take.any() else None})
        norms.append({"min":float(en[take].min()),"q25":float(np.quantile(en[take],.25)),"median":float(np.median(en[take])),"q75":float(np.quantile(en[take],.75)),"max":float(en[take].max())} if take.any() else None)
    return coverage,norms


def stage_a1(force:bool=False)->dict[str,Any]:
    c=cfg(); gate=stage_a0(False)
    if not gate["pass"]: raise RuntimeError("STOP: ADP4-A0 failed")
    out=OUT/"a1_group_matching"; final=out/"final_metrics.json"
    if final.exists() and not force: return json.loads(final.read_text(encoding="utf-8"))
    if (out/"round_metrics.jsonl").exists(): raise FileExistsError(f"Refusing overwrite: {out}")
    (out/"checkpoints").mkdir(parents=True,exist_ok=True); (out/"assignments").mkdir(parents=True,exist_ok=True); dump(out/"config.json",c)
    St,dt,gt=load_units("train"); Sv,dv,gv=load_units("val"); hv=load_unit_hidden("val")
    base.seed_everything(c["optimization_seed"]); bank=base.SetMechanismBank(5,c["mechanism_dim"],c["mechanism_heads"]).to(DEVICE)
    opt=torch.optim.AdamW(bank.parameters(),lr=c["a1"]["m_step_learning_rate"],weight_decay=c["a1"]["m_step_weight_decay"])
    previous=None; rows=[]; started=time.time(); torch.cuda.reset_peak_memory_stats() if DEVICE.type=="cuda" else None
    for r in range(c["a1"]["rounds"]):
        tr=unit_e_step(bank,St,dt,gt,c,c["optimization_seed"]+r*1000003); change=1.0 if previous is None else float(np.mean(np.any(tr["permutations"]!=previous,axis=1)))
        ms=m_step(bank,St,dt,tr["assignment"],tr["v"],c,opt,r); train_pred=prediction(bank,St,tr["assignment"],tr["v"]); train_mse=float(np.mean((train_pred-dt)**2))
        va=unit_e_step(bank,Sv,dv,gv,c,c["optimization_seed"]+70000001+r*1000003); val_pred=prediction(bank,Sv,va["assignment"],va["v"]); val_nrmse=base.nrmse(dv,val_pred)
        func=probe_functional_matrix(bank,va["v_all"],hv,c); ev=assignment_eval(func,va["assignment"],hv); counts=np.bincount(tr["assignment"],minlength=5); cov,norms=coverage_stats(Sv,va["assignment"],val_pred)
        torch.save(bank.state_dict(),out/"checkpoints"/f"round_{r:03d}_bank.pt"); np.savez_compressed(out/"assignments"/f"round_{r:03d}.npz",candidate=tr["assignment"],v=tr["v"],base_permutation=tr["permutations"],chosen_error=tr["chosen_error"])
        row={"round":r,"train_unit_mse":train_mse,"val_unit_nrmse":val_nrmse,"candidate_usage":(counts/len(St)).tolist(),"candidate_unit_count":counts.tolist(),
             "base_assignment_change_rate":change,"base_exact_agreement":tr["base_exact_agreement"],"functional_r2_matrix":func.tolist(),**ev,
             "candidate_state_coverage":cov,"candidate_effect_norm_range":norms,"alive":[True]*5,"num_alive":5,"merge_proposals":[],"merge_accepts":[],
             "comparison_graph_precision":None,"comparison_graph_recall":None,"comparison_graph_f1":None,
             "e_step_seconds":tr["seconds"]+va["seconds"],"m_step_seconds":ms["seconds"],"merge_seconds":0.0,"round_seconds":tr["seconds"]+va["seconds"]+ms["seconds"],
             "cuda_peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20 if DEVICE.type=="cuda" else 0.0}
        append_jsonl(out/"round_metrics.jsonl",row); rows.append(row); previous=tr["permutations"].copy()
        print(f"[A1] r={r:02d} nrmse={val_nrmse:.4f} acc={ev['unit_matching_accuracy']:.4f} func={np.round(ev['matched_functional_r2'],3).tolist()} split={ev['mean_sign_split']:.3f} change={change:.3f} usage={np.round(row['candidate_usage'],3).tolist()}",flush=True)
    last=rows[-1]; last3=float(np.mean([x["base_assignment_change_rate"] for x in rows[-3:]])); th=c["pass_thresholds"]
    checks={"unit_accuracy":last["unit_matching_accuracy"]>th["unit_hungarian_accuracy"],"functional_each":all(x>th["best_functional_r2_each"] for x in last["matched_functional_r2"]),
            "functional_two_strong":sum(x>0.85 for x in last["matched_functional_r2"])>=th["num_functional_r2_above_0_85"],"stability":last3<th["last3_base_assignment_change"],"sign_split":last["mean_sign_split"]<th["mean_sign_split"]}
    result={**meta("adp4_a1",c),"rounds_completed":len(rows),"last3_base_assignment_change":last3,"last_round":last,"checks":checks,"pass":bool(all(checks.values())),"stop_required":not all(checks.values()),"wall_seconds":time.time()-started,"cuda_peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20 if DEVICE.type=="cuda" else 0.0}
    dump(final,result); (out/"summary.md").write_text(f"# ADP4-A1\n\n- PASS: **{result['pass']}**\n- accuracy: {last['unit_matching_accuracy']}\n- functional R2: {last['matched_functional_r2']}\n- mean sign split: {last['mean_sign_split']}\n- last3 change: {last3}\n",encoding="utf-8")
    print(json.dumps({"pass":result["pass"],"checks":checks},indent=2),flush=True); return result


def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--stage",choices=["adp4_0","adp4_a0","adp4_a1"],required=True); p.add_argument("--seed",type=int,default=0); p.add_argument("--force",action="store_true"); a=p.parse_args()
    if a.seed!=0: raise ValueError("Development stage fixes seed 0")
    result={"adp4_0":stage_0,"adp4_a0":stage_a0,"adp4_a1":stage_a1}[a.stage](a.force)
    print(json.dumps({"stage":result["stage"],"pass":result["pass"]},indent=2),flush=True)


if __name__=="__main__": main()
