"""Local literature-method transfer battery (descriptive, paired-seven).

Reads only the local verified stage outputs and local raw ``data_repro_20260903``.
No model training or causal/biomarker inference is performed.
"""
from __future__ import annotations
import hashlib, itertools, json, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
RAW = ROOT / "data_repro_20260903"
STAGE = ROOT / "outputs/analysis/stage_20260904_eye_fnirs_imu_verified"
OUT = ROOT / "outputs/literature_method_transfer_20260904_local"
SUBJECTS = ["2026080901","2026081401","2026081403","2026081501","2026081502","2026081503","2026081701"]
TASKS = ["摸头","摸膝","摸腰"]

def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def signflip_p(x):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if len(x)<3:return np.nan
    obs=abs(x.mean()); null=[abs(np.mean(x*np.asarray(s))) for s in itertools.product([-1,1],repeat=len(x))]
    return float(np.mean(np.asarray(null)>=obs-1e-12))

def summarize_delta(df, key, value, label):
    df=df.copy()
    if "subject" in df: df["subject"]=df["subject"].astype(str)
    w=df.pivot_table(index=key,columns="session",values=value,aggfunc="mean")
    if not {"base_line","sec_test"}.issubset(w.columns): return pd.DataFrame()
    d=(w.sec_test-w.base_line)
    if isinstance(d.index,pd.MultiIndex): d=d.groupby(level=0).mean()
    d=d.reindex(SUBJECTS); x=d.to_numpy(float); x=x[np.isfinite(x)]
    return pd.DataFrame([{ "method":label,"feature":value,"n":len(x),"mean_delta":np.nanmean(x) if len(x) else np.nan,"median_delta":np.nanmedian(x) if len(x) else np.nan,"positive":int((x>0).sum()),"negative":int((x<0).sum()),"direction_fraction":max((x>0).sum(),(x<0).sum())/len(x) if len(x) else np.nan,"signflip_p":signflip_p(x)}])

def fnirs_methods(rows):
    # Recompute a three-task RSA from STAGE task-level fNIRS response vectors.
    sp=pd.read_csv(STAGE/"movement_subtask_summary.csv"); sp["subject"]=sp.subject.astype(str)
    vec=sp.pivot_table(index=["subject","session"],columns="task_name",values="mean_response_delta",aggfunc="mean").reindex(columns=TASKS)
    rsa_rows=[]
    for (sub,ses),g in vec.iterrows():
        v=g.to_numpy(float); sep=np.nan
        if np.isfinite(v).sum()==3:
            cs=[]
            for i in range(3):
                for j in range(i+1,3):
                    den=np.linalg.norm(v[[i,j]][:1])*np.linalg.norm(v[[i,j]][1:]); cs.append(1-(v[i]*v[j]/(abs(v[i])*abs(v[j])) if v[i] and v[j] else np.nan))
            sep=float(np.nanmean(cs))
        rsa_rows.append({"subject":sub,"session":ses,"sep":sep})
    rsa=pd.DataFrame(rsa_rows)
    # Scalar task summaries cannot form channel-wise representational geometry;
    # report RSA as unavailable and use explicit head-minus-knee contrasts below.
    rs=pd.DataFrame([{"method":"fNIRS RSA task separation","feature":"not_available_channel_level_vectors","n":0,"mean_delta":np.nan,"median_delta":np.nan,"positive":0,"negative":0,"direction_fraction":np.nan,"signflip_p":np.nan}])
    st=pd.read_csv(STAGE/"fnirs_event_states.csv")
    st["subject"]=st.subject.astype(str)
    met=[]
    for (sub,ses),g in st.groupby(["subject","session"]):
        s=g.sort_values("event_index").state.to_numpy(int); n=len(s); p=np.array([(s==1).mean(),(s==2).mean()]); ent=float(-(p[p>0]*np.log2(p[p>0])).sum()); sw=float(np.mean(s[1:]!=s[:-1])) if n>1 else np.nan
        trans=np.array([[np.sum((s[:-1]==i)&(s[1:]==j)) for j in (1,2)] for i in (1,2)],float) if n>1 else np.zeros((2,2)); tp=trans/trans.sum() if trans.sum() else trans; tp=tp[tp>0]; tent=float(-(tp*np.log2(tp)).sum()/2.0) if tp.size else 0.0
        met.append({"subject":sub,"session":ses,"occupancy_entropy_norm":ent,"switch_rate":sw,"transition_entropy_norm":tent})
    sm=pd.DataFrame(met)
    for f in ["occupancy_entropy_norm","switch_rate","transition_entropy_norm"]:
        z=summarize_delta(sm,["subject"],f,"fNIRS dynamic state")
        if len(z): rows.extend(z.to_dict("records"))
    if len(rs): rows.extend(rs.to_dict("records"))
    sp=sp.copy(); wide=sp.pivot_table(index=["subject","session"],columns="task_name",values="mean_response_delta",aggfunc="mean")
    if set(["摸头","摸膝"]).issubset(wide.columns):
        taskdelta=wide.reset_index(); taskdelta["head_minus_knee_response"]=taskdelta["摸头"]-taskdelta["摸膝"]; taskdelta["head_minus_knee_abs"]=taskdelta["head_minus_knee_response"].abs()
        for f in ["head_minus_knee_response","head_minus_knee_abs"]:
            z=summarize_delta(taskdelta,["subject"],f,"fNIRS task-space pair contrast")
            if len(z): rows.extend(z.to_dict("records"))
    for f in ["mean_response_delta","mean_left_right_asymmetry"]:
        z=summarize_delta(sp,["subject","task_name"],f,"fNIRS overall response proxy")
        if len(z): rows.extend(z.to_dict("records"))
    return rsa, st, sp

def emg_methods(rows):
    emg=pd.read_csv(STAGE/"movement_subtask_emg_summary.csv")
    emg["subject"]=emg.subject.astype(str)
    for f in ["mean_envelope","coactivation","mean_envelope_cv"]:
        for task in TASKS:
            z=summarize_delta(emg[emg.task_name==task],["subject"],f,f"EMG {task}")
            if len(z): z["feature"] = f; rows.extend(z.to_dict("records"))
    # block-dynamics descriptors already aggregate execution trends per subject/session/task
    b=emg.copy(); b["subject"]=b.subject.astype(str)
    return emg,b

def coupling(rows):
    c=pd.read_csv(STAGE/"movement_subtask_bridge_summary.csv")
    c["subject"]=c.subject.astype(str)
    c["brain_response_emg_abs_rho"]=pd.to_numeric(c["spearman_rho"],errors="coerce").abs()
    for f in ["brain_response_emg_abs_rho"]:
        z=summarize_delta(c,["subject","task_name"],f,"fNIRS-EMG same-task coupling")
        if len(z): rows.extend(z.to_dict("records"))
    return c

def fusion():
    d=pd.read_csv(STAGE/"motor_multimodal_features.csv"); d["subject"]=d["subject"].astype(str)
    qraw=d.groupby("subject").mean(numeric_only=True).reindex(SUBJECTS)
    # Build subject-level baseline->sec deltas from STAGE only.
    num=["fnirs_event_proxy_delta","fnirs_event_left_right_asymmetry","emg_global_coactivation","bridge_event_spearman_rho"]
    wide=d.pivot_table(index="subject",columns="session",values=[x for x in num if x in d],aggfunc="mean"); delta=pd.DataFrame(index=SUBJECTS)
    for f in num:
        if (f,"base_line") in wide and (f,"sec_test") in wide: delta[f]=wide[(f,"sec_test")]-wide[(f,"base_line")]
    d=delta
    qv=pd.DataFrame(index=SUBJECTS)
    def qc(name): return pd.to_numeric(qraw[name],errors="coerce").fillna(0) if name in qraw else pd.Series(0.0,index=SUBJECTS)
    qv["fnirs_quality_weight"]=(qc("fnirs_event_count_used")/15.0*qc("fnirs_sample_valid_coverage")).clip(0,1)
    qv["emg_quality_weight"]=(qc("emg_mapping_valid_for_primary_features")*qc("emg_coactivation_muscle_count")/7.0).clip(0,1)
    qv["emg_bridge_quality_weight"]=(qc("bridge_event_pairs")/5.0).clip(0,1)
    groups={"fNIRS":["fnirs_event_proxy_delta","fnirs_event_left_right_asymmetry"],"EMG":["emg_global_coactivation"],"bridge":["bridge_event_spearman_rho"]}
    scores={}
    for name,cols in groups.items():
        cols=[c for c in cols if c in d]
        z=d[cols].apply(lambda x:(x-x.mean())/(x.std(ddof=1) or 1),axis=0)
        scores[name]=z.mean(axis=1)
    weights=pd.DataFrame({"fNIRS":qv.fnirs_quality_weight,"EMG":qv.emg_quality_weight,"bridge":qv.emg_bridge_quality_weight})
    raw=pd.DataFrame(scores); weighted=(raw*weights).sum(axis=1)/weights.where(raw.notna()).sum(axis=1)
    rows=[]
    for name,s in {"late_equal":raw.mean(axis=1),"late_quality_weighted":weighted}.items():
        x=s.to_numpy(float); loo=[]
        for i in range(len(x)):
            keep=np.arange(len(x))!=i; loo.append(float(np.corrcoef(x[keep],s.to_numpy(float)[keep])[0,1]))
        n=int(np.isfinite(x).sum()); rows.append({"fusion":name,"n":n,"mean_score":float(np.nanmean(x)) if n else np.nan,"positive":int((x>0).sum()),"negative":int((x<0).sum()),"direction_fraction":float(max((x>0).sum(),(x<0).sum())/n) if n else np.nan,"signflip_p":signflip_p(x),"loo_self_corr_min":float(np.nanmin(loo)) if loo else np.nan,"loo_self_corr_median":float(np.nanmedian(loo)) if loo else np.nan})
    pd.DataFrame({"subject":SUBJECTS,**scores,"quality_fnirs":weights.fnIRS if False else weights["fNIRS"],"quality_emg":weights.EMG,"quality_bridge":weights.bridge,"late_equal":raw.mean(axis=1),"late_quality_weighted":weighted}).to_csv(OUT/"fusion_subject_scores.csv",index=False,encoding="utf-8-sig")
    return pd.DataFrame(rows)

def mbll_audit():
    import fnirs_qc
    rows=[]
    for p in sorted(RAW.rglob("*.nirs")):
        try:
            r=fnirs_qc.read_fnirs(p,None); waves=sorted({c.wavelength_a_nm for c in r.channels}|{c.wavelength_b_nm for c in r.channels})
            d=[]
            for c in r.channels:
                s=r.source_positions[c.source-1,:2]; t=r.detector_positions[c.detector-1,:2]; d.append(float(np.linalg.norm(s-t)))
            # Standard two-wavelength coefficient matrix; reported only as a relative sensitivity audit.
            E=np.array([[0.3896,1.79],[2.03,1.22]])
            rows.append({"file":str(p.relative_to(RAW)),"wavelengths_nm":";".join(map(str,waves)),"n_channels":len(r.channels),"distance_median_raw":float(np.median(d)),"distance_min_raw":float(np.min(d)),"coefficient_matrix_condition":float(np.linalg.cond(E)),"status":"relative_MBLL_matrix_only","reason":"DPF, extinction-coefficient provenance, coordinate units and calibrated optical path are undeclared; no absolute HbO/HbR claim"})
        except Exception as e: rows.append({"file":str(p.relative_to(RAW)),"status":"not_run","reason":str(e)})
    return pd.DataFrame(rows)

def association_summary():
    s=pd.read_csv(STAGE/"movement_subtask_summary.csv"); s["subject"]=s.subject.astype(str)
    w=s.pivot_table(index=["subject","session"],columns="task_name",values="mean_response_delta",aggfunc="mean"); w["head_minus_knee_response"]=w["摸头"]-w["摸膝"]; w["overall_fnirs"]=w[["摸头","摸膝","摸腰"]].mean(axis=1)
    e=pd.read_csv(STAGE/"movement_subtask_emg_summary.csv"); e["subject"]=e.subject.astype(str); eg=e.groupby(["subject","session"]).coactivation.mean().rename("emg_global_coactivation")
    st=pd.read_csv(STAGE/"fnirs_event_states.csv"); st["subject"]=st.subject.astype(str)
    def oe(x):
        p=np.array([(x==1).mean(),(x==2).mean()]); p=p[p>0]; return float(-(p*np.log2(p)).sum())
    occ=st.groupby(["subject","session"]).state.apply(oe).rename("occupancy_entropy_norm")
    b=pd.read_csv(STAGE/"movement_subtask_bridge_summary.csv"); b["subject"]=b.subject.astype(str); coup=b.groupby(["subject","session"]).spearman_rho.mean().abs().rename("coupling_abs_rho")
    z=pd.concat([w[["head_minus_knee_response","overall_fnirs"]],eg,occ,coup],axis=1); delta=z.xs("sec_test",level=1)-z.xs("base_line",level=1)
    pairs=[("head_minus_knee_response","emg_global_coactivation"),("occupancy_entropy_norm","head_minus_knee_response"),("overall_fnirs","emg_global_coactivation"),("coupling_abs_rho","emg_global_coactivation")]; out=[]
    for a,bn in pairs:
        x,y=delta[a],delta[bn]; good=x.notna()&y.notna(); x=x[good].to_numpy(float); y=y[good].to_numpy(float); n=len(x)
        if n<4 or np.std(x)==0 or np.std(y)==0: continue
        rho=float(spearmanr(x,y).statistic); vals=[]
        for perm in itertools.permutations(y): vals.append(abs(float(spearmanr(x,perm).statistic)))
        p=float(np.mean(np.asarray(vals)>=abs(rho)-1e-12)); loo=[]
        for i in range(n): loo.append(float(spearmanr(np.delete(x,i),np.delete(y,i)).statistic))
        out.append({"x":a,"y":bn,"n":n,"spearman_rho":rho,"exact_permutation_p":p,"loo_rho_min":min(loo),"loo_rho_max":max(loo),"loo_sign_stable":bool(all(np.sign(v)==np.sign(rho) or v==0 for v in loo))})
    return pd.DataFrame(out)

def main():
    OUT.mkdir(parents=True,exist_ok=True); method=[]
    rsa,st,sp=fnirs_methods(method); emg,b=emg_methods(method); c=coupling(method); fs=fusion(); mb=mbll_audit()
    pd.DataFrame(method).to_csv(OUT/"method_summary.csv",index=False,encoding="utf-8-sig"); fs.to_csv(OUT/"fusion_summary.csv",index=False,encoding="utf-8-sig"); mb.to_csv(OUT/"mbll_sensitivity_audit.csv",index=False,encoding="utf-8-sig"); association_summary().to_csv(OUT/"association_summary.csv",index=False,encoding="utf-8-sig")
    # Compact overview figure
    m=pd.DataFrame(method); fig,ax=plt.subplots(figsize=(10,4)); mm=m.groupby("method").mean(numeric_only=True).sort_values("mean_delta"); ax.barh(mm.index,mm.mean_delta,color=["#2a9d8f" if x>=0 else "#e76f51" for x in mm.mean_delta]); ax.axvline(0,color="0.4",lw=.8); ax.set_xlabel("mean paired descriptor / delta (descriptive)"); fig.tight_layout(); fig.savefig(OUT/"method_overview.png",dpi=300); plt.close(fig)
    files=[STAGE/"fnirs_event_states.csv",STAGE/"movement_subtask_emg_summary.csv",STAGE/"movement_subtask_summary.csv",STAGE/"movement_subtask_bridge_summary.csv",STAGE/"motor_multimodal_features.csv"]
    raw_files=sorted(RAW.rglob("*.nirs")); raw_manifest=[{"path":str(p.relative_to(RAW)),"sha256":sha256(p)} for p in raw_files]
    meta={"raw_root":str(RAW.resolve()),"stage_root":str(STAGE.resolve()),"subjects":SUBJECTS,"method_script_sha256":sha256(Path(__file__)),"input_sha256":{str(p):sha256(p) for p in files if p.exists()},"raw_nirs_manifest":raw_manifest,"command":"python scripts/literature_method_transfer_local.py","fusion_quality_formula":{"fnirs":"clip(fnirs_event_count_used/15 * fnirs_sample_valid_coverage,0,1)","emg":"clip(emg_mapping_valid_for_primary_features * emg_coactivation_muscle_count/7,0,1)","bridge":"clip(bridge_event_pairs/5,0,1)","missing":"0 (conservative)"},"mbll_note":"relative sensitivity audit only"}
    (OUT/"metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"README.md").write_text("# Literature method transfer battery (local)\n\nAll tabular inputs are from the local verified stage `outputs/analysis/stage_20260904_eye_fnirs_imu_verified`; raw fNIRS audit files are under `data_repro_20260903`. This is a descriptive migration-validity experiment on the paired-seven subset. Independent unit is subject; blocks/events are reduced before comparison. Outputs cover fNIRS task-space contrasts and dynamic states, same-task fNIRS-EMG coupling, EMG envelope/coactivation/CV, transparent late fusion with LOO self-stability and exact sign-flip permutation, and an MBLL sensitivity audit. Fusion weights: fNIRS=clip(fnirs_event_count_used/15 * fnirs_sample_valid_coverage,0,1); EMG=clip(emg_mapping_valid_for_primary_features * emg_coactivation_muscle_count/7,0,1); bridge=clip(bridge_event_pairs/5,0,1); missing values are zero. Channel-level RSA is marked unavailable because STAGE has no channel-wise task vectors. No FGANet/GIN/closed-loop training, causal claims or biomarker claims are made. MBLL is relative matrix-only because DPF, coefficient provenance, coordinate units and calibrated optical path are undeclared. See `method_summary.csv`, `fusion_summary.csv`, `mbll_sensitivity_audit.csv`, `metadata.json`.\n",encoding="utf-8")
    print(f"Wrote {len(method)} method rows, fusion={len(fs)}, MBLL files={len(mb)} to {OUT}")

if __name__=="__main__": main()
