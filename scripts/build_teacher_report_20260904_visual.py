# -*- coding: utf-8 -*-
"""Build the anonymized, STAGE-only visual teacher report for 2026-09-04."""
from __future__ import annotations
import argparse, csv, hashlib, itertools, json, math, os, shutil, subprocess, sys, tempfile
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
except ImportError:
    plt = None
    font_manager = None
try:
    from docx import Document
    from docx.shared import Cm, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
except ImportError:
    Document = None

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "outputs/analysis/stage_20260904_eye_fnirs_imu_verified"
TRANSFER = ROOT / "outputs/literature_method_transfer_20260904_local"
PREPROCESS = ROOT / "outputs/preprocessed/current/preprocessing_inventory.csv"
OUT = ROOT / "outputs/teacher_report_20260904_visual"
FIG = OUT / "figures"
SUBJECTS = ["2026080901", "2026081401", "2026081403", "2026081501", "2026081502", "2026081503", "2026081701"]
LABELS = {sid: f"S{i:02d}" for i, sid in enumerate(SUBJECTS, 1)}
TASKS = ["摸头", "摸膝", "摸腰"]
FONT = "Noto Sans SC"
PLOT_FONT_NAME = FONT
CJK_FONT_CANDIDATES = [Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"), Path(r"C:\Windows\Fonts\Deng.ttf"), Path(r"C:\Windows\Fonts\simhei.ttf")]
COLORS = {"fnirs": "#1f6f8b", "emg": "#c65d3b", "bridge": "#5d7f55", "warn": "#b8860b", "ink": "#263238", "muted": "#718096"}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()

def resolve_cjk_font() -> Path:
    for path in CJK_FONT_CANDIDATES:
        if path.exists() and path.stat().st_size > 0:
            return path
    checked = "; ".join(str(p) for p in CJK_FONT_CANDIDATES)
    raise RuntimeError(f"未找到可用 CJK 字体，无法安全生成中文图文报告。检查：{checked}")

def setup_plotting():
    global PLOT_FONT_NAME
    if plt is None: return
    font_path=resolve_cjk_font()
    font_manager.fontManager.addfont(str(font_path))
    PLOT_FONT_NAME=font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"font.family": PLOT_FONT_NAME, "axes.unicode_minus": False, "svg.fonttype": "none", "figure.facecolor": "white", "axes.facecolor": "white"})

def save(fig, name: str):
    for ext, kwargs in (("png", {"dpi": 220}), ("svg", {})):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)

def paired_delta(frame: pd.DataFrame, value: str, group: Optional[List[str]] = None) -> pd.Series:
    keys = ["subject", "session"] + (group or [])
    w = frame.pivot_table(index=["subject"] + (group or []), columns="session", values=value, aggfunc="mean")
    d = w["sec_test"] - w["base_line"]
    return d.groupby(level="subject").mean().reindex(SUBJECTS)

def state_metrics(states: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (subject, session), g in states.groupby(["subject", "session"]):
        seq=g.sort_values("event_index").state.to_numpy(int)
        p=np.array([(seq==1).mean(), (seq==2).mean()]); p=p[p>0]
        rows.append({"subject": str(subject), "session": session, "occupancy_entropy_norm": float(-(p*np.log2(p)).sum()), "switch_rate": float(np.mean(seq[1:] != seq[:-1])) if len(seq)>1 else np.nan})
    return pd.DataFrame(rows)

def association_sources(stage: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    task = pd.read_csv(stage / "movement_subtask_summary.csv"); task.subject=task.subject.astype(str)
    task_w=task.pivot_table(index=["subject","session"], columns="task_name", values="mean_response_delta", aggfunc="mean")
    task_w["head_minus_knee_response"] = task_w["摸头"] - task_w["摸膝"]
    task_w["overall_fnirs"] = task_w[TASKS].mean(axis=1)
    emg=pd.read_csv(stage / "movement_subtask_emg_summary.csv"); emg.subject=emg.subject.astype(str)
    emg_s=emg.groupby(["subject","session"]).coactivation.mean().rename("emg_global_coactivation")
    state=state_metrics(pd.read_csv(stage / "fnirs_event_states.csv"))
    state_s=state.set_index(["subject","session"]).occupancy_entropy_norm
    bridge=pd.read_csv(stage / "movement_subtask_bridge_summary.csv"); bridge.subject=bridge.subject.astype(str)
    bridge_s=bridge.groupby(["subject","session"]).spearman_rho.mean().abs().rename("coupling_abs_rho")
    z=pd.concat([task_w[["head_minus_knee_response","overall_fnirs"]], emg_s, state_s, bridge_s], axis=1)
    d=(z.xs("sec_test", level="session") - z.xs("base_line", level="session")).reindex(SUBJECTS)
    d.index.name="subject"
    d.insert(0,"anonymous_id",[LABELS[s] for s in d.index])
    return d, state

def plot_workflow():
    fig, ax=plt.subplots(figsize=(10.8,4.2)); ax.set_xlim(0,10); ax.set_ylim(0,4); ax.axis("off")
    nodes=[(0.5,2.65,"本地原始数据\n7 名配对受试者",COLORS["ink"]),(3.0,2.65,"任务/信号/对齐\n三层质量门控",COLORS["warn"]),(5.5,2.65,"受试者级特征\nfNIRS | EMG | bridge",COLORS["fnirs"]),(8.0,2.65,"探索性迁移\n趋势、关联、融合",COLORS["bridge"])]
    for x,y,t,c in nodes:
        ax.text(x+0.8,y,t,ha="center",va="center",fontsize=12,color="white",bbox=dict(boxstyle="round,pad=.65",facecolor=c,edgecolor="none"))
    for x in (2.25,4.75,7.25): ax.annotate("",xy=(x+.55,2.65),xytext=(x,2.65),arrowprops=dict(arrowstyle="->",lw=1.8,color=COLORS["muted"]))
    ax.text(5,0.85,"独立单位：受试者；Δ = sec_test − base_line\n事件/任务窗只用于受试者内汇总，不作为独立样本",ha="center",va="center",fontsize=12,color=COLORS["ink"],bbox=dict(boxstyle="round,pad=.55",facecolor="#edf3f5",edgecolor="#cbd5e0"))
    save(fig,"01_workflow")

def plot_coverage(emg: pd.DataFrame, bridge: pd.DataFrame):
    inventory=pd.read_csv(PREPROCESS)
    inventory["modality"]=inventory.modality.astype(str).str.upper()
    core=inventory[inventory.subject.astype(str).isin(SUBJECTS)]
    statuses=[
        ("EEG", int(((core.modality=="EEG") & (core.status=="READY_WITH_MASKS")).sum()), int(((core.modality=="EEG") & (core.status=="FAILED")).sum()), 0, "就绪（带 mask）", "失败未纳入", ""),
        ("fNIRS", int(((core.modality=="FNIRS") & (core.analysis_branch=="STANDARD_MASKED")).sum()), 0, int(((core.modality=="FNIRS") & (core.analysis_branch=="EYE_IR_DEGRADED_MASKED")).sum()), "标准 mask", "", "眼动 IR 降级 mask"),
        ("EMG", int(((core.modality=="EMG") & (core.status=="READY_CONTINUOUS")).sum()), 0, 0, "连续就绪", "", ""),
    ]
    paired_emg=emg[emg.subject.astype(str).isin(SUBJECTS)]; paired_bridge=bridge[bridge.subject.astype(str).isin(SUBJECTS)]
    cards=[("配对受试者",len(SUBJECTS),COLORS["fnirs"]),("配对访视",14,COLORS["fnirs"]),("EMG 任务窗",len(paired_emg),COLORS["emg"]),("脑-肌 bridge",len(paired_bridge),COLORS["bridge"])]
    fig=plt.figure(figsize=(11.2,6.1)); grid=fig.add_gridspec(2,1,height_ratios=[.9,1.25],hspace=.38)
    ax=fig.add_subplot(grid[0]); ax.axis("off")
    for i,(label,value,color) in enumerate(cards):
        x=.015+i*.247
        ax.add_patch(plt.Rectangle((x,.12),.225,.76,facecolor="#f7fafc",edgecolor=color,lw=1.5,transform=ax.transAxes))
        ax.text(x+.112,.60,str(value),ha="center",va="center",fontsize=23,fontweight="bold",color=color,transform=ax.transAxes)
        ax.text(x+.112,.30,label,ha="center",va="center",fontsize=10.5,color=COLORS["ink"],transform=ax.transAxes)
    ax.text(.5,.01,"四个数字来自不同记录层级；推断和趋势图的独立单位始终为受试者。",ha="center",fontsize=10,color=COLORS["muted"],transform=ax.transAxes)
    ax=fig.add_subplot(grid[1]); y=np.arange(len(statuses)); ready=np.array([r[1] for r in statuses]); failed=np.array([r[2] for r in statuses]); degraded=np.array([r[3] for r in statuses])
    ax.barh(y,ready,color=COLORS["fnirs"],height=.5,label="主分支可用")
    ax.barh(y,failed,left=ready,color="#c94c4c",height=.5,label="失败，未纳入")
    ax.barh(y,degraded,left=ready+failed,color="#d9a441",height=.5,label="降级，保留敏感性")
    for i,(mod,a,b,c,a_label,b_label,c_label) in enumerate(statuses):
        text=f"{a} {a_label}" + (f"  +  {b} {b_label}" if b else "") + (f"  +  {c} {c_label}" if c else "")
        ax.text(a+b+c+.45,i,text,va="center",fontsize=10,color=COLORS["ink"])
    ax.set_yticks(y,[r[0] for r in statuses],fontsize=11); ax.invert_yaxis(); ax.set_xlim(0,max(ready+failed+degraded)+15)
    ax.set_xlabel("预处理清单中的记录数（与上方分析层级不同）")
    ax.set_title("预处理状态：EYE_IR_DEGRADED_MASKED 保留为敏感性分支",loc="left",fontsize=12,pad=8)
    ax.spines[["top","right","left"]].set_visible(False); ax.grid(axis="x",alpha=.16); ax.legend(loc="lower right",frameon=False,fontsize=9)
    save(fig,"02_coverage_quality")

def trend_sources(stage: Path) -> pd.DataFrame:
    task=pd.read_csv(stage / "movement_subtask_summary.csv"); task.subject=task.subject.astype(str)
    task_w=task.pivot_table(index=["subject","session"],columns="task_name",values="mean_response_delta",aggfunc="mean")
    task_w["head_minus_knee_response"]=task_w["摸头"]-task_w["摸膝"]
    task_delta=task_w.head_minus_knee_response.unstack("session").sec_test-task_w.head_minus_knee_response.unstack("session").base_line
    state=state_metrics(pd.read_csv(stage / "fnirs_event_states.csv")).set_index(["subject","session"])
    state_delta=state.occupancy_entropy_norm.unstack("session").sec_test-state.occupancy_entropy_norm.unstack("session").base_line
    emg=pd.read_csv(stage / "movement_subtask_emg_summary.csv"); emg.subject=emg.subject.astype(str)
    emg_head=emg[emg.task_name.eq("摸头")].groupby(["subject","session"]).mean_envelope_cv.mean()
    emg_delta=emg_head.unstack("session").sec_test-emg_head.unstack("session").base_line
    bridge=pd.read_csv(stage / "movement_subtask_bridge_summary.csv"); bridge.subject=bridge.subject.astype(str)
    bridge["brain_response_emg_abs_rho"]=pd.to_numeric(bridge.spearman_rho,errors="coerce").abs()
    # Aggregate within each subject/session across available task-level bridge rows,
    # then form the paired delta. These are the exact values shown in the panel.
    bridge_session=bridge.groupby(["subject","session","task_name"]).brain_response_emg_abs_rho.mean().groupby(level=[0,1]).mean()
    bridge_delta=bridge_session.unstack("session").sec_test-bridge_session.unstack("session").base_line
    delta=pd.DataFrame({"head_minus_knee_response":task_delta,"occupancy_entropy_norm":state_delta,"mean_envelope_cv":emg_delta,"brain_response_emg_abs_rho":bridge_delta}).reindex(SUBJECTS)
    delta.insert(0,"anonymous_id",[LABELS[s] for s in delta.index])
    return delta

def exact_signflip_p(values: pd.Series) -> float:
    x=pd.to_numeric(values,errors="coerce").dropna().to_numpy(float)
    if len(x)<3: return float("nan")
    observed=abs(float(x.mean()))
    signs=np.asarray(list(itertools.product([-1.0,1.0],repeat=len(x))))
    return float(np.mean(np.abs((signs*x).mean(axis=1)) >= observed-1e-12))

def trend_statistics(values: pd.Series) -> dict:
    x=pd.to_numeric(values,errors="coerce").dropna().to_numpy(float)
    return {"n":int(len(x)),"mean":float(np.mean(x)) if len(x) else float("nan"),"median":float(np.median(x)) if len(x) else float("nan"),"positive":int((x>0).sum()),"negative":int((x<0).sum()),"zero":int((x==0).sum()),"signflip_p":exact_signflip_p(pd.Series(x))}

def plot_trends(trends: pd.DataFrame):
    focus=[("head_minus_knee_response","摸头−摸膝任务对比",COLORS["fnirs"]),("occupancy_entropy_norm","状态占用熵",COLORS["fnirs"]),("mean_envelope_cv","EMG 包络 CV（摸头）",COLORS["emg"]),("brain_response_emg_abs_rho","脑-肌 |rho|",COLORS["bridge"])]
    fig,axs=plt.subplots(2,2,figsize=(10.6,8.8));
    for ax,(feat,title,color) in zip(axs.ravel(),focus):
        q=trends[["anonymous_id",feat]].dropna(); ys=np.arange(len(q))
        stats=trend_statistics(q[feat]); val=stats["mean"]
        span=max(float(np.nanmax(np.abs(q[feat].to_numpy(float))))*1.22,.06)
        ax.axvline(0,color="#94a3b8",lw=.9); ax.hlines(ys,0,q[feat],color="#cbd5e0",lw=1.1); ax.scatter(q[feat],ys,color=color,s=48,zorder=3)
        ax.axvline(val,color=color,lw=1.4,ls="--"); ax.scatter([val],[len(q)+.12],color=color,s=52,marker="D",zorder=4,label="平均")
        ax.scatter([stats["median"]],[len(q)+.12],color=COLORS["ink"],s=42,marker="|",zorder=4,label="中位数")
        ax.set_xlim(-span,span); ax.set_ylim(-.7,len(q)+.65); ax.set_yticks(ys,list(q.anonymous_id),fontsize=8.5); ax.set_title(title,fontsize=12,pad=10)
        ax.text(.03,.95,f"n={stats['n']}; {stats['positive']}↑ / {stats['negative']}↓ / {stats['zero']}=0\n平均 {val:+.4g}；exact sign-flip p={stats['signflip_p']:.3g}",transform=ax.transAxes,ha="left",va="top",fontsize=8.8,bbox=dict(boxstyle="round,pad=.28",facecolor="white",edgecolor="#cbd5e0"))
        ax.set_xlabel("Δ = sec_test − base_line"); ax.spines[["top","right","left"]].set_visible(False); ax.grid(axis="x",alpha=.16)
    fig.suptitle("候选趋势：每点为一名匿名受试者；exact sign-flip p 为探索性诊断",fontsize=14,y=.99)
    fig.tight_layout(); save(fig,"03_paired_trends")

def scatter_association(data: pd.DataFrame, x: str, y: str, stats: pd.Series, title: str, name: str, color: str, xlab: str, ylab: str):
    fig,ax=plt.subplots(figsize=(6.6,5.3)); q=data[["anonymous_id",x,y]].dropna();
    ax.axhline(0,color="#cbd5e0",lw=.8); ax.axvline(0,color="#cbd5e0",lw=.8)
    ax.scatter(q[x],q[y],s=72,facecolor="white",edgecolor=color,linewidth=1.8,marker="o")
    for _,r in q.iterrows(): ax.annotate(r.anonymous_id,(r[x],r[y]),xytext=(5,4),textcoords="offset points",fontsize=9,color=COLORS["ink"])
    if len(q)>=2:
        co=np.polyfit(q[x],q[y],1); xx=np.linspace(q[x].min(),q[x].max(),100); ax.plot(xx,np.polyval(co,xx),color=color,lw=1.2,ls="--")
    ax.set_xlabel("Δ " + xlab); ax.set_ylabel("Δ " + ylab); ax.set_title(title + "（Δ = sec_test − base_line）",fontsize=12)
    stable="是" if bool(stats.loo_sign_stable) else "否"
    ax.text(.03,.05,f"Spearman ρ={stats.spearman_rho:+.2f}; n={int(stats.n)}\nexact permutation p={stats.exact_permutation_p:.3f}\nLOO：{stats.loo_rho_min:+.2f} 至 {stats.loo_rho_max:+.2f}\nLOO方向稳定：{stable}",transform=ax.transAxes,va="bottom",fontsize=9.4,bbox=dict(boxstyle="round,pad=.35",facecolor="white",edgecolor="#cbd5e0"))
    ax.spines[["top","right"]].set_visible(False); ax.grid(alpha=.14); fig.tight_layout(); save(fig,name)

def plot_method_matrix(method: pd.DataFrame):
    rows=[
        ("[已运行] Kwak 2022\nFGANet", "特征分组与融合思路", "透明 late fusion 原型", "不训练深网；不报性能"),
        ("[可运行代理] Li 2023\nEEG–EMG GIN", "脑-肌图关系", "受试者级关联与 LOO", "不训练 GIN；不作因果"),
        ("[审计完成] Huo 2024\nfNIRS neuromodulation review", "fNIRS 生理/测量边界", "MBLL 与双波长审计", "不报告绝对 HbO/HbR"),
        ("[可运行代理] Lorenz 2024\nneuroimaging + motion capture review", "神经影像-动作对应", "EMG/IMU 执行 proxy", "不称绝对姿态或关节运动学"),
        ("[已运行候选] Beste 2023\nBRAC", "动态状态描述", "状态占用熵、切换率", "会话内数学状态，非认知标签"),
        ("[原型] Su 2024\nadaptive hybrid BCI", "质量/缺失感知融合", "可审计质量权重", "fNIRS 权重=0；无监督目标"),
        ("[已运行候选] Yang 2024\nfNIRS–EMG", "同任务跨模态耦合", "|rho| 配对差，n=6", "小样本候选，不作神经驱动"),
        ("[概念借鉴] Li 2022\nbrain controllability", "网络动力学概念", "质量门控动态特征", "不作源级可控性解释"),
    ]
    fig,ax=plt.subplots(figsize=(11.5,8.0)); ax.axis("off")
    table=ax.table(cellText=rows,colLabels=["文献方法组件","迁移内容","本地可运行输出","边界/取舍"],loc="center",cellLoc="left",colLoc="left",colWidths=[.28,.22,.23,.27])
    table.auto_set_font_size(False); table.set_fontsize(9.2); table.scale(1,2.85)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e0"); cell.get_text().set_fontname(PLOT_FONT_NAME)
        if r==0: cell.set_facecolor(COLORS["ink"]); cell.get_text().set_color("white"); cell.get_text().set_fontweight("bold")
        elif r%2==0: cell.set_facecolor("#f7fafc")
    ax.set_title("用户指定八篇文献：方法组件的本地迁移取舍",fontsize=14,pad=16)
    save(fig,"06_method_transfer_matrix")

def plot_roadmap():
    fig,ax=plt.subplots(figsize=(10.5,4.8)); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,5)
    for x,title,items,color in [(0.4,"现在：保守主线",["固定主要任务对比","持续三层质量门控","逐人报告有效 n/Δ","仅受试者级验证"],COLORS["fnirs"]),(5.25,"扩样后：创新升级",["20–30 配对受试者","预注册主要终点","LOSO/跨访视验证","再比较图模型与时序模型"],COLORS["bridge"])]:
        ax.add_patch(plt.Rectangle((x,.75),4.1,3.45,facecolor="#f7fafc",edgecolor=color,lw=1.7))
        ax.text(x+.22,3.75,title,fontsize=14,fontweight="bold",color=color)
        for i,item in enumerate(items): ax.text(x+.35,3.05-i*.55,"• "+item,fontsize=11.5,color=COLORS["ink"])
    ax.annotate("",xy=(5.05,2.45),xytext=(4.55,2.45),arrowprops=dict(arrowstyle="->",lw=1.8,color=COLORS["muted"]))
    ax.text(5,.2,"升级条件：独立验证稳定后，才讨论候选标志物或受试者级模型",ha="center",fontsize=11,color=COLORS["warn"])
    save(fig,"07_roadmap")

def set_cell_shading(cell, fill):
    tcPr=cell._tc.get_or_add_tcPr(); shd=OxmlElement("w:shd"); shd.set(qn("w:fill"),fill); tcPr.append(shd)

def set_repeat_table_header(row):
    trPr=row._tr.get_or_add_trPr(); tblHeader=OxmlElement("w:tblHeader"); tblHeader.set(qn("w:val"),"true"); trPr.append(tblHeader)

def style_doc(doc):
    styles=doc.styles; styles["Normal"].font.name="等线"; styles["Normal"]._element.rPr.rFonts.set(qn("w:eastAsia"),"等线"); styles["Normal"].font.size=Pt(10.2)
    for name,size,color in [("Title",22,"263238"),("Heading 1",16,"1F6F8B"),("Heading 2",12,"263238")]:
        styles[name].font.name="等线"; styles[name]._element.rPr.rFonts.set(qn("w:eastAsia"),"等线"); styles[name].font.size=Pt(size); styles[name].font.color.rgb=RGBColor.from_string(color)

def add_title(doc,title,subtitle=""):
    p=doc.add_paragraph(); p.style="Title"; p.alignment=WD_ALIGN_PARAGRAPH.LEFT; p.add_run(title)
    if subtitle:
        p=doc.add_paragraph(subtitle); p.alignment=WD_ALIGN_PARAGRAPH.LEFT; p.runs[0].font.color.rgb=RGBColor(90,105,115); p.runs[0].font.size=Pt(10.5)

def add_bullets(doc, items):
    for item in items: doc.add_paragraph(item, style="List Bullet")

def add_figure(doc,name,width=16.8,caption=None):
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.add_run().add_picture(str(FIG/f"{name}.png"),width=Cm(width))
    if caption:
        # Keep the formal caption distinct from plain-language interpretation so
        # readers can first identify the plot and then understand its implication.
        lines=str(caption).splitlines()
        p=doc.add_paragraph(lines[0]); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.italic=True; run.font.size=Pt(8.8); run.font.color.rgb=RGBColor(90,105,115)
        for line in lines[1:]:
            p=doc.add_paragraph(line); p.alignment=WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_after=Pt(1.5)
            for run in p.runs:
                run.font.size=Pt(9.1); run.font.color.rgb=RGBColor(38,50,56)

def add_table(doc, headers, rows, widths=None):
    t=doc.add_table(rows=1, cols=len(headers)); t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.style="Table Grid"; hdr=t.rows[0]
    for i,h in enumerate(headers): hdr.cells[i].text=str(h); set_cell_shading(hdr.cells[i],"263238");
    for c in hdr.cells:
        for r in c.paragraphs[0].runs: r.font.color.rgb=RGBColor(255,255,255); r.font.bold=True; r.font.size=Pt(8.5)
        c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
    set_repeat_table_header(hdr)
    for row in rows:
        cells=t.add_row().cells
        for i,val in enumerate(row):
            cells[i].text=str(val); cells[i].vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for p in cells[i].paragraphs:
                for r in p.runs: r.font.size=Pt(8.2)
    if widths:
        for row in t.rows:
            for i,w in enumerate(widths): row.cells[i].width=Cm(w)
    return t

def page(doc): doc.add_page_break()

def build_docx(method, assoc, fusion, counts):
    if Document is None: raise RuntimeError("python-docx is required for --document-only")
    doc=Document(); sec=doc.sections[0]; sec.top_margin=Cm(1.55); sec.bottom_margin=Cm(1.45); sec.left_margin=Cm(1.6); sec.right_margin=Cm(1.6); style_doc(doc)
    add_title(doc,"质量门控多模态分析：\n阶段性可复现汇报","本地 STAGE-only 数据 | 2026-09-04 | 教师快速阅读版（匿名化）")
    doc.add_heading("执行摘要",level=1)
    add_bullets(doc,["质量门控流程已完整跑通：7 名 base_line/sec_test 配对受试者（14 访视）；汇总 206 个 EMG 任务窗与 40 个脑-肌 bridge。", "文献方法迁移得到候选趋势和跨模态关联，但均为探索性描述，不是已验证标志物、疗效或因果机制。", "当前最清晰的任务结构候选是 Δ(摸头−摸膝响应)：n=7，均值 +0.0275，5/2 同向，exact sign-flip p=0.406。", "脑-肌 coupling 与 EMG coactivation 的关联在 n=6 为 ρ=−0.77、exact permutation p=0.103，LOO 方向稳定；仅作为扩样优先验证线索。", "质量融合可运行，但 STAGE 缺少 fnirs_sample_valid_coverage，按预设保守公式将 fNIRS 融合权重置 0；质量加权结果不应被解释为融合性能提升。"])
    doc.add_paragraph("结论口径：目前可汇报“质量门控、可解释、可追溯的脑—肌探索性工作流已跑通”，不能汇报“已发现针刺疗效、生物标志物、闭环机制或深度学习诊断模型”。主终点也尚未冻结：STAGE 未提供通道级任务向量，需本地重导后复核。")
    page(doc)
    add_title(doc,"研究设计与数据流")
    add_figure(doc,"01_workflow",16.7,"图 1. 本地数据到探索性迁移结论的流程。Δ=sec_test−base_line；独立单位=受试者。\n怎么看：从左向右看四个步骤；任务语义按外层文件夹判定，EMG 均视为运动任务；每一步都要通过前一步的质量门控。\n看到了什么：EEG 当前主要用于事件标记、跨模态时间对齐和质量门控；EMG 标记依据 EEG/fNIRS 标记反推；IMU 只提供重力参考 roll/pitch 与运动强度 proxy。\n意味着什么：对于每个具体汇总指标，每名受试者最终贡献一个配对变化量，减少把事件窗误当成独立样本的风险。\n不能推出什么：EEG 尚未作为主要终点；流程图说明“如何分析”，不等于已经证明干预疗效、实时因果或闭环机制。")
    doc.add_heading("分析原则",level=1)
    add_bullets(doc,["固定组合针刺/电针为研究背景；当前设计不含 sham、单穴随机或逐次刺激日志。", "任务名来自既定固定协议与自动 marker；跨模态配对仅描述同任务窗口，不代表实时因果。", "所有核心数值来自本地 verified STAGE 与本地 method-transfer 输出；不使用旧 J 盘阶段数值。"])
    page(doc)
    add_title(doc,"数据覆盖与质量边界")
    add_figure(doc,"02_coverage_quality",16.5,"图 2. 数据量与层级。蓝色=主分支可用；红/橙色=失败、未纳入；黄色=降级但保留敏感性分析。\n怎么看：上方 7/14/206/40 是不同记录层级，下方横条是预处理状态；数字不能相加，也不能替代受试者级 n。\n看到了什么：41 EEG ready + 3 failed；33 fNIRS standard + 16 eye-IR degraded；14 EMG ready。黄色分支不是“正常数据”，而是带明确干扰边界的敏感性分支。\n意味着什么：当前数据足以建立可复现的探索性流程，并能定位失败记录与降级记录对后续扩样的影响。\n不能推出什么：可用记录数多不代表统计把握度高；失败记录未进入主分支，降级记录也不能与标准分支等价。")
    add_bullets(doc,["眼动 fNIRS：EYE_IR_DEGRADED_MASKED，不整体排除；仅作为红外干扰/遮挡后降级敏感性分支。", "EMG：S01/base_line 使用显式 shifted_v1 映射例外；编号对齐不等于信号修复，质量门控仍独立执行。", "IMU：仅 gravity-referenced roll/pitch 与运动强度 proxy；不称绝对姿态、绝对航向或关节运动学。"])
    page(doc)
    add_title(doc,"配对候选趋势")
    add_figure(doc,"03_paired_trends",16.7,"图 3. 受试者级配对变化；Δ=sec_test−base_line；p 为 exact sign-flip 探索性诊断。\n怎么看：每行是一名匿名受试者，0 右侧表示 sec_test 增加，左侧表示降低；菱形为均值，短竖线为中位数。\n看到了什么：四个面板分别表示：摸头−摸膝 fNIRS signed contrast（任务差异，均值 +0.02753，5↑/2↓/0=0，p=0.40625）；状态占用熵（无监督数学状态复杂度，不等于认知能力，均值 −0.15635，2↑/4↓/1=0，p=0.21875）；EMG 包络 CV（相对波动，不等于动作改善，均值 −0.06849，2↑/5↓/0=0，p=0.2031）；脑—肌 |rho|（关联强度且不保留方向，均值 −0.06111，2↑/4↓/0=0，p=0.46875）。\n意味着什么：任务对比和执行稳定性可能是后续主线的候选表型，可先用于扩样时的优先级排序。\n不能推出什么：四个面板横轴尺度和单位不同，不能互相比较大小；方向一致不等于显著或稳定，也不能冻结为标志物。")
    page(doc)
    add_title(doc,"关联一：状态占用熵与任务对比")
    add_figure(doc,"04_assoc_entropy_task",13.7,"图 4. 状态占用熵与摸头−摸膝 fNIRS 响应的受试者级关联。\n怎么看：横轴和纵轴都是同一受试者从 base_line 到 sec_test 的变化量；左上/右下象限表示两者方向相反，右上/左下表示同向；虚线仅为描述性辅助线，不是预测模型。\n看到了什么：ρ=−0.46（n=7），点云整体偏向反向变化，但离散度大。这里的“状态”是由信号数学规则得到的无监督编号，不等于认知能力或心理量表。\n意味着什么：它提供了一条把动态状态与任务反应放入同一候选认知—运动框架的可检验线索。\n不能推出什么：exact permutation p=0.302；LOO 符号稳定也不能说明认知导致运动变化，更不能作为干预机制证据。")
    page(doc)
    add_title(doc,"关联二：脑-肌耦合与 EMG 共激活")
    add_figure(doc,"05_assoc_coupling_emg",13.7,"图 5. 脑—肌耦合与 EMG 共激活的受试者级关联（n=6）。\n怎么看：先在每名受试者/访视内，用最多 5 个配对任务窗计算 fNIRS–EMG rho，再取 |rho| 形成横轴；纵轴是 EMG 全局共激活变化；每个点代表一名共同可用受试者。\n看到了什么：图中 −0.77 是 6 名受试者两个 delta 的跨受试者 Spearman rho，不是某一条任务窗内的 rho；LOO 后方向保持。S01 因不完整配对未进入该关联。\n意味着什么：耦合强度变化与肌肉共激活变化可能存在反向关系，是当前最值得扩样优先验证的候选脑—肌标志物关系。\n不能推出什么：|rho| 不保留方向，也不表示神经驱动；exact permutation p=0.103，且无行为/力学金标准，不能称因果或已验证标志物。")
    page(doc)
    add_title(doc,"用户指定八篇文献的方法迁移矩阵")
    add_figure(doc,"06_method_transfer_matrix",17.1,"图 6. 用户指定八篇文献的方法组件及其本地取舍。RSA 不是八篇文献之一；因 STAGE 缺少通道级任务向量，相关补充分析明确标记 unavailable。\n怎么看：逐行看“文献方法组件—迁移内容—本地可运行输出—边界/取舍”，最后一列比前两列更重要，因为它标出当前证据的上限。\n看到了什么：已运行的是透明特征融合、脑—肌关联、动态状态熵、双波长/MBLL 审计和质量/缺失感知原型；需要通道级向量或更大样本的方法被标记为 unavailable 或不训练深网。\n意味着什么：文献中的创新点可以转化为可审计的小样本原型，先验证“是否在本地数据上产生稳定趋势”，再决定是否升级模型。\n不能推出什么：方法迁移成功只代表计算可执行，不代表复现原论文性能，也不代表本地已经证明机制或临床价值。")
    doc.add_paragraph("MBLL 审计确认 730/850 nm 双波长与探头几何存在，但 DPF、消光系数溯源、坐标单位和绝对光程未声明，因此不报告绝对 HbO/HbR。")
    page(doc)
    add_title(doc,"对老师三点的回应")
    add_table(doc,["老师关注点","当前可回答的内容","保留边界"],[
        ["认知—运动逻辑","以任务控制链组织：状态/任务 fNIRS → EMG 执行 proxy；报告关联与个体差异。","认知与运动非同一实时 trial；不写为因果路径。"],
        ["穴位—经络—脑网络主线","固定组合干预背景下，可描述纵向脑—任务—执行的共同变化。","无单穴随机、sham、源定位；不做单穴/经络特异机制。"],
        ["人工智能价值","质量门控、缺失感知、透明 late fusion、LOO 与逐人可追溯。","n=7 不训练 FGANet/GIN/CNN-LSTM，不报临床 accuracy。"]], [3.4,6.5,6.4])
    doc.add_heading("质量加权融合的如实解释",level=1)
    add_bullets(doc,["公式：fNIRS=clip(event_count_used/15 × sample_valid_coverage,0,1)；EMG=clip(mapping_valid × muscle_count/7,0,1)；bridge=clip(event_pairs/5,0,1)。", "STAGE 当前缺 fnirs_sample_valid_coverage，因此 fNIRS 权重按缺失规则保守为 0；这不是“fNIRS 无效”的结论，而是数据接口不充分时不让它主导融合。", f"等权 fusion：{int(fusion.iloc[0].n)} 人、{int(fusion.iloc[0].positive)}↑/{int(fusion.iloc[0].negative)}↓、p={fusion.iloc[0].signflip_p:.3f}；质量加权：{int(fusion.iloc[1].positive)}↑/{int(fusion.iloc[1].negative)}↓、p={fusion.iloc[1].signflip_p:.3f}。", "融合无监督目标、无性能比较；fusion_summary 中的 self-correlation LOO 结构上接近 1，不作为模型稳健性证据。"])
    page(doc)
    add_title(doc,"扩样路线图：保守与创新并行")
    add_figure(doc,"07_roadmap",16.7,"图 7. 扩样路线图：先固定终点和质量契约，再在足够配对样本上升级模型。\n怎么看：左框是当前样本即可执行的保守路线，右框是达到样本量和质量条件后才进入的创新路线；箭头表示证据门槛，不是时间承诺。\n看到了什么：当前优先固定任务对比、三层质量门控、逐人 Δ 和有效 n；扩到约 20–30 名配对受试者后，才比较 LOSO、层级模型、规则图和时序图模型。\n意味着什么：新增受试者可以沿同一预处理契约增量加入，减少后续流程漂移，并把 AI 用在可解释、可验证的环节。\n不能推出什么：路线图不是保证显著结果的计划；模型升级前仍需独立验证和预注册主要终点。")
    doc.add_heading("建议下一步",level=1)
    add_bullets(doc,["第一优先：在本地从原始数据重导通道级三任务向量，复核 analysis_contract 所列的“摸头−摸膝 task-space magnitude”主终点；在此之前不冻结。", "保守路线：固定预设任务对比、质量门控、逐人 Δ 与有效 n；新增样本只增量进入相同契约。", "创新路线：达到 20–30 名配对受试者后，使用受试者级 LOSO/跨访视验证，比较规则图、层级模型、pyRiemann 或时序图模型。"])
    page(doc)
    add_title(doc,"不能宣称的边界与可复现性")
    doc.add_heading("当前不能宣称",level=1)
    add_bullets(doc,["不能宣称针刺/电针疗效、单穴或经络因果、脑区源定位机制、已验证生物标志物或临床预测。", "不能把 fNIRS–EMG 相关称作神经驱动或实时因果；不能把 IMU proxy 称作绝对姿态/关节运动学。", "不能把通道、事件、任务窗、网络边当独立样本；不能把探索性 exact p 当确认性检验。"])
    doc.add_heading("输入与追溯",level=1)
    add_bullets(doc,["核心表：STAGE 的 fnirs_event_states、movement_subtask_summary、movement_subtask_emg_summary、movement_subtask_bridge_summary、motor_multimodal_features。", "迁移统计：literature_method_transfer 的 method_summary、association_summary、fusion_summary 与 metadata。", "报告 metadata 保存相对输入路径、SHA-256、匿名规则与匿名标签、生成命令与 QA；正文不显示原始姓名或绝对路径。"])
    p=doc.add_paragraph("图注统一说明：Δ=sec_test−base_line；独立单位=受试者；n 和 exact sign-flip/permutation p 均为探索性。")
    p.runs[0].font.bold=True
    doc.save(OUT/"teacher_report_20260904_visual.docx")

def build_pdf(method, assoc, fusion):
    """Create an eight-page condensed visual briefing without Word/LibreOffice."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.lib.utils import ImageReader
    font_path = resolve_cjk_font()
    pdfmetrics.registerFont(TTFont("CJK", str(font_path)))
    pdf = Canvas(str(OUT/"teacher_report_20260904_visual.pdf"), pagesize=A4)
    W,H=A4
    def wrap(text, width, size=10):
        import re
        tokens=re.findall(r"[A-Za-z0-9_./=+−–-]+|.",text); lines=[]; current=""
        for token in tokens:
            candidate=current+token
            if current and pdfmetrics.stringWidth(candidate,"CJK",size)>width:
                lines.append(current); current=token
            else: current=candidate
        if current: lines.append(current)
        return lines
    def header(title, page_no, subtitle="浓缩视觉简报，不等同于 DOCX 详版"):
        pdf.setFillColor(colors.HexColor("#263238")); pdf.setFont("CJK",17); pdf.drawString(1.45*cm,H-1.6*cm,title)
        pdf.setFillColor(colors.HexColor("#718096")); pdf.setFont("CJK",8.5); pdf.drawRightString(W-1.45*cm,H-1.52*cm,subtitle)
        pdf.setStrokeColor(colors.HexColor("#cbd5e0")); pdf.line(1.45*cm,H-1.82*cm,W-1.45*cm,H-1.82*cm)
        pdf.setFont("CJK",8); pdf.setFillColor(colors.HexColor("#718096")); pdf.drawRightString(W-1.45*cm,1.0*cm,f"第 {page_no} / 8 页 | Δ = sec_test − base_line | 探索性")
    def bullets(items, y, width=17.7*cm, size=10.3):
        pdf.setFont("CJK",size); pdf.setFillColor(colors.HexColor("#263238"))
        for item in items:
            for j,line in enumerate(wrap(item,width-0.45*cm,size)):
                pdf.drawString(1.65*cm,y,("• " if j==0 else "  ")+line); y-=.52*cm
            y-=.08*cm
        return y
    def image(name, top, bottom=1.45*cm, max_width=17.8*cm):
        path=FIG/name
        img=ImageReader(str(path)); iw,ih=img.getSize(); available=top-bottom
        scale=min(max_width/iw,available/ih); width,height=iw*scale,ih*scale
        x=(W-width)/2; pdf.drawImage(img,x,top-height,width,height,preserveAspectRatio=True,mask="auto")
        return top-height
    def table(rows, x, y, widths, row_h=1.05*cm, size=8.5):
        for r,row in enumerate(rows):
            xx=x
            for c,value in enumerate(row):
                fill=colors.HexColor("#263238") if r==0 else (colors.HexColor("#f7fafc") if r%2==0 else colors.white)
                pdf.setFillColor(fill); pdf.rect(xx,y-r*row_h,widths[c],row_h,fill=1,stroke=0)
                pdf.setStrokeColor(colors.HexColor("#cbd5e0")); pdf.rect(xx,y-r*row_h,widths[c],row_h,fill=0,stroke=1)
                pdf.setFillColor(colors.white if r==0 else colors.HexColor("#263238")); pdf.setFont("CJK",size)
                lines=wrap(str(value),widths[c]-.2*cm,size)
                base=y-r*row_h+row_h-.34*cm
                for line in lines[:3]: pdf.drawString(xx+.1*cm,base,line); base-=.29*cm
                xx+=widths[c]
        return y-len(rows)*row_h
    header("质量门控多模态分析：阶段性可复现汇报",1,"STAGE-only | 2026-09-04 | 匿名化浓缩视觉简报")
    pdf.setFont("CJK",10.5); pdf.setFillColor(colors.HexColor("#263238")); pdf.drawString(1.55*cm,H-2.55*cm,"当前结论：可复现工作流已跑通；候选趋势与关联均仅用于探索性验证排序。")
    table([["配对受试者","访视","EMG 任务窗","脑-肌 bridge"],["7","14","206","40"]],1.55*cm,H-4.15*cm,[4.1*cm,3.7*cm,4.6*cm,4.2*cm],row_h=1.0*cm,size=10)
    bullets(["任务对比：Δ(摸头−摸膝响应) n=7，均值 +0.0275，5/2 同向，exact sign-flip p=0.406。", "脑-肌 coupling 与 EMG coactivation：n=6，Spearman rho=-0.77，exact permutation p=0.103；仅为扩样候选。", "主终点未冻结：STAGE 缺通道级任务向量，须在本地重导后复核 task-space magnitude。"],H-6.8*cm)
    pdf.showPage()
    header("流程、覆盖与质量边界",2)
    y=bullets(["怎么看：从左向右看流程；任务语义按外层文件夹判定，EMG 均为运动任务，只有通过任务、信号和对齐门控才进入受试者级趋势。", "看到了什么：EEG 主要用于事件标记、跨模态时间对齐和质量门控；EMG 标记依据 EEG/fNIRS 标记反推；IMU 只提供重力参考 roll/pitch 与运动强度 proxy。", "意味着什么：对于每个具体汇总指标，每名受试者最终贡献一个配对变化量，避免把任务窗数量误当作样本量。", "不能推出什么：EEG 尚未作为主要终点；流程说明可复现性，不等于干预疗效、实时因果或闭环机制。"],H-2.12*cm,width=17.2*cm,size=8.35)
    y=image("01_workflow.png",y-.05*cm,bottom=H-8.85*cm,max_width=17.4*cm)
    y=bullets(["质量图怎么看：蓝色=主分支可用，红/橙色=失败未纳入，黄色=降级但保留敏感性；上方数字看规模，下方横条看状态。", "关键观察：41 EEG ready+3 failed；33 fNIRS standard+16 eye-IR degraded；14 EMG ready。黄色分支带有明确干扰边界，不与标准分支等价。", "不能推出什么：上方 7/14/206/40 来自不同记录层级，不能相加或替代受试者 n。"],y-.12*cm,width=17.2*cm,size=8.15)
    image("02_coverage_quality.png",y-.08*cm,bottom=1.5*cm,max_width=17.4*cm)
    pdf.showPage()
    header("受试者级配对趋势",3)
    y=bullets(["怎么看：每行一个匿名受试者；0 右侧表示 sec_test 增加，左侧表示降低；菱形为均值、短线为中位数。四个面板横轴尺度和单位不同，不能跨面板比较大小。", "看到了什么：摸头−摸膝 fNIRS signed contrast 均值 +0.02753（5↑/2↓/0=0，p=0.40625）；状态占用熵均值 −0.15635（2↑/4↓/1=0，p=0.21875），是无监督数学状态复杂度而非认知能力；EMG 包络 CV 均值 −0.06849（2↑/5↓/0=0，p=0.2031），表示相对波动而非动作改善；脑—肌 |rho| 均值 −0.06111（2↑/4↓/0=0，p=0.46875），表示关联强度且不保留方向。", "意味着什么：任务对比和执行稳定性可作为扩样时的候选表型。", "不能推出什么：p 值均为 exact sign-flip 探索性诊断；方向一致不等于已验证标志物。"],H-2.25*cm,size=8.05)
    image("03_paired_trends.png",y-.12*cm,bottom=1.45*cm,max_width=17.6*cm)
    pdf.showPage()
    header("关联一：状态占用熵与任务对比",4)
    y=bullets(["怎么看：每点是一名受试者的配对变化；横轴为状态占用熵变化，纵轴为摸头−摸膝 fNIRS 响应变化。左上/右下表示方向相反，右上/左下表示同向；虚线只是描述性辅助线，不是预测模型。", "看到了什么：ρ=−0.46（n=7），点云整体偏向反向变化；状态是无监督数学编号，不等于认知能力。", "意味着什么：可把动态状态与任务反应放进同一候选认知—运动分析框架。", "不能推出什么：exact permutation p=0.302；LOO 符号稳定也不能说明因果或干预机制。"],H-2.25*cm,size=8.15)
    image("04_assoc_entropy_task.png",y-.12*cm,bottom=1.45*cm,max_width=15.5*cm)
    pdf.showPage()
    header("关联二：脑-肌耦合与 EMG 共激活",5)
    y=bullets(["怎么看：每名受试者/访视内用最多 5 个配对任务窗计算 fNIRS–EMG rho，再取 |rho| 形成横轴；纵轴是 EMG 共激活变化。", "看到了什么：−0.77（n=6）是两个 delta 的跨受试者 Spearman rho，不是单条任务窗 rho；S01 因不完整配对未进入，LOO 后方向保持。", "意味着什么：耦合强度与肌肉共激活可能反向，是当前最值得扩样的候选关系。", "不能推出什么：|rho| 不保留方向，也不表示神经驱动；exact permutation p=0.103，不能称因果或已验证标志物。"],H-2.25*cm,size=8.15)
    image("05_assoc_coupling_emg.png",y-.12*cm,bottom=1.45*cm,max_width=15.5*cm)
    pdf.showPage()
    header("用户指定八篇文献的方法迁移矩阵",6)
    y=bullets(["怎么看：逐行看‘方法组件—迁移内容—本地输出—边界/取舍’，并先读每行的状态标签；最后一列定义证据上限。", "看到了什么：标签包括[已运行]、[审计完成]、[可运行代理]、[原型]、[已运行候选]、[概念借鉴]；透明融合、脑—肌关联、动态状态熵和 MBLL 审计已产生本地输出。", "意味着什么：文献创新点先转为可审计小样本原型，用于排序扩样优先级。", "不能推出什么：可运行不等于复现原论文性能；RSA 仍是另行标记的 unavailable 补充分析，不是八篇文献条目。"],H-2.25*cm,size=7.8)
    image("06_method_transfer_matrix.png",y-.12*cm,bottom=1.45*cm,max_width=17.6*cm)
    pdf.showPage()
    header("老师的三点问题：可答与不可答",7)
    end=table([["问题","目前可回答","目前不可回答"],["认知—运动逻辑","状态/任务 fNIRS 与 EMG 执行 proxy 的共同变化和关联。","实时因果路径或神经驱动。"],["穴位—经络—脑网络","固定组合干预背景下的纵向共同变化。","单穴、经络特异或源定位机制。"],["人工智能价值","质量门控、缺失感知、透明融合、逐人可追溯。","n=7 的深网训练、临床 accuracy。"]],1.25*cm,H-3.7*cm,[3.4*cm,7.2*cm,7.0*cm],row_h=1.55*cm,size=8.7)
    end=bullets(["怎么看：表格左列是老师的问题，中列是当前数据能支持的回答，右列是必须保留的边界；路线图中的 20–30 是规划门槛，不是已完成样本量或功效结果。", "看到了什么：三点都已有可汇报的‘方法层’回应：保守路线对应任务对比、质量门控、逐人有效 n/Δ 和受试者级验证。", "意味着什么：穴位—经络只能作为固定组合干预的背景假设，扩样也不能推出单穴因果；AI 先体现在质量门控、逐人追溯和透明融合。", "不能推出什么：当前回答不能替代单穴随机、源定位或临床验证。"],end-.25*cm,width=17.2*cm,size=7.65)
    image("07_roadmap.png",end-.18*cm,bottom=1.45*cm,max_width=17.3*cm)
    pdf.showPage()
    header("边界、融合解释与下一步",8)
    bullets(["怎么看：本页先看三条数据边界，再看融合解释和下一步，最后核对‘不能宣称’清单。", "看到了什么：眼动 fNIRS 作为降级敏感性分支保留；IMU 只给重力参考姿态/运动强度 proxy；fNIRS 质量字段缺失时权重按规则置 0。", "意味着什么：系统优先保证可审计和可追溯，避免让缺失字段制造虚假的融合优势。", "下一步：重导通道级三任务向量并冻结终点；20–30 名配对受试者只是规划门槛，不是已完成样本量或功效结论，达到后再做 LOSO/跨访视和模型比较。", "不能推出什么：疗效、单穴/经络因果、源定位机制、已验证标志物或临床预测。"],H-2.35*cm,size=9.0)
    pdf.save()

def write_qa():
    import zipfile
    from PIL import Image
    from pypdf import PdfReader
    docx_path=OUT/"teacher_report_20260904_visual.docx"; pdf_path=OUT/"teacher_report_20260904_visual.pdf"
    with zipfile.ZipFile(docx_path) as z:
        xml=z.read("word/document.xml").decode("utf-8")
        embedded=sum(n.startswith("word/media/") for n in z.namelist())
        headings=sum(xml.count(t) for t in ["执行摘要","研究设计与数据流","数据覆盖与质量边界","配对候选趋势","关联一","关联二","用户指定八篇文献","对老师三点的回应","扩样路线图","不能宣称的边界"])
        requested_breaks=xml.count('w:type="page"')
        table_count=xml.count("<w:tbl>")
        caption_labels={label: xml.count(label) for label in ["怎么看：","看到了什么：","意味着什么：","不能推出什么："]}
    png=[]
    for p in sorted(FIG.glob("*.png")):
        with Image.open(p) as im: png.append({"file":p.name,"pixels":list(im.size),"bytes":p.stat().st_size,"nonblank":p.stat().st_size>1024 and im.size[0]>100 and im.size[1]>100})
    pdf_pages=len(PdfReader(str(pdf_path)).pages)
    pdf_reader=PdfReader(str(pdf_path))
    pdf_text="\n".join((p.extract_text() or "") for p in pdf_reader.pages)
    pdf_caption_labels={label: pdf_text.count(label) for label in ["怎么看：","看到了什么：","意味着什么：","不能推出什么："]}
    poppler=shutil.which("pdftoppm") or r"C:\Users\Weediia\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe"
    if not Path(poppler).exists(): raise RuntimeError("pdftoppm is required for independent PDF render QA")
    rendered=[]
    with tempfile.TemporaryDirectory(prefix="teacher_report_pdf_qa_") as temp:
        prefix=Path(temp)/"page"
        subprocess.run([poppler,"-png","-r","120",str(pdf_path),str(prefix)],check=True,capture_output=True,text=True)
        rendered_paths=sorted(Path(temp).glob("page-*.png"))
        if len(rendered_paths) != pdf_pages: raise RuntimeError(f"PDF render count mismatch: expected {pdf_pages}, got {len(rendered_paths)}")
        qa_pages=OUT/"qa_pages"
        if qa_pages.resolve().parent != OUT.resolve(): raise RuntimeError("Refusing to clean QA pages outside the report package")
        qa_pages.mkdir(exist_ok=True)
        for stale in qa_pages.glob("page-*.png"): stale.unlink()
        for stale in OUT.glob("qa_page-*.png"): stale.unlink()
        for i,p in enumerate(rendered_paths,1):
            with Image.open(p) as im:
                pixels=list(im.size); rgb=im.convert("RGB")
                extrema=rgb.getextrema(); nonempty=any(lo != hi for lo,hi in extrema)
            destination=qa_pages/f"page-{i:02d}.png"
            shutil.copyfile(p,destination)
            rendered.append({"page":i,"pixels":pixels,"nonempty":nonempty,"qa_page_file":str(destination.relative_to(OUT))})
    qa={
        "docx_explicit_page_breaks":requested_breaks,
        "docx_render_status":"not_rendered_in_current_session",
        "docx_embedded_images":embedded,
        "docx_table_count":table_count,
        "docx_title_count":headings,
        "docx_caption_labels":caption_labels,
        "pdf_pages":pdf_pages,
        "pdf_expected_pages":8,
        "pdf_page_count_pass":pdf_pages==8,
        "pdf_caption_labels":pdf_caption_labels,
        "pdf_rendered_pages":rendered,
        "pdf_render_pass":len(rendered)==8 and all(x["nonempty"] for x in rendered),
        "qa_pages_count":len(list((OUT/"qa_pages").glob("page-*.png"))),
        "png":png,
        "all_png_nonblank":all(x["nonblank"] for x in png),
        "pdf_producer":"ReportLab direct"
    }
    (OUT/"qa_report.json").write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding="utf-8")

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--figures-only",action="store_true"); parser.add_argument("--document-only",action="store_true"); args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True); setup_plotting()
    method=pd.read_csv(TRANSFER/"method_summary.csv"); assoc=pd.read_csv(TRANSFER/"association_summary.csv"); fusion=pd.read_csv(TRANSFER/"fusion_summary.csv")
    emg=pd.read_csv(STAGE/"movement_subtask_emg_summary.csv"); bridge=pd.read_csv(STAGE/"movement_subtask_bridge_summary.csv")
    assoc_data,_=association_sources(STAGE); assoc_data.to_csv(OUT/"association_source_data_anonymized.csv",index=False,encoding="utf-8-sig")
    if not args.document_only:
        if plt is None: raise RuntimeError("matplotlib is required for --figures-only")
        trends=trend_sources(STAGE)
        plot_workflow(); plot_coverage(emg,bridge); plot_trends(trends); plot_method_matrix(method); plot_roadmap()
        a1=assoc[(assoc.x=="occupancy_entropy_norm") & (assoc.y=="head_minus_knee_response")].iloc[0]
        a2=assoc[(assoc.x=="coupling_abs_rho") & (assoc.y=="emg_global_coactivation")].iloc[0]
        scatter_association(assoc_data,"occupancy_entropy_norm","head_minus_knee_response",a1,"状态占用熵 vs 摸头−摸膝响应", "04_assoc_entropy_task",COLORS["fnirs"],"状态占用熵","摸头−摸膝 fNIRS 响应")
        scatter_association(assoc_data,"coupling_abs_rho","emg_global_coactivation",a2,"脑-肌 coupling vs EMG 共激活", "05_assoc_coupling_emg",COLORS["bridge"],"脑-肌 |rho|","EMG 全局共激活")
    if not args.figures_only:
        build_docx(method,assoc,fusion,{"emg_windows":len(emg),"bridge":len(bridge)}); build_pdf(method,assoc,fusion); write_qa()
    inputs=[STAGE/"fnirs_event_states.csv",STAGE/"movement_subtask_summary.csv",STAGE/"movement_subtask_emg_summary.csv",STAGE/"movement_subtask_bridge_summary.csv",STAGE/"motor_multimodal_features.csv",STAGE/"eye_fnirs_features.csv",TRANSFER/"method_summary.csv",TRANSFER/"association_summary.csv",TRANSFER/"fusion_summary.csv",TRANSFER/"metadata.json",PREPROCESS]
    meta={"report_scope":"teacher quick report; STAGE-only core numbers; anonymized S01-S07","anonymization":{"rule":"Raw subject IDs are represented as S01-S07 in external report artifacts.","labels":[f"S{i:02d}" for i in range(1,8)]},"input_files_relative_to_project":[{"path":str(p.relative_to(ROOT)),"sha256":sha256(p)} for p in inputs],"report_script_sha256":sha256(Path(__file__)),"figure_count":len(list(FIG.glob("*.png"))),"caption_framework":["怎么看","看到了什么","意味着什么","不能推出什么"],"caption_scope":"All seven figures in DOCX and all figure-bearing PDF pages include plain-language interpretation and explicit evidence boundaries.","docx":"teacher_report_20260904_visual.docx","pdf_role":"condensed_visual_briefing","pdf_producer":"ReportLab direct"}
    (OUT/"report_metadata.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"docx={OUT/'teacher_report_20260904_visual.docx'} pdf={OUT/'teacher_report_20260904_visual.pdf'} figures={len(list(FIG.glob('*.png')))}")

if __name__=="__main__": main()
