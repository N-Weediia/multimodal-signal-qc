"""Build the compact leadership release for the corrected paired-seven run."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import markdown
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
try:  # Support both ``python scripts/...`` and ``import scripts...``.
    from .paired7_config import CANONICAL_ROOT, PAIRED_SUBJECTS, RELEASE_ROOT, RUN_ROOT
except ImportError:  # pragma: no cover - direct script execution
    from paired7_config import CANONICAL_ROOT, PAIRED_SUBJECTS, RELEASE_ROOT, RUN_ROOT


FIGURES = [
    ("phase16/formal_dashboard", "01_formal_dashboard", "Three-layer formal dashboard"),
    ("phase17/briefing_figure", "02_briefing_figure", "Single-page leadership summary"),
    ("phase18/cohort_expansion_stress_test", "03_cohort_expansion_stress_test", "Frozen n=4 versus corrected n=7"),
    ("phase11/nof1_response_fingerprint", "04_nof1_response_fingerprint", "Missing-aware participant fingerprints"),
    ("phase14/task_spatial_redistribution", "05_task_spatial_redistribution", "Task spatial magnitude summaries"),
    ("phase14_stability/fixed_three_marker_stability", "06_three_layer_loo_stability", "Leave-one-participant-out stability"),
    ("phase9/tcm_eye_anchor", "07_tcm_eye_quantitative_anchor", "Eye-process and raw four-diagnosis coverage"),
    ("phase17_eeg_quality/eeg_quality_sensitivity", "08_eeg_quality_sensitivity", "Legacy versus segment-gated EEG sensitivity"),
    ("phase17_eeg_quality/eeg_broad_threshold_sensitivity", "09_eeg_threshold_sensitivity", "EEG threshold coverage trade-off"),
    ("phase19/quality_gated_multimodal_graph", "10_quality_gated_multimodal_graph", "Theory-led quality-gated cross-modal graph"),
    ("phase19/quality_gate_matrix", "11_quality_gate_matrix", "Paired quality gates used by graph"),
]

REPRESENTATIVE_FIGURES = [
    ("phase5/joint_eeg_fnirs_2026081403.png", "10_representative_eeg_fnirs_1403.png", "Representative non-coregistered multimodal view"),
    ("phase5/joint_eeg_fnirs_2026081501.png", "11_representative_eeg_fnirs_1501.png", "Representative non-coregistered multimodal view"),
]

DATA_FILES = [
    ("phase16/resting_fnirs_network_panel.csv", "resting_fnirs_network_n7.csv", "formal rest panel"),
    ("phase16/task_space_panel.csv", "task_space_n7.csv", "formal task-space panel"),
    ("phase16/dynamic_state_panel.csv", "dynamic_state_n7.csv", "formal dynamic-state panel"),
    ("phase16/candidate_summary.csv", "open_candidate_summary_n7.csv", "open candidate diagnostics"),
    ("phase17/source_data.csv", "briefing_source_data_n7.csv", "single-page figure source data"),
    ("phase18/cohort_expansion_comparison.csv", "cohort_expansion_comparison.csv", "frozen n=4 versus corrected n=7"),
    ("phase11/nof1_delta_matrix.csv", "nof1_delta_matrix_n7.csv", "participant-level multimodal deltas"),
    ("phase11/marker_summary.csv", "marker_summary_n7.csv", "participant-level marker diagnostics"),
    ("phase14_stability/fixed_three_marker_summary.csv", "three_layer_marker_loo_n7.csv", "full and leave-one-out summaries"),
    ("phase14/task_spatial_redistribution_summary.csv", "task_spatial_redistribution_n7.csv", "task spatial source table"),
    ("phase15/task_pairwise_contrasts.csv", "task_pairwise_contrasts_n7.csv", "task contrast source table"),
    ("phase6/marker_summary.csv", "crossmodal_bridge_diagnostics_n7.csv", "rank and leave-one-out diagnostics"),
    ("phase9/paired_eye_process_coverage.csv", "eye_process_coverage_n7.csv", "eye report/process coverage"),
    ("phase9/paired_eye_process_delta.csv", "eye_process_delta_n7.csv", "task-aligned process deltas"),
    ("phase9/tcm_eye_raw_quantitative_tidy.csv", "tcm_eye_raw_quantitative_tidy.csv", "raw quantitative anchors only"),
    ("phase17_eeg_quality/eeg_quality_gated_subject_summary.csv", "eeg_quality_gated_subject_summary.csv", "EEG visit-level coverage"),
    ("phase17_eeg_quality/eeg_quality_sensitivity_comparison.csv", "eeg_quality_sensitivity_comparison.csv", "EEG estimator sensitivity"),
    ("phase17_eeg_quality/threshold_sensitivity.csv", "eeg_threshold_sensitivity.csv", "EEG threshold sensitivity"),
    ("phase19/node_matrix.csv", "phase19_node_matrix.csv", "Graph node delta matrix"),
    ("phase19/paired_quality_gates.csv", "phase19_paired_quality_gates.csv", "Paired quality gates"),
    ("phase19/edge_statistics.csv", "phase19_edge_statistics.csv", "Graph edge rank and LOO diagnostics"),
    ("phase19/edge_contributions.csv", "phase19_edge_contributions.csv", "Participant edge attributions"),
    ("phase19/subject_graph_explanations.csv", "phase19_subject_graph_explanations.csv", "Participant graph explanations"),
]

CANONICAL_DATA_FILES = [
    ("movement_emg_markers.csv", "movement_emg_markers.csv", "EEG/fNIRS-inferred EMG movement marker audit"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bool_mask(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.lower().eq("true")


def loo_mean(values: pd.Series) -> tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    estimates = [float(np.delete(clean, i).mean()) for i in range(len(clean))]
    return min(estimates), max(estimates)


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def collect_release_files() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for stem, target, role in FIGURES:
        for extension in ("png", "svg", "pdf"):
            source = RUN_ROOT / f"{stem}.{extension}"
            destination = RELEASE_ROOT / "figures" / f"{target}.{extension}"
            copy_file(source, destination)
            rows.append({"release_file": destination.relative_to(RELEASE_ROOT).as_posix(), "source": f"paired7/{stem}.{extension}", "role": role})
    for source_rel, target, role in REPRESENTATIVE_FIGURES:
        source = RUN_ROOT / source_rel
        destination = RELEASE_ROOT / "figures" / target
        copy_file(source, destination)
        rows.append({"release_file": destination.relative_to(RELEASE_ROOT).as_posix(), "source": f"paired7/{source_rel}", "role": role})
    for source_rel, target, role in DATA_FILES:
        source = RUN_ROOT / source_rel
        destination = RELEASE_ROOT / "data" / target
        copy_file(source, destination)
        rows.append({"release_file": destination.relative_to(RELEASE_ROOT).as_posix(), "source": f"paired7/{source_rel}", "role": role})
    for source_rel, target, role in CANONICAL_DATA_FILES:
        source = CANONICAL_ROOT / source_rel
        destination = RELEASE_ROOT / "data" / target
        copy_file(source, destination)
        rows.append({"release_file": destination.relative_to(RELEASE_ROOT).as_posix(), "source": f"canonical/{source_rel}", "role": role})
    for source, target, role in [
        (RUN_ROOT / "phase16" / "formal_analysis_dictionary.json", "formal_analysis_dictionary.json", "formal panel dictionary"),
        (RUN_ROOT / "phase17_eeg_quality" / "report.md", "eeg_quality_method_and_limits.md", "EEG quality branch report"),
        (RUN_ROOT / "phase9" / "review.md", "tcm_eye_method_and_limits.md", "TCM/eye coverage report"),
        (RUN_ROOT / "phasetcm" / "paired7_sec_test_report_parseability.md", "report_parseability_audit.md", "report extraction audit"),
        (RUN_ROOT / "phase18" / "review.md", "cohort_expansion_review.md", "cohort refresh interpretation"),
        (RUN_ROOT / "phase19" / "review.md", "phase19_graph_method_and_limits.md", "quality-gated graph method and limits"),
        (RUN_ROOT / "phase19" / "metadata.json", "phase19_metadata.json", "quality-gated graph provenance metadata"),
        (ROOT / "outputs" / "literature_20260903" / "methods_adaptation_matrix.md", "methods_adaptation_matrix.md", "literature-to-data adaptation review"),
        (ROOT / "outputs" / "literature_20260903" / "external_search_20260903.md", "external_search_20260903.md", "external literature and code search log"),
        (RUN_ROOT / "provenance" / "cleanup_applied.md", "cleanup_applied.md", "applied cleanup audit"),
    ]:
        destination = RELEASE_ROOT / "provenance" / target
        copy_file(source, destination)
        rows.append({"release_file": destination.relative_to(RELEASE_ROOT).as_posix(), "source": source.relative_to(ROOT).as_posix(), "role": role})
    pd.DataFrame(rows).to_csv(RELEASE_ROOT / "provenance" / "source_map.csv", index=False, encoding="utf-8-sig")
    return pd.DataFrame(rows)


def analysis_dictionary() -> pd.DataFrame:
    rows = [
        ["acupoint-meridian-brain framework", "rest_network", "signed resting fNIRS network integration delta", "signed", 7, "resting_fnirs_network_n7.csv", "Primary candidate; longitudinal association, not efficacy or single-point attribution"],
        ["cognition-motor", "movement_fnirs", "quality-aware movement fNIRS response delta", "signed proxy", 7, "nof1_delta_matrix_n7.csv", "Optical-density proxy; not HbO/HbR activation localization"],
        ["cognition-motor", "head_minus_knee_magnitude", "touch-head minus touch-knee mean absolute channel-change magnitude", "magnitude only", 7, "task_pairwise_contrasts_n7.csv", "Primary task fingerprint; does not encode activation direction"],
        ["cognition-motor", "task_spatial_magnitude", "mean absolute paired fNIRS channel change per protocol task", "magnitude only", "head=7;knee=7;waist=6", "task_space_n7.csv", "Probe-coordinate summary; waist missing for one participant"],
        ["dynamic AI descriptor", "occupancy_entropy_norm_delta", "change in normalized occupancy entropy over common first 19 events", "signed", 7, "dynamic_state_n7.csv", "Session-local mathematical states; exploratory"],
        ["dynamic AI descriptor", "state_switch_rate", "change in session-local state switch rate", "signed", 7, "dynamic_state_n7.csv", "Old negative direction not replicated; downgraded"],
        ["cross-layer bridge", "rest_to_movement_rho", "Spearman association between resting-network and movement-fNIRS deltas", "signed rank", 7, "open_candidate_summary_n7.csv", "Hypothesis-generating; no causal direction"],
        ["cross-layer bridge", "entropy_to_task_space_rho", "Spearman association between occupancy-entropy and head-minus-knee deltas", "signed rank", 7, "open_candidate_summary_n7.csv", "Hypothesis-generating; state labels are session-local"],
        ["TCM/cognitive anchor", "eye_process_delta", "exact-task quantitative eye-process follow-up minus baseline", "metric-specific", "executive=3;memory=1", "eye_process_delta_n7.csv", "Manual visual transcription pending second review; report scores excluded"],
        ["TCM raw anchor", "four_diagnosis_raw", "raw quantitative tongue/pulse report fields", "metric-specific", "paired=1", "tcm_eye_raw_quantitative_tidy.csv", "No syndrome inference; insufficient paired coverage"],
        ["EEG sensitivity", "event_theta", "legacy continuous versus segment-gated event counterpart", "signed", "paired=3/7", "eeg_quality_sensitivity_comparison.csv", "Estimators are not numerically identical; secondary quality evidence"],
        ["EEG sensitivity", "alpha_beta_ratio", "legacy continuous versus segment-gated execution counterpart", "signed", "paired=5/7", "eeg_quality_sensitivity_comparison.csv", "Estimators are not numerically identical; secondary quality evidence"],
        ["AI integration", "nof1_fingerprint", "missing-aware robust-z organization of participant-level multimodal deltas", "descriptive", 7, "nof1_delta_matrix_n7.csv", "Not a clinical score, classifier, or treatment recommendation"],
        ["AI integration", "quality_gated_multimodal_graph", "pre-specified cross-layer graph with LOO edge weights, paired QC gates and participant edge contributions", "exploratory attribution", "effective n varies by edge", "phase19_edge_statistics.csv; phase19_subject_graph_explanations.csv", "Transparent descriptor; no trained classifier, causal edge or acupuncture-point localization"],
        ["Synchronization", "movement_emg_markers", "EMG movement markers inferred from the selected EEG/fNIRS marker stream; marker 1 defines EMG frame 0", "audit metadata", "all EMG movement records", "movement_emg_markers.csv", "Secondary-stream disagreement is reported per marker; inferred timing is not an independent hardware trigger"],
    ]
    columns = ["domain", "marker", "definition", "directionality", "effective_n", "source_data", "guardrail"]
    out = pd.DataFrame(rows, columns=columns)
    out.to_csv(RELEASE_ROOT / "data" / "analysis_dictionary.csv", index=False, encoding="utf-8-sig")
    return out


def metrics() -> dict[str, object]:
    comparison = pd.read_csv(RUN_ROOT / "phase18" / "cohort_expansion_comparison.csv").set_index("metric")
    nof1 = pd.read_csv(RUN_ROOT / "phase11" / "nof1_delta_matrix.csv")
    eye = pd.read_csv(RUN_ROOT / "phase9" / "paired_eye_process_delta.csv")
    eeg = pd.read_csv(RUN_ROOT / "phase17_eeg_quality" / "eeg_quality_sensitivity_comparison.csv")
    threshold = pd.read_csv(RUN_ROOT / "phase17_eeg_quality" / "threshold_sensitivity.csv")
    movement_loo = loo_mean(nof1["movement_fnirs"])
    event_valid = bool_mask(eeg["event_delta_comparison_valid"])
    ratio_valid = bool_mask(eeg["ratio_comparison_valid"])
    event_agree = int((np.sign(eeg.loc[event_valid, "old_eeg_event_theta_frontal_delta_change"]) == np.sign(eeg.loc[event_valid, "new_qg_eeg_event_theta_frontal_delta_change"])).sum())
    ratio_agree = int((np.sign(eeg.loc[ratio_valid, "old_eeg_global_alpha_beta_ratio_change"]) == np.sign(eeg.loc[ratio_valid, "new_qg_eeg_global_alpha_beta_ratio_execution_change"])).sum())
    strict = threshold[threshold.threshold.eq(.35)]
    eye_error = eye[(eye.task.eq("执行功能与注意力")) & (eye.metric.eq("antisaccade_error_percent"))]
    return {
        "comparison": comparison,
        "movement_mean": float(nof1.movement_fnirs.mean()),
        "movement_loo": movement_loo,
        "eye_error_values": eye_error.delta_followup_minus_baseline.tolist(),
        "event_n": int(event_valid.sum()),
        "event_agree": event_agree,
        "ratio_n": int(ratio_valid.sum()),
        "ratio_agree": ratio_agree,
        "strict_baseline_blocks": float(strict.baseline_valid_blocks.mean()),
        "strict_followup_blocks": float(strict.followup_valid_blocks.mean()),
    }


def build_report(values: dict[str, object]) -> str:
    c = values["comparison"]
    rest = c.loc["rest network"]
    spatial = c.loc["task spatial magnitude"]
    occupancy = c.loc["state occupancy entropy"]
    switch = c.loc["state switch rate"]
    head = c.loc["head minus knee magnitude"]
    waist = c.loc["waist minus knee magnitude"]
    bridge = c.loc["rest-to-movement rho"]
    entropy_bridge = c.loc["entropy-to-head-knee rho"]
    eye_deltas = ", ".join(f"{value:+.1f}" for value in values["eye_error_values"])
    subjects = "、".join(PAIRED_SUBJECTS)
    return f"""# 针刺/电针后多模态脑—运动研究

## 阶段性进展汇报（更正后的 n=7 配对队列）

**固定口径**：纳入 7 名具有 `base_line → sec_test` 的受试者（{subjects}），两次采集间隔约两周；所有受试者接受同一固定组合针刺/电针方案。当前结果是小样本纵向探索和候选标志物筛选，不是即时刺激效应、临床疗效或单穴位/单经络因果证据。

## 一句话结论

更正 `2026081401` 并扩展到 7 名配对受试者后，**静息 fNIRS 网络、运动 fNIRS 响应和“摸头相对摸膝”的任务空间幅度仍保留可复核的方向性线索；动态状态中的占用熵下降保留、切换率下降未复现；新增眼动过程量提供了 3 名认知锚点，而分段 EEG 仍应作为质量敏感性证据。** AI 的主要价值不是训练小样本分类器，而是完成任务分解、跨层桥接、逐人指纹和留一法压力测试。

## 领导先看：队列扩充后哪些结果站住了

![队列扩充压力测试](figures/03_cohort_expansion_stress_test.png)

| 结果 | 冻结 n=4 | 更正 n=7 | 判断 |
|---|---:|---:|---|
| 静息网络均值 | {rest.old_estimate:+.4f} | {rest.new_estimate:+.4f}；LOO {rest.new_loo_min:+.4f}～{rest.new_loo_max:+.4f} | 减弱但均值方向在 LOO 中保留 |
| 运动 fNIRS 均值 | +0.1588 | {values['movement_mean']:+.4f}；LOO {values['movement_loo'][0]:+.4f}～{values['movement_loo'][1]:+.4f} | 保留；仍是光密度 proxy |
| 摸头−摸膝空间幅度 | {head.old_estimate:+.4f}，4/4 正 | {head.new_estimate:+.4f}，6/7 正 | 当前最直观的任务候选 |
| 摸腰−摸膝空间幅度 | {waist.old_estimate:+.4f}，3/4 正 | {waist.new_estimate:+.4f}，4/6 正 | 明显减弱，降级 |
| 占用熵变化 | {occupancy.old_estimate:+.4f} | {occupancy.new_estimate:+.4f}；LOO 全为负均值 | 保留为动态结构候选 |
| 状态切换率变化 | {switch.old_estimate:+.4f} | {switch.new_estimate:+.4f} | 旧负向结果未复现，移出主结论 |

这张图最重要的意义是：新增样本没有简单放大旧结果，而是帮助区分“可保留线索”和“原先可能由极小样本驱动的线索”。

## 三层正式总览

![三层正式总览](figures/01_formal_dashboard.png)

### 1. 认知—运动：任务空间指纹 + 量化眼动过程

- 三个动作按自动 marker 和固定小任务规则处理，分析单位始终是受试者，不把 block、通道或事件当成额外样本。
- 所有 EMG 均按运动任务处理；EMG marker 由选定的 EEG/fNIRS marker 流推断，首个 marker 映射为 EMG frame 0，逐标记差异见 `data/movement_emg_markers.csv`。
- `摸头 − 摸膝` 的绝对 fNIRS 通道变化幅度为 {head.new_estimate:+.4f}，6/7 名受试者为正；它描述**变化幅度**，不代表激活增强。
- 任务有效覆盖为摸头 7、摸膝 7、摸腰 6；`2026081701` 摸腰没有有效通道，未做填补。
- 新增眼动报告只使用过程量。执行功能/注意力任务有 3 名完整配对；反向眼跳错误率的三个 `sec_test − base_line` 值为 {eye_deltas} 个百分点。该同向现象值得后续复核，但来自图像报告人工转录，仍需第二位研究人员核对。

![眼动与四诊量化锚点](figures/07_tcm_eye_quantitative_anchor.png)

### 2. 穴位—经络—脑网络：建立可检验框架，不做单穴归因

当前可以防守的主线是：

> 固定组合针刺/电针背景 → 静息网络整合变化 → 运动任务空间变化 → 动态结构描述。

- 静息网络与运动 fNIRS 变化的 Spearman `rho` 从 n=4 的 {bridge.old_estimate:+.2f} 降为 n=7 的 {bridge.new_estimate:+.2f}，但新 LOO 范围仍为 {bridge.new_loo_min:+.2f}～{bridge.new_loo_max:+.2f}，方向均为正。
- 占用熵变化与“摸头−摸膝”空间幅度差的 `rho={entropy_bridge.new_estimate:+.2f}`，LOO {entropy_bridge.new_loo_min:+.2f}～{entropy_bridge.new_loo_max:+.2f}；只作为动态—任务桥接假设。
- 所有人接受合谷、四神聪、百会等固定组合操作，没有单穴位、单经络对照，因此不能把任何变化归因到某一个穴位或经络。
- 四诊目前只有 1 名原始量化字段前后配对，不推断证型，也不进入相关或机器学习模型。

### 3. AI：做质量感知的组织与发现，不做虚假的预测

本阶段 AI/算法的实际作用包括：

- 依据实验协议把运动拆成摸头、摸膝、摸腰和重复 block，形成任务空间指纹；
- 用会话内动态状态、占用熵、转移熵和驻留结构描述时间组织变化；
- 用缺失感知的 robust-z 将多模态变化组织为 N-of-1 反应指纹；
- 用精确任务对齐、有效 `n`、方向计数和留一法识别小样本过拟合；
- 用分段 EEG 质量门控区分“局部坏导联”和“整段广泛伪迹”。

![逐人多模态反应指纹](figures/04_nof1_response_fingerprint.png)

当前 `n=7` 不适合训练深度学习疗效分类器。这里的“AI 发挥作用”应体现在可解释的多模态结构发现、质量控制和下一轮可验证假设，而不是给每个人生成一个看似精确的疗效分数。

## EEG 分段质量增强：覆盖增加，但仍不能升为主证据

![EEG 质量敏感性](figures/08_eeg_quality_sensitivity.png)

- 14 个受试者—访视单元中，13 个具备完整协议时间覆盖；`2026081502/base_line` 的 EEG 仅约 46 秒，虽有完整 fNIRS marker，也不能被反推为完整 EEG 访视。
- 严格阈值 `broad_bad_epoch_fraction ≤ 0.35` 下，完整访视平均有效 `pre+execution` block 为 baseline {values['strict_baseline_blocks']:.1f}、sec_test {values['strict_followup_blocks']:.1f}。
- 旧连续记录与分段门控 event-theta 仅 {values['event_n']}/7 可比较，方向一致 {values['event_agree']}/{values['event_n']}；alpha/beta ratio 为 {values['ratio_n']}/7 可比较，方向一致 {values['ratio_agree']}/{values['ratio_n']}。
- 两条分支的滤波、聚合和时间窗定义不同，是**估计规则敏感性对照**，不是同一指标的复测验证。

因此 EEG 当前最有价值的发现是“任务/访视间质量负担不均、分段保留优于整次一票否决”，而不是某个稳定的疗效标志物。

## 目前可向领导汇报的四个“有意思但不过度”的发现

1. **任务空间排序保留**：摸头的空间变化幅度高于摸膝，n=7 中 6/7 同向，是目前最直观的认知—运动任务候选。
2. **网络—运动桥接仍同向**：相关强度从 `+0.80` 回落到 `+0.54`，但 LOO 始终为正，更符合“稳定方向、强度待验证”的表述。
3. **动态结果被筛选**：占用熵下降可保留，状态切换率下降未复现。这证明扩样和压力测试确实在淘汰不稳结果。
4. **认知锚点开始形成**：3 名眼动过程量完整配对中，反向眼跳错误率均为负向变化；只能作为认知过程候选，不能称为疗效改善。

## 必须主动说明的边界

1. 独立样本量最多是 7 名受试者；不同指标的实际有效 `n` 可低至 1、3、5 或 6。
2. `base_line → sec_test` 是约两周纵向变化，不是针刺当下的即时效应。
3. 固定组合方案没有单穴位对照，不能宣称合谷、四神聪或百会中的单一穴位效应。
4. fNIRS 当前是光密度 proxy；探头坐标为设备空间代理，尚未完成短距离回归和个体 EEG/fNIRS 配准。
5. 动态状态是会话内数学标签，不能命名为认知状态、经络状态或治疗状态。
6. bootstrap、sign-flip、Spearman 和 LOO 都是小样本敏感性诊断，不构成确认性统计。

## 建议下一步

1. 冻结本发布包的指标定义、任务顺序、EEG 阈值和有效 `n` 规则，后续新增配对数据直接做外部复核。
2. 对眼动人工转录和四诊原始字段完成第二人核对；在配对覆盖增加前不做多变量相关。
3. EEG 优先解决采集完整性、帽体位移和电极接触，不再无边界叠加预处理算法；分段门控作为固定敏感性分支保留。
4. 若原始 fNIRS 条件允许，下一技术升级应是短距离通道回归、HbO/HbR 转换和空间配准，而不是直接训练黑箱模型。

## 汇报顺序

先展示 `03 队列扩充压力测试`，再展示 `01 三层正式总览`、`05 任务空间图`、`04 个体指纹`；眼动/四诊和 EEG 放在主结果之后，作为“中医量化接口”和“主动质量控制”两条补充分支。所有源 CSV、分析字典和哈希清单位于同目录的 `data/` 与 `provenance/`。
"""


def write_readme() -> None:
    text = """# 2026-09-03 corrected paired-seven release

This is the compact leadership release for the corrected n=7 longitudinal queue.

- `01_领导进展报告.md` and `.html`: presentation-ready Chinese report.
- `figures/`: formal figures in PNG/SVG/PDF; representative multimodal views are PNG only.
- `data/`: source tables used by the report and an analysis dictionary with effective n and guardrails.
- `data/movement_emg_markers.csv`: audit table for EEG/fNIRS-derived movement markers mapped to EMG, with marker 1 defined as EMG frame 0.
- `data/phase19_*`: quality-gated graph node, edge, quality-gate and participant attribution tables; this branch is exploratory and is not part of the formal efficacy panels.
- `provenance/`: source map, figure contract, method audits, cleanup record, and SHA-256 manifest.

The independent unit is the participant. Baseline to follow-up is approximately two weeks under a fixed combination acupuncture/electroacupuncture protocol. No result is a clinical efficacy, immediate stimulation, single-point, meridian-localization, or confirmatory statistical claim.
"""
    (RELEASE_ROOT / "00_README.md").write_text(text, encoding="utf-8")


def write_figure_contract() -> None:
    text = """# Figure contract

- Core conclusion: cohort expansion to corrected paired n=7 preserves selected network/task directions while exposing unstable dynamic and EEG results.
- Archetype: quantitative grid with one leadership summary and subordinate robustness/coverage panels.
- Backend: Python/matplotlib for every plot, preview, and export.
- Evidence hierarchy: cohort stress test; three-layer dashboard; task and participant fingerprints; TCM/eye and EEG sensitivity supplements.
- Exploratory AI supplement: phase19 theory-led graph with paired quality gates and leave-one-subject-out edge attributions; it is explicitly separated from the formal dashboard.
- Statistics: participant-level mean/median, sign count, Spearman, and leave-one-participant-out diagnostics; no confirmatory p-value claim.
- Source data: every quantitative release figure maps to a CSV in `data/` or its source map.
- Image integrity: no microscopy or selective image adjustment; representative EEG/fNIRS panels are descriptive and not spatially co-registered.
- Reviewer risks: tiny n, unequal effective n, optical-density proxy, session-local state labels, manual eye-report transcription, and non-equivalent EEG estimators are stated in the report.
"""
    (RELEASE_ROOT / "provenance" / "figure_contract.md").write_text(text, encoding="utf-8")


def write_html(report: str) -> None:
    body = markdown.markdown(report, extensions=["tables", "fenced_code"])
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>n=7 阶段性进展汇报</title><style>
body{{font-family:"Microsoft YaHei","Noto Sans CJK SC",Arial,sans-serif;max-width:1100px;margin:32px auto;padding:0 24px;color:#20252b;line-height:1.72}}h1,h2,h3{{line-height:1.3;color:#17324d}}h1{{border-bottom:3px solid #2a9d8f;padding-bottom:12px}}h2{{margin-top:34px;border-bottom:1px solid #d6dde3;padding-bottom:7px}}img{{display:block;max-width:100%;height:auto;margin:18px auto 26px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border:1px solid #ccd4dc;padding:8px 10px;text-align:left;vertical-align:top}}th{{background:#eef3f5}}blockquote{{margin:18px 0;padding:10px 18px;border-left:4px solid #2a9d8f;background:#f5f8f8}}code{{background:#f0f2f4;padding:2px 5px;border-radius:3px}}@media print{{body{{max-width:none;margin:0}}h2{{break-after:avoid}}img,table{{break-inside:avoid}}}}
</style></head><body>{body}</body></html>"""
    (RELEASE_ROOT / "01_领导进展报告.html").write_text(html, encoding="utf-8")


def write_manifest() -> None:
    rows = []
    manifest = RELEASE_ROOT / "provenance" / "release_file_manifest.csv"
    for path in sorted(RELEASE_ROOT.rglob("*")):
        if path.is_file() and path != manifest:
            rows.append({"relative_path": path.relative_to(RELEASE_ROOT).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(rows).to_csv(manifest, index=False, encoding="utf-8-sig")


def main(refresh: bool = False) -> None:
    if RELEASE_ROOT.exists() and any(RELEASE_ROOT.iterdir()) and not refresh:
        raise SystemExit(f"Release directory already exists and is not empty: {RELEASE_ROOT}")
    (RELEASE_ROOT / "figures").mkdir(parents=True, exist_ok=True)
    (RELEASE_ROOT / "data").mkdir(parents=True, exist_ok=True)
    (RELEASE_ROOT / "provenance").mkdir(parents=True, exist_ok=True)
    collect_release_files()
    analysis_dictionary()
    report = build_report(metrics())
    (RELEASE_ROOT / "01_领导进展报告.md").write_text(report, encoding="utf-8")
    write_html(report)
    write_readme()
    write_figure_contract()
    write_manifest()
    metadata = {
        "cohort": list(PAIRED_SUBJECTS),
        "n": len(PAIRED_SUBJECTS),
        "sessions": ["base_line", "sec_test"],
        "interval": "approximately two weeks",
        "intervention": "fixed combination acupuncture/electroacupuncture",
        "unit_of_analysis": "participant",
    }
    (RELEASE_ROOT / "provenance" / "release_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_manifest()
    print(f"Wrote {RELEASE_ROOT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh enumerated release files in place without deleting audit records.",
    )
    main(refresh=parser.parse_args().refresh)
