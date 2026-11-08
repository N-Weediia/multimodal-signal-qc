# 项目目录与数据产物资产全景

本文档系统盘点多模态脑机与运动功能评估系统（EEF_analyze）的代码组织、工具脚本分类、测试覆盖范围、产物目录层级及归档规范。

---

## 1. 根目录核心模块职责分工

| 文件/目录 | 职责定位 | 核心输入/依赖 | 关键输出/作用 |
| :--- | :--- | :--- | :--- |
| [`main.py`](file:///d:/Python_File/EEF_analyze/main.py) | 多模态离线质量评估 (QC) 主入口 | 原始 BDF, CSV, NIRS | `outputs/qc/current/` (报告、诊断图、inventory) |
| [`preprocess.py`](file:///d:/Python_File/EEF_analyze/preprocess.py) | 规范化去噪预处理与掩码提取入口 | 原始数据 | `outputs/preprocessed/` (`*_cleaned.npz`, index.json) |
| [`emg_qc.py`](file:///d:/Python_File/EEF_analyze/emg_qc.py) | 表面肌电与 IMU 质量评估底层算法库 | Delsys CSV 时序 | 肌肉映射对齐、工频谐波比、倾角代理 |
| [`fnirs_qc.py`](file:///d:/Python_File/EEF_analyze/fnirs_qc.py) | 近红外脑血氧质量评估底层算法库 | Homer `.nirs` | 双波长匹配、SCI、CV、红外干扰识别 |
| [`low_snr_recovery.py`](file:///d:/Python_File/EEF_analyze/low_snr_recovery.py) | 低信噪比信号实验性恢复探索 | 原始记录 | Hampel/TDDR 阶跃修正、谐波回归分支 |
| [`expand_all_cases.py`](file:///d:/Python_File/EEF_analyze/expand_all_cases.py) | 全量病例特征展开与提取 | 原始数据 + QC Inventory | 受试者级动作 Block 特征表 |
| [`summarize_expanded_results.py`](file:///d:/Python_File/EEF_analyze/summarize_expanded_results.py) | 特征表多任务汇总与配对统计 | 展开特征表 | 任务响应均值、质控合格率 |
| [`quality_gated_phenotype.py`](file:///d:/Python_File/EEF_analyze/quality_gated_phenotype.py) | 质量门控表型构建与综合矩阵 | 展开特征 + QC 评分 | 降维表型、清洗后特征矩阵 |
| [`integrate_tcm_reports.py`](file:///d:/Python_File/EEF_analyze/integrate_tcm_reports.py) | 中医四诊与眼动仪量化报告整合 | 报告文件 (PDF/XLSX) | 多源表型关联清单 |
| [`mbll_relative_sensitivity_reviewed_20260908.py`](file:///d:/Python_File/EEF_analyze/mbll_relative_sensitivity_reviewed_20260908.py) | 相对 MBLL 敏感性分析脚本 | 原始数据 + 预处理 OD | HbO/HbR 敏感性网格评估 |
| [`plot_eeg_montage.py`](file:///d:/Python_File/EEF_analyze/plot_eeg_montage.py) | 64导 EEG 国际 10-10 拓扑排布绘图 | 电极物理坐标与 ROI 分区 | 矢量与高清脑电排布图 |
| [`plot_eeg_montage_hires.py`](file:///d:/Python_File/EEF_analyze/plot_eeg_montage_hires.py) | 高分印刷排版脑电图生成 | 同上 | 300 DPI 印刷级图版 |
| [`build_best_subject_preprocessing_comparison_20260914.py`](file:///d:/Python_File/EEF_analyze/build_best_subject_preprocessing_comparison_20260914.py) | 最佳受试者预处理效果展示生成器 | `review_20260914` 产物 | 6 组对比图版与指标对比表 |
| [`build_preprocess_steps_and_figure_guide_20260914.py`](file:///d:/Python_File/EEF_analyze/build_preprocess_steps_and_figure_guide_20260914.py) | 预处理技术指南 Word 构建脚本 | 最佳受试者对比图 | Word 格式技术图解指南 |
| [`build_document_and_pdf.py`](file:///d:/Python_File/EEF_analyze/build_document_and_pdf.py) | 实验方案与多模态规格整合排版 | 方案文本、图表资产 | 方案 Word 修订版与无头渲染 PDF |

---

## 2. `scripts/` 工具与阶段性脚本分类

`scripts/` 目录下收纳了全项目在探索与发布各阶段沉淀的模块化脚本：

### 2.1 原始数据与同步工具
- [`scripts/scan_old_data.py`](file:///d:/Python_File/EEF_analyze/scripts/scan_old_data.py)：扫描并比对 J: 盘与本地目录差异，分析重复哈希。
- [`scripts/import_raw_subset.py`](file:///d:/Python_File/EEF_analyze/scripts/import_raw_subset.py)：受控的原始数据哈希校验拉取工具（支持 `--apply` 与 dry-run）。
- [`scripts/paired7_config.py`](file:///d:/Python_File/EEF_analyze/scripts/paired7_config.py)：7 名配对受试者公共配置与阶段输出路径解析工具。
- [`scripts/build_incremental_analysis_contract.py`](file:///d:/Python_File/EEF_analyze/scripts/build_incremental_analysis_contract.py)：增量分析契约与哈希账本生成工具。
- [`scripts/build_marker_alignment_audit.py`](file:///d:/Python_File/EEF_analyze/scripts/build_marker_alignment_audit.py)：跨模态时间事件标记对齐审计。
- [`scripts/phase1_audit.py`](file:///d:/Python_File/EEF_analyze/scripts/phase1_audit.py) & [`scripts/phase1_cleanup.py`](file:///d:/Python_File/EEF_analyze/scripts/phase1_cleanup.py)：第一阶段数据清理账本与哈希验证。

### 2.2 Phase 1 ~ 19 分析脚本（探索与敏感性分支）
- [`scripts/phase5_spatial_visuals.py`](file:///d:/Python_File/EEF_analyze/scripts/phase5_spatial_visuals.py) & [`scripts/phase5_task_rsa.py`](file:///d:/Python_File/EEF_analyze/scripts/phase5_task_rsa.py)：头皮空间分布与任务表征相似度分析（RSA）。
- [`scripts/phase6_crossmodal_markers.py`](file:///d:/Python_File/EEF_analyze/scripts/phase6_crossmodal_markers.py)：跨模态标志物关联。
- [`scripts/phase7_block_dynamics.py`](file:///d:/Python_File/EEF_analyze/scripts/phase7_block_dynamics.py)：各动作 Block 内部与跨 Block 动态衰减分析。
- [`scripts/phase8_state_transition_entropy.py`](file:///d:/Python_File/EEF_analyze/scripts/phase8_state_transition_entropy.py)：多稳态脑网络状态转移熵计算。
- [`scripts/phase9_tcm_eye_anchor.py`](file:///d:/Python_File/EEF_analyze/scripts/phase9_tcm_eye_anchor.py)：眼动指标与中医证候锚点关联。
- [`scripts/phase10_latent_brain_muscle_coupling.py`](file:///d:/Python_File/EEF_analyze/scripts/phase10_latent_brain_muscle_coupling.py)：皮质-肌电潜在耦合分解（CMC）。
- [`scripts/phase11_nof1_response_fingerprint.py`](file:///d:/Python_File/EEF_analyze/scripts/phase11_nof1_response_fingerprint.py)：个体化 N-of-1 响应指纹分析。
- [`scripts/phase14_fixed_three_marker_stability.py`](file:///d:/Python_File/EEF_analyze/scripts/phase14_fixed_three_marker_stability.py) & [`scripts/phase14_task_spatial_redistribution.py`](file:///d:/Python_File/EEF_analyze/scripts/phase14_task_spatial_redistribution.py)：三动作稳定性与空间拓扑重构。
- [`scripts/phase15_task_pairwise_contrasts.py`](file:///d:/Python_File/EEF_analyze/scripts/phase15_task_pairwise_contrasts.py)：动作间成对对比统计。
- [`scripts/phase16_formal_dashboard.py`](file:///d:/Python_File/EEF_analyze/scripts/phase16_formal_dashboard.py) & [`scripts/phase16_open_exploration.py`](file:///d:/Python_File/EEF_analyze/scripts/phase16_open_exploration.py)：形式化仪表板与探索性指标集。
- [`scripts/phase17_briefing_figure.py`](file:///d:/Python_File/EEF_analyze/scripts/phase17_briefing_figure.py) & [`scripts/phase17_eeg_quality_sensitivity.py`](file:///d:/Python_File/EEF_analyze/scripts/phase17_eeg_quality_sensitivity.py)：运动脑电质量门控敏感性评估。
- [`scripts/phase18_refresh_comparison.py`](file:///d:/Python_File/EEF_analyze/scripts/phase18_refresh_comparison.py)：前后访视刷新对比。
- [`scripts/phase19_quality_gated_multimodal_graph.py`](file:///d:/Python_File/EEF_analyze/scripts/phase19_quality_gated_multimodal_graph.py)：跨模态假说网络图（留一受试者稳健检验）。
- [`scripts/analyze_emg_imu.py`](file:///d:/Python_File/EEF_analyze/scripts/analyze_emg_imu.py)：IMU 字段盘点与 1 秒运动学姿态代理提取。
- [`scripts/segment_eeg_quality.py`](file:///d:/Python_File/EEF_analyze/scripts/segment_eeg_quality.py)：运动任务分段脑电质量审计。

### 2.3 报告构建与发布辅助工具
- [`scripts/build_paired7_release.py`](file:///d:/Python_File/EEF_analyze/scripts/build_paired7_release.py) & [`scripts/validate_paired7_release.py`](file:///d:/Python_File/EEF_analyze/scripts/validate_paired7_release.py)：配对 7 人发布包封装与完整性哈希校验。
- [`scripts/build_plain_teacher_report_20260908.py`](file:///d:/Python_File/EEF_analyze/scripts/build_plain_teacher_report_20260908.py) & [`scripts/build_teacher_report_20260904_visual.py`](file:///d:/Python_File/EEF_analyze/scripts/build_teacher_report_20260904_visual.py)：汇报图文报告自动构建。
- [`scripts/render_plain_report_word.ps1`](file:///d:/Python_File/EEF_analyze/scripts/render_plain_report_word.ps1) 等：Word 文档锁定检测与格式渲染脚本。

---

## 3. `tests/` 自动化测试清单与覆盖范围

系统在 `tests/` 下建立了一套基于 `pytest` 的单元测试与冒烟测试集：

| 测试文件 | 目标模块 / 测试覆盖要点 |
| :--- | :--- |
| [`tests/test_multimodal_qc.py`](file:///d:/Python_File/EEF_analyze/tests/test_multimodal_qc.py) | 核心质控测试：覆盖 EEG 59/31 导判定、中位数重参考、sEMG 谐波比、fNIRS SCI/CV 及 `main.py` 批处理 |
| [`tests/test_analyze_emg_imu.py`](file:///d:/Python_File/EEF_analyze/tests/test_analyze_emg_imu.py) | IMU 解析测试：检验 Acc/Gyro 字段占有率、1 秒倾角代理及重力范数门控 |
| [`tests/test_explore_emg_mapping.py`](file:///d:/Python_File/EEF_analyze/tests/test_explore_emg_mapping.py) | 肌电肌肉映射测试：检验标准 7 通道映射与 `2026080901` 的 `shifted_v1` 映射转换 |
| [`tests/test_explore_eye_fnirs.py`](file:///d:/Python_File/EEF_analyze/tests/test_explore_eye_fnirs.py) | 近红外眼动干扰测试：检验 `READY_WITH_MASKS_KNOWN_IR_INTERFERENCE` 状态触发与掩码 |
| [`tests/test_marker_alignment_audit.py`](file:///d:/Python_File/EEF_analyze/tests/test_marker_alignment_audit.py) | 跨模态时间对齐测试：运动首标对齐、眼动 EEG/fNIRS 对齐及静息态无标处理 |
| [`tests/test_marker_protocol_labels.py`](file:///d:/Python_File/EEF_analyze/tests/test_marker_protocol_labels.py) | 事件协议分配测试：摸头、摸膝、摸腰按顺序赋值与五标眼动反推逻辑 |
| [`tests/test_low_snr_recovery.py`](file:///d:/Python_File/EEF_analyze/tests/test_low_snr_recovery.py) | 低信噪比恢复测试：阶跃检测、Hampel 滤波、EMG 谐波回归验证 |
| [`tests/test_mbll_relative_sensitivity_reviewed.py`](file:///d:/Python_File/EEF_analyze/tests/test_mbll_relative_sensitivity_reviewed.py) | 相对 MBLL 敏感性测试：消光系数转换、滤波网格与 DPF 稳定性 |
| [`tests/run_mbll_relative_sensitivity_reviewed_smoke.py`](file:///d:/Python_File/EEF_analyze/tests/run_mbll_relative_sensitivity_reviewed_smoke.py) | 零外部依赖同解释器 MBLL 冒烟测试 |
| [`tests/test_import_raw_subset.py`](file:///d:/Python_File/EEF_analyze/tests/test_import_raw_subset.py) | 数据子集导入测试：dry-run 规划、哈希冲突不覆盖及扩展名过滤 |
| [`tests/test_incremental_analysis_contract.py`](file:///d:/Python_File/EEF_analyze/tests/test_incremental_analysis_contract.py) | 增量分析契约测试：任务账本生成与三层门控逻辑 |
| [`tests/test_phase19_quality_gated_multimodal_graph.py`](file:///d:/Python_File/EEF_analyze/tests/test_phase19_quality_gated_multimodal_graph.py) | 跨模态图模型测试：LOO 稳健 z 分数与网络边稳定性 |
| 其他阶段测试 (`test_phase*.py`, `test_quality_gated_phenotype.py` 等) | 阶段 1~18 各特征计算函数的单元测试 |

---

## 4. `outputs/` 输出目录层次规范

`outputs/` 统一存放所有程序输出，其层次结构如下：

```text
outputs/
├── analysis/               # 特征提取与高级分析产物
│   ├── 20260903_all_cases_canonical/  # 18名基线全样本标准化分析
│   └── paired7_20260903/              # 配对 7 人分阶段分析 (phase5~19)
├── qc/                     # 质控输出根目录
│   ├── current/            # main.py 默认输出目录
│   └── local_repro_20260903/ # 本地复现运行质控
├── preprocessed/           # 预处理数据根目录
│   ├── current/            # preprocess.py 默认输出目录
│   └── stage_20260904_...  # 历史验证过的阶段预处理数据
├── release_20260903_paired7/ # 正式对外汇报发布包 (含报告、图表、源数据与哈希)
├── release_20260903/       # 旧版 n=4 冻结归档发布包 (仅作审计留痕)
├── doc_figures/            # 实验方案文档中引用的核心矢量/位图资源
├── imu_posture_*/          # IMU 运动学与姿态代理提取产物
├── mbll_reviewed_20260908/ # 相对 MBLL 敏感性分析产物 (含 run_manifest.json)
└── _scratch/               # 临时审查、调试与中间生成物目录 (Git 忽略，可安全清空)
    ├── plain_tmp/          # 临时报告渲染排版生成物
    └── tmp/                # 临时图表与手工审查缓存
```

---

## 5. 阶段性产物目录说明与版本演进

项目根目录下保留了数个关键历史阶段的产物目录，其版本与状态口径如下：

### 5.1 `best_subject_preprocessing_comparison_20260914/`
- **定位**：预处理算法展示标杆案例。
- **代表受试者**：`2026081502/sec_test`（综合质控评分 98.563）。
- **内容**：01 受试者筛选分布、02 EEG 时域前后对比、03 EEG 节律能量比、04 fNIRS OD与相对MBLL、05 sEMG 包络、06 IMU 姿态代理。
- **解释红线**：这是算法技术效果展示，**不构成**任何临床疗效推断。

### 5.2 `review_20260914/` 与 `optimized_20260914/`
- **数据源与规模**：均覆盖 7 名配对受试者的 107 条记录。
- **Review 态 (`review_20260914`)**：
  - 流程版本：`2026-09-14.v5`；
  - 统计：41 `READY_WITH_MASKS`、33 `READY_CHANNEL_FILTERED`、14 `READY_CONTINUOUS`、16 `READY_WITH_MASKS_KNOWN_IR_INTERFERENCE`、3 `FAILED`。
- **Optimized 态 (`optimized_20260914`)**：
  - 流程版本：`2026-09-14.v6-quality-tiered-recovery`；
  - 统计：42、33、14、16、2（`FAILED` 减少 1 条）；
  - **唯一状态变化**：`2026081401\base_line\EEG\运动\data.bdf` 由 `FAILED` 成功优化恢复为 `READY_WITH_MASKS`。

### 5.3 `mbll_relative_sensitivity_20260908/`
- **定位**：配对 7 人运动任务的 MBLL 敏感性原型及汇报幻灯片（含 Node.js 渲染脚本与 PPTX）。
- **核心数据**：包含 14 条运动记录、15,600 事件行与多网格灵敏度对比。

---

## 6. 临时与历史归档管理

1. **`archive/` 目录**：
   - 存放历史过时分析产物与阶段性日志备份（如旧版 outputs 归档）。保持制度性冻结，不在此写入新数据。
2. **`outputs/_scratch/` 目录**：
   - 统一收纳所有临时调试、试跑输出、中间中间文档（如原根目录下的 `plain_tmp`、`tmp`）。
   - 该目录已在 `.gitignore` 中声明，不会污染 Git 历史，可随时整体清理。
