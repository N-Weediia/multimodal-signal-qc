# 项目目录与数据产物资产全景

本文档系统盘点多模态脑机与运动功能评估系统（EEF_analyze）的代码组织、工具脚本分类、测试覆盖范围、产物目录层级及归档规范。

---

## 1. 架构升级与代码模块化分层 (`src/`)

自系统版本 `v8.0.0` 起，项目全面实行**模态与阶段正交分离**的模块化架构，核心底层算法与执行流水线统一归整于 `src/` 目录下，根目录保留经典无破坏门面（Facade）：

```text
src/
├── modalities/                 # 【按模态划分：物理与生理信号处理算法】
│   ├── eeg/                    # 脑电模态 (qc.py, preprocess.py, features.py, montage.py, montage_hires.py)
│   ├── emg/                    # 表面肌电模态 (qc.py, preprocess.py, imu.py, mapping.py)
│   ├── fnirs/                  # 近红外模态 (qc.py, preprocess.py, mbll.py, mbll_v1.py)
│   └── multimodal/             # 跨模态整合 (alignment.py, cmc.py, cross_network.py, tcm_eye.py)
│
├── stages/                     # 【按业务阶段划分：全流程执行流水线】
│   ├── stage0_audit_sync/      # Stage 0: 原始数据只读审计、哈希校验与受控拉取
│   ├── stage1_qc/              # Stage 1: 多模态独立离线质量筛查 (pipeline.py)
│   ├── stage2_preprocess/      # Stage 2: 规范预处理与清洗导出 (pipeline.py)
│   ├── stage3_recovery/        # Stage 3: 低信噪比与伪迹分级恢复 (low_snr_recovery.py, subspace_shrinkage.py)
│   ├── stage4_features/        # Stage 4: 全量病例特征展开与表型整合 (expand_cases, summarize, phenotype)
│   ├── stage5_analysis/        # Stage 5: 专项分析分支 (mbll_sensitivity, multimodal_graph, exploratory_phases)
│   └── stage6_reporting/       # Stage 6: 汇报文档排版与图表指南构建 (teacher_reports, best_subject, protocol)
│
├── tools/                      # 【诊断与维护工具】
│   ├── sync_emg_summaries.py   # NPZ 与 JSON 摘要字段同步
│   └── check_*.py              # 案例与通道有效性独立核验工具
│
├── utils/                      # 【通用支持库】
│   ├── safety.py               # J: 严格只读断言与输出路径边界核查
│   ├── config.py               # 路径常量、会话清单与任务 Marker 协议
│   └── visualization.py        # 中文字体、统一配色与高分辨率画布配置
│
└── version.py                  # 全局版本常数与演进历史 (v8.0.0)
```

### 根目录经典门面接口映射

为保证现有文档手册与命令行 100% 稳定运行，根目录继续提供兼容门面：

| 根目录门面文件 | 模块化映射目标 | 职责定位 |
| :--- | :--- | :--- |
| [`main.py`](file:///d:/Python_File/EEF_analyze/main.py) | `src.stages.stage1_qc.pipeline` | 多模态离线质量评估 (QC) 主入口 |
| [`preprocess.py`](file:///d:/Python_File/EEF_analyze/preprocess.py) | `src.stages.stage2_preprocess.pipeline` | 规范化去噪预处理与掩码提取主入口 (v8) |
| [`emg_qc.py`](file:///d:/Python_File/EEF_analyze/emg_qc.py) | `src.modalities.emg.qc` | 表面肌电与 IMU 质量评估底层算法库 |
| [`fnirs_qc.py`](file:///d:/Python_File/EEF_analyze/fnirs_qc.py) | `src.modalities.fnirs.qc` | 近红外脑血氧质量评估底层算法库 |
| [`low_snr_recovery.py`](file:///d:/Python_File/EEF_analyze/low_snr_recovery.py) | `src.stages.stage3_recovery.low_snr_recovery` | 低信噪比信号实验性恢复探索 |
| [`expand_all_cases.py`](file:///d:/Python_File/EEF_analyze/expand_all_cases.py) | `src.stages.stage4_features.expand_cases` | 全量病例特征展开与提取 |
| [`summarize_expanded_results.py`](file:///d:/Python_File/EEF_analyze/summarize_expanded_results.py) | `src.stages.stage4_features.summarize_features` | 特征表多任务汇总与配对统计 |
| [`quality_gated_phenotype.py`](file:///d:/Python_File/EEF_analyze/quality_gated_phenotype.py) | `src.stages.stage4_features.quality_gated_phenotype` | 质量门控表型构建与综合矩阵 |
| [`integrate_tcm_reports.py`](file:///d:/Python_File/EEF_analyze/integrate_tcm_reports.py) | `src.modalities.multimodal.tcm_eye` | 中医四诊与眼动仪量化报告整合 |
| [`plot_eeg_montage.py`](file:///d:/Python_File/EEF_analyze/plot_eeg_montage.py) | `src.modalities.eeg.montage` | 64导 EEG 国际 10-10 拓扑排布绘图 |

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

## 4. `outputs/` 统一归类与结果归纳

所有生成物严格归纳于 `outputs/` 的规范分类目录中，杜绝版本混淆。详见 [06_RESULTS_CATALOG.md](file:///d:/Python_File/EEF_analyze/docs/06_RESULTS_CATALOG.md)：

```text
outputs/
├── qc/                              # [Stage 1] 多模态离线质量控制产物
│   ├── current/                     # 全样本质控评估默认输出 (recording_inventory.csv + 逐记录诊断图)
│   ├── local_repro_20260903/        # 配对 7 人本地副本复现质控评估产物
│   └── 20260903_summary/            # 历史汇总快照
├── preprocessed/                    # [Stage 2] 规范预处理清洗数据 (NPZ + index.json)
│   ├── current/                     # 当前最新预处理成果 (v8: TDDR + WOLA + PASS)
│   ├── stage_20260904_...           # 阶段验证预处理数据
│   ├── v5_review_20260914/          # 2026-09-14 Review 态 (107记录，3 FAILED)
│   └── v6_optimized_20260914/       # 2026-09-14 Optimized 态 (107记录，2 FAILED)
├── analysis/                        # [Stage 4 & 5] 特征提取与专项假说分析
│   ├── 20260903_all_cases_canonical/# 18名基线全样本标准化动作 Block 特征表
│   ├── paired7_20260903/            # 配对 7 人分阶段分析 (Phase 1~19, 含 Phase 19 跨模态网络)
│   ├── mbll_reviewed_20260908/      # 配对 7 人相对 MBLL 敏感性分析产物
│   └── imu_posture_20260904/        # 表面肌电 Avanti IMU 1秒姿态角与运动强度代理
├── reports/                         # [Stage 6] 成果汇报、图版指南与实验方案排布
│   ├── protocols_and_montage/       # 实验方案 Word/PDF 修订版与 64 导 EEG 拓扑排布图
│   ├── briefings/                   # 导师汇报简报与阶段总结
│   ├── comparisons/                 # 最佳受试者 (2026081502/sec_test) 预处理前后对比全套图解
│   └── teacher_reports/             # 阶段性导师汇报图文成果包 (20260904, 20260908)
├── audit_ledgers/                   # [Stage 0] 原始数据扫描、安全同步与哈希审计账本
│   ├── j_old_data_scan_20260902/    # J: 盘与本地目录哈希比对与重复分析
│   ├── j_drive_data_inventory_20260902.csv # J: 盘只读资产总表
│   └── local_data_inventory_20260902.csv   # 本地资产总表
├── release/                         # 正式发布打包与哈希封存 (release_20260903_paired7 等)
├── literature/                      # 外部方法学文献对照、迁移矩阵与搜索记录
└── _scratch/                        # 临时审查、调试与中间沙盒 (Git 忽略，可安全清空)
    └── recovery_v8_validation/      # v8 升级算法 (TDDR/WOLA/PASS) 诊断验证表与图表
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
