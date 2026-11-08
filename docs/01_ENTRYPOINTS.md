# 全流程与各阶段入口清单及执行手册

本手册系统梳理多模态脑机与运动功能评估系统（EEF_analyze）从 Stage 0 到 Stage 6 的全部关键业务执行入口，提供标准命令行、参数解析、核心输入输出及避坑指引。

---

## 阶段概览与数据流

| 阶段 | 核心任务 | 主要脚本入口 | 核心输入 | 核心输出 |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 0** | 数据审计与安全同步 | [`scripts/scan_old_data.py`](file:///d:/Python_File/EEF_analyze/scripts/scan_old_data.py)<br>[`scripts/import_raw_subset.py`](file:///d:/Python_File/EEF_analyze/scripts/import_raw_subset.py) | `J:\2026二康\数据` (只读) | `outputs/j_old_data_scan_20260902/`<br>`data/` (本地原始子集) |
| **Stage 1** | 多模态独立质量控制 (QC) | [`main.py`](file:///d:/Python_File/EEF_analyze/main.py) | `data/` 或 `data_repro_20260903/` | `outputs/qc/current/`<br>`recording_inventory.csv` |
| **Stage 2** | 规范预处理与质量掩码 | [`preprocess.py`](file:///d:/Python_File/EEF_analyze/preprocess.py) | `data/` 或 `data_repro_20260903/` | `outputs/preprocessed/current/`<br>`preprocessing_index.json` |
| **Stage 3** | 低信噪比实验性恢复 | [`low_snr_recovery.py`](file:///d:/Python_File/EEF_analyze/low_snr_recovery.py) | 原始运动记录 | `outputs/_scratch/low_snr_20260903/` |
| **Stage 4** | 特征提取与表型整合 | [`expand_all_cases.py`](file:///d:/Python_File/EEF_analyze/expand_all_cases.py)<br>[`summarize_expanded_results.py`](file:///d:/Python_File/EEF_analyze/summarize_expanded_results.py)<br>[`quality_gated_phenotype.py`](file:///d:/Python_File/EEF_analyze/quality_gated_phenotype.py)<br>[`integrate_tcm_reports.py`](file:///d:/Python_File/EEF_analyze/integrate_tcm_reports.py) | 原始数据 + QC Inventory + 四诊/眼动报告 | `outputs/analysis/`<br>`outputs/phase4_20260901/` |
| **Stage 5** | 专项分析分支 | [`mbll_relative_sensitivity_reviewed_20260908.py`](file:///d:/Python_File/EEF_analyze/mbll_relative_sensitivity_reviewed_20260908.py)<br>[`scripts/analyze_emg_imu.py`](file:///d:/Python_File/EEF_analyze/scripts/analyze_emg_imu.py)<br>[`scripts/segment_eeg_quality.py`](file:///d:/Python_File/EEF_analyze/scripts/segment_eeg_quality.py)<br>[`scripts/phase19_quality_gated_multimodal_graph.py`](file:///d:/Python_File/EEF_analyze/scripts/phase19_quality_gated_multimodal_graph.py) | 预处理 NPZ、IMU CSV、差值特征 | `outputs/mbll_reviewed_20260908/`<br>`outputs/imu_posture_20260904/`<br>`outputs/analysis/.../phase19/` |
| **Stage 6** | 报告与可视化排布 | [`plot_eeg_montage.py`](file:///d:/Python_File/EEF_analyze/plot_eeg_montage.py)<br>[`plot_eeg_montage_hires.py`](file:///d:/Python_File/EEF_analyze/plot_eeg_montage_hires.py)<br>[`build_best_subject_preprocessing_comparison_20260914.py`](file:///d:/Python_File/EEF_analyze/build_best_subject_preprocessing_comparison_20260914.py)<br>[`build_preprocess_steps_and_figure_guide_20260914.py`](file:///d:/Python_File/EEF_analyze/build_preprocess_steps_and_figure_guide_20260914.py)<br>[`build_document_and_pdf.py`](file:///d:/Python_File/EEF_analyze/build_document_and_pdf.py) | 预处理图表、排布坐标、实验方案源图 | `outputs/eeg_64ch_montage_layout.png`<br>`best_subject_preprocessing_comparison_20260914/`<br>Word/PDF 方案文档 |

---

## Stage 0: 原始数据审计与受控同步

### 1.1 数据盘扫描与重复比对: `scripts/scan_old_data.py`
- **功能定位**：非破坏性盘点 J: 盘目录与本地 `data/` 的路径/大小/哈希差异，识别重复组与异常命名。
- **前置条件**：J: 盘挂载可用；本地存在核对表与基础 inventory。
- **运行命令**：
  ```powershell
  python scripts/scan_old_data.py --j-root "J:\2026二康\数据" --local-root data --output outputs/j_old_data_scan_20260902
  ```
- **核心输入**：`J:\2026二康\数据`（只读）、`outputs/j_drive_data_inventory_20260902.csv`、`outputs/local_data_inventory_20260902.csv`。
- **核心输出**：`outputs/j_old_data_scan_20260902/`（`scan_report.md`, `duplicate_groups.csv`, `safe_redundancy_candidates.csv`, `manual_review.csv`）。
- **避坑指南**：严禁向 J: 盘写出报告；脚本默认写出至 `outputs/`。

### 1.2 受控原始数据子集导入: `scripts/import_raw_subset.py`
- **功能定位**：从 J: 盘安全复制指定受试者原始数据至本地 `data/`，包含 SHA-256 校验，防止覆盖已有不一致文件。
- **前置条件**：确认本地磁盘空间充足。
- **运行命令**：
  ```powershell
  # 步骤 1：试运行 (Dry-run)，核对计划复制的文件与大小
  python scripts/import_raw_subset.py --source-root "J:\2026二康\数据" --dest-root data --subjects 2026080901,2026081401

  # 步骤 2：正式执行复制 (--apply)
  python scripts/import_raw_subset.py --source-root "J:\2026二康\数据" --dest-root data --subjects 2026080901,2026081401 --apply
  ```
- **核心输入**：`J:\2026二康\数据`。
- **核心输出**：`data/<subject>/...` 以及 `outputs/phase2_20260901/raw_copy_manifest.csv`。
- **避坑指南**：默认不带 `--apply` 仅为 dry-run；若目标文件已存在且哈希不一致，会记录 `conflict` 且**决不覆盖**。

---

## Stage 1: 多模态独立质量控制 (QC)

### 核心入口: `main.py`
- **功能定位**：多模态数据离线采集质量初筛。独立评估 EEG（59/31导）、sEMG（7通道及工频谐波）、fNIRS（双波长配对、SCI、CV）。生成可视化诊断图与汇总清单。
- **模块依赖**：[`emg_qc.py`](file:///d:/Python_File/EEF_analyze/emg_qc.py), [`fnirs_qc.py`](file:///d:/Python_File/EEF_analyze/fnirs_qc.py)。
- **前置条件**：原始记录存放于指定数据目录。
- **常用运行命令**：
  ```powershell
  # 全量评估本地 data 目录
  python main.py --overwrite

  # 指定干净本地副本评估，并输出到独立结果目录
  python main.py --data-root data_repro_20260903 --results-root outputs/qc/local_repro_20260903 --overwrite

  # 仅评估单个受试者
  python main.py --subject "2026080901" --overwrite

  # 快速巡检（仅前 60 秒）
  python main.py --max-seconds 60 --overwrite

  # 数值质控（不画图以提升速度）
  python main.py --summary-only --overwrite
  ```
- **核心输入**：`--data-root` 下的 `data.bdf`, `*.csv` (EMG), `*.nirs`。
- **核心输出**：
  - `outputs/qc/current/recording_inventory.csv`：所有记录质量总表。
  - `outputs/qc/current/analysis_index.json`：汇总索引与状态统计。
  - `outputs/qc/current/<subject>/<session>/<task>/`：逐记录包含 `quality_report.md`、`summary.json`、`signal_preview.png`、`power_spectra.png`。
- **避坑指南**：
  - **`--overwrite` 会递归删除对应的目标目录**（`shutil.rmtree`），运行前必须核对工作目录与 `--results-root`。
  - QC 的状态（如 `SUCCESS`）仅代表完成评估，`READY` 代表数据完整可进入预处理，不代表临床有效或无需人工复核。

---

## Stage 2: 规范预处理与质量掩码

### 核心入口: `preprocess.py`
- **功能定位**：生成数学可分析且具备严格审计掩码的去噪信号。**绝不修改原始数据**，只输出清洗后的 NPZ 与参数索引。
- **前置条件**：已完成原始数据校验；环境安装 numpy、scipy。
- **常用运行命令**：
  ```powershell
  # 预处理写入指定输出目录
  python preprocess.py --output-root outputs/preprocessed/current --overwrite

  # 针对复现队列运行，输出至临时审查目录
  python preprocess.py --data-root data_repro_20260903 --output-root outputs/_scratch/preprocessed --overwrite

  # 仅预处理单人
  python preprocess.py --subject "2026081401" --output-root outputs/preprocessed/current --overwrite
  ```
- **核心输入**：`data/` 中的原始记录。
- **核心输出**：
  - EEG: `eeg_cleaned.npz`（含 `data_uv`, `labels`, `sampling_rate`, `bad_channel_mask`, `interpolated_channel_mask`, `valid_2s_epoch_mask`）。
  - sEMG: `emg_cleaned.npz`（含 `data`, `time_s`, `sampling_rate`, `channel_names`, `logical_muscle_indices`, `mapping_profile`）。
  - fNIRS: `fnirs_cleaned_od.npz`（含双波长光密度 OD、`wavelength_valid_sample_mask`, `motion_artifact_mask`, `repair_mask`, `analysis_sample_mask`）。
  - 全局索引：`preprocessing_inventory.csv` 与 `preprocessing_index.json`。
- **避坑指南**：
  - 预处理内部具备缓存有效性检查（对比源文件 SHA-256、流程版本和参数哈希），未改动文件会自动跳过。若需强制刷新需传 `--overwrite`。
  - 下游分析**必须**消费 NPZ 中的 mask，绝不能将插值导联或无效 epoch 视为真实测量值。

---

## Stage 3: 低信噪比实验性恢复

### 核心入口: `low_snr_recovery.py`
- **功能定位**：针对高难度运动伪影、局部阶跃与严重工频干扰的敏感性分支探索（包含 Hampel 滤波、TDDR 阶跃恢复、EMG 谐波回归、ASR-like 原型）。
- **前置条件**：已了解该实验分支的局限性；该分支**不能**作为常规临床分析的主管线。
- **运行命令**：
  ```powershell
  python low_snr_recovery.py --out outputs/_scratch/low_snr_20260903 --data-root data_repro_20260903

  # 若需要包含耗时较长的 EEG ASR-like 试验
  python low_snr_recovery.py --out outputs/_scratch/low_snr_20260903 --data-root data_repro_20260903 --include-eeg
  ```
- **核心输入**：原始记录。
- **核心输出**：`outputs/_scratch/low_snr_20260903/` 下的敏感性对比数据与 `low_snr_recovery_metadata.json`。
- **避坑指南**：
  - 实验输出建议放在 `outputs/_scratch/` 下，复核后可清理。
  - ASR-like 与无短通道校正属于探索性质，禁止在正式论文或报告中声称“完全消除伪影”。

---

## Stage 4: 模态特征提取与表型整合

### 4.1 全量多任务特征展开: `expand_all_cases.py`
- **功能定位**：直接从原始数据目录查找并读取 BDF/NIRS/CSV，提取运动任务各动作 block 与静息态的特征指标。
- **运行命令**：
  ```powershell
  python expand_all_cases.py --data-root data_repro_20260903 --qc-inventory outputs/qc/local_repro_20260903/recording_inventory.csv --out outputs/analysis/local_repro_20260903
  ```
- **核心输出**：`expanded_analysis_metadata.json`, `expanded_subtask_features.csv`, `expanded_state_features.csv`。
- **避坑指南**：它直接读取原始文件，**不**自动消费 `preprocess.py` 的 NPZ；需显式通过 `--qc-inventory` 指定 QC 清单。

### 4.2 特征汇总与配对统计: `summarize_expanded_results.py`
- **功能定位**：基于 `expand_all_cases.py` 的产物，生成受试者级配对差值、任务响应度及质控分布。
- **运行命令**：
  ```powershell
  python summarize_expanded_results.py --out outputs/analysis/local_repro_20260903
  ```
- **核心输出**：`expanded_quality_summary.csv`, `expanded_subtask_summary.csv` 等。

### 4.3 质量门控表型构建: `quality_gated_phenotype.py`
- **功能定位**：结合 QC 质量分数进行加权或门控，剔除低质窗口后生成稳健受试者表型矩阵。
- **运行命令**：
  ```powershell
  python quality_gated_phenotype.py --analysis-root outputs/analysis/20260903_all_cases_canonical --output-root outputs/phase4_20260901
  ```

### 4.4 中医四诊与眼动报告整合: `integrate_tcm_reports.py`
- **功能定位**：清点并试点接入中医四诊（问卷/脉象）与眼动仪器报告（PDF/Excel），建立受试者级多源数据关联。
- **运行命令**：
  ```powershell
  python integrate_tcm_reports.py --out outputs/experiments/tcm_20260901 --data-root data_repro_20260903
  ```
- **核心输出**：`tcm_report_inventory.csv`, `eye_report_pilot_features.csv`, `four_diagnosis_pilot_features.csv`。

---

## Stage 5: 专项分支分析

### 5.1 相对 MBLL 光密度敏感性分析: `mbll_relative_sensitivity_reviewed_20260908.py`
- **功能定位**：针对配对 7 人的运动任务 fNIRS，基于自然对数光密度 OD 计算相对 HbO/HbR，网格化评估不同 DPF 与滤波参数的影响。
- **运行命令**：
  ```powershell
  python mbll_relative_sensitivity_reviewed_20260908.py --output outputs/mbll_reviewed_20260908
  ```
- **核心输出**：`mbll_event_responses.csv`, `mbll_subject_task_session_summary.csv`, `run_manifest.json`。
- **避坑指南**：仅评估“摸头、摸膝、摸腰”3个规范动作，第 21 标后的握拳动作因协议边界不全仅做审计不作事件估算。

### 5.2 EMG 附带 IMU 运动学代理分析: `scripts/analyze_emg_imu.py`
- **功能定位**：解析 Delsys Avanti CSV 中的加速度计、陀螺仪与磁力计列，提取 1 秒重力姿态代理（Roll/Pitch）与运动强度 RMS。
- **运行命令**：
  ```powershell
  python scripts/analyze_emg_imu.py --data-root data_repro_20260903 --output-root outputs/imu_posture_repro_20260903
  ```
- **核心输出**：`imu_stream_summary.csv`, `imu_kinematic_proxy_1s.csv`, `imu_field_inventory.json`。
- **避坑指南**：未进行磁校准与空间解算，严禁将结果解读为“绝对关节角度”或“航向角（Yaw）”。

### 5.3 动作分段脑电质量审计: `scripts/segment_eeg_quality.py`
- **功能定位**：根据自动事件标划分 `pre_action` (5s), `execution` (5s), `post_action_rest` (15s)，评估各时段脑电伪迹分布。
- **运行命令**：
  ```powershell
  python scripts/segment_eeg_quality.py --output outputs/segment_eeg_quality_paired7_20260903
  ```

### 5.4 质量门控跨模态图探索: `scripts/phase19_quality_gated_multimodal_graph.py`
- **功能定位**：构建 n=7 的认知-运动假说响应图。使用留一受试者（LOO）稳健 z 分数与 Spearman 秩相关。
- **运行命令**：
  ```powershell
  python scripts/phase19_quality_gated_multimodal_graph.py
  ```
- **核心输出**：`outputs/analysis/paired7_20260903/phase19/`（网络图 PNG/SVG、边统计表）。

---

## Stage 6: 报告与可视化排布

### 6.1 EEG 64 导电极拓扑与脑区分区绘制
- **标准版本**: [`plot_eeg_montage.py`](file:///d:/Python_File/EEF_analyze/plot_eeg_montage.py)
- **高分印刷版**: [`plot_eeg_montage_hires.py`](file:///d:/Python_File/EEF_analyze/plot_eeg_montage_hires.py)
- **运行命令**：
  ```powershell
  python plot_eeg_montage.py
  python plot_eeg_montage_hires.py
  ```
- **核心输出**：`outputs/eeg_64ch_montage_layout.png`, `.pdf`, `.svg`。

### 6.2 最佳受试者预处理对比图表: `build_best_subject_preprocessing_comparison_20260914.py`
- **功能定位**：以最高分受试者（`2026081502/sec_test`）为展示范式，生成时域、频域（Theta/Alpha/Beta）、fNIRS、EMG、IMU 前后对比图。
- **运行命令**：
  ```powershell
  python build_best_subject_preprocessing_comparison_20260914.py
  ```
- **核心输出**：`best_subject_preprocessing_comparison_20260914/`（01~06 组图及 `run_manifest.json`）。

### 6.3 预处理步骤与图文指南生成: `build_preprocess_steps_and_figure_guide_20260914.py`
- **功能定位**：汇集对比图与算法表格，生成 Word 文档指南。
- **运行命令**：
  ```powershell
  python build_preprocess_steps_and_figure_guide_20260914.py
  ```
- **核心输出**：`preprocess_steps_and_figure_guide_20260914.docx`。

### 6.4 实验方案修订版整合与排版: `build_document_and_pdf.py`
- **功能定位**：排版中医药现代化课题四实验方案，内嵌电极布局图与参数表，并借助 Edge/Chrome 生成无头打印 PDF。
- **运行命令**：
  ```powershell
  python build_document_and_pdf.py
  ```
- **核心输出**：`outputs/中医药现代化课题四实验方案_修订版.docx` 与 `.pdf`。
