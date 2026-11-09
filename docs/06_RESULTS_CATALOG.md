# 项目结果全景归纳与资产使用指南 (Results Catalog)

本文档系统归纳多模态脑机与运动功能评估系统（EEF_analyze）所产生的所有数据产物、特征矩阵、诊断图表及文档报告，明确标注其**模态类型**、**处理阶段**、**版本标签**、**适用队列范围**及**解释边界**，彻底杜绝数据混用与错误归因。

---

## 1. 输出目录全景分类架构 (`outputs/`)

为防止不同模态、不同阶段的产物混杂，所有生成结果遵循严格的分类归档体系：

```text
outputs/
├── qc/                              # 【Stage 1】多模态独立离线质量控制产物
│   ├── current/                     # 当前全样本评估默认输出 (recording_inventory.csv + 逐记录诊断图)
│   ├── local_repro_20260903/        # 配对 7 人本地原始数据复现质控评估产物
│   └── 20260903_summary/            # 历史汇总快照与质控状态分布
│
├── preprocessed/                    # 【Stage 2】规范化去噪预处理与掩码产物 (NPZ + index.json)
│   ├── current/                     # 当前最新预处理产物 (v8: Hann WOLA + Fishburn TDDR + PASS)
│   ├── stage_20260904_eye_fnirs_imu_verified/ # 2026-09-04 阶段验证快照
│   ├── v5_review_20260914/          # 2026-09-14 Review 态历史快照 (107记录，3 FAILED)
│   └── v6_optimized_20260914/       # 2026-09-14 Optimized 态历史快照 (107记录，2 FAILED)
│
├── analysis/                        # 【Stage 4 & 5】特征提取与专项假说分析
│   ├── 20260903_all_cases_canonical/ # 18 名受试者基线标准化动作 Block 特征集
│   ├── paired7_20260903/            # 配对 7 人 Phase 1~19 各阶段分析产物 (含 Phase 19 跨模态网络)
│   ├── mbll_reviewed_20260908/      # 配对 7 人相对 MBLL 敏感性分析产物 (消光系数/DPF/滤波网格)
│   ├── imu_posture_20260904/        # 表面肌电 Avanti IMU 1秒姿态角与运动强度代理
│   └── segment_eeg_quality_paired7_20260903/ # 运动任务分段脑电信号质量审计产物
│
├── reports/                         # 【Stage 6】成果汇报、图版指南与实验方案排布
│   ├── protocols_and_montage/       # 实验方案 Word/PDF 终版修订稿与 64 导 EEG 拓扑排布图
│   ├── briefings/                   # 导师汇报简报、阶段汇报 Markdown 与汇总表
│   ├── comparisons/                 # 最佳受试者 (2026081502/sec_test) 预处理前后对比全套图解与文档
│   └── teacher_reports/             # 2026-09-04 与 2026-09-08 导师汇报图文成果包
│
├── audit_ledgers/                   # 【Stage 0】原始数据比对、安全同步与哈希审计账本
│   ├── j_old_data_scan_20260902/    # J: 盘与本地目录哈希扫描报告与重复文件清单
│   ├── j_drive_data_inventory_20260902.csv # J: 盘只读扫描资产全量总表
│   ├── local_data_inventory_20260902.csv   # 本地数据资产全量总表
│   └── cleanup_20261025_manifest.*  # 2026-10-25 数据治理清理明细账本
│
├── release/                         # 正式发布打包与哈希校验封存
│   ├── release_20260903/            # 2026-09-03 全样本发布归档包
│   └── release_20260903_paired7/    # 配对 7 人发布包及 SHA-256 完整性清单
│
├── literature/                      # 外部方法学文献对照、迁移矩阵与搜索记录
│   ├── literature_20260903/         # 初版文献检索与方法学对照矩阵
│   ├── literature_20260904/         # 进阶多模态脑肌耦合文献笔记
│   └── literature_method_transfer_20260904_local/ # 本地方法迁移验证输出
│
└── _scratch/                        # 实验性算法沙盒与临时验证产物 (严禁用于最终汇报)
    └── recovery_v8_validation/      # v8 升级算法 (TDDR/WOLA/PASS) 诊断对照表与图表
```

---

## 2. 各处理阶段产物详细归纳表

### 2.1 Stage 0: 数据审计与受控导入账本
| 资产目录 / 文件 | 模态范围 | 适用队列 | 生成脚本 | 核心作用与关键字段 |
| :--- | :--- | :--- | :--- | :--- |
| `outputs/audit_ledgers/j_old_data_scan_20260902/` | 全模态 | J: 盘原始数据 | `scripts/scan_old_data.py` | 识别 J: 盘内重复记录组，给出安全同步建议 |
| `outputs/audit_ledgers/j_drive_data_inventory_20260902.csv` | 全模态 | J: 盘全量 | 同上 | 记录大小、修改时间与 SHA-256 哈希 |
| `outputs/audit_ledgers/cleanup_20261025_manifest.json` | 全模态 | 本地工程 | `scripts/phase1_cleanup.py` | 记录清理、移动与规整操作的不可篡改日志 |

### 2.2 Stage 1: 多模态独立质量控制 (QC)
| 资产目录 / 文件 | 模态范围 | 适用队列 | 管线版本 | 核心作用与使用指引 |
| :--- | :--- | :--- | :--- | :--- |
| `outputs/qc/current/recording_inventory.csv` | EEG, EMG, fNIRS | 全样本 18 人 | `2026-10-25.v7-full-cohort-qc` | **全记录质控总表**。包含各模态状态（`READY`, `CAUTION`, `FAILED`）与详细得分 |
| `outputs/qc/current/analysis_index.json` | 全模态 | 全样本 18 人 | 同上 | 质控统计摘要、总通过率与异常记录列表 |
| `outputs/qc/current/<subj>/<sess>/<task>/` | 单项模态 | 单次采集 | 同上 | `quality_report.md`、`summary.json`、`signal_preview.png` 时域图与频谱图 |
| `outputs/qc/local_repro_20260903/` | EEG, EMG, fNIRS | 配对 7 人 | 同上 | 针对干净本地副本运行的独立质控快照 |

> [!IMPORTANT]
> **QC 产物的解释边界**：
> - QC 评估仅用于离线硬件连接与生理特征粗筛（如 50Hz 工频干扰比、心搏脉动 SCI）。
> - 状态为 `READY` 或 `SUCCESS` **仅表示数据完整可供分析，绝不代表临床有效或无需人工复核**。

### 2.3 Stage 2: 规范预处理与清洗导出 (Cleaned NPZ)
| 资产目录 / 文件 | 模态范围 | 适用队列 | 管线版本 | 核心作用与包含数组 |
| :--- | :--- | :--- | :--- | :--- |
| `outputs/preprocessed/current/` | EEG, EMG, fNIRS | 全样本 18 人 | **`2026-11-09.v8-quality-tiered-recovery`** | **当前最新预处理成果库**。包含每个受试者/任务的 `*_cleaned.npz` 与质量掩码 |
| `outputs/preprocessed/current/preprocessing_index.json` | 全模态 | 全样本 18 人 | 同上 | 预处理记录清单，包含数据质量状态、掩码统计与时间同步参数 |
| `outputs/preprocessed/v6_optimized_20260914/` | EEG, EMG, fNIRS | 配对 7 人 (107条) | `2026-09-14.v6-quality-tiered-recovery` | 历史基准：42 掩码就绪、33 通道滤波、14 连续就绪、16 红外干扰、**2 FAILED** |
| `outputs/preprocessed/v5_review_20260914/`<br>(`review_20260914/`) | EEG, EMG, fNIRS | 配对 7 人 (107条) | `2026-09-14.v5-initial-review` | 历史复核版：41 掩码就绪、33 通道滤波、14 连续就绪、16 红外干扰、**3 FAILED** |

> [!CAUTION]
> **预处理产物版本绝对隔离红线**：
> 1. `review_20260914` (v5) 与 `optimized_20260914` (v6) 的状态数仅有一条差异（`2026081401\base_line\EEG\运动\data.bdf` 在 v5 中失败，在 v6 中恢复），**严禁将两组数字拼合作为总数**。
> 2. `outputs/preprocessed/current/` (v8) 引入了 TDDR 和 Hann WOLA，导出的 NPZ 数据分布与 v5/v6 有差异，新分析必须完全基于同一版本读取。

### 2.4 Stage 5: 模态专项分析与建模产物
| 资产目录 / 文件 | 模态范围 | 适用队列 | 理论假说与方法学 | 核心输出与解释限制 |
| :--- | :--- | :--- | :--- | :--- |
| `outputs/mbll_reviewed_20260908/` | fNIRS (双波长) | 配对 7 人 (14条运动) | 相对 MBLL 光密度自然对数敏感性 | 包含消光系数敏感性表、DPF 网格分析。**仅为相对估计，缺乏短间距物理回归与源定位，未连接 EOG/ECG** |
| `outputs/imu_posture_20260904/` | EMG (Avanti IMU) | 配对 7 人 (14条运动) | 1 秒运动学姿态倾角代理与重力范数 | 提取前后倾角与加速度强度，作为动作执行粗代理，**非完整光学运动捕捉** |
| `outputs/analysis/paired7_20260903/phase19/` | EEG + fNIRS + EMG | 配对 7 人 | 留一受试者 (LOO) 稳健性跨模态假说网络 | 基于 baseline→sec_test 受试者级差值构建的网络图，**为探索性图模型，样本量小，不可用于临床分类推断** |

### 2.5 Stage 6: 汇报图文、电极排布与实验方案文档
| 资产目录 / 文件 | 成果类型 | 核心内容 | 关联脚本 |
| :--- | :--- | :--- | :--- |
| `outputs/reports/protocols_and_montage/eeg_64ch_montage_layout.*` | 高清矢量/位图图版 | Neuracle 64 导国际 10-10 系统标准电极拓扑与 ROI 分区图 (含 300 DPI 印刷版) | `plot_eeg_montage.py`<br>`plot_eeg_montage_hires.py` |
| `outputs/reports/protocols_and_montage/中医药现代化课题四实验方案_修订版.docx` | 课题方案规范 | Word 修订版实验技术规范与多模态操作流程 | `build_document_and_pdf.py` |
| `outputs/reports/comparisons/` (及 `best_subject_preprocessing_comparison_20260914/`) | 预处理对比套图 | 代表性受试者 (`2026081502/sec_test`) 预处理前后 6 组全景时频域对比图与指标表 | `build_best_subject_preprocessing_comparison_20260914.py` |
| `outputs/reports/teacher_reports/` | 导师汇报图文包 | 2026-09-04 与 2026-09-08 阶段汇报图文资料与成果胶片 | `build_teacher_report_20260908.py` |
| `outputs/reports/briefings/` | 质量简报矩阵 | 包含全队列各模态质量合格率矩阵、红外眼动干扰说明及阶段报告 | `scripts/build_cohort_quality_briefing_20261025.py` |

---

## 3. 四大核心安全与解释红线

1. **红线一：`J:\2026二康` 严格只读**
   - 外部数据源绝对禁止写出任何日志、缓存或结果文件；所有输出必须严格落在 `D:\Python_File\EEF_analyze\outputs\`。
2. **红线二：工作目录 (cwd) 必须位于项目根目录**
   - 相对输出路径（如 `outputs/qc/current`）在不同 cwd 启动会导致产物写错位置，运行前务必确认 cwd 为 `D:\Python_File\EEF_analyze`。
3. **红线三：绝不向任何原始数据中插入伪造恢复**
   - 严重损坏无法物理还原的通道或片段，必须打上 `FAILED` 状态或置入有效性掩码（Mask），严禁通过线性插值伪造有效信号参与统计。
4. **红线四：严格恪守科学结论的解释边界**
   - 单受试者展示（如 2026081502）仅用于预处理效果说明，**不可作群体疗效推断**；
   - 相对 MBLL 仅为相对变动代理，**不能等同于绝对血氧饱和度**；
   - 跨模态网络相关图（Phase 19）为假说探索，**不可声明临床诊断结论**。
