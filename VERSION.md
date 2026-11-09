# 项目版本管理规范与演进历程 (Version Management)

本项目（EEF_analyze）采用三层版本管理规范：**代码与系统语义化版本 (System Semantic Version)**、**流水线数据处理版本标签 (Pipeline Provenance Tag)** 以及 **Git 提交与发布标签 (Git Milestones & Tags)**。

---

## 1. 版本体系与标识契约

| 维度 | 标识格式 | 当前版本 | 职责与作用范围 |
| :--- | :--- | :--- | :--- |
| **系统版本** | `vMAJOR.MINOR.PATCH` | `v8.0.0` | 代码库整体架构与核心算法版本 |
| **预处理管线版本** | `YYYY-MM-DD.v<N>-<feature-slug>` | `2026-11-09.v8-quality-tiered-recovery` | 写入各 NPZ 与 `preprocessing_index.json` 的不可篡改溯源标签 |
| **质控管线版本** | `YYYY-MM-DD.v<N>-<feature-slug>` | `2026-10-25.v7-full-cohort-qc` | 写入 `recording_inventory.csv` 与 `analysis_index.json` 的质控标准标签 |
| **Git 发布标签** | `v<MAJOR>.<MINOR>.<PATCH>` | `v8.0.0` | 代码仓库可溯源里程碑锚点 |

---

## 2. 预处理管线演进对比矩阵

系统自 2026 年 9 月至今历经四次核心处理管线演进。**严禁跨版本混拼状态分布或直接合并指标数组**：

| 版本标签 | 发布日期 | 队列范围 | 核心改进算法 | 记录状态分布 (n=107) | 产物目录位置 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`v5`**<br>`2026-09-14.v5-initial-review` | 2026-09-14 | 配对 7 人<br>(202608 批次) | 初始保守质控与初版清洗掩码 | READY_WITH_MASKS: 41<br>READY_CHANNEL_FILTERED: 33<br>READY_CONTINUOUS: 14<br>KNOWN_IR_INTERFERENCE: 16<br>**FAILED: 3** | `outputs/preprocessed/v5_review_20260914/`<br>`review_20260914/` |
| **`v6`**<br>`2026-09-14.v6-quality-tiered-recovery` | 2026-09-14 | 配对 7 人<br>(202608 批次) | 针对基线 EEG 离群段微调恢复策略 | READY_WITH_MASKS: 42<br>READY_CHANNEL_FILTERED: 33<br>READY_CONTINUOUS: 14<br>KNOWN_IR_INTERFERENCE: 16<br>**FAILED: 2** (成功恢复 1401 baseline EEG) | `outputs/preprocessed/v6_optimized_20260914/` |
| **`v7`**<br>`2026-10-25.v7-full-cohort-quality-provenance` | 2026-10-25 | 全样本 18 人<br>(全队列拓展) | 扩展至全病例，建立完整质量溯源矩阵与自动化简报 | 纳入 18 名受试者所有基线与随访记录，生成全队列质控诊断矩阵 | `outputs/preprocessed/v7_cohort_20261025/` |
| **`v8`**<br>`2026-11-09.v8-quality-tiered-recovery` | 2026-11-09 | 全样本 18 人 / 配对 7 人 | **多模态分级恢复重大升级**：<br>1. **EEG**: 引入去偏置中位数重参考与断连自适应门限，杜绝 50Hz 工频共模误切；协议感知子空间收缩 (PASS)。<br>2. **sEMG**: 引入 Hann WOLA 重叠相加谐波相消，50Hz 工频抑制 >90% 且保全肌电生理高频能量。<br>3. **fNIRS**: 引入 Fishburn TDDR (时间导数分布修复)，阶跃伪迹压制 >70%。 | 全队列保留率显著提升，极重度工频污染信号实现物理可审计恢复 | `outputs/preprocessed/current/` |

---

## 3. 代码模块化分层版本规范 (`src/`)

为避免模态和处理阶段代码交织混杂，自 `v8.0.0` 起全面推行**模态与阶段正交分离**：

```text
src/
├── modalities/                 # 【模态底层库：按物理生理信号类型划分】
│   ├── eeg/                    # 脑电：电极排布 (montage)、稳健重参考、断连判定、空间特征
│   ├── emg/                    # 肌电：逻辑肌肉映射、Hann WOLA 谐波消解、IMU 姿态与加速度
│   ├── fnirs/                  # 近红外：双波长 OD 转换、Fishburn TDDR、相对 MBLL 敏感性
│   └── multimodal/             # 跨模态：Marker 1 同步协议、CMC 脑肌耦合、跨模态网络
│
├── stages/                     # 【处理阶段流水线：按业务执行阶段划分】
│   ├── stage0_audit_sync/      # Stage 0: 原始数据只读审计、哈希校验与受控拉取
│   ├── stage1_qc/              # Stage 1: 多模态独立离线质量筛查 (main.py 引擎)
│   ├── stage2_preprocess/      # Stage 2: 规范预处理与清洗导出 (preprocess.py 引擎)
│   ├── stage3_recovery/        # Stage 3: 低信噪比与伪迹分级恢复 (TDDR, WOLA, PASS)
│   ├── stage4_features/        # Stage 4: 全量病例特征展开与表型整合
│   ├── stage5_analysis/        # Stage 5: 专项分析分支 (Phase 1~19, MBLL 敏感性)
│   └── stage6_reporting/       # Stage 6: 汇报文档排版与图表指南构建
│
└── utils/                      # 【系统通用支撑层】
    ├── safety.py               # J: 严格只读断言与输出路径边界核查
    ├── config.py               # 路径常量、会话清单与任务 Marker 协议
    └── visualization.py        # 中文字体、统一配色与高分辨率画布配置
```

---

## 4. 版本管理与使用铁律

> [!CAUTION]
> 1. **严禁混拼不同版本的统计数字**：9/14 的 Review (v5) 与 Optimized (v6) 有精确的状态计数差异（FAILED 从 3 降为 2），绝不能在同一张表或报告中把两者数字随意相加或偷换。
> 2. **预处理产物版本一致性**：读取 `outputs/preprocessed/` 下的 NPZ 时，必须核对 `preprocessing_index.json` 或 NPZ 元数据中的 `pipeline_version`，确保同一项分析消费的预处理数据版本完全一致。
> 3. **向下兼容保证**：根目录核心执行入口（`main.py`, `preprocess.py`, `emg_qc.py`, `fnirs_qc.py` 等）作为标准门面接口长期维持稳定，所有新旧脚本与自动化测试集必须保证 100% 通过。
