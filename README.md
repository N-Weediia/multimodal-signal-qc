# EEG、sEMG 与 fNIRS 多模态脑机与运动功能评估系统

> [!CAUTION]
> **【安全红线：`J:\2026二康` 严格只读】**
> - 外部数据源 `J:\2026二康`、其所有子目录及任何解析到该路径的目标，**严格仅允许读取**。
> - 严禁在 J: 盘创建、修改、覆盖、删除任何文件或写入日志/缓存；所有输出必须保存在 `D:\Python_File\EEF_analyze`。
> - 运行任何分析程序前，必须确保当前工作目录（cwd）位于 `D:\Python_File\EEF_analyze`。详见 [05_SAFETY_AND_GOVERNANCE.md](file:///d:/Python_File/EEF_analyze/docs/05_SAFETY_AND_GOVERNANCE.md)。

---

## 1. 项目定位与核心能力

本项目针对中医药现代化及脑机康复评估需求，构建了规范化的多模态生物医学信号处理与表型分析系统，覆盖 **博睿康 Neuracle 64导 EEG**、**Delsys Avanti 7通道 sEMG (含 IMU)** 和 **丹阳慧创 NirSmart 双波长 fNIRS**。

### 核心系统能力
- **独立离线质量控制 (QC)**：基于物理与生理特性的独立筛查（EEG 59/31 导稳健中位数重参考、EMG 50Hz 工频谐波比、fNIRS 心搏 SCI 与 CV 门控），生成逐记录诊断报告与整体资产清单。
- **规范预处理与审计掩码**：绝不篡改原始数据，保守去噪并导出清洗后的标准 NPZ（`eeg_cleaned.npz`, `emg_cleaned.npz`, `fnirs_cleaned_od.npz`），全流程绑定通道掩码与时段有效性掩码。
- **跨模态时间对齐契约**：严格规范 Marker 1 的同步与边界含义，自动对齐运动首标与 EMG 第 0 帧，支持眼动五标缺失边界反推与静息态整段无标分析。
- **专项分析与理论假说图**：包含相对 MBLL（光密度自然对数）敏感性分析、IMU 1 秒运动学姿态代理提取，以及基于配对 7 人的跨模态响应网络图（LOO 稳健验证）。

---

## 2. 文档中心快速导航

项目已建立系统化文档管理体系，详细技术规范与操作指引请查阅 [`docs/`](file:///d:/Python_File/EEF_analyze/docs/README.md) 专题文档：

| 专题文档 | 核心内容 | 重点指引 |
| :--- | :--- | :--- |
| 📖 [**文档中心首页**](file:///d:/Python_File/EEF_analyze/docs/README.md) | 文档导航总览、全流程系统架构图、文档更新与维护规范 | 快速总览系统全景 |
| 🚀 [**01 执行入口清单**](file:///d:/Python_File/EEF_analyze/docs/01_ENTRYPOINTS.md) | Stage 0 到 Stage 6 核心业务执行入口、命令参数、输入输出及避坑指南 | 运行实验与批处理前必读 |
| 🔬 [**02 多模态处理规范**](file:///d:/Python_File/EEF_analyze/docs/02_MULTIMODAL_PIPELINE.md) | EEG 64/32导配置、sEMG 逻辑肌肉映射、fNIRS 双波长 OD 与红外眼动干扰契约 | 信号清洗与门控算法标准 |
| 📂 [**03 目录与资产全景**](file:///d:/Python_File/EEF_analyze/docs/03_DIRECTORY_LAYOUT.md) | 根目录结构、`scripts/` 分类表、`tests/` 测试集、`outputs/` 层次及历史阶段产物 | 代码资产与输出结构索引 |
| ⏱️ [**04 事件标记与对齐**](file:///d:/Python_File/EEF_analyze/docs/04_EVENT_MARKERS.md) | Marker 1 定义、运动/眼动/静息任务标记结构、五标眼动反推算法与审计字段 | 跨模态时间同步核心标准 |
| 🛡️ [**05 安全红线与治理**](file:///d:/Python_File/EEF_analyze/docs/05_SAFETY_AND_GOVERNANCE.md) | J: 只读红线、cwd 相对路径陷阱、`--overwrite` 递归删除风险及幂等复现准则 | 数据安全与运行前自查 |

---

## 3. 核心常用命令速查表

在终端（PowerShell）中执行命令前，请确保当前工作目录为项目根目录 `D:\Python_File\EEF_analyze`。

### 3.1 环境初始化与数据质控 (QC)
```powershell
# 1. 安装基础依赖
python -m pip install -r requirements.txt

# 2. 运行多模态独立质控 (main.py)
# 对本地副本全量运行并生成诊断图
python main.py --data-root data_repro_20260903 --results-root outputs/qc/local_repro_20260903 --overwrite

# 快速检查（仅前 60 秒，用于测试新数据通道状态）
python main.py --max-seconds 60 --overwrite

# 仅评估单个受试者
python main.py --subject "2026081401" --overwrite
```

### 3.2 规范化去噪预处理 (Preprocess)
```powershell
# 运行独立预处理管线 (生成清洗后的 NPZ 与质量掩码，不修改原始数据)
python preprocess.py --data-root data_repro_20260903 --output-root outputs/preprocessed/current --overwrite
```

### 3.3 专项分析与敏感性验证
```powershell
# 运行配对 7 人的相对 MBLL 光密度敏感性分析 (自然对数 OD、滤波网格与 DPF 比较)
python mbll_relative_sensitivity_reviewed_20260908.py --output outputs/mbll_reviewed_20260908

# 提取 sEMG 伴随的 Avanti IMU 1秒姿态倾角与运动强度代理
python scripts/analyze_emg_imu.py --data-root data_repro_20260903 --output-root outputs/imu_posture_repro_20260903

# 运行跨模态响应网络图探索 (Phase 19, 基于受试者级差值与 LOO 稳健检验)
python scripts/phase19_quality_gated_multimodal_graph.py
```

### 3.4 报告与电极排布图绘制
```powershell
# 绘制国际 10-10 系统 64 导 EEG 电极排布与 ROI 功能脑区图
python plot_eeg_montage.py

# 绘制 300 DPI 印刷级高分电极排布图
python plot_eeg_montage_hires.py

# 生成最佳受试者 (2026081502/sec_test) 预处理前后效果全套对比图表
python build_best_subject_preprocessing_comparison_20260914.py
```

---

## 4. 顶级目录结构图解

```text
D:\Python_File\EEF_analyze
├── docs/                       # 系统化技术规范与执行手册 (01~05 专题)
├── scripts/                    # 阶段性分析脚本 (Phase 1~19)、数据导入与报告工具
├── tests/                      # pytest 自动化单元测试与冒烟测试集
├── outputs/                    # 统一输出目录 (包含 analysis, qc, preprocessed, release, _scratch)
├── data/                       # 本地原始数据工作目录 (由 .gitignore 忽略)
├── data_repro_20260903/        # 配对 7 人干净原始数据本地复现副本 (196 文件 + 核对表)
├── main.py                     # 多模态独立采集质控 (QC) 核心主程序
├── preprocess.py               # 规范去噪预处理与掩码提取核心主程序
├── low_snr_recovery.py         # 低信噪比实验性恢复探索层
├── expand_all_cases.py         # 多任务动作 Block 特征展开与提取
└── requirements.txt            # 项目 Python 依赖声明
```

---

## 5. 工作队列与状态口径说明

1. **纵向队列范围**：
   当前配对纵向分析队列严格限定为确认的 7 名受试者：`2026080901`, `2026081401`, `2026081403`, `2026081501`, `2026081502`, `2026081503`, `2026081701`。其余仅有基线访视的受试者保留在基线样本库中，严禁冒充前后配对样本。
2. **预处理版本口径**：
   - 2026-09-14 产物覆盖 107 条记录：Review 态为 v5 流程；Optimized 态为 v6 流程（`2026-09-14.v6-quality-tiered-recovery`）。
   - 两版本唯一状态差异是 `2026081401\base_line\EEG\运动\data.bdf` 在 Optimized 态由 `FAILED` 修复为 `READY_WITH_MASKS`。两组数字严禁混编。
3. **解释边界红线**：
   所有输出图表与指标均为技术处理和数据表征，`READY` 状态代表已生成合规数组与掩码，**绝不代表临床有效性**，不能替代预先设定的统计排除标准或波形人工目检。
