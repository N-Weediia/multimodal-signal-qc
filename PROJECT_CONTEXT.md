# 项目接管快照（2026-10-25）

> 长期约束以 [AGENTS.md](AGENTS.md:3) 为准：`J:\2026二康`、其子目录及解析到该处的链接/别名目标严格只读；项目生成物只写入 `D:\Python_File\EEF_analyze` 或用户另行明确授权的位置。执行任何写入入口前，先核对当前工作目录（cwd）及解析后的输出目标。本文件记录现状，不是新研究计划。

## 工作入口与数据流

- QC 和 preprocess 是两个独立入口：`main.py` 评估采集质量并生成报告/inventory（README.md:22-25、main.py:815-822）；`preprocess.py` 另行生成预处理数组、mask 和 preprocessing inventory/index（README.md:70-82、preprocess.py:701-718）。QC 结果不会自动变成预处理数组；`READY` 不代表临床有效，也不能替代人工波形复核（README.md:101）。
- expanded 特征链直接从 `RAW_ROOT` 查找并读取原始 BDF/NIRS/CSV（expand_all_cases.py:31-35、92-117），不自动消费 preprocess 的数组。它默认读取历史 `outputs/qc/20260830/recording_inventory.csv` 作为可选 QC 辅助表（expand_all_cases.py:34、121-130、313）；CLI 有 `--qc-inventory` 参数（354-363），复跑应显式指定对应 inventory。
- `data_repro_20260903` 是 README 记录的本地原始副本（README.md:15）；配对分析配置优先用该副本，不存在时才回退到 J: 输入根（scripts/paired7_config.py:8-14）。

## 9/14 预处理结果与图表版本

- 数字来自实际读取并汇总的两份索引，不以报告生成脚本的硬编码文字为数据源：
  - Review：`review_20260914/preprocessing_index.json`（第2-3行起）和 `preprocessing_inventory.csv`。共107条：41 `READY_WITH_MASKS`、33 `READY_CHANNEL_FILTERED`、14 `READY_CONTINUOUS`、16 `READY_WITH_MASKS_KNOWN_IR_INTERFERENCE`、3 `FAILED`。复核摘要也记录此组数字（review_20260914_summary.md:2）。
  - Optimized：`optimized_20260914/preprocessing_index.json`（第2-3行起）和 `preprocessing_inventory.csv`。共107条：42、33、14、16、2；流程版本 `2026-09-14.v6-quality-tiered-recovery`（index第13行、inventory第2行）。
  - 唯一状态变化是 `2026081401\base_line\EEG\运动\data.bdf`：review `FAILED`，optimized `READY_WITH_MASKS`（两份 inventory 第10行）。Review 对应 v5（review inventory第10行）；切勿把两组状态数拼在一起。
- 9/14 图构建器绑定 review：`build_best_subject_preprocessing_comparison_20260914.py:6-10` 将原始输入设为 `data_repro_20260903`、`STAGE` 设为 `review_20260914`、输出设为 `best_subject_preprocessing_comparison_20260914`。图和指标在该输出目录：`README.md:5-18,30-41`、`figure_metrics.csv`、`subject_selection.csv`、`run_manifest.json:1-15`。代表访视为 `2026081502/sec_test`，分数98.563；这是处理展示，不是疗效或群体推断。
- Review 摘要第6行记录的事项是按受试者×访视×任务去重、重复导出先人工确认、失败 EEG 不造伪恢复、统计继续带 mask。它是既有复核意见，不等同于新的用户待办；optimized 已改变一条 EEG 状态。
- 图文说明生成器设定输出 `preprocess_steps_and_figure_guide_20260914.docx`（build_preprocess_steps_and_figure_guide_20260914.py:10-12、120）；本轮未重跑。

## 已有分析与解释边界

- `outputs/mbll_reviewed_20260908/` 是配对7人的质量门控相对 MBLL 分支。`run_manifest.json:2-45` 是历史运行记录：分析完成、14条运动记录、15,600事件行、3,120汇总行、1,482配对行、108敏感性行；历史 smoke 通过，pytest 证据来自不同 Python 环境，不代表本轮运行或测试。
- MBLL 脚本固定读取本地 raw 副本和旧 stage（mbll_relative_sensitivity_reviewed_20260908.py:30-39）。结果是依赖消光系数、DPF、光程假设的相对估计；850 nm/年龄公式存在外推，物理探头位置未由元数据确认，无短程回归。只分析三个完成的固定运动动作，握拳边界未定故只审计不估算；EOG/ECG 未连接、无校正（MBLL脚本:35-65,873-885）。
- `outputs/analysis/paired7_20260903/phase19/` 是已存在的探索性跨模态图（README.md:14；phase19/review.md:3-17），单位为受试者级 baseline→sec_test 差值，n=7；不是训练分类器或临床预测。报告列出的限制包括 fNIRS OD proxy、无短程回归/源定位、EEG/fNIRS 未配准和小样本（phase19/review.md:5-15）。
- 事件规则见 `事件标记使用说明.md`：marker `1` 是同步/边界，不是任务编码（第7、13-21行）；运动首标是三模态同步并对应 EMG 第0帧（第27-40行）；眼动只对齐 EEG/fNIRS（第42-48、153-160行）；静息无标保留整段（第162-172行）。眼动五标缺失边界需明确保存为推断（第130-151、174-198行）。
- 图表解读限制见 `best_subject_preprocessing_comparison_20260914/figure_contract.md:3-9` 和该目录 README.md:15-19,25-28,39-41：个例不能作群体推断；相对 MBLL 有假设；EMG 缺可信动作标记；EOG/ECG 未连接；IMU 不是完整运动学。

## 写入和运行注意事项

- **cwd 会影响相对路径。** `main.py` 的默认 `data/`、`outputs/qc/current` 是相对 cwd 的 Path（main.py:68-70）；`preprocess.py` 也是相对路径默认值（703、706行）。因此只有在项目根 `D:\Python_File\EEF_analyze` 为 cwd 时，这些默认输出才落在 D:；从 J: 工作目录启动可能把相对输出写进 J:。low-SNR、phenotype、summarize、integrate 也有相对默认路径（low_snr_recovery.py:519-523；quality_gated_phenotype.py:219-225；summarize_expanded_results.py:114-120；integrate_tcm_reports.py:301-307）。执行前核对 cwd 与 `resolve()` 后的输出目标。
- 硬编码的 J: 引用指向数据源/输入/测试，不是硬编码写出目标：`expand_all_cases.py:31-35`、`explore_202608_cases.py:42-44`、`scripts/paired7_config.py:8-14`、`scripts/import_raw_subset.py:20-22`、`scripts/scan_old_data.py:155-160`、`tests/test_multimodal_qc.py:178-179`；`scripts/phase1_audit.py:245` 只是说明把历史 J: 路径作为文本证据，不访问该路径。`AGENTS.md:3-9` 规定只读边界。显式输出参数普遍没有阻止 J: 写入的代码级保护：import 的 `--dest-root`/`--output-dir`（scripts/import_raw_subset.py:91-125,129-173），QC `--results-root`（main.py:820,891-902），preprocess `--output-root`（preprocess.py:706-713），expanded `--out`（expand_all_cases.py:352-363），MBLL `--output`（MBLL脚本:960-968）。
- `main.py --overwrite` 和 `preprocess.py --overwrite` 会递归删除已有结果目标（main.py:822,848-851；preprocess.py:707,727-736）；不能在未核对解析路径时运行。
- `make_report.py` 在模块顶层读取、改写并替换 cwd 下 `preprocess_review_20260914.docx`（make_report.py:3,22-27），导入即有写入副作用；cwd 若在 J:，风险同样适用。`scripts/phase7_block_dynamics.py` 导入时会创建输出目录（第24-27行）。
- 未发现用户明确指定的新阶段目标。文档中的建议只是既有材料内容，不构成研究方向授权。历史 manifest 的成功记录不等于本轮测试通过；本轮没有运行分析、报告脚本或 pytest，也没有改动原始数据。
