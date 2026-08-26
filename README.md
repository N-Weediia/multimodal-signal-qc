# EEG、sEMG 与 fNIRS 多模态数据质量评估

本程序递归筛查 `data/` 中的三类文件：博睿康 `data.bdf` EEG、Delsys/Avanti CSV sEMG，以及 NirSmart/Homer `.nirs` fNIRS。三类信号采用独立读取、阈值和报告，不会把 EEG 的重参考或 uV 阈值套到 sEMG/fNIRS，也不会修改原始文件。

## 运行环境

```powershell
python -m pip install -r requirements.txt
python main.py --overwrite
```

默认评估 `data/` 下的全部受试者文件夹。当前目录可发现 4 名受试者；其中庞金明目录只发现 EEG 文件，程序会在批次清单中如实标注“未发现 EMG/fNIRS 支持文件”，不会把它推断成采集遗漏。

评估其他受试者时，传入受试者文件夹名称：

```powershell
python main.py --subject "其他受试者" --overwrite
```

如需只评估一名受试者，传入受试者文件夹名称：

```powershell
python main.py --subject "082101" --overwrite
```

如需快速检查每段记录的前一分钟：

```powershell
python main.py --max-seconds 60 --overwrite
```

## 生成可分析的去噪数据

质量评估完成后，运行独立预处理命令。它不会修改 `data/` 内的原始 BDF、CSV 或 NIRS 文件：

```powershell
python preprocess.py --output-root preprocessed_20260822_ready
```

预处理按模态保守处理，并把无法可靠恢复的内容保留为掩码或排除记录：

- EEG：仅使用固定 31 个头皮导联；先稳健参考以识别平线/持续高幅导联，再以可用导联平均参考，50 Hz 陷波、0.5--45 Hz 零相位带通并降采样至 250 Hz。坏导联可作局部空间插值，但会始终留在 `bad_channel_mask` 和 `interpolated_channel_mask` 中；统计分析不得把它们作为原始观测值。`valid_2s_epoch_mask=False` 的两秒窗口必须排除。
- sEMG：保留源幅值单位，20--450 Hz 零相位带通、50 Hz 及谐波陷波，并重采样到 1000 Hz。源 CSV 没有任务事件，输出仍是连续信号，任务分段必须使用外部同步日志。
- fNIRS：仅对无红外眼动仪干扰的记录输出双波长光密度，剔除 SCI < 0.50 或双波长 CV > 15% 的物理通道，修复突发导数尖峰并进行 0.01--0.2 Hz 滤波。未确认 DPF、探头距离和消光系数前，不自动转换 HbO/HbR。眼动任务的 fNIRS 写入 `EXCLUDED_KNOWN_IR_INTERFERENCE`，不会生成伪恢复数据。

输出目录包括：

- `preprocessing_index.json` 和 `preprocessing_inventory.csv`：每段数据的处理状态与来源。
- EEG `eeg_cleaned.npz`：`data_uv`、`labels`、采样率、坏/插值导联掩码及两秒有效窗口掩码。
- sEMG `emg_cleaned.npz`：连续去噪数组、传感器编号和时间轴。
- fNIRS `fnirs_cleaned_od.npz`：保留物理通道的双波长光密度、时间轴和采样率。

`READY` 只代表已生成可追溯的预处理数组和分析掩码，不等同于临床有效性，也不能替代预先规定的统计排除标准或人工波形复核。

也可以指定单个 BDF/CSV/NIRS/HCX 文件或目录。传入 `.hcx` 时，如存在同名 `.nirs`，程序会自动评估 `.nirs`：

```powershell
python main.py --input ".\data\傅开强\EMG" --overwrite
python main.py --input ".\data\傅开强\FNIRS\03傅开强\某段记录.nirs" --overwrite
```

## 输出结果

结果路径保留受试者、模态、源目录和源文件名。例如重复导出的 fNIRS 眼动记录分别写入独立子目录；同一 EEG 任务目录中的 `data.bdf` 和 `data.1.bdf` 也分别保存。这样不会以最后处理的文件覆盖先前结果。

每段记录均包含：

- `quality_report.md`：中文摘要和需要复核的通道。
- EEG `channel_metrics.csv`、sEMG `emg_channel_metrics.csv` 或 fNIRS `fnirs_channel_metrics.csv`。
- `summary.json`：记录级摘要和程序实际使用的阈值。
- 质量图、`signal_preview.png` 和 `power_spectra.png`。

本次运行的完整审计清单写入：

- `results/analysis_index.json`：全部成功、无法评分和跳过记录的机器可读索引。
- `results/recording_inventory.csv`：可直接打开检查的文件级目录、任务归类、评分状态和结果路径。
- `results/data_inventory.json` 与 `results/batch_report.md`：四名受试者的模态覆盖、任务覆盖和无法评分原因。

CSV 没有任务事件，因此 sEMG 只按连续会话评估，不会虚构静息态、眼动和运动任务边界。短于 2 秒的 EEG 片段没有足够窗口估计工频与伪迹，将保留为 `FAILED / UNASSESSABLE`，而不是被当作“质量差”或从总表中消失。

## 评分边界

### EEG

程序固定使用32通道缩减方案：`Fpz、Fp1、Fp2、Fz、F3、F4、F7、F8、FCz、FC1、FC2、FC5、FC6、Cz、C3、C4、T7、T8、CP1、CP2、CP5、CP6、Pz、P3、P4、P7、P8、POz、Oz、O1、O2` 共31个头皮导联，加上系统参考/地电极构成32通道。未打导电膏的其他头皮电极、`ECG`、`HEOR`、`HEOL`、`VEOU`、`VEOL` 等辅助导联均不参与评分或公共参考。

程序只用启用头皮导联构造逐采样中位数参考，检查残余50 Hz、异常高幅、平线、量程饱和和采集阻抗。中位数参考对少数噪声/脱连接导联稳健，不会像截尾均值那样被几个坏导联污染、再把工频扩散到全部导联。报告会分别列出未启用、缺失和辅助导联，未启用导联不会被误记为 `POOR`。

### sEMG

逐传感器检查时间戳丢样/倒退、整段平线、重复极值，以及 50 Hz 与谐波在 20-450 Hz 中的功率比例。低于 20 Hz 的功率只作提示，因为连续 CSV 中没有足够事件信息区分主动动作与运动伪迹。

根据现场校正，七块目标肌肉整体后移到实际采集单元序列 `2、3、4、5、9、10、11`：依次为斜方肌、三角肌前束、三角肌中束、肱二头肌、肱三头肌、肱桡肌和尺侧腕屈肌。CSV 表头未声明幅值单位，RMS 和稳健标准差保留源文件单位。

### fNIRS

每个物理通道配对 730/850 nm，检查缺失或非正强度、双波长 CV、0.5-2.5 Hz 心搏耦合 SCI、突发光密度阶跃、平线和重复极值。SCI < 0.80 或任一波长 CV > 7.5% 进入复核；SCI < 0.50 或 CV > 15% 给予更重扣分。

采集方法要求预采集后将红色/灰色节点调整为绿色或黄色。离线程序不能可靠还原厂商未公开的界面状态码，因此离线质量分不等同于 NirSmart 节点颜色。

眼动任务同步使用红外眼动仪，其发射光会干扰 fNIRS 原始光强。程序会把此类记录标记为已知红外干扰背景，并使会话状态至少为 `CAUTION`；通道异常不会被简单解释为头发遮挡、耦合不良或运动，也不会通过软件分数抵消这一真实污染。

## 使用限制

所有结果都是采集质量筛查，不能替代人工目视检查、预先规定的伪迹处理和生理/临床解释。阈值集中在各模块顶部，只应在有设备量程、幅值单位和研究方案依据时调整。
