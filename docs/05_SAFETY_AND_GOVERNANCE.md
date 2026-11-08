# 安全红线与数据治理规范

本规范规定多模态脑机与运动功能评估系统（EEF_analyze）在开发、运行、数据处理与智能体协作过程中的**最高安全红线**与**数据治理准则**。所有在此代码库工作的开发人员与智能体，必须严格无条件遵守。

---

## 1. 外部数据源绝对只读铁律 (`J:\2026二康`)

### 1.1 铁律条款
> **【安全红线：`J:\2026二康` 严格只读】**
> - `J:\2026二康`、其所有子目录，以及解析到该路径的任何别名、符号链接或其他路径形式，**均严格仅允许读取**。
> - **严禁**在上述位置创建、覆盖、追加、删除、重命名或移动任何文件或目录；
> - **严禁**向该路径写入缓存（如 `.cache`、`__pycache__`）、日志、临时文件、锁文件（如 Word/Excel 临时锁）；
> - **严禁**修改该路径的权限、安全描述符（ACL）或任何元数据属性；
> - 所有项目生成物、分析报告、缓存和中间调试内容**必须**且**只能**写入 `D:\Python_File\EEF_analyze` 或获准写入的目录。

### 1.2 设立背景与临床责任
- `J:\2026二康` 存储的是临床多中心试验采集的医院原版第一手数据（原始 BDF、NIRS、CSV 与受试者信息核对表），属于不可再生的原始生物医学研究证据。
- 移动存储设备的意外写入、断电中断或误操作极易引发文件系统扇区损坏或静默数据损坏（Silent Data Corruption），导致法医学与学术审查维度的不可逆事故。
- 访问原始数据必须采取受控只读模式，生产计算应优先使用本地干净副本（如 `data_repro_20260903/`）。

---

## 2. 当前工作目录 (cwd) 与相对路径陷阱

### 2.1 陷阱成因剖析
在 Python 及跨平台脚本中，未解析的相对路径（如 `Path("data")` 或 `Path("outputs/qc/current")`）完全依赖启动时操作系统的**当前工作目录 (Current Working Directory, cwd)** 进行绝对化定位：

```python
# 风险代码范例：
DEFAULT_RESULTS_ROOT = Path("outputs") / "qc" / "current"
...
destination = output_path_for(recording, args.data_root, args.results_root)
```

- 若在 `D:\Python_File\EEF_analyze` 下启动，`outputs/qc/current` 被安全解析为 `D:\Python_File\EEF_analyze\outputs\qc\current`；
- **致命陷阱**：若终端、IDE 或智能体将当前工作目录切换至 `J:\2026二康\数据` 并执行脚本，上述相对路径将直接被解析为 `J:\2026二康\数据\outputs\qc\current`，导致脚本**强行在只读盘创建目录并写出数据**，直接击穿安全红线！

### 2.2 操作规范与防御检查
1. **工作目录锁定**：运行任何命令、测试或启动 Python 前，必须确保当前工作目录位于 `D:\Python_File\EEF_analyze`。严禁执行任何 `cd J:\...` 指令。
2. **运行前解析断言**：在编写或调用带有写出操作的脚本时，必须检查并断言输出路径：
   ```python
   destination = destination.resolve()
   assert not str(destination).startswith("J:"), "安全阻断：禁止向 J: 盘写出数据！"
   ```

---

## 3. `--overwrite` 参数的递归删除风险与防线

### 3.1 强行递归删除风险 (`shutil.rmtree`)
在 [`main.py`](file:///d:/Python_File/EEF_analyze/main.py) 与 [`preprocess.py`](file:///d:/Python_File/EEF_analyze/preprocess.py) 中，`--overwrite` 标志通过调用 Python 的 `shutil.rmtree(destination)` 实现目标目录覆盖：

```python
if destination.exists() and args.overwrite:
    shutil.rmtree(destination)
```

`shutil.rmtree` 执行的是操作系统的**无提示物理递归删除**（不经过操作系统回收站）。
若以下情况发生，将造成不可挽回的灾难性破坏：
- 传入了错误的 `--results-root` 或 `--output-root`（例如误传为根目录、源数据目录或父目录）；
- 路径拼接逻辑出现异常，导致 `destination` 解析为空或根驱动器。

### 3.2 执行 `--overwrite` 前置核查标准
在终端输入或通过自动化脚本传递 `--overwrite` 之前，必须执行以下三层核查：
1. **边界确认**：确认目标目录确实位于 `D:\Python_File\EEF_analyze\outputs\` 下，且绝非 `data/`、`data_repro_*/` 或 `J:\`；
2. **级别隔离**：尽量指定单受试者（如 `--subject 2026080901`）进行逐人局部刷新，避免无差别的全库递归重建；
3. **备份已验证成果**：关键发布包（如 `outputs/release_*/`）或已发表阶段成果必须独立归档，不得将其指定为当前 `--output-root`。

---

## 4. 幂等性、可复现性与数据溯源机制

为了保证科学研究的严格可复验性与计算经济性，系统内置了完善的数据治理框架：

### 4.1 幂等性设计 (Idempotency)
- **缓存有效性检验**：
  [`preprocess.py`](file:///d:/Python_File/EEF_analyze/preprocess.py) 在处理前会同步比对三项要素：
  1. 原始文件当前 SHA-256 哈希值；
  2. 预处理管线版本号（`PREPROCESS_PIPELINE_VERSION`，如 `2026-09-14.v6-quality-tiered-recovery`）；
  3. 预处理关键参数综合哈希（`preprocess_parameter_hash`）。
- **无副作用复用**：若三者完全一致且已有输出完整，程序自动判定为最新并跳过重复计算，确保多次运行结果严格恒等。

### 4.2 科学可复现性 (Reproducibility)
- **参数显式固化**：所有滤波截止频率、重采样率、窗长、特征提取阶数一律以常数形式显式定义在脚本头部，严禁使用未声明的魔法数字（Magic Numbers）。
- **运行清单 (Run Manifest)**：阶段性重点分析（如 Stage 5 相对 MBLL、Stage 6 最佳受试者展示）必须自动输出 `run_manifest.json`，其中详细记录：
  - 本地执行时间戳与时区；
  - 启动命令的完整命令行参数（`argv`）；
  - Python、NumPy、SciPy、Matplotlib 等关键环境版本；
  - 执行主脚本本身的 SHA-256 校验和。

### 4.3 证据分级与不造伪原则
- **缺失与异常明确留痕**：
  遇到短记录（<2s）、文件损坏或脱落导联，程序输出 `FAILED` 或 `UNASSESSABLE`，**绝不通过插值或伪数据捏造“正常”信号**。
- **掩码隔离**：
  坏导联插值仅用于维持空间矩阵维度，在计算能量、特征或统计差异时，**必须带上 `bad_channel_mask` 进行门控或剔除**。
- **推断状态标定**：
  任何算法反推的标记或估计边界（如眼动五标反推），必须在字段中明确标注为 `inferred`，严禁冒充硬件采集的原始数据。
