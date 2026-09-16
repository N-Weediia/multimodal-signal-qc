# Multimodal Signal Quality Control

面向 EEG、表面肌电（sEMG）和 fNIRS 的离线质量评估核心模块。项目强调“按模态独立评估、原始文件只读、每个结论可审计”，适合作为多模态生理信号处理和研究数据工程的公开 Demo。

## 能力证明

- sEMG：采样时间轴、平线、重复极值、RMS、稳健波动和 50 Hz 及谐波干扰评估。
- fNIRS：双波长配对、光强有效性、波长耦合指数（SCI）、变异系数、运动伪迹和质量分级。
- 显式保留质量掩码和失败原因，避免把不可用记录静默混入下游分析。
- 提供 `.nirs` 读取与合成信号测试，不需要真实受试者数据即可验证核心逻辑。

## 运行

```powershell
python -m pip install -r requirements.txt
python demo_qc.py
python -m pytest -q
```

真实项目中的原始数据入口、受试者级分析和报告发布脚本没有放入公开 Demo；本仓库只保留不依赖内部数据的质量控制核心。

## 简历描述

> 搭建 EEG/sEMG/fNIRS 多模态质量控制流水线，针对不同信号独立设计阈值、频域指标和质量掩码，输出可追溯的通道级质量分级与失败原因。
