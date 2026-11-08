import zipfile,html
from pathlib import Path
p=Path('preprocess_review_20260914.docx'); tmp=Path('preprocess_review_20260914.tmp.docx')
paras=[
'2026年9月14日 数据抢救型预处理复核与多层特征报告',
'本报告针对当前采集质量偏低的问题，评估哪些信号可以修复、哪些只能降权使用，以及处理后能够稳定提取的人眼可读特征和后续模型特征。新增的 sec_test 第二测试数据已纳入本轮复核。',
'一 数据覆盖与复跑结果',
'从 data 目录重新扫描 7 位受试者，共 107 条记录。预处理状态为：READY_WITH_MASKS 41 条、READY_CHANNEL_FILTERED 33 条、READY_CONTINUOUS 14 条、眼动任务红外干扰降级 16 条、EEG 直接信号失败 3 条。新增第二测试数据被自动识别，结果与既有索引一致。',
'二 目前流程仍可优化的地方',
'建议把“硬删除”改成三级质量分层：A级为可直接统计，B级为保留但按质量权重进入模型，C级为仅保留原始和诊断特征。EEG 对坏段使用 epoch mask，坏导联在空间邻近且可用导联足够时插值；fNIRS 保留突变修复前后的质量指标，眼动任务单独建模红外干扰；EMG 和 IMU 保留时间戳不强行伪造同步。这样可以抢救更多信息，同时避免把修复值当成真实观测。',
'三 处理前后可以直接看懂的一级特征',
'EEG：波形振幅范围、异常尖峰比例、平线比例、工频峰、θ/α/β频带功率和动作窗有效率。fNIRS：光强正值比例、光密度突变数、有效通道比例、HbO/HbR相对变化方向。EMG：整流包络高低、RMS、峰峰值、活动持续时间和共激活。IMU：动态加速度强度、角速度强度、姿态倾角代理和动作周期。',
'四 后续特征提取后的二级与三级特征',
'二级特征是可输入传统机器学习的窗口统计量：均值、标准差、RMS、峰度、频带功率、谱熵、斜率、恢复时间、HbO-HbR差值、EMG包络积分、肌间相关和脑肌耦合。三级特征是跨模态和受试者层面的结构量：任务分类概率、前后变化量、个体基线归一化、脑肌同步网络、状态转移熵、受试者质量加权的纵向响应指纹。三级特征必须保留来源窗口、质量权重和缺失标记。',
'五 图形化解读原则',
'报告中的处理前后图应优先展示同一通道同一时间窗，叠加原始波形、滤波波形、质量遮罩和频谱。图中任何“更平滑”只能解释为噪声下降或可分析性提高，不能直接解释为疗效。对于低质量记录，建议同时显示原始、修复后和被遮罩区间，老师可以据图判断抢救是否合理。',
'六 当前结论',
'现有流程的核心方向是正确的，但可以通过质量分层、保留修复前后双版本、增加窗口级权重和跨模态缺失标记来扩大可用特征范围。3 条 EEG 直接信号失败仍不建议生成伪恢复数据；其余记录应尽量保留质量指标，供后续模型学习“信号可信度”。'
]

def esc(s): return html.escape(s)
with zipfile.ZipFile(p) as z: files={n:z.read(n) for n in z.namelist()}
xml=files['word/document.xml'].decode('utf-8'); insert=''.join(f'<w:p><w:r><w:t>{esc(x)}</w:t></w:r></w:p>' for x in paras); xml=xml.replace('</w:body>',''+insert+'</w:body>'); files['word/document.xml']=xml.encode('utf-8')
with zipfile.ZipFile(tmp,'w') as z:
 for n,b in files.items(): z.writestr(n,b)
tmp.replace(p)
print(p)
