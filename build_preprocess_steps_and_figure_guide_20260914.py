from pathlib import Path
import csv
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parent
ASSET = ROOT / "best_subject_preprocessing_comparison_20260914"
OUT = ROOT / "preprocess_steps_and_figure_guide_20260914.docx"

def font(run, size=10.5, bold=False, color="000000"):
    run.font.name = "Microsoft YaHei"
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    run.font.size = Pt(size); run.bold = bold; run.font.color.rgb = RGBColor.from_string(color)

def para(doc, text="", size=10.5, bold=False, align=None, after=6):
    p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(after); p.paragraph_format.line_spacing = 1.15
    if align is not None: p.alignment = align
    if text: font(p.add_run(text), size, bold)
    return p

def heading(doc, text, level=1):
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(11); p.paragraph_format.space_after = Pt(5); p.paragraph_format.keep_with_next = True
    font(p.add_run(text), 15 if level == 1 else 12, True); return p

def shade(cell, fill):
    shd = OxmlElement("w:shd"); shd.set(qn("w:fill"), fill); cell._tc.get_or_add_tcPr().append(shd)

def table(doc, headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers)); t.autofit = False
    for i, h in enumerate(headers):
        c=t.rows[0].cells[i]; c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
        if widths: c.width=Inches(widths[i])
        shade(c,"1F4E79"); p=c.paragraphs[0]; p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        font(p.add_run(h),9,True,"FFFFFF")
    for ri,row in enumerate(rows):
        cs=t.add_row().cells
        for i,v in enumerate(row):
            c=cs[i]; c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if widths: c.width=Inches(widths[i])
            if ri%2: shade(c,"F3F6F8")
            p=c.paragraphs[0]; p.paragraph_format.space_after=Pt(0); font(p.add_run(str(v)),8.8)
    return t

PANEL_NOTES = {
    "01_subject_selection.png": [
        ("面板：质量排名", "现象：各完整访视的综合分存在差异，本记录位于最高位。处理作用：把完整性、伪迹比例和通道保留率合并为可审计的选择依据。可提取特征：质量分、模态完整性、坏段比例和有效通道数。限制：这是质量筛选，不是受试者生理能力或疗效排名。"),
    ],
    "02_eeg_time_domain_before_after.png": [
        ("上方面板：处理前 C3", "现象：原始电位中可见工频纹波、尖峰和较大的慢漂移。处理作用：记录原始问题，避免把清理效果误认为生理变化。可提取特征：峰峰值、斜率、漂移、异常尖峰密度。限制：原始大幅变化既可能是伪迹，也可能包含真实活动，不能只凭振幅判定。"),
        ("下方面板：处理后 C3", "现象：波形更适合观察节律，异常 2 秒窗由掩码管理。处理作用：重参考、50 Hz 陷波和 0.5–45 Hz 带通降低共模、工频和带外噪声。可提取特征：窗口均值、方差、峰度、有效比例和任务期变化。限制：未连接 EOG/ECG，眼动和心电伪迹不能被宣称已完全去除。"),
    ],
    "03_eeg_theta_alpha_beta.png": [
        ("上方面板：频段包络", "现象：θ、α、β 包络随时间起伏。处理作用：把难以直接阅读的快速电位转换为频段能量的时间轨迹。可提取特征：各频段均值、峰值、斜率、任务/静息比值和频段间比值。限制：单个频段变大不等于认知改善，必须有预设比较和质量权重。"),
        ("下方面板：动作期与恢复期频谱", "现象：不同时间段的功率谱峰和能量分布可比较。处理作用：滤波和分段减少带外干扰，使频率结构更稳定。可提取特征：δ/θ/α/β 功率、峰频、谱熵和 1/f 斜率。限制：5 秒动作窗较短，频率分辨率有限；运动伪迹可能仍影响高频功率。"),
    ],
    "04_fnirs_intensity_od_mbll.png": [
        ("上方面板：双波长原始光强", "现象：两波长光强存在缓慢变化和局部突跳。处理作用：保留探头接触、遮挡和运动问题的直接证据。可提取特征：光强中位数、变异系数、突跳数量和双波长相关性。限制：原始光强不是血氧浓度，设备尺度不能直接跨人比较。"),
        ("中方面板：处理后光密度 OD", "现象：突变被修复，曲线进入可分析的慢变化范围。处理作用：光强转 OD、突变处理和 0.01–0.2 Hz 滤波，并由样本掩码标出无效区间。可提取特征：OD 均值、斜率、峰值、恢复时间和有效样本率。限制：无短程通道，系统生理噪声仍可能存在。"),
        ("下方面板：相对 MBLL", "现象：ΔHbO 与 ΔHbR 可在事件基线下观察相对变化。处理作用：按光路距离、DPF 和消光系数把双波长变化转为相对浓度估计。可提取特征：峰值、面积、时间到峰、HbO-HbR 协同和任务/基线差。限制：这是相对估计，不是绝对浓度或单独的脑区激活证据。"),
    ],
    "05_emg_before_after_envelope.png": [
        ("上方面板：原始 sEMG", "现象：原始波形快速振荡，夹杂工频和宽带噪声。处理作用：保留传感器原始状态，便于追溯接触和饱和问题。可提取特征：幅值分布、零交叉、异常比例和基线噪声。限制：CSV 未提供经过校准的物理幅值单位。"),
        ("下方面板：滤波波形与包络", "现象：滤波后的肌电快速成分和橙色包络更易阅读。处理作用：20–450 Hz、陷波、整流和 5 Hz 低通将肌肉活动强度显式化。可提取特征：RMS、包络均值/峰值、活动持续时间、起止沿和肌肉共激活。限制：缺少可信动作标记时，不能把峰值精确归因到某次动作。"),
    ],
    "06_imu_motion_intensity_tilt_proxy.png": [
        ("上方面板：动态加速度 RMS", "现象：1 秒窗运动强度随记录时间变化。处理作用：用加速度模长的动态部分描述运动，不进行会漂移的陀螺积分。可提取特征：RMS、峰值数、jerk、活动占比和运动复杂度。限制：源单位和坐标系信息有限，跨设备绝对比较需谨慎。"),
        ("下方面板：roll/pitch 姿态代理", "现象：重力占主导时可见姿态角度代理变化。处理作用：用加速度方向估计倾斜趋势。可提取特征：角度范围、速度、稳定性和左右对称性。限制：不能得到绝对 yaw、关节角或完整运动学。"),
    ],
}

def figure_page(doc, filename, title, text):
    p=heading(doc,title); p.paragraph_format.page_break_before=True
    para(doc,text,10.2)
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.add_run().add_picture(str(ASSET/filename),width=Inches(6.75))
    for panel, note in PANEL_NOTES.get(filename, []):
        p=doc.add_paragraph(); p.paragraph_format.space_after=Pt(5); p.paragraph_format.line_spacing=1.1
        font(p.add_run(panel+"。"),10.1,True)
        font(p.add_run(note),10.1)

def main():
    d=Document(); s=d.sections[0]; s.top_margin=Inches(.62); s.bottom_margin=Inches(.58); s.left_margin=Inches(.7); s.right_margin=Inches(.7)
    d.styles["Normal"].font.name="Microsoft YaHei"; d.styles["Normal"].font.size=Pt(10.5)
    title=d.add_paragraph(); title.alignment=WD_ALIGN_PARAGRAPH.CENTER; title.paragraph_format.space_before=Pt(34)
    font(title.add_run("代表性受试者数据预处理前后对比"),21,True)
    para(d,"从原始信号到可解释分析数据的图文汇报",12,False,WD_ALIGN_PARAGRAPH.CENTER,16)
    para(d,"数据版本：review_20260914　代表性访视：2026081502 / sec_test",10.5,False,WD_ALIGN_PARAGRAPH.CENTER,16)
    p=para(d,"先说结论：本次选择的完整多模态访视综合质量分为 98.563/100。预处理显著降低 EEG 工频干扰，保留 44/44 条 fNIRS 通道和 99.302% 的分析样本，并保留 EMG 与 IMU 的可解释活动指标。图形用于说明数据质量和特征可读性，不能单独证明疗效或因果机制。",11,False,None,10)
    heading(d,"一、为什么选这条记录")
    para(d,"先按完整性准入，再按质量排名：EEG、fNIRS、EMG、IMU 四种模态均可用；EEG 和 fNIRS 的核心运动标记完整；EMG 7 个传感器映射有效；每个传感器均有 Acc 和 Gyro。综合分由 EEG 伪迹、坏导联、fNIRS 通道/样本保留率和 EMG 质量组成。",10.5)
    table(d,["指标","结果","人眼可理解的含义"],[["综合质量分","98.563/100","在完整访视中排名第一"],["EEG","218/253 个有效 2 秒窗；坏导联 4.52%","大部分时间窗可进入统计"],["fNIRS","44/44 通道；有效样本 99.302%","通道和时间覆盖都很完整"],["EMG + IMU","7/7 映射；7/7 Acc+Gyro","肌肉活动和运动强度可同步作为辅助特征"]],[1.5,1.55,3.7])
    heading(d,"二、处理流程与质量分层")
    table(d,["模态","处理前主要问题","处理后保留内容"],[["EEG","约 50 Hz 工频和短时大幅伪迹","重参考、陷波、0.5–45 Hz；异常窗/导联显式掩码"],["fNIRS","双波长光强尺度不统一、突变和无效点","OD、突变修复、0.01–0.2 Hz；通道和样本掩码"],["EMG","工频、宽带噪声、原始波形难读","20–450 Hz、陷波、整流包络"],["IMU","多轴曲线难直接比较、积分会漂移","动态加速度、1 秒运动强度、重力主导时的姿态代理"]],[1.0,2.4,3.35])
    para(d,"建议按质量分层进入模型：A 层（高质量，完整且有效比例高）直接用于主要分析；B 层（局部污染或部分掩码）保留有效窗并携带质量权重；C 层（长时间缺失、关键模态缺失或无法同步）用于探索性分析或仅提取仍可信的单模态特征。所有层级都保留原始文件、处理后数组、掩码和处理参数。",10.5)
    heading(d,"三、从图上能读出的特征层级")
    table(d,["层级","具体特征","来源/解释"],[["一级：直接可见","振幅、峰谷、基线漂移、突变、工频纹波、有效区间","处理前后时域图；描述信号是否连续、是否有明显污染"],["二级：窗口统计","EEG θ/α/β 功率与包络、fNIRS ΔHbO/ΔHbR、EMG RMS/包络、IMU 动态 RMS/jerk","按 1–2 秒窗计算均值、标准差、峰值、斜率、有效比例，并附质量权重"],["三级：跨模态/纵向","脑-肌耦合、脑-运动延迟、任务-静息差、基线-复测变化、受试者稳定性","在受试者分组交叉验证后提取；只在时间对齐和质量门控通过时解释"]],[1.05,2.55,3.15])
    para(d,"一级特征适合老师看图说事；二级特征适合统计和传统机器学习；三级特征适合研究问题建模，但必须防止把运动伪迹当成脑信号。任何二级、三级特征都应同时保存 valid_fraction、artifact_fraction、interpolation_fraction 和 quality_tier。",10.5)
    heading(d,"四、处理前后质量变化的定量证据")
    para(d,"下面的数字来自同一条代表性访视，不是对全体受试者的推断。它们回答的是：经过处理后，哪些干扰被压低，哪些信息被保留下来，以及哪些时间窗仍需谨慎使用。",10.5)
    table(d,["对象","处理前可见问题","处理后结果","解释"],[["EEG 工频","50 Hz 附近中位干扰 177.773 µV","处理后 0.074 µV","工频干扰显著降低，频谱更适合分析"],["EEG 时间窗","总计 253 个 2 秒窗","有效 218 个，约 86.2%","保留大部分数据，35 个窗需掩码/降权"],["fNIRS 通道/样本","原始双波长光强含突跳和无效点","44/44 通道；99.302% 样本可分析","空间和时间覆盖均较完整"],["EMG","快速振荡且不易观察活动强度","7/7 映射；质量分 95.714","包络可用于窗口活动强度"],["IMU","多轴曲线难直接量化","7/7 具备 Acc+Gyro","可提取运动强度和姿态代理"]],[1.05,2.1,1.65,2.0])
    para(d,"图形展示的是可读性提升和质量可追溯性。处理不会凭空增加不存在的信息；掩码、缺失和不确定性应与信号值一起进入后续分析。",10.5)
    heading(d,"五、质量分层如何进入模型")
    table(d,["层级","判定原则","允许使用的内容","建模策略"],[["A 高质量","完整模态、有效比例高、局部伪迹少","全部有效窗及常规二级特征","主分析；质量权重接近 1"],["B 可抢救","局部污染、少量插值或部分坏导联","有效窗、掩码后的窗口统计、单模态特征","保留；显式加入质量分和有效比例"],["C 探索性","长缺失、关键模态缺失、同步不可靠","仍可信的单模态一级/二级特征","只做探索或敏感性分析，不进入核心结论"]],[1.0,2.0,2.25,1.55])
    para(d,"质量等级必须在切分数据前确定，并写入每个窗口和每个特征。不能用标签信息或测试集结果反向调整质量阈值，否则会造成评估偏差。",10.5)
    heading(d,"六、图形化结果")
    figure_page(d,"01_subject_selection.png","图1　代表性访视的质量排名","横轴是综合质量分。蓝色为本次代表性记录；它同时满足四模态完整性和核心运动标记要求，因此适合做处理前后展示。排名只用于选择教学个例，不代表受试者间的生理差异或疗效差异。")
    figure_page(d,"02_eeg_time_domain_before_after.png","图2　EEG 处理前后时域波形","灰线是原始 C3，蓝线是处理后同一导联。重点观察工频纹波、异常尖峰、基线漂移和有效窗。处理后的曲线更适合观察节律和统计，但不能把更平滑直接解释为功能改善。")
    figure_page(d,"03_eeg_theta_alpha_beta.png","图3　EEG 节律和频谱特征","θ、α、β 包络是二级特征的直观示例；频谱峰表示某频率附近能量集中。应在预先规定的任务/静息和受试者分组比较中解释。")
    figure_page(d,"04_fnirs_intensity_od_mbll.png","图4　fNIRS 从光强到相对血红蛋白变化","上、中、下三行分别展示原始双波长光强、OD 处理结果和相对 MBLL。ΔHbO/ΔHbR 是带有 DPF、路径长度和基线假设的相对估计，不是绝对浓度，也不能单独称为脑区激活。")
    figure_page(d,"05_emg_before_after_envelope.png","图5　EMG 波形与肌肉活动包络","蓝线保留肌电频带，橙线为整流低通包络。包络均值、峰值、活动持续时间是二级特征；没有可信动作标记时，不应把某个峰归因于某一次动作。")
    figure_page(d,"06_imu_motion_intensity_tilt_proxy.png","图6　IMU 运动强度和姿态代理","动态加速度 RMS 是运动强度的一级/二级特征，roll/pitch 是重力主导时的姿态代理。当前资料不足以支持绝对 yaw、完整关节角或完整运动学重建。")
    heading(d,"七、复核结论与后续建模约束")
    for text in ["EEG 伪迹应优先做窗口级掩码和降权，避免整条导联删除造成特征损失。", "fNIRS 短缺失可以插值，长缺失保留 NaN；插值比例和最长连续插值时长必须进入质量特征。", "多模态模型按受试者划分训练、验证和测试集；同一受试者的不同访视不得随机拆分，避免信息泄漏。", "保留 raw、filtered、repaired 三个版本和参数哈希，任何图和特征都能追溯到输入。", "本报告使用的结构化指标见 figure_metrics.csv，完整选择审计见 subject_selection.csv。"]: para(d,"• "+text,10.5,False,None,3)
    heading(d,"八、下一步实施顺序")
    para(d,"第一步，把 EEG 质量判断从整条记录下沉到 1–2 秒窗口；短时污染做掩码或降权，持续平线和断线才判整条导联不可用。第二步，对 fNIRS 限制长距离插值，保留 NaN 和插值比例。第三步，为四种模态统一输出质量分、有效比例、伪迹比例和处理版本。第四步，在受试者层面切分训练、验证和测试集，并在训练集内拟合标准化参数。",10.5)
    para(d,"最终建议：主结果只使用 A 层和高质量 B 层；C 层用于敏感性分析。任何三级跨模态特征都应先证明事件对齐，再检查去除运动高伪迹窗后方向是否稳定。",10.5)
    d.save(OUT); print(OUT)

if __name__ == "__main__": main()
