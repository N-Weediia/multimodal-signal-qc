from __future__ import annotations

from pathlib import Path
import csv
import re

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "best_subject_preprocessing_comparison_20260910"
DOCX = OUT / "preprocess_steps_and_figure_guide_20260910.docx"


def set_run_font(run, name="Microsoft YaHei", size=10.8, bold=False, color="000000"):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
    run.font.size = Pt(size)
    run.bold = bold
    run.font.color.rgb = RGBColor.from_string(color)


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=110, start=130, bottom=110, end=130):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, v in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table, color="D9D9D9", size="6"):
    tbl = table._tbl
    tbl_pr = tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = f"w:{edge}"
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), color)


def keep_with_next(paragraph):
    paragraph.paragraph_format.keep_with_next = True


def add_para(doc, text="", style=None, size=10.8, bold=False, color="000000", before=0, after=6, line=1.15, align=None):
    p = doc.add_paragraph(style=style)
    if text:
        r = p.add_run(text)
        set_run_font(r, size=size, bold=bold, color=color)
    pf = p.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = line
    if align is not None:
        p.alignment = align
    return p


def add_rich_para(doc, pieces, size=10.8, before=0, after=6, line=1.15, align=None):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = line
    if align is not None:
        p.alignment = align
    for piece in pieces:
        if isinstance(piece, tuple):
            text, bold = piece
        else:
            text, bold = piece, False
        r = p.add_run(text)
        set_run_font(r, size=size, bold=bold)
    return p


def add_bullet(doc, lead, text, size=10.6, level=0):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Inches(0.18 + 0.2 * level)
    p.paragraph_format.first_line_indent = Inches(-0.14)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.12
    r = p.add_run(lead)
    set_run_font(r, size=size, bold=True)
    r = p.add_run(text)
    set_run_font(r, size=size)
    return p


def add_heading(doc, text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run(text)
    set_run_font(r, size=15.5 if level == 1 else 12.5, bold=True)
    return p


def add_table(doc, headers, rows, widths=None, font_size=9.4):
    table = doc.add_table(rows=1, cols=len(headers))
    table.autofit = False
    set_table_borders(table)
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        if widths:
            cell.width = Inches(widths[i])
        set_cell_shading(cell, "1F4E79")
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        set_cell_margins(cell)
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        set_run_font(r, size=font_size, bold=True, color="FFFFFF")
    for ri, row in enumerate(rows):
        cells = table.add_row().cells
        for i, value in enumerate(row):
            cell = cells[i]
            if widths:
                cell.width = Inches(widths[i])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)
            if ri % 2 == 1:
                set_cell_shading(cell, "F3F6F8")
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.08
            r = p.add_run(str(value))
            set_run_font(r, size=font_size)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return table


def add_figure_page(doc, figure_name, title, intro, look, meaning, can_say, cannot_say, script, width=6.65):
    # Put the figure heading on a new page without inserting a standalone
    # page-break paragraph. A standalone break can itself spill to the next
    # page when the preceding figure ends close to the footer, creating a
    # completely blank page in Word.
    heading = add_heading(doc, title, 1)
    heading.paragraph_format.page_break_before = True
    add_para(doc, intro, size=10.6, after=8)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(3)
    run = p.add_run()
    run.add_picture(str(OUT / figure_name), width=Inches(width))
    cap = add_para(doc, f"图：{title}", size=9.2, bold=True, color="555555", after=8, align=WD_ALIGN_PARAGRAPH.CENTER)
    for h, body in (("先看哪里", look), ("图上的东西代表什么", meaning), ("可以说什么", can_say), ("不能说什么", cannot_say), ("建议口头讲稿", script)):
        add_rich_para(doc, [(h + "：", True), body], size=10.25, after=5, line=1.12)


def set_field(paragraph, instruction):
    run = paragraph.add_run()
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = instruction
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "end")
    run._r.append(fld_char1)
    run._r.append(instr)
    run._r.append(fld_char2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Inches(8.5)
    sec.page_height = Inches(11)
    sec.top_margin = Inches(0.63)
    sec.bottom_margin = Inches(0.6)
    sec.left_margin = Inches(0.72)
    sec.right_margin = Inches(0.72)
    sec.header_distance = Inches(0.25)
    sec.footer_distance = Inches(0.28)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Microsoft YaHei"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.font.size = Pt(10.8)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.15
    for name, size in (("Title", 21), ("Heading 1", 15.5), ("Heading 2", 12.5), ("Heading 3", 11.2)):
        style = styles[name]
        style.font.name = "Microsoft YaHei"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
    # Word's built-in Title style can carry a blue paragraph border. Remove it
    # explicitly so the title remains a plain black heading as required.
    title_style_ppr = styles["Title"]._element.get_or_add_pPr()
    title_style_border = title_style_ppr.find(qn("w:pBdr"))
    if title_style_border is not None:
        title_style_ppr.remove(title_style_border)

    header = sec.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = header.add_run("多模态数据预处理展示  受试者 2026081502")
    set_run_font(r, size=8.2, color="666666")
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = footer.add_run("阶段性汇报材料  |  第 ")
    set_run_font(r, size=8.1, color="666666")
    set_field(footer, "PAGE")
    r = footer.add_run(" 页")
    set_run_font(r, size=8.1, color="666666")

    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(46)
    title.paragraph_format.space_after = Pt(12)
    r = title.add_run("代表性受试者数据预处理说明")
    set_run_font(r, size=21, bold=True)
    title_ppr = title._p.get_or_add_pPr()
    title_border = title_ppr.find(qn("w:pBdr"))
    if title_border is not None:
        title_ppr.remove(title_border)
    sub = add_para(doc, "从原始信号到可解释分析数据的图文汇报", size=12.5, color="555555", after=18, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc, "汇报对象：课题组教师和非技术汇报者", size=10.5, color="555555", after=2, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_para(doc, "代表性记录：受试者 2026081502 的复测访视", size=10.5, color="555555", after=22, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_rich_para(doc, [("先说结论：", True), "这份材料主要展示我们怎样检查、清理和解释 EEG、fNIRS、EMG 与 IMU 数据。处理后曲线更容易观察，异常数据也有了明确的标记；但单个受试者的图只能证明处理流程可用，不能证明治疗有效或证明某个脑区、穴位和经络之间存在因果关系。"], size=11.2, after=12, line=1.22)
    add_rich_para(doc, [("一个必须分清的区别：", True), "图中写的“处理前”和“处理后”，指同一份复测记录经过算法处理前后，不是治疗前和治疗后。只有比较不同访视并在受试者层面汇总，才是在讨论治疗前后变化。"], size=10.7, after=10, line=1.18)
    add_para(doc, "本文档只使用受试者编号，不呈现姓名、生日和原始文件名。", size=9.7, color="666666", after=12)
    add_heading(doc, "一页先说清楚我们已经做了什么", 1)
    add_bullet(doc, "先把数据分清楚。", "任务按外层文件夹判定，不再依据文件名猜任务；运动任务中的 EEG、fNIRS 和 EMG 按统一规则整理。", 10.55)
    add_bullet(doc, "再把时间对齐。", "fNIRS 和 EEG 的第一个标记作为 EMG 第 0 帧的时间起点，再根据已经确认的动作顺序推算 EMG 动作边界；不把不确定的边界写成真实标记。", 10.55)
    add_bullet(doc, "把传感器对应关系纠正。", "EMG 使用 2、3、4、5、9、10、11 号传感器，分别对应斜方肌上部、三角肌前束、三角肌中束、肱二头肌、肱三头肌、肱桡肌和尺侧腕屈肌。", 10.55)
    add_bullet(doc, "每种信号分别做质量检查。", "坏导联、坏通道、突变样本、缺失样本和不完整动作都明确记录；不把不能确认的数据悄悄补成好数据。", 10.55)
    add_bullet(doc, "把 fNIRS 多算了一条相对 HbO 和 HbR 结果。", "在原来的光密度变化之外，增加了使用消光系数、年龄相关 DPF 和源探距离的相对 MBLL 估计，并比较两种方法的方向是否一致。", 10.55)
    add_bullet(doc, "把脑和肌肉放到同一条分析链上。", "当前重点是运动任务中的脑信号变化和肌肉电活动是否共同变化；认知任务作为辅助观察线，不把它夸大为已经证明的因果链。", 10.55)

    doc.add_page_break()
    add_heading(doc, "一 代表性记录为什么选它", 1)
    add_para(doc, "为了让老师看到一个完整的处理过程，我们不是凭肉眼挑一条最漂亮的曲线，而是先规定准入条件，再在满足条件的记录中比较质量分。完整性优先：缺少关键动作、缺少某种模态、EMG 映射不清楚或 IMU 的加速度和陀螺仪不完整，即使某一项曲线很好，也不作为代表性完整记录。", size=10.7, after=7)
    add_para(doc, "最终选中受试者 2026081502 的复测访视，综合质量分 98.6/100。它有完整的 15 个核心运动动作标记，EEG 动作期广泛伪迹比例为 0%，fNIRS 保留 44/44 个通道、平均有效样本约 99.3%，7 个标准 EMG 传感器映射有效，7 个传感器都具备 Acc 和 Gyro。", size=10.7, after=7)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after = Pt(3)
    p.add_run().add_picture(str(OUT / "01_subject_selection.png"), width=Inches(6.75))
    add_para(doc, "图 1 代表性记录的客观筛选结果", size=9.3, bold=True, color="555555", after=7, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_rich_para(doc, [("怎么看：", True), "先看蓝色横条，再看右侧的伪迹比例和 fNIRS 保留通道。蓝色记录同时满足完整性要求和较高质量分，因此适合作为“处理前后对比的示范样本”。"], size=10.35, after=5)
    add_rich_para(doc, [("不能这样讲：", True), "98.6 分不是疗效分，也不是受试者能力分；它只表示这条记录更完整、更适合展示数据处理。"], size=10.35, after=5)
    add_rich_para(doc, [("口头讲法：", True), "我们先规定完整性门槛，再在完整记录中按质量评分。最后选出的不是最能证明疗效的人，而是最适合把处理流程讲清楚的一条记录。"], size=10.35, after=5)

    doc.add_page_break()
    add_heading(doc, "二 统一的数据处理流程", 1)
    add_para(doc, "四种信号的处理目标不完全一样：EEG 主要处理工频和异常导联，fNIRS 主要处理光强突变、坏通道和有效样本，EMG 主要保留肌肉电活动并提取包络，IMU 主要给出运动强度和保守的倾角代理。所有处理都保留原始文件不动，另外生成带质量标记的分析数据。", size=10.7, after=8)
    add_para(doc, "原始记录  →  按文件夹确认任务  →  检查时间和通道  →  去除或标记明显异常  →  保留可解释的信号  →  形成任务窗和受试者级指标", size=11.0, bold=True, color="1F4E79", after=9, align=WD_ALIGN_PARAGRAPH.CENTER)
    add_table(doc, ["信号", "实际做了什么", "为什么这样做", "处理后如何使用"], [
        ["EEG 脑电", "质量检查先用逐采样中位数参考；正式分析用有效导联均值参考；50 Hz 陷波；四阶零相位 0.5–45 Hz 带通；2 秒窗检查异常和坏导联；必要时记录可视化插值。", "先用稳健参考识别异常，再用有效导联均值形成正式分析信号；去掉共同偏移和工频，保留常用脑节律范围；把异常段标出来，避免把坏段当成脑活动。", "用有效导联和有效 2 秒窗计算节律和任务指标。EOG、ECG 虽有通道名，但实际未接电极，不能校正。"],
        ["fNIRS 脑血氧", "检查非正和缺失光强；双波长通道质量检查；把光强变成 OD；检测并修复突变；0.01–0.2 Hz 滤波；保留通道和样本掩码；另算相对 MBLL。", "先保证数据方向和通道质量，再降低突跳和高频噪声；相对 MBLL 让两种波长能进一步表示 HbO 和 HbR 的相对变化。", "默认只用通过通道和样本质量筛选的数据。眼动 fNIRS 单独保留并标注红外干扰，不和普通运动 fNIRS 混用。"],
        ["EMG 肌电", "按逻辑肌肉校正 2、3、4、5、9、10、11 号传感器；50 Hz 及谐波陷波；20–450 Hz 带通；统一到约 1000 Hz；展示时整流并用 5 Hz 低通得到包络。", "保留肌肉电活动的主要频率范围，压低工频；包络比原始快速振荡更容易看出肌肉活动强弱。", "包络用于肌肉活动强弱和波动比较。EMG 是肌肉激活信号，不是完整关节运动学；当前 CSV 没有独立可靠的任务标记。"],
        ["IMU 惯性数据", "按传感器自己的时间戳读取 Acc 和 Gyro；用 1 秒窗计算动态加速度 RMS；在重力占主导的时段计算 roll/pitch 倾角代理。", "不直接积分陀螺仪，避免没有校准时的漂移；先提供最保守、最容易解释的运动强度和姿态变化。", "用于辅助判断“有没有明显运动”和“姿态是否发生粗略改变”。不能解释为绝对方向、yaw、关节角或完整轨迹。"],
    ], widths=[0.82, 2.45, 2.35, 2.0], font_size=8.55)
    add_heading(doc, "为什么没有做 EOG 和 ECG 校正", 2)
    add_para(doc, "EEG 文件中确实能看到 ECG、HEOR、HEOL、VEOU、VEOL 等通道名称，但根据采集情况，这些通道实际没有接入有效电极。它们没有记录可靠的眼电或心电波形，所以不能拿来做回归、ICA 成分识别或心搏校正。我们采用的做法是：如实记录“通道名存在但没有有效参考信号”，并在报告中把它作为限制，而不是把通道名误当成已完成校正。", size=10.55, after=5)
    add_heading(doc, "fNIRS 相对 MBLL 具体增加了什么", 2)
    add_para(doc, "fNIRS 原始数据有 730 nm 和 850 nm 两种波长。先用自然对数把光强变化写成光密度变化，再用有出处的消光系数和年龄相关 DPF，把两种波长的变化分解成相对 ΔHbO 与 ΔHbR。这里的结果是“相对变化”，不是绝对血红蛋白浓度；源探距离来自文件声明为 cm 的坐标计算，但仍需确认坐标是不是帽子上的实际物理距离。", size=10.55, after=5)
    add_para(doc, "这一步的意义是把原来的单一 OD 曲线分成两条更接近生理含义的曲线，同时保留参数假设。不同 DPF、源探距离和滤波范围下，任务方向在本次检查中保持一致，但幅度会变化，因此结论重点放在方向和可重复性，不能只看一个绝对数值。", size=10.55, after=5)
    add_table(doc, ["相对 MBLL 参数", "本次使用", "汇报时如何理解"], [
        ["两个波长", "730 nm 和 850 nm", "用两种波长的变化区分 HbO 和 HbR 的相对变化。"],
        ["消光系数", "730 nm：HbO 390、Hb 1102.2；850 nm：HbO 1058、Hb 691.32（cm⁻¹/M）", "表示不同血红蛋白对两种光的吸收不同；参数来源固定后，结果才可复算。"],
        ["年龄相关 DPF", "730 nm 约 7.294；850 nm 约 6.217", "表示光在组织中的实际路程比源探直线距离更长；会影响幅度。"],
        ["源探距离", "约 5.101 cm，来自文件声明为 cm 的坐标计算", "这是坐标推得的距离，仍需确认是否等于帽子上的实际物理距离。"],
        ["输出", "相对 ΔHbO 和相对 ΔHbR，单位按换算结果展示", "用于看相对方向和时间变化，不宣称绝对血红蛋白浓度。"],
    ], widths=[1.45, 2.75, 3.5], font_size=8.9)

    add_figure_page(doc, "02_eeg_time_domain_before_after.png", "三 EEG 时域信号处理前后", "这张图用同一条 EEG 导联、同一段时间，直接展示处理前后发生了什么。它回答的是“数据有没有变得更适合分析”，不是“治疗有没有效果”。", "先看上下两幅图的纵轴。上图原始信号有很大的整体偏移，细小脑电活动被挤在大偏移上；下图经过参考处理和滤波后围绕 0 附近波动，更容易看到有效变化。再看橙色 0–5 秒，这是动作期；蓝色 5–20 秒，是动作后静息期。", "灰色线是尚未重参考、尚未滤波的原始单导联。蓝色线是用有效导联均值构造分析参考后，再经过 50 Hz 陷波和 0.5–45 Hz 带通的信号。中位数参考只用于前面的稳健质量检查。波峰和波谷只是电位随时间的瞬时起伏；它们可能来自脑活动，也可能受到肌电、动作、接触变化等影响。", "可以说：处理降低了共同偏移和工频影响，使波形更容易阅读；处理后的 2 秒窗还会经过异常幅度和接近平线检查，不能用的窗不进入统计。", "不能说：蓝线更整洁就代表治疗有效；也不能把某一个波峰对应到某一个动作或脑区，更不能说眼电和心电已经被校正。", "这页主要证明我们做了三件事：先用稳健参考帮助检查异常，再用有效导联均值形成分析参考，接着去掉 50 Hz 工频并保留 0.5 到 45 Hz 的脑电分析范围。处理后波形更容易看，但这只是数据质量改善，不是疗效证据。")

    add_figure_page(doc, "03_eeg_theta_alpha_beta.png", "四 EEG 的 theta alpha beta 节律怎么看", "这张图把处理后的 EEG 拆成三个常用频段，帮助我们说明 EEG 不只看一条杂乱曲线，还可以看不同频率范围的变化。", "先看上图三条包络线，再看下图两个时间段的频谱。上图横轴是时间，阴影仍然标出动作期和动作后静息期；下图横轴是频率，橙色和蓝色分别比较动作期与动作后静息期。", "上图包络线越高，表示该频段在当时的瞬时振幅更大，不是说神经元数量更多。下图某个频率附近出现峰，表示这个时间段在该频率附近的振荡能量更集中。theta 为 4–8 Hz，常在注意、工作记忆和控制任务中作为低频节律观察；alpha 为 8–13 Hz，常用于观察清醒静息、感觉抑制和任务状态变化；beta 为 13–30 Hz，常用于观察运动和认知状态相关的较快节律。它们都是分析频段，不是单独的疾病诊断标签。", "可以说：这条 C3 导联在本示例片段中，alpha 包络在不少时段高于 theta，beta 在动作段有若干较明显起伏；因此后续可以在预先规定的任务、静息和前后访视之间比较频段功率或比值。", "不能说：某一个 alpha 或 beta 峰就是认知改善、运动改善或疗效；不能把单导联个例图当成全脑结论；也不能忽略年龄、闭眼睁眼、肌电和动作伪迹对频段的影响。", "这张图不是在说某个峰值就是疗效，而是在说明我们把脑电按频率拆开了。theta、alpha、beta 的高低要在同一任务和同一统计规则下比较，单张个例图只能说明这条记录里哪些频率成分更突出。")

    add_figure_page(doc, "04_fnirs_intensity_od_mbll.png", "五 fNIRS 从光强到 OD 和相对 HbO HbR", "这张图把 fNIRS 的三个层次放在一起：仪器直接记录的光强、光密度 OD，以及进一步用相对 MBLL 换算出的 HbO 和 HbR。", "从上到下看三层。第一层是 730 nm 和 850 nm 原始光强的片段归一化曲线；第二层是经过突变处理和 0.01–0.2 Hz 滤波的 OD；第三层是相对 ΔHbO 和相对 ΔHbR。橙色 0–5 秒为摸头动作，蓝色 5–20 秒为动作后静息。", "原始光强越低，OD 通常越高，因为 OD 是光被吸收变化的对数表达。OD 仍是光学变化指标；相对 MBLL 用两种波长、消光系数、DPF 和源探距离把变化分解成两条曲线。图中 ΔHbO 橙线在动作后有明显上升，ΔHbR 蓝线变化较小，这只能描述这个通道在所选基线下的相对估计。", "可以说：我们已经不只看原始光强，还检查了通道质量、突变和有效样本，并增加了相对 HbO/HbR 计算；本条记录保留 44/44 通道，平均有效样本约 99.3%。", "不能说：HbO 上升就等于某个已定位脑区激活或治疗有效；不能把坐标标签直接当成 M1、SMA 等脑区；不能把相对浓度变化说成绝对浓度；0.01–0.2 Hz 滤波也不等于完全去掉心跳和头皮血流影响。", "这一页的重点是展示计算链条。仪器先给光强，我们把它变成 OD，再用相对 MBLL 得到 HbO 和 HbR。这样生理解释更接近，但因为没有短程通道和完整系统生理参考，当前仍然只能作相对、探索性的结果。", width=6.0)

    add_figure_page(doc, "05_emg_before_after_envelope.png", "六 EMG 肌肉电活动和包络怎么看", "这张图展示三角肌前束的一小段连续 EMG。它的作用是说明如何从快速、密集的肌电波形中提取容易解释的肌肉活动强度。", "先看上图的灰色原始曲线：正负快速摆动很多，难以直接比较。再看下图：蓝线是 20–450 Hz 带通并去工频谐波后的肌电，橙线是整流后再低通得到的包络。这里的 6 秒片段是连续记录中的展示片段，不把它强行标成某一次动作。", "原始 EMG 正负摆动是肌肉电信号方向和瞬时振荡。包络线越高，通常表示这段时间内该肌肉的电活动更强；包络线变平，表示活动相对稳定或减弱。传感器 3 的肌肉标签来自已经确认的对应表，不表示它一定是动作的主要发力肌。", "可以说：处理后仍保留了快速肌电，同时用包络把肌肉活动的强弱趋势显示出来；这为比较不同任务或不同访视中的肌肉激活和波动提供了基础。", "不能说：EMG 包络就是关节位移、动作幅度或完整运动学；不能把图中任意一个峰归因于某次摸头动作；不能忽略电极贴合、皮下脂肪、串扰和传感器放置对幅值的影响。", "EMG 这一页展示的是从原始肌电到包络的过程。我们能比较肌肉电活动强弱，但 EMG 不是摄像机，也不是关节角度传感器；要把某个峰说成某个动作，必须有可靠同步事件。")

    add_figure_page(doc, "06_imu_motion_intensity_tilt_proxy.png", "七 IMU 能告诉我们什么", "IMU 能补充肌肉数据没有直接提供的运动强度和姿态变化信息，但在缺少坐标系、绑带方向和校准资料时，必须使用保守解释。", "上图是 1 秒窗的动态加速度 RMS，可以先看哪里出现明显运动；下图是由重力方向计算的 roll 和 pitch 倾角代理。图中约 121 秒附近动态运动强度明显升高，同时 roll 发生较大改变。", "动态加速度 RMS 越高，表示这一秒内加速度变化更大，说明运动更明显。roll/pitch 不是完整姿态，而是根据重力方向估算的粗略倾角代理；只有重力占主导时才有较清楚的姿态含义。", "可以说：本记录的 IMU 具备完整 Acc 和 Gyro 字段，因此可以做运动强度和保守姿态辅助检查；它能帮助我们判断某段 EMG 活动是否伴随较明显身体运动。", "不能说：roll 或 pitch 的绝对数值就是身体的真实关节角；不能从这张图得到绝对 yaw、完整姿态或手臂运动轨迹；也不能单凭 IMU 变化证明脑信号变化来自某个机制。", "IMU 这一页的结论很简单：它能告诉我们什么时候动得更明显，能给一个粗略倾角参考，但还不能还原完整关节运动学。因此它是 EMG 和 fNIRS 的辅助证据，不是独立的运动疗效终点。")

    doc.add_page_break()
    add_heading(doc, "八 处理完的数据应该怎样看", 1)
    add_para(doc, "看处理后曲线时，先问“这一条线是什么量”，再问“它在什么时间段变化”，最后才问“它能不能和任务联系起来”。下面的顺序适用于老师现场看图。", size=10.7, after=8)
    add_table(doc, ["先确认", "看什么", "稳妥的解释"], [
        ["时间", "动作期 0–5 秒、动作后静息 5–20 秒，或明确的任务窗。", "先确认变化是否发生在预先定义的时间段，避免把背景波动当成任务反应。"],
        ["方向", "曲线向上还是向下，是否在多个重复动作或多个受试者中一致。", "单次上升或下降只是该通道和该基线下的变化；要看重复性和群体方向。"],
        ["幅度", "变化有多大，单位是什么，是否受参数影响。", "OD 和相对 MBLL 的幅度受处理参数影响，方向稳定不代表幅度稳定。"],
        ["波峰波谷", "峰值出现的时间和持续时间。", "峰值可以提示短时间内活动增强，但不能单独说明原因、脑区或疗效。"],
        ["阴影与标记", "动作期、静息期、坏段和不可用通道。", "阴影是时间范围，mask 是质量记录；它们不是生理信号本身。"],
        ["跨模态", "EEG、fNIRS、EMG、IMU 是否在同一时间和同一任务下共同变化。", "共同变化可以形成候选关联，但没有独立行为和运动学金标准时，不能写成因果链。"],
    ], widths=[1.1, 2.55, 3.95], font_size=9.15)
    doc.add_page_break()
    add_heading(doc, "九 当前结果能回答到哪一步", 1)
    add_para(doc, "当前工作已经把“数据能不能用、怎样处理、怎样解释”这件事做成了可重复的流程。它可以支持下一步的受试者级统计，但代表性个例图本身不是群体结果。现阶段更合适的论文主线是固定治疗方案前后，多模态脑反应与肌肉执行的探索性关联。", size=10.7, after=7)
    add_bullet(doc, "关于认知和运动。", "目前可以把逻辑改写为：同一受试者在固定干预前后，先观察认知任务中的脑状态，再观察运动任务中的脑反应和肌肉执行，最后在受试者层面检验这些变化是否一起变化。由于认知任务和运动任务不是同一个实时 trial，也缺少完整的眼动准确率和反应时，因此这是“跨任务关联”，不是“认知导致运动改善”。", 10.45)
    add_bullet(doc, "关于穴位、经络和脑网络。", "目前可以把固定穴位组合或经络相关治疗方案作为干预背景，研究其前后脑信号、脑血氧和肌肉活动的共同变化。但没有单穴随机、sham 对照、不同经络组合比较和源级定位，就不能证明某一个穴位通过某条经络改变了某个脑网络。", 10.45)
    add_bullet(doc, "关于人工智能。", "现阶段人工智能的实际作用主要是自动整理任务、做质量筛选、管理缺失、完成跨模态对齐、生成受试者级指标和做留一法检查。它不是已经训练出一个临床诊断模型，也不是已经完成实时闭环。", 10.45)
    add_bullet(doc, "关于当前最值得继续验证的现象。", "已有分析中，OD proxy 与相对 MBLL 在 19 个可比较的受试者级方向中有 18 个一致；36 组参数敏感性分析的 108/108 个任务设置保持默认方向。但方向一致不等于幅度不变，更不等于疗效已经得到验证。", 10.45)
    add_bullet(doc, "关于当前数据边界。", "7 名配对受试者、14 次访视、200 个完整运动任务窗适合做流程验证和探索性趋势排序，不适合训练深度网络或做临床预测。眼动 fNIRS 单独作为有红外干扰的降级分支；EOG/ECG 不做校正；没有短程通道，因此不能做短程回归。", 10.45)

    doc.add_page_break()
    add_heading(doc, "十 现场三到五分钟汇报稿", 1)
    script = [
        "老师您好，这次汇报不是先展示疗效，而是先把我们目前的数据处理工作讲清楚。我们面对的是 EEG、fNIRS、EMG 和 IMU 四种信号，第一步先按外层文件夹确认任务，避免根据文件名误判。随后把 EEG 和 fNIRS 的第一个标记作为 EMG 第 0 帧的时间起点，再根据已经确认的动作顺序推算 EMG 动作边界。EMG 的传感器对应关系也已经按照实际情况改正为 2、3、4、5、9、10、11 号。",
        "为了展示处理效果，我们先规定完整性门槛，再按 EEG、fNIRS、EMG 和 IMU 的质量指标选择代表性记录。受试者 2026081502 的复测访视综合分 98.6 分，15 个核心动作完整，EEG 动作期广泛伪迹为 0%，fNIRS 44 个通道全部保留，平均有效样本约 99.3%，EMG 和 IMU 字段也完整。因此它适合展示流程，但这个分数不是疗效分。",
        "EEG 先做重参考、50 Hz 陷波和 0.5 到 45 Hz 带通，再按 2 秒窗检查异常。图中处理后波形围绕 0 附近，说明共同偏移和工频影响被压低。接着把 EEG 分成 theta、alpha 和 beta 三个频段，包络反映某个频段当时的振幅，频谱峰反映某个频率附近的能量更集中。这里不能根据单个峰说认知改善或治疗有效，只能在预先规定的任务和前后访视中做受试者级比较。",
        "fNIRS 先检查光强、坏通道、缺失和突变，再计算光密度 OD，并增加相对 MBLL，把两种波长换算成相对 HbO 和 HbR。这样比只看原始光强更接近生理解释，但仍然受 DPF、源探距离和滤波参数影响。我们在本记录中还做了敏感性分析，方向比较稳定，但幅度会变化。因此当前只能说相对方向和可重复性，不能把 HbO 上升直接说成某个脑区激活或疗效。",
        "EMG 经过工频处理和 20 到 450 Hz 带通，再用整流加低通得到包络。包络高表示肌肉电活动较强，但 EMG 不是完整运动学。IMU 用 1 秒窗给出运动强度和保守的 roll、pitch 倾角代理，能辅助判断有没有明显运动，但不能还原完整关节角和绝对姿态。",
        "关于老师提出的认知和运动逻辑，目前我们的回答是：旧的展示确实把认知和运动分开列出来，缺少可检验的连接。现在可以用同一受试者前后变化，把认知任务脑状态、运动任务脑反应和 EMG 肌肉执行放到同一条关联分析中。但因为两类任务不是同一个实时 trial，也缺少完整行为金标准，我们只能说共同变化，不能说认知导致运动改善。",
        "关于穴位、经络和脑网络，目前可以把固定穴位组合或经络相关治疗方案作为背景，观察干预前后的脑血氧、脑状态和肌肉活动是否共同变化。由于没有单穴或 sham 对照，也没有源级脑网络定位，我们不能声称某一个穴位通过某条经络改变了某个脑网络。论文应当先把主线写成固定方案下的多模态脑肌响应，等样本和元数据足够，再验证更强的机制假设。",
        "最后，人工智能目前主要用在自动整理、质量筛选、缺失感知、跨模态对齐和可解释的受试者级指标上，而不是在 7 名受试者上训练黑箱模型。下一步先固定这套预处理和主要终点，把新增受试者沿相同规则加入，再在样本量足够时比较更复杂的可解释模型。",
    ]
    for para in script:
        add_para(doc, para, size=10.6, after=7, line=1.17)

    add_heading(doc, "十一 术语白话表", 1)
    add_table(doc, ["术语", "白话解释", "汇报时建议怎么说"], [
        ["重参考", "用多条相对可靠的 EEG 导联估计共同电位，再从每条导联中减掉。", "把大家共同受到的偏移先压下去。"],
        ["陷波", "专门压低 50 Hz 及 EMG 中的工频谐波。", "减少电源工频干扰。"],
        ["带通滤波", "只保留某一段频率，抑制太慢或太快的成分。", "保留当前模态真正需要看的频率范围。"],
        ["theta alpha beta", "EEG 中常用的 4–8、8–13、13–30 Hz 频段。", "不同频率范围的节律强度，不能从单个峰值判断疗效。"],
        ["光密度 OD", "根据光强变化计算出的相对吸收变化指标。", "先把仪器光强变成可比较的光学变化。"],
        ["HbO HbR", "氧合和脱氧血红蛋白的相对变化估计。", "相对 ΔHbO 和 ΔHbR，不是绝对浓度。"],
        ["MBLL", "利用两种波长和消光系数把光学变化分解成 HbO 与 HbR 的方法。", "让 fNIRS 结果更接近生理含义，但需要参数假设。"],
        ["DPF", "光在头皮和脑组织里实际走过的路程相对直线距离的倍数。", "它会影响 HbO 和 HbR 的幅度，所以要做敏感性分析。"],
        ["EMG 包络", "把快速正负摆动的肌电整流后平滑，得到活动强弱趋势。", "包络高通常表示肌肉电活动更强，不是关节位移。"],
        ["IMU 倾角代理", "用加速度中重力方向估出的粗略 roll/pitch。", "只能作姿态参考，不能代替完整运动学。"],
        ["质量掩码", "记录哪些通道或时间点可以进入统计。", "坏段被明确排除，不是悄悄修成好数据。"],
        ["受试者级指标", "每名受试者先汇总成一个或几个变化量，再比较受试者之间的结果。", "避免把很多时间窗误当成很多受试者。"],
    ], widths=[1.25, 3.45, 2.9], font_size=9.25)

    add_heading(doc, "十二 下一步工作顺序", 1)
    add_bullet(doc, "第一步，固定预处理契约。", "保持任务按外层文件夹判定、统一标记逻辑、EMG 传感器映射、EEG/fNIRS/EMG/IMU 质量门控和相对 MBLL 参数记录不变。", 10.45)
    add_bullet(doc, "第二步，固定主要比较。", "建议以运动任务的脑血氧变化加 EMG 肌肉活动作为主线，认知任务和眼动 fNIRS 作为辅助或探索分支。", 10.45)
    add_bullet(doc, "第三步，做受试者级检验。", "每个人先得到自己的前后变化，再报告有效样本数、方向、置信区间和留一法稳定性，不把时间窗数量当成样本量。", 10.45)
    add_bullet(doc, "第四步，样本增加后再做 AI。", "在样本量足够时比较透明规则模型、分层模型和可解释的图模型；现阶段不以深度网络精度作为主要成果。", 10.45)
    add_bullet(doc, "第五步，保留边界。", "不宣称闭环机制、单穴或经络因果、源级脑区机制、已验证标志物、绝对姿态或临床预测。", 10.45)

    add_heading(doc, "参考依据", 1)
    add_para(doc, "相对 fNIRS 血红蛋白变化采用 Scholkmann F and Wolf M. General equation for the differential pathlength factor of the frontal human head depending on wavelength and age. Journal of Biomedical Optics. 2013;18(10):105004. DOI: 10.1117/1.JBO.18.10.105004。消光系数采用工程中固定的 730 nm 和 850 nm 双波长参数。该 DPF 公式的拟合范围与当前年龄、波长和非额区应用之间存在外推边界，因此报告中只使用相对变化并明确这一限制。", size=9.4, color="555555", after=4, line=1.12)
    add_para(doc, "本材料中的图表和数值来自工程目录中已经生成的代表性记录结果。图表用于说明数据处理和解释边界，不作为单个受试者的疗效结论。", size=9.4, color="555555", after=4, line=1.12)
    add_heading(doc, "汇报前的三句话", 1)
    add_bullet(doc, "第一句。", "这次展示的是同一份记录的算法处理前后，不是治疗前后。", 10.2)
    add_bullet(doc, "第二句。", "我们已经把任务、时间、通道、异常和参数记录清楚，使后续受试者可以按同一套规则处理。", 10.2)
    add_bullet(doc, "第三句。", "目前得到的是质量改善和探索性关联，疗效、单穴经络因果和脑区机制还需要更严格的群体证据。", 10.2)

    doc.core_properties.title = "代表性受试者数据预处理说明"
    doc.core_properties.subject = "EEG fNIRS EMG IMU preprocessing and figure guide"
    doc.core_properties.author = ""
    doc.core_properties.keywords = "EEG; fNIRS; EMG; IMU; preprocessing"
    doc.save(DOCX)
    print(DOCX)


if __name__ == "__main__":
    main()
