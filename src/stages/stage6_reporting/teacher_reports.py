from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.section import WD_SECTION_START
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_LINE_SPACING
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_ROW_HEIGHT_RULE
import json
import zipfile
import re

ROOT = Path(r"D:\Python_File\EEF_analyze")
OUT = ROOT / "outputs" / "teacher_report_20260908_reviewed"
DOCX = OUT / "teacher_report_20260908_reviewed.docx"
ASSET = ROOT / "outputs" / "mbll_reviewed_20260908" / "teacher_assets"
OLD_FIG = ROOT / "outputs" / "teacher_report_20260904_visual" / "figures"

BLUE = "1F4E79"
TEAL = "2B7A8B"
GREEN = "548235"
LIGHT_BLUE = "D9EAF7"
LIGHT_GRAY = "F2F4F7"
BORDER = "D9D9D9"
TEXT = "222222"
MUTED = "5B6573"

def set_font(run, name="Microsoft YaHei", size=10.5, bold=False, color=TEXT, italic=False):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = RGBColor.from_string(color)

def set_para(paragraph, before=0, after=6, line=1.15, keep=False):
    fmt = paragraph.paragraph_format
    fmt.space_before = Pt(before)
    fmt.space_after = Pt(after)
    fmt.line_spacing = line
    if keep:
        fmt.keep_with_next = True

def set_cell_shading(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = tcPr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tcPr.append(shd)
    shd.set(qn("w:fill"), fill)

def set_cell_border(cell, color=BORDER, sz="6"):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    borders = tcPr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tcPr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = "w:" + edge
        el = borders.find(qn(tag))
        if el is None:
            el = OxmlElement(tag)
            borders.append(el)
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), sz)
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), color)

def set_cell_margins(cell, top=100, start=120, bottom=100, end=120):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcMar = tcPr.first_child_found_in("w:tcMar")
    if tcMar is None:
        tcMar = OxmlElement("w:tcMar")
        tcPr.append(tcMar)
    for m, v in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tcMar.find(qn("w:" + m))
        if node is None:
            node = OxmlElement("w:" + m)
            tcMar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")

def mark_header_row(row):
    trPr = row._tr.get_or_add_trPr()
    tblHeader = OxmlElement("w:tblHeader")
    tblHeader.set(qn("w:val"), "true")
    trPr.append(tblHeader)

def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    fldChar1 = OxmlElement("w:fldChar")
    fldChar1.set(qn("w:fldCharType"), "begin")
    instrText = OxmlElement("w:instrText")
    instrText.set(qn("xml:space"), "preserve")
    instrText.text = " PAGE "
    fldChar2 = OxmlElement("w:fldChar")
    fldChar2.set(qn("w:fldCharType"), "end")
    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)
    set_font(run, size=9, color=MUTED)

def remove_style_borders(style):
    pPr = style._element.get_or_add_pPr()
    pBdr = pPr.find(qn("w:pBdr"))
    if pBdr is not None:
        pPr.remove(pBdr)

def add_text(doc, text, style=None, before=0, after=6, line=1.15, bold_prefix=None):
    p = doc.add_paragraph(style=style)
    set_para(p, before=before, after=after, line=line)
    if bold_prefix and text.startswith(bold_prefix):
        r1 = p.add_run(bold_prefix)
        set_font(r1, size=10.5, bold=True)
        r2 = p.add_run(text[len(bold_prefix):])
        set_font(r2, size=10.5)
    else:
        r = p.add_run(text)
        set_font(r, size=10.5)
    return p

def add_bullet(doc, text, level=0):
    p = doc.add_paragraph(style="List Bullet" if level == 0 else "List Bullet 2")
    set_para(p, after=3, line=1.12)
    r = p.add_run(text)
    set_font(r, size=10.5)
    return p

def add_heading(doc, text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.paragraph_format.keep_with_next = True
    set_para(p, before=10 if level == 1 else 7, after=4, line=1.0, keep=True)
    r = p.add_run(text)
    set_font(r, size=15 if level == 1 else 12, bold=True, color="000000")
    return p

def add_table(doc, headers, rows, widths=None, font_size=9.2):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    hdr = table.rows[0]
    mark_header_row(hdr)
    for j, h in enumerate(headers):
        cell = hdr.cells[j]
        if widths:
            cell.width = Inches(widths[j])
        set_cell_shading(cell, BLUE)
        set_cell_border(cell)
        set_cell_margins(cell)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_para(p, after=0, line=1.0)
        r = p.add_run(str(h))
        set_font(r, size=font_size, bold=True, color="FFFFFF")
    for i, rowdata in enumerate(rows):
        cells = table.add_row().cells
        for j, val in enumerate(rowdata):
            cell = cells[j]
            if widths:
                cell.width = Inches(widths[j])
            set_cell_border(cell)
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if i % 2 == 1:
                set_cell_shading(cell, LIGHT_GRAY)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j >= 1 else WD_ALIGN_PARAGRAPH.CENTER
            set_para(p, after=0, line=1.08)
            r = p.add_run(str(val))
            set_font(r, size=font_size)
    set_para(doc.add_paragraph(), after=2)
    return table

def add_figure(doc, path, caption, width=6.4, alt=""):
    path = Path(path)
    if not path.exists():
        return False
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_para(p, before=4, after=2, line=1.0)
    run = p.add_run()
    inline = run.add_picture(str(path), width=Inches(width))
    if alt:
        inline._inline.docPr.set("descr", alt)
        inline._inline.docPr.set("title", alt)
    cp = doc.add_paragraph()
    cp.alignment = WD_ALIGN_PARAGRAPH.LEFT
    set_para(cp, before=2, after=8, line=1.08)
    rr = cp.add_run(caption)
    set_font(rr, size=9.1, color=MUTED)
    return True

def add_reference(doc, text):
    p = doc.add_paragraph(style="List Number")
    set_para(p, after=3, line=1.1)
    r = p.add_run(text)
    set_font(r, size=9.5)
    return p

def add_source_note(doc, text):
    p = doc.add_paragraph()
    set_para(p, before=0, after=0, line=1.0)
    r = p.add_run(text)
    set_font(r, size=7.6, color=MUTED, italic=True)
    return p

doc = Document()
sec = doc.sections[0]
sec.page_width = Inches(8.5)
sec.page_height = Inches(11)
sec.top_margin = Inches(0.68)
sec.bottom_margin = Inches(0.62)
sec.left_margin = Inches(0.72)
sec.right_margin = Inches(0.72)
sec.header_distance = Inches(0.3)
sec.footer_distance = Inches(0.3)

# Base styles
styles = doc.styles
for sname in ["Normal", "Body Text"]:
    st = styles[sname]
    st.font.name = "Microsoft YaHei"
    st._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    st.font.size = Pt(10.5)
    st.font.color.rgb = RGBColor.from_string(TEXT)
    st.paragraph_format.space_after = Pt(6)
    st.paragraph_format.line_spacing = 1.15
title_style = styles["Title"]
remove_style_borders(title_style)
title_style.font.color.rgb = RGBColor(0, 0, 0)
for level, size in [(1, 15), (2, 12)]:
    st = styles[f"Heading {level}"]
    st.font.name = "Microsoft YaHei"
    st._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    st.font.size = Pt(size)
    st.font.bold = True
    st.font.color.rgb = RGBColor(0, 0, 0)
    st.paragraph_format.space_before = Pt(10 if level == 1 else 7)
    st.paragraph_format.space_after = Pt(4)
    st.paragraph_format.keep_with_next = True
# Footer
footer_p = sec.footer.paragraphs[0]
add_page_number(footer_p)

# Cover
p = doc.add_paragraph(style="Title")
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
set_para(p, before=22, after=10, line=1.0)
r = p.add_run("固定组合干预背景下多模态脑肌响应阶段性报告")
set_font(r, size=22, bold=True, color="000000")

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
set_para(p, after=18, line=1.0)
r = p.add_run("面向老师的可审计分析说明")
set_font(r, size=13, color=MUTED)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
set_para(p, after=24, line=1.0)
r = p.add_run("2026 年 9 月 8 日")
set_font(r, size=10.5, color=MUTED)

add_heading(doc, "先看结论", 1)
add_text(doc, "当前数据最稳妥的交付是：在固定组合干预背景下，建立一套质量感知、缺失感知、可解释的离线 EEG-fNIRS-EMG 多模态表型流程，并对静息网络、运动任务、肌肉执行和眼动降级分支进行探索性描述。当前结果不能证明针刺疗效、单穴或经络特异性、神经血氧机制、闭环 BCI，也不能称为已验证临床标志物。", after=10)

add_table(doc, ["当前事实", "教师版口径"], [
    ["配对范围", "7 名受试者有 baseline 与 follow-up；纵向独立单位是受试者"],
    ["运动任务", "14 条配对运动记录；200 个完整固定动作窗；9 个不完整握拳扩展窗排除"],
    ["MBLL 结果", "默认 HbO 为后处理敏感性分支；摸腰方向较一致，但 8 个探索检验未多重校正"],
    ["眼动与 EEG", "14 条眼动均为 EYE_IR_DEGRADED_MASKED；14 条运动 EEG 的 EOG/ECG 未连接，未校正"],
    ["核心限制", "无短程通道；坐标声明为 cm 但实体帽距离未核实；认知与运动不是同一 trial"]
], widths=[1.35, 5.3], font_size=9.5)

add_text(doc, "报告中的所有趋势均为探索性。图中的点、线、事件和 block 只用于受试者内整理或受试者级汇总，不增加独立样本量。", after=6)
doc.add_page_break()

# 1
add_heading(doc, "执行摘要", 1)
add_text(doc, "这次整理把主线收敛为“固定组合干预背景—质量门控—多模态任务表型—谨慎验证”。fNIRS 的 OD proxy 与新增 MBLL 分支分别报告；OD 作为既有阶段基线，MBLL 用于检验参数假设下的敏感性。EMG 描述外周执行，EEG 在质量允许的范围内描述快速控制 proxy，眼动 fNIRS 单独保留为降级敏感性分支。")
add_bullet(doc, "MBLL 默认 HbO 结果中，摸头 n=7，均值 -0.02691 uM，中位数 +0.00874 uM，4 人上升、3 人下降；未校正 exact sign-flip p=0.609375。")
add_bullet(doc, "摸膝 n=6，均值 -0.01460 uM，中位数 +0.00277 uM，3 人上升、3 人下降；未校正 p=0.96875。")
add_bullet(doc, "摸腰 n=6，均值 -0.14411 uM，中位数 -0.11405 uM，0 人上升、6 人下降；未校正 p=0.03125。该结果不能直接称为显著疗效，因为共有 8 个探索检验且未作多重校正，同时存在 DPF 外推和无短程通道限制。")
add_bullet(doc, "摸头减摸膝的纵向 HbO 对比 n=6，exact sign-flip p=1.0；不能据此宣称头部动作优于膝部动作。")
add_bullet(doc, "OD proxy 与 MBLL 的纵向方向 raw OD 为 18/19，历史 z-OD 为 17/19；这是方向一致性统计，不是神经特异性验证。")
add_bullet(doc, "108/108 个任务乘敏感性设置的相对默认符号一致，说明方向在这些设置中没有改变，但幅度仍可能变化。")
add_text(doc, "因此，面向老师的结论应是：流程和证据边界已经固定，MBLL 为可追溯的敏感性增强；若现有已采集但尚未导入的数据最终达到条件，则按冻结流程增量处理配对样本，同时补齐元数据并做独立验证；不把当前探索性统计升级为治疗或临床结论。")

# 2
add_heading(doc, "数据规模与现有数据边界", 1)
add_text(doc, "当前可用于纵向比较的是 7 名同时拥有 baseline 和 follow-up 的配对受试者。运动任务共有 14 条配对记录。完整固定动作窗只包括摸头、摸膝和摸腰，共 200 个；原有动作之后出现的 9 个不完整或边界不确定握拳扩展窗不进入这组核心汇总。")
add_table(doc, ["数据层级", "当前范围", "如何解释"], [
    ["受试者", "7 名配对受试者", "纵向独立统计单位"],
    ["运动记录", "14 条 baseline/follow-up 记录", "记录覆盖，不等于 n=14 受试者"],
    ["完整固定动作窗", "200 个", "受试者内任务分段，不是独立样本"],
    ["握拳扩展", "9 个不完整/边界不确定窗排除", "保留为数据边界，不强行补齐"],
    ["S07 follow-up", "去重后 7 个 marker，5 个动作 marker，仅摸头", "摸膝和摸腰为协议事件缺失，因此相应 n=6"]
], widths=[1.55, 2.0, 3.1], font_size=9.2)
add_text(doc, "事件值 1 只表示同步或时间边界。运动任务名称来自外层目录和确认的协议顺序：摸头、摸膝、摸腰；新增段单独标为握拳。不能把 marker 值或第几个 marker 当作任务类型编码，也不能把缺失的边界补写成观测事实。")
add_text(doc, "认知任务和运动任务不是同一实时 trial。认知任务可用于控制/过程 proxy 的独立描述；它与运动 fNIRS/EMG 的关系只能写成跨模块关联或候选响应结构，不能串成因果链。")

# 3
add_heading(doc, "统一预处理与质量控制", 1)
add_figure(doc, OLD_FIG / "01_workflow.png",
           "图 1 统一分析流程。怎么看：从左到右看数据登记、任务与信号对齐、受试者级特征和探索性迁移。观察到什么：流程把质量门控放在多模态特征和纵向变化之前。说明什么：当前可交付的核心是可追溯的离线分析流程；若已有已采集但尚未导入的数据满足条件，可按冻结流程增量处理配对记录。不能说明什么：流程图不是临床预测器、治疗机制图或已经训练完成的人工智能模型；框内的受试者和特征均不增加独立样本量，也不表示新增硬件采集。", width=6.45, alt="统一目录登记、三层质量门控和受试者级多模态特征流程图")

add_heading(doc, "记录登记与三层 gate", 2)
add_bullet(doc, "外层目录决定受试者、访视、模态和任务语义；文件名只作审计信息。")
add_bullet(doc, "path gate 检查文件存在且可解析，signal gate 检查通道、窗口和任务质量，alignment gate 检查同 block 跨模态时间残差。")
add_bullet(doc, "缺失保留 NaN 和排除原因，不用 0 填补；事件、block、通道和网络边只在受试者内部作为整理单位。")
add_bullet(doc, "运动任务第一个 EEG/fNIRS marker 定义 EMG frame 0；marker disagreement 记录为审计字段，不能升级为硬件 TTL。")

add_heading(doc, "EEG", 2)
add_text(doc, "EEG 采用稳健参考、50 Hz 陷波、0.5 到 45 Hz 零相位带通和 250 Hz 重采样，并保留坏导联与有效窗 mask。EEG 只作质量门控的探索性控制 proxy。14 条运动 EEG 记录的 EOG/ECG 通道虽有通道名，但采集时未连接，不能做 EOG/ECG 校正；因此不能把“有通道名”写成“完成伪迹校正”。")

add_heading(doc, "fNIRS", 2)
add_text(doc, "fNIRS 主基线是光密度 OD proxy，包含通道 SCI/CV 质量控制、突发导数修复、0.01 到 0.2 Hz 滤波和运动/有效样本 mask。该频带处理只能衰减部分较高频生理成分和心搏频段，不能写成完成心脏干扰去除；在没有短程通道和生理参考时，也不能分离浅表系统噪声。MBLL HbO/HbR 是独立敏感性分支。无短程通道，因此不能声称完成 short-separation regression。眼动 fNIRS 受红外干扰并做过遮挡，统一保留为 EYE_IR_DEGRADED_MASKED；该分支不与运动数据混合，也不转换为行为正确率。")

add_heading(doc, "EMG 与 IMU", 2)
add_text(doc, "EMG 的跨记录比较使用逻辑肌肉索引，标准传感器顺序为 2、3、4、5、9、10、11；唯一明确的 shifted_v1 记录单独登记。编号修正只解决可追溯映射，不等于信号质量修复。处理包括 20 到 450 Hz、50 Hz 及谐波陷波和 1000 Hz 重采样。无 MVC 时不写 MVC 归一化。EMG 是肌肉激活执行 proxy，IMU 只作姿态变化或动态强度 proxy，不等于关节角度和完整运动学。")

# 4 MBLL
add_heading(doc, "新增 MBLL 方法与适用域", 1)
add_text(doc, "本节使用通过验收的匿名 MBLL 教师资产。MBLL 是由双波长光密度和几何参数得到 HbO/HbR 的后处理分支，不是直接新增的神经血氧测量。报告同时保留 OD proxy，以便区分“既有光密度表型”和“参数化血氧敏感性”。")
add_table(doc, ["参数维度", "报告中的处理", "必须保留的限制"], [
    ["波长", "使用 730/850 nm 双波长信息进入 MBLL 分支", "850 nm 及波长相关参数要结合适用域解释"],
    ["DPF", "明确记录默认值与敏感性条件", "大于 50 岁人群需 DPF 适用域警告；不把理论范围当固定 cutoff"],
    ["距离", "坐标声明单位为 cm，计算距离范围 4.50 到 5.78 cm", "实体帽的实际源探测器距离未核实；不能从理论距离推断脑区深度"],
    ["滤波", "按已记录条件比较方向稳定性和幅度变化", "敏感性结果不能替代独立验证"],
    ["短程通道", "未提供 short-separation 通道", "不能宣称完成短程回归或充分去除浅层生理混杂"],
    ["统计单位", "受试者级 baseline 到 follow-up 变化", "事件、动作窗、通道和敏感性行不是独立样本"]
], widths=[1.1, 3.1, 2.45], font_size=9.0)
add_text(doc, "对于大于 50 岁、850 nm 和非额区通道，教师版方法中保留 DPF 适用域警告。MBLL 数值可用于检验方向和幅度对参数的敏感程度，但不能凭后处理结果命名脑区、穴位或经络，也不能把 HbO 方向直接翻译成疗效。")

# 5 New result
add_heading(doc, "默认 MBLL 纵向结果", 1)
add_text(doc, "默认分支定义为 follow-up 减 baseline 的受试者级事件响应变化，单位为 uM。每一行对应一名受试者的配对结果汇总；S07 的 follow-up 没有摸膝和摸腰的协议事件，因此这两项 n=6 是事件缺失，而非把一行统计结果随机删除。")
add_table(doc, ["协议动作", "有效 n", "均值 uM", "中位数 uM", "方向", "未校正 exact sign flip"], [
    ["摸头", "7", "-0.02691", "+0.00874", "4 上升 / 3 下降", "p=0.609375"],
    ["摸膝", "6", "-0.01460", "+0.00277", "3 上升 / 3 下降", "p=0.96875"],
    ["摸腰", "6", "-0.14411", "-0.11405", "0 上升 / 6 下降", "p=0.03125"],
    ["摸头减摸膝", "6", "未报告均值", "未报告中位数", "任务对比", "p=1.0"]
], widths=[1.45, 0.7, 1.15, 1.25, 1.45, 1.5], font_size=8.8)
add_figure(doc, ASSET / "default_mbll_hbo_paired.png",
            "图 2 默认 MBLL HbO 受试者级配对图。怎么看：每个小图对应一个协议动作，横轴是 baseline 与 follow-up，纵轴是事件响应变化量 ΔHbO，单位为 uM；蓝点表示 baseline，红点表示 follow-up，线段连接同一匿名受试者。观察到什么：摸腰面板显示 6/6 个有效配对方向向下，摸头和摸膝的方向更混合。说明什么：在当前默认参数下，这是 6/6 同向下降的候选趋势，需扩大样本并预注册验证。不能说明什么：它不是疗效证明、不是神经特异性 HbO 证据，也不能说明某个动作或穴位产生因果变化；摸腰的 p=0.03125 未经过 8 个探索检验的多重校正。", width=6.45, alt="默认 MBLL HbO 三种运动动作的受试者级配对图")

add_text(doc, "统计解释必须同时看效应方向、有效 n、个体点和参数条件。摸腰的单项未校正 sign-flip p 值不能单独作为显著性结论；在本参数和本配对样本中，只能写为“6/6 同向下降的候选趋势，需扩大样本并预注册验证”。在没有对照、没有短程通道、DPF 适用域受限以及纵向间隔存在的情况下，不能把这一候选趋势解释为疗效、已验证标志物或因果变化。")

# 6 consistency
add_heading(doc, "OD proxy 与 MBLL 一致性", 1)
add_text(doc, "OD proxy 与 MBLL 不同量纲、不同解释层级。当前只比较预先定义的受试者级纵向方向，不把方向相同视为神经特异性。raw OD 的方向一致为 18/19，历史 z-OD 为 17/19；分母 19 是任务/条件层面的有效比较行，不能当 19 名受试者。")
add_figure(doc, ASSET / "od_mbll_longitudinal_direction_agreement.png",
           "图 3 OD proxy 与 MBLL 纵向方向一致性汇总。怎么看：横轴是两种 OD 表示，纵轴是与 MBLL 方向相同的比较行比例；柱顶标注为一致行数/总行数。观察到什么：raw OD 为 18/19，历史 z-OD 为 17/19。说明什么：在这些已定义的任务和配对条件下，方向层面大体相近，可支持继续复核。不能说明什么：这不是 HbO/HbR 的独立验证，不证明神经特异性，不证明疗效，也不能掩盖每一行的有效 n、幅度差异和缺失原因。", width=6.25, alt="OD proxy 与 MBLL 纵向方向一致性柱状图")
add_figure(doc, ASSET / "sensitivity_sign_stability.png",
           "图 4 MBLL 敏感性符号与幅度稳定性。怎么看：左半图横轴是摸头、摸膝和摸腰，纵轴是相对默认设置保持同方向的比例；右半图横轴仍是三个协议动作，纵轴是各设置相对默认结果的幅度比，虚线 1.0 表示与默认幅度相同。观察到什么：108/108 个任务乘敏感性设置保持相同符号，但幅度比大约分布在 0.77 到 1.60。说明什么：在当前参数网格内方向没有翻转，方向对设置相对稳健；同时幅度仍会变化。不能说明什么：方向稳定不等于幅度稳定，参数也没有因此成为唯一正确选择；这仍是后处理敏感性，不是外部验证或疗效证据。", width=6.25, alt="三个运动任务在多种 MBLL 敏感性设置下的方向和幅度稳定性")

# 7 older trends
add_heading(doc, "既有阶段性探索趋势", 1)
add_text(doc, "以下图表来自已经核实的旧阶段 OD/状态/EMG 探索结果，仅用于说明分析结构和候选线索。它们不替代本次 MBLL 教师资产，也不应与 MBLL 数值混写。")
add_figure(doc, OLD_FIG / "03_paired_trends.png",
             "图 5 旧阶段多面板受试者级探索图。怎么看：四个面板的横轴都表示 follow-up 减 baseline 的变化（Δ=sec_test−base_line），纵轴为匿名受试者；每条水平线连接同一受试者的配对值，虚线为面板内的平均变化，单位随指标而异。左上是摸头−摸膝任务对比的 OD proxy，右上是状态占用熵，左下是摸头 EMG 包络 CV，右下是脑−肌绝对相关 |rho|（右下有效 n=6，其余面板各自按图中 n 解释）。观察到什么：左上和右上方向存在个体差异，左下和右下也不是所有受试者同向。说明什么：这些面板展示了旧阶段如何把 OD、动态状态、EMG 执行和脑−肌关系整理为候选表型。不能说明什么：它们不是 MBLL 结果、不是疗效或因果证据；状态标签没有外部语义，EMG/脑−肌指标也不能替代行为或完整运动学；认知任务与运动任务不是同一 trial。", width=5.7, alt="旧阶段OD、状态熵、EMG和脑肌指标的四面板受试者级探索图")
add_figure(doc, OLD_FIG / "04_assoc_entropy_task.png",
           "图 6 旧阶段状态结构与任务响应的探索性关系。怎么看：横轴和纵轴都是受试者级 baseline 到 follow-up 变化，点旁为匿名 ID。观察到什么：不同受试者的方向并不完全一致，留一法范围与拟合方向只能作为敏感性诊断。说明什么：动态状态特征可以进入候选表型池，等待更大样本和预注册定义。不能说明什么：PCA 和 k-means 状态没有外部语义，不能命名心理、经络或治疗状态，也不能据此提出因果路径。", width=5.65, alt="旧阶段动态状态与任务响应关联散点图")
add_figure(doc, OLD_FIG / "05_assoc_coupling_emg.png",
           "图 7 旧阶段脑肌 coupling 与 EMG 共激活的探索性关系。怎么看：横轴是脑肌相关指标的纵向变化，纵轴是 EMG 全局共激活的纵向变化，点为匿名受试者。观察到什么：个体点分布分散，描述性拟合方向受有效配对和同步质量影响。说明什么：在共同有效 block 和同步 gate 通过时，fNIRS/EMG 可以提供跨模态执行线索。不能说明什么：这不是脑肌因果、不是疗效、不是完整运动学；EMG 是肌肉执行 proxy，不能替代关节角度和轨迹。", width=5.65, alt="旧阶段脑肌相关指标与 EMG 共激活变化散点图")

# 8 Eye and EEG
add_heading(doc, "眼动降级分支与 EEG 质量边界", 1)
add_text(doc, "14 条眼动记录均保留为 EYE_IR_DEGRADED_MASKED。红外干扰和遮挡导致部分样本被 mask，因此该分支只用于质量覆盖和 OD proxy 的敏感性描述。它不能恢复视线轨迹、目标选择、行为正确率或精确眼动任务标签，也不与运动 fNIRS/EMG 混合。")
add_text(doc, "14 条运动 EEG 记录中 EOG/ECG 采集时未连接，因此没有 EOG/ECG 校正。现有导出中的相关命名通道只能在字段语义和记录含义得到澄清后作为审计线索，不能代替眼动仪行为输出，也不通过补做接线来改变既有数据。")
add_table(doc, ["分支", "当前可做", "当前不可做"], [
    ["眼动 fNIRS", "独立报告 EYE_IR_DEGRADED_MASKED 的有效样本覆盖和 OD proxy", "不能恢复 gaze、目标选择、正确率，不能与运动数据混合"],
    ["运动 EEG", "质量门控后作快速控制 proxy 敏感性分析", "不能声称完成 EOG/ECG 校正；不完整访视不接入主脑肌节点"],
    ["事件 marker", "按观察时间、相邻间隔和协议顺序分段", "不能用 marker 1 或事件编号命名任务"],
    ["IMU", "描述 roll/pitch 和动态加速度/角速度强度 proxy", "不能给出绝对 yaw、关节角度或完整运动学"]
], widths=[1.15, 3.0, 2.7], font_size=9.0)

# 9 teacher 3 points
add_heading(doc, "逐条回应老师三点", 1)
add_heading(doc, "认知与运动如何统一", 2)
add_text(doc, "建议采用“认知控制—任务相关网络—外周执行”的分层表型语言。认知任务提供目标、选择和抑制等控制过程的理论背景；EEG 是时间分辨率较高的控制 proxy；fNIRS 是较慢时间尺度的任务/网络 proxy；EMG 是动作执行 proxy。由于认知任务和运动任务不是同一实时 trial，当前只写跨模块关联或候选响应结构，不写认知状态驱动运动变化的因果链。事件值 1 只服务于同步和时间边界，不用于任务命名。")
add_heading(doc, "穴位经络与脑网络如何连接", 2)
add_text(doc, "固定组合干预可以作为研究背景和先验假设，当前可检验的链条是“固定干预背景—静息网络—运动任务空间—肌肉执行”。但是现有材料缺少单穴对照、sham、逐次刺激日志、刺激参数分层和充分的空间定位；因此不能把结果归因于某个穴位、某条经络或某个脑区，也不能把探头中点直接命名为穴位或经络。")
add_heading(doc, "AI 的实际贡献是什么", 2)
add_text(doc, "AI 相关内容必须分三层说明。第一层是工程自动化：目录登记、哈希追溯、质量 mask、缺失记录和跨模态 gate。第二层是统计推断：受试者级配对变化、exact sign-flip/排列诊断、留一受试者范围、self-excluded attribution 和质量加权的透明图。第三层是真正学习模型：当前没有训练完成的 CNN、GIN、FGANet、CNN-LSTM、临床预测器或 SHAP 因果解释。当前可以称为透明的状态/融合探索，不应把质控、排序和排列检验包装成 AI 创新。")

# 10 mainline
add_heading(doc, "论文主线与两条路线", 1)
add_text(doc, "建议论文主线写为：固定组合干预背景下，质量感知、缺失感知和可解释的离线 EEG-fNIRS-EMG 多模态表型。主结果以运动任务 fNIRS 表型和 OD/MBLL 敏感性为中心，静息网络、EMG 执行和脑肌 bridge 作为探索性扩展；眼动 fNIRS 和 EEG 质量分支单独报告。")
add_heading(doc, "保守路线", 2)
add_bullet(doc, "固定主要任务对比、窗口、DPF 记录方式、质量阈值和受试者级统计，不因新结果方向改变而事后改定义。")
add_bullet(doc, "若现有已采集但尚未导入的数据最终达到更多配对受试者，按冻结流程增量处理，并报告每人的原始变化、有效 n、排除原因和敏感性范围；不新增硬件采集。")
add_bullet(doc, "样本扩大后再考虑 elastic-net、pyRiemann 等强基线，并采用 subject-level LOSO 或跨访视验证。")
add_bullet(doc, "EOG/ECG 已确认采集时未接线；仅向厂家/采集人员补问这些命名通道的导出/字段语义、参考/GND 和阻抗记录含义（如可获得），同时补问源探测器实际距离、短程通道和同步 TTL。上述是元数据澄清，不是补做实验。")
add_heading(doc, "创新路线", 2)
add_bullet(doc, "在固定节点/边定义上比较规则图、层级模型、pyRiemann 和时序图网络，并用模态消融检验增量价值。")
add_bullet(doc, "把认知、任务和执行组织成多层图或状态空间模型，报告稳定性和个体差异，不把路径写成因果方向。")
add_bullet(doc, "只有在更大样本和独立验证中稳定的特征，才考虑升级为候选标志物；当前透明图是可审计组织框架，不是深度模型。")
add_text(doc, "路线图中的‘20–30 配对受试者’是条件式门槛：仅当现有已采集但尚未导入的数据最终达到该规模时，才按冻结流程进入后续验证；不表示新增受试者采集。")
add_figure(doc, OLD_FIG / "07_roadmap.png",
             "图 8 保守路线与创新路线。怎么看：左侧是当前阶段的固定任务、持续质量门控和受试者级验证，右侧是条件式的预注册、LOSO 和模型比较；图中文字的‘20–30 配对受试者’仅在现有已采集但尚未导入数据最终达到该规模时成立。观察到什么：创新升级依赖独立验证和更大的配对样本。说明什么：当前最重要的交付是把主要终点和质量契约固定下来，并在已有数据范围内补问元数据。不能说明什么：右侧计划不代表已经完成的训练或验证，也不表示新增硬件采集，更不代表当前已有临床预测能力。", width=6.35, alt="当前保守路线和条件式创新升级路线图")

# 11 limits and questions
add_heading(doc, "不可宣称边界", 1)
for item in [
    "无 sham 或其他对照，不能写针刺相对于自然变化的疗效。",
    "无单穴/经络分层和逐次刺激日志，不能写单穴、经络或脑区特异性。",
    "认知任务与运动任务不是同一 trial，不能写实时因果串联。",
    "EOG/ECG 未连接，不能写完成 EOG/ECG 校正。",
    "无短程通道，不能写完成 short-separation regression。",
    "MBLL 是后处理敏感性分支，不能写成直接神经血氧或唯一正确参数。",
    "EMG 和 IMU 是执行/运动强度 proxy，不能替代完整运动学。",
    "当前样本量和质量覆盖不能支持临床预测、疗效分层或已验证生物标志物。"
]:
    add_bullet(doc, item)

add_heading(doc, "可向厂家和实验人员补问的元数据", 1)
for item in [
      "EOG/ECG 已确认未接线；仅补问这些命名通道的导出/字段语义、参考/GND、阻抗记录含义（如可获得）以及导出设置。",
    "fNIRS 的设备/软件版本、真实 source-detector 距离、坐标单位、增益、采样率、短程通道和遮挡时间段。",
    "MBLL 所用 DPF、消光系数、几何单位、年龄/人群适用域及其文献出处。",
    "EEG、fNIRS 和 EMG 的硬件 TTL、时钟偏移、marker 发送路径、EMG frame 0 定义和重复事件去重规则。",
    "EMG 传感器实际贴附位置、左右侧、激活确认、增益、单位、MVC、采样率和 IMU 轴向/磁校准。",
    "眼动原始轨迹、校准日志、红外干扰/遮挡日志、任务呈现日志和 trial-level 行为输出。",
    "固定组合干预的逐次刺激日志、穴位/经络分层、刺激参数、治疗间隔和对照条件。"
]:
    add_bullet(doc, item)

# 12 refs
add_heading(doc, "参考文献", 1)
refs = [
    "Yücel M A et al. Best practices for fNIRS publications. Neurophotonics. 2021;8(1):012101. DOI: 10.1117/1.nph.8.1.012101.",
    "Pernet C R et al. COBIDAS-MEEG: Best practices in data analysis and sharing in neuroimaging using MEEG. Nature Neuroscience. 2020. DOI: 10.1038/s41593-020-00709-0.",
      "Prahl S. Tabulated Molar Extinction Coefficient for Hemoglobin in Water. OMLC. 730 nm: HbO 390, HbR 1102.2; 850 nm: HbO 1058, HbR 691.32. https://omlc.org/spectra/hemoglobin/summary.html.",
      "Scholkmann F, Wolf M. General equation for the differential pathlength factor of the frontal human head depending on wavelength and age. Journal of Biomedical Optics. 2013;18(10):105004. DOI: 10.1117/1.JBO.18.10.105004.",
      "Jiang Y-C et al. Characterization of Bimanual Cyclical Tasks From Single-Trial EEG-fNIRS Measurements. IEEE Transactions on Neural Systems and Rehabilitation Engineering. 2022;30:146-156. DOI: 10.1109/TNSRE.2022.3144216.",
    "The interaction between changes of muscle activation and cortical network dynamics during isometric elbow contraction. Frontiers in Bioengineering and Biotechnology. 2023. DOI: 10.3389/fbioe.2023.1176054.",
    "Beste C et al. Binding and Retrieval in Action Control. Communications Biology. 2023. DOI: 10.1038/s42003-023-04531-9.",
    "Vabalas A et al. Machine learning algorithm validation with a limited sample size. PLOS ONE. 2019;14(11):e0224365. DOI: 10.1371/journal.pone.0224365."
]
for ref in refs:
    add_reference(doc, ref)

add_heading(doc, "图注阅读提示", 1)
add_text(doc, "教师版每张图都按五步写图注：先说明图看什么，再说明横纵轴、点线颜色和单位，随后说明观察到的方向和有效 n，再给出探索性意义，最后明确不能说明疗效、因果、神经特异性、任务行为恢复或标志物验证。")
# core properties privacy
doc.core_properties.author = "Research team"
doc.core_properties.last_modified_by = "Research team"
doc.core_properties.title = "固定组合干预背景下多模态脑肌响应阶段性报告"
doc.core_properties.subject = "教师版阶段性分析报告"
doc.core_properties.keywords = "EEG fNIRS EMG MBLL quality control exploratory"
doc.core_properties.comments = ""

OUT.mkdir(parents=True, exist_ok=True)
doc.save(DOCX)
print(str(DOCX))
