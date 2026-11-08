from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_LINE_SPACING
from pathlib import Path
import json


ROOT = Path(r"D:\Python_File\EEF_analyze")
OUT = ROOT / "outputs" / "teacher_report_20260908_plain"
OUT.mkdir(parents=True, exist_ok=True)
DOCX = ROOT / "plain_tmp" / "teacher_report_plain_20260908.docx"
DOCX.parent.mkdir(parents=True, exist_ok=True)
ASSET = ROOT / "outputs" / "mbll_reviewed_20260908" / "teacher_assets"
OLD = ROOT / "outputs" / "analysis" / "stage_20260904_eye_fnirs_imu_verified"


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn('w:shd'))
    if shd is None:
        shd = OxmlElement('w:shd')
        tc_pr.append(shd)
    shd.set(qn('w:fill'), fill)


def set_cell_border(cell, color='D9D9D9', size='6'):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    borders = tcPr.first_child_found_in('w:tcBorders')
    if borders is None:
        borders = OxmlElement('w:tcBorders')
        tcPr.append(borders)
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        tag = 'w:' + edge
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn('w:val'), 'single')
        element.set(qn('w:sz'), size)
        element.set(qn('w:space'), '0')
        element.set(qn('w:color'), color)


def set_cell_margins(cell, top=100, start=120, bottom=100, end=120):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcMar = tcPr.first_child_found_in('w:tcMar')
    if tcMar is None:
        tcMar = OxmlElement('w:tcMar')
        tcPr.append(tcMar)
    for m, v in [('top', top), ('start', start), ('bottom', bottom), ('end', end)]:
        node = tcMar.find(qn('w:' + m))
        if node is None:
            node = OxmlElement('w:' + m)
            tcMar.append(node)
        node.set(qn('w:w'), str(v))
        node.set(qn('w:type'), 'dxa')


def set_repeat_table_header(row):
    trPr = row._tr.get_or_add_trPr()
    tblHeader = OxmlElement('w:tblHeader')
    tblHeader.set(qn('w:val'), 'true')
    trPr.append(tblHeader)


def set_keep_with_next(paragraph, value=True):
    pPr = paragraph._p.get_or_add_pPr()
    k = pPr.find(qn('w:keepNext'))
    if value and k is None:
        pPr.append(OxmlElement('w:keepNext'))
    elif not value and k is not None:
        pPr.remove(k)


def set_font(run, name='Microsoft YaHei', size=10.8, bold=False, color='222222', italic=False):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn('w:ascii'), name)
    run._element.get_or_add_rPr().rFonts.set(qn('w:hAnsi'), name)
    run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), name)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = RGBColor.from_string(color)


def style_paragraph(p, before=0, after=6, line=1.15, align=None):
    pf = p.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = line
    if align is not None:
        p.alignment = align


def add_para(doc, text='', style=None, before=0, after=6, line=1.15, align=None, size=10.8, bold=False, color='222222', italic=False):
    p = doc.add_paragraph(style=style) if style else doc.add_paragraph()
    style_paragraph(p, before, after, line, align)
    if text:
        r = p.add_run(text)
        set_font(r, size=size, bold=bold, color=color, italic=italic)
    return p


def add_rich_para(doc, segments, before=0, after=6, line=1.15, align=None):
    p = doc.add_paragraph()
    style_paragraph(p, before, after, line, align)
    for text, kwargs in segments:
        r = p.add_run(text)
        set_font(r, **kwargs)
    return p


def add_heading(doc, text, level=1, page_break=False):
    if page_break:
        doc.add_page_break()
    p = doc.add_paragraph(style=f'Heading {level}')
    style_paragraph(p, before=4 if level == 1 else 2, after=8 if level == 1 else 4, line=1.05)
    r = p.add_run(text)
    set_font(r, size=16 if level == 1 else 12.5, bold=True, color='000000')
    set_keep_with_next(p, True)
    return p


def add_bullet(doc, text, level=0, after=3):
    p = doc.add_paragraph(style='List Bullet' if level == 0 else 'List Bullet 2')
    style_paragraph(p, before=0, after=after, line=1.1)
    r = p.add_run(text)
    set_font(r, size=10.6)
    return p


def add_number(doc, label, text, after=5):
    p = doc.add_paragraph()
    style_paragraph(p, before=0, after=after, line=1.12)
    r = p.add_run(label + ' ')
    set_font(r, size=10.8, bold=True, color='1F4E79')
    r = p.add_run(text)
    set_font(r, size=10.8)
    return p


def add_table(doc, headers, rows, widths=None, font_size=9.6, header_fill='1F4E79'):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    hdr = table.rows[0]
    set_repeat_table_header(hdr)
    for j, h in enumerate(headers):
        c = hdr.cells[j]
        if widths:
            c.width = Inches(widths[j])
        set_cell_shading(c, header_fill)
        set_cell_border(c)
        set_cell_margins(c)
        c.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        p = c.paragraphs[0]
        style_paragraph(p, after=0, line=1.0, align=WD_ALIGN_PARAGRAPH.CENTER)
        r = p.add_run(str(h))
        set_font(r, size=font_size, bold=True, color='FFFFFF')
    for i, row in enumerate(rows):
        cells = table.add_row().cells
        for j, val in enumerate(row):
            c = cells[j]
            if widths:
                c.width = Inches(widths[j])
            set_cell_border(c)
            set_cell_margins(c, top=100, start=120, bottom=100, end=120)
            c.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if i % 2 == 1:
                set_cell_shading(c, 'F3F6F9')
            p = c.paragraphs[0]
            style_paragraph(p, after=0, line=1.08, align=WD_ALIGN_PARAGRAPH.LEFT)
            r = p.add_run(str(val))
            set_font(r, size=font_size)
    add_para(doc, '', after=2)
    return table


def add_image(doc, path, width=6.7, caption=None):
    if not Path(path).exists():
        add_para(doc, f'图表文件暂未找到：{Path(path).name}', italic=True, color='7F6000')
        return
    p = doc.add_paragraph()
    style_paragraph(p, before=2, after=3, line=1.0, align=WD_ALIGN_PARAGRAPH.CENTER)
    p.add_run().add_picture(str(path), width=Inches(width))
    if caption:
        # Keep the figure and its explanatory caption on the same page. This
        # prevents a caption from becoming detached at the top of the next
        # page when Word repaginates the document.
        set_keep_with_next(p, True)
        p2 = add_para(doc, caption, after=8, line=1.08, align=WD_ALIGN_PARAGRAPH.LEFT, size=9.5, color='555555', italic=True)
        set_keep_with_next(p2, False)


def add_page_number(paragraph):
    run = paragraph.add_run()
    fldChar1 = OxmlElement('w:fldChar')
    fldChar1.set(qn('w:fldCharType'), 'begin')
    instrText = OxmlElement('w:instrText')
    instrText.set(qn('xml:space'), 'preserve')
    instrText.text = ' PAGE '
    fldChar2 = OxmlElement('w:fldChar')
    fldChar2.set(qn('w:fldCharType'), 'end')
    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)
    set_font(run, size=9, color='777777')


def setup_doc():
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Inches(0.62)
    sec.bottom_margin = Inches(0.58)
    sec.left_margin = Inches(0.72)
    sec.right_margin = Inches(0.72)
    sec.header_distance = Inches(0.25)
    sec.footer_distance = Inches(0.28)
    styles = doc.styles
    normal = styles['Normal']
    normal.font.name = 'Microsoft YaHei'
    normal._element.rPr.rFonts.set(qn('w:eastAsia'), 'Microsoft YaHei')
    normal.font.size = Pt(10.8)
    for name, size in [('Title', 24), ('Heading 1', 16), ('Heading 2', 12.5)]:
        st = styles[name]
        st.font.name = 'Microsoft YaHei'
        st._element.rPr.rFonts.set(qn('w:eastAsia'), 'Microsoft YaHei')
        st.font.size = Pt(size)
        st.font.color.rgb = RGBColor(0, 0, 0)
        st.font.bold = True
    # Word's built-in Title style can carry a blue paragraph border. The
    # document skill requires a clean title separated by whitespace only.
    title_ppr = styles['Title']._element.pPr
    if title_ppr is not None:
        title_border = title_ppr.find(qn('w:pBdr'))
        if title_border is not None:
            title_ppr.remove(title_border)
    footer = sec.footer
    p = footer.paragraphs[0]
    style_paragraph(p, before=0, after=0, line=1.0, align=WD_ALIGN_PARAGRAPH.RIGHT)
    r = p.add_run('阶段性汇报  |  ')
    set_font(r, size=9, color='777777')
    add_page_number(p)
    return doc


def build():
    doc = setup_doc()

    # Cover and conclusion
    p = doc.add_paragraph(style='Title')
    style_paragraph(p, before=20, after=12, line=1.0, align=WD_ALIGN_PARAGRAPH.LEFT)
    r = p.add_run('EEG fNIRS EMG 数据目前做到了什么')
    set_font(r, size=23, bold=True, color='000000')
    # Explicitly suppress inherited borders from Word's built-in Title style.
    # Some Word installations render a blue line even after the style border
    # is removed, so set every paragraph border edge to nil on this paragraph.
    title_ppr = p._p.get_or_add_pPr()
    old_pbdr = title_ppr.find(qn('w:pBdr'))
    if old_pbdr is not None:
        title_ppr.remove(old_pbdr)
    title_pbdr = OxmlElement('w:pBdr')
    for edge in ('top', 'left', 'bottom', 'right', 'between', 'bar'):
        edge_el = OxmlElement(f'w:{edge}')
        edge_el.set(qn('w:val'), 'nil')
        title_pbdr.append(edge_el)
    title_ppr.append(title_pbdr)
    add_para(doc, '面向老师的阶段性汇报', after=18, size=12.5, color='555555')
    add_para(doc, '先说结论', style='Heading 1', before=4, after=8)
    add_para(doc, '我们现在已经把一套容易出错的数据，整理成了一套可以重复执行的处理方法。最重要的改变不是“得到了一个很漂亮的结果”，而是先把每一步做了什么、为什么保留或排除、结果能解释到哪一步，全部说清楚。', after=8, size=11.2)
    add_para(doc, '当前最稳妥的论文主线是：在固定治疗方案前后，分别观察认知侧和运动侧有没有变化，最后只在同一名受试者层面检查两边是否同步。这个主线可以做成一篇有数据基础的探索性论文，但不能写成某个穴位已经被证明通过某个脑区改善了运动。', after=10, size=11.2)
    add_para(doc, '当前数据有 7 名同时有治疗前和复测数据的受试者，共 14 条运动近红外记录。我们把能确认完整的动作留下，把边界不完整的 9 个握拳动作窗排除。', after=12, size=11.2)
    add_table(doc, ['现在已经确认', '现在看到的线索', '现在还不能说'], [
        ['任务按外层文件夹判断；EEG、fNIRS、EMG 的时间关系已经统一；EMG 传感器对应关系已纠正；质量检查和排除规则已经固定。', '光学密度（OD，表示光被吸收多少的相对指标）与 MBLL（把两种波长的光强变化换算成相对 HbO/HbR 的方法）对受试者复测前后的方向大多一致；摸腰任务的 HbO 方向在 6 名受试者中全部下降；部分脑与肌肉指标有一起变化的线索。', '不能说已经证明疗效、穴位特异性、经络特异性或脑区因果关系；不能称为已验证标志物或闭环系统。'],
    ], widths=[2.15, 2.15, 2.15], font_size=9.7)
    add_para(doc, '给老师汇报时可以把这项工作概括成一句话：我们先把数据质量和处理逻辑理顺，再用同一条流程去看脑和肌肉是否朝同一个方向变化。现在得到的是可以继续扩大样本验证的线索。', after=0, size=11.1, bold=True, color='1F4E79')

    # Page 2
    add_heading(doc, '一 我们到底做了什么', level=1, page_break=True)
    add_para(doc, '这部分只回答一个问题：原始数据拿到手以后，我们具体做了哪些工作。', after=8, size=11)
    add_table(doc, ['工作', '我们实际做的事', '这样做的原因'], [
        ['数据整理', '把本阶段要用的受试者数据复制到工程目录，删除 data 中确定用不到的文件；保留旧数据和旧结果，不覆盖。', '后续同一套流程可以直接运行，避免移动硬盘路径变化导致结果不可复现。'],
        ['任务判断', '以外层文件夹代表的任务为准，不再根据文件名里出现的“静息态”等字样判断任务。', '文件名曾经出现过误导，外层目录才是当前整理后采用的任务依据。'],
        ['时间同步', 'fNIRS 和 EEG 的第一个标记作为 EMG 的 0 帧起点，再用 fNIRS/EEG 标记推算 EMG 的动作边界；发现不同步时记录下来，不把缺失边界补成真实动作。', '三种信号的时间起点不同，必须先对齐，才能比较同一次动作。'],
        ['EMG 通道', '使用 2、3、4、5、9、10、11 号传感器。依次对应斜方肌上部、三角肌前束、三角肌中束、肱二头肌、肱三头肌、肱桡肌、尺侧腕屈肌。', '原来的编号整体向前错了一位；1 号是未配置通道，11 号不能漏掉。'],
        ['EEG', '保留头皮脑电，做参考、滤波和质量检查；没有用名义上的 EOG、ECG 做校正。', '这两个通道虽然出现在文件名里，但采集时没有接传感器，不能当作真实眼电或心电参考。'],
        ['fNIRS', '先检查光强、坏通道和突变，再计算光学密度（OD，表示光被吸收多少的相对指标）；同时增加 MBLL（把两种波长的光强变化换算成相对 HbO/HbR 的方法）。', 'OD 适合看相对变化，MBLL 可以进一步区分 HbO 和 HbR，但在当前数据上只能作相对结果。'],
        ['EMG 和 IMU', 'EMG 看肌肉激活强弱和波动；IMU（惯性测量单元）看身体或肢体姿态变化和运动强度的粗略指标。', 'EMG 不是完整运动学；IMU 也不足以还原关节角度和完整运动轨迹。'],
        ['眼动数据', '14 条眼动相关 fNIRS 记录单独保留，考虑已有遮挡处理，但同时标记红外眼动仪可能带来的干扰。', '眼动数据有价值，但不能和普通运动 fNIRS 混在一起，也不能说红外干扰已经被完全去除。'],
    ], widths=[1.0, 3.25, 2.2], font_size=8.8)
    add_para(doc, '处理流程最后形成四个结果层次：一是数据和通道能不能用，二是脑和肌肉的单项变化，三是脑和肌肉是否一起变化，四是哪些变化值得在更多受试者中验证。', after=0, size=10.8, bold=True, color='1F4E79')

    # Page 3 data and results
    add_heading(doc, '二 目前数据告诉了我们什么', level=1, page_break=True)
    add_heading(doc, '先看已经确认的数据事实', level=2)
    add_table(doc, ['项目', '目前确认的数量或情况', '怎么理解'], [
        ['配对受试者', '7 名；都有治疗前和复测记录。', '受试者数量仍然很少，统计结果只能作为探索。'],
        ['运动近红外记录', '14 条。', '7 名受试者各有一次治疗前和一次复测。'],
        ['完整动作窗', '200 个。', '只保留能清楚确认摸头、摸膝、摸腰边界的动作。'],
        ['排除动作窗', '9 个不完整或边界不确定的握拳窗。', '没有把不确定的时间段硬算成有效动作。'],
        ['受试者 S07', '复测中没有摸膝和摸腰的有效标记。', '所以摸膝和摸腰的配对比较只有 6 名，不是随意删掉一行。'],
        ['fNIRS 通道距离', '文件声明单位为 cm，源探距离约 4.50 到 5.78 cm。', '没有短程通道，不能用短程回归专门去掉头皮和血管的系统波动。'],
        ['EOG 和 ECG', '文件中有通道名称，但采集时实际未接线。', '不能用作眼电回归、心电回归或 ICA 识别参考。'],
    ], widths=[1.25, 2.45, 2.75], font_size=9.0)
    add_heading(doc, '再把结果分成三类', level=2)
    add_table(doc, ['类别', '目前的说法'], [
        ['已经做实的', '数据整理规则、任务判断规则、同步方法、EMG 通道对应关系、质量排除原则，以及 OD 和 MBLL 两种计算方式都已经明确。'],
        ['看到的趋势', 'OD 与 MBLL 的纵向方向分别有 18/19 和 17/19 的一致；108/108 个参数组合方向一致；摸腰 HbO 在 6/6 名受试者中下降，未校正的精确检验 p=0.03125；部分脑与肌肉指标有同向变化线索。'],
        ['不能下的结论', '不能说治疗已经被证明有效，不能说某个穴位或经络特异性有效，不能说某个脑区导致了动作改善，不能称为已经验证的标志物，也不能称为闭环 BCI。'],
    ], widths=[1.25, 5.2], font_size=9.3)
    add_para(doc, '特别提醒：摸腰 HbO 的 p=0.03125 是未做多重比较校正的探索结果。当前一共看了 8 项类似的探索检验，因此它只能作为“优先复核的候选趋势”，不能单独当作最终结论。', after=0, size=10.8, bold=True, color='7F6000')

    # Page 4 MBLL
    add_heading(doc, '三 我们为什么又做了 MBLL', level=1, page_break=True)
    add_para(doc, '近红外设备原始记录的是 730 nm 和 850 nm 两种光的强度。我们先计算光学密度（OD，表示光被吸收多少的相对指标）。它适合先看前后变化，但不能直接告诉我们是 HbO 还是 HbR 在变。', after=8)
    add_para(doc, 'MBLL 是“修正的 Beer Lambert 定律”的简称，也就是把两种波长的光强变化，结合光源到探测器的距离、组织中的光程和血红蛋白的吸光能力，换算成相对 HbO 和 HbR 变化的方法。这里得到的单位是相对变化，不是抽血意义上的绝对血红蛋白浓度。', after=8)
    add_table(doc, ['计算中用到的东西', '本次采用的内容', '需要注意的地方'], [
        ['两种波长', '730 nm 和 850 nm。', '按设备文件中的波长计算。'],
        ['消光系数', '采用 OMLC 公开表格的 HbO 和 HbR 数据，并按自然对数的光密度转换。', '来源可追溯，但不同表格的对数约定不能混用。'],
        ['年龄相关 DPF', '采用 Scholkmann 和 Wolf 2013 年提出的年龄相关差分路径因子（DPF，表示光在组织中实际走过的路比直线距离更长的倍数）公式。', '公式原本主要基于 0 到 50 岁、690 到 832 nm、额部或额颞部资料；850 nm、高龄和非额区属于外推使用。'],
        ['源探距离', '直接使用文件中声明为 cm 的坐标计算，约 4.50 到 5.78 cm。', '还需向厂家确认这些坐标是帽子上的真实物理距离，还是模板坐标。'],
        ['短程通道', '没有发现 1.5 cm 左右的短程通道。', '所以不能从数据中单独估计并回归掉浅层血流和头皮噪声。'],
    ], widths=[1.35, 3.35, 2.0], font_size=8.9)
    add_para(doc, '因此，MBLL 在本项目中的定位是：让相对变化多一个 HbO/HbR 的解释角度，帮助检查 OD 看到的方向是否稳定；它不是把当前数据变成了可以直接做临床诊断的绝对血氧值。', after=8, size=10.8, bold=True, color='1F4E79')
    add_image(doc, ASSET / 'default_mbll_hbo_paired.png', width=6.25,
              caption='图 1 默认参数下的 HbO 前后比较。每条线代表一名受试者，左边是治疗前，右边是复测；纵轴是相对 HbO 变化，单位为微摩尔。看图时先看同一个人的线是向上还是向下，而不是把不同人的绝对高度当成临床水平。摸腰任务中 6/6 名受试者的线向下，是当前最值得复核的候选趋势；但样本量小、参数有外推、没有多重校正，所以图不能证明疗效，也不能证明某个穴位造成了这个变化。')

    # Page 5 comparison and sensitivity
    add_heading(doc, '四 OD 和 MBLL 是否讲出了同一个方向', level=1, page_break=False)
    add_para(doc, '我们没有把 OD 和 MBLL 的数值直接硬比，因为两者的单位和计算方式不同。我们只比较一个更可靠的问题：同一名受试者从治疗前到复测，是向上还是向下。', after=8)
    add_image(doc, ASSET / 'od_mbll_longitudinal_direction_agreement.png', width=6.35,
              caption='图 2 OD 和 MBLL 的方向一致性。每一格表示一个受试者和一个任务的前后变化方向；绿色表示两种方法方向相同，浅色表示方向不同。raw OD 与 MBLL 的方向有 18/19 一致，历史 z-OD 与 MBLL 有 17/19 一致。它说明主要方向不是只由某一种计算方法偶然造成的，但这不是对 MBLL 精度的独立验证，也不能代替扩大样本和处理质量检查。')
    add_para(doc, '这件事对论文很重要：我们可以说“方向在两种计算方法下大体重复出现”，但不能说“OD 已经证明 MBLL 一定正确”。', after=8, size=10.9, bold=True, color='1F4E79')
    add_image(doc, ASSET / 'sensitivity_sign_stability.png', width=6.35,
              caption='图 3 参数敏感性。我们改变 DPF、源探距离和滤波范围，得到 108 个参数组合。108/108 个组合的方向没有翻转，说明当前主要方向在这组参数范围内较稳；但是变化幅度约为默认结果的 0.77 到 1.60 倍，说明“方向比较稳”不等于“数值大小很精确”。因此后续应把方向作为主要探索线索，避免把微小的幅度差解释成绝对生理差异。')

    # Page 6 cognitive motor
    add_heading(doc, '五 老师的问题一 认知和运动为什么没有连起来', level=1, page_break=False)
    add_para(doc, '先直接回答：这个问题是存在的。我们现在可以把认知任务和运动任务放进同一个受试者层面的分析框架，但暂时不能证明“认知变化导致了运动变化”。原因不是没有脑电或肌电，而是实验中的认知任务和运动任务不是同一次实时任务，且认知行为结果不完整。', after=8, size=11)
    add_number(doc, '老师指出的问题：', '现在的结果像是认知做一块、运动做一块，看不出两者之间有什么关系。')
    add_number(doc, '为什么以前看起来没连起来：', '以前的结果主要按设备分别展示。认知任务是眼跳和目标搜索，运动任务是摸头、摸膝、摸腰；它们不是同一个动作过程。眼动 fNIRS 还受到红外眼动仪影响，眼动正确率和反应时间也没有作为完整的行为金标准导入。')
    add_number(doc, '现有数据能做到哪一步：', '运动任务可以同时比较脑血氧、脑电、肌肉激活和 IMU 的粗略姿态变化。认知任务可以单独描述眼动相关的脑反应和质量情况。然后在同一个受试者层面比较：认知指标变了的人，运动指标是否也往同一方向变。')
    add_number(doc, '现在把主线改成这样：', '固定治疗前后分成两条并行观察线。认知侧看脑状态和眼动任务；运动侧看任务脑血氧、EMG 肌肉激活和 IMU 有限姿态。最后只在同一名受试者层面检查两边是否同步变化。这叫建立可以检验的联系，不是证明认知导致运动。')
    add_number(doc, '每种信号用大白话解释：', 'EEG 看脑电活动的快速变化；fNIRS 看脑血氧变化，反应较慢；EMG 看肌肉有没有更强地收缩；IMU 看肢体姿态或运动强度有没有变化。四种信号放在一起，才有机会描述“脑的变化是否伴随肌肉执行变化”。')
    add_number(doc, '目前已经看到的线索：', '摸头和摸膝的 fNIRS 任务差异、静息状态分组、EMG 激活波动、脑与肌肉的相关变化都已经算过。旧阶段中，脑状态熵与任务差异的相关约为 rho=-0.464，脑肌相关变化与 EMG 共激活变化的相关约为 rho=-0.771；但有效受试者只有 7 名或 6 名，精确检验也没有达到确认性证据。')
    add_number(doc, '不能说什么：', '不能说认知训练或认知能力变化导致了运动改善；不能把 EMG 叫作完整运动学；不能把眼动 fNIRS 的结果当作无干扰的认知脑网络证据。')
    add_number(doc, '下一步怎么做：', '把已经采集但尚未导入的数据继续按同一套规则处理，优先补齐每名受试者的认知任务质量和运动任务脑肌对应关系；如果现有文件里没有正确率和反应时间，就明确写成“认知脑反应的辅助分析”，不要硬补。')

    # Page 7 acupuncture
    # Continue naturally after the cognitive section so the final two
    # cognitive points do not occupy an almost empty page by themselves.
    add_heading(doc, '六 老师的问题二 穴位 经络和脑网络为什么没有连成主线', level=1, page_break=False)
    add_para(doc, '先直接回答：老师的判断在当前证据层面是对的，我们目前没有证明“穴位—经络—脑网络”这条机制链。不是因为没有脑信号，而是所有受试者接受的是固定治疗方案，没有把穴位和经络分别拆开比较。', after=8, size=11)
    add_number(doc, '问题在哪里：', '如果最后只展示“某个通道变了、某块肌肉变了”，老师当然看不到穴位、经络和脑网络之间的关系。')
    add_number(doc, '为什么现在还不能做特异性归因：', '目前没有单穴组、不同经络组合组、假针或其他对照，也没有完整的逐次刺激日志。因此观察到的变化只能归到“固定治疗方案前后”，不能归到某个单独穴位。提供的实验方案正文中也没有写出具体穴位和经络名称，所以报告不猜名称，也不把名称写成数据验证结果。')
    add_number(doc, '现有数据能做到哪一步：', '可以把固定治疗方案当作共同背景，把治疗前后脑电状态、连接候选和任务血氧变化，和运动任务中的肌肉激活变化放在一起看。这里的“脑网络”目前只表示脑电状态或通道连接的候选变化，不能写成已经建立的网络机制、精确源定位或脑区因果。')
    add_number(doc, '建议论文主线：', '固定治疗方案作为共同干预背景；分别观察治疗前后脑侧反应和肌肉执行变化；最后检查脑和肌肉是否共同变化。论文讲的是“固定方案下的多模态响应”，不是“某穴位特异性机制”。')
    add_number(doc, '目前有什么证据：', 'MBLL 下摸腰 HbO 在 6/6 名受试者中下降；OD 和 MBLL 的主要纵向方向大体一致；部分静息状态、运动任务和脑肌关系出现可继续验证的线索。这些结果可以帮助我们提出“固定方案下可能存在脑血氧和运动执行共同变化”的假设。')
    add_number(doc, '不能说什么：', '不能说某穴位已经被证明有效，不能说某条经络特异性调节了某个脑网络，不能说某个脑区导致了动作改善，也不能把相关关系写成治疗机制。')
    add_number(doc, '下一步怎么做：', '先把已采数据全部按固定流程完成，再按受试者层面形成“脑变化—肌肉变化”的结果表。若实验日志中已经记录了治疗方案、时间或穴位组合，补充整理这些已有信息；若没有，向实验人员或厂家补问字段含义即可，不增加新的数据采集。')

    # Page 8 AI
    add_heading(doc, '七 人工智能目前到底做了什么', level=1, page_break=True)
    add_para(doc, '现在最准确的说法是：人工智能和自动化帮助我们把数据整理得更稳、更透明，但还没有训练出一个可以独立诊断、预测疗效或实时控制的深度学习模型。', after=8, size=11)
    add_table(doc, ['已经完成的工作', '它的作用', '能不能叫成 AI 创新'], [
        ['自动识别外层任务、整理文件和统一时间标记', '减少人为把任务或时间点看错的机会。', '这是自动化，不能单独当作论文 AI 创新。'],
        ['质量检查和缺失记录', '发现坏通道、突变、无效动作窗，并保留排除原因。', '这是质量控制，是可靠分析的基础。'],
        ['把质量信息带入脑和肌肉联合分析', '不是把所有数据一股脑平均，而是让不同受试者、不同信号的可靠程度影响结果解释。', '这是当前最有价值的计算思路，可以作为方法亮点。'],
        ['受试者级状态分组和透明关联', '把静息状态、任务脑反应、EMG 和 IMU 变化放到同一张结果表里，寻找可解释的共同变化。', '可以称为可解释的多模态分析原型，不能夸成已完成深度学习系统。'],
        ['FGANet、图神经网络、CNN-LSTM、SHAP', '目前没有在本数据上完成可靠训练和验证。', '暂时不能写成已经完成的工作；样本量增加后再考虑。'],
    ], widths=[2.1, 2.8, 2.0], font_size=9.0)
    add_para(doc, '如果老师问“人工智能的作用在哪里”，可以直接回答：当前 AI 的作用主要是自动整理、质量控制和把多模态信息合在一起；真正的预测模型还没有做，因为现在只有 7 名配对受试者，直接训练复杂模型很容易把噪声当成规律。', after=0, size=10.9, bold=True, color='1F4E79')

    # Page 9 script
    add_heading(doc, '八 建议跟老师怎么说', level=1, page_break=True)
    add_para(doc, '下面这段话可以直接照着汇报，时间大约 3 到 5 分钟。', after=8, size=11)
    script = [
        '老师，我们这一阶段先没有急着下疗效结论，而是先把数据处理流程固定下来。之前数据里有几个容易出错的地方：文件名和实际任务不完全一致，EMG 传感器编号整体错了一位，三种设备的时间起点也不一样。现在已经按外层文件夹判断任务，统一了 EEG、fNIRS 和 EMG 的时间关系，并把 2、3、4、5、9、10、11 号 EMG 对应到七块目标肌肉。',
        '目前有 7 名受试者同时有治疗前和复测数据，共 14 条运动近红外记录。能确认完整的动作留下，不完整的 9 个动作窗排除。近红外方面，我们同时保留了原来的 OD 相对指标，又增加了 MBLL，把两种波长换算成相对 HbO 和 HbR，看看不同计算方法下方向是否一致。',
        '结果显示，OD 和 MBLL 在受试者前后方向上大部分一致；改变 DPF、源探距离和滤波范围后，108 个参数组合的方向都没有翻转，但幅度会变，所以现在更适合把它作为稳定方向的探索结果，而不是绝对数值。摸腰任务的 HbO 在 6 名可比较的受试者中都是下降，未校正 p=0.03125。这个结果值得下一步优先复核，但还不能叫疗效或标志物。',
        '关于认知和运动，老师指出的问题是对的。我们现在把它们放进同一个受试者层面框架，但不把它们硬连成一条因果链：固定治疗前后，认知侧单独看脑状态和眼动任务，运动侧看任务脑血氧、EMG 肌肉激活和 IMU 有限姿态，最后检查两边是否同步变化。但是认知和运动不是同一次实时任务，眼动行为指标也不完整，所以不能说认知变化导致运动改善。',
        '关于穴位、经络和脑网络，老师的判断在当前证据层面是对的，我们目前没有证明这条机制链。所有人接受的是固定治疗方案，没有单穴、假针或不同组合对照；提供的实验方案正文也没有列出具体穴位和经络名称。因此我们能写的是固定方案下的脑侧反应和肌肉响应，不能把变化归因到某一个穴位或某一条经络，也不能说某个脑区产生了因果作用。',
        '所以我建议论文先走一条稳妥的主线：固定干预前后脑血氧、肌肉激活和有限姿态变化的质量控制及多模态探索。等已经采集的数据全部导入、受试者数量增加后，再判断是否有足够证据做可解释的预测模型。',
    ]
    for i, t in enumerate(script, 1):
        add_number(doc, f'{i}.', t, after=7)

    # Page 10 next steps
    add_heading(doc, '九 接下来按什么顺序做', level=1, page_break=True)
    add_heading(doc, '只用现有数据和已经采集但尚未导入的数据', level=2)
    add_table(doc, ['顺序', '下一步工作', '完成标志'], [
        ['第一步', '把当前的文件整理、任务判断、时间同步、EMG 映射和质量排除规则固定成唯一版本。', '以后新增文件只按外层目录放入，同一脚本可以重复运行。'],
        ['第二步', '对每条 fNIRS 同时保存 OD 和相对 MBLL 的结果；继续保留眼动 fNIRS 单独分析。', '每个结果都有方法、参数、有效样本数和排除原因。'],
        ['第三步', '把已采未导入的受试者逐个加入，先看数据完整性，再做受试者级前后比较。', '不把不完整标记、无效通道或缺失动作补成有效数据。'],
        ['第四步', '把脑、EMG 和 IMU 的结果放在同一张受试者结果表中，优先复核摸腰 HbO 和脑肌关联。', '从“单设备结果”变成“脑变化是否伴随肌肉执行变化”。'],
        ['第五步', '样本量足够后再做可解释模型；先做简单、可验证的模型，再考虑深度学习。', '模型必须有受试者独立验证和清楚的特征贡献解释。'],
    ], widths=[0.8, 4.1, 2.0], font_size=9.0)
    add_heading(doc, '可以补问的资料 但不需要补实验', level=2)
    add_bullet(doc, '向厂家确认 fNIRS 坐标是否是真实帽上距离，以及光强阈值、自动增益和通道状态的含义。')
    add_bullet(doc, '确认 fNIRS 文件中 analog_sig、aux 和 s 字段分别代表什么。')
    add_bullet(doc, '整理已有的刺激记录、任务标记表和每次治疗方案；如果没有，就保留为缺失，不回填。')
    add_bullet(doc, '确认 EEG 文件中 EOG 和 ECG 名义通道的导出逻辑；它们实际未接线，因此不能补做校正。')
    add_para(doc, '实验已经结束，后续计划不包括补接 EOG、ECG、短程 fNIRS，也不包括重做 TTL。我们能做的是把现有数据和已有记录处理得更干净，并把限制写清楚。', after=8, size=10.8, bold=True, color='7F6000')
    add_heading(doc, '论文题目和主线建议', level=2)
    add_table(doc, ['版本', '建议题目', '适用条件'], [
        ['保守版', '固定干预前后脑血氧与肌肉激活变化的多模态探索', '当前 7 名配对受试者也可以先形成阶段性结果；重点写数据质量、方向一致性和候选趋势。'],
        ['创新版', '质量感知和可解释的脑肌多模态状态变化研究', '需要继续加入更多已采数据，并完成受试者独立验证后再使用。'],
    ], widths=[1.0, 3.0, 2.9], font_size=9.1)

    # References and reading note
    add_heading(doc, '十 本阶段主要方法依据', level=1, page_break=True)
    add_para(doc, '本阶段只列出直接影响计算方法的两个来源，方便后续写论文时追溯。', after=8)
    add_number(doc, '血红蛋白消光系数：', '采用 Oregon Medical Laser Center 公开的血红蛋白光谱表。730 nm 和 850 nm 的 HbO/HbR 数值已经在程序中记录，并统一了对数约定。')
    add_para(doc, '来源：https://omlc.org/spectra/hemoglobin/summary.html', after=8, size=9.6, color='555555')
    add_number(doc, '年龄相关 DPF：', '采用 Scholkmann 和 Wolf 2013 年提出的年龄相关公式。报告中把 850 nm、高龄和非额区使用明确写为外推，不能把外推结果说成直接测得的绝对血氧。')
    add_para(doc, '来源：https://doi.org/10.1117/1.JBO.18.10.105004', after=8, size=9.6, color='555555')
    add_heading(doc, '最后一句话', level=2)
    add_para(doc, '当前阶段最有价值的成果，是把“数据能不能用、怎么对齐、怎么算、哪些结果只是线索”说清楚。论文先围绕固定方案下的脑血氧和肌肉执行变化建立一条可信的多模态主线，等样本量增加后，再决定是否把它发展成真正的可解释人工智能模型。', after=0, size=11.2, bold=True, color='1F4E79')

    # Core properties
    props = doc.core_properties
    props.title = 'EEG fNIRS EMG 数据目前做到了什么'
    props.subject = '面向老师的阶段性汇报'
    props.author = '研究团队'
    props.comments = '教师版匿名汇报文档'
    doc.save(DOCX)
    print(DOCX)


if __name__ == '__main__':
    build()
