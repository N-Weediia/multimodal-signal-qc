"""
生成完全符合原 PDF 布局与格式的中医药现代化课题四实验方案修订版文档（PDF 与 DOCX）。
严格包含 EEG 64 导电极拓扑、fNIRS 44 通道更新与 sEMG 7 通道精细描述，大字号高可读性排版。
"""

import os
import subprocess
from pathlib import Path
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
import fitz

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUTPUT_DIR / "doc_figures"

HTML_PATH = OUTPUT_DIR / "方案_preview.html"
PDF_PATH = OUTPUT_DIR / "中医药现代化课题四实验方案_修订版.pdf"
DOCX_PATH = OUTPUT_DIR / "中医药现代化课题四实验方案_修订版.docx"
RENDER_DIR = OUTPUT_DIR / "rendered_pdf_pages"
RENDER_DIR.mkdir(parents=True, exist_ok=True)

def get_browser_exe():
    candidates = [
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("未找到 Chrome 或 Edge 浏览器用于无头渲染 PDF。")

def generate_html():
    fig1_uri = (FIG_DIR / "fig1_devices.jpg").resolve().as_uri()
    fig2_uri = (FIG_DIR / "fig2_actions.jpg").resolve().as_uri()
    fig3_uri = (FIG_DIR / "fig3_motor_protocol.png").resolve().as_uri()
    fig4_uri = (FIG_DIR / "fig4_eeg_layout.png").resolve().as_uri()
    fig5_uri = (FIG_DIR / "fig5_fnirs_layout.png").resolve().as_uri()
    fig6_uri = (FIG_DIR / "fig6_emg_muscles.jpg").resolve().as_uri()
    fig7_uri = (FIG_DIR / "fig7_cognitive_scene.jpg").resolve().as_uri()
    fig8_uri = (FIG_DIR / "fig8_cognitive_protocol.png").resolve().as_uri()

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>基于多源信息智能感知与中西医多模态特征融合的监测-评估数智平台研发实验方案</title>
<style>
  @page {{
    size: A4 portrait;
    margin: 0;
  }}
  * {{
    box-sizing: border-box;
  }}
  body {{
    margin: 0;
    padding: 0;
    background-color: #f0f0f0;
    font-family: "SimSun", "Songti SC", "STSong", "SimSun-ExtB", serif;
    font-size: 11pt;
    line-height: 1.62;
    color: #111;
    -webkit-print-color-adjust: exact;
  }}
  .page {{
    width: 210mm;
    height: 297mm;
    padding: 22mm 24mm 20mm 24mm;
    margin: 0 auto 10px auto;
    background: #ffffff;
    position: relative;
    page-break-after: always;
    overflow: hidden;
  }}
  @media print {{
    body {{
      background: none;
    }}
    .page {{
      margin: 0;
      page-break-after: always;
    }}
  }}
  .page-footer {{
    position: absolute;
    bottom: 12mm;
    left: 0;
    width: 100%;
    text-align: center;
    font-size: 10pt;
    font-family: "SimSun", serif;
    color: #444;
  }}
  h1.doc-title {{
    font-family: "SimHei", "Heiti SC", "Microsoft YaHei", sans-serif;
    font-size: 16.5pt;
    font-weight: bold;
    text-align: center;
    line-height: 1.45;
    margin: 0 0 20px 0;
    letter-spacing: 0.5px;
  }}
  h2.section-title {{
    font-family: "SimHei", "Heiti SC", "Microsoft YaHei", sans-serif;
    font-size: 13pt;
    font-weight: bold;
    margin: 14px 0 6px 0;
    color: #000;
  }}
  h3.sub-title {{
    font-family: "SimHei", "Heiti SC", "Microsoft YaHei", sans-serif;
    font-size: 11.5pt;
    font-weight: bold;
    margin: 10px 0 4px 0;
  }}
  h4.level4-title {{
    font-family: "SimHei", "Heiti SC", "Microsoft YaHei", sans-serif;
    font-size: 11pt;
    font-weight: bold;
    margin: 8px 0 4px 0;
  }}
  p {{
    margin: 3px 0 5px 0;
    text-align: justify;
    text-justify: inter-ideograph;
    line-height: 1.62;
  }}
  .indent {{
    text-indent: 2em;
  }}
  .figure-box {{
    text-align: center;
    margin: 8px 0 10px 0;
  }}
  .figure-box img {{
    max-width: 100%;
    height: auto;
    display: inline-block;
  }}
  .figure-caption {{
    font-family: "SimSun", "Songti SC", serif;
    font-size: 10pt;
    color: #222;
    text-align: center;
    margin-top: 5px;
    margin-bottom: 2px;
  }}
  .bold {{
    font-family: "SimHei", "Heiti SC", "Microsoft YaHei", sans-serif;
    font-weight: bold;
  }}
</style>
</head>
<body>

  <!-- ==================== 第 1 页 ==================== -->
  <div class="page">
    <h1 class="doc-title">基于多源信息智能感知与中西医多模态特征融合的<br>监测-评估数智平台研发实验方案</h1>

    <h2 class="section-title">一、入组条件</h2>
    <h3 class="sub-title">1.1 纳入标准</h3>
    <p>（1）单侧卒中；（2）具备完成任务所需的认知能力；（3）患侧功能性运动功能障碍；（4）卒中发病时间为 7 天至 2 年；（5）招募部分有高危基础病（高血压/房颤/糖尿病等）患者用于疾病预警模型构建。</p>

    <h3 class="sub-title">1.2 排除标准</h3>
    <p>（1）视力障碍或全色盲；（2）存在其他神经系统疾病，例如帕金森病或阿尔茨海默病；（3）存在严重的认知障碍，不能配合完成实验者。</p>

    <h2 class="section-title" style="margin-top: 14px;">二、方案整体介绍</h2>
    <p class="indent">本实验方案由两个相对独立的实验模块组成，分别为“运动评估实验”和“认知评估实验”。每位患者入组后接受中医四诊仪数据采集，之后在同一次测试中依次完成“运动任务”和“认知任务”。</p>

    <h3 class="sub-title" style="margin-top: 12px;">1. 运动评估实验</h3>
    <h4 class="level4-title">1.1 采集设备</h4>
    <p class="indent">近红外 fNIRS +脑电 EEG+肌电 sEMG 同步采集</p>

    <div class="figure-box">
      <img src="{fig1_uri}" style="max-height: 180px;" alt="图 1 采集设备和实验场景">
      <div class="figure-caption">图 1 采集设备和实验场景</div>
    </div>

    <h4 class="level4-title">1.2 动作设计</h4>
    <p class="indent">被试取坐位，背部挺直，双手自然垂于体侧，掌心朝向身体。</p>

    <div class="page-footer">1</div>
  </div>

  <!-- ==================== 第 2 页 ==================== -->
  <div class="page">
    <p><span class="bold">动作 1：手掌上举摸头</span><br>做法：缓慢上举手臂，手掌触摸同侧头顶，随后，缓慢将上肢恢复至初始位置。</p>
    <p><span class="bold">动作 2：手心触摸对侧膝部</span><br>做法：手臂缓慢伸向对侧膝盖，手触摸对侧膝盖，随后，缓慢将上肢恢复到初始位置。</p>
    <p><span class="bold">动作 3：手背触摸后腰部</span><br>做法：缓慢将手臂向后摆动，手背朝向身体后方，让手背触摸同侧后腰部，随后，缓慢将上肢恢复到初始位置。</p>

    <div class="figure-box">
      <img src="{fig2_uri}" style="max-height: 175px;" alt="图 2 三个采集动作">
      <div class="figure-caption">图 2 三个采集动作</div>
    </div>

    <p class="indent" style="margin-top: 8px;">目前已采集数据为上述动作任务，后续考虑增加下肢任务（屈髋屈膝、踝背屈、睁眼/闭眼坐站）以及平衡任务（TUG 行走试验）等。</p>

    <h4 class="level4-title" style="margin-top: 14px;">1.3 采集流程</h4>
    <p class="indent">实验开始首先采集 6min 静息态数据：被试舒适地坐在指定椅子上安静闭眼休息。</p>
    <p class="indent">运动任务执行顺序：（动作 1 摸头、动作 2 摸膝、动作 3 摸腰)。</p>
    <p class="indent">提示进行动作 5s，休息 15s，作为一个 block 任务。每个动作重复 5 次 block 任务，不同类型动作任务之间休息 2min。运动评估总实验流程约 15min。</p>

    <div class="figure-box" style="margin-top: 12px;">
      <img src="{fig3_uri}" style="max-height: 130px;" alt="图 3 实验任务流程图">
      <div class="figure-caption">图 3 实验任务流程图</div>
    </div>

    <div class="page-footer">2</div>
  </div>

  <!-- ==================== 第 3 页：脑电专题大图页 ==================== -->
  <div class="page">
    <h4 class="level4-title" style="margin-top: 4px; font-size: 12pt;">1.4 数据收集</h4>
    <p class="indent"><span class="bold">EEG 信号采集配置：</span>实验设备采用博睿康（Neuracle）NeuSen W 64通道无线脑电采集系统，配套 EEGCap64-V3.0 脑电极帽进行脑电信号采集，数据以 24-bit 精度存储，采样频率为 1000 Hz。电极排布基于国际标准扩展 10-10 系统，共包含 64 个物理记录通道：</p>
    <p class="indent">（1）59 个头皮脑电记录导联：覆盖前额区（Fpz、Fp1、Fp2、AF3、AF4、AF7、AF8）、额区（Fz、F1~F8）、额中央区（FCz、FC1~FC6、FT7、FT8）、中央区（Cz、C1~C6、T7、T8）、中央顶区（CP1~CP6、TP7、TP8）、顶区（Pz、P3~P8）、顶枕区（POz、PO3~PO8）及枕区（Oz、O1、O2）；</p>
    <p class="indent">（2）5 个外加生理辅助导联：心电（ECG）、水平眼电（HEOR、HEOL）、垂直眼电（VEOU、VEOL）；在线参考电极（REF）置于 CPz，接地电极（GND）置于 AFz，如图 4 所示。</p>

    <div class="figure-box" style="margin: 12px 0 10px 0;">
      <img src="{fig4_uri}" style="max-height: 480px;" alt="图 4 EEG 通道排布图">
      <div class="figure-caption" style="font-size: 10.5pt; font-weight: bold; margin-top: 8px;">图 4 EEG 通道排布图</div>
    </div>

    <div class="page-footer">3</div>
  </div>

  <!-- ==================== 第 4 页：近红外与肌电专题页 ==================== -->
  <div class="page">
    <p class="indent" style="margin-top: 4px;"><span class="bold">fNIRS 监测脑区选择：</span>根据空间定位及脑电 10-10 系统，将 18 个光源和 15 个探测器组成的 44 个通道覆盖双侧前额叶和双侧运动区 10 个感兴趣区域（ROI）：前额叶皮层（PFC）、运动前区（PMA）、辅助运动区（SMA）、初级运动区（M1）和感觉运动区（SMC），如图 5 所示。</p>

    <div class="figure-box" style="margin: 8px 0 14px 0;">
      <img src="{fig5_uri}" style="max-height: 230px;" alt="图 5 fNIRS 通道排布图">
      <div class="figure-caption">图 5 fNIRS 通道排布图</div>
    </div>

    <p class="indent" style="margin-top: 14px;"><span class="bold">EMG 信号采集位置：</span>实验设备采用 Delsys 公司 Trigno™ Wireless EMG 无线肌电采集系统，在患侧肩部及上肢共 7 块肌肉上放置表面肌电电极，具体包括：斜方肌（上部）、三角肌（前束、中束）、肱二头肌、肱三头肌、肱桡肌、尺侧腕屈肌。</p>

    <div class="figure-box" style="margin: 8px 0 10px 0;">
      <img src="{fig6_uri}" style="max-height: 230px;" alt="图 6 肌电信号采集位置图">
      <div class="figure-caption">图 6 肌电信号采集位置图</div>
    </div>

    <div class="page-footer">4</div>
  </div>

  <!-- ==================== 第 5 页：认知评估实验 ==================== -->
  <div class="page">
    <h3 class="sub-title" style="margin-top: 4px; font-size: 12.5pt;">2. 认知评估实验</h3>
    <h4 class="level4-title">2.1 采集设备</h4>
    <p class="indent">近红外 fNIRS +脑电 EEG+眼动设备同步采集</p>

    <div class="figure-box" style="margin: 8px 0 14px 0;">
      <img src="{fig7_uri}" style="max-height: 210px;" alt="图 7 认知评估实验场景">
      <div class="figure-caption">图 7 认知评估实验场景</div>
    </div>

    <h4 class="level4-title" style="margin-top: 12px;">2.2 任务设计</h4>
    <p><span class="bold">任务 A：眼跳任务</span><br>
    眼跳任务包括对照任务、追随任务、正向眼跳任务和反向眼跳任务四个任务。<br>
    <span class="bold">对照任务：</span>记录静息态眼跳次数与基线注视特征。屏幕中央呈现一个小注视点。被试指示：保持双眼注视屏幕中央，尽量自然睁眼，不刻意扫描。<br>
    <span class="bold">追随任务：</span>评估平滑追踪与追随时出现的补偿性追随眼跳次数。刺激：一个小圆点（目标）以恒定速度在水平轴上左右往返。被试指示：尽量平滑地用眼跟随移动点，不要用头移动。<br>
    <span class="bold">正向眼跳任务：</span>测量被试对外显视觉刺激的反应速度。刺激：中央注视点出现，注视点消失，同时在左右任一方向显示周边刺激作为目标。被试指示：要求尽快向出现的周边刺激看去。<br>
    <span class="bold">反向眼跳任务：</span>评估抑制功能与执行控制。刺激：与正向任务相同的注视与外显刺激设置，但被试指令不同：当周边刺激出现时不要看向刺激，而是尽快看向相反方向等距位置（镜像点）。</p>

    <div class="page-footer">5</div>
  </div>

  <!-- ==================== 第 6 页：认知流程与测试安排 ==================== -->
  <div class="page">
    <p style="margin-top: 4px;"><span class="bold">任务 B：目标搜索任务</span><br>
    目标搜索任务要求被试者从 3×3 的方块图像矩阵中找出目标图像。<br>
    每试次呈现一个 3×3 方格，每格显示一张图片。目标图像在每个试次 9 个位置中某一步骤出现，其余 8 个为干扰项。被试需要用眼睛注视以选择目标位置。</p>

    <h4 class="level4-title" style="margin-top: 14px;">2.3 实验流程</h4>
    <p class="indent">参与者坐在舒适的椅子上，并被要求尽可能避免头部移动。依次进行任务 A 和任务 B。</p>
    <p class="indent">任务 A：对照→追随→正向→反向，任务时间约 2min</p>
    <p class="indent">任务 B：图像试次共 16 次，任务时间约 2min。</p>
    <p class="indent">总时长约 5min</p>

    <div class="figure-box" style="margin: 12px 0 18px 0;">
      <img src="{fig8_uri}" style="max-height: 140px;" alt="图 8 认知评估实验流程">
      <div class="figure-caption">图 8 认知评估实验流程</div>
    </div>

    <h2 class="section-title" style="margin-top: 24px;">三、被试参与时间与测试安排</h2>
    <p class="indent">每名患者入组后完成首次测试 1 次；约 3-4 周干预治疗后完成第二次复测 1 次。</p>
    <p class="indent">单次正式实验任务约 20min，其中运动评估约 15min，认知评估约 5min。</p>
    <p class="indent">考虑设备佩戴、校准、任务说明、休息与突发情况处理，每次到场测试预计约 30-40min；如患者疲劳，可适当延长休息时间。</p>

    <div class="page-footer">6</div>
  </div>

</body>
</html>
"""
    HTML_PATH.write_text(html_content, encoding="utf-8")
    print(f"HTML 生成成功: {HTML_PATH}")


def compile_pdf():
    browser = get_browser_exe()
    cmd = [
        str(browser),
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF_PATH}",
        str(HTML_PATH.resolve())
    ]
    print("正在通过无头浏览器编译 PDF...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("Browser compile error:", res.stderr)
    else:
        print(f"PDF 编译成功: {PDF_PATH}")


def render_pdf_to_images():
    doc = fitz.open(PDF_PATH)
    print(f"PDF 总页数: {len(doc)}")
    image_paths = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=150)
        img_file = RENDER_DIR / f"page_{i+1}.png"
        pix.save(img_file)
        image_paths.append(img_file)
        print(f"第 {i+1} 页已渲染: {img_file}")
    return image_paths


def generate_docx():
    """生成完全对应的 Word DOCX 文档"""
    doc = Document()

    for section in doc.sections:
        section.top_margin = Inches(0.85)
        section.bottom_margin = Inches(0.85)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)

    normal_style = doc.styles['Normal']
    normal_style.font.name = '宋体'
    normal_style._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(0, 0, 0)

    def add_p(text, bold=False, align=WD_ALIGN_PARAGRAPH.LEFT, indent=False, space_after=4):
        p = doc.add_paragraph()
        p.alignment = align
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.line_spacing = 1.3
        if indent:
            p.paragraph_format.first_line_indent = Inches(0.28)
        run = p.add_run(text)
        run.bold = bold
        run.font.name = '宋体'
        run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
        return p

    def add_heading_1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(6)
        run = p.add_run(text)
        run.bold = True
        run.font.name = '黑体'
        run._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
        run.font.size = Pt(14)
        return p

    def add_heading_2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(4)
        run = p.add_run(text)
        run.bold = True
        run.font.name = '黑体'
        run._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
        run.font.size = Pt(12)
        return p

    def add_figure(img_path, caption, width=Inches(5.4)):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run()
        run.add_picture(str(img_path), width=width)
        
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.space_after = Pt(8)
        c_run = cap.add_run(caption)
        c_run.font.name = '宋体'
        c_run._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
        c_run.font.size = Pt(10)

    # 标题
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_p.paragraph_format.space_after = Pt(16)
    t_run = title_p.add_run("基于多源信息智能感知与中西医多模态特征融合的\n监测-评估数智平台研发实验方案")
    t_run.bold = True
    t_run.font.name = '黑体'
    t_run._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
    t_run.font.size = Pt(16)

    # 一、入组条件
    add_heading_1("一、入组条件")
    add_heading_2("1.1 纳入标准")
    add_p("（1）单侧卒中；（2）具备完成任务所需的认知能力；（3）患侧功能性运动功能障碍；（4）卒中发病时间为 7 天至 2 年；（5）招募部分有高危基础病（高血压/房颤/糖尿病等）患者用于疾病预警模型构建。")
    add_heading_2("1.2 排除标准")
    add_p("（1）视力障碍或全色盲；（2）存在其他神经系统疾病，例如帕金森病或阿尔茨海默病；（3）存在严重的认知障碍，不能配合完成实验者。")

    # 二、方案整体介绍
    add_heading_1("二、方案整体介绍")
    add_p("本实验方案由两个相对独立的实验模块组成，分别为“运动评估实验”和“认知评估实验”。每位患者入组后接受中医四诊仪数据采集，之后在同一次测试中依次完成“运动任务”和“认知任务”。", indent=True)

    add_heading_2("1. 运动评估实验")
    add_p("1.1 采集设备", bold=True)
    add_p("近红外 fNIRS +脑电 EEG+肌电 sEMG 同步采集", indent=True)
    add_figure(FIG_DIR / "fig1_devices.jpg", "图 1 采集设备和实验场景", width=Inches(5.8))

    add_p("1.2 动作设计", bold=True)
    add_p("被试取坐位，背部挺直，双手自然垂于体侧，掌心朝向身体。", indent=True)
    add_p("动作 1：手掌上举摸头\n做法：缓慢上举手臂，手掌触摸同侧头顶，随后，缓慢将上肢恢复至初始位置。")
    add_p("动作 2：手心触摸对侧膝部\n做法：手臂缓慢伸向对侧膝盖，手触摸对侧膝盖，随后，缓慢将上肢恢复到初始位置。")
    add_p("动作 3：手背触摸后腰部\n做法：缓慢将手臂向后摆动，手背朝向身体后方，让手背触摸同侧后腰部，随后，缓慢将上肢恢复到初始位置。")
    add_figure(FIG_DIR / "fig2_actions.jpg", "图 2 三个采集动作", width=Inches(5.8))
    add_p("目前已采集数据为上述动作任务，后续考虑增加下肢任务（屈髋屈膝、踝背屈、睁眼/闭眼坐站）以及平衡任务（TUG 行走试验）等。", indent=True)

    add_p("1.3 采集流程", bold=True)
    add_p("实验开始首先采集 6min 静息态数据：被试舒适地坐在指定椅子上安静闭眼休息。", indent=True)
    add_p("运动任务执行顺序：（动作 1 摸头、动作 2 摸膝、动作 3 摸腰)。", indent=True)
    add_p("提示进行动作 5s，休息 15s，作为一个 block 任务。每个动作重复 5 次 block 任务，不同类型动作任务之间休息 2min。运动评估总实验流程约 15min。", indent=True)
    add_figure(FIG_DIR / "fig3_motor_protocol.png", "图 3 实验任务流程图", width=Inches(5.8))

    add_p("1.4 数据收集", bold=True)
    add_p("EEG 信号采集配置：实验设备采用博睿康（Neuracle）NeuSen W 64通道无线脑电采集系统，配套 EEGCap64-V3.0 脑电极帽进行脑电信号采集，数据以 24-bit 精度存储，采样频率为 1000 Hz。电极排布基于国际标准扩展 10-10 系统，共包含 64 个物理记录通道，其中包含 59 个头皮脑电记录导联（覆盖前额区、额区、额中央区、中央区、顶区、枕区及颞区），以及 5 个外加生理辅助导联（心电 ECG、水平眼电 HEOR/HEOL、垂直眼电 VEOU/VEOL），在线参考电极（REF）置于 CPz，接地电极（GND）置于 AFz，如图 4 所示。", indent=True)
    add_figure(FIG_DIR / "fig4_eeg_layout.png", "图 4 EEG 通道排布图", width=Inches(5.4))

    add_p("fNIRS 监测脑区选择：根据空间定位及脑电 10-10 系统，将 18 个光源和 15 个探测器组成的 44 个通道覆盖双侧前额叶和双侧运动区 10 个感兴趣区域（ROI）：前额叶皮层（PFC）、运动前区（PMA）、辅助运动区（SMA）、初级运动区（M1）和感觉运动区（SMC），如图 5 所示。", indent=True)
    add_figure(FIG_DIR / "fig5_fnirs_layout.png", "图 5 fNIRS 通道排布图", width=Inches(4.5))

    add_p("EMG 信号采集位置：实验设备采用 Delsys 公司 Trigno™ Wireless EMG 无线肌电采集系统，在患侧肩部及上肢共 7 块肌肉上放置表面肌电电极，具体包括：斜方肌（上部）、三角肌（前束、中束）、肱二头肌、肱三头肌、肱桡肌、尺侧腕屈肌。", indent=True)
    add_figure(FIG_DIR / "fig6_emg_muscles.jpg", "图 6 肌电信号采集位置图", width=Inches(5.0))

    # 2. 认知评估实验
    add_heading_2("2. 认知评估实验")
    add_p("2.1 采集设备", bold=True)
    add_p("近红外 fNIRS +脑电 EEG+眼动设备同步采集", indent=True)
    add_figure(FIG_DIR / "fig7_cognitive_scene.jpg", "图 7 认知评估实验场景", width=Inches(4.2))

    add_p("2.2 任务设计", bold=True)
    add_p("任务 A：眼跳任务\n眼跳任务包括对照任务、追随任务、正向眼跳任务和反向眼跳任务四个任务。\n对照任务：记录静息态眼跳次数与基线注视特征。屏幕中央呈现一个小注视点。被试指示：保持双眼注视屏幕中央，尽量自然睁眼，不刻意扫描。\n追随任务：评估平滑追踪与追随时出现的补偿性追随眼跳次数。刺激：一个小圆点（目标）以恒定速度在水平轴上左右往返。被试指示：尽量平滑地用眼跟随移动点，不要用头移动。\n正向眼跳任务：测量被试对外显视觉刺激的反应速度。刺激：中央注视点出现，注视点消失，同时在左右任一方向显示周边刺激作为目标。被试指示：要求尽快向出现的周边刺激看去。\n反向眼跳任务：评估抑制功能与执行控制。刺激：与正向任务相同的注视与外显刺激设置，但被试指令不同：当周边刺激出现时不要看向刺激，而是尽快看向相反方向等距位置（镜像点）。")
    add_p("任务 B：目标搜索任务\n目标搜索任务要求被试者从 3×3 的方块图像矩阵中找出目标图像。\n每试次呈现一个 3×3 方格，每格显示一张图片。目标图像在每个试次 9 个位置中某一步骤出现，其余 8 个为干扰项。被试需要用眼睛注视以选择目标位置。")

    add_p("2.3 实验流程", bold=True)
    add_p("参与者坐在舒适的椅子上，并被要求尽可能避免头部移动。依次进行任务 A 和任务 B。\n任务 A：对照→追随→正向→反向，任务时间约 2min\n任务 B：图像试次共 16 次，任务时间约 2min。\n总时长约 5min", indent=True)
    add_figure(FIG_DIR / "fig8_cognitive_protocol.png", "图 8 认知评估实验流程", width=Inches(5.8))

    # 三、被试参与时间与测试安排
    add_heading_1("三、被试参与时间与测试安排")
    add_p("每名患者入组后完成首次测试 1 次；约 3-4 周干预治疗后完成第二次复测 1 次。", indent=True)
    add_p("单次正式实验任务约 20min，其中运动评估约 15min，认知评估约 5min。", indent=True)
    add_p("考虑设备佩戴、校准、任务说明、休息与突发情况处理，每次到场测试预计约 30-40min；如患者疲劳，可适当延长休息时间。", indent=True)

    try:
        doc.save(DOCX_PATH)
        print(f"DOCX 生成成功: {DOCX_PATH}")
    except PermissionError:
        alt_docx = DOCX_PATH.with_name("中医药现代化课题四实验方案_修订版_v2.docx")
        doc.save(alt_docx)
        print(f"原 DOCX 文件被外部占用，已另存为: {alt_docx}")

if __name__ == "__main__":
    generate_html()
    compile_pdf()
    generate_docx()
    render_pdf_to_images()
