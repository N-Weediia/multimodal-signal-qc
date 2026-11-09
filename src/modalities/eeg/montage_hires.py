"""
生成高可读性、大字号的 64 导 EEG 大脑电极拓扑排布图。
特点：
1. 大脑与电极充满画布 90% 以上面积，去除多余内嵌标题，字号与正文 11pt 严格等大。
2. 59 个头皮电极使用标准的均匀等距 10-10 系统几何坐标，杜绝局部挤压重叠。
3. 电极字号采用 10.5~11pt 纯黑加粗，在文档中具有极佳的视觉辨识度。
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

# 经过等距几何优化的标准 10-10 系统头皮坐标（最小间距 >= 0.20，确保 0.078 半径圆圈完全不重叠）
COORDS = {}

# 1. 0层: T7..T8 (y = 0.0)
for label, x in zip(["T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8"],
                    np.linspace(-1.08, 1.08, 9)):
    COORDS[label] = (float(x), 0.0)

# 2. FC层: y = 0.28
for label, x in zip(["FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8"],
                    np.linspace(-1.02, 1.02, 9)):
    COORDS[label] = (float(x), 0.28)

# 3. CP层: y = -0.28
for label, x in zip(["TP7", "CP5", "CP3", "CP1", "CP2", "CP4", "CP6", "TP8"],
                    [-0.98, -0.70, -0.42, -0.20, 0.20, 0.42, 0.70, 0.98]):
    COORDS[label] = (float(x), -0.28)
COORDS["REF"] = (0.0, -0.28)

# 4. F层: y = 0.56
for label, x in zip(["F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8"],
                    np.linspace(-0.90, 0.90, 9)):
    COORDS[label] = (float(x), 0.56)

# 5. P层: y = -0.56
for label, x in zip(["P7", "P5", "P3", "Pz", "P4", "P6", "P8"],
                    np.linspace(-0.85, 0.85, 7)):
    COORDS[label] = (float(x), -0.56)

# 6. AF层: y = 0.82
for label, x in zip(["AF7", "AF3", "AF4", "AF8"], [-0.68, -0.26, 0.26, 0.68]):
    COORDS[label] = (float(x), 0.82)
COORDS["GND"] = (0.0, 0.82)

# 7. PO层: y = -0.82
for label, x in zip(["PO7", "PO5", "PO3", "POz", "PO4", "PO6", "PO8"],
                    np.linspace(-0.66, 0.66, 7)):
    COORDS[label] = (float(x), -0.82)

# 8. Fp层: y = 1.05
COORDS["Fp1"] = (-0.32, 1.05)
COORDS["Fpz"] = (0.0, 1.08)
COORDS["Fp2"] = (0.32, 1.05)

# 9. O层: y = -1.05
COORDS["O1"] = (-0.26, -1.05)
COORDS["Oz"] = (0.0, -1.08)
COORDS["O2"] = (0.26, -1.05)

# 脑区分类与色彩
ROI_GROUPS = {
    "前额叶 (PFC)": {
        "channels": ["Fpz", "Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8"],
        "color": "#E65100",  # 醒目橙
    },
    "额区/运动准备 (Frontal)": {
        "channels": ["Fz", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8",
                     "FCz", "FC1", "FC2", "FC3", "FC4", "FC5", "FC6"],
        "color": "#0D47A1",  # 深蓝
    },
    "感觉运动区 (SMC/M1)": {
        "channels": ["Cz", "C1", "C2", "C3", "C4", "C5", "C6",
                     "CP1", "CP2", "CP3", "CP4", "CP5", "CP6"],
        "color": "#C2185B",  # 洋红
    },
    "顶叶区 (Parietal)": {
        "channels": ["Pz", "P3", "P4", "P5", "P6", "P7", "P8",
                     "POz", "PO3", "PO4", "PO5", "PO6", "PO7", "PO8"],
        "color": "#1B5E20",  # 深绿
    },
    "枕叶区 (Occipital)": {
        "channels": ["Oz", "O1", "O2"],
        "color": "#6A1B9A",  # 紫色
    },
    "颞叶区 (Temporal)": {
        "channels": ["FT7", "FT8", "T7", "T8", "TP7", "TP8"],
        "color": "#00838F",  # 青蓝
    },
}

def get_channel_color(ch):
    for roi, info in ROI_GROUPS.items():
        if ch in info["channels"]:
            return info["color"]
    return "#455A64"


def plot_high_readability_eeg(output_path: Path):
    # 画布大小设为与文档插入宽度吻合的 6.5 x 6.5 英寸
    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=300)
    ax.set_aspect("equal")
    ax.axis("off")

    # 1. 头部几何底图
    # 外圈头部轮廓
    head = Circle((0, 0), 1.25, facecolor="#F8FAFC", edgecolor="#1E293B", linewidth=2.5, zorder=1)
    ax.add_patch(head)

    # 鼻子 (Nose)
    nose_path = [
        (MplPath.MOVETO, (-0.16, 1.22)),
        (MplPath.CURVE3, (0.0, 1.46)),
        (MplPath.CURVE3, (0.16, 1.22)),
    ]
    codes, verts = zip(*nose_path)
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor="#F1F5F9", edgecolor="#1E293B", lw=2.5, zorder=2))

    # 耳朵 (Ears)
    left_ear = FancyBboxPatch((-1.36, -0.22), 0.12, 0.44, boxstyle="round,pad=0.03",
                              facecolor="#F1F5F9", edgecolor="#1E293B", lw=2.2, zorder=0)
    right_ear = FancyBboxPatch((1.24, -0.22), 0.12, 0.44, boxstyle="round,pad=0.03",
                               facecolor="#F1F5F9", edgecolor="#1E293B", lw=2.2, zorder=0)
    ax.add_patch(left_ear)
    ax.add_patch(right_ear)

    # 纵裂沟与冠状中线参考线
    ax.plot([0, 0], [-1.22, 1.22], color="#CBD5E1", lw=1.8, linestyle="--", zorder=2)
    ax.plot([-1.22, 1.22], [0, 0], color="#E2E8F0", lw=1.4, linestyle=":", zorder=2)

    # 大脑左右半球阴影区
    left_hemi = Circle((-0.02, 0), 1.15, facecolor="#F1F5F9", edgecolor="#E2E8F0", lw=1.0, alpha=0.5, zorder=1)
    right_hemi = Circle((0.02, 0), 1.15, facecolor="#F1F5F9", edgecolor="#E2E8F0", lw=1.0, alpha=0.5, zorder=1)
    ax.add_patch(left_hemi)
    ax.add_patch(right_hemi)

    # 2. 绘制 10-10 拓扑网格微弱连线
    grid_pairs = [
        # 纵线
        ("Fpz", "AFz"), ("AFz", "Fz"), ("Fz", "FCz"), ("FCz", "Cz"), ("Cz", "REF"), ("REF", "Pz"), ("Pz", "POz"), ("POz", "Oz"),
        ("Fp1", "AF3"), ("AF3", "F1"), ("F1", "FC1"), ("FC1", "C1"), ("C1", "CP1"), ("CP1", "P3"), ("P3", "PO3"), ("PO3", "O1"),
        ("Fp2", "AF4"), ("AF4", "F2"), ("F2", "FC2"), ("FC2", "C2"), ("C2", "CP2"), ("CP2", "P4"), ("P4", "PO4"), ("PO4", "O2"),
        # 横线
        ("Fpz", "Fp1"), ("Fpz", "Fp2"),
        ("AF7", "AF3"), ("AF3", "GND"), ("GND", "AF4"), ("AF4", "AF8"),
        ("F7", "F5"), ("F5", "F3"), ("F3", "F1"), ("F1", "Fz"), ("Fz", "F2"), ("F2", "F4"), ("F4", "F6"), ("F6", "F8"),
        ("FT7", "FC5"), ("FC5", "FC3"), ("FC3", "FC1"), ("FC1", "FCz"), ("FCz", "FC2"), ("FC2", "FC4"), ("FC4", "FC6"), ("FC6", "FT8"),
        ("T7", "C5"), ("C5", "C3"), ("C3", "C1"), ("C1", "Cz"), ("Cz", "C2"), ("C2", "C4"), ("C4", "C6"), ("C6", "T8"),
        ("TP7", "CP5"), ("CP5", "CP3"), ("CP3", "CP1"), ("CP1", "REF"), ("REF", "CP2"), ("CP2", "CP4"), ("CP4", "CP6"), ("CP6", "TP8"),
        ("P7", "P5"), ("P5", "P3"), ("P3", "Pz"), ("Pz", "P4"), ("P4", "P6"), ("P6", "P8"),
        ("PO7", "PO5"), ("PO5", "PO3"), ("PO3", "POz"), ("POz", "PO4"), ("PO4", "PO6"), ("PO6", "PO8"),
        ("O1", "Oz"), ("Oz", "O2")
    ]
    for p1, p2 in grid_pairs:
        k1 = "GND" if p1 == "AFz" else p1
        k2 = "GND" if p2 == "AFz" else p2
        k1 = "REF" if k1 == "CPz" else k1
        k2 = "REF" if k2 == "CPz" else k2
        if k1 in COORDS and k2 in COORDS:
            x1, y1 = COORDS[k1]
            x2, y2 = COORDS[k2]
            ax.plot([x1, x2], [y1, y2], color="#94A3B8", lw=0.9, linestyle="-", alpha=0.5, zorder=3)

    # 3. 绘制 59 个头皮记录电极（大圆圈 + 大字号，确保与正文字体视觉等大）
    r_circle = 0.086
    for label, (x, y) in COORDS.items():
        if label in ("REF", "GND"):
            continue
        color = get_channel_color(label)
        # 白色饱满圆圈底，彩色加粗边框
        circle = Circle((x, y), r_circle, facecolor="white", edgecolor=color, lw=2.6, zorder=5)
        ax.add_patch(circle)

        # 字体大小：3字符设为 10.2pt，2字符设为 12.5pt，粗体易读且绝不贴边
        fsize = 10.2 if len(label) >= 3 else 12.5
        ax.text(x, y, label, ha="center", va="center", fontsize=fsize, fontweight="bold",
                color="#000000", zorder=6)

    # 4. 绘制参考电极 REF 与接地电极 GND
    # GND (AFz)
    gx, gy = COORDS["GND"]
    gnd_box = FancyBboxPatch((gx - 0.082, gy - 0.072), 0.164, 0.144, boxstyle="round,pad=0.015",
                             facecolor="#F3E8FF", edgecolor="#7E22CE", lw=2.4, linestyle="--", zorder=5)
    ax.add_patch(gnd_box)
    ax.text(gx, gy, "GND", ha="center", va="center", fontsize=11.0, fontweight="heavy", color="#581C87", zorder=6)

    # REF (CPz)
    rx, ry = COORDS["REF"]
    ref_box = FancyBboxPatch((rx - 0.082, ry - 0.072), 0.164, 0.144, boxstyle="round,pad=0.015",
                             facecolor="#F3E8FF", edgecolor="#7E22CE", lw=2.4, linestyle="--", zorder=5)
    ax.add_patch(ref_box)
    ax.text(rx, ry, "REF", ha="center", va="center", fontsize=11.0, fontweight="heavy", color="#581C87", zorder=6)

    # 5. 方位标记（清晰易读，留出足够空间绝不与电极触碰）
    ax.text(0, 1.52, "前 (鼻侧)", ha="center", va="center", fontsize=12.0, fontweight="bold", color="#1E293B")
    ax.text(0, -1.35, "后 (枕侧)", ha="center", va="center", fontsize=12.0, fontweight="bold", color="#1E293B")
    ax.text(-1.50, 0, "左\n耳", ha="center", va="center", fontsize=11.5, fontweight="bold", color="#1E293B")
    ax.text(1.50, 0, "右\n耳", ha="center", va="center", fontsize=11.5, fontweight="bold", color="#1E293B")

    # 6. 紧凑精炼的底部图例
    legend_elements = [
        mpatches.Patch(facecolor=info["color"], edgecolor="none", label=roi.split()[0])
        for roi, info in ROI_GROUPS.items()
    ]
    legend_elements.append(mpatches.Patch(facecolor="#F3E8FF", edgecolor="#7E22CE", linestyle="--", label="基准/地"))
    
    ax.legend(handles=legend_elements, loc="lower center", bbox_to_anchor=(0.5, -0.16),
              ncol=4, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1",
              fontsize=10.5, columnspacing=1.2, handletextpad=0.5)

    ax.set_xlim(-1.65, 1.65)
    ax.set_ylim(-1.55, 1.62)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"大字号高可读性 EEG 拓扑图已生成至: {output_path}")

if __name__ == "__main__":
    out_dir = Path("outputs/doc_figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_img = out_dir / "fig4_eeg_layout.png"
    plot_high_readability_eeg(out_img)
