"""
绘制与 fNIRS 通道图风格一致的 64 导 EEG 大脑电极排布与功能脑区分区示意图。
基于国际 10-10 系统，标定 59 个头皮记录导联 + 参考/地电极 + 辅助导联。
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, PathPatch
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath

# 字体配置
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

# 59个头皮电极的标准10-10拓扑坐标
POSITION_MAP = {
    # 额极与前额区
    "Fp1": (-0.35, 1.05), "Fpz": (0.0, 1.12), "Fp2": (0.35, 1.05),
    "AF7": (-0.72, 0.82), "AF3": (-0.28, 0.77), "AF4": (0.28, 0.77), "AF8": (0.72, 0.82),
    # 额区
    "F7": (-0.91, 0.52), "F5": (-0.62, 0.47), "F3": (-0.35, 0.42), "F1": (-0.12, 0.39),
    "Fz": (0.0, 0.39), "F2": (0.12, 0.39), "F4": (0.35, 0.42), "F6": (0.62, 0.47), "F8": (0.91, 0.52),
    # 额中央与额颞区
    "FT7": (-1.03, 0.17), "FC5": (-0.72, 0.14), "FC3": (-0.43, 0.12), "FC1": (-0.15, 0.10),
    "FCz": (0.0, 0.10), "FC2": (0.15, 0.10), "FC4": (0.43, 0.12), "FC6": (0.72, 0.14), "FT8": (1.03, 0.17),
    # 中央区与颞区
    "T7": (-1.10, -0.18), "C5": (-0.74, -0.18), "C3": (-0.43, -0.18), "C1": (-0.15, -0.18),
    "Cz": (0.0, -0.18), "C2": (0.15, -0.18), "C4": (0.43, -0.18), "C6": (0.74, -0.18), "T8": (1.10, -0.18),
    # 中央顶区与顶颞区
    "TP7": (-1.04, -0.48), "CP5": (-0.72, -0.45), "CP3": (-0.43, -0.45), "CP1": (-0.15, -0.45),
    "CP2": (0.15, -0.45), "CP4": (0.43, -0.45), "CP6": (0.72, -0.45), "TP8": (1.04, -0.48),
    # 顶区
    "P7": (-0.91, -0.72), "P5": (-0.62, -0.68), "P3": (-0.35, -0.68), "Pz": (0.0, -0.68),
    "P4": (0.35, -0.68), "P6": (0.62, -0.68), "P8": (0.91, -0.72),
    # 顶枕区
    "PO7": (-0.69, -0.93), "PO5": (-0.48, -0.88), "PO3": (-0.26, -0.91), "POz": (0.0, -0.94),
    "PO4": (0.26, -0.91), "PO6": (0.48, -0.88), "PO8": (0.69, -0.93),
    # 枕区
    "O1": (-0.25, -1.10), "Oz": (0.0, -1.15), "O2": (0.25, -1.10),
}

# 虚拟标记点（参考与地电极）
REF_GND_MAP = {
    "GND (AFz)": (0.0, 0.77),
    "REF (CPz)": (0.0, -0.45),
}

# 脑区分类及色系
ROI_GROUPS = {
    "前额叶 (PFC)": {
        "channels": ["Fpz", "Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8"],
        "color": "#E65100",  # 琥珀橙
        "bg_color": "#FFE0B2",
    },
    "额区/运动准备 (Frontal)": {
        "channels": ["Fz", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8",
                     "FCz", "FC1", "FC2", "FC3", "FC4", "FC5", "FC6"],
        "color": "#1565C0",  # 钴蓝
        "bg_color": "#BBDEFB",
    },
    "感觉运动区 (SMC / M1)": {
        "channels": ["Cz", "C1", "C2", "C3", "C4", "C5", "C6",
                     "CP1", "CP2", "CP3", "CP4", "CP5", "CP6"],
        "color": "#C2185B",  # 洋红/深粉
        "bg_color": "#F8BBD0",
    },
    "顶叶区 (Parietal)": {
        "channels": ["Pz", "P3", "P4", "P5", "P6", "P7", "P8",
                     "POz", "PO3", "PO4", "PO5", "PO6", "PO7", "PO8"],
        "color": "#2E7D32",  # 森林绿
        "bg_color": "#C8E6C9",
    },
    "枕叶区 (Occipital)": {
        "channels": ["Oz", "O1", "O2"],
        "color": "#6A1B9A",  # 紫色
        "bg_color": "#E1BEE7",
    },
    "颞叶区 (Temporal)": {
        "channels": ["FT7", "FT8", "T7", "T8", "TP7", "TP8"],
        "color": "#00838F",  # 青蓝
        "bg_color": "#B2EBF2",
    },
}

def get_channel_roi(ch):
    for roi, info in ROI_GROUPS.items():
        if ch in info["channels"]:
            return roi, info["color"]
    return "其他", "#455A64"


def draw_brain_background(ax):
    """绘制高质感的大脑轮廓（含脑沟裂、大脑外形、鼻子、耳朵）"""
    # 1. 头部外轮廓（淡灰高雅底色）
    head_outer = Circle((0, -0.05), 1.25, facecolor="#F8FAFC", edgecolor="#334155", linewidth=2.2, zorder=1)
    ax.add_patch(head_outer)

    # 2. 鼻子（Nose）
    nose_path = [
        (MplPath.MOVETO, (-0.14, 1.20)),
        (MplPath.CURVE3, (0.0, 1.40)),
        (MplPath.CURVE3, (0.14, 1.20)),
    ]
    codes, verts = zip(*nose_path)
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor="#F1F5F9", edgecolor="#334155", lw=2.2, zorder=2))

    # 3. 左耳与右耳（Ears）
    left_ear = FancyBboxPatch((-1.35, -0.22), 0.12, 0.36, boxstyle="round,pad=0.04",
                              facecolor="#F1F5F9", edgecolor="#334155", lw=2.0, zorder=0)
    right_ear = FancyBboxPatch((1.23, -0.22), 0.12, 0.36, boxstyle="round,pad=0.04",
                               facecolor="#F1F5F9", edgecolor="#334155", lw=2.0, zorder=0)
    ax.add_patch(left_ear)
    ax.add_patch(right_ear)

    # 4. 大脑半球与皮层皱褶轮廓（半透明水墨解剖质感）
    theta = np.linspace(0, 2*np.pi, 200)
    # 左大脑半球剪影
    r_left = 1.08 + 0.04 * np.sin(4 * theta)
    x_l = -0.58 + 0.52 * np.cos(theta)
    y_l = -0.05 + 1.05 * np.sin(theta)
    
    # 纵裂（中央纵缝）
    ax.plot([0, 0], [-1.15, 1.15], color="#CBD5E1", lw=1.6, linestyle="--", zorder=2)
    # 冠状缝参考线（经过Cz）
    ax.plot([-1.18, 1.18], [-0.18, -0.18], color="#E2E8F0", lw=1.2, linestyle=":", zorder=2)

    # 大脑左右半球阴影区
    left_hemi = Circle((-0.03, -0.05), 1.12, facecolor="#F1F5F9", edgecolor="#E2E8F0", lw=1.0, alpha=0.6, zorder=1)
    right_hemi = Circle((0.03, -0.05), 1.12, facecolor="#F1F5F9", edgecolor="#E2E8F0", lw=1.0, alpha=0.6, zorder=1)
    ax.add_patch(left_hemi)
    ax.add_patch(right_hemi)


def plot_eeg_montage(output_path: Path):
    fig, ax = plt.subplots(figsize=(11, 11), dpi=300)
    ax.set_aspect("equal")
    ax.axis("off")

    # 绘制大脑与头部底图
    draw_brain_background(ax)

    # 绘制 10-10 导联之间的微弱连接拓扑网格线（增加专业工程美感）
    adj_pairs = [
        ("Fpz", "Fp1"), ("Fpz", "Fp2"), ("Fp1", "AF3"), ("Fp2", "AF4"), ("AF7", "AF3"), ("AF4", "AF8"),
        ("AF7", "F7"), ("AF3", "F3"), ("AF4", "F4"), ("AF8", "F8"),
        ("F7", "F5"), ("F5", "F3"), ("F3", "F1"), ("F1", "Fz"), ("Fz", "F2"), ("F2", "F4"), ("F4", "F6"), ("F6", "F8"),
        ("F7", "FT7"), ("F3", "FC3"), ("Fz", "FCz"), ("F4", "FC4"), ("F8", "FT8"),
        ("FT7", "FC5"), ("FC5", "FC3"), ("FC3", "FC1"), ("FC1", "FCz"), ("FCz", "FC2"), ("FC2", "FC4"), ("FC4", "FC6"), ("FC6", "FT8"),
        ("FT7", "T7"), ("FC3", "C3"), ("FCz", "Cz"), ("FC4", "C4"), ("FT8", "T8"),
        ("T7", "C5"), ("C5", "C3"), ("C3", "C1"), ("C1", "Cz"), ("Cz", "C2"), ("C2", "C4"), ("C4", "C6"), ("C6", "T8"),
        ("T7", "TP7"), ("C3", "CP3"), ("Cz", "CPz" if "CPz" in REF_GND_MAP else "Pz"), ("C4", "CP4"), ("T8", "TP8"),
        ("TP7", "CP5"), ("CP5", "CP3"), ("CP3", "CP1"), ("CP1", "CP2"), ("CP2", "CP4"), ("CP4", "CP6"), ("CP6", "TP8"),
        ("TP7", "P7"), ("CP3", "P3"), ("CP4", "P4"), ("TP8", "P8"),
        ("P7", "P5"), ("P5", "P3"), ("P3", "Pz"), ("Pz", "P4"), ("P4", "P6"), ("P6", "P8"),
        ("P7", "PO7"), ("P3", "PO3"), ("Pz", "POz"), ("P4", "PO4"), ("P8", "PO8"),
        ("PO7", "PO5"), ("PO5", "PO3"), ("PO3", "POz"), ("POz", "PO4"), ("PO4", "PO6"), ("PO6", "PO8"),
        ("PO3", "O1"), ("POz", "Oz"), ("PO4", "O2"),
        ("O1", "Oz"), ("Oz", "O2")
    ]
    all_points = dict(POSITION_MAP)
    all_points.update({"CPz": REF_GND_MAP["REF (CPz)"], "AFz": REF_GND_MAP["GND (AFz)"]})

    for p1, p2 in adj_pairs:
        if p1 in all_points and p2 in all_points:
            x1, y1 = all_points[p1]
            x2, y2 = all_points[p2]
            ax.plot([x1, x2], [y1, y2], color="#94A3B8", lw=0.75, linestyle="-", alpha=0.45, zorder=3)

    # 绘制 59 个记录电极
    for ch, (x, y) in POSITION_MAP.items():
        roi_name, color = get_channel_roi(ch)
        # 电极外圈阴影与圆圈
        circle_bg = Circle((x, y), 0.052, facecolor="white", edgecolor=color, lw=1.8, zorder=4)
        ax.add_patch(circle_bg)
        # 内部微小彩点或字
        ax.text(x, y, ch, ha="center", va="center", fontsize=7.2, fontweight="bold",
                color="#0F172A", zorder=5)

    # 绘制参考电极 (REF) 与 接地电极 (GND)
    for label, (x, y) in REF_GND_MAP.items():
        rect = FancyBboxPatch((x - 0.05, y - 0.045), 0.10, 0.09, boxstyle="round,pad=0.015",
                              facecolor="#F3E8FF", edgecolor="#7E22CE", lw=1.8, linestyle="--", zorder=6)
        ax.add_patch(rect)
        short_lbl = "REF" if "REF" in label else "GND"
        ax.text(x, y, short_lbl, ha="center", va="center", fontsize=7.0, fontweight="heavy",
                color="#581C87", zorder=7)

    # 标题与副标题（严格符合课题四实验方案规范）
    ax.text(0, 1.48, "博睿康 NeuSen W 64导脑电（EEG）电极拓扑排布图", ha="center", va="center",
            fontsize=15, fontweight="bold", color="#1E293B")
    ax.text(0, 1.40, "基于国际标准 10-10 系统（59个头皮记录导联 + REF/GND + 5个外加生理导联）",
            ha="center", va="center", fontsize=9.5, color="#64748B")

    # 顶部方向标识
    ax.text(0, 1.25, "前（Front / 鼻侧）", ha="center", va="center", fontsize=8.5, color="#64748B", fontweight="medium")
    ax.text(-1.38, -0.05, "左耳\n(Left)", ha="center", va="center", fontsize=8.0, color="#64748B")
    ax.text(1.38, -0.05, "右耳\n(Right)", ha="center", va="center", fontsize=8.0, color="#64748B")
    ax.text(0, -1.33, "后（Back / 枕侧）", ha="center", va="center", fontsize=8.5, color="#64748B", fontweight="medium")

    # 左下角/右下角图例区
    # 脑区分区图例
    legend_elements = [
        mpatches.Patch(facecolor=info["color"], edgecolor="none", label=f"{roi} ({len(info['channels'])}导)")
        for roi, info in ROI_GROUPS.items()
    ]
    legend_elements.append(mpatches.Patch(facecolor="#F3E8FF", edgecolor="#7E22CE", linestyle="--", label="REF / GND (基准/地)"))
    
    leg = ax.legend(handles=legend_elements, loc="lower left", bbox_to_anchor=(-0.16, -0.15),
              ncol=4, frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0",
              title="脑电功能感兴趣区 (ROI) 分类与导联统计", title_fontsize=9.5, fontsize=8.5)
    leg.get_title().set_fontweight("bold")

    # 底部外加辅助导联说明框
    aux_text = (
        "【5个外加生理辅助导联】:  心电: ECG (1导)  |  水平眼电: HEOR, HEOL (2导)  |  垂直眼电: VEOU, VEOL (2导)\n"
        "【硬件参数】: 设备型号: Neuracle NeuSen W 64  |  电极帽: EEGCap64-V3.0  |  采样率: 1000 Hz  |  精度: 24-bit BDF"
    )
    ax.text(0, -1.48, aux_text, ha="center", va="center", fontsize=8.5,
            color="#334155", bbox=dict(boxstyle="round,pad=0.5", facecolor="#F1F5F9", edgecolor="#CBD5E1", lw=1.0))

    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.62, 1.56)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    # 同时保存一份 pdf 和 svg
    pdf_path = output_path.with_suffix(".pdf")
    svg_path = output_path.with_suffix(".svg")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")
    plt.close()
    print(f"示意图已成功生成至: {output_path}")

if __name__ == "__main__":
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_img = out_dir / "eeg_64ch_montage_layout.png"
    plot_eeg_montage(out_img)
