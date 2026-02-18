"""
XAI手法 × NNアーキテクチャ 構造図 v2

上段: CNN アーキテクチャ (画像系: Problem 2, 3, 5)
  - 入力画像を6×6ピクセルグリッド + グロー効果で表現
  - Conv層を3Dパース(台形+影)特徴マップで表現、Poolingによるサイズ漸減
  - Flatten: 扇形に広がる点線
  - FC層: サンプリング接続
  - 出力ニューロン: 明るめ色 + 外縁グロー

下段: 決定木アンサンブル (表形式: Problem 1, 4)
  - 決定木: 分岐ノード(菱形/丸) vs 葉ノード(四角)を区別
  - 各木の葉→集約への線束

各XAI手法がモデルのどの層・パラメータにアクセスするかを
色分けした矢印 + ミニアイコン + 数式記号で示す。
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon
from matplotlib.collections import PatchCollection
from matplotlib.path import Path as MPath
import numpy as np
from pathlib import Path
import matplotlib.font_manager as fm
import platform


# ── Font setup ──────────────────────────────────────────
def setup_japanese_font():
    system = platform.system()
    candidates = []
    if system == "Darwin":
        candidates = ["Hiragino Sans", "Hiragino Kaku Gothic Pro", "Osaka"]
    elif system == "Windows":
        candidates = ["Yu Gothic", "Meiryo", "MS Gothic"]
    else:
        candidates = ["Noto Sans CJK JP", "IPAGothic", "VL Gothic"]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            return font
    plt.rcParams["font.family"] = "sans-serif"
    return "sans-serif"

setup_japanese_font()

# ── Colors ──────────────────────────────────────────────
C = {
    "bg":       "#0d1117",
    "panel":    "#161b22",
    "card":     "#1c2333",
    "border":   "#30363d",
    "text":     "#e6edf3",
    "dim":      "#8b949e",
    "accent":   "#58a6ff",
    # XAI categories
    "grad":     "#ff6b6b",
    "pert":     "#4a9eff",
    "game":     "#4ecdc4",
    "vis":      "#ffd93d",
    # Neuron fills
    "neuron":   "#2a3444",
    "neuron_e": "#4a5568",
    "fmap":     "#1e293b",
    "fmap_e":   "#3b5068",
    # Output neuron (brighter)
    "out_fc":   "#5a2020",
    "out_ec":   "#cc6666",
    "out_glow": "#ff6b6b",
}


# ══════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════

def draw_neuron(ax, cx, cy, r=0.012, fc=C["neuron"], ec=C["neuron_e"],
                lw=1.0, zorder=5, glow=False, glow_color=None):
    """Draw a single neuron. Optional outer glow ring."""
    if glow:
        gc = glow_color or ec
        glow_c = Circle((cx, cy), r * 1.6, facecolor=gc, edgecolor="none",
                         alpha=0.12, zorder=zorder - 1)
        ax.add_patch(glow_c)
    c = Circle((cx, cy), r, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(c)
    return (cx, cy)


def draw_neuron_column(ax, cx, y_center, n, spacing=0.032, r=0.012, **kw):
    """Draw a vertical column of neurons, return list of (cx, cy)."""
    positions = []
    total_h = (n - 1) * spacing
    y_start = y_center + total_h / 2
    for i in range(n):
        y = y_start - i * spacing
        draw_neuron(ax, cx, y, r=r, **kw)
        positions.append((cx, y))
    return positions


def connect_layers_sampled(ax, layer_a, layer_b, sample_rate=0.4,
                           color=C["dim"], lw=0.4, alpha=0.25, zorder=2,
                           band_alpha=0.06):
    """Draw sampled connections + faint band for undrawn connections."""
    rng = np.random.RandomState(123)
    # Draw a faint filled region between leftmost-A and rightmost-B
    if layer_a and layer_b:
        ya_top = max(y for _, y in layer_a)
        ya_bot = min(y for _, y in layer_a)
        yb_top = max(y for _, y in layer_b)
        yb_bot = min(y for _, y in layer_b)
        xa = layer_a[0][0]
        xb = layer_b[0][0]
        band_verts = [(xa, ya_top), (xb, yb_top), (xb, yb_bot), (xa, ya_bot)]
        band = Polygon(band_verts, closed=True, facecolor=color,
                       edgecolor="none", alpha=band_alpha, zorder=zorder - 1)
        ax.add_patch(band)
    # Sample a few connections
    for (x1, y1) in layer_a:
        for (x2, y2) in layer_b:
            if rng.random() < sample_rate:
                ax.plot([x1, x2], [y1, y2], color=color, lw=lw,
                        alpha=alpha, zorder=zorder)


def draw_feature_map_3d(ax, cx, cy, n_maps, map_w, map_h,
                         offset_x=0.004, offset_y=0.004,
                         fc=C["fmap"], ec=C["fmap_e"], lw=0.8,
                         zorder=3, perspective=True):
    """Draw 3D-perspective stacked feature maps with shadow.

    If perspective=True, the front face is drawn as a slight trapezoid
    to convey depth. A shadow polygon is placed behind the stack.
    """
    # Shadow
    shadow_dx, shadow_dy = 0.005, -0.005
    total_ox = (n_maps - 1) * offset_x
    total_oy = (n_maps - 1) * offset_y
    sx = cx - map_w / 2 + shadow_dx + total_ox
    sy = cy - map_h / 2 + shadow_dy - total_oy
    shadow = FancyBboxPatch(
        (sx, sy), map_w, map_h,
        boxstyle="round,pad=0.003",
        facecolor="#000000", edgecolor="none", alpha=0.25,
        linewidth=0, zorder=zorder - 1,
    )
    ax.add_patch(shadow)

    rects = []
    for i in range(n_maps):
        x = cx - map_w / 2 + i * offset_x
        y = cy - map_h / 2 - i * offset_y
        a = 0.55 + 0.45 * (i / max(n_maps - 1, 1))
        if perspective and i == n_maps - 1:
            # Front face: slight trapezoid (top narrower)
            shrink = 0.003
            verts = [
                (x + shrink, y + map_h),           # top-left
                (x + map_w - shrink, y + map_h),    # top-right
                (x + map_w, y),                      # bot-right
                (x, y),                              # bot-left
            ]
            trap = Polygon(verts, closed=True, facecolor=fc, edgecolor=ec,
                           linewidth=lw, alpha=a, zorder=zorder + i)
            ax.add_patch(trap)
            rects.append(trap)
        else:
            rect = FancyBboxPatch(
                (x, y), map_w, map_h,
                boxstyle="round,pad=0.002",
                facecolor=fc, edgecolor=ec, linewidth=lw,
                zorder=zorder + i, alpha=a,
            )
            ax.add_patch(rect)
            rects.append(rect)
    # Connection points
    last_x = cx - map_w / 2 + (n_maps - 1) * offset_x
    last_y = cy - (n_maps - 1) * offset_y
    right = (last_x + map_w, last_y + map_h / 2)
    left = (cx - map_w / 2, cy + map_h / 2)
    center = (cx + (n_maps - 1) * offset_x / 2,
              cy - (n_maps - 1) * offset_y / 2 + map_h / 2)
    # The "heatmap region" for Grad-CAM overlay
    front_x = last_x
    front_y = last_y
    return {"right": right, "left": left, "center": center,
            "front_xy": (front_x, front_y), "front_wh": (map_w, map_h)}


def draw_input_image(ax, cx, cy, size=0.08, grid=6):
    """Draw a 6×6 pixel-grid input image with outer glow."""
    rng = np.random.RandomState(42)
    cell = size / grid
    x0 = cx - size / 2
    y0 = cy - size / 2
    for r in range(grid):
        for c_ in range(grid):
            val = rng.uniform(0.15, 0.85)
            color = plt.cm.viridis(val)
            rect = Rectangle((x0 + c_ * cell, y0 + r * cell), cell, cell,
                              facecolor=color, edgecolor=C["border"],
                              linewidth=0.3, zorder=4)
            ax.add_patch(rect)
    # Glow border
    glow = FancyBboxPatch(
        (x0 - 0.006, y0 - 0.006), size + 0.012, size + 0.012,
        boxstyle="round,pad=0.005",
        facecolor=C["accent"], edgecolor="none", alpha=0.08,
        linewidth=0, zorder=3)
    ax.add_patch(glow)
    # Crisp border
    border = FancyBboxPatch(
        (x0 - 0.002, y0 - 0.002), size + 0.004, size + 0.004,
        boxstyle="round,pad=0.003",
        facecolor="none", edgecolor=C["accent"], linewidth=1.5, zorder=5)
    ax.add_patch(border)
    return {"right": (cx + size / 2 + 0.004, cy), "center": (cx, cy),
            "top": (cx, cy + size / 2 + 0.004),
            "bot": (cx, cy - size / 2 - 0.004),
            "x0": x0, "y0": y0, "size": size}


def draw_flow_arrow(ax, p1, p2, color=C["dim"], lw=1.5, zorder=4, style="-|>"):
    arr = FancyArrowPatch(p1, p2, arrowstyle=style, color=color,
                          linewidth=lw, mutation_scale=14, zorder=zorder)
    ax.add_patch(arr)


def draw_flatten_fan(ax, src_right, dst_neurons, color=C["dim"],
                     lw=0.5, alpha=0.25, zorder=2):
    """Draw fan-out dotted lines from a single point to neuron column."""
    for (nx, ny) in dst_neurons:
        ax.plot([src_right[0], nx], [src_right[1], ny],
                color=color, lw=lw, alpha=alpha, ls=":", zorder=zorder)


def draw_heatmap_overlay(ax, x, y, w, h, zorder=8):
    """Draw a Grad-CAM style heatmap overlay on a feature map."""
    rng = np.random.RandomState(7)
    n = 4
    cw, ch = w / n, h / n
    for r in range(n):
        for c_ in range(n):
            val = rng.uniform(0.0, 1.0)
            # Center-biased heat
            dist = ((r - n / 2 + 0.5) ** 2 + (c_ - n / 2 + 0.5) ** 2) / (n * 0.8)
            heat = max(0, 1 - dist) * 0.7 + val * 0.3
            rgba = plt.cm.hot(heat)
            rect = Rectangle((x + c_ * cw, y + r * ch), cw, ch,
                              facecolor=rgba, edgecolor="none",
                              alpha=0.55, zorder=zorder)
            ax.add_patch(rect)


def draw_occlusion_patch(ax, x0, y0, size, zorder=9):
    """Draw a small grey occlusion patch on the input image."""
    pw = size * 0.3
    ph = size * 0.3
    px = x0 + size * 0.35
    py = y0 + size * 0.3
    patch = Rectangle((px, py), pw, ph, facecolor="#888888",
                       edgecolor="#aaaaaa", linewidth=1.0,
                       alpha=0.85, zorder=zorder)
    ax.add_patch(patch)
    # cross pattern
    cx, cy2 = px + pw / 2, py + ph / 2
    d = pw * 0.25
    ax.plot([cx - d, cx + d], [cy2 - d, cy2 + d], color="#cccccc",
            lw=1.0, zorder=zorder + 1)
    ax.plot([cx - d, cx + d], [cy2 + d, cy2 - d], color="#cccccc",
            lw=1.0, zorder=zorder + 1)


def draw_tree_symbol_v2(ax, root_x, root_y, width, height,
                         color_node, color_edge, depth=3):
    """Draw a symbolic decision tree with distinct node shapes.

    Internal nodes: circles (decision)
    Leaf nodes: small squares
    """
    nodes = []
    leaf_positions = []

    def _draw(x, y, level, spread):
        is_leaf = (level >= depth)
        if is_leaf:
            # Leaf: small filled square
            s = 0.008
            rect = Rectangle((x - s / 2, y - s / 2), s, s,
                              facecolor=C["game"], edgecolor=color_edge,
                              linewidth=0.8, zorder=6, alpha=0.85)
            ax.add_patch(rect)
            nodes.append((x, y))
            leaf_positions.append((x, y))
        else:
            # Internal: circle (decision node)
            r = 0.008
            c = Circle((x, y), r, facecolor=color_node, edgecolor=color_edge,
                        linewidth=0.9, zorder=6, alpha=0.9)
            ax.add_patch(c)
            nodes.append((x, y))
            dy = height / (depth + 0.5)
            dx = spread / 2
            for child_x in [x - dx, x + dx]:
                child_y = y - dy
                ax.plot([x, child_x], [y - 0.008, child_y + 0.008],
                        color=color_edge, lw=0.9, alpha=0.6, zorder=5)
                _draw(child_x, child_y, level + 1, spread / 2)

    _draw(root_x, root_y, 0, width / 2)
    return nodes, leaf_positions


# ── Mini-icon drawing functions ─────────────────────────

def draw_icon_heatmap(ax, cx, cy, s=0.012, zorder=10):
    """Mini heatmap icon (gradient square)."""
    n = 3
    cell = s / n
    x0, y0 = cx - s / 2, cy - s / 2
    for r in range(n):
        for c_ in range(n):
            heat = (r + c_) / (2 * n - 2)
            rgba = plt.cm.hot(heat)
            rect = Rectangle((x0 + c_ * cell, y0 + r * cell), cell, cell,
                              facecolor=rgba, edgecolor="none",
                              alpha=0.9, zorder=zorder)
            ax.add_patch(rect)
    border = Rectangle((x0, y0), s, s, facecolor="none",
                        edgecolor="#ffffff", linewidth=0.5, zorder=zorder + 1)
    ax.add_patch(border)


def draw_icon_integral(ax, cx, cy, color, fontsize=12, zorder=10):
    """Integral symbol ∫."""
    ax.text(cx, cy, "∫", ha="center", va="center",
            fontsize=fontsize, color=color, fontweight="bold", zorder=zorder)


def draw_icon_grey_patch(ax, cx, cy, s=0.012, zorder=10):
    """Grey occlusion patch icon."""
    rect = Rectangle((cx - s / 2, cy - s / 2), s, s,
                      facecolor="#888888", edgecolor="#bbbbbb",
                      linewidth=0.8, alpha=0.9, zorder=zorder)
    ax.add_patch(rect)
    d = s * 0.3
    ax.plot([cx - d, cx + d], [cy - d, cy + d], color="#dddddd",
            lw=0.8, zorder=zorder + 1)
    ax.plot([cx - d, cx + d], [cy + d, cy - d], color="#dddddd",
            lw=0.8, zorder=zorder + 1)


def draw_icon_tree_shapley(ax, cx, cy, color, fontsize=8, zorder=10):
    """TreeSHAP icon: small tree + φ."""
    # Tiny tree
    r = 0.004
    ax.plot([cx - 0.008, cx], [cy - 0.005, cy + 0.005], color=color,
            lw=0.8, zorder=zorder)
    ax.plot([cx + 0.008, cx], [cy - 0.005, cy + 0.005], color=color,
            lw=0.8, zorder=zorder)
    c = Circle((cx, cy + 0.005), r, facecolor=color, edgecolor="none",
               zorder=zorder)
    ax.add_patch(c)
    ax.text(cx + 0.015, cy, "φ", ha="center", va="center",
            fontsize=fontsize, color=color, fontstyle="italic",
            fontweight="bold", zorder=zorder)


def draw_icon_scatter_line(ax, cx, cy, color, s=0.012, zorder=10):
    """LIME icon: scatter dots + fitted line."""
    rng = np.random.RandomState(11)
    for _ in range(5):
        dx = rng.uniform(-s / 2, s / 2)
        dy = rng.uniform(-s / 2, s / 2)
        c = Circle((cx + dx, cy + dy), 0.002, facecolor=color,
                    edgecolor="none", alpha=0.7, zorder=zorder)
        ax.add_patch(c)
    ax.plot([cx - s / 2, cx + s / 2], [cy - s / 3, cy + s / 3],
            color=color, lw=1.2, alpha=0.9, zorder=zorder + 1)


def draw_icon_shuffle(ax, cx, cy, color, s=0.010, zorder=10):
    """Permutation Importance icon: shuffle arrows ⇄."""
    # Draw shuffle arrows with lines instead of Unicode
    d = s * 0.5
    ax.annotate("", xy=(cx + d, cy + d * 0.3), xytext=(cx - d, cy - d * 0.3),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2),
                zorder=zorder)
    ax.annotate("", xy=(cx - d, cy + d * 0.3), xytext=(cx + d, cy - d * 0.3),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2),
                zorder=zorder)


def draw_icon_mini_curve(ax, cx, cy, color, s=0.014, zorder=10):
    """PDP/ICE icon: small rising curve."""
    xs = np.linspace(cx - s / 2, cx + s / 2, 20)
    ys = cy - s / 3 + (s * 0.7) * (1 / (1 + np.exp(-12 * (xs - cx) / s)))
    ax.plot(xs, ys, color=color, lw=1.4, alpha=0.9, zorder=zorder)
    # Axes
    ax.plot([cx - s / 2, cx + s / 2], [cy - s / 3, cy - s / 3],
            color=color, lw=0.5, alpha=0.5, zorder=zorder)
    ax.plot([cx - s / 2, cx - s / 2], [cy - s / 3, cy + s / 2.5],
            color=color, lw=0.5, alpha=0.5, zorder=zorder)


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════
def generate_diagram():
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(24, 16),
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.10},
    )
    fig.patch.set_facecolor(C["bg"])

    for ax in (ax_top, ax_bot):
        ax.set_facecolor(C["bg"])
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.12, 1.08)
        ax.set_aspect("equal")
        ax.axis("off")

    # ════════════════════════════════════════════════════════
    #  UPPER PANEL – CNN
    # ════════════════════════════════════════════════════════
    ax_top.add_patch(FancyBboxPatch(
        (-0.02, -0.11), 1.04, 1.17,
        boxstyle="round,pad=0.012", facecolor=C["panel"],
        edgecolor=C["border"], linewidth=1, zorder=0))

    ax_top.text(0.01, 1.04, "CNN アーキテクチャ（画像系: Problem 2, 3, 5）",
                fontsize=13, color=C["accent"], fontweight="bold",
                va="bottom", zorder=10)

    y_mid = 0.48
    layer_label_y = 0.10

    # ── 1. Input image (6×6 + glow) ──
    inp = draw_input_image(ax_top, 0.06, y_mid, size=0.09, grid=6)
    ax_top.text(0.06, layer_label_y, "入力画像\nH×W×C", ha="center",
                fontsize=8, color=C["dim"], va="top", zorder=10)

    # ── 2. Conv blocks (3D perspective feature maps) ──
    conv_specs = [
        {"cx": 0.19, "n": 6,  "w": 0.055, "h": 0.080, "label": "Conv1+ReLU\n+Pool",
         "size": "32@24×24"},
        {"cx": 0.32, "n": 8,  "w": 0.045, "h": 0.065, "label": "Conv2+ReLU\n+Pool",
         "size": "64@12×12"},
        {"cx": 0.45, "n": 10, "w": 0.038, "h": 0.052, "label": "Conv3+ReLU\n+Pool",
         "size": "128@6×6"},
    ]
    conv_points = []
    prev_right = inp["right"]
    for spec in conv_specs:
        pts = draw_feature_map_3d(
            ax_top, spec["cx"], y_mid, spec["n"],
            spec["w"], spec["h"],
            offset_x=0.0035, offset_y=0.0035, perspective=True)
        conv_points.append(pts)
        draw_flow_arrow(ax_top, prev_right, pts["left"], lw=1.3)
        prev_right = pts["right"]
        label_x = spec["cx"] + spec["n"] * 0.0035 / 2
        ax_top.text(label_x, layer_label_y,
                    spec["label"], ha="center", fontsize=7.5,
                    color=C["dim"], va="top", zorder=10, linespacing=1.2)
        ax_top.text(label_x, layer_label_y - 0.045,
                    spec["size"], ha="center", fontsize=7,
                    color=C["accent"], va="top", zorder=10)

    # ── 3. Flatten (fan-out) ──
    flat_x = 0.55
    ax_top.text(flat_x, y_mid, "Flatten", ha="center", va="center",
                fontsize=7.5, color=C["dim"], rotation=90, zorder=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=C["card"],
                          edgecolor=C["border"], linewidth=0.8))
    draw_flow_arrow(ax_top, prev_right, (flat_x - 0.015, y_mid), lw=1.3)

    # ── 4. FC layers (sampled connections) ──
    fc_specs = [
        {"cx": 0.62, "n": 7, "label": "FC1 (512)", "spacing": 0.028},
        {"cx": 0.72, "n": 5, "label": "FC2 (128)", "spacing": 0.028},
    ]
    fc_layers = []
    for spec in fc_specs:
        neurons = draw_neuron_column(ax_top, spec["cx"], y_mid, spec["n"],
                                     spacing=spec["spacing"], r=0.010,
                                     fc=C["neuron"], ec=C["neuron_e"])
        fc_layers.append(neurons)
        ax_top.text(spec["cx"], layer_label_y, spec["label"], ha="center",
                    fontsize=8, color=C["dim"], va="top", zorder=10)

    # Fan-out: Flatten → FC1
    flat_right = (flat_x + 0.018, y_mid)
    draw_flatten_fan(ax_top, flat_right, fc_layers[0],
                     color=C["dim"], lw=0.5, alpha=0.25)
    ax_top.text(flat_x, layer_label_y, "Flatten\n9216→512", ha="center",
                fontsize=7, color=C["accent"], va="top", zorder=10)

    # FC1 → FC2: sampled connections
    connect_layers_sampled(ax_top, fc_layers[0], fc_layers[1],
                           sample_rate=0.35, lw=0.4, alpha=0.22)

    # ── 5. Output neurons (bright + glow) ──
    out_cx = 0.82
    out_neurons = []
    out_n = 3
    out_spacing = 0.040
    total_h = (out_n - 1) * out_spacing
    y_start = y_mid + total_h / 2
    for i in range(out_n):
        ny = y_start - i * out_spacing
        draw_neuron(ax_top, out_cx, ny, r=0.013,
                    fc=C["out_fc"], ec=C["out_ec"], lw=1.5,
                    glow=True, glow_color=C["out_glow"])
        out_neurons.append((out_cx, ny))

    connect_layers_sampled(ax_top, fc_layers[1], out_neurons,
                           sample_rate=0.5, lw=0.4, alpha=0.22)
    ax_top.text(out_cx, layer_label_y, "出力\n(Class Score)", ha="center",
                fontsize=8, color=C["dim"], va="top", zorder=10,
                linespacing=1.2)

    # Softmax label + class labels
    ax_top.text(out_cx + 0.028, y_mid, "softmax", ha="left", va="center",
                fontsize=7, color=C["dim"], zorder=10, rotation=-90)
    class_labels = ["cat", "dog", "..."]
    for i, (nx, ny) in enumerate(out_neurons):
        ax_top.text(nx + 0.023, ny, class_labels[i], ha="left", va="center",
                    fontsize=6.5, color=C["dim"], zorder=10)

    # ════════════════════════════════════════════════════════
    #  CNN XAI ANNOTATIONS
    # ════════════════════════════════════════════════════════

    # --- Grad-CAM ---
    gc = C["grad"]
    conv3_c = conv_points[2]["center"]
    front_xy = conv_points[2]["front_xy"]
    front_wh = conv_points[2]["front_wh"]
    out_top = out_neurons[0]

    # Heatmap overlay on Conv3 front face
    draw_heatmap_overlay(ax_top, front_xy[0], front_xy[1],
                         front_wh[0], front_wh[1], zorder=8)

    # Gradient backward arrow: output → Conv3
    gc_y_line = 0.80
    ax_top.annotate("", xy=(conv3_c[0], gc_y_line),
                    xytext=(out_top[0], gc_y_line),
                    arrowprops=dict(arrowstyle="-|>", color=gc, lw=2.5,
                                   connectionstyle="arc3,rad=0.0"),
                    zorder=8)
    ax_top.plot([conv3_c[0], conv3_c[0]],
                [conv3_c[1] + 0.035, gc_y_line],
                color=gc, lw=1.8, ls="--", alpha=0.7, zorder=7)
    ax_top.plot([out_top[0], out_top[0]],
                [out_top[1] + 0.018, gc_y_line],
                color=gc, lw=1.8, ls="--", alpha=0.7, zorder=7)

    # Grad-CAM label + icon
    gc_label_x = (conv3_c[0] + out_top[0]) / 2
    draw_icon_heatmap(ax_top, gc_label_x - 0.065, gc_y_line + 0.035,
                      s=0.014, zorder=10)
    ax_top.text(gc_label_x, gc_y_line + 0.038,
                "Grad-CAM", ha="center", fontsize=11, color=gc,
                fontweight="bold", zorder=10)
    ax_top.text(gc_label_x, gc_y_line + 0.013,
                "dy(c)/dA(k) 特徴マップの勾配重み付け平均",
                ha="center", fontsize=7.5, color=gc, alpha=0.85, zorder=10)

    # Highlight Conv3
    c3 = conv_specs[2]
    hl = FancyBboxPatch(
        (c3["cx"] - c3["w"] / 2 - 0.008, y_mid - c3["h"] / 2 - 0.015),
        c3["w"] + c3["n"] * 0.0035 + 0.016,
        c3["h"] + 0.030,
        boxstyle="round,pad=0.005", facecolor="none", edgecolor=gc,
        linewidth=2.0, linestyle="--", zorder=7)
    ax_top.add_patch(hl)

    # --- Integrated Gradients ---
    ig = C["grad"]
    ig_y = y_mid - 0.18
    # Thick gradient arrow from input to output
    ax_top.annotate("", xy=(out_cx + 0.02, ig_y),
                    xytext=(0.04, ig_y),
                    arrowprops=dict(arrowstyle="-|>", color=ig, lw=3.0,
                                   alpha=0.5),
                    zorder=6)
    # Integral symbol
    draw_icon_integral(ax_top, 0.25, ig_y + 0.002, ig, fontsize=14, zorder=10)
    # α labels
    ax_top.text(0.06, ig_y + 0.020, "α=0 (基準)", ha="left",
                fontsize=6.5, color=ig, alpha=0.8, zorder=10)
    ax_top.text(out_cx - 0.03, ig_y + 0.020, "α=1 (実入力)", ha="right",
                fontsize=6.5, color=ig, alpha=0.8, zorder=10)
    # Label
    ax_top.text(0.44, ig_y - 0.025,
                "Integrated Gradients: 基準→実入力まで勾配を経路積分",
                ha="center", fontsize=8.5, color=ig, fontweight="bold", zorder=10)
    # Connecting vertical lines
    ax_top.annotate("", xy=(0.06, y_mid - 0.06),
                    xytext=(0.06, ig_y + 0.01),
                    arrowprops=dict(arrowstyle="-|>", color=ig, lw=1.0,
                                   alpha=0.4), zorder=5)
    ax_top.annotate("", xy=(out_cx, y_mid - 0.06),
                    xytext=(out_cx, ig_y + 0.01),
                    arrowprops=dict(arrowstyle="-|>", color=ig, lw=1.0,
                                   alpha=0.4), zorder=5)

    # --- Occlusion ---
    oc = C["pert"]
    oc_y = ig_y - 0.09
    # Occlusion patch on input image
    draw_occlusion_patch(ax_top, inp["x0"], inp["y0"], inp["size"], zorder=9)
    # Black-box call arrow: input → output
    ax_top.annotate("", xy=(out_cx + 0.02, oc_y),
                    xytext=(0.04, oc_y),
                    arrowprops=dict(arrowstyle="-|>", color=oc, lw=2.0,
                                   linestyle="dashed", alpha=0.6),
                    zorder=6)
    # Icon
    draw_icon_grey_patch(ax_top, 0.14, oc_y, s=0.013, zorder=10)
    # f(x) label on arrow
    ax_top.text(0.44, oc_y + 0.008, "f(x)", ha="center", va="bottom",
                fontsize=8, color=oc, fontstyle="italic", zorder=10)
    # Label
    ax_top.text(0.44, oc_y - 0.025,
                "Occlusion: 入力の一部を遮蔽(スライディングウィンドウ) → 出力変化を測定",
                ha="center", fontsize=8.5, color=oc, fontweight="bold", zorder=10)
    # Connecting vertical line from input
    ax_top.annotate("", xy=(0.06, ig_y - 0.04),
                    xytext=(0.06, oc_y + 0.01),
                    arrowprops=dict(arrowstyle="-|>", color=oc, lw=1.0,
                                   alpha=0.4), zorder=5)

    # ════════════════════════════════════════════════════════
    #  LOWER PANEL – Tree Ensemble
    # ════════════════════════════════════════════════════════
    ax_bot.add_patch(FancyBboxPatch(
        (-0.02, -0.11), 1.04, 1.17,
        boxstyle="round,pad=0.012", facecolor=C["panel"],
        edgecolor=C["border"], linewidth=1, zorder=0))

    ax_bot.text(0.01, 1.04, "決定木アンサンブル（表形式: Problem 1, 4）",
                fontsize=13, color=C["accent"], fontweight="bold",
                va="bottom", zorder=10)

    tree_y_mid = 0.52

    # ── Input features ──
    inp_cx = 0.07
    inp_n = 6
    inp_neurons = draw_neuron_column(ax_bot, inp_cx, tree_y_mid, inp_n,
                                     spacing=0.028, r=0.009,
                                     fc="#1e293b", ec="#4a5568")
    feat_labels = ["x1", "x2", "x3", "x4", "...", "xM"]
    for i, (nx, ny) in enumerate(inp_neurons):
        ax_bot.text(nx - 0.020, ny, feat_labels[i], ha="right", va="center",
                    fontsize=7, color=C["dim"], zorder=10)
    ax_bot.text(inp_cx, tree_y_mid - 0.14, "入力特徴量\nM次元", ha="center",
                fontsize=9, color=C["dim"], va="top", zorder=10)

    # ── Decision trees (v2: distinct node shapes) ──
    tree_specs = [
        {"cx": 0.22, "label": "Tree 1"},
        {"cx": 0.35, "label": "Tree 2"},
        {"cx": 0.48, "label": "Tree 3"},
    ]

    tree_width = 0.09
    tree_height = 0.11
    all_tree_leaf_positions = []
    tree_root_positions = []

    for spec in tree_specs:
        root_y = tree_y_mid + tree_height / 2
        nodes, leaves = draw_tree_symbol_v2(
            ax_bot, spec["cx"], root_y,
            tree_width, tree_height,
            color_node=C["card"], color_edge=C["neuron_e"], depth=3)
        all_tree_leaf_positions.extend(leaves)
        tree_root_positions.append((spec["cx"], root_y))
        ax_bot.text(spec["cx"], tree_y_mid - 0.14, spec["label"], ha="center",
                    fontsize=8.5, color=C["dim"], va="top", zorder=10)

        # Connections from input neurons to tree root
        for (nx, ny) in inp_neurons:
            ax_bot.plot([nx + 0.009, spec["cx"] - tree_width / 2 - 0.005],
                        [ny, root_y],
                        color=C["dim"], lw=0.3, alpha=0.15, zorder=2)

    # "..." between Tree 3 and Tree N
    ax_bot.text(0.565, tree_y_mid + 0.01, "...", fontsize=20, color=C["dim"],
                ha="center", va="center", zorder=10)

    # Tree N
    tn_cx = 0.64
    root_y_n = tree_y_mid + tree_height / 2
    nodes_n, leaves_n = draw_tree_symbol_v2(
        ax_bot, tn_cx, root_y_n,
        tree_width, tree_height,
        color_node=C["card"], color_edge=C["neuron_e"], depth=3)
    all_tree_leaf_positions.extend(leaves_n)
    tree_root_positions.append((tn_cx, root_y_n))
    ax_bot.text(tn_cx, tree_y_mid - 0.14, "Tree N", ha="center",
                fontsize=8.5, color=C["dim"], va="top", zorder=10)
    for (nx, ny) in inp_neurons:
        ax_bot.plot([nx + 0.009, tn_cx - tree_width / 2 - 0.005],
                    [ny, root_y_n],
                    color=C["dim"], lw=0.3, alpha=0.15, zorder=2)

    # ── Aggregation neuron (enhanced) ──
    agg_cx = 0.77
    # Outer decorative ring
    agg_outer = Circle((agg_cx, tree_y_mid), 0.024, facecolor=C["accent"],
                        edgecolor="none", alpha=0.08, zorder=4)
    ax_bot.add_patch(agg_outer)
    draw_neuron(ax_bot, agg_cx, tree_y_mid, r=0.020,
                fc="#1e293b", ec=C["accent"], lw=2.0)
    ax_bot.text(agg_cx, tree_y_mid, "Σ", ha="center", va="center",
                fontsize=12, color=C["accent"], fontweight="bold", zorder=10)
    ax_bot.text(agg_cx, tree_y_mid - 0.14, "集約\n(平均/投票)", ha="center",
                fontsize=8, color=C["dim"], va="top", zorder=10,
                linespacing=1.2)

    # Connect leaf nodes → aggregation (line bundles)
    # Sample some leaf positions per tree
    for spec in tree_specs + [{"cx": tn_cx}]:
        # Leaves near this tree
        tree_leaves = [(lx, ly) for (lx, ly) in all_tree_leaf_positions
                       if abs(lx - spec["cx"]) < tree_width]
        for (lx, ly) in tree_leaves:
            ax_bot.plot([lx, agg_cx - 0.022],
                        [ly, tree_y_mid],
                        color=C["dim"], lw=0.5, alpha=0.25, zorder=2)

    # ── Output neuron (bright + glow) ──
    out_cx_tree = 0.87
    draw_neuron(ax_bot, out_cx_tree, tree_y_mid, r=0.018,
                fc=C["out_fc"], ec=C["out_ec"], lw=1.8,
                glow=True, glow_color=C["out_glow"])
    draw_flow_arrow(ax_bot, (agg_cx + 0.022, tree_y_mid),
                    (out_cx_tree - 0.020, tree_y_mid), lw=1.8)
    ax_bot.text(out_cx_tree, tree_y_mid - 0.14, "予測値 y", ha="center",
                fontsize=9, color=C["dim"], va="top", zorder=10)

    # ════════════════════════════════════════════════════════
    #  TREE XAI ANNOTATIONS
    # ════════════════════════════════════════════════════════

    # --- TreeSHAP ---
    ts = C["game"]
    hl_x1 = tree_specs[0]["cx"] - tree_width / 2 - 0.018
    hl_x2 = tn_cx + tree_width / 2 + 0.018
    hl_y1 = tree_y_mid - tree_height / 2 - 0.030
    hl_y2 = tree_y_mid + tree_height / 2 + 0.030

    # Highlight box around all trees
    tree_hl = FancyBboxPatch(
        (hl_x1, hl_y1), hl_x2 - hl_x1, hl_y2 - hl_y1,
        boxstyle="round,pad=0.008", facecolor=ts, edgecolor=ts,
        linewidth=2.0, linestyle="--", alpha=0.12, zorder=1)
    ax_bot.add_patch(tree_hl)
    ax_bot.add_patch(FancyBboxPatch(
        (hl_x1, hl_y1), hl_x2 - hl_x1, hl_y2 - hl_y1,
        boxstyle="round,pad=0.008", facecolor="none", edgecolor=ts,
        linewidth=2.0, linestyle="--", zorder=7))

    # Star marker on one internal node (Tree 1, 2nd level)
    marker_x = tree_specs[0]["cx"] - tree_width / 8
    marker_y = tree_y_mid + tree_height / 2 - tree_height / 3.5
    ax_bot.plot(marker_x, marker_y, marker="*", markersize=10,
                color=ts, zorder=11)

    ts_label_y = hl_y2 + 0.05
    draw_icon_tree_shapley(ax_bot, (hl_x1 + hl_x2) / 2 - 0.07,
                           ts_label_y + 0.032, ts, fontsize=9, zorder=10)
    ax_bot.text((hl_x1 + hl_x2) / 2, ts_label_y + 0.038,
                "TreeSHAP", ha="center", fontsize=11, color=ts,
                fontweight="bold", zorder=10)
    ax_bot.text((hl_x1 + hl_x2) / 2, ts_label_y + 0.010,
                "ノード分割値・葉ノード値に直接アクセス  O(TLD²)",
                ha="center", fontsize=7.5, color=ts, alpha=0.85, zorder=10)
    ax_bot.annotate("", xy=((hl_x1 + hl_x2) / 2, hl_y2 + 0.005),
                    xytext=((hl_x1 + hl_x2) / 2, ts_label_y),
                    arrowprops=dict(arrowstyle="-|>", color=ts, lw=1.8),
                    zorder=8)

    # --- LIME ---
    lime_y = tree_y_mid - 0.23
    # Scatter dots around input (perturbation samples)
    rng = np.random.RandomState(77)
    for (nx, ny) in inp_neurons[:4]:
        for _ in range(3):
            dx = rng.uniform(-0.012, 0.012)
            dy = rng.uniform(-0.008, 0.008)
            c = Circle((nx + dx, ny + dy), 0.003, facecolor=C["pert"],
                        edgecolor="none", alpha=0.35, zorder=3)
            ax_bot.add_patch(c)
    # Arrow: input → output (black-box call)
    ax_bot.annotate("", xy=(out_cx_tree + 0.02, lime_y),
                    xytext=(inp_cx - 0.02, lime_y),
                    arrowprops=dict(arrowstyle="-|>", color=C["pert"], lw=2.0,
                                   alpha=0.5), zorder=6)
    ax_bot.text(0.44, lime_y + 0.008, "f(x)", ha="center", va="bottom",
                fontsize=8, color=C["pert"], fontstyle="italic", zorder=10)
    # Icon + label
    draw_icon_scatter_line(ax_bot, 0.15, lime_y, C["pert"], s=0.013, zorder=10)
    ax_bot.text(0.44, lime_y - 0.025,
                "LIME: 入力周辺を摂動 → ブラックボックス呼び出し → 局所線形モデルで近似",
                ha="center", fontsize=8.5, color=C["pert"],
                fontweight="bold", zorder=10)

    # --- Permutation Importance ---
    pi_y = lime_y - 0.09
    # Shuffle mark on one input neuron
    shuf_neuron = inp_neurons[1]
    draw_icon_shuffle(ax_bot, shuf_neuron[0] + 0.025, shuf_neuron[1],
                      C["pert"], zorder=10)
    # Arrow
    ax_bot.annotate("", xy=(out_cx_tree + 0.02, pi_y),
                    xytext=(inp_cx - 0.02, pi_y),
                    arrowprops=dict(arrowstyle="-|>", color=C["pert"], lw=2.0,
                                   alpha=0.5), zorder=6)
    ax_bot.text(out_cx_tree - 0.02, pi_y + 0.008, "Δ性能", ha="right",
                va="bottom", fontsize=7.5, color=C["pert"], zorder=10)
    # Icon + label
    ax_bot.text(0.44, pi_y - 0.025,
                "Permutation Importance: 特徴量列をシャッフル → モデル評価 → 性能低下量を測定",
                ha="center", fontsize=8.5, color=C["pert"],
                fontweight="bold", zorder=10)

    # --- PDP / ICE ---
    pdp_y = pi_y - 0.09
    # Mini curve icon near input
    draw_icon_mini_curve(ax_bot, 0.15, pdp_y, C["vis"], s=0.016, zorder=10)
    # Arrow
    ax_bot.annotate("", xy=(out_cx_tree + 0.02, pdp_y),
                    xytext=(inp_cx - 0.02, pdp_y),
                    arrowprops=dict(arrowstyle="-|>", color=C["vis"], lw=2.0,
                                   alpha=0.5), zorder=6)
    ax_bot.text(0.15, pdp_y + 0.018, "x(j) を変動", ha="center",
                fontsize=7, color=C["vis"], zorder=10)
    # Label
    ax_bot.text(0.44, pdp_y - 0.025,
                "PDP / ICE: 1特徴量をグリッド上で変動 → 予測値の応答曲線を記録",
                ha="center", fontsize=8.5, color=C["vis"],
                fontweight="bold", zorder=10)

    # ════════════════════════════════════════════════════════
    #  LEGEND (figure-level, right column) – simplified
    # ════════════════════════════════════════════════════════
    leg_x = 0.91
    fy = 0.94
    fig.text(leg_x, fy, "凡例 / Legend", fontsize=12, color=C["text"],
             fontweight="bold", ha="center", va="top")
    fy -= 0.035

    legend = [
        ("勾配ベース",  C["grad"], "Grad-CAM\nIntegrated Gradients"),
        ("摂動ベース",  C["pert"], "Occlusion, LIME\nPermutation Importance"),
        ("ゲーム理論",  C["game"], "TreeSHAP"),
        ("可視化系",    C["vis"],  "PDP / ICE"),
    ]
    for cat, color, methods in legend:
        fig.patches.append(mpatches.FancyBboxPatch(
            (leg_x - 0.035, fy - 0.008), 0.008, 0.012,
            boxstyle="round,pad=0.002", facecolor=color, edgecolor=color,
            alpha=0.9, transform=fig.transFigure, zorder=10))
        fig.text(leg_x - 0.022, fy, cat, fontsize=9.5, color=color,
                 fontweight="bold", va="center")
        fig.text(leg_x - 0.022, fy - 0.017, methods, fontsize=7,
                 color=C["dim"], va="top", linespacing=1.3)
        fy -= 0.060

    # Mini-icon legend
    fy -= 0.010
    fig.text(leg_x, fy, "手法シンボル", fontsize=10, color=C["text"],
             fontweight="bold", ha="center", va="top")
    fy -= 0.028

    icon_legend = [
        ("Grad-CAM",     C["grad"], "ヒートマップ重畳"),
        ("Integ. Grad",  C["grad"], "∫ 経路積分"),
        ("Occlusion",    C["pert"], "遮蔽窓パッチ"),
        ("TreeSHAP",     C["game"], "φ Shapley値"),
        ("LIME",         C["pert"], "散布+直線 局所近似"),
        ("Perm. Imp.",   C["pert"], "シャッフル交差"),
        ("PDP/ICE",      C["vis"],  "応答曲線プロット"),
    ]
    for name, color, desc in icon_legend:
        fig.text(leg_x - 0.030, fy, name, fontsize=7.5, color=color,
                 fontweight="bold", va="top")
        fig.text(leg_x + 0.012, fy, desc, fontsize=6.5, color=C["dim"],
                 va="top")
        fy -= 0.025

    # ════════════════════════════════════════════════════════
    #  TITLE & FOOTER
    # ════════════════════════════════════════════════════════
    fig.suptitle(
        "XAI手法 × ニューラルネットワーク構造 ─ アクセスポイント俯瞰図",
        fontsize=17, color=C["text"], fontweight="bold", y=0.995)
    fig.text(0.5, 0.975,
             "Explainable AI Methods × Neural Network Architectures — Access Point Overview",
             ha="center", fontsize=10, color=C["dim"], fontstyle="italic")

    # Footer
    fig.text(0.5, 0.008,
             "Fig. XAI-NN  |  7手法 × 2アーキテクチャ (CNN + 決定木アンサンブル)  |"
             "  勾配ベース / 摂動ベース / ゲーム理論 / 可視化系",
             ha="center", fontsize=8, color=C["dim"])

    # ════════════════════════════════════════════════════════
    #  SAVE
    # ════════════════════════════════════════════════════════
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "xai_nn_architecture.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=C["bg"], edgecolor="none", pad_inches=0.3)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


# ══════════════════════════════════════════════════════════
#  REUSABLE BASE DRAWING FUNCTIONS
# ══════════════════════════════════════════════════════════

def _draw_cnn_base(ax, y_mid=0.50, layer_label_y=0.12):
    """Draw CNN architecture base. Returns dict of positions for XAI annotations."""
    ax.add_patch(FancyBboxPatch(
        (-0.02, -0.08), 1.04, 1.13,
        boxstyle="round,pad=0.012", facecolor=C["panel"],
        edgecolor=C["border"], linewidth=1, zorder=0))

    ax.text(0.01, 1.01, "CNN アーキテクチャ",
            fontsize=12, color=C["accent"], fontweight="bold",
            va="bottom", zorder=10)

    # Input image
    inp = draw_input_image(ax, 0.06, y_mid, size=0.09, grid=6)
    ax.text(0.06, layer_label_y, "入力画像\nH×W×C", ha="center",
            fontsize=8, color=C["dim"], va="top", zorder=10)

    # Conv blocks
    conv_specs = [
        {"cx": 0.19, "n": 6,  "w": 0.055, "h": 0.080,
         "label": "Conv1+ReLU\n+Pool", "size": "32@24×24"},
        {"cx": 0.32, "n": 8,  "w": 0.045, "h": 0.065,
         "label": "Conv2+ReLU\n+Pool", "size": "64@12×12"},
        {"cx": 0.45, "n": 10, "w": 0.038, "h": 0.052,
         "label": "Conv3+ReLU\n+Pool", "size": "128@6×6"},
    ]
    conv_points = []
    prev_right = inp["right"]
    for spec in conv_specs:
        pts = draw_feature_map_3d(
            ax, spec["cx"], y_mid, spec["n"], spec["w"], spec["h"],
            offset_x=0.0035, offset_y=0.0035, perspective=True)
        conv_points.append(pts)
        draw_flow_arrow(ax, prev_right, pts["left"], lw=1.3)
        prev_right = pts["right"]
        lx = spec["cx"] + spec["n"] * 0.0035 / 2
        ax.text(lx, layer_label_y, spec["label"], ha="center",
                fontsize=7.5, color=C["dim"], va="top", zorder=10, linespacing=1.2)
        ax.text(lx, layer_label_y - 0.045, spec["size"], ha="center",
                fontsize=7, color=C["accent"], va="top", zorder=10)

    # Flatten
    flat_x = 0.55
    ax.text(flat_x, y_mid, "Flatten", ha="center", va="center",
            fontsize=7.5, color=C["dim"], rotation=90, zorder=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=C["card"],
                      edgecolor=C["border"], linewidth=0.8))
    draw_flow_arrow(ax, prev_right, (flat_x - 0.015, y_mid), lw=1.3)

    # FC layers
    fc_specs = [
        {"cx": 0.62, "n": 7, "label": "FC1 (512)", "spacing": 0.028},
        {"cx": 0.72, "n": 5, "label": "FC2 (128)", "spacing": 0.028},
    ]
    fc_layers = []
    for spec in fc_specs:
        neurons = draw_neuron_column(ax, spec["cx"], y_mid, spec["n"],
                                     spacing=spec["spacing"], r=0.010,
                                     fc=C["neuron"], ec=C["neuron_e"])
        fc_layers.append(neurons)
        ax.text(spec["cx"], layer_label_y, spec["label"], ha="center",
                fontsize=8, color=C["dim"], va="top", zorder=10)

    flat_right = (flat_x + 0.018, y_mid)
    draw_flatten_fan(ax, flat_right, fc_layers[0], color=C["dim"], lw=0.5, alpha=0.25)
    ax.text(flat_x, layer_label_y, "Flatten\n9216→512", ha="center",
            fontsize=7, color=C["accent"], va="top", zorder=10)
    connect_layers_sampled(ax, fc_layers[0], fc_layers[1],
                           sample_rate=0.35, lw=0.4, alpha=0.22)

    # Output neurons
    out_cx = 0.82
    out_neurons = []
    for i in range(3):
        ny = y_mid + 0.040 - i * 0.040
        draw_neuron(ax, out_cx, ny, r=0.013,
                    fc=C["out_fc"], ec=C["out_ec"], lw=1.5,
                    glow=True, glow_color=C["out_glow"])
        out_neurons.append((out_cx, ny))
    connect_layers_sampled(ax, fc_layers[1], out_neurons,
                           sample_rate=0.5, lw=0.4, alpha=0.22)
    ax.text(out_cx, layer_label_y, "出力\n(Class Score)", ha="center",
            fontsize=8, color=C["dim"], va="top", zorder=10, linespacing=1.2)
    ax.text(out_cx + 0.028, y_mid, "softmax", ha="left", va="center",
            fontsize=7, color=C["dim"], zorder=10, rotation=-90)
    for i, (nx, ny) in enumerate(out_neurons):
        ax.text(nx + 0.023, ny, ["cat", "dog", "..."][i], ha="left",
                va="center", fontsize=6.5, color=C["dim"], zorder=10)

    return {
        "inp": inp, "conv_specs": conv_specs, "conv_points": conv_points,
        "fc_layers": fc_layers, "out_neurons": out_neurons,
        "out_cx": out_cx, "y_mid": y_mid, "flat_x": flat_x,
    }


def _draw_tree_base(ax, tree_y_mid=0.52):
    """Draw Tree Ensemble architecture base. Returns dict of positions."""
    ax.add_patch(FancyBboxPatch(
        (-0.02, -0.08), 1.04, 1.13,
        boxstyle="round,pad=0.012", facecolor=C["panel"],
        edgecolor=C["border"], linewidth=1, zorder=0))

    ax.text(0.01, 1.01, "決定木アンサンブル",
            fontsize=12, color=C["accent"], fontweight="bold",
            va="bottom", zorder=10)

    # Input features
    inp_cx = 0.07
    inp_neurons = draw_neuron_column(ax, inp_cx, tree_y_mid, 6,
                                     spacing=0.028, r=0.009,
                                     fc="#1e293b", ec="#4a5568")
    for i, (nx, ny) in enumerate(inp_neurons):
        ax.text(nx - 0.020, ny, ["x1", "x2", "x3", "x4", "...", "xM"][i],
                ha="right", va="center", fontsize=7, color=C["dim"], zorder=10)
    ax.text(inp_cx, tree_y_mid - 0.14, "入力特徴量\nM次元", ha="center",
            fontsize=9, color=C["dim"], va="top", zorder=10)

    # Decision trees
    tree_specs = [
        {"cx": 0.22, "label": "Tree 1"},
        {"cx": 0.35, "label": "Tree 2"},
        {"cx": 0.48, "label": "Tree 3"},
    ]
    tree_width, tree_height = 0.09, 0.11
    all_leaves = []
    tree_root_positions = []

    for spec in tree_specs:
        root_y = tree_y_mid + tree_height / 2
        nodes, leaves = draw_tree_symbol_v2(
            ax, spec["cx"], root_y, tree_width, tree_height,
            color_node=C["card"], color_edge=C["neuron_e"], depth=3)
        all_leaves.extend(leaves)
        tree_root_positions.append((spec["cx"], root_y))
        ax.text(spec["cx"], tree_y_mid - 0.14, spec["label"], ha="center",
                fontsize=8.5, color=C["dim"], va="top", zorder=10)
        for (nx, ny) in inp_neurons:
            ax.plot([nx + 0.009, spec["cx"] - tree_width / 2 - 0.005],
                    [ny, root_y], color=C["dim"], lw=0.3, alpha=0.15, zorder=2)

    ax.text(0.565, tree_y_mid + 0.01, "...", fontsize=20, color=C["dim"],
            ha="center", va="center", zorder=10)

    tn_cx = 0.64
    root_y_n = tree_y_mid + tree_height / 2
    nodes_n, leaves_n = draw_tree_symbol_v2(
        ax, tn_cx, root_y_n, tree_width, tree_height,
        color_node=C["card"], color_edge=C["neuron_e"], depth=3)
    all_leaves.extend(leaves_n)
    tree_root_positions.append((tn_cx, root_y_n))
    ax.text(tn_cx, tree_y_mid - 0.14, "Tree N", ha="center",
            fontsize=8.5, color=C["dim"], va="top", zorder=10)
    for (nx, ny) in inp_neurons:
        ax.plot([nx + 0.009, tn_cx - tree_width / 2 - 0.005],
                [ny, root_y_n], color=C["dim"], lw=0.3, alpha=0.15, zorder=2)

    # Aggregation
    agg_cx = 0.77
    agg_outer = Circle((agg_cx, tree_y_mid), 0.024, facecolor=C["accent"],
                        edgecolor="none", alpha=0.08, zorder=4)
    ax.add_patch(agg_outer)
    draw_neuron(ax, agg_cx, tree_y_mid, r=0.020, fc="#1e293b", ec=C["accent"], lw=2.0)
    ax.text(agg_cx, tree_y_mid, "Σ", ha="center", va="center",
            fontsize=12, color=C["accent"], fontweight="bold", zorder=10)
    ax.text(agg_cx, tree_y_mid - 0.14, "集約\n(平均/投票)", ha="center",
            fontsize=8, color=C["dim"], va="top", zorder=10, linespacing=1.2)

    for spec in tree_specs + [{"cx": tn_cx}]:
        tree_lvs = [(lx, ly) for (lx, ly) in all_leaves
                    if abs(lx - spec["cx"]) < tree_width]
        for (lx, ly) in tree_lvs:
            ax.plot([lx, agg_cx - 0.022], [ly, tree_y_mid],
                    color=C["dim"], lw=0.5, alpha=0.25, zorder=2)

    # Output
    out_cx_tree = 0.87
    draw_neuron(ax, out_cx_tree, tree_y_mid, r=0.018,
                fc=C["out_fc"], ec=C["out_ec"], lw=1.8,
                glow=True, glow_color=C["out_glow"])
    draw_flow_arrow(ax, (agg_cx + 0.022, tree_y_mid),
                    (out_cx_tree - 0.020, tree_y_mid), lw=1.8)
    ax.text(out_cx_tree, tree_y_mid - 0.14, "予測値 y", ha="center",
            fontsize=9, color=C["dim"], va="top", zorder=10)

    return {
        "inp_cx": inp_cx, "inp_neurons": inp_neurons,
        "tree_specs": tree_specs, "tn_cx": tn_cx,
        "tree_width": tree_width, "tree_height": tree_height,
        "all_leaves": all_leaves, "tree_root_positions": tree_root_positions,
        "agg_cx": agg_cx, "out_cx_tree": out_cx_tree,
        "tree_y_mid": tree_y_mid,
    }


# ══════════════════════════════════════════════════════════
#  INDIVIDUAL DIAGRAM GENERATORS  v3 — dual-panel layout
#  Left: architecture schematic  |  Right: method-specific visualization
# ══════════════════════════════════════════════════════════

def _save_individual(fig, name, out_dir):
    out_path = out_dir / f"xai_individual_{name}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor=C["bg"], edgecolor="none", pad_inches=0.3)
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


def _make_dual_fig(title, title_color):
    """Create a 2-panel figure: left=architecture, right=visualization."""
    fig = plt.figure(figsize=(36, 15))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], wspace=0.07)
    ax_l = fig.add_subplot(gs[0])
    ax_r = fig.add_subplot(gs[1])
    fig.patch.set_facecolor(C["bg"])
    for ax in (ax_l, ax_r):
        ax.set_facecolor(C["bg"])
    fig.suptitle(title, fontsize=20, color=title_color,
                 fontweight="bold", y=0.98)
    return fig, ax_l, ax_r


def _setup_schem(ax, title):
    """Setup schematic axes (left panel)."""
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.axis("off")
    ax.set_title(title, fontsize=14, color=C["accent"],
                 fontweight="bold", pad=12, loc="left")
    ax.add_patch(FancyBboxPatch(
        (0.0, 0.0), 1.0, 1.0,
        boxstyle="round,pad=0.015", facecolor=C["panel"],
        edgecolor=C["border"], linewidth=1.2, zorder=0))


def _setup_viz(ax, title, color):
    """Setup visualization axes (right panel)."""
    ax.set_facecolor(C["panel"])
    ax.set_title(title, fontsize=14, color=color,
                 fontweight="bold", pad=12, loc="left")
    for spine in ax.spines.values():
        spine.set_color(C["border"])
        spine.set_linewidth(1.2)
    ax.tick_params(colors=C["dim"], labelsize=9)


def _sbox(ax, x, y, w, h, label, color=C["accent"], fontsize=10,
          sublabel=None, highlight=False, hl_color=None):
    """Draw a labeled box in schematic."""
    ec = (hl_color or color) if highlight else color
    lw = 2.5 if highlight else 1.3
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.012", facecolor=C["card"],
        edgecolor=ec, linewidth=lw, zorder=5,
        linestyle="--" if highlight else "-"))
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fontsize, color=ec, fontweight="bold", zorder=10)
    if sublabel:
        ax.text(x, y - h / 2 - 0.025, sublabel, ha="center", va="top",
                fontsize=7.5, color=C["dim"], zorder=10)


def _sarrow(ax, x1, y1, x2, y2, color=C["dim"], lw=1.8, style="-|>"):
    """Draw arrow in schematic."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=6)


def _formula_text(ax, text, x, y, color, fontsize=11):
    """Draw formula text block in right panel."""
    ax.text(x, y, text, fontsize=fontsize, color=color,
            va="top", ha="left", transform=ax.transAxes, zorder=12,
            linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#0d111788",
                      edgecolor=color, linewidth=1.8, alpha=0.95))


# ─── CNN schematic (for Grad-CAM, IG) ─────────────────
def _schem_cnn(ax, highlight_layer=None, hl_color=None):
    """Simplified CNN flow diagram."""
    _setup_schem(ax, "CNN (畳み込みニューラルネットワーク)")
    specs = [
        (0.08, "入力画像\nH x W x C", 0.11, 0.18, C["accent"]),
        (0.25, "Conv1\n+ ReLU\n+ Pool", 0.12, 0.16, C["neuron_e"]),
        (0.42, "Conv2\n+ ReLU\n+ Pool", 0.12, 0.14, C["neuron_e"]),
        (0.59, "Conv3\n+ ReLU\n+ Pool", 0.12, 0.12, C["neuron_e"]),
        (0.76, "FC\nFlatten\n+ Dense", 0.10, 0.11, C["neuron_e"]),
        (0.92, "出力\ny(c)", 0.09, 0.10, C["out_ec"]),
    ]
    layer_names = ["input", "conv1", "conv2", "conv3", "fc", "output"]
    sizes = ["H x W x C", "32@24x24", "64@12x12", "128@6x6", "512", "C classes"]
    positions = {}
    for i, (x, label, w, h, col) in enumerate(specs):
        hl = (highlight_layer == layer_names[i]) if highlight_layer else False
        _sbox(ax, x, 0.52, w, h, label, col, fontsize=8.5,
              sublabel=sizes[i], highlight=hl, hl_color=hl_color)
        positions[layer_names[i]] = (x, 0.52, w, h)
    for i in range(len(specs) - 1):
        x1 = specs[i][0] + specs[i][2] / 2
        x2 = specs[i + 1][0] - specs[i + 1][2] / 2
        _sarrow(ax, x1, 0.52, x2, 0.52)
    # Explanations below
    notes = [
        (0.08, "入力テンソル:\n画像のピクセル値"),
        (0.25, "畳み込み層:\nフィルタで特徴抽出"),
        (0.59, "特徴マップ:\nフィルタの応答結果"),
        (0.76, "全結合層:\nベクトルに変換"),
        (0.92, "Softmax:\nクラス確率を出力"),
    ]
    for nx, txt in notes:
        ax.text(nx, 0.18, txt, ha="center", fontsize=7.5,
                color=C["dim"], va="top", zorder=10, linespacing=1.3)
    return positions


# ─── Tree schematic (for TreeSHAP) ────────────────────
def _schem_tree(ax, highlight_trees=False, hl_color=None):
    """Simplified tree ensemble flow diagram."""
    _setup_schem(ax, "決定木アンサンブル (Random Forest / GBM)")
    # Input
    _sbox(ax, 0.08, 0.52, 0.11, 0.22, "入力\nx1..xM", C["accent"], 9,
          sublabel="M個の特徴量")
    # Trees
    tree_data = [(0.28, "Tree 1"), (0.43, "Tree 2"), (0.58, "Tree N")]
    tree_xs = [t[0] for t in tree_data]
    for i, (tx, tl) in enumerate(tree_data):
        hl = highlight_trees
        tri = Polygon([(tx, 0.72), (tx - 0.06, 0.35), (tx + 0.06, 0.35)],
                      closed=True, facecolor=C["card"],
                      edgecolor=(hl_color if hl else C["neuron_e"]),
                      linewidth=(2.5 if hl else 1.5),
                      linestyle=("--" if hl else "-"), zorder=5)
        ax.add_patch(tri)
        ax.text(tx, 0.50, tl, ha="center", va="center",
                fontsize=9, color=C["neuron_e"], zorder=10)
        _sarrow(ax, 0.135, 0.52, tx - 0.06, 0.52, lw=0.9)
        if i == 1:
            ax.text(tx + 0.075, 0.52, "...", fontsize=18, color=C["dim"],
                    ha="center", va="center", zorder=10)
    # Tree structure explanation
    ax.text(0.28, 0.25, "各ノードで\n\"x(j) <= t ?\"\nで分岐", ha="center",
            fontsize=7.5, color=C["dim"], va="top", zorder=10, linespacing=1.2)
    ax.text(0.58, 0.25, "葉ノードで\n予測値を出力", ha="center",
            fontsize=7.5, color=C["dim"], va="top", zorder=10, linespacing=1.2)
    # Aggregation
    agg_x = 0.76
    ax.add_patch(Circle((agg_x, 0.52), 0.04, facecolor=C["card"],
                         edgecolor=C["accent"], linewidth=2, zorder=5))
    ax.text(agg_x, 0.52, "Sigma", ha="center", va="center",
            fontsize=11, color=C["accent"], fontweight="bold", zorder=10)
    ax.text(agg_x, 0.25, "集約:\nN本の木の\n平均 / 投票", ha="center",
            fontsize=7.5, color=C["dim"], va="top", zorder=10, linespacing=1.2)
    for tx in tree_xs:
        _sarrow(ax, tx + 0.06, 0.52, agg_x - 0.04, 0.52, lw=0.9)
    # Output
    out_x = 0.92
    ax.add_patch(Circle((out_x, 0.52), 0.035, facecolor=C["out_fc"],
                         edgecolor=C["out_ec"], linewidth=2, zorder=5))
    ax.text(out_x, 0.52, "y", ha="center", va="center",
            fontsize=12, color=C["out_ec"], fontweight="bold", zorder=10)
    ax.text(out_x, 0.25, "最終予測", ha="center",
            fontsize=8, color=C["dim"], va="top", zorder=10)
    _sarrow(ax, agg_x + 0.04, 0.52, out_x - 0.035, 0.52)
    return tree_xs


# ─── Black-box schematic (for model-agnostic methods) ──
def _schem_blackbox(ax, method_name=""):
    """Simplified black-box model for model-agnostic methods."""
    _setup_schem(ax, "任意のモデル (モデル非依存)")
    # Input
    _sbox(ax, 0.12, 0.52, 0.16, 0.22, "入力 x\n(画像 / 表 / ...)",
          C["accent"], 9, sublabel="任意のデータ形式")
    # Black box
    bx, by, bw, bh = 0.50, 0.52, 0.28, 0.30
    ax.add_patch(FancyBboxPatch(
        (bx - bw / 2, by - bh / 2), bw, bh,
        boxstyle="round,pad=0.02", facecolor="#1a1a2e",
        edgecolor=C["dim"], linewidth=3.0, linestyle="--", zorder=5))
    ax.text(bx, by + 0.03, "f(x)", ha="center", va="center",
            fontsize=22, color=C["text"], fontweight="bold", zorder=10)
    ax.text(bx, by - 0.07, "Black Box", ha="center", va="center",
            fontsize=10, color=C["dim"], zorder=10)
    # Output
    _sbox(ax, 0.88, 0.52, 0.14, 0.18, "出力 y\n(予測値)",
          C["out_ec"], 9, sublabel="回帰 / 分類")
    # Arrows
    _sarrow(ax, 0.20, 0.52, bx - bw / 2, 0.52, lw=2.5)
    _sarrow(ax, bx + bw / 2, 0.52, 0.81, 0.52, lw=2.5)
    # Note
    ax.text(0.50, 0.12,
            "CNN / 決定木 / SVM / NN ... 内部構造を問わず適用可能",
            ha="center", fontsize=9.5, color=C["dim"], zorder=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=C["card"],
                      edgecolor=C["border"], linewidth=0.8))


# ═══════════════════════════════════════════════════════
#  RIGHT PANEL VISUALIZATIONS (unique per method)
# ═══════════════════════════════════════════════════════

def _viz_gradcam(ax):
    """Heatmap overlay visualization for Grad-CAM."""
    gc = C["grad"]
    _setup_viz(ax, "Grad-CAM: ヒートマップの生成過程", gc)
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-3.5, 9.5)
    ax.set_aspect("equal")
    ax.axis("off")
    # Step 1: Feature maps A(k) (small grids)
    ax.text(1.0, 9.0, "1. Conv3 の特徴マップ A(k)", fontsize=11,
            color=C["text"], fontweight="bold", zorder=10)
    rng = np.random.RandomState(42)
    for k in range(4):
        data = rng.uniform(0, 1, (4, 4))
        x0 = k * 2.0
        for i in range(4):
            for j in range(4):
                v = data[i, j]
                ax.add_patch(Rectangle((x0 + j * 0.42, 7.5 - i * 0.42),
                             0.40, 0.40, facecolor=plt.cm.viridis(v),
                             edgecolor=C["border"], linewidth=0.3, zorder=5))
        ax.text(x0 + 0.8, 5.8, f"A({k + 1})", ha="center", fontsize=8,
                color=C["dim"], zorder=10)

    # Step 2: Gradient weights a_k
    ax.text(1.0, 5.2, "2. 勾配から重み a_k を計算", fontsize=11,
            color=C["text"], fontweight="bold", zorder=10)
    ax.annotate("", xy=(3.5, 5.0), xytext=(3.5, 5.6),
                arrowprops=dict(arrowstyle="-|>", color=gc, lw=2), zorder=6)
    weights = [0.72, 0.15, 0.08, 0.05]
    for k, w in enumerate(weights):
        x0 = k * 2.0
        bar_w = 1.2 * w / max(weights)
        ax.add_patch(FancyBboxPatch((x0, 4.2), bar_w, 0.5,
                     boxstyle="round,pad=0.02", facecolor=gc, alpha=0.7,
                     edgecolor="none", zorder=5))
        ax.text(x0 + bar_w + 0.15, 4.45, f"a_{k + 1}={w:.2f}",
                fontsize=8, color=gc, va="center", zorder=10)

    # Step 3: Weighted sum → heatmap L
    ax.text(1.0, 3.6, "3. 加重平均 + ReLU → ヒートマップ L", fontsize=11,
            color=C["text"], fontweight="bold", zorder=10)
    ax.annotate("", xy=(3.5, 3.2), xytext=(3.5, 3.5),
                arrowprops=dict(arrowstyle="-|>", color=gc, lw=2), zorder=6)
    # Synthetic heatmap
    hm = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            hm[i, j] = np.exp(-((i - 2.5) ** 2 + (j - 3) ** 2) / 3.5)
    hm += rng.uniform(0, 0.08, (6, 6))
    hm = np.clip(hm, 0, 1)
    for i in range(6):
        for j in range(6):
            ax.add_patch(Rectangle((j * 0.65 + 1.0, 1.5 - i * 0.65 + 0.9),
                         0.63, 0.63, facecolor=plt.cm.jet(hm[i, j]),
                         edgecolor=C["border"], linewidth=0.3, zorder=5))
    ax.text(5.5, 1.6, "= L (Class\n   Activation\n   Map)",
            fontsize=10, color=gc, fontweight="bold", va="center", zorder=10)

    # Formula
    ax.text(0.5, -2.8,
            "数式:  L = ReLU( Sum_k  a_k * A(k) )\n"
            "a_k = (1/Z) Sum_{i,j} dY(c)/dA(k)_{i,j}\n\n"
            "Y(c): 出力クラス c のスコア    A(k): k番目の特徴マップ\n"
            "a_k: 勾配の空間平均 (重み)     L: 最終ヒートマップ",
            fontsize=10, color=gc, va="top", zorder=10, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#0d111788",
                      edgecolor=gc, linewidth=1.8))


def _viz_ig(ax):
    """Interpolation path visualization for Integrated Gradients."""
    ig = C["grad"]
    _setup_viz(ax, "Integrated Gradients: 基準→入力の経路積分", ig)
    ax.axis("off")
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-3.5, 9)
    ax.set_aspect("equal")

    rng = np.random.RandomState(7)
    # Show interpolation steps
    ax.text(0.0, 8.5, "補間経路:  x' + a * (x - x')   a = 0 → 1",
            fontsize=11, color=C["text"], fontweight="bold", zorder=10)
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    base = np.zeros((4, 4))
    target = rng.uniform(0.3, 1.0, (4, 4))
    for ai, alpha in enumerate(alphas):
        img = base + alpha * (target - base)
        x0 = ai * 1.9
        for i in range(4):
            for j in range(4):
                ax.add_patch(Rectangle((x0 + j * 0.42, 6.5 - i * 0.42),
                             0.40, 0.40, facecolor=plt.cm.viridis(img[i, j]),
                             edgecolor=C["border"], linewidth=0.3, zorder=5))
        ax.text(x0 + 0.8, 4.8, f"a={alpha:.2f}", ha="center",
                fontsize=8, color=ig, zorder=10)
        if ai < len(alphas) - 1:
            ax.annotate("", xy=(x0 + 1.75, 5.7), xytext=(x0 + 1.4, 5.7),
                        arrowprops=dict(arrowstyle="-|>", color=ig, lw=1.5),
                        zorder=6)
    ax.text(0.3, 4.3, "x' (黒=基準)", fontsize=8, color=C["dim"], zorder=10)
    ax.text(8.0, 4.3, "x (実入力)", fontsize=8, color=C["dim"], zorder=10)

    # Gradient accumulation
    ax.text(0.0, 3.6, "各ステップの勾配 dF/dx を蓄積:", fontsize=11,
            color=C["text"], fontweight="bold", zorder=10)
    ax.annotate("", xy=(4.5, 3.0), xytext=(4.5, 3.4),
                arrowprops=dict(arrowstyle="-|>", color=ig, lw=2), zorder=6)
    # Attribution map
    attrib = np.abs(rng.randn(4, 4))
    attrib = attrib / attrib.max()
    for i in range(4):
        for j in range(4):
            ax.add_patch(Rectangle((j * 0.65 + 2.5, 1.2 - i * 0.65 + 0.4),
                         0.63, 0.63, facecolor=plt.cm.hot(attrib[i, j]),
                         edgecolor=C["border"], linewidth=0.3, zorder=5))
    ax.text(5.5, 1.5, "= IG(x)\n  帰属マップ\n  (各ピクセルの寄与度)",
            fontsize=10, color=ig, fontweight="bold", va="center", zorder=10)

    # Formula
    ax.text(0.5, -2.5,
            "数式:  IG_i(x) = (x_i - x'_i) * Integral_0^1 (dF/dx_i)(x' + a*(x - x')) da\n\n"
            "x': 基準 (黒画像)   x: 実入力   F: モデル出力\n"
            "a: 補間係数 0→1   dF/dx: 各ピクセルの勾配   IG_i: ピクセルiの寄与",
            fontsize=10, color=ig, va="top", zorder=10, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#0d111788",
                      edgecolor=ig, linewidth=1.8))


def _viz_occlusion(ax):
    """Sliding window sensitivity map for Occlusion."""
    oc = C["pert"]
    _setup_viz(ax, "Occlusion: 遮蔽窓をスライドして感度マップを作成", oc)
    ax.axis("off")
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-3.5, 9)
    ax.set_aspect("equal")

    rng = np.random.RandomState(33)
    # Original image
    ax.text(0.0, 8.5, "1. 入力画像の各領域を遮蔽 (グレーパッチ)",
            fontsize=11, color=C["text"], fontweight="bold", zorder=10)
    base_img = rng.uniform(0.4, 1.0, (5, 5))
    patches = [(0, 0), (0, 2), (2, 1), (3, 3)]
    for pi, (pr, pc) in enumerate(patches):
        x0 = pi * 2.3
        for i in range(5):
            for j in range(5):
                val = base_img[i, j]
                fc = plt.cm.viridis(val)
                if i == pr and j == pc:
                    fc = (0.5, 0.5, 0.5, 1.0)  # grey patch
                ax.add_patch(Rectangle((x0 + j * 0.35, 6.5 - i * 0.35),
                             0.33, 0.33, facecolor=fc,
                             edgecolor=C["border"], linewidth=0.3, zorder=5))
        ax.text(x0 + 0.8, 4.6, f"patch @({pr},{pc})", ha="center",
                fontsize=7, color=oc, zorder=10)

    # Sensitivity map
    ax.text(0.0, 4.0, "2. 出力変化を記録 → 感度マップ S(p)",
            fontsize=11, color=C["text"], fontweight="bold", zorder=10)
    ax.annotate("", xy=(4.0, 3.5), xytext=(4.0, 3.9),
                arrowprops=dict(arrowstyle="-|>", color=oc, lw=2), zorder=6)
    sens = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            sens[i, j] = np.exp(-((i - 2) ** 2 + (j - 2) ** 2) / 3)
    sens += rng.uniform(0, 0.1, (5, 5))
    sens = sens / sens.max()
    for i in range(5):
        for j in range(5):
            ax.add_patch(Rectangle((j * 0.65 + 1.5, 1.5 - i * 0.65 + 0.5),
                         0.63, 0.63, facecolor=plt.cm.hot(sens[i, j]),
                         edgecolor=C["border"], linewidth=0.3, zorder=5))
    ax.text(5.5, 1.8, "S(p): 感度マップ\n暖色 = 遮蔽で\n出力が大きく変化\n= 重要な領域",
            fontsize=10, color=oc, fontweight="bold", va="center", zorder=10)

    # Formula
    ax.text(0.5, -2.5,
            "数式:  S(p) = f(x) - f( x_mask(p) )\n\n"
            "x: 元の入力   x_mask(p): 位置pにグレーパッチ配置\n"
            "f: モデル (内部にアクセス不要)   S(p): 位置pの重要度\n"
            "特徴: モデル非依存。画像データ向き。計算コスト高い",
            fontsize=10, color=oc, va="top", zorder=10, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#0d111788",
                      edgecolor=oc, linewidth=1.8))


def _viz_treeshap(ax):
    """SHAP waterfall / bar chart for TreeSHAP."""
    ts = C["game"]
    _setup_viz(ax, "TreeSHAP: 各特徴量のShapley寄与度", ts)
    features = ["x1 (面積)", "x2 (築年数)", "x3 (駅距離)",
                "x4 (階数)", "x5 (部屋数)", "x6 (築構造)"]
    shap_vals = [0.35, -0.22, 0.18, -0.08, 0.12, -0.05]
    y_pos = np.arange(len(features))
    colors = [ts if v >= 0 else "#ff6b6b" for v in shap_vals]
    ax.barh(y_pos, shap_vals, height=0.6, color=colors, alpha=0.8,
            edgecolor="none", zorder=5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=11, color=C["text"])
    ax.set_xlabel("Shapley value (phi_i)", fontsize=11, color=C["dim"])
    ax.axvline(0, color=C["border"], lw=1.5, zorder=3)
    ax.invert_yaxis()
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color(C["border"])
    ax.tick_params(axis="x", colors=C["dim"], labelsize=9)
    # Value labels
    for i, v in enumerate(shap_vals):
        offset = 0.02 if v >= 0 else -0.02
        ha = "left" if v >= 0 else "right"
        ax.text(v + offset, i, f"{v:+.2f}", va="center", ha=ha,
                fontsize=10, color=C["text"], fontweight="bold", zorder=10)
    # Annotations
    ax.text(0.35, 6.5, "正 (緑) = 予測を上げる\n負 (赤) = 予測を下げる",
            fontsize=9, color=C["dim"], va="top", zorder=10,
            transform=ax.transData)
    # Formula text at bottom
    ax.text(0.02, -0.08,
            "phi_i = Sum_{S} |S|!(M-|S|-1)!/M! * [f(S+{i}) - f(S)]\n"
            "木の内部構造を利用して O(TLD^2) で厳密計算  (T=木数, L=葉数, D=深さ)",
            fontsize=9, color=ts, va="top", zorder=10, linespacing=1.5,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#0d111788",
                      edgecolor=ts, linewidth=1.5))


def _viz_lime(ax):
    """Scatter + local linear fit for LIME."""
    lm = C["pert"]
    _setup_viz(ax, "LIME: 局所近傍のサンプルで線形モデルを学習", lm)
    rng = np.random.RandomState(55)
    n = 40
    # Generate perturbed samples around x
    x_center = 5.0
    xs = x_center + rng.randn(n) * 1.5
    noise = rng.randn(n) * 0.3
    ys = 0.4 * xs + 1.0 + noise + 0.1 * (xs - x_center) ** 2
    # Distance-based weights (closer = larger)
    dists = np.abs(xs - x_center)
    weights = np.exp(-dists ** 2 / 2)
    sizes = 30 + weights * 150
    ax.scatter(xs, ys, s=sizes, c=lm, alpha=0.4, edgecolors="none", zorder=5)
    # Original point
    y_orig = 0.4 * x_center + 1.0
    ax.scatter([x_center], [y_orig], s=200, c="#ff6b6b", edgecolors="white",
               linewidth=2, marker="*", zorder=8, label="x (対象入力)")
    # Local linear fit
    x_line = np.linspace(2, 8, 50)
    y_line = 0.4 * x_line + 1.0
    ax.plot(x_line, y_line, color=lm, lw=2.5, ls="--", zorder=6,
            label="g(x) 局所線形モデル")
    ax.set_xlabel("特徴量空間", fontsize=11, color=C["dim"])
    ax.set_ylabel("モデル出力 f(x)", fontsize=11, color=C["dim"])
    ax.legend(fontsize=9, loc="upper left", facecolor=C["card"],
              edgecolor=C["border"], labelcolor=C["text"])
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    # Annotations
    ax.annotate("近い = 重み大\n(大きいドット)", xy=(x_center + 0.3, y_orig - 0.1),
                xytext=(x_center + 2, y_orig - 1),
                fontsize=9, color=C["text"],
                arrowprops=dict(arrowstyle="-|>", color=lm, lw=1.2), zorder=10)
    ax.annotate("遠い = 重み小\n(小さいドット)", xy=(2.5, 0.4 * 2.5 + 0.8),
                xytext=(1.5, 4.0),
                fontsize=9, color=C["text"],
                arrowprops=dict(arrowstyle="-|>", color=C["dim"], lw=1.0), zorder=10)
    # Formula
    ax.text(0.02, -0.08,
            "xi(x) = argmin_{g} L(f, g, pi_x) + Omega(g)\n"
            "f: 元モデル (black box)   g: 局所線形モデル   pi_x: 近傍重み関数",
            fontsize=9, color=lm, va="top", zorder=10, linespacing=1.5,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#0d111788",
                      edgecolor=lm, linewidth=1.5))


def _viz_pi(ax):
    """Bar chart of feature importance with shuffle illustration for PI."""
    pi_c = C["pert"]
    _setup_viz(ax, "Permutation Importance: シャッフルによる性能低下", pi_c)
    features = ["x1 (面積)", "x2 (築年数)", "x3 (駅距離)",
                "x4 (階数)", "x5 (部屋数)", "x6 (築構造)"]
    importance = [0.32, 0.25, 0.18, 0.10, 0.08, 0.03]
    y_pos = np.arange(len(features))
    ax.barh(y_pos, importance, height=0.6, color=pi_c, alpha=0.75,
            edgecolor="none", zorder=5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=11, color=C["text"])
    ax.set_xlabel("重要度 I_j = score(元) - score(shuffle)", fontsize=11,
                  color=C["dim"])
    ax.invert_yaxis()
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    # Value labels
    for i, v in enumerate(importance):
        ax.text(v + 0.005, i, f"{v:.2f}", va="center", ha="left",
                fontsize=10, color=C["text"], fontweight="bold", zorder=10)
    # Shuffle illustration
    ax.text(0.25, 6.2, "手順: 特徴量 j の列をランダムにシャッフル\n"
            "→ モデルを再評価 → 性能低下量 = 重要度",
            fontsize=9, color=C["dim"], va="top", zorder=10,
            transform=ax.transData)
    # Formula
    ax.text(0.02, -0.08,
            "I_j = s_orig - (1/K) Sum_k s_{j,k}\n"
            "s_orig: 元の性能   s_{j,k}: j列シャッフル後のk回目の性能   K: 反復回数",
            fontsize=9, color=pi_c, va="top", zorder=10, linespacing=1.5,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#0d111788",
                      edgecolor=pi_c, linewidth=1.5))


def _viz_pdp_ice(ax):
    """PDP bold line + ICE thin lines for PDP/ICE."""
    vc = C["vis"]
    _setup_viz(ax, "PDP / ICE: 特徴量と予測の応答曲線", vc)
    rng = np.random.RandomState(99)
    x_vals = np.linspace(0, 10, 50)
    # ICE curves (individual)
    n_ice = 15
    for i in range(n_ice):
        offset = rng.uniform(-1.5, 1.5)
        noise = rng.randn(50) * 0.15
        y_ice = 2.0 + 0.3 * x_vals + offset + noise + 0.05 * np.sin(x_vals)
        ax.plot(x_vals, y_ice, color=vc, alpha=0.15, lw=1.0, zorder=3)
    # PDP curve (average)
    y_pdp = 2.0 + 0.3 * x_vals + 0.05 * np.sin(x_vals)
    ax.plot(x_vals, y_pdp, color=vc, lw=3.5, zorder=6, label="PDP (平均)")
    ax.fill_between(x_vals, y_pdp - 0.5, y_pdp + 0.5,
                    alpha=0.08, color=vc, zorder=2)
    # Highlight one ICE
    y_one = 2.0 + 0.3 * x_vals + 0.8 + rng.randn(50) * 0.1
    ax.plot(x_vals, y_one, color="#ff6b6b", lw=2.0, ls="--", zorder=5,
            label="ICE (個別サンプル)")
    ax.set_xlabel("x(j): 注目する特徴量の値", fontsize=11, color=C["dim"])
    ax.set_ylabel("f(x): モデルの予測値", fontsize=11, color=C["dim"])
    ax.legend(fontsize=10, loc="upper left", facecolor=C["card"],
              edgecolor=C["border"], labelcolor=C["text"])
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    # Annotations
    ax.annotate("各薄線 = 1サンプル\nの応答 (ICE)", xy=(8, 4.8),
                xytext=(6, 6), fontsize=9, color=C["text"],
                arrowprops=dict(arrowstyle="-|>", color=vc, lw=1.0), zorder=10)
    ax.annotate("太線 = 全サンプル\nの平均 (PDP)", xy=(7, y_pdp[35]),
                xytext=(2, 5.5), fontsize=9, color=C["text"],
                arrowprops=dict(arrowstyle="-|>", color=vc, lw=1.5), zorder=10)
    # Formula
    ax.text(0.02, -0.08,
            "PDP: f_S(x_S) = (1/n) Sum_i f(x_S, x_C(i))\n"
            "ICE: f(i)(x_S) = f(x_S, x_C(i))   x_S: 注目特徴量  x_C: 残りの特徴量",
            fontsize=9, color=vc, va="top", zorder=10, linespacing=1.5,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#0d111788",
                      edgecolor=vc, linewidth=1.5))


# ═══════════════════════════════════════════════════════
#  7 GENERATORS
# ═══════════════════════════════════════════════════════

def gen_gradcam(out_dir):
    gc = C["grad"]
    fig, ax_l, ax_r = _make_dual_fig(
        "Grad-CAM — 勾配による特徴マップの重要度可視化 [CNN専用 / 勾配ベース]", gc)
    pos = _schem_cnn(ax_l, highlight_layer="conv3", hl_color=gc)
    # Highlight annotation on left panel
    cx, cy = pos["conv3"][:2]
    ax_l.annotate("Grad-CAM\nはここに\nアクセス", xy=(cx, cy + 0.06),
                  xytext=(cx, 0.90), fontsize=10, color=gc,
                  fontweight="bold", ha="center",
                  arrowprops=dict(arrowstyle="-|>", color=gc, lw=2.0),
                  zorder=12)
    ox = pos["output"][0]
    ax_l.annotate("出力 Y(c)\nから勾配\nを逆伝播", xy=(ox, cy + 0.05),
                  xytext=(ox, 0.90), fontsize=9, color=gc, ha="center",
                  arrowprops=dict(arrowstyle="-|>", color=gc, lw=1.5),
                  zorder=12)
    _viz_gradcam(ax_r)
    return _save_individual(fig, "gradcam", out_dir)


def gen_integrated_gradients(out_dir):
    ig = C["grad"]
    fig, ax_l, ax_r = _make_dual_fig(
        "Integrated Gradients — 経路積分による入力帰属 [微分可能モデル / 勾配ベース]", ig)
    pos = _schem_cnn(ax_l, highlight_layer="input", hl_color=ig)
    ix = pos["input"][0]
    ox = pos["output"][0]
    ax_l.annotate("基準 x'\n(黒画像)", xy=(ix, 0.52 - 0.09),
                  xytext=(ix, 0.15), fontsize=9, color=ig,
                  fontweight="bold", ha="center",
                  arrowprops=dict(arrowstyle="-|>", color=ig, lw=1.5),
                  zorder=12)
    ax_l.annotate("", xy=(ox - 0.045, 0.52), xytext=(ix + 0.055, 0.52),
                  arrowprops=dict(arrowstyle="-|>", color=ig, lw=3.0,
                                  alpha=0.3), zorder=4)
    ax_l.text(0.50, 0.92, "入力→出力の全経路で勾配を積分",
              fontsize=10, color=ig, fontweight="bold", ha="center", zorder=12)
    _viz_ig(ax_r)
    return _save_individual(fig, "integrated_gradients", out_dir)


def gen_occlusion(out_dir):
    oc = C["pert"]
    fig, ax_l, ax_r = _make_dual_fig(
        "Occlusion — 遮蔽窓による感度分析 [モデル非依存 / 摂動ベース]", oc)
    _schem_blackbox(ax_l)
    # Occlusion-specific annotation
    ax_l.text(0.12, 0.88, "入力の一部を\nグレーで遮蔽",
              fontsize=10, color=oc, fontweight="bold", ha="center",
              zorder=12)
    ax_l.add_patch(Rectangle((0.07, 0.66), 0.06, 0.06, facecolor="grey",
                              edgecolor=oc, linewidth=2, zorder=12))
    ax_l.annotate("", xy=(0.12, 0.63), xytext=(0.12, 0.73),
                  arrowprops=dict(arrowstyle="-|>", color=oc, lw=1.5),
                  zorder=12)
    _viz_occlusion(ax_r)
    return _save_individual(fig, "occlusion", out_dir)


def gen_treeshap(out_dir):
    ts = C["game"]
    fig, ax_l, ax_r = _make_dual_fig(
        "TreeSHAP — 木構造を利用したShapley値の厳密計算 [決定木専用 / ゲーム理論]", ts)
    _schem_tree(ax_l, highlight_trees=True, hl_color=ts)
    ax_l.text(0.43, 0.90, "木の内部ノードを辿って\nShapley値を厳密計算",
              fontsize=10, color=ts, fontweight="bold", ha="center", zorder=12)
    _viz_treeshap(ax_r)
    return _save_individual(fig, "treeshap", out_dir)


def gen_lime(out_dir):
    lm = C["pert"]
    fig, ax_l, ax_r = _make_dual_fig(
        "LIME — 局所的な解釈可能モデルによる説明 [モデル非依存 / 摂動ベース]", lm)
    _schem_blackbox(ax_l)
    ax_l.text(0.50, 0.90, "入力近傍を摂動して\nf(x) を繰り返し呼び出し",
              fontsize=10, color=lm, fontweight="bold", ha="center", zorder=12)
    # Perturbation dots
    rng = np.random.RandomState(88)
    for _ in range(12):
        dx = rng.uniform(-0.06, 0.06)
        dy = rng.uniform(-0.08, 0.08)
        ax_l.add_patch(Circle((0.12 + dx, 0.52 + dy), 0.012,
                               facecolor=lm, edgecolor="none",
                               alpha=0.3, zorder=3))
    _viz_lime(ax_r)
    return _save_individual(fig, "lime", out_dir)


def gen_permutation_importance(out_dir):
    pi_c = C["pert"]
    fig, ax_l, ax_r = _make_dual_fig(
        "Permutation Importance — 特徴量シャッフルによる重要度 [モデル非依存 / 摂動ベース]",
        pi_c)
    _schem_blackbox(ax_l)
    ax_l.text(0.50, 0.90, "特徴量 j の列を\nランダムにシャッフル",
              fontsize=10, color=pi_c, fontweight="bold", ha="center", zorder=12)
    # Shuffle arrows
    ax_l.annotate("", xy=(0.08, 0.70), xytext=(0.16, 0.76),
                  arrowprops=dict(arrowstyle="<->", color=pi_c, lw=2.0),
                  zorder=12)
    ax_l.text(0.12, 0.80, "shuffle", fontsize=8, color=pi_c,
              ha="center", zorder=12)
    _viz_pi(ax_r)
    return _save_individual(fig, "permutation_importance", out_dir)


def gen_pdp_ice(out_dir):
    vc = C["vis"]
    fig, ax_l, ax_r = _make_dual_fig(
        "PDP / ICE — 部分依存プロットと個別条件付き期待値 [モデル非依存 / 可視化系]",
        vc)
    _schem_blackbox(ax_l)
    ax_l.text(0.50, 0.90, "1つの特徴量 x(j) を\n系統的に変動させて応答を記録",
              fontsize=10, color=vc, fontweight="bold", ha="center", zorder=12)
    # Mini curve in left panel
    ax_l.annotate("x(j) を変動", xy=(0.12, 0.66),
                  xytext=(0.12, 0.80), fontsize=9, color=vc,
                  fontweight="bold", ha="center",
                  arrowprops=dict(arrowstyle="-|>", color=vc, lw=1.5),
                  zorder=12)
    _viz_pdp_ice(ax_r)
    return _save_individual(fig, "pdp_ice", out_dir)


def generate_individual_diagrams():
    """Generate 7 individual XAI method diagrams."""
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    print("Generating individual XAI method diagrams...")

    generators = [
        gen_gradcam,
        gen_integrated_gradients,
        gen_occlusion,
        gen_treeshap,
        gen_lime,
        gen_permutation_importance,
        gen_pdp_ice,
    ]
    paths = []
    for gen_fn in generators:
        paths.append(gen_fn(out_dir))

    print(f"Done: {len(paths)} individual diagrams generated.")
    return paths


# ═══════════════════════════════════════════════════════
#  ARCHITECTURE-FOCUSED DIAGRAMS (detailed NN structure + XAI overlay)
# ═══════════════════════════════════════════════════════

def _make_arch_fig(title, title_color):
    """Create a single-panel architecture figure (landscape)."""
    fig, ax = plt.subplots(1, 1, figsize=(26, 13))
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.08, 1.05)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.suptitle(title, fontsize=18, color=title_color,
                 fontweight="bold", y=0.99)
    return fig, ax


def _save_arch(fig, name, out_dir):
    out_path = out_dir / f"xai_arch_{name}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor=C["bg"], edgecolor="none", pad_inches=0.2)
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


def arch_gradcam(out_dir):
    gc = C["grad"]
    fig, ax = _make_arch_fig("Grad-CAM  [CNN / 勾配ベース]", gc)
    d = _draw_cnn_base(ax)
    y_mid = d["y_mid"]
    conv3_c = d["conv_points"][2]["center"]
    front_xy = d["conv_points"][2]["front_xy"]
    front_wh = d["conv_points"][2]["front_wh"]
    out_top = d["out_neurons"][0]
    c3 = d["conv_specs"][2]
    # Heatmap on Conv3
    draw_heatmap_overlay(ax, front_xy[0], front_xy[1],
                         front_wh[0], front_wh[1], zorder=8)
    # Gradient backward arrow
    gc_y = 0.85
    ax.annotate("", xy=(conv3_c[0], gc_y), xytext=(out_top[0], gc_y),
                arrowprops=dict(arrowstyle="-|>", color=gc, lw=3), zorder=8)
    ax.plot([conv3_c[0], conv3_c[0]], [conv3_c[1] + 0.04, gc_y],
            color=gc, lw=2, ls="--", alpha=0.7, zorder=7)
    ax.plot([out_top[0], out_top[0]], [out_top[1] + 0.02, gc_y],
            color=gc, lw=2, ls="--", alpha=0.7, zorder=7)
    # Highlight Conv3
    ax.add_patch(FancyBboxPatch(
        (c3["cx"] - c3["w"] / 2 - 0.01, y_mid - c3["h"] / 2 - 0.02),
        c3["w"] + c3["n"] * 0.0035 + 0.02, c3["h"] + 0.04,
        boxstyle="round,pad=0.005", facecolor="none", edgecolor=gc,
        linewidth=2.5, linestyle="--", zorder=7))
    mid_x = (conv3_c[0] + out_top[0]) / 2
    draw_icon_heatmap(ax, mid_x - 0.08, gc_y + 0.04, s=0.018, zorder=10)
    ax.text(mid_x, gc_y + 0.045, "Grad-CAM: 勾配逆伝播 → 特徴マップ加重平均",
            ha="center", fontsize=13, color=gc, fontweight="bold", zorder=10)
    ax.text(0.5, -0.06,
            "a_k = GAP(dY(c)/dA(k))    L = ReLU(Sum_k a_k * A(k))",
            ha="center", fontsize=11, color=gc, zorder=10)
    return _save_arch(fig, "gradcam", out_dir)


def arch_ig(out_dir):
    ig = C["grad"]
    fig, ax = _make_arch_fig("Integrated Gradients  [CNN / 勾配ベース]", ig)
    d = _draw_cnn_base(ax)
    y_mid, out_cx, inp = d["y_mid"], d["out_cx"], d["inp"]
    ig_y = y_mid - 0.18
    ax.annotate("", xy=(out_cx + 0.02, ig_y), xytext=(0.04, ig_y),
                arrowprops=dict(arrowstyle="-|>", color=ig, lw=5, alpha=0.5),
                zorder=6)
    draw_icon_integral(ax, 0.25, ig_y + 0.003, ig, fontsize=18, zorder=10)
    ax.text(0.06, ig_y + 0.025, "x' (基準: a=0)", ha="left",
            fontsize=10, color=ig, zorder=10)
    ax.text(out_cx - 0.03, ig_y + 0.025, "x (実入力: a=1)", ha="right",
            fontsize=10, color=ig, zorder=10)
    ax.text(0.44, ig_y - 0.03,
            "基準 x' → 実入力 x を補間しながら勾配を経路積分",
            ha="center", fontsize=12, color=ig, fontweight="bold", zorder=10)
    ax.text(0.5, -0.06,
            "IG_i(x) = (x_i - x'_i) * Integral_0^1 (dF/dx_i)(x' + a*(x-x')) da",
            ha="center", fontsize=11, color=ig, zorder=10)
    return _save_arch(fig, "integrated_gradients", out_dir)


def arch_occlusion(out_dir):
    oc = C["pert"]
    fig, ax = _make_arch_fig("Occlusion  [モデル非依存 / 摂動ベース]", oc)
    d = _draw_cnn_base(ax)
    y_mid, out_cx, inp = d["y_mid"], d["out_cx"], d["inp"]
    draw_occlusion_patch(ax, inp["x0"], inp["y0"], inp["size"], zorder=9)
    oc_y = y_mid - 0.18
    ax.annotate("", xy=(out_cx + 0.02, oc_y), xytext=(0.04, oc_y),
                arrowprops=dict(arrowstyle="-|>", color=oc, lw=3,
                               linestyle="dashed", alpha=0.7), zorder=6)
    draw_icon_grey_patch(ax, 0.14, oc_y, s=0.018, zorder=10)
    ax.text(0.44, oc_y + 0.015, "f(x): ブラックボックス呼び出し",
            ha="center", fontsize=11, color=oc, zorder=10)
    ax.text(0.44, oc_y - 0.03,
            "入力の一部をグレーで遮蔽 → 出力変化を観測 → 感度マップ",
            ha="center", fontsize=12, color=oc, fontweight="bold", zorder=10)
    ax.text(0.5, -0.06,
            "S(p) = f(x) - f(x_mask(p))    ※モデル内部にアクセス不要",
            ha="center", fontsize=11, color=oc, zorder=10)
    return _save_arch(fig, "occlusion", out_dir)


def arch_treeshap(out_dir):
    ts = C["game"]
    fig, ax = _make_arch_fig("TreeSHAP  [決定木専用 / ゲーム理論]", ts)
    d = _draw_tree_base(ax)
    tm = d["tree_y_mid"]
    tw, th = d["tree_width"], d["tree_height"]
    t_specs = d["tree_specs"]
    hl_x1 = t_specs[0]["cx"] - tw / 2 - 0.02
    hl_x2 = d["tn_cx"] + tw / 2 + 0.02
    hl_y1 = tm - th / 2 - 0.03
    hl_y2 = tm + th / 2 + 0.03
    ax.add_patch(FancyBboxPatch(
        (hl_x1, hl_y1), hl_x2 - hl_x1, hl_y2 - hl_y1,
        boxstyle="round,pad=0.008", facecolor=ts, edgecolor=ts,
        linewidth=2.5, linestyle="--", alpha=0.12, zorder=1))
    ax.add_patch(FancyBboxPatch(
        (hl_x1, hl_y1), hl_x2 - hl_x1, hl_y2 - hl_y1,
        boxstyle="round,pad=0.008", facecolor="none", edgecolor=ts,
        linewidth=2.5, linestyle="--", zorder=7))
    for spec in t_specs:
        mx = spec["cx"] - tw / 8
        my = tm + th / 2 - th / 3.5
        ax.plot(mx, my, marker="*", markersize=14, color=ts, zorder=11)
    lbl_y = hl_y2 + 0.05
    draw_icon_tree_shapley(ax, (hl_x1 + hl_x2) / 2 - 0.08,
                           lbl_y + 0.035, ts, fontsize=12, zorder=10)
    ax.text((hl_x1 + hl_x2) / 2, lbl_y + 0.045,
            "TreeSHAP: 木の内部構造を辿ってShapley値を厳密計算",
            ha="center", fontsize=13, color=ts, fontweight="bold", zorder=10)
    ax.text(0.5, -0.06,
            "phi_i = Sum_S |S|!(M-|S|-1)!/M! * [f(S+{i}) - f(S)]    O(TLD^2)",
            ha="center", fontsize=11, color=ts, zorder=10)
    return _save_arch(fig, "treeshap", out_dir)


def arch_lime(out_dir):
    lm = C["pert"]
    fig, ax = _make_arch_fig("LIME  [モデル非依存 / 摂動ベース]", lm)
    d = _draw_tree_base(ax)
    tm = d["tree_y_mid"]
    inp_cx, inp_neurons = d["inp_cx"], d["inp_neurons"]
    out_cx_tree = d["out_cx_tree"]
    rng = np.random.RandomState(77)
    for (nx, ny) in inp_neurons[:4]:
        for _ in range(6):
            dx = rng.uniform(-0.018, 0.018)
            dy = rng.uniform(-0.012, 0.012)
            ax.add_patch(Circle((nx + dx, ny + dy), 0.005, facecolor=lm,
                                edgecolor="none", alpha=0.30, zorder=3))
    lime_y = tm - 0.22
    ax.annotate("", xy=(out_cx_tree + 0.02, lime_y),
                xytext=(inp_cx - 0.02, lime_y),
                arrowprops=dict(arrowstyle="-|>", color=lm, lw=3, alpha=0.6),
                zorder=6)
    draw_icon_scatter_line(ax, 0.15, lime_y, lm, s=0.018, zorder=10)
    ax.text(0.44, lime_y + 0.015, "f(x): ブラックボックスを繰り返し呼び出し",
            ha="center", fontsize=11, color=lm, zorder=10)
    ax.text(0.44, lime_y - 0.03,
            "入力近傍を摂動 → f(x) で評価 → 局所線形モデル g で近似",
            ha="center", fontsize=12, color=lm, fontweight="bold", zorder=10)
    ax.text(0.5, -0.06,
            "xi(x) = argmin_g L(f, g, pi_x) + Omega(g)    ※任意のモデルに適用可",
            ha="center", fontsize=11, color=lm, zorder=10)
    return _save_arch(fig, "lime", out_dir)


def arch_pi(out_dir):
    pi_c = C["pert"]
    fig, ax = _make_arch_fig(
        "Permutation Importance  [モデル非依存 / 摂動ベース]", pi_c)
    d = _draw_tree_base(ax)
    tm = d["tree_y_mid"]
    inp_cx, inp_neurons = d["inp_cx"], d["inp_neurons"]
    out_cx_tree = d["out_cx_tree"]
    for neuron in inp_neurons[1:4]:
        draw_icon_shuffle(ax, neuron[0] + 0.03, neuron[1], pi_c,
                          s=0.014, zorder=10)
    pi_y = tm - 0.22
    ax.annotate("", xy=(out_cx_tree + 0.02, pi_y),
                xytext=(inp_cx - 0.02, pi_y),
                arrowprops=dict(arrowstyle="-|>", color=pi_c, lw=3, alpha=0.6),
                zorder=6)
    ax.text(out_cx_tree - 0.02, pi_y + 0.015, "Delta score",
            ha="right", fontsize=10, color=pi_c, zorder=10)
    ax.text(0.44, pi_y - 0.03,
            "特徴量 j をシャッフル → モデル再評価 → 性能低下量 = 重要度",
            ha="center", fontsize=12, color=pi_c, fontweight="bold", zorder=10)
    ax.text(0.5, -0.06,
            "I_j = s_orig - (1/K) Sum_k s_{j,k}    ※任意のモデルに適用可",
            ha="center", fontsize=11, color=pi_c, zorder=10)
    return _save_arch(fig, "permutation_importance", out_dir)


def arch_pdp_ice(out_dir):
    vc = C["vis"]
    fig, ax = _make_arch_fig(
        "PDP / ICE  [モデル非依存 / 可視化系]", vc)
    d = _draw_tree_base(ax)
    tm = d["tree_y_mid"]
    inp_cx, inp_neurons = d["inp_cx"], d["inp_neurons"]
    out_cx_tree = d["out_cx_tree"]
    draw_icon_mini_curve(ax, 0.15, tm - 0.22, vc, s=0.022, zorder=10)
    pdp_y = tm - 0.22
    ax.annotate("", xy=(out_cx_tree + 0.02, pdp_y),
                xytext=(inp_cx - 0.02, pdp_y),
                arrowprops=dict(arrowstyle="-|>", color=vc, lw=3, alpha=0.6),
                zorder=6)
    ax.text(0.15, pdp_y + 0.03, "x(j) を系統的に変動",
            ha="center", fontsize=10, color=vc, zorder=10)
    ax.text(0.44, pdp_y - 0.03,
            "1つの特徴量を変動 → 予測値の応答曲線を記録 (PDP=平均, ICE=個別)",
            ha="center", fontsize=12, color=vc, fontweight="bold", zorder=10)
    ax.text(0.5, -0.06,
            "PDP: f_S(x_S) = (1/n) Sum_i f(x_S, x_C(i))    ICE: f(i)(x_S) = f(x_S, x_C(i))",
            ha="center", fontsize=11, color=vc, zorder=10)
    return _save_arch(fig, "pdp_ice", out_dir)


def generate_arch_diagrams():
    """Generate 7 architecture-focused XAI diagrams (detailed NN structure)."""
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    print("Generating architecture-focused diagrams...")
    generators = [
        arch_gradcam, arch_ig, arch_occlusion, arch_treeshap,
        arch_lime, arch_pi, arch_pdp_ice,
    ]
    paths = []
    for gen_fn in generators:
        paths.append(gen_fn(out_dir))
    print(f"Done: {len(paths)} architecture diagrams generated.")
    return paths


if __name__ == "__main__":
    generate_diagram()
    generate_individual_diagrams()
    generate_arch_diagrams()
