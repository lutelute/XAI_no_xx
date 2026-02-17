"""
電力系統の系統図を生成する。
10母線系統のトポロジーと正常/異常状態を図示。
"""

import matplotlib
matplotlib.use('Agg')

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.style.use('dark_background')
plt.rcParams['font.family'] = ['Hiragino Sans', 'sans-serif']

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# 系統図 1: 10母線系統トポロジー
# ============================================================
def draw_system_topology():
    fig, ax = plt.subplots(1, 1, figsize=(14, 10), dpi=150)

    # 母線の位置 (10母線をリング状に配置)
    bus_positions = {
        1: (3, 9),    # 発電機1 (Slack)
        2: (11, 9),   # 発電機2
        3: (1, 6.5),  # 負荷
        4: (5, 6.5),
        5: (9, 6.5),
        6: (13, 6.5),
        7: (1, 3.5),
        8: (5, 3.5),
        9: (9, 3.5),
        10: (13, 3.5),
    }

    # 送電線の接続 (10本)
    lines = [
        (1, 3), (1, 4), (2, 5), (2, 6),
        (3, 7), (4, 5), (4, 8), (5, 9),
        (6, 10), (7, 8),
    ]

    # 送電線を描画
    for idx, (b1, b2) in enumerate(lines):
        x1, y1 = bus_positions[b1]
        x2, y2 = bus_positions[b2]
        ax.plot([x1, x2], [y1, y2], '-', color='#4a9eff', linewidth=2.5, alpha=0.7, zorder=1)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.25, f'L{idx+1}', fontsize=7, color='#6ab0ff',
                ha='center', va='bottom', alpha=0.8)

    # 母線を描画
    for bus_id, (x, y) in bus_positions.items():
        if bus_id <= 2:
            # 発電機母線
            color = '#ff6b6b'
            circle = plt.Circle((x, y), 0.55, color=color, alpha=0.9, zorder=3)
            ax.add_patch(circle)
            ax.text(x, y, f'G{bus_id}', fontsize=11, fontweight='bold',
                    ha='center', va='center', color='white', zorder=4)
            # 発電機シンボル
            ax.text(x, y + 0.9, f'母線{bus_id}\n(発電機)', fontsize=8,
                    ha='center', va='bottom', color='#ffaaaa')
        else:
            # 負荷母線
            color = '#4ecdc4'
            rect = FancyBboxPatch((x - 0.45, y - 0.35), 0.9, 0.7,
                                   boxstyle="round,pad=0.1",
                                   facecolor=color, alpha=0.9, zorder=3)
            ax.add_patch(rect)
            ax.text(x, y, f'{bus_id}', fontsize=11, fontweight='bold',
                    ha='center', va='center', color='white', zorder=4)
            # 負荷シンボル（下向き矢印）
            ax.annotate('', xy=(x, y - 0.6), xytext=(x, y - 0.35),
                       arrowprops=dict(arrowstyle='->', color='#aaa', lw=1.5), zorder=2)
            ax.text(x, y - 0.8, f'負荷', fontsize=7, ha='center', color='#aadddd')

    # 凡例
    gen_patch = mpatches.Patch(color='#ff6b6b', label='発電機母線 (PV制御)')
    load_patch = mpatches.Patch(color='#4ecdc4', label='負荷母線 (PQ負荷)')
    line_patch = mpatches.Patch(color='#4a9eff', label='送電線 (L1-L10)')
    ax.legend(handles=[gen_patch, load_patch, line_patch],
              loc='lower right', fontsize=10, framealpha=0.8,
              facecolor='#1a1a2e', edgecolor='#555')

    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(1.5, 10.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('10母線電力系統モデル — トポロジー図', fontsize=16, pad=20)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'system_topology.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"系統図保存: {path}")


# ============================================================
# 系統図 2: 正常 vs 異常状態の比較
# ============================================================
def draw_normal_vs_anomaly():
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=150)

    bus_positions = {
        1: (3, 9), 2: (11, 9), 3: (1, 6.5), 4: (5, 6.5),
        5: (9, 6.5), 6: (13, 6.5), 7: (1, 3.5), 8: (5, 3.5),
        9: (9, 3.5), 10: (13, 3.5),
    }
    lines = [
        (1, 3), (1, 4), (2, 5), (2, 6),
        (3, 7), (4, 5), (4, 8), (5, 9),
        (6, 10), (7, 8),
    ]

    scenarios = [
        {
            'title': '正常運転状態',
            'subtitle': '全母線電圧 0.95-1.05 pu\n全線路負荷率 20-70%',
            'line_colors': ['#4a9eff'] * 10,
            'line_widths': [2.5] * 10,
            'bus_voltages': [1.02, 1.01, 0.99, 0.98, 0.98, 0.97, 0.97, 0.96, 0.96, 0.95],
            'line_loadings': [45, 42, 48, 40, 38, 50, 44, 46, 42, 40],
            'alert_buses': [],
            'tripped_lines': [],
        },
        {
            'title': '異常1: 線路開放 (N-1事故)',
            'subtitle': 'L6 (母線4-5間) が開放\n→ 隣接線路 L5, L7 に過負荷',
            'line_colors': ['#4a9eff', '#4a9eff', '#4a9eff', '#4a9eff',
                           '#ff4444', '#666666', '#ff4444', '#4a9eff',
                           '#4a9eff', '#4a9eff'],
            'line_widths': [2.5, 2.5, 2.5, 2.5, 5, 1, 5, 2.5, 2.5, 2.5],
            'bus_voltages': [1.02, 1.01, 0.95, 0.93, 0.94, 0.97, 0.93, 0.91, 0.95, 0.95],
            'line_loadings': [45, 42, 48, 40, 85, 0, 88, 46, 42, 40],
            'alert_buses': [4, 8],
            'tripped_lines': [5],
        },
        {
            'title': '異常2: 発電機脱落',
            'subtitle': 'G2 (母線2) がトリップ\n→ 全系統で電圧低下・電力不均衡',
            'line_colors': ['#4a9eff', '#4a9eff', '#ffaa00', '#ffaa00',
                           '#4a9eff', '#4a9eff', '#4a9eff', '#ffaa00',
                           '#ffaa00', '#4a9eff'],
            'line_widths': [2.5, 2.5, 3.5, 3.5, 2.5, 2.5, 2.5, 3.5, 3.5, 2.5],
            'bus_voltages': [1.02, 0.0, 0.95, 0.94, 0.88, 0.86, 0.94, 0.92, 0.85, 0.83],
            'line_loadings': [50, 48, 72, 75, 42, 55, 50, 70, 78, 44],
            'alert_buses': [5, 6, 9, 10],
            'tripped_lines': [],
        },
        {
            'title': '異常3: 電圧崩壊前兆',
            'subtitle': '無効電力不足による広域電圧低下\n→ 複数母線で電圧 < 0.90 pu',
            'line_colors': ['#ffaa00'] * 10,
            'line_widths': [3.5] * 10,
            'bus_voltages': [0.98, 0.97, 0.88, 0.86, 0.85, 0.84, 0.82, 0.80, 0.79, 0.78],
            'line_loadings': [60, 58, 65, 62, 55, 68, 60, 70, 72, 55],
            'alert_buses': [7, 8, 9, 10],
            'tripped_lines': [],
        },
    ]

    for ax, scenario in zip(axes.flat, scenarios):
        # 送電線
        for idx, (b1, b2) in enumerate(lines):
            x1, y1 = bus_positions[b1]
            x2, y2 = bus_positions[b2]
            style = '--' if idx in scenario['tripped_lines'] else '-'
            ax.plot([x1, x2], [y1, y2], style,
                    color=scenario['line_colors'][idx],
                    linewidth=scenario['line_widths'][idx],
                    alpha=0.8, zorder=1)
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            loading = scenario['line_loadings'][idx]
            load_color = '#ff4444' if loading > 80 else '#ffaa00' if loading > 60 else '#aaaaaa'
            ax.text(mx, my + 0.3, f'{loading}%', fontsize=6, color=load_color,
                    ha='center', va='bottom')

        # 母線
        for bus_id, (x, y) in bus_positions.items():
            v = scenario['bus_voltages'][bus_id - 1]
            is_alert = bus_id in scenario['alert_buses']

            if bus_id <= 2:
                if v < 0.1:  # トリップ
                    color = '#555555'
                    label = 'X'
                else:
                    color = '#ff6b6b'
                    label = f'G{bus_id}'
                circle = plt.Circle((x, y), 0.5, color=color,
                                   alpha=0.9, zorder=3,
                                   edgecolor='#ff0000' if is_alert else 'none',
                                   linewidth=3 if is_alert else 0)
                ax.add_patch(circle)
                ax.text(x, y, label, fontsize=10, fontweight='bold',
                        ha='center', va='center', color='white', zorder=4)
            else:
                if v < 0.90:
                    color = '#ff6b6b'
                elif v < 0.95:
                    color = '#ffaa00'
                else:
                    color = '#4ecdc4'
                rect = FancyBboxPatch((x - 0.4, y - 0.3), 0.8, 0.6,
                                       boxstyle="round,pad=0.1",
                                       facecolor=color, alpha=0.9, zorder=3,
                                       edgecolor='#ff0000' if is_alert else 'none',
                                       linewidth=3 if is_alert else 0)
                ax.add_patch(rect)
                ax.text(x, y, f'{bus_id}', fontsize=10, fontweight='bold',
                        ha='center', va='center', color='white', zorder=4)

            # 電圧表示
            if v > 0.1:
                v_color = '#ff4444' if v < 0.90 else '#ffaa00' if v < 0.95 else '#aadddd'
                ax.text(x, y - 0.7, f'{v:.2f}pu', fontsize=7, ha='center',
                        color=v_color, zorder=4)

        ax.set_xlim(-0.5, 14.5)
        ax.set_ylim(2.5, 10.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(scenario['title'], fontsize=13, fontweight='bold', pad=10)
        ax.text(7, 2.8, scenario['subtitle'], fontsize=9, ha='center',
                color='#aaaaaa', style='italic')

    # 凡例
    fig.text(0.5, 0.02,
             '母線色: 緑=正常(V>0.95) / 黄=注意(0.90<V<0.95) / 赤=危険(V<0.90)　　'
             '線路: 太さ=負荷率 / 赤=過負荷(>80%) / 黄=高負荷(>60%)',
             fontsize=9, ha='center', color='#999999')

    fig.suptitle('10母線系統 — 正常運転 vs 異常パターン', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    path = os.path.join(RESULTS_DIR, 'system_normal_vs_anomaly.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"正常/異常比較図保存: {path}")


# ============================================================
# 系統図 3: Case 2 電圧安定性の概念図
# ============================================================
def draw_voltage_stability_concept():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # 左: PV曲線
    ax = axes[0]
    P = np.linspace(0, 1.2, 200)
    V_nose = np.sqrt(np.clip(1 - P**2 * 0.8, 0, None))
    V_upper = np.where(P < 0.95, V_nose + 0.05 * np.sin(P * np.pi), V_nose)
    V_upper = np.clip(V_upper, 0, 1.05)

    ax.plot(P[:160], V_upper[:160], '-', color='#4ecdc4', linewidth=2.5, label='運転領域 (上側)')
    ax.plot(P[155:], V_upper[155:], '--', color='#ff6b6b', linewidth=2, label='不安定領域 (下側)')

    # 運転点
    ax.plot(0.6, 0.92, 'o', color='#4a9eff', markersize=12, zorder=5)
    ax.annotate('現在の運転点', xy=(0.6, 0.92), xytext=(0.3, 0.75),
               fontsize=10, color='#4a9eff',
               arrowprops=dict(arrowstyle='->', color='#4a9eff', lw=1.5))

    # 崩壊点
    ax.plot(0.95, 0.55, 's', color='#ff6b6b', markersize=12, zorder=5)
    ax.annotate('電圧崩壊点\n(ノーズポイント)', xy=(0.95, 0.55), xytext=(0.75, 0.35),
               fontsize=10, color='#ff6b6b',
               arrowprops=dict(arrowstyle='->', color='#ff6b6b', lw=1.5))

    # マージン矢印
    ax.annotate('', xy=(0.95, 0.88), xytext=(0.6, 0.88),
               arrowprops=dict(arrowstyle='<->', color='#ffaa00', lw=2))
    ax.text(0.775, 0.91, '電圧安定性余裕\n(予測対象)', fontsize=9, ha='center',
            color='#ffaa00', fontweight='bold')

    ax.set_xlabel('有効電力 P [pu]', fontsize=11)
    ax.set_ylabel('電圧 V [pu]', fontsize=11)
    ax.set_title('PV曲線と電圧安定性余裕', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xlim(-0.05, 1.3)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.2)

    # 右: 特徴量と余裕の関係概念
    ax = axes[1]
    features = ['総負荷 MW', '無効電力\n予備 Mvar', '最大線路\n負荷率 %',
                '最弱母線\n電圧', '発電機数', '調相設備\nMvar']
    effects = [-0.8, 0.6, -0.5, 0.4, 0.3, 0.5]
    colors = ['#ff6b6b' if e < 0 else '#4ecdc4' for e in effects]

    bars = ax.barh(range(len(features)), effects, color=colors, alpha=0.8, height=0.6)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=10)
    ax.set_xlabel('電圧安定性余裕への影響\n(←悪化 | 改善→)', fontsize=11)
    ax.set_title('特徴量が余裕に与える影響 (概念図)', fontsize=13, fontweight='bold')
    ax.axvline(x=0, color='white', linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.2, axis='x')

    # 注釈
    ax.text(0.95, 0.05, '※ 概念的な方向性を示す\n　実際の影響度はXAI分析を参照',
            transform=ax.transAxes, fontsize=8, color='#888888',
            ha='right', va='bottom')

    fig.suptitle('Case 2: 電圧安定性限界予測 — 問題設定', fontsize=15, y=1.02)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'voltage_stability_concept.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"電圧安定性概念図保存: {path}")


if __name__ == '__main__':
    draw_system_topology()
    draw_normal_vs_anomaly()
    draw_voltage_stability_concept()
    print("\n系統図の生成完了!")
