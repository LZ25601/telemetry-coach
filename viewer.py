import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d
import os
import sys
import glob

# ── Config ──────────────────────────────────────────────────
RESAMPLE_POINTS = 1000   # interpolation points across track
SMOOTH_SIGMA    = 4      # gaussian smoothing (higher = smoother lines)
OUTPUT_DIR      = "comparisons"
# ────────────────────────────────────────────────────────────

BG    = '#0d0d0d'
PANEL = '#161616'
GRID  = '#252525'
TEXT  = '#b0b0b0'
REF_C = '#00bcd4'
LAP_C = '#ff6b35'
POS_C = '#4caf50'
NEG_C = '#f44336'

CHANNELS = [
    ('speed_kmh',   'Speed km/h'),
    ('gas',         'Throttle'),
    ('brake',       'Brake'),
    ('steer_angle', 'Steering'),
    ('g_lat',       'G Lateral'),
]


def load_and_resample(filepath, n=RESAMPLE_POINTS):
    df = pd.read_csv(filepath)
    df = df.sort_values('norm_pos').drop_duplicates(subset='norm_pos')
    pos_grid = np.linspace(0, 1, n)
    resampled = {'norm_pos': pos_grid}
    for col, _ in CHANNELS:
        raw = np.interp(pos_grid, df['norm_pos'], df[col])
        resampled[col] = gaussian_filter1d(raw, sigma=SMOOTH_SIGMA)
    return pd.DataFrame(resampled)


def plot_comparison(ref_path, lap_path, output_path):
    ref = load_and_resample(ref_path)
    lap = load_and_resample(lap_path)

    ref_name = os.path.splitext(os.path.basename(ref_path))[0]
    lap_name = os.path.splitext(os.path.basename(lap_path))[0]
    pos      = ref['norm_pos']
    xticks   = np.linspace(0, 1, 11)
    xlabels  = [f'{int(x*100)}%' for x in xticks]

    fig = plt.figure(figsize=(20, 16), facecolor=BG)
    fig.text(0.5, 0.975, 'Telemetry Comparison', ha='center', va='top',
             color='white', fontsize=15, fontweight='bold')
    fig.text(0.5, 0.958,
             f'{lap_name}  (orange)  vs  {ref_name}  (blue)'
             '  ·  green = faster/more than ref  ·  red = slower/less',
             ha='center', va='top', color=TEXT, fontsize=9)

    gs = gridspec.GridSpec(5, 1, hspace=0.04,
                           left=0.07, right=0.97, top=0.945, bottom=0.055)

    for i, (col, ylabel) in enumerate(CHANNELS):
        ax = fig.add_subplot(gs[i], facecolor=PANEL)
        r  = ref[col].values
        l  = lap[col].values
        d  = l - r

        ax.plot(pos, r, color=REF_C, linewidth=1.5, alpha=0.95, zorder=3)
        ax.plot(pos, l, color=LAP_C, linewidth=1.5, alpha=0.95, zorder=3,
                linestyle='--', dashes=(6, 2))
        ax.fill_between(pos, r, l, where=d >= 0, alpha=0.18,
                        color=POS_C, interpolate=True, zorder=2)
        ax.fill_between(pos, r, l, where=d <  0, alpha=0.18,
                        color=NEG_C, interpolate=True, zorder=2)

        all_vals = np.concatenate([r, l])
        ymin, ymax = all_vals.min(), all_vals.max()
        pad = (ymax - ymin) * 0.12 if ymax != ymin else 0.5
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_xlim(0, 1)

        ax.set_xticks(xticks)
        ax.set_xticklabels(
            xlabels if i == len(CHANNELS) - 1 else ['' for _ in xticks],
            color=TEXT, fontsize=8
        )
        ax.set_yticks(np.linspace(ymin, ymax, 4))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1f}'))
        ax.tick_params(colors=TEXT, labelsize=8, length=3, width=0.5)
        ax.set_ylabel(ylabel, color=TEXT, fontsize=9, labelpad=6)

        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
            spine.set_linewidth(0.6)
        ax.grid(axis='x', color=GRID, linewidth=0.5, linestyle='--', zorder=1)
        ax.grid(axis='y', color=GRID, linewidth=0.3, linestyle=':', zorder=1)

        if i == 0:
            ax.legend(
                handles=[
                    plt.Line2D([0], [0], color=REF_C, linewidth=2, label=ref_name),
                    plt.Line2D([0], [0], color=LAP_C, linewidth=2,
                               linestyle='--', dashes=(6, 2), label=lap_name),
                ],
                loc='upper right', fontsize=8.5,
                facecolor='#1e1e1e', edgecolor='#333', labelcolor='white',
                ncol=2, handlelength=2, borderpad=0.5
            )

    fig.get_axes()[-1].set_xlabel('Track Position', color=TEXT, fontsize=9)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, facecolor=BG, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def find_latest_session():
    sessions = sorted(glob.glob(os.path.join("sessions", "*")))
    return sessions[-1] if sessions else None


def main():
    # Usage:
    #   python viewer.py                          → auto: latest session, all laps vs reference
    #   python viewer.py ref.csv lap.csv          → explicit pair
    #   python viewer.py sessions/2026-05-18_...  → all laps in a specific session

    if len(sys.argv) == 3:
        ref_path = sys.argv[1]
        lap_path = sys.argv[2]
        out = os.path.join(OUTPUT_DIR,
            f"compare_{os.path.splitext(os.path.basename(lap_path))[0]}.png")
        plot_comparison(ref_path, lap_path, out)

    elif len(sys.argv) == 2:
        session_folder = sys.argv[1]
        ref_path = os.path.join(session_folder, "reference_lap.csv")
        if not os.path.exists(ref_path):
            print(f"ERROR: No reference_lap.csv in {session_folder}")
            return
        laps = sorted(glob.glob(os.path.join(session_folder, "lap_*.csv")))
        if not laps:
            print("No lap_*.csv files found.")
            return
        for lap_path in laps:
            lap_name = os.path.splitext(os.path.basename(lap_path))[0]
            out = os.path.join(OUTPUT_DIR,
                f"{os.path.basename(session_folder)}_{lap_name}.png")
            print(f"  Comparing: {lap_name}")
            plot_comparison(ref_path, lap_path, out)

    else:
        session_folder = find_latest_session()
        if not session_folder:
            print("No sessions folder found. Run recorder.py first.")
            return
        print(f"Using latest session: {session_folder}\n")
        ref_path = os.path.join(session_folder, "reference_lap.csv")
        if not os.path.exists(ref_path):
            print(f"ERROR: No reference_lap.csv in {session_folder}")
            return
        laps = sorted(glob.glob(os.path.join(session_folder, "lap_*.csv")))
        if not laps:
            print("No lap_*.csv files found. Drive some laps first.")
            return
        for lap_path in laps:
            lap_name = os.path.splitext(os.path.basename(lap_path))[0]
            out = os.path.join(OUTPUT_DIR,
                f"{os.path.basename(session_folder)}_{lap_name}.png")
            print(f"  Comparing: {lap_name}")
            plot_comparison(ref_path, lap_path, out)

    print(f"\nDone. Charts saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
