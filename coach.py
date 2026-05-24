import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import os
import sys
import glob
import json
from datetime import datetime

# ── Config ──────────────────────────────────────────────────────────────────
RESAMPLE_POINTS    = 1000
SMOOTH_SIGMA       = 4
G_LAT_THRESHOLD    = 0.3    # g to count as cornering
MIN_CORNER_LEN     = 20     # min grid points for a valid corner
MERGE_GAP          = 15     # merge corners separated by fewer than this pts
BRAKE_LATE_THRESH  = 8      # pts — late braking tolerance
BRAKE_EARLY_THRESH = 8      # pts — early braking tolerance
THROTTLE_THRESH    = 10     # pts — late/early throttle tolerance
SPEED_DEFICIT_KMH  = 5.0    # km/h minimum speed deficit to flag
G_DELTA_THRESH     = 0.25   # g delta to flag oversteer/understeer
# ────────────────────────────────────────────────────────────────────────────


def load_and_resample(filepath, n=RESAMPLE_POINTS):
    df = pd.read_csv(filepath)
    df = df.sort_values("norm_pos").drop_duplicates(subset="norm_pos")
    pos_grid = np.linspace(0, 1, n)
    out = {"norm_pos": pos_grid}
    for col in ["speed_kmh", "gas", "brake", "steer_angle", "g_lat"]:
        out[col] = gaussian_filter1d(
            np.interp(pos_grid, df["norm_pos"], df[col]),
            sigma=SMOOTH_SIGMA,
        )
    return pd.DataFrame(out)


def detect_corners(ref_df):
    g_abs     = np.abs(gaussian_filter1d(ref_df["g_lat"].values, sigma=6))
    in_corner = g_abs > G_LAT_THRESHOLD
    corners   = []
    start     = None

    for i, v in enumerate(in_corner):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= MIN_CORNER_LEN:
                corners.append([start, i])
            start = None
    if start is not None and len(ref_df) - start >= MIN_CORNER_LEN:
        corners.append([start, len(ref_df)])

    # Merge close corners
    merged = []
    for c in corners:
        if merged and c[0] - merged[-1][1] < MERGE_GAP:
            merged[-1][1] = c[1]
        else:
            merged.append(c)
    return merged


def _find_braking_point(brake, start, end):
    for i, b in enumerate(brake[start:end]):
        if b > 0.05:
            return start + i
    return None


def _find_throttle_reapply(gas, start, end):
    seg     = gas[start:end]
    min_idx = int(np.argmin(seg))
    for i in range(min_idx + 1, len(seg)):
        if seg[i] > 0.1:
            return start + min_idx, start + i
    return start + min_idx, None


def _analyse_corner(idx, ref, lap, start, end, pos_grid):
    findings    = []
    label       = f"T{idx:02d}"
    pos_mid     = float(pos_grid[int((start + end) / 2)])
    brake_start = max(0, start - 40)

    r_brake = ref["brake"].values
    l_brake = lap["brake"].values
    r_gas   = ref["gas"].values
    l_gas   = lap["gas"].values
    r_speed = ref["speed_kmh"].values
    l_speed = lap["speed_kmh"].values
    r_glat  = ref["g_lat"].values
    l_glat  = lap["g_lat"].values

    # 1. Braking point
    r_bp = _find_braking_point(r_brake, brake_start, end)
    l_bp = _find_braking_point(l_brake, brake_start, end)
    if r_bp is not None and l_bp is not None:
        delta = l_bp - r_bp
        dist  = abs(pos_grid[l_bp] - pos_grid[r_bp]) * 100
        if delta > BRAKE_LATE_THRESH:
            findings.append({"corner": label, "pos": pos_mid, "type": "late_brake",
                "msg": f"{label} — Braked too late (~{dist:.1f}% later than reference)"})
        elif delta < -BRAKE_EARLY_THRESH:
            findings.append({"corner": label, "pos": pos_mid, "type": "early_brake",
                "msg": f"{label} — Braked too early (~{dist:.1f}% earlier than reference)"})

    # 2. Throttle reapplication
    _, r_reapply = _find_throttle_reapply(r_gas, start, end)
    _, l_reapply = _find_throttle_reapply(l_gas, start, end)
    if r_reapply is not None and l_reapply is not None:
        delta = l_reapply - r_reapply
        if delta > THROTTLE_THRESH:
            findings.append({"corner": label, "pos": pos_mid, "type": "late_throttle",
                "msg": f"{label} — Late throttle reapplication (losing exit speed)"})
        elif delta < -THROTTLE_THRESH:
            findings.append({"corner": label, "pos": pos_mid, "type": "early_throttle",
                "msg": f"{label} — Throttle applied too early (risk of understeer)"})

    # 3. Minimum corner speed
    speed_deficit = float(r_speed[start:end].min() - l_speed[start:end].min())
    if speed_deficit > SPEED_DEFICIT_KMH:
        findings.append({"corner": label, "pos": pos_mid, "type": "speed_deficit",
            "msg": f"{label} — Corner speed deficit: -{speed_deficit:.1f} km/h vs reference"})

    # 4. Lateral G
    r_g = float(np.abs(r_glat[start:end]).max())
    l_g = float(np.abs(l_glat[start:end]).max())
    g_delta = l_g - r_g
    if g_delta > G_DELTA_THRESH:
        findings.append({"corner": label, "pos": pos_mid, "type": "oversteer",
            "msg": f"{label} — High lateral G spike (possible oversteer)"})
    elif g_delta < -G_DELTA_THRESH:
        findings.append({"corner": label, "pos": pos_mid, "type": "understeer",
            "msg": f"{label} — Low lateral G (possible understeer / missed apex)"})

    return findings


def run_rule_engine(ref, lap, corners):
    all_findings = []
    pos_grid     = ref["norm_pos"].values
    for i, (start, end) in enumerate(corners):
        all_findings.extend(_analyse_corner(i + 1, ref, lap, start, end, pos_grid))
    return all_findings


def format_report(findings, ref_name, lap_name, lap_time_secs=None):
    lines = []
    lines.append("=" * 60)
    lines.append("  TELEMETRY COACH — Rule-Based Analysis")
    lines.append("=" * 60)
    lines.append(f"  Reference : {ref_name}")
    lines.append(f"  Lap       : {lap_name}")
    if lap_time_secs:
        lines.append(f"  Lap Time  : {lap_time_secs:.3f}s")
    lines.append("")

    if not findings:
        lines.append("  ✓ No significant mistakes detected.")
    else:
        type_icons = {
            "late_brake":    "🔴",
            "early_brake":   "🟡",
            "late_throttle": "🔴",
            "early_throttle":"🟡",
            "speed_deficit": "🔵",
            "oversteer":     "🟠",
            "understeer":    "🟠",
        }
        for f in findings:
            icon = type_icons.get(f["type"], "⚪")
            lines.append(f"  {icon}  {f['msg']}")

    lines.append("")
    lines.append(f"  Total findings: {len(findings)}")
    lines.append("=" * 60)
    return "\n".join(lines)


def get_lap_time(filepath):
    try:
        df = pd.read_csv(filepath)
        # last_lap_time or lap_time_ms column
        if "lap_time_ms" in df.columns:
            return df["lap_time_ms"].max() / 1000.0
    except Exception:
        pass
    return None


def process_pair(ref_path, lap_path, save_json=True):
    ref = load_and_resample(ref_path)
    lap = load_and_resample(lap_path)
    corners  = detect_corners(ref)
    findings = run_rule_engine(ref, lap, corners)

    ref_name = os.path.splitext(os.path.basename(ref_path))[0]
    lap_name = os.path.splitext(os.path.basename(lap_path))[0]
    lap_time = get_lap_time(lap_path)

    report = format_report(findings, ref_name, lap_name, lap_time)
    print(report)

    if save_json:
        out_dir  = "coaching_reports"
        os.makedirs(out_dir, exist_ok=True)
        session  = os.path.basename(os.path.dirname(lap_path))
        out_path = os.path.join(out_dir, f"{session}_{lap_name}.json")
        with open(out_path, "w") as f:
            json.dump({
                "ref": ref_name, "lap": lap_name,
                "lap_time_secs": lap_time,
                "corners_detected": len(corners),
                "findings": findings,
                "generated": datetime.now().isoformat(),
            }, f, indent=2)
        print(f"  Report saved: {out_path}\n")

    return findings


def find_latest_session():
    sessions = sorted(glob.glob(os.path.join("sessions", "*")))
    return sessions[-1] if sessions else None


def main():
    # Usage:
    #   python coach.py                          → auto: latest session, all laps vs reference
    #   python coach.py ref.csv lap.csv          → explicit pair
    #   python coach.py sessions/2026-05-18_...  → all laps in a session folder

    if len(sys.argv) == 3:
        process_pair(sys.argv[1], sys.argv[2])

    elif len(sys.argv) == 2:
        session_folder = sys.argv[1]
        ref_path = os.path.join(session_folder, "reference_lap.csv")
        if not os.path.exists(ref_path):
            print(f"ERROR: No reference_lap.csv in {session_folder}")
            return
        for lap_path in sorted(glob.glob(os.path.join(session_folder, "lap_*.csv"))):
            process_pair(ref_path, lap_path)

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
            process_pair(ref_path, lap_path)


if __name__ == "__main__":
    main()
