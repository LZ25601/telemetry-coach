"""
agent2.py — Offline SAC-based Mistake Detection Agent
------------------------------------------------------
Usage:
  python agent2.py train  sessions/2026-05-18_.../   # train on a session folder
  python agent2.py eval   ref.csv  lap.csv            # evaluate a lap
  python agent2.py report ref.csv  lap.csv            # full coaching report
"""

import os
import sys
import glob
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
import torch

# ── Config ──────────────────────────────────────────────────────────────────
RESAMPLE_POINTS    = 1000
SMOOTH_SIGMA       = 4
MODEL_PATH         = "models/agent2_sac"

# Rule engine thresholds (must match coach.py)
G_LAT_THRESHOLD    = 0.3
MIN_CORNER_LEN     = 20
MERGE_GAP          = 15
BRAKE_LATE_THRESH  = 8
BRAKE_EARLY_THRESH = 8
THROTTLE_THRESH    = 10
SPEED_DEFICIT_KMH  = 5.0
G_DELTA_THRESH     = 0.25

# Mistake type labels
MISTAKE_TYPES = [
    "none",
    "late_brake",
    "early_brake",
    "late_throttle",
    "early_throttle",
    "speed_deficit",
    "oversteer",
    "understeer",
]
MISTAKE_ICONS = {
    "none":           "✓",
    "late_brake":     "🔴",
    "early_brake":    "🟡",
    "late_throttle":  "🔴",
    "early_throttle": "🟡",
    "speed_deficit":  "🔵",
    "oversteer":      "🟠",
    "understeer":     "🟠",
}
# ────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  Data utilities (shared with coach.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_and_resample(filepath, n=RESAMPLE_POINTS):
    df = pd.read_csv(filepath)
    df = df.sort_values("norm_pos").drop_duplicates(subset="norm_pos")
    pos_grid = np.linspace(0, 1, n)
    out = {"norm_pos": pos_grid}
    for col in ["speed_kmh", "gas", "brake", "steer_angle", "g_lat"]:
        out[col] = gaussian_filter1d(
            np.interp(pos_grid, df["norm_pos"], df[col]), sigma=SMOOTH_SIGMA
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
    merged = []
    for c in corners:
        if merged and c[0] - merged[-1][1] < MERGE_GAP:
            merged[-1][1] = c[1]
        else:
            merged.append(c)
    return merged


# ══════════════════════════════════════════════════════════════════════════════
#  Rule engine (copied from coach.py — single source of truth for labels)
# ══════════════════════════════════════════════════════════════════════════════

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


def rule_engine_labels(ref, lap, corners):
    """Returns list of (corner_idx, mistake_type_str) for every corner."""
    labels   = []
    pos_grid = ref["norm_pos"].values
    r_brake  = ref["brake"].values;   l_brake = lap["brake"].values
    r_gas    = ref["gas"].values;     l_gas   = lap["gas"].values
    r_speed  = ref["speed_kmh"].values; l_speed = lap["speed_kmh"].values
    r_glat   = ref["g_lat"].values;   l_glat  = lap["g_lat"].values

    for i, (start, end) in enumerate(corners):
        mistake   = "none"
        bp_start  = max(0, start - 40)

        r_bp = _find_braking_point(r_brake, bp_start, end)
        l_bp = _find_braking_point(l_brake, bp_start, end)
        if r_bp is not None and l_bp is not None:
            delta = l_bp - r_bp
            if delta > BRAKE_LATE_THRESH:
                mistake = "late_brake"
            elif delta < -BRAKE_EARLY_THRESH:
                mistake = "early_brake"

        if mistake == "none":
            _, r_re = _find_throttle_reapply(r_gas, start, end)
            _, l_re = _find_throttle_reapply(l_gas, start, end)
            if r_re is not None and l_re is not None:
                delta = l_re - r_re
                if delta > THROTTLE_THRESH:
                    mistake = "late_throttle"
                elif delta < -THROTTLE_THRESH:
                    mistake = "early_throttle"

        if mistake == "none":
            deficit = float(r_speed[start:end].min() - l_speed[start:end].min())
            if deficit > SPEED_DEFICIT_KMH:
                mistake = "speed_deficit"

        if mistake == "none":
            r_g = float(np.abs(r_glat[start:end]).max())
            l_g = float(np.abs(l_glat[start:end]).max())
            if l_g - r_g > G_DELTA_THRESH:
                mistake = "oversteer"
            elif l_g - r_g < -G_DELTA_THRESH:
                mistake = "understeer"

        labels.append((i, mistake))
    return labels


# ══════════════════════════════════════════════════════════════════════════════
#  Feature extraction — one feature vector per corner
# ══════════════════════════════════════════════════════════════════════════════

def extract_corner_features(ref, lap, start, end):
    """
    Returns a 10-dim feature vector for one corner:
    [speed_delta_min, speed_delta_mean, brake_delta_max, brake_point_shift,
     throttle_delta_min, throttle_reapply_shift, g_lat_delta_max, g_lat_delta_mean,
     steer_delta_max, steer_delta_mean]
    """
    bp_start = max(0, start - 40)
    r        = ref.iloc[start:end]
    l        = lap.iloc[start:end]
    r_b      = ref["brake"].values; l_b = lap["brake"].values
    r_g      = ref["gas"].values;   l_g = lap["gas"].values

    speed_delta = l["speed_kmh"].values - r["speed_kmh"].values
    brake_delta = l_b[start:end] - r_b[start:end]
    gas_delta   = l_g[start:end] - r_g[start:end]
    glat_delta  = l["g_lat"].values - r["g_lat"].values
    steer_delta = l["steer_angle"].values - r["steer_angle"].values

    r_bp = _find_braking_point(r_b, bp_start, end)
    l_bp = _find_braking_point(l_b, bp_start, end)
    bp_shift = (l_bp - r_bp) / 50.0 if (r_bp and l_bp) else 0.0

    _, r_re = _find_throttle_reapply(r_g, start, end)
    _, l_re = _find_throttle_reapply(l_g, start, end)
    tr_shift = (l_re - r_re) / 50.0 if (r_re and l_re) else 0.0

    return np.array([
        float(speed_delta.min())  / 50.0,
        float(speed_delta.mean()) / 20.0,
        float(brake_delta.max())  if len(brake_delta) else 0.0,
        float(bp_shift),
        float(gas_delta.min()),
        float(tr_shift),
        float(np.abs(glat_delta).max()),
        float(glat_delta.mean()),
        float(np.abs(steer_delta).max()),
        float(steer_delta.mean()),
    ], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  Gymnasium Environment
# ══════════════════════════════════════════════════════════════════════════════

class LapComparisonEnv(gym.Env):
    """
    Each episode = one corner from a lap pair.
    Observation: 10-dim feature vector of deltas for that corner.
    Action:      continuous 8-dim vector — one score per mistake class.
    Reward:      +1 if argmax(action) matches rule engine label, -0.5 otherwise.
    """
    metadata = {}

    def __init__(self, lap_pairs, ref_df, corners):
        super().__init__()
        self.lap_pairs = lap_pairs   # list of (ref_df, lap_df) pairs
        self.ref_df    = ref_df
        self.corners   = corners
        self.n_classes = len(MISTAKE_TYPES)

        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(10,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_classes,), dtype=np.float32
        )
        self._current_features = None
        self._current_label    = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Pick a random lap pair and random corner
        ref, lap = self.lap_pairs[np.random.randint(len(self.lap_pairs))]
        corner_idx = np.random.randint(len(self.corners))
        start, end = self.corners[corner_idx]

        features = extract_corner_features(ref, lap, start, end)
        labels   = rule_engine_labels(ref, lap, self.corners)
        mistake  = labels[corner_idx][1]

        self._current_features = features
        self._current_label    = MISTAKE_TYPES.index(mistake)
        return features, {}

    def step(self, action):
        predicted = int(np.argmax(action))
        correct   = self._current_label
        reward    = 1.0 if predicted == correct else -0.5

        # Partial credit for related mistakes (brake family / throttle family)
        brake_family    = {MISTAKE_TYPES.index("late_brake"), MISTAKE_TYPES.index("early_brake")}
        throttle_family = {MISTAKE_TYPES.index("late_throttle"), MISTAKE_TYPES.index("early_throttle")}
        if predicted != correct:
            if {predicted, correct} <= brake_family or {predicted, correct} <= throttle_family:
                reward = 0.0  # partial — right family, wrong direction

        terminated = True   # each episode = one corner classification
        return self._current_features, reward, terminated, False, {"correct": predicted == correct}


# ══════════════════════════════════════════════════════════════════════════════
#  Training
# ══════════════════════════════════════════════════════════════════════════════

def load_session(session_folder):
    ref_path = os.path.join(session_folder, "reference_lap.csv")
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"No reference_lap.csv in {session_folder}")
    ref = load_and_resample(ref_path)
    lap_paths = sorted(glob.glob(os.path.join(session_folder, "lap_*.csv")))
    laps = [load_and_resample(p) for p in lap_paths]
    print(f"  Loaded reference + {len(laps)} laps from {session_folder}")
    return ref, laps


def train(session_folder, timesteps=50_000):
    print("\n── Agent 2 Training ────────────────────────────────")
    ref, laps = load_session(session_folder)
    corners   = detect_corners(ref)
    print(f"  Corners detected: {len(corners)}")

    # Build lap pairs: reference vs each lap
    lap_pairs = [(ref, lap) for lap in laps]
    if not lap_pairs:
        print("ERROR: No lap_*.csv files found. Record some laps first.")
        return

    env      = Monitor(LapComparisonEnv(lap_pairs, ref, corners))
    eval_env = Monitor(LapComparisonEnv(lap_pairs, ref, corners))

    model = SAC(
        "MlpPolicy", env,
        learning_rate=3e-4,
        buffer_size=50_000,
        batch_size=256,
        gamma=0.99,
        verbose=1,
        policy_kwargs=dict(net_arch=[128, 128]),
        device="cpu",
    )

    os.makedirs("models", exist_ok=True)
    stop_cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=10, verbose=1)
    eval_cb = EvalCallback(
        eval_env, best_model_save_path="models/",
        log_path="models/logs/", eval_freq=2000,
        n_eval_episodes=50, callback_after_eval=stop_cb, verbose=1
    )

    print(f"  Training for up to {timesteps:,} timesteps...")
    model.learn(total_timesteps=timesteps, callback=eval_cb, progress_bar=False)
    model.save(MODEL_PATH)
    print(f"\n  Model saved: {MODEL_PATH}.zip")
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  Inference
# ══════════════════════════════════════════════════════════════════════════════

def predict_lap(ref_path, lap_path, model_path=MODEL_PATH):
    ref     = load_and_resample(ref_path)
    lap     = load_and_resample(lap_path)
    corners = detect_corners(ref)

    if not os.path.exists(model_path + ".zip"):
        print(f"ERROR: No trained model found at {model_path}.zip")
        print("Run:  python agent2.py train <session_folder>")
        return []

    model = SAC.load(model_path, device="cpu")
    predictions = []
    for i, (start, end) in enumerate(corners):
        features  = extract_corner_features(ref, lap, start, end)
        action, _ = model.predict(features, deterministic=True)
        predicted = MISTAKE_TYPES[int(np.argmax(action))]
        pos_mid   = float(ref["norm_pos"].iloc[int((start + end) / 2)])
        predictions.append({
            "corner": f"T{i+1:02d}",
            "pos":    pos_mid,
            "type":   predicted,
        })
    return predictions


def full_report(ref_path, lap_path, model_path=MODEL_PATH):
    ref      = load_and_resample(ref_path)
    lap      = load_and_resample(lap_path)
    corners  = detect_corners(ref)
    ref_name = os.path.splitext(os.path.basename(ref_path))[0]
    lap_name = os.path.splitext(os.path.basename(lap_path))[0]

    # Rule engine labels
    rule_labels = {c: m for c, m in rule_engine_labels(ref, lap, corners)}

    # Agent predictions
    agent_preds = []
    if os.path.exists(model_path + ".zip"):
        model = SAC.load(model_path, device="cpu")
        for i, (start, end) in enumerate(corners):
            features  = extract_corner_features(ref, lap, start, end)
            action, _ = model.predict(features, deterministic=True)
            agent_preds.append(MISTAKE_TYPES[int(np.argmax(action))])
    else:
        agent_preds = [MISTAKE_TYPES[rule_labels.get(i, 0)] for i in range(len(corners))]

    # Lap time
    try:
        df_lap   = pd.read_csv(lap_path)
        lap_time = df_lap["lap_time_ms"].max() / 1000.0
    except Exception:
        lap_time = None

    lines = []
    lines.append("=" * 65)
    lines.append("  TELEMETRY COACH — Agent 2 Report")
    lines.append("=" * 65)
    lines.append(f"  Reference : {ref_name}")
    lines.append(f"  Lap       : {lap_name}")
    if lap_time:
        lines.append(f"  Lap Time  : {lap_time:.3f}s")
    lines.append("")
    lines.append(f"  {'Corner':<8} {'Rule Engine':<22} {'RL Agent':<22} {'Match'}")
    lines.append(f"  {'-'*8} {'-'*22} {'-'*22} {'-'*5}")

    matches = 0
    for i, (start, end) in enumerate(corners):
        label  = f"T{i+1:02d}"
        rule_m = rule_labels.get(i, 'none')
        agent_m = agent_preds[i]
        match  = "✓" if rule_m == agent_m else "✗"
        if rule_m == agent_m:
            matches += 1
        rule_str  = f"{MISTAKE_ICONS.get(rule_m,'')} {rule_m}"
        agent_str = f"{MISTAKE_ICONS.get(agent_m,'')} {agent_m}"
        lines.append(f"  {label:<8} {rule_str:<22} {agent_str:<22} {match}")

    accuracy = matches / len(corners) * 100 if corners else 0
    lines.append("")
    lines.append(f"  Agent accuracy vs rule engine: {accuracy:.0f}% ({matches}/{len(corners)} corners)")
    lines.append("")

    # Actionable feedback (agent predictions only)
    feedback = [p for p in agent_preds if p != "none"]
    if feedback:
        lines.append("  ── Coaching Feedback ──")
        for i, m in enumerate(agent_preds):
            if m != "none":
                icon = MISTAKE_ICONS.get(m, "")
                msg  = _mistake_message(f"T{i+1:02d}", m)
                lines.append(f"  {icon}  {msg}")
    else:
        lines.append("  ✓ No significant mistakes detected.")

    lines.append("=" * 65)
    report = "\n".join(lines)
    print(report)

    # Save JSON
    out_dir = "coaching_reports"
    os.makedirs(out_dir, exist_ok=True)
    session = os.path.basename(os.path.dirname(lap_path))
    out_path = os.path.join(out_dir, f"{session}_{lap_name}_agent2.json")
    with open(out_path, "w") as f:
        json.dump({
            "ref": ref_name, "lap": lap_name, "lap_time_secs": lap_time,
            "corners": len(corners), "accuracy_vs_rules": accuracy,
            "predictions": [
                {"corner": f"T{i+1:02d}", "rule": rule_labels.get(i, 'none'), "agent": agent_preds[i]}
                for i in range(len(corners))
            ]
        }, f, indent=2)
    print(f"  Report saved: {out_path}\n")


def _mistake_message(label, mistake):
    msgs = {
        "late_brake":     f"{label} — Braked too late (losing time under braking)",
        "early_brake":    f"{label} — Braked too early (leaving speed on the table)",
        "late_throttle":  f"{label} — Late throttle reapplication (losing exit speed)",
        "early_throttle": f"{label} — Throttle applied too early (risk of understeer)",
        "speed_deficit":  f"{label} — Corner speed deficit vs reference",
        "oversteer":      f"{label} — High lateral G spike (possible oversteer)",
        "understeer":     f"{label} — Low lateral G (possible understeer / missed apex)",
    }
    return msgs.get(mistake, f"{label} — {mistake}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def find_latest_session():
    sessions = sorted(glob.glob(os.path.join("sessions", "*")))
    return sessions[-1] if sessions else None


def main():
    # python agent2.py train  <session_folder>        train on session
    # python agent2.py train                          train on latest session
    # python agent2.py eval   <ref.csv> <lap.csv>     predict mistakes
    # python agent2.py report <ref.csv> <lap.csv>     full side-by-side report

    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"

    if cmd == "train":
        folder = sys.argv[2] if len(sys.argv) > 2 else find_latest_session()
        if not folder:
            print("No session folder found.")
            return
        train(folder, timesteps=100_000)

    elif cmd == "eval":
        if len(sys.argv) < 4:
            print("Usage: python agent2.py eval <ref.csv> <lap.csv>")
            return
        preds = predict_lap(sys.argv[2], sys.argv[3])
        for p in preds:
            icon = MISTAKE_ICONS.get(p["type"], "")
            print(f"  {icon}  {p['corner']} — {p['type']}")

    elif cmd == "report":
        if len(sys.argv) < 4:
            print("Usage: python agent2.py report <ref.csv> <lap.csv>")
            return
        full_report(sys.argv[2], sys.argv[3])

    else:
        print(__doc__)


if __name__ == "__main__":
    main()
