import mmap
import ctypes
import ctypes.wintypes
import time
import csv
import os
from datetime import datetime

# ── Config ──────────────────────────
MIN_FRAMES_PER_LAP = 3000   # ~50s at 60fps — filters spawn segment
MIN_LAP_TIME_SECS  = 30.0   # filters invalid/off-track laps (AC sets iLastTime=0)

# ─────────────────────────────────────────────
# AC Shared Memory Structures (official layout)
# Source: Assetto Corsa SDK documentation
# ─────────────────────────────────────────────

class SPageFilePhysics(ctypes.Structure):
    _fields_ = [
        ("packetId",            ctypes.c_int),
        ("gas",                 ctypes.c_float),
        ("brake",               ctypes.c_float),
        ("fuel",                ctypes.c_float),
        ("gear",                ctypes.c_int),
        ("rpms",                ctypes.c_int),
        ("steerAngle",          ctypes.c_float),
        ("speedKmh",            ctypes.c_float),
        ("velocity",            ctypes.c_float * 3),
        ("accG",                ctypes.c_float * 3),
        ("wheelSlip",           ctypes.c_float * 4),
        ("wheelLoad",           ctypes.c_float * 4),
        ("wheelPressure",       ctypes.c_float * 4),
        ("wheelAngularSpeed",   ctypes.c_float * 4),
        ("tyreWear",            ctypes.c_float * 4),
        ("tyreDirtyLevel",      ctypes.c_float * 4),
        ("tyreCoreTemp",        ctypes.c_float * 4),
        ("camberRAD",           ctypes.c_float * 4),
        ("suspensionTravel",    ctypes.c_float * 4),
        ("drs",                 ctypes.c_float),
        ("tc",                  ctypes.c_float),
        ("heading",             ctypes.c_float),
        ("pitch",               ctypes.c_float),
        ("roll",                ctypes.c_float),
        ("cgHeight",            ctypes.c_float),
        ("carDamage",           ctypes.c_float * 5),
        ("numberOfTyresOut",    ctypes.c_int),
        ("pitLimiterOn",        ctypes.c_int),
        ("abs",                 ctypes.c_float),
    ]

class SPageFileGraphic(ctypes.Structure):
    _fields_ = [
        ("packetId",                ctypes.c_int),
        ("status",                  ctypes.c_int),
        ("session",                 ctypes.c_int),
        ("currentTime",             ctypes.c_wchar * 15),
        ("lastTime",                ctypes.c_wchar * 15),
        ("bestTime",                ctypes.c_wchar * 15),
        ("split",                   ctypes.c_wchar * 15),
        ("completedLaps",           ctypes.c_int),
        ("position",                ctypes.c_int),
        ("iCurrentTime",            ctypes.c_int),
        ("iLastTime",               ctypes.c_int),
        ("iBestTime",               ctypes.c_int),
        ("sessionTimeLeft",         ctypes.c_float),
        ("distanceTraveled",        ctypes.c_float),
        ("isInPit",                 ctypes.c_int),
        ("currentSectorIndex",      ctypes.c_int),
        ("lastSectorTime",          ctypes.c_int),
        ("numberOfLaps",            ctypes.c_int),
        ("tyreCompound",            ctypes.c_wchar * 33),
        ("replayTimeMultiplier",    ctypes.c_float),
        ("normalizedCarPosition",   ctypes.c_float),
        ("carCoordinates",          ctypes.c_float * 3),
    ]


def open_shared_memory():
    physics_mm  = mmap.mmap(-1, ctypes.sizeof(SPageFilePhysics),  "Local\\acpmf_physics")
    graphics_mm = mmap.mmap(-1, ctypes.sizeof(SPageFileGraphic), "Local\\acpmf_graphics")
    return physics_mm, graphics_mm


def read_physics(mm):
    mm.seek(0)
    return SPageFilePhysics.from_buffer_copy(mm.read(ctypes.sizeof(SPageFilePhysics)))


def read_graphics(mm):
    mm.seek(0)
    return SPageFileGraphic.from_buffer_copy(mm.read(ctypes.sizeof(SPageFileGraphic)))


def save_lap(frames, label, session_folder):
    filepath = os.path.join(session_folder, f"{label}.csv")
    keys = frames[0].keys()
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(frames)
    return filepath


def main():
    print("Connecting to Assetto Corsa shared memory...")
    print("Make sure AC is running in a hotlap/practice session.")
    print("Press Ctrl+C to stop.\n")

    try:
        mm_physics, mm_graphics = open_shared_memory()
    except Exception as e:
        print(f"ERROR: Could not connect — {e}")
        print("Is Assetto Corsa running?")
        return

    session_time   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_folder = os.path.join("sessions", session_time)
    os.makedirs(session_folder, exist_ok=True)
    print(f"Session folder: {session_folder}\n")

    current_frames  = []
    saved_lap_count = 0
    last_pos        = None

    try:
        while True:
            phys = read_physics(mm_physics)
            gfx  = read_graphics(mm_graphics)

            speed = phys.speedKmh
            pos   = gfx.normalizedCarPosition
            wx    = gfx.carCoordinates[0]
            wz    = gfx.carCoordinates[2]

            frame = {
                'gas':          phys.gas,
                'brake':        phys.brake,
                'steer_angle':  phys.steerAngle,
                'speed_kmh':    speed,
                'g_lat':        phys.accG[0],
                'g_lon':        phys.accG[2],
                'world_x':      wx,
                'world_z':      wz,
                'norm_pos':     pos,
                'lap_time_ms':  gfx.iCurrentTime,
                'completed_laps': gfx.completedLaps,
                'timestamp_ms': int(time.time() * 1000),
            }

            # Skip frames if in pit lane
            if gfx.isInPit:
                current_frames = []
                last_pos = None
                continue

            # Only record when moving
            if speed > 1.0:
                current_frames.append(frame)

            # Live feedback every ~1 second
            if len(current_frames) % 60 == 0 and len(current_frames) > 0:
                print(f"  speed: {speed:6.1f} km/h | "
                      f"pos: {pos:.4f} | "
                      f"frames: {len(current_frames):5d} | "
                      f"laps saved: {saved_lap_count}", end='\r')

            if last_pos is not None:
                if last_pos > 0.9 and pos < 0.1:
                    # Wait one extra frame for AC to update iLastTime
                    time.sleep(0.05)
                    gfx = read_graphics(mm_graphics)  # re-read after delay
                    lap_secs = gfx.iLastTime / 1000.0

                    if len(current_frames) < MIN_FRAMES_PER_LAP:
                        print(f"\n  ✗ Discarded: spawn segment — starting clean lap recording...")
                        current_frames = []
                    elif lap_secs < MIN_LAP_TIME_SECS:
                        print(f"\n  ✗ Discarded: invalid/off-track ({lap_secs:.1f}s) — starting clean lap recording...")
                        current_frames = []
                    else:
                        label    = "reference_lap" if saved_lap_count == 0 else f"lap_{saved_lap_count:03d}"
                        filepath = save_lap(current_frames, label, session_folder)
                        print(f"\n  ✓ Saved: {label}.csv  "
                            f"({len(current_frames)} frames, "
                            f"lap time: {lap_secs:.3f}s)")
                        current_frames  = []
                        saved_lap_count += 1

            last_pos = pos
            time.sleep(1 / 60)

    except KeyboardInterrupt:
        print(f"\n\nStopped. Total laps saved: {saved_lap_count}")
        if current_frames:
            print(f"Incomplete lap discarded ({len(current_frames)} frames).")
    finally:
        mm_physics.close()
        mm_graphics.close()
        print("Done.")


if __name__ == "__main__":
    main()
