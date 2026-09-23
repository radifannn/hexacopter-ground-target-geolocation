import csv
import math
import re
import subprocess
import time

import numpy as np
from pymavlink import mavutil

import geolocation_geometry as geo


# ============================================================
# Experiment configuration
# ============================================================

SAMPLES = 100
SAMPLE_INTERVAL = 0.20  # seconds

# Current detector measurement
PIXEL_U = 319.0
PIXEL_V = 226.0

MAVLINK_CONNECTION = "tcp:127.0.0.1:5762"

GAZEBO_POSE_TOPIC = (
    "/world/hexacopter_runway/dynamic_pose/info"
)

TARGET_GPS_TOPIC = "/target/gps"

OUTPUT_CSV = (
    "/home/rifqi-radifan/gz_ws/geolocation/"
    "static_validation_100samples.csv"
)


# ============================================================
# Gazebo pose parsing
# ============================================================

def extract_pose_blocks(text):

    blocks = []

    for match in re.finditer(r"pose\s*\{", text):

        start = match.start()
        brace_start = text.find("{", start)

        depth = 0
        end = None

        for i in range(brace_start, len(text)):

            if text[i] == "{":
                depth += 1

            elif text[i] == "}":
                depth -= 1

                if depth == 0:
                    end = i + 1
                    break

        if end is not None:
            blocks.append(text[start:end])

    return blocks


def parse_pose(block):

    name_match = re.search(
        r'name:\s*"([^"]+)"',
        block
    )

    if not name_match:
        return None

    name = name_match.group(1)

    position_match = re.search(
        r'position\s*\{(.*?)\}',
        block,
        re.DOTALL
    )

    orientation_match = re.search(
        r'orientation\s*\{(.*?)\}',
        block,
        re.DOTALL
    )

    if not position_match or not orientation_match:
        return None

    position = position_match.group(1)
    orientation = orientation_match.group(1)

    def value(pattern, source, default=0.0):

        match = re.search(pattern, source)

        if match:
            return float(match.group(1))

        return default

    return {
        "name": name,

        "position": np.array([
            value(r'x:\s*([-+0-9.eE]+)', position),
            value(r'y:\s*([-+0-9.eE]+)', position),
            value(r'z:\s*([-+0-9.eE]+)', position)
        ]),

        "quaternion": np.array([
            value(r'x:\s*([-+0-9.eE]+)', orientation),
            value(r'y:\s*([-+0-9.eE]+)', orientation),
            value(r'z:\s*([-+0-9.eE]+)', orientation),
            value(r'w:\s*([-+0-9.eE]+)', orientation, 1.0)
        ])
    }


def read_gazebo_state():

    result = subprocess.run(
        [
            "timeout",
            "0.35s",
            "gz",
            "topic",
            "-e",
            "-t",
            GAZEBO_POSE_TOPIC
        ],
        capture_output=True,
        text=True
    )

    poses = {}

    for block in extract_pose_blocks(result.stdout):

        pose = parse_pose(block)

        if pose is not None:
            poses[pose["name"]] = pose

    if "hexacopter_with_ardupilot" not in poses:
        raise RuntimeError("UAV pose not found.")

    if "pitch_link" not in poses:
        raise RuntimeError("pitch_link pose not found.")

    return poses


# ============================================================
# Target GPS reference
# ============================================================

def read_target_gps():

    result = subprocess.run(
        [
            "timeout",
            "1.0s",
            "gz",
            "topic",
            "-e",
            "-t",
            TARGET_GPS_TOPIC
        ],
        capture_output=True,
        text=True
    )

    text = result.stdout

    latitudes = re.findall(
        r"latitude_deg:\s*([-+0-9.eE]+)",
        text
    )

    longitudes = re.findall(
        r"longitude_deg:\s*([-+0-9.eE]+)",
        text
    )

    if not latitudes or not longitudes:
        raise RuntimeError(
            "Target GPS reference unavailable."
        )

    return float(latitudes[-1]), float(longitudes[-1])


# ============================================================
# Persistent MAVLink connection
# ============================================================

def connect_mavlink():

    print("Connecting to ArduPilot SITL...")

    master = mavutil.mavlink_connection(
        MAVLINK_CONNECTION
    )

    master.wait_heartbeat()

    print(
        f"Connected: system={master.target_system}, "
        f"component={master.target_component}"
    )

    # ATTITUDE: 10 Hz
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,
        100000,
        0, 0, 0, 0, 0
    )

    # GLOBAL_POSITION_INT: 5 Hz
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
        200000,
        0, 0, 0, 0, 0
    )

    return master


def read_latest_telemetry(master):

    attitude = None
    gps = None

    deadline = time.monotonic() + 0.18

    while time.monotonic() < deadline:

        msg = master.recv_match(
            type=[
                "ATTITUDE",
                "GLOBAL_POSITION_INT"
            ],
            blocking=False
        )

        if msg is None:
            time.sleep(0.005)
            continue

        if msg.get_type() == "ATTITUDE":

            attitude = {
                "roll": msg.roll,
                "pitch": msg.pitch,
                "yaw": msg.yaw
            }

        elif msg.get_type() == "GLOBAL_POSITION_INT":

            gps = {
                "lat": msg.lat / 1e7,
                "lon": msg.lon / 1e7,
                "rel_alt": msg.relative_alt / 1000.0
            }

    return attitude, gps


# ============================================================
# GPS error
# ============================================================

def gps_error_m(
    estimated_lat,
    estimated_lon,
    reference_lat,
    reference_lon
):

    R = 6378137.0

    north_error = (
        math.radians(
            estimated_lat - reference_lat
        ) * R
    )

    east_error = (
        math.radians(
            estimated_lon - reference_lon
        )
        * R
        * math.cos(
            math.radians(reference_lat)
        )
    )

    horizontal_error = math.hypot(
        north_error,
        east_error
    )

    return (
        north_error,
        east_error,
        horizontal_error
    )


# ============================================================
# One estimator sample
# ============================================================

def estimate_one(master):

    # Pixel -> camera ray
    ray_camera = geo.pixel_to_camera_ray(
        PIXEL_U,
        PIXEL_V
    )

    # Gazebo state
    poses = read_gazebo_state()

    uav_pose = poses[
        "hexacopter_with_ardupilot"
    ]

    pitch_pose = poses["pitch_link"]

    uav_position = uav_pose["position"]

    pitch_quaternion = pitch_pose["quaternion"]

    # UAV telemetry
    attitude, gps = read_latest_telemetry(master)

    if attitude is None or gps is None:
        return None

    roll = attitude["roll"]
    pitch = attitude["pitch"]
    yaw = attitude["yaw"]

    # Camera -> NED
    ray_ned = geo.camera_ray_to_ned_static(
        ray_camera,
        pitch_quaternion,
        roll,
        pitch,
        yaw
    )

    # Ground intersection
    height = uav_position[2] - geo.GROUND_Z

    target_ned = geo.ray_ground_intersection(
        ray_ned,
        height
    )

    # NED -> GPS
    target_lat, target_lon = geo.ned_to_gps(
        gps["lat"],
        gps["lon"],
        target_ned[0],
        target_ned[1]
    )

    return {
        "u": PIXEL_U,
        "v": PIXEL_V,

        "north": target_ned[0],
        "east": target_ned[1],
        "down": target_ned[2],

        "lat": target_lat,
        "lon": target_lon,

        "uav_lat": gps["lat"],
        "uav_lon": gps["lon"],
        "uav_alt": height,

        "roll_deg": math.degrees(roll),
        "pitch_deg": math.degrees(pitch),
        "yaw_deg": math.degrees(yaw),

        "gimbal_pitch_deg": math.degrees(
            geo.static_pitch_from_quaternion(
                pitch_quaternion
            )
        ),

        "ray_n": ray_ned[0],
        "ray_e": ray_ned[1],
        "ray_d": ray_ned[2]
    }


# ============================================================
# Main experiment
# ============================================================

def main():

    print()
    print("=" * 70)
    print("STATIC GEOLOCATION — 100 SAMPLE VALIDATION")
    print("=" * 70)

    print()
    print("Fixed conditions:")
    print(f"  Pixel      = ({PIXEL_U}, {PIXEL_V})")
    print(f"  Samples    = {SAMPLES}")
    print(f"  Interval   = {SAMPLE_INTERVAL} s")
    print("  Target     = stationary")
    print("  RC7        = 1400")

    # Independent target reference
    target_ref_lat, target_ref_lon = read_target_gps()

    print()
    print("Target GPS reference:")
    print(f"  Latitude  = {target_ref_lat:.12f}")
    print(f"  Longitude = {target_ref_lon:.12f}")

    master = connect_mavlink()

    rows = []

    print()
    print("Collecting samples...")

    for i in range(SAMPLES):

        sample_start = time.monotonic()

        try:

            result = estimate_one(master)

            if result is None:
                print(
                    f"  [{i+1:03d}/{SAMPLES}] "
                    "SKIPPED — telemetry unavailable"
                )
                time.sleep(SAMPLE_INTERVAL)
                continue

            north_error, east_error, horizontal_error = (
                gps_error_m(
                    result["lat"],
                    result["lon"],
                    target_ref_lat,
                    target_ref_lon
                )
            )

            result["sample"] = i + 1
            result["north_error"] = north_error
            result["east_error"] = east_error
            result["horizontal_error"] = horizontal_error

            rows.append(result)

            print(
                f"  [{i+1:03d}/{SAMPLES}] "
                f"N={result['north']:+.3f} m  "
                f"E={result['east']:+.3f} m  "
                f"error={horizontal_error:.4f} m"
            )

        except Exception as exc:

            print(
                f"  [{i+1:03d}/{SAMPLES}] "
                f"ERROR: {exc}"
            )

        elapsed = time.monotonic() - sample_start

        remaining = (
            SAMPLE_INTERVAL - elapsed
        )

        if remaining > 0:
            time.sleep(remaining)

    if not rows:
        raise RuntimeError(
            "No valid samples collected."
        )

    # ========================================================
    # Save CSV
    # ========================================================

    fieldnames = list(rows[0].keys())

    with open(
        OUTPUT_CSV,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    # ========================================================
    # Statistics
    # ========================================================

    horizontal_errors = np.array([
        row["horizontal_error"]
        for row in rows
    ])

    north_errors = np.array([
        row["north_error"]
        for row in rows
    ])

    east_errors = np.array([
        row["east_error"]
        for row in rows
    ])

    north_positions = np.array([
        row["north"]
        for row in rows
    ])

    east_positions = np.array([
        row["east"]
        for row in rows
    ])

    horizontal_rmse = math.sqrt(
        np.mean(horizontal_errors ** 2)
    )

    horizontal_mae = np.mean(
        np.abs(horizontal_errors)
    )

    north_rmse = math.sqrt(
        np.mean(north_errors ** 2)
    )

    east_rmse = math.sqrt(
        np.mean(east_errors ** 2)
    )

    print()
    print("=" * 70)
    print("STATIC VALIDATION RESULTS")
    print("=" * 70)

    print()
    print(f"Valid samples       = {len(rows)}")

    print()
    print("Estimated position:")
    print(
        f"  North mean        = "
        f"{np.mean(north_positions):+.4f} m"
    )
    print(
        f"  North std         = "
        f"{np.std(north_positions):.4f} m"
    )
    print(
        f"  East mean         = "
        f"{np.mean(east_positions):+.4f} m"
    )
    print(
        f"  East std          = "
        f"{np.std(east_positions):.4f} m"
    )

    print()
    print("Position error:")
    print(
        f"  North RMSE        = "
        f"{north_rmse:.4f} m"
    )
    print(
        f"  East RMSE         = "
        f"{east_rmse:.4f} m"
    )
    print(
        f"  Horizontal MAE    = "
        f"{horizontal_mae:.4f} m"
    )
    print(
        f"  Horizontal RMSE   = "
        f"{horizontal_rmse:.4f} m"
    )
    print(
        f"  Horizontal max    = "
        f"{np.max(horizontal_errors):.4f} m"
    )
    print(
        f"  Horizontal min    = "
        f"{np.min(horizontal_errors):.4f} m"
    )

    print()
    print(f"CSV saved to:")
    print(f"  {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
