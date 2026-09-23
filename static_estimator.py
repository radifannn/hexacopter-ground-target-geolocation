import math
import numpy as np
import time
import re
import subprocess

from pymavlink import mavutil

import geolocation_geometry as geo


MAVLINK_CONNECTION = "tcp:127.0.0.1:5762"

PIXEL_U = 319.0
PIXEL_V = 226.0

GROUND_Z = 0.0

GAZEBO_POSE_TOPIC = (
    "/world/hexacopter_runway/dynamic_pose/info"
)


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

    def get_value(pattern, source, default=0.0):

        match = re.search(pattern, source)

        if match:
            return float(match.group(1))

        return default

    x = get_value(r'x:\s*([-+0-9.eE]+)', position)
    y = get_value(r'y:\s*([-+0-9.eE]+)', position)
    z = get_value(r'z:\s*([-+0-9.eE]+)', position)

    qx = get_value(r'x:\s*([-+0-9.eE]+)', orientation)
    qy = get_value(r'y:\s*([-+0-9.eE]+)', orientation)
    qz = get_value(r'z:\s*([-+0-9.eE]+)', orientation)
    qw = get_value(r'w:\s*([-+0-9.eE]+)', orientation, 1.0)

    return {
        "name": name,
        "position": np.array([x, y, z]),
        "quaternion": np.array([qx, qy, qz, qw])
    }


def read_gazebo_state():

    result = subprocess.run(
        [
            "timeout",
            "1",
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


def read_uav_telemetry():

    print("Connecting to ArduPilot SITL...")

    master = mavutil.mavlink_connection(
        MAVLINK_CONNECTION
    )

    master.wait_heartbeat()

    print(
        f"Connected: system={master.target_system}, "
        f"component={master.target_component}"
    )

    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,
        100000,
        0, 0, 0, 0, 0
    )

    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
        200000,
        0, 0, 0, 0, 0
    )

    attitude = None
    gps = None

    start = time.time()

    while time.time() - start < 2.0:

        msg = master.recv_match(
            type=["ATTITUDE", "GLOBAL_POSITION_INT"],
            blocking=False
        )

        if msg is None:
            time.sleep(0.01)
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

    if attitude is None:
        raise RuntimeError("ATTITUDE unavailable.")

    if gps is None:
        raise RuntimeError(
            "GLOBAL_POSITION_INT unavailable."
        )

    return attitude, gps


def estimate_target():

    # -------------------------------------------------------
    # 1. Camera observation
    # -------------------------------------------------------

    ray_camera = geo.pixel_to_camera_ray(
        PIXEL_U,
        PIXEL_V
    )

    # -------------------------------------------------------
    # 2. Gazebo state
    # -------------------------------------------------------

    poses = read_gazebo_state()

    uav_pose = poses[
        "hexacopter_with_ardupilot"
    ]

    pitch_pose = poses["pitch_link"]

    uav_position = uav_pose["position"]

    pitch_quaternion = pitch_pose["quaternion"]

    # -------------------------------------------------------
    # 3. UAV attitude + GPS
    # -------------------------------------------------------

    attitude, gps = read_uav_telemetry()

    roll = attitude["roll"]
    pitch = attitude["pitch"]
    yaw = attitude["yaw"]

    # -------------------------------------------------------
    # 4. Camera -> NED
    # -------------------------------------------------------

    ray_ned = geo.camera_ray_to_ned_static(
        ray_camera,
        pitch_quaternion,
        roll,
        pitch,
        yaw
    )

    # -------------------------------------------------------
    # 5. Ground intersection
    # -------------------------------------------------------

    height = uav_position[2] - GROUND_Z

    target_ned = geo.ray_ground_intersection(
        ray_ned,
        height
    )

    # -------------------------------------------------------
    # 6. NED -> GPS
    # -------------------------------------------------------

    target_lat, target_lon = geo.ned_to_gps(
        gps["lat"],
        gps["lon"],
        target_ned[0],
        target_ned[1]
    )

    return {
        "ray_camera": ray_camera,
        "ray_ned": ray_ned,
        "target_ned": target_ned,
        "target_lat": target_lat,
        "target_lon": target_lon,
        "uav_lat": gps["lat"],
        "uav_lon": gps["lon"],
        "uav_alt": height,
        "pitch_angle": geo.static_pitch_from_quaternion(
            pitch_quaternion
        ),
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw
    }


def main():

    print()
    print("=" * 60)
    print("STATIC TARGET GEOLOCATION")
    print("=" * 60)

    result = estimate_target()

    print()
    print("Camera observation:")
    print(
        f"  Pixel = "
        f"({PIXEL_U:.1f}, {PIXEL_V:.1f})"
    )

    print()
    print("NED ray:")
    print(
        f"  [{result['ray_ned'][0]:+.6f}, "
        f"{result['ray_ned'][1]:+.6f}, "
        f"{result['ray_ned'][2]:+.6f}]"
    )

    print()
    print("Estimated target position:")
    print(
        f"  North = {result['target_ned'][0]:+.4f} m"
    )
    print(
        f"  East  = {result['target_ned'][1]:+.4f} m"
    )
    print(
        f"  Down  = {result['target_ned'][2]:+.4f} m"
    )

    print()
    print("Estimated target GPS:")
    print(
        f"  Latitude  = "
        f"{result['target_lat']:.12f}"
    )
    print(
        f"  Longitude = "
        f"{result['target_lon']:.12f}"
    )

    print()
    print("Current orientation:")
    print(
        f"  UAV roll  = "
        f"{math.degrees(result['roll']):+.3f}°"
    )
    print(
        f"  UAV pitch = "
        f"{math.degrees(result['pitch']):+.3f}°"
    )
    print(
        f"  UAV yaw   = "
        f"{math.degrees(result['yaw']):+.3f}°"
    )
    print(
        f"  Gimbal pitch = "
        f"{math.degrees(result['pitch_angle']):+.3f}°"
    )


if __name__ == "__main__":
    main()
