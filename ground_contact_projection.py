import math
import re
import subprocess
import time

import numpy as np
from pymavlink import mavutil


# ============================================================
# Camera calibration from Gazebo camera_info
# ============================================================

FX = 205.46962738037109
FY = 205.46965599060059
CX = 320.0
CY = 240.0


# ============================================================
# Observed detector bbox
# ============================================================

BBOX_X = 348.0
BBOX_Y = 216.0
BBOX_W = 22.0
BBOX_H = 16.0

OBSERVED_U = BBOX_X + BBOX_W / 2.0
OBSERVED_V = BBOX_Y + BBOX_H


# ============================================================
# Target ground-contact point
# ============================================================

TARGET_GROUND_WORLD = np.array([
    2.0,
    5.0,
    0.0
], dtype=float)


# ============================================================
# Rotations
# ============================================================

def Rx(a):
    c = math.cos(a)
    s = math.sin(a)

    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ], dtype=float)


def Ry(a):
    c = math.cos(a)
    s = math.sin(a)

    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ], dtype=float)


def Rz(a):
    c = math.cos(a)
    s = math.sin(a)

    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ], dtype=float)


def rpy_to_R(roll, pitch, yaw):
    return Rz(yaw) @ Ry(pitch) @ Rx(roll)


def quaternion_to_R(q):
    x, y, z, w = q

    q = np.asarray(q, dtype=float)
    q /= np.linalg.norm(q)

    x, y, z, w = q

    return np.array([
        [
            1 - 2 * (y*y + z*z),
            2 * (x*y - z*w),
            2 * (x*z + y*w)
        ],
        [
            2 * (x*y + z*w),
            1 - 2 * (x*x + z*z),
            2 * (y*z - x*w)
        ],
        [
            2 * (x*z + y*w),
            2 * (y*z - x*w),
            1 - 2 * (x*x + y*y)
        ]
    ], dtype=float)


# ============================================================
# Parse gz model output
# ============================================================

def run_gz_model(model, link=None):

    cmd = [
        "gz",
        "model",
        "-m",
        model
    ]

    if link is not None:
        cmd += ["--link", link]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip()
        )

    return result.stdout


def parse_named_pose(text, name):

    pattern = (
        rf"- Name:\s*{re.escape(name)}\s*$"
        r".*?"
        r"- Pose \[ XYZ \(m\) \] \[ RPY \(rad\) \]:\s*"
        r"\n\s*\[([^\]]+)\]\s*"
        r"\n\s*\[([^\]]+)\]"
    )

    match = re.search(
        pattern,
        text,
        re.DOTALL | re.MULTILINE
    )

    if not match:
        raise RuntimeError(
            f"Could not parse pose for {name}"
        )

    position = np.array([
        float(v)
        for v in match.group(1).split()
    ])

    rpy = np.array([
        float(v)
        for v in match.group(2).split()
    ])

    return position, rpy


# ============================================================
# Get current UAV + gimbal camera pose
# ============================================================

def get_camera_world_pose():

    # UAV
    uav_text = run_gz_model(
        "hexacopter_with_ardupilot"
    )

    uav_pos, uav_rpy = parse_named_pose(
        uav_text,
        "hexacopter_with_ardupilot"
    )

    R_world_uav = rpy_to_R(
        *uav_rpy
    )

    # Gimbal
    gimbal_text = run_gz_model("gimbal")

    gimbal_pos, gimbal_rpy = parse_named_pose(
        gimbal_text,
        "gimbal"
    )

    R_uav_gimbal = rpy_to_R(
        *gimbal_rpy
    )

    # Pitch link
    pitch_text = run_gz_model(
        "gimbal",
        "pitch_link"
    )

    pitch_pos, pitch_rpy = parse_named_pose(
        pitch_text,
        "pitch_link"
    )

    R_gimbal_pitch = rpy_to_R(
        *pitch_rpy
    )

    # Camera has zero translation from pitch_link.
    camera_pos_pitch = np.zeros(3)

    camera_pos_gimbal = (
        pitch_pos
        + R_gimbal_pitch
        @ camera_pos_pitch
    )

    camera_pos_uav = (
        gimbal_pos
        + R_uav_gimbal
        @ camera_pos_gimbal
    )

    camera_pos_world = (
        uav_pos
        + R_world_uav
        @ camera_pos_uav
    )

    # --------------------------------------------------------
    # Current camera orientation
    # --------------------------------------------------------

    # Simplified camera frame -> Gazebo camera frame
    R_gazebo_from_camera = np.array([
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0]
    ], dtype=float)

    # Fixed SDF mounting rotations
    R_outer = (
        Rz(math.radians(90.0))
        @ Rx(math.radians(90.0))
    )

    R_sensor = (
        Ry(math.radians(-90.0))
        @ Rx(math.radians(-90.0))
    )

    R_pitch = Rx(
        pitch_rpy[0]
    )

    R_gazebo_body_from_camera = (
        R_outer
        @ R_pitch
        @ R_sensor
    )

    # UAV body -> world
    R_ned_from_body = R_world_uav

    # Gazebo body -> ArduPilot body
    R_ardupilot_from_gazebo = np.diag([
        1.0,
        -1.0,
        -1.0
    ])

    # NED -> Gazebo world for position.
    # Gazebo world:
    #   X = East
    #   Y = North
    #   Z = Up
    #
    # NED:
    #   N = +Y
    #   E = +X
    #   D = -Z

    return {
        "camera_pos_world": camera_pos_world,
        "uav_pos_world": uav_pos,
        "R_world_uav": R_world_uav,
        "R_gazebo_from_camera": R_gazebo_from_camera,
        "R_gazebo_body_from_camera":
            R_gazebo_body_from_camera,
        "R_ardupilot_from_gazebo":
            R_ardupilot_from_gazebo,
    }


# ============================================================
# Read UAV attitude
# ============================================================

def read_attitude():

    master = mavutil.mavlink_connection(
        "tcp:127.0.0.1:5762"
    )

    master.wait_heartbeat()

    start = time.monotonic()

    while time.monotonic() - start < 2.0:

        msg = master.recv_match(
            type="ATTITUDE",
            blocking=True,
            timeout=0.5
        )

        if msg is not None:
            return (
                msg.roll,
                msg.pitch,
                msg.yaw
            )

    raise RuntimeError(
        "Could not receive ATTITUDE."
    )


# ============================================================
# World Gazebo point -> NED relative UAV
# ============================================================

def world_to_ned(point_world, uav_world):

    delta = (
        point_world
        - uav_world
    )

    return np.array([
        delta[1],      # North
        delta[0],      # East
        -delta[2]      # Down
    ])


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 65)
    print("GROUND-CONTACT POINT PROJECTION")
    print("=" * 65)

    # Current camera/world pose
    camera = get_camera_world_pose()

    # Current UAV attitude
    roll, pitch, yaw = read_attitude()

    # --------------------------------------------------------
    # Target point in NED
    # --------------------------------------------------------

    target_ned = world_to_ned(
        TARGET_GROUND_WORLD,
        camera["uav_pos_world"]
    )

    # Camera position relative UAV in NED
    camera_ned = world_to_ned(
        camera["camera_pos_world"],
        camera["uav_pos_world"]
    )

    # Vector from camera to target in NED
    target_from_camera_ned = (
        target_ned - camera_ned
    )

    ray_ned = (
        target_from_camera_ned
        / np.linalg.norm(
            target_from_camera_ned
        )
    )

    # --------------------------------------------------------
    # NED -> body
    # --------------------------------------------------------

    R_body_from_ned = (
        Rz(yaw)
        @ Ry(pitch)
        @ Rx(roll)
    ).T

    ray_body = (
        R_body_from_ned
        @ ray_ned
    )

    # --------------------------------------------------------
    # ArduPilot body -> Gazebo body
    # --------------------------------------------------------

    R_gazebo_from_ardupilot = (
        camera["R_ardupilot_from_gazebo"].T
    )

    ray_gazebo_body = (
        R_gazebo_from_ardupilot
        @ ray_body
    )

    # --------------------------------------------------------
    # Gazebo body -> Gazebo camera
    # --------------------------------------------------------

    ray_gazebo_camera = (
        camera["R_gazebo_body_from_camera"].T
        @ ray_gazebo_body
    )

    # --------------------------------------------------------
    # Gazebo camera -> our camera convention
    # --------------------------------------------------------

    ray_camera = (
        camera["R_gazebo_from_camera"].T
        @ ray_gazebo_camera
    )

    ray_camera /= np.linalg.norm(
        ray_camera
    )

    if ray_camera[2] <= 0:

        raise RuntimeError(
            "Ground-contact point is behind camera."
        )

    # --------------------------------------------------------
    # Project into image
    # --------------------------------------------------------

    predicted_u = (
        FX
        * ray_camera[0]
        / ray_camera[2]
        + CX
    )

    predicted_v = (
        FY
        * ray_camera[1]
        / ray_camera[2]
        + CY
    )

    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    du = predicted_u - OBSERVED_U
    dv = predicted_v - OBSERVED_V

    error = math.hypot(
        du,
        dv
    )

    print()
    print("Target ground-contact point:")
    print(
        f"  World = "
        f"[{TARGET_GROUND_WORLD[0]:+.3f}, "
        f"{TARGET_GROUND_WORLD[1]:+.3f}, "
        f"{TARGET_GROUND_WORLD[2]:+.3f}] m"
    )

    print()
    print("Camera world position:")
    print(
        f"  [{camera['camera_pos_world'][0]:+.4f}, "
        f"{camera['camera_pos_world'][1]:+.4f}, "
        f"{camera['camera_pos_world'][2]:+.4f}] m"
    )

    print()
    print("True ground-point NED:")
    print(
        f"  [{target_ned[0]:+.4f}, "
        f"{target_ned[1]:+.4f}, "
        f"{target_ned[2]:+.4f}] m"
    )

    print()
    print("Camera ray:")
    print(
        f"  [{ray_camera[0]:+.6f}, "
        f"{ray_camera[1]:+.6f}, "
        f"{ray_camera[2]:+.6f}]"
    )

    print()
    print("Pixel comparison:")
    print(
        f"  Predicted = "
        f"({predicted_u:.3f}, {predicted_v:.3f})"
    )

    print(
        f"  Observed  = "
        f"({OBSERVED_U:.3f}, {OBSERVED_V:.3f})"
    )

    print()
    print(
        f"  Pixel error = {error:.3f} px"
    )

    print(
        f"  Δu = {du:+.3f} px"
    )

    print(
        f"  Δv = {dv:+.3f} px"
    )


if __name__ == "__main__":
    main()
