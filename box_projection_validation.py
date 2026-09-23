import math
import subprocess
import re

import numpy as np


# ============================================================
# Camera
# ============================================================

FX = 205.46962738037109
FY = 205.46965599060059
CX = 320.0
CY = 240.0


# ============================================================
# Observed detector result
# ============================================================

OBSERVED_BBOX = {
    "x": 348.0,
    "y": 216.0,
    "w": 22.0,
    "h": 16.0,
}


# ============================================================
# Target geometry
# ============================================================

TARGET_CENTER = np.array([
    2.0,
    5.0,
    0.25
])

TARGET_SIZE = np.array([
    1.0,
    0.7,
    0.5
])


# ============================================================
# Rotation matrices
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

    return (
        Rz(yaw)
        @ Ry(pitch)
        @ Rx(roll)
    )


# ============================================================
# Quaternion
# ============================================================

def quaternion_to_R(q):

    x, y, z, w = q

    n = np.linalg.norm(q)

    x, y, z, w = q / n

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
            2 * (x*z - y*w),
            2 * (y*z + x*w),
            1 - 2 * (x*x + y*y)
        ]
    ], dtype=float)


# ============================================================
# Gazebo model output parser
# ============================================================

def run_gz_model(model, link=None):

    command = [
        "gz",
        "model",
        "-m",
        model
    ]

    if link is not None:
        command += [
            "--link",
            link
        ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr
        )

    return result.stdout


def parse_model_pose(text, name):

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
            f"Pose not found for {name}"
        )

    position = np.array([
        float(x)
        for x in match.group(1).split()
    ])

    rpy = np.array([
        float(x)
        for x in match.group(2).split()
    ])

    return position, rpy


# ============================================================
# Gazebo state
# ============================================================

def get_camera_pose():

    # UAV
    uav_text = run_gz_model(
        "hexacopter_with_ardupilot"
    )

    uav_pos, uav_rpy = parse_model_pose(
        uav_text,
        "hexacopter_with_ardupilot"
    )

    R_world_uav = rpy_to_R(
        *uav_rpy
    )

    # Gimbal
    gimbal_text = run_gz_model(
        "gimbal"
    )

    gimbal_pos, gimbal_rpy = parse_model_pose(
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

    pitch_pos, pitch_rpy = parse_model_pose(
        pitch_text,
        "pitch_link"
    )

    R_gimbal_pitch = rpy_to_R(
        *pitch_rpy
    )

    # Camera is at zero translation relative
    # to pitch_link according to our SDF.
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
    # Camera orientation
    # --------------------------------------------------------

    R_outer = (
        Rz(math.radians(90.0))
        @ Rx(math.radians(90.0))
    )

    R_sensor = (
        Ry(math.radians(-90.0))
        @ Rx(math.radians(-90.0))
    )

    pitch_angle = pitch_rpy[0]

    R_pitch = Rx(pitch_angle)

    R_gazebo_body_from_camera = (
        R_outer
        @ R_pitch
        @ R_sensor
    )

    # Gazebo camera -> NED
    R_gazebo_from_camera = np.array([
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0]
    ], dtype=float)

    R_ardupilot_from_gazebo = np.diag([
        1.0,
        -1.0,
        -1.0
    ])

    R_ned_body = rpy_to_R(
        0,
        0,
        0
    )

    # We'll replace UAV attitude below using ArduPilot
    # only for the orientation.

    return (
        uav_pos,
        uav_rpy,
        camera_pos_world,
        R_world_uav,
        R_gazebo_body_from_camera,
        R_gazebo_from_camera,
        R_ardupilot_from_gazebo
    )


# ============================================================
# Read UAV attitude from MAVLink
# ============================================================

def read_uav_attitude():

    from pymavlink import mavutil

    master = mavutil.mavlink_connection(
        "tcp:127.0.0.1:5762"
    )

    master.wait_heartbeat()

    start = time.time()

    while time.time() - start < 2:

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
        "ATTITUDE telemetry unavailable."
    )


# ============================================================
# Gazebo world -> NED
# ============================================================

def gazebo_world_to_ned(
    point_world,
    uav_world
):

    delta = (
        point_world
        - uav_world
    )

    return np.array([
        delta[1],
        delta[0],
        -delta[2]
    ])


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 65)
    print("3D TARGET BOX → IMAGE BOUNDING BOX VALIDATION")
    print("=" * 65)

    # --------------------------------------------------------
    # Get camera pose
    # --------------------------------------------------------

    (
        uav_pos,
        uav_rpy,
        camera_pos_world,
        _,
        R_gbc,
        R_gfc,
        R_ag
    ) = get_camera_pose()

    # --------------------------------------------------------
    # Get UAV attitude
    # --------------------------------------------------------

    roll, pitch, yaw = read_uav_attitude()

    R_ned_body = (
        Rz(yaw)
        @ Ry(pitch)
        @ Rx(roll)
    )

    # --------------------------------------------------------
    # Combined camera -> NED
    # --------------------------------------------------------

    R_camera_to_ned = (
        R_ned_body
        @ R_ag
        @ R_gbc
        @ R_gfc
    )

    # --------------------------------------------------------
    # Generate 8 target corners
    # --------------------------------------------------------

    half = TARGET_SIZE / 2.0

    corners = []

    for sx in [-1, 1]:

        for sy in [-1, 1]:

            for sz in [-1, 1]:

                corners.append(
                    TARGET_CENTER
                    + np.array([
                        sx * half[0],
                        sy * half[1],
                        sz * half[2]
                    ])
                )

    pixels = []

    for corner in corners:

        # World Gazebo -> NED relative UAV
        corner_ned = gazebo_world_to_ned(
            corner,
            uav_pos
        )

        # Need position relative to camera.
        # Camera position in Gazebo world -> NED.
        camera_ned = gazebo_world_to_ned(
            camera_pos_world,
            uav_pos
        )

        point_from_camera_ned = (
            corner_ned
            - camera_ned
        )

        ray_ned = (
            point_from_camera_ned
            / np.linalg.norm(
                point_from_camera_ned
            )
        )

        # NED -> camera
        ray_camera = (
            R_camera_to_ned.T
            @ ray_ned
        )

        if ray_camera[2] <= 0:
            continue

        u = (
            FX
            * ray_camera[0]
            / ray_camera[2]
            + CX
        )

        v = (
            FY
            * ray_camera[1]
            / ray_camera[2]
            + CY
        )

        pixels.append(
            (u, v)
        )

    if len(pixels) < 4:

        raise RuntimeError(
            "Too few visible target corners."
        )

    # --------------------------------------------------------
    # Predicted 2D bounding box
    # --------------------------------------------------------

    min_u = min(
        p[0] for p in pixels
    )

    max_u = max(
        p[0] for p in pixels
    )

    min_v = min(
        p[1] for p in pixels
    )

    max_v = max(
        p[1] for p in pixels
    )

    predicted_w = (
        max_u - min_u
    )

    predicted_h = (
        max_v - min_v
    )

    predicted_cx = (
        min_u + max_u
    ) / 2.0

    predicted_cy = (
        min_v + max_v
    ) / 2.0

    # --------------------------------------------------------
    # Compare with detector
    # --------------------------------------------------------

    observed_x = OBSERVED_BBOX["x"]
    observed_y = OBSERVED_BBOX["y"]

    observed_w = OBSERVED_BBOX["w"]
    observed_h = OBSERVED_BBOX["h"]

    observed_cx = (
        observed_x
        + observed_w / 2.0
    )

    observed_cy = (
        observed_y
        + observed_h / 2.0
    )

    center_error = math.hypot(
        predicted_cx - observed_cx,
        predicted_cy - observed_cy
    )

    size_error = math.hypot(
        predicted_w - observed_w,
        predicted_h - observed_h
    )

    print()
    print("Camera world position:")
    print(
        f"  [{camera_pos_world[0]:+.4f}, "
        f"{camera_pos_world[1]:+.4f}, "
        f"{camera_pos_world[2]:+.4f}] m"
    )

    print()
    print("Predicted bounding box:")
    print(
        f"  x = {min_u:.3f}"
    )
    print(
        f"  y = {min_v:.3f}"
    )
    print(
        f"  w = {predicted_w:.3f}"
    )
    print(
        f"  h = {predicted_h:.3f}"
    )

    print()
    print("Predicted bbox center:")
    print(
        f"  ({predicted_cx:.3f}, "
        f"{predicted_cy:.3f})"
    )

    print()
    print("Observed detector:")
    print(
        f"  x = {observed_x:.3f}"
    )
    print(
        f"  y = {observed_y:.3f}"
    )
    print(
        f"  w = {observed_w:.3f}"
    )
    print(
        f"  h = {observed_h:.3f}"
    )

    print()
    print("Observed bbox center:")
    print(
        f"  ({observed_cx:.3f}, "
        f"{observed_cy:.3f})"
    )

    print()
    print("Comparison:")
    print(
        f"  Center error = "
        f"{center_error:.3f} px"
    )

    print(
        f"  Size error   = "
        f"{size_error:.3f} px"
    )


if __name__ == "__main__":
    import time
    main()
