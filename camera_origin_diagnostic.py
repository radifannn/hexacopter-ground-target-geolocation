import math
import re
import subprocess

import numpy as np


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
    return Rz(yaw) @ Ry(pitch) @ Rx(roll)


# ============================================================
# Run gz model command
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
            result.stderr.strip()
        )

    return result.stdout


# ============================================================
# Extract the first Pose [ XYZ ] [ RPY ] belonging to a
# specific Name entry.
# ============================================================

def parse_named_pose(text, target_name):

    # Find the "- Name: ..." line.
    name_match = re.search(
        rf"- Name:\s*{re.escape(target_name)}\s*$",
        text,
        re.MULTILINE
    )

    if not name_match:
        raise RuntimeError(
            f"Could not find Name: {target_name}"
        )

    remaining = text[name_match.end():]

    pose_match = re.search(
        r"- Pose \[ XYZ \(m\) \] \[ RPY \(rad\) \]:\s*"
        r"\n\s*\[([^\]]+)\]\s*"
        r"\n\s*\[([^\]]+)\]",
        remaining
    )

    if not pose_match:
        raise RuntimeError(
            f"Could not find Pose for {target_name}"
        )

    position = np.array([
        float(x)
        for x in pose_match.group(1).split()
    ])

    rpy = np.array([
        float(x)
        for x in pose_match.group(2).split()
    ])

    return position, rpy


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 65)
    print("CAMERA OPTICAL CENTER DIAGNOSTIC")
    print("=" * 65)

    # --------------------------------------------------------
    # UAV world pose
    # --------------------------------------------------------

    uav_text = run_gz_model(
        "hexacopter_with_ardupilot"
    )

    uav_pos, uav_rpy = parse_named_pose(
        uav_text,
        "hexacopter_with_ardupilot"
    )

    R_world_uav = rpy_to_R(*uav_rpy)

    # --------------------------------------------------------
    # Gimbal pose relative to UAV
    # --------------------------------------------------------

    gimbal_text = run_gz_model(
        "gimbal"
    )

    gimbal_pos, gimbal_rpy = parse_named_pose(
        gimbal_text,
        "gimbal"
    )

    R_uav_gimbal = rpy_to_R(*gimbal_rpy)

    # --------------------------------------------------------
    # Pitch link pose relative to gimbal
    # --------------------------------------------------------

    pitch_text = run_gz_model(
        "gimbal",
        "pitch_link"
    )

    pitch_pos, pitch_rpy = parse_named_pose(
        pitch_text,
        "pitch_link"
    )

    R_gimbal_pitch = rpy_to_R(*pitch_rpy)

    # --------------------------------------------------------
    # Camera sensor position relative pitch_link
    #
    # From the SDF:
    #   <pose>0 0 0 -1.57 -1.57 0</pose>
    #
    # Translation = zero.
    # --------------------------------------------------------

    camera_pos_pitch = np.array([
        0.0,
        0.0,
        0.0
    ])

    # --------------------------------------------------------
    # Compose position hierarchy
    #
    # UAV world
    #   -> gimbal
    #      -> pitch_link
    #         -> camera
    # --------------------------------------------------------

    camera_pos_gimbal = (
        pitch_pos
        + R_gimbal_pitch @ camera_pos_pitch
    )

    camera_pos_uav = (
        gimbal_pos
        + R_uav_gimbal @ camera_pos_gimbal
    )

    camera_pos_world = (
        uav_pos
        + R_world_uav @ camera_pos_uav
    )

    lever_arm = (
        camera_pos_world
        - uav_pos
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print()
    print("UAV world pose:")
    print(
        f"  Position = "
        f"[{uav_pos[0]:+.6f}, "
        f"{uav_pos[1]:+.6f}, "
        f"{uav_pos[2]:+.6f}] m"
    )

    print(
        f"  RPY = "
        f"[{math.degrees(uav_rpy[0]):+.3f}, "
        f"{math.degrees(uav_rpy[1]):+.3f}, "
        f"{math.degrees(uav_rpy[2]):+.3f}] deg"
    )

    print()
    print("Gimbal pose relative UAV:")
    print(
        f"  Position = "
        f"[{gimbal_pos[0]:+.6f}, "
        f"{gimbal_pos[1]:+.6f}, "
        f"{gimbal_pos[2]:+.6f}] m"
    )

    print(
        f"  RPY = "
        f"[{math.degrees(gimbal_rpy[0]):+.3f}, "
        f"{math.degrees(gimbal_rpy[1]):+.3f}, "
        f"{math.degrees(gimbal_rpy[2]):+.3f}] deg"
    )

    print()
    print("Pitch link relative gimbal:")
    print(
        f"  Position = "
        f"[{pitch_pos[0]:+.6f}, "
        f"{pitch_pos[1]:+.6f}, "
        f"{pitch_pos[2]:+.6f}] m"
    )

    print(
        f"  RPY = "
        f"[{math.degrees(pitch_rpy[0]):+.3f}, "
        f"{math.degrees(pitch_rpy[1]):+.3f}, "
        f"{math.degrees(pitch_rpy[2]):+.3f}] deg"
    )

    print()
    print("Camera optical center:")
    print(
        f"  World = "
        f"[{camera_pos_world[0]:+.6f}, "
        f"{camera_pos_world[1]:+.6f}, "
        f"{camera_pos_world[2]:+.6f}] m"
    )

    print()
    print("Camera lever arm from UAV:")
    print(
        f"  Δ = "
        f"[{lever_arm[0]:+.6f}, "
        f"{lever_arm[1]:+.6f}, "
        f"{lever_arm[2]:+.6f}] m"
    )

    print(
        f"  Magnitude = "
        f"{np.linalg.norm(lever_arm):.6f} m"
    )


if __name__ == "__main__":
    main()
