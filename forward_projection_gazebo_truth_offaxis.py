import math
import numpy as np

import static_geolocation_gazebo as geo


# Actual detector measurement
OBSERVED_U = 357.0
OBSERVED_V = 227.0

# Exact target position from our world.sdf
TARGET_WORLD = np.array([
    2.0,
    5.0,
    0.01
], dtype=float)


def main():

    print()
    print("=" * 65)
    print("FORWARD PROJECTION — GAZEBO TRUTH VALIDATION")
    print("=" * 65)

    # --------------------------------------------------
    # 1. Read actual Gazebo UAV and gimbal state
    # --------------------------------------------------

    poses = geo.read_gazebo_state()

    uav_world = poses["hexacopter_with_ardupilot"]["position"]

    pitch_angle = geo.pitch_from_quaternion(
        poses["pitch_link"]["quaternion"]
    )

    # --------------------------------------------------
    # 2. Read UAV attitude
    # --------------------------------------------------

    attitude, gps = geo.read_uav_telemetry()

    roll = attitude["roll"]
    pitch = attitude["pitch"]
    yaw = attitude["yaw"]

    # --------------------------------------------------
    # 3. Exact target displacement in Gazebo world
    # --------------------------------------------------

    camera_world = uav_world + np.array([0.0101, 0.0103, -0.1052])
    target_vector_gazebo = TARGET_WORLD - camera_world

    # For our current world:
    #
    # Gazebo X -> NED East
    # Gazebo Y -> NED North
    # Gazebo Z -> NED Up
    #
    # Therefore:

    target_ned = np.array([
        target_vector_gazebo[1],     # North
        target_vector_gazebo[0],     # East
        -target_vector_gazebo[2]     # Down
    ])

    ray_ned = (
        target_ned /
        np.linalg.norm(target_ned)
    )

    print()
    print("1. Exact Gazebo geometry")

    print(
        f"   UAV world = "
        f"[{uav_world[0]:+.4f}, "
        f"{uav_world[1]:+.4f}, "
        f"{uav_world[2]:+.4f}] m"
    )

    print(
        f"   Target world = "
        f"[{TARGET_WORLD[0]:+.4f}, "
        f"{TARGET_WORLD[1]:+.4f}, "
        f"{TARGET_WORLD[2]:+.4f}] m"
    )

    print(
        f"   True target NED = "
        f"[{target_ned[0]:+.4f}, "
        f"{target_ned[1]:+.4f}, "
        f"{target_ned[2]:+.4f}] m"
    )

    print(
        f"   True NED ray = "
        f"[{ray_ned[0]:+.6f}, "
        f"{ray_ned[1]:+.6f}, "
        f"{ray_ned[2]:+.6f}]"
    )

    print()
    print("2. Current orientation")

    print(
        f"   Roll  = {math.degrees(roll):+.3f}°"
    )

    print(
        f"   Pitch = {math.degrees(pitch):+.3f}°"
    )

    print(
        f"   Yaw   = {math.degrees(yaw):+.3f}°"
    )

    print(
        f"   Gimbal pitch = "
        f"{math.degrees(pitch_angle):+.3f}°"
    )

    # --------------------------------------------------
    # 4. Same validated transformation chain
    # --------------------------------------------------

    R_gazebo_from_camera = np.array([
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0]
    ], dtype=float)

    R_outer = (
        geo.Rz(math.radians(90.0))
        @ geo.Rx(math.radians(90.0))
    )

    R_sensor = (
        geo.Ry(math.radians(-90.0))
        @ geo.Rx(math.radians(-90.0))
    )

    R_pitch = geo.Rx(pitch_angle)

    R_gazebo_body_from_camera = (
        R_outer
        @ R_pitch
        @ R_sensor
    )

    R_ardupilot_from_gazebo = np.diag([
        1.0,
        -1.0,
        -1.0
    ])

    R_ned_from_body = (
        geo.Rz(yaw)
        @ geo.Ry(pitch)
        @ geo.Rx(roll)
    )

    # --------------------------------------------------
    # 5. NED -> camera
    # --------------------------------------------------

    ray_body = (
        R_ned_from_body.T
        @ ray_ned
    )

    ray_gazebo_body = (
        R_ardupilot_from_gazebo.T
        @ ray_body
    )

    ray_gazebo_camera = (
        R_gazebo_body_from_camera.T
        @ ray_gazebo_body
    )

    ray_camera = (
        R_gazebo_from_camera.T
        @ ray_gazebo_camera
    )

    ray_camera /= np.linalg.norm(ray_camera)

    # --------------------------------------------------
    # 6. Camera ray -> image pixel
    # --------------------------------------------------

    if ray_camera[2] <= 0:

        raise RuntimeError(
            "Target is behind the camera."
        )

    predicted_u = (
        geo.FX
        * ray_camera[0]
        / ray_camera[2]
        + geo.CX
    )

    predicted_v = (
        geo.FY
        * ray_camera[1]
        / ray_camera[2]
        + geo.CY
    )

    error_u = predicted_u - OBSERVED_U
    error_v = predicted_v - OBSERVED_V

    pixel_error = math.hypot(
        error_u,
        error_v
    )

    print()
    print("3. Forward projection")

    print(
        f"   Predicted pixel = "
        f"({predicted_u:.3f}, {predicted_v:.3f})"
    )

    print(
        f"   Observed pixel  = "
        f"({OBSERVED_U:.3f}, {OBSERVED_V:.3f})"
    )

    print()
    print(
        f"   Pixel error = "
        f"{pixel_error:.3f} px"
    )

    print(
        f"   Δu = {error_u:+.3f} px"
    )

    print(
        f"   Δv = {error_v:+.3f} px"
    )


if __name__ == "__main__":
    main()
