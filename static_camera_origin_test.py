import math
import numpy as np

import geolocation_geometry as geo
import static_batch_validation as batch


# ------------------------------------------------------------
# Use bbox center, not contour centroid
# ------------------------------------------------------------

PIXEL_U = 357.0
PIXEL_V = 227.0

# Actual target center plane
TARGET_Z = 0.01


def find_target_pose(poses):

    for name, pose in poses.items():

        if "moving_target" in name:

            return pose

    raise RuntimeError(
        f"Target pose not found. Available: {list(poses)}"
    )


def main():

    print()
    print("=" * 65)
    print("STATIC GEOLOCATION — CAMERA ORIGIN TEST")
    print("=" * 65)

    # --------------------------------------------------------
    # Gazebo state
    # --------------------------------------------------------

    poses = batch.read_gazebo_state()

    uav_pose = poses["hexacopter_with_ardupilot"]
    pitch_pose = poses["pitch_link"]
    target_pose = find_target_pose(poses)

    uav_world = uav_pose["position"]
    target_world = target_pose["position"]

    # --------------------------------------------------------
    # Camera world position
    #
    # Same hierarchy as our previous diagnostic:
    # UAV -> gimbal -> pitch_link -> camera
    # --------------------------------------------------------

    import camera_origin_diagnostic as cod

    gimbal_text = cod.run_gz_model("gimbal")

    gimbal_pos, gimbal_rpy = cod.parse_named_pose(
        gimbal_text,
        "gimbal"
    )

    pitch_text = cod.run_gz_model(
        "gimbal",
        "pitch_link"
    )

    pitch_pos, pitch_rpy = cod.parse_named_pose(
        pitch_text,
        "pitch_link"
    )

    uav_text = cod.run_gz_model(
        "hexacopter_with_ardupilot"
    )

    uav_check_pos, uav_rpy = cod.parse_named_pose(
        uav_text,
        "hexacopter_with_ardupilot"
    )

    R_world_uav = cod.rpy_to_R(*uav_rpy)
    R_uav_gimbal = cod.rpy_to_R(*gimbal_rpy)
    R_gimbal_pitch = cod.rpy_to_R(*pitch_rpy)

    camera_pos_uav = (
        gimbal_pos
        + R_uav_gimbal @ (
            pitch_pos
        )
    )

    camera_world = (
        uav_check_pos
        + R_world_uav @ camera_pos_uav
    )

    # --------------------------------------------------------
    # UAV attitude
    # --------------------------------------------------------

    master = batch.connect_mavlink()

    attitude, _ = batch.read_latest_telemetry(
        master
    )

    if attitude is None:

        raise RuntimeError(
            "Could not read UAV attitude."
        )

    # --------------------------------------------------------
    # Pixel -> camera ray
    # --------------------------------------------------------

    ray_camera = geo.pixel_to_camera_ray(
        PIXEL_U,
        PIXEL_V
    )

    # --------------------------------------------------------
    # Camera ray -> NED
    # --------------------------------------------------------

    ray_ned = geo.camera_ray_to_ned_static(
        ray_camera,
        pitch_pose["quaternion"],
        attitude["roll"],
        attitude["pitch"],
        attitude["yaw"]
    )

    # --------------------------------------------------------
    # Camera position relative UAV in NED
    #
    # Gazebo:
    #   X = East
    #   Y = North
    #   Z = Up
    #
    # NED:
    #   N = +Y
    #   E = +X
    #   D = -Z
    # --------------------------------------------------------

    camera_delta = (
        camera_world - uav_world
    )

    camera_ned = np.array([
        camera_delta[1],
        camera_delta[0],
        -camera_delta[2]
    ])

    # --------------------------------------------------------
    # Target horizontal truth
    # --------------------------------------------------------

    target_delta = (
        target_world - uav_world
    )

    true_target_ned = np.array([
        target_delta[1],
        target_delta[0],
        -target_delta[2]
    ])

    # --------------------------------------------------------
    # Target plane relative to UAV
    # --------------------------------------------------------

    target_down = (
        uav_world[2] - TARGET_Z
    )

    # Camera's own down coordinate relative UAV
    camera_down = camera_ned[2]

    # Distance from camera to target plane
    camera_to_plane_down = (
        target_down - camera_down
    )

    if camera_to_plane_down <= 0:

        raise RuntimeError(
            "Target plane is not below camera."
        )

    if ray_ned[2] <= 0:

        raise RuntimeError(
            "Ray does not point toward ground."
        )

    # --------------------------------------------------------
    # Camera-origin ray intersection
    # --------------------------------------------------------

    scale = (
        camera_to_plane_down
        / ray_ned[2]
    )

    estimated_ned = (
        camera_ned
        + scale * ray_ned
    )

    error_n = (
        estimated_ned[0]
        - true_target_ned[0]
    )

    error_e = (
        estimated_ned[1]
        - true_target_ned[1]
    )

    error_h = math.hypot(
        error_n,
        error_e
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print()
    print("Pixel:")
    print(
        f"  ({PIXEL_U:.1f}, {PIXEL_V:.1f})"
    )

    print()
    print("Camera world:")
    print(
        f"  [{camera_world[0]:+.4f}, "
        f"{camera_world[1]:+.4f}, "
        f"{camera_world[2]:+.4f}]"
    )

    print()
    print("Camera NED relative UAV:")
    print(
        f"  [{camera_ned[0]:+.4f}, "
        f"{camera_ned[1]:+.4f}, "
        f"{camera_ned[2]:+.4f}] m"
    )

    print()
    print("True target NED:")
    print(
        f"  [{true_target_ned[0]:+.4f}, "
        f"{true_target_ned[1]:+.4f}, "
        f"{true_target_ned[2]:+.4f}] m"
    )

    print()
    print("NED ray:")
    print(
        f"  [{ray_ned[0]:+.6f}, "
        f"{ray_ned[1]:+.6f}, "
        f"{ray_ned[2]:+.6f}]"
    )

    print()
    print("Camera -> target plane:")
    print(
        f"  Down distance = "
        f"{camera_to_plane_down:.4f} m"
    )

    print()
    print("Estimated target NED:")
    print(
        f"  [{estimated_ned[0]:+.4f}, "
        f"{estimated_ned[1]:+.4f}, "
        f"{estimated_ned[2]:+.4f}] m"
    )

    print()
    print("Error:")
    print(
        f"  North = {error_n:+.4f} m"
    )
    print(
        f"  East  = {error_e:+.4f} m"
    )
    print(
        f"  Horizontal = {error_h:.4f} m"
    )


if __name__ == "__main__":
    main()
