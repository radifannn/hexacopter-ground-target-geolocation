import csv
import math
import time

import numpy as np

import geolocation_geometry as geo
import static_batch_validation as batch


# ============================================================
# Experiment configuration
# ============================================================

SAMPLES = 100
SAMPLE_INTERVAL = 0.20

# Detector measurement from the current static test
PIXEL_U = 357.9
PIXEL_V = 223.6

OUTPUT_CSV = (
    "/home/rifqi-radifan/gz_ws/geolocation/"
    "static_truth_validation_100samples.csv"
)


# ============================================================
# Find moving target pose
# ============================================================

def find_target_pose(poses):
    """
    Find the moving_target pose from Gazebo dynamic_pose data.

    Gazebo may expose the model/link using different naming
    conventions, so do not require one exact string.
    """

    # Prefer names that clearly identify the target base link.
    preferred = [
        "moving_target::base_link",
        "moving_target",
        "moving_target/base_link",
    ]

    for name in preferred:
        if name in poses:
            return poses[name]

    # Fallback: any pose containing "moving_target".
    candidates = [
        (name, pose)
        for name, pose in poses.items()
        if "moving_target" in name
    ]

    if candidates:
        # Prefer a base_link if multiple target-related poses exist.
        for name, pose in candidates:
            if "base_link" in name:
                return pose

        return candidates[0][1]

    raise RuntimeError(
        "Moving target pose not found.\n"
        f"Available pose names: {list(poses.keys())}"
    )


# ============================================================
# Estimate one sample
# ============================================================

def estimate_one(poses, attitude):
    """
    Estimate target position from the camera pixel.

    Reference position is NOT used here.
    """

    uav_pose = poses["hexacopter_with_ardupilot"]
    pitch_pose = poses["pitch_link"]

    # --------------------------------------------------------
    # Pixel -> camera ray
    # --------------------------------------------------------

    ray_camera = geo.pixel_to_camera_ray(
        PIXEL_U,
        PIXEL_V
    )

    # --------------------------------------------------------
    # Camera -> NED
    # --------------------------------------------------------

    ray_ned = geo.camera_ray_to_ned_static(
        ray_camera,
        pitch_pose["quaternion"],
        attitude["roll"],
        attitude["pitch"],
        attitude["yaw"]
    )

    # --------------------------------------------------------
    # Use the exact target plane from Gazebo.
    #
    # Target base_link is at its actual Z coordinate.
    # --------------------------------------------------------

    target_pose = find_target_pose(poses)

    uav_z = uav_pose["position"][2]
    target_z = target_pose["position"][2]

    vertical_distance = uav_z - target_z

    target_est_ned = geo.ray_plane_intersection(
        ray_ned,
        vertical_distance
    )

    return {
        "estimate_ned": target_est_ned,
        "ray_ned": ray_ned,
        "pitch_pose": pitch_pose,
        "uav_pose": uav_pose,
        "target_pose": target_pose
    }


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)
    print("STATIC GEOLOCATION — GAZEBO TRUTH VALIDATION")
    print("=" * 70)

    print()
    print("Conditions:")
    print(f"  Pixel      = ({PIXEL_U}, {PIXEL_V})")
    print(f"  Samples    = {SAMPLES}")
    print(f"  Interval   = {SAMPLE_INTERVAL} s")
    print("  Target     = stationary")
    print("  RC7        = 1400")

    # --------------------------------------------------------
    # Connect once
    # --------------------------------------------------------

    master = batch.connect_mavlink()

    rows = []

    print()
    print("Collecting samples...")

    for i in range(SAMPLES):

        sample_start = time.monotonic()

        try:

            # Read one Gazebo snapshot.
            poses = batch.read_gazebo_state()

            # Read UAV attitude.
            attitude, _ = batch.read_latest_telemetry(
                master
            )

            if attitude is None:
                raise RuntimeError(
                    "UAV attitude unavailable."
                )

            result = estimate_one(
                poses,
                attitude
            )

            # ------------------------------------------------
            # Extract exact Gazebo truth
            # ------------------------------------------------

            uav_world = result["uav_pose"]["position"]
            target_world = result["target_pose"]["position"]

            # Gazebo world:
            #   X -> East
            #   Y -> North
            #   Z -> Up
            #
            # NED:
            #   X -> North
            #   Y -> East
            #   Z -> Down

            true_north = (
                target_world[1]
                - uav_world[1]
            )

            true_east = (
                target_world[0]
                - uav_world[0]
            )

            true_down = (
                uav_world[2]
                - target_world[2]
            )

            estimated_north = (
                result["estimate_ned"][0]
            )

            estimated_east = (
                result["estimate_ned"][1]
            )

            estimated_down = (
                result["estimate_ned"][2]
            )

            # ------------------------------------------------
            # Errors
            # ------------------------------------------------

            north_error = (
                estimated_north
                - true_north
            )

            east_error = (
                estimated_east
                - true_east
            )

            down_error = (
                estimated_down
                - true_down
            )

            horizontal_error = math.hypot(
                north_error,
                east_error
            )

            # ------------------------------------------------
            # Gimbal pitch for logging
            # ------------------------------------------------

            gimbal_pitch = (
                geo.static_pitch_from_quaternion(
                    result["pitch_pose"]["quaternion"]
                )
            )

            row = {
                "sample": i + 1,

                "u": PIXEL_U,
                "v": PIXEL_V,

                "estimated_north":
                    estimated_north,

                "estimated_east":
                    estimated_east,

                "estimated_down":
                    estimated_down,

                "true_north":
                    true_north,

                "true_east":
                    true_east,

                "true_down":
                    true_down,

                "north_error":
                    north_error,

                "east_error":
                    east_error,

                "down_error":
                    down_error,

                "horizontal_error":
                    horizontal_error,

                "uav_x":
                    uav_world[0],

                "uav_y":
                    uav_world[1],

                "uav_z":
                    uav_world[2],

                "target_x":
                    target_world[0],

                "target_y":
                    target_world[1],

                "target_z":
                    target_world[2],

                "roll_deg":
                    math.degrees(
                        attitude["roll"]
                    ),

                "pitch_deg":
                    math.degrees(
                        attitude["pitch"]
                    ),

                "yaw_deg":
                    math.degrees(
                        attitude["yaw"]
                    ),

                "gimbal_pitch_deg":
                    math.degrees(
                        gimbal_pitch
                    ),

                "ray_n":
                    result["ray_ned"][0],

                "ray_e":
                    result["ray_ned"][1],

                "ray_d":
                    result["ray_ned"][2],
            }

            rows.append(row)

            print(
                f"  [{i+1:03d}/{SAMPLES}] "
                f"true N={true_north:+.3f} m  "
                f"true E={true_east:+.3f} m  "
                f"est N={estimated_north:+.3f} m  "
                f"est E={estimated_east:+.3f} m  "
                f"error={horizontal_error:.4f} m"
            )

        except Exception as exc:

            print(
                f"  [{i+1:03d}/{SAMPLES}] "
                f"ERROR: {exc}"
            )

        elapsed = (
            time.monotonic()
            - sample_start
        )

        remaining = (
            SAMPLE_INTERVAL
            - elapsed
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

    north_errors = np.array([
        row["north_error"]
        for row in rows
    ])

    east_errors = np.array([
        row["east_error"]
        for row in rows
    ])

    down_errors = np.array([
        row["down_error"]
        for row in rows
    ])

    horizontal_errors = np.array([
        row["horizontal_error"]
        for row in rows
    ])

    estimated_north = np.array([
        row["estimated_north"]
        for row in rows
    ])

    estimated_east = np.array([
        row["estimated_east"]
        for row in rows
    ])

    true_north = np.array([
        row["true_north"]
        for row in rows
    ])

    true_east = np.array([
        row["true_east"]
        for row in rows
    ])

    # Bias
    north_bias = np.mean(north_errors)
    east_bias = np.mean(east_errors)
    down_bias = np.mean(down_errors)

    # Standard deviation
    north_std = np.std(north_errors)
    east_std = np.std(east_errors)
    down_std = np.std(down_errors)

    # MAE
    north_mae = np.mean(
        np.abs(north_errors)
    )

    east_mae = np.mean(
        np.abs(east_errors)
    )

    horizontal_mae = np.mean(
        horizontal_errors
    )

    # RMSE
    north_rmse = math.sqrt(
        np.mean(north_errors ** 2)
    )

    east_rmse = math.sqrt(
        np.mean(east_errors ** 2)
    )

    horizontal_rmse = math.sqrt(
        np.mean(horizontal_errors ** 2)
    )

    print()
    print("=" * 70)
    print("GAZEBO TRUTH VALIDATION RESULTS")
    print("=" * 70)

    print()
    print(f"Valid samples = {len(rows)}")

    print()
    print("TRUE POSITION:")
    print(
        f"  North mean = "
        f"{np.mean(true_north):+.4f} m"
    )
    print(
        f"  East mean  = "
        f"{np.mean(true_east):+.4f} m"
    )

    print()
    print("ESTIMATED POSITION:")
    print(
        f"  North mean = "
        f"{np.mean(estimated_north):+.4f} m"
    )
    print(
        f"  East mean  = "
        f"{np.mean(estimated_east):+.4f} m"
    )

    print()
    print("SYSTEMATIC BIAS:")
    print(
        f"  North bias = "
        f"{north_bias:+.4f} m"
    )
    print(
        f"  East bias  = "
        f"{east_bias:+.4f} m"
    )
    print(
        f"  Down bias  = "
        f"{down_bias:+.4f} m"
    )

    print()
    print("ERROR STANDARD DEVIATION:")
    print(
        f"  North std = "
        f"{north_std:.4f} m"
    )
    print(
        f"  East std  = "
        f"{east_std:.4f} m"
    )
    print(
        f"  Down std  = "
        f"{down_std:.4f} m"
    )

    print()
    print("MAE:")
    print(
        f"  North MAE      = "
        f"{north_mae:.4f} m"
    )
    print(
        f"  East MAE       = "
        f"{east_mae:.4f} m"
    )
    print(
        f"  Horizontal MAE = "
        f"{horizontal_mae:.4f} m"
    )

    print()
    print("RMSE:")
    print(
        f"  North RMSE      = "
        f"{north_rmse:.4f} m"
    )
    print(
        f"  East RMSE       = "
        f"{east_rmse:.4f} m"
    )
    print(
        f"  Horizontal RMSE = "
        f"{horizontal_rmse:.4f} m"
    )

    print()
    print(
        f"Horizontal max = "
        f"{np.max(horizontal_errors):.4f} m"
    )

    print(
        f"Horizontal min = "
        f"{np.min(horizontal_errors):.4f} m"
    )

    print()
    print("CSV saved to:")
    print(f"  {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
