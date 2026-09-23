Hexacopter Ground-Target Geolocation

Research progress checkpoint for a final-year project on estimating the world/GPS coordinate of a ground target detected by a hexacopter camera.

1. Project objective

The main objective is:

Estimate the world/GPS coordinate of a ground target from the hexacopter's camera observation.

Important project interpretation:

The target's own GPS is ground truth/reference only for validation.

The target GPS is not an estimator input.

The camera detects/localizes the target in the image.

The gimbal is a 2-DOF system: roll + pitch.

The gimbal's intended role is to stabilize camera pointing relative to the Earth/world frame during UAV movement.

The gimbal is not intended to visually chase the target.

Static target geolocation is the current development stage.

Dynamic target estimation comes later.

MPC is a later supporting/control layer, not the current research focus.

EKF should only be introduced after the underlying static geometric geolocation is validated.

2. Intended processing pipeline

Camera
  ↓
Object detection / localization
  ↓
Pixel (u, v)
  ↓
Camera intrinsics
  ↓
Camera-frame ray
  ↓
Camera + gimbal + UAV attitude transform
  ↓
Earth / NED ray
  ↓
Ray-ground / ray-plane intersection
  ↓
Raw target position estimate
  ↓
Later: EKF / target-state estimation
  ↓
Validation against target GPS
  ↓
Later: dynamic target
  ↓
Later: MPC / UAV control

The immediate goal is to make the static geometric pipeline internally consistent before moving to dynamic estimation.

3. Simulation baseline

The project uses:

ArduPilot SITL

Gazebo Harmonic

Hexacopter model with gimbal and camera

Python

OpenCV / NumPy

MAVLink telemetry

Gazebo:

cd ~/gz_ws/src/ardupilot_gazebo
gz sim -v4 -r worlds/hexacopter_runway.sdf

ArduPilot SITL:

cd ~/ardupilot
source ~/venv-ardupilot/bin/activate
sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --map --console

The hexacopter motor mapping was configured and basic takeoff, hover, forward/reverse motion, and landing were validated.

Current mapping:

ch0 → rotor_4_joint → -838
ch1 → rotor_1_joint → +838
ch2 → rotor_0_joint → -838
ch3 → rotor_3_joint → +838
ch4 → rotor_5_joint → +838
ch5 → rotor_2_joint → -838

4. Camera intrinsics

Gazebo camera:

Image width  = 640 px
Image height = 480 px
Horizontal FOV = 2.0 rad

The actual Gazebo camera_info output was inspected and used as the authoritative calibration.

Current intrinsics:

fx = 205.46962738037109
fy = 205.46965599060059
cx = 320.0
cy = 240.0

Distortion coefficients are zero.

These values are used in:

geolocation/geolocation_geometry.py
geolocation/camera_intrinsics.py

5. Pixel → camera ray

The image point is converted to a normalized ray using:

ray = K^-1 [u, v, 1]^T

The Python-side simplified camera convention currently used is:

X = right
Y = down
Z = forward

Do not confuse this with Gazebo's native camera convention used in the simulation:

X = forward
Y = left
Z = up

Frame conversion is therefore a critical part of the implementation.

6. UAV telemetry

ArduPilot telemetry is read through MAVLink over:

tcp:127.0.0.1:5762

Current telemetry includes:

UAV GPS

roll

pitch

yaw

altitude

The project uses telemetry to determine UAV state for geolocation.

Relevant file:

geolocation/read_telemetry.py

7. Camera/gimbal/world transformation

A static transform chain was developed:

camera
  ↓
pitch_link
  ↓
gimbal
  ↓
hexacopter body
  ↓
NED

The static case has been tested with the UAV approximately level and the gimbal at a fixed pitch.

Observed directional behavior:

image center → approximately Down
image right  → approximately +East
image left   → approximately -East
image above  → approximately +North
image below  → approximately -North

This is an important milestone, but the implementation is currently a static-case prototype rather than the final general dynamic 2-DOF gimbal solution.

8. Ray → target plane

The ray is intersected with a horizontal target plane.

Conceptually:

t = vertical_distance / ray_down
target_ned = camera_origin_ned + t * ray_ned

A ground-plane/plane-intersection implementation exists in:

geolocation/geolocation_geometry.py

9. NED → GPS

For the small simulation area, local conversion is used:

dlat = North / R
dlon = East / (R cos(latitude))

with:

R = 6378137 m

The conversion is implemented in:

geolocation/geolocation_geometry.py

10. Static target experiments

10.1 Initial near-axis static target

An initial small stationary target near the image center was tested.

The resulting geometric geolocation was very accurate in that favorable configuration.

An exact Gazebo truth validation later produced approximately:

Horizontal RMSE ≈ 2.35 cm

This demonstrated that the overall static geometry can work accurately in an appropriate configuration.

10.2 Off-axis target

The target was then moved to approximately:

X = 2 m
Y = 5 m

The target appeared off-axis in the image.

This exposed a systematic error larger than the initial near-axis experiment.

This triggered investigation of:

camera intrinsics

camera/gimbal rotation

camera origin / lever arm

physical target point definition

bounding-box center interpretation

11. Target model simplification

The original target was a larger 3D box.

To remove ambiguity between:

object center

visible face center

ground contact point

bounding-box center

the target was simplified to a flat symmetric red marker.

Current marker:

World position:
X = 2.0 m
Y = 5.0 m
Z = 0.01 m

Size:
0.5 × 0.5 × 0.02 m

The lower surface therefore touches the ground.

Modified Gazebo files:

simulation/worlds/hexacopter_runway.sdf
simulation/models/moving_target/model.sdf

12. Current detector result

The red target detector is currently very stable.

Repeated output:

bbox   = (352, 222, 10, 10)
center = (357, 227)

The center remained stable over many consecutive frames in the static experiment.

Detector:

vision/red_target_detector.py

13. Camera optical-center investigation

A diagnostic script was created to inspect the camera's physical location relative to the UAV:

geolocation/camera_origin_diagnostic.py

A representative diagnostic gave a camera optical center roughly:

~0.106 m

from the UAV reference origin.

This confirms that the camera is not exactly collocated with the UAV reference origin.

However, a temporary approximate camera-origin correction was tested in the forward-projection script and did not solve the pixel mismatch. Therefore, the camera lever arm is known to exist, but it has not been shown to be the dominant source of the remaining error.

14. Current static camera-origin test

With the simplified marker and detector pixel:

Pixel = (357, 227)
Target plane Z = 0.01 m

a recent test produced approximately:

True target NED:
[+4.9953, +1.9914, +10.1802] m

Estimated target NED:
[+4.9360, +2.0879, +10.1802] m

Errors:

North = -0.0593 m
East  = +0.0965 m
Horizontal = 0.1132 m

Therefore the simplified marker did not remove the off-axis error.

15. Forward-projection investigation

A forward model was used to ask:

If the target world position is known exactly, what camera pixel does the current geometric model predict?

Current detector measurement:

Observed pixel = (357, 227)

The clean UAV-origin forward projection predicted approximately:

Predicted pixel = (355.059, 226.219)

giving:

Pixel error ≈ 2.09 px
Δu ≈ -1.94 px
Δv ≈ -0.78 px

This establishes a concrete mismatch between the known Gazebo geometry and the current camera projection model.

A temporary approximate camera-origin correction was also tested, but it increased the mismatch to about:

2.56 px

That correction was temporary and should not be treated as part of the validated algorithm.

16. Historical contour-centroid experiment

A contour-centroid measurement was tested as an alternative to bounding-box center.

The experiment produced:

TRUE POSITION:
North mean = +4.9985 m
East mean  = +1.9958 m

ESTIMATED POSITION:
North mean = +5.0879 m
East mean  = +2.0296 m

SYSTEMATIC BIAS:
North bias = +0.0894 m
East bias  = +0.0339 m

Horizontal MAE  = 0.0957 m
Horizontal RMSE = 0.0957 m

This experiment made the result worse in that configuration.

Therefore:

Contour centroid is not currently considered the solution.

17. Main unresolved problem

The current research problem is:

Why does the forward projection differ from the actual detected pixel by roughly 2 pixels, and why does that mismatch propagate into roughly centimeter-to-decimeter static position error at ~10 m range?

Current hypotheses:

A. Camera coordinate convention

Gazebo camera axes and the simplified Python camera axes differ.

A small remaining rotation/sign/order mistake could produce the observed angular bias.

B. Exact gimbal rotation chain

The nested SDF hierarchy includes gimbal, roll, and pitch transformations.

The static implementation may still contain a small mismatch in:

joint axis interpretation

SDF rotation order

fixed sensor rotation

pitch sign

frame conversion

C. Camera origin / lever arm

The camera is physically offset from the UAV origin.

This effect exists, but the temporary approximate correction did not explain the observed mismatch.

D. Image measurement definition

The detector returns a bounding-box center:

(357, 227)

This may not correspond exactly to the physical 3D point used as the target reference.

E. Forward/inverse consistency

The forward and inverse mathematical models need to be checked as true inverses using the same exact coordinate frames and origins.

18. Current important files

Geolocation

geolocation/
├── geolocation_geometry.py
├── camera_intrinsics.py
├── read_telemetry.py
├── static_estimator.py
├── static_batch_validation.py
├── static_truth_validation.py
├── static_camera_origin_test.py
├── forward_projection_gazebo_truth_offaxis.py
├── box_projection_validation.py
├── ground_contact_projection.py
├── camera_origin_diagnostic.py
└── red_detector_centroid_test.py

Vision

vision/
└── red_target_detector.py

Simulation files modified during this project

simulation/
├── worlds/
│   └── hexacopter_runway.sdf
└── models/
    └── moving_target/
        └── model.sdf

The existing repository/version of the hexacopter model should also be retained if needed to reproduce the gimbal/camera hierarchy.

19. What has been validated vs. what is still experimental

Demonstrated / working

ArduPilot SITL + Gazebo hexacopter baseline

Hexacopter motor mapping

Camera stream

Gazebo camera intrinsics

Pixel → camera ray

UAV MAVLink telemetry

Static camera/gimbal → NED directional behavior

Ray → horizontal plane intersection

NED → GPS conversion

Initial near-axis static geolocation

Stable target detection

Still being investigated

Exact off-axis camera projection

Exact camera optical-center handling

Full general 2-DOF gimbal transform

Exact physical meaning of detected pixel center

Forward/inverse transform consistency

Off-axis static accuracy

Not the current focus

EKF tuning

Dynamic target

MPC tuning

autonomous target chasing

global path planning

obstacle avoidance

20. Recommended next step

Do not immediately tune thresholds, EKF parameters, or MPC parameters.

The next investigation should be:

1. Restore the forward-projection script to the clean
   UAV-origin version if necessary.

2. Reconstruct the exact Gazebo camera pose from the
   full UAV → gimbal → roll → pitch → camera chain.

3. Write down every coordinate-frame convention explicitly.

4. Build one authoritative rotation chain.

5. Forward-project the known target point.

6. Invert the same transform back to the target.

7. Compare predicted pixel vs detected pixel.

8. Determine which physical 3D point corresponds to the
   chosen detector pixel.

9. Only after static geometry is internally consistent:
      → batch static validation
      → dynamic target
      → EKF
      → later MPC

The priority is therefore:

Solve the static geometric inconsistency first.
