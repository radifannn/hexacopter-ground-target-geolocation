import math

# ------------------------------------------------------------
# Camera parameters from the Gazebo camera model
# ------------------------------------------------------------

WIDTH = 640
HEIGHT = 480
HORIZONTAL_FOV = 2.0  # radians


# ------------------------------------------------------------
# Calculate focal length in pixels
# ------------------------------------------------------------

fx = WIDTH / (2.0 * math.tan(HORIZONTAL_FOV / 2.0))

# Assume square pixels
fy = fx


# ------------------------------------------------------------
# Principal point
# ------------------------------------------------------------

cx = (WIDTH - 1) / 2.0
cy = (HEIGHT - 1) / 2.0


# ------------------------------------------------------------
# Intrinsic matrix
# ------------------------------------------------------------

K = [
    [fx, 0.0, cx],
    [0.0, fy, cy],
    [0.0, 0.0, 1.0]
]


# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

print("Camera resolution:")
print(f"  Width  = {WIDTH}")
print(f"  Height = {HEIGHT}")

print("\nHorizontal FOV:")
print(f"  HFOV = {HORIZONTAL_FOV:.4f} rad")
print(f"  HFOV = {math.degrees(HORIZONTAL_FOV):.2f} deg")

print("\nCamera intrinsic parameters:")
print(f"  fx = {fx:.4f} px")
print(f"  fy = {fy:.4f} px")
print(f"  cx = {cx:.4f} px")
print(f"  cy = {cy:.4f} px")

print("\nIntrinsic matrix K:")
for row in K:
    print(" ", row)
