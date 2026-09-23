import cv2
import numpy as np
import time
import os


# --------------------------------------------------
# Gazebo camera -> GStreamer -> OpenCV
# --------------------------------------------------
pipeline = (
    "udpsrc port=5600 caps=\"application/x-rtp, media=video, "
    "encoding-name=H264, payload=96\" ! "
    "rtph264depay ! h264parse ! avdec_h264 ! "
    "videoconvert ! appsink"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Could not open camera stream.")
    raise SystemExit(1)

print("Camera stream opened.")
print("Press 'q' to quit.")
print("Press 's' to save the current frame.")


last_print = 0.0

while True:
    ret, frame = cap.read()

    if not ret:
        print("No frame received.")
        continue

    # --------------------------------------------------
    # Convert BGR -> HSV
    # --------------------------------------------------
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --------------------------------------------------
    # Red color ranges
    # Red wraps around HSV hue = 0/180,
    # so we use two masks.
    # --------------------------------------------------
    lower_red_1 = np.array([0, 100, 80])
    upper_red_1 = np.array([10, 255, 255])

    lower_red_2 = np.array([170, 100, 80])
    upper_red_2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

    mask = cv2.bitwise_or(mask1, mask2)

    # --------------------------------------------------
    # Clean up the mask
    # --------------------------------------------------
    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # --------------------------------------------------
    # Find contours
    # --------------------------------------------------
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    target_found = False

    best_contour = None
    best_area = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        # Ignore tiny noise
        if area < 10:
            continue

        if area > best_area:
            best_area = area
            best_contour = contour

    # --------------------------------------------------
    # Process detected target
    # --------------------------------------------------
    if best_contour is not None:

        x, y, w, h = cv2.boundingRect(best_contour)

        center_x = x + w // 2
        center_y = y + h // 2

        target_found = True

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Draw target center
        cv2.circle(
            frame,
            (center_x, center_y),
            4,
            (0, 255, 0),
            -1
        )

        # Label
        cv2.putText(
            frame,
            "RED TARGET",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

        # Print detection information periodically
        current_time = time.time()

        if current_time - last_print > 0.5:
            print(
                f"Target detected | "
                f"bbox=({x}, {y}, {w}, {h}) | "
                f"center=({center_x}, {center_y}) | "
                f"area={best_area:.1f}"
            )
            last_print = current_time

    else:
        current_time = time.time()

        if current_time - last_print > 1.0:
            print("Target not detected.")
            last_print = current_time

    # --------------------------------------------------
    # Draw camera center
    # --------------------------------------------------
    height, width = frame.shape[:2]

    image_center_x = width // 2
    image_center_y = height // 2

    cv2.drawMarker(
        frame,
        (image_center_x, image_center_y),
        (255, 255, 255),
        cv2.MARKER_CROSS,
        20,
        1
    )

    # --------------------------------------------------
    # Show result
    # --------------------------------------------------
    cv2.imshow("Red Target Detector", frame)

    # --------------------------------------------------
    # Keyboard
    # --------------------------------------------------
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        save_path = os.path.expanduser("~/gz_ws/red_target_test.jpg")
        cv2.imwrite(save_path, frame)
        print(f"Frame saved to: {save_path}")

    elif key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
