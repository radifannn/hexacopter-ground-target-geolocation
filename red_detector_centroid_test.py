import cv2
import numpy as np
import math


PIPELINE = (
    'udpsrc port=5600 caps="application/x-rtp, '
    'media=video, encoding-name=H264, payload=96" ! '
    'rtph264depay ! h264parse ! avdec_h264 ! '
    'videoconvert ! appsink'
)


LOWER_RED_1 = np.array([0, 100, 80])
UPPER_RED_1 = np.array([10, 255, 255])

LOWER_RED_2 = np.array([170, 100, 80])
UPPER_RED_2 = np.array([180, 255, 255])


def main():

    cap = cv2.VideoCapture(
        PIPELINE,
        cv2.CAP_GSTREAMER
    )

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open camera stream."
        )

    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV
        )

        mask1 = cv2.inRange(
            hsv,
            LOWER_RED_1,
            UPPER_RED_1
        )

        mask2 = cv2.inRange(
            hsv,
            LOWER_RED_2,
            UPPER_RED_2
        )

        mask = cv2.bitwise_or(
            mask1,
            mask2
        )

        kernel = np.ones(
            (3, 3),
            np.uint8
        )

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

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:

            contour = max(
                contours,
                key=cv2.contourArea
            )

            area = cv2.contourArea(contour)

            if area >= 10:

                x, y, w, h = cv2.boundingRect(
                    contour
                )

                # Bounding-box center
                bbox_cx = x + w / 2.0
                bbox_cy = y + h / 2.0

                # Contour centroid
                M = cv2.moments(contour)

                if M["m00"] != 0:

                    centroid_x = (
                        M["m10"] / M["m00"]
                    )

                    centroid_y = (
                        M["m01"] / M["m00"]
                    )

                else:

                    centroid_x = bbox_cx
                    centroid_y = bbox_cy

                print(
                    f"bbox_center="
                    f"({bbox_cx:.2f}, {bbox_cy:.2f}) | "
                    f"contour_centroid="
                    f"({centroid_x:.2f}, {centroid_y:.2f}) | "
                    f"bbox={w}x{h} | "
                    f"area={area:.1f}"
                )

                display = frame.copy()

                cv2.rectangle(
                    display,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                cv2.circle(
                    display,
                    (
                        int(round(bbox_cx)),
                        int(round(bbox_cy))
                    ),
                    4,
                    (255, 0, 0),
                    -1
                )

                cv2.circle(
                    display,
                    (
                        int(round(centroid_x)),
                        int(round(centroid_y))
                    ),
                    4,
                    (0, 0, 255),
                    -1
                )

                cv2.putText(
                    display,
                    "Blue = bbox center | Red = contour centroid",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.imshow(
                    "Detector comparison",
                    display
                )

        key = cv2.waitKey(1)

        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
