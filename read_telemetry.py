from pymavlink import mavutil
import math
import time


# ============================================================
# CONNECTION
# ============================================================

CONNECTION = "tcp:127.0.0.1:5762"


# ============================================================
# CONNECT
# ============================================================

print("Connecting to ArduPilot SITL...")

master = mavutil.mavlink_connection(CONNECTION)

print("Waiting for heartbeat...")

master.wait_heartbeat()

print(
    f"Connected: system={master.target_system}, "
    f"component={master.target_component}"
)


# ============================================================
# REQUEST TELEMETRY RATES
# ============================================================

def request_message_interval(message_id, interval_us):
    """
    Request a MAVLink message at a specific interval.

    interval_us:
        100000 = 10 Hz
        200000 = 5 Hz
        1000000 = 1 Hz
    """

    master.mav.command_long_send(
        master.target_system,
        0,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        message_id,
        interval_us,
        0,
        0,
        0,
        0,
        0
    )


print("Requesting GLOBAL_POSITION_INT at 5 Hz...")
request_message_interval(
    mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT,
    200000
)

print("Requesting ATTITUDE at 10 Hz...")
request_message_interval(
    mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,
    100000
)


# Give ArduPilot a moment to process requests
time.sleep(1)

print("\nReading telemetry...\n")


# ============================================================
# TELEMETRY LOOP
# ============================================================

while True:

    msg = master.recv_match(
        type=[
            "GLOBAL_POSITION_INT",
            "ATTITUDE"
        ],
        blocking=True,
        timeout=2
    )

    if msg is None:
        print("No telemetry received.")
        continue

    msg_type = msg.get_type()


    # --------------------------------------------------------
    # GLOBAL POSITION
    # --------------------------------------------------------

    if msg_type == "GLOBAL_POSITION_INT":

        latitude = msg.lat / 1e7
        longitude = msg.lon / 1e7

        # MSL altitude
        altitude = msg.alt / 1000.0

        # Relative altitude above home
        relative_altitude = msg.relative_alt / 1000.0

        print(
            f"GPS      "
            f"lat={latitude:.7f}, "
            f"lon={longitude:.7f}, "
            f"alt={altitude:.2f} m, "
            f"rel_alt={relative_altitude:.2f} m"
        )


    # --------------------------------------------------------
    # ATTITUDE
    # --------------------------------------------------------

    elif msg_type == "ATTITUDE":

        roll = math.degrees(msg.roll)
        pitch = math.degrees(msg.pitch)
        yaw = math.degrees(msg.yaw)

        print(
            f"ATTITUDE "
            f"roll={roll:+7.2f}°, "
            f"pitch={pitch:+7.2f}°, "
            f"yaw={yaw:+7.2f}°"
        )
