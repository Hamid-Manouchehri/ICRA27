"""
Move a UR5e from a specified joint pose, then translate its TCP by +1 cm in
base-frame x, wait for the configured pause, and translate by +1 cm in
base-frame y.

Inputs:
    UR5e IP address and six initial joint angles [deg].

Outputs:
    One moveJ followed by two blocking Cartesian moveL commands. The final TCP
    target is initial_tcp_position + [0.01, 0.01, 0.0] m, with unchanged tool
    orientation. A separate thread sends the measured TCP position to a
    CoppeliaSim marker through the ZeroMQ Remote API.

User-adjustable parameters:
    Edit robot_ip and the joint/linear speed and acceleration settings below.
    Verify the initial pose, Cartesian path, tool, workspace, and emergency-stop
    access before running this hardware-control script.

Dependency:
    pip install --user ur_rtde coppeliasim-zmqremoteapi-client
"""

import math
import threading
import time

try:
    import rtde_control
    import rtde_receive
except ImportError:
    rtde_control = None
    rtde_receive = None


# -------------------- User Settings --------------------
robot_ip = "192.168.1.100"  # TODO UR5e IP/hostname [valid robot address]

start_joint_q_deg = [  # TODO joint angles [deg], within robot joint limits
    -77.64,
    -98.5,
    -95.80,
    -75.75,
    90.3,
    -2.75,
]
start_joint_q = [math.radians(angle_deg) for angle_deg in start_joint_q_deg]

joint_speed_rad_s = 0.25  # TODO moveJ speed [rad/s], recommended 0.1 to 0.5
joint_acceleration_rad_s2 = 0.5  # TODO moveJ acceleration [rad/s^2], 0.1 to 1

linear_speed_m_s = 0.025  # TODO moveL speed [m/s], recommended 0.01 to 0.05
linear_acceleration_m_s2 = 0.1  # TODO moveL acceleration [m/s^2], 0.05 to 0.5

translation_x_m = 0.01  # TODO base-frame +x translation [m], requested 0.01
translation_y_m = 0.01  # TODO base-frame +y translation [m], requested 0.01
pause_between_moves_s = 2.0  # TODO pause after +x move [s], valid >= 0

# CoppeliaSim streaming is separate from the thread issuing robot moves.
send_coppeliasim_trajectory = True  # TODO bool; requires CoppeliaSim running
coppeliasim_host = "127.0.0.1"  # TODO ZeroMQ host [hostname or IPv4 address]
coppeliasim_port = 23000  # TODO ZeroMQ port [integer], default is 23000
coppeliasim_object_path = "/Sphere"  # TODO exact marker path in scene tree
coppeliasim_reference_object_path = ""  # TODO empty string means world frame
coppeliasim_update_frequency_hz = 25.0  # TODO update rate [Hz], 1 to 100
coppeliasim_ready_timeout_s = 4.0  # TODO startup wait [s], recommended 1 to 10

coppeliasim_position_offset_m = [  # TODO world-frame [x, y, z] offset [m]
    0.0,
    0.0,
    0.0,
]
coppeliasim_position_sign = [  # TODO UR5e-to-simulator axis signs, -1 or +1
    1.0,
    1.0,
    1.0,
]


def map_tcp_position_to_coppeliasim(tcp_pose):
    """Map a measured UR5e base-frame TCP pose to simulator position [m]."""

    return [
        coppeliasim_position_offset_m[0] +
        coppeliasim_position_sign[0] * tcp_pose[0],
        coppeliasim_position_offset_m[1] +
        coppeliasim_position_sign[1] * tcp_pose[1],
        coppeliasim_position_offset_m[2] +
        coppeliasim_position_sign[2] * tcp_pose[2],
    ]


def stream_tcp_position_to_coppeliasim(
    rtde_receive_interface,
    stop_event,
    ready_event,
):
    """Update the CoppeliaSim marker from a separate background thread."""

    try:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    except ImportError as error:
        print("CoppeliaSim streaming unavailable:", error)
        ready_event.set()
        return

    try:
        print(
            "Connecting to CoppeliaSim at",
            coppeliasim_host,
            "port",
            coppeliasim_port,
        )
        client = RemoteAPIClient(
            host=coppeliasim_host,
            port=coppeliasim_port,
        )
        sim = client.require("sim")
        target_handle = sim.getObject(coppeliasim_object_path)

        reference_handle = -1
        if coppeliasim_reference_object_path:
            reference_handle = sim.getObject(
                coppeliasim_reference_object_path
            )

        update_period_s = 1.0 / coppeliasim_update_frequency_hz

        initial_tcp_pose = rtde_receive_interface.getActualTCPPose()
        initial_marker_position = map_tcp_position_to_coppeliasim(
            initial_tcp_pose
        )
        sim.setObjectPosition(
            target_handle,
            reference_handle,
            initial_marker_position,
        )
        print(
            "CoppeliaSim marker streaming started at",
            coppeliasim_update_frequency_hz,
            "Hz",
        )
        ready_event.set()

        while not stop_event.is_set():
            update_start_s = time.monotonic()
            tcp_pose = rtde_receive_interface.getActualTCPPose()
            marker_position = map_tcp_position_to_coppeliasim(tcp_pose)
            sim.setObjectPosition(
                target_handle,
                reference_handle,
                marker_position,
            )

            update_elapsed_s = time.monotonic() - update_start_s
            remaining_period_s = update_period_s - update_elapsed_s
            if remaining_period_s > 0.0:
                stop_event.wait(remaining_period_s)
    except Exception as error:
        print("CoppeliaSim marker streaming stopped:", error)
        ready_event.set()


def main():
    """Execute the configured joint move and two Cartesian translations."""

    if rtde_control is None or rtde_receive is None:
        print("Missing dependency: install with `pip install --user ur_rtde`")
        return

    print("Connecting to UR5e at", robot_ip)
    rtde_control_interface = rtde_control.RTDEControlInterface(robot_ip)
    rtde_receive_interface = rtde_receive.RTDEReceiveInterface(robot_ip)
    coppeliasim_stop_event = None
    coppeliasim_thread = None

    try:

        print("Moving to initial joint configuration [deg]:")
        print(start_joint_q_deg)
        rtde_control_interface.moveJ(
            start_joint_q,
            joint_speed_rad_s,
            joint_acceleration_rad_s2,
        )

        time.sleep(pause_between_moves_s)

        initial_tcp_pose = list(rtde_receive_interface.getActualTCPPose())

        if send_coppeliasim_trajectory:
            coppeliasim_stop_event = threading.Event()
            coppeliasim_ready_event = threading.Event()
            coppeliasim_thread = threading.Thread(
                target=stream_tcp_position_to_coppeliasim,
                args=(
                    rtde_receive_interface,
                    coppeliasim_stop_event,
                    coppeliasim_ready_event,
                ),
                name="coppeliasim_tcp_stream",
                daemon=True,
            )
            coppeliasim_thread.start()
            if not coppeliasim_ready_event.wait(coppeliasim_ready_timeout_s):
                print(
                    "CoppeliaSim connection is still pending; "
                    "continuing robot motion."
                )

        target_x_pose = initial_tcp_pose.copy()
        target_x_pose[0] += translation_x_m
        print("Moving TCP +1 cm in base-frame x...")
        rtde_control_interface.moveL(
            target_x_pose,
            linear_speed_m_s,
            linear_acceleration_m_s2,
        )

        time.sleep(pause_between_moves_s)

        target_xy_pose = target_x_pose.copy()
        target_xy_pose[1] += translation_y_m
        print("Moving TCP +1 cm in base-frame y...")
        rtde_control_interface.moveL(
            target_xy_pose,
            linear_speed_m_s,
            linear_acceleration_m_s2,
        )

        time.sleep(pause_between_moves_s)

        target_xy_pose = target_xy_pose.copy()
        target_xy_pose[1] -= translation_y_m
        print("Moving TCP -1 cm in base-frame y...")
        rtde_control_interface.moveL(
            target_xy_pose,
            linear_speed_m_s,
            linear_acceleration_m_s2,
        )

        time.sleep(pause_between_moves_s)

        target_x_pose = target_xy_pose.copy()
        target_x_pose[0] -= translation_x_m
        print("Moving TCP -1 cm in base-frame x...")
        rtde_control_interface.moveL(
            target_x_pose,
            linear_speed_m_s,
            linear_acceleration_m_s2,
        )

    finally:
        if coppeliasim_stop_event is not None:
            coppeliasim_stop_event.set()
        if coppeliasim_thread is not None:
            coppeliasim_thread.join(timeout=2.0)
        rtde_control_interface.stopScript()

    print("UR5e Cartesian moves completed.")


if __name__ == "__main__":
    main()
