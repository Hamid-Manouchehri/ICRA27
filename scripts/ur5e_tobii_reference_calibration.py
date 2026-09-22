"""
Collect one spatial reference between Tobii UCS and CoppeliaSim/UR5e coordinates.

Inputs:
    A known UR5e joint configuration [deg], its expected base-frame TCP
    position [m], a CoppeliaSim marker position [m], and three seconds of gaze.

Outputs:
    In-memory robot_calibration_samples and tobii_calibration_samples lists.
    The calculated manual_tobii_reference_xy_mm and
    manual_coppelia_reference_xy_m lists are printed in MATLAB-ready form and
    written to calibration_json_path, overwriting any previous calibration.

Processing:
    The robot moves to the known joint pose. CoppeliaSim displays the reference
    marker at the known Cartesian position. After a preparation delay, valid
    left/right Tobii gaze points in the User Coordinate System (UCS) are
    combined per sample, and their median over the instructed three-second
    fixation is used as the Tobii reference [mm].

Important:
    This script commands physical robot motion. Verify the joint pose, TCP/tool
    configuration, path, workspace, and emergency-stop access before running.

Dependencies:
    pip install --user ur_rtde tobii-research
    pip install --user coppeliasim-zmqremoteapi-client
"""

import json
import math
import statistics
import time
from pathlib import Path

try:
    import rtde_control
    import rtde_receive
except ImportError:
    rtde_control = None
    rtde_receive = None

try:
    import tobii_research as tobii_research_api
except ImportError:
    tobii_research_api = None


# -------------------- User Settings --------------------
robot_ip = "192.168.1.100"  # TODO UR5e IP/hostname [valid robot address]

# calibration output path [file path]; overwritten after each
# successful run with the newly measured reference points.
calibration_json_path = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "calibrations"
    # / "tobii_coppelia_reference_calibration_narrow_fine_gross_shape_counter_clockwise_s24.json"
    # / "tobii_coppelia_reference_calibration_wide_fine_gross_shape_counter_clockwise_s24.json"
    # / "tobii_coppelia_reference_calibration_narrow_ellipse_shape_counter_clockwise_s24.json"
    # / "tobii_coppelia_reference_calibration_wide_ellipse_shape_counter_clockwise_s24.json"
    # / "tobii_coppelia_reference_calibration_narrow_pear_shape_counter_clockwise_s24.json"
    # / "tobii_coppelia_reference_calibration_wide_pear_shape_counter_clockwise_s24.json"
    / "tobii_coppelia_reference_calibration_test.json"
)

start_joint_q_deg = [  # TODO known joint angles [deg], within joint limits
    -77.64,
    -98.5,
    -95.80,
    -75.75,
    90.3,
    -2.75,
]
start_joint_q = [math.radians(angle_deg) for angle_deg in start_joint_q_deg]

expected_robot_tcp_position_m = [  # TODO known UR5e TCP expressed in base frame [x, y, z] [m]
    -0.014,
    -0.558,
    0.232,
]

coppeliasim_reference_position_m = [  # TODO calibration marker [x, y, z] [m]
    -0.014,
    -0.650,
    0.232,
]

joint_speed_rad_s = 0.25  # TODO moveJ speed [rad/s], recommended 0.1 to 0.5
joint_acceleration_rad_s2 = 0.5  # TODO moveJ acceleration [rad/s^2], 0.1 to 1
max_tcp_reference_error_m = 0.020  # TODO allowed position error [m], > 0

subject_preparation_time_s = 2.0  # TODO look-at-target preparation [s], >= 0
calibration_recording_time_s = 3.0  # TODO fixation recording [s], > 0
robot_recording_frequency_hz = 125.0  # TODO robot sample rate [Hz], 25 to 500

coppeliasim_host = "127.0.0.1"  # TODO ZeroMQ host [hostname or IPv4 address]
coppeliasim_port = 23000  # TODO ZeroMQ port [integer], default is 23000
coppeliasim_object_path = "/calibration_marker"  # TODO exact yellow marker scene path
coppeliasim_reference_object_path = ""  # TODO empty string means world frame
set_marker_color_yellow = False  # TODO bool; ignored if object is not a shape


# -------------------- Calibration Results --------------------
# These lists are populated by main() and remain available when this module is
# run interactively or imported and main() is called.
robot_calibration_samples = []
tobii_calibration_samples = []
manual_tobii_reference_xy_mm = [math.nan, math.nan]
manual_coppelia_reference_xy_m = coppeliasim_reference_position_m[:2]


def is_valid_ucs_point(gaze_sample, eye_name):
    """Return true for a valid, finite three-component UCS gaze point."""

    validity_key = f"{eye_name}_gaze_point_validity"
    point_key = f"{eye_name}_gaze_point_in_user_coordinate_system"
    point = gaze_sample.get(point_key)

    return (
        gaze_sample.get(validity_key) == 1 and
        isinstance(point, (list, tuple)) and
        len(point) == 3 and
        all(math.isfinite(float(value)) for value in point)
    )


def combine_valid_gaze_points(gaze_samples):
    """Average valid left/right UCS gaze points for each Tobii sample [mm]."""

    combined_points_ucs_mm = []
    for gaze_sample in gaze_samples:
        valid_eye_points = []
        for eye_name in ("left", "right"):
            if is_valid_ucs_point(gaze_sample, eye_name):
                point_key = (
                    f"{eye_name}_gaze_point_in_user_coordinate_system"
                )
                valid_eye_points.append(gaze_sample[point_key])

        if not valid_eye_points:
            continue

        eye_count = len(valid_eye_points)
        combined_point = [
            sum(float(point[axis]) for point in valid_eye_points) / eye_count
            for axis in range(3)
        ]
        combined_points_ucs_mm.append(combined_point)

    return combined_points_ucs_mm


def median_xyz(samples_xyz):
    """Return the component-wise median of N three-component samples."""

    return [
        statistics.median(sample[axis] for sample in samples_xyz)
        for axis in range(3)
    ]


def write_calibration_json(tobii_reference_xy_mm, coppelia_reference_xy_m):
    """Overwrite calibration_json_path with the latest reference points.

    Input:
        tobii_reference_xy_mm: [x, y] Tobii UCS reference point [mm].
        coppelia_reference_xy_m: [x, y] CoppeliaSim reference point [m].

    Output:
        calibration_json_path is created (with parent directories) or
        overwritten with
        {"manual_tobii_reference_xy_mm", "manual_coppelia_reference_xy_m"}.
    """

    calibration_data = {
        "manual_tobii_reference_xy_mm": [
            round(tobii_reference_xy_mm[0], 6),
            round(tobii_reference_xy_mm[1], 6),
        ],
        "manual_coppelia_reference_xy_m": list(coppelia_reference_xy_m),
    }

    calibration_json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(calibration_json_path, "w", encoding="utf-8") as json_file:
        json.dump(calibration_data, json_file, indent=2)
        json_file.write("\n")

    print("Calibration JSON written to:", calibration_json_path)


def connect_coppeliasim_marker():
    """Connect to CoppeliaSim and place the marker at the known position."""

    from coppeliasim_zmqremoteapi_client import RemoteAPIClient

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
    marker_handle = sim.getObject(coppeliasim_object_path)

    reference_handle = -1
    if coppeliasim_reference_object_path:
        reference_handle = sim.getObject(coppeliasim_reference_object_path)

    sim.setObjectPosition(
        marker_handle,
        reference_handle,
        coppeliasim_reference_position_m,
    )

    if set_marker_color_yellow:
        try:
            sim.setShapeColor(
                marker_handle,
                "",
                sim.colorcomponent_ambient_diffuse,
                [1.0, 1.0, 0.0],
            )
        except Exception as error:
            print("Could not set marker color; continuing:", error)

    print(
        "CoppeliaSim reference marker [m]:",
        coppeliasim_reference_position_m,
    )
    return client


def record_robot_samples(rtde_receive_interface):
    """Record stationary TCP/joint data using Tobii's system clock domain."""

    robot_samples = []
    recording_start_s = time.monotonic()
    next_sample_s = recording_start_s
    sample_period_s = 1.0 / robot_recording_frequency_hz

    while True:
        current_time_s = time.monotonic()
        recording_time_s = current_time_s - recording_start_s
        if recording_time_s >= calibration_recording_time_s:
            break

        system_time_stamp_us = tobii_research_api.get_system_time_stamp()
        tcp_pose = list(rtde_receive_interface.getActualTCPPose())
        joint_position = list(rtde_receive_interface.getActualQ())
        robot_samples.append(
            [system_time_stamp_us, recording_time_s] +
            tcp_pose +
            joint_position
        )

        next_sample_s += sample_period_s
        remaining_time_s = next_sample_s - time.monotonic()
        if remaining_time_s > 0.0:
            time.sleep(remaining_time_s)

    return robot_samples


def main():
    """Move to the reference pose and collect one gaze/robot calibration."""

    if rtde_control is None or rtde_receive is None:
        print("Missing dependency: install with `pip install --user ur_rtde`")
        return
    if tobii_research_api is None:
        print("Missing dependency: install with `pip install tobii-research`")
        return

    trackers = tobii_research_api.find_all_eyetrackers()
    if not trackers:
        print("No Tobii eye tracker found.")
        return
    tracker = trackers[0]

    try:
        coppeliasim_client = connect_coppeliasim_marker()
    except Exception as error:
        print("Calibration stopped: CoppeliaSim marker is unavailable:", error)
        return

    print("Tobii model:", tracker.model)
    print("Tobii serial:", tracker.serial_number)
    print("Connecting to UR5e at", robot_ip)

    rtde_control_interface = rtde_control.RTDEControlInterface(robot_ip)
    rtde_receive_interface = rtde_receive.RTDEReceiveInterface(robot_ip)

    try:
        print("Moving to calibration joint configuration [deg]:")
        print(start_joint_q_deg)
        rtde_control_interface.moveJ(
            start_joint_q,
            joint_speed_rad_s,
            joint_acceleration_rad_s2,
        )

        measured_tcp_pose = list(rtde_receive_interface.getActualTCPPose())
        position_error_m = math.sqrt(sum(
            (
                measured_tcp_pose[axis] -
                expected_robot_tcp_position_m[axis]
            ) ** 2
            for axis in range(3)
        ))
        print("Measured TCP position [m]:", measured_tcp_pose[:3])
        print("Expected robot TCP position [m]:", expected_robot_tcp_position_m)
        print("Position difference [m]:", position_error_m)

        if position_error_m > max_tcp_reference_error_m:
            raise RuntimeError(
                "Measured TCP position is outside the calibration tolerance."
            )

        print()
        print("LOOK AT THE YELLOW MARKER AND KEEP FIXATING ON ITS CENTER.")
        print(
            "Calibration recording starts in",
            subject_preparation_time_s,
            "seconds.",
        )
        time.sleep(subject_preparation_time_s)

        robot_calibration_samples.clear()
        tobii_calibration_samples.clear()
        manual_tobii_reference_xy_mm[:] = [math.nan, math.nan]
        manual_coppelia_reference_xy_m[:] = (
            coppeliasim_reference_position_m[:2]
        )

        def gaze_callback(data):
            tobii_calibration_samples.append(dict(data))

        tracker.subscribe_to(
            tobii_research_api.EYETRACKER_GAZE_DATA,
            gaze_callback,
            as_dictionary=True,
        )
        try:
            print(
                "Recording robot and Tobii data for",
                calibration_recording_time_s,
                "seconds...",
            )
            robot_calibration_samples.extend(
                record_robot_samples(rtde_receive_interface)
            )
        finally:
            tracker.unsubscribe_from(
                tobii_research_api.EYETRACKER_GAZE_DATA,
                gaze_callback,
            )

        if not tobii_calibration_samples:
            raise RuntimeError("No Tobii gaze samples were received.")

        combined_gaze_ucs_mm = combine_valid_gaze_points(
            tobii_calibration_samples
        )
        if not combined_gaze_ucs_mm:
            raise RuntimeError(
                "No finite valid left/right Tobii UCS gaze points were found."
            )

        median_tobii_ucs_mm = median_xyz(combined_gaze_ucs_mm)
        median_robot_tcp_m = median_xyz([
            robot_sample[2:5]
            for robot_sample in robot_calibration_samples
        ])
        manual_tobii_reference_xy_mm[:] = median_tobii_ucs_mm[:2]

        print()
        print("Calibration completed.")
        print(
            "Median combined Tobii fixation point in UCS [mm]:",
            median_tobii_ucs_mm,
        )
        print(
            "manual_tobii_reference_xy_mm = "
            f"[{manual_tobii_reference_xy_mm[0]:.6f}, "
            f"{manual_tobii_reference_xy_mm[1]:.6f}];"
        )
        print(
            "manual_coppelia_reference_xy_m = "
            f"[{manual_coppelia_reference_xy_m[0]:.6f}, "
            f"{manual_coppelia_reference_xy_m[1]:.6f}];"
        )
        print("Median measured robot TCP [m]:", median_robot_tcp_m)
        print(
            "Valid combined gaze samples:",
            len(combined_gaze_ucs_mm),
            "of",
            len(tobii_calibration_samples),
        )

        write_calibration_json(
            manual_tobii_reference_xy_mm,
            manual_coppelia_reference_xy_m,
        )

        return (
            manual_tobii_reference_xy_mm,
            manual_coppelia_reference_xy_m,
        )
    finally:
        rtde_control_interface.stopScript()


if __name__ == "__main__":
    main()
