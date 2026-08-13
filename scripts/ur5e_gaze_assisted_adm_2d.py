"""Run gaze-assisted planar admittance control on a UR5e.

Inputs:
    UR5e base-frame TCP force and pose from RTDE.
    Real-time Tobii gaze points and origins in the Tobii UCS [mm].
    A Tobii-to-CoppeliaSim calibration JSON.

Outputs:
    Optional UR5e speedL command [vx, vy, 0, 0, 0, 0].
    Optional `/Sphere` TCP marker trajectory in CoppeliaSim through a
    separate ZeroMQ process.
    CSV data containing force, TCP motion, fixation, intent, confidence,
    damping, and rectangular CBF results.

Controller logic:
    Gaze suggests a signed planar intent direction.
    Human-applied force drives the admittance dynamics.
    Fixation confidence blends fixed damping with directional damping.
    A velocity-level control barrier function (CBF) keeps the TCP inside an
    axis-aligned rectangular workspace.

Main tunable parameters:
    Workspace corners, CBF alpha, fixed/intent damping, virtual mass,
    fixation velocity/duration thresholds, and confidence thresholds are in
    User Settings. Check all frames, signs, limits, and the robot path before
    setting send_robot_commands = True.

Dependencies:
    ur_rtde and tobii-research.

Assumptions:
    The calibrated CoppeliaSim x-y plane and UR5e base x-y plane are related
    by the explicit axis-order, sign, and offset settings below.
"""

import csv
import json
import math
import multiprocessing
import queue
import signal
import threading
import time
import traceback
from collections import deque
from pathlib import Path

import rtde_control
import rtde_receive
import tobii_research


# -------------------- User Settings --------------------
# TODO Keep False until force signs, gaze mapping, workspace, and CBF behavior
# have been checked. True sends physical speedL commands.
send_robot_commands = True

robot_ip = "192.168.1.100"  # TODO UR5e IPv4 address or hostname

# TODO Workspace offsets [x, y] from the startup TCP [m]. Recommended motion
# should remain comfortably inside the robot's safe reachable workspace.
workspace_lower_corner_offset_m = [-0.20, -0.20]
workspace_upper_corner_offset_m = [0.25, 0.25]

# TODO CBF approach rate [1/s], recommended 1 to 20. Larger values intervene
# closer to the boundary; smaller values slow outward motion earlier.
cbf_alpha_per_s = 5.0

max_cart_speed_m_s = 0.10  # TODO planar speed limit [m/s], 0.01 to 0.15
speedl_acceleration_m_s2 = 0.5  # TODO speedL acceleration [m/s^2], 0.1 to 2
speed_stop_deceleration_m_s2 = 2.0  # TODO stop deceleration [m/s^2], 0.5 to 5

# Fixed fallback admittance: M*v_dot + B*v = F.
virtual_mass_x_kg = 10.0  # TODO x virtual mass [kg], recommended 1 to 20
virtual_mass_y_kg = 10.0  # TODO y virtual mass [kg], recommended 1 to 20
fixed_damping_x_n_s_m = 200.0  # TODO fallback x damping [N s/m], 20 to 120
fixed_damping_y_n_s_m = 200.0  # TODO fallback y damping [N s/m], 20 to 120

# At high gaze confidence, forward motion along intent receives lower damping.
# Perpendicular and opposite motion receive the non-intent damping.
intent_forward_damping_n_s_m = 15.0  # TODO [N s/m], positive
intent_other_damping_n_s_m = 200.0  # TODO [N s/m], positive

force_lpf_alpha = 0.2  # TODO force filter alpha [0, 1]
force_deadband_n = 1.0  # TODO ignore force magnitude below [N], 0 to 5
force_sign_x = 1.0  # TODO base x force sign, -1 or +1
force_sign_y = 1.0  # TODO base y force sign, -1 or +1
force_bias_samples = 250  # TODO unloaded startup samples, recommended >= 100

# TODO I-VT angular-velocity threshold [deg/s], recommended 20 to 70.
fixation_velocity_threshold_deg_s = 30.0
# TODO minimum real-time fixation duration [s], recommended 0.06 to 0.20.
minimum_fixation_duration_s = 0.10
# TODO duration that produces confidence 1 [s], greater than the minimum.
full_confidence_fixation_duration_s = 0.40
# TODO confidence below which gaze has no damping influence [0, 1).
minimum_intent_confidence = 0.20
# TODO discard a fixation not updated within this time [s], 0.2 to 1.0.
maximum_fixation_age_s = 0.50
# TODO recent TCP samples averaged before forming intent, recommended 5 to 50.
intent_tcp_average_sample_count = 25
# TODO avoid normalizing a nearly zero TCP-to-fixation vector [m].
minimum_intent_distance_m = 0.005

# TODO calibration file for this subject/setup.
calibration_json_path = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "calibrations"
    / "tobii_coppelia_reference_protractor_shape.json"
)
# TODO physical screen motion / robot motion [unitless], must be positive.
screen_motion_per_robot_motion = 1.0
# TODO Tobii UCS x/y signs in CoppeliaSim, each -1 or +1.
tobii_to_coppelia_axis_sign = [1.0, 1.0]

# CoppeliaSim axis c is built from robot axis
# coppeliasim_axis_from_robot[c]. Current setup is x<-x and y<-y.
# For legacy x_c=-y_r, y_c=x_r use [1, 0] and [-1, 1].
coppeliasim_axis_from_robot = [0, 1]  # TODO permutation of [0, 1]
coppeliasim_sign_from_robot = [1.0, 1.0]  # TODO each -1 or +1
coppeliasim_offset_xy_m = [0.0, 0.0]  # TODO CoppeliaSim offset [m]
coppeliasim_z_sign_from_robot = 1.0  # TODO z sign, -1 or +1
coppeliasim_z_offset_m = 0.0  # TODO CoppeliaSim z offset [m]

# CoppeliaSim marker bridge. All ZeroMQ calls run in a separate process.
send_coppeliasim_trajectory = True  # TODO requires CoppeliaSim running
coppeliasim_host = "127.0.0.1"  # TODO ZeroMQ host or IPv4 address
coppeliasim_port = 23000  # TODO ZeroMQ port, default 23000
coppeliasim_object_path = "/Sphere"  # TODO exact scene-tree marker path
# TODO empty means CoppeliaSim world frame.
coppeliasim_reference_object_path = ""
coppeliasim_update_frequency_hz = 25.0  # TODO marker update rate [Hz]

# TODO require each contributing eye point to lie in display area [0, 1]^2.
require_gaze_on_display = True

move_to_start_configuration = True  # TODO False skips the startup move
start_joint_q_deg = [  # TODO joint angles [deg], within robot limits
    -77.64,
    -98.5,
    -95.80,
    -75.75,
    90.3,
    -2.75,
]
start_joint_speed_rad_s = 0.5  # TODO moveJ speed [rad/s], 0.1 to 1
start_joint_acceleration_rad_s2 = 0.5  # TODO [rad/s^2], 0.1 to 2

control_frequency_hz = 500.0  # TODO UR5e e-series control rate [Hz]
subject_preparation_time_s = 5.0  # TODO preparation delay [s], >= 0
maximum_runtime_s = 120.0  # TODO automatic stop time [s], positive

# TODO controller output CSV path. Raw source files are not modified.
write_data_csv_path = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw_data"
    / "ur5e_gaze_assisted_adm_2d_protractor_test.csv"
)


# -------------------- Shared State --------------------
keep_running = True
force_bias_n = [0.0, 0.0]
force_raw_n = [0.0, 0.0]
force_filtered_n = [0.0, 0.0]
cart_velocity_command_m_s = [0.0, 0.0]

fixation_lock = threading.Lock()
latest_fixation = {
    "valid": False,
    "position_robot_xy_m": [math.nan, math.nan],
    "duration_s": 0.0,
    "update_time_s": math.nan,
    "angular_velocity_deg_s": math.nan,
}


def handle_stop_signal(_signal_number, _frame):
    """Stop the controller when Ctrl+C is pressed."""

    global keep_running
    keep_running = False


def clamp_value(value, lower_limit, upper_limit):
    """Limit one scalar to a closed interval."""

    if value < lower_limit:
        return lower_limit

    if value > upper_limit:
        return upper_limit

    return value


def vector_norm_2d(vector_xy):
    """Return the Euclidean norm of a 2-D vector."""

    x_squared = vector_xy[0] * vector_xy[0]
    y_squared = vector_xy[1] * vector_xy[1]
    norm_squared = x_squared + y_squared
    return math.sqrt(norm_squared)


def vector_norm_3d(vector_xyz):
    """Return the Euclidean norm of a 3-D vector."""

    x_squared = vector_xyz[0] * vector_xyz[0]
    y_squared = vector_xyz[1] * vector_xyz[1]
    z_squared = vector_xyz[2] * vector_xyz[2]
    norm_squared = x_squared + y_squared + z_squared
    return math.sqrt(norm_squared)


def normalize_vector_3d(vector_xyz):
    """Return a unit 3-D vector."""

    vector_norm = vector_norm_3d(vector_xyz)
    return [
        vector_xyz[0] / vector_norm,
        vector_xyz[1] / vector_norm,
        vector_xyz[2] / vector_norm,
    ]


def limit_vector_norm_2d(vector_xy, maximum_norm):
    """Limit a 2-D magnitude while preserving direction."""

    vector_norm = vector_norm_2d(vector_xy)

    if vector_norm <= maximum_norm:
        return list(vector_xy)

    scale = maximum_norm / vector_norm
    limited_x = scale * vector_xy[0]
    limited_y = scale * vector_xy[1]
    return [limited_x, limited_y]


def is_finite_vector(vector, expected_length):
    """Return True for a finite vector with the expected length."""

    if vector is None or len(vector) != expected_length:
        return False

    return all(math.isfinite(float(value)) for value in vector)


def load_spatial_calibration():
    """Load Tobii UCS [mm] and CoppeliaSim x-y [m] reference points."""

    with open(calibration_json_path, "r", encoding="utf-8") as json_file:
        calibration = json.load(json_file)

    tobii_reference_xy_mm = calibration[
        "manual_tobii_reference_xy_mm"
    ]
    coppelia_reference_xy_m = calibration[
        "manual_coppelia_reference_xy_m"
    ]

    return tobii_reference_xy_mm, coppelia_reference_xy_m


def map_coppelia_xy_to_robot_base(coppelia_xy_m):
    """Invert the configured robot-to-CoppeliaSim planar transform."""

    robot_xy_m = [0.0, 0.0]

    for coppelia_axis in range(2):
        robot_axis = coppeliasim_axis_from_robot[coppelia_axis]
        shifted_position_m = (
            coppelia_xy_m[coppelia_axis]
            - coppeliasim_offset_xy_m[coppelia_axis]
        )
        signed_position_m = (
            shifted_position_m
            / coppeliasim_sign_from_robot[coppelia_axis]
        )
        robot_xy_m[robot_axis] = signed_position_m

    return robot_xy_m


def map_tcp_position_to_coppeliasim(tcp_position_xyz_m):
    """Map UR5e base-frame TCP position [m] to CoppeliaSim [m]."""

    robot_position_xy_m = tcp_position_xyz_m[:2]
    coppelia_position_xy_m = [0.0, 0.0]

    for coppelia_axis in range(2):
        robot_axis = coppeliasim_axis_from_robot[coppelia_axis]
        robot_position_m = robot_position_xy_m[robot_axis]
        signed_position_m = (
            coppeliasim_sign_from_robot[coppelia_axis]
            * robot_position_m
        )
        coppelia_position_xy_m[coppelia_axis] = (
            coppeliasim_offset_xy_m[coppelia_axis]
            + signed_position_m
        )

    coppelia_z_m = (
        coppeliasim_z_offset_m
        + coppeliasim_z_sign_from_robot * tcp_position_xyz_m[2]
    )
    return [
        coppelia_position_xy_m[0],
        coppelia_position_xy_m[1],
        coppelia_z_m,
    ]


def coppeliasim_trajectory_process(trajectory_queue, stop_event):
    """Update the CoppeliaSim marker from an independent process."""

    try:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    except ImportError as error:
        print("CoppeliaSim bridge import failed:", error, flush=True)
        return

    try:
        print(
            "CoppeliaSim bridge connecting to",
            coppeliasim_host,
            "port",
            coppeliasim_port,
            flush=True,
        )
        client = RemoteAPIClient(
            host=coppeliasim_host,
            port=coppeliasim_port,
        )
        sim = client.require("sim")
        target_object_handle = sim.getObject(coppeliasim_object_path)

        reference_object_handle = -1

        if coppeliasim_reference_object_path:
            reference_object_handle = sim.getObject(
                coppeliasim_reference_object_path
            )

        print(
            "CoppeliaSim bridge found",
            coppeliasim_object_path,
            "with handle",
            target_object_handle,
            flush=True,
        )

        update_period_s = 1.0 / coppeliasim_update_frequency_hz
        next_update_time_s = 0.0
        first_sample = True

        while not stop_event.is_set():
            try:
                trajectory_sample = trajectory_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            while True:
                try:
                    trajectory_sample = trajectory_queue.get_nowait()
                except queue.Empty:
                    break

            current_time_s = time.monotonic()

            if current_time_s < next_update_time_s:
                continue

            tcp_position_xyz_m = trajectory_sample[1]
            coppelia_position_xyz_m = map_tcp_position_to_coppeliasim(
                tcp_position_xyz_m
            )
            sim.setObjectPosition(
                target_object_handle,
                reference_object_handle,
                coppelia_position_xyz_m,
            )

            if first_sample:
                print(
                    "CoppeliaSim first marker position [m]:",
                    coppelia_position_xyz_m,
                    flush=True,
                )
                first_sample = False

            next_update_time_s = current_time_s + update_period_s
    except Exception as error:
        print("CoppeliaSim bridge stopped:", repr(error), flush=True)
        traceback.print_exc()


def start_coppeliasim_trajectory_process():
    """Start the non-blocking CoppeliaSim marker process."""

    if not send_coppeliasim_trajectory:
        return None, None, None

    process_context = multiprocessing.get_context("spawn")
    trajectory_queue = process_context.Queue(maxsize=1)
    stop_event = process_context.Event()
    trajectory_process = process_context.Process(
        target=coppeliasim_trajectory_process,
        args=(trajectory_queue, stop_event),
        name="coppeliasim_trajectory_process",
    )
    trajectory_process.start()

    print(
        "Started CoppeliaSim bridge process with PID",
        trajectory_process.pid,
        flush=True,
    )
    return trajectory_queue, stop_event, trajectory_process


def publish_coppeliasim_trajectory_sample(
    trajectory_queue,
    controller_time_s,
    tcp_position_xyz_m,
):
    """Queue only the newest TCP sample without blocking robot control."""

    if trajectory_queue is None:
        return

    try:
        while True:
            trajectory_queue.get_nowait()
    except queue.Empty:
        pass

    try:
        trajectory_queue.put_nowait(
            (controller_time_s, list(tcp_position_xyz_m))
        )
    except queue.Full:
        pass


def stop_coppeliasim_trajectory_process(
    trajectory_queue,
    stop_event,
    trajectory_process,
):
    """Stop and release the CoppeliaSim marker process."""

    if trajectory_process is None:
        return

    stop_event.set()
    trajectory_process.join(timeout=2.0)

    if trajectory_process.is_alive():
        trajectory_process.terminate()
        trajectory_process.join()

    trajectory_queue.close()


def map_gaze_ucs_to_robot_base(
    gaze_point_ucs_mm,
    tobii_reference_xy_mm,
    coppelia_reference_xy_m,
):
    """Map one Tobii UCS gaze point [mm] to UR5e base x-y [m]."""

    gaze_delta_x_mm = gaze_point_ucs_mm[0] - tobii_reference_xy_mm[0]
    gaze_delta_y_mm = gaze_point_ucs_mm[1] - tobii_reference_xy_mm[1]

    coppelia_m_per_tobii_mm = 1e-3 / screen_motion_per_robot_motion

    gaze_delta_coppelia_x_m = (
        gaze_delta_x_mm
        * tobii_to_coppelia_axis_sign[0]
        * coppelia_m_per_tobii_mm
    )
    gaze_delta_coppelia_y_m = (
        gaze_delta_y_mm
        * tobii_to_coppelia_axis_sign[1]
        * coppelia_m_per_tobii_mm
    )

    gaze_coppelia_x_m = (
        coppelia_reference_xy_m[0]
        + gaze_delta_coppelia_x_m
    )
    gaze_coppelia_y_m = (
        coppelia_reference_xy_m[1]
        + gaze_delta_coppelia_y_m
    )
    gaze_coppelia_xy_m = [
        gaze_coppelia_x_m,
        gaze_coppelia_y_m,
    ]

    return map_coppelia_xy_to_robot_base(gaze_coppelia_xy_m)


def read_valid_eye_measurement(gaze_sample, eye_name):
    """Read one valid eye's UCS point and unit gaze direction."""

    point_validity_key = f"{eye_name}_gaze_point_validity"
    origin_validity_key = f"{eye_name}_gaze_origin_validity"
    point_ucs_key = f"{eye_name}_gaze_point_in_user_coordinate_system"
    origin_ucs_key = f"{eye_name}_gaze_origin_in_user_coordinate_system"
    display_point_key = f"{eye_name}_gaze_point_on_display_area"

    point_valid = gaze_sample[point_validity_key] == 1
    origin_valid = gaze_sample[origin_validity_key] == 1

    if not point_valid or not origin_valid:
        return None

    point_ucs_mm = gaze_sample[point_ucs_key]
    origin_ucs_mm = gaze_sample[origin_ucs_key]

    if not is_finite_vector(point_ucs_mm, 3):
        return None

    if not is_finite_vector(origin_ucs_mm, 3):
        return None

    if require_gaze_on_display:
        display_point = gaze_sample[display_point_key]

        if not is_finite_vector(display_point, 2):
            return None

        gaze_on_display_x = 0.0 <= display_point[0] <= 1.0
        gaze_on_display_y = 0.0 <= display_point[1] <= 1.0

        if not gaze_on_display_x or not gaze_on_display_y:
            return None

    gaze_direction = [
        point_ucs_mm[0] - origin_ucs_mm[0],
        point_ucs_mm[1] - origin_ucs_mm[1],
        point_ucs_mm[2] - origin_ucs_mm[2],
    ]
    gaze_direction_norm = vector_norm_3d(gaze_direction)

    if gaze_direction_norm == 0.0:
        return None

    unit_gaze_direction = normalize_vector_3d(gaze_direction)
    return list(point_ucs_mm), unit_gaze_direction


def combine_valid_eye_measurements(gaze_sample):
    """Average all valid eye points and normalize their mean direction."""

    eye_measurements = []

    for eye_name in ("left", "right"):
        eye_measurement = read_valid_eye_measurement(
            gaze_sample,
            eye_name,
        )

        if eye_measurement is not None:
            eye_measurements.append(eye_measurement)

    if not eye_measurements:
        return None

    eye_count = len(eye_measurements)
    point_sum_ucs_mm = [0.0, 0.0, 0.0]
    direction_sum = [0.0, 0.0, 0.0]

    for point_ucs_mm, unit_direction in eye_measurements:
        for axis in range(3):
            point_sum_ucs_mm[axis] += point_ucs_mm[axis]
            direction_sum[axis] += unit_direction[axis]

    combined_point_ucs_mm = [
        point_sum_ucs_mm[0] / eye_count,
        point_sum_ucs_mm[1] / eye_count,
        point_sum_ucs_mm[2] / eye_count,
    ]
    combined_unit_direction = normalize_vector_3d(direction_sum)

    return combined_point_ucs_mm, combined_unit_direction


def create_realtime_fixation_callback(
    trial_clock,
    tobii_reference_xy_mm,
    coppelia_reference_xy_m,
):
    """Create a non-RTDE Tobii callback that updates the latest fixation."""

    detector = {
        "previous_time_s": None,
        "previous_direction": None,
        "previous_point_robot_xy_m": None,
        "candidate_start_time_s": None,
        "candidate_point_sum_robot_xy_m": [0.0, 0.0],
        "candidate_sample_count": 0,
    }

    def reset_candidate():
        detector["candidate_start_time_s"] = None
        detector["candidate_point_sum_robot_xy_m"] = [0.0, 0.0]
        detector["candidate_sample_count"] = 0

    def invalidate_fixation():
        with fixation_lock:
            latest_fixation["valid"] = False

    def gaze_callback(gaze_sample):
        host_time_ns = time.perf_counter_ns()
        trial_start_time_ns = trial_clock["start_time_ns"]

        if trial_start_time_ns is None:
            return

        gaze_time_s = (host_time_ns - trial_start_time_ns) * 1e-9
        combined_measurement = combine_valid_eye_measurements(gaze_sample)

        if combined_measurement is None:
            # Do not invalidate here: keep the last completed fixation and
            # let calculate_gaze_intent() age it out via
            # maximum_fixation_age_s instead of snapping to fixed damping
            # on every blink.
            reset_candidate()
            detector["previous_time_s"] = None
            detector["previous_direction"] = None
            detector["previous_point_robot_xy_m"] = None
            return

        gaze_point_ucs_mm = combined_measurement[0]
        gaze_direction = combined_measurement[1]
        gaze_point_robot_xy_m = map_gaze_ucs_to_robot_base(
            gaze_point_ucs_mm,
            tobii_reference_xy_mm,
            coppelia_reference_xy_m,
        )

        previous_time_s = detector["previous_time_s"]
        previous_direction = detector["previous_direction"]
        previous_point_robot_xy_m = detector[
            "previous_point_robot_xy_m"
        ]

        if previous_time_s is None:
            detector["previous_time_s"] = gaze_time_s
            detector["previous_direction"] = gaze_direction
            detector["previous_point_robot_xy_m"] = (
                gaze_point_robot_xy_m
            )
            return

        sample_interval_s = gaze_time_s - previous_time_s
        direction_dot_product = (
            previous_direction[0] * gaze_direction[0]
            + previous_direction[1] * gaze_direction[1]
            + previous_direction[2] * gaze_direction[2]
        )
        direction_dot_product = clamp_value(
            direction_dot_product,
            -1.0,
            1.0,
        )
        angular_step_rad = math.acos(direction_dot_product)
        angular_step_deg = math.degrees(angular_step_rad)
        angular_velocity_deg_s = angular_step_deg / sample_interval_s

        is_fixation_candidate = (
            angular_velocity_deg_s
            <= fixation_velocity_threshold_deg_s
        )

        if is_fixation_candidate:
            if detector["candidate_start_time_s"] is None:
                detector["candidate_start_time_s"] = previous_time_s
                detector["candidate_point_sum_robot_xy_m"] = [
                    previous_point_robot_xy_m[0]
                    + gaze_point_robot_xy_m[0],
                    previous_point_robot_xy_m[1]
                    + gaze_point_robot_xy_m[1],
                ]
                detector["candidate_sample_count"] = 2
            else:
                detector["candidate_point_sum_robot_xy_m"][0] += (
                    gaze_point_robot_xy_m[0]
                )
                detector["candidate_point_sum_robot_xy_m"][1] += (
                    gaze_point_robot_xy_m[1]
                )
                detector["candidate_sample_count"] += 1

            fixation_duration_s = (
                gaze_time_s - detector["candidate_start_time_s"]
            )

            if fixation_duration_s >= minimum_fixation_duration_s:
                sample_count = detector["candidate_sample_count"]
                point_sum = detector[
                    "candidate_point_sum_robot_xy_m"
                ]
                mean_fixation_x_m = point_sum[0] / sample_count
                mean_fixation_y_m = point_sum[1] / sample_count

                with fixation_lock:
                    latest_fixation["valid"] = True
                    latest_fixation["position_robot_xy_m"] = [
                        mean_fixation_x_m,
                        mean_fixation_y_m,
                    ]
                    latest_fixation["duration_s"] = fixation_duration_s
                    latest_fixation["update_time_s"] = gaze_time_s
                    latest_fixation["angular_velocity_deg_s"] = (
                        angular_velocity_deg_s
                    )
        else:
            # Do not invalidate here: keep the last completed fixation and
            # let calculate_gaze_intent() age it out via
            # maximum_fixation_age_s instead of snapping to fixed damping
            # on every saccade.
            reset_candidate()

        detector["previous_time_s"] = gaze_time_s
        detector["previous_direction"] = gaze_direction
        detector["previous_point_robot_xy_m"] = gaze_point_robot_xy_m

    return gaze_callback


def start_tobii_fixation_stream(
    trial_clock,
    tobii_reference_xy_mm,
    coppelia_reference_xy_m,
):
    """Subscribe the real-time fixation callback to the first Tobii tracker."""

    trackers = tobii_research.find_all_eyetrackers()

    if not trackers:
        raise RuntimeError("No Tobii eye tracker found.")

    tracker = trackers[0]
    gaze_callback = create_realtime_fixation_callback(
        trial_clock,
        tobii_reference_xy_mm,
        coppelia_reference_xy_m,
    )
    tracker.subscribe_to(
        tobii_research.EYETRACKER_GAZE_DATA,
        gaze_callback,
        as_dictionary=True,
    )

    print("Tobii model:", tracker.model)
    print("Tobii serial:", tracker.serial_number)
    print("Tobii frequency:", tracker.get_gaze_output_frequency(), "Hz")

    return tracker, gaze_callback


def stop_tobii_fixation_stream(tracker, gaze_callback):
    """Unsubscribe the Tobii callback."""

    tracker.unsubscribe_from(
        tobii_research.EYETRACKER_GAZE_DATA,
        gaze_callback,
    )


def read_latest_fixation():
    """Copy the latest fixation without holding its lock in the RTDE loop."""

    with fixation_lock:
        fixation = {
            "valid": latest_fixation["valid"],
            "position_robot_xy_m": list(
                latest_fixation["position_robot_xy_m"]
            ),
            "duration_s": latest_fixation["duration_s"],
            "update_time_s": latest_fixation["update_time_s"],
            "angular_velocity_deg_s": latest_fixation[
                "angular_velocity_deg_s"
            ],
        }

    return fixation


def read_tcp_position_xyz(rtde_receive_interface):
    """Read UR5e base-frame TCP position [x, y, z] in metres."""

    tcp_pose = rtde_receive_interface.getActualTCPPose()
    return [tcp_pose[0], tcp_pose[1], tcp_pose[2]]


def read_tcp_actual_velocity(rtde_receive_interface):
    """Read actual TCP velocity [m/s, rad/s] from the UR5e."""

    tcp_speed = rtde_receive_interface.getActualTCPSpeed()
    return list(tcp_speed)


def read_tcp_force_xy(rtde_receive_interface):
    """Read signed base-frame TCP force [Fx, Fy] in newtons."""

    tcp_wrench = rtde_receive_interface.getActualTCPForce()
    force_x_n = force_sign_x * tcp_wrench[0]
    force_y_n = force_sign_y * tcp_wrench[1]
    return [force_x_n, force_y_n]


def estimate_force_bias(rtde_receive_interface, sample_period_s):
    """Average unloaded planar force to estimate sensor bias [N]."""

    force_bias_n[0] = 0.0
    force_bias_n[1] = 0.0

    for _sample_index in range(force_bias_samples):
        measured_force_n = read_tcp_force_xy(rtde_receive_interface)
        force_bias_n[0] += measured_force_n[0]
        force_bias_n[1] += measured_force_n[1]
        time.sleep(sample_period_s)

    force_bias_n[0] /= force_bias_samples
    force_bias_n[1] /= force_bias_samples


def apply_force_deadband(force_n):
    """Return zero inside the configured force deadband."""

    if abs(force_n) < force_deadband_n:
        return 0.0

    return force_n


def update_force_filter(rtde_receive_interface):
    """Read, bias, deadband, and low-pass filter planar force [N]."""

    measured_force_n = read_tcp_force_xy(rtde_receive_interface)

    for axis in range(2):
        unbiased_force_n = measured_force_n[axis] - force_bias_n[axis]
        force_raw_n[axis] = apply_force_deadband(unbiased_force_n)

        old_force_n = force_filtered_n[axis]
        new_force_n = force_raw_n[axis]
        retained_force_n = (1.0 - force_lpf_alpha) * old_force_n
        added_force_n = force_lpf_alpha * new_force_n
        force_filtered_n[axis] = retained_force_n + added_force_n


def calculate_workspace_corners(start_position_xy_m):
    """Return absolute rectangular workspace corners in robot base [m]."""

    lower_corner_m = [
        start_position_xy_m[0] + workspace_lower_corner_offset_m[0],
        start_position_xy_m[1] + workspace_lower_corner_offset_m[1],
    ]
    upper_corner_m = [
        start_position_xy_m[0] + workspace_upper_corner_offset_m[0],
        start_position_xy_m[1] + workspace_upper_corner_offset_m[1],
    ]
    return lower_corner_m, upper_corner_m


def point_is_inside_rectangle(point_xy_m, lower_corner_m, upper_corner_m):
    """Return True when a 2-D point is inside the closed rectangle."""

    inside_x = lower_corner_m[0] <= point_xy_m[0]
    inside_x = inside_x and point_xy_m[0] <= upper_corner_m[0]
    inside_y = lower_corner_m[1] <= point_xy_m[1]
    inside_y = inside_y and point_xy_m[1] <= upper_corner_m[1]
    return inside_x and inside_y


def calculate_fixation_confidence(fixation_duration_s):
    """Map retained fixation duration to confidence in [0, 1]."""

    duration_above_minimum_s = (
        fixation_duration_s - minimum_fixation_duration_s
    )
    confidence_duration_range_s = (
        full_confidence_fixation_duration_s
        - minimum_fixation_duration_s
    )
    raw_confidence = (
        duration_above_minimum_s / confidence_duration_range_s
    )
    return clamp_value(raw_confidence, 0.0, 1.0)


def calculate_intent_influence(fixation_confidence):
    """Map confidence to gaze damping influence in [0, 1]."""

    if fixation_confidence < minimum_intent_confidence:
        return 0.0

    return fixation_confidence


def calculate_gaze_intent(
    recent_tcp_positions_xy_m,
    fixation,
    current_time_s,
    workspace_lower_corner_m,
    workspace_upper_corner_m,
):
    """Calculate normalized TCP-to-fixation intent in robot base x-y."""

    intent = {
        "target_xy_m": [math.nan, math.nan],
        "direction_xy": [0.0, 0.0],
        "distance_m": math.nan,
        "fixation_age_s": math.nan,
        "confidence": 0.0,
        "influence": 0.0,
        "target_valid": False,
        "active": False,
    }

    if not fixation["valid"]:
        return intent

    fixation_age_s = current_time_s - fixation["update_time_s"]
    intent["fixation_age_s"] = fixation_age_s

    if fixation_age_s > maximum_fixation_age_s:
        return intent

    fixation_target_xy_m = fixation["position_robot_xy_m"]
    intent["target_xy_m"] = fixation_target_xy_m

    target_inside_workspace = point_is_inside_rectangle(
        fixation_target_xy_m,
        workspace_lower_corner_m,
        workspace_upper_corner_m,
    )

    if not target_inside_workspace:
        return intent

    tcp_position_sum_x_m = 0.0
    tcp_position_sum_y_m = 0.0

    for tcp_position_xy_m in recent_tcp_positions_xy_m:
        tcp_position_sum_x_m += tcp_position_xy_m[0]
        tcp_position_sum_y_m += tcp_position_xy_m[1]

    tcp_sample_count = len(recent_tcp_positions_xy_m)
    mean_tcp_x_m = tcp_position_sum_x_m / tcp_sample_count
    mean_tcp_y_m = tcp_position_sum_y_m / tcp_sample_count

    intent_displacement_x_m = fixation_target_xy_m[0] - mean_tcp_x_m
    intent_displacement_y_m = fixation_target_xy_m[1] - mean_tcp_y_m
    intent_displacement_xy_m = [
        intent_displacement_x_m,
        intent_displacement_y_m,
    ]
    intent_distance_m = vector_norm_2d(intent_displacement_xy_m)
    intent["distance_m"] = intent_distance_m

    if intent_distance_m < minimum_intent_distance_m:
        return intent

    intent_direction_x = intent_displacement_x_m / intent_distance_m
    intent_direction_y = intent_displacement_y_m / intent_distance_m
    intent["direction_xy"] = [
        intent_direction_x,
        intent_direction_y,
    ]
    intent["target_valid"] = True

    confidence = calculate_fixation_confidence(fixation["duration_s"])
    influence = calculate_intent_influence(confidence)
    intent["confidence"] = confidence
    intent["influence"] = influence
    intent["active"] = influence > 0.0
    return intent


def calculate_damping_force(intent):
    """Calculate fixed or confidence-weighted directional damping force."""

    velocity_x_m_s = cart_velocity_command_m_s[0]
    velocity_y_m_s = cart_velocity_command_m_s[1]

    fixed_damping_force_x_n = (
        fixed_damping_x_n_s_m * velocity_x_m_s
    )
    fixed_damping_force_y_n = (
        fixed_damping_y_n_s_m * velocity_y_m_s
    )

    if not intent["active"]:
        return [fixed_damping_force_x_n, fixed_damping_force_y_n]

    intent_x = intent["direction_xy"][0]
    intent_y = intent["direction_xy"][1]
    signed_intent_speed_m_s = (
        velocity_x_m_s * intent_x
        + velocity_y_m_s * intent_y
    )

    forward_speed_m_s = max(signed_intent_speed_m_s, 0.0)
    opposite_speed_m_s = min(signed_intent_speed_m_s, 0.0)

    forward_velocity_x_m_s = forward_speed_m_s * intent_x
    forward_velocity_y_m_s = forward_speed_m_s * intent_y
    opposite_velocity_x_m_s = opposite_speed_m_s * intent_x
    opposite_velocity_y_m_s = opposite_speed_m_s * intent_y

    parallel_velocity_x_m_s = signed_intent_speed_m_s * intent_x
    parallel_velocity_y_m_s = signed_intent_speed_m_s * intent_y
    perpendicular_velocity_x_m_s = (
        velocity_x_m_s - parallel_velocity_x_m_s
    )
    perpendicular_velocity_y_m_s = (
        velocity_y_m_s - parallel_velocity_y_m_s
    )

    intent_damping_force_x_n = (
        intent_forward_damping_n_s_m * forward_velocity_x_m_s
    )
    intent_damping_force_x_n += (
        intent_other_damping_n_s_m * opposite_velocity_x_m_s
    )
    intent_damping_force_x_n += (
        intent_other_damping_n_s_m * perpendicular_velocity_x_m_s
    )

    intent_damping_force_y_n = (
        intent_forward_damping_n_s_m * forward_velocity_y_m_s
    )
    intent_damping_force_y_n += (
        intent_other_damping_n_s_m * opposite_velocity_y_m_s
    )
    intent_damping_force_y_n += (
        intent_other_damping_n_s_m * perpendicular_velocity_y_m_s
    )

    influence = intent["influence"]
    fixed_weight = 1.0 - influence
    damping_force_x_n = fixed_weight * fixed_damping_force_x_n
    damping_force_x_n += influence * intent_damping_force_x_n
    damping_force_y_n = fixed_weight * fixed_damping_force_y_n
    damping_force_y_n += influence * intent_damping_force_y_n

    return [damping_force_x_n, damping_force_y_n]


def integrate_admittance(intent, sample_period_s):
    """Integrate M*v_dot + damping_force = measured_force."""

    global cart_velocity_command_m_s

    damping_force_n = calculate_damping_force(intent)

    net_force_x_n = force_filtered_n[0] - damping_force_n[0]
    net_force_y_n = force_filtered_n[1] - damping_force_n[1]
    acceleration_x_m_s2 = net_force_x_n / virtual_mass_x_kg
    acceleration_y_m_s2 = net_force_y_n / virtual_mass_y_kg

    velocity_change_x_m_s = acceleration_x_m_s2 * sample_period_s
    velocity_change_y_m_s = acceleration_y_m_s2 * sample_period_s
    new_velocity_x_m_s = (
        cart_velocity_command_m_s[0] + velocity_change_x_m_s
    )
    new_velocity_y_m_s = (
        cart_velocity_command_m_s[1] + velocity_change_y_m_s
    )
    nominal_velocity_m_s = [
        new_velocity_x_m_s,
        new_velocity_y_m_s,
    ]
    nominal_velocity_m_s = limit_vector_norm_2d(
        nominal_velocity_m_s,
        max_cart_speed_m_s,
    )
    cart_velocity_command_m_s = nominal_velocity_m_s

    return damping_force_n, list(nominal_velocity_m_s)


def apply_rectangular_cbf(
    tcp_position_xy_m,
    nominal_velocity_xy_m_s,
    lower_corner_m,
    upper_corner_m,
):
    """Project nominal velocity onto rectangular CBF constraints.

    For h_lower_x = x - x_min and h_upper_x = x_max - x:
        vx >= -alpha*h_lower_x
        vx <=  alpha*h_upper_x
    The same equations are applied independently to y. For an axis-aligned
    rectangle this clipping is the minimum-change CBF velocity projection.
    """

    distance_from_lower_x_m = (
        tcp_position_xy_m[0] - lower_corner_m[0]
    )
    distance_from_upper_x_m = (
        upper_corner_m[0] - tcp_position_xy_m[0]
    )
    distance_from_lower_y_m = (
        tcp_position_xy_m[1] - lower_corner_m[1]
    )
    distance_from_upper_y_m = (
        upper_corner_m[1] - tcp_position_xy_m[1]
    )

    minimum_safe_vx_m_s = (
        -cbf_alpha_per_s * distance_from_lower_x_m
    )
    maximum_safe_vx_m_s = (
        cbf_alpha_per_s * distance_from_upper_x_m
    )
    minimum_safe_vy_m_s = (
        -cbf_alpha_per_s * distance_from_lower_y_m
    )
    maximum_safe_vy_m_s = (
        cbf_alpha_per_s * distance_from_upper_y_m
    )

    safe_velocity_x_m_s = clamp_value(
        nominal_velocity_xy_m_s[0],
        minimum_safe_vx_m_s,
        maximum_safe_vx_m_s,
    )
    safe_velocity_y_m_s = clamp_value(
        nominal_velocity_xy_m_s[1],
        minimum_safe_vy_m_s,
        maximum_safe_vy_m_s,
    )
    safe_velocity_xy_m_s = [
        safe_velocity_x_m_s,
        safe_velocity_y_m_s,
    ]

    cbf_modified_velocity = (
        safe_velocity_xy_m_s != nominal_velocity_xy_m_s
    )
    return safe_velocity_xy_m_s, cbf_modified_velocity


def send_speed_command(rtde_control_interface, velocity_xy_m_s, period_s):
    """Send planar speedL velocity while keeping z and rotation at zero."""

    speed_command = [
        velocity_xy_m_s[0],
        velocity_xy_m_s[1],
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    if send_robot_commands:
        rtde_control_interface.speedL(
            speed_command,
            speedl_acceleration_m_s2,
            period_s,
        )


def move_to_start_joint_configuration(rtde_control_interface):
    """Move to the configured start joints when robot commands are enabled."""

    if not send_robot_commands or not move_to_start_configuration:
        return

    start_joint_q_rad = []

    for joint_angle_deg in start_joint_q_deg:
        joint_angle_rad = math.radians(joint_angle_deg)
        start_joint_q_rad.append(joint_angle_rad)

    rtde_control_interface.moveJ(
        start_joint_q_rad,
        start_joint_speed_rad_s,
        start_joint_acceleration_rad_s2,
    )


def write_csv_header(csv_writer):
    """Write controller, gaze-intent, damping, and CBF column names."""

    csv_writer.writerow([
        "time_s",
        "host_monotonic_time_ns",
        "Fx_raw_N",
        "Fy_raw_N",
        "Fx_filtered_N",
        "Fy_filtered_N",
        "tcp_x_m",
        "tcp_y_m",
        "tcp_z_m",
        "tcp_vx_actual_m_s",
        "tcp_vy_actual_m_s",
        "fixation_x_robot_m",
        "fixation_y_robot_m",
        "fixation_duration_s",
        "fixation_valid",
        "fixation_age_s",
        "fixation_angular_velocity_deg_s",
        "intent_target_valid",
        "intent_active",
        "intent_confidence",
        "intent_influence",
        "intent_direction_x",
        "intent_direction_y",
        "intent_distance_m",
        "damping_force_x_N",
        "damping_force_y_N",
        "vx_nominal_m_s",
        "vy_nominal_m_s",
        "vx_cbf_m_s",
        "vy_cbf_m_s",
        "cbf_modified_velocity",
        "send_robot_commands",
    ])


def write_csv_row(
    csv_writer,
    time_s,
    host_time_ns,
    tcp_position_xyz_m,
    tcp_actual_velocity,
    fixation,
    intent,
    damping_force_n,
    nominal_velocity_xy_m_s,
    safe_velocity_xy_m_s,
    cbf_modified_velocity,
):
    """Write one synchronized controller sample."""

    fixation_position_xy_m = fixation["position_robot_xy_m"]

    csv_writer.writerow([
        time_s,
        host_time_ns,
        force_raw_n[0],
        force_raw_n[1],
        force_filtered_n[0],
        force_filtered_n[1],
        tcp_position_xyz_m[0],
        tcp_position_xyz_m[1],
        tcp_position_xyz_m[2],
        tcp_actual_velocity[0],
        tcp_actual_velocity[1],
        fixation_position_xy_m[0],
        fixation_position_xy_m[1],
        fixation["duration_s"],
        int(fixation["valid"]),
        intent["fixation_age_s"],
        fixation["angular_velocity_deg_s"],
        int(intent["target_valid"]),
        int(intent["active"]),
        intent["confidence"],
        intent["influence"],
        intent["direction_xy"][0],
        intent["direction_xy"][1],
        intent["distance_m"],
        damping_force_n[0],
        damping_force_n[1],
        nominal_velocity_xy_m_s[0],
        nominal_velocity_xy_m_s[1],
        safe_velocity_xy_m_s[0],
        safe_velocity_xy_m_s[1],
        int(cbf_modified_velocity),
        int(send_robot_commands),
    ])


def main():
    """Run real-time gaze-assisted planar admittance control."""

    global cart_velocity_command_m_s

    signal.signal(signal.SIGINT, handle_stop_signal)
    sample_period_s = 1.0 / control_frequency_hz

    if sorted(coppeliasim_axis_from_robot) != [0, 1]:
        raise ValueError("coppeliasim_axis_from_robot must permute [0, 1].")

    if full_confidence_fixation_duration_s <= minimum_fixation_duration_s:
        raise ValueError("Full-confidence duration must exceed the minimum.")

    tobii_reference_xy_mm, coppelia_reference_xy_m = (
        load_spatial_calibration()
    )

    print("Connecting to UR5e at", robot_ip)
    rtde_control_interface = rtde_control.RTDEControlInterface(robot_ip)
    rtde_receive_interface = rtde_receive.RTDEReceiveInterface(robot_ip)
    tracker = None
    gaze_callback = None
    trajectory_queue = None
    simulator_stop_event = None
    trajectory_process = None

    try:
        move_to_start_joint_configuration(rtde_control_interface)

        print("Keep the robot unloaded. Estimating planar force bias...")
        estimate_force_bias(rtde_receive_interface, sample_period_s)

        start_position_xyz_m = read_tcp_position_xyz(
            rtde_receive_interface
        )
        start_position_xy_m = start_position_xyz_m[:2]
        workspace_lower_corner_m, workspace_upper_corner_m = (
            calculate_workspace_corners(start_position_xy_m)
        )

        if not point_is_inside_rectangle(
            start_position_xy_m,
            workspace_lower_corner_m,
            workspace_upper_corner_m,
        ):
            raise ValueError(
                "The startup TCP must lie inside the CBF rectangle."
            )

        print("Workspace lower corner [m]:", workspace_lower_corner_m)
        print("Workspace upper corner [m]:", workspace_upper_corner_m)
        print("Subject preparation time [s]:", subject_preparation_time_s)
        time.sleep(subject_preparation_time_s)

        (
            trajectory_queue,
            simulator_stop_event,
            trajectory_process,
        ) = start_coppeliasim_trajectory_process()

        trial_clock = {"start_time_ns": None}
        tracker, gaze_callback = start_tobii_fixation_stream(
            trial_clock,
            tobii_reference_xy_mm,
            coppelia_reference_xy_m,
        )
        recent_tcp_positions_xy_m = deque(
            maxlen=intent_tcp_average_sample_count
        )

        write_data_csv_path.parent.mkdir(parents=True, exist_ok=True)

        with open(write_data_csv_path, "w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            write_csv_header(csv_writer)

            trial_start_time_ns = time.perf_counter_ns()
            trial_clock["start_time_ns"] = trial_start_time_ns
            maximum_step_count = int(
                maximum_runtime_s * control_frequency_hz
            )

            print("Gaze-assisted controller started.")
            print("send_robot_commands =", send_robot_commands)
            print("Press Ctrl+C to stop.")

            for _step_index in range(maximum_step_count):
                if not keep_running:
                    break

                period_start = rtde_control_interface.initPeriod()
                host_time_ns = time.perf_counter_ns()
                time_s = (host_time_ns - trial_start_time_ns) * 1e-9

                tcp_position_xyz_m = read_tcp_position_xyz(
                    rtde_receive_interface
                )
                tcp_position_xy_m = tcp_position_xyz_m[:2]
                tcp_actual_velocity = read_tcp_actual_velocity(
                    rtde_receive_interface
                )
                recent_tcp_positions_xy_m.append(tcp_position_xy_m)

                fixation = read_latest_fixation()
                intent = calculate_gaze_intent(
                    recent_tcp_positions_xy_m,
                    fixation,
                    time_s,
                    workspace_lower_corner_m,
                    workspace_upper_corner_m,
                )

                update_force_filter(rtde_receive_interface)
                damping_force_n, nominal_velocity_xy_m_s = (
                    integrate_admittance(intent, sample_period_s)
                )
                safe_velocity_xy_m_s, cbf_modified_velocity = (
                    apply_rectangular_cbf(
                        tcp_position_xy_m,
                        nominal_velocity_xy_m_s,
                        workspace_lower_corner_m,
                        workspace_upper_corner_m,
                    )
                )
                cart_velocity_command_m_s = safe_velocity_xy_m_s

                send_speed_command(
                    rtde_control_interface,
                    safe_velocity_xy_m_s,
                    sample_period_s,
                )
                write_csv_row(
                    csv_writer,
                    time_s,
                    host_time_ns,
                    tcp_position_xyz_m,
                    tcp_actual_velocity,
                    fixation,
                    intent,
                    damping_force_n,
                    nominal_velocity_xy_m_s,
                    safe_velocity_xy_m_s,
                    cbf_modified_velocity,
                )

                # Queue only; the ZeroMQ API runs in the child process.
                publish_coppeliasim_trajectory_sample(
                    trajectory_queue,
                    time_s,
                    tcp_position_xyz_m,
                )

                rtde_control_interface.waitPeriod(period_start)
    finally:
        if tracker is not None:
            stop_tobii_fixation_stream(tracker, gaze_callback)

        stop_coppeliasim_trajectory_process(
            trajectory_queue,
            simulator_stop_event,
            trajectory_process,
        )

        if send_robot_commands:
            rtde_control_interface.speedStop(
                speed_stop_deceleration_m_s2
            )

        rtde_control_interface.stopScript()

    print("Stopped UR5e gaze-assisted admittance controller.")
    print("CSV saved to:", write_data_csv_path)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
