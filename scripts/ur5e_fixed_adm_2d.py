"""
Fixed-admittance planar controller for UR5e motion in the base-frame x-y plane.

Inputs:
    UR5e TCP wrench from RTDE, using only [Fx, Fy] [N].
    UR5e TCP position from RTDE, using only [x, y] [m].
    UR5e actual TCP velocity from RTDE as [vx, vy, vz, wx, wy, wz],
    with linear components [m/s] and angular components [rad/s].

Outputs:
    Optional UR5e Cartesian velocity command [vx, vy, 0, 0, 0, 0].
    CSV log with planar force, TCP position, measured TCP velocity,
    commanded velocity, and fixed admittance data.
    Tobii gaze CSV containing every field returned by the SDK callback.
    Both CSV files use the same host-monotonic time_s origin. The Tobii CSV
    also preserves the SDK device and system timestamps without modification.
    Tobii time_s marks callback arrival on the host; the SDK timestamps remain
    available when device-acquisition timing or callback latency is required.
    Optional CoppeliaSim end-effector position trajectory through a separate
    Python process using the ZeroMQ Remote API.

Controller:
    Fixed planar admittance for each axis:

        M_x * x_ddot + B_x * x_dot = F_x
        M_y * y_ddot + B_y * y_dot = F_y

User-adjustable parameters:
    Set M_inertia_x/y [kg] and B_damping_x/y [Ns/m] near the top of the
    file. Positive mass and damping are required; practical starting ranges
    are 1 to 20 kg and 20 to 120 Ns/m. Force filtering, velocity, workspace,
    robot, startup, and logging settings are also user-adjustable there.

Dependency:
    pip install --user ur_rtde tobii-research coppeliasim-zmqremoteapi-client
"""

import csv
import json
import math
import multiprocessing
from pathlib import Path
import queue
import signal
import threading
import time
import traceback

try:
    import rtde_control
    import rtde_receive
except ImportError:
    rtde_control = None
    rtde_receive = None


# -------------------- User Settings --------------------
robot_ip = "192.168.1.100"  # TODO UR5e IP/hostname [valid robot address]
write_data_csv_path = (
    Path(__file__).resolve().parent.parent /
    "data" /
    "raw_data" /
    # "ur5e_fixed_adm_2d_test.csv"
    "ur5e_fixed_adm_2d_irregular_shape_2_d50.csv"
)
# TODO output CSV path [file path]; resolved from this script, not cwd.
tobii_data_csv_path = (
    Path(__file__).resolve().parent.parent /
    "data" /
    "raw_data" /
    # "tobii_fixed_adm_2d_test.csv"
    "tobii_fixed_adm_2d_irregular_shape_2_d50.csv"
)
# TODO Tobii output CSV path [file path]; one row per Tobii callback sample.

enable_tobii_recording = True  # TODO bool; requires a connected Tobii tracker
tobii_queue_size = 50000  # TODO callback buffer [samples], recommended >= 10000

send_robot_commands = True  # TODO bool; use True only after checking signs

move_to_start_configuration = True  # TODO bool; False skips startup move
start_joint_q_deg = [  # TODO joint angles [deg], within robot joint limits
    -77.64,
    -98.5,
    -95.80,
    -75.75,
    90.3,
    -2.75,
]
# start_joint_q_deg = [  # TODO joint angles [deg], lower z, close to table for calibration stuffs
#     -77.7,
#     -115.7,
#     -113.0,
#     -42.0,
#     90.22,
#     -2.7,
# ]  # end-effector close to table; lower z
start_joint_q = [q_deg * math.pi / 180.0 for q_deg in start_joint_q_deg]
start_joint_speed = 0.5  # TODO moveJ speed [rad/s], recommended 0.1 to 1
start_joint_acceleration = 0.5  # TODO moveJ acceleration [rad/s^2], 0.1 to 2

control_frequency = 500.0  # TODO UR5e native control rate [Hz], use 500
dt = 1.0 / control_frequency
speedl_acceleration = 0.5  # TODO speedL acceleration [m/s^2], 0.1 to 2
speed_stop_deceleration = 2.0  # TODO speedStop deceleration [m/s^2], 0.5 to 5

# Fixed planar admittance settings. Both mass and damping must be positive.
M_inertia_x = 10.0  # TODO virtual x mass [kg], recommended 1 to 20
M_inertia_y = 10.0  # TODO virtual y mass [kg], recommended 1 to 20
B_damping_x = 10.0  # TODO fixed x damping [Ns/m], recommended 20 to 120
B_damping_y = 10.0  # TODO fixed y damping [Ns/m], recommended 20 to 120

force_lpf_alpha = 0.2  # TODO filter alpha [0, 1]; smaller is smoother
force_deadband = 1.0  # TODO ignore force noise below this value [N], 0 to 5

max_cart_speed = 0.15  # TODO planar speed limit [m/s], recommended 0.01 to 0.15

# Axis-aligned rectangular workspace in the UR5e base-frame x-y plane.
# TODO lower [x, y] corner offsets from startup TCP [m]. Each value must be
# smaller than the corresponding upper-corner value.
workspace_lower_corner_offset_m = [-0.2, -0.2]
# TODO upper [x, y] corner offsets from startup TCP [m].
workspace_upper_corner_offset_m = [0.25, 0.25]

if (
    len(workspace_lower_corner_offset_m) != 2 or
    len(workspace_upper_corner_offset_m) != 2
):
    raise ValueError("Workspace corner offsets must contain [x, y].")

if any(
    lower_value >= upper_value
    for lower_value, upper_value in zip(
        workspace_lower_corner_offset_m,
        workspace_upper_corner_offset_m,
    )
):
    raise ValueError(
        "Each workspace lower-corner offset must be less than its "
        "upper-corner offset."
    )

force_bias_samples = 250  # TODO unloaded samples [count], recommended >= 100
max_runtime_seconds = 120  # TODO automatic stop time [s], must be positive
# TODO subject preparation delay before control/recording [s], valid >= 0.
subject_preparation_time_s = 1.0  # seconds

# TODO set signs so positive push directions produce positive base-frame motion.
force_sign_x = 1.0  # TODO x sign [unitless], valid values: -1 or +1
force_sign_y = 1.0  # TODO y sign [unitless], valid values: -1 or +1

# CoppeliaSim trajectory bridge. It runs in a separate process and never
# blocks the RTDE control loop. The C++ code uses a 40 ms simulator update.
send_coppeliasim_trajectory = True  # TODO bool; requires CoppeliaSim running
coppeliasim_host = "127.0.0.1"  # TODO ZeroMQ host [hostname or IPv4 address]
coppeliasim_port = 23000  # TODO ZeroMQ port [integer], default is 23000
coppeliasim_object_path = "/Sphere"  # TODO exact marker path in scene tree
coppeliasim_reference_object_path = ""  # TODO empty string means world frame
coppeliasim_update_frequency = 25.0  # TODO simulator update [Hz], recommended 25

# Match scripts/admittance_pb_velMode.cpp:
# CoppeliaSim x = -robot y, CoppeliaSim y = robot x, CoppeliaSim z = robot z.
coppeliasim_position_offset = [  # TODO world-frame offset [m]
    0.0,
    0.0,
    0.0,
]
coppeliasim_position_signs = [  # TODO axis signs [unitless], each is -1 or +1
    1.0,
    1.0,
    1.0,
]


# -------------------- Controller State --------------------
keep_running = True

force_bias = [0.0, 0.0]
force_raw = [0.0, 0.0]
force_filtered = [0.0, 0.0]
cart_velocity = [0.0, 0.0]
start_position = [0.0, 0.0]


def handle_stop_signal(_signal_number, _frame):
    """Stop the control loop when Ctrl+C is pressed."""

    global keep_running
    keep_running = False


def clamp_value(value, lower_limit, upper_limit):
    """Return a scalar limited to [lower_limit, upper_limit]."""

    if value < lower_limit:
        return lower_limit

    if value > upper_limit:
        return upper_limit

    return value


def vec2_norm(vector):
    """Return the Euclidean norm of a 2-D vector."""

    return math.sqrt(
        vector[0] * vector[0] +
        vector[1] * vector[1]
    )


def deadband_value(value, deadband):
    """Set a small force value to zero."""

    if abs(value) < deadband:
        return 0.0

    return value


def limit_vector_norm(vector, max_norm):
    """Limit a 2-D vector magnitude while preserving its direction."""

    vector_norm = vec2_norm(vector)

    if vector_norm <= max_norm:
        return vector

    scale = max_norm / vector_norm
    return [
        scale * vector[0],
        scale * vector[1],
    ]


def read_tcp_force_xy(rtde_receive_interface):
    """Read base-frame TCP force [Fx, Fy] in N from the UR5e."""

    tcp_wrench = rtde_receive_interface.getActualTCPForce()

    return [
        force_sign_x * tcp_wrench[0],
        force_sign_y * tcp_wrench[1],
    ]


def read_tcp_position_xy(rtde_receive_interface):
    """Read base-frame TCP position [x, y] in m from the UR5e."""

    tcp_position_xyz = read_tcp_position_xyz(rtde_receive_interface)

    return [
        tcp_position_xyz[0],
        tcp_position_xyz[1],
    ]


def read_tcp_position_xyz(rtde_receive_interface):
    """Read base-frame TCP position [x, y, z] in m from the UR5e."""

    tcp_pose = rtde_receive_interface.getActualTCPPose()

    return [
        tcp_pose[0],
        tcp_pose[1],
        tcp_pose[2],
    ]


def read_tcp_actual_velocity(rtde_receive_interface):
    """Read actual base-frame TCP velocity from the UR5e.

    Returns:
        [vx, vy, vz, wx, wy, wz], where linear components are [m/s]
        and angular components are [rad/s].
    """

    tcp_actual_speed = rtde_receive_interface.getActualTCPSpeed()

    return [
        tcp_actual_speed[0],
        tcp_actual_speed[1],
        tcp_actual_speed[2],
        tcp_actual_speed[3],
        tcp_actual_speed[4],
        tcp_actual_speed[5],
    ]


def map_tcp_position_to_coppeliasim(tcp_position_xyz):
    """Map UR5e base-frame position [m] to the CoppeliaSim frame [m]."""

    return [
        coppeliasim_position_offset[0] +
        coppeliasim_position_signs[0] * tcp_position_xyz[0],
        coppeliasim_position_offset[1] +
        coppeliasim_position_signs[1] * tcp_position_xyz[1],
        coppeliasim_position_offset[2] +
        coppeliasim_position_signs[2] * tcp_position_xyz[2],
    ]


def coppeliasim_trajectory_process(trajectory_queue, stop_event):
    """Send the latest TCP position to CoppeliaSim from a child process.

    Input:
        trajectory_queue: multiprocessing queue containing
            (controller_time_s, tcp_position_xyz_m) samples.
        stop_event: multiprocessing event set by the robot process at exit.

    Output:
        The configured CoppeliaSim object receives setObjectPosition() calls.

    Assumptions:
        CoppeliaSim is already running with its ZeroMQ Remote API service
        enabled. Simulator communication errors terminate only this child.
    """

    try:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    except ImportError as error:
        print(
            "CoppeliaSim bridge disabled: cannot import ZeroMQ client:",
            error,
            flush=True,
        )
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
            "CoppeliaSim bridge found target",
            coppeliasim_object_path,
            "with handle",
            target_object_handle,
            flush=True,
        )

        update_period_s = 1.0 / coppeliasim_update_frequency
        next_update_time = 0.0
        sent_sample_count = 0

        print(
            "CoppeliaSim bridge connected; updating",
            coppeliasim_object_path,
            "at",
            coppeliasim_update_frequency,
            "Hz",
        )

        while not stop_event.is_set():
            try:
                trajectory_sample = trajectory_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Discard stale samples so simulator lag cannot build a backlog.
            while True:
                try:
                    trajectory_sample = trajectory_queue.get_nowait()
                except queue.Empty:
                    break

            current_time = time.monotonic()
            if current_time < next_update_time:
                continue

            tcp_position_xyz = trajectory_sample[1]
            coppeliasim_position = map_tcp_position_to_coppeliasim(
                tcp_position_xyz
            )

            # Non-stepping mode: this process updates the object only. It does
            # not start, stop, or step the CoppeliaSim simulation.
            sim.setObjectPosition(
                target_object_handle,
                reference_object_handle,
                coppeliasim_position,
            )

            if sent_sample_count == 0:
                print(
                    "CoppeliaSim first marker position:",
                    coppeliasim_position,
                    flush=True,
                )
            sent_sample_count += 1

            next_update_time = current_time + update_period_s

    except Exception as error:
        print("CoppeliaSim bridge stopped:", repr(error), flush=True)
        traceback.print_exc()


def start_coppeliasim_trajectory_process():
    """Start the independent CoppeliaSim trajectory process if enabled."""

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
    time_s,
    tcp_position_xyz,
):
    """Enqueue only the newest TCP position without blocking robot control."""

    if trajectory_queue is None:
        return

    try:
        while True:
            trajectory_queue.get_nowait()
    except queue.Empty:
        pass

    try:
        trajectory_queue.put_nowait((time_s, list(tcp_position_xyz)))
    except queue.Full:
        # The child process is consuming the queue. A later sample will retry.
        pass


def stop_coppeliasim_trajectory_process(
    trajectory_queue,
    stop_event,
    trajectory_process,
):
    """Stop and release the independent CoppeliaSim trajectory process."""

    if trajectory_process is None:
        return

    stop_event.set()
    trajectory_process.join(timeout=2.0)

    if trajectory_process.is_alive():
        trajectory_process.terminate()
        trajectory_process.join()

    trajectory_queue.close()


def make_tobii_json_safe(value):
    """Convert SDK values into JSON-compatible values without data loss."""

    if isinstance(value, dict):
        return {
            str(key): make_tobii_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [make_tobii_json_safe(item) for item in value]

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def make_tobii_csv_value(value):
    """Serialize compound SDK values into one CSV cell."""

    if isinstance(value, (dict, list, tuple)):
        return json.dumps(
            make_tobii_json_safe(value),
            separators=(",", ":"),
        )

    return value


def write_tobii_csv(gaze_queue, writer_stop_event):
    """Write synchronized Tobii callback samples to CSV in a thread."""

    with open(tobii_data_csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_field_names = None

        while not writer_stop_event.is_set() or not gaze_queue.empty():
            try:
                (
                    time_s,
                    host_monotonic_time_ns,
                    gaze_sample,
                ) = gaze_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if csv_field_names is None:
                csv_field_names = sorted(gaze_sample.keys())
                csv_writer.writerow(
                    ["time_s", "host_monotonic_time_ns"] +
                    csv_field_names +
                    ["sdk_data_json"]
                )

            csv_row = [time_s, host_monotonic_time_ns] + [
                make_tobii_csv_value(gaze_sample.get(field_name))
                for field_name in csv_field_names
            ]
            csv_row.append(
                json.dumps(
                    make_tobii_json_safe(gaze_sample),
                    separators=(",", ":"),
                )
            )
            csv_writer.writerow(csv_row)


def create_tobii_gaze_callback(
    gaze_queue,
    recording_active_event,
    trial_clock,
):
    """Queue SDK gaze data with the shared host-monotonic trial time."""

    def gaze_callback(data):
        """Timestamp and queue one SDK dictionary without file I/O."""

        if not recording_active_event.is_set():
            return

        host_monotonic_time_ns = time.perf_counter_ns()
        trial_start_time_ns = trial_clock["start_time_ns"]

        if trial_start_time_ns is None:
            return

        gaze_sample = dict(data)
        time_s = (
            host_monotonic_time_ns - trial_start_time_ns
        ) * 1e-9

        # The second check prevents a callback that overlaps trial shutdown
        # from adding a post-stop sample to the queue.
        if not recording_active_event.is_set():
            return

        try:
            gaze_queue.put_nowait(
                (
                    time_s,
                    host_monotonic_time_ns,
                    gaze_sample,
                )
            )
        except queue.Full:
            # Do not block the Tobii SDK callback if the writer falls behind.
            pass

    return gaze_callback


def start_tobii_recording(recording_active_event, trial_clock):
    """Arm Tobii acquisition and its synchronized CSV writer."""

    if not enable_tobii_recording:
        print("Tobii recording disabled by user setting.")
        return None

    try:
        import tobii_research as tobii_research_api
    except ImportError:
        print(
            "Tobii recording disabled: install with "
            "`pip install --user tobii-research`"
        )
        return None

    trackers = tobii_research_api.find_all_eyetrackers()

    if not trackers:
        print("No Tobii eye tracker found. Tobii recording disabled.")
        return None

    tracker = trackers[0]
    gaze_queue = queue.Queue(maxsize=tobii_queue_size)
    writer_stop_event = threading.Event()
    writer_thread = threading.Thread(
        target=write_tobii_csv,
        args=(gaze_queue, writer_stop_event),
        name="tobii_csv_writer",
        daemon=True,
    )
    gaze_callback = create_tobii_gaze_callback(
        gaze_queue,
        recording_active_event,
        trial_clock,
    )

    writer_thread.start()
    tracker.subscribe_to(
        tobii_research_api.EYETRACKER_GAZE_DATA,
        gaze_callback,
        as_dictionary=True,
    )

    print("Tobii model:", tracker.model)
    print("Tobii serial:", tracker.serial_number)
    print(
        "Tobii frequency:",
        tracker.get_gaze_output_frequency(),
        "Hz",
    )
    print("Tobii CSV:", tobii_data_csv_path)

    return (
        tobii_research_api,
        tracker,
        gaze_callback,
        writer_stop_event,
        writer_thread,
    )


def stop_tobii_recording(tobii_recording):
    """Unsubscribe from Tobii and flush the remaining gaze queue."""

    if tobii_recording is None:
        return

    (
        tobii_research_api,
        tracker,
        gaze_callback,
        writer_stop_event,
        writer_thread,
    ) = tobii_recording

    try:
        tracker.unsubscribe_from(
            tobii_research_api.EYETRACKER_GAZE_DATA,
            gaze_callback,
        )
    finally:
        writer_stop_event.set()
        writer_thread.join(timeout=2.0)


def estimate_force_bias(rtde_receive_interface):
    """Average unloaded startup force to estimate the planar force bias."""

    global force_bias

    force_bias = [0.0, 0.0]

    for _sample_idx in range(force_bias_samples):
        measured_force = read_tcp_force_xy(rtde_receive_interface)

        force_bias[0] += measured_force[0]
        force_bias[1] += measured_force[1]

        time.sleep(dt)

    force_bias[0] /= force_bias_samples
    force_bias[1] /= force_bias_samples


def update_force_filter(rtde_receive_interface):
    """Read, bias, deadband, and low-pass filter planar force in N."""

    measured_force = read_tcp_force_xy(rtde_receive_interface)

    for axis in range(2):
        force_raw[axis] = measured_force[axis] - force_bias[axis]
        force_raw[axis] = deadband_value(force_raw[axis], force_deadband)

        force_filtered[axis] = (
            (1.0 - force_lpf_alpha) * force_filtered[axis] +
            force_lpf_alpha * force_raw[axis]
        )


def integrate_fixed_admittance():
    """Integrate M*v_dot + B*v = F for the planar x-y axes."""

    global cart_velocity

    acceleration_x = (force_filtered[0] - B_damping_x * cart_velocity[0]) / M_inertia_x
    acceleration_y = (force_filtered[1] - B_damping_y * cart_velocity[1]) / M_inertia_y

    cart_velocity[0] += acceleration_x * dt
    cart_velocity[1] += acceleration_y * dt

    cart_velocity = limit_vector_norm(cart_velocity, max_cart_speed)


def apply_workspace_limit(tcp_position):
    """Remove velocity components pointing outside the x-y rectangle.

    Input:
        tcp_position: measured [x, y] TCP position [m] in the UR5e base frame.

    The rectangular corners are user-defined offsets from start_position [m].
    Inward and edge-tangential velocity components remain available.
    """

    workspace_lower_corner_m = [
        start_position[0] + workspace_lower_corner_offset_m[0],
        start_position[1] + workspace_lower_corner_offset_m[1],
    ]
    workspace_upper_corner_m = [
        start_position[0] + workspace_upper_corner_offset_m[0],
        start_position[1] + workspace_upper_corner_offset_m[1],
    ]

    at_lower_x = tcp_position[0] <= workspace_lower_corner_m[0]
    at_upper_x = tcp_position[0] >= workspace_upper_corner_m[0]
    at_lower_y = tcp_position[1] <= workspace_lower_corner_m[1]
    at_upper_y = tcp_position[1] >= workspace_upper_corner_m[1]

    if (at_lower_x and cart_velocity[0] < 0.0) or (
        at_upper_x and cart_velocity[0] > 0.0
    ):
        cart_velocity[0] = 0.0

    if (at_lower_y and cart_velocity[1] < 0.0) or (
        at_upper_y and cart_velocity[1] > 0.0
    ):
        cart_velocity[1] = 0.0


def send_speed_command(rtde_control_interface):
    """Send planar TCP velocity [m/s] and zero rotation through speedL()."""

    speed_command = [
        cart_velocity[0],
        cart_velocity[1],
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    if send_robot_commands:
        rtde_control_interface.speedL(
            speed_command,
            speedl_acceleration,
            dt
        )


def move_to_start_joint_configuration(rtde_control_interface):
    """Move UR5e to the user-defined joint configuration."""

    if not send_robot_commands:
        print("Skipping start joint move because send_robot_commands=False")
        return

    if not move_to_start_configuration:
        print("Skipping start joint move by user setting.")
        return

    print("Moving UR5e to start_joint_q [rad]:")
    print(start_joint_q)

    rtde_control_interface.moveJ(
        start_joint_q,
        start_joint_speed,
        start_joint_acceleration
    )


def write_csv_header(csv_writer):
    """Write names and units for the planar controller CSV columns."""

    csv_writer.writerow([
        "time_s",
        "host_monotonic_time_ns",
        "Fx_raw_N",
        "Fy_raw_N",
        "Fx_filt_N",
        "Fy_filt_N",
        "x_m",
        "y_m",
        "z_m",
        "coppeliasim_x_m",
        "coppeliasim_y_m",
        "coppeliasim_z_m",
        "tcp_vx_actual_m_s",
        "tcp_vy_actual_m_s",
        "tcp_vz_actual_m_s",
        "tcp_wx_actual_rad_s",
        "tcp_wy_actual_rad_s",
        "tcp_wz_actual_rad_s",
        "vx_cmd_m_s",
        "vy_cmd_m_s",
        "speed_cmd_m_s",
        "M_x_kg",
        "M_y_kg",
        "B_x_Ns_m",
        "B_y_Ns_m",
        "send_robot_commands",
    ])


def write_csv_row(
    csv_writer,
    time_s,
    host_monotonic_time_ns,
    tcp_position,
    tcp_actual_velocity,
):
    """Write one controller sample with measured TCP state to CSV."""

    coppeliasim_position = map_tcp_position_to_coppeliasim(tcp_position)

    csv_writer.writerow([
        time_s,
        host_monotonic_time_ns,
        force_raw[0],
        force_raw[1],
        force_filtered[0],
        force_filtered[1],
        tcp_position[0],
        tcp_position[1],
        tcp_position[2],
        coppeliasim_position[0],
        coppeliasim_position[1],
        coppeliasim_position[2],
        tcp_actual_velocity[0],
        tcp_actual_velocity[1],
        tcp_actual_velocity[2],
        tcp_actual_velocity[3],
        tcp_actual_velocity[4],
        tcp_actual_velocity[5],
        cart_velocity[0],
        cart_velocity[1],
        vec2_norm(cart_velocity),
        M_inertia_x,
        M_inertia_y,
        B_damping_x,
        B_damping_y,
        int(send_robot_commands),
    ])


def main():
    """Run the UR5e fixed planar x-y admittance loop."""

    global start_position

    if rtde_control is None or rtde_receive is None:
        print("Missing dependency: install with `pip install --user ur_rtde`")
        return

    signal.signal(signal.SIGINT, handle_stop_signal)

    print("Connecting to UR5e at", robot_ip)
    rtde_control_interface = rtde_control.RTDEControlInterface(robot_ip)
    rtde_receive_interface = rtde_receive.RTDEReceiveInterface(robot_ip)

    move_to_start_joint_configuration(rtde_control_interface)

    print("Keep the robot unloaded. Estimating planar force bias...")
    estimate_force_bias(rtde_receive_interface)

    start_position = read_tcp_position_xy(rtde_receive_interface)

    print(
        "Subject preparation: control and recording start in",
        subject_preparation_time_s,
        "seconds."
    )
    time.sleep(subject_preparation_time_s)

    (
        trajectory_queue,
        simulator_stop_event,
        trajectory_process,
    ) = start_coppeliasim_trajectory_process()
    recording_active_event = threading.Event()
    trial_clock = {"start_time_ns": None}
    tobii_recording = start_tobii_recording(
        recording_active_event,
        trial_clock,
    )

    try:
        with open(write_data_csv_path, "w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            write_csv_header(csv_writer)

            max_steps = int(max_runtime_seconds / dt)
            trial_start_time_ns = time.perf_counter_ns()
            trial_clock["start_time_ns"] = trial_start_time_ns
            recording_active_event.set()

            print("Controller and synchronized recording started.")
            print("Press Ctrl+C to stop.")
            print("send_robot_commands =", send_robot_commands)
            print("M_inertia_x/y [kg] =", M_inertia_x, M_inertia_y)
            print("B_damping_x/y [Ns/m] =", B_damping_x, B_damping_y)

            for step_idx in range(max_steps):
                if not keep_running:
                    break

                period_start = rtde_control_interface.initPeriod()
                host_monotonic_time_ns = time.perf_counter_ns()
                time_s = (
                    host_monotonic_time_ns - trial_start_time_ns
                ) * 1e-9

                tcp_position_xyz = read_tcp_position_xyz(
                    rtde_receive_interface
                )
                tcp_actual_velocity = read_tcp_actual_velocity(
                    rtde_receive_interface
                )
                tcp_position_xy = [
                    tcp_position_xyz[0],
                    tcp_position_xyz[1],
                ]

                update_force_filter(rtde_receive_interface)
                integrate_fixed_admittance()
                apply_workspace_limit(tcp_position_xy)
                send_speed_command(rtde_control_interface)

                write_csv_row(
                    csv_writer,
                    time_s,
                    host_monotonic_time_ns,
                    tcp_position_xyz,
                    tcp_actual_velocity,
                )

                # Queue-only operation: no CoppeliaSim API call occurs here.
                publish_coppeliasim_trajectory_sample(
                    trajectory_queue,
                    time_s,
                    tcp_position_xyz,
                )

                rtde_control_interface.waitPeriod(period_start)
    finally:
        recording_active_event.clear()
        stop_tobii_recording(tobii_recording)

        stop_coppeliasim_trajectory_process(
            trajectory_queue,
            simulator_stop_event,
            trajectory_process,
        )

        if send_robot_commands:
            rtde_control_interface.speedStop(speed_stop_deceleration)

        rtde_control_interface.stopScript()

    print("Stopped UR5e fixed planar admittance controller.")
    print("CSV saved to:", write_data_csv_path)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
