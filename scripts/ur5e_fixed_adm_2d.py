"""
Fixed-admittance planar controller for UR5e motion in the base-frame x-y plane.

Inputs:
    UR5e TCP wrench from RTDE, using only [Fx, Fy] [N].
    UR5e TCP position from RTDE, using only [x, y] [m].

Outputs:
    Optional UR5e Cartesian velocity command [vx, vy, 0, 0, 0, 0].
    CSV log with planar force, position, velocity, and fixed admittance data.

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
    pip install --user ur_rtde
"""

import csv
import math
import signal
import time

try:
    import rtde_control
    import rtde_receive
except ImportError:
    rtde_control = None
    rtde_receive = None


# -------------------- User Settings --------------------
robot_ip = "192.168.1.100"  # TODO UR5e IP/hostname [valid robot address]
write_data_csv_path = "../data/raw_data/ur5e_fixed_adm_2d_log.csv"
# TODO output CSV path [file path]; keep raw data in data/raw_data.

send_robot_commands = True  # TODO bool; use True only after checking signs

move_to_start_configuration = True  # TODO bool; False skips startup move
start_joint_q_deg = [  # TODO joint angles [deg], within robot joint limits
    -85.0,
    -97.0,
    -97.0,
    -76.0,
    90.0,
    -10.0,
]
start_joint_q = [q_deg * math.pi / 180.0 for q_deg in start_joint_q_deg]
start_joint_speed = 0.5  # TODO moveJ speed [rad/s], recommended 0.1 to 1
start_joint_acceleration = 0.5  # TODO moveJ acceleration [rad/s^2], 0.1 to 2

control_frequency = 125.0  # TODO control rate [Hz], recommended 125 to 500
dt = 1.0 / control_frequency
speedl_acceleration = 0.5  # TODO speedL acceleration [m/s^2], 0.1 to 2
speed_stop_deceleration = 2.0  # TODO speedStop deceleration [m/s^2], 0.5 to 5

# Fixed planar admittance settings. Both mass and damping must be positive.
M_inertia_x = 10.0  # TODO virtual x mass [kg], recommended 1 to 20
M_inertia_y = 10.0  # TODO virtual y mass [kg], recommended 1 to 20
B_damping_x = 60.0  # TODO fixed x damping [Ns/m], recommended 20 to 120
B_damping_y = 60.0  # TODO fixed y damping [Ns/m], recommended 20 to 120

force_lpf_alpha = 0.2  # TODO filter alpha [0, 1]; smaller is smoother
force_deadband = 1.0  # TODO ignore force noise below this value [N], 0 to 5

max_cart_speed = 0.15  # TODO planar speed limit [m/s], recommended 0.01 to 0.15
max_displacement_from_start = 0.25  # TODO planar radius [m], must be positive

force_bias_samples = 250  # TODO unloaded samples [count], recommended >= 100
max_runtime_seconds = 120  # TODO automatic stop time [s], must be positive

# TODO set signs so positive push directions produce positive base-frame motion.
force_sign_x = 1.0  # TODO x sign [unitless], valid values: -1 or +1
force_sign_y = 1.0  # TODO y sign [unitless], valid values: -1 or +1


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

    tcp_pose = rtde_receive_interface.getActualTCPPose()

    return [
        tcp_pose[0],
        tcp_pose[1],
    ]


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
    """Remove outward planar velocity beyond the start-position radius."""

    displacement = [
        tcp_position[0] - start_position[0],
        tcp_position[1] - start_position[1],
    ]

    distance = vec2_norm(displacement)

    if distance < max_displacement_from_start or distance < 1e-6:
        return

    outward_velocity = (
        cart_velocity[0] * displacement[0] +
        cart_velocity[1] * displacement[1]
    ) / distance

    if outward_velocity <= 0.0:
        return

    cart_velocity[0] -= outward_velocity * displacement[0] / distance
    cart_velocity[1] -= outward_velocity * displacement[1] / distance


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
        "Fx_raw_N",
        "Fy_raw_N",
        "Fx_filt_N",
        "Fy_filt_N",
        "x_m",
        "y_m",
        "vx_cmd_m_s",
        "vy_cmd_m_s",
        "speed_cmd_m_s",
        "M_x_kg",
        "M_y_kg",
        "B_x_Ns_m",
        "B_y_Ns_m",
        "send_robot_commands",
    ])


def write_csv_row(csv_writer, time_s, tcp_position):
    """Write one planar controller sample to the CSV file."""

    csv_writer.writerow([
        time_s,
        force_raw[0],
        force_raw[1],
        force_filtered[0],
        force_filtered[1],
        tcp_position[0],
        tcp_position[1],
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

    print("Controller started. Press Ctrl+C to stop.")
    print("send_robot_commands =", send_robot_commands)
    print("M_inertia_x/y [kg] =", M_inertia_x, M_inertia_y)
    print("B_damping_x/y [Ns/m] =", B_damping_x, B_damping_y)

    with open(write_data_csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        write_csv_header(csv_writer)

        max_steps = int(max_runtime_seconds / dt)

        for step_idx in range(max_steps):
            if not keep_running:
                break

            period_start = rtde_control_interface.initPeriod()

            tcp_position = read_tcp_position_xy(rtde_receive_interface)

            update_force_filter(rtde_receive_interface)
            integrate_fixed_admittance()
            apply_workspace_limit(tcp_position)
            send_speed_command(rtde_control_interface)

            time_s = step_idx * dt
            write_csv_row(csv_writer, time_s, tcp_position)

            rtde_control_interface.waitPeriod(period_start)

    if send_robot_commands:
        rtde_control_interface.speedStop(speed_stop_deceleration)

    rtde_control_interface.stopScript()

    print("Stopped UR5e fixed planar admittance controller.")
    print("CSV saved to:", write_data_csv_path)


if __name__ == "__main__":
    main()
