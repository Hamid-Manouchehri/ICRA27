"""
Simple direct-intention variable admittance controller for UR5e.

Inputs:
    UR5e built-in TCP wrench from RTDE: [Fx Fy Fz Tx Ty Tz].
    UR5e TCP pose from RTDE: [x y z rx ry rz].

Outputs:
    Optional UR5e Cartesian velocity command through speedL().
    CSV log with raw force, biased/filtered force, damping, and velocity.

Controller:
    Direct intention only:

        M * x_ddot + B * x_dot = F_h

    Kang-style variable damping:

        B = Fstd / |x_dot|

    The damping is clamped between B_min and B_max. Small velocity gives
    high damping for precision. Larger velocity gives lower damping for
    easier/faster motion.

Dependency:
    pip install --user ur_rtde
"""

import csv
import math
import signal
import time
from pathlib import Path

try:
    import rtde_control
    import rtde_receive
except ImportError:
    rtde_control = None
    rtde_receive = None


# -------------------- User Settings --------------------
robot_ip = "192.168.1.100"  # TODO UR5e IP address

write_data_csv_path = (
    Path(__file__).resolve().parent.parent /
    "data" /
    "raw_data" /
    "ur5e_var_adm_direct_3d_log.csv"
)
# TODO output CSV path [file path]; resolved from this script, not cwd.

send_robot_commands = True  # TODO set True only after checking signs

move_to_start_configuration = True  # TODO set False to skip startup move
start_joint_q_deg = [  # TODO UR5e start joint angles [deg]
    -85.0,
    -97.0,
    -97.0,
    -76.0,
    90.0,
    -10.0,
]
start_joint_q = [q_deg * math.pi / 180.0 for q_deg in start_joint_q_deg]
start_joint_speed = 0.5  # TODO moveJ joint speed [rad/s]
start_joint_acceleration = 0.5  # TODO moveJ joint acceleration [rad/s^2]

control_frequency = 125.0  # TODO [Hz], start low; UR e-series can do 500
dt = 1.0 / control_frequency
speedl_acceleration = 0.5  # TODO speedL acceleration [m/s^2]
speed_stop_deceleration = 2.0  # TODO speedStop deceleration [m/s^2]

M_inertia = 10.0  # TODO virtual mass [kg], try 5 to 20
B_min = 20.0  # TODO low damping [Ns/m], fast/light
B_max = 120.0  # TODO high damping [Ns/m], slow/stable
B_rate_max = 300.0  # TODO damping slew rate [Ns/m/s]

Fstd = 3.0  # TODO standard force [N], try 2 to 5
speed_eps = 0.02  # TODO avoids huge B near zero speed [m/s]

force_lpf_alpha = 0.2  # TODO smaller=smoother, range 0 to 1
force_deadband = 1.0  # TODO ignore small force noise [N]

max_cart_speed = 0.15  # TODO translational speed limit [m/s]
max_displacement_from_start = 0.25  # TODO workspace radius [m]

force_bias_samples = 250  # TODO keep robot unloaded at startup
max_runtime_seconds = 120  # TODO automatic stop time [s]

# TODO set signs if robot moves opposite to your push direction.
force_sign_x = 1.0
force_sign_y = 1.0
force_sign_z = 1.0


# -------------------- Controller State --------------------
keep_running = True

force_bias = [0.0, 0.0, 0.0]
force_raw = [0.0, 0.0, 0.0]
force_filtered = [0.0, 0.0, 0.0]
cart_velocity = [0.0, 0.0, 0.0]
start_position = [0.0, 0.0, 0.0]

B_desired = B_max
B_applied = B_max


def handle_stop_signal(_signal_number, _frame):
    """Stop the control loop when Ctrl+C is pressed."""

    global keep_running
    keep_running = False


def clamp_value(value, lower_limit, upper_limit):
    """Limit a scalar value between lower_limit and upper_limit."""

    if value < lower_limit:
        return lower_limit

    if value > upper_limit:
        return upper_limit

    return value


def vec3_norm(vector):
    """Return Euclidean norm of a 3-D vector."""

    return math.sqrt(
        vector[0] * vector[0] +
        vector[1] * vector[1] +
        vector[2] * vector[2]
    )


def deadband_value(value, deadband):
    """Set small force values to zero."""

    if abs(value) < deadband:
        return 0.0

    return value


def limit_vector_norm(vector, max_norm):
    """Limit a 3-D vector magnitude."""

    vector_norm = vec3_norm(vector)

    if vector_norm <= max_norm:
        return vector

    scale = max_norm / vector_norm
    return [
        scale * vector[0],
        scale * vector[1],
        scale * vector[2],
    ]


def compute_variable_damping(speed):
    """Compute Kang direct-intention damping B = Fstd / |velocity|."""

    safe_speed = max(speed, speed_eps)
    damping = Fstd / safe_speed

    return clamp_value(damping, B_min, B_max)


def rate_limit_damping(desired_damping, previous_damping):
    """Limit how fast damping can change between control steps."""

    max_step = B_rate_max * dt
    damping_step = desired_damping - previous_damping
    damping_step = clamp_value(damping_step, -max_step, max_step)

    return previous_damping + damping_step


def read_tcp_force_xyz(rtde_receive_interface):
    """Read UR TCP force and keep only translational force [N]."""

    tcp_wrench = rtde_receive_interface.getActualTCPForce()

    return [
        force_sign_x * tcp_wrench[0],
        force_sign_y * tcp_wrench[1],
        force_sign_z * tcp_wrench[2],
    ]


def read_tcp_position_xyz(rtde_receive_interface):
    """Read UR TCP position in the robot base frame [m]."""

    tcp_pose = rtde_receive_interface.getActualTCPPose()

    return [
        tcp_pose[0],
        tcp_pose[1],
        tcp_pose[2],
    ]


def estimate_force_bias(rtde_receive_interface):
    """Average startup force and use it as the no-touch offset."""

    global force_bias

    force_bias = [0.0, 0.0, 0.0]

    for _sample_idx in range(force_bias_samples):
        measured_force = read_tcp_force_xyz(rtde_receive_interface)

        force_bias[0] += measured_force[0]
        force_bias[1] += measured_force[1]
        force_bias[2] += measured_force[2]

        time.sleep(dt)

    force_bias[0] /= force_bias_samples
    force_bias[1] /= force_bias_samples
    force_bias[2] /= force_bias_samples


def update_force_filter(rtde_receive_interface):
    """Read force, subtract bias, apply deadband, then low-pass filter."""

    measured_force = read_tcp_force_xyz(rtde_receive_interface)

    for axis in range(3):
        force_raw[axis] = measured_force[axis] - force_bias[axis]
        force_raw[axis] = deadband_value(force_raw[axis], force_deadband)

        force_filtered[axis] = (
            (1.0 - force_lpf_alpha) * force_filtered[axis] +
            force_lpf_alpha * force_raw[axis]
        )


def integrate_direct_admittance():
    """Integrate M*x_ddot + B*x_dot = F_h."""

    global B_desired
    global B_applied
    global cart_velocity

    speed = vec3_norm(cart_velocity)

    B_desired = compute_variable_damping(speed)
    B_applied = rate_limit_damping(B_desired, B_applied)

    for axis in range(3):
        acceleration = (
            force_filtered[axis] -
            B_applied * cart_velocity[axis]
        ) / M_inertia

        cart_velocity[axis] += acceleration * dt

    cart_velocity = limit_vector_norm(cart_velocity, max_cart_speed)


def apply_workspace_limit(tcp_position):
    """Prevent motion farther than a radius from the start position."""

    displacement = [
        tcp_position[0] - start_position[0],
        tcp_position[1] - start_position[1],
        tcp_position[2] - start_position[2],
    ]

    distance = vec3_norm(displacement)

    if distance < max_displacement_from_start or distance < 1e-6:
        return

    outward_velocity = (
        cart_velocity[0] * displacement[0] +
        cart_velocity[1] * displacement[1] +
        cart_velocity[2] * displacement[2]
    ) / distance

    if outward_velocity <= 0.0:
        return

    cart_velocity[0] -= outward_velocity * displacement[0] / distance
    cart_velocity[1] -= outward_velocity * displacement[1] / distance
    cart_velocity[2] -= outward_velocity * displacement[2] / distance


def send_speed_command(rtde_control_interface):
    """Send Cartesian velocity command to UR5e with speedL."""

    speed_command = [
        cart_velocity[0],
        cart_velocity[1],
        cart_velocity[2],
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
    """Write CSV column names."""

    csv_writer.writerow([
        "time_s",
        "Fx_raw",
        "Fy_raw",
        "Fz_raw",
        "Fx_filt",
        "Fy_filt",
        "Fz_filt",
        "x",
        "y",
        "z",
        "vx_cmd",
        "vy_cmd",
        "vz_cmd",
        "speed_cmd",
        "B_desired",
        "B_applied",
        "send_robot_commands",
    ])


def write_csv_row(csv_writer, time_s, tcp_position):
    """Write one controller sample to CSV."""

    csv_writer.writerow([
        time_s,
        force_raw[0],
        force_raw[1],
        force_raw[2],
        force_filtered[0],
        force_filtered[1],
        force_filtered[2],
        tcp_position[0],
        tcp_position[1],
        tcp_position[2],
        cart_velocity[0],
        cart_velocity[1],
        cart_velocity[2],
        vec3_norm(cart_velocity),
        B_desired,
        B_applied,
        int(send_robot_commands),
    ])


def main():
    """Run the UR5e direct variable admittance loop."""

    global start_position

    if rtde_control is None or rtde_receive is None:
        print("Missing dependency: install with `pip install --user ur_rtde`")
        return

    signal.signal(signal.SIGINT, handle_stop_signal)

    print("Connecting to UR5e at", robot_ip)
    rtde_control_interface = rtde_control.RTDEControlInterface(robot_ip)
    rtde_receive_interface = rtde_receive.RTDEReceiveInterface(robot_ip)

    move_to_start_joint_configuration(rtde_control_interface)

    print("Keep the robot unloaded. Estimating force bias...")
    estimate_force_bias(rtde_receive_interface)

    start_position = read_tcp_position_xyz(rtde_receive_interface)

    print("Controller started. Press Ctrl+C to stop.")
    print("send_robot_commands =", send_robot_commands)

    with open(write_data_csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        write_csv_header(csv_writer)

        max_steps = int(max_runtime_seconds / dt)

        for step_idx in range(max_steps):
            if not keep_running:
                break

            period_start = rtde_control_interface.initPeriod()

            tcp_position = read_tcp_position_xyz(rtde_receive_interface)

            update_force_filter(rtde_receive_interface)
            integrate_direct_admittance()
            apply_workspace_limit(tcp_position)
            send_speed_command(rtde_control_interface)

            time_s = step_idx * dt
            write_csv_row(csv_writer, time_s, tcp_position)

            rtde_control_interface.waitPeriod(period_start)

    if send_robot_commands:
        rtde_control_interface.speedStop(speed_stop_deceleration)

    rtde_control_interface.stopScript()

    print("Stopped UR5e direct admittance controller.")
    print("CSV saved to:", write_data_csv_path)


if __name__ == "__main__":
    main()
