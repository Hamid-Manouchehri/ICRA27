"""
Kang-style variable admittance controller for planar UR5e pHRI.

Paper:
    G. Kang, H. S. Oh, J. K. Seo, U. Kim, H. R. Choi, "Variable
    Admittance Control of Robot Manipulators Based on Human Intention",
    IEEE/ASME Trans. Mechatronics 24(3), 2019, pp. 1023-1032.
    See doc/Variable_Admittance_Control_of_Robot_Manipulators_Based_on_
    Human_Intention.pdf

Inputs:
    UR5e base-frame TCP wrench from RTDE, using only [Fx, Fy] [N].
    UR5e base-frame TCP pose from RTDE, using only [x, y, z] [m].
    UR5e actual TCP velocity from RTDE [vx, vy, vz, wx, wy, wz], with
    linear components [m/s] and angular components [rad/s].

Outputs:
    Optional UR5e Cartesian velocity command [vx, vy, 0, 0, 0, 0].
    CSV log with force, TCP motion, the reference-variable estimate,
    the expected force, the guidance direction, and the full local and
    global damping/mass quantities.
    Optional Tobii gaze CSV sharing the same time_s origin, so trials
    stay comparable with the rest of this project's raw_data.
    Optional CoppeliaSim TCP marker through a separate process.

Controller (paper equation numbers in parentheses):
    Planar mass-damper admittance (3), reducing (1)-(2)'s general model
    to direct teaching (k=0, x0=v0=a0=0):

        m * x_ddot + c * x_dot = Fs

    1. Direct intention (7): the force magnitude sets an isotropic
       damping c(v) = Fstd / |v|, clamped to [c_min_n_s_m,
       c_max_n_s_m]. Small force means precise motion, so damping is
       high; large force means fast motion, so damping is low.

    2. Indirect intention (11)-(16): human intention is modeled as
       motion along a plane curve of constant curvature (a straight
       line is the zero-curvature case). The signed reference variable
       kappa*B_hat is estimated every decimated step from consecutive
       unit tangent vectors of the *commanded* velocity (Fig. 5(b)
       feeds the admittance model's own output back into the
       estimator) and low-pass filtered. In this planar controller it
       reduces to one signed scalar, because the unit binormal B_hat
       is always +-z for in-plane motion. See
       update_reference_variable() for the full derivation.

    3. Force expectation (17): the force needed to hold the estimated
       curve while maintaining the current speed,

           F_expected = m * |v|^2 / rho * N_hat + c * |v| * T_hat

       a centripetal term (vanishes at zero curvature) plus the
       direct-intention viscous term along the direction of motion.

    4. Force guidance (18)-(20): admittance is made anisotropic in the
       frame whose x axis is the (rate-limited) direction of
       F_expected: the aligned axis keeps the direct-intention (c, m),
       the perpendicular axis is stiffened to (beta*c, beta*m) with

           beta = 1 + 10 * |v|                                   (19)

       growing with speed. Unlike direct intention, this implementation
       blends beta continuously from 1 up to its full value over a
       speed range (indirect_min_speed_m_s to indirect_full_speed_m_s)
       instead of switching, and the guidance direction is rate-limited,
       so neither the resistance nor its axis can step.

Implementation notes (choices the paper leaves open):
    1. The unit tangent vector is taken from the admittance model's own
       commanded velocity cart_velocity_command_m_s, not the measured
       TCP velocity. Fig. 5(b) of the paper feeds "x_d_dot" (the
       admittance output) into both the Force Expectation and Force
       Guiding blocks, and a commanded velocity is inherently smoother
       than a measured one (it is itself the state of a first-order
       filter), which matters because (15) differentiates it.
    2. The reference-variable estimator runs decimated at
       estimate_frequency_hz rather than every control step. A 2 ms
       central difference of a direction vector amplifies noise before
       any filtering can help; this project's Chen-style controller
       (ur5e_chen_var_adm_2d.py) hit the identical problem with its
       curvature estimate and fixed it the same way. 125 Hz also
       happens to match the paper's own reported control rate (8 ms).
    3. Paper (16) low-pass filters with alpha weighting the *previous*
       value ("= alpha*(k-1) + (1-alpha)*(k)"), the opposite of every
       other filter in this project's controllers, which weight the
       *new* sample by alpha. reference_variable_lpf_alpha and
       expected_force_lpf_alpha here follow this project's convention
       instead (new sample weighted by alpha); the resulting filter
       behavior is identical, only the parameter's meaning is mirrored.
    4. The paper gives no numeric value for (16)'s weighting factor or
       for how quickly the expected force should be filtered. Defaults
       follow this project's other low-pass filters rather than the
       paper.
    5. Section IV has no fit-quality analog to gate on: the reference
       variable is a simple recursive filter, always defined, unlike
       Chen's windowed power-law fit. The only gate here is a speed
       ramp (indirect_weight), because the unit tangent is undefined at
       zero velocity.
    6. (18)-(20) are implemented by rotating the measured force and the
       commanded velocity into the guidance-aligned frame, integrating
       each axis independently with its own scalar (c, m), then
       rotating the result back, rather than forming and inverting the
       2x2 matrices directly. Because R is orthogonal and the local
       matrices are diagonal, this is exactly M_i^-1 (F - B_i v); see
       integrate_planar_admittance(). The global-frame B_i and M_i are
       still assembled, for logging only.
    7. Because beta multiplies *both* m and c in (18), it cancels out
       of the local discrete-stability ratio c_local/m_local for every
       value of beta -- see the stability check below. Guidance
       strength therefore cannot destabilize the integration by itself.
    8. Fs in (7) is this controller's own force_filtered_n (biased,
       deadbanded, low-pass filtered), matching how every other
       controller in this project conditions the raw sensor signal;
       the paper does not describe its own conditioning in this detail.
    9. (12)'s discrete B_hat (current tangent cross previous tangent)
       points opposite the standard continuous Frenet-Serret B for
       genuine CCW motion, so N_hat = B_hat x T_hat would point away
       from the estimated center of curvature rather than toward it.
       update_expected_force() corrects this with an explicit sign
       flip rather than silently inheriting a centripetal term that
       pushes the wrong way; see that function's docstring for the
       hand and numeric verification. reference_variable_filt_1_m
       itself is left exactly as (15)-(16) define it -- only where it
       is *used* to build a direction is the sign corrected.

Known behavior worth checking before trusting a trial:
    The guidance direction is defined even on a perfectly straight push
    (the tangential term of (17) alone gives it a direction), so
    indirect intention engages on straight-line motion too, once moving
    faster than indirect_min_speed_m_s -- not only on curves. This
    matches the paper's own theory (a straight line is the rho ->
    infinity case of a constant-curvature curve, Section IV-A) even
    though the paper's own experiments evaluate indirect intention only
    on curved and rectangular tracking paths (Section V). If "curves
    only" guidance is wanted instead, also gate indirect_weight on
    abs(reference_variable_filt_1_m).

Main tunable parameters:
    Virtual mass, the damping range [c_min_n_s_m, c_max_n_s_m],
    standard_force_n, whether indirect intention is enabled, and
    beta_speed_gain are all in User Settings, sorted most to least
    important. Check force signs, workspace, and the start pose before
    setting send_robot_commands = True.

Dependency:
    pip install --user ur_rtde tobii-research coppeliasim-zmqremoteapi-client

Assumptions:
    Motion is planar in the UR5e base-frame x-y plane at fixed z and
    orientation, matching this project's other planar controllers. The
    human pushes the tool directly, so the measured TCP wrench is the
    human force. Human intention is estimated purely from the commanded
    velocity and measured force; unlike ur5e_gaze_assisted_adm_2d.py
    this controller does not use gaze for control, though Tobii data is
    still recorded for cross-trial comparability (see
    ur5e_chen_var_adm_2d.py, which does the same).
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
from pathlib import Path

try:
    import rtde_control
    import rtde_receive
except ImportError:
    rtde_control = None
    rtde_receive = None


# -------------------- User Settings --------------------
robot_ip = "192.168.1.100"  # TODO UR5e IP/hostname [valid robot address]

# TODO bool; True sends physical speedL commands. Keep False until force
# signs, workspace corners, and the start pose have been checked.
send_robot_commands = True

# TODO controller output CSV path [file path]; resolved from this
# script, not from the current working directory. Rename per trial.
write_data_csv_path = (
    Path(__file__).resolve().parent.parent /
    "data" /
    "raw_data" /
    "ur5e_kang_var_adm_2d_log.csv"
)

# TODO Tobii output CSV path [file path]; one row per Tobii callback.
# Shares the time_s origin of the controller CSV, so the pair matches
# the <ur5e_*, tobii_*> convention used by this project's MATLAB scripts.
tobii_data_csv_path = (
    Path(__file__).resolve().parent.parent /
    "data" /
    "raw_data" /
    "tobii_kang_var_adm_2d_log.csv"
)

# --- Admittance model (3); direct intention (7) ---
# TODO virtual mass m [kg], recommended 1 to 20. Matches both the
# paper's own simulation-verification example and this project's other
# controllers.
virtual_mass_kg = 10.0

# TODO damping bounds [N s/m] for c(v) in (7), both positive with
# c_min < c_max. Paper's single-DOF verification uses [20, 60].
c_min_n_s_m = 20.0
c_max_n_s_m = 60.0

# TODO standard force Fstd [N] in (7), positive. The comfortable force
# level the operator sets by pushing harder or softer. Paper examples
# use 2 to 3 N.
standard_force_n = 2.5

# TODO speed floor [m/s] used only to avoid dividing by ~0 in (7).
# Matches the same guard in ur5e_var_adm_direct_3d.py.
direct_damping_speed_eps_m_s = 0.02

# TODO damping slew-rate limit [N s/m per s], recommended 100 to 600.
# Bounds how fast the isotropic damping may change so the operator does
# not feel a step in resistance.
damping_rate_max_n_s_m_per_s = 300.0

# --- Indirect intention / force guidance (11)-(20) ---
# TODO bool; the paper itself labels indirect intention "(Optional)" in
# its block diagram (Fig. 5(b)). False reduces this controller to
# direct intention only, matching ur5e_var_adm_direct_3d.py's scope.
enable_indirect_intention = True

# TODO speed gain in beta = 1 + beta_speed_gain*|v| (19), non-negative.
# Paper value is 10.
beta_speed_gain = 10.0

# TODO low-pass alpha for the reference variable (16) [0, 1]; smaller
# is smoother. Weights the *new* sample, per implementation note 3
# above. Not given a numeric value in the paper.
reference_variable_lpf_alpha = 0.10

# TODO rate at which the reference variable is estimated and the
# guidance geometry is refreshed [Hz], recommended 60 to 200, must be
# <= control_frequency_hz. Estimating from raw 2 ms differences is
# noisy even before filtering; see implementation note 2 above.
estimate_frequency_hz = 125.0

# TODO speed below which indirect intention is fully off [m/s],
# recommended 0.003 to 0.02. Below this the unit tangent used by (15)
# and (17) is undefined.
indirect_min_speed_m_s = 0.005

# TODO speed at which indirect intention reaches full strength [m/s],
# must exceed indirect_min_speed_m_s.
indirect_full_speed_m_s = 0.03

# TODO maximum rotation rate of the guidance direction [rad/s],
# positive. Keeps the damping/mass frame from snapping when the
# expected force passes near zero.
guidance_direction_max_rate_rad_s = 6.0

# TODO low-pass alpha for the expected force direction [0, 1]. Not
# specified by the paper; smooths (17) before it sets the guidance
# direction, the same role intended_force_lpf_alpha plays in
# ur5e_chen_var_adm_2d.py.
expected_force_lpf_alpha = 0.10

# --- Force conditioning ---
force_lpf_alpha = 0.2  # TODO filter alpha [0, 1]; smaller is smoother
force_deadband_n = 1.0  # TODO radial force deadband [N], 0 to 5
force_bias_samples = 250  # TODO unloaded samples [count], >= 100

# TODO set signs so a positive push produces positive base-frame motion.
force_sign_x = 1.0  # TODO x sign [unitless], valid values: -1 or +1
force_sign_y = 1.0  # TODO y sign [unitless], valid values: -1 or +1

# --- Motion limits and workspace ---
max_cart_speed_m_s = 0.15  # TODO planar speed limit [m/s], 0.01 to 0.15

# Axis-aligned rectangular workspace in the UR5e base-frame x-y plane.
# TODO lower [x, y] corner offsets from the startup TCP [m]. Each value
# must be smaller than the matching upper-corner value.
workspace_lower_corner_offset_m = [-0.2, -0.2]
# TODO upper [x, y] corner offsets from the startup TCP [m].
workspace_upper_corner_offset_m = [0.25, 0.25]

# --- Robot, timing, and startup ---
control_frequency_hz = 500.0  # TODO UR5e native control rate [Hz]
speedl_acceleration_m_s2 = 0.5  # TODO speedL acceleration [m/s^2]
speed_stop_deceleration_m_s2 = 2.0  # TODO stop deceleration [m/s^2]

max_runtime_s = 120.0  # TODO automatic stop time [s], must be positive
subject_preparation_time_s = 1.0  # TODO delay before recording [s], >= 0

move_to_start_configuration = True  # TODO bool; False skips startup move
start_joint_q_deg = [  # TODO joint angles [deg], within robot limits
    -77.64,
    -98.5,
    -95.80,
    -75.75,
    90.3,
    -2.75,
]
start_joint_speed_rad_s = 0.5  # TODO moveJ speed [rad/s], 0.1 to 1
start_joint_acceleration_rad_s2 = 0.5  # TODO moveJ accel [rad/s^2]

# --- Tobii recording ---
enable_tobii_recording = True  # TODO bool; requires a connected tracker
tobii_queue_size = 50000  # TODO callback buffer [samples], >= 10000

# --- CoppeliaSim marker bridge ---
# Runs in a separate process so ZeroMQ latency can never stall the
# control loop.
send_coppeliasim_trajectory = True  # TODO requires CoppeliaSim running
coppeliasim_host = "127.0.0.1"  # TODO ZeroMQ host [hostname or IPv4]
coppeliasim_port = 23000  # TODO ZeroMQ port [integer], default 23000
coppeliasim_object_path = "/Sphere"  # TODO exact marker path in scene
coppeliasim_reference_object_path = ""  # TODO empty means world frame
coppeliasim_update_frequency_hz = 25.0  # TODO marker update rate [Hz]

coppeliasim_position_offset_m = [0.0, 0.0, 0.0]  # TODO offset [m]
coppeliasim_position_signs = [1.0, 1.0, 1.0]  # TODO signs, each -1 or +1


# -------------------- Derived Settings --------------------
dt = 1.0 / control_frequency_hz
start_joint_q = [math.radians(angle_deg) for angle_deg in start_joint_q_deg]

# One estimator update every estimate_period_steps control cycles.
estimate_period_steps = max(1, int(round(
    control_frequency_hz / estimate_frequency_hz)))
estimate_dt = estimate_period_steps * dt


# -------------------- Setting Checks --------------------
# Only conditions that would silently produce wrong physics or unsafe
# motion are checked here, per this project's "no defensive scaffolding"
# rule.
if c_min_n_s_m <= 0.0 or c_max_n_s_m <= c_min_n_s_m:
    raise ValueError("Require 0 < c_min_n_s_m < c_max_n_s_m.")

if standard_force_n <= 0.0:
    raise ValueError("Require standard_force_n > 0.")

if virtual_mass_kg <= 0.0:
    raise ValueError("Require virtual_mass_kg > 0.")

if estimate_frequency_hz > control_frequency_hz:
    raise ValueError(
        "Require estimate_frequency_hz <= control_frequency_hz.")

if indirect_full_speed_m_s <= indirect_min_speed_m_s:
    raise ValueError(
        "Require indirect_full_speed_m_s > indirect_min_speed_m_s.")

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

# Explicit discrete-stability check for the local-frame admittance
# integration. The forward-Euler update v <- v + (F - c*v)/m * dt is
# stable only while c*dt/m < 2 on each local axis; the factor 2 margin
# keeps it well inside that bound. beta multiplies both c and m in
# (18), so it cancels out of this ratio for every axis and every value
# of beta -- only c_max_n_s_m and virtual_mass_kg matter here.
if c_max_n_s_m * dt / virtual_mass_kg > 1.0:
    raise ValueError(
        "Admittance integration is too stiff for this control period: "
        "reduce c_max_n_s_m, raise virtual_mass_kg, or raise "
        "control_frequency_hz. beta cancels out of this ratio, so "
        "guidance strength cannot fix instability here."
    )


# -------------------- Controller State --------------------
keep_running = True

force_bias_n = [0.0, 0.0]
force_raw_n = [0.0, 0.0]
force_filtered_n = [0.0, 0.0]

# Virtual admittance velocity state; this is what is commanded, and
# also what the indirect-intention estimator differentiates.
cart_velocity_command_m_s = [0.0, 0.0]

start_position_m = [0.0, 0.0]

c_desired_n_s_m = c_max_n_s_m
c_applied_n_s_m = c_max_n_s_m

# Reference-variable estimator state (11)-(16): the previous decimated
# unit tangent and speed, carried step to step. No window/history is
# needed because (16) is a simple recursive filter, not a batch fit.
tangent_prev_xy = [1.0, 0.0]
speed_prev_m_s = 0.0
have_tangent_history = False
reference_variable_raw_1_m = 0.0
reference_variable_filt_1_m = 0.0

# Force expectation (17) and guidance direction (18)-(20) state.
expected_force_n = [0.0, 0.0]
guidance_direction_angle_rad = 0.0
guidance_direction_xy = [1.0, 0.0]
indirect_weight = 0.0
beta_speed_raw = 1.0
beta_effective = 1.0

# Local (guidance-frame) damping and mass, (18): [along, across].
c_local_n_s_m = [c_max_n_s_m, c_max_n_s_m]
m_local_kg = [virtual_mass_kg, virtual_mass_kg]

# Row-major 2x2 global-frame matrices, (20), logged only.
damping_matrix_n_s_m = [
    c_max_n_s_m, 0.0,
    0.0, c_max_n_s_m,
]
mass_matrix_kg = [
    virtual_mass_kg, 0.0,
    0.0, virtual_mass_kg,
]


def handle_stop_signal(_signal_number, _frame):
    """Stop the control loop when Ctrl+C is pressed."""

    global keep_running
    keep_running = False


# -------------------- Small Math Helpers --------------------
def clamp_value(value, lower_limit, upper_limit):
    """Return a scalar limited to [lower_limit, upper_limit]."""

    if value < lower_limit:
        return lower_limit

    if value > upper_limit:
        return upper_limit

    return value


def vec2_norm(vector):
    """Return the Euclidean norm of a 2-D vector."""

    return math.sqrt(vector[0] * vector[0] + vector[1] * vector[1])


def vec2_cross(vector_a, vector_b):
    """Return the scalar cross product a_x*b_y - a_y*b_x."""

    return vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0]


def vec2_unit(vector, fallback_unit_vector):
    """Return the unit vector, or the fallback if too close to zero."""

    vector_norm = vec2_norm(vector)

    if vector_norm < 1e-9:
        return list(fallback_unit_vector)

    return [vector[0] / vector_norm, vector[1] / vector_norm]


def vec2_left_normal(vector):
    """Return the vector rotated by +90 deg in the x-y plane."""

    return [-vector[1], vector[0]]


def limit_vector_norm_2d(vector, max_norm):
    """Limit a 2-D vector magnitude while preserving its direction."""

    vector_norm = vec2_norm(vector)

    if vector_norm <= max_norm:
        return list(vector)

    scale = max_norm / vector_norm
    return [scale * vector[0], scale * vector[1]]


def linear_ramp(value, ramp_start, ramp_end):
    """Map value onto [0, 1], ramping linearly between the two bounds.

    Returns 0 at or below ramp_start and 1 at or above ramp_end. Used
    everywhere a continuous gate is needed instead of a hard threshold,
    so no controller quantity can step discontinuously.
    """

    if ramp_end <= ramp_start:
        return 1.0 if value >= ramp_end else 0.0

    return clamp_value(
        (value - ramp_start) / (ramp_end - ramp_start), 0.0, 1.0)


def wrap_angle_rad(angle_rad):
    """Wrap an angle into [-pi, pi]."""

    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


# -------------------- Robot I/O --------------------
def read_tcp_force_xy(rtde_receive_interface):
    """Read base-frame TCP force [Fx, Fy] in N from the UR5e.

    getActualTCPForce() already reports the wrench in the robot base
    frame, so no rotation by the end-effector orientation is applied
    here.
    """

    tcp_wrench_n = rtde_receive_interface.getActualTCPForce()

    return [
        force_sign_x * tcp_wrench_n[0],
        force_sign_y * tcp_wrench_n[1],
    ]


def read_tcp_position_xyz(rtde_receive_interface):
    """Read base-frame TCP position [x, y, z] in m from the UR5e."""

    tcp_pose = rtde_receive_interface.getActualTCPPose()

    return [tcp_pose[0], tcp_pose[1], tcp_pose[2]]


def read_tcp_actual_velocity(rtde_receive_interface):
    """Read the actual base-frame TCP velocity from the UR5e.

    Returns:
        [vx, vy, vz, wx, wy, wz] with linear components [m/s] and
        angular components [rad/s]. Logged for comparison against the
        commanded velocity only; the controller itself never reads it.
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


def estimate_force_bias(rtde_receive_interface):
    """Average unloaded startup force to estimate the planar bias [N]."""

    global force_bias_n

    force_bias_n = [0.0, 0.0]

    for _sample_idx in range(force_bias_samples):
        measured_force_n = read_tcp_force_xy(rtde_receive_interface)

        force_bias_n[0] += measured_force_n[0]
        force_bias_n[1] += measured_force_n[1]

        time.sleep(dt)

    force_bias_n[0] /= force_bias_samples
    force_bias_n[1] /= force_bias_samples


def update_force_filter(rtde_receive_interface):
    """Bias, radially deadband, then low-pass the planar force [N].

    The deadband acts on the planar magnitude rather than each axis
    independently:

        f_out = f_raw * max(0, |f_raw| - deadband) / |f_raw|

    This shrinks the vector toward zero without changing its direction
    and is continuous at the threshold, unlike a per-axis hard cutoff.
    """

    measured_force_n = read_tcp_force_xy(rtde_receive_interface)

    force_raw_n[0] = measured_force_n[0] - force_bias_n[0]
    force_raw_n[1] = measured_force_n[1] - force_bias_n[1]

    raw_magnitude_n = vec2_norm(force_raw_n)

    if raw_magnitude_n <= force_deadband_n:
        deadbanded_force_n = [0.0, 0.0]
    else:
        shrink_scale = (
            raw_magnitude_n - force_deadband_n) / raw_magnitude_n
        deadbanded_force_n = [
            shrink_scale * force_raw_n[0],
            shrink_scale * force_raw_n[1],
        ]

    for axis in range(2):
        force_filtered_n[axis] = (
            (1.0 - force_lpf_alpha) * force_filtered_n[axis] +
            force_lpf_alpha * deadbanded_force_n[axis]
        )


# -------------------- Direct Intention (paper 3, 7) ------------------
def compute_direct_damping(speed_m_s):
    """Direct intention, paper (7): c(v) = Fstd / |v|, clamped.

    Inputs:
        speed_m_s: current commanded planar speed [m/s].

    Returns:
        Isotropic virtual damping [N s/m] in [c_min_n_s_m, c_max_n_s_m].
    """

    safe_speed_m_s = max(speed_m_s, direct_damping_speed_eps_m_s)
    damping_n_s_m = standard_force_n / safe_speed_m_s

    return clamp_value(damping_n_s_m, c_min_n_s_m, c_max_n_s_m)


def rate_limit_damping(desired_n_s_m, previous_n_s_m):
    """Limit the per-step change of the isotropic damping [N s/m]."""

    max_step_n_s_m = damping_rate_max_n_s_m_per_s * dt
    damping_step = clamp_value(
        desired_n_s_m - previous_n_s_m, -max_step_n_s_m, max_step_n_s_m)

    return previous_n_s_m + damping_step


def update_direct_damping():
    """Update the isotropic direct-intention damping, paper (7).

    Outputs (module state):
        c_desired_n_s_m, c_applied_n_s_m.
    """

    global c_desired_n_s_m, c_applied_n_s_m

    speed_m_s = vec2_norm(cart_velocity_command_m_s)

    c_desired_n_s_m = compute_direct_damping(speed_m_s)
    c_applied_n_s_m = rate_limit_damping(c_desired_n_s_m, c_applied_n_s_m)


# -------------------- Indirect Intention: Reference Variable ---------
# -------------------- (paper 11-16) -----------------------------------
def update_reference_variable(step_idx):
    """Estimate the signed planar reference variable kappa*Bz, (11)-(16).

    For a curve confined to a plane, the unit binormal B_hat is always
    perpendicular to that plane, i.e. always +-z here. The general 3-D
    reference variable (1/rho)*B_hat therefore collapses to one signed
    scalar: its magnitude is the curvature 1/rho and its sign is the
    turn direction (CCW/CW). Substituting T_hat = v_hat (paper: "the
    unit tangent vector is equal to the unit velocity vector") into
    (12)-(15) and keeping only the z-component gives, for two
    consecutive decimated unit tangents,

        kappa*Bz(k) = (T_hat(k) x T_hat(k-1)) / (|v(k-1)| * estimate_dt)

    a plain 2-D scalar cross product, i.e. exactly (15) written out in
    the plane. This is then low-pass filtered per (16), with the alpha
    convention flipped as noted in implementation note 3 above.

    Runs only every estimate_period_steps control cycles: a 2 ms
    central difference of a direction vector is dominated by noise
    before any filtering can help, the same problem
    ur5e_chen_var_adm_2d.py's curvature estimate had.

    Outputs (module state):
        reference_variable_raw_1_m, reference_variable_filt_1_m,
        tangent_prev_xy, speed_prev_m_s, have_tangent_history.
    """

    global tangent_prev_xy, speed_prev_m_s, have_tangent_history
    global reference_variable_raw_1_m, reference_variable_filt_1_m

    if step_idx % estimate_period_steps != 0:
        return

    tangent_now_xy = vec2_unit(cart_velocity_command_m_s, tangent_prev_xy)
    speed_now_m_s = vec2_norm(cart_velocity_command_m_s)

    if have_tangent_history and speed_prev_m_s >= indirect_min_speed_m_s:
        reference_variable_raw_1_m = vec2_cross(
            tangent_now_xy, tangent_prev_xy) / (
                speed_prev_m_s * estimate_dt)
    else:
        # No usable previous tangent, or the robot was too slow for it
        # to carry a reliable direction: no turn evidence this step.
        reference_variable_raw_1_m = 0.0

    reference_variable_filt_1_m = (
        (1.0 - reference_variable_lpf_alpha) * reference_variable_filt_1_m +
        reference_variable_lpf_alpha * reference_variable_raw_1_m
    )

    tangent_prev_xy = tangent_now_xy
    speed_prev_m_s = speed_now_m_s
    have_tangent_history = True


# -------------------- Force Expectation (paper 17) --------------------
def update_expected_force():
    """Expected force to hold the estimated curve, paper (17):

        F_expected = (m * |v|^2 / rho) * N_hat + c * |v| * T_hat

    In the planar reduction the centripetal term becomes
    -m*|v|^2*reference_variable_filt_1_m*left_normal(T_hat) (note the
    minus sign, explained below), and no separate rho/N_hat extraction
    is needed since the sign of the reference variable already carries
    the turn direction -- see update_reference_variable(). At zero
    curvature (straight line, the paper's rho -> infinity case) this
    term vanishes and F_expected reduces to the tangential term alone,
    i.e. guidance aligns with the current direction of motion.

    On the minus sign: (12) defines B_hat from T_hat(k) x T_hat(k-1),
    current cross previous. For genuine CCW planar motion this points
    the *opposite* way from the standard continuous Frenet-Serret B
    (dT/ds = kappa*N with kappa > 0, B = T x N), because a discrete
    "new cross old" is the negative of the "old cross new" order a
    forward difference would suggest -- verified by hand (prev=(1,0),
    now=(cos d, sin d) for small d>0 gives cross = -sin(d) < 0) and
    numerically in scratch verification. The textbook identity
    N_hat = B_hat x T_hat therefore points *away* from the estimated
    center of curvature if B_hat is taken directly from (12). Since
    the entire point of this term is to guide the human toward that
    center, the sign is corrected here rather than propagating (12)'s
    convention into a centripetal force that pushes the wrong way.

    Outputs (module state):
        expected_force_n, low-pass filtered.
    """

    speed_m_s = vec2_norm(cart_velocity_command_m_s)

    if speed_m_s < indirect_min_speed_m_s:
        expected_force_raw_n = [0.0, 0.0]
    else:
        tangent_xy = vec2_unit(
            cart_velocity_command_m_s, guidance_direction_xy)
        normal_xy = vec2_left_normal(tangent_xy)

        centripetal_n = (
            -virtual_mass_kg * speed_m_s * speed_m_s *
            reference_variable_filt_1_m
        )
        tangential_n = c_applied_n_s_m * speed_m_s

        expected_force_raw_n = [
            centripetal_n * normal_xy[0] + tangential_n * tangent_xy[0],
            centripetal_n * normal_xy[1] + tangential_n * tangent_xy[1],
        ]

    for axis in range(2):
        expected_force_n[axis] = (
            (1.0 - expected_force_lpf_alpha) * expected_force_n[axis] +
            expected_force_lpf_alpha * expected_force_raw_n[axis]
        )


# -------------------- Force Guidance (paper 18-20) --------------------
def update_indirect_weight():
    """Update the continuous speed gate on indirect intention.

    The paper switches indirect intention on and off between free and
    tracking motion (Fig. 5(b), Section V). This implementation ramps
    it continuously with speed instead, so beta cannot step, and
    disables it entirely when enable_indirect_intention is False.

    Outputs (module state):
        indirect_weight in [0, 1].
    """

    global indirect_weight

    if not enable_indirect_intention:
        indirect_weight = 0.0
        return

    speed_m_s = vec2_norm(cart_velocity_command_m_s)
    indirect_weight = linear_ramp(
        speed_m_s, indirect_min_speed_m_s, indirect_full_speed_m_s)


def update_guidance_direction():
    """Update the rate-limited direction of the guidance frame's x axis.

    Target direction: the expected force while indirect intention is
    active and well-defined, otherwise the measured force, otherwise
    hold the current angle -- the same fallback chain
    ur5e_chen_var_adm_2d.py uses for its intended direction, and for
    the same reason: the guidance frame must never be undefined.

    Outputs (module state):
        guidance_direction_angle_rad, guidance_direction_xy.
    """

    global guidance_direction_angle_rad, guidance_direction_xy

    if vec2_norm(expected_force_n) > 1e-9 and indirect_weight > 0.0:
        target_angle_rad = math.atan2(
            expected_force_n[1], expected_force_n[0])
    elif vec2_norm(force_filtered_n) > 1e-9:
        target_angle_rad = math.atan2(
            force_filtered_n[1], force_filtered_n[0])
    else:
        target_angle_rad = guidance_direction_angle_rad

    angle_error_rad = wrap_angle_rad(
        target_angle_rad - guidance_direction_angle_rad)
    max_angle_step_rad = guidance_direction_max_rate_rad_s * dt
    angle_step_rad = clamp_value(
        angle_error_rad, -max_angle_step_rad, max_angle_step_rad)

    guidance_direction_angle_rad = wrap_angle_rad(
        guidance_direction_angle_rad + angle_step_rad)
    guidance_direction_xy = [
        math.cos(guidance_direction_angle_rad),
        math.sin(guidance_direction_angle_rad),
    ]


def integrate_planar_admittance():
    """Integrate the force-guided admittance model, paper (18)-(20).

    In the frame whose x axis is guidance_direction_xy,

        c_local = [c_applied_n_s_m, beta_effective * c_applied_n_s_m]
        m_local = [virtual_mass_kg, beta_effective * virtual_mass_kg]

    Rather than assembling Bd = R*B_local*R^T and Mi = R*M_local*R^T
    and inverting Mi, the measured force and the commanded velocity are
    rotated into this local frame, each axis is integrated
    independently with its own scalar (c, m), and the result is rotated
    back. Because R is orthogonal (R^-1 = R^T) and B_local/M_local are
    diagonal,

        Mi^-1 (F_global - Bd*v_global) = R * M_local^-1 (F_local -
                                                           B_local*v_local)

    so this is exactly the paper's model, just without ever forming or
    inverting a 2x2 matrix. beta_effective blends continuously from 1
    (isotropic, indirect_weight = 0) up to beta_speed_raw (19) as
    indirect_weight ramps to 1, so guidance fades in and out instead of
    switching.

    Outputs (module state):
        cart_velocity_command_m_s, beta_speed_raw, beta_effective,
        c_local_n_s_m, m_local_kg, damping_matrix_n_s_m (logged only),
        mass_matrix_kg (logged only).
    """

    global cart_velocity_command_m_s
    global beta_speed_raw, beta_effective

    speed_m_s = vec2_norm(cart_velocity_command_m_s)
    beta_speed_raw = 1.0 + beta_speed_gain * speed_m_s
    beta_effective = 1.0 + indirect_weight * (beta_speed_raw - 1.0)

    cosine = guidance_direction_xy[0]
    sine = guidance_direction_xy[1]

    c_local_n_s_m[0] = c_applied_n_s_m
    c_local_n_s_m[1] = beta_effective * c_applied_n_s_m
    m_local_kg[0] = virtual_mass_kg
    m_local_kg[1] = beta_effective * virtual_mass_kg

    # Global-to-local rotation is R^T.
    force_local_n = [
        cosine * force_filtered_n[0] + sine * force_filtered_n[1],
        -sine * force_filtered_n[0] + cosine * force_filtered_n[1],
    ]
    velocity_local_m_s = [
        cosine * cart_velocity_command_m_s[0] +
        sine * cart_velocity_command_m_s[1],
        -sine * cart_velocity_command_m_s[0] +
        cosine * cart_velocity_command_m_s[1],
    ]

    acceleration_local_m_s2 = [
        (force_local_n[0] - c_local_n_s_m[0] * velocity_local_m_s[0]) /
        m_local_kg[0],
        (force_local_n[1] - c_local_n_s_m[1] * velocity_local_m_s[1]) /
        m_local_kg[1],
    ]

    velocity_local_new_m_s = [
        velocity_local_m_s[0] + acceleration_local_m_s2[0] * dt,
        velocity_local_m_s[1] + acceleration_local_m_s2[1] * dt,
    ]

    # Local-to-global rotation is R.
    cart_velocity_command_m_s = [
        cosine * velocity_local_new_m_s[0] -
        sine * velocity_local_new_m_s[1],
        sine * velocity_local_new_m_s[0] +
        cosine * velocity_local_new_m_s[1],
    ]

    # Clamping the state itself, not just the output, keeps the virtual
    # model from winding up while the limit is active.
    cart_velocity_command_m_s = limit_vector_norm_2d(
        cart_velocity_command_m_s, max_cart_speed_m_s)

    # Global-frame Bd = R*B_local*R^T and Mi = R*M_local*R^T, logged
    # only; the integration above never forms them directly.
    damping_matrix_n_s_m[0] = (
        c_local_n_s_m[0] * cosine * cosine +
        c_local_n_s_m[1] * sine * sine
    )
    damping_matrix_n_s_m[1] = (
        c_local_n_s_m[0] - c_local_n_s_m[1]) * cosine * sine
    damping_matrix_n_s_m[2] = damping_matrix_n_s_m[1]
    damping_matrix_n_s_m[3] = (
        c_local_n_s_m[0] * sine * sine +
        c_local_n_s_m[1] * cosine * cosine
    )

    mass_matrix_kg[0] = (
        m_local_kg[0] * cosine * cosine + m_local_kg[1] * sine * sine)
    mass_matrix_kg[1] = (m_local_kg[0] - m_local_kg[1]) * cosine * sine
    mass_matrix_kg[2] = mass_matrix_kg[1]
    mass_matrix_kg[3] = (
        m_local_kg[0] * sine * sine + m_local_kg[1] * cosine * cosine)


def apply_workspace_limit(tcp_position_xy_m):
    """Remove velocity components pointing outside the x-y rectangle.

    Inputs:
        tcp_position_xy_m: measured [x, y] TCP position [m] in the UR5e
            base frame.

    The rectangle corners are the user-defined offsets from
    start_position_m. Inward and edge-tangential motion stay available.
    """

    workspace_lower_corner_m = [
        start_position_m[0] + workspace_lower_corner_offset_m[0],
        start_position_m[1] + workspace_lower_corner_offset_m[1],
    ]
    workspace_upper_corner_m = [
        start_position_m[0] + workspace_upper_corner_offset_m[0],
        start_position_m[1] + workspace_upper_corner_offset_m[1],
    ]

    for axis in range(2):
        at_lower_limit = (
            tcp_position_xy_m[axis] <= workspace_lower_corner_m[axis])
        at_upper_limit = (
            tcp_position_xy_m[axis] >= workspace_upper_corner_m[axis])

        if at_lower_limit and cart_velocity_command_m_s[axis] < 0.0:
            cart_velocity_command_m_s[axis] = 0.0

        if at_upper_limit and cart_velocity_command_m_s[axis] > 0.0:
            cart_velocity_command_m_s[axis] = 0.0


def send_speed_command(rtde_control_interface):
    """Send the planar TCP velocity [m/s] through speedL()."""

    speed_command = [
        cart_velocity_command_m_s[0],
        cart_velocity_command_m_s[1],
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    if send_robot_commands:
        rtde_control_interface.speedL(
            speed_command, speedl_acceleration_m_s2, dt)


def move_to_start_joint_configuration(rtde_control_interface):
    """Move the UR5e to the user-defined start joint configuration."""

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
        start_joint_speed_rad_s,
        start_joint_acceleration_rad_s2,
    )


# -------------------- CoppeliaSim Bridge --------------------
def map_tcp_position_to_coppeliasim(tcp_position_xyz_m):
    """Map UR5e base-frame position [m] to the CoppeliaSim frame [m]."""

    return [
        coppeliasim_position_offset_m[0] +
        coppeliasim_position_signs[0] * tcp_position_xyz_m[0],
        coppeliasim_position_offset_m[1] +
        coppeliasim_position_signs[1] * tcp_position_xyz_m[1],
        coppeliasim_position_offset_m[2] +
        coppeliasim_position_signs[2] * tcp_position_xyz_m[2],
    ]


def coppeliasim_trajectory_process(trajectory_queue, stop_event):
    """Send the latest TCP position to CoppeliaSim from a child process.

    Input:
        trajectory_queue: multiprocessing queue holding
            (controller_time_s, tcp_position_xyz_m) samples.
        stop_event: multiprocessing event set by the robot process.

    Output:
        The configured CoppeliaSim object receives setObjectPosition()
        calls.

    Assumptions:
        CoppeliaSim is already running with its ZeroMQ Remote API
        enabled. Simulator errors terminate only this child process, so
        the control loop is never blocked by simulator latency.
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
        client = RemoteAPIClient(
            host=coppeliasim_host, port=coppeliasim_port)
        sim = client.require("sim")
        target_object_handle = sim.getObject(coppeliasim_object_path)

        reference_object_handle = -1
        if coppeliasim_reference_object_path:
            reference_object_handle = sim.getObject(
                coppeliasim_reference_object_path)

        update_period_s = 1.0 / coppeliasim_update_frequency_hz
        next_update_time_s = 0.0

        print(
            "CoppeliaSim bridge connected; updating",
            coppeliasim_object_path,
            "at",
            coppeliasim_update_frequency_hz,
            "Hz",
            flush=True,
        )

        while not stop_event.is_set():
            try:
                trajectory_sample = trajectory_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Drop stale samples so simulator lag cannot build a backlog.
            while True:
                try:
                    trajectory_sample = trajectory_queue.get_nowait()
                except queue.Empty:
                    break

            current_time_s = time.monotonic()
            if current_time_s < next_update_time_s:
                continue

            coppeliasim_position_m = map_tcp_position_to_coppeliasim(
                trajectory_sample[1])

            sim.setObjectPosition(
                target_object_handle,
                reference_object_handle,
                coppeliasim_position_m,
            )

            next_update_time_s = current_time_s + update_period_s

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

    return trajectory_queue, stop_event, trajectory_process


def publish_coppeliasim_trajectory_sample(
    trajectory_queue,
    time_s,
    tcp_position_xyz_m,
):
    """Enqueue only the newest TCP position without blocking control."""

    if trajectory_queue is None:
        return

    try:
        while True:
            trajectory_queue.get_nowait()
    except queue.Empty:
        pass

    try:
        trajectory_queue.put_nowait((time_s, list(tcp_position_xyz_m)))
    except queue.Full:
        # The child is draining the queue; a later sample will retry.
        pass


def stop_coppeliasim_trajectory_process(
    trajectory_queue,
    stop_event,
    trajectory_process,
):
    """Stop and release the CoppeliaSim trajectory process."""

    if trajectory_process is None:
        return

    stop_event.set()
    trajectory_process.join(timeout=2.0)

    if trajectory_process.is_alive():
        trajectory_process.terminate()
        trajectory_process.join()

    trajectory_queue.close()


# -------------------- Tobii Recording --------------------
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
            make_tobii_json_safe(value), separators=(",", ":"))

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
            csv_row.append(json.dumps(
                make_tobii_json_safe(gaze_sample), separators=(",", ":")))
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
        time_s = (host_monotonic_time_ns - trial_start_time_ns) * 1e-9

        # Re-checked so a callback overlapping shutdown cannot append a
        # post-stop sample.
        if not recording_active_event.is_set():
            return

        try:
            gaze_queue.put_nowait(
                (time_s, host_monotonic_time_ns, gaze_sample))
        except queue.Full:
            # Never block the Tobii SDK callback thread.
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
        gaze_queue, recording_active_event, trial_clock)

    writer_thread.start()
    tracker.subscribe_to(
        tobii_research_api.EYETRACKER_GAZE_DATA,
        gaze_callback,
        as_dictionary=True,
    )

    print("Tobii model:", tracker.model)
    print("Tobii serial:", tracker.serial_number)
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
            tobii_research_api.EYETRACKER_GAZE_DATA, gaze_callback)
    finally:
        writer_stop_event.set()
        writer_thread.join(timeout=2.0)


# -------------------- Logging --------------------
def write_csv_header(csv_writer):
    """Write names and units for the controller CSV columns."""

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
        "vx_cmd_m_s",
        "vy_cmd_m_s",
        "speed_cmd_m_s",
        "c_desired_Ns_m",
        "c_applied_Ns_m",
        "reference_variable_raw_1_m",
        "reference_variable_filt_1_m",
        "beta_speed_raw",
        "indirect_weight",
        "beta_effective",
        "expected_force_x_N",
        "expected_force_y_N",
        "guidance_dir_x",
        "guidance_dir_y",
        "c_local_along_Ns_m",
        "c_local_across_Ns_m",
        "m_local_along_kg",
        "m_local_across_kg",
        "B11_Ns_m",
        "B12_Ns_m",
        "B21_Ns_m",
        "B22_Ns_m",
        "M11_kg",
        "M12_kg",
        "M21_kg",
        "M22_kg",
        "send_robot_commands",
    ])


def write_csv_row(
    csv_writer,
    time_s,
    host_monotonic_time_ns,
    tcp_position_xyz_m,
    tcp_actual_velocity,
):
    """Write one controller sample with measured TCP state to CSV."""

    coppeliasim_position_m = map_tcp_position_to_coppeliasim(
        tcp_position_xyz_m)

    csv_writer.writerow([
        time_s,
        host_monotonic_time_ns,
        force_raw_n[0],
        force_raw_n[1],
        force_filtered_n[0],
        force_filtered_n[1],
        tcp_position_xyz_m[0],
        tcp_position_xyz_m[1],
        tcp_position_xyz_m[2],
        coppeliasim_position_m[0],
        coppeliasim_position_m[1],
        coppeliasim_position_m[2],
        tcp_actual_velocity[0],
        tcp_actual_velocity[1],
        cart_velocity_command_m_s[0],
        cart_velocity_command_m_s[1],
        vec2_norm(cart_velocity_command_m_s),
        c_desired_n_s_m,
        c_applied_n_s_m,
        reference_variable_raw_1_m,
        reference_variable_filt_1_m,
        beta_speed_raw,
        indirect_weight,
        beta_effective,
        expected_force_n[0],
        expected_force_n[1],
        guidance_direction_xy[0],
        guidance_direction_xy[1],
        c_local_n_s_m[0],
        c_local_n_s_m[1],
        m_local_kg[0],
        m_local_kg[1],
        damping_matrix_n_s_m[0],
        damping_matrix_n_s_m[1],
        damping_matrix_n_s_m[2],
        damping_matrix_n_s_m[3],
        mass_matrix_kg[0],
        mass_matrix_kg[1],
        mass_matrix_kg[2],
        mass_matrix_kg[3],
        int(send_robot_commands),
    ])


# -------------------- Main --------------------
def main():
    """Run the UR5e Kang-style planar variable admittance loop."""

    global start_position_m

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

    tcp_position_xyz_m = read_tcp_position_xyz(rtde_receive_interface)
    start_position_m = [tcp_position_xyz_m[0], tcp_position_xyz_m[1]]

    print(
        "Subject preparation: control and recording start in",
        subject_preparation_time_s,
        "seconds.",
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
        recording_active_event, trial_clock)

    try:
        with open(write_data_csv_path, "w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            write_csv_header(csv_writer)

            max_steps = int(max_runtime_s / dt)
            trial_start_time_ns = time.perf_counter_ns()
            trial_clock["start_time_ns"] = trial_start_time_ns
            recording_active_event.set()

            print("Controller and synchronized recording started.")
            print("Press Ctrl+C to stop.")
            print("send_robot_commands =", send_robot_commands)
            print("virtual_mass_kg =", virtual_mass_kg)
            print(
                "damping range [N s/m] =", c_min_n_s_m, c_max_n_s_m)
            print("standard_force_n =", standard_force_n)
            print("enable_indirect_intention =", enable_indirect_intention)
            print(
                "indirect-intention estimator:",
                control_frequency_hz / estimate_period_steps,
                "Hz, beta_speed_gain =", beta_speed_gain,
            )

            for step_idx in range(max_steps):
                if not keep_running:
                    break

                period_start = rtde_control_interface.initPeriod()
                host_monotonic_time_ns = time.perf_counter_ns()
                time_s = (
                    host_monotonic_time_ns - trial_start_time_ns) * 1e-9

                tcp_position_xyz_m = read_tcp_position_xyz(
                    rtde_receive_interface)
                tcp_actual_velocity = read_tcp_actual_velocity(
                    rtde_receive_interface)
                tcp_position_xy_m = [
                    tcp_position_xyz_m[0],
                    tcp_position_xyz_m[1],
                ]

                update_force_filter(rtde_receive_interface)

                # Paper data flow: direct intention (7) sets c first,
                # since both the tangential term of (17) and the local
                # along-axis damping of (18) reuse it; the reference
                # variable (11)-(16) is decimated; the expected force
                # (17) and guidance direction then run every control
                # step from the latest filtered reference variable;
                # (18)-(20) finally integrate the admittance model in
                # the guidance-aligned local frame.
                update_direct_damping()
                update_reference_variable(step_idx)
                update_expected_force()
                update_indirect_weight()
                update_guidance_direction()
                integrate_planar_admittance()

                apply_workspace_limit(tcp_position_xy_m)
                send_speed_command(rtde_control_interface)

                write_csv_row(
                    csv_writer,
                    time_s,
                    host_monotonic_time_ns,
                    tcp_position_xyz_m,
                    tcp_actual_velocity,
                )

                # Queue-only: no CoppeliaSim API call happens here.
                publish_coppeliasim_trajectory_sample(
                    trajectory_queue, time_s, tcp_position_xyz_m)

                rtde_control_interface.waitPeriod(period_start)
    finally:
        recording_active_event.clear()
        stop_tobii_recording(tobii_recording)

        stop_coppeliasim_trajectory_process(
            trajectory_queue, simulator_stop_event, trajectory_process)

        if send_robot_commands:
            rtde_control_interface.speedStop(
                speed_stop_deceleration_m_s2)

        rtde_control_interface.stopScript()

    print("Stopped UR5e Kang-style variable admittance controller.")
    print("CSV saved to:", write_data_csv_path)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
