"""
Ying-style fuzzy variable admittance controller for planar UR5e pHRI.

Paper:
    K. Ying, C. Wang, C.-Y. Chen, X. Pan, L. Chen, "A Human Intention
    Based Fuzzy Variable Admittance Control System for Physical
    Human-Robot Interaction", 2022 IEEE/ASME Int. Conf. on Advanced
    Intelligent Mechatronics (AIM), pp. 202-207.
    See doc/A_Human_Intention_Based_Fuzzy_Variable_Admittance_Control_
    System_for_Physical_HumanRobot_Interaction.pdf

Inputs:
    UR5e base-frame TCP wrench from RTDE, using only [Fx, Fy] [N].
    UR5e base-frame TCP pose from RTDE, using only [x, y, z] [m].
    UR5e actual TCP velocity from RTDE [vx, vy, vz, wx, wy, wz], with
    linear components [m/s] and angular components [rad/s].

Outputs:
    Optional UR5e Cartesian velocity command [vx, vy, 0, 0, 0, 0].
    CSV log with force, TCP motion, the interaction power, both axes'
    force-alignment cosines, and the per-axis fuzzy damping.
    Optional Tobii gaze CSV sharing the same time_s origin, so trials
    stay comparable with the rest of this project's raw_data.
    Optional CoppeliaSim TCP marker through a separate process.

Controller (paper equation/table numbers in parentheses):
    Planar mass-damper admittance (3), the same direct-teaching
    reduction of (2) used throughout this project (k=0, x0=v0=a0=0):

        Md * x_ddot + Dd * x_dot = Fh

    with Dd fuzzy-inferred independently per axis (x, y); Md is held
    constant (the paper varies only Dd, citing prior work that damping
    matters more than mass for human perception in pHRI).

    1. Direct intention: the scalar interaction power

           P = Fh . V

       the dot product of the filtered force and the *measured*
       (low-pass filtered) TCP velocity -- matching Fig. 1's block
       diagram, which feeds the robot's actual V back into the fuzzy
       rule system, not the admittance model's own commanded output
       (contrast ur5e_kang_var_adm_2d.py, whose paper's own diagram
       feeds back the commanded value instead). P > 0 means the human
       is injecting kinetic energy (accelerating along the push);
       P < 0 means dissipating it (decelerating/resisting).

    2. Indirect intention: for each axis i in {x, y},

           |cos(theta_i)| = |Fi| / |F|                             (1)

       the alignment of that axis with the *current* force direction:
       1 means the force points straight along axis i, 0 means it is
       perpendicular to it. The paper's (1) sums over 3 axes (x, y,
       z); here |F| uses only the planar (x, y) magnitude, matching
       this project's exclusively-planar setup -- see implementation
       note 3.

    3. Fuzzy rule (Rule A, Table I / Fig. 2): P and |cos(theta_i)| are
       looked up in the paper's published 9 (P) x 5 (|cos|) table of
       output fuzzy-set indices (1-9), independently for each axis
       (shared P, axis-specific |cos(theta_i)|), then mapped onto
       [damping_min_n_s_m, damping_movement_max_n_s_m] = [5, 150]
       N s/m. See RULE_A_TABLE and evaluate_fuzzy_rule_a() below for
       how this implementation evaluates the table.

    Unlike ur5e_chen_var_adm_2d.py and ur5e_kang_var_adm_2d.py, there
    is no rotated damping frame anywhere in this method: (1) is
    defined per *base-frame coordinate axis*, so Dx and Dy integrate
    independently in the base frame with no rotation matrix at all --
    the simplest of the three controllers' admittance integrations.

Implementation notes (choices the paper leaves open):
    1. Full Mamdani inference (explicit triangular membership functions
       for P, |cos(theta_i)|, and Dd, min-max composition, centroid
       defuzzification) is not reproduced, because the paper gives the
       *rule table* (Table I) but not the membership function shapes
       or widths for any of the three variables -- reconstructing
       those would mean guessing details the paper never states.
       Instead, Table I is used directly: the output fuzzy-set index
       is found by *bilinear interpolation* of the published 9x5 table
       at the query (P, |cos(theta_i)|) point, then mapped linearly to
       N s/m. This reproduces the table exactly at its own grid points
       (P integer, |cos(theta_i)| in {0, 0.25, 0.5, 0.75, 1.0}), varies
       smoothly between them, and matches the smooth "fuzzy surface"
       the paper itself plots from this same table (Fig. 2) -- which
       is itself evidence the authors' own surface came from some form
       of interpolation/defuzzification swept across a fine grid, not
       a blocky nearest-rule lookup.
    2. "The range of damping is set as [5, 200] ... minimum damping in
       the movement is 5 N s/m and maximum damping in the movement is
       150 N s/m, and the 200 N s/m is used to prevent deviant
       [behavior]" is read as: the 9 output fuzzy-set indices span
       [5, 150] evenly (used by the interpolation above), and 200 is a
       separate, outer safety clamp (damping_safety_max_n_s_m) applied
       after that mapping, not one of the 9 MF centers itself. This
       reconciles both sentences without inventing a non-uniform MF
       spacing.
    3. |F| in (1) is the *planar* force magnitude hypot(Fx, Fy), not
       the paper's 3D sum over i = 1..3, matching this project's
       exclusively-planar setup (no controller in this project ever
       reads or controls Fz).
    4. P's calibration range ("changes in the range of [-4, 4] in
       normal interaction") is specific to the paper's Franka Panda
       setup and human subjects. It is exposed here as power_range_w,
       which defaults to a UR5e-scale guess, not [-4, 4] -- the 9 P
       columns of Table I are always treated as evenly spanning
       [-power_range_w, +power_range_w], so recalibrating
       power_range_w reshapes the effective P axis of the lookup
       proportionally. Recalibrate from a real trial's
       Fx*vx + Fy*vy before trusting the fuzzy surface's shape at this
       project's actual force/speed ranges.
    5. Md (virtual mass) is not given an explicit value in the paper
       text; virtual_mass_kg defaults to this project's standard
       10 kg (see ur5e_chen_var_adm_2d.py, ur5e_kang_var_adm_2d.py).
    6. Neither a force-direction fallback nor damping rate-limiting is
       described by the paper. Below minimum_force_for_direction_n,
       |cos(theta_i)| is undefined (0/0), so both axes default
       directly to damping_movement_max_n_s_m (precise/stiff at rest,
       matching every other controller in this project's low-force
       behavior) rather than evaluating the table at an arbitrary
       fallback angle. Both Dx and Dy are also slew-rate limited
       (damping_rate_max_n_s_m_per_s), the same hardening
       ur5e_chen_var_adm_2d.py and ur5e_kang_var_adm_2d.py apply for
       the same reason: an unlimited step in Dd is felt as a jolt.

Known behavior worth checking before trusting a trial:
    Because P is shared across both axes but |cos(theta_i)| is
    axis-specific, a force pushing hard along x alone (P large,
    cos_x approx 1, cos_y approx 0) drives Dx toward the low end of
    the table (row cos=1 favors low Dd at high positive P) while Dy is
    driven by the table's cos=0 row, which stays relatively high (7-9
    fuzzy index) across the full P range. This is by design (resist
    unintended off-axis drift) but means Dy does not fall toward
    damping_min_n_s_m just because the robot is moving fast overall;
    only an axis the force is actually aligned with becomes compliant.

Main tunable parameters:
    Virtual mass, the damping bounds [damping_min_n_s_m,
    damping_movement_max_n_s_m, damping_safety_max_n_s_m], and
    power_range_w are all in User Settings, sorted most to least
    important. Check force signs, workspace, and the start pose before
    setting send_robot_commands = True.

Dependency:
    pip install --user ur_rtde tobii-research coppeliasim-zmqremoteapi-client

Assumptions:
    Motion is planar in the UR5e base-frame x-y plane at fixed z and
    orientation, matching this project's other planar controllers. The
    human pushes the tool directly, so the measured TCP wrench is the
    human force. Human intention is estimated purely from filtered
    force and measured velocity; unlike ur5e_gaze_assisted_adm_2d.py
    this controller does not use gaze for control, though Tobii data is
    still recorded for cross-trial comparability (see
    ur5e_chen_var_adm_2d.py and ur5e_kang_var_adm_2d.py, which do the
    same).
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
    "ur5e_ying_var_adm_2d_fine_gross_shape_s0.csv"
)

# TODO Tobii output CSV path [file path]; one row per Tobii callback.
# Shares the time_s origin of the controller CSV, so the pair matches
# the <ur5e_*, tobii_*> convention used by this project's MATLAB scripts.
tobii_data_csv_path = (
    Path(__file__).resolve().parent.parent /
    "data" /
    "raw_data" /
    "tobii_ying_var_adm_2d_fine_gross_shape_s0.csv"
)

# --- Admittance model (2), (3); fuzzy damping (1), Table I "Rule A" ---
# TODO virtual mass Md [kg], recommended 1 to 20. Not given explicitly
# by the paper; matches this project's other controllers.
virtual_mass_kg = 10.0

# TODO minimum fuzzy damping [N s/m], positive. Paper: "minimum damping
# in the movement is 5 N s/m".
damping_min_n_s_m = 10.0

# TODO maximum fuzzy damping reached by the 9-index table's smooth
# interpolation [N s/m], must exceed damping_min_n_s_m. Paper: "maximum
# damping in the movement is 150 N s/m".
damping_movement_max_n_s_m = 100.0

# TODO outer safety clamp applied after the table lookup [N s/m], must
# be >= damping_movement_max_n_s_m. Paper: "200 N s/m is used to
# prevent deviant [behavior]"; see implementation note 2 above.
damping_safety_max_n_s_m = 200.0

# TODO half-width of the interaction-power axis of Table I [W],
# positive. The paper's own calibration ("P changes in the range of
# [-4, 4] in normal interaction") is specific to their Franka Panda
# setup and subjects; this UR5e-scale guess needs recalibration from a
# real trial's Fx*vx + Fy*vy. See implementation note 4 above.
power_range_w = 2.5

# TODO force magnitude below which |cos(theta_i)| is undefined and both
# axes default to damping_movement_max_n_s_m instead [N], smaller than
# force_deadband_n. See implementation note 6 above.
minimum_force_for_direction_n = 0.5

# TODO damping slew-rate limit [N s/m per s], recommended 100 to 600.
# Bounds how fast either axis's damping may change so the operator does
# not feel a step in resistance. Not described by the paper.
damping_rate_max_n_s_m_per_s = 300.0

# --- Force conditioning ---
force_lpf_alpha = 0.2  # TODO filter alpha [0, 1]; smaller is smoother
force_deadband_n = 1.0  # TODO radial force deadband [N], 0 to 5
force_bias_samples = 250  # TODO unloaded samples [count], >= 100

# TODO set signs so a positive push produces positive base-frame motion.
force_sign_x = 1.0  # TODO x sign [unitless], valid values: -1 or +1
force_sign_y = 1.0  # TODO y sign [unitless], valid values: -1 or +1

# --- Velocity conditioning ---
# TODO low-pass alpha for the measured TCP velocity [0, 1]. Used only
# for the interaction-power P = Fh . V, never for the admittance
# integration itself (which uses its own commanded velocity state).
velocity_lpf_alpha = 0.2

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


# -------------------- Fuzzy Rule Table (Rule A, paper Table I) -------
# Rows are |cos(theta_i)| indices 0..4, mapping linearly to cos values
# 0.00, 0.25, 0.50, 0.75, 1.00. Columns are P indices matching the
# paper's own P labels -4..4 directly, rescaled at query time onto
# [-power_range_w, +power_range_w] (see evaluate_fuzzy_rule_a()). Cell
# values are output fuzzy-set indices 1-9 (not N s/m directly),
# transcribed exactly from the paper's Table I.
RULE_A_TABLE = [
    # P =        -4  -3  -2  -1   0   1   2   3   4
    [9, 9, 8, 7, 7, 7, 8, 9, 9],  # |cos(theta_i)| = 0.00
    [9, 8, 7, 6, 6, 6, 6, 7, 7],  # |cos(theta_i)| = 0.25
    [8, 7, 7, 6, 6, 6, 5, 5, 4],  # |cos(theta_i)| = 0.50
    [8, 7, 7, 6, 5, 5, 4, 3, 2],  # |cos(theta_i)| = 0.75
    [7, 6, 6, 5, 5, 4, 3, 2, 1],  # |cos(theta_i)| = 1.00
]
RULE_A_TABLE_ROW_COUNT = len(RULE_A_TABLE)
RULE_A_TABLE_COL_COUNT = len(RULE_A_TABLE[0])
RULE_A_MIN_INDEX = 1.0
RULE_A_MAX_INDEX = 9.0


# -------------------- Derived Settings --------------------
dt = 1.0 / control_frequency_hz
start_joint_q = [math.radians(angle_deg) for angle_deg in start_joint_q_deg]


# -------------------- Setting Checks --------------------
# Only conditions that would silently produce wrong physics or unsafe
# motion are checked here, per this project's "no defensive scaffolding"
# rule.
if damping_min_n_s_m <= 0.0:
    raise ValueError("Require damping_min_n_s_m > 0.")

if damping_movement_max_n_s_m <= damping_min_n_s_m:
    raise ValueError(
        "Require damping_min_n_s_m < damping_movement_max_n_s_m.")

if damping_safety_max_n_s_m < damping_movement_max_n_s_m:
    raise ValueError(
        "Require damping_safety_max_n_s_m >= damping_movement_max_n_s_m.")

if power_range_w <= 0.0:
    raise ValueError("Require power_range_w > 0.")

if virtual_mass_kg <= 0.0:
    raise ValueError("Require virtual_mass_kg > 0.")

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

# Explicit discrete-stability check for the per-axis admittance
# integration. The forward-Euler update v <- v + (F - D*v)/M * dt is
# stable only while D*dt/M < 2 on each axis; the factor 2 margin keeps
# it well inside that bound. damping_safety_max_n_s_m is the true worst
# case since it is the outer clamp applied after the fuzzy lookup.
if damping_safety_max_n_s_m * dt / virtual_mass_kg > 1.0:
    raise ValueError(
        "Admittance integration is too stiff for this control period: "
        "reduce damping_safety_max_n_s_m, raise virtual_mass_kg, or "
        "raise control_frequency_hz."
    )


# -------------------- Controller State --------------------
keep_running = True

force_bias_n = [0.0, 0.0]
force_raw_n = [0.0, 0.0]
force_filtered_n = [0.0, 0.0]

# Measured TCP velocity, raw and low-pass filtered; used only for the
# interaction power P, never for the admittance integration itself.
velocity_measured_m_s = [0.0, 0.0]
velocity_filtered_m_s = [0.0, 0.0]

# Virtual admittance velocity state; this is what is commanded.
cart_velocity_command_m_s = [0.0, 0.0]

start_position_m = [0.0, 0.0]

power_w = 0.0
cos_theta_x = 0.0
cos_theta_y = 0.0

damping_x_desired_n_s_m = damping_movement_max_n_s_m
damping_x_applied_n_s_m = damping_movement_max_n_s_m
damping_y_desired_n_s_m = damping_movement_max_n_s_m
damping_y_applied_n_s_m = damping_movement_max_n_s_m


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


def vec2_dot(vector_a, vector_b):
    """Return the dot product of two 2-D vectors."""

    return vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]


def limit_vector_norm_2d(vector, max_norm):
    """Limit a 2-D vector magnitude while preserving its direction."""

    vector_norm = vec2_norm(vector)

    if vector_norm <= max_norm:
        return list(vector)

    scale = max_norm / vector_norm
    return [scale * vector[0], scale * vector[1]]


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
        angular components [rad/s].
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


def update_velocity_filter(tcp_actual_velocity):
    """Low-pass the measured planar TCP velocity [m/s].

    The filtered velocity feeds the interaction power P = Fh . V only.
    The admittance integration uses its own virtual velocity state, so
    this filter cannot add lag to the commanded motion.
    """

    velocity_measured_m_s[0] = tcp_actual_velocity[0]
    velocity_measured_m_s[1] = tcp_actual_velocity[1]

    for axis in range(2):
        velocity_filtered_m_s[axis] = (
            (1.0 - velocity_lpf_alpha) * velocity_filtered_m_s[axis] +
            velocity_lpf_alpha * velocity_measured_m_s[axis]
        )


# -------------------- Fuzzy Damping (paper 1, Table I) ----------------
def bilinear_interpolate_table(
    table,
    row_value,
    col_value,
    row_min,
    row_max,
    col_min,
    col_max,
):
    """Bilinearly interpolate a 2-D table at a continuous query point.

    Inputs:
        table: row-major 2-D list, table[row][col].
        row_value, col_value: query point, clamped into
            [row_min, row_max] and [col_min, col_max].
        row_min, row_max, col_min, col_max: the value each table edge
            (first/last row or column) corresponds to; the rows and
            columns in between are assumed evenly spaced across that
            span.

    Returns:
        The interpolated table value at (row_value, col_value).
    """

    row_count = len(table)
    col_count = len(table[0])

    row_query = clamp_value(row_value, row_min, row_max)
    col_query = clamp_value(col_value, col_min, col_max)

    row_fraction = (
        (row_query - row_min) / (row_max - row_min) * (row_count - 1))
    col_fraction = (
        (col_query - col_min) / (col_max - col_min) * (col_count - 1))

    row_lo = int(math.floor(row_fraction))
    row_hi = min(row_lo + 1, row_count - 1)
    col_lo = int(math.floor(col_fraction))
    col_hi = min(col_lo + 1, col_count - 1)

    row_t = row_fraction - row_lo
    col_t = col_fraction - col_lo

    top_value = (
        table[row_lo][col_lo] * (1.0 - col_t) +
        table[row_lo][col_hi] * col_t
    )
    bottom_value = (
        table[row_hi][col_lo] * (1.0 - col_t) +
        table[row_hi][col_hi] * col_t
    )

    return top_value * (1.0 - row_t) + bottom_value * row_t


def evaluate_fuzzy_rule_a(power_query_w, cos_theta_axis):
    """Evaluate Rule A (Table I) at one axis's query point, paper (1).

    Inputs:
        power_query_w: interaction power P [W], clamped internally to
            [-power_range_w, power_range_w].
        cos_theta_axis: |cos(theta_i)| for this axis, in [0, 1].

    Returns:
        Fuzzy damping [N s/m], mapped from the interpolated output
        index (see implementation note 1 and 2 in the module
        docstring) and clamped to [damping_min_n_s_m,
        damping_safety_max_n_s_m].
    """

    fuzzy_index = bilinear_interpolate_table(
        RULE_A_TABLE,
        cos_theta_axis,
        power_query_w,
        0.0,
        1.0,
        -power_range_w,
        power_range_w,
    )

    index_fraction = (
        (fuzzy_index - RULE_A_MIN_INDEX) /
        (RULE_A_MAX_INDEX - RULE_A_MIN_INDEX)
    )
    damping_n_s_m = damping_min_n_s_m + index_fraction * (
        damping_movement_max_n_s_m - damping_min_n_s_m)

    return clamp_value(
        damping_n_s_m, damping_min_n_s_m, damping_safety_max_n_s_m)


def rate_limit_damping(desired_n_s_m, previous_n_s_m):
    """Limit the per-step change of one axis's damping [N s/m]."""

    max_step_n_s_m = damping_rate_max_n_s_m_per_s * dt
    damping_step = clamp_value(
        desired_n_s_m - previous_n_s_m, -max_step_n_s_m, max_step_n_s_m)

    return previous_n_s_m + damping_step


def update_fuzzy_damping():
    """Update the interaction power, both cosines, and Dx/Dy, (1).

    Below minimum_force_for_direction_n, |cos(theta_i)| is undefined
    (0/0) and both axes default directly to
    damping_movement_max_n_s_m instead of evaluating the table at an
    arbitrary fallback angle; see implementation note 6 in the module
    docstring.

    Outputs (module state):
        power_w, cos_theta_x, cos_theta_y, damping_x_desired_n_s_m,
        damping_x_applied_n_s_m, damping_y_desired_n_s_m,
        damping_y_applied_n_s_m.
    """

    global power_w, cos_theta_x, cos_theta_y
    global damping_x_desired_n_s_m, damping_x_applied_n_s_m
    global damping_y_desired_n_s_m, damping_y_applied_n_s_m

    force_magnitude_n = vec2_norm(force_filtered_n)
    power_w = vec2_dot(force_filtered_n, velocity_filtered_m_s)
    power_w = clamp_value(power_w, -power_range_w, power_range_w)

    if force_magnitude_n < minimum_force_for_direction_n:
        cos_theta_x = 0.0
        cos_theta_y = 0.0
        damping_x_desired_n_s_m = damping_movement_max_n_s_m
        damping_y_desired_n_s_m = damping_movement_max_n_s_m
    else:
        cos_theta_x = clamp_value(
            abs(force_filtered_n[0]) / force_magnitude_n, 0.0, 1.0)
        cos_theta_y = clamp_value(
            abs(force_filtered_n[1]) / force_magnitude_n, 0.0, 1.0)
        damping_x_desired_n_s_m = evaluate_fuzzy_rule_a(
            power_w, cos_theta_x)
        damping_y_desired_n_s_m = evaluate_fuzzy_rule_a(
            power_w, cos_theta_y)

    damping_x_applied_n_s_m = rate_limit_damping(
        damping_x_desired_n_s_m, damping_x_applied_n_s_m)
    damping_y_applied_n_s_m = rate_limit_damping(
        damping_y_desired_n_s_m, damping_y_applied_n_s_m)


def integrate_planar_admittance():
    """Integrate Md*x_ddot + Dd*x_dot = Fh independently per axis, (3).

    No rotation is involved: Dx and Dy already apply directly along the
    base-frame x and y axes (see the module docstring's "Controller"
    section).

    Outputs (module state):
        cart_velocity_command_m_s.
    """

    global cart_velocity_command_m_s

    acceleration_x_m_s2 = (
        force_filtered_n[0] -
        damping_x_applied_n_s_m * cart_velocity_command_m_s[0]
    ) / virtual_mass_kg
    acceleration_y_m_s2 = (
        force_filtered_n[1] -
        damping_y_applied_n_s_m * cart_velocity_command_m_s[1]
    ) / virtual_mass_kg

    cart_velocity_command_m_s[0] += acceleration_x_m_s2 * dt
    cart_velocity_command_m_s[1] += acceleration_y_m_s2 * dt

    # Clamping the state itself, not just the output, keeps the virtual
    # model from winding up while the limit is active.
    cart_velocity_command_m_s = limit_vector_norm_2d(
        cart_velocity_command_m_s, max_cart_speed_m_s)


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
        "tcp_vx_filt_m_s",
        "tcp_vy_filt_m_s",
        "vx_cmd_m_s",
        "vy_cmd_m_s",
        "speed_cmd_m_s",
        "power_W",
        "cos_theta_x",
        "cos_theta_y",
        "Dx_desired_Ns_m",
        "Dx_applied_Ns_m",
        "Dy_desired_Ns_m",
        "Dy_applied_Ns_m",
        "M_kg",
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
        velocity_filtered_m_s[0],
        velocity_filtered_m_s[1],
        cart_velocity_command_m_s[0],
        cart_velocity_command_m_s[1],
        vec2_norm(cart_velocity_command_m_s),
        power_w,
        cos_theta_x,
        cos_theta_y,
        damping_x_desired_n_s_m,
        damping_x_applied_n_s_m,
        damping_y_desired_n_s_m,
        damping_y_applied_n_s_m,
        virtual_mass_kg,
        int(send_robot_commands),
    ])


# -------------------- Main --------------------
def main():
    """Run the UR5e Ying-style planar fuzzy variable admittance loop."""

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
                "damping range [N s/m] =",
                damping_min_n_s_m,
                damping_movement_max_n_s_m,
                "(safety clamp", damping_safety_max_n_s_m, ")",
            )
            print("power_range_w =", power_range_w)

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
                update_velocity_filter(tcp_actual_velocity)

                # Paper data flow: the fuzzy rule (1) sets Dx and Dy
                # from the current force and measured velocity, then
                # (3) integrates the admittance model per axis with no
                # rotation involved.
                update_fuzzy_damping()
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

    print("Stopped UR5e Ying-style fuzzy variable admittance controller.")
    print("CSV saved to:", write_data_csv_path)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
