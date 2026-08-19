# Context Handoff: Building a Damping-Prediction Framework

## 0. Purpose of this document

This describes an existing robotics research codebase (UR5e robot arm +
Tobii eye tracker, physical human-robot interaction / pHRI) so that a
**separate, new framework/submodule** can be planned and built to predict
the admittance-controller damping coefficient from behavioral and physical
signals. That new framework is **not part of this repository** and should
not modify anything here. It only needs to **read** the data described
below (treat `data/raw_data/` and `data/calibrations/` as read-only source
data) and reproduce/extend the feature-extraction logic referenced here in
its own codebase.

If you (the agent reading this) are scoping or implementing that new
framework, this document should give you everything needed to decide what
raw signals exist, what per-trial features can be engineered from them,
and what "ground truth" signal (mean force, for this first pass — see
§8.2-8.3 for why squared jerk was ruled out and why mean force alone is
probably not the whole story) is available to tune/score the predictor's
output.

## 1. Project overview

ICRA 2027 pHRI research project. A human physically pushes/guides the
end-effector (TCP) of a UR5e robot arm in a horizontal plane while a
**variable admittance controller** governs how the robot responds to that
force. Damping (`B` below) is the key tunable: low damping feels
compliant/easy to push but is prone to overshoot and jerkiness; high
damping feels stiff/resistive but smooths the motion. The project studies
whether **eye-gaze behavior** (via a Tobii Pro Nano eye tracker mounted
near the workspace) can inform what damping should be at any moment, on
the theory that where/how a person looks reveals their movement intent
before force alone would show it.

Two physical instruments are used together and recorded with a shared
host-monotonic clock:
- **UR5e robot arm**, controlled/read via the `ur_rtde` Python API
  (`rtde_control`, `rtde_receive`) at a native 500 Hz control loop.
- **Tobii Pro Nano eye tracker**, read via the `tobii_research` Python SDK
  through an async callback (~60 Hz native rate, but callback-driven and
  not a fixed period; each trial's actual rate is recorded).
- **CoppeliaSim** is used only as a visualization/marker bridge (a sphere
  object tracks the TCP position in a simulated scene) via a separate
  ZeroMQ process; it has no control authority and does not affect the
  physics.

## 2. Repository layout

```
scripts/                     Python control/recording scripts (see §3-5)
matlab/                      MATLAB analysis/plotting scripts (self-contained,
                              never call each other — every script that needs
                              a shared algorithm, e.g. I-VT fixation detection,
                              copies it in as a local function)
data/raw_data/                Raw per-trial CSVs from the Python scripts (see §6)
data/calibrations/            Per-(band,shape) Tobii<->CoppeliaSim calibration
                              JSON (see §6.5)
data/processed_data/          MATLAB-generated derived CSVs, including the
                              existing pilot feature table (see §6.6)
data/shapes/                  (workspace/task shape definitions, not detailed here)
AI_GUIDELINES.md             This project's coding/style conventions
doc/                          Reference papers (variable admittance control,
                              gaze-assisted teleoperation, power-law speed/
                              curvature literature)
```

## 3. The control framework and what "damping" means

Both controllers implement the same planar admittance equation per axis,
integrated at `dt = 1/500 s`:

```
M * v_dot + B * v = F
```

`M` [kg] is virtual inertia, `B` [N·s/m] is damping, `F` [N] is the
human-applied force measured at the TCP (robot force/torque sensor, base
frame, biased against an unloaded baseline, deadbanded, then low-pass
filtered with `filtered = (1-alpha)*filtered + alpha*raw`, `alpha=0.2`).
The controller integrates this to a commanded Cartesian velocity, clamped
to a max speed, and sends it to the robot via `speedL()`. **Damping `B` is
the central quantity this new framework is meant to predict/tune.**

### 3.1 `scripts/ur5e_fixed_adm_2d.py` — fixed-damping controller

`B_damping_x`/`B_damping_y` are constant for the whole trial (set once in
User Settings, e.g. 10, 50, or 100 N·s/m in the pilot dataset — see §7).
This is the script used to record the **pilot dataset**, which is the
primary candidate source of clean, controlled training data because
damping is an explicit independent variable (not something already being
adapted in response to the user, so its effect on force/smoothness is not
confounded by a closed feedback loop on gaze).

### 3.2 `scripts/ur5e_gaze_assisted_adm_2d.py` — variable-damping controller (background only, not a data source)

**Out of scope as a data source for this framework** — its CSV has no
`coppeliasim_x/y_m` columns (§6.3), so it cannot be placed in the same
spatial frame as gaze without extra, unverified work. Kept here only as
background: it is the project's existing hand-tuned (non-learned)
approach to *setting* damping from gaze, conceptually the same problem
this new framework addresses, and a possible eventual target to replace
or augment. Useful for the signals it treats as meaningful:

- I-VT (velocity-threshold) fixation detection runs on live gaze to find
  the current fixation point, mapped into the robot's frame via the
  calibration (§5).
- Up to `intent_history_max_fixation_count` (default 5) recent, still-fresh
  fixations are kept as **candidate intent targets**, not just the latest
  one.
- Each candidate gets a weight = `gaze_influence` (confidence, ramps 0->1
  over fixation duration) x `distance_weight` (ramps 0->1 as TCP-to-fixation
  distance grows, avoiding noisy short vectors) x an **alignment weight**
  vs. the instantaneous filtered force direction (1 at 0 deg apart, 0 at
  `force_intent_alignment_cutoff_deg`, default 80 deg apart) — i.e. gaze
  only gets trusted when the user is actually pushing toward it.
- Candidates combine via a noisy-OR: `total_influence = 1 - prod(1 - w_i)`,
  a scalar in [0, 1].
- Damping is anisotropic and blended: at `total_influence = 0` it is the
  isotropic `fixed_damping_{x,y}_n_s_m` (default 80 N·s/m); at
  `total_influence = 1` it uses `intent_forward_damping_n_s_m` (default 10,
  i.e. compliant) along the force direction and
  `intent_other_damping_n_s_m` (default 80, i.e. stiff) perpendicular/
  opposite to it. Intermediate `total_influence` linearly blends the
  resulting **damping force**, not the coefficient directly (see
  `calculate_damping_force()` in the script for the exact vector math).
- **Important:** the logged CSV for this script does not contain a single
  scalar "effective damping" column — only `total_influence`,
  `damping_force_x_N`/`damping_force_y_N`, and `intent_direction_x/y`. An
  effective scalar damping-in-the-direction-of-motion would need to be
  reconstructed from those plus the known constants above.
- Only exploratory/debug recordings exist for this controller so far (see
  §7) — there is no clean multi-condition dataset from it yet.

## 4. Eye-tracking data (Tobii Pro Nano)

Gaze is reported in the **Tobii User Coordinate System (UCS)**, millimeters,
a 3-D coordinate frame fixed to the tracker/screen setup — not the robot
frame (see §5 for the mapping). Each SDK callback reports **per-eye**
values (left and right are independent and can each be valid or invalid on
any given sample):
- `*_gaze_point_in_user_coordinate_system`: 3-D point [mm] in UCS where
  that eye's gaze intersects the tracked plane.
- `*_gaze_origin_in_user_coordinate_system`: 3-D estimated eyeball position
  [mm] in UCS.
- `*_gaze_point_on_display_area`: 2-D point normalized to `[0, 1]^2`
  (screen-relative; used to test "is this gaze actually on the active
  display").
- `*_gaze_point_validity`, `*_gaze_origin_validity`: 1 = valid, else invalid.
- `*_pupil_diameter`: pupil diameter [mm]; `*_pupil_validity`: 1 = valid.

**Eye direction** for one eye is
`direction = normalize(gaze_point_ucs - gaze_origin_ucs)`, valid only when
both origin and point are valid and the vector is nonzero. **Two-eye
combination** has two different conventions used in this codebase — do not
conflate them:
- *Fallback average* (used for I-VT fixation detection and for the gaze-
  assisted controller's live intent tracking): average whichever eye(s)
  are valid; falls back to a single eye if only one is valid.
- *Strict average* (used when a genuine two-eye physiological/spatial
  signal is wanted, e.g. pupil diameter, or a "clean" raw gaze cloud):
  requires **both** eyes valid, `avg = 0.5*(left + right)`, otherwise the
  sample is excluded — no single-eye substitution.

**I-VT fixation detection** (the algorithm used everywhere fixations are
computed in this codebase — see e.g.
`matlab/compute_plot_similarity_fixation_direction_force_vel.m`):
1. Combine left/right eye directions (fallback-average convention) into a
   unit gaze direction per sample.
2. Angular velocity between consecutive valid samples:
   `angle_deg = acosd(dot(dir_i, dir_i+1))`, `deg/s = angle_deg / dt`.
3. A sample is a fixation candidate when combined gaze is valid and
   `angular_velocity_deg_s <= velocity_threshold_deg_s` (typically 20-30
   deg/s in this codebase).
4. Contiguous candidate runs shorter than `minimum_fixation_duration_s`
   (typically 0.06-0.10 s) are discarded; the rest become fixations, each
   summarized by its mean UCS gaze point and its duration.

## 5. Tobii <-> CoppeliaSim calibration

One calibration JSON exists per (band, shape) task condition (see §7),
shared across that condition's damping trials, produced by
`scripts/ur5e_tobii_reference_calibration.py`. Schema:

```json
{
  "manual_tobii_reference_xy_mm": [x, y],
  "manual_coppelia_reference_xy_m": [x, y]
}
```

The script moves the robot to a known joint pose (known CoppeliaSim/robot
position), has the subject fixate a reference marker for ~3 s, and records
the median combined UCS gaze point over that window as the Tobii-side
reference. Mapping any UCS point to the CoppeliaSim/robot x-y frame (used
throughout the MATLAB analysis scripts and inside the gaze-assisted
controller):

```
m_per_mm = 1e-3 / screen_motion_per_robot_motion   # typically screen_motion_per_robot_motion = 1.0
displacement_mm = ucs_xy_mm - manual_tobii_reference_xy_mm
displacement_m  = displacement_mm .* axis_sign .* m_per_mm   # axis_sign in {-1,+1} per axis, typically [1,1]
xy_m = manual_coppelia_reference_xy_m + displacement_m
```

This is a **pure translation + optional per-axis sign flip**, no rotation
or scale beyond the fixed `screen_motion_per_robot_motion` factor — assumes
the Tobii UCS x/y axes are already aligned with the CoppeliaSim/robot x/y
axes (a stated assumption throughout).

## 6. Data inventory

### 6.1 Directory tree & file-naming convention

```
data/raw_data/ur5e_<controller>_<condition>.csv     robot-side log
data/raw_data/tobii_<controller>_<condition>.csv    gaze-side log (same trial)
data/calibrations/tobii_coppelia_reference_calibration_<band>_<shape>.json
```
`<controller>` is `fixed_adm_2d` or `gaze_assisted_adm_2d`. Every
`ur5e_*.csv` has a matching `tobii_*.csv` from the same trial, sharing the
same host-monotonic `time_s` origin (both zero at that trial's recording
start), so they can be joined/interpolated on `time_s` directly — **the
Tobii CSV is not synchronously sampled with the robot CSV** (independent
~60 Hz async callback vs. 500 Hz robot loop), so any per-sample join needs
interpolation, not row alignment.

### 6.2 Robot CSV schema — `ur5e_fixed_adm_2d_*.csv`

```
time_s, host_monotonic_time_ns,
Fx_raw_N, Fy_raw_N, Fx_filt_N, Fy_filt_N,
x_m, y_m, z_m,                         # robot base frame, measured TCP position
coppeliasim_x_m, coppeliasim_y_m, coppeliasim_z_m,   # see note below
tcp_vx_actual_m_s, tcp_vy_actual_m_s, tcp_vz_actual_m_s,
tcp_wx_actual_rad_s, tcp_wy_actual_rad_s, tcp_wz_actual_rad_s,
vx_cmd_m_s, vy_cmd_m_s, speed_cmd_m_s,  # controller's own commanded velocity
M_x_kg, M_y_kg, B_x_Ns_m, B_y_Ns_m,     # constant for the whole trial in this script
send_robot_commands                      # 1 if commands were actually sent to hardware
```
500 Hz nominal sample rate (one row per control-loop iteration).
`tcp_v*_actual_m_s`/`w*` come from `rtde_receive.getActualTCPSpeed()` — the
robot controller's own (already-filtered, not finite-differenced by this
script) velocity estimate. The existing MATLAB feature-extraction pipeline
(§6.6) does **not** trust this column for derivatives; it re-derives
velocity/acceleration by differentiating a filtered position trace
instead — a good reason to prefer force-derived signals over any
further-differentiated kinematic quantity (§8.2-8.3). `coppeliasim_x/y/z_m`
is just `x/y/z_m` passed through an axis-sign/offset map
(`map_tcp_position_to_coppeliasim`); in the pilot dataset this map is the
identity ( `coppeliasim_position_signs=[1,1,1]`,
`coppeliasim_position_offset=[0,0,0]` ), so `coppeliasim_x_m == x_m` and
`coppeliasim_y_m == y_m` for that data — but use `coppeliasim_x_m/y_m` for
anything that needs to line up with gaze/fixation coordinates, since that
is the frame the calibration (§5) maps into.

### 6.3 `ur5e_gaze_assisted_adm_2d_*.csv` — excluded from this framework's scope

This schema has no `coppeliasim_x/y_m` columns (only `tcp_x_m`/`tcp_y_m`
in the robot base frame), so it cannot be aligned with gaze/fixation
coordinates without replicating that script's own
`coppeliasim_axis_from_robot`/`coppeliasim_sign_from_robot`/
`coppeliasim_offset_xy_m` mapping — unverified, extra work. **Do not use
these files as data for this framework.** They are also not a clean
dataset anyway (§7): only ad hoc test/debug recordings exist, no
controlled damping variation. See §3.2 for why the controller that
produced them is still relevant background.

### 6.4 Gaze CSV schema — `tobii_*.csv` (both controllers)

```
time_s, host_monotonic_time_ns, device_time_stamp,
left_gaze_origin_in_user_coordinate_system, left_gaze_origin_validity,
left_gaze_point_in_user_coordinate_system, left_gaze_point_on_display_area,
left_gaze_point_validity, left_pupil_diameter, left_pupil_validity,
right_gaze_origin_in_user_coordinate_system, right_gaze_origin_validity,
right_gaze_point_in_user_coordinate_system, right_gaze_point_on_display_area,
right_gaze_point_validity, right_pupil_diameter, right_pupil_validity,
system_time_stamp, sdk_data_json
```
The `*_in_user_coordinate_system`/`*_on_display_area` columns are stored as
JSON-array-like text in a single CSV cell (e.g. `[x,y,z]`) — parse them,
don't treat them as scalars. `sdk_data_json` is the full original SDK
dictionary for that sample, redundant with the named columns but useful as
a fallback/audit trail. `time_s` shares its zero-point with the paired
robot CSV's `time_s` (both trial-relative, host-monotonic).

### 6.5 Calibration JSON schema

See §5. One file per `(band, shape)` pair,
`tobii_coppelia_reference_calibration_<band>_<shape>.json`.

### 6.6 Existing processed feature table — `data/processed_data/pilot_condition_table.csv`

Already built by `matlab/build_pilot_condition_table.m` (one row per pilot
trial, §7). This is the **closest existing prior art for feature
engineering** in this codebase — reuse its computation methods rather than
reinventing them. Columns:

```
trial_id, band, shape, damping_n_s_m,
movement_time_s, path_length_m,
mean_speed_m_s, peak_speed_m_s,
mean_force_n, peak_force_n, impulse_n_s,
gaze_n_samples, gaze_valid_pct, gaze_hz,
fixation_count, fixation_rate_hz, mean_fixation_duration_s,
median_curvature_1_m,
power_law_beta, power_law_r2, power_law_n_used, power_law_frac_used,
calibration_id,
fixation_density_curvature_bin1_hz, fixation_density_curvature_bin2_hz,
fixation_density_curvature_bin3_hz
```
Notable computation details (see `process_condition()` in that file for
exact code):
- `mean_force_n` / `peak_force_n` = mean/max of
  `hypot(Fx_filt_N, Fy_filt_N)` over the trial; `impulse_n_s` =
  `trapz(time_s, force_magnitude_n)`. **`mean_force_n` is already exactly
  one of the two ground-truth signals requested for this new framework —
  no new computation needed for it.**
- `path_length_m`, `mean_speed_m_s`, `peak_speed_m_s`, and
  `median_curvature_1_m` all come from `prepare_tcp_trajectory()`, which:
  removes non-finite/duplicate-time samples, resamples `coppeliasim_x/y_m`
  onto a uniform time grid (median sample interval), applies a
  moving-median filter (window 0.02 s) to reject outliers, then a
  moving-mean filter (window 0.08 s) to smooth, differentiates
  (`gradient`) to get velocity, smooths velocity the same way, then
  differentiates again for acceleration. Curvature is
  `kappa = (vx*ay - vy*ax) / (vx^2+vy^2)^1.5`, only where speed exceeds a
  0.002 m/s floor (curvature is undefined/noisy near-zero speed).
- `power_law_beta`/`power_law_r2`: fits `log(speed) = log(k) - beta*log(kappa)`
  (the empirical speed-curvature power law from the motor-control
  literature referenced in `doc/`) via linear regression in log-log space,
  restricted to a curvature range with >= 20 samples.
- Fixation-related columns reuse the I-VT algorithm from §4.
- `fixation_density_curvature_bin{1,2,3}_hz`: fixation rate [Hz] split by
  which pooled-curvature tertile (bin edges computed globally across all
  trials) the TCP was passing through at each fixation's timestamp.

## 7. The pilot dataset (the dataset in scope)

This framework uses **only** the clean 2 (band) x 3 (shape) x 3 (damping)
= **18-condition factorial grid**, one trial each (no repeats, no
multiple subjects — this is pilot data), recorded with
`ur5e_fixed_adm_2d.py` — confirmed present: all 18 `ur5e_*.csv`/
`tobii_*.csv` pairs and all 6 calibration JSONs exist in `data/raw_data/`
and `data/calibrations/`. Every one of these robot CSVs has
`coppeliasim_x_m`/`coppeliasim_y_m` columns (§6.2), which is what makes
it usable here.

- `band` in `{narrow, wide}` — path/target width.
- `shape` in `{circle, curved, edge}` — task path geometry.
- `damping` in `{10, 50, 100}` N·s/m — the fixed `B_x = B_y` for that
  trial (this is the independent variable the eventual predictor should
  learn to set).

File stem pattern: `<band>_<shape>_shape_d<damping>`, e.g.
`ur5e_fixed_adm_2d_narrow_circle_shape_d10.csv` /
`tobii_fixed_adm_2d_narrow_circle_shape_d10.csv`, calibration
`tobii_coppelia_reference_calibration_narrow_circle.json` (shared by that
band+shape's 3 damping trials). All 18 `ur5e_*`/`tobii_*` pairs and all 6
calibration JSONs are present in `data/raw_data/` and
`data/calibrations/` respectively.

Established pilot finding (from prior exploratory analysis in this
project): **damping strongly affects force magnitude (~5x from
B=10 to B=100) but barely affects path speed/timing** — informally,
"damping is an effort knob, not a trajectory knob" for this pilot. This is
relevant context for how force and smoothness (jerk) might trade off
against damping.

`data/raw_data/` also contains many other CSVs from earlier, less
structured development sessions (shapes named `eight`, `ellipse`,
`irregular_shape_2`, `random_shape`, `s1_irregular_shape`,
`sym_random_shape_*`, etc., and several `ur5e_gaze_assisted_adm_2d_*test*`
recordings). **These are excluded from this framework** — not a
systematic damping sweep, and (for the gaze-assisted ones) missing
`coppeliasim_x/y_m` per §6.3. Do not read them.

## 8. What the damping-prediction submodule needs

**The goal, restated:** the classifier is not meant to hit one fixed,
hand-designed cost function. It is meant to **learn from data how best to
trade off speed and accuracy** when picking damping — low damping favors
speed (fast, cheap to push, per the "damping is an effort knob" pilot
finding in §7) but risks poor accuracy (jittery, hard-to-control fine
motion, per §8.3); high damping favors accuracy/stability but costs speed
and effort. Where the best point on that trade-off sits is exactly what
should come from data rather than being asserted up front. `mean_force_n`
and `force_jitter_n` (§8.3) are this project's current best proxies for
the two ends of that trade-off — effort/speed and accuracy/stability,
respectively — not two independent, unrelated metrics to juggle.

**This requires more data than currently exists.** The 18-trial pilot
tests only 3 discrete damping values per condition, one trial each, no
repeats — enough to see that a trade-off exists (§8.3) but not enough to
learn its shape, i.e. not enough to answer "how best" to trade the two
off for a given situation. Before or alongside building the predictor,
plan what additional data collection this needs: candidates include a
finer damping sweep per condition (more than 3 points, to trace the
trade-off curve rather than compare 3 samples on it), repeated trials per
condition (to get variance, since every number in §8.3's table is
currently an n=1 estimate), and/or more subjects (the current pilot is a
single person). Do not treat the 18-trial pilot as sufficient to *train*
a generalizable model — treat it as sufficient to prototype the feature/
objective pipeline this document describes, and to motivate what the next
data-collection round should target.

### 8.1 Candidate input features (per trial, or per short time-window if the eventual model runs online)

From the gaze stream (§4, §6.4/6.6):
- Fixation count, fixation rate [Hz], mean/median fixation duration.
- Saccade length (Euclidean distance between consecutive fixation
  centers, in the calibrated CoppeliaSim/robot frame) — distribution
  stats (mean, median, spread).
- Gaze data quality: `gaze_valid_pct`, `gaze_hz`.
- Pupil diameter (strict two-eye average, §4): mean/variance over the
  trial, and its spatial pattern (already exposed as
  `plot_pilot_trajectories_grid.m`'s temporal/spatial pupil figures, see
  matlab/ directory) as candidate features — pupil dilation is a known
  cognitive-load/arousal proxy in the eye-tracking literature.
- Gaze-vs-TCP alignment/lead-lag (see
  `matlab/analyze_gaze_tcp_lead_lag_curvature_v2.m` and
  `matlab/compute_plot_similarity_fixation_direction_force_vel.m` for an
  existing implementation of comparing gaze direction to force/velocity
  direction over time).

From the force stream (§6.2):
- `Fx_filt_N`/`Fy_filt_N` magnitude and direction statistics (mean, peak,
  impulse — already in §6.6) and higher-order shape (e.g. force rate of
  change, force direction reversals, the `force_jitter_n`-style roughness
  measure introduced in §8.3) as candidates for capturing "how
  hard/erratically is the person pushing."

From the TCP kinematics (§6.2, via the `prepare_tcp_trajectory()`
method in §6.6):
- Speed, curvature, path length, movement time, power-law beta/R^2.

From task identity (§7): `band`, `shape` — categorical, known in advance
of/independent of any within-trial behavior, so these do not carry the
confounding risk described in §8.4.

### 8.2 Ground-truth / tuning signal: mean force (for now)

**Squared jerk is deferred — too noisy.** It would require a third
time-derivative of position, and this codebase's own methodology already
treats *first*-derivative velocity as unreliable enough that it
re-differentiates a filtered position trace rather than trust the logged
`tcp_vx_actual_m_s` (§6.6's `prepare_tcp_trajectory()`). A third
derivative compounds that noise amplification again. Not used for now.

**Mean force is the sole ground-truth signal for this first pass.**
Already computed: `mean_force_n` in
`data/processed_data/pilot_condition_table.csv` (§6.6), or recomputable
from any pilot robot CSV as `mean(hypot(Fx_filt_N, Fy_filt_N))` over the
trial.

### 8.3 Mean force alone is very likely an incomplete objective — read before designing a training target

This section exists to answer a question the user raised directly, so
resolve it with the user before committing to "minimize mean force" as
the sole thing a model is trained toward.

**The concern, restated:** `F ~= B*v` in steady motion (from
`M*v_dot + B*v = F`), so at a given speed, higher damping mechanically
requires more force — this is exactly the established pilot finding in
§7 ("damping is an effort knob"). A model that only minimizes mean force
would therefore trivially learn "always predict the lowest damping,"
regardless of gaze or anything else, because mean force increases with
damping in essentially every condition. Separately, under-damped ("low
B") control of *fine*, precise motion is expected to be jittery/
oscillatory — a real control-theoretic effect (an under-damped mass-
damper system driven by a human's own neuromuscular feedback loop can
ring/overshoot) — and that failure mode is not what "low mean force" by
itself would penalize.

**This was checked empirically against the 18-trial pilot data** (not
just theorized): for each trial, in addition to `mean_force_n`, a cheap
force-roughness proxy was computed directly from the already-filtered
force channel — no differentiation of position needed, so none of the
noise-amplification concern that ruled out jerk in §8.2:

```
force_jitter_n = mean(|force_mag[i+1] - force_mag[i]|)   # consecutive-sample force change, N
force_mag = hypot(Fx_filt_N, Fy_filt_N)
```

Averaged across all 6 (band, shape) cells per damping level:

| B [N s/m] | mean(mean_force_n) [N] | mean(force_jitter_n) [N/sample] |
|-----------|------------------------|----------------------------------|
| 10        | 0.891                  | 0.0475                           |
| 50        | 2.267                  | 0.0346                           |
| 100       | 4.085                  | 0.0347                           |

**Mean force rises ~4.6x from B=10 to B=100 while force-jitter falls (most
of the drop between B=10 and B=50, then flat).** These two signals move
in *opposite* directions as damping increases, confirming they are not
redundant, and confirming the user's concern: an objective built from
mean force alone would push toward minimum damping and actively ignore
the jitter cost that also matters.

One thing this quick check did **not** confirm: the specific
`narrow`-vs-`wide` (fine-vs-gross movement) split the user hypothesized.
Comparing `force_jitter_n` between narrow and wide at the same damping did
not show a clean, consistent difference in this data (e.g. at B=10,
narrow/wide jitter across the 3 shapes was close and inconsistently
ordered) — plausibly because there is only one trial per condition (no
repeats, so this comparison is not statistically defensible yet), or
because `band` alone is too coarse a proxy for local movement "fineness"
(a trial's *within-trial* local speed/curvature, already available via
`prepare_tcp_trajectory()` in §6.6, is probably a better proxy for
gross-vs-fine than the whole-trial `band` label). Do not hard-code the
fine-motion-needs-high-damping assumption into a feature or objective
until this is checked more carefully — treat it as a hypothesis, not a
confirmed effect, in this pilot data.

**Recommendation:** once past this "mean force only" first pass, build a
combined objective, e.g. `cost = mean_force_n + lambda * force_jitter_n`
(z-score both across the 18 trials first so neither term dominates by
scale alone), and for each of the 6 (band, shape) cells, find which of
the 3 tested dampings minimizes it — that per-condition "best observed
damping" is a more defensible training target than a raw force value.
`force_jitter_n` (or a similar force-based roughness/oscillation measure,
e.g. its standard deviation, or a high-pass-filtered version of it) is
recommended specifically because it is computed directly from a measured
signal, unlike jerk. With only 18 trials and no repeats, treat any of
this as hypothesis-generating, not as data a final model should be
trusted to generalize from without more trials.

### 8.4 Known methodological risk — please reason about this explicitly, do not silently assume it away

Gaze and force behavior recorded *during* a pilot trial were themselves
produced *under* that trial's fixed damping setting. That means naively
training "predict the damping that was used, from gaze/force features
measured during that same trial" risks learning damping's *effect* on
behavior rather than a genuine *predictive* signal that would generalize
to choosing damping in advance or in real time. Concretely: at B=100 the
person necessarily pushes harder (confirmed pilot finding, §7), so
`mean_force_n` trivially "predicts" `damping_n_s_m` in the pilot data by
construction, not because it's a useful input signal for a real-time
damping controller. Before building a supervised model, resolve (or ask
the user to resolve) questions such as:
- Should the model predict damping from features available *before* the
  trial starts (task identity, subject calibration, etc.) — a much weaker
  and more honest problem than using within-trial behavioral features?
- Should the target be "the pilot's actual damping label," or should mean
  force (and, per §8.3, likely a force-roughness term too) instead define
  an **objective/cost** used to pick the *best* damping per condition
  (e.g. per band+shape, which of the 3 tested dampings minimizes a
  combined cost) and predict *that* from pre-trial features? This second
  framing is closer to how the user described mean force: as a value "to
  tune the output," i.e. an objective/scoring function, not necessarily a
  direct supervised label.
- If the eventual system is meant to run online (adjust damping
  continuously, like `ur5e_gaze_assisted_adm_2d.py` already attempts
  heuristically, §3.2), features and any cost signal need to be computed
  from short causal windows only (no look-ahead), which the pilot dataset
  (fixed damping for the whole trial) cannot directly supervise — it can
  only supply trial-level, not moment-level, ground truth.

This document does not resolve this; it exists so the new framework's
design explicitly accounts for it rather than discovering it after
training something on confounded features.

## 9. Project conventions worth respecting

From `AI_GUIDELINES.md` (this repo's own coding-style source of truth —
not binding on a separate framework, but useful defaults if you want the
two codebases to feel consistent):
- Always state units and frames in identifier names
  (`force_n`, `position_m`, `angular_velocity_deg_s`, etc.).
- Distinguish measured/filtered/commanded/desired values in names.
- Keep formulas visible/recognizable in code, not buried in dense
  one-liners; prefer explicit intermediate variables.
- `data/raw_data/` must not be modified by anything that reads it.

## 10. Constraints on this new work

- Do not modify any file in this repository as part of building the new
  framework — this document is reference/planning material only.
- Treat `data/raw_data/`, `data/calibrations/`, and
  `data/processed_data/` as read-only inputs.
- The new framework should live in its own directory/repository, separate
  from this one, and can freely choose its own language/library stack
  (this repo uses Python + MATLAB, but nothing requires the new framework
  to match that).
