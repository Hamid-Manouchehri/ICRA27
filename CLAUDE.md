# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ICRA 2027 paper work: combining variable admittance control (VAC), a power-law
speed/curvature model, and Tobii eye-tracking to modulate a UR5e's physical
compliance during human-robot interaction. Primary languages are Python
(real-time robot/eye-tracker control) and MATLAB (offline analysis of the
CSVs those scripts produce). There is no package manager, build system, or
test suite — this is a loose collection of standalone research scripts, each
runnable independently.

**Full project conventions live in `AI_GUIDELINES.md` — read it before
making non-trivial edits.** It is the authoritative style/workflow guide for
this repo (naming, User Settings format, permission rules, etc.) and takes
precedence over generic defaults. The most load-bearing rules from it:

- **Never execute hardware-control commands (`rtde_control`/`rtde_receive`
  scripts) or run MATLAB without explicit permission.** These scripts drive
  a real 6-DOF arm.
- Every runnable script has a `# -------------------- Start User Settings --------------------`
  block near the top with `TODO`-commented, unit-annotated, range-annotated
  tunables, sorted most- to least-important. Follow this pattern in new
  scripts; don't scatter tunables through the logic.
  And the block would ends with `# -------------------- End User Settings --------------------`
- Variable names carry units and frames as suffixes throughout: `_m`, `_m_s`,
  `_n` (newtons), `_deg`, `_deg_s`, `_rad_s`, `_xy_m`, `_robot_xy_m`. Keep
  this up when adding code — it's how frame/unit bugs get caught by reading.
- Lines fill to ~80 chars before wrapping (pack arguments onto one line up
  to the limit; only split onto continuation lines when it would overflow —
  don't force one-argument-per-line).
- `data/raw_data/` must not be modified or overwritten by anything other
  than the controller script that produced it. This means Claude must
  NEVER touch it — no edits, no overwrites, no deletes, no moves, not
  even as a side effect of another change. Read-only access is fine.
- Changing a script's `write_csv_header`/`write_csv_row` columns is a schema
  change: every downstream MATLAB script that reads that CSV (see the data
  flow below) needs matching updates, or it fails with "unrecognized table
  variable name" on the next run.
- Ask before: adding/deleting files, deleting code, changing CSV column
  names, adding dependencies, running install/build/system-modifying
  commands, creating/switching git branches, or committing/pushing.

## Verification

There's no test runner. The practical verification loop used throughout
this project:
- `python -m py_compile scripts/<name>.py` — syntax-only check that doesn't
  execute anything (safe to run freely; real execution is not).
- For MATLAB `.m` files, there's no non-interactive checker available here —
  changes are reasoned through manually and confirmed by the user running
  the script themselves.
- Dependencies aren't pinned anywhere (no requirements.txt/environment.yml);
  each script's docstring states its own `pip install` line.

## Architecture

### Two parallel tracks joined by CSV/JSON files

`scripts/` (Python) runs real-time hardware control and writes data.
`matlab/` (MATLAB) reads that data back offline for analysis/plotting. The
scripts never call MATLAB and vice versa — the only coupling is file
formats, so column/schema changes on one side must be manually mirrored on
the other. `matlab/README.md` documents the full MATLAB-side pipeline graph
(which script consumes which CSV/JSON and what it emits) — read it before
touching more than one `.m` file.

### Controller family (`scripts/`)

Three admittance controllers, increasing in sophistication, all structured
the same way (User Settings → shared mutable state → helper functions →
`main()` with an RTDE `initPeriod()`/`waitPeriod()` control loop, default
500 Hz):

- `ur5e_fixed_adm_2d.py` — baseline fixed-damping planar admittance.
- `ur5e_var_adm_direct_3d.py` — Kang-style direct-intention variable
  damping in 3D: `B = Fstd / |velocity|`, clamped and rate-limited, no gaze.
- `ur5e_gaze_assisted_adm_2d.py` — the active/complex one. Anisotropic
  damping is applied along the *instantaneous filtered force direction*
  (not the gaze direction), blended toward isotropic fixed damping unless
  gaze corroborates that push. Gaze intent is not a single latest fixation:
  a short history of recent distinct fixations
  (`intent_history_max_fixation_count`) is kept, each scored by its own
  confidence/distance/force-alignment weight, and combined into one
  `total_influence` via a noisy-OR (`1 - Π(1 - w_i)`) rather than a hard
  winner-take-all pick — this avoids discontinuous jumps in commanded
  damping when gaze moves between fixations. `calculate_blended_intent()`
  is the core of this logic.

Concurrency pattern shared by the gaze-assisted (and some standalone move)
scripts, worth understanding before editing them: the 500 Hz RTDE loop runs
on the main thread; Tobii gaze arrives via an SDK callback on its own
background *thread*, writing into lock-guarded shared state
(`fixation_lock`); the CoppeliaSim visualization marker is updated from a
separate `multiprocessing.Process` (not a thread) communicating over a
bounded `Queue`, specifically so ZeroMQ I/O latency can never stall the
control loop.

`ur5e_tobii_reference_calibration.py` performs one robot-move +
gaze-fixation calibration and writes the Tobii-UCS↔CoppeliaSim/robot-base
mapping to `data/calibrations/*.json`
(`manual_tobii_reference_xy_mm`, `manual_coppelia_reference_xy_m`) — this
JSON is what the gaze-assisted controller and several MATLAB scripts load
to convert Tobii gaze into robot base-frame coordinates.

`ur5e_move_xy_1cm.py` is for adjusting the zooming of coppeliasim with actual robot movement, after executing the command robot moves exactly 1cm in x and y direction. By measuring that on screen we can make sure the ratio of movement is 1.

`test.py`, `test_tobii.py` are standalone smoke-test/manual-jog utilities, not part of the experiment pipeline.

### Data layout (`data/`)

- `raw_data/` — CSVs written directly by the Python controllers (one row
  per control-loop sample). Read-only from everything else.
- `processed_data/` — derived CSVs written by MATLAB analysis scripts
  (fixation transitions, lead/lag, etc.).
- `calibrations/` — per-setup Tobii↔CoppeliaSim/robot JSON from the
  calibration script.
- `shapes/` — CoppeliaSim scene (`.ttt`) and mesh (`.obj`) files for each
  experiment shape (ellipse, protractor, irregular, random, ...).

File naming encodes the experiment: `<script>_<shape>_s<subject>_d<param>.csv`
(e.g. `ur5e_fixed_adm_2d_ellipse_s1_d50.csv`), and each `ur5e_*` raw CSV has
a matching `tobii_*` CSV from the same trial (same `time_s` origin) — MATLAB
scripts expect the pair, not either file alone.

## Git

`main` is the stable line. Experimental work happens on feature branches
(e.g. `fixation_buffering`) branched off `main`. Before assuming a branch
has drifted far, check the actual topology —
`git rev-list --count main..<branch>` / `<branch>..main` — divergence here
has so far been small (a handful of commits), not a deep fork. Independent
fixes (touching files/logic identical on both branches) propagate cleanly
with `git cherry-pick`; changes to branch-specific logic need to be
hand-applied on each branch separately.
