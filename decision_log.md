# Decision Log

** chain of reasoning that justifies a modeling choice! **

Running record of decisions, findings, assumptions, actions and outputs,
kept as the project's context backup. Claude appends an entry after each
one, without asking, and reads this file first when session context is
lost. Git history still records *what* changed in code; entries here
record the why, the evidence, and where the output lives.

Entry format: `### YYYY-MM-DD [tag] — title`, then 2-6 terse lines.
Tags: decision, finding, assumption, action, open. Every number names its
source (script + file). Numbers from chat or scratchpad analysis are
marked "not paper-grade" (see the 2026-09-12 rule below).

Entries before 2026-09-11 are reconstructed retroactively from project
history, not contemporaneous. 2026-09-14 to 2026-11-01 were not logged
(see git log). Entries 2026-11-02 to 2026-11-05 were backfilled on
2026-11-05 from session notes; their days are approximate.

## Current state (overwritten in place; last updated 2026-11-05)
- Pipeline now: hand labels (0) -> W 1 s features (1) -> hold-out split
  (2) -> logistic regression (3), all offline; controller runs the model
  online (4); MATLAB analysis of those sessions (5), offline.
  PIPELINE.md covers only the older LOSO/LOSHO pipeline.
- Decided 2026-11-05: "Fine" = labeler judgment. A trial's own hand
  labels are its ground truth; fine_mask_fun geometry is only a proxy.
- Open: hand-label the 66 Gated test trials (the labeling tool needs a
  path/naming change first; user said not yet, 2026-11-05).
- Open: all labels are from one labeler, the user (an author); no
  agreement data exists. An independent second labeler on a subset is
  needed for the paper.
- Open: paper aims to predict regime changes before the force does; the
  lag of P(Fine) vs force has not been measured.
- In progress (2026-11-05): new training target (label at window end +
  lead, 0.1 s step, no purity filter, small margin around the target
  time). User drafts the build_modality_feature_datasets.py change; Claude
  debugs. Test = offline evaluation on held-out windows, then online.
- Later candidates: recency weighting inside the window (lag only);
  threshold toward Fine (asymmetry).
- Next: user runs d_1029 (ROC) and d_1019_feature_timing_at_regime_change.m;
  fix the s24/s23 path mismatch; check the September lead-time sweep.
- Not paper-grade yet: the 0.6 s lag and online AUC numbers.
- On hold, resume only if asked: eq. (7)/(11) offline replay; all-trials
  guidance analysis.
- Offered, not started: all-trials AUC script; lag-shifted ROC in d_1029;
  offline replay script; current-pipeline section in PIPELINE.md; moving
  the lag-check scripts into the repo.

---

### 2026-09-07 — Switch from Otsu/automatic labeling to manual labeling
Damping `B` is a deterministic function of filtered force magnitude (Chen
controller law), so an Otsu threshold on force-derived signal produces
labels that are circular with force-derived features — the classifier
would just decode the control law, not detect task demand. Switched to
hand-labeling from trace/video replay instead.

### 2026-09-08 — Exclude positional + force features (robot, and mirrored gaze positional)
Same root cause as above: TCP position and force enter the admittance law
directly, so leaving them in lets the classifier trivially recover the
label from the control law rather than from task-demand signal. Split into
`POSITIONAL_FEATURE_NAMES` / `FORCE_FEATURE_NAMES` with independent
toggles so future confound categories can be added without touching call
sites.

### 2026-09-08 — Rank features per-fold, not on pooled data, before top-N selection
Ranking on pooled data before splitting leaks the held-out LOSO subject's
information into the feature list. Every optimal-feature-count and top-N
result reflects only the training folds.

### 2026-09-09 — Add LOSHO (leave-one-shape-out) alongside LOSO
LOSO tests generalization to a new subject on shapes already seen in
training — it doesn't test generalization to unseen task geometry, which
is the actual claim of interest. Per-shape class balance is uneven
(ellipse 41/59, fine_gross 77/23, pear 20/80), so per-shape majority
baseline is reported alongside accuracy; a single global baseline would be
misleading given the imbalance.

### 2026-09-09 — Merge build_modality_feature_dataset_all.py into build_modality_feature_datasets.py
Reduces pipeline complexity by one fewer script to keep in sync. Verified
byte-identical output (64 windows x 262 columns, 0 cell differences)
before switching over.

### 2026-09-11 — Add persistence baseline ("assume label didn't change")
Majority-class baseline alone is misleading for anticipation: at short
lead times the label mostly hasn't changed yet, so persistence reaches
~100% and is the real bar the classifier must beat, not majority-class.
Required for the lead-time sweep and for LOSHO's imbalanced per-shape test
sets.

### 2026-09-11 — Fix lead_time_sweep_s / window_step_s mismatch in compare_loso_feature_sets.py
`window_step_s` changed to 2.0 s but the lead-time sweep still had a stale
local value (0.5), silently doubling every reported lead time. Separately,
`round(lead_s / window_step_s)` was collapsing sweep points that should
stay distinct (1s->0, 3s->2), so sweep points were changed to (0, 2, 4, 6)
s — exact multiples of window_step_s.

### 2026-09-11 — Extend compare_loso_feature_sets.py in place for lead-time analysis, not a new script
Initially built as a standalone script; reverted to modifying the existing
one, since duplicate lead-time logic living in two places drifts out of
sync (e.g. the window_step_s bug above would then need fixing twice).

### 2026-09-11 — Removed "significantly improves the VAC controller" claim from abstract draft
Checked against raw_data: 0 trials exist under the gaze-assisted
closed-loop controller (`ur5e_gaze_assisted_adm_2d`) — only the reactive
Chen controller has trials. No closed-loop experiment exists to support a
controller-improvement claim. Replaced with the supported claim: gaze
extends the horizon over which task demand is predictable (W_2s lead-time
sweep: +2.7 pts at 4s, +4.4 pts at 6s over robot-only).

### 2026-09-12 — Persistence baseline is degenerate at window offsets 0 and 1
At offset 0 it scores the label against itself. At offset 1, with contiguous
non-overlapping windows (window_step_s >= window_size_s), the purity filter
removes every window containing a label change, so no surviving consecutive
pair can disagree. Confirmed: exactly 1.000 in all 16 fold-rows at 0 s and
2 s, versus 0.48-0.71 at 4 s and 6 s. build_paper_results.py flags these
horizons and writes NaN for the vs-persistence statistics instead of
quoting a baseline that carries no information.

### 2026-09-12 — Paper numbers come from files, never from analysis run in chat
Several reported values (Chen damping AUC, the Fine/Gross damping
difference, every p-value) existed only in a terminal transcript and were
not reproducible or checkable. build_paper_results.py now regenerates every
quoted number into paper_fig*/paper_tab* CSVs plus paper_claims_W_2s.csv,
which maps each claim to its source file, column, and row filter.

### 2026-09-12 — Report per-fold mean, not pooled accuracy
Pooled (micro) LOSO accuracy concatenates every fold's predictions and
scores once (robot: 98.01%); the mean of per-fold accuracies is 98.11%.
Only per-fold values support the paired across-subject tests, so the paper
uses per-fold mean +/- SD throughout and never mixes the two conventions.

### 2026-09-12 — Analysed trial count is 86, not 103
103 trials were recorded; 12 belong to excluded s18; of the remaining 91,
five produced zero windows in the built dataset: s10 narrow/ellipse/cw,
s11 wide/ellipse/cw, s13 wide/fine_gross/cw, s15 narrow/pear/cw and
narrow/pear/ccw. s13's is explained (no tobii label file). The other four
have valid labels, 94.6-99.8% gaze validity, full 120 s duration, and
normal robot/gaze stream overlap -- cause not yet identified.

### 2026-09-13 — s18 reinstated with a new recording session
A new s18 session was recorded and manually labeled today: 24 raw files
+ 24 labeled files, full 12-condition coverage (3 shapes x 2 bandwidths
x 2 directions), all robot/tobii pairs present -- verified clean, unlike
the exhaustive-check issues found in some s10-s17 trials. This is a
DIFFERENT recording session from the s18 excluded on 2026-09-11 (that
exclusion reason -- uniform "fine" behavior throughout, performance well
below other subjects -- was never independently confirmed as a
pre-registered, classifier-independent criterion; see that entry). s18
is included going forward: expected_subjects in build_paper_results.py
updated to (10..18). Steps 2-5 must be re-run to fold it into any
result; nothing downstream reflects it yet.

### 2026-11-02 [finding] — 1win_ahead label shift is a no-op (hold-out pipeline)
`num_win_label_moveforward = 1` in build_holdout_feature_dataset.py
changes 0 of 4,953 window labels in features_all_{train,test}_W_1s.csv
and drops 2,054 windows (29%). Same mechanism as the 2026-09-12
persistence entry: the purity filter plus the 0.5 s post-change margin
leave no adjacent pair with different labels. Every
*_1win_ahead_W_1s.joblib model, including those run online, is a
current-phase classifier with no anticipation.

### 2026-11-02 [finding] — Online P(Fine) lags the TCP by ~0.6 s
Per Gated trial: AUC of P(Fine)(t + L) vs a reference r(t), for L = -3..3 s
in 0.1 s steps; best lag = argmax. Geometry reference (fine_mask_fun),
66 trials: median 0.6 s, 62/66 positive, AUC 0.83 at L = 0, 0.92 at best L.
Hand labels moved by position (50-NN majority, s11, s14-18, 36 trials):
median 0.6 s (0.3-1.0), AUC 0.84 -> 0.95. Not paper-grade: scratchpad
check_pfine.py and check_labels.py (temp dir, not in repo).

### 2026-11-03 [finding] — The lag comes from the training target
Each window is labeled with its own label, so online the target is the
majority label of the trailing 1 s window: lag ~ W/2 + 10 Hz hold
(0.57 s measured). Aligned on label changes, P(Fine) crosses 0.5 at
+0.76 s entering Fine and +0.28 s leaving it. Removing the transition
filter alone would not remove the lag. Not paper-grade: scratchpad
check_window_lag.py.

### 2026-11-03 [finding] — Offline hold-out AUC overstates online performance
Held-out-shape test AUC is 0.96-0.97 (predictions CSVs), but those test
windows exclude every transition, where the online errors are. Online vs
geometry: 0.83 at L = 0. Outliers: s21 (P(Fine) high everywhere); s19
and s20 narrow pear have the lowest AUCs (0.65-0.71).

### 2026-11-04 [decision] — Hand labels dropped from d_1029; geometry is the test reference
Hand labels exist only for the training-session Chen recordings (s10-18),
not the test trials, and s23 has none. Moving them by position onto a
different recording was judged meaningless (user's call), so all label
code was removed. Gated test trials are scored against fine_mask_fun
(fine_gross, pear: x > 0.06; ellipse: x > 0.08 or x < -0.11).

### 2026-11-04 [action] — d_1029_plot_damping_vs_position_chen_vs_gated.m updated
Added a per-lap figure (laps from lap_times_by_start_crossing_fun) and a
per-trial ROC of Gated P(Fine) vs the geometric fine area (perfcurve);
removed the label code. Output: figures only. Uncommitted.

### 2026-11-05 [open] — Test plan for a retrained classifier (recommended, not confirmed)
First an offline replay: run the model every 0.1 s on trailing windows
over the hand-labeled s10-18 Chen recordings (same-trial labels); measure
switch times, AUC with transitions, false switches. The old model should
reproduce ~0.6 s. Then new online Gated sessions, which alone support
controller-performance claims. The replay script does not exist yet.

### 2026-11-05 [open] — What "Fine" means: labeler judgment or geometric region
Sets the training target and whether test trials need hand labels. On the
labeled recordings, labels vs geometry agree 0.80 on fine_gross (65% vs
49% Fine) and ~0.90 on pear and ellipse, with a 0.0-0.1 s timing offset.
The labeling tool shows only the TCP path. If labeler judgment: label the
Gated trials blind, with a second labeler on 10-20%. No inter-rater data
is known for the training labels.

### 2026-11-05 [open] — Controller output paths mismatch (s24 robot, s23 Tobii)
ur5e_gaze_assisted_chen_vac_2d.py line 275 writes the
..._s24_on_narrow_pear_shape robot CSV; line 290 writes the
..._s23_on_narrow_pear_shape Tobii CSV. Fix before the next recording.

### 2026-11-05 [assumption] — Which model each test subject ran
From the controller's comments: s11 and s14-18 ran held-out shape+subject
models; s19-23 ran held-out shape only. Not checked against session logs.

### 2026-11-05 [open] — Check the September lead-time sweep before a new target
compare_loso_feature_sets.py already labels at a lead (0/2/4/6 s, W 2 s);
4 and 6 s were non-degenerate (2026-09-11 and 09-12 entries). Check how
it built its targets before writing a new one.

### 2026-11-05 [decision] — decision_log.md becomes the context backup
User asked that decisions, findings, assumptions, actions and outputs be
logged here from now on, without asking, and read after context loss.
Header rewritten for the wider scope; "Current state" block added.

### 2026-11-05 [decision] — "Fine" is the labeler's judgment; geometry is a proxy
User's decision. Matches writing/test/main.tex line 313 (labels judged
visually "with the local geometry of the shape as reference"). A trial's
own hand labels are its ground truth; fine_mask_fun results (0.6 s lag,
AUC 0.83) remain a proxy check. Supersedes the reference choice in the
2026-11-04 entry; not using labels moved from other recordings stands.
Consequence: scoring the online classifier needs hand-labeled Gated trials.

### 2026-11-05 [finding] — Labeling tool can't read the test-session files as is
manual_label_chen_var_adm_2d_by_live_replay.py reads data/raw_data/ with
the training-session file naming; test trials are in
raw_data/test_subjects/ under other names. Labeled files exist for s10-18
only. Current playback settings: 80x, stride 80 samples; whether the
training labels used these settings is unknown.

### 2026-11-05 [open] — Paper's anticipation aim vs force not yet measured
main.tex line 296 (as of 2026-11-05): "aiming to predict regime changes
before they appear in the interaction force". Online P(Fine) lags the TCP
by ~0.6 s (geometry and labels); its timing relative to force changes has
not been measured.

### 2026-11-05 [finding] — All hand labels come from one labeler, the user
s10-18 were labeled by the user alone, who is also an author and knows the
hypotheses. No inter-rater data exists. An independent second labeler on
a subset (training and test trials) is needed to report agreement;
re-labeling by the same person only gives intra-rater consistency.

### 2026-11-05 [decision] — Labeling tool not adapted for test trials yet
User's call: keep manual_label_chen_var_adm_2d_by_live_replay.py as is
for now. The Gated test trials stay unlabeled until this is revisited.

### 2026-11-05 [finding] — The model is dominated by a reactive robot feature
Standardized LR coefficients (3 ellipse held-out models; coef_ is for
"Gross"): robot_direction_concentration +4.3 to +4.6; gaze path
efficiency +1.2, gaze p90 gaze-EE distance +1.3 to +1.5, robot
std_curvature -1.1 to -1.2. Medians over all windows, Fine vs Gross:
direction concentration 0.83 vs 0.99, p90 distance 0.019 vs 0.034 m,
path efficiency 0.21 vs 0.40, std curvature 335 vs 59 (1/m). Direction
concentration measures how much the TCP turned in the last window, so it
can only react after the TCP is in the curvy section.

### 2026-11-05 [open] — Recency weighting inside the window (user's idea)
Weight samples by age (linear ramp, newest = 1, oldest = 0). For a feature
that is a plain average, a step reaches half weight at 0.29 W instead of
0.5 W, and 3/4 weight at 0.5 W instead of 0.75 W. Noise std is x1.15,
the same as a uniform 0.75 W window. The four deployed features aren't
plain averages, so this is a guide only. Expected: lag ~0.6 -> ~0.4 s,
no anticipation. Works with pure-window training (no target change).
Needs the same weights in classifier_feature_bank for training and the
controller, then a retrain. Not started.

### 2026-11-05 [decision] — Retrain with a target at the window end (+ lead)
User chose this over weighting first. Plan for build_modality_feature_
datasets.py: step 0.1 s (the controller's 10 Hz); target = per-sample
label at window end + lead (lead 0 first, then a sweep); drop the purity
filter (windows may contain a change); drop only windows whose target is
None or within a small margin of a change. Keep W = 1 s on the first run
so only the target changes. In stage 2 set num_win_label_moveforward = 0
(the lead now lives in stage 1). User drafts, Claude debugs.

### 2026-11-05 [finding] — A 0.5 s post-change margin would defeat the new target
With the target at the window end, dropping targets in the 0.5 s after a
change removes exactly the "just entered Fine" windows the model must
learn to call early. Labels vs geometry differ by only 0.0-0.1 s, so a
small margin (about +/-0.2 s, or none) is enough. Test windows should
keep everything except None targets, so their predictions act as the
offline replay.
