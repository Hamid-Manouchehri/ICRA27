# Decision Log

** chain of reasoning that justifies a modeling choice! **

Why, not what. Manifests in `data/json/` and git history already capture
*what* ran and *what* changed; this file is for the reasoning behind a
choice, which doesn't survive anywhere else. Add an entry whenever a
modeling/methodology choice is made, right after making it. Keep entries
to 2-4 lines: what was decided + the actual reason.

Entries before 2026-09-11 are reconstructed retroactively from project
history, not contemporaneous.

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
