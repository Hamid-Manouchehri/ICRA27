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
