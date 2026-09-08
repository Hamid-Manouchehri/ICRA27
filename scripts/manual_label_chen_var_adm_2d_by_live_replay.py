"""manual_label_chen_var_adm_2d_by_live_replay.py -- Live keyboard-driven
Fine/Gross labeling for ONE manually-selected Chen trial, applied
simultaneously to both the ur5e (robot) and tobii (gaze) CSVs.

Python counterpart to matlab/manual_label_chen_var_adm_2d_by_live_
replay.m -- same interaction model (single press, not hold: press
fine_key once and every sample from then on is Fine until you press
again). Visualization is intentionally minimal -- ONE plot, the TCP
path colored live by your label, nothing else. A damping panel and a
gaze-position overlay were both dropped on request once ~100 trials
needed labeling back to back: every extra plotted artist adds real
per-frame redraw cost, and redraw cost -- not playback_speed_
multiplier -- is what actually limits how fast a trial can run once
the multiplier is already high (see SPEED below).

The tobii gaze CSV is still loaded and labeled/saved alongside the
robot CSV (label transferred by nearest robot sample time) -- only its
VISUAL overlay was removed, not the labeling itself. Since nothing
from it is plotted anymore, its calibration JSON isn't needed here at
all (that JSON only ever fed the now-removed overlay's coordinate
mapping).

WHY A SEPARATE KEYBOARD THREAD, AND TWO KEY-HANDLING PATHS: the MATLAB
version's keystrokes were serviced by its GUI event loop
(drawnow/pause), which under load could intermittently miss a press.
keyboard.Listener (pynput) runs on its OWN OS-level thread, decoupled
from matplotlib's render loop, so a press is captured immediately
regardless of redraw timing -- but it's a SYSTEM-WIDE hook, not
window-scoped (don't type fine_key/gross_key/none_key/quit_key in
another app while this runs), and on some Windows/keyboard-layout
combinations pynput's reported key can come through without a usable
.char. So matplotlib's OWN key_press_event ALSO drives labeling
(on_mpl_key) as a second, independent path -- whichever one actually
receives an event applies the label; both call the same
apply_key_label(). quit_key goes through both paths too, so -- unlike
Ctrl+C, whose delivery depends on which window has OS focus -- it
reliably ends the replay regardless of focus.

SPEED, for getting through ~100 trials -- two independent knobs:
  - playback_speed_multiplier: the WALL-CLOCK TARGET ("finish in
    1/multiplier of the trial's real duration"), enforced by skipping
    pause() when the loop is behind schedule. Past a certain point
    (redraw cost exceeds the per-chunk time budget), raising it
    further does nothing -- the loop is already going as fast as it
    can draw, so the "target" stops being the limiting factor.
  - animation_stride: how many samples advance per redraw. THIS is
    the knob that matters once you're redraw-bound (likely already
    true at high playback_speed_multiplier): fewer, bigger jumps per
    redraw means fewer total redraws for the same trial, and that's
    where the wall-clock time actually goes. Doubling animation_stride
    roughly halves wall-clock time in that regime.
  Real tradeoff on animation_stride: too high and the TCP trail jumps
  in visibly large chunks, making it harder to judge exactly when a
  regime changed -- it's not free speed, it trades against how well
  you can actually watch and decide.

Only ONE trial per run, filename set manually below (this project's
batch/discovery convention lives in the .m version and in
build_modality_feature_datasets.py; this script is for hand-picking a
specific trial, not iterating a whole dataset).

Inputs: data/raw_data/ur5e_chen_var_adm_2d_<trial>.csv (set manually
  below) + its tobii_* counterpart.
Output: data/processed_data/{ur5e,tobii}_chen_var_adm_2d_live_labeled_
  <trial>.csv, written once, when the replay ends (finished, window
  closed early, quit_key, or Ctrl+C).

Dependencies: numpy, pandas, matplotlib, pynput
  pip install numpy pandas matplotlib pynput
"""

import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pynput import keyboard

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIRECTORY.parent
DATA_DIRECTORY = PROJECT_ROOT / "data"
RAW_DATA_DIRECTORY = DATA_DIRECTORY / "raw_data"
PROCESSED_DATA_DIRECTORY = DATA_DIRECTORY / "processed_data"

# -------------------- User Settings --------------------
# TODO the ur5e CSV filename to label, exactly as it appears in
# data/raw_data/ -- the matching tobii_* CSV is derived automatically
# (same trial token, standard naming).
# robot_csv_filename = "ur5e_chen_var_adm_2d_narrow_ellipse_shape_clockwise_s18.csv"  # 1
# robot_csv_filename = "ur5e_chen_var_adm_2d_wide_ellipse_shape_clockwise_s18.csv"  # 2
# robot_csv_filename = "ur5e_chen_var_adm_2d_narrow_ellipse_shape_counter_clockwise_s18.csv"  # 3
# robot_csv_filename = "ur5e_chen_var_adm_2d_wide_ellipse_shape_counter_clockwise_s18.csv"  # 4 
# robot_csv_filename = "ur5e_chen_var_adm_2d_narrow_fine_gross_shape_clockwise_s18.csv"  # 5
# robot_csv_filename = "ur5e_chen_var_adm_2d_wide_fine_gross_shape_clockwise_s18.csv"  # 6
# robot_csv_filename = "ur5e_chen_var_adm_2d_narrow_fine_gross_shape_counter_clockwise_s18.csv"  # 7
# robot_csv_filename = "ur5e_chen_var_adm_2d_wide_fine_gross_shape_counter_clockwise_s18.csv"  # 8
# robot_csv_filename = "ur5e_chen_var_adm_2d_narrow_pear_shape_clockwise_s18.csv"  # 9
# robot_csv_filename = "ur5e_chen_var_adm_2d_wide_pear_shape_clockwise_s18.csv"  # 10
# robot_csv_filename = "ur5e_chen_var_adm_2d_narrow_pear_shape_counter_clockwise_s18.csv"  # 11
robot_csv_filename = "ur5e_chen_var_adm_2d_wide_pear_shape_counter_clockwise_s18.csv"  # 12

overwrite_existing = True  # TODO refuse to run if the output CSV exists

fine_key = "d"
gross_key = "g"
none_key = "n"  # explicit unsure/transition; also the starting state
quit_key = "q"  # saves + ends the replay -- use this, NOT Ctrl+C (see docstring)
fine_label_text = "Fine"
gross_label_text = "Gross"
none_label_text = "None"

debug_key_events = False  # TODO True prints every raw key event (both paths)

fine_color = (0.85, 0.10, 0.10)
gross_color = (0.10, 0.65, 0.20)
none_color = (0.55, 0.55, 0.55)

tcp_marker_size = 10.0

# TODO see the module docstring's SPEED section -- animation_stride is
# the more effective knob once playback_speed_multiplier is already
# high (redraw cost, not the multiplier, becomes the limit).
playback_speed_multiplier = 80.0  # >1 = faster than real time
animation_stride = 80  # samples advanced per redraw -- raise for less wall-clock time
redraw_pause_s = 0.005  # unconditional yield per frame; keeps key capture reliable

# ======================================================================
# Path setup + small CSV/label helpers
# ======================================================================


def get_numeric_column(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def resample_labels_nearest(
        raw_time_s: np.ndarray, raw_label: np.ndarray,
        query_time_s: np.ndarray) -> np.ndarray:
    """Nearest-in-time label transfer, matching the .m version's
    resample_labels_nearest. raw_time_s may be unsorted/duplicated."""
    valid = np.isfinite(raw_time_s)
    clean_time_s = raw_time_s[valid]
    clean_label = raw_label[valid]

    order = np.argsort(clean_time_s, kind="stable")
    clean_time_s = clean_time_s[order]
    clean_label = clean_label[order]
    unique_time_s, unique_indices = np.unique(clean_time_s, return_index=True)
    clean_label = clean_label[unique_indices]

    insert_idx = np.searchsorted(unique_time_s, query_time_s)
    insert_idx = np.clip(insert_idx, 1, len(unique_time_s) - 1)
    left_idx = insert_idx - 1
    left_closer = (query_time_s - unique_time_s[left_idx]) <= \
        (unique_time_s[insert_idx] - query_time_s)
    nearest_idx = np.where(left_closer, left_idx, insert_idx)
    return clean_label[nearest_idx]


def trial_token_from_robot_filename(filename: str) -> str:
    prefix, suffix = "ur5e_chen_var_adm_2d_", ".csv"
    if not (filename.startswith(prefix) and filename.endswith(suffix)):
        raise ValueError(
            f"robot_csv_filename '{filename}' doesn't match the expected "
            f"'{prefix}<trial>{suffix}' pattern.")
    return filename[len(prefix):-len(suffix)]


# ======================================================================
# Thread-safe label/quit state, written by the keyboard listener thread
# AND matplotlib's own key handler, read by the main (plotting) thread
# once per animation frame.
# ======================================================================

class ReplayState:
    def __init__(self, initial_label: str):
        self._lock = threading.Lock()
        self._label = initial_label
        self._quit_requested = False

    def set_label(self, label: str) -> None:
        with self._lock:
            self._label = label

    def request_quit(self) -> None:
        with self._lock:
            self._quit_requested = True

    def snapshot(self) -> tuple:
        with self._lock:
            return self._label, self._quit_requested


def apply_key_label(char: str, state: ReplayState) -> bool:
    """Shared single-press-toggle logic (see the .m version's on_key_
    press): char already lowercased. Returns True if char matched one
    of fine_key/gross_key/none_key/quit_key, so both key-handling paths
    below apply IDENTICAL matching -- one definition, not two copies
    that could silently drift apart."""

    if char == fine_key:
        state.set_label(fine_label_text)
    elif char == gross_key:
        state.set_label(gross_label_text)
    elif char == none_key:
        state.set_label(none_label_text)
    elif char == quit_key:
        state.request_quit()
    else:
        return False
    return True


def make_on_press(state: ReplayState):
    """KeyPressFcn equivalent -- single press sets state going forward,
    same semantics as the .m version's on_key_press (no hold, no
    release needed). Runs on pynput's own listener thread, SYSTEM-WIDE
    (not window-scoped -- see module docstring)."""

    def on_press(key):
        char = getattr(key, "char", None)
        if debug_key_events:
            print(f"[pynput] raw event: {key!r}  ->  char={char!r}")
        if char is None:
            return
        matched = apply_key_label(char.lower(), state)
        if debug_key_events and not matched:
            print(f"[pynput] {char!r} did not match any of "
                  f"{fine_key!r}/{gross_key!r}/{none_key!r}/{quit_key!r}")

    return on_press


def main() -> None:
    robot_csv_file = RAW_DATA_DIRECTORY / robot_csv_filename
    if not robot_csv_file.is_file():
        raise FileNotFoundError(f"Not found: {robot_csv_file}")
    trial_token = trial_token_from_robot_filename(robot_csv_filename)

    gaze_csv_file = RAW_DATA_DIRECTORY / f"tobii_chen_var_adm_2d_{trial_token}.csv"

    write_robot_csv_file = (
        PROCESSED_DATA_DIRECTORY /
        f"ur5e_chen_var_adm_2d_live_labeled_{trial_token}.csv")
    write_gaze_csv_file = (
        PROCESSED_DATA_DIRECTORY /
        f"tobii_chen_var_adm_2d_live_labeled_{trial_token}.csv")
    if not overwrite_existing and write_robot_csv_file.is_file():
        raise FileExistsError(
            f"{write_robot_csv_file} already exists -- set overwrite_existing "
            f"= True to relabel this trial.")
    PROCESSED_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print(f"--- {trial_token} ---")
    robot_table = pd.read_csv(robot_csv_file, keep_default_na=False)
    robot_time_s = get_numeric_column(robot_table["time_s"])
    coppeliasim_x_m = get_numeric_column(robot_table["coppeliasim_x_m"])
    coppeliasim_y_m = get_numeric_column(robot_table["coppeliasim_y_m"])
    sample_count = len(robot_table)

    gaze_available = gaze_csv_file.is_file()
    gaze_table = gaze_time_s = None
    if gaze_available:
        gaze_table = pd.read_csv(gaze_csv_file, keep_default_na=False)
        gaze_time_s = get_numeric_column(gaze_table["time_s"])
    else:
        print(f"WARNING: {trial_token}: no matching tobii CSV -- "
              f"only the robot CSV will be labeled.")

    robot_label = np.full(sample_count, none_label_text, dtype=object)

    # ---- Figure: TCP path only ----
    plt.ion()
    fig, path_axes = plt.subplots(figsize=(7, 7))
    fig.canvas.manager.set_window_title(f"Label: {trial_token}")

    fine_line, = path_axes.plot([], [], ".", color=fine_color,
                                 markersize=tcp_marker_size, label=fine_label_text)
    gross_line, = path_axes.plot([], [], ".", color=gross_color,
                                  markersize=tcp_marker_size, label=gross_label_text)
    none_line, = path_axes.plot([], [], ".", color=none_color,
                                 markersize=tcp_marker_size, label=none_label_text)
    current_marker, = path_axes.plot(
        [], [], "o", markersize=tcp_marker_size + 6,
        markerfacecolor=none_color, markeredgecolor="k")
    path_axes.set_xlim(coppeliasim_x_m.min() - 0.02, coppeliasim_x_m.max() + 0.02)
    path_axes.set_ylim(coppeliasim_y_m.min() - 0.02, coppeliasim_y_m.max() + 0.02)
    path_axes.set_aspect("equal")
    path_axes.set_xlabel("x [m]")
    path_axes.set_ylabel("y [m]")
    path_axes.grid(True)
    path_axes.legend(loc="best", fontsize=8)
    path_axes.set_title(
        f"Click this window, then press any key to start -- "
        f"press '{fine_key}'={fine_label_text}, '{gross_key}'={gross_label_text}, "
        f"'{none_key}'={none_label_text} (starts as {none_label_text}), "
        f"'{quit_key}'=quit+save")

    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()

    state = ReplayState(none_label_text)
    listener = keyboard.Listener(on_press=make_on_press(state))
    listener.start()

    def save_labeled_csvs() -> None:
        robot_table["label"] = robot_label
        robot_table.to_csv(write_robot_csv_file, index=False)

        saved_gaze = False
        if gaze_available:
            gaze_table["label"] = resample_labels_nearest(
                robot_time_s, robot_label, gaze_time_s)
            gaze_table.to_csv(write_gaze_csv_file, index=False)
            saved_gaze = True

        labels, counts = np.unique(robot_label, return_counts=True)
        counts_text = ", ".join(f"{l}={c}" for l, c in zip(labels, counts))
        print(f"Saved {trial_token}: {counts_text}")
        print(f"  -> {write_robot_csv_file}")
        if saved_gaze:
            print(f"  -> {write_gaze_csv_file}")

    print("Click the plot window, then press any key to start replay.")
    print(f"Press once: {fine_key}={fine_label_text}, {gross_key}={gross_label_text}, "
          f"{none_key}={none_label_text}, {quit_key}=quit+save.")
    if debug_key_events:
        print("[debug] debug_key_events=True: every raw key event will be "
              "printed below, from both the pynput and matplotlib paths.")

    def on_mpl_key(event) -> None:
        """SECOND, independent key-handling path -- see module
        docstring's WHY A SEPARATE KEYBOARD THREAD section."""
        started.set()
        key_name = (event.key or "").lower()
        if debug_key_events:
            print(f"[matplotlib] raw event: key={event.key!r}")
        if not key_name:
            return
        matched = apply_key_label(key_name, state)
        if debug_key_events and not matched:
            print(f"[matplotlib] {event.key!r} did not match any of "
                  f"{fine_key!r}/{gross_key!r}/{none_key!r}/{quit_key!r}")

    try:
        # Focus-safe start gate: wait for the user to click the figure
        # or press a key ON it, mirroring the .m version's
        # waitforbuttonpress, before playback (and its wall-clock
        # pacing) begins. key_press_event also drives labeling from
        # here on (on_mpl_key just above).
        started = threading.Event()
        fig.canvas.mpl_connect("button_press_event", lambda _event: started.set())
        fig.canvas.mpl_connect("key_press_event", on_mpl_key)
        while not started.is_set() and plt.fignum_exists(fig.number):
            fig.canvas.flush_events()
            time.sleep(0.02)

        replay_start_time = time.perf_counter()
        trial_start_s = robot_time_s[0]

        for sample_idx in range(0, sample_count, animation_stride):
            if not plt.fignum_exists(fig.number):
                break
            label, quit_requested = state.snapshot()
            if quit_requested:
                break

            chunk_end_idx = min(sample_idx + animation_stride, sample_count) - 1
            robot_label[sample_idx:chunk_end_idx + 1] = label

            shown_x = coppeliasim_x_m[:chunk_end_idx + 1]
            shown_y = coppeliasim_y_m[:chunk_end_idx + 1]
            shown_label = robot_label[:chunk_end_idx + 1]

            fine_line.set_data(
                np.where(shown_label == fine_label_text, shown_x, np.nan),
                np.where(shown_label == fine_label_text, shown_y, np.nan))
            gross_line.set_data(
                np.where(shown_label == gross_label_text, shown_x, np.nan),
                np.where(shown_label == gross_label_text, shown_y, np.nan))
            none_line.set_data(
                np.where(shown_label == none_label_text, shown_x, np.nan),
                np.where(shown_label == none_label_text, shown_y, np.nan))

            current_color = {fine_label_text: fine_color,
                              gross_label_text: gross_color,
                              none_label_text: none_color}[label]
            current_marker.set_data(
                [coppeliasim_x_m[chunk_end_idx]], [coppeliasim_y_m[chunk_end_idx]])
            current_marker.set_markerfacecolor(current_color)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            # Unconditional yield every frame regardless of pacing slack
            # below (see module docstring): at high playback_speed_
            # multiplier there is often no slack at all, and without
            # SOME sleep here the OS/GUI event queues can still starve
            # intermittently.
            time.sleep(redraw_pause_s)

            now_s = robot_time_s[chunk_end_idx]
            trial_time_elapsed_s = now_s - trial_start_s
            target_wall_time_s = trial_time_elapsed_s / playback_speed_multiplier
            remaining_pause_s = target_wall_time_s - (
                time.perf_counter() - replay_start_time)
            if remaining_pause_s > 0:
                time.sleep(remaining_pause_s)

        if plt.fignum_exists(fig.number):
            _, quit_requested = state.snapshot()
            path_axes.set_title(
                "Saved -- quitting (close this window)." if quit_requested
                else "Replay finished -- saved. Close this window when ready.")
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
    finally:
        # Runs on normal completion, an early window close, quit_key,
        # AND Ctrl+C (KeyboardInterrupt unwinds through this try/finally
        # just like any other exception) -- the one guaranteed save path.
        save_labeled_csvs()
        listener.stop()


if __name__ == "__main__":
    main()
