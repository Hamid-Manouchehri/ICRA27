import time
import tobii_research as tr


def gaze_callback(data):
    left = data["left_gaze_point_on_display_area"]
    right = data["right_gaze_point_on_display_area"]

    left_valid = data["left_gaze_point_validity"]
    right_valid = data["right_gaze_point_validity"]

    print(
        f"time={data['system_time_stamp']} | "
        f"left={left}, valid={left_valid} | "
        f"right={right}, valid={right_valid}"
    )


trackers = tr.find_all_eyetrackers()

if not trackers:
    raise RuntimeError("No Tobii eye tracker found.")

tracker = trackers[0]

print("Model:", tracker.model)
print("Serial:", tracker.serial_number)
print("Frequency:", tracker.get_gaze_output_frequency(), "Hz")

tracker.subscribe_to(
    tr.EYETRACKER_GAZE_DATA,
    gaze_callback,
    as_dictionary=True,
)

try:
    time.sleep(10)
finally:
    tracker.unsubscribe_from(
        tr.EYETRACKER_GAZE_DATA,
        gaze_callback,
    )