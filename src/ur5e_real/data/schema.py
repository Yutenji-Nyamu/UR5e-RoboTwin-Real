SCHEMA_VERSION = 1

ACTION_COLUMNS = [
    "controller_time_s",
    "tcp_x",
    "tcp_y",
    "tcp_z",
    "tcp_rx",
    "tcp_ry",
    "tcp_rz",
    "gripper_state",
    "gripper_event_counter",
]

# RoboTwin ACT expects three camera keys. The current workcell has two physical
# cameras, so the left wrist observation intentionally duplicates the wrist view.
CAMERA_GROUP_MAP = {
    "head_camera": "head",
    "right_camera": "wrist",
    "left_camera": "wrist",
}
