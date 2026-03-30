import numpy as np


LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


def _safe_distance(points, confs, idx_a, idx_b, conf_threshold):
    if confs[idx_a] <= conf_threshold or confs[idx_b] <= conf_threshold:
        return None
    return float(np.linalg.norm(points[idx_a] - points[idx_b]))


def _body_scale(points, confs, conf_threshold):
    shoulder_width = _safe_distance(points, confs, LEFT_SHOULDER, RIGHT_SHOULDER, conf_threshold)
    if shoulder_width is not None and shoulder_width > 1e-6:
        return shoulder_width

    hip_width = _safe_distance(points, confs, LEFT_HIP, RIGHT_HIP, conf_threshold)
    if hip_width is not None and hip_width > 1e-6:
        return hip_width

    return 1.0


def normalize_keypoints(points, confs, conf_threshold=0.5):
    points = np.asarray(points, dtype=np.float32)
    confs = np.asarray(confs, dtype=np.float32)

    if (
        confs[LEFT_HIP] > conf_threshold
        and confs[RIGHT_HIP] > conf_threshold
    ):
        center = (points[LEFT_HIP] + points[RIGHT_HIP]) / 2.0
    elif (
        confs[LEFT_SHOULDER] > conf_threshold
        and confs[RIGHT_SHOULDER] > conf_threshold
    ):
        center = (points[LEFT_SHOULDER] + points[RIGHT_SHOULDER]) / 2.0
    else:
        center = np.mean(points, axis=0)

    scale = _body_scale(points, confs, conf_threshold)
    normalized = (points - center) / max(scale, 1e-6)
    return normalized


def extract_window_features(window_points, window_confs, conf_threshold=0.5):
    window_points = np.asarray(window_points, dtype=np.float32)
    window_confs = np.asarray(window_confs, dtype=np.float32)

    normalized_frames = [
        normalize_keypoints(points, confs, conf_threshold)
        for points, confs in zip(window_points, window_confs)
    ]
    normalized_frames = np.asarray(normalized_frames, dtype=np.float32)

    left_ankle_y = normalized_frames[:, LEFT_ANKLE, 1]
    right_ankle_y = normalized_frames[:, RIGHT_ANKLE, 1]
    mean_ankle_y = (left_ankle_y + right_ankle_y) / 2.0

    left_wrist = normalized_frames[:, LEFT_WRIST, :]
    right_wrist = normalized_frames[:, RIGHT_WRIST, :]
    left_elbow = normalized_frames[:, LEFT_ELBOW, :]
    right_elbow = normalized_frames[:, RIGHT_ELBOW, :]
    left_knee = normalized_frames[:, LEFT_KNEE, :]
    right_knee = normalized_frames[:, RIGHT_KNEE, :]

    left_forearm_len = np.linalg.norm(left_wrist - left_elbow, axis=1)
    right_forearm_len = np.linalg.norm(right_wrist - right_elbow, axis=1)
    left_shank_len = np.linalg.norm(normalized_frames[:, LEFT_ANKLE, :] - left_knee, axis=1)
    right_shank_len = np.linalg.norm(normalized_frames[:, RIGHT_ANKLE, :] - right_knee, axis=1)

    wrist_center_y = (left_wrist[:, 1] + right_wrist[:, 1]) / 2.0
    wrist_center_x = (left_wrist[:, 0] + right_wrist[:, 0]) / 2.0
    ankle_delta_y = np.diff(mean_ankle_y)
    wrist_delta_y = np.diff(wrist_center_y)
    wrist_delta_x = np.diff(wrist_center_x)

    features = np.array(
        [
            np.mean(mean_ankle_y),
            np.std(mean_ankle_y),
            np.max(mean_ankle_y) - np.min(mean_ankle_y),
            np.mean(np.abs(ankle_delta_y)),
            np.std(ankle_delta_y) if len(ankle_delta_y) else 0.0,
            np.mean(left_forearm_len),
            np.std(left_forearm_len),
            np.mean(right_forearm_len),
            np.std(right_forearm_len),
            np.mean(np.abs(np.diff(left_forearm_len))),
            np.mean(np.abs(np.diff(right_forearm_len))),
            np.mean(wrist_center_y),
            np.std(wrist_center_y),
            np.mean(np.abs(wrist_delta_y)),
            np.mean(np.abs(wrist_delta_x)),
            np.std(wrist_delta_y) if len(wrist_delta_y) else 0.0,
            np.std(wrist_delta_x) if len(wrist_delta_x) else 0.0,
            np.mean(left_shank_len),
            np.mean(right_shank_len),
            np.mean(np.abs(left_ankle_y - right_ankle_y)),
            np.mean(np.abs(left_wrist[:, 1] - right_wrist[:, 1])),
            np.mean(window_confs[:, LEFT_WRIST]),
            np.mean(window_confs[:, RIGHT_WRIST]),
            np.mean(window_confs[:, LEFT_ANKLE]),
            np.mean(window_confs[:, RIGHT_ANKLE]),
        ],
        dtype=np.float32,
    )
    return features


FEATURE_NAMES = [
    "ankle_y_mean",
    "ankle_y_std",
    "ankle_y_range",
    "ankle_y_speed_mean",
    "ankle_y_speed_std",
    "left_forearm_len_mean",
    "left_forearm_len_std",
    "right_forearm_len_mean",
    "right_forearm_len_std",
    "left_forearm_change_mean",
    "right_forearm_change_mean",
    "wrist_center_y_mean",
    "wrist_center_y_std",
    "wrist_center_y_speed_mean",
    "wrist_center_x_speed_mean",
    "wrist_center_y_speed_std",
    "wrist_center_x_speed_std",
    "left_shank_len_mean",
    "right_shank_len_mean",
    "ankle_lr_gap_mean",
    "wrist_lr_gap_mean",
    "left_wrist_conf_mean",
    "right_wrist_conf_mean",
    "left_ankle_conf_mean",
    "right_ankle_conf_mean",
]
