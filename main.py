import cv2
import numpy as np
from ultralytics import YOLO
from scipy.signal import find_peaks

# ==========================
# 1. 初始化模型
# ==========================
# 使用轻量pose模型（后续可以换成RKNN版本）
model = YOLO("yolov8n-pose.pt")

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

# ==========================
# 2. 参数设置
# ==========================
WINDOW_SIZE = 30        # 滑动窗口长度（帧）
PEAK_DISTANCE = 10      # 峰之间最小距离（防止重复计数）
PEAK_PROMINENCE = 15    # 峰值显著性（过滤噪声）
KEYPOINT_CONF_THRESHOLD = 0.5
WRIST_DISTANCE_CHANGE_THRESHOLD = 0.03   # 归一化距离变化阈值，越大说明手腕摆动越明显
WRIST_WINDOW_RADIUS = 4              # 在跳跃峰附近检查手腕动作

# 存储脚踝Y坐标（时间序列）
ankle_y_history = []
wrist_motion_history = []

prev_left_wrist_distance = None
prev_right_wrist_distance = None

# 计数器
jump_count = 0
rope_jump_count = 0
last_peak_index = -100  # 防止重复计数


# 旧方案：用手肘到手腕连线的角度变化来判断甩绳
# def angle_diff_rad(current_angle, previous_angle):
#     diff = current_angle - previous_angle
#     return np.arctan2(np.sin(diff), np.cos(diff))
#
#
# def get_limb_angle(keypoints, confs, start_idx, end_idx):
#     if confs[start_idx] <= KEYPOINT_CONF_THRESHOLD or confs[end_idx] <= KEYPOINT_CONF_THRESHOLD:
#         return None
#
#     start_point = keypoints[start_idx]
#     end_point = keypoints[end_idx]
#     return np.arctan2(end_point[1] - start_point[1], end_point[0] - start_point[0])


def get_point_distance(keypoints, confs, start_idx, end_idx):
    if confs[start_idx] <= KEYPOINT_CONF_THRESHOLD or confs[end_idx] <= KEYPOINT_CONF_THRESHOLD:
        return None

    start_point = keypoints[start_idx]
    end_point = keypoints[end_idx]
    return float(np.linalg.norm(end_point - start_point))


def get_body_scale(keypoints, confs):
    left_shoulder = 5
    right_shoulder = 6
    left_hip = 11
    right_hip = 12

    if confs[left_shoulder] > KEYPOINT_CONF_THRESHOLD and confs[right_shoulder] > KEYPOINT_CONF_THRESHOLD:
        return float(np.linalg.norm(keypoints[left_shoulder] - keypoints[right_shoulder]))

    if confs[left_hip] > KEYPOINT_CONF_THRESHOLD and confs[right_hip] > KEYPOINT_CONF_THRESHOLD:
        return float(np.linalg.norm(keypoints[left_hip] - keypoints[right_hip]))

    return 1.0

# ==========================
# 3. 视频读取
# ==========================
#cap = cv2.VideoCapture(0)  # 摄像头
cap = cv2.VideoCapture("D:\\work\\Intern\\Dataset\\SingleMan.mp4")  # 视频文件
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # ==========================
    # 4. Pose检测
    # ==========================
    results = model(frame, verbose=False)

    if len(results[0].keypoints) > 0:
        # 取第一个人（单人阶段）
        keypoints = results[0].keypoints.xy[0].cpu().numpy()
        keypoint_conf = results[0].keypoints.conf[0].cpu().numpy()

        # 先画骨架，再画关键点
        for start_idx, end_idx in SKELETON:
            if keypoint_conf[start_idx] > KEYPOINT_CONF_THRESHOLD and keypoint_conf[end_idx] > KEYPOINT_CONF_THRESHOLD:
                start_point = tuple(keypoints[start_idx].astype(int))
                end_point = tuple(keypoints[end_idx].astype(int))
                cv2.line(frame, start_point, end_point, (255, 200, 0), 2)

        for idx, (x, y) in enumerate(keypoints):
            if keypoint_conf[idx] > KEYPOINT_CONF_THRESHOLD:
                point = (int(x), int(y))
                cv2.circle(frame, point, 4, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    f"{idx}",
                    (point[0] + 5, point[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1
                )

        # ==========================
        # 5. 手腕摆动检测
        # ==========================
        # 旧方案：角度变化检测
        # left_wrist_angle = get_limb_angle(keypoints, keypoint_conf, 7, 9)
        # right_wrist_angle = get_limb_angle(keypoints, keypoint_conf, 8, 10)
        #
        # left_wrist_speed = 0.0
        # right_wrist_speed = 0.0
        #
        # if left_wrist_angle is not None and prev_left_wrist_angle is not None:
        #     left_wrist_speed = abs(angle_diff_rad(left_wrist_angle, prev_left_wrist_angle))
        # if right_wrist_angle is not None and prev_right_wrist_angle is not None:
        #     right_wrist_speed = abs(angle_diff_rad(right_wrist_angle, prev_right_wrist_angle))
        #
        # prev_left_wrist_angle = left_wrist_angle
        # prev_right_wrist_angle = right_wrist_angle
        # wrist_motion_score = (left_wrist_speed + right_wrist_speed) / 2.0

        body_scale = max(get_body_scale(keypoints, keypoint_conf), 1.0)

        left_wrist_distance = get_point_distance(keypoints, keypoint_conf, 7, 9)
        right_wrist_distance = get_point_distance(keypoints, keypoint_conf, 8, 10)

        left_distance_change = 0.0
        right_distance_change = 0.0

        if left_wrist_distance is not None and prev_left_wrist_distance is not None:
            left_distance_change = abs(left_wrist_distance - prev_left_wrist_distance) / body_scale
        if right_wrist_distance is not None and prev_right_wrist_distance is not None:
            right_distance_change = abs(right_wrist_distance - prev_right_wrist_distance) / body_scale

        prev_left_wrist_distance = left_wrist_distance
        prev_right_wrist_distance = right_wrist_distance

        wrist_motion_score = (left_distance_change + right_distance_change) / 2.0
        wrist_motion_history.append(wrist_motion_score)

        # COCO关键点索引：
        # 左脚踝 = 15, 右脚踝 = 16
        left_ankle_y = keypoints[15][1]
        right_ankle_y = keypoints[16][1]

        # 取平均（更稳）
        ankle_y = (left_ankle_y + right_ankle_y) / 2

        ankle_y_history.append(ankle_y)

        # ==========================
        # 6. 滑动窗口峰值检测
        # ==========================
        if len(ankle_y_history) > WINDOW_SIZE:
            # 只保留最近窗口
            window = np.array(ankle_y_history[-WINDOW_SIZE:])

            # ⚠️ 关键：反转（因为图像y向下）
            signal = -window

            peaks, properties = find_peaks(
                signal,
                distance=PEAK_DISTANCE,
                prominence=PEAK_PROMINENCE
            )

            # ==========================
            # 7. 计数逻辑
            # ==========================
            for peak in peaks:
                global_index = frame_id - WINDOW_SIZE + peak

                # 防止重复计数
                if global_index - last_peak_index > PEAK_DISTANCE:
                    jump_count += 1
                    last_peak_index = global_index

                    peak_center = len(wrist_motion_history) - WINDOW_SIZE + peak
                    wrist_start = max(0, peak_center - WRIST_WINDOW_RADIUS)
                    wrist_end = min(len(wrist_motion_history), peak_center + WRIST_WINDOW_RADIUS + 1)
                    wrist_window = wrist_motion_history[wrist_start:wrist_end]
                    wrist_motion_mean = float(np.mean(wrist_window)) if wrist_window else 0.0
                    has_rope_swing = wrist_motion_mean > WRIST_DISTANCE_CHANGE_THRESHOLD

                    if has_rope_swing:
                        rope_jump_count += 1

                    print(
                        f"检测到跳跃！总跳跃: {jump_count}, "
                        f"跳绳判定: {'是' if has_rope_swing else '否'}, "
                        f"跳绳计数: {rope_jump_count}"
                    )

    # ==========================
    # 8. 可视化
    # ==========================
    cv2.putText(
        frame,
        f"Jump Count: {jump_count}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Rope Jump Count: {rope_jump_count}",
        (30, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    wrist_status = "Distance changing" if wrist_motion_history and wrist_motion_history[-1] > WRIST_DISTANCE_CHANGE_THRESHOLD else "Not enough"
    cv2.putText(
        frame,
        f"Wrist Motion: {wrist_status}",
        (30, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow("Jump Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()