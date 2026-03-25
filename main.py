import cv2
import numpy as np
from ultralytics import YOLO
from scipy.signal import find_peaks

# ==========================
# 1. 初始化模型
# ==========================
# 使用轻量pose模型（后续可以换成RKNN版本）
model = YOLO("yolov8n-pose.pt")

# ==========================
# 2. 参数设置
# ==========================
WINDOW_SIZE = 30        # 滑动窗口长度（帧）
PEAK_DISTANCE = 10      # 峰之间最小距离（防止重复计数）
PEAK_PROMINENCE = 15    # 峰值显著性（过滤噪声）

# 存储脚踝Y坐标（时间序列）
ankle_y_history = []

# 计数器
jump_count = 0
last_peak_index = -100  # 防止重复计数

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

        # COCO关键点索引：
        # 左脚踝 = 15, 右脚踝 = 16
        left_ankle_y = keypoints[15][1]
        right_ankle_y = keypoints[16][1]

        # 取平均（更稳）
        ankle_y = (left_ankle_y + right_ankle_y) / 2

        ankle_y_history.append(ankle_y)

        # ==========================
        # 5. 滑动窗口峰值检测
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
            # 6. 计数逻辑
            # ==========================
            for peak in peaks:
                global_index = frame_id - WINDOW_SIZE + peak

                # 防止重复计数
                if global_index - last_peak_index > PEAK_DISTANCE:
                    jump_count += 1
                    last_peak_index = global_index

                    print(f"检测到跳跃！当前计数: {jump_count}")

    # ==========================
    # 7. 可视化
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

    cv2.imshow("Jump Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()