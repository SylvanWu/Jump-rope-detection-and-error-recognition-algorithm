import cv2
import numpy as np
from ultralytics import YOLO
from scipy.signal import find_peaks
from collections import defaultdict, deque
import pickle
from pose_features import extract_window_features
import subprocess
import torch

# ==========================
# 1. 初始化模型
# ==========================
# 检测并设置设备
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")
if device == "cuda:0":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

pose_model = YOLO("yolov8n-pose.pt")
# 设置模型使用GPU
pose_model.to(device)

ENABLE_CLASSIFIER = False  # 设为True启用分类器，False只计数
CLASSIFIER_PATH = "models/jump_rope_rf_multiclass_v2.pkl"
with open(CLASSIFIER_PATH, "rb") as f:
    classifier_package = pickle.load(f)
    classifier = classifier_package["classifier"]
    label_names = {int(k): v for k, v in classifier_package["labels"].items()}
    window_size = classifier_package["window_size"]
    stride = classifier_package["stride"]

print(f"已加载分类器: {CLASSIFIER_PATH}")
print(f"类别: {label_names}")
print(f"窗口大小: {window_size}, 步长: {stride}")

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
WRIST_DISTANCE_CHANGE_THRESHOLD = 0.03   # 归一化距离变化阈值
WRIST_WINDOW_RADIUS = 4              # 在跳跃峰附近检查手腕动作
MAX_PERSONS = 20  # 最多支持的人数
TRACKING_DISTANCE_THRESHOLD = 100  # 追踪距离阈值（像素）

# ==========================
# 3. 人员追踪器（简单的基于距离的追踪）
# ==========================
class PersonTracker:
    def __init__(self):
        self.persons = {}  # person_id -> person_data
        self.next_id = 0
    
    def update(self, detections):
        """
        detections: list of (keypoints, confs)
        返回: list of (person_id, keypoints, confs)
        """
        if not detections:
            # 移除所有超过30帧未更新的人员
            to_remove = [pid for pid, data in self.persons.items() 
                        if data['last_seen'] < self.next_id - 30]
            for pid in to_remove:
                del self.persons[pid]
            return []
        
        # 获取活跃的人员
        active_persons = {pid: data for pid, data in self.persons.items() 
                         if data['last_seen'] >= self.next_id - 30}
        
        # 简单的基于髋部中心的距离匹配
        matched = set()
        assignments = []
        
        for det_keypoints, det_confs in detections:
            det_center = self._get_center(det_keypoints, det_confs)
            if det_center is None:
                continue
            
            best_pid = None
            best_dist = TRACKING_DISTANCE_THRESHOLD
            
            for pid, data in active_persons.items():
                if pid in matched:
                    continue
                if data['last_center'] is None:
                    continue
                
                dist = np.linalg.norm(det_center - data['last_center'])
                if dist < best_dist:
                    best_dist = dist
                    best_pid = pid
            
            if best_pid is not None:
                matched.add(best_pid)
                assignments.append((best_pid, det_keypoints, det_confs))
                self.persons[best_pid]['last_center'] = det_center
                self.persons[best_pid]['last_seen'] = self.next_id
            else:
                # 创建新人员
                new_id = self.next_id
                self.persons[new_id] = {
                    'last_center': det_center,
                    'last_seen': self.next_id,
                    'ankle_y_history': deque(maxlen=WINDOW_SIZE * 3),
                    'wrist_motion_history': deque(maxlen=WINDOW_SIZE * 3),
                    'prev_left_wrist_distance': None,
                    'prev_right_wrist_distance': None,
                    'jump_count': 0,
                    'rope_jump_count': 0,
                    'last_peak_index': -100,
                    # 新增：用于分类器的姿态缓冲区
                    'pose_buffer': deque(maxlen=window_size),
                    'conf_buffer': deque(maxlen=window_size),
                    'jump_details': [],  # 记录每次跳跃的详细信息
                    'latest_prediction': None,
                    'latest_probability': 0.0,
                }
                assignments.append((new_id, det_keypoints, det_confs))
        
        self.next_id += 1
        return assignments
    
    def _get_center(self, keypoints, confs):
        """获取髋部中心作为追踪点"""
        left_hip = 11
        right_hip = 12
        if confs[left_hip] > KEYPOINT_CONF_THRESHOLD and confs[right_hip] > KEYPOINT_CONF_THRESHOLD:
            return (keypoints[left_hip] + keypoints[right_hip]) / 2.0
        
        left_shoulder = 5
        right_shoulder = 6
        if confs[left_shoulder] > KEYPOINT_CONF_THRESHOLD and confs[right_shoulder] > KEYPOINT_CONF_THRESHOLD:
            return (keypoints[left_shoulder] + keypoints[right_shoulder]) / 2.0
        
        return None

# ==========================
# 4. 辅助函数
# ==========================
def get_point_distance(keypoints, confs, start_idx, end_idx):
    if confs[start_idx] <= KEYPOINT_CONF_THRESHOLD or confs[end_idx] <= KEYPOINT_CONF_THRESHOLD:
        return None
    return float(np.linalg.norm(keypoints[end_idx] - keypoints[start_idx]))


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


def get_gpu_stats():
    """获取GPU使用率（NVIDIA）"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            gpu_util = float(parts[0].strip())
            mem_used = float(parts[1].strip())
            mem_total = float(parts[2].strip())
            temp = float(parts[3].strip())
            power = float(parts[4].strip()) if len(parts) > 4 else 0
            
            return {
                'utilization': gpu_util,
                'memory_used': mem_used,
                'memory_total': mem_total,
                'memory_percent': (mem_used / mem_total * 100) if mem_total > 0 else 0,
                'temperature': temp,
                'power': power
            }
    except:
        pass
    return None


def get_rk3588_estimate(frame_time, num_persons, pose_model_type="yolov8n-pose"):
    """
    估算RK3588的性能
    RK3588 NPU: 6 TOPS INT8
    RK3588 GPU: Mali-G610 MP4
    
    YOLOv8n-pose on RK3588 NPU (RKNN量化后):
    - 理论推理时间: ~15-25ms (40-67 FPS)
    - 但实际会有预处理/后处理开销
    """
    # RK3588相对于桌面GPU的减速系数（估算）
    # CPU模式: 5-10x 慢
    # NPU模式: 2-3x 慢（相比桌面GPU）
    rk3588_slowdown_factor = 3.0  # NPU量化后
    
    estimated_time_rk3588 = frame_time * rk3588_slowdown_factor
    estimated_fps = 1.0 / estimated_time_rk3588 if estimated_time_rk3588 > 0 else 0
    
    # RK3588实际能力评估
    if num_persons <= 5:
        feasible = estimated_fps >= 15
        performance_level = "GOOD" if estimated_fps >= 25 else "OK" if estimated_fps >= 15 else "LOW"
    elif num_persons <= 10:
        feasible = estimated_fps >= 10
        performance_level = "GOOD" if estimated_fps >= 20 else "OK" if estimated_fps >= 10 else "LOW"
    else:
        feasible = estimated_fps >= 5
        performance_level = "GOOD" if estimated_fps >= 15 else "OK" if estimated_fps >= 5 else "LOW"
    
    return {
        'estimated_fps': estimated_fps,
        'feasible': feasible,
        'performance_level': performance_level,
        'frame_time_ms': estimated_time_rk3588 * 1000
    }


def detect_jumps(person_data, frame_id):
    """检测单个人的跳跃并使用分类器识别类型"""
    ankle_history = list(person_data['ankle_y_history'])
    wrist_history = list(person_data['wrist_motion_history'])
    
    if len(ankle_history) < WINDOW_SIZE:
        return False, False, None
    
    window = np.array(ankle_history[-WINDOW_SIZE:])
    signal = -window  # 反转（因为图像y向下）
    
    peaks, properties = find_peaks(
        signal,
        distance=PEAK_DISTANCE,
        prominence=PEAK_PROMINENCE
    )
    
    has_jump = False
    has_rope = False
    jump_type = None
    
    for peak in peaks:
        global_index = frame_id - WINDOW_SIZE + peak
        
        if global_index - person_data['last_peak_index'] > PEAK_DISTANCE:
            has_jump = True
            person_data['last_peak_index'] = global_index
            person_data['jump_count'] += 1
            
            # 检查手腕动作
            peak_center = len(wrist_history) - WINDOW_SIZE + peak
            wrist_start = max(0, peak_center - WRIST_WINDOW_RADIUS)
            wrist_end = min(len(wrist_history), peak_center + WRIST_WINDOW_RADIUS + 1)
            wrist_window = wrist_history[wrist_start:wrist_end]
            wrist_motion_mean = float(np.mean(wrist_window)) if wrist_window else 0.0
            has_rope_swing = wrist_motion_mean > WRIST_DISTANCE_CHANGE_THRESHOLD
            
            if has_rope_swing:
                has_rope = True
                person_data['rope_jump_count'] += 1
                
                # 使用分类器识别跳跃类型（可选，消耗CPU）
                if ENABLE_CLASSIFIER and person_data['rope_jump_count'] % 2 == 1:
                    pose_buffer = list(person_data['pose_buffer'])
                    conf_buffer = list(person_data['conf_buffer'])
                    
                    if len(pose_buffer) >= window_size:
                        try:
                            feature_vector = extract_window_features(
                                pose_buffer[-window_size:],
                                conf_buffer[-window_size:]
                            ).reshape(1, -1)
                            
                            prediction = int(classifier.predict(feature_vector)[0])
                            probability = float(np.max(classifier.predict_proba(feature_vector)[0]))
                            
                            jump_type = label_names.get(prediction, f"class_{prediction}")
                            person_data['latest_prediction'] = jump_type
                            person_data['latest_probability'] = probability
                            
                            # 记录这次跳跃的详细信息
                            person_data['jump_details'].append({
                                'frame': global_index,
                                'type': jump_type,
                                'confidence': probability
                            })
                            
                            # 只保留最近20次跳跃记录
                            if len(person_data['jump_details']) > 20:
                                person_data['jump_details'] = person_data['jump_details'][-20:]
                            
                        except Exception as e:
                            print(f"分类器预测错误: {e}")
            
            break  # 只处理最新的峰
    
    return has_jump, has_rope, jump_type


# ==========================
# 5. 主循环
# ==========================
tracker = PersonTracker()

# 视频输入（可改为0使用摄像头）
cap = cv2.VideoCapture("D:\\work\\Intern\\Dataset\\10WomanCross.mp4")
frame_id = 0

# 性能监控
import time
frame_times = deque(maxlen=30)  # 记录最近30帧的处理时间
fps_display = 0

# 颜色映射（为不同人员分配不同颜色）
COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (0, 128, 255),
    (255, 128, 0), (128, 255, 0), (255, 0, 128), (0, 255, 128),
]

while True:
    start_time = time.time()
    
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_id += 1
    
    # ==========================
    # 6. Pose检测
    # ==========================
    results = pose_model(frame, verbose=False, device=device)
    
    # 收集所有检测到的人
    detections = []
    if len(results[0].keypoints) > 0:
        num_persons = len(results[0].keypoints.xy)
        for person_idx in range(num_persons):
            keypoints = results[0].keypoints.xy[person_idx].cpu().numpy()
            keypoint_conf = results[0].keypoints.conf[person_idx].cpu().numpy()
            detections.append((keypoints, keypoint_conf))
    
    # 追踪人员
    assignments = tracker.update(detections)
    
    # ==========================
    # 7. 更新每个人的状态并计数
    # ==========================
    for person_id, keypoints, keypoint_conf in assignments:
        person_data = tracker.persons[person_id]
        
        # 更新姿态缓冲区（用于分类器）
        person_data['pose_buffer'].append(keypoints.copy())
        person_data['conf_buffer'].append(keypoint_conf.copy())
        
        # 计算手腕运动
        body_scale = max(get_body_scale(keypoints, keypoint_conf), 1.0)
        
        left_wrist_distance = get_point_distance(keypoints, keypoint_conf, 7, 9)
        right_wrist_distance = get_point_distance(keypoints, keypoint_conf, 8, 10)
        
        left_distance_change = 0.0
        right_distance_change = 0.0
        
        if left_wrist_distance is not None and person_data['prev_left_wrist_distance'] is not None:
            left_distance_change = abs(left_wrist_distance - person_data['prev_left_wrist_distance']) / body_scale
        if right_wrist_distance is not None and person_data['prev_right_wrist_distance'] is not None:
            right_distance_change = abs(right_wrist_distance - person_data['prev_right_wrist_distance']) / body_scale
        
        person_data['prev_left_wrist_distance'] = left_wrist_distance
        person_data['prev_right_wrist_distance'] = right_wrist_distance
        
        wrist_motion_score = (left_distance_change + right_distance_change) / 2.0
        person_data['wrist_motion_history'].append(wrist_motion_score)
        
        # 获取脚踝Y坐标
        left_ankle_y = keypoints[15][1] if keypoint_conf[15] > KEYPOINT_CONF_THRESHOLD else None
        right_ankle_y = keypoints[16][1] if keypoint_conf[16] > KEYPOINT_CONF_THRESHOLD else None
        
        if left_ankle_y is not None and right_ankle_y is not None:
            ankle_y = (left_ankle_y + right_ankle_y) / 2
            person_data['ankle_y_history'].append(ankle_y)
        elif left_ankle_y is not None:
            person_data['ankle_y_history'].append(left_ankle_y)
        elif right_ankle_y is not None:
            person_data['ankle_y_history'].append(right_ankle_y)
        
        # 检测跳跃
        has_jump, has_rope, jump_type = detect_jumps(person_data, frame_id)
        
        # ==========================
        # 8. 可视化
        # ==========================
        color = COLORS[person_id % len(COLORS)]
        
        # 画骨架
        for start_idx, end_idx in SKELETON:
            if keypoint_conf[start_idx] > KEYPOINT_CONF_THRESHOLD and keypoint_conf[end_idx] > KEYPOINT_CONF_THRESHOLD:
                start_point = tuple(keypoints[start_idx].astype(int))
                end_point = tuple(keypoints[end_idx].astype(int))
                cv2.line(frame, start_point, end_point, color, 2)
        
        # 画关键点
        for idx, (x, y) in enumerate(keypoints):
            if keypoint_conf[idx] > KEYPOINT_CONF_THRESHOLD:
                point = (int(x), int(y))
                cv2.circle(frame, point, 3, color, -1)
        
        # 显示人员ID和计数信息
        hip_center = None
        if keypoint_conf[11] > KEYPOINT_CONF_THRESHOLD and keypoint_conf[12] > KEYPOINT_CONF_THRESHOLD:
            hip_center = ((keypoints[11] + keypoints[12]) / 2).astype(int)
        
        if hip_center is not None:
            # 人员ID
            cv2.putText(
                frame, f"Person {person_id}",
                (hip_center[0] - 40, hip_center[1] - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
            
            # 跳跃计数
            cv2.putText(
                frame, f"Jumps: {person_data['jump_count']}",
                (hip_center[0] - 40, hip_center[1] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )
            
            # 跳绳计数
            cv2.putText(
                frame, f"Rope: {person_data['rope_jump_count']}",
                (hip_center[0] - 40, hip_center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
            )
            
            # 最新跳跃类型
            if person_data['latest_prediction'] is not None:
                type_text = f"{person_data['latest_prediction']} ({person_data['latest_probability']:.2f})"
                cv2.putText(
                    frame, type_text,
                    (hip_center[0] - 60, hip_center[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1
                )
    
    # ==========================
    # 9. 显示汇总信息
    # ==========================
    y_offset = 30
    cv2.putText(
        frame, f"Frame: {frame_id} | Persons: {len(assignments)}",
        (20, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
    )
    y_offset += 30
    
    # 显示所有人的统计
    for person_id in sorted(tracker.persons.keys()):
        if tracker.persons[person_id]['last_seen'] >= frame_id - 30:
            data = tracker.persons[person_id]
            color = COLORS[person_id % len(COLORS)]
            text = f"P{person_id}: Jumps={data['jump_count']}, Rope={data['rope_jump_count']}"
            if data['latest_prediction'] is not None:
                text += f", Type={data['latest_prediction']}({data['latest_probability']:.2f})"
            cv2.putText(
                frame, text,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1
            )
            y_offset += 25
    
    # 显示最近跳跃事件（终端输出）
    for person_id in sorted(tracker.persons.keys()):
        data = tracker.persons[person_id]
        if data['jump_details'] and data['jump_details'][-1]['frame'] == frame_id:
            latest_jump = data['jump_details'][-1]
            print(f"[Frame {frame_id}] Person {person_id}: {latest_jump['type']} (confidence: {latest_jump['confidence']:.3f})")
    
    # ==========================
    # 10. 性能监控
    # ==========================
    end_time = time.time()
    frame_time = end_time - start_time
    frame_times.append(frame_time)
    fps_display = 1.0 / np.mean(frame_times) if frame_times else 0
    
    # 获取GPU状态
    gpu_stats = get_gpu_stats()
    
    # RK3588性能估算
    rk3588_info = get_rk3588_estimate(frame_time, len(assignments))
    
    # 显示性能信息（右上角）
    perf_y = 30
    perf_x = frame.shape[1] - 450
    
    # 当前FPS
    cv2.putText(
        frame, f"FPS: {fps_display:.1f} | Frame Time: {frame_time*1000:.1f}ms",
        (perf_x, perf_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
    )
    perf_y += 25
    
    # GPU信息
    if gpu_stats:
        gpu_color = (0, 255, 0) if gpu_stats['utilization'] < 70 else (0, 255, 255) if gpu_stats['utilization'] < 90 else (0, 0, 255)
        cv2.putText(
            frame, f"GPU: {gpu_stats['utilization']:.0f}% | Mem: {gpu_stats['memory_used']:.0f}/{gpu_stats['memory_total']:.0f}MB",
            (perf_x, perf_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, gpu_color, 2
        )
        perf_y += 25
        cv2.putText(
            frame, f"Temp: {gpu_stats['temperature']:.0f}C | Power: {gpu_stats['power']:.1f}W",
            (perf_x, perf_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )
        perf_y += 25
    else:
        cv2.putText(
            frame, "GPU: N/A (CPU mode or no nvidia-smi)",
            (perf_x, perf_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
        )
        perf_y += 25
    
    # RK3588估算
    rk_color = (0, 255, 0) if rk3588_info['feasible'] else (0, 0, 255)
    cv2.putText(
        frame, f"RK3588 Est: {rk3588_info['estimated_fps']:.1f} FPS ({rk3588_info['performance_level']})",
        (perf_x, perf_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, rk_color, 2
    )
    perf_y += 25
    cv2.putText(
        frame, f"RK3588 Feasible: {'YES' if rk3588_info['feasible'] else 'NO'}",
        (perf_x, perf_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, rk_color, 2
    )
    
    # 终端输出性能信息（每30帧）
    if frame_id % 30 == 0:
        print(f"\n{'='*60}")
        print(f"[Performance] Frame {frame_id}")
        print(f"  Current FPS: {fps_display:.1f}")
        print(f"  Frame Time: {frame_time*1000:.1f}ms")
        print(f"  Persons: {len(assignments)}")
        if gpu_stats:
            print(f"  GPU Utilization: {gpu_stats['utilization']:.1f}%")
            print(f"  GPU Memory: {gpu_stats['memory_used']:.0f}/{gpu_stats['memory_total']:.0f} MB ({gpu_stats['memory_percent']:.1f}%)")
            print(f"  GPU Temperature: {gpu_stats['temperature']:.1f}°C")
            print(f"  GPU Power: {gpu_stats['power']:.1f}W")
        print(f"  RK3588 Estimated FPS: {rk3588_info['estimated_fps']:.1f}")
        print(f"  RK3588 Performance: {rk3588_info['performance_level']}")
        print(f"  RK3588 Feasible: {'YES' if rk3588_info['feasible'] else 'NO'}")
        print(f"{'='*60}\n")
    
    cv2.imshow("Multi-Person Jump Rope Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
