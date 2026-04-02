import argparse
import pickle
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

from pose_features import extract_window_features

PROGRESS_EVERY_FRAMES = 60
PROGRESS_EVERY_WINDOWS = 40


def normalize_label_map(raw_label_map):
    normalized = {}
    for key, value in raw_label_map.items():
        normalized[int(key)] = value
    return normalized


def extract_pose_sequence(model, video_path, device):
    capture = cv2.VideoCapture(video_path)
    all_points = []
    all_confs = []
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0

    print(
        f"[pose] start extraction, total_frames={total_frames}, device={device}",
        flush=True,
    )

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        processed_frames += 1
        results = model(frame, verbose=False, device=device)
        if len(results[0].keypoints) == 0:
            if processed_frames % PROGRESS_EVERY_FRAMES == 0:
                if total_frames > 0:
                    progress = processed_frames / total_frames * 100.0
                    print(
                        f"[pose] frame {processed_frames}/{total_frames} "
                        f"({progress:.1f}%), valid_pose_frames={len(all_points)}",
                        flush=True,
                    )
                else:
                    print(
                        f"[pose] frame {processed_frames}, valid_pose_frames={len(all_points)}",
                        flush=True,
                    )
            continue

        points = results[0].keypoints.xy[0].cpu().numpy()
        confs = results[0].keypoints.conf[0].cpu().numpy()
        all_points.append(points)
        all_confs.append(confs)

        if processed_frames % PROGRESS_EVERY_FRAMES == 0:
            if total_frames > 0:
                progress = processed_frames / total_frames * 100.0
                print(
                    f"[pose] frame {processed_frames}/{total_frames} "
                    f"({progress:.1f}%), valid_pose_frames={len(all_points)}",
                    flush=True,
                )
            else:
                print(
                    f"[pose] frame {processed_frames}, valid_pose_frames={len(all_points)}",
                    flush=True,
                )

    capture.release()
    print(f"[pose] extraction done, valid_pose_frames={len(all_points)}", flush=True)
    return all_points, all_confs


def draw_overlay(frame, lines):
    overlay = frame.copy()
    panel_height = 30 + 28 * len(lines)
    cv2.rectangle(overlay, (10, 10), (760, panel_height), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    y = 38
    for line in lines:
        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
        )
        y += 28
    return frame


def run_prediction(video_path, pose_model_path, classifier_path, device, show):
    with open(classifier_path, "rb") as model_file:
        package = pickle.load(model_file)

    classifier = package["classifier"]
    window_size = package["window_size"]
    stride = package["stride"]
    label_names = normalize_label_map(package["labels"])

    pose_model = YOLO(pose_model_path)
    capture = cv2.VideoCapture(video_path)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    points_buffer = deque(maxlen=window_size)
    confs_buffer = deque(maxlen=window_size)
    predictions = []
    probabilities = []
    processed_frames = 0
    valid_pose_frames = 0
    windows_done = 0
    valid_pose_since_last_window = 0
    latest_prob = 0.0
    latest_pred = 0

    print(
        f"[run] start, total_frames={total_frames}, window_size={window_size}, "
        f"stride={stride}, show={show}, device={device}",
        flush=True,
    )

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        processed_frames += 1
        results = pose_model(frame, verbose=False, device=device)
        if len(results[0].keypoints) > 0:
            points = results[0].keypoints.xy[0].cpu().numpy()
            confs = results[0].keypoints.conf[0].cpu().numpy()
            points_buffer.append(points)
            confs_buffer.append(confs)
            valid_pose_frames += 1
            valid_pose_since_last_window += 1

            if len(points_buffer) == window_size and valid_pose_since_last_window >= stride:
                feature_vector = extract_window_features(
                    list(points_buffer),
                    list(confs_buffer),
                ).reshape(1, -1)
                latest_pred = int(classifier.predict(feature_vector)[0])
                latest_prob = float(np.max(classifier.predict_proba(feature_vector)[0]))
                predictions.append(latest_pred)
                probabilities.append(latest_prob)
                windows_done += 1
                valid_pose_since_last_window = 0

                if windows_done % PROGRESS_EVERY_WINDOWS == 0:
                    print(f"[cls] windows_done={windows_done}", flush=True)

        if processed_frames % PROGRESS_EVERY_FRAMES == 0:
            if total_frames > 0:
                progress = processed_frames / total_frames * 100.0
                print(
                    f"[pose] frame {processed_frames}/{total_frames} ({progress:.1f}%), "
                    f"valid_pose_frames={valid_pose_frames}, windows={windows_done}",
                    flush=True,
                )
            else:
                print(
                    f"[pose] frame {processed_frames}, valid_pose_frames={valid_pose_frames}, "
                    f"windows={windows_done}",
                    flush=True,
                )

        if show:
            if predictions:
                class_votes_now = np.bincount(np.asarray(predictions, dtype=np.int32))
                final_pred_now = int(np.argmax(class_votes_now))
                top_class_ratio_now = float(class_votes_now[final_pred_now] / len(predictions))
                mean_conf_now = float(np.mean(probabilities))
            else:
                top_class_ratio_now = 0.0
                mean_conf_now = 0.0
                final_pred_now = latest_pred

            lines = [
                f"Frame: {processed_frames}/{total_frames if total_frames > 0 else '?'}",
                f"Window: {windows_done}  Latest: {label_names.get(latest_pred, str(latest_pred))} ({latest_prob:.3f})",
                f"Top class ratio: {top_class_ratio_now:.3f}",
                f"Mean confidence: {mean_conf_now:.3f}",
                f"Final prediction: {label_names.get(final_pred_now, str(final_pred_now))}",
            ]
            vis_frame = draw_overlay(frame, lines)
            cv2.imshow("Jump Rope Classifier", vis_frame)

            # ESC to stop early
            if cv2.waitKey(1) & 0xFF == 27:
                break

    capture.release()
    if show:
        cv2.destroyAllWindows()

    if not predictions:
        raise RuntimeError(
            "No classification windows were generated. "
            "Video may be too short or valid pose frames are insufficient."
        )

    class_votes = np.bincount(np.asarray(predictions, dtype=np.int32))
    final_label = int(np.argmax(class_votes))
    top_class_ratio = float(class_votes[final_label] / len(predictions))
    mean_confidence = float(np.mean(probabilities))

    print(f"Video: {video_path}")
    print(f"Windows: {len(predictions)}")
    print(f"Top class ratio: {top_class_ratio:.3f}")
    print(f"Mean confidence: {mean_confidence:.3f}")
    print(f"Final prediction: {label_names.get(final_label, str(final_label))}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run jump-rope / non-jump-rope classification on a video.")
    parser.add_argument("--video", required=True, help="Video to classify")
    parser.add_argument("--classifier", required=True, help="Path to trained classifier pickle")
    parser.add_argument("--pose-model", default="yolov8n-pose.pt", help="Ultralytics pose model path")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device for YOLO pose, e.g. cpu or cuda:0",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show realtime visualization with overlayed window statistics",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_prediction(
        video_path=args.video,
        pose_model_path=args.pose_model,
        classifier_path=args.classifier,
        device=args.device,
        show=args.show,
    )
