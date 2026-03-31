import argparse
import pickle

import cv2
import numpy as np
from ultralytics import YOLO

from pose_features import extract_window_features

PROGRESS_EVERY_FRAMES = 60
PROGRESS_EVERY_WINDOWS = 40


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


def run_prediction(video_path, pose_model_path, classifier_path, device):
    with open(classifier_path, "rb") as model_file:
        package = pickle.load(model_file)

    classifier = package["classifier"]
    window_size = package["window_size"]
    stride = package["stride"]
    label_names = package["labels"]

    pose_model = YOLO(pose_model_path)
    all_points, all_confs = extract_pose_sequence(pose_model, video_path, device)

    if len(all_points) < window_size:
        raise RuntimeError(f"Video is too short. Need at least {window_size} valid pose frames.")

    predictions = []
    probabilities = []
    total_windows = (len(all_points) - window_size) // stride + 1
    print(
        f"[cls] start window classification, total_windows={total_windows}, "
        f"window_size={window_size}, stride={stride}",
        flush=True,
    )

    for idx, start in enumerate(range(0, len(all_points) - window_size + 1, stride), start=1):
        end = start + window_size
        feature_vector = extract_window_features(all_points[start:end], all_confs[start:end])
        feature_vector = feature_vector.reshape(1, -1)
        pred = int(classifier.predict(feature_vector)[0])
        prob = float(np.max(classifier.predict_proba(feature_vector)[0]))
        predictions.append(pred)
        probabilities.append(prob)

        if idx % PROGRESS_EVERY_WINDOWS == 0 or idx == total_windows:
            progress = idx / total_windows * 100.0
            print(f"[cls] window {idx}/{total_windows} ({progress:.1f}%)", flush=True)

    positive_ratio = float(np.mean(np.asarray(predictions) == 1))
    mean_confidence = float(np.mean(probabilities))
    final_label = 1 if positive_ratio >= 0.5 else 0

    print(f"Video: {video_path}")
    print(f"Windows: {len(predictions)}")
    print(f"Positive ratio: {positive_ratio:.3f}")
    print(f"Mean confidence: {mean_confidence:.3f}")
    print(f"Final prediction: {label_names[final_label]}")


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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_prediction(
        video_path=args.video,
        pose_model_path=args.pose_model,
        classifier_path=args.classifier,
        device=args.device,
    )
