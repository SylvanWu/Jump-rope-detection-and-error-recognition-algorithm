import argparse
import csv
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

from pose_features import FEATURE_NAMES, extract_window_features

PROGRESS_EVERY_FRAMES = 60


def load_manifest(manifest_path):
    samples = []
    with open(manifest_path, "r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            video_path = row["video_path"].strip()
            label = int(row["label"])
            samples.append({"video_path": video_path, "label": label})
    return samples


def extract_pose_sequence(model, video_path, device):
    capture = cv2.VideoCapture(video_path)
    window_points = []
    window_confs = []
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0
    video_name = Path(video_path).name

    print(
        f"[{video_name}] start pose extraction, total_frames={total_frames}",
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
                        f"[{video_name}] frame {processed_frames}/{total_frames} "
                        f"({progress:.1f}%), valid_pose_frames={len(window_points)}",
                        flush=True,
                    )
                else:
                    print(
                        f"[{video_name}] frame {processed_frames}, "
                        f"valid_pose_frames={len(window_points)}",
                        flush=True,
                    )
            continue

        points = results[0].keypoints.xy[0].cpu().numpy()
        confs = results[0].keypoints.conf[0].cpu().numpy()
        window_points.append(points)
        window_confs.append(confs)

        if processed_frames % PROGRESS_EVERY_FRAMES == 0:
            if total_frames > 0:
                progress = processed_frames / total_frames * 100.0
                print(
                    f"[{video_name}] frame {processed_frames}/{total_frames} "
                    f"({progress:.1f}%), valid_pose_frames={len(window_points)}",
                    flush=True,
                )
            else:
                print(
                    f"[{video_name}] frame {processed_frames}, "
                    f"valid_pose_frames={len(window_points)}",
                    flush=True,
                )

    capture.release()
    print(
        f"[{video_name}] pose extraction done, valid_pose_frames={len(window_points)}",
        flush=True,
    )
    return window_points, window_confs


def build_samples_from_video(model, video_path, label, window_size, stride, device):
    all_points, all_confs = extract_pose_sequence(model, video_path, device)
    dataset = []

    if len(all_points) < window_size:
        return dataset

    for start in range(0, len(all_points) - window_size + 1, stride):
        end = start + window_size
        feature_vector = extract_window_features(
            all_points[start:end],
            all_confs[start:end],
        )
        dataset.append((feature_vector, label))

    print(
        f"[{Path(video_path).name}] generated_windows={len(dataset)} "
        f"(window_size={window_size}, stride={stride})",
        flush=True,
    )
    return dataset


def train_classifier(manifest_path, model_path, output_path, window_size, stride, device):
    manifest = load_manifest(manifest_path)
    pose_model = YOLO(model_path)

    features = []
    labels = []

    for item in manifest:
        video_path = item["video_path"]
        label = item["label"]
        print(f"Processing: {video_path} label={label}", flush=True)
        video_samples = build_samples_from_video(
            pose_model,
            video_path,
            label,
            window_size,
            stride,
            device,
        )

        for feature_vector, sample_label in video_samples:
            features.append(feature_vector)
            labels.append(sample_label)

    if not features:
        raise RuntimeError("No training samples were generated. Check your manifest or window size.")

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    print(
        f"Collected samples: total={len(x)}, positive={int(np.sum(y == 1))}, "
        f"negative={int(np.sum(y == 0))}",
        flush=True,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
    )
    classifier.fit(x_train, y_train)

    predictions = classifier.predict(x_test)
    print(classification_report(y_test, predictions, digits=4))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as model_file:
        pickle.dump(
            {
                "classifier": classifier,
                "feature_names": FEATURE_NAMES,
                "window_size": window_size,
                "stride": stride,
                "labels": {0: "not_jump_rope", 1: "jump_rope"},
            },
            model_file,
        )

    summary_path = output_path.with_suffix(".json")
    summary = {
        "model_path": str(output_path),
        "pose_model": model_path,
        "device": device,
        "window_size": window_size,
        "stride": stride,
        "feature_names": FEATURE_NAMES,
        "num_samples": int(len(x)),
        "num_positive": int(np.sum(y == 1)),
        "num_negative": int(np.sum(y == 0)),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved classifier to: {output_path}")
    print(f"Saved summary to: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a jump-rope / non-jump-rope classifier from YOLO pose keypoints.")
    parser.add_argument("--manifest", required=True, help="CSV file with columns: video_path,label")
    parser.add_argument("--pose-model", default="yolov8n-pose.pt", help="Ultralytics pose model path")
    parser.add_argument("--output", default="models/jump_rope_rf.pkl", help="Where to save the trained classifier")
    parser.add_argument("--window-size", type=int, default=30, help="Number of frames per sample window")
    parser.add_argument("--stride", type=int, default=10, help="Sliding window step")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device for YOLO pose extraction, e.g. cpu or cuda:0",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_classifier(
        manifest_path=args.manifest,
        model_path=args.pose_model,
        output_path=args.output,
        window_size=args.window_size,
        stride=args.stride,
        device=args.device,
    )
