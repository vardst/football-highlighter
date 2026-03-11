"""
train_model.py — Download soccer datasets and train a custom YOLOv8s model

Datasets:
    - SoccerNet_v3_H250 (Zenodo): 19K images, 2 classes (ball, person)
    - Soccana (HuggingFace): 25K images, 3 classes (player, ball, referee)

Unified class mapping:
    0: ball
    1: player
    2: goalkeeper  (reserved for future datasets)
    3: referee

Usage:
    python train_model.py                    # full pipeline: download + train
    python train_model.py --download-only    # just download and merge datasets
    python train_model.py --skip-download    # train on already-downloaded data
    python train_model.py --epochs 50        # custom epoch count
    python train_model.py --test-soccana     # quick test with pre-trained Soccana model
"""

import argparse
import os
import shutil
import glob
import zipfile
import subprocess
import sys

# Unified class scheme
UNIFIED_CLASSES = {
    0: "ball",
    1: "player",
    2: "goalkeeper",
    3: "referee",
}

# SoccerNet_v3_H250 class mapping: original -> unified
SOCCERNET_MAP = {
    0: 0,   # ball -> ball
    1: 1,   # person -> player
}

# Soccana class mapping: original -> unified
# Soccana classes: 0=Player, 1=Referee, 2=Ball
SOCCANA_MAP = {
    0: 1,   # player -> player
    1: 3,   # referee -> referee
    2: 0,   # ball -> ball
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def download_soccernet(data_dir):
    """Download SoccerNet_v3_H250 from Zenodo."""
    dest = os.path.join(data_dir, "soccernet")
    if os.path.isdir(dest) and len(os.listdir(dest)) > 0:
        print(f"  [download] SoccerNet already exists at {dest}, skipping")
        return dest

    os.makedirs(dest, exist_ok=True)

    # SoccerNet_v3_H250 Zenodo URL (YOLO format)
    url = "https://zenodo.org/records/7808511/files/YOLO.zip?download=1"
    zip_path = os.path.join(data_dir, "SoccerNet_v3_H250.zip")

    print(f"  [download] Downloading SoccerNet_v3_H250 from Zenodo...")
    print(f"             This is ~2.5GB, may take a while")

    # Use curl with resume support — more reliable for large files
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        print(f"  [download] Attempt {attempt}/{max_retries}...")
        result = subprocess.run(
            ["curl", "-L", "-C", "-", "-o", zip_path,
             "--connect-timeout", "30",
             "--retry", "3", "--retry-delay", "5",
             url],
        )
        if result.returncode == 0:
            break
        print(f"  [download] curl exited {result.returncode}, retrying...")
    else:
        raise RuntimeError(f"Failed to download SoccerNet after {max_retries} attempts")

    print(f"  [download] Extracting SoccerNet_v3_H250...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)

    os.remove(zip_path)
    print(f"  [download] SoccerNet_v3_H250 ready at {dest}")
    return dest


def download_soccana(data_dir):
    """Download Soccana dataset from HuggingFace."""
    dest = os.path.join(data_dir, "soccana")
    if os.path.isdir(dest) and len(os.listdir(dest)) > 0:
        print(f"  [download] Soccana already exists at {dest}, skipping")
        return dest

    os.makedirs(dest, exist_ok=True)

    print(f"  [download] Downloading Soccana from HuggingFace...")

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="Adit-jain/Soccana_player_ball_detection_v1",
            repo_type="dataset",
            local_dir=dest,
            allow_patterns=["*.yaml", "*.yml",
                           "V1/images/**", "V1/labels/**",
                           "images/**", "labels/**"],
        )
    except Exception as e:
        print(f"  [download] huggingface_hub failed ({e})")
        print(f"  [download] Trying git clone...")
        subprocess.run(
            ["git", "clone",
             "https://huggingface.co/datasets/Adit-jain/Soccana_player_ball_detection_v1",
             dest],
            check=True,
        )

    print(f"  [download] Soccana ready at {dest}")
    return dest


def download_soccana_pretrained(models_dir):
    """Download pre-trained Soccana YOLOv11n model from HuggingFace."""
    dest = os.path.join(models_dir, "soccana_yolov11n.pt")
    if os.path.isfile(dest):
        print(f"  [download] Soccana pretrained model already exists at {dest}")
        return dest

    os.makedirs(models_dir, exist_ok=True)

    print(f"  [download] Downloading pre-trained Soccana YOLOv11n...")

    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id="Adit-jain/soccana",
            filename="Models/Trained/yolov11_sahi_1280/Model/weights/best.pt",
            repo_type="model",
            local_dir=models_dir,
        )
        # Rename to our convention
        if os.path.isfile(downloaded) and downloaded != dest:
            shutil.copy2(downloaded, dest)
    except Exception as e:
        print(f"  [download] Failed to download pre-trained model: {e}")
        print(f"  [download] You can manually download from:")
        print(f"             https://huggingface.co/Adit-jain/soccana")
        return None

    print(f"  [download] Pre-trained model saved to {dest}")
    return dest


def _check_split(base, split):
    """Check common YOLO layouts under a base directory for a given split."""
    # Layout 1: base/images/split/, base/labels/split/
    img = os.path.join(base, "images", split)
    lbl = os.path.join(base, "labels", split)
    if os.path.isdir(img) and os.path.isdir(lbl):
        return {"images": img, "labels": lbl}
    # Layout 2: base/split/images/, base/split/labels/
    img = os.path.join(base, split, "images")
    lbl = os.path.join(base, split, "labels")
    if os.path.isdir(img) and os.path.isdir(lbl):
        return {"images": img, "labels": lbl}
    return None


def find_dataset_structure(dataset_dir):
    """
    Detect the YOLO dataset structure within a downloaded dataset.
    Searches up to 2 levels deep for images/labels directories.
    Returns dict with 'train', 'val', 'test' paths for images and labels.
    """
    structure = {}

    for split in ["train", "valid", "val", "test"]:
        split_key = "val" if split == "valid" else split
        if split_key in structure:
            continue

        # Check at root level
        found = _check_split(dataset_dir, split)
        if found:
            structure[split_key] = found
            continue

        # Check one level deeper (e.g., V1/, YOLO/, etc.)
        if os.path.isdir(dataset_dir):
            for subdir in os.listdir(dataset_dir):
                sub = os.path.join(dataset_dir, subdir)
                if not os.path.isdir(sub):
                    continue
                found = _check_split(sub, split)
                if found:
                    structure[split_key] = found
                    break

            # Check two levels deeper
            if split_key not in structure:
                for subdir in os.listdir(dataset_dir):
                    sub = os.path.join(dataset_dir, subdir)
                    if not os.path.isdir(sub):
                        continue
                    for subsubdir in os.listdir(sub):
                        subsub = os.path.join(sub, subsubdir)
                        if not os.path.isdir(subsub):
                            continue
                        found = _check_split(subsub, split)
                        if found:
                            structure[split_key] = found
                            break
                    if split_key in structure:
                        break

    return structure


def remap_label_file(src_path, dst_path, class_map, max_boxes=100):
    """
    Read a YOLO label file, remap class IDs, and write to destination.
    Skips lines with unmapped classes.
    Caps total boxes to max_boxes to avoid MPS shape mismatch bugs.
    """
    with open(src_path, "r") as f:
        lines = f.readlines()

    remapped = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        old_cls = int(parts[0])
        if old_cls not in class_map:
            continue
        new_cls = class_map[old_cls]
        remapped.append(f"{new_cls} {' '.join(parts[1:])}\n")

    # Cap annotations per image — too many triggers MPS TAL shape mismatch
    if len(remapped) > max_boxes:
        remapped = remapped[:max_boxes]

    with open(dst_path, "w") as f:
        f.writelines(remapped)


def merge_datasets(soccernet_dir, soccana_dir, merged_dir):
    """
    Merge SoccerNet and Soccana into a unified dataset with remapped classes.
    """
    print(f"\n[merge] Scanning dataset structures...")

    soccernet_struct = find_dataset_structure(soccernet_dir)
    soccana_struct = find_dataset_structure(soccana_dir)

    print(f"  SoccerNet splits: {list(soccernet_struct.keys())}")
    print(f"  Soccana splits:   {list(soccana_struct.keys())}")

    # Create merged directory structure
    for split in ["train", "val"]:
        os.makedirs(os.path.join(merged_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(merged_dir, "labels", split), exist_ok=True)

    total_images = 0

    # Process each dataset
    for name, struct, class_map in [
        ("soccernet", soccernet_struct, SOCCERNET_MAP),
        ("soccana", soccana_struct, SOCCANA_MAP),
    ]:
        print(f"\n[merge] Processing {name}...")

        for split in ["train", "val", "test"]:
            if split not in struct:
                continue

            # Map test split into val for merged dataset
            target_split = "val" if split == "test" else split

            img_src = struct[split]["images"]
            lbl_src = struct[split]["labels"]

            img_dst = os.path.join(merged_dir, "images", target_split)
            lbl_dst = os.path.join(merged_dir, "labels", target_split)

            image_files = [f for f in os.listdir(img_src)
                          if f.lower().endswith((".jpg", ".jpeg", ".png"))]

            count = 0
            for img_file in image_files:
                # Prefix to avoid name collisions
                new_name = f"{name}_{img_file}"
                lbl_file = os.path.splitext(img_file)[0] + ".txt"
                new_lbl = f"{name}_{lbl_file}"

                src_img = os.path.join(img_src, img_file)
                src_lbl = os.path.join(lbl_src, lbl_file)

                dst_img = os.path.join(img_dst, new_name)
                dst_lbl = os.path.join(lbl_dst, new_lbl)

                # Copy image (symlink to save space)
                if not os.path.exists(dst_img):
                    os.symlink(os.path.abspath(src_img), dst_img)

                # Remap and copy label
                if os.path.isfile(src_lbl):
                    remap_label_file(src_lbl, dst_lbl, class_map)
                else:
                    # No label file = no annotations, write empty
                    open(dst_lbl, "w").close()

                count += 1

            print(f"  {name}/{split}: {count} images -> merged/{target_split}")
            total_images += count

    print(f"\n[merge] Total merged images: {total_images}")
    return total_images


def generate_data_yaml(merged_dir):
    """Generate the data.yaml config for Ultralytics training."""
    yaml_path = os.path.join(merged_dir, "data.yaml")

    content = f"""# Soccer detection dataset — merged SoccerNet + Soccana
path: {os.path.abspath(merged_dir)}
train: images/train
val: images/val

nc: {len(UNIFIED_CLASSES)}
names:
"""
    for idx in sorted(UNIFIED_CLASSES.keys()):
        content += f"  {idx}: {UNIFIED_CLASSES[idx]}\n"

    with open(yaml_path, "w") as f:
        f.write(content)

    print(f"[config] Generated {yaml_path}")
    return yaml_path


def train_model(data_yaml, epochs=20, batch=16, imgsz=640, device=None,
                resume=False):
    """Fine-tune YOLOv8s on the merged dataset."""
    from ultralytics import YOLO

    if device is None:
        import torch
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "0"
        else:
            device = "cpu"

    last_pt = os.path.join(BASE_DIR, "runs", "soccer_v1", "weights", "last.pt")

    if resume and os.path.isfile(last_pt):
        print(f"\n{'='*60}")
        print(f"  Resuming training from checkpoint")
        print(f"  Checkpoint: {last_pt}")
        print(f"  Device: {device}")
        print(f"{'='*60}\n")
        model = YOLO(last_pt)
        results = model.train(resume=True)
    else:
        print(f"\n{'='*60}")
        print(f"  Training YOLOv8s")
        print(f"  Device: {device}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch:  {batch}")
        print(f"  Imgsz:  {imgsz}")
        print(f"{'='*60}\n")

        model = YOLO("yolov8s.pt")

        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            project=os.path.join(BASE_DIR, "runs"),
            name="soccer_v1",
            exist_ok=True,
            # Augmentation for small object detection
            mosaic=1.0,
            mixup=0.15,
            copy_paste=0.1,
            scale=0.5,        # random scale for small object variety
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            # Training params
            patience=20,
            save=True,
            save_period=1,    # save checkpoint every epoch for resume
            val=True,
            plots=True,
            verbose=True,
        )

    # Copy best weights to models/
    os.makedirs(MODELS_DIR, exist_ok=True)
    best_pt = os.path.join(BASE_DIR, "runs", "soccer_v1", "weights", "best.pt")
    dest_pt = os.path.join(MODELS_DIR, "soccer_yolov8s.pt")

    if os.path.isfile(best_pt):
        shutil.copy2(best_pt, dest_pt)
        print(f"\n[train] Best model copied to {dest_pt}")
    else:
        # Try last.pt as fallback
        last_pt = os.path.join(BASE_DIR, "runs", "soccer_v1", "weights", "last.pt")
        if os.path.isfile(last_pt):
            shutil.copy2(last_pt, dest_pt)
            print(f"\n[train] Last model copied to {dest_pt}")
        else:
            print(f"\n[train] WARNING: Could not find trained weights")

    return dest_pt


def test_soccana_pretrained(video_path=None):
    """Quick test: download and run pre-trained Soccana model on a frame."""
    model_path = download_soccana_pretrained(MODELS_DIR)
    if model_path is None:
        return

    from ultralytics import YOLO
    import cv2

    model = YOLO(model_path)
    print(f"\n[test] Soccana model classes: {model.names}")

    if video_path and os.path.isfile(video_path):
        cap = cv2.VideoCapture(video_path)
        # Skip to 30 seconds in (likely gameplay)
        cap.set(cv2.CAP_PROP_POS_MSEC, 30000)
        ret, frame = cap.read()
        cap.release()

        if ret:
            results = model(frame, verbose=False)
            boxes = results[0].boxes
            print(f"[test] Detections on sample frame:")
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[cls]
                print(f"  {name}: conf={conf:.3f}")
            print(f"[test] Total: {len(boxes)} detections")
    else:
        print(f"[test] No video provided — model loaded successfully")
        print(f"[test] Run with: python train_model.py --test-soccana --video match.mp4")


def main():
    parser = argparse.ArgumentParser(description="Train custom soccer YOLO model")
    parser.add_argument("--download-only", action="store_true",
                        help="Only download and merge datasets, don't train")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, train on existing data/merged/")
    parser.add_argument("--test-soccana", action="store_true",
                        help="Download and test pre-trained Soccana model")
    parser.add_argument("--video", default=None,
                        help="Video path for --test-soccana")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Training epochs (default: 20)")
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size (default: 8, lower is safer on MPS)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size (default: 640)")
    parser.add_argument("--device", default=None,
                        help="Device: mps, 0 (cuda), cpu (default: auto)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint (runs/soccer_v1/weights/last.pt)")
    args = parser.parse_args()

    if args.test_soccana:
        test_soccana_pretrained(args.video)
        return

    merged_dir = os.path.join(DATA_DIR, "merged")

    if not args.skip_download:
        os.makedirs(DATA_DIR, exist_ok=True)

        print("=" * 60)
        print("  Phase 1: Download datasets")
        print("=" * 60)

        soccernet_dir = download_soccernet(DATA_DIR)
        soccana_dir = download_soccana(DATA_DIR)

        print("\n" + "=" * 60)
        print("  Phase 2: Merge and remap classes")
        print("=" * 60)

        if os.path.isdir(merged_dir):
            print(f"  Removing existing merged dir...")
            shutil.rmtree(merged_dir)

        os.makedirs(merged_dir, exist_ok=True)
        merge_datasets(soccernet_dir, soccana_dir, merged_dir)

    if not os.path.isdir(merged_dir):
        print(f"Error: merged dataset not found at {merged_dir}")
        print(f"Run without --skip-download first")
        sys.exit(1)

    data_yaml = generate_data_yaml(merged_dir)

    if args.download_only:
        print("\n[done] Datasets downloaded and merged. Run without --download-only to train.")
        return

    print("\n" + "=" * 60)
    print("  Phase 3: Train YOLOv8s")
    print("=" * 60)

    model_path = train_model(
        data_yaml,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        resume=args.resume,
    )

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Model: {model_path}")
    print(f"  Use:   python pipeline.py --model {model_path} ...")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
