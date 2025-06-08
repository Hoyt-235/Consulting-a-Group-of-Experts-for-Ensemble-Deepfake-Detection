import os
import sys
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv

# Insert the project’s “src” directory into sys.path so that absolute imports work.
src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from dataset.EnsembleDataset import DeepfakeEnsembleDataset, collate_ensemble
from models.jury.stil_detector     import STILDetector
from models.jury.ucf_detector      import UCFDetector
from models.jury.uia_vit_detector  import UIAViTDetector
from models.jury.spsl_detector     import SpslDetector as SPSLDetector
from models.ensembleModel import Judge
from models.CNNImageBranch import CNNImageBranch, CNNImageBranch_Lite

def run_single_test_epoch(model, dataloader, device):
    """
    Runs inference on one dataset and returns a list of tuples:
        (index, true_label, pred_label, pred_prob)
    """
    model.eval()
    results = []
    sample_idx = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing", leave=False):
            img1, img2, img3, img4, video, labels = batch
            img1 = img1.to(device)
            img2 = img2.to(device)
            img3 = img3.to(device)
            img4 = img4.to(device)
            video = video.to(device)
            labels = labels.long().view(-1)  # shape [B], dtype=torch.int64

            output = model(img1, img2, img3, img4, video, labels=labels)
            logits = output["logits"]               # shape [B, 2]
            probs = F.softmax(logits, dim=1)[:, 1]  # [B], probability of “fake”
            preds = (probs >= 0.5).long()           # 0 or 1

            batch_size = preds.size(0)
            true_labels = labels.cpu().numpy().astype(int).tolist()
            for i in range(batch_size):
                results.append((
                    sample_idx + i,
                    true_labels[i],
                    preds[i].item(),
                    probs[i].item()
                ))
            sample_idx += batch_size

    return results

def main(config_path: str):
    # 1) Load YAML config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) Load detector configs, instantiate branches
    yaml_paths = [
        "/workspace/src/training/config/detector/ucf.yaml",
        "/workspace/src/training/config/detector/spsl.yaml",
        "/workspace/src/training/config/detector/uia_vit.yaml",
        "/workspace/src/training/config/detector/stil.yaml",
    ]
    detector_configs = []
    for ypath in yaml_paths:
        with open(ypath, "r") as yf:
            detector_configs.append(yaml.safe_load(yf))

    branches = {}

    # UCF branch
    ucf = UCFDetector(detector_configs[0])
    ucf.to(device).eval()
    branches["ucf"] = ucf

    # SPSL branch
    spsl = SPSLDetector(detector_configs[1])
    spsl.to(device).eval()
    branches["spsl"] = spsl

    # UIA-ViT branch
    uiavit = UIAViTDetector(detector_configs[2])
    uiavit.to(device).eval()
    branches["uiavit"] = uiavit

    # STIL branch
    stil = STILDetector(detector_configs[3])
    stil.to(device).eval()
    branches["stil"] = stil

    # Image branch
    img_branch = CNNImageBranch()
    img_branch.to(device).eval()
    branches["img_branch"] = img_branch

    # 3) Build the Judge (ensemble) and move it to device
    model = Judge(branches, fusion_level=config["fusion_level"]).to(device)

    # 4) Load ensemble weights (contains all branch + fusion parameters)
    if not config.get("ensemble_weights"):
        raise ValueError("Config must specify 'ensemble_weights'.")
    ckpt = torch.load(config["ensemble_weights"], map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 5) Prepare test datasets
    test_meta = config["dataset"].get("test_metadata")
    if test_meta is None:
        raise ValueError("Config must specify 'dataset.test_metadata'.")

    if isinstance(test_meta, dict):
        test_meta = [test_meta]

    batch_size = config["testing"]["batch_size"]
    num_workers = config["testing"]["num_workers"]

    for td in test_meta:
        name = td.get("name", "testset")
        metadata_path = td["path"]
        class_keys = td["class_keys"]

        test_set = DeepfakeEnsembleDataset(
            metadata_path=metadata_path,
            seq_len=config["dataset"]["sequence_length"],
            img_size=config["dataset"]["img_size"],
            video_size=config["dataset"]["video_size"],
            pick_frame=config["dataset"].get("pick_frame", "middle"),
            class_keys=class_keys,
            subset_ratio=1.0,
            augmentation=None  # No augmentation at test time
        )
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_ensemble,
            pin_memory=True
        )

        # 6) Run inference
        print(f"Running test on dataset '{name}' ({len(test_set)} samples).")
        results = run_single_test_epoch(model, test_loader, device)

        # 7) Save predictions + true labels to CSV
        output_dir = config["testing"].get("output_dir", "./predictions")
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"{name}.csv")

        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["index", "true_label", "pred_label", "pred_prob"])
            for idx, true_lbl, pred_lbl, prob in results:
                writer.writerow([idx, true_lbl, pred_lbl, prob])

        print(f"Saved predictions for '{name}' to: {csv_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Test the EnsembleModel on one or more datasets and save predictions (with true labels) to CSV."
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()
    main(args.config)
