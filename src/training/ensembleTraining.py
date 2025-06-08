import os, sys

# Insert the project’s “src” directory into sys.path so that absolute imports work.
# Adjust this path if your script lives elsewhere.
src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  
if src_root not in sys.path:
    sys.path.insert(0, src_root)
import os
import time
import copy
import torch
import torch.nn as nn
import yaml
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import albumentations as A
import cv2
import os
import pprint
import torch
from albumentations.pytorch import ToTensorV2
from torchvision.transforms.functional import to_tensor
import numpy as np
from torchvision import transforms
from dataset.EnsembleDataset import DeepfakeEnsembleDataset, collate_ensemble
from models.jury.stil_detector     import STILDetector
from models.jury.ucf_detector      import UCFDetector
from models.jury.uia_vit_detector  import UIAViTDetector  
from models.jury.spsl_detector     import SpslDetector    as SPSLDetector
from models.ensembleModel import Judge, JudgeLoss
from models.CNNImageBranch import CNNImageBranch
from tqdm import tqdm
#import torch.backends.cudnn as cudnn
#cudnn.benchmark = True

# Stochastic data augmentation pipeline:

dataAugmentation = A.Compose([
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=(-0.5, 0.5), contrast_limit=0.0, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=(-0.5, 0.5), p=1.0),
        ], p=1.0),

        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 19), p=1.0),
            A.Blur(blur_limit=(3, 19), p=1.0),
        ], p=1.0),

        A.OneOf([
            A.FancyPCA(alpha=0.5, p=1.0),
            A.CoarseDropout(num_holes_range=(1,3),
                            hole_height_range=(0.1,0.2),
                            hole_width_range=(0.1,0.2),
                            fill="random",
                            fill_mask=None,
                            p=1.0),
        ], p=1.0),

        A.OneOf([
            A.HorizontalFlip(p=1.0),
            A.Rotate(limit=(-90, 90),
                     border_mode=cv2.BORDER_REPLICATE,
                     p=1.0),
            A.VerticalFlip(p=1.0),
        ], p=1.0),

        A.OneOf([
            A.GaussNoise(std_range=(0.1,0.1), mean_range=(0,0), p=0.1),
            A.ImageCompression(quality_range=(1,10), p=0.9),
        ], p=1.0),
      A.Resize(224, 224),
      A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
      A.ToTensorV2()
    ], p=1.0,  # apply the top-level montage with probability 1
       seed=1024)


def train_ensemble(
    model,
    train_loader,
    val_loader,
    device,
    learning_rate=None,
    criterion=None,
    optimizer=None,
    scheduler=None,
    weight_decay=None,
    T_max=None,
    eta_min=None,
    last_epoch=None,
    num_epochs=50,
    patience= None,
    checkpoint_dir="checkpoints",
    log_dir="/workspace/runs/ensemble_v2_w_aug",
    start_epoch=0
):
    """
    Robust training + validation loop for an ensemble deepfake detection model.

    Features:
    - Checkpointing (saves best model based on validation loss)
    - Early stopping (stops if no improvement for `patience` epochs)
    - TensorBoard logging (train/val losses, accuracies, and parameter histograms)
    - Graceful handling of KeyboardInterrupt (saves last state)

    Args:
        model (nn.Module): PyTorch model (ensemble) to train.
        train_loader (DataLoader): DataLoader for training set.
        val_loader (DataLoader): DataLoader for validation set.
        device (torch.device): CPU or CUDA device.
        criterion (nn.Module, optional): Loss function. Defaults to BCEWithLogitsLoss.
        optimizer (torch.optim.Optimizer, optional): Optimizer. Defaults to Adam.
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): LR scheduler. Defaults to None.
        num_epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience (in epochs).
        checkpoint_dir (str): Directory to save checkpoints.
        log_dir (str): Directory for TensorBoard logs.
        start_epoch (int): Epoch to start from (for resuming).
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    

    if criterion is None:
        # Binary classification: use BCEWithLogitsLoss
        criterion = nn.BCEWithLogitsLoss()

    if not optimizer:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if scheduler == "CosineAnnealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=T_max,
                                                               eta_min=eta_min,
                                                               last_epoch=last_epoch)
        

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    epochs_no_improve = 0


    try:
        
        for epoch in range(start_epoch, num_epochs + 1):
            epoch_start = time.time()
            print(f"=== Epoch {epoch+1}/{num_epochs} ===")

            # --- Training Phase ---
            model.train()
            train_losses = []
            train_preds = []
            train_labels = []

            pbar = tqdm(train_loader, desc="Train", leave=False)
            for batch in pbar:
                img1, img2, img3, img4, video, labels = batch
                img1 = img1.to(device)
                img2 = img2.to(device)
                img3 = img3.to(device)
                img4 = img4.to(device)
                video  = video.to(device)
                labels = labels.to(device).long().view(-1)  # → shape [B], dtype=torch.int64
                optimizer.zero_grad()
                output = model(img1, img2, img3, img4, video,labels)
                loss, loss_dict = criterion(output, labels)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

                # predictions for accuracy
                logits = output["logits"]              # shape [B, 2]
                probs = F.softmax(logits, dim=1)[:, 1]  # → [B], probability of “fake” class
                preds = (probs >= 0.5).long()          # 0 or 1
                train_preds.extend(preds.tolist())
                train_labels.extend(labels.cpu().numpy().astype(int).flatten().tolist())
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            # Average training metrics
            avg_train_loss = sum(train_losses) / len(train_losses)
            train_acc = accuracy_score(train_labels, train_preds)

            writer.add_scalar("Loss/train", avg_train_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)


            # --- Validation Phase ---
            model.eval()
            val_losses = []
            val_preds = []
            val_labels = []

            with torch.no_grad():
                pbar = tqdm(val_loader, desc="Val", leave=False)
                for batch in pbar:
                    img1, img2, img3, img4, video, labels = batch
                    img1 = img1.to(device)
                    img2 = img2.to(device)
                    img3 = img3.to(device)
                    img4 = img4.to(device)
                    video  = video.to(device)
                    labels = labels.to(device).long().view(-1)  # → shape [B], dtype=torch.int64
                    output = model(img1, img2, img3, img4, video, labels)
                    loss, loss_dict = criterion(output, labels)
                    val_losses.append(loss.item())


                    logits = output["logits"]              # shape [B, 2]
                    probs = F.softmax(logits, dim=1)[:, 1]  # → [B], probability of “fake” class
                    preds = (probs >= 0.5).long()          # 0 or 1
                    val_preds.extend(preds.tolist())
                    val_labels.extend(labels.cpu().numpy().astype(int).flatten().tolist())

                    pbar.set_postfix(val_loss=f"{loss.item():.4f}")

            avg_val_loss = sum(val_losses) / len(val_losses)
            val_acc = accuracy_score(val_labels, val_preds)

            writer.add_scalar("Loss/val", avg_val_loss, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)

            print(
                f"Epoch {epoch+1} | "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f} | "
                f"Time: {(time.time()-epoch_start):.1f}s"
            )

            # Check for improvement
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0

                ckpt_path = os.path.join(
                    checkpoint_dir, f"best_epoch_{epoch+1:03d}_val_loss_{avg_val_loss:.4f}.pth"
                )
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "val_acc": val_acc
                }, ckpt_path)
                print(f"  ✓ Checkpoint saved to {ckpt_path}")
            else:
                epochs_no_improve += 1
                print(f"  No improvement for {epochs_no_improve}/{patience} epochs.")

            # Early stopping
            if int(epochs_no_improve) >= int(patience):
                print(f"Early stopping at epoch {epoch+1}.")
                break

            # Step LR scheduler (if any)
            if scheduler:
                scheduler.step()

    except KeyboardInterrupt:
        print("\n KeyboardInterrupt caught. Saving last model state...")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        interrupt_ckpt = os.path.join(checkpoint_dir, f"interrupt_{timestamp}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss
        }, interrupt_ckpt)
        print(f"  Last model state saved to {interrupt_ckpt}. Exiting.")

    # Load best model weights before returning
    model.load_state_dict(best_model_wts)
    return model, best_val_loss


def main(config_path: str):
    from torch.utils.data import DataLoader

    # 1) Load YAML
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)


    # 3) Instantiate models, build ensemble and move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1) Load all four detector configs
    yaml_paths = [
        "/workspace/src/training/config/detector/ucf.yaml",
        "/workspace/src/training/config/detector/spsl.yaml",
        "/workspace/src/training/config/detector/uia_vit.yaml",
        "/workspace/src/training/config/detector/stil.yaml",
    ]
    configs =[]
    for file in yaml_paths:
        with open(file,'r') as y:
            configs.append(yaml.safe_load(y))
    branches = {}

        # SPSL branch
    spsl_cfg = configs[1]
    spsl = SPSLDetector(spsl_cfg)
    if config["spsl_weights"]:
        ckpt = torch.load(config["spsl_weights"], map_location=device)
        spsl.load_state_dict(ckpt)
    spsl.to(device).eval()
    branches["spsl"] = spsl
    
    # UCF branch
    ucf_cfg = configs[0]
    ucf = UCFDetector(ucf_cfg)           # instantiate
    if config["ucf_weights"]:
        ckpt = torch.load(config["ucf_weights"], map_location=device)
        ucf.load_state_dict(ckpt)
    ucf.to(device).eval()
    branches["ucf"] = ucf
    
    # UIA‐ViT branch
    uiavit_cfg = configs[2]
    uiavit = UIAViTDetector(uiavit_cfg)
    if config["uia_vit_weights"]:
        ckpt = torch.load(config["uia_vit_weights"], map_location=device)
        uiavit.load_state_dict(ckpt)
    uiavit.to(device).eval()
    branches["uiavit"] = uiavit
    
    # STIL branch
    stil_cfg = configs[3]
    stil = STILDetector(stil_cfg)
    if config["stil_weights"]:
        ckpt = torch.load(config["stil_weights"], map_location=device)
        stil.load_state_dict(ckpt)
    stil.to(device).eval()
    branches["stil"] = stil

    #Image Branch
    img_branch = CNNImageBranch()
    img_branch.to(device)
    branches["img_branch"] = img_branch
    
    # 3) Finally, build the Judge (ensemble) and move it to the same device
    model = Judge(branches, fusion_level=config["fusion_level"]).to(device)
    # 1) Freeze every sub‐detector entirely:
    for name, submodel in model.branches.items():
        for p in submodel.parameters():
            p.requires_grad = False
    # 3) Confirm which parameters remain trainable:
    trainable_params = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            trainable_params.append((name, p))
    print(">>> Trainable parameters:")
    for name, p in trainable_params:
        print("   ", name, "\t", p.shape)

    criterion = JudgeLoss(fusion_mode=config["fusion_level"],
                         λ=config["loss_weights"]["lambda_balance"],
                         μ=config["loss_weights"]["mu_alignment"])
    optimizer = torch.optim.Adam([p for _, p in trainable_params], lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"])


    # 4) Create DataLoaders 
    train_set = DeepfakeEnsembleDataset(
        metadata_path=config["dataset"]["train_metadata"]["path"],
        seq_len=config["dataset"]["sequence_length"],
        img_size=config["dataset"]["img_size"],
        video_size=config["dataset"]["video_size"],
        pick_frame=config["dataset"].get("pick_frame", "middle"),
        class_keys= config["dataset"]["train_metadata"]["class_keys"],
        subset_ratio=1.0,
        augmentation=dataAugmentation
    )
    train_loader = DataLoader(
            train_set,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=config["training"]["num_workers"],
            collate_fn=collate_ensemble,
            pin_memory=True
    )
    val_set = DeepfakeEnsembleDataset(
        metadata_path=config["dataset"]["val_metadata"]["path"],
        seq_len=config["dataset"]["sequence_length"],
        img_size=config["dataset"]["img_size"],
        video_size=config["dataset"]["video_size"],
        pick_frame=config["dataset"].get("pick_frame", "middle"),
        class_keys= config["dataset"]["val_metadata"]["class_keys"],
        subset_ratio=1.0
    )
    val_loader = DataLoader(
            val_set,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=config["training"]["num_workers"],
            collate_fn=collate_ensemble,
            pin_memory=True
    )

        
    # 5) Call the training function with parsed config values
    trained_model, best_loss = train_ensemble(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=config["training"].get("scheduler", None),
        weight_decay=config["training"].get("weight_decay", 0.0),
        T_max=config["training"]["T_max"],
        eta_min=config["training"]["eta_min"],
        last_epoch=config["training"]["last_epoch"],
        num_epochs=config["training"]["epochs"],
        patience=config["training"]["patience"],
        learning_rate=config["training"]["learning_rate"],
        checkpoint_dir=config["logging"]["checkpoint_dir"],
        log_dir=config["logging"]["log_dir"],
        start_epoch=0
    )

    print(f"Training complete. Best validation loss: {best_loss:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Train the EnsembleModel using a single YAML config file."
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()
    main(args.config)
