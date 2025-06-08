import json
import random
from pathlib import Path
from typing import List, Dict
import albumentations as A
import cv2
import os
import pprint
import torch
from albumentations.pytorch import ToTensorV2
from torchvision.transforms.functional import to_tensor
import numpy as np
from torchvision import transforms
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class DeepfakeEnsembleDataset(Dataset):
    def __init__(
        self,
        metadata_path: str,
        seq_len: int,
        img_size: int = 300,
        video_size: int = 84,
        pick_frame: str = "middle",   # "middle" or "random"
        class_keys: List[str] = None,
        subset_ratio: float = 1.0,     # fraction of the dataset to keep (0 < ratio <= 1.0)
        augmentation: A.Compose = None,  # optional A.Compose pipeline
    ):
        """
        Args:
          metadata_path : path to JSON metadata (FaceForensics++ / DFD / Celeb-DF formats)
          seq_len       : exactly how many frames per video‐chunk (e.g. 8)
          img_size      : input size for image‐based branches (e.g. 224 or 300)
          video_size    : input size for video branch (e.g. 84)
          pick_frame    : which frame from the chunk to send into image branches ("middle" or "random")
          class_keys    : top‐level keys in JSON to treat as classes
          subset_ratio  : fraction of total samples to keep (for fast testing). Must be in (0,1].
          augmentation  : optional albumentations A.compose pipeline for data augmentation
        """
        super().__init__()
        self.seq_len    = seq_len
        self.pick_frame = pick_frame
        self.augmentation = augmentation

        # 1) Load the JSON metadata
        with open(metadata_path, "r") as f:
            full_meta = json.load(f)

        # 2) Expect exactly one top‐level key
        top_keys = list(full_meta.keys())
        if len(top_keys) != 1:
            raise ValueError(f"Expected exactly one top‐level key in {metadata_path}, got {top_keys}")
        dataset_name  = top_keys[0]
        dataset_block = full_meta[dataset_name]

        # 3) Determine which subset‐names to use
        if class_keys is None:
            class_keys = list(dataset_block.keys())
        else:
            for cls in class_keys:
                if cls not in dataset_block:
                    raise KeyError(f"Subset '{cls}' not found in '{dataset_name}'")
        self.class_keys = class_keys

        # 4) Build binary labels: any subset ending in "_real" or exactly "FF-real" → 0, else 1
        self._class_to_idx = {}
        for cls in class_keys:
            lower = cls.lower()
            if lower == "ff-real" or lower.endswith("_real"):
                self._class_to_idx[cls] = 0
            else:
                self._class_to_idx[cls] = 1

        # 5) Flatten each subset into non‐overlapping seq_len‐chunks
        all_samples = []
        for cls in class_keys:
            label_idx    = self._class_to_idx[cls]
            subset_block = dataset_block[cls]  # e.g. { "train": {...}, "val": {...}, ... }

            for split in ("train","val","test"):
                if split not in subset_block:
                    continue
                split_block = subset_block[split]

                # Detect whether there's a compression layer
                first_key, first_val = next(iter(split_block.items()))
                if isinstance(first_val, dict) and "frames" in first_val:
                    compression_layer = False
                else:
                    compression_layer = True

                if not compression_layer:
                    # Format (a): { video_id: { "frames":[…] }, … }
                    for video_id, vid_info in split_block.items():
                        frame_list = vid_info.get("frames", [])
                        N = len(frame_list)
                        for start in range(0, N, seq_len):
                            end = start + seq_len
                            if end > N:
                                break
                            clip = frame_list[start:end]
                            if len(clip) == seq_len:
                                all_samples.append({
                                    "frames": clip,
                                    "label":  label_idx,
                                    "split":  split
                                })
                else:
                    # Format (b): there's a compression subfolder
                    for comp_name, comp_videos in split_block.items():
                        for video_id, vid_info in comp_videos.items():
                            frame_list = vid_info.get("frames", [])
                            N = len(frame_list)
                            for start in range(0, N, seq_len):
                                end = start + seq_len
                                if end > N:
                                    break
                                clip = frame_list[start:end]
                                if len(clip) == seq_len:
                                    all_samples.append({
                                        "frames": clip,
                                        "label":  label_idx,
                                        "split":  split
                                    })

        # 6) Optionally reduce dataset size
        if not (0.0 < subset_ratio <= 1.0):
            raise ValueError("subset_ratio must be in (0.0, 1.0].")
        if subset_ratio < 1.0:
            keep_count = int(len(all_samples) * subset_ratio)
            # random shuffle and keep only the first keep_count
            random.shuffle(all_samples)
            self.samples = all_samples[:keep_count]
        else:
            self.samples = all_samples

        # 7) Define transforms
        self.img_tfms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])
        self.video_tfms = transforms.Compose([
            transforms.Resize((video_size, video_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        samp   = self.samples[idx]
        frames = samp["frames"]  # list[str], exactly seq_len long
        label  = samp["label"]   # 0 or 1
        T      = len(frames)     # should equal self.seq_len

        # 1) Build the video tensor WITHOUT flattening channels and time
        vids = []
        for fpath in frames:
            img = Image.open(fpath).convert("RGB")
            vids.append(self.video_tfms(img))  # → each is (3, H, W)

        # Stack along a NEW “time” dimension:
        #    vid_tensor = [ T × (3, H, W) ] → shape (T, 3, H, W)
        vid_tensor = torch.stack(vids, dim=0)

        # 2) Choose one “middle” (or random) frame for the 4 image branches
        if self.pick_frame == "middle":
            fi = T // 2
        else:
            fi = random.randrange(T)
        chosen_img = Image.open(frames[fi]).convert("RGB")
        
         # 3) Apply Albumentations (if provided), else fallback to torchvision
        if self.augmentation is not None:
            # Albumentations expects NumPy array in HWC, uint8
            np_img = np.array(chosen_img)  # shape (H, W, 3), dtype=uint8
            augmented = self.augmentation(image=np_img)
            img_tensor = augmented["image"]  # already a torch.Tensor via ToTensorV2()
        else:
            img_tensor = self.img_tfms(chosen_img)

        # 3) Replicate that same single‐frame into 4 image branches
        img1 = img2 = img3 = img4 = img_tensor.float()  # each is (3, img_size, img_size)

        return img1, img2, img3, img4, vid_tensor, label


def collate_ensemble(batch):
    """
    batch is a list of tuples:
      (img1, img2, img3, img4, vid_tensor, label)

    We want to return batched tensors of shape:
      - imgs1: [B, 3, img_size, img_size]
      - imgs2: [B, 3, img_size, img_size]
      - imgs3: [B, 3, img_size, img_size]
      - imgs4: [B, 3, img_size, img_size]
      - vids : [B, T, 3, H, W]
      - labs : [B] (LongTensor)
    """
    imgs1, imgs2, imgs3, imgs4, vids, labs = zip(*batch)

    # Stack each image‐branch
    batched_img1 = torch.stack(imgs1, dim=0)
    batched_img2 = torch.stack(imgs2, dim=0)
    batched_img3 = torch.stack(imgs3, dim=0)
    batched_img4 = torch.stack(imgs4, dim=0)

    # `vids` is a tuple of `B` tensors, each shape (T,3,H,W).
    # We want one big tensor of shape [B, T, 3, H, W]:
    batched_vids = torch.stack(vids, dim=0)

    batched_labs = torch.tensor(labs, dtype=torch.long)
    return batched_img1, batched_img2, batched_img3, batched_img4, batched_vids, batched_labs
