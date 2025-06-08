# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: Abstract Base Class for all types of deepfake datasets.

import sys

import lmdb

sys.path.append('.')

import os
import math
import yaml
import glob
import json

import numpy as np
from copy import deepcopy
import cv2
import random
from PIL import Image
from collections import defaultdict

import torch
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms as T

import albumentations as A

from .albu import IsotropicResize

FFpp_pool=['FaceForensics++','FaceShifter','DeepFakeDetection','FF-DF','FF-F2F','FF-FS','FF-NT']#

def all_in_pool(inputs,pool):
    for each in inputs:
        if each not in pool:
            return False
    return True


class DeepfakeAbstractBaseDataset(data.Dataset):
    """
    Abstract base class for all deepfake datasets.
    """
    def __init__(self, config=None, mode='train'):
        """Initializes the dataset object.

        Args:
            config (dict): A dictionary containing configuration parameters.
            mode (str): A string indicating the mode (train or test).

        Raises:
            NotImplementedError: If mode is not train or test.
        """
        
        # Set the configuration and mode
        self.config = config
        self.mode = mode
        self.compression = config['compression']
        self.frame_num = config['frame_num'][mode]

        # Check if 'video_mode' exists in config, otherwise set video_level to False
        self.video_level = config.get('video_mode', False)
        self.clip_size = config.get('clip_size', None)
        self.lmdb = config.get('lmdb', False)
        # Dataset dictionary
        self.image_list = []
        self.label_list = []
        
        # Set the dataset dictionary based on the mode
        if mode == 'train':
            dataset_list = config['train_dataset']
            # Training data should be collected together for training
            image_list, label_list = [], []
            for one_data in dataset_list:
                tmp_image, tmp_label, tmp_name = self.collect_img_and_label_for_one_dataset(one_data)
                image_list.extend(tmp_image)
                label_list.extend(tmp_label)
            if self.lmdb:
                if len(dataset_list)>1:
                    if all_in_pool(dataset_list,FFpp_pool):
                        lmdb_path = os.path.join(config['lmdb_dir'], f"FaceForensics++_lmdb")
                        self.env = lmdb.open(lmdb_path, create=False, subdir=True, readonly=True, lock=False)
                    else:
                        raise ValueError('Training with multiple dataset and lmdb is not implemented yet.')
                else:
                    lmdb_path = os.path.join(config['lmdb_dir'], f"{dataset_list[0] if dataset_list[0] not in FFpp_pool else 'FaceForensics++'}_lmdb")
                    self.env = lmdb.open(lmdb_path, create=False, subdir=True, readonly=True, lock=False)
        elif mode == 'test':
            one_data = config['test_datasets']
            # Test dataset should be evaluated separately. So collect only one dataset each time
            image_list, label_list, name_list = self.collect_img_and_label_for_one_dataset(one_data)
            if self.lmdb:
                lmdb_path = os.path.join(config['lmdb_dir'], f"{one_data}_lmdb" if one_data not in FFpp_pool else 'FaceForensics++_lmdb')
                self.env = lmdb.open(lmdb_path, create=False, subdir=True, readonly=True, lock=False)
        else:
            raise NotImplementedError('Only train and test modes are supported.')

        assert len(image_list)!=0 and len(label_list)!=0, f"Collect nothing for {mode} mode!"
        self.image_list, self.label_list = image_list, label_list


        # Create a dictionary containing the image and label lists
        self.data_dict = {
            'image': self.image_list, 
            'label': self.label_list, 
        }
        
        self.transform = self.init_data_aug_method()
        

    def init_data_aug_method(self):
        """
        Initialize albumentations data augmentation pipeline.
        If `with_landmark` is True, apply a single isotropic resize to preserve keypoint consistency.
        Otherwise, randomly choose one of several isotropic resize modes.
        """
        aug_list = [
            A.HorizontalFlip(p=self.config['data_aug']['flip_prob']),
            A.Rotate(limit=self.config['data_aug']['rotate_limit'], p=self.config['data_aug']['rotate_prob']),
            A.GaussianBlur(blur_limit=self.config['data_aug']['blur_limit'], p=self.config['data_aug']['blur_prob']),
        ]

        # Conditional isotropic resize
        if self.config['with_landmark']:
            resize_aug = A.Resize(height=self.config['resolution'],
                         width=self.config['resolution'],
                                interpolation=cv2.INTER_AREA,
                                mask_interpolation=cv2.INTER_CUBIC)
        else:
            resize_aug = A.OneOf([
                A.Resize(height=self.config['resolution'],
                         width=self.config['resolution'],
                                interpolation=cv2.INTER_AREA,
                                mask_interpolation=cv2.INTER_CUBIC),
                A.Resize(height=self.config['resolution'],
                         width=self.config['resolution'],
                                interpolation=cv2.INTER_AREA,
                                mask_interpolation=cv2.INTER_LINEAR),
                A.Resize(height=self.config['resolution'],
                         width=self.config['resolution'],
                                interpolation=cv2.INTER_LINEAR,
                                mask_interpolation=cv2.INTER_LINEAR),
            ], p=1.0)
        aug_list.append(resize_aug)

        # Color/image augmentations
        aug_list.extend([
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=self.config['data_aug']['brightness_limit'],
                                        contrast_limit=self.config['data_aug']['contrast_limit']),
                A.FancyPCA(p=0.5),
                A.HueSaturationValue(p=0.5),
            ], p=0.5),
            A.ImageCompression(quality_range=(self.config['data_aug']['quality_lower'],
                                              self.config['data_aug']['quality_upper']),
                            p=0.5),
        ])

        return A.Compose(
            aug_list,
            keypoint_params=A.KeypointParams(format='xy') if self.config['with_landmark'] else None
        )


    def rescale_landmarks(self, landmarks, original_size=256, new_size=224):
        scale_factor = new_size / original_size
        rescaled_landmarks = landmarks * scale_factor
        return rescaled_landmarks

    
    def collect_img_and_label_for_one_dataset(self, dataset_name: str):
        """Collects image and label lists.
    
        Args:
            dataset_name (str): A string like 'FF-F2F' or 'FaceForensics++_c40'.
    
        Returns:
            frame_path_list: list of (for frame‐level) strings or (for video‐level) lists of strings
            label_list: list of corresponding integer labels
            video_name_list: list of “unique_video_name” strings (one per frame or per clip)
    
        Raises:
            ValueError: If no valid frames/labels are found, or if a label is missing in config.
        """
        label_list = []
        frame_path_list = []
        video_name_list = []
    
        # Locate the JSON folder; apply fallback if needed
        if not os.path.exists(self.config["dataset_json_folder"]):
            self.config["dataset_json_folder"] = \
            self.config["dataset_json_folder"].replace(
                    "/Youtu_Pangu_Security_Public",
                    "/Youtu_Pangu_Security/public"
                )
    
        # Load the dataset JSON
        try:
            with open(
                os.path.join(self.config["dataset_json_folder"], dataset_name + ".json"),
                "r"
            ) as f:
                dataset_info = json.load(f)
        except Exception as e:
            print(e)
            raise ValueError(f"dataset {dataset_name} does not exist!")
    
        # Handle any “_c40” suffix by stripping it and remembering compression “cp”
        cp = None
        if dataset_name.endswith("_c40"):
            base = dataset_name.rsplit("_", 1)[0]
            dataset_name = base
            cp = "c40"
    
        # For each label (“real” / “fake”) in the top‐level JSON:
        for label_str in dataset_info[dataset_name]:
            sub_dataset_info = dataset_info[dataset_name][label_str][self.mode]
    
            # If this dataset requires choosing a compression type:
            if (
                cp is None
                and dataset_name
                in [
                    "FF-DF",
                    "FF-F2F",
                    "FF-FS",
                    "FF-NT",
                    "FaceForensics++",
                    "DeepFakeDetection",
                    "FaceShifter",
                ]
            ):
                sub_dataset_info = sub_dataset_info[self.compression]
            elif (
                cp == "c40"
                and dataset_name
                in [
                    "FF-DF",
                    "FF-F2F",
                    "FF-FS",
                    "FF-NT",
                    "FaceForensics++",
                    "DeepFakeDetection",
                    "FaceShifter",
                ]
            ):
                sub_dataset_info = sub_dataset_info["c40"]
    
            # Iterate over each video in that sub‐dictionary
            for video_name, video_info in sub_dataset_info.items():
                unique_video_name = f"{video_info['label']}_{video_name}"
    
                # Lookup the integer label from config
                orig_label_str = video_info["label"]
                if orig_label_str not in self.config["label_dict"]:
                    raise ValueError(
                        f"Label {orig_label_str} not found in configuration."
                    )
                label_int = self.config["label_dict"][orig_label_str]
    
                # Grab the raw list of “frame‐paths” (strings) from JSON
                raw_frame_paths = video_info.get("frames", [])
                if not raw_frame_paths:
                    # No frames listed—skip this video
                    continue
    
                # Filter out any paths whose filename stem isn’t an integer
                valid_frame_paths = []
                for p in raw_frame_paths:
                    # Normalize slashes, then split off the filename
                    stem = p.replace("\\", "/").split("/")[-1].split(".")[0]
                    try:
                        _ = int(stem)  # if this fails, skip path
                        valid_frame_paths.append(p)
                    except ValueError:
                        continue
    
                if not valid_frame_paths:
                    # After filtering, no valid numeric‐stem frames remain—skip
                    continue
    
                # Now sort by that integer stem:
                def extract_index(path_str: str) -> int:
                    fname = path_str.replace("\\", "/").split("/")[-1]
                    return int(fname.split(".")[0])
    
                valid_frame_paths.sort(key=extract_index)
                total_frames = len(valid_frame_paths)
    
                # If there are more frames than self.frame_num, subsample:
                if self.frame_num < total_frames:
                    # In both modes, we will end up with exactly self.frame_num paths
                    if self.video_level:
                        # video‐level: pick a continuous clip of length frame_num
                        start_max = total_frames - self.frame_num
                        if self.mode == "train":
                            start_frame = random.randint(0, start_max)
                        else:
                            start_frame = 0
                        valid_frame_paths = valid_frame_paths[
                            start_frame : start_frame + self.frame_num
                        ]
                        total_frames = self.frame_num
                    else:
                        # frame‐level: select evenly spaced frames
                        step = total_frames // self.frame_num
                        selected = []
                        for i in range(0, total_frames, step):
                            selected.append(valid_frame_paths[i])
                            if len(selected) == self.frame_num:
                                break
                        valid_frame_paths = selected
                        total_frames = self.frame_num
    
                # If we are doing “video_level” (i.e. learning from clips of size clip_size):
                if self.video_level:
                    if self.clip_size is None:
                        raise ValueError(
                            "clip_size must be specified when video_level is True."
                        )
    
                    if total_frames >= self.clip_size:
                        # We will slice this sorted valid_frame_paths into contiguous clips
                        # of length clip_size, possibly multiple clips in one video.
                        selected_clips = []
                        num_clips = total_frames // self.clip_size
    
                        if num_clips > 1:
                            clip_step = (total_frames - self.clip_size) // (num_clips - 1)
                            for i in range(num_clips):
                                if self.mode == "train":
                                    start_frame = random.randrange(
                                        i * clip_step,
                                        min((i + 1) * clip_step, total_frames - self.clip_size + 1),
                                    )
                                else:
                                    start_frame = i * clip_step
                                contig = valid_frame_paths[
                                    start_frame : start_frame + self.clip_size
                                ]
                                # Should always be exactly clip_size long
                                assert len(contig) == self.clip_size, (
                                    "clip_size mismatch"
                                )
                                selected_clips.append(contig)
                        else:
                            # Only one clip possible
                            if self.mode == "train":
                                start_frame = random.randrange(0, total_frames - self.clip_size + 1)
                            else:
                                start_frame = 0
                            contig = valid_frame_paths[
                                start_frame : start_frame + self.clip_size
                            ]
                            assert len(contig) == self.clip_size, "clip_size mismatch"
                            selected_clips.append(contig)
    
                        # Append each clip separately, with the same label:
                        label_list.extend([label_int] * len(selected_clips))
                        frame_path_list.extend(selected_clips)
                        video_name_list.extend([unique_video_name] * len(selected_clips))
                    else:
                        # Not enough frames to form even one clip
                        print(
                            f"Skipping video {unique_video_name} "
                            f"because it has less than clip_size ({self.clip_size}) frames ({total_frames})."
                        )
                else:
                    # Frame‐level mode: treat each of the “total_frames” as a separate sample
                    label_list.extend([label_int] * total_frames)
                    frame_path_list.extend(valid_frame_paths)
                    video_name_list.extend([unique_video_name] * total_frames)
    
        # After collecting from all videos, make sure we actually have some data
        if len(frame_path_list) == 0 or len(label_list) == 0:
            raise ValueError(
                f"No valid frames/labels found for dataset {dataset_name}. "
                "Check your JSON file or frame naming."
            )
    
        # Finally, shuffle everything in unison:
        combined = list(zip(label_list, frame_path_list, video_name_list))
        random.shuffle(combined)
        label_list, frame_path_list, video_name_list = zip(*combined)
        return frame_path_list, label_list, video_name_list


     
    def load_rgb(self, file_path):
        """
        Load an RGB image from a file path and resize it to a specified resolution.

        Args:
            file_path: A string indicating the path to the image file.

        Returns:
            An Image object containing the loaded and resized image.

        Raises:
            ValueError: If the loaded image is None.
        """
        size = self.config['resolution'] # if self.mode == "train" else self.config['resolution']
        if not self.lmdb:
            if not file_path[0] == '.':
                file_path =  file_path
            assert os.path.exists(file_path), f"{file_path} does not exist"
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError('Loaded image is None: {}'.format(file_path))
        elif self.lmdb:
            with self.env.begin(write=False) as txn:
                # transfer the path format from rgb-path to lmdb-key
                if file_path[0]=='.':
                    file_path=file_path.replace('./datasets\\','')

                image_bin = txn.get(file_path.encode())
                image_buf = np.frombuffer(image_bin, dtype=np.uint8)
                img = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(np.array(img, dtype=np.uint8))


    def load_mask(self, file_path):
        """
        Load a binary mask image from a file path and resize it to a specified resolution.

        Args:
            file_path: A string indicating the path to the mask file.

        Returns:
            A numpy array containing the loaded and resized mask.

        Raises:
            None.
        """
        size = self.config['resolution']
        if file_path is None:
            return np.zeros((size, size, 1))
        if not self.lmdb:
            if not file_path[0] == '.':
                file_path =  f'./{self.config["rgb_dir"]}\\'+file_path
            if os.path.exists(file_path):
                mask = cv2.imread(file_path, 0)
                if mask is None:
                    mask = np.zeros((size, size))
            else:
                return np.zeros((size, size, 1))
        else:
            with self.env.begin(write=False) as txn:
                # transfer the path format from rgb-path to lmdb-key
                if file_path[0]=='.':
                    file_path=file_path.replace('./datasets\\','')

                image_bin = txn.get(file_path.encode())
                if image_bin is None:
                    mask = np.zeros((size, size,3))
                else:
                    image_buf = np.frombuffer(image_bin, dtype=np.uint8)
                    # cv2.IMREAD_GRAYSCALE为灰度图，cv2.IMREAD_COLOR为彩色图
                    mask = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
        mask = cv2.resize(mask, (size, size)) / 255
        mask = np.expand_dims(mask, axis=2)
        return np.float32(mask)

    def load_landmark(self, file_path):
        """
        Load 2D facial landmarks from a file path.

        Args:
            file_path: A string indicating the path to the landmark file.

        Returns:
            A numpy array containing the loaded landmarks.

        Raises:
            None.
        """
        if file_path is None:
            return np.zeros((81, 2))
        if not self.lmdb:
            if not file_path[0] == '.':
                file_path =  f'{self.config["rgb_dir"]}/'+file_path
            if os.path.exists(file_path):
                landmark = np.load(file_path)
            else:
                return np.zeros((81, 2))
        else:
            with self.env.begin(write=False) as txn:
                # transfer the path format from rgb-path to lmdb-key
                if file_path[0]=='.':
                    file_path=file_path.replace('./datasets\\','')
                binary = txn.get(file_path.encode())
                landmark = np.frombuffer(binary, dtype=np.uint32).reshape((81, 2))
                landmark=self.rescale_landmarks(np.float32(landmark), original_size=256, new_size=self.config['resolution'])
        return landmark

    def to_tensor(self, img):
        """
        Convert an image to a PyTorch tensor.
        """
        return T.ToTensor()(img)

    def normalize(self, img):
        """
        Normalize an image.
        """
        mean = self.config['mean']
        std = self.config['std']
        normalize = T.Normalize(mean=mean, std=std)
        return normalize(img)

    def data_aug(self, img, landmark=None, mask=None, augmentation_seed=None):
        """
        Apply data augmentation to an image, landmark, and mask.

        Args:
            img: An Image object containing the image to be augmented.
            landmark: A numpy array containing the 2D facial landmarks to be augmented.
            mask: A numpy array containing the binary mask to be augmented.

        Returns:
            The augmented image, landmark, and mask.
        """

        # Set the seed for the random number generator
        if augmentation_seed is not None:
            random.seed(augmentation_seed)
            np.random.seed(augmentation_seed)
        
        # Create a dictionary of arguments
        kwargs = {'image': img}
        
        # Check if the landmark and mask are not None
        if landmark is not None:
            kwargs['keypoints'] = landmark
            kwargs['keypoint_params'] = A.KeypointParams(format='xy')
        if mask is not None:
            mask = mask.squeeze(2)
            if mask.max() > 0:
                kwargs['mask'] = mask

        # Apply data augmentation
        transformed = self.transform(**kwargs)
        
        # Get the augmented image, landmark, and mask
        augmented_img = transformed['image']
        augmented_landmark = transformed.get('keypoints')
        augmented_mask = transformed.get('mask',mask)

        # Convert the augmented landmark to a numpy array
        if augmented_landmark is not None:
            augmented_landmark = np.array(augmented_landmark)

        # Reset the seeds to ensure different transformations for different videos
        if augmentation_seed is not None:
            random.seed()
            np.random.seed()

        return augmented_img, augmented_landmark, augmented_mask

    def __getitem__(self, index, no_norm=False):
        """
        Returns the data point at the given index.

        Args:
            index (int): The index of the data point.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Get the image paths and label
        image_paths = self.data_dict['image'][index]
        label = self.data_dict['label'][index]

        if not isinstance(image_paths, list):
            image_paths = [image_paths]  # for the image-level IO, only one frame is used

        image_tensors = []
        landmark_tensors = []
        mask_tensors = []
        augmentation_seed = None

        for image_path in image_paths:
            # Initialize a new seed for data augmentation at the start of each video
            if self.video_level and image_path == image_paths[0]:
                augmentation_seed = random.randint(0, 2**32 - 1)

            # Get the mask and landmark paths
            mask_path = image_path.replace('frames', 'masks')  # Use .png for mask
            landmark_path = image_path.replace('frames', 'landmarks').replace('.png', '.npy')  # Use .npy for landmark

            # Load the image
            try:
                image = self.load_rgb(image_path.replace('\\', '/'))
            except Exception as e:
                # Skip this image and return the first one
                print(f"Error loading image at index {index}: {e}")
                return self.__getitem__(0)
            image = np.array(image)  # Convert to numpy array for data augmentation

            # Load mask and landmark (if needed)
            if self.config['with_mask']:
                mask = self.load_mask(mask_path)
            else:
                mask = None
            if self.config['with_landmark']:
                landmarks = self.load_landmark(landmark_path)
            else:
                landmarks = None

            # Do Data Augmentation
            if self.mode == 'train' and self.config['use_data_augmentation']:
                image_trans, landmarks_trans, mask_trans = self.data_aug(image, landmarks, mask, augmentation_seed)
            else:
                image_trans, landmarks_trans, mask_trans = deepcopy(image), deepcopy(landmarks), deepcopy(mask)
            

            # To tensor and normalize
            if not no_norm:
                image_trans = self.normalize(self.to_tensor(image_trans))
                if self.config['with_landmark']:
                    landmarks_trans = torch.from_numpy(landmarks)
                if self.config['with_mask']:
                    mask_trans = torch.from_numpy(mask_trans)

            image_tensors.append(image_trans)
            landmark_tensors.append(landmarks_trans)
            mask_tensors.append(mask_trans)

        if self.video_level:
            # Stack image tensors along a new dimension (time)
            image_tensors = torch.stack(image_tensors, dim=0)
            # Stack landmark and mask tensors along a new dimension (time)
            if not any(landmark is None or (isinstance(landmark, list) and None in landmark) for landmark in landmark_tensors):
                landmark_tensors = torch.stack(landmark_tensors, dim=0)
            if not any(m is None or (isinstance(m, list) and None in m) for m in mask_tensors):
                mask_tensors = torch.stack(mask_tensors, dim=0)
        else:
            # Get the first image tensor
            image_tensors = image_tensors[0]
            # Get the first landmark and mask tensors
            if not any(landmark is None or (isinstance(landmark, list) and None in landmark) for landmark in landmark_tensors):
                landmark_tensors = landmark_tensors[0]
            if not any(m is None or (isinstance(m, list) and None in m) for m in mask_tensors):
                mask_tensors = mask_tensors[0]

        return image_tensors, label, landmark_tensors, mask_tensors
    
    @staticmethod
    def collate_fn(batch):
        """
        Collate a batch of data points.

        Args:
            batch (list): A list of tuples containing the image tensor, the label tensor,
                          the landmark tensor, and the mask tensor.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Separate the image, label, landmark, and mask tensors
        images, labels, landmarks, masks = zip(*batch)
        
        # Stack the image, label, landmark, and mask tensors
        images = torch.stack(images, dim=0)
        labels = torch.LongTensor(labels)
        
        # Special case for landmarks and masks if they are None
        if not any(landmark is None or (isinstance(landmark, list) and None in landmark) for landmark in landmarks):
            landmarks = torch.stack(landmarks, dim=0)
        else:
            landmarks = None

        if not any(m is None or (isinstance(m, list) and None in m) for m in masks):
            masks = torch.stack(masks, dim=0)
        else:
            masks = None

        # Create a dictionary of the tensors
        data_dict = {}
        data_dict['image'] = images
        data_dict['label'] = labels
        data_dict['landmark'] = landmarks
        data_dict['mask'] = masks
        return data_dict

    def __len__(self):
        """
        Return the length of the dataset.

        Args:
            None.

        Returns:
            An integer indicating the length of the dataset.

        Raises:
            AssertionError: If the number of images and labels in the dataset are not equal.
        """
        assert len(self.image_list) == len(self.label_list), 'Number of images and labels are not equal'
        return len(self.image_list)


if __name__ == "__main__":
    with open('/data/home/zhiyuanyan/DeepfakeBench/training/config/detector/video_baseline.yaml', 'r') as f:
        config = yaml.safe_load(f)
    train_set = DeepfakeAbstractBaseDataset(
                config = config,
                mode = 'train', 
            )
    train_data_loader = \
        torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config['train_batchSize'],
            shuffle=True, 
            num_workers=0,
            collate_fn=train_set.collate_fn,
        )
    from tqdm import tqdm
    for iteration, batch in enumerate(tqdm(train_data_loader)):
        # print(iteration)
        ...
        # if iteration > 10:
        #     break
