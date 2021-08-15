
import os

import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image, ImageFile
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from wtfml.utils.utils import get_one_hot_target_for_training
from wtfml.data_loaders.image.image_augumentation import image_augmentations, mask_augmentations
from wtfml.data_loaders.image.landmark_augumentation import rotation_landmark

ImageFile.LOAD_TRUNCATED_IMAGES = True



class FaceMaskDataset(data.Dataset):
    def __init__(
        self,
        image_paths,
        dataframe,
        mask_folder,
        model_name="ir50",
        image_augmentations=image_augmentations,
        mask_augmentations=mask_augmentations,
        channel_first=False,
        mode="train",
    ):
        self.image_paths = image_paths
        self.mask_folder = mask_folder
        self.mask_augmentations = mask_augmentations
        self.image_augmentations = image_augmentations
        self.channel_first = channel_first
        self.mode = mode
        self.dataframe = dataframe
        self.model_name = model_name

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image_path = self.image_paths[item]
        image = Image.open(image_path)
        image = np.array(image)
        if self.image_augmentations is not None:
            image = self.image_augmentations(image, mode=self.mode, model_name=self.model_name, image_type = "face")
        if self.channel_first:
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        mask_path = os.path.join(
            self.mask_folder,
            "/".join(image_path.split("/")[-2:]).split(".")[0] + ".png",
        )
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask.astype(float)
        mask = self.mask_augmentations(mask)
        target_series = self.dataframe.iloc[item]
        target = get_one_hot_target_for_training(target_series, sub_adjustment=0.6)
        return image, mask, target

class FaceMaskDatasetWithEye(data.Dataset):
    def __init__(
        self,
        image_paths,
        dataframe,
        mask_folder,
        eye_folder, 
        model_name="ir50",
        image_augmentations=image_augmentations,
        mask_augmentations=mask_augmentations,
        channel_first=False,
        mode="train",
    ):
        self.image_paths = image_paths
        self.mask_folder = mask_folder
        self.mask_augmentations = mask_augmentations
        self.image_augmentations = image_augmentations
        self.channel_first = channel_first
        self.mode = mode
        self.dataframe = dataframe
        self.model_name = model_name
        self.eye_folder = eye_folder

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image_path = self.image_paths[item]
        image = Image.open(image_path)
        image = np.array(image)
        
        right_eye_image_path = os.path.join(
            self.eye_folder,
            "/".join([image_path.split("/")[-2],"right_eye_" + image_path.split("/")[-1]]),
        )
        right_eye_image = Image.open(right_eye_image_path)
        right_eye_image = np.array(right_eye_image)

        left_eye_image_path = os.path.join(
            self.eye_folder,
            "/".join([image_path.split("/")[-2],"left_eye_" + image_path.split("/")[-1]]),
        )
        left_eye_image = Image.open(left_eye_image_path)
        left_eye_image = np.array(left_eye_image)


        if self.image_augmentations is not None:
            image = self.image_augmentations(image, mode=self.mode, model_name=self.model_name,image_type = "face")
            right_eye_image = self.image_augmentations(right_eye_image, mode=self.mode, image_type = "eye")
            left_eye_image = self.image_augmentations(left_eye_image, mode=self.mode, image_type = "eye")

        if self.channel_first:
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)


        mask_path = os.path.join(
            self.mask_folder,
            "/".join(image_path.split("/")[-2:]).split(".")[0] + ".png",
        )
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask.astype(float)
        mask = self.mask_augmentations(mask)


        target_series = self.dataframe.iloc[item]
        target = get_one_hot_target_for_training(target_series, sub_adjustment=0.6)
        return image,right_eye_image , left_eye_image, mask, target
    
    
    
    
class FaceMaskDatasetWithEyeAndLandmarks(data.Dataset):
    def __init__(
        self,
        image_paths,
        dataframe,
        mask_folder,
        eye_folder, 
        model_name="ir50",
        image_augmentations=image_augmentations,
        mask_augmentations=mask_augmentations,
        channel_first=False,
        mode="train",
    ):
        self.image_paths = image_paths
        self.mask_folder = mask_folder
        self.mask_augmentations = mask_augmentations
        self.image_augmentations = image_augmentations
        self.channel_first = channel_first
        self.mode = mode
        self.dataframe = dataframe
        self.model_name = model_name
        self.eye_folder = eye_folder

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image_path = self.image_paths[item]
        image = Image.open(image_path)
        image = np.array(image)
        
        right_eye_image_path = os.path.join(
            self.eye_folder,
            "/".join([image_path.split("/")[-2],"right_eye_" + image_path.split("/")[-1]]),
        )
        right_eye_image = Image.open(right_eye_image_path)
        right_eye_image = np.array(right_eye_image)

        left_eye_image_path = os.path.join(
            self.eye_folder,
            "/".join([image_path.split("/")[-2],"left_eye_" + image_path.split("/")[-1]]),
        )
        left_eye_image = Image.open(left_eye_image_path)
        left_eye_image = np.array(left_eye_image)


        if self.image_augmentations is not None:
            image = self.image_augmentations(image, mode=self.mode, model_name=self.model_name,image_type = "face")
            right_eye_image = self.image_augmentations(right_eye_image, mode=self.mode, image_type = "eye")
            left_eye_image = self.image_augmentations(left_eye_image, mode=self.mode, image_type = "eye")

        if self.channel_first:
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)


        mask_path = os.path.join(
            self.mask_folder,
            "/".join(image_path.split("/")[-2:]).split(".")[0] + ".png",
        )
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask.astype(float)
        mask = self.mask_augmentations(mask)
        
        

        target_series = self.dataframe.iloc[item]
        landmarks = rotation_landmark(target_series.copy(), mode = self.mode)
        landmarks = torch.tensor(landmarks, dtype = torch.float)
        
        target = get_one_hot_target_for_training(target_series, sub_adjustment=0.6)
        return image,right_eye_image , left_eye_image,landmarks, mask, target