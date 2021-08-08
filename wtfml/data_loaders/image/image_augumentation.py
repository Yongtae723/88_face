import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchvision import transforms as T
import numpy as np
from PIL import Image

import cv2


image_base_albumentations = A.Compose(
    [
        A.RandomBrightnessContrast(brightness_limit= .3, contrast_limit=.3),
        A.HueSaturationValue(),
        A.RGBShift(),
        A.RandomGamma(),
        # A.GaussianBlur(p=0.3, blur_limit=(3, 3)),
        # A.ElasticTransform(),
        # A.ShiftScaleRotate(),
        A.ToGray(p = .2),
        A.ImageCompression(quality_lower=95, p=0.3),
        # A.CLAHE(tile_grid_size=(4, 4)),
        # A.ISONoise(),
        # A.HorizontalFlip(p=0.5),
        # A.Resize(height=160, width=160, interpolation=cv2.INTER_LANCZOS4, p=1),
        # ToTensorV2(),
        # fixed_image_standardization,
    ]
)

facenet_resize_totensor_transform = T.Compose(
    [
        T.ToPILImage(mode=None),
        T.Resize((160, 160), interpolation=Image.LANCZOS),
        np.float32,
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

ir50_resize_totensor_transform = T.Compose(
    [
        T.ToPILImage(mode=None),
        T.Resize((112, 112), interpolation=Image.LANCZOS),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

eyeimage_resize_totensor_transform = T.Compose(
    [
        T.ToPILImage(mode=None),
        T.Resize((64, 64), interpolation=Image.LANCZOS),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)

def image_augmentations(
    image, image_albumentations=image_base_albumentations, model_name="facenet", mode="train", image_type = "face",
):
    if mode == "train":
        augmented = image_albumentations(image=image)
        image = augmented["image"]
    if image_type == "face": 
        if model_name == "facenet":
            return facenet_resize_totensor_transform(image)
        return ir50_resize_totensor_transform(image)
    elif image_type == "eye":
        return eyeimage_resize_totensor_transform(image)
    
    
def mask_augmentations(image):
    mask_albumentations = A.Compose(
    [
        A.Resize(height=160, width=160, interpolation=cv2.INTER_LANCZOS4, p=1),
        ToTensorV2(),
        
    ]
    )
    image = image / np.max(image)
    image = mask_albumentations(image=image)["image"].float()
    return image