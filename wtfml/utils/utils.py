import os
from os import listdir
from os.path import isdir, isfile, join
from typing import List
from wtfml.utils import config
import numpy as np
import random


def basename(path: str) -> str:
    """
    get '17asdfasdf2d_0_0.jpg' from 'train_folder/train/o/17asdfasdf2d_0_0.jpg

    Args:
        path (str): [description]

    Returns:
        str: [description]
    """
    return path.split("/")[-1]


# get 'train_folder/train/o' from 'train_folder/train/o/17asdfasdf2d_0_0.jpg'
def basefolder(path: str) -> str:
    """
    get 'train_folder/train/o' from 'train_folder/train/o/17asdfasdf2d_0_0.jpg'

    Args:
        path (str): [description]

    Returns:
        str: [description]
    """
    return "/".join(path.split("/")[:-1])


def get_image_paths(folder: str) -> List[str]:
    """
    get full image paths

    Args:
        folder (str): [description]

    Returns:
        List[str]: [description]
    """
    image_paths = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
    if join(folder, ".DS_Store") in image_paths:
        image_paths.remove(join(folder, ".DS_Store"))
    for path in reversed(image_paths):
        if (
            basename(path)[-4:].lower() != ".jpg"
            and basename(path)[-4:].lower() != ".png"
        ):
            image_paths.remove(path)

    image_paths = sorted(image_paths)
    return image_paths


def get_subfolder_paths(folder: str) -> List[str]:
    """
    get subfolders

    Args:
        folder (str): [description]

    Returns:
        List[str]: [description]
    """
    subfolder_paths = [
        join(folder, f)
        for f in listdir(folder)
        if (isdir(join(folder, f)) and f[0] != ".")
    ]
    if join(folder, ".DS_Store") in subfolder_paths:
        subfolder_paths.remove(join(folder, ".DS_Store"))
    subfolder_paths = sorted(subfolder_paths)
    return subfolder_paths


# create an output folder if it does not already exist
def confirm_output_folder(output_folder: str) -> None:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


def get_one_hot_target_for_training(
    target_series, sub_adjustment, task="multi_label_classification"
):
    return_label = config.circle_count_df.copy()
    target = np.zeros(23)
    if task == "multi_label_classification":
        for label in ["main_1", "main_2", "main_3", "sub_1", "sub_2"]:
            if "main" in label:
                intency = 1
            else:
                intency = sub_adjustment
            classes = target_series[label]
            if isinstance(classes, str):
                class_num = int(np.where(return_label.classes == classes)[0])
                target[class_num] += intency
        return target
    else:
        classes_list = []
        weight = []
        for label in ["main_1", "main_2", "main_3", "sub_1", "sub_2"]:
            if "main" in label:
                intency = 1
            else:
                intency = sub_adjustment
            classes = target_series[label]

            if isinstance(classes, str):
                class_num = int(np.where(return_label.classes == classes)[0])
                classes_list.append(class_num)
                weight.append(intency)
        target_num = random.choices(classes_list, weights=weight)[0]
        target[target_num] += 1
        return target