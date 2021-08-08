import math
import os

import numpy as np
import torch
import torch.nn as nn
from wtfml.utils import config
import torch.nn.functional as F

from torchvision import transforms
import cv2
from matplotlib import pyplot as plt




def get_mean_prediction_from_mask(pred, target=None, metrix = "mean", radius_intention = 1):
    image_shape = pred.shape[2]
    pred_answer = []
    target_answer = []
    for class_name, circle_position in config.circle_position_df.iterrows():
        x = int(circle_position["x"] / 1000 * image_shape)
        y = int(circle_position["y"] / 1000 * image_shape)
        r = int(circle_position["r"] / 1000 * image_shape) * radius_intention
        position_map = np.zeros((image_shape, image_shape), dtype=float)
        position_map = cv2.circle(position_map, (x, y), int(r), 1, -1)
        if metrix == "mean":
            pred_answer.append(torch.mean(pred[:, 0, position_map == 1], axis=1))
        elif metrix == "max":
            pred_answer.append(torch.max(pred[:, 0, position_map == 1], axis=1)[0])
            # print(torch.max(pred[:, 0, position_map == 1], axis=1))
        
        if target is not None:
            if metrix == "mean":
                target_answer.append(torch.mean(target[:, 0, position_map == 1], axis=1))
            elif metrix == "max":
                target_answer.append(torch.max(target[:, 0, position_map == 1], axis=1)[0])
            target_answer = torch.stack([x for x in target_answer])
            target_answer = torch.transpose(target_answer, 0, 1)
    pred_answer = torch.stack([x for x in pred_answer])
    pred_answer = torch.transpose(pred_answer, 0, 1)
    return pred_answer, target_answer



def get_recall_from_mask(mask, targets, threshold=0.75, radius_intention = 1, metrix = "mean"):
    prediction_from_mask, _ = get_mean_prediction_from_mask(mask, radius_intention = radius_intention, metrix=metrix)
    result = []
    for num, target in enumerate(targets):
        prediction = []
        for class_num in range(len(target)):
            if target[class_num] != 0:
                if prediction_from_mask[num][class_num] > threshold * target[class_num]:
                    prediction.append(1)
                else:
                    prediction.append(0)

        result.append(sum(prediction) / len(prediction))

    return result


def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


def de_preprocess(tensor):

    return tensor * 0.5 + 0.5


hflip = transforms.Compose([
            de_preprocess,
            transforms.ToPILImage(),
            transforms.functional.hflip,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

def hflip_batch(imgs_tensor):
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)

    return hfliped_imgs