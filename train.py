#%%

import os
import sys
from wtfml.data_loaders.pl_data_module.data_module import plDataModule
from wtfml.utils.utils import basefolder
sys.path.append("/".join(os.getcwd().split("/")[:-1]))
sys.path.append(os.getcwd())

import pandas as pd
import torch
import pytorch_lightning as pl
from glob import glob
import datetime
from efficientnet_pytorch import EfficientNet

from wtfml.cross_validation.fold_generator import FoldGenerator
from wtfml.data_loaders.image.face_88_mapping import FaceMaskDatasetWithEyeAndLandmarks
from wtfml.engine.image.embedder.model import classification_model
from wtfml.engine.image.segmentation.model import FacetypeLocationGenerator
from wtfml.engine.pl_engine.PL_face_mapper import SegmentationPLEngineWithEyeAndLandmark
from wtfml.utils.utils import *
import shutil


#%%



# csv_path = "/content/88_face_classification/pretreated_csv_for_AI_2021_04_19.csv"
# base_iamge_folder = "/content/FaceClassificationAi"
# base_mask_folder = "/content/FaceClassificationAi/mask_png"
# num_splits = 10
famous_wight = 1
csv_path = "/Users/yongtae/Documents/cinderella-planning/dataset/face/csv/train_data/pretreated_csv_2021-08-07_15_49_with_landmark.csv"
base_image_folder = "/Users/yongtae/Documents/cinderella-planning/dataset/face/image/face_image/cropped/non_padding_margin_0"
base_mask_folder = "/Users/yongtae/Documents/cinderella-planning/dataset/face/image/mask/mask_png_gauss_1.3_h"
eye_folder = "/Users/yongtae/Documents/cinderella-planning/dataset/face/image/eye"

# save_model_path = "/content/drive/MyDrive/88_face_classification/result/MixMatch/2021_05_24/facenet_pretrained_epoch-150_new_mixsigmoid/lambda75/0/model_best_acc.pth.tar"
# save_folder = "/content/drive/MyDrive/88_face_classification/result/2021_05_29/prediction_image_stringness_{}".format(famous_wight)
save_folder = "/content/result"
loss_method = 'L1'
model_name = 'ir50'
save_model_path = "/Users/yongtae/Documents/cinderella-planning/dataset/face/free_dataset/backbone_ir50_asia.pth" if model_name == "ir50" else None
sigma = 1.3
is_padding = False
num_splits = 5
d_today = datetime.date.today()

# model_name = "facenet"
# save_model_path = "/content/drive/MyDrive/88_face_classification/backbone_ir50_asia.pth" if model_name == "ir50" else None

input_df = pd.read_csv(csv_path)
input_df = input_df.dropna(subset=["original_image_path"])
input_df = input_df[input_df.have_label == True]
input_df.reset_index(drop=True, inplace=True)
normal_df = input_df[input_df.is_famous == False]
normal_df.reset_index(drop=True, inplace=True)
fold_generator = FoldGenerator(
    targets = normal_df,
    task="multilabel_classification",
    num_splits=num_splits,
    shuffle=False,
)

#%%



fold = 0
# for fold in range(num_splits):
(
    _,
    _,
    input_train,
    input_valid,
    target_train,
    target_valid,
) = fold_generator.get_fold(data=normal_df, fold=fold)
famous_df = input_df[input_df.is_famous == True]
sample_number_list = [input_train.shape[0], famous_df.shape[0]]
input_train = pd.concat([input_train, famous_df]).reset_index()
input_valid = input_valid.reset_index()
weights = [1] * sample_number_list[0] + [
    1 * famous_wight
] * sample_number_list[1]

image_path_train = [
    os.path.join(base_image_folder, image_path)
    for image_path in list(input_train.cropped_image_place.values)
]
image_path_valid = [
    os.path.join(base_image_folder, image_path)
    for image_path in list(input_valid.cropped_image_place.values)
]
train_dataset = FaceMaskDatasetWithEyeAndLandmarks(
    image_path_train, mask_folder=base_mask_folder, mode="train", dataframe=input_train, model_name=model_name,eye_folder = eye_folder
)

val_dataset = FaceMaskDatasetWithEyeAndLandmarks(
    image_path_valid, mask_folder=base_mask_folder, mode="valid", dataframe=input_valid, model_name=model_name, eye_folder = eye_folder
)

data_module = plDataModule(train_dataset=train_dataset, val_dataset=val_dataset, train_batch_size=16)
#%%
face_calssification_model = classification_model(
    transfer_learning=False,
    classify=False,
    save_model_path=save_model_path,
    model_name=model_name,
)  # TODO:ここのclasifyは考える必要あり。
location_decoder = FacetypeLocationGenerator(mid_layer_num=64, input_size=535 + 1404)
eye_embedder = EfficientNet.from_pretrained('efficientnet-b0', num_classes=23)

pl_model = SegmentationPLEngineWithEyeAndLandmark(
    face_encoder=face_calssification_model,
    location_decoder=location_decoder,
    image_loss_method = loss_method,
    eye_encoder = eye_embedder,
    lr = 0.005,
)

callbacks_path = os.path.join(save_folder,"{}/{}/{}/sigma_{}/margin_{}/{}".format(model_name, loss_method,'normalize' if is_normalization else 'non_normalization', sigma, margin,fold))
if not os.path.exists(callbacks_path):
    os.makedirs(callbacks_path)
input_valid.to_csv(os.path.join(callbacks_path, "valid_image_path.csv"))
callbacks_loss = pl.callbacks.ModelCheckpoint(
    dirpath = callbacks_path,
    filename = "{epoch}-{valid_loss:.4f}-{acc:.4f}",
    monitor = "valid_loss",
    mode = "min",
    save_top_k = 1,
    save_last = True,
    )
callbacks_recall_just = pl.callbacks.ModelCheckpoint(
    dirpath = callbacks_path,
    filename = "{epoch}-{recall_list_just:.4f}-{acc_just:.4f}",
    monitor = "recall_list_just",
    mode = "max",
    save_top_k = 1,
    )
callbacks_recall_wide = pl.callbacks.ModelCheckpoint(
    dirpath = callbacks_path,
    filename = "{epoch}-{recall_list_wide:.4f}-{acc_wide:.4f}",
    monitor = "recall_list_wide",
    mode = "max",
    save_top_k = 1,
    )
callbacks_acc= pl.callbacks.ModelCheckpoint(
    dirpath = callbacks_path,
    filename = "{epoch}-{acc_just:.4f}",
    monitor = "acc_just",
    mode = "max",
    save_top_k = 1,
    )

trainer = pl.Trainer(gpus=1, max_epochs = 110, callbacks = [callbacks_loss, callbacks_recall_just, callbacks_recall_wide, callbacks_acc])
trainer.fit(pl_model, datamodule=data_module)
# padding_name = 'padding' if is_padding else 'non_padding'
# shutil.copytree(basefolder(callbacks_path),"/content/drive/MyDrive/88_face_classification/result/{}/WithEye/{}/{}/margin_{}/{}/sigma_{}".format(d_today,model_name,padding_name,margin,'normalize' if is_normalization else 'non_normalization',sigma))