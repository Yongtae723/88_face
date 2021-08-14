"""
__author__: Abhishek Thakur
"""

import datetime

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules import BCEWithLogitsLoss
from tqdm import tqdm

from wtfml.engine.image.embedder.utils import (
    get_mean_prediction_from_mask,
    get_recall_from_mask,
    hflip_batch,
    l2_norm,
)



class SegmentationPLEngine(pl.LightningModule):
    def __init__(
        self,
        face_encoder,
        location_decoder,
        is_train_encoder=False,
        lr=0.001,
        image_loss_method="MSE",
        class_loss_fn=nn.BCEWithLogitsLoss(),
        lamb=1,
        F_score_metrix=smp.utils.metrics.Fscore(threshold=0.5),
        normalize = False,
    ):
        super(SegmentationPLEngine, self).__init__()
        self.face_encoder = face_encoder
        if not is_train_encoder:
            for name, module in self.face_encoder._modules.items():
                module.requires_grad = False  # 全ての層を凍結
        self.location_decoder = location_decoder
        self.scaler = None
        if image_loss_method == "MSE":
            self.image_loss_fn = nn.MSELoss()
        elif image_loss_method == "BCE":
            self.image_loss_fn = nn.BCEWithLogitsLoss()
        elif image_loss_method == "L1":
            self.image_loss_fn = nn.L1Loss()
        else:
            raise ValueError("image_loss_method should be MSE or L1 or BCE")

        self.class_loss_fn = class_loss_fn
        self.F_score_metrix = F_score_metrix
        self.lamb = lamb
        self.lr = lr
        self.normalize = normalize

    def forward(self, x):
        self.face_encoder.eval()
        with torch.no_grad():
            fliped = hflip_batch(x)
            emb_batch = self.face_encoder(x) + self.face_encoder(fliped)
            if self.normalize:
                representations = l2_norm(emb_batch)
            if not self.normalize:
                representations = emb_batch /2


        x = self.location_decoder(representations)
        return x

    def training_step(self, batch, batch_idx):
        # REQUIRED
        image, mask, target = batch
        # target = main_target + sub_target * self.sub_adjustment
        pred_batch_train = self.forward(image)
        train_loss = self.image_loss_fn(pred_batch_train, mask)
        F_score_image = self.F_score_metrix(pred_batch_train, mask)

        # pred_class, _ = get_mean_prediction_from_mask(pred_batch_train)
        # class_loss = self.class_loss_fn(pred_class, target)
        self.log(
            "train_batch_F_score_image",
            F_score_image,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "train_batch_loss",
            train_loss,
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            logger=True,
        )
        return {"loss": train_loss}

    def validation_step(self, batch, batch_idx):
        image, mask, target = batch
        # target = main_target + sub_target * self.sub_adjustment
        pred_batch_valid = self.forward(image)
        recall_list_just_list =  get_recall_from_mask(pred_batch_valid, target, threshold=0.5, radius_intention=1, metrix="max")
        recall_list_wide_list =  get_recall_from_mask(pred_batch_valid, target, threshold=0.5, radius_intention=2, metrix="max")

        if self.current_epoch < 35:
            recall_list_just = 0
            loss = np.inf
            recall_list_wide = 0
            acc_just = 0
            acc_wide = 0
        else:
            recall_list_just = sum(recall_list_just_list) / len(recall_list_just_list)
            loss = self.image_loss_fn(pred_batch_valid, mask) 
            recall_list_wide = sum(recall_list_wide_list) / len(recall_list_wide_list)
            acc_just = len(np.where(np.array(recall_list_just_list) > 0)[0]) / len(recall_list_just_list)
            acc_wide = len(np.where(np.array(recall_list_wide_list) > 0)[0]) / len(recall_list_wide_list)
        self.log(
            "recall_list_just",
            recall_list_just,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "valid_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "recall_list_wide",
            recall_list_wide,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "acc_wide",
            acc_wide,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "acc_just",
            acc_just,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False,
        )
        return {
            "val_loss": loss,
            # "acc": acc,
        }

    def configure_optimizers(self):
        # REQUIRED
        opt = optim.Adam(
            self.location_decoder.parameters(),
            lr=self.lr,
        )
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max= 20)
        return [opt], [sch]






class SegmentationPLEngineWithEye(pl.LightningModule):
    def __init__(
        self,
        face_encoder,
        location_decoder,
        eye_encoder,
        is_train_encoder=False,

        lr=0.001,
        image_loss_method="MSE",
        class_loss_fn=nn.BCEWithLogitsLoss(),
        lamb=1,
        F_score_metrix=smp.utils.metrics.Fscore(threshold=0.5),
        normalize = False,
    ):
        super(SegmentationPLEngineWithEye, self).__init__()
        self.face_encoder = face_encoder
        if not is_train_encoder:
            for name, module in self.face_encoder._modules.items():
                module.requires_grad = False  # 全ての層を凍結
        self.location_decoder = location_decoder
        self.scaler = None
        if image_loss_method == "MSE":
            self.image_loss_fn = nn.MSELoss()
        elif image_loss_method == "BCE":
            self.image_loss_fn = nn.BCEWithLogitsLoss()
        elif image_loss_method == "L1":
            self.image_loss_fn = nn.L1Loss()
        else:
            raise ValueError("image_loss_method should be MSE or L1 or BCE")
        self.eye_encoder = eye_encoder
        self.class_loss_fn = class_loss_fn
        self.F_score_metrix = F_score_metrix
        self.lamb = lamb
        self.lr = lr
        self.normalize = normalize

    def forward(self, face_image : torch.Tensor, right_eye_image : torch.Tensor, left_eye_image : torch.Tensor):
        self.face_encoder.eval()
        with torch.no_grad():
            fliped = hflip_batch(face_image,)
            emb_batch = self.face_encoder(face_image,) + self.face_encoder(fliped)
            if self.normalize:
                representations = l2_norm(emb_batch)
            if not self.normalize:
                representations = emb_batch /2
        right_vector = self.eye_encoder(right_eye_image)
        left_vector = self.eye_encoder(left_eye_image)

        eye_vector = (right_vector + left_vector)/2
        representations = torch.cat([representations, eye_vector], dim = 1)

        x = self.location_decoder(representations)
        return x

    def training_step(self, batch, batch_idx):
        # REQUIRED
        image,right_eye_image, left_eye_image, mask, target = batch
        pred_batch_train = self.forward(image, right_eye_image, left_eye_image)
        # target = main_target + sub_target * self.sub_adjustment
        
        train_loss = self.image_loss_fn(pred_batch_train, mask)
        F_score_image = self.F_score_metrix(pred_batch_train, mask)


        self.log(
            "train_batch_F_score_image",
            F_score_image,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "train_batch_loss",
            train_loss,
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            logger=True,
        )
        return {"loss": train_loss}

    def validation_step(self, batch, batch_idx):
        image,right_eye_image, left_eye_image, mask, target = batch
        pred_batch_valid = self.forward(image, right_eye_image, left_eye_image)

        recall_list_just_list =  get_recall_from_mask(pred_batch_valid, target, threshold=0.5, radius_intention=1, metrix="max")
        recall_list_wide_list =  get_recall_from_mask(pred_batch_valid, target, threshold=0.5, radius_intention=2, metrix="max")

        if self.current_epoch < 35:
            recall_list_just = 0
            loss = np.inf
            recall_list_wide = 0
            acc_just = 0
            acc_wide = 0
        else:
            recall_list_just = sum(recall_list_just_list) / len(recall_list_just_list)
            loss = self.image_loss_fn(pred_batch_valid, mask) 
            recall_list_wide = sum(recall_list_wide_list) / len(recall_list_wide_list)
            acc_just = len(np.where(np.array(recall_list_just_list) > 0)[0]) / len(recall_list_just_list)
            acc_wide = len(np.where(np.array(recall_list_wide_list) > 0)[0]) / len(recall_list_wide_list)
        self.log(
            "recall_list_just",
            recall_list_just,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "valid_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "recall_list_wide",
            recall_list_wide,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "acc_wide",
            acc_wide,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "acc_just",
            acc_just,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False,
        )
        return {
            "val_loss": loss,
            # "acc": acc,
        }

    def configure_optimizers(self):
        # REQUIRED
        # opt_generator = optim.Adam(
        #     self.location_decoder.parameters(),
        #     lr=self.lr,
        # )

        # opt_eye_image = optim.Adam(
        #     self.eye_encoder.parameters(),
        #     lr=self.lr,
        # )
        opt = optim.Adam(
            list(self.location_decoder.parameters()) + list(self.eye_encoder.parameters()),
            lr=self.lr,
        )

        # sch_generator = optim.lr_scheduler.CosineAnnealingLR(opt_generator, T_max= 20)
        # sch_eye_image = optim.lr_scheduler.CosineAnnealingLR(opt_eye_image, T_max= 20)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max= 20)
        return [opt], [sch]









class SegmentationPLEngineWithEyeAndLandmark(pl.LightningModule):
    def __init__(
        self,
        face_encoder,
        location_decoder,
        eye_encoder,
        is_train_encoder=False,

        lr=0.001,
        image_loss_method="MSE",
        class_loss_fn=nn.BCEWithLogitsLoss(),
        lamb=1,
        F_score_metrix=smp.utils.metrics.Fscore(threshold=0.5),
        normalize = True,
    ):
        super(SegmentationPLEngineWithEyeAndLandmark, self).__init__()
        self.face_encoder = face_encoder
        if not is_train_encoder:
            for name, module in self.face_encoder._modules.items():
                module.requires_grad = False  # 全ての層を凍結
        self.location_decoder = location_decoder
        self.scaler = None
        if image_loss_method == "MSE":
            self.image_loss_fn = nn.MSELoss()
        elif image_loss_method == "BCE":
            self.image_loss_fn = nn.BCEWithLogitsLoss()
        elif image_loss_method == "L1":
            self.image_loss_fn = nn.L1Loss()
        else:
            raise ValueError("image_loss_method should be MSE or L1 or BCE")
        self.eye_encoder = eye_encoder
        self.class_loss_fn = class_loss_fn
        self.F_score_metrix = F_score_metrix
        self.lamb = lamb
        self.lr = lr
        self.normalize = normalize

    def forward(self, face_image : torch.Tensor, right_eye_image : torch.Tensor, left_eye_image : torch.Tensor, landmark_point:torch.Tensor):
        self.face_encoder.eval()
        with torch.no_grad():
            fliped = hflip_batch(face_image,)
            emb_batch = self.face_encoder(face_image,) + self.face_encoder(fliped)
            if self.normalize:
                representations = l2_norm(emb_batch)
            if not self.normalize:
                representations = emb_batch /2
        right_vector = self.eye_encoder(right_eye_image)
        left_vector = self.eye_encoder(left_eye_image)

        eye_vector = (right_vector + left_vector)/2
        representations = torch.cat([representations, eye_vector, landmark_point], dim = 1)

        x = self.location_decoder(representations)
        return x

    def training_step(self, batch, batch_idx):
        # REQUIRED
        image,right_eye_image, left_eye_image, landmark_point,mask, target = batch
        pred_batch_train = self.forward(image, right_eye_image, left_eye_image, landmark_point)
        # target = main_target + sub_target * self.sub_adjustment
        
        train_loss = self.image_loss_fn(pred_batch_train, mask)
        F_score_image = self.F_score_metrix(pred_batch_train, mask)


        self.log(
            "train_batch_F_score_image",
            F_score_image,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "train_batch_loss",
            train_loss,
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            logger=True,
        )
        return {"loss": train_loss}

    def validation_step(self, batch, batch_idx):
        image,right_eye_image, left_eye_image,landmark_point, mask, target = batch
        pred_batch_valid = self.forward(image, right_eye_image, left_eye_image, landmark_point)

        recall_list_just_list =  get_recall_from_mask(pred_batch_valid, target, threshold=0.5, radius_intention=1, metrix="max")
        recall_list_wide_list =  get_recall_from_mask(pred_batch_valid, target, threshold=0.5, radius_intention=2, metrix="max")

        if self.current_epoch < 35:
            recall_list_just = 0
            loss = np.inf
            recall_list_wide = 0
            acc_just = 0
            acc_wide = 0
        else:
            recall_list_just = sum(recall_list_just_list) / len(recall_list_just_list)
            loss = self.image_loss_fn(pred_batch_valid, mask) 
            recall_list_wide = sum(recall_list_wide_list) / len(recall_list_wide_list)
            acc_just = len(np.where(np.array(recall_list_just_list) > 0)[0]) / len(recall_list_just_list)
            acc_wide = len(np.where(np.array(recall_list_wide_list) > 0)[0]) / len(recall_list_wide_list)
        self.log(
            "recall_list_just",
            recall_list_just,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "valid_loss",
            loss,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "recall_list_wide",
            recall_list_wide,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "acc_wide",
            acc_wide,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "acc_just",
            acc_just,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False,
        )
        return {
            "val_loss": loss,
            # "acc": acc,
        }

    def configure_optimizers(self):
        # REQUIRED
        # opt_generator = optim.Adam(
        #     self.location_decoder.parameters(),
        #     lr=self.lr,
        # )

        # opt_eye_image = optim.Adam(
        #     self.eye_encoder.parameters(),
        #     lr=self.lr,
        # )
        opt = optim.Adam(
            list(self.location_decoder.parameters()) + list(self.eye_encoder.parameters()),
            lr=self.lr,
        )

        # sch_generator = optim.lr_scheduler.CosineAnnealingLR(opt_generator, T_max= 20)
        # sch_eye_image = optim.lr_scheduler.CosineAnnealingLR(opt_eye_image, T_max= 20)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max= 20)
        return [opt], [sch]