import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.models import yolo
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, OBBModel, PoseModel, SegmentationModel
from .predictor import DetectionSoftPredictor

class YOLOSoft(YOLO):
    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "classify": {
                "model": ClassificationModel,
                "trainer": yolo.classify.ClassificationTrainer,
                "validator": yolo.classify.ClassificationValidator,
                "predictor": yolo.classify.ClassificationPredictor,
            },
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": DetectionSoftPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": yolo.segment.SegmentationTrainer,
                "validator": yolo.segment.SegmentationValidator,
                "predictor": yolo.segment.SegmentationPredictor,
            },
            "pose": {
                "model": PoseModel,
                "trainer": yolo.pose.PoseTrainer,
                "validator": yolo.pose.PoseValidator,
                "predictor": yolo.pose.PosePredictor,
            },
            "obb": {
                "model": OBBModel,
                "trainer": yolo.obb.OBBTrainer,
                "validator": yolo.obb.OBBValidator,
                "predictor": yolo.obb.OBBPredictor,
            },
        }


class MLP(torch.nn.Module):

    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config
        self.class_embed = nn.Embedding(config.MODEL.VOCAB_DIM, config.MODEL.EMBEDDING_DIM)
        self.MLP = nn.Sequential()
        for idx, dim in enumerate(self.config.MODEL.MLP_CONFIG):
            if idx == 0:
                self.MLP.append(nn.Linear(config.MODEL.EMBEDDING_DIM*2+8, dim))
                self.MLP.append(nn.ReLU())
            else:
                self.MLP.append(nn.Linear(config.MODEL.MLP_CONFIG[idx-1], dim))
                self.MLP.append(nn.ReLU())
        self.head = nn.Linear(config.MODEL.MLP_CONFIG[-1], 1)

    def forward(self, batch, apply_sigmoid=False):
        source_cls_embed = self.class_embed(batch['source_class'])
        target_cls_embed = self.class_embed(batch['target_class'])
        source_embed = torch.cat([batch['source_bbox'], source_cls_embed], dim=-1)
        target_embed = torch.cat([batch['target_bbox'], target_cls_embed], dim=-1)

        combined_embed = torch.cat([source_embed, target_embed], dim=-1)
        combined_embed = self.MLP(combined_embed)
        output = self.head(combined_embed)
        if apply_sigmoid:
            return torch.sigmoid(output)
        else:
            return output


class MLPwithSoftClass(torch.nn.Module):

    def __init__(self, config):
        super(MLPwithSoftClass, self).__init__()
        self.config = config
        self.class_mlp = nn.Linear(config.MODEL.VOCAB_DIM, config.MODEL.EMBEDDING_DIM)
        self.MLP = nn.Sequential()
        for idx, dim in enumerate(self.config.MODEL.MLP_CONFIG):
            if idx == 0:
                self.MLP.append(nn.Linear(config.MODEL.EMBEDDING_DIM*2+8, dim))
                self.MLP.append(nn.ReLU())
            else:
                self.MLP.append(nn.Linear(config.MODEL.MLP_CONFIG[idx-1], dim))
                self.MLP.append(nn.ReLU())
        self.head = nn.Linear(config.MODEL.MLP_CONFIG[-1], 1)

    def forward(self, batch, apply_sigmoid=False):
        # print(batch['source_bbox'])
        source_cls_embed = self.class_mlp(batch['source_class'])
        target_cls_embed = self.class_mlp(batch['target_class'])
        source_embed = torch.cat([batch['source_bbox'], source_cls_embed], dim=-1)
        target_embed = torch.cat([batch['target_bbox'], target_cls_embed], dim=-1)

        combined_embed = torch.cat([source_embed, target_embed], dim=-1)
        combined_embed = self.MLP(combined_embed)
        output = self.head(combined_embed)
        if apply_sigmoid:
            return torch.sigmoid(output)
        else:
            return output


class MLPwithSoftClassExtraMLP(torch.nn.Module):

    def __init__(self, config):
        super(MLPwithSoftClassExtraMLP, self).__init__()
        self.config = config
        self.class_intermediate = nn.Linear(config.MODEL.VOCAB_DIM, config.MODEL.VOCAB_DIM)
        self.class_embed = nn.Linear(config.MODEL.VOCAB_DIM, config.MODEL.EMBEDDING_DIM // 2)
        self.bbox_mapping = nn.Linear(4, 4)
        self.bbox_embed = nn.Linear(4, config.MODEL.EMBEDDING_DIM // 2)
        self.MLP = nn.Sequential()
        for idx, dim in enumerate(self.config.MODEL.MLP_CONFIG):
            if idx == 0:
                self.MLP.append(nn.Linear(config.MODEL.EMBEDDING_DIM*2, dim))
                self.MLP.append(nn.ReLU())
            else:
                self.MLP.append(nn.Linear(config.MODEL.MLP_CONFIG[idx-1], dim))
                self.MLP.append(nn.ReLU())
        self.head = nn.Linear(config.MODEL.MLP_CONFIG[-1], 1)

    def forward(self, batch, apply_sigmoid=False):
        source_cls_embed = self.class_intermediate(batch['source_class'])
        source_cls_embed = F.relu(self.class_embed(source_cls_embed))
        target_cls_embed = self.class_intermediate(batch['target_class'])
        target_cls_embed = F.relu(self.class_embed(target_cls_embed))
        source_bbox_embed = self.bbox_embed(F.relu(self.bbox_mapping(batch['source_bbox'])))
        target_bbox_embed = self.bbox_embed(F.relu(self.bbox_mapping(batch['target_bbox'])))
        source_embed = torch.cat([source_bbox_embed, source_cls_embed], dim=-1)
        target_embed = torch.cat([target_bbox_embed, target_cls_embed], dim=-1)

        combined_embed = torch.cat([source_embed, target_embed], dim=-1)
        combined_embed = self.MLP(combined_embed)
        output = self.head(combined_embed)
        if apply_sigmoid:
            return torch.sigmoid(output)
        else:
            return output
