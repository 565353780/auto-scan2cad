#!/usr/bin/env python
# -*- coding: utf-8 -*-

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor

from detectron2_detect.Config.configs import CONFIGS


class Detector(object):

    def __init__(self,
                 model_file_path=None,
                 config_name="X_101_32x8d_FPN_3x",
                 confidence_threshold=None):
        self.model_file_path = model_file_path
        self.config_name = config_name
        self.confidence_threshold = confidence_threshold

        self.cfg = None
        self.predictor = None

        if None not in [model_file_path, config_name]:
            self.loadModel(model_file_path, config_name)
        return

    def loadModel(self, model_file_path, config_name):
        self.model_file_path = model_file_path
        self.config_name = config_name

        assert config_name in CONFIGS.keys()

        config_file = CONFIGS[config_name]

        print("start loading model...", end="")
        self.cfg = get_cfg()
        self.cfg.merge_from_file(config_file)
        self.cfg.MODEL.WEIGHTS = model_file_path

        if self.confidence_threshold is not None:
            self.cfg.MODEL.RETINANET.SCORE_THRESH_TEST = self.confidence_threshold
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
            self.cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = self.confidence_threshold

        self.cfg.freeze()
        self.predictor = DefaultPredictor(self.cfg)
        print("SUCCESS!")
        return True

    def detect_image(self, image):
        assert self.predictor is not None

        result = self.predictor(image)

        pred_boxes = result["instances"].pred_boxes.tensor.cpu().numpy()
        scores = result["instances"].scores.cpu().numpy()
        pred_classes = result["instances"].pred_classes.cpu().numpy()
        pred_masks = result["instances"].pred_masks.cpu().numpy()

        result_dict = {}
        result_dict["pred_boxes"] = pred_boxes
        result_dict["scores"] = scores
        result_dict["pred_classes"] = pred_classes
        result_dict["pred_masks"] = pred_masks
        return result_dict
