#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2

from detectron2_detect.Module.detector import Detector


def demo():
    #  model_path = "/home/chli/chLi/detectron2/model_final_f96b26.pkl"
    #  config_name = "R_101_FPN_3x"
    model_file_path = "/home/chli/chLi/detectron2/model_final_2d9806.pkl"
    config_name = "X_101_32x8d_FPN_3x"

    detector = Detector(model_file_path, config_name)

    image_path = "/home/chli/chLi/detectron2/test1.jpg"
    image = cv2.imread(image_path)

    result_dict = detector.detect_image(image)
    print(result_dict)

    for box in result_dict["pred_boxes"].astype(int):
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255),
                      3)
    cv2.imshow("result", image)
    cv2.waitKey(5000)
    return True


def demo_video():
    #  model_path = "/home/chli/chLi/detectron2/model_final_f96b26.pkl"
    #  config_name = "R_101_FPN_3x"
    model_file_path = "/home/chli/chLi/detectron2/model_final_2d9806.pkl"
    config_name = "X_101_32x8d_FPN_3x"

    detector = Detector(model_file_path, config_name)

    video_path = "/home/chli/videos/robot-1.mp4"

    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result_dict = detector.detect_image(frame)
        print(result_dict)

        for box in result_dict["pred_boxes"].astype(int):
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]),
                          (0, 0, 255), 3)
        cv2.imshow("result", frame)
        cv2.waitKey(1)
    return True
