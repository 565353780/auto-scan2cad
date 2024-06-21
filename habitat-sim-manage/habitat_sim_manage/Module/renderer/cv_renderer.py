#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from PIL import Image
from copy import deepcopy
from habitat_sim.utils.common import d3_40_colors_rgb


class CVRenderer(object):
    def __init__(self, window_name="CVRenderer"):
        self.window_name = window_name
        return

    def reset(self):
        return True

    def init(self, window_name="CVRenderer"):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        return True

    def getRGBImage(self, observations):
        if observations is None:
            return None

        if "color_sensor" not in observations.keys():
            return None
        rgb_obs = deepcopy(observations["color_sensor"])

        rgb_image = rgb_obs[..., :3][..., ::-1] / 255.0
        return rgb_image

    def getDepthImage(self, observations):
        if observations is None:
            return None

        if "depth_sensor" not in observations.keys():
            return None
        depth_obs = deepcopy(observations["depth_sensor"])

        depth_clip = np.clip(depth_obs, 0.0, 10.0)
        depth_image = depth_clip / 10.0
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)
        return depth_image

    def getSemanticImage(self, observations):
        if observations is None:
            return None

        if "semantic_sensor" not in observations.keys():
            return None
        semantic_obs = deepcopy(observations["semantic_sensor"])

        semantic_image = Image.new(
            "P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_image.putpalette(d3_40_colors_rgb.flatten())
        semantic_image.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_image = semantic_image.convert("RGBA")

        semantic_image = np.array(semantic_image)
        semantic_image = semantic_image[..., 0:3][..., ::-1] / 255.0
        return semantic_image

    def getImage(self, observations):
        rgb_image = self.getRGBImage(observations)
        depth_image = self.getDepthImage(observations)
        semantic_image = self.getSemanticImage(observations)

        image_list = []
        if rgb_image is not None:
            image_list.append(rgb_image)
        if depth_image is not None:
            image_list.append(depth_image)
        if semantic_image is not None:
            image_list.append(semantic_image)

        if len(image_list) == 0:
            return None

        image = np.hstack(image_list)
        return image

    def renderFrame(self, observations, return_image=False):
        image = self.getImage(observations)
        if image is None:
            print("[ERROR][CVRenderer::renderFrame]")
            print("\t image is None!")
            if return_image:
                return self.getRGBImage(observations)
            return False

        cv2.imshow(self.window_name, image)

        if return_image:
            return self.getRGBImage(observations)

        return True

    def close(self):
        cv2.destroyAllWindows()
        return True

    def waitKey(self, wait_key):
        cv2.waitKey(wait_key)
        return True


def demo():
    cv_renderer = CVRenderer()

    cv_renderer.init()
    cv_renderer.waitKey(1)
    cv_renderer.close()
    return True
