#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from habitat_sim.utils.common import d3_40_colors_rgb

class PltRenderer(object):
    def __init__(self):
        return

    def reset(self):
        return True

    def init(self):
        plt.figure(figsize=(24, 8))
        plt.ion()
        return True

    def renderFrame(self, observations):
        if observations is None:
            return True

        observations_keys = observations.keys()

        rgb_obs = None
        depth_obs = None
        semantic_obs = None

        if "color_sensor" in observations_keys:
            rgb_obs = observations["color_sensor"]
        if "depth_sensor" in observations_keys:
            depth_obs = observations["depth_sensor"]
        if "semantic_sensor" in observations_keys:
            semantic_obs = observations["semantic_sensor"]

        if rgb_obs is None and \
                depth_obs is None and \
                semantic_obs is None:
            return True

        plt.cla()

        arr = []
        titles = []

        if rgb_obs is not None:
            rgb_obs = rgb_obs[..., 0:3]
            arr.append(rgb_obs)
            titles.append('rgb')

        if depth_obs is not None:
            #  depth_img = np.clip(depth_obs, 0, 10) / 10.0
            arr.append(depth_obs)
            titles.append('depth')

        if semantic_obs is not None:

            semantic_img = Image.new("P",
                (semantic_obs.shape[1], semantic_obs.shape[0]))
            semantic_img.putpalette(d3_40_colors_rgb.flatten())
            semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
            semantic_img = semantic_img.convert("RGBA")

            arr.append(semantic_obs)
            titles.append('semantic')

        for i, data in enumerate(arr):
            ax = plt.subplot(1, 3, i+1)
            ax.axis('off')
            ax.set_title(titles[i])
            plt.imshow(data)
        return True

    def close(self):
        plt.ioff()
        return True

    def waitKey(self, pause_time):
        plt.pause(pause_time)
        return True

def demo():
    plt_renderer = PltRenderer()

    plt_renderer.init()
    plt_renderer.waitKey(0.001)
    plt_renderer.close()
    return True

