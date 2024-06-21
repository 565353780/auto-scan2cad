#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import habitat_sim

from habitat_sim_manage.Config.config import SIM_SETTING

from habitat_sim_manage.Method.actions import register_actions
register_actions()

def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]

    sensor_specs = []

    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [settings["height"], settings["width"]]
    color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(color_sensor_spec)

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(depth_sensor_spec)

    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(semantic_sensor_spec)

    move_dist = settings["move_dist"]
    rotate_angle = settings["rotate_angle"]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=move_dist)),
        "move_left": habitat_sim.agent.ActionSpec(
            "move_left", habitat_sim.agent.ActuationSpec(amount=move_dist)),
        "move_right": habitat_sim.agent.ActionSpec(
            "move_right", habitat_sim.agent.ActuationSpec(amount=move_dist)),
        "move_backward": habitat_sim.agent.ActionSpec(
            "move_backward", habitat_sim.agent.ActuationSpec(amount=move_dist)),
        "move_up": habitat_sim.agent.ActionSpec(
            "my_move_up", habitat_sim.agent.ActuationSpec(amount=move_dist)),
        "move_down": habitat_sim.agent.ActionSpec(
            "my_move_down", habitat_sim.agent.ActuationSpec(amount=move_dist)),
        "turn_left": habitat_sim.agent.ActionSpec(
            "my_turn_left", habitat_sim.agent.ActuationSpec(amount=rotate_angle)),
        "turn_right": habitat_sim.agent.ActionSpec(
            "my_turn_right", habitat_sim.agent.ActuationSpec(amount=rotate_angle)),
        "turn_up": habitat_sim.agent.ActionSpec(
            "turn_up", habitat_sim.agent.ActuationSpec(amount=rotate_angle)),
        "turn_down": habitat_sim.agent.ActionSpec(
            "turn_down", habitat_sim.agent.ActuationSpec(amount=rotate_angle)),
        "head_left": habitat_sim.agent.ActionSpec(
            "my_head_left", habitat_sim.agent.ActuationSpec(amount=rotate_angle)),
        "head_right": habitat_sim.agent.ActionSpec(
            "my_head_right", habitat_sim.agent.ActuationSpec(amount=rotate_angle)),
    }
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def makeGLBConfig(glb_file_path):
    if not os.path.exists(glb_file_path):
        print("[ERROR][configs::makeGLBConfig]")
        print("\t glb_file not exist!")
        return None

    sim_settings = SIM_SETTING
    sim_settings["scene"] = glb_file_path
    return make_cfg(sim_settings)
