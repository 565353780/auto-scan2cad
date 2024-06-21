#!/usr/bin/env python
# -*- coding: utf-8 -*-

import habitat_sim

from habitat_sim_manage.Config.config import SIM_SETTING

from habitat_sim_manage.Method.infos import print_scene_recur
from habitat_sim_manage.Method.configs import makeGLBConfig


class SimLoader(object):

    def __init__(self):
        self.cfg = None
        self.sim = None
        self.action_names = None
        self.observations = None
        return

    def reset(self):
        self.cfg = None
        if self.sim is not None:
            self.sim.close()
        self.sim = None
        return True

    def loadSettings(self, glb_file_path):
        self.reset()

        self.cfg = makeGLBConfig(glb_file_path)
        self.sim = habitat_sim.Simulator(self.cfg)

        self.initAgent()
        return True

    def initAgent(self):
        self.agent = self.sim.initialize_agent(SIM_SETTING["default_agent"])
        self.action_names = list(
            self.cfg.agents[SIM_SETTING["default_agent"]].action_space.keys())

        self.updateObservations()
        return True

    def updateObservations(self):
        self.observations = self.sim.get_sensor_observations()
        return True

    def stepAction(self, action):
        if action not in self.action_names:
            print("[ERROR][ActionController::stepAction]")
            print("\t action out of range!")
            return False
        self.observations = self.sim.step(action)
        return True

    def getSemanticScene(self):
        if self.sim is None:
            return None
        semantic_scene = self.sim.semantic_scene
        print_scene_recur(semantic_scene)
        return semantic_scene

    def setAgentState(self, agent_state):
        self.agent.set_state(agent_state)
        self.observations = self.sim.get_sensor_observations()
        return True

    def getAgentState(self):
        agent_state = self.agent.get_state()
        return agent_state


def demo():
    glb_file_path = \
        "/home/chli/chLi/ScanNet/scans/scene0474_02/scene0474_02_vh_clean.glb"

    sim_loader = SimLoader()
    sim_loader.loadSettings(glb_file_path)
    print("[INFO][sim_loader::demo]")
    print("\t load scene success!")
    return True
