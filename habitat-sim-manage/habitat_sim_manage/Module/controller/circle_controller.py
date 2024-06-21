#!/usr/bin/env python
# -*- coding: utf-8 -*-

from habitat_sim_manage.Config.init_pose import INIT_RADIUS
from habitat_sim_manage.Config.input_map import INPUT_KEY_DICT

from habitat_sim_manage.Method.poses import \
    getMoveForwardPose, getMoveLeftPose, \
    getMoveRightPose, getMoveBackwardPose, \
    getMoveUpPose, getMoveDownPose
from habitat_sim_manage.Method.circle_poses import \
    getCircleTurnLeftPose, getCircleTurnRightPose, \
    getCircleTurnUpPose, getCircleTurnDownPose, \
    getCircleHeadLeftPose, getCircleHeadRightPose

from habitat_sim_manage.Module.controller.base_pose_controller import BasePoseController

class CircleController(BasePoseController):
    def __init__(self):
        super(CircleController, self).__init__()
        self.radius = INIT_RADIUS
        return

    def reset(self):
        super().reset()
        self.radius = INIT_RADIUS
        return True

    def updateCenterPose(self):
        return True

    def updatePose(self):
        return True

    def moveForward(self, move_dist):
        self.pose = getMoveForwardPose(self.pose, move_dist)
        return True

    def moveLeft(self, move_dist):
        self.pose = getMoveLeftPose(self.pose, move_dist)
        return True

    def moveRight(self, move_dist):
        self.pose = getMoveRightPose(self.pose, move_dist)
        return True

    def moveBackward(self, move_dist):
        self.pose = getMoveBackwardPose(self.pose, move_dist)
        return True

    def moveUp(self, move_dist):
        self.pose = getMoveUpPose(self.pose, move_dist)
        return True

    def moveDown(self, move_dist):
        self.pose = getMoveDownPose(self.pose, move_dist)
        return True

    def turnLeft(self, rotate_angle):
        self.pose = getCircleTurnLeftPose(self.pose, self.radius, rotate_angle)
        return True

    def turnRight(self, rotate_angle):
        self.pose = getCircleTurnRightPose(self.pose, self.radius, rotate_angle)
        return True

    def turnUp(self, rotate_angle):
        self.pose = getCircleTurnUpPose(self.pose, self.radius, rotate_angle)
        return True

    def turnDown(self, rotate_angle):
        self.pose = getCircleTurnDownPose(self.pose, self.radius, rotate_angle)
        return True

    def headLeft(self, rotate_angle):
        self.pose = getCircleHeadLeftPose(self.pose, self.radius, rotate_angle)
        return True

    def headRight(self, rotate_angle):
        self.pose = getCircleHeadRightPose(self.pose, self.radius, rotate_angle)
        return True

    def moveClose(self, move_dist):
        self.radius = max(self.radius - move_dist, 0.0)
        return True

    def moveFar(self, move_dist):
        self.radius += move_dist
        return True

    def test(self):
        super().test()

        input_key_list = INPUT_KEY_DICT.keys()
        for input_key in input_key_list:
            agent_state = self.getAgentStateByKey(input_key)
            print("[INFO][CircleController::test]")
            print("\t getAgentStateByKey")
            print("\t agent_state: position", agent_state.position,
                  "rotation", agent_state.rotation)
        return True

def demo():
    circle_controller = CircleController()

    circle_controller.test()
    return True

