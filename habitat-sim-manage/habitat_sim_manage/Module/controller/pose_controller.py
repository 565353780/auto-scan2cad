#!/usr/bin/env python
# -*- coding: utf-8 -*-

from habitat_sim_manage.Config.input_map import INPUT_KEY_DICT

from habitat_sim_manage.Method.poses import \
    getMoveForwardPose, getMoveLeftPose, \
    getMoveRightPose, getMoveBackwardPose, \
    getMoveUpPose, getMoveDownPose, \
    getTurnLeftPose, getTurnRightPose, \
    getTurnUpPose, getTurnDownPose, \
    getHeadLeftPose, getHeadRightPose

from habitat_sim_manage.Module.controller.base_pose_controller import BasePoseController

class PoseController(BasePoseController):
    def __init__(self):
        super(PoseController, self).__init__()
        return

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
        self.pose = getTurnLeftPose(self.pose, rotate_angle)
        return True

    def turnRight(self, rotate_angle):
        self.pose = getTurnRightPose(self.pose, rotate_angle)
        return True

    def turnUp(self, rotate_angle):
        self.pose = getTurnUpPose(self.pose, rotate_angle)
        return True

    def turnDown(self, rotate_angle):
        self.pose = getTurnDownPose(self.pose, rotate_angle)
        return True

    def headLeft(self, rotate_angle):
        self.pose = getHeadLeftPose(self.pose, rotate_angle)
        return True

    def headRight(self, rotate_angle):
        self.pose = getHeadRightPose(self.pose, rotate_angle)
        return True

    def moveClose(self, move_dist):
        return self.moveForward(move_dist)

    def moveFar(self, move_dist):
        return self.moveBackward(move_dist)

    def test(self):
        super().test()

        input_key_list = INPUT_KEY_DICT.keys()
        for input_key in input_key_list:
            agent_state = self.getAgentStateByKey(input_key)
            print("[INFO][PoseController::test]")
            print("\t getAgentStateByKey")
            print("\t agent_state: position", agent_state.position,
                  "rotation", agent_state.rotation)
        return True

def demo():
    pose_controller = PoseController()

    pose_controller.test()
    return True

