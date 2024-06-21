#!/usr/bin/env python
# -*- coding: utf-8 -*-

from habitat_sim import AgentState

from habitat_sim_manage.Config.config import SIM_SETTING
from habitat_sim_manage.Config.input_map import INPUT_KEY_DICT
from habitat_sim_manage.Config.init_pose import INIT_POSE

from habitat_sim_manage.Data.point import Point
from habitat_sim_manage.Data.pose import Pose

from habitat_sim_manage.Method.rotations import \
    getRadFromDirection, getRotationFromRad

class BasePoseController(object):
    def __init__(self):
        self.input_key_dict = INPUT_KEY_DICT
        self.input_key_list = self.input_key_dict.keys()
        self.move_func_dict = {
            "move_forward": self.moveForward,
            "move_left": self.moveLeft,
            "move_right": self.moveRight,
            "move_backward": self.moveBackward,
            "move_up": self.moveUp,
            "move_down": self.moveDown,
        }
        self.rotate_func_dict = {
            "turn_left": self.turnLeft,
            "turn_right": self.turnRight,
            "turn_up": self.turnUp,
            "turn_down": self.turnDown,
            "head_left": self.headLeft,
            "head_right": self.headRight,
        }
        self.radius_func_dict = {
            "move_close": self.moveClose,
            "move_far": self.moveFar,
        }
        self.move_func_list = self.move_func_dict.keys()
        self.rotate_func_list = self.rotate_func_dict.keys()

        self.pose = INIT_POSE
        return

    def reset(self):
        self.input_key_dict = INPUT_KEY_DICT
        self.input_key_list = self.input_key_dict.keys()

        self.pose = INIT_POSE
        return True

    def resetPose(self):
        self.pose = INIT_POSE
        return True

    def getPoseByLookAt(self, position, look_at):
        direction_x = look_at.x - position.x
        direction_y = look_at.y - position.y
        direction_z = look_at.z - position.z
        direction = Point(direction_x, direction_y, direction_z)
        rad = getRadFromDirection(direction)
        pose = Pose(position, rad)
        return pose

    def getPoseFromLookAt(self, look_at, move_direction):
        position_x = look_at.x + move_direction.x
        position_y = look_at.y + move_direction.y
        position_z = look_at.z + move_direction.z
        position = Point(position_x, position_y, position_z)
        pose = self.getPoseByLookAt(position, look_at)
        return pose

    def getAgentStateFromPose(self, pose):
        agent_state = AgentState()
        agent_state.position = pose.position.toArray()
        agent_state.rotation = getRotationFromRad(pose.rad)
        return agent_state

    def getInitAgentState(self):
        return self.getAgentStateFromPose(INIT_POSE)

    def getAgentStateByAgentLookAt(self, position, look_at):
        pose = self.getPoseByLookAt(position, look_at)
        return self.getAgentStateFromPose(pose)

    def getAgentStateFromAgentLookAt(self, look_at, move_direction):
        pose = self.getPoseFromLookAt(look_at, move_direction)
        return self.getAgentStateFromPose(pose)

    def getAgentState(self):
        return self.getAgentStateFromPose(self.pose)

    def moveForward(self, move_dist):
        print("[ERROR][BasePoseController::moveForward]")
        print("\t this function not defined!")
        return False

    def moveLeft(self, move_dist):
        print("[ERROR][BasePoseController::moveLeft]")
        print("\t this function not defined!")
        return False

    def moveRight(self, move_dist):
        print("[ERROR][BasePoseController::moveRight]")
        print("\t this function not defined!")
        return False

    def moveBackward(self, move_dist):
        print("[ERROR][BasePoseController::moveBackward]")
        print("\t this function not defined!")
        return False

    def moveUp(self, move_dist):
        print("[ERROR][BasePoseController::moveUp]")
        print("\t this function not defined!")
        return False

    def moveDown(self, move_dist):
        print("[ERROR][BasePoseController::moveDown]")
        print("\t this function not defined!")
        return False

    def turnLeft(self, rotate_angle):
        print("[ERROR][BasePoseController::turnLeft]")
        print("\t this function not defined!")
        return False

    def turnRight(self, rotate_angle):
        print("[ERROR][BasePoseController::turnRight]")
        print("\t this function not defined!")
        return False

    def turnUp(self, rotate_angle):
        print("[ERROR][BasePoseController::turnUp]")
        print("\t this function not defined!")
        return False

    def turnDown(self, rotate_angle):
        print("[ERROR][BasePoseController::turnDown]")
        print("\t this function not defined!")
        return False

    def headLeft(self, rotate_angle):
        print("[ERROR][BasePoseController::headLeft]")
        print("\t this function not defined!")
        return False

    def headRight(self, rotate_angle):
        print("[ERROR][BasePoseController::headRight]")
        print("\t this function not defined!")
        return False

    def moveClose(self, move_dist):
        print("[ERROR][BasePoseController::moveClose]")
        print("\t this function not defined!")
        return False

    def movaFar(self, move_dist):
        print("[ERROR][BasePoseController::movaFar]")
        print("\t this function not defined!")
        return False

    def getAgentStateByKey(self,
                           input_key,
                           move_dist=SIM_SETTING['move_dist'],
                           rotate_angle=SIM_SETTING['rotate_angle'],
                           radius_move_dist=SIM_SETTING['radius_move_dist']):
        if input_key not in self.input_key_list:
            print("[WARN][BasePoseController::getAgentStateByKey]")
            print("\t input_key not valid!")
            return self.getAgentState()

        action = self.input_key_dict[input_key]
        if action in self.move_func_list:
            self.move_func_dict[action](move_dist)
            return self.getAgentState()
        if action in self.rotate_func_list:
            self.rotate_func_dict[action](rotate_angle)
            return self.getAgentState()
        if action in self.radius_func_dict:
            self.radius_func_dict[action](radius_move_dist)
            return self.getAgentState()

        print("[WARN][BasePoseController::getAgentStateByKey]")
        print("\t action not valid!")
        return self.getAgentState()

    def test(self):
        position = Point(2.7, 1.5, -3.0)
        look_at = Point(1.0, 0.5, -5.5)
        move_direction = Point(1.0, 1.0, 3.0)

        pose = self.getPoseByLookAt(position, look_at)
        print("[INFO][BasePoseController::test]")
        print("\t getPoseByLookAt")
        pose.outputInfo(1)

        pose = self.getPoseFromLookAt(look_at, move_direction)
        print("[INFO][BasePoseController::test]")
        print("\t getPoseFromLookAt")
        pose.outputInfo(1)

        agent_state = self.getAgentStateByAgentLookAt(position, look_at)
        print("[INFO][BasePoseController::test]")
        print("\t getAgentStateByAgentLookAt")
        print("\t agent_state: position", agent_state.position,
              "rotation", agent_state.rotation)

        agent_state = self.getAgentStateFromAgentLookAt(look_at, move_direction)
        print("[INFO][BasePoseController::test]")
        print("\t getAgentStateFromAgentLookAt")
        print("\t agent_state: position", agent_state.position,
              "rotation", agent_state.rotation)
        return True

def demo():
    base_pose_controller = BasePoseController()

    base_pose_controller.test()
    return True

