#!/usr/bin/env python
# -*- coding: utf-8 -*-

from habitat_sim_manage.Config.input_map import INPUT_KEY_DICT

class ActionController(object):
    def __init__(self):
        self.input_key_dict = INPUT_KEY_DICT
        self.input_key_list = self.input_key_dict.keys()
        return

    def reset(self):
        self.input_key_dict = INPUT_KEY_DICT
        self.input_key_list = self.input_key_dict.keys()
        return True

    def getAction(self, input_key):
        if input_key not in self.input_key_list:
            print("[WARN][ActionController::getAction]")
            print("\t input_key not valid!")
            return None

        action = self.input_key_dict[input_key]
        return action

def demo():
    input_key_list = INPUT_KEY_DICT.keys()

    action_controller = ActionController()

    for input_key in input_key_list:
        action = action_controller.getAction(input_key)
        print(input_key, "->", action)
    return True

