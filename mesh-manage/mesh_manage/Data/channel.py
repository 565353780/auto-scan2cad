#!/usr/bin/env python
# -*- coding: utf-8 -*-

class Channel(object):
    def __init__(self, name="", value=0):
        self.name = name
        self.value = value
        self.size = 0
        self.type = ""
        self.count = 1

        self.updateName()
        self.updateValue()
        return

    def updateName(self):
        if self.name == "":
            return True

        if self.name in [
                "x", "y", "z",
                "nx","ny", "nz"]:
            self.size = 4
            self.type = "F"
            self.count = 1
            return True

        if self.name in [
                "r", "g", "b", "rgb", "red", "green", "blue",
                "label", "semantic_label", "instance_label"]:
            self.size = 4
            self.type = "U"
            self.count = 1
            return True

        self.size = 4
        self.type = "F"
        self.count = 1
        return True

    def updateValue(self):
        if self.type == "":
            return True

        if self.type == "F":
            self.value = float(self.value)
            return True
        if self.type == "U":
            self.value = int(self.value)
            return True
        if self.type == "I":
            self.value = int(self.value)
            return True
        return True

    def setName(self, name):
        self.name = name
        if not self.updateName():
            print("[ERROR][Channel::setName]")
            print("\t updateName failed!")
            return False
        if not self.updateValue():
            print("[ERROR][Channel::setName]")
            print("\t updateValue failed!")
            return False
        return True

    def setValue(self, value):
        self.value = value
        if not self.updateValue():
            print("[ERROR][Channel::setValue]")
            print("\t updateValue failed!")
            return False
        return True

    def outputInfo(self, info_level=0):
        line_start = "\t" * info_level
        print(line_start + "[Channel]")
        print(line_start + "\t name =", self.name)
        print(line_start + "\t value =", self.value)
        print(line_start + "\t size =",self.size)
        print(line_start + "\t type =", self.type)
        print(line_start + "\t count =", self.count)
        return True

