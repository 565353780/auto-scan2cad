#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mesh_manage.Data.channel import Channel

class ChannelPoint(object):
    def __init__(self, channel_name_list=[], channel_value_list=[]):
        self.channel_list = []

        self.setChannelValueList(channel_name_list, channel_value_list)
        return

    def updateFloatRGB(self):
        r = self.getChannelValue("r")
        g = self.getChannelValue("g")
        b = self.getChannelValue("b")
        if r is None or g is None or b is None:
            return True
        rgb = r << 16 | g << 8 | b | 1<<24
        self.setChannelValue("rgb", rgb)
        return True

    def setChannelValue(self, channel_name, channel_value):
        if len(self.channel_list) == 0:
            new_channel = Channel(channel_name, channel_value)
            self.channel_list.append(new_channel)
            return True

        for exist_channel in self.channel_list:
            if exist_channel.name == channel_name:
                exist_channel.setValue(channel_value)
                if channel_name != "rgb":
                    self.updateFloatRGB()
                return True

        new_channel = Channel(channel_name, channel_value)
        self.channel_list.append(new_channel)
        if channel_name != "rgb":
            self.updateFloatRGB()
        return True

    def setChannelValueList(self, channel_name_list, channel_value_list):
        for i in range(len(channel_name_list)):
            self.setChannelValue(channel_name_list[i], channel_value_list[i])
        return True

    def getChannelNameList(self):
        if len(self.channel_list) == 0:
            return []
        channel_name_list = [channel.name for channel in self.channel_list]
        return channel_name_list

    def getChannelValue(self, channel_name):
        if len(self.channel_list) == 0:
            return None

        for exist_channel in self.channel_list:
            if exist_channel.name != channel_name:
                continue
            return exist_channel.value
        return None

    def getChannelValueList(self, channel_name_list):
        channel_value_list = [self.getChannelValue(channel_name) for channel_name in channel_name_list]
        return channel_value_list

    def outputInfo(self, info_level=0):
        line_start = "\t" * info_level
        print(line_start + "[ChannelPoint]")
        for channel in self.channel_list:
            channel.outputInfo(info_level + 1)
        return True

