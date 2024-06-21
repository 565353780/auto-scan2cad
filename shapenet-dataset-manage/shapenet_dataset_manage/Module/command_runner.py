#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from tqdm import tqdm
from subprocess import Popen


class CommandRunner(object):

    def __init__(self, process_num=24):
        self.process_num = process_num

        self.command_list = []
        self.next_start_command_idx = 0
        self.finished_command_idx = -1

        self.dev_null = open(os.devnull, "w")
        return

    def reset(self):
        self.command_list = []
        self.next_start_command_idx = 0
        self.finished_command_idx = -1
        return True

    def addCommand(self, command):
        self.command_list.append(command)
        return True

    def addCommandList(self, command_list):
        self.command_list += command_list
        return True

    def getCommand(self):
        if len(self.command_list) == 0:
            return None

        if self.next_start_command_idx >= len(self.command_list):
            return None

        command = self.command_list[self.next_start_command_idx]
        self.next_start_command_idx += 1
        return command

    def finishOneCommand(self):
        self.finished_command_idx += 1
        return True

    def getFinishedCommandNum(self):
        return self.finished_command_idx + 1

    def getRunningCommandNum(self):
        return self.next_start_command_idx - self.finished_command_idx

    def getTotalCommandNum(self):
        return len(self.command_list)

    def getRunningCommandList(self):
        first_running_command_idx = self.finished_command_idx + 1
        last_running_command_idx = self.next_start_command_idx - 1
        return self.command_list[
            first_running_command_idx:last_running_command_idx]

    def start(self):
        print("[INFO][CommandRunner::start]")
        print("\t start running with", self.process_num, "processes...")

        pbar = tqdm(total=self.getTotalCommandNum())

        process_list = []
        while True:
            if len(process_list) < self.process_num:
                command = self.getCommand()
                if command is not None:
                    process_list.append(
                        Popen(command, stdout=self.dev_null, shell=True))
            if len(process_list) == 0:
                break
            for i in range(len(process_list) - 1, -1, -1):
                if Popen.poll(process_list[i]) is not None:
                    process_list.pop(i)
                    self.finishOneCommand()
                    pbar.update(1)
        pbar.close()
        return True
