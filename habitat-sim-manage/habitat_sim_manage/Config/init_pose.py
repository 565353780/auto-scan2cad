#!/usr/bin/env python
# -*- coding: utf-8 -*-

from habitat_sim_manage.Data.point import Point
from habitat_sim_manage.Data.rad import Rad
from habitat_sim_manage.Data.pose import Pose

INIT_POSITION = Point(0.0, 1.5, 0.0)
INIT_RAD = Rad(0.0, 0.0, 0.0)
INIT_POSE = Pose(INIT_POSITION, INIT_RAD)
INIT_RADIUS = 2.0

