#!/usr/bin/env python
import math
import numpy as np
import rospy
import geodesy.utm
from std_msgs.msg import Header
from geographic_msgs.msg import GeoPoint, GeoPose
from sensor_msgs.msg import Range
from visualization_msgs.msg import Marker
import fm_plottools



def sample_cost_fun(cf, x):
    y = cf(x[0], x[1]) + random.normalvariate(0, 0.25)
    return y

def calc_true_path_cost(cost_fun, path, *args, **kwargs):
    true_cost,true_var = cost_fun(path[:,0], path[:,1], *args, **kwargs)
    return true_cost.sum()

class OperatingRegion:
    def __init__(self, V, start_node, goal_node):
        self.vertices = V
        self.start_node = start_node
        self.goal_node = goal_node
        Vmin = np.floor(self.vertices.min(0)).astype(int) # [minx, miny]
        Vmax = np.ceil(self.vertices.max(0)).astype(int) # [maxx, maxy]
        self.left = Vmin[0]
        self.bottom = Vmin[1]
        self.right = Vmax[0]
        self.top = Vmax[1]
        self.width = self.right - self.left+1
        self.height = self.top - self.bottom+1
        self.build_obstacles()

    def build_obstacles(self):
        # Construct obstacles
        lx = np.arange(self.left, self.right+1, dtype='int')
        ly = np.arange(self.bottom, self.top+1, dtype='int')
        self.obstacles = [(x,y) for x in lx for y in ly if not inPoly((x,y), self.vertices)]
