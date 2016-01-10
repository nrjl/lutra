# Testing script for breakpoints
import numpy as np
import os

import fm_plottools
import fm_graphtools
import bfm_explorer
import fast_marcher
import pickle
import os.path
import matplotlib.pyplot as plt

from lutra_fm_ireland import OperatingRegion

def GP_cost_function(x, y, max_depth=6.0, mean_depth=0.0):
    # Cost function shold be strictly positive (depth < 10)
    X = np.array([np.ravel(x), np.ravel(y)]).transpose()
    mean,var = GP_model.predict(X)
    mean = max_depth-(mean+mean_depth)
    mean[mean < 0.1] = 0.1
    if len(mean) == 1:
        return mean[0]
    else:
        return mean

if 'GP_model' not in locals():
    V_Ireland = np.array([[0,0], [-43,-38],[-70,-94], [-60,-150],[0,-180],[54,-152],[85,-70],[0,0]])
    start_node = (0,-2)
    goal_node = (-30,-150)
    model_file = os.path.expanduser("~")+'/catkin_ws/src/ros_lutra/data/IrelandLnModel.pkl'
    fh = open(model_file, 'rb')
    GP_model = pickle.load(fh)
    mean_depth = pickle.load(fh)
    fh.close()
    op_region = OperatingRegion(V_Ireland, start_node, goal_node)
    true_g = fm_graphtools.CostmapGridFixedObs(op_region.width, op_region.height, obstacles=op_region.obstacles, bl_corner=[op_region.left, op_region.bottom])

explorer_cost = bfm_explorer.mat_cost_function_GP(true_g, GP_cost_function, max_depth=4.5, mean_depth=mean_depth)
true_g.cost_fun = explorer_cost.calc_cost

tFM = fast_marcher.FullBiFastMarcher(true_g)
tFM.set_goal(goal_node)
tFM.set_start(start_node)
tFM.search()
tFM.pull_path()

f0, a0 = fm_plottools.init_fig()
f1, a1 = fm_plottools.init_fig()
f2, a2 = fm_plottools.init_fig()
fm_plottools.draw_grid(a0, true_g, tFM.path)
fm_plottools.draw_costmap(a1, true_g, tFM.FastMarcherSG.cost_to_come, tFM.path)
fm_plottools.draw_fbfmcost(a2, true_g, tFM.path_cost, tFM.path)
plt.show()