#!/usr/bin/env python
import math
import numpy as np
import random
import time
import rospy
import geodesy.utm
from geographic_msgs.msg import GeoPose
import fm_plottools
import fm_graphtools
import bfm_explorer
import matplotlib.pyplot as plt
# import matplotlib.animation as animation
import fast_marcher
import copy
import pickle

MEAN_VALUE = 3
SAMPLE_RANGE = 4.5
MAP_BLOBS = [[30, 20, 16, 10],
             [60, 40, 6, 15],
             [10, 40, 32, 12],
             [60, 5, 15, 9],
             [25, 35, 8, 15],
             [70, 30, 12, 12],
             [80, 40, 16, 15],
             [5, 60, 30, 7]]

def explore_cost_function(a, b, blobs=MAP_BLOBS):
    cost = MEAN_VALUE
    for i in range(np.shape(blobs)[0]):
        cost += blobs[i][3]*math.exp(-math.sqrt((a-blobs[i][0])**2 + (b-blobs[i][1])**2)/blobs[i][2])
    return cost

def sample_cost_fun(cf, x):
    y = cf(x[0], x[1]) + random.normalvariate(0, 0.25)
    return y

def calc_true_path_cost(cost_fun, path, *args):
    true_cost = 0
    for i in range(len(path)):
        true_cost += cost_fun(path[i][0], path[i][1], *args)
    return true_cost

def calc_est_path_cost(gp_model, mean_val, path):
    true_cost,var_cost = gp_model.predict(np.asarray(path))
    true_cost += mean_val
    return true_cost.sum(), var_cost.sum()


class LutraFastMarchingExplorer:

    def __init__(self, nwp, gridsize):
        print "Creating Lutra Fast Marching Explorer object"
        self.pos_sub_ = rospy.Subscriber('/crw_waypoint_reached', GeoPose, self.pose_callback)
        self.wp_pub_ = rospy.Publisher('/crw_waypoint_sub', GeoPose, queue_size=10)
        self.total_waypoints = nwp
        self.gridsize = gridsize
        self.delta_costs = [-1, 1]
        self.num_visited = 0
        self.nowstr = time.strftime("%Y_%m_%d-%H_%M")

        self.fig, self.ax = plt.subplots(1, 2, sharex=True, sharey=True)
        self.fig.set_size_inches(15, 7)
        for i in range(2):
            self.ax[i].set_aspect('equal', 'datalim')
            self.ax[i].tick_params(labelbottom='on',labeltop='off')
            self.ax[i].set_xlabel('x')
            self.ax[i].set_ylabel('y')
            self.ax[i].autoscale(tight=True)
        self.ax[0].set_title("True cost field")
        self.ax[1].set_title("Estimated cost - FM sampling")
        self.best_cost_plot=[]
        self.cost_plot_matrix=[]
        self.video_frames = []
        print "Waiting for first waypoint. The next waypoint reached will "
        print "be assumed to represent the local origin of the search area."

    def get_local_coords(self, utm_pose):
        return np.array([math.floor(utm_pose.easting - self.zero_utm.easting), math.floor(utm_pose.northing - self.zero_utm.northing)])

    def get_utm_coords(self, local_pose):
        out_pose = copy.copy(self.zero_utm)
        out_pose.easting += local_pose[0]
        out_pose.northing += local_pose[1]
        return out_pose

    def previously_sampled(self, point):
        for ii in range(self.fm_sampling_explorer.X.shape[0]):
            if np.sqrt(((self.fm_sampling_explorer.X[ii,:] - point)**2).sum()) < SAMPLE_RANGE:
                return True
        return False

    def pose_callback(self, msg):
        print "Waypoint {0} reached.".format(self.num_visited)
        self.cgeopose_ = msg
        self.cpose_ = msg.position
        self.cquat_ = msg.orientation
        pp = geodesy.utm.fromMsg(self.cpose_)
        self.num_visited += 1

        if self.num_visited <= 1:
            print "Arrived at first waypoint, creating fast march explorer."
            self.zero_utm = pp
            self.test_gridx = range(2, self.gridsize[0], 10);
            self.test_gridy = range(2, self.gridsize[1], 10);

            self.true_g = fm_graphtools.CostmapGrid(self.gridsize[0], self.gridsize[1], explore_cost_function)
            explorer_cost = bfm_explorer.mat_cost_function(self.true_g, explore_cost_function)
            self.true_g.cost_fun = explorer_cost.calc_cost

            start_node = (3,3)
            end_node = (self.gridsize[0]-3, self.gridsize[1]-3)

            # Search over true field
            tFM = fast_marcher.FullBiFastMarcher(self.true_g)
            tFM.set_start(start_node)
            tFM.set_goal(end_node)
            tFM.search()
            tFM.pull_path()
            self.best_path = tFM.path
            self.best_path_cost = calc_true_path_cost(explore_cost_function, self.best_path)

            # Initial sample set
            X = np.array([self.get_local_coords(pp)])
            Y = np.zeros((1, 1))
            Y[0] = sample_cost_fun(explore_cost_function, X[0,:])
            self.fm_sampling_explorer = bfm_explorer.fast_marching_explorer(self.gridsize, start_node, end_node, X, Y, MEAN_VALUE, self.true_g.obstacles)

        elif self.num_visited == self.total_waypoints:
            print "Arrived at final waypoint, saving data."
            fh = open('lutra_fastmarchlog_'+self.nowstr+'.p', 'wb')
            pickle.dump(self.fm_sampling_explorer.X, fh)
            pickle.dump(self.fm_sampling_explorer.Y, fh)
            fh.close()
            self.plot_current_path(self.get_local_coords(pp))
            # ani1 = animation.ArtistAnimation(self.fig, self.video_frames, interval=1000, repeat_delay=0)
            # ani1.save('fm_explorer_'+self.nowstr+'.mp4', writer = 'avconv', fps=1, bitrate=1500)
            return

        else:
            clocalpos = self.get_local_coords(pp)
            self.fm_sampling_explorer.add_observation(clocalpos, sample_cost_fun(explore_cost_function, clocalpos))

        # Find next sample point
        fm_best_cost = -1

        for tx in self.test_gridx:
            for ty in self.test_gridy:
                if  ((tx,ty) in self.true_g.obstacles):
                    continue

                if not self.previously_sampled([tx,ty]):
                    current_value = 0
                    for td in self.delta_costs:
                        stdY = math.sqrt(self.fm_sampling_explorer.varYfull[ty*self.gridsize[0]+tx])
                        cost_update =fm_graphtools.polynomial_cost_modifier(self.fm_sampling_explorer.GP_cost_graph, tx, ty, 15, td*stdY)
                        current_value += self.fm_sampling_explorer.cost_update(cost_update)
                    if fm_best_cost == -1 or (current_value < fm_best_cost):
                        fm_best_cost = current_value
                        fm_bestX = [tx,ty]
        self.plot_current_path(fm_bestX)
        target_utm = self.get_utm_coords(fm_bestX)
        print "Next target point selected: E = {0}m, N = {1}m.".format(fm_bestX[0], fm_bestX[1])
        self.pub_point(target_utm)

    def pub_point(self, pp):
        self.cpose_ = pp.toMsg()
        self.cgeopose_.position = self.cpose_
        self.wp_pub_.publish(self.cgeopose_)

    def plot_current_path(self, nexttarget):
        graph_frame = []
        if not self.best_cost_plot:
            self.best_cost_plot, barlims = fm_plottools.draw_grid(self.ax[0], self.true_g, self.best_path, 19)
            graph_frame.extend(self.best_cost_plot)
        while self.ax[1].lines:
            self.ax[1].lines.pop()
        self.cost_plot_matrix, barlims = fm_plottools.draw_grid(self.ax[1], self.fm_sampling_explorer.GP_cost_graph, self.fm_sampling_explorer.fbFM.path, 19)
        self.cost_plot_matrix.append(self.ax[1].plot(self.fm_sampling_explorer.X[:,0], self.fm_sampling_explorer.X[:,1], 'rx')[0])
        self.cost_plot_matrix.append(self.ax[1].plot(nexttarget[0], nexttarget[1], 'bo')[0])
        graph_frame.extend(self.cost_plot_matrix)
        self.video_frames.append(graph_frame)
        self.fig.savefig('fm_explorer_{0}_S{1:02d}.pdf'.format(self.nowstr, self.num_visited-1), bbox_inches='tight')
        # plt.draw()

if __name__ == '__main__':

    rospy.init_node('fm_explorer', anonymous=False)
    nwaypoints = rospy.get_param('/explorer_waypoints', 31)
    eastwidth = rospy.get_param('/explorer_width', 80)
    northheight = rospy.get_param('/explorer_height', 60)
    fmex = LutraFastMarchingExplorer(nwaypoints, [eastwidth, northheight])
    rospy.spin()
