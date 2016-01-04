#!/usr/bin/env python
import math
import numpy as np
import random
import time
import rospy
import geodesy.utm
from geographic_msgs.msg import GeoPose
from sensor_msgs.msg import Range
import fm_plottools
import fm_graphtools
import bfm_explorer
import matplotlib.pyplot as plt
# import matplotlib.animation as animation
import fast_marcher
import copy
import pickle
import os.path
import sys
sys.path.insert(0, '../utilities')
from point_in_polygon import wn_PnPoly as inPoly

SAMPLE_RANGE = 4.0

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

    def __init__(self, nwp, nloc, V, sgnodes, GPm, mean_depth):
        print "Creating Lutra Fast Marching Explorer object"
        
        # ROS pub/subs
        self.wp_sub_ = rospy.Subscriber('/crw_waypoint_reached', GeoPose, self.waypoint_reached_callback)
        self.pos_sub_ = rospy.Subscriber('/crw_geopose_pub', GeoPose, self.pose_callback)
        self.depth_sub_ = rospy.Subscriber('/crw_sonar_pub', Range, self.sonar_callback)
        self.wp_pub_ = rospy.Publisher('/crw_waypoint_sub', GeoPose, queue_size=10)
        
        # FM search area
        self.n_samples = nloc
        self.total_waypoints = nwp
        self.operating_area = V
        self.start_node = sgnodes[0]
        self.end_node = sgnodes[1]
        self.gridsize = [V[:,0].max() - V[:,0].min(), V[:,1].max() - V[:,1].min()]
        self.origin_offset = [V[:,0].min(), V[:,1].min()]
        
        # GP
        self.GPm = GPm
        self.mean_depth = mean_depth
        self.delta_costs = [-1, 1]
        
        self.num_visited = 0
        self.nowstr = time.strftime("%Y_%m_%d-%H_%M")

        # Plots
        self.fig, self.ax = plt.subplots(1, 2, sharex=True, sharey=True)
        self.fig.set_size_inches(15, 7)
        for axx in self.ax:
            axx.set_aspect('equal', 'datalim')
            axx.tick_params(labelbottom='on',labeltop='off')
            axx.set_xlabel('Easting (m)')
            axx.set_ylabel('Northing (m)')
            axx.autoscale(tight=True)
            axx.plot(self.operating_area[:,0], self.operating_area[:,1], 'k-')
        self.ax[0].set_title("True cost field")
        self.ax[1].set_title("Estimated cost - FM sampling")
        
        self.create_random_samples()
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
        
    def create_random_samples(self):
        # Rejection sampling of points in V region
        self.sample_locations = np.zeros((self.n_samples, 2))
        numsamp = 0
        while numsamp < self.n_samples:
            Px = np.random.uniform(self.V[:,0].min(), self.V[:,0].max())
            Py = np.random.uniform(self.V[:,0].min(), self.V[:,0].max())
            if not self.previously_sampled([Px,Py]) and inPoly([Px,Py], self.operating_area):
                self.sample_locations[numsamp] = [Px,Py]
                numsamp += 1

    def sonar_callback(self, msg):
        self.sonar_val = msg.range
        self.sonar_time = rospy.get_time()
    
    def pose_callback(self, msg):
        None
        
    def GP_cost_function(self, x, y):
        offset_pos = [x+self.origin_offset[0], y+self.origin_offset[1]]
        return self.GPm.predict([offset_pos])
                
    def waypoint_reached_callback(self, msg):
        print "Waypoint {0} reached.".format(self.num_visited)
        # Wait for valid/recent sonar data
        waiter = True
        while not hasattr(self, 'sonar_time') or rospy.get_time()-self.sonar_time > 0.5:
            if waiter:
                waiter=False
                print "Waiting for valid sonar data"
            time.sleep(0.1)
        print "Sonar data recorded, depth = {0}m".format(self.sonar_depth)
        self.cgeopose_ = msg
        self.cpose_ = msg.position
        self.cquat_ = msg.orientation
        pp = geodesy.utm.fromMsg(self.cpose_)
        self.num_visited += 1

        # Current position in world frame 
        clocalpos = self.get_local_coords(pp)
        
        # Current position in grid frame
        cX = np.array([clocalpos[0]-self.origin_offset[0], clocalpos[1]-self.origin_offset[1]])
        
        if self.num_visited <= 1:
            print "Arrived at first waypoint, creating fast march explorer."
            self.zero_utm = pp

            self.true_g = fm_graphtools.CostmapGrid(self.gridsize[0], self.gridsize[1], self.GP_cost_function)
            explorer_cost = bfm_explorer.mat_cost_function(self.true_g, self.GP_cost_function)
            self.true_g.cost_fun = explorer_cost.calc_cost
            
            # Grid location transform
            start_node = (self.start_node[0]-self.origin_offset[0], self.start_node[1]-self.origin_offset[1])
            goal_node = (self.end_node[0]-self.origin_offset[0], self.end_node[1]-self.origin_offset[1])
            
            # Search over true field
            tFM = fast_marcher.FullBiFastMarcher(self.true_g)
            tFM.set_start(start_node)
            tFM.set_goal(goal_node)
            tFM.search()
            tFM.pull_path()
            self.best_path = tFM.path
            self.best_path_cost = calc_true_path_cost(self.GP_cost_function, self.best_path)

            # Initial sample set
            X = np.array([cX])
            Y = np.zeros((1, 1))
            Y[0] = self.sonar_depth
            self.fm_sampling_explorer = bfm_explorer.fast_marching_explorer(self.gridsize, start_node, goal_node, X, Y, self.GPmean, self.true_g.obstacles)

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
            self.fm_sampling_explorer.add_observation(cX, self.sonar_depth)

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

    V_Ireland = np.array([[0,0], [-43,-38],[-70,-94], [-60,-150],[0,-180],[54,-152],[85,-70],[0,0]])
    start_end = np.array([[0,-2], [-30,-150]])
    rospy.init_node('fm_explorer', anonymous=False)
    nwaypoints = rospy.get_param('/explorer_waypoints', 51)
    model_file = os.path.expanduser("~")+'/catkin_ws/src/ros_lutra/data/IrelandLnModel.pkl'
    fh = open(model_file, 'rb')
    GP_model = pickle.load(fh)
    mean_depth = pickle.load(fh)
    fh.close()    
    fmex = LutraFastMarchingExplorer(nwaypoints, 100, V_Ireland, start_end, GP_model, mean_depth)
    rospy.spin()
