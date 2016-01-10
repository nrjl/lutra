#!/usr/bin/env python
import math
import numpy as np
import random
import time
import rospy
import geodesy.utm
from std_msgs.msg import Header
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
sys.path.insert(0, os.path.expanduser("~")+'/catkin_ws/src/ros_lutra/utilities')
from point_in_polygon import wn_PnPoly as inPoly

SAMPLE_RANGE = 4.0

def sample_cost_fun(cf, x):
    y = cf(x[0], x[1]) + random.normalvariate(0, 0.25)
    return y

def calc_true_path_cost(cost_fun, path, *args, **kwargs):
    true_cost = 0
    for i in range(len(path)):
        true_cost += cost_fun(path[i][0], path[i][1], *args, **kwargs)
    return true_cost

def calc_est_path_cost(gp_model, mean_val, path):
    true_cost,var_cost = gp_model.predict(np.asarray(path))
    true_cost += mean_val
    return true_cost.sum(), var_cost.sum()

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
                
class LutraFastMarchingExplorer:

    def __init__(self, total_waypoints, n_samples, operating_region, GPm, mean_depth=0, fake_sonar=False, max_depth=20.0):
        print "Creating Lutra Fast Marching Explorer object"
        print "Total number of waypoints to be visited: {0}".format(total_waypoints)
        print "Number of random samples: {0}".format(n_samples)
        print "Max depth for min graph cost: {0}".format(max_depth)
        print "Fake sonar activated: {0}".format(fake_sonar)
                
        # FM search area
        self.n_samples = n_samples
        self.total_waypoints = total_waypoints
        self.operating_region = operating_region
        self.start_node = self.operating_region.start_node
        self.goal_node = self.operating_region.goal_node
        self.gridsize = [self.operating_region.width, self.operating_region.height]
        
        # GP
        self.GPm = GPm
        self.mean_depth = mean_depth
        self.max_depth = max_depth
        self.delta_costs = [-1, 1]
        
        self.num_visited = 0
        self.nowstr = time.strftime("%Y_%m_%d-%H_%M")
        
        print "Setting up truth FM graph..."
        self.true_g = fm_graphtools.CostmapGridFixedObs(self.gridsize[0], self.gridsize[1], 
            obstacles=self.operating_region.obstacles, 
            bl_corner=[self.operating_region.left, self.operating_region.bottom])
        explorer_cost = bfm_explorer.mat_cost_function_GP(self.true_g, 
            cost_fun=bfm_explorer.GP_cost_function,
            GPm = self.GPm,
            max_depth=self.max_depth, 
            mean_depth=mean_depth)
        self.true_g.cost_fun = explorer_cost.calc_cost
        
        # Search over true field
        tFM = fast_marcher.FullBiFastMarcher(self.true_g)
        tFM.set_start(self.start_node)
        tFM.set_goal(self.goal_node)
        tFM.search()
        tFM.pull_path()
        self.best_path = tFM.path
        self.best_path_cost = calc_true_path_cost(bfm_explorer.GP_cost_function, 
            self.best_path,
            GPm = self.GPm,
            max_depth=self.max_depth, 
            mean_depth=self.mean_depth)

        # Plots
        self.fig, self.ax = plt.subplots(1, 2, sharex=True, sharey=True)
        self.fig.set_size_inches(15, 7)
        for axx in self.ax:
            axx.set_aspect('equal', 'datalim')
            axx.tick_params(labelbottom='on',labeltop='off')
            axx.set_xlabel('Easting (m)')
            axx.set_ylabel('Northing (m)')
            axx.autoscale(tight=True)
            axx.plot(self.operating_region.vertices[:,0], self.operating_region.vertices[:,1], 'k-')
        self.ax[0].set_title("True cost field")
        self.ax[1].set_title("Estimated cost - FM sampling")
        
        self.create_random_samples()
        self.best_cost_plot=[]
        self.cost_plot_matrix=[]
        self.video_frames = []
        
        # ROS pub/subs
        print "Setting up publishers and subscribers..."
        self.wp_sub_ = rospy.Subscriber('/crw_waypoint_reached', GeoPose, self.waypoint_reached_callback)
        self.pos_sub_ = rospy.Subscriber('/crw_geopose_pub', GeoPose, self.pose_callback)
        self.depth_sub_ = rospy.Subscriber('/crw_sonar_pub', Range, self.sonar_callback)
        self.wp_pub_ = rospy.Publisher('/crw_waypoint_sub', GeoPose, queue_size=10)
        
        self.sonar_time = rospy.Time.now()-rospy.Duration(10)
        
        # Fake sonar
        self.fake_sonar = fake_sonar
        if self.fake_sonar:
            print "Setting up simulated sonar..."
            self.depth_pub_ = rospy.Publisher('/crw_sonar_pub', Range, queue_size=10)
            self.next_fakesonar = rospy.Time.now()
            newhead = Header(seq = 0, stamp = rospy.Time.now(), frame_id='sonar')
            self.fake_sonar_msg = Range(header=newhead,min_range=0.5, max_range=100.0)
                
        print "Setup complete, waiting for first waypoint. The next waypoint reached "
        print "will be assumed to represent the local origin of the search area."

    def get_local_coords(self, utm_pose):
        return np.array([math.floor(utm_pose.easting - self.zero_utm.easting), math.floor(utm_pose.northing - self.zero_utm.northing)])

    def local_to_grid(self, point):
        px = np.min(np.max(point[0] - self.origin_offset[0], 0), self.gridsize[0])
        py = np.min(np.max(point[1] - self.origin_offset[1], 0), self.gridsize[1])
        return np.array([px,py])
        
    def grid_to_local(self, point):
        px = point[0] + self.origin_offset[0]
        py = point[1] + self.origin_offset[1]
        return np.array([px,py])    

    def get_utm_coords(self, local_pose):
        out_pose = copy.copy(self.zero_utm)
        out_pose.easting += local_pose[0]
        out_pose.northing += local_pose[1]
        return out_pose

    def previously_sampled(self, point):
        if self.num_visited >= 1:
            for ii in range(self.fm_sampling_explorer.X.shape[0]):
                if np.sqrt(((self.fm_sampling_explorer.X[ii,:] - point)**2).sum()) < SAMPLE_RANGE:
                    return True
        return False
        
    def create_random_samples(self):
        # Rejection sampling of points in operating region - IN LOCAL COORDINATES
        self.sample_locations = np.zeros((self.n_samples, 2))
        numsamp = 0
        while numsamp < self.n_samples:
            Px = np.random.uniform(self.operating_region.left, self.operating_region.right)
            Py = np.random.uniform(self.operating_region.bottom, self.operating_region.top)
            if not self.previously_sampled([Px,Py]) and inPoly([Px,Py], self.operating_region.vertices):
                self.sample_locations[numsamp] = [Px,Py]
                numsamp += 1

    def sonar_callback(self, msg):
        self.sonar_val = msg.range
        self.sonar_time = rospy.Time.now()
    
    def pose_callback(self, msg):
        self.cgeopose_ = msg
        self.cpose_ = msg.position
        self.cquat_ = msg.orientation
        
        if self.num_visited > 0 and self.fake_sonar and rospy.Time.now() > self.next_fakesonar:
            self.next_fakesonar = rospy.Time.now() + rospy.Duration(1.0)
            pp = geodesy.utm.fromMsg(self.cpose_)
            clocalpos = self.get_local_coords(pp)
            depth = self.GPm.predict([clocalpos])
            self.fake_sonar_msg.range = depth
            self.fake_sonar_msg.header.seq += 1 
            self.depth_pub_.publish(self.fake_sonar_msg)
                
    def waypoint_reached_callback(self, msg):
        print "Waypoint {0} reached.".format(self.num_visited)
        self.cgeopose_ = msg
        self.cpose_ = msg.position
        self.cquat_ = msg.orientation
        pp = geodesy.utm.fromMsg(self.cpose_)
        
        if self.num_visited == 0:
            print "Waypoint 0: defining local origin."
            self.zero_utm = pp
        self.num_visited += 1
        
        # Wait for valid/recent sonar data
        waiter = True
        while rospy.Time.now() > self.sonar_time + rospy.Duration(0.5):
            if waiter:
                waiter=False
                print "Waiting for valid sonar data"
            time.sleep(0.1)
        print "Sonar data recorded, depth = {0}m".format(self.sonar_depth)
        
        # Current position in world frame 
        cX = self.get_local_coords(pp)
        
        # Current position in grid frame
        #cX = np.array([clocalpos[0]-self.origin_offset[0], clocalpos[1]-self.origin_offset[1]])
        
        if self.num_visited <= 1:
            # Initial sample set
            X = np.array([cX])
            Y = np.zeros((1, 1))
            Y[0] = self.sonar_depth
            self.fm_sampling_explorer = bfm_explorer.fast_marching_explorer(self.gridsize, 
                self.start_node, self.goal_node, X, Y, 
                obs=self.true_g.obstacles, 
                mean_value=self.GPmean, 
                GP_l=self.GPm.kern.lengthscale[0], 
                GP_sv=self.GPm.kern.variance[0], 
                GP_sn=self.GPm.Gaussian_noise.variance[0],
                max_depth=self.max_depth, 
                mean_depth=self.mean_depth
                )

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

        for [tx,ty] in self.sample_locations:

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
    
    nwaypoints = rospy.get_param('~explorer_waypoints', 51)
    nsamp = rospy.get_param('~explorer_samples', 100)
    max_depth = rospy.get_param('~max_depth', 20.0)
    fake_sonar = rospy.get_param('~fake_sonar', False)
    
    V_Ireland = np.array([[0,0], [-43,-38],[-70,-94], [-60,-150],[0,-180],[54,-152],[85,-70],[0,0]])
    start_node = (0,-2)
    goal_node = (-30,-150)
    model_file = os.path.expanduser("~")+'/catkin_ws/src/ros_lutra/data/IrelandLnModel.pkl'
    fh = open(model_file, 'rb')
    GP_model = pickle.load(fh)
    mean_depth = pickle.load(fh)
    fh.close()
    op_region = OperatingRegion(V_Ireland, start_node, goal_node)
    fmex = LutraFastMarchingExplorer(nwaypoints, nsamp, op_region, GP_model, 
        mean_depth=mean_depth, 
        max_depth=max_depth,
        fake_sonar=fake_sonar)
    rospy.spin()
