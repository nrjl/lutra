#!/usr/bin/env python
import math
import numpy as np
import random
import time
import rospy
import geodesy.utm
from std_msgs.msg import Header
from geographic_msgs.msg import GeoPoint, GeoPose
from sensor_msgs.msg import Range
import fm_plottools
import fm_graphtools
import bfm_explorer
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import fast_marcher
import copy
import pickle
import os.path
import sys
import scipy
import collections
sys.path.insert(0, os.path.expanduser("~")+'/catkin_ws/src/ros_lutra/utilities')
from point_in_polygon import wn_PnPoly as inPoly

SAMPLE_RANGE = 4.0

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
                
class LutraFastMarchingExplorer:

    def __init__(self, total_waypoints, n_samples, operating_region, GPm, mean_depth=0, 
        fake_sonar=False, max_depth=20.0, pond_image=None, continuous_sonar=False, swath_samples=False):
        print "Creating Lutra Fast Marching Explorer object"
        print "Total number of waypoints to be visited: {0}".format(total_waypoints)
        print "Number of random samples: {0}".format(n_samples)
        print "Max depth for min graph cost: {0}".format(max_depth)
        print "Fake sonar activated: {0}".format(fake_sonar)
        print "Continous sonar measurements: {0}".format(continuous_sonar)
        print "Swath samples: {0}".format(swath_samples)
        
        self.continuous_sonar=continuous_sonar
        self.swath_samples=swath_samples
        self.sonar_buffer = collections.deque(maxlen=20)
        
        # FM search area
        self.n_samples = n_samples
        self.total_waypoints = total_waypoints
        self.operating_region = operating_region
        self.start_node = self.operating_region.start_node
        self.goal_node = self.operating_region.goal_node
        self.gridsize = [self.operating_region.width, self.operating_region.height]
        self.observations = np.zeros((n_samples, 6)) # Time, lat, lon, x, y, depth
        
        # GP
        self.GPm = GPm
        self.mean_depth = mean_depth
        self.max_depth = max_depth
        self.delta_costs = [-1]
        
        self.num_visited = 0
        self.nowstr = time.strftime("%Y_%m_%d-%H_%M")
        
        print "Setting up truth FM graph..."
        self.true_g = fm_graphtools.CostmapGridFixedObs(self.gridsize[0], self.gridsize[1], 
            obstacles=self.operating_region.obstacles, 
            bl_corner=[self.operating_region.left, self.operating_region.bottom])
        print "Building truth cost matrix..."
        explorer_cost = bfm_explorer.mat_cost_function_GP(self.true_g, 
            cost_fun=bfm_explorer.GP_cost_function,
            GPm = self.GPm,
            max_depth=self.max_depth, 
            mean_depth=mean_depth)
        self.true_g.cost_fun = explorer_cost.calc_cost
        
        # Search over true field
        print "Creating full searcher over true field..."
        tFM = fast_marcher.FullBiFastMarcher(self.true_g)
        tFM.set_start(self.start_node)
        tFM.set_goal(self.goal_node)
        print "Searching over true field..."
        tFM.search()
        tFM.pull_path()
        self.best_path = tFM.path
        print "Calculating best cost cost over true field..."
        self.best_path_cost = calc_true_path_cost(bfm_explorer.GP_cost_function, 
            np.array(self.best_path),
            GPm = self.GPm,
            max_depth=self.max_depth, 
            mean_depth=self.mean_depth)

        # Plots
        print "Creating plots..."
        self.fig, self.ax = plt.subplots(1, 3, sharex=True, sharey=True)
        self.fig.set_size_inches(15, 5)
        self.cost_alpha=1.0
        if pond_image:
            self.pond = scipy.misc.imread(pond_image+'.png')
            pond_locations = np.genfromtxt(pond_image+'.csv', delimiter=',')
            pond_origin = GeoPoint(pond_locations[2,0],pond_locations[2,1],pond_locations[2,2])
            self.zero_utm = geodesy.utm.fromMsg(pond_origin)
            self.pond_bl = self.get_local_coords(geodesy.utm.fromMsg(GeoPoint(pond_locations[0,0],pond_locations[0,1],pond_locations[0,2])))
            self.pond_tr = self.get_local_coords(geodesy.utm.fromMsg(GeoPoint(pond_locations[1,0],pond_locations[1,1],pond_locations[1,2])))
            self.cost_alpha = 0.5

        for axx in self.ax:
            axx.set_aspect('equal', 'datalim')
            axx.tick_params(labelbottom='on',labeltop='off')
            axx.set_xlabel('Easting (m)')
            axx.set_ylabel('Northing (m)')
            axx.autoscale(tight=True)
            if pond_image:
                axx.imshow(self.pond, extent = (self.pond_bl[0],self.pond_tr[0],self.pond_bl[1],self.pond_tr[1]))
            axx.plot(self.operating_region.vertices[:,0], self.operating_region.vertices[:,1], 'k-')
            axx.set_xlim([self.true_g.left-20, self.true_g.right+20])
            axx.set_ylim([self.true_g.bottom-20, self.true_g.top+20])
        self.ax[0].set_title("True cost field")
        self.ax[1].set_title("Estimated cost - FMEx sampling")
        self.ax[2].set_title("GP Variance - FMEx Sampling")
        
        print "Generating random samples..."
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
                
        print "Setup complete, waiting for first waypoint."
        if not pond_image:
            print "The next waypoint reached will be assumed to "
            print "represent the local origin of the search area."

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
        self.sample_locations = np.zeros((self.n_samples, 2), dtype='int')
        numsamp = 0
        while numsamp < self.n_samples:
            Px = np.random.randint(self.operating_region.left, self.operating_region.right)
            Py = np.random.randint(self.operating_region.bottom, self.operating_region.top)
            if (Px,Py) not in self.operating_region.obstacles and ((Px-self.goal_node[0])**2 + (Py-self.goal_node[1])**2) > 100: #not self.previously_sampled([Px,Py]) and inPoly([Px,Py], self.operating_region.vertices):
                self.sample_locations[numsamp] = [Px,Py]
                numsamp += 1

    def sonar_callback(self, msg):
        self.sonar_depth = msg.range
        self.sonar_time = rospy.Time.now()
        if hasattr(self, 'zero_utm'):
            self.sonar_buffer.append([self.clocalpos[0], self.clocalpos[1], self.sonar_depth])
    
    def pose_callback(self, msg):
        self.cgeopose_ = msg
        self.cpose_ = msg.position
        self.cquat_ = msg.orientation
        self.pp = geodesy.utm.fromMsg(self.cpose_)
        if hasattr(self, 'zero_utm'):
            self.clocalpos = self.get_local_coords(self.pp)
        
        if hasattr(self, 'zero_utm') and self.fake_sonar and rospy.Time.now() > self.next_fakesonar:
            depth,dvar = self.GPm.predict(np.reshape(self.clocalpos, (1,2)))
            self.fake_sonar_msg.range = depth[0]+self.mean_depth
            self.fake_sonar_msg.header.seq += 1 
            self.depth_pub_.publish(self.fake_sonar_msg)
            self.next_fakesonar = rospy.Time.now() + rospy.Duration(1.0)
            
                
    def waypoint_reached_callback(self, msg):
        print "Waypoint {0} reached.".format(self.num_visited)
        self.cgeopose_ = msg
        self.cpose_ = msg.position
        self.cquat_ = msg.orientation
        self.pp = geodesy.utm.fromMsg(self.cpose_)
        
        if not hasattr(self, 'zero_utm'):
            print "Waypoint 0: defining local origin."
            self.zero_utm = self.pp
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
        self.clocalpos = self.get_local_coords(self.pp)
        print "Current local position: {0}, {1}".format(self.clocalpos[0], self.clocalpos[1])
        self.observations[self.num_visited-1,:] = np.array(
            [rospy.Time.now().secs, self.cpose_.latitude, self.cpose_.longitude,
            self.clocalpos[0], self.clocalpos[1], self.sonar_depth])
        tt = time.time()
        
        # Current position in grid frame
        #cX = np.array([clocalpos[0]-self.origin_offset[0], clocalpos[1]-self.origin_offset[1]])
        
        if self.num_visited <= 1:            
            # Initial sample set
            X = np.reshape(self.clocalpos, (1,2))
            Y = np.zeros((1, 1))
            Y[0] = self.sonar_depth
            self.fm_sampling_explorer = bfm_explorer.fast_marching_explorer(self.gridsize, 
                self.start_node, self.goal_node, X, Y, 
                obs=self.true_g.obstacles, 
                mean_value=self.mean_depth, 
                GP_l=self.GPm.kern.lengthscale[0]*1.0, 
                GP_sv=self.GPm.kern.variance[0]*1.0, 
                GP_sn=self.GPm.Gaussian_noise.variance[0]*5.0,
                max_depth=self.max_depth, 
                mean_depth=self.mean_depth,
                bl_corner=[self.operating_region.left, self.operating_region.bottom]
                )
            self.fm_sampling_explorer.search()
            self.poly_cost = fm_graphtools.polynomial_precompute_cost_modifier(self.true_g, 14, min_val=0.001)
            print "Initial setup took {0:0.2f}s".format(time.time()-tt)
                

        elif self.num_visited == self.total_waypoints:
            print "Arrived at final waypoint, saving data."
            fh = open('lutra_fastmarchlog_'+self.nowstr+'.p', 'wb')
            pickle.dump(self.fm_sampling_explorer.X, fh)
            pickle.dump(self.fm_sampling_explorer.Y, fh)
            pickle.dump(self.observations, fh)
            fh.close()
            #self.plot_current_path(self.get_local_coords(pp))
            #try:
            #    ani1 = animation.ArtistAnimation(self.fig, self.video_frames, interval=1000, repeat_delay=0)
            #    ani1.save('fm_explorer_'+self.nowstr+'.mp4', writer = 'avconv', fps=1, bitrate=1500)
            #except:
            #    print "Saving video directly failed."
            #
            #fh = open('lutra_videoframes_'+self.nowstr+'.p', 'wb')
            #try:
            #    pickle.dump(self.fig, fh)
            #    pickle.dump(self.video_frames)
            #except:
            #    print "Dumping video frames to file failed"
            #fh.close()
            return

        else:
            if self.continuous_sonar: #pop off all the recent sonar data
                oX = []
                oY = []
                while True:
                    try:
                        cO = self.sonar_buffer.pop()
                        oX.append([cO[0], cO[1]])
                        oY.append(cO[2])
                    except IndexError:
                        break
                oX = np.array(oX)
                oY = np.reshape(oY, (len(oY),1))
            else:
                oX = [self.clocalpos]; oY=[[self.sonar_depth]]
            self.fm_sampling_explorer.add_observation(oX,oY)
            print "Adding {1} new sample(s) took {0:0.2f}s".format(time.time()-tt, len(oY))            

        # Find next sample point
        fm_best_cost = -1
        tt = time.time()
        for ii in range(self.sample_locations.shape[0]):
            [tx,ty] = self.sample_locations[ii]
            #print self.fm_sampling_explorer.cmodel.var_dict
            #if  ((tx,ty) not in self.true_g.obstacles) and not self.previously_sampled([tx,ty]):
            current_value = 0
            for td in self.delta_costs:
                stdY = self.fm_sampling_explorer.GP_cost_graph.var_fun(tx,ty) #math.sqrt()
                cost_update =self.poly_cost.calc_cost(tx, ty, td*stdY)
                current_value += self.fm_sampling_explorer.cost_update_new(cost_update)
            #print "Sample {0:2d} at ({1:4d},{2:4d}), std={3:0.3f}, pcost={4:0.3f}".format(ii,tx,ty,stdY,current_value)
            if fm_best_cost == -1 or (current_value < fm_best_cost):
                fm_best_cost = current_value
                fm_bestX = [tx,ty]
                fm_bestVar = stdY
                fm_besti = ii
        print "Finding best sample took {0:0.2f}s.".format(time.time()-tt)
        target_utm = self.get_utm_coords(fm_bestX)
        print "Next target point selected: E = {0}m, N = {1}m (c={2:0.3f}, std={3:0.3f}, i={4}).".format(
            fm_bestX[0], fm_bestX[1], fm_best_cost, fm_bestVar,fm_besti)
        self.pub_point(target_utm)
        self.sonar_buffer.clear()
        tt = time.time()
        self.plot_current_path(fm_bestX)
        print "Plotting took {0:0.2f}s.".format(time.time()-tt)
        self.create_random_samples()

    def pub_point(self, pp):
        self.cpose_ = pp.toMsg()
        self.cgeopose_.position = self.cpose_
        self.wp_pub_.publish(self.cgeopose_)

    def plot_current_path(self, nexttarget):
        c_best_cost = calc_true_path_cost(bfm_explorer.GP_cost_function, 
            np.array(self.fm_sampling_explorer.fbFM.path),
            GPm = self.GPm,
            max_depth=self.max_depth, 
            mean_depth=self.mean_depth)
        if not self.best_cost_plot:
            self.best_cost_plot, barlims = fm_plottools.draw_grid(self.ax[0], self.true_g, path=self.best_path, 
                min_cost=0.0, max_cost=self.max_depth, alpha=self.cost_alpha)
            self.ax[0].text(self.true_g.left, self.true_g.bottom-10, "Best path cost = {0:0.3f}".format(self.best_path_cost),color='w')
            self.text1 = self.ax[1].text(self.true_g.left, self.true_g.bottom-10, "Current path cost = {0:0.3f}".format(c_best_cost),color='w')
            self.est_cost_plot, barlims = fm_plottools.draw_grid(self.ax[1], 
                self.fm_sampling_explorer.GP_cost_graph, self.fm_sampling_explorer.fbFM.path, 
                max_cost=self.max_depth, min_cost=0.0, alpha=self.cost_alpha)
            self.est_cost_plot.append(self.ax[1].plot(self.fm_sampling_explorer.X[:,0], self.fm_sampling_explorer.X[:,1], 'rx')[0])
            self.est_cost_plot.append(self.ax[1].plot(nexttarget[0], nexttarget[1], 'bo')[0])
            self.var_plot, barlims = fm_plottools.draw_grid(self.ax[2], 
                self.fm_sampling_explorer.GP_cost_graph,
                min_cost=0.0, alpha=self.cost_alpha, 
                cost_fun=lambda(x):self.fm_sampling_explorer.GP_cost_graph.var_fun(x[0],x[1]))
            self.var_plot.append(self.ax[2].plot(self.sample_locations[:,0], self.sample_locations[:,1], 'k.')[0])
            self.ep = self.est_cost_plot[0].get_array()
            self.ev = self.var_plot[0].get_array()
            for axx in self.ax:
                axx.set_xlim([self.true_g.left-20, self.true_g.right+20])
                axx.set_ylim([self.true_g.bottom-20, self.true_g.top+20])
        else:
            self.text1.set_text("Current path cost = {0:0.3f}".format(c_best_cost))
            # Array is transposed!
            for locx,locy in self.fm_sampling_explorer.cmodel.cost_dict:
                self.ep[locy-self.true_g.bottom, locx-self.true_g.left] = self.fm_sampling_explorer.cmodel.cost_dict[(locx,locy)]
                self.ev[locy-self.true_g.bottom, locx-self.true_g.left] = self.fm_sampling_explorer.cmodel.var_dict[(locx,locy)]
            self.est_cost_plot[0].set_array(self.ep)
            xp, yp = zip(*self.fm_sampling_explorer.fbFM.path)
            self.est_cost_plot[1].set_data(xp, yp)
            self.est_cost_plot[2].set_data(xp[0], yp[0])
            self.est_cost_plot[3].set_data(xp[-1], yp[-1])
            self.est_cost_plot[4].set_data(self.fm_sampling_explorer.X[:,0], self.fm_sampling_explorer.X[:,1])
            self.est_cost_plot[5].set_data(nexttarget[0], nexttarget[1])
            self.var_plot[0].set_array(self.ev)
            self.var_plot[1].set_data(self.sample_locations[:,0], self.sample_locations[:,1])            
        try:
            self.fig.savefig('fm_explorer_{0}_S{1:02d}.pdf'.format(self.nowstr, self.num_visited-1), bbox_inches='tight')
        except:
            print "Save figure failed, continuing search."
        # plt.draw()

if __name__ == '__main__':

    rospy.init_node('fm_explorer', anonymous=False)
    
    nwaypoints = rospy.get_param('~explorer_waypoints', 31)
    nsamp = rospy.get_param('~explorer_samples', 100)
    max_depth = rospy.get_param('~max_depth', 10.0)
    fake_sonar = rospy.get_param('~fake_sonar', False)
    continuous_sonar = rospy.get_param('~continuous_sonar', False)
    swath_samples = rospy.get_param('~swath_samples', False)
    
    V_Ireland = np.array([[0,0], [-43,-38],[-70,-94], [-60,-170],[0,-180],[54,-152],[85,-70],[0,0]])
    start_node = (0,-2)
    goal_node = (-30,-160)
    model_file = os.path.expanduser("~")+'/catkin_ws/src/ros_lutra/data/IrelandLnModel.pkl'
    fh = open(model_file, 'rb')
    GP_model = pickle.load(fh)
    mean_depth = pickle.load(fh)
    fh.close()
    op_region = OperatingRegion(V_Ireland, start_node, goal_node)
    pond_image = os.path.expanduser("~")+'/catkin_ws/src/ros_lutra/data/ireland_lane_pond'
    fmex = LutraFastMarchingExplorer(nwaypoints, nsamp, op_region, GP_model, 
        mean_depth=mean_depth, 
        max_depth=max_depth,
        fake_sonar=fake_sonar,
        pond_image=pond_image,
        continuous_sonar=continuous_sonar,
        swath_samples=swath_samples)
    rospy.spin()
