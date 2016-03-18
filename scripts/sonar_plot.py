#!/usr/bin/env python
import math
import numpy as np
import rospy
import geodesy.utm
from std_msgs.msg import Header, ColorRGBA
from geographic_msgs.msg import GeoPoint, GeoPose
from geometry_msgs.msg import Point
from sensor_msgs.msg import Range
from visualization_msgs.msg import Marker
import tf
import os.path
import pickle
import GPy
from matplotlib import cm

def sample_cost_fun(cf, x):
    y = cf(x[0], x[1]) + random.normalvariate(0, 0.25)
    return y
    
def simple_marker_init():
    newmarker = Marker()
    newmarker.header.seq = 0
    newmarker.header.stamp = rospy.Time.now()
    newmarker.id=0
    newmarker.type = newmarker.CUBE
    newmarker.action = newmarker.ADD
    newmarker.scale.x = 1.0
    newmarker.scale.y = 1.0
    newmarker.scale.z = 1.0
    newmarker.color.a = 1.0
    newmarker.color.r = 1.0
    newmarker.color.g = 1.0
    newmarker.color.b = 0.0
    newmarker.pose.orientation.w = 1.0
    newmarker.pose.position.x = 0.0
    newmarker.pose.position.y = 0.0
    newmarker.pose.position.z = 0.0
    return newmarker    
        
class LutraSonar:

    def __init__(self, max_sonar_points, sonar_marker_timeout, mean_depth=0.0, fake_sonar_freq=-1.0, GPm=None, pond_image=None, online_GP=False,GP_l=14.0,GP_sv=45.0,GP_sn=1.0):
        print "Creating Lutra Sonar object"
        print "Mean depth estimate: {0}".format(mean_depth)
        print "Fake sonar frequency: {0}".format(fake_sonar_freq)
                
        self.num_visited = 0
        self.mean_depth = mean_depth
        
        if pond_image:
            pond_locations = np.genfromtxt(pond_image+'.csv', delimiter=',')
            pond_origin = GeoPoint(pond_locations[2,0],pond_locations[2,1],pond_locations[2,2])
            self.zero_utm = geodesy.utm.fromMsg(pond_origin)     
            self.pond_bl = self.get_local_coords(geodesy.utm.fromMsg(GeoPoint(pond_locations[0,0],pond_locations[0,1],pond_locations[0,2])))
            self.pond_tr = self.get_local_coords(geodesy.utm.fromMsg(GeoPoint(pond_locations[1,0],pond_locations[1,1],pond_locations[1,2])))
            
        
        # Marker message initialisation
        self.depth_marker = simple_marker_init()
        self.depth_marker.header.frame_id ='sonar'
        self.depth_marker.ns='sonar_markers'
        self.depth_marker.scale.x = 0.1
        self.depth_marker.lifetime = rospy.Duration(sonar_marker_timeout)
        self.max_sonar_points = max_sonar_points
        
        # Fake sonar
        self.fake_sonar_freq = fake_sonar_freq
        if self.fake_sonar_freq > 0.0:
            print "Setting up simulated sonar..."
            self.GPm = GPm
            self.depth_pub_ = rospy.Publisher('/crw_sonar_pub', Range, queue_size=10)
            self.next_fakesonar = rospy.Time.now()
            newhead = Header(seq = 0, stamp = rospy.Time.now(), frame_id='sonar')
            self.fake_sonar_msg = Range(header=newhead,min_range=0.5, max_range=100.0,field_of_view=math.pi/180.0*10.0)
        
        self.online_GP = False
        if online_GP:
            if pond_image:
                print "Creating sonar GP"
                self.online_GP = True
                self.max_obs = 500
                self.GP_obs = np.zeros((self.max_obs,3))
                self.n_obs = 0
                kern = GPy.kern.RBF(2, variance=GP_sn, lengthscale=GP_l)
                self.sonar_GP = GPy.models.GPRegression(np.zeros((1,2)),np.zeros((1,1)),kern)
                self.sonar_GP.Gaussian_noise.variance = GP_sn
                lx = np.linspace(self.pond_bl[0],self.pond_tr[0],50)
                ly = np.linspace(self.pond_bl[1],self.pond_tr[1],50)
                self.X_star = np.array([[x,y] for x in lx for y in ly])
                self.Y_star = np.zeros((self.X_star.shape[0], 1))
                self.GP_marker = simple_marker_init()
                self.GP_marker.header.frame_id ='world'
                self.GP_marker.type = self.GP_marker.CUBE_LIST
                self.GP_marker.ns='sonar_markers'
                self.GP_marker.scale.z = 0.1
                self.GP_marker.scale.x = lx[1]-lx[0]
                self.GP_marker.scale.y = ly[1]-ly[0]
                for i in range(self.X_star.shape[0]):
                    self.GP_marker.points.append(Point(self.X_star[i,0],self.X_star[i,1],self.Y_star[i]))
                    self.GP_marker.colors.append(ColorRGBA(1.0,0.0,0.0,1.0))
            else:
                print "Cannot create GP plot without pond_image bounds"

        
        # ROS pub/subs
        print "Setting up publishers and subscribers..."
        self.depth_marker_pub_ = rospy.Publisher('/sonar_marker', Marker, queue_size=1)
        if self.online_GP:
            self.GP_pub_=rospy.Publisher('/sonar_GP',Marker,queue_size=1)
        self.wp_sub_ = rospy.Subscriber('/crw_waypoint_reached', GeoPose, self.waypoint_reached_callback)
        self.pos_sub_ = rospy.Subscriber('/crw_geopose_pub', GeoPose, self.pose_callback)
        self.depth_sub_ = rospy.Subscriber('/crw_sonar_pub', Range, self.sonar_callback)
        
        # TF broadcaster and listener
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.tf_listener =  tf.TransformListener()
                
        print "Setup complete."
        if not pond_image:
            print "The next waypoint reached will be assumed to "
            print "represent the local origin of the search area."

    def get_local_coords(self, utm_pose):
        return np.array([utm_pose.easting-self.zero_utm.easting, utm_pose.northing-self.zero_utm.northing])

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

    def sonar_callback(self, msg):
        self.depth_marker.header.seq += 1
        self.depth_marker.header.stamp = rospy.Time.now()
        self.depth_marker.id = (self.depth_marker.id+1) % self.max_sonar_points
        self.depth_marker.pose.position.x = msg.range
        self.depth_marker_pub_.publish(self.depth_marker)
        
        if self.online_GP:
            try:
                (trans,rot) = self.tf_listener.lookupTransform('/world', '/sonar', rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                return
            self.GP_obs[self.n_obs%self.max_obs] = np.array([trans[0],trans[1],msg.range])
            self.n_obs = (self.n_obs+1)
            obs_dex = min(self.max_obs,self.n_obs)
            X = np.atleast_2d(self.GP_obs[0:obs_dex,0:2])
            Y = np.atleast_2d(self.GP_obs[0:obs_dex,2]).transpose()
            self.sonar_GP.set_XY(X,Y-self.mean_depth)
            y_star,y_var = self.sonar_GP.predict(self.X_star)
            y_star = y_star + self.mean_depth
            cols = cm.jet((np.ravel(y_var) - y_var.min())/(y_var.max()-y_var.min()))
            for i in range(self.X_star.shape[0]):
                self.GP_marker.points[i].z = -y_star[i]
                self.GP_marker.colors[i].r = cols[i,0]
                self.GP_marker.colors[i].g = cols[i,1]
                self.GP_marker.colors[i].b = cols[i,2]
            self.GP_marker.header.seq += 1
            self.GP_marker.header.stamp = rospy.Time.now()
            self.GP_pub_.publish(self.GP_marker)
    
    def pose_callback(self, msg):
        self.cgeopose_ = msg
        self.cpose_ = msg.position
        self.cquat_ = msg.orientation
        euler = tf.transformations.euler_from_quaternion((msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w))
        # print "R: {0}, P: {1}, Y: {2}".format(*euler)
        self.pp = geodesy.utm.fromMsg(self.cpose_)
        tnow = rospy.Time.now()
        if hasattr(self, 'zero_utm'):
            self.clocalpos = self.get_local_coords(self.pp)
            self.tf_broadcaster.sendTransform((self.clocalpos[0], self.clocalpos[1], 0),
                tf.transformations.quaternion_from_euler(0, 0, math.pi-euler[1]), tnow, "base_link", "world")
                
            if (self.fake_sonar_freq > 0) and tnow > self.next_fakesonar:
                depth,dvar = self.GPm.predict(np.reshape(self.clocalpos, (1,2)))
                self.fake_sonar_msg.range = depth[0]+self.mean_depth
                self.fake_sonar_msg.header.seq += 1 
                self.fake_sonar_msg.header.stamp = tnow 
                self.depth_pub_.publish(self.fake_sonar_msg)
                self.next_fakesonar = tnow + rospy.Duration(1.0/self.fake_sonar_freq)
            
                
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

if __name__ == '__main__':

    rospy.init_node('sonar_plot', anonymous=False)
    
    fake_sonar_freq = rospy.get_param('~fake_sonar_freq', -1.0)
    max_points = rospy.get_param('~max_sonar_points', 100)
    timeout = rospy.get_param('~sonar_marker_timeout', 60.0)
    online_GP = rospy.get_param('~online_GP', False)
    
    model_file = os.path.expanduser("~")+'/catkin_ws/src/ros_lutra/data/IrelandLnModel.pkl'
    fh = open(model_file, 'rb')
    GP_model = pickle.load(fh)
    mean_depth = pickle.load(fh)
    fh.close()
    pond_image = os.path.expanduser("~")+'/catkin_ws/src/ros_lutra/data/ireland_lane_pond'
    sonar_plotter = LutraSonar(max_points, timeout, mean_depth, fake_sonar_freq, GPm=GP_model, pond_image=pond_image, online_GP=online_GP)
    rospy.spin()
