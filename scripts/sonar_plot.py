#!/usr/bin/env python
import math
import numpy as np
import rospy
import geodesy.utm
from std_msgs.msg import Header
from geographic_msgs.msg import GeoPoint, GeoPose
from sensor_msgs.msg import Range
from visualization_msgs.msg import Marker
import tf
import os.path
import pickle


def sample_cost_fun(cf, x):
    y = cf(x[0], x[1]) + random.normalvariate(0, 0.25)
    return y
        
class LutraSonar:

    def __init__(self, max_sonar_points, sonar_marker_timeout, mean_depth, fake_sonar_freq=-1.0, GPm=None, pond_image=None):
        print "Creating Lutra Sonar object"
        print "Mean depth estimate: {0}".format(mean_depth)
        print "Fake sonar frequency: {0}".format(fake_sonar_freq)
                
        self.num_visited = 0
        
        if pond_image:
            pond_locations = np.genfromtxt(pond_image+'.csv', delimiter=',')
            pond_origin = GeoPoint(pond_locations[2,0],pond_locations[2,1],pond_locations[2,2])
            self.zero_utm = geodesy.utm.fromMsg(pond_origin)        
        
        # Marker message initialisation
        self.depth_marker = Marker()
        self.depth_marker.header.seq = 0
        self.depth_marker.header.frame_id ='sonar'
        self.depth_marker.header.stamp = rospy.Time.now()
        self.depth_marker.ns='sonar_markers'
        self.depth_marker.id=0
        self.depth_marker.type = self.depth_marker.CUBE
        self.depth_marker.action = self.depth_marker.ADD
        self.depth_marker.scale.x = 0.1
        self.depth_marker.scale.y = 1.0
        self.depth_marker.scale.z = 1.0
        self.depth_marker.color.a = 1.0
        self.depth_marker.color.r = 1.0
        self.depth_marker.color.g = 1.0
        self.depth_marker.color.b = 0.0
        self.depth_marker.pose.orientation.w = 1.0
        self.depth_marker.pose.position.x = 0.0
        self.depth_marker.pose.position.y = 0.0
        self.depth_marker.pose.position.z = 0.0
        self.depth_marker.lifetime = rospy.Duration(sonar_marker_timeout)
        self.max_sonar_points = max_sonar_points
        
        # Fake sonar
        self.fake_sonar_freq = fake_sonar_freq
        if self.fake_sonar_freq > 0.0:
            print "Setting up simulated sonar..."
            self.GPm = GPm
            self.mean_depth = mean_depth
            self.depth_pub_ = rospy.Publisher('/crw_sonar_pub', Range, queue_size=10)
            self.next_fakesonar = rospy.Time.now()
            newhead = Header(seq = 0, stamp = rospy.Time.now(), frame_id='sonar')
            self.fake_sonar_msg = Range(header=newhead,min_range=0.5, max_range=100.0,field_of_view=math.pi/180.0*10.0)
        
        # ROS pub/subs
        print "Setting up publishers and subscribers..."
        self.depth_marker_pub_ = rospy.Publisher('/sonar_marker', Marker, queue_size=1)
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
    
    model_file = os.path.expanduser("~")+'/catkin_ws/src/ros_lutra/data/IrelandLnModel.pkl'
    fh = open(model_file, 'rb')
    GP_model = pickle.load(fh)
    mean_depth = pickle.load(fh)
    fh.close()
    pond_image = os.path.expanduser("~")+'/catkin_ws/src/ros_lutra/data/ireland_lane_pond'
    sonar_plotter = LutraSonar(max_points, timeout, mean_depth, fake_sonar_freq, GPm=GP_model, pond_image=pond_image)
    rospy.spin()
