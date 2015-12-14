#!/usr/bin/env python
import math
import numpy as np
import rospy
import geodesy.utm
from geographic_msgs.msg import GeoPose
from sensor_msgs.msg import Temperature, Range
import copy
import cPickle as pickle


class LutraBagReader:

    def __init__(self, offpos=[0.0,0.0]):
        print "Creating Lutra Bag Reader"
        self.pose_sub_ = rospy.Subscriber('/crw_geopose_pub', GeoPose, self.pose_callback)
        self.temp_sub_ = rospy.Subscriber('/crw_temp_pub', Temperature, self.temp_callback)
        self.sonar_sub_ = rospy.Subscriber('/crw_sonar_pub', Range, self.sonar_callback)
        self.poses = np.array([])
        self.sonar = np.array([])
        self.temps = np.array([])

        self.start_time = 0
        self.offpos = offpos

        print "Initialised."
        
    def get_local_coords(self, utm_pose):
        return np.array([utm_pose.easting - self.zero_utm.easting, utm_pose.northing - self.zero_utm.northing])

    def get_utm_coords(self, local_pose):
        out_pose = copy.copy(self.zero_utm)
        out_pose.easting += local_pose[0]
        out_pose.northing += local_pose[1]
        return out_pose
        
    def add_pose(self, msg):
        ctime = rospy.get_time()
        pp = geodesy.utm.fromMsg(msg.position)
        clocalpos = self.get_local_coords(pp)
        if self.poses.size:
            self.poses = np.append(self.poses, [[ctime-self.start_time, clocalpos[0], clocalpos[1]]],axis=0)
        else:
            self.poses = np.array([[ctime-self.start_time, clocalpos[0], clocalpos[1]]])

    def pose_callback(self, msg):
        self.cgeopose_ = msg
        self.cpose_ = msg.position
        self.cquat_ = msg.orientation
        pp = geodesy.utm.fromMsg(self.cpose_)
        tnow = rospy.get_time()
                
        if not self.poses.size:
            print "First pose record, initialising origin here."
            origin = copy.copy(pp)
            origin.easting -= self.offpos[0]
            origin.northing -= self.offpos[1]
            self.zero_utm = origin
            self.start_time = tnow
            
        self.add_pose(msg)
            

    def save_output(self):
        print "Saving data."
        fh = open('sonar_data.pkl', 'wb')
        pickle.dump(self.poses, fh)
        pickle.dump(self.sonar, fh)
        pickle.dump(self.temps, fh)
        fh.close()
        print "Done, shutting down."
            
            
    def sonar_callback(self, msg):
        if not self.poses.size:
            return
             
        ctime = rospy.get_time()-self.start_time
        if self.sonar.size:
            self.sonar = np.append(self.sonar, [[ctime, self.poses[-1,1], self.poses[-1,2], msg.range]],axis=0)
        else:
            self.sonar = np.array([ [ctime, self.poses[-1,1], self.poses[-1,2], msg.range] ])
        #print "Sonar observation value: {0} at time: {1}".format(msg.range, ctime)
        
    def temp_callback(self, msg):
        if not self.poses.size or msg.temperature <= 0.1:
            return
             
        ctime = rospy.get_time()-self.start_time
        if self.temps.size:
            self.temps = np.append(self.temps, [[ctime, self.poses[-1,1], self.poses[-1,2], msg.temperature]],axis=0)
        else:
            self.temps = np.array([ [ctime, self.poses[-1,1], self.poses[-1,2], msg.temperature] ])
        #print "Temperature observation value: {0} at time: {1}".format(msg.temperature, ctime)

if __name__ == '__main__':
    rospy.init_node('lutra_bag_reader', anonymous=False)
    while not rospy.is_shutdown():
        fmex = LutraBagReader(offpos=[0.0, 0.0])
        rospy.spin()
    fmex.save_output()