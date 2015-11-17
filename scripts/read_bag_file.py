#!/usr/bin/env python
import math
import numpy as np
import rospy
import geodesy.utm
from geographic_msgs.msg import GeoPose
import copy
import pickle


class LutraBagReader:

    def __init__(self, nwp, dt, X, Y, offset=0, offpos=[0.0,0.0]):
        print "Creating Lutra Bag Reader"
        self.pos_sub_ = rospy.Subscriber('/crw_waypoint_reached', GeoPose, self.pose_callback)
        self.wp_sub_ = rospy.Subscriber('/crw_waypoint_sub', GeoPose, self.waypoint_callback)
        self.pose_sub_ = rospy.Subscriber('/crw_geopose_pub', GeoPose, self.geopose_callback)
        self.total_waypoints = nwp
        self.start_time = 0
        self.frame_trigger = 0
        self.dt = dt
        self.X = X
        self.Y = Y
        self.poses = np.array([])
        self.targets = np.array([])
        self.current_waypoint = 0
        self.offset = offset
        self.offpos = offpos

        print "Waiting for first waypoint."
        
    def get_local_coords(self, utm_pose):
        return np.array([utm_pose.easting - self.zero_utm.easting, utm_pose.northing - self.zero_utm.northing])

    def get_utm_coords(self, local_pose):
        out_pose = copy.copy(self.zero_utm)
        out_pose.easting += local_pose[0]
        out_pose.northing += local_pose[1]
        return out_pose
    
    def pose_callback(self, msg):
        self.cgeopose_ = msg
        self.cpose_ = msg.position
        self.cquat_ = msg.orientation
        pp = geodesy.utm.fromMsg(self.cpose_)
        tnow = rospy.get_time()
                
        if self.current_waypoint <= self.offset:
            print "Arrived at first waypoint, creating fast march explorer."
            origin = copy.copy(pp)
            origin.easting -= self.X[self.offset, 0]+self.offpos[0]
            origin.northing -= self.X[self.offset, 1]+self.offpos[1]
            self.zero_utm = origin
            self.start_time = tnow
            self.frame_trigger = self.start_time + self.dt
            self.samples = np.array([[0, self.X[0,0], self.X[0,1], self.Y[0]]])
            for i in range(1, self.offset+1):
                self.samples = np.append(self.samples, [[0, self.X[i,0], self.X[i,1], self.Y[i]]], axis=0)
            self.current_waypoint = self.offset+1

        elif self.current_waypoint == self.total_waypoints:
            print "Arrived at final waypoint, saving data."
            fh = open('timed_log_trial2.p', 'wb')
            pickle.dump(self.samples, fh)
            pickle.dump(self.poses, fh)
            pickle.dump(self.targets, fh)
            fh.close()
            self.current_waypoint+=1
        elif self.current_waypoint < self.total_waypoints:
            clocalpos = self.get_local_coords(pp)
            self.samples = np.append(self.samples, [[tnow-self.start_time, clocalpos[0], clocalpos[1], self.Y[self.current_waypoint]]], axis=0)
            self.current_waypoint+=1
        else:
            print "Extra waypoint reached!"
        print "Waypoint {0} reached at t={1}, x={2}, y={3}, obs={4}".format(self.current_waypoint, *self.samples[-1,:])
            
            
    def geopose_callback(self, msg):
        ctime = rospy.get_time()
        if self.start_time>0 and ctime>self.frame_trigger:
            self.frame_trigger = self.frame_trigger+self.dt
            pp = geodesy.utm.fromMsg(msg.position)
            clocalpos = self.get_local_coords(pp)
            if self.poses.size:
                self.poses = np.append(self.poses, [[ctime-self.start_time, clocalpos[0], clocalpos[1]]],axis=0)
            else:
                self.poses = np.array([[ctime-self.start_time, clocalpos[0], clocalpos[1]]])
            #print "New pose, t={0}, x={1}, y={2}".format(*self.poses[-1,:])

    def waypoint_callback(self, msg):        
        pp = geodesy.utm.fromMsg(msg.position)
        if self.targets.size:
            clocalpos = self.get_local_coords(pp)
            self.targets = np.append(self.targets, [[rospy.get_time()-self.start_time, clocalpos[0], clocalpos[1]]],axis=0)
        else:
            # If it is the first sent waypoint, reset the origin to relative to this point
            #origin = copy.copy(pp)
            #origin.easting -= self.X[self.offset, 0]
            #origin.northing -= self.X[self.offset, 1]
            #self.zero_utm = origin
            clocalpos = self.get_local_coords(pp)
            self.targets = np.array([[rospy.get_time()-self.start_time, clocalpos[0], clocalpos[1]]])
        print "New target, t={0}, x={1}, y={2}".format(*self.targets[-1,:])

if __name__ == '__main__':
    rospy.init_node('lutra_bag_reader', anonymous=False)
    nwaypoints = rospy.get_param('/explorer_waypoints', 30)
    fh = open('lutra_fastmarchlog_2015_08_04-12_40.p', 'rb')
    X = pickle.load(fh)
    Y = pickle.load(fh)
    fh.close()
    fmex = LutraBagReader(nwaypoints, 0.5, X, Y, offset=3, offpos=[0.01,-0.3])
    rospy.spin()