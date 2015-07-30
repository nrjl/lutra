#!/usr/bin/env python

import time
import rospy
import geodesy.utm
from geographic_msgs.msg import GeoPose
# from geographic_msgs.msg import GeoPoint
from std_msgs.msg import String
# from geometry_msgs.msg import Quaternion

class WaypointBoxPublisher:

    def __init__(self):
        print "Creating WaypointBoxPublisher object"
        rospy.init_node('waypoint_boxer', anonymous=False)
        self.pos_sub_ = rospy.Subscriber('/crw_geopose_pub', GeoPose, self.pose_callback)
        self.wp_pub_ = rospy.Publisher('/crw_waypoint_sub', GeoPose, queue_size=10)
        self.clear_pub_ = rospy.Publisher('/clear_waypoints', String, queue_size=10)
        
    def pose_callback(self, msg):
        # print "Pose message received!"
        self.cgeopose_ = msg
        self.cpose_ = msg.position
        self.cquat_ = msg.orientation        
        
    def input_loop(self):
        out = 'a'
        print "Options: b - new box, c - clear current waypoint, a - abort, x - exit"
        while ((not rospy.is_shutdown()) and (out != 'x')):
            out = raw_input("Enter command: ")
            if (out.lower() == 'b'):
                boxer.send_box()
            elif (out.lower() == 'c'):
                self.clear_pub_.publish("clear")
            elif (out.lower() == 'a'):
                self.clear_pub_.publish("abort")
            # rospy.spin()
        print 'Exiting, goodbye!'
        
    def send_box(self):
        if hasattr(self, 'cgeopose_'):
            pp = geodesy.utm.fromMsg(self.cpose_)
            pp.northing += 20.0
            self.pub_point(pp)
            pp.easting += 20.0
            self.pub_point(pp)
            pp.northing += - 20.0
            self.pub_point(pp)
            pp.easting += - 20.0
            self.pub_point(pp)
        else:
            print "No pose found yet, not publishing"                
            
    def pub_point(self, pp):
        self.cpose_ = pp.toMsg()
        self.cgeopose_.position = self.cpose_
        self.wp_pub_.publish(self.cgeopose_) 
        time.sleep(0.2)
        
    def send_clear(self):
        self.clear_pub_.publish("clear")    

if __name__ == '__main__':
    boxer = WaypointBoxPublisher()
    try:
        boxer.input_loop()
    except KeyboardInterrupt:
        print "Interrupt encountered, exiting..."
    except:
        print "Other exception caught, exiting..."
