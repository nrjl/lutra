#!/usr/bin/env python

import numpy as np
import rospy
import geodesy.utm
from geographic_msgs.msg import GeoPose,GeoPoint
import geodesy.utm
import os.path

#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String

def talker(zero_utm):
    pub = rospy.Publisher('/crw_geopose_pub', GeoPose, queue_size=10)
    rospy.init_node('simple_simulator', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    pp = GeoPose()
    while not rospy.is_shutdown():
        zero_utm.northing -= 0.1
        pp.position = zero_utm.toMsg()
        pub.publish(pp)
        rate.sleep()

if __name__ == '__main__':
    pond_image=os.path.expanduser("~")+'/catkin_ws/src/ros_lutra/data/ireland_lane_pond'
    pond_locations = np.genfromtxt(pond_image+'.csv', delimiter=',')
    pond_origin = GeoPoint(pond_locations[2,0],pond_locations[2,1],pond_locations[2,2])
    zero_utm = geodesy.utm.fromMsg(pond_origin)
    try:
        talker(zero_utm)
    except rospy.ROSInterruptException:
        pass
