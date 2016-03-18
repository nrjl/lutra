cmake_minimum_required(VERSION 2.8.3)
project(ros_lutra)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  geographic_msgs
  geometry_msgs
  rospy
  std_msgs
  geodesy
  visualization_msgs
  tf
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES lutra
  CATKIN_DEPENDS geographic_msgs geometry_msgs rospy std_msgs geodesy visualization_msgs tf
#  DEPENDS system_lib
)

###########
## Build ##
###########

# include_directories(include)
include_directories( ${catkin_INCLUDE_DIRS} )
