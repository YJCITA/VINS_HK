cmake_minimum_required(VERSION 2.8.3)
project(feature_tracker)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    sensor_msgs
    cv_bridge
    camera_model
    )

# SET(OpenCV_DIR "/home/yj/bak/packet/opencv/opencv3.1/install/share/OpenCV")
# FIND_PACKAGE(OpenCV 3.1.0 REQUIRED)
SET(OpenCV_DIR "/home/yj/bak/packet/opencv/opencv-2.4.13/install/share/OpenCV")
FIND_PACKAGE(OpenCV 2.4.13 REQUIRED)
# find_package(OpenCV REQUIRED)


SET(Ceres_DIR "/home/yj/bak/packet/ceres_solver/install/share/Ceres")
find_package(Ceres REQUIRED)
MESSAGE(STATUS "feature_tracker: CERES_INCLUDE_DIRS: ${CERES_INCLUDE_DIRS}")
include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})

catkin_package()

include_directories(
    ${catkin_INCLUDE_DIRS}
    )

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(/home/yj/bak/packet/eigen3_new/script_eigen.cmake) # eigen new
#find_package(Eigen3  REQUIRED)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

add_executable(feature_tracker
    src/feature_tracker_node.cpp
    src/parameters.cpp
    src/feature_tracker.cpp
    )

target_link_libraries(feature_tracker ${catkin_LIBRARIES} ${OpenCV_LIBS})
