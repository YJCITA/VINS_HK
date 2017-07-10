cmake_minimum_required(VERSION 2.8.3)
project(camera_model)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fPIC")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    )

find_package(Boost REQUIRED COMPONENTS filesystem program_options system)
include_directories(${Boost_INCLUDE_DIRS})

#SET(OpenCV_DIR "/home/yj/bak/packet/opencv/opencv3.1/install/share/OpenCV")
#FIND_PACKAGE(OpenCV 3.1.0 REQUIRED)
SET(OpenCV_DIR "/home/yj/bak/packet/opencv/opencv-2.4.13/install/share/OpenCV")
FIND_PACKAGE(OpenCV 2.4.13 REQUIRED)
#find_package(OpenCV REQUIRED)

# set(EIGEN_INCLUDE_DIR "/usr/local/include/eigen3")
SET(Ceres_DIR "/home/yj/bak/packet/ceres_solver/install/share/Ceres")
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})
MESSAGE(STATUS "CERES_INCLUDE_DIRS: ${CERES_INCLUDE_DIRS}")

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES camera_model
    CATKIN_DEPENDS roscpp std_msgs
#    DEPENDS system_lib
    )

include_directories(
    ${catkin_INCLUDE_DIRS}
    )

include_directories("include")


# add_executable(Calibration 
#     src/intrinsic_calib.cc
#     src/chessboard/Chessboard.cc
#     src/calib/CameraCalibration.cc
#     src/camera_models/Camera.cc
#     src/camera_models/CameraFactory.cc
#     src/camera_models/CostFunctionFactory.cc
#     src/camera_models/PinholeCamera.cc
#     src/camera_models/CataCamera.cc
#     src/camera_models/EquidistantCamera.cc
#     src/camera_models/ScaramuzzaCamera.cc
#     src/sparse_graph/Transform.cc
#     src/gpl/gpl.cc
#     src/gpl/EigenQuaternionParameterization.cc)

add_library(camera_model STATIC
    src/chessboard/Chessboard.cc
    src/calib/CameraCalibration.cc
    src/camera_models/Camera.cc
    src/camera_models/CameraFactory.cc
    src/camera_models/CostFunctionFactory.cc
    src/camera_models/PinholeCamera.cc
    src/camera_models/CataCamera.cc
    src/camera_models/EquidistantCamera.cc
    src/camera_models/ScaramuzzaCamera.cc
    src/sparse_graph/Transform.cc
    src/gpl/gpl.cc
    src/gpl/EigenQuaternionParameterization.cc)

# target_link_libraries(Calibration ${Boost_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})
target_link_libraries(camera_model ${Boost_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES})
