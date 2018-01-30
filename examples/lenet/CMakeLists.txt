cmake_minimum_required(VERSION 2.8)
project(LENET_TEST)

file( GLOB TEST_SRCS "lenet_test.cpp" )

SET(OPENCV_PATH "/home/hzwangjialei/commlibs/opencv-2.4.13")
#SET(OPENCV_PATH "/home/hzwangjialei/commlibs/opencv-3.1.0")
SET(OPENCV_INC "${OPENCV_PATH}/include")
SET(OPENCV_LIB "${OPENCV_PATH}/lib")

SET(NCNN_LIB "/home/hzwangjialei/project/ncnn/build/install/lib")
SET(NCNN_INC "/home/hzwangjialei/project/ncnn/build/install/include")

INCLUDE_DIRECTORIES(${OPENCV_INC} ${NCNN_INC})
LINK_DIRECTORIES(   ${OPENCV_LIB} ${NCNN_LIB})

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -std=c++11 -O2 -D__STDC_CONSTANT_MACROS" )

add_executable(lenet_test ${TEST_SRCS})
#target_link_libraries(lenet_test "-lopencv_core  -lopencv_imgproc -lopencv_imgcodecs -lglog -lncnn")
target_link_libraries(lenet_test "-lncnn -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_features2d -lopencv_legacy")
