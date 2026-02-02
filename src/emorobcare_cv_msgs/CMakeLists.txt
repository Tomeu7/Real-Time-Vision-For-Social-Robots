cmake_minimum_required(VERSION 3.8)
project(emorobcare_cv_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rosidl_generator_py REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
# point for gesture
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ObjectDetection.msg"
  "msg/ObjectDetections.msg"
  "msg/PointingDetection.msg"
  "msg/GestureRecognition.msg"
  "msg/GestureRecognitions.msg"
  "msg/ProcessingTime.msg"
  "msg/EngagementDetail.msg"
  "msg/MultiUserEngagementLevel.msg"
  "msg/EngagementGeometry.msg"
  DEPENDENCIES builtin_interfaces std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_export_dependencies(rosidl_generator_py)


ament_package()
