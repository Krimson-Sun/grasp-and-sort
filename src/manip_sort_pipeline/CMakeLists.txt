cmake_minimum_required(VERSION 3.8)
project(manip_sort_pipeline)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(control_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(manip_sort_interfaces REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(shape_msgs REQUIRED)
find_package(simulation_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)

add_executable(five_point_moveit_demo src/five_point_moveit_demo.cpp)
ament_target_dependencies(
  five_point_moveit_demo
  control_msgs
  geometry_msgs
  moveit_core
  moveit_msgs
  moveit_ros_planning_interface
  rclcpp
  rclcpp_action
  std_msgs
  tf2
  tf2_geometry_msgs
  trajectory_msgs
)

add_library(sort_scene_demo_lib STATIC
  src/sort_demo/gazebo.cpp
  src/sort_demo/motion.cpp
  src/sort_demo/runner.cpp
  src/sort_demo/scene.cpp
  src/sort_demo/sorting.cpp
  src/vision/config.cpp
  src/vision/executor.cpp
  src/vision/manager.cpp
)
target_include_directories(sort_scene_demo_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(
  sort_scene_demo_lib
  control_msgs
  geometry_msgs
  manip_sort_interfaces
  moveit_core
  moveit_msgs
  moveit_ros_planning_interface
  rclcpp
  rclcpp_action
  shape_msgs
  simulation_interfaces
  std_msgs
  tf2
  tf2_geometry_msgs
  trajectory_msgs
)

add_executable(sort_scene_demo src/sort_scene_demo.cpp)
target_link_libraries(sort_scene_demo sort_scene_demo_lib)

add_executable(vision_sort_manager src/vision_sort_manager.cpp)
target_link_libraries(vision_sort_manager sort_scene_demo_lib)
ament_target_dependencies(
  vision_sort_manager
  control_msgs
  geometry_msgs
  manip_sort_interfaces
  moveit_core
  moveit_msgs
  moveit_ros_planning_interface
  rclcpp
  rclcpp_action
  shape_msgs
  simulation_interfaces
  std_msgs
  tf2
  tf2_geometry_msgs
  trajectory_msgs
)

install(TARGETS
  five_point_moveit_demo
  sort_scene_demo
  vision_sort_manager
  sort_scene_demo_lib
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/ DESTINATION include)

install(
  DIRECTORY launch
            config
            worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
