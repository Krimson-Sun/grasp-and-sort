cmake_minimum_required(VERSION 3.8)
project(manip_sort_description)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config srdf urdf
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
