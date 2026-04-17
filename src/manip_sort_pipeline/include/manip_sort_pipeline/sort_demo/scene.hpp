#pragma once

#include <map>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <geometry_msgs/msg/pose.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <rclcpp/rclcpp.hpp>

#include "manip_sort_pipeline/sort_demo/types.hpp"

namespace manip_sort_pipeline::sort_demo
{

geometry_msgs::msg::Pose make_pose(
  double x,
  double y,
  double z,
  const geometry_msgs::msg::Quaternion& orientation);

geometry_msgs::msg::Quaternion make_quaternion_from_axes(
  const Eigen::Vector3d& x_axis,
  const Eigen::Vector3d& y_axis,
  const Eigen::Vector3d& z_axis);

std::vector<std::string> get_touch_links();
std::map<std::string, geometry_msgs::msg::Pose> get_initial_cube_poses();
std::vector<geometry_msgs::msg::Quaternion> get_vertical_grasp_orientations();

bool wait_for_objects(
  const rclcpp::Logger& logger,
  moveit::planning_interface::PlanningSceneInterface& planning_scene_interface,
  const std::vector<std::string>& object_ids);

bool set_grasp_object_collisions(
  const rclcpp::Logger& logger,
  moveit::planning_interface::PlanningSceneInterface& planning_scene_interface,
  const rclcpp::Client<GetPlanningScene>::SharedPtr& get_planning_scene_client,
  const std::string& object_id,
  const std::vector<std::string>& touch_links,
  bool allowed);

bool set_surface_collisions(
  const rclcpp::Logger& logger,
  moveit::planning_interface::PlanningSceneInterface& planning_scene_interface,
  const rclcpp::Client<GetPlanningScene>::SharedPtr& get_planning_scene_client,
  const std::string& surface_id,
  const std::vector<std::string>& touch_links,
  bool allowed);

bool set_collision_pair(
  const rclcpp::Logger& logger,
  moveit::planning_interface::PlanningSceneInterface& planning_scene_interface,
  const rclcpp::Client<GetPlanningScene>::SharedPtr& get_planning_scene_client,
  const std::string& first_id,
  const std::string& second_id,
  bool allowed);

std::vector<moveit_msgs::msg::CollisionObject> build_scene_objects();
std::vector<std::string> build_scene_ids();

}  // namespace manip_sort_pipeline::sort_demo
