#pragma once

#include <vector>

#include <geometry_msgs/msg/pose.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>

#include "manip_sort_pipeline/sort_demo/types.hpp"

namespace manip_sort_pipeline::sort_demo
{

std::vector<SortingTask> build_sorting_tasks(
  const geometry_msgs::msg::Quaternion& orientation,
  double approach_z,
  double gripper_base_to_grasp_plane,
  double grasp_surface_clearance);

bool sort_single_cube(
  const rclcpp::Logger& logger,
  moveit::planning_interface::MoveGroupInterface& arm_group,
  const GripperActionClient::SharedPtr& gripper_action_client,
  GazeboDetachInterface& gazebo_detach_interface,
  moveit::planning_interface::PlanningSceneInterface& planning_scene_interface,
  const rclcpp::Client<GetPlanningScene>::SharedPtr& get_planning_scene_client,
  const SortingTask& task,
  const std::vector<std::string>& touch_links,
  const SortExecutionOptions& options);

}  // namespace manip_sort_pipeline::sort_demo
