#pragma once

#include <string>
#include <vector>

#include <manip_sort_interfaces/msg/grasp_candidate.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>
#include <rclcpp/rclcpp.hpp>

#include "manip_sort_pipeline/vision/types.hpp"

namespace manip_sort_pipeline::vision
{

bool execute_vision_sort_task(
  const rclcpp::Logger& logger,
  moveit::planning_interface::MoveGroupInterface& arm_group,
  const sort_demo::GripperActionClient::SharedPtr& gripper_action_client,
  sort_demo::GazeboDetachInterface& gazebo_detach_interface,
  moveit::planning_interface::PlanningSceneInterface& planning_scene_interface,
  const rclcpp::Client<sort_demo::GetPlanningScene>::SharedPtr& get_planning_scene_client,
  const VisionExecutionTask& task,
  const std::vector<std::string>& touch_links,
  const VisionManagerConfig& config,
  std::string& failure_reason);

}  // namespace manip_sort_pipeline::vision
