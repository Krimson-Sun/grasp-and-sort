#pragma once

#include <string>
#include <vector>

#include <geometry_msgs/msg/pose.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>
#include <rclcpp/rclcpp.hpp>

#include "manip_sort_pipeline/sort_demo/types.hpp"

namespace manip_sort_pipeline::sort_demo
{

bool wait_for_move_group(
  const rclcpp::Logger& logger,
  moveit::planning_interface::MoveGroupInterface& move_group,
  const std::string& group_name);

bool wait_for_current_state(
  const rclcpp::Logger& logger,
  moveit::planning_interface::MoveGroupInterface& move_group,
  const std::string& group_name);

bool wait_for_gripper_controller(
  const rclcpp::Logger& logger,
  const GripperActionClient::SharedPtr& client,
  const std::string& action_name);

bool plan_and_execute_named_target(
  const rclcpp::Logger& logger,
  moveit::planning_interface::MoveGroupInterface& move_group,
  const std::string& target_name,
  const std::string& pipeline_id = "ompl",
  const std::string& planner_id = "");

bool execute_gripper_action_target(
  const rclcpp::Logger& logger,
  const GripperActionClient::SharedPtr& client,
  bool open,
  double duration_seconds = 1.0);

bool plan_and_execute_pose_target(
  const rclcpp::Logger& logger,
  moveit::planning_interface::MoveGroupInterface& move_group,
  const geometry_msgs::msg::Pose& target_pose,
  const std::string& label,
  const std::string& pipeline_id,
  const std::string& planner_id,
  bool position_only = false,
  bool allow_approximate_ik = true,
  bool allow_position_fallback = false);

bool plan_and_execute_best_pose_target(
  const rclcpp::Logger& logger,
  moveit::planning_interface::MoveGroupInterface& move_group,
  const geometry_msgs::msg::Pose& target_pose,
  const std::string& label,
  const std::vector<PlannerCandidate>& planner_candidates,
  const TransferScoreWeights& score_weights);

bool execute_cartesian_segment(
  const rclcpp::Logger& logger,
  moveit::planning_interface::MoveGroupInterface& move_group,
  const geometry_msgs::msg::Pose& target_pose,
  const std::string& label,
  double eef_step = 0.005,
  double min_fraction = 0.98,
  bool avoid_collisions = true);

bool find_cartesian_pick_pose_with_clearance_retry(
  const rclcpp::Logger& logger,
  moveit::planning_interface::MoveGroupInterface& move_group,
  const geometry_msgs::msg::Pose& target_pose,
  const std::string& label,
  double eef_step,
  double min_fraction,
  bool avoid_collisions,
  double retry_step,
  int retry_count,
  geometry_msgs::msg::Pose& selected_pose);

bool execute_cartesian_pick_with_clearance_retry(
  const rclcpp::Logger& logger,
  moveit::planning_interface::MoveGroupInterface& move_group,
  const geometry_msgs::msg::Pose& target_pose,
  const std::string& label,
  double eef_step,
  double min_fraction,
  bool avoid_collisions,
  double retry_step,
  int retry_count);

bool execute_cartesian_waypoint_sequence(
  const rclcpp::Logger& logger,
  moveit::planning_interface::MoveGroupInterface& move_group,
  const std::vector<geometry_msgs::msg::Pose>& waypoints,
  const std::string& prefix,
  double eef_step = 0.005,
  double min_fraction = 0.98,
  bool avoid_collisions = false);

}  // namespace manip_sort_pipeline::sort_demo
