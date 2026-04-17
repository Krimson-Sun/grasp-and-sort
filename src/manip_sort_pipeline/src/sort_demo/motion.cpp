#include "manip_sort_pipeline/sort_demo/motion.hpp"

#include <chrono>
#include <map>
#include <string>
#include <thread>

#include <builtin_interfaces/msg/duration.hpp>
#include <moveit/utils/moveit_error_code.hpp>
#include <moveit_msgs/msg/move_it_error_codes.hpp>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>

using namespace std::chrono_literals;

namespace manip_sort_pipeline::sort_demo
{
namespace
{

std::map<std::string, double> get_gripper_joint_targets(bool open)
{
  if (open)
  {
    return {
      {"robotiq_85_left_knuckle_joint", 0.0},
      {"robotiq_85_right_knuckle_joint", 0.0},
      {"robotiq_85_left_inner_knuckle_joint", 0.0},
      {"robotiq_85_right_inner_knuckle_joint", 0.0},
      {"robotiq_85_left_finger_tip_joint", 0.0},
      {"robotiq_85_right_finger_tip_joint", 0.0},
    };
  }

  return {
    {"robotiq_85_left_knuckle_joint", 0.7929},
    {"robotiq_85_right_knuckle_joint", 0.7929},
    {"robotiq_85_left_inner_knuckle_joint", 0.7929},
    {"robotiq_85_right_inner_knuckle_joint", 0.7929},
    {"robotiq_85_left_finger_tip_joint", -0.7929},
    {"robotiq_85_right_finger_tip_joint", -0.7929},
  };
}

std::vector<std::string> get_gripper_joint_names()
{
  return {
    "robotiq_85_left_knuckle_joint",
    "robotiq_85_right_knuckle_joint",
    "robotiq_85_left_inner_knuckle_joint",
    "robotiq_85_right_inner_knuckle_joint",
    "robotiq_85_left_finger_tip_joint",
    "robotiq_85_right_finger_tip_joint",
  };
}

}  // namespace

bool wait_for_move_group(
  const rclcpp::Logger& logger,
  moveit::planning_interface::MoveGroupInterface& move_group,
  const std::string& group_name)
{
  for (int attempt = 1; attempt <= 30; ++attempt)
  {
    if (move_group.getMoveGroupClient().wait_for_action_server(1s))
    {
      RCLCPP_INFO(logger, "MoveGroup action server is ready for '%s'.", group_name.c_str());
      return true;
    }

    RCLCPP_INFO(
      logger, "Waiting for MoveGroup action server for '%s'... (%d/30)", group_name.c_str(),
      attempt);
  }

  return false;
}

bool wait_for_current_state(
  const rclcpp::Logger& logger,
  moveit::planning_interface::MoveGroupInterface& move_group,
  const std::string& group_name)
{
  for (int attempt = 1; attempt <= 20; ++attempt)
  {
    move_group.startStateMonitor(1.0);
    if (move_group.getCurrentState(1.0))
    {
      RCLCPP_INFO(logger, "Current robot state is available for '%s'.", group_name.c_str());
      return true;
    }

    RCLCPP_INFO(logger, "Waiting for current state for '%s'... (%d/20)", group_name.c_str(), attempt);
    std::this_thread::sleep_for(1s);
  }

  return false;
}

bool wait_for_gripper_controller(
  const rclcpp::Logger& logger,
  const GripperActionClient::SharedPtr& client,
  const std::string& action_name)
{
  constexpr int max_attempts = 90;
  for (int attempt = 1; attempt <= max_attempts; ++attempt)
  {
    if (client->wait_for_action_server(1s))
    {
      RCLCPP_INFO(logger, "Gripper action server '%s' is ready.", action_name.c_str());
      return true;
    }

    RCLCPP_INFO(
      logger, "Waiting for gripper action server '%s'... (%d/%d)", action_name.c_str(), attempt,
      max_attempts);
  }

  return false;
}

bool plan_and_execute_named_target(
  const rclcpp::Logger& logger,
  moveit::planning_interface::MoveGroupInterface& move_group,
  const std::string& target_name,
  const std::string& pipeline_id,
  const std::string& planner_id)
{
  move_group.setStartStateToCurrentState();
  move_group.setPlanningPipelineId(pipeline_id);
  move_group.setPlannerId(planner_id);

  if (!move_group.setNamedTarget(target_name))
  {
    RCLCPP_ERROR(logger, "Named target '%s' is not available.", target_name.c_str());
    return false;
  }

  moveit::planning_interface::MoveGroupInterface::Plan plan;
  const auto plan_result = move_group.plan(plan);
  if (!plan_result)
  {
    RCLCPP_ERROR(
      logger, "Planning to named target '%s' failed: %s", target_name.c_str(),
      moveit::core::errorCodeToString(plan_result).c_str());
    return false;
  }

  const auto exec_result = move_group.execute(plan);
  if (!exec_result)
  {
    RCLCPP_ERROR(
      logger, "Execution of named target '%s' failed: %s", target_name.c_str(),
      moveit::core::errorCodeToString(exec_result).c_str());
    return false;
  }

  RCLCPP_INFO(logger, "Reached named target '%s'.", target_name.c_str());
  return true;
}

bool execute_gripper_action_target(
  const rclcpp::Logger& logger,
  const GripperActionClient::SharedPtr& client,
  bool open,
  double duration_seconds)
{
  const auto target_values = get_gripper_joint_targets(open);
  const auto label = open ? "open" : "closed";
  const auto joint_names = get_gripper_joint_names();

  control_msgs::action::FollowJointTrajectory::Goal goal;
  goal.trajectory.joint_names = joint_names;

  trajectory_msgs::msg::JointTrajectoryPoint point;
  point.positions.reserve(joint_names.size());
  for (const auto& joint_name : joint_names)
  {
    point.positions.push_back(target_values.at(joint_name));
  }

  builtin_interfaces::msg::Duration time_from_start;
  time_from_start.sec = static_cast<int32_t>(duration_seconds);
  time_from_start.nanosec = static_cast<uint32_t>((duration_seconds - time_from_start.sec) * 1e9);
  point.time_from_start = time_from_start;
  goal.trajectory.points.push_back(point);

  auto goal_handle_future = client->async_send_goal(goal);
  if (goal_handle_future.wait_for(5s) != std::future_status::ready)
  {
    RCLCPP_ERROR(logger, "Timed out sending explicit gripper target '%s'.", label);
    return false;
  }

  auto goal_handle = goal_handle_future.get();
  if (!goal_handle)
  {
    RCLCPP_ERROR(logger, "Gripper controller rejected target '%s'.", label);
    return false;
  }

  auto result_future = client->async_get_result(goal_handle);
  if (result_future.wait_for(std::chrono::duration<double>(duration_seconds + 5.0)) !=
      std::future_status::ready)
  {
    RCLCPP_ERROR(logger, "Timed out waiting for explicit gripper target '%s'.", label);
    return false;
  }

  const auto wrapped_result = result_future.get();
  if (wrapped_result.code != rclcpp_action::ResultCode::SUCCEEDED)
  {
    const auto error_string =
      wrapped_result.result ? wrapped_result.result->error_string : std::string("unknown");

    const bool is_grasp_contact_abort =
      !open && wrapped_result.code == rclcpp_action::ResultCode::ABORTED &&
      (error_string.find("tolerance violation") != std::string::npos ||
       error_string.find("Tolerance violation") != std::string::npos);

    if (is_grasp_contact_abort)
    {
      RCLCPP_WARN(
        logger,
        "Explicit gripper target '%s' stopped on contact before full closure: %s. "
        "Treating this as a successful grasp.",
        label, error_string.c_str());
    }
    else
    {
      RCLCPP_ERROR(
        logger, "Explicit gripper target '%s' failed with action result %d: %s", label,
        static_cast<int>(wrapped_result.code), error_string.c_str());
      return false;
    }
  }

  std::string state_summary;
  for (std::size_t i = 0; i < joint_names.size() && i < point.positions.size(); ++i)
  {
    if (!state_summary.empty())
    {
      state_summary += ", ";
    }
    state_summary += joint_names[i] + "=" + std::to_string(point.positions[i]);
  }

  RCLCPP_INFO(
    logger, "Reached explicit gripper target '%s'. Commanded joints: %s", label,
    state_summary.c_str());
  return true;
}

bool plan_and_execute_pose_target(
  const rclcpp::Logger& logger,
  moveit::planning_interface::MoveGroupInterface& move_group,
  const geometry_msgs::msg::Pose& target_pose,
  const std::string& label,
  const std::string& pipeline_id,
  const std::string& planner_id,
  bool position_only,
  bool allow_approximate_ik,
  bool allow_position_fallback)
{
  move_group.setStartStateToCurrentState();
  move_group.clearPoseTargets();
  move_group.setPlanningPipelineId(pipeline_id);
  move_group.setPlannerId(planner_id);

  auto set_target = [&](bool use_position_only, bool use_approximate_ik) -> bool {
    move_group.clearPoseTargets();

    if (use_position_only)
    {
      return move_group.setPositionTarget(
        target_pose.position.x, target_pose.position.y, target_pose.position.z,
        move_group.getEndEffectorLink());
    }

    if (use_approximate_ik)
    {
      return move_group.setApproximateJointValueTarget(
        target_pose, move_group.getEndEffectorLink());
    }

    return move_group.setPoseTarget(target_pose);
  };

  struct PlanningAttempt
  {
    bool use_position_only;
    bool use_approximate_ik;
    const char* description;
  };

  std::vector<PlanningAttempt> attempts;
  attempts.push_back({position_only, false, position_only ? "position target" : "pose target"});
  if (!position_only && allow_approximate_ik)
  {
    attempts.push_back({false, true, "approximate IK target"});
  }
  if (!position_only && allow_position_fallback)
  {
    attempts.push_back({true, false, "position-only fallback"});
  }

  RCLCPP_INFO(
    logger, "Planning to '%s' using %s.", label.c_str(),
    position_only ? "position target" : "pose target");

  moveit::planning_interface::MoveGroupInterface::Plan plan;
  moveit::core::MoveItErrorCode plan_result =
    moveit::core::MoveItErrorCode(moveit_msgs::msg::MoveItErrorCodes::FAILURE);
  bool planned = false;

  for (const auto& attempt : attempts)
  {
    if (!set_target(attempt.use_position_only, attempt.use_approximate_ik))
    {
      RCLCPP_WARN(
        logger, "Unable to set %s for '%s'. Trying next strategy.", attempt.description,
        label.c_str());
      continue;
    }

    if (std::string(attempt.description) != "pose target" || position_only)
    {
      RCLCPP_INFO(logger, "Retrying '%s' using %s.", label.c_str(), attempt.description);
    }

    plan_result = move_group.plan(plan);
    if (plan_result)
    {
      planned = true;
      break;
    }

    RCLCPP_WARN(
      logger, "Planning to '%s' with %s failed: %s", label.c_str(), attempt.description,
      moveit::core::errorCodeToString(plan_result).c_str());
  }

  move_group.clearPoseTargets();

  if (!planned)
  {
    RCLCPP_ERROR(
      logger, "Planning to '%s' failed after all strategies: %s", label.c_str(),
      moveit::core::errorCodeToString(plan_result).c_str());
    return false;
  }

  const auto exec_result = move_group.execute(plan);
  if (!exec_result)
  {
    RCLCPP_ERROR(
      logger, "Execution to '%s' failed: %s", label.c_str(),
      moveit::core::errorCodeToString(exec_result).c_str());
    return false;
  }

  RCLCPP_INFO(
    logger, "Reached '%s' at x=%.3f y=%.3f z=%.3f.", label.c_str(), target_pose.position.x,
    target_pose.position.y, target_pose.position.z);
  return true;
}

bool execute_cartesian_segment(
  const rclcpp::Logger& logger,
  moveit::planning_interface::MoveGroupInterface& move_group,
  const geometry_msgs::msg::Pose& target_pose,
  const std::string& label,
  double eef_step,
  double min_fraction,
  bool avoid_collisions)
{
  move_group.setStartStateToCurrentState();
  const auto current_pose = move_group.getCurrentPose(move_group.getEndEffectorLink()).pose;
  RCLCPP_INFO(
    logger,
    "Cartesian segment '%s' for link '%s': current pos=(%.3f, %.3f, %.3f) target pos=(%.3f, %.3f, %.3f) "
    "current quat=(%.3f, %.3f, %.3f, %.3f) target quat=(%.3f, %.3f, %.3f, %.3f)",
    label.c_str(), move_group.getEndEffectorLink().c_str(), current_pose.position.x, current_pose.position.y, current_pose.position.z,
    target_pose.position.x, target_pose.position.y, target_pose.position.z,
    current_pose.orientation.x, current_pose.orientation.y, current_pose.orientation.z,
    current_pose.orientation.w, target_pose.orientation.x, target_pose.orientation.y,
    target_pose.orientation.z, target_pose.orientation.w);

  std::vector<geometry_msgs::msg::Pose> waypoints;
  waypoints.push_back(target_pose);

  moveit_msgs::msg::RobotTrajectory trajectory;
  moveit_msgs::msg::MoveItErrorCodes error_code;
  const double fraction =
    move_group.computeCartesianPath(waypoints, eef_step, trajectory, avoid_collisions, &error_code);

  if (fraction < min_fraction)
  {
    RCLCPP_ERROR(
      logger, "Cartesian path to '%s' achieved only %.3f of the requested path.", label.c_str(),
      fraction);
    return false;
  }

  const auto exec_result = move_group.execute(trajectory);
  if (!exec_result)
  {
    RCLCPP_ERROR(
      logger, "Execution of cartesian segment '%s' failed: %s", label.c_str(),
      moveit::core::errorCodeToString(exec_result).c_str());
    return false;
  }

  RCLCPP_INFO(
    logger, "Reached cartesian target '%s' at x=%.3f y=%.3f z=%.3f.", label.c_str(),
    target_pose.position.x, target_pose.position.y, target_pose.position.z);
  return true;
}

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
  geometry_msgs::msg::Pose& selected_pose)
{
  auto attempt_pose = target_pose;
  for (int attempt = 0; attempt <= retry_count; ++attempt)
  {
    if (attempt > 0)
    {
      attempt_pose.position.z = target_pose.position.z + retry_step * static_cast<double>(attempt);
      RCLCPP_WARN(
        logger,
        "Cartesian pick '%s' is retrying %.3f m above the nominal grasp height (attempt %d/%d).",
        label.c_str(), attempt_pose.position.z - target_pose.position.z, attempt, retry_count);
    }

    move_group.setStartStateToCurrentState();
    std::vector<geometry_msgs::msg::Pose> waypoints;
    waypoints.push_back(attempt_pose);

    moveit_msgs::msg::RobotTrajectory trajectory;
    moveit_msgs::msg::MoveItErrorCodes error_code;
    const double fraction = move_group.computeCartesianPath(
      waypoints, eef_step, trajectory, avoid_collisions, &error_code);

    if (fraction >= min_fraction)
    {
      selected_pose = attempt_pose;
      if (attempt > 0)
      {
        RCLCPP_WARN(
          logger,
          "Cartesian pick '%s' succeeded only after raising the grasp height by %.3f m.",
          label.c_str(), attempt_pose.position.z - target_pose.position.z);
      }
      return true;
    }
  }

  RCLCPP_ERROR(
    logger,
    "Cartesian pick '%s' failed even after raising the target by up to %.3f m.",
    label.c_str(), retry_step * static_cast<double>(retry_count));
  return false;
}

bool execute_cartesian_pick_with_clearance_retry(
  const rclcpp::Logger& logger,
  moveit::planning_interface::MoveGroupInterface& move_group,
  const geometry_msgs::msg::Pose& target_pose,
  const std::string& label,
  double eef_step,
  double min_fraction,
  bool avoid_collisions,
  double retry_step,
  int retry_count)
{
  geometry_msgs::msg::Pose selected_pose;
  if (!find_cartesian_pick_pose_with_clearance_retry(
        logger, move_group, target_pose, label, eef_step, min_fraction, avoid_collisions,
        retry_step, retry_count, selected_pose))
  {
    return false;
  }

  return execute_cartesian_segment(
    logger, move_group, selected_pose, label, eef_step, min_fraction, avoid_collisions);
}

bool execute_cartesian_waypoint_sequence(
  const rclcpp::Logger& logger,
  moveit::planning_interface::MoveGroupInterface& move_group,
  const std::vector<geometry_msgs::msg::Pose>& waypoints,
  const std::string& prefix,
  double eef_step,
  double min_fraction,
  bool avoid_collisions)
{
  for (std::size_t i = 0; i < waypoints.size(); ++i)
  {
    if (!execute_cartesian_segment(
          logger, move_group, waypoints[i], prefix + "_" + std::to_string(i + 1), eef_step,
          min_fraction, avoid_collisions))
    {
      return false;
    }
  }

  return true;
}

}  // namespace manip_sort_pipeline::sort_demo
