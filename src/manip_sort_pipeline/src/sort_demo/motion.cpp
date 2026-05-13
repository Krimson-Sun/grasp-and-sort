#include "manip_sort_pipeline/sort_demo/motion.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <builtin_interfaces/msg/duration.hpp>
#include <moveit/robot_state/robot_state.hpp>
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

double pose_position_error(
  const geometry_msgs::msg::Pose& left,
  const geometry_msgs::msg::Pose& right)
{
  const auto dx = left.position.x - right.position.x;
  const auto dy = left.position.y - right.position.y;
  const auto dz = left.position.z - right.position.z;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double duration_to_seconds(const builtin_interfaces::msg::Duration& duration)
{
  return static_cast<double>(duration.sec) + static_cast<double>(duration.nanosec) * 1e-9;
}

struct TransferPlanScore
{
  double total = std::numeric_limits<double>::infinity();
  double tcp_path_length = 0.0;
  double straight_distance = 0.0;
  double detour_ratio = std::numeric_limits<double>::infinity();
  double joint_path_length = 0.0;
  double duration = 0.0;
  double z_overshoot = 0.0;
};

struct ScoredPlan
{
  moveit::planning_interface::MoveGroupInterface::Plan plan;
  PlannerCandidate planner;
  int attempt = 1;
  TransferPlanScore score;
};

std::string planner_label(const PlannerCandidate& planner)
{
  return planner.pipeline_id + "/" + (planner.planner_id.empty() ? "default" : planner.planner_id);
}

std::optional<geometry_msgs::msg::Point> tcp_point_for_trajectory_point(
  const moveit::core::RobotState& reference_state,
  const trajectory_msgs::msg::JointTrajectory& trajectory,
  const trajectory_msgs::msg::JointTrajectoryPoint& point,
  const std::string& tip_link)
{
  if (point.positions.size() < trajectory.joint_names.size())
  {
    return std::nullopt;
  }

  moveit::core::RobotState state(reference_state);
  try
  {
    state.setVariablePositions(trajectory.joint_names, point.positions);
  }
  catch (const std::exception&)
  {
    return std::nullopt;
  }
  state.update();

  if (!state.getRobotModel()->hasLinkModel(tip_link))
  {
    return std::nullopt;
  }

  const auto& transform = state.getGlobalLinkTransform(tip_link);
  geometry_msgs::msg::Point result;
  result.x = transform.translation().x();
  result.y = transform.translation().y();
  result.z = transform.translation().z();
  return result;
}

std::optional<TransferPlanScore> score_transfer_plan(
  const moveit::planning_interface::MoveGroupInterface::Plan& plan,
  const moveit::core::RobotState& start_state,
  const geometry_msgs::msg::Pose& target_pose,
  const std::string& tip_link,
  const TransferScoreWeights& weights)
{
  const auto& trajectory = plan.trajectory.joint_trajectory;
  if (trajectory.points.empty() || trajectory.joint_names.empty())
  {
    return std::nullopt;
  }

  TransferPlanScore score;
  if (!start_state.getRobotModel()->hasLinkModel(tip_link))
  {
    return std::nullopt;
  }

  const auto& start_transform = start_state.getGlobalLinkTransform(tip_link);
  geometry_msgs::msg::Point start_tcp;
  start_tcp.x = start_transform.translation().x();
  start_tcp.y = start_transform.translation().y();
  start_tcp.z = start_transform.translation().z();

  std::optional<geometry_msgs::msg::Point> previous_tcp = start_tcp;
  double max_z = start_tcp.z;

  for (const auto& point : trajectory.points)
  {
    const auto current_tcp =
      tcp_point_for_trajectory_point(start_state, trajectory, point, tip_link);
    if (!current_tcp)
    {
      return std::nullopt;
    }

    if (previous_tcp)
    {
      const auto dx = current_tcp->x - previous_tcp->x;
      const auto dy = current_tcp->y - previous_tcp->y;
      const auto dz = current_tcp->z - previous_tcp->z;
      score.tcp_path_length += std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    max_z = std::max(max_z, current_tcp->z);
    previous_tcp = current_tcp;
  }

  std::vector<double> previous_joint_positions;
  previous_joint_positions.reserve(trajectory.joint_names.size());
  try
  {
    for (const auto& joint_name : trajectory.joint_names)
    {
      previous_joint_positions.push_back(start_state.getVariablePosition(joint_name));
    }
  }
  catch (const std::exception&)
  {
    return std::nullopt;
  }

  for (const auto& current : trajectory.points)
  {
    const auto joint_count = std::min(previous_joint_positions.size(), current.positions.size());
    for (std::size_t joint_index = 0; joint_index < joint_count; ++joint_index)
    {
      score.joint_path_length +=
        std::abs(current.positions[joint_index] - previous_joint_positions[joint_index]);
      previous_joint_positions[joint_index] = current.positions[joint_index];
    }
  }

  if (!trajectory.points.empty())
  {
    score.duration = duration_to_seconds(trajectory.points.back().time_from_start);
  }

  const auto straight_dx = target_pose.position.x - start_tcp.x;
  const auto straight_dy = target_pose.position.y - start_tcp.y;
  const auto straight_dz = target_pose.position.z - start_tcp.z;
  score.straight_distance =
    std::sqrt(straight_dx * straight_dx + straight_dy * straight_dy + straight_dz * straight_dz);
  score.detour_ratio =
    score.straight_distance > 1e-6 ? score.tcp_path_length / score.straight_distance : 1.0;
  score.z_overshoot = std::max(0.0, max_z - std::max(start_tcp.z, target_pose.position.z));
  score.total = weights.detour_ratio * score.detour_ratio +
                weights.joint_path_length * score.joint_path_length +
                weights.duration * score.duration +
                weights.z_overshoot * score.z_overshoot;
  return score;
}

std::string transfer_plan_rejection_reason(
  const TransferPlanScore& score,
  const TransferScoreWeights& weights)
{
  if (weights.max_detour_ratio > 0.0 && score.detour_ratio > weights.max_detour_ratio)
  {
    return "detour_ratio";
  }
  if (weights.max_z_overshoot > 0.0 && score.z_overshoot > weights.max_z_overshoot)
  {
    return "z_overshoot";
  }
  if (weights.max_duration > 0.0 && score.duration > weights.max_duration)
  {
    return "duration";
  }
  return "";
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

    const bool is_tolerance_abort =
      wrapped_result.code == rclcpp_action::ResultCode::ABORTED &&
      (error_string.find("tolerance violation") != std::string::npos ||
       error_string.find("Tolerance violation") != std::string::npos);
    const bool is_grasp_contact_abort = !open && is_tolerance_abort;
    const bool is_open_tolerance_abort = open && is_tolerance_abort;

    if (is_grasp_contact_abort)
    {
      RCLCPP_WARN(
        logger,
        "Explicit gripper target '%s' stopped on contact before full closure: %s. "
        "Treating this as a successful grasp.",
        label, error_string.c_str());
    }
    else if (is_open_tolerance_abort)
    {
      RCLCPP_WARN(
        logger,
        "Explicit gripper target '%s' stopped near the open target: %s. "
        "Treating this as successful enough to continue after release.",
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

    return move_group.setPoseTarget(target_pose, move_group.getEndEffectorLink());
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
    logger,
    "Planning to '%s' using %s. target pos=(%.3f, %.3f, %.3f) target quat=(%.3f, %.3f, %.3f, %.3f)",
    label.c_str(), position_only ? "position target" : "pose target",
    target_pose.position.x, target_pose.position.y, target_pose.position.z,
    target_pose.orientation.x, target_pose.orientation.y,
    target_pose.orientation.z, target_pose.orientation.w);

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
      if (attempt.use_position_only)
      {
        RCLCPP_WARN(
          logger,
          "Planning to '%s' succeeded only with position-only fallback. Orientation was not enforced.",
          label.c_str());
      }
      else
      {
        RCLCPP_INFO(
          logger, "Planning to '%s' succeeded with %s.", label.c_str(), attempt.description);
      }
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

  move_group.getCurrentState(1.0);
  const auto actual_pose = move_group.getCurrentPose(move_group.getEndEffectorLink()).pose;
  const auto position_error = pose_position_error(actual_pose, target_pose);
  constexpr double max_pose_target_position_error = 0.04;
  if (position_error > max_pose_target_position_error)
  {
    RCLCPP_ERROR(
      logger,
      "Execution to '%s' reported success, but TCP '%s' is %.3f m from target. "
      "actual=(%.3f, %.3f, %.3f), target=(%.3f, %.3f, %.3f).",
      label.c_str(), move_group.getEndEffectorLink().c_str(), position_error,
      actual_pose.position.x, actual_pose.position.y, actual_pose.position.z,
      target_pose.position.x, target_pose.position.y, target_pose.position.z);
    return false;
  }

  RCLCPP_INFO(
    logger, "Reached '%s' at x=%.3f y=%.3f z=%.3f.", label.c_str(), actual_pose.position.x,
    actual_pose.position.y, actual_pose.position.z);
  return true;
}

bool plan_and_execute_best_pose_target(
  const rclcpp::Logger& logger,
  moveit::planning_interface::MoveGroupInterface& move_group,
  const geometry_msgs::msg::Pose& target_pose,
  const std::string& label,
  const std::vector<PlannerCandidate>& planner_candidates,
  const TransferScoreWeights& score_weights)
{
  if (planner_candidates.empty())
  {
    RCLCPP_ERROR(logger, "No planner candidates configured for best-plan target '%s'.", label.c_str());
    return false;
  }

  move_group.clearPoseTargets();
  const auto fixed_start_state = move_group.getCurrentState(1.0);
  if (!fixed_start_state)
  {
    RCLCPP_ERROR(logger, "Unable to read current state before planning '%s'.", label.c_str());
    return false;
  }

  RCLCPP_INFO(
    logger,
    "Planning best pose target '%s' with %zu planner candidates. target pos=(%.3f, %.3f, %.3f) "
    "target quat=(%.3f, %.3f, %.3f, %.3f)",
    label.c_str(), planner_candidates.size(), target_pose.position.x, target_pose.position.y,
    target_pose.position.z, target_pose.orientation.x, target_pose.orientation.y,
    target_pose.orientation.z, target_pose.orientation.w);

  std::optional<ScoredPlan> best_plan;
  std::size_t successful_candidates = 0;
  std::size_t rejected_candidates = 0;

  for (const auto& planner : planner_candidates)
  {
    const int attempts = std::max(1, planner.attempts);
    for (int attempt = 1; attempt <= attempts; ++attempt)
    {
      move_group.setStartState(*fixed_start_state);
      move_group.clearPoseTargets();
      move_group.setPlanningPipelineId(planner.pipeline_id);
      move_group.setPlannerId(planner.planner_id);

      if (!move_group.setPoseTarget(target_pose, move_group.getEndEffectorLink()))
      {
        RCLCPP_WARN(
          logger, "Unable to set pose target for transfer candidate '%s' attempt %d/%d.",
          planner_label(planner).c_str(), attempt, attempts);
        continue;
      }

      moveit::planning_interface::MoveGroupInterface::Plan plan;
      const auto plan_result = move_group.plan(plan);
      if (!plan_result)
      {
        RCLCPP_WARN(
          logger, "Plan candidate '%s' attempt %d/%d failed to plan: %s",
          planner_label(planner).c_str(), attempt, attempts,
          moveit::core::errorCodeToString(plan_result).c_str());
        continue;
      }

      const auto score = score_transfer_plan(
        plan, *fixed_start_state, target_pose, move_group.getEndEffectorLink(), score_weights);
      if (!score)
      {
        RCLCPP_WARN(
          logger,
          "Plan candidate '%s' attempt %d/%d planned, but its trajectory could not be scored.",
          planner_label(planner).c_str(), attempt, attempts);
        continue;
      }

      const auto rejection_reason = transfer_plan_rejection_reason(*score, score_weights);
      RCLCPP_INFO(
        logger,
        "Plan candidate '%s' attempt %d/%d: score=%.3f detour=%.3f tcp_len=%.3f "
        "straight=%.3f joint_len=%.3f duration=%.3f z_over=%.3f%s%s",
        planner_label(planner).c_str(), attempt, attempts, score->total, score->detour_ratio,
        score->tcp_path_length, score->straight_distance, score->joint_path_length,
        score->duration, score->z_overshoot, rejection_reason.empty() ? "" : " rejected=",
        rejection_reason.c_str());

      if (!rejection_reason.empty())
      {
        ++rejected_candidates;
        continue;
      }

      ++successful_candidates;
      if (!best_plan || score->total < best_plan->score.total)
      {
        best_plan = ScoredPlan{plan, planner, attempt, *score};
      }
    }
  }

  move_group.clearPoseTargets();

  if (!best_plan)
  {
    RCLCPP_ERROR(
      logger,
      "No acceptable transfer plan found for '%s'. successful=%zu rejected=%zu",
      label.c_str(), successful_candidates, rejected_candidates);
    return false;
  }

  RCLCPP_INFO(
    logger,
    "Selected plan candidate '%s' attempt %d for '%s': score=%.3f detour=%.3f "
    "tcp_len=%.3f joint_len=%.3f duration=%.3f z_over=%.3f",
    planner_label(best_plan->planner).c_str(), best_plan->attempt, label.c_str(),
    best_plan->score.total, best_plan->score.detour_ratio, best_plan->score.tcp_path_length,
    best_plan->score.joint_path_length, best_plan->score.duration, best_plan->score.z_overshoot);

  const auto exec_result = move_group.execute(best_plan->plan);
  if (!exec_result)
  {
    RCLCPP_ERROR(
      logger, "Execution of selected plan '%s' failed: %s", label.c_str(),
      moveit::core::errorCodeToString(exec_result).c_str());
    return false;
  }

  move_group.getCurrentState(1.0);
  const auto actual_pose = move_group.getCurrentPose(move_group.getEndEffectorLink()).pose;
  const auto position_error = pose_position_error(actual_pose, target_pose);
  constexpr double max_pose_target_position_error = 0.04;
  if (position_error > max_pose_target_position_error)
  {
    RCLCPP_ERROR(
      logger,
      "Selected plan '%s' executed, but TCP '%s' is %.3f m from target. "
      "actual=(%.3f, %.3f, %.3f), target=(%.3f, %.3f, %.3f).",
      label.c_str(), move_group.getEndEffectorLink().c_str(), position_error,
      actual_pose.position.x, actual_pose.position.y, actual_pose.position.z,
      target_pose.position.x, target_pose.position.y, target_pose.position.z);
    return false;
  }

  RCLCPP_INFO(
    logger, "Reached best-plan target '%s' at x=%.3f y=%.3f z=%.3f.",
    label.c_str(), actual_pose.position.x, actual_pose.position.y, actual_pose.position.z);
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
