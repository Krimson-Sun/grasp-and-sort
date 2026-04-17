#include <chrono>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit/utils/moveit_error_code.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

using namespace std::chrono_literals;

namespace
{

struct DemoTarget
{
  std::string name;
  std::vector<double> joint_offsets;
  std::string gripper_command;
};

bool wait_for_move_group(
  const rclcpp::Logger& logger,
  moveit::planning_interface::MoveGroupInterface& move_group)
{
  for (int attempt = 1; attempt <= 30; ++attempt)
  {
    if (move_group.getMoveGroupClient().wait_for_action_server(1s))
    {
      RCLCPP_INFO(logger, "MoveGroup action server is ready.");
      return true;
    }

    RCLCPP_INFO(logger, "Waiting for MoveGroup action server... (%d/30)", attempt);
  }

  return false;
}

bool wait_for_current_state(
  const rclcpp::Logger& logger,
  moveit::planning_interface::MoveGroupInterface& move_group)
{
  for (int attempt = 1; attempt <= 20; ++attempt)
  {
    move_group.startStateMonitor(1.0);
    if (move_group.getCurrentState(1.0))
    {
      RCLCPP_INFO(logger, "Current robot state is available.");
      return true;
    }

    RCLCPP_INFO(logger, "Waiting for current robot state... (%d/20)", attempt);
    std::this_thread::sleep_for(1s);
  }

  return false;
}

bool plan_and_execute_named_target(
  const rclcpp::Logger& logger,
  moveit::planning_interface::MoveGroupInterface& move_group,
  const std::string& target_name)
{
  move_group.setStartStateToCurrentState();
  if (!move_group.setNamedTarget(target_name))
  {
    RCLCPP_ERROR(logger, "Named target '%s' is not available.", target_name.c_str());
    return false;
  }

  const auto move_result = move_group.move();
  if (!move_result)
  {
    RCLCPP_ERROR(
      logger,
      "Motion to named target '%s' failed: %s",
      target_name.c_str(),
      moveit::core::errorCodeToString(move_result).c_str());
    return false;
  }

  RCLCPP_INFO(logger, "Reached named target '%s'.", target_name.c_str());
  return true;
}

bool plan_and_execute_joint_target(
  const rclcpp::Logger& logger,
  moveit::planning_interface::MoveGroupInterface& move_group,
  const std::vector<double>& target_joints,
  const std::string& end_effector_link,
  const std::string& target_name)
{
  move_group.setStartStateToCurrentState();
  move_group.clearPoseTargets();

  if (!move_group.setJointValueTarget(target_joints))
  {
    RCLCPP_ERROR(
      logger,
      "MoveIt rejected joint target '%s'.",
      target_name.c_str());
    return false;
  }

  const auto move_result = move_group.move();
  move_group.clearPoseTargets();
  if (!move_result)
  {
    RCLCPP_ERROR(
      logger,
      "Motion to joint target '%s' failed: %s",
      target_name.c_str(),
      moveit::core::errorCodeToString(move_result).c_str());
    return false;
  }

  const auto current_pose = move_group.getCurrentPose(end_effector_link).pose;
  RCLCPP_INFO(
    logger,
    "Reached %s at x=%.3f y=%.3f z=%.3f.",
    target_name.c_str(),
    current_pose.position.x,
    current_pose.position.y,
    current_pose.position.z);
  return true;
}

bool wait_for_gripper_action(
  const rclcpp::Logger& logger,
  const rclcpp_action::Client<control_msgs::action::FollowJointTrajectory>::SharedPtr& client)
{
  for (int attempt = 1; attempt <= 30; ++attempt)
  {
    if (client->wait_for_action_server(1s))
    {
      RCLCPP_INFO(logger, "Gripper action server is ready.");
      return true;
    }

    RCLCPP_INFO(logger, "Waiting for gripper action server... (%d/30)", attempt);
  }

  return false;
}

bool command_gripper(
  const rclcpp::Logger& logger,
  const rclcpp_action::Client<control_msgs::action::FollowJointTrajectory>::SharedPtr& client,
  const std::vector<std::string>& joint_names,
  const std::vector<double>& joint_multipliers,
  double target_position,
  double move_duration_seconds,
  const std::string& command_name)
{
  control_msgs::action::FollowJointTrajectory::Goal goal;
  goal.trajectory.joint_names = joint_names;

  trajectory_msgs::msg::JointTrajectoryPoint point;
  point.positions.reserve(joint_names.size());
  for (size_t i = 0; i < joint_names.size(); ++i)
  {
    const double multiplier = i < joint_multipliers.size() ? joint_multipliers[i] : 1.0;
    point.positions.push_back(target_position * multiplier);
  }
  point.time_from_start.sec = static_cast<int32_t>(move_duration_seconds);
  point.time_from_start.nanosec =
    static_cast<uint32_t>((move_duration_seconds - point.time_from_start.sec) * 1e9);
  goal.trajectory.points.push_back(point);

  auto goal_future = client->async_send_goal(goal);
  if (goal_future.wait_for(5s) != std::future_status::ready)
  {
    RCLCPP_ERROR(logger, "Timed out while sending gripper command '%s'.", command_name.c_str());
    return false;
  }

  auto goal_handle = goal_future.get();
  if (!goal_handle)
  {
    RCLCPP_ERROR(logger, "Gripper command '%s' was rejected.", command_name.c_str());
    return false;
  }

  auto result_future = client->async_get_result(goal_handle);
  const auto timeout = std::chrono::duration<double>(move_duration_seconds + 5.0);
  if (result_future.wait_for(timeout) != std::future_status::ready)
  {
    RCLCPP_ERROR(logger, "Timed out while executing gripper command '%s'.", command_name.c_str());
    return false;
  }

  const auto wrapped_result = result_future.get();
  if (wrapped_result.code != rclcpp_action::ResultCode::SUCCEEDED)
  {
    RCLCPP_ERROR(logger, "Gripper command '%s' failed with result code %d.", command_name.c_str(), static_cast<int>(wrapped_result.code));
    return false;
  }

  RCLCPP_INFO(logger, "Executed gripper command '%s' to %.3f.", command_name.c_str(), target_position);
  return true;
}

}  // namespace

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<rclcpp::Node>("five_point_moveit_demo");
  auto logger = node->get_logger();

  const std::string planning_group =
    node->declare_parameter<std::string>("planning_group", "ur_manipulator");
  const std::string end_effector_link =
    node->declare_parameter<std::string>("end_effector_link", "gripper_tcp");
  const std::string pose_reference_frame =
    node->declare_parameter<std::string>("pose_reference_frame", "base_link");
  const double planning_time =
    node->declare_parameter<double>("planning_time", 10.0);
  const int planning_attempts =
    node->declare_parameter<int>("planning_attempts", 10);
  const double velocity_scaling =
    node->declare_parameter<double>("velocity_scaling", 0.2);
  const double acceleration_scaling =
    node->declare_parameter<double>("acceleration_scaling", 0.2);
  const double goal_position_tolerance =
    node->declare_parameter<double>("goal_position_tolerance", 0.01);
  const double goal_orientation_tolerance =
    node->declare_parameter<double>("goal_orientation_tolerance", 0.05);
  const double dwell_seconds =
    node->declare_parameter<double>("dwell_seconds", 1.0);
  const double gripper_move_duration =
    node->declare_parameter<double>("gripper_move_duration", 2.0);
  const std::string gripper_action_name =
    node->declare_parameter<std::string>("gripper_action_name", "/gripper_controller/follow_joint_trajectory");
  const double gripper_open_position =
    node->declare_parameter<double>("gripper_open_position", 0.7929);
  const double gripper_closed_position =
    node->declare_parameter<double>("gripper_closed_position", 0.0);
  const std::vector<std::string> gripper_joint_names =
    node->declare_parameter<std::vector<std::string>>(
      "gripper_joint_names",
      {
        "robotiq_85_left_knuckle_joint",
        "robotiq_85_right_knuckle_joint",
        "robotiq_85_left_inner_knuckle_joint",
        "robotiq_85_right_inner_knuckle_joint",
        "robotiq_85_left_finger_tip_joint",
        "robotiq_85_right_finger_tip_joint"});
  const std::vector<double> gripper_joint_multipliers =
    node->declare_parameter<std::vector<double>>(
      "gripper_joint_multipliers",
      {1.0, 1.0, 1.0, 1.0, -1.0, -1.0});

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  std::thread spinner([&executor]() { executor.spin(); });

  auto gripper_action_client =
    rclcpp_action::create_client<control_msgs::action::FollowJointTrajectory>(
      node, gripper_action_name);

  moveit::planning_interface::MoveGroupInterface move_group(node, planning_group);
  move_group.setPoseReferenceFrame(pose_reference_frame);
  if (!move_group.setEndEffectorLink(end_effector_link))
  {
    RCLCPP_WARN(
      logger,
      "MoveIt did not confirm end-effector link '%s'. The demo will still try to use it explicitly.",
      end_effector_link.c_str());
  }
  move_group.setPlanningTime(planning_time);
  move_group.setNumPlanningAttempts(static_cast<unsigned int>(planning_attempts));
  move_group.setMaxVelocityScalingFactor(velocity_scaling);
  move_group.setMaxAccelerationScalingFactor(acceleration_scaling);
  move_group.setGoalPositionTolerance(goal_position_tolerance);
  move_group.setGoalOrientationTolerance(goal_orientation_tolerance);

  RCLCPP_INFO(
    logger,
    "Starting five-point MoveIt demo for group '%s' using link '%s' in frame '%s'.",
    planning_group.c_str(),
    end_effector_link.c_str(),
    pose_reference_frame.c_str());

  bool success = false;
  do
  {
    if (!wait_for_move_group(logger, move_group))
    {
      RCLCPP_ERROR(logger, "MoveGroup action server did not become ready in time.");
      break;
    }

    if (!wait_for_current_state(logger, move_group))
    {
      RCLCPP_ERROR(logger, "Robot state monitor did not become ready in time.");
      break;
    }

    if (!wait_for_gripper_action(logger, gripper_action_client))
    {
      RCLCPP_ERROR(logger, "Gripper action server did not become ready in time.");
      break;
    }

    if (!plan_and_execute_named_target(logger, move_group, "up"))
    {
      break;
    }

    std::this_thread::sleep_for(std::chrono::duration<double>(dwell_seconds));

    const std::vector<double> anchor_joints = move_group.getCurrentJointValues();

    const std::vector<DemoTarget> targets = {
      {"point_1", {0.10, 0.10, 0.20, -0.05, 0.05, 0.00}, "open"},
      {"point_2", {0.20, 0.18, 0.35, -0.15, 0.08, 0.15}, "close"},
      {"point_3", {-0.15, 0.22, 0.30, -0.10, -0.08, -0.10}, "open"},
      {"point_4", {0.25, 0.30, 0.50, -0.25, 0.18, 0.25}, "close"},
      {"point_5", {0.00, 0.38, 0.60, -0.35, 0.00, 0.00}, "open"},
    };

    bool completed_all_targets = true;
    for (const auto& target : targets)
    {
      if (target.joint_offsets.size() != anchor_joints.size())
      {
        RCLCPP_ERROR(
          logger,
          "Joint target '%s' has %zu values, expected %zu.",
          target.name.c_str(),
          target.joint_offsets.size(),
          anchor_joints.size());
        completed_all_targets = false;
        break;
      }

      std::vector<double> target_joints = anchor_joints;
      for (std::size_t i = 0; i < target_joints.size(); ++i)
      {
        target_joints[i] += target.joint_offsets[i];
      }

      if (!plan_and_execute_joint_target(
            logger, move_group, target_joints, end_effector_link, target.name))
      {
        completed_all_targets = false;
        break;
      }

      std::this_thread::sleep_for(std::chrono::duration<double>(dwell_seconds));
      const double gripper_target =
        target.gripper_command == "close" ? gripper_closed_position : gripper_open_position;
    if (!command_gripper(
          logger,
          gripper_action_client,
          gripper_joint_names,
          gripper_joint_multipliers,
          gripper_target,
          gripper_move_duration,
          target.gripper_command))
      {
        completed_all_targets = false;
        break;
      }
    }
    success = completed_all_targets;
  } while (false);

  move_group.stop();
  executor.cancel();
  if (spinner.joinable())
  {
    spinner.join();
  }
  rclcpp::shutdown();

  return success ? 0 : 1;
}
