#include "manip_sort_pipeline/sort_demo/runner.hpp"

#include <thread>
#include <vector>

#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>

#include "manip_sort_pipeline/sort_demo/gazebo.hpp"
#include "manip_sort_pipeline/sort_demo/motion.hpp"
#include "manip_sort_pipeline/sort_demo/scene.hpp"
#include "manip_sort_pipeline/sort_demo/sorting.hpp"

namespace manip_sort_pipeline::sort_demo
{
namespace
{

void configure_arm_group(
  const SortDemoConfig& config,
  moveit::planning_interface::MoveGroupInterface& arm_group,
  const rclcpp::Logger& logger)
{
  arm_group.setPoseReferenceFrame(config.pose_reference_frame);
  arm_group.setPlanningTime(config.planning_time);
  arm_group.setNumPlanningAttempts(static_cast<unsigned int>(config.planning_attempts));
  arm_group.setMaxVelocityScalingFactor(config.arm_velocity_scaling);
  arm_group.setMaxAccelerationScalingFactor(config.arm_acceleration_scaling);
  arm_group.setGoalPositionTolerance(config.goal_position_tolerance);
  arm_group.setGoalOrientationTolerance(config.goal_orientation_tolerance);
  arm_group.setWorkspace(-0.2, -1.2, 0.0, 1.4, 1.2, 1.6);

  if (!arm_group.setEndEffectorLink(config.end_effector_link))
  {
    RCLCPP_WARN(
      logger, "MoveIt did not confirm end-effector link '%s'. Continuing anyway.",
      config.end_effector_link.c_str());
  }
}

void configure_gripper_group(
  const SortDemoConfig& config,
  moveit::planning_interface::MoveGroupInterface& gripper_group)
{
  gripper_group.setPlanningTime(5.0);
  gripper_group.setNumPlanningAttempts(5);
  gripper_group.setMaxVelocityScalingFactor(config.gripper_velocity_scaling);
  gripper_group.setMaxAccelerationScalingFactor(config.gripper_acceleration_scaling);
  gripper_group.setGoalJointTolerance(config.gripper_joint_tolerance);
}

void initialize_gazebo_interface(
  const std::shared_ptr<rclcpp::Node>& node,
  GazeboDetachInterface& gazebo_detach_interface)
{
  for (const auto& cube_id : {"cube_small", "cube_medium", "cube_large"})
  {
    gazebo_detach_interface.attach_publishers[cube_id] =
      node->create_publisher<std_msgs::msg::Empty>("/" + std::string(cube_id) + "/attach", 10);
    gazebo_detach_interface.detach_publishers[cube_id] =
      node->create_publisher<std_msgs::msg::Empty>("/" + std::string(cube_id) + "/detach", 10);
    gazebo_detach_interface.state_subscribers[cube_id] =
      node->create_subscription<std_msgs::msg::Bool>(
        "/" + std::string(cube_id) + "/state", 10,
        [&gazebo_detach_interface, cube_id](const std_msgs::msg::Bool::SharedPtr message) {
          std::scoped_lock lock(gazebo_detach_interface.state_mutex);
          gazebo_detach_interface.attached_states[cube_id] = message->data;
          gazebo_detach_interface.state_received[cube_id] = true;
        });
  }
}

}  // namespace

SortDemoConfig load_sort_demo_config(rclcpp::Node& node)
{
  SortDemoConfig config;

  config.arm_group_name =
    node.declare_parameter<std::string>("planning_group", config.arm_group_name);
  config.gripper_group_name =
    node.declare_parameter<std::string>("gripper_group", config.gripper_group_name);
  config.end_effector_link =
    node.declare_parameter<std::string>("end_effector_link", config.end_effector_link);
  config.pose_reference_frame =
    node.declare_parameter<std::string>("pose_reference_frame", config.pose_reference_frame);
  config.planning_time = node.declare_parameter<double>("planning_time", config.planning_time);
  config.planning_attempts =
    node.declare_parameter<int>("planning_attempts", config.planning_attempts);
  config.arm_velocity_scaling =
    node.declare_parameter<double>("arm_velocity_scaling", config.arm_velocity_scaling);
  config.arm_acceleration_scaling =
    node.declare_parameter<double>("arm_acceleration_scaling", config.arm_acceleration_scaling);
  config.gripper_velocity_scaling =
    node.declare_parameter<double>("gripper_velocity_scaling", config.gripper_velocity_scaling);
  config.gripper_acceleration_scaling = node.declare_parameter<double>(
    "gripper_acceleration_scaling", config.gripper_acceleration_scaling);
  config.gripper_action_name =
    node.declare_parameter<std::string>("gripper_action_name", config.gripper_action_name);
  config.goal_position_tolerance = node.declare_parameter<double>(
    "goal_position_tolerance", config.goal_position_tolerance);
  config.goal_orientation_tolerance = node.declare_parameter<double>(
    "goal_orientation_tolerance", config.goal_orientation_tolerance);
  config.execution_options.dwell_seconds =
    node.declare_parameter<double>("dwell_seconds", config.execution_options.dwell_seconds);
  config.execution_options.approach_z =
    node.declare_parameter<double>("approach_z", config.execution_options.approach_z);
  config.gripper_base_to_grasp_plane = node.declare_parameter<double>(
    "gripper_base_to_grasp_plane", config.gripper_base_to_grasp_plane);
  config.grasp_surface_clearance = node.declare_parameter<double>(
    "grasp_surface_clearance", config.grasp_surface_clearance);
  config.execution_options.cartesian_avoid_collisions = node.declare_parameter<bool>(
    "cartesian_avoid_collisions", config.execution_options.cartesian_avoid_collisions);
  config.execution_options.pick_retry_step =
    node.declare_parameter<double>("pick_retry_step", config.execution_options.pick_retry_step);
  config.execution_options.pick_retry_count =
    node.declare_parameter<int>("pick_retry_count", config.execution_options.pick_retry_count);

  return config;
}

bool run_sort_scene_demo(const std::shared_ptr<rclcpp::Node>& node)
{
  const auto logger = node->get_logger();
  const auto config = load_sort_demo_config(*node);

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  std::thread spinner([&executor]() { executor.spin(); });

  moveit::planning_interface::MoveGroupInterface arm_group(node, config.arm_group_name);
  moveit::planning_interface::MoveGroupInterface gripper_group(node, config.gripper_group_name);
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
  auto gripper_action_client = rclcpp_action::create_client<control_msgs::action::FollowJointTrajectory>(
    node, config.gripper_action_name);
  auto get_planning_scene_client = node->create_client<GetPlanningScene>("get_planning_scene");
  GazeboDetachInterface gazebo_detach_interface;

  initialize_gazebo_interface(node, gazebo_detach_interface);
  configure_arm_group(config, arm_group, logger);
  configure_gripper_group(config, gripper_group);

  RCLCPP_INFO(
    logger,
    "Starting hardcoded sorting demo with group '%s', gripper '%s' and end effector '%s'.",
    config.arm_group_name.c_str(), config.gripper_group_name.c_str(),
    config.end_effector_link.c_str());
  RCLCPP_INFO(
    logger,
    "Grasp tuning: base_to_grasp_plane=%.3f m, surface_clearance=%.3f m, "
    "cartesian_avoid_collisions=%s, pick_retry_step=%.3f m, pick_retry_count=%d.",
    config.gripper_base_to_grasp_plane, config.grasp_surface_clearance,
    config.execution_options.cartesian_avoid_collisions ? "true" : "false",
    config.execution_options.pick_retry_step, config.execution_options.pick_retry_count);

  bool success = false;
  do
  {
    if (!wait_for_move_group(logger, arm_group, config.arm_group_name))
    {
      RCLCPP_ERROR(logger, "MoveGroup action server for the arm is not ready.");
      break;
    }

    if (!wait_for_current_state(logger, arm_group, config.arm_group_name))
    {
      RCLCPP_ERROR(logger, "Arm current state is not ready.");
      break;
    }

    if (!wait_for_current_state(logger, gripper_group, config.gripper_group_name))
    {
      RCLCPP_ERROR(logger, "Gripper current state is not ready.");
      break;
    }

    if (!wait_for_gripper_controller(logger, gripper_action_client, config.gripper_action_name))
    {
      RCLCPP_ERROR(logger, "Gripper controller action server is not ready.");
      break;
    }

    const std::vector<std::string> detachable_cube_ids = {"cube_small", "cube_medium", "cube_large"};
    if (!wait_for_gazebo_detach_interface(node, logger, gazebo_detach_interface, detachable_cube_ids))
    {
      RCLCPP_ERROR(logger, "Gazebo detachable-joint interface is not ready.");
      break;
    }

    if (!reset_detachable_cubes_at_startup(logger, gazebo_detach_interface, detachable_cube_ids))
    {
      RCLCPP_ERROR(logger, "Failed to reset Gazebo detachable cubes at startup.");
      break;
    }

    const auto scene_ids = build_scene_ids();
    planning_scene_interface.removeCollisionObjects(scene_ids);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    if (!planning_scene_interface.applyCollisionObjects(build_scene_objects()))
    {
      RCLCPP_ERROR(logger, "Failed to apply collision objects to the planning scene.");
      break;
    }

    if (!wait_for_objects(logger, planning_scene_interface, scene_ids))
    {
      RCLCPP_ERROR(logger, "Planning scene objects did not appear in time.");
      break;
    }

    if (!plan_and_execute_named_target(logger, arm_group, "up"))
    {
      break;
    }

    if (!execute_gripper_action_target(logger, gripper_action_client, true, 1.0))
    {
      break;
    }

    const auto reference_pose = arm_group.getCurrentPose(config.end_effector_link).pose;
    const auto tasks = build_sorting_tasks(
      reference_pose.orientation, config.execution_options.approach_z,
      config.gripper_base_to_grasp_plane, config.grasp_surface_clearance);
    const auto touch_links = get_touch_links();

    bool all_tasks_done = true;
    for (const auto& task : tasks)
    {
      if (!sort_single_cube(
            logger, arm_group, gripper_action_client, gazebo_detach_interface,
            planning_scene_interface, get_planning_scene_client, task, touch_links,
            config.execution_options))
      {
        all_tasks_done = false;
        break;
      }

      if (!plan_and_execute_named_target(logger, arm_group, "up"))
      {
        RCLCPP_WARN(
          logger,
          "Returning to named target 'up' failed after '%s'. Continuing with the next task.",
          task.cube_id.c_str());
      }
    }

    success = all_tasks_done;
  } while (false);

  arm_group.stop();
  gripper_group.stop();
  executor.cancel();
  if (spinner.joinable())
  {
    spinner.join();
  }

  return success;
}

}  // namespace manip_sort_pipeline::sort_demo
