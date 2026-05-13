#include "manip_sort_pipeline/vision/executor.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <thread>
#include <vector>

#include "manip_sort_pipeline/sort_demo/gazebo.hpp"
#include "manip_sort_pipeline/sort_demo/motion.hpp"
#include "manip_sort_pipeline/sort_demo/scene.hpp"

namespace manip_sort_pipeline::vision
{
namespace
{

geometry_msgs::msg::Pose offset_pose_z(const geometry_msgs::msg::Pose& pose, double offset)
{
  auto result = pose;
  result.position.z += offset;
  return result;
}

double distance_xy(
  const geometry_msgs::msg::Point& left,
  const geometry_msgs::msg::Point& right)
{
  return std::hypot(left.x - right.x, left.y - right.y);
}

bool current_tcp_is_near_xy(
  const rclcpp::Logger& logger,
  moveit::planning_interface::MoveGroupInterface& arm_group,
  const geometry_msgs::msg::Pose& target_pose,
  const std::string& label,
  double max_xy_error)
{
  arm_group.getCurrentState(1.0);
  const auto current_pose = arm_group.getCurrentPose(arm_group.getEndEffectorLink()).pose;
  const auto xy_error = distance_xy(current_pose.position, target_pose.position);
  if (xy_error <= max_xy_error)
  {
    return true;
  }

  RCLCPP_ERROR(
    logger,
    "Refusing recovery detach for '%s': current TCP xy=(%.3f, %.3f) is %.3f m from target xy=(%.3f, %.3f).",
    label.c_str(), current_pose.position.x, current_pose.position.y, xy_error,
    target_pose.position.x, target_pose.position.y);
  return false;
}

void best_effort_open_gripper(
  const rclcpp::Logger& logger,
  const sort_demo::GripperActionClient::SharedPtr& gripper_action_client)
{
  sort_demo::execute_gripper_action_target(logger, gripper_action_client, true, 1.0);
}

bool release_attached_object_before_opening(
  const rclcpp::Logger& logger,
  moveit::planning_interface::MoveGroupInterface& arm_group,
  const sort_demo::GripperActionClient::SharedPtr& gripper_action_client,
  moveit::planning_interface::PlanningSceneInterface& planning_scene_interface,
  sort_demo::GazeboDetachInterface& gazebo_detach_interface,
  const std::string& object_id)
{
  RCLCPP_WARN(
    logger,
    "Recovering with '%s' attached. Keeping the gripper closed until Gazebo detach is confirmed.",
    object_id.c_str());

  if (!sort_demo::request_cube_attach_state(logger, gazebo_detach_interface, object_id, false, 3, false))
  {
    RCLCPP_ERROR(
      logger,
      "Failed to confirm Gazebo detach for '%s'. Leaving the gripper closed to avoid dropping or "
      "dragging an attached object with open fingers.",
      object_id.c_str());
    return false;
  }

  arm_group.detachObject(object_id);
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  if (!sort_demo::remove_collision_object(logger, planning_scene_interface, object_id))
  {
    RCLCPP_WARN(
      logger,
      "Continuing recovery for '%s' with its gripper collisions still allowed because the "
      "collision object removal was not confirmed.",
      object_id.c_str());
  }
  best_effort_open_gripper(logger, gripper_action_client);
  return true;
}

bool same_orientation(
  const geometry_msgs::msg::Quaternion& left,
  const geometry_msgs::msg::Quaternion& right)
{
  constexpr double tolerance = 1e-4;
  return std::abs(left.x - right.x) < tolerance && std::abs(left.y - right.y) < tolerance &&
         std::abs(left.z - right.z) < tolerance && std::abs(left.w - right.w) < tolerance;
}

std::vector<geometry_msgs::msg::Quaternion> build_preplace_orientation_options(
  const geometry_msgs::msg::Quaternion& preferred_orientation)
{
  std::vector<geometry_msgs::msg::Quaternion> options;
  options.push_back(preferred_orientation);

  for (const auto& orientation : sort_demo::get_vertical_grasp_orientations())
  {
    const auto duplicate = std::any_of(
      options.begin(), options.end(),
      [&orientation](const auto& existing) { return same_orientation(existing, orientation); });
    if (!duplicate)
    {
      options.push_back(orientation);
    }
  }

  return options;
}

bool plan_and_execute_transfer_to_preplace(
  const rclcpp::Logger& logger,
  moveit::planning_interface::MoveGroupInterface& arm_group,
  const geometry_msgs::msg::Pose& preplace,
  const std::string& label,
  const VisionManagerConfig& config)
{
  if (sort_demo::plan_and_execute_best_pose_target(
        logger, arm_group, preplace, label, config.transfer_planners,
        config.transfer_score_weights))
  {
    return true;
  }

  RCLCPP_WARN(
    logger,
    "Best-plan transfer failed for '%s'. Trying the configured planner portfolio again with "
    "approximate IK and position-only fallback before giving up with the object attached.",
    label.c_str());

  for (const auto& planner : config.transfer_planners)
  {
    const int attempts = std::max(1, planner.attempts);
    for (int attempt = 1; attempt <= attempts; ++attempt)
    {
      if (sort_demo::plan_and_execute_pose_target(
            logger, arm_group, preplace, label + "_robust_fallback",
            planner.pipeline_id, planner.planner_id, false, true, true))
      {
        return true;
      }
    }
  }

  return false;
}

}  // namespace

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
  std::string& failure_reason)
{
  const auto pregrasp = offset_pose_z(task.grasp_pose, config.pregrasp_z_offset);
  const auto lift_pose = offset_pose_z(task.grasp_pose, config.lift_z_offset);
  auto place_pose = task.place_pose;
  auto preplace = offset_pose_z(place_pose, config.place_z_offset);
  auto retreat = offset_pose_z(place_pose, config.retreat_z_offset);
  bool released_collision_object_removed = false;
  RCLCPP_INFO(
    logger,
    "Executing task for '%s': grasp=(%.3f, %.3f, %.3f) grasp_quat=(%.3f, %.3f, %.3f, %.3f) "
    "pregrasp=(%.3f, %.3f, %.3f) pregrasp_quat=(%.3f, %.3f, %.3f, %.3f) "
    "place=(%.3f, %.3f, %.3f) place_quat=(%.3f, %.3f, %.3f, %.3f) "
    "preplace=(%.3f, %.3f, %.3f) preplace_quat=(%.3f, %.3f, %.3f, %.3f) dry_run=%s",
    task.object_id.c_str(),
    task.grasp_pose.position.x, task.grasp_pose.position.y, task.grasp_pose.position.z,
    task.grasp_pose.orientation.x, task.grasp_pose.orientation.y,
    task.grasp_pose.orientation.z, task.grasp_pose.orientation.w,
    pregrasp.position.x, pregrasp.position.y, pregrasp.position.z,
    pregrasp.orientation.x, pregrasp.orientation.y,
    pregrasp.orientation.z, pregrasp.orientation.w,
    place_pose.position.x, place_pose.position.y, place_pose.position.z,
    place_pose.orientation.x, place_pose.orientation.y,
    place_pose.orientation.z, place_pose.orientation.w,
    preplace.position.x, preplace.position.y, preplace.position.z,
    preplace.orientation.x, preplace.orientation.y,
    preplace.orientation.z, preplace.orientation.w,
    config.dry_run ? "true" : "false");

  if (!config.dry_run)
  {
    sort_demo::request_cube_attach_state(logger, gazebo_detach_interface, task.object_id, false, 1, false);
    arm_group.detachObject(task.object_id);
  }

  if (!sort_demo::execute_gripper_action_target(logger, gripper_action_client, true, 1.0))
  {
    failure_reason = "open_gripper_failed";
    return false;
  }

  if (!sort_demo::plan_and_execute_best_pose_target(
        logger, arm_group, pregrasp, task.object_id + "_pregrasp_approach",
        config.pregrasp_planners, config.pregrasp_score_weights))
  {
    RCLCPP_WARN(
      logger,
      "Best-plan pregrasp approach for '%s' failed. Falling back to the legacy OMPL pregrasp planner.",
      task.object_id.c_str());
    if (!sort_demo::plan_and_execute_pose_target(
          logger, arm_group, pregrasp, task.object_id + "_pregrasp", "ompl", "",
          false, true, false))
    {
      failure_reason = "plan_failed";
      return false;
    }
  }

  if (!sort_demo::set_grasp_object_collisions(
        logger, planning_scene_interface, get_planning_scene_client, task.object_id, touch_links,
        true))
  {
    failure_reason = "planning_scene_failed";
    return false;
  }

  if (!sort_demo::set_surface_collisions(
        logger, planning_scene_interface, get_planning_scene_client, "source_table", touch_links,
        true))
  {
    sort_demo::set_grasp_object_collisions(
      logger, planning_scene_interface, get_planning_scene_client, task.object_id, touch_links,
      false);
    failure_reason = "planning_scene_failed";
    return false;
  }

  if (!sort_demo::execute_cartesian_pick_with_clearance_retry(
        logger, arm_group, task.grasp_pose, task.object_id + "_grasp", 0.005, 0.98,
        config.sort_demo_config.execution_options.cartesian_avoid_collisions,
        config.sort_demo_config.execution_options.pick_retry_step,
        config.sort_demo_config.execution_options.pick_retry_count))
  {
    sort_demo::set_surface_collisions(
      logger, planning_scene_interface, get_planning_scene_client, "source_table", touch_links,
      false);
    sort_demo::set_grasp_object_collisions(
      logger, planning_scene_interface, get_planning_scene_client, task.object_id, touch_links,
      false);
    failure_reason = "plan_failed";
    return false;
  }

  if (!config.dry_run)
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    if (!sort_demo::execute_gripper_action_target(logger, gripper_action_client, false, 1.5))
    {
      best_effort_open_gripper(logger, gripper_action_client);
      sort_demo::set_surface_collisions(
        logger, planning_scene_interface, get_planning_scene_client, "source_table", touch_links,
        false);
      sort_demo::set_grasp_object_collisions(
        logger, planning_scene_interface, get_planning_scene_client, task.object_id, touch_links,
        false);
      failure_reason = "close_gripper_failed";
      return false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    if (!sort_demo::request_cube_attach_state(
          logger, gazebo_detach_interface, task.object_id, true, 3, false))
    {
      best_effort_open_gripper(logger, gripper_action_client);
      sort_demo::set_surface_collisions(
        logger, planning_scene_interface, get_planning_scene_client, "source_table", touch_links,
        false);
      sort_demo::set_grasp_object_collisions(
        logger, planning_scene_interface, get_planning_scene_client, task.object_id, touch_links,
        false);
      failure_reason = "attach_failed";
      return false;
    }

    arm_group.attachObject(task.object_id, arm_group.getEndEffectorLink(), touch_links);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  else
  {
    RCLCPP_INFO(logger, "Dry-run active: skipping physical close/attach for '%s'.", task.object_id.c_str());
  }

  if (!sort_demo::execute_cartesian_segment(
        logger, arm_group, lift_pose, task.object_id + "_lift", 0.005, 0.98,
        config.sort_demo_config.execution_options.cartesian_avoid_collisions))
  {
    if (!config.dry_run && !release_attached_object_before_opening(
          logger, arm_group, gripper_action_client, planning_scene_interface,
          gazebo_detach_interface, task.object_id))
    {
      failure_reason = "attached_recovery_failed";
      return false;
    }
    if (config.dry_run)
    {
      best_effort_open_gripper(logger, gripper_action_client);
    }
    sort_demo::set_surface_collisions(
      logger, planning_scene_interface, get_planning_scene_client, "source_table", touch_links,
      false);
    sort_demo::set_grasp_object_collisions(
      logger, planning_scene_interface, get_planning_scene_client, task.object_id, touch_links,
      false);
    failure_reason = "lift_failed";
    return false;
  }

  sort_demo::set_surface_collisions(
    logger, planning_scene_interface, get_planning_scene_client, "source_table", touch_links,
    false);

  bool reached_preplace = false;
  std::size_t preplace_orientation_index = 0;
  const auto preplace_orientation_options =
    build_preplace_orientation_options(task.place_pose.orientation);
  for (const auto& orientation : preplace_orientation_options)
  {
    ++preplace_orientation_index;
    place_pose.orientation = orientation;
    preplace = offset_pose_z(place_pose, config.place_z_offset);
    retreat = offset_pose_z(place_pose, config.retreat_z_offset);

    if (preplace_orientation_index > 1)
    {
      RCLCPP_WARN(
        logger,
        "Retrying preplace for '%s' with vertical orientation option %zu/%zu while keeping the object attached.",
        task.object_id.c_str(), preplace_orientation_index, preplace_orientation_options.size());
    }

    if (plan_and_execute_transfer_to_preplace(
          logger, arm_group, preplace, task.object_id + "_preplace_transfer", config))
    {
      reached_preplace = true;
      break;
    }
  }

  if (!reached_preplace)
  {
    if (!config.dry_run)
    {
      failure_reason = "attached_recovery_failed";
      RCLCPP_ERROR(
        logger,
        "Failed to reach verified preplace for '%s'. Keeping the object attached instead of "
        "detaching away from the target bin.",
        task.object_id.c_str());
      return false;
    }
    if (config.dry_run)
    {
      best_effort_open_gripper(logger, gripper_action_client);
    }
    failure_reason = "place_failed";
    return false;
  }

  if (!sort_demo::execute_cartesian_segment(
        logger, arm_group, place_pose, task.object_id + "_place", 0.005, 0.98,
        config.sort_demo_config.execution_options.cartesian_avoid_collisions))
  {
    if (!config.dry_run && !current_tcp_is_near_xy(
          logger, arm_group, place_pose, task.object_id + "_place_recovery", 0.08))
    {
      failure_reason = "attached_recovery_failed";
      RCLCPP_ERROR(
        logger,
        "Cartesian place for '%s' failed before the TCP reached the target bin. Keeping the object "
        "attached for manual recovery.",
        task.object_id.c_str());
      return false;
    }
    if (!config.dry_run && !release_attached_object_before_opening(
          logger, arm_group, gripper_action_client, planning_scene_interface,
          gazebo_detach_interface, task.object_id))
    {
      failure_reason = "attached_recovery_failed";
      return false;
    }
    if (config.dry_run)
    {
      best_effort_open_gripper(logger, gripper_action_client);
    }
    failure_reason = "place_failed";
    return false;
  }

  if (!config.dry_run)
  {
    if (!sort_demo::request_cube_attach_state(logger, gazebo_detach_interface, task.object_id, false, 3, false))
    {
      failure_reason = "detach_failed";
      return false;
    }
    arm_group.detachObject(task.object_id);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    released_collision_object_removed =
      sort_demo::remove_collision_object(logger, planning_scene_interface, task.object_id);
    if (!released_collision_object_removed)
    {
      RCLCPP_WARN(
        logger,
        "Keeping allowed collisions between '%s' and the gripper links because the released "
        "object still exists in the planning scene.",
        task.object_id.c_str());
    }

    if (!sort_demo::execute_gripper_action_target(logger, gripper_action_client, true, 1.0))
    {
      failure_reason = "open_gripper_failed";
      return false;
    }
  }
  else
  {
    RCLCPP_INFO(logger, "Dry-run active: skipping detach/open sequence for '%s'.", task.object_id.c_str());
  }

  if (!sort_demo::execute_cartesian_segment(
        logger, arm_group, retreat, task.object_id + "_retreat", 0.005, 0.98,
        config.sort_demo_config.execution_options.cartesian_avoid_collisions))
  {
    RCLCPP_WARN(
      logger,
      "Retreat after placing '%s' failed, but the object was already detached and the gripper was opened. "
      "Treating the sort as complete; the manager will still try to return to the scan pose.",
      task.object_id.c_str());
  }

  if (config.dry_run || released_collision_object_removed)
  {
    sort_demo::set_grasp_object_collisions(
      logger, planning_scene_interface, get_planning_scene_client, task.object_id, touch_links,
      false);
  }
  else
  {
    RCLCPP_WARN(
      logger,
      "Leaving allowed collisions for released object '%s' with the gripper links enabled to "
      "avoid a false start-state collision on the next plan.",
      task.object_id.c_str());
  }

  failure_reason.clear();
  RCLCPP_INFO(logger, "Task for '%s' completed successfully.", task.object_id.c_str());
  return true;
}

}  // namespace manip_sort_pipeline::vision
