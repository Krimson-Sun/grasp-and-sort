#include "manip_sort_pipeline/sort_demo/sorting.hpp"

#include <chrono>
#include <thread>

#include "manip_sort_pipeline/sort_demo/gazebo.hpp"
#include "manip_sort_pipeline/sort_demo/motion.hpp"
#include "manip_sort_pipeline/sort_demo/scene.hpp"

namespace manip_sort_pipeline::sort_demo
{

std::vector<SortingTask> build_sorting_tasks(
  const geometry_msgs::msg::Quaternion& orientation,
  double approach_z,
  double gripper_base_to_grasp_plane,
  double grasp_surface_clearance)
{
  constexpr double work_surface_z = 0.0;
  const auto grasp_z = [&](double cube_height) {
    return work_surface_z + cube_height + gripper_base_to_grasp_plane + grasp_surface_clearance;
  };

  return {
    {
      "cube_small",
      make_pose(0.52, -0.40, grasp_z(0.04), orientation),
      make_pose(0.62, 0.24, grasp_z(0.04), orientation),
      {
        make_pose(0.40, -0.52, 0.30, orientation),
        make_pose(0.44, -0.05, 0.42, orientation),
        make_pose(0.54, 0.12, approach_z, orientation),
      },
    },
    {
      "cube_medium",
      make_pose(0.62, -0.30, grasp_z(0.05), orientation),
      make_pose(0.62, 0.36, grasp_z(0.05), orientation),
      {
        make_pose(0.50, -0.12, 0.40, orientation),
        make_pose(0.58, 0.12, 0.42, orientation),
      },
    },
    {
      "cube_large",
      make_pose(0.72, -0.20, grasp_z(0.06), orientation),
      make_pose(0.62, 0.48, grasp_z(0.06), orientation),
      {
        make_pose(0.82, -0.02, 0.32, orientation),
        make_pose(0.80, 0.28, 0.40, orientation),
        make_pose(0.70, 0.58, approach_z, orientation),
      },
    },
  };
}

bool sort_single_cube(
  const rclcpp::Logger& logger,
  moveit::planning_interface::MoveGroupInterface& arm_group,
  const GripperActionClient::SharedPtr& gripper_action_client,
  GazeboDetachInterface& gazebo_detach_interface,
  moveit::planning_interface::PlanningSceneInterface& planning_scene_interface,
  const rclcpp::Client<GetPlanningScene>::SharedPtr& get_planning_scene_client,
  const SortingTask& task,
  const std::vector<std::string>& touch_links,
  const SortExecutionOptions& options)
{
  constexpr double contact_settle_seconds = 0.3;
  constexpr double breakaway_lift_distance = 0.006;
  constexpr double pregrasp_lift_distance = 0.027;
  constexpr double grasp_settle_seconds = 0.4;

  const auto orientation_candidates = get_vertical_grasp_orientations();
  const auto nominal_pick_approach = make_pose(
    task.pick_pose.position.x, task.pick_pose.position.y, options.approach_z, task.pick_pose.orientation);
  auto pick_approach = nominal_pick_approach;
  auto place_approach = make_pose(
    task.place_pose.position.x, task.place_pose.position.y, options.approach_z, task.place_pose.orientation);

  RCLCPP_INFO(logger, "Sorting '%s'.", task.cube_id.c_str());

  if (!execute_gripper_action_target(logger, gripper_action_client, true, 1.0))
  {
    return false;
  }

  bool orientation_validated = false;
  auto cartesian_pick = pick_approach;
  for (std::size_t i = 0; i < orientation_candidates.size(); ++i)
  {
    auto candidate_pick_approach = nominal_pick_approach;
    candidate_pick_approach.orientation = orientation_candidates[i];

    RCLCPP_INFO(
      logger, "Trying vertical orientation candidate %zu for '%s'.", i + 1,
      (task.cube_id + "_pick_approach").c_str());

    if (!plan_and_execute_pose_target(
          logger, arm_group, candidate_pick_approach, task.cube_id + "_pick_approach", "ompl", "",
          false, true, true))
    {
      continue;
    }

    const auto reached_pick_pose = arm_group.getCurrentPose(arm_group.getEndEffectorLink()).pose;
    candidate_pick_approach.orientation = reached_pick_pose.orientation;

    auto candidate_cartesian_pick = candidate_pick_approach;
    candidate_cartesian_pick.position.z = task.pick_pose.position.z;

    if (!set_grasp_object_collisions(
          logger, planning_scene_interface, get_planning_scene_client, task.cube_id, touch_links,
          true))
    {
      return false;
    }

    geometry_msgs::msg::Pose validated_cartesian_pick;
    const bool descent_available = find_cartesian_pick_pose_with_clearance_retry(
      logger, arm_group, candidate_cartesian_pick, task.cube_id + "_pick", 0.005, 0.98,
      options.cartesian_avoid_collisions, options.pick_retry_step, options.pick_retry_count,
      validated_cartesian_pick);

    if (!descent_available)
    {
      set_grasp_object_collisions(
        logger, planning_scene_interface, get_planning_scene_client, task.cube_id, touch_links,
        false);
      RCLCPP_WARN(
        logger,
        "Vertical orientation candidate %zu for '%s' was rejected because no safe downward "
        "Cartesian approach was found.",
        i + 1, (task.cube_id + "_pick_approach").c_str());
      continue;
    }

    pick_approach = candidate_pick_approach;
    place_approach.orientation = candidate_pick_approach.orientation;
    cartesian_pick = validated_cartesian_pick;
    orientation_validated = true;
    RCLCPP_INFO(
      logger, "Selected vertical orientation candidate %zu for '%s'.", i + 1,
      (task.cube_id + "_pick_approach").c_str());
    break;
  }

  if (!orientation_validated)
  {
    RCLCPP_ERROR(
      logger, "No vertical orientation candidate provided a safe downward approach for '%s'.",
      task.cube_id.c_str());
    return false;
  }

  if (!execute_cartesian_pick_with_clearance_retry(
        logger, arm_group, cartesian_pick, task.cube_id + "_pick", 0.005, 0.98,
        options.cartesian_avoid_collisions, options.pick_retry_step, options.pick_retry_count))
  {
    set_grasp_object_collisions(
      logger, planning_scene_interface, get_planning_scene_client, task.cube_id, touch_links, false);
    return false;
  }

  if (!set_surface_collisions(
        logger, planning_scene_interface, get_planning_scene_client, "source_table", touch_links,
        true))
  {
    return false;
  }

  std::this_thread::sleep_for(std::chrono::duration<double>(contact_settle_seconds));

  if (!execute_gripper_action_target(logger, gripper_action_client, false, 1.5))
  {
    set_surface_collisions(
      logger, planning_scene_interface, get_planning_scene_client, "source_table", touch_links,
      false);
    return false;
  }

  std::this_thread::sleep_for(std::chrono::duration<double>(grasp_settle_seconds));

  auto breakaway_lift = cartesian_pick;
  breakaway_lift.position.z += breakaway_lift_distance;
  if (!execute_cartesian_segment(
        logger, arm_group, breakaway_lift, task.cube_id + "_grasp_breakaway", 0.005, 0.98, false))
  {
    RCLCPP_WARN(
      logger,
      "Breakaway lift before Gazebo attach failed for '%s'. Continuing with direct attach attempt.",
      task.cube_id.c_str());
  }

  if (!request_cube_attach_state(logger, gazebo_detach_interface, task.cube_id, true, 2, false))
  {
    set_surface_collisions(
      logger, planning_scene_interface, get_planning_scene_client, "source_table", touch_links,
      false);
    return false;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  arm_group.attachObject(task.cube_id, arm_group.getEndEffectorLink(), touch_links);
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  auto pregrasp_lift = cartesian_pick;
  pregrasp_lift.position.z += pregrasp_lift_distance;
  if (!execute_cartesian_segment(
        logger, arm_group, pregrasp_lift, task.cube_id + "_pregrasp_lift", 0.005, 0.98,
        options.cartesian_avoid_collisions))
  {
    return false;
  }

  if (!set_collision_pair(
        logger, planning_scene_interface, get_planning_scene_client, task.cube_id, "source_table",
        true))
  {
    return false;
  }

  if (!set_surface_collisions(
        logger, planning_scene_interface, get_planning_scene_client, "source_table", touch_links,
        false))
  {
    return false;
  }

  auto cartesian_lift = pick_approach;
  cartesian_lift.position.z = pick_approach.position.z;
  if (!execute_cartesian_segment(
        logger, arm_group, cartesian_lift, task.cube_id + "_lift", 0.005, 0.98,
        options.cartesian_avoid_collisions))
  {
    return false;
  }

  if (!set_collision_pair(
        logger, planning_scene_interface, get_planning_scene_client, task.cube_id, "source_table",
        false))
  {
    return false;
  }

  auto transfer_waypoints = task.transfer_waypoints;
  for (auto& waypoint : transfer_waypoints)
  {
    waypoint.orientation = pick_approach.orientation;
  }

  if (!execute_cartesian_waypoint_sequence(
        logger, arm_group, transfer_waypoints, task.cube_id + "_transfer", 0.005, 0.98,
        options.cartesian_avoid_collisions))
  {
    return false;
  }

  place_approach.orientation = pick_approach.orientation;
  if (!execute_cartesian_segment(
        logger, arm_group, place_approach, task.cube_id + "_place_approach", 0.005, 0.98,
        options.cartesian_avoid_collisions))
  {
    return false;
  }

  if (!execute_gripper_action_target(logger, gripper_action_client, true, 1.0))
  {
    return false;
  }

  if (!request_cube_attach_state(logger, gazebo_detach_interface, task.cube_id, false))
  {
    return false;
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  arm_group.detachObject(task.cube_id);
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  if (!set_grasp_object_collisions(
        logger, planning_scene_interface, get_planning_scene_client, task.cube_id, touch_links,
        false))
  {
    return false;
  }

  std::this_thread::sleep_for(std::chrono::duration<double>(options.dwell_seconds));
  return true;
}

}  // namespace manip_sort_pipeline::sort_demo
