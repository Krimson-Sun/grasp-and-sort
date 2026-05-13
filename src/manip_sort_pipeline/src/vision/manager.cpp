#include "manip_sort_pipeline/vision/manager.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iterator>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <manip_sort_interfaces/msg/detected_object_array.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <std_msgs/msg/string.hpp>

#include "manip_sort_pipeline/sort_demo/gazebo.hpp"
#include "manip_sort_pipeline/sort_demo/motion.hpp"
#include "manip_sort_pipeline/sort_demo/runner.hpp"
#include "manip_sort_pipeline/sort_demo/scene.hpp"
#include "manip_sort_pipeline/vision/config.hpp"
#include "manip_sort_pipeline/vision/executor.hpp"

using namespace std::chrono_literals;

namespace manip_sort_pipeline::vision
{
namespace
{

class DetectedObjectsCache
{
public:
  void update(const manip_sort_interfaces::msg::DetectedObjectArray::SharedPtr& message)
  {
    std::scoped_lock lock(mutex_);
    latest_ = *message;
    received_ = true;
    last_update_ = std::chrono::steady_clock::now();
  }

  bool get_fresh_copy_after(
    manip_sort_interfaces::msg::DetectedObjectArray& output,
    const std::chrono::duration<double>& stale_after,
    const std::chrono::steady_clock::time_point& min_update_time) const
  {
    std::scoped_lock lock(mutex_);
    if (!received_)
    {
      return false;
    }

    if (last_update_ < min_update_time)
    {
      return false;
    }

    if (std::chrono::steady_clock::now() - last_update_ > stale_after)
    {
      return false;
    }

    output = latest_;
    return true;
  }

  bool has_received() const
  {
    std::scoped_lock lock(mutex_);
    return received_;
  }

  double age_seconds() const
  {
    std::scoped_lock lock(mutex_);
    if (!received_)
    {
      return -1.0;
    }
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - last_update_).count();
  }

private:
  mutable std::mutex mutex_;
  manip_sort_interfaces::msg::DetectedObjectArray latest_;
  bool received_ = false;
  std::chrono::steady_clock::time_point last_update_;
};

double object_distance_xy(const manip_sort_interfaces::msg::DetectedObject& object)
{
  return std::hypot(object.centroid_world.x, object.centroid_world.y);
}

VisionExecutionTask make_task(
  const manip_sort_interfaces::msg::DetectedObject& object,
  const manip_sort_interfaces::msg::GraspCandidate& candidate,
  const VisionObjectConfig& config,
  const VisionManagerConfig& manager_config)
{
  VisionExecutionTask task;
  task.object_id = object.object_id;
  task.grasp_pose = candidate.pose;
  task.place_pose = manager_config.bin_place_poses.at(config.bin_name);
  task.place_pose.orientation = candidate.pose.orientation;
  return task;
}

bool wait_for_fresh_objects(
  const rclcpp::Logger& logger,
  const DetectedObjectsCache& cache,
  const VisionManagerConfig& config,
  manip_sort_interfaces::msg::DetectedObjectArray& result,
  const std::chrono::steady_clock::time_point& min_update_time)
{
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(config.scan_timeout);
  while (std::chrono::steady_clock::now() < deadline)
  {
    if (cache.get_fresh_copy_after(
          result, std::chrono::duration<double>(config.perception_stale_after), min_update_time))
    {
      return true;
    }

    std::this_thread::sleep_for(100ms);
  }

  if (!cache.has_received())
  {
    RCLCPP_WARN(
      logger,
      "Timed out waiting for a fresh perception result on '%s'. No perception messages have been received yet.",
      config.perception_topic.c_str());
  }
  else
  {
    RCLCPP_WARN(
      logger,
      "Timed out waiting for a fresh perception result on '%s'. Latest cached message age is %.2f s.",
      config.perception_topic.c_str(), cache.age_seconds());
  }
  return false;
}

std::string format_xyz(const geometry_msgs::msg::Point& point)
{
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(3)
         << "(" << point.x << ", " << point.y << ", " << point.z << ")";
  return stream.str();
}

std::string format_pose_xyz(const geometry_msgs::msg::Pose& pose)
{
  return format_xyz(pose.position);
}

bool refresh_object_from_new_frame(
  const rclcpp::Logger& logger,
  const DetectedObjectsCache& cache,
  const VisionManagerConfig& config,
  const std::string& object_id,
  manip_sort_interfaces::msg::DetectedObject& refreshed_object)
{
  manip_sort_interfaces::msg::DetectedObjectArray latest_objects;
  const auto validation_time = std::chrono::steady_clock::now();
  if (!wait_for_fresh_objects(logger, cache, config, latest_objects, validation_time))
  {
    RCLCPP_WARN(
      logger,
      "Skipping '%s' because no new perception frame arrived before execution validation.",
      object_id.c_str());
    return false;
  }

  for (const auto& object : latest_objects.objects)
  {
    if (object.object_id != object_id)
    {
      continue;
    }
    if (object.candidates.empty())
    {
      RCLCPP_WARN(
        logger,
        "Skipping '%s' because the latest perception frame has no candidates for it.",
        object_id.c_str());
      return false;
    }

    refreshed_object = object;
    return true;
  }

  RCLCPP_WARN(
    logger,
    "Skipping stale target '%s': it is not present in the latest perception frame.",
    object_id.c_str());
  return false;
}

bool object_present_in_new_frame(
  const rclcpp::Logger& logger,
  const DetectedObjectsCache& cache,
  const VisionManagerConfig& config,
  const std::string& object_id,
  bool& present)
{
  manip_sort_interfaces::msg::DetectedObjectArray latest_objects;
  const auto validation_time = std::chrono::steady_clock::now();
  if (!wait_for_fresh_objects(logger, cache, config, latest_objects, validation_time))
  {
    RCLCPP_WARN(
      logger,
      "Could not confirm whether '%s' is still visible after the failed candidate.",
      object_id.c_str());
    return false;
  }

  present = std::any_of(
    latest_objects.objects.begin(), latest_objects.objects.end(),
    [&object_id](const auto& object) { return object.object_id == object_id; });
  return true;
}

void configure_arm_group(
  const sort_demo::SortDemoConfig& config,
  moveit::planning_interface::MoveGroupInterface& arm_group)
{
  arm_group.setPoseReferenceFrame(config.pose_reference_frame);
  arm_group.setPlanningTime(config.planning_time);
  arm_group.setNumPlanningAttempts(static_cast<unsigned int>(config.planning_attempts));
  arm_group.setMaxVelocityScalingFactor(config.arm_velocity_scaling);
  arm_group.setMaxAccelerationScalingFactor(config.arm_acceleration_scaling);
  arm_group.setGoalPositionTolerance(config.goal_position_tolerance);
  arm_group.setGoalOrientationTolerance(config.goal_orientation_tolerance);
  arm_group.setWorkspace(-0.2, -1.2, 0.0, 1.4, 1.2, 1.6);
  arm_group.setEndEffectorLink(config.end_effector_link);
}

void configure_gripper_group(
  const sort_demo::SortDemoConfig& config,
  moveit::planning_interface::MoveGroupInterface& gripper_group)
{
  gripper_group.setPlanningTime(5.0);
  gripper_group.setNumPlanningAttempts(5);
  gripper_group.setMaxVelocityScalingFactor(config.gripper_velocity_scaling);
  gripper_group.setMaxAccelerationScalingFactor(config.gripper_acceleration_scaling);
  gripper_group.setGoalJointTolerance(config.gripper_joint_tolerance);
}

}  // namespace

bool run_vision_sort_manager(const std::shared_ptr<rclcpp::Node>& node)
{
  const auto logger = node->get_logger();
  const auto config = load_vision_manager_config(*node);
  const auto object_lookup = build_object_lookup(config);
  const auto object_ids = get_object_ids(config);
  std::unordered_map<std::string, std::size_t> object_priority;
  for (std::size_t index = 0; index < config.objects.size(); ++index)
  {
    object_priority[config.objects[index].object_id] = index;
  }

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  std::thread spinner([&executor]() { executor.spin(); });

  DetectedObjectsCache cache;
  auto subscription = node->create_subscription<manip_sort_interfaces::msg::DetectedObjectArray>(
    config.perception_topic, 10,
    [&cache](const manip_sort_interfaces::msg::DetectedObjectArray::SharedPtr message) {
      cache.update(message);
    });
  (void)subscription;
  auto decision_capture_publisher = node->create_publisher<std_msgs::msg::String>(
    config.decision_capture_topic, 10);

  moveit::planning_interface::MoveGroupInterface arm_group(node, config.sort_demo_config.arm_group_name);
  moveit::planning_interface::MoveGroupInterface gripper_group(node, config.sort_demo_config.gripper_group_name);
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
  auto gripper_action_client =
    rclcpp_action::create_client<control_msgs::action::FollowJointTrajectory>(
      node, config.sort_demo_config.gripper_action_name);
  auto get_planning_scene_client =
    node->create_client<sort_demo::GetPlanningScene>("get_planning_scene");
  sort_demo::GazeboDetachInterface gazebo_detach_interface;

  initialize_gazebo_interface_for_objects(node, gazebo_detach_interface, object_ids);
  configure_arm_group(config.sort_demo_config, arm_group);
  configure_gripper_group(config.sort_demo_config, gripper_group);
  if (config.sort_demo_config.end_effector_link != "gripper_tcp")
  {
    RCLCPP_WARN(
      logger,
      "Vision sort is configured with end_effector_link='%s'. Grasp poses are generated for "
      "'gripper_tcp', so using a different link will introduce a systematic XY offset at pick time.",
      config.sort_demo_config.end_effector_link.c_str());
  }
  RCLCPP_INFO(
    logger,
    "Vision sort manager configured with %zu objects, perception topic '%s', end_effector_link='%s', "
    "pose_reference_frame='%s', dry_run=%s, capture_decision_frames=%s, "
    "return_to_scan_after_success=%s, return_to_scan_after_candidate_failure=%s, "
    "recover_to_scan_after_skipped_success_timeout=%s.",
    object_ids.size(), config.perception_topic.c_str(),
    config.sort_demo_config.end_effector_link.c_str(),
    config.sort_demo_config.pose_reference_frame.c_str(),
    config.dry_run ? "true" : "false",
    config.capture_decision_frames ? "true" : "false",
    config.return_to_scan_after_success ? "true" : "false",
    config.return_to_scan_after_candidate_failure ? "true" : "false",
    config.recover_to_scan_after_skipped_success_timeout ? "true" : "false");
  {
    std::ostringstream stream;
    for (const auto& planner : config.pregrasp_planners)
    {
      if (stream.tellp() > 0)
      {
        stream << ", ";
      }
      stream << planner.pipeline_id << "/"
             << (planner.planner_id.empty() ? "default" : planner.planner_id)
             << "x" << planner.attempts;
    }
    RCLCPP_INFO(
      logger,
      "Pregrasp best-plan scoring enabled with planners [%s]. weights: detour=%.2f, "
      "joint=%.2f, duration=%.2f, z_over=%.2f; reject: max_detour=%.2f, "
      "max_z_over=%.2f, max_duration=%.2f.",
      stream.str().c_str(), config.pregrasp_score_weights.detour_ratio,
      config.pregrasp_score_weights.joint_path_length,
      config.pregrasp_score_weights.duration, config.pregrasp_score_weights.z_overshoot,
      config.pregrasp_score_weights.max_detour_ratio,
      config.pregrasp_score_weights.max_z_overshoot,
      config.pregrasp_score_weights.max_duration);
  }
  {
    std::ostringstream stream;
    for (const auto& planner : config.transfer_planners)
    {
      if (stream.tellp() > 0)
      {
        stream << ", ";
      }
      stream << planner.pipeline_id << "/"
             << (planner.planner_id.empty() ? "default" : planner.planner_id)
             << "x" << planner.attempts;
    }
    RCLCPP_INFO(
      logger,
      "Transfer best-plan scoring enabled with planners [%s]. weights: detour=%.2f, "
      "joint=%.2f, duration=%.2f, z_over=%.2f; reject: max_detour=%.2f, "
      "max_z_over=%.2f, max_duration=%.2f.",
      stream.str().c_str(), config.transfer_score_weights.detour_ratio,
      config.transfer_score_weights.joint_path_length,
      config.transfer_score_weights.duration, config.transfer_score_weights.z_overshoot,
      config.transfer_score_weights.max_detour_ratio,
      config.transfer_score_weights.max_z_overshoot,
      config.transfer_score_weights.max_duration);
  }

  bool success = false;
  do
  {
    if (!sort_demo::wait_for_move_group(logger, arm_group, config.sort_demo_config.arm_group_name) ||
        !sort_demo::wait_for_current_state(logger, arm_group, config.sort_demo_config.arm_group_name) ||
        !sort_demo::wait_for_current_state(logger, gripper_group, config.sort_demo_config.gripper_group_name) ||
        !sort_demo::wait_for_gripper_controller(
          logger, gripper_action_client, config.sort_demo_config.gripper_action_name) ||
        !sort_demo::wait_for_gazebo_detach_interface(node, logger, gazebo_detach_interface, object_ids))
    {
      break;
    }

    if (!reset_objects_at_startup(logger, gazebo_detach_interface, config))
    {
      RCLCPP_ERROR(logger, "Failed to reset the Gazebo sorting objects at startup.");
      break;
    }

    const auto scene_ids = build_vision_scene_ids(config);
    planning_scene_interface.removeCollisionObjects(scene_ids);
    std::this_thread::sleep_for(500ms);

    if (!planning_scene_interface.applyCollisionObjects(build_vision_scene_objects(config)) ||
        !sort_demo::wait_for_objects(logger, planning_scene_interface, scene_ids))
    {
      RCLCPP_ERROR(logger, "Failed to initialize the planning scene for vision sorting.");
      break;
    }

    if (!sort_demo::set_collision_pair(
          logger, planning_scene_interface, get_planning_scene_client, "ground_plane",
          "source_table", true) ||
        !sort_demo::set_collision_pair(
          logger, planning_scene_interface, get_planning_scene_client, "ground_plane",
          "target_table", true))
    {
      RCLCPP_ERROR(logger, "Failed to allow expected table contacts with the Gazebo ground plane.");
      break;
    }

    if (!sort_demo::plan_and_execute_named_target(logger, arm_group, config.scan_named_target))
    {
      break;
    }

    std::unordered_set<std::string> completed_object_ids;
    std::unordered_set<std::string> deferred_object_ids;
    std::size_t consecutive_perception_timeouts = 0;
    bool skipped_scan_after_success = false;
    const auto touch_links = sort_demo::get_touch_links();
    while (rclcpp::ok())
    {
      manip_sort_interfaces::msg::DetectedObjectArray detected_objects;
      const auto scan_ready_time = std::chrono::steady_clock::now();
      if (!wait_for_fresh_objects(logger, cache, config, detected_objects, scan_ready_time))
      {
        ++consecutive_perception_timeouts;
        RCLCPP_WARN(
          logger,
          "Perception timeout #%zu. Waiting for the next frame instead of shutting down.",
          consecutive_perception_timeouts);
        if (skipped_scan_after_success && config.recover_to_scan_after_skipped_success_timeout)
        {
          RCLCPP_WARN(
            logger,
            "Perception timed out after skipping the post-sort scan pose. Returning to '%s' to recover camera visibility.",
            config.scan_named_target.c_str());
          sort_demo::plan_and_execute_named_target(logger, arm_group, config.scan_named_target);
          skipped_scan_after_success = false;
        }
        else if (skipped_scan_after_success)
        {
          RCLCPP_WARN(
            logger,
            "Perception timed out after skipping the post-sort scan pose. Staying at the current pose because scan recovery is disabled.");
        }
        continue;
      }
      consecutive_perception_timeouts = 0;

      std::vector<manip_sort_interfaces::msg::DetectedObject> valid_objects;
      for (const auto& object : detected_objects.objects)
      {
        if (object_lookup.find(object.object_id) == object_lookup.end())
        {
          RCLCPP_WARN(
            logger, "Skipping perception object '%s' because it is not present in the object registry.",
            object.object_id.c_str());
          continue;
        }
        if (completed_object_ids.find(object.object_id) != completed_object_ids.end())
        {
          RCLCPP_INFO(
            logger, "Skipping perception object '%s' because it was already sorted.",
            object.object_id.c_str());
          continue;
        }
        if (object.candidates.empty())
        {
          RCLCPP_WARN(logger, "Skipping '%s' because perception returned no candidates.", object.object_id.c_str());
          continue;
        }
        valid_objects.push_back(object);
      }

      if (valid_objects.empty())
      {
        RCLCPP_INFO(logger, "No valid objects remain in the perception result. Finishing sort loop.");
        success = true;
        break;
      }

      std::vector<manip_sort_interfaces::msg::DetectedObject> pending_objects;
      pending_objects.reserve(valid_objects.size());
      std::copy_if(
        valid_objects.begin(), valid_objects.end(), std::back_inserter(pending_objects),
        [&deferred_object_ids](const auto& object) {
          return deferred_object_ids.find(object.object_id) == deferred_object_ids.end();
        });

      if (pending_objects.empty())
      {
        RCLCPP_INFO(
          logger,
          "All %zu valid objects are temporarily deferred after recent failures. Retrying deferred objects.",
          valid_objects.size());
        deferred_object_ids.clear();
        pending_objects = valid_objects;
      }

      std::sort(
        pending_objects.begin(), pending_objects.end(),
        [&object_priority](const auto& left, const auto& right) {
          const auto left_priority = object_priority.at(left.object_id);
          const auto right_priority = object_priority.at(right.object_id);
          if (left_priority != right_priority)
          {
            return left_priority < right_priority;
          }
          return object_distance_xy(left) < object_distance_xy(right);
        });
      {
        std::ostringstream stream;
        for (const auto& object : pending_objects)
        {
          if (stream.tellp() > 0)
          {
            stream << ", ";
          }
          stream << object.object_id << "@d=" << std::fixed << std::setprecision(3)
                 << object_distance_xy(object) << " candidates=" << object.candidates.size();
        }
        RCLCPP_INFO(logger, "Current cycle pending objects ordered by priority: [%s]", stream.str().c_str());
      }

      bool cycle_success = false;
      bool cycle_abort = false;
      bool cycle_rescan = false;
      for (const auto& object : pending_objects)
      {
        manip_sort_interfaces::msg::DetectedObject active_object;
        if (!refresh_object_from_new_frame(logger, cache, config, object.object_id, active_object))
        {
          cycle_rescan = true;
          break;
        }

        const auto& object_config = object_lookup.at(active_object.object_id);
        RCLCPP_INFO(
          logger, "Trying object '%s' for bin '%s'. centroid_world=%s best_score=%.3f candidates=%zu",
          active_object.object_id.c_str(), object_config.bin_name.c_str(),
          format_xyz(active_object.centroid_world).c_str(), active_object.best_grasp_score,
          active_object.candidates.size());
        if (config.capture_decision_frames)
        {
          std_msgs::msg::String capture_request;
          capture_request.data = active_object.object_id;
          decision_capture_publisher->publish(capture_request);
          RCLCPP_INFO(
            logger,
            "Requested automatic capture of the decision frame for '%s' on topic '%s'.",
            active_object.object_id.c_str(), config.decision_capture_topic.c_str());
        }
        std::size_t candidate_index = 0;
        for (const auto& candidate : active_object.candidates)
        {
          ++candidate_index;
          RCLCPP_INFO(
            logger,
            "Trying candidate %zu/%zu for '%s': grasp=%s quat=(%.3f, %.3f, %.3f, %.3f) "
            "score=%.3f width=%.3f support=%.3f flatness=%.4f symmetry=%.3f",
            candidate_index, active_object.candidates.size(), active_object.object_id.c_str(),
            format_pose_xyz(candidate.pose).c_str(),
            candidate.pose.orientation.x, candidate.pose.orientation.y,
            candidate.pose.orientation.z, candidate.pose.orientation.w,
            candidate.score, candidate.width,
            candidate.support_length, candidate.flatness, candidate.symmetry);
          const auto task = make_task(active_object, candidate, object_config, config);
          std::string failure_reason;
          if (execute_vision_sort_task(
                logger, arm_group, gripper_action_client, gazebo_detach_interface,
                planning_scene_interface, get_planning_scene_client, task, touch_links, config,
                failure_reason))
          {
            cycle_success = true;
            completed_object_ids.insert(active_object.object_id);
            deferred_object_ids.erase(active_object.object_id);
            RCLCPP_INFO(
              logger, "Sorted '%s' into '%s'. Completed objects: %zu.", active_object.object_id.c_str(),
              object_config.bin_name.c_str(), completed_object_ids.size());
            break;
          }

          RCLCPP_WARN(
            logger,
            "Candidate %zu for '%s' failed with reason '%s'. Waiting for a fresh frame from the current pose.",
            candidate_index, active_object.object_id.c_str(), failure_reason.c_str());
          if (failure_reason == "attached_recovery_failed")
          {
            RCLCPP_ERROR(
              logger,
              "Stopping the vision sort loop because '%s' may still be attached. Manual reset is "
              "safer than opening the gripper or planning the next candidate.",
              active_object.object_id.c_str());
            cycle_abort = true;
            break;
          }
          if (config.return_to_scan_after_candidate_failure)
          {
            RCLCPP_WARN(
              logger, "Returning to scan pose after candidate failure because it is enabled.");
            sort_demo::plan_and_execute_named_target(logger, arm_group, config.scan_named_target);
            skipped_scan_after_success = false;
          }

          bool object_still_visible = true;
          if (object_present_in_new_frame(
                logger, cache, config, active_object.object_id, object_still_visible) &&
              !object_still_visible)
          {
            cycle_success = true;
            completed_object_ids.insert(active_object.object_id);
            deferred_object_ids.erase(active_object.object_id);
            RCLCPP_INFO(
              logger,
              "Treating '%s' as sorted after candidate failure because it disappeared from the latest "
              "perception frame. Completed objects: %zu.",
              active_object.object_id.c_str(), completed_object_ids.size());
            break;
          }

          deferred_object_ids.insert(active_object.object_id);
          RCLCPP_WARN(
            logger,
            "Deferring '%s' after failed candidate so other visible objects can be attempted first.",
            active_object.object_id.c_str());
          cycle_rescan = true;
          break;
        }

        if (cycle_abort)
        {
          break;
        }

        if (cycle_rescan)
        {
          break;
        }

        if (cycle_success)
        {
          if (config.return_to_scan_after_success)
          {
            sort_demo::plan_and_execute_named_target(logger, arm_group, config.scan_named_target);
            skipped_scan_after_success = false;
          }
          else
          {
            skipped_scan_after_success = true;
            RCLCPP_INFO(
              logger,
              "Skipping post-sort return to scan pose. The next object will be attempted from the current retreat pose.");
          }
          break;
        }
      }

      if (cycle_abort)
      {
        break;
      }

      if (!cycle_success)
      {
        RCLCPP_WARN(logger, "No candidate succeeded in the current cycle. Waiting for a fresh frame.");
        continue;
      }
    }
  } while (false);

  subscription.reset();
  arm_group.stop();
  gripper_group.stop();
  executor.cancel();
  if (spinner.joinable())
  {
    spinner.join();
  }

  return success;
}

}  // namespace manip_sort_pipeline::vision
