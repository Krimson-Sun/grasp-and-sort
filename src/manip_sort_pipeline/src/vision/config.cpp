#include "manip_sort_pipeline/vision/config.hpp"

#include <algorithm>
#include <chrono>
#include <future>
#include <sstream>
#include <string>
#include <thread>

#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/empty.hpp>

#include "manip_sort_pipeline/sort_demo/gazebo.hpp"
#include "manip_sort_pipeline/sort_demo/runner.hpp"

using namespace std::chrono_literals;

namespace manip_sort_pipeline::vision
{
namespace
{

geometry_msgs::msg::Pose pose_from_list(const std::vector<double>& values)
{
  geometry_msgs::msg::Pose pose;
  if (values.size() >= 3)
  {
    pose.position.x = values[0];
    pose.position.y = values[1];
    pose.position.z = values[2];
  }
  if (values.size() >= 7)
  {
    pose.orientation.x = values[3];
    pose.orientation.y = values[4];
    pose.orientation.z = values[5];
    pose.orientation.w = values[6];
  }
  else
  {
    pose.orientation.w = 1.0;
  }
  return pose;
}

std::vector<std::string> serialize_planner_candidates(
  const std::vector<sort_demo::PlannerCandidate>& candidates)
{
  std::vector<std::string> specs;
  specs.reserve(candidates.size());
  for (const auto& candidate : candidates)
  {
    specs.push_back(
      candidate.pipeline_id + ":" + candidate.planner_id + ":" +
      std::to_string(candidate.attempts));
  }
  return specs;
}

bool parse_planner_candidate(
  const std::string& spec,
  sort_demo::PlannerCandidate& candidate)
{
  std::vector<std::string> fields;
  std::stringstream stream(spec);
  std::string field;
  while (std::getline(stream, field, ':'))
  {
    fields.push_back(field);
  }

  if (fields.empty() || fields[0].empty())
  {
    return false;
  }

  candidate.pipeline_id = fields[0];
  candidate.planner_id = fields.size() > 1 ? fields[1] : "";
  candidate.attempts = 1;
  if (fields.size() > 2 && !fields[2].empty())
  {
    try
    {
      candidate.attempts = std::max(1, std::stoi(fields[2]));
    }
    catch (const std::exception&)
    {
      return false;
    }
  }

  return true;
}

std::vector<sort_demo::PlannerCandidate> load_planner_candidates(
  rclcpp::Node& node,
  const std::string& parameter_name,
  const std::vector<sort_demo::PlannerCandidate>& defaults)
{
  const auto planner_specs = node.declare_parameter<std::vector<std::string>>(
    parameter_name, serialize_planner_candidates(defaults));
  std::vector<sort_demo::PlannerCandidate> result;
  for (const auto& spec : planner_specs)
  {
    sort_demo::PlannerCandidate candidate;
    if (parse_planner_candidate(spec, candidate))
    {
      result.push_back(candidate);
      continue;
    }

    RCLCPP_WARN(
      node.get_logger(),
      "Ignoring invalid %s spec '%s'. Expected 'pipeline:planner:attempts'.",
      parameter_name.c_str(), spec.c_str());
  }

  if (result.empty())
  {
    RCLCPP_WARN(
      node.get_logger(),
      "No valid %s specs were provided. Falling back to ompl with one attempt.",
      parameter_name.c_str());
    result.push_back({"ompl", "", 1});
  }

  return result;
}

sort_demo::TransferScoreWeights load_score_weights(
  rclcpp::Node& node,
  const std::string& prefix,
  const sort_demo::TransferScoreWeights& defaults)
{
  sort_demo::TransferScoreWeights result;
  result.detour_ratio =
    node.declare_parameter<double>(prefix + "_detour_ratio_weight", defaults.detour_ratio);
  result.joint_path_length =
    node.declare_parameter<double>(prefix + "_joint_path_length_weight", defaults.joint_path_length);
  result.duration =
    node.declare_parameter<double>(prefix + "_duration_weight", defaults.duration);
  result.z_overshoot =
    node.declare_parameter<double>(prefix + "_z_overshoot_weight", defaults.z_overshoot);
  result.max_detour_ratio =
    node.declare_parameter<double>(prefix + "_max_detour_ratio", defaults.max_detour_ratio);
  result.max_z_overshoot =
    node.declare_parameter<double>(prefix + "_max_z_overshoot", defaults.max_z_overshoot);
  result.max_duration =
    node.declare_parameter<double>(prefix + "_max_duration", defaults.max_duration);
  return result;
}

moveit_msgs::msg::CollisionObject make_box(
  const std::string& id,
  const geometry_msgs::msg::Pose& pose,
  const sort_demo::BoxDimensions& dimensions,
  const std::string& frame_id = "base_link")
{
  moveit_msgs::msg::CollisionObject object;
  object.id = id;
  object.header.frame_id = frame_id;

  shape_msgs::msg::SolidPrimitive primitive;
  primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
  primitive.dimensions = {dimensions.x, dimensions.y, dimensions.z};

  object.primitives.push_back(primitive);
  object.primitive_poses.push_back(pose);
  object.operation = moveit_msgs::msg::CollisionObject::ADD;
  return object;
}

}  // namespace

VisionManagerConfig load_vision_manager_config(rclcpp::Node& node)
{
  VisionManagerConfig config;
  config.sort_demo_config = sort_demo::load_sort_demo_config(node);

  config.scan_named_target =
    node.declare_parameter<std::string>("scan_named_target", config.scan_named_target);
  config.perception_topic =
    node.declare_parameter<std::string>("perception_topic", config.perception_topic);
  config.scan_timeout = node.declare_parameter<double>("scan_timeout", config.scan_timeout);
  config.perception_stale_after =
    node.declare_parameter<double>("perception_stale_after", config.perception_stale_after);
  config.pregrasp_z_offset =
    node.declare_parameter<double>("pregrasp_z_offset", config.pregrasp_z_offset);
  config.lift_z_offset = node.declare_parameter<double>("lift_z_offset", config.lift_z_offset);
  config.place_z_offset = node.declare_parameter<double>("place_z_offset", config.place_z_offset);
  config.retreat_z_offset =
    node.declare_parameter<double>("retreat_z_offset", config.retreat_z_offset);
  config.pregrasp_planners =
    load_planner_candidates(node, "pregrasp_planners", config.pregrasp_planners);
  config.pregrasp_score_weights =
    load_score_weights(node, "pregrasp_score", config.pregrasp_score_weights);
  config.transfer_planners =
    load_planner_candidates(node, "transfer_planners", config.transfer_planners);
  config.transfer_score_weights =
    load_score_weights(node, "transfer_score", config.transfer_score_weights);
  config.dry_run = node.declare_parameter<bool>("dry_run", config.dry_run);
  config.reset_objects_on_startup =
    node.declare_parameter<bool>("reset_objects_on_startup", config.reset_objects_on_startup);
  config.capture_decision_frames =
    node.declare_parameter<bool>("capture_decision_frames", config.capture_decision_frames);
  config.return_to_scan_after_success = node.declare_parameter<bool>(
    "return_to_scan_after_success", config.return_to_scan_after_success);
  config.return_to_scan_after_candidate_failure = node.declare_parameter<bool>(
    "return_to_scan_after_candidate_failure", config.return_to_scan_after_candidate_failure);
  config.recover_to_scan_after_skipped_success_timeout = node.declare_parameter<bool>(
    "recover_to_scan_after_skipped_success_timeout",
    config.recover_to_scan_after_skipped_success_timeout);
  config.decision_capture_topic = node.declare_parameter<std::string>(
    "decision_capture_topic", config.decision_capture_topic);

  const auto object_ids =
    node.declare_parameter<std::vector<std::string>>("object_ids", std::vector<std::string>{});
  const auto bin_names =
    node.declare_parameter<std::vector<std::string>>("bin_names", std::vector<std::string>{});

  for (const auto& bin_name : bin_names)
  {
    config.bin_place_poses[bin_name] = pose_from_list(
      node.declare_parameter<std::vector<double>>("bins." + bin_name + ".place_pose", {0, 0, 0, 0, 0, 0, 1}));
  }

  for (const auto& object_id : object_ids)
  {
    VisionObjectConfig object;
    object.object_id = object_id;
    object.class_name =
      node.declare_parameter<std::string>("objects." + object_id + ".class_name", object_id);
    object.color_name =
      node.declare_parameter<std::string>("objects." + object_id + ".color_name", object_id);
    object.bin_id = node.declare_parameter<int>("objects." + object_id + ".bin_id", 0);
    object.bin_name =
      node.declare_parameter<std::string>("objects." + object_id + ".bin_name", "");
    object.initial_pose = pose_from_list(
      node.declare_parameter<std::vector<double>>(
        "objects." + object_id + ".initial_pose", {0, 0, 0, 0, 0, 0, 1}));
    const auto dimensions = node.declare_parameter<std::vector<double>>(
      "objects." + object_id + ".collision_dimensions", {0.05, 0.05, 0.02});
    if (dimensions.size() >= 3)
    {
      object.collision_dimensions = {dimensions[0], dimensions[1], dimensions[2]};
    }
    config.objects.push_back(object);
  }

  return config;
}

std::vector<std::string> get_object_ids(const VisionManagerConfig& config)
{
  std::vector<std::string> ids;
  ids.reserve(config.objects.size());
  for (const auto& object : config.objects)
  {
    ids.push_back(object.object_id);
  }
  return ids;
}

std::map<std::string, VisionObjectConfig> build_object_lookup(const VisionManagerConfig& config)
{
  std::map<std::string, VisionObjectConfig> lookup;
  for (const auto& object : config.objects)
  {
    lookup[object.object_id] = object;
  }
  return lookup;
}

std::vector<moveit_msgs::msg::CollisionObject> build_vision_scene_objects(const VisionManagerConfig& config)
{
  constexpr double table_height = 0.76;
  constexpr double table_size_x = 0.60;
  constexpr double table_size_y = 0.50;
  constexpr double work_surface_z = -0.002;
  constexpr double wall_height = 0.10;
  constexpr double wall_thickness = 0.02;

  std::vector<moveit_msgs::msg::CollisionObject> objects;
  objects.reserve(10 + config.objects.size());

  geometry_msgs::msg::Pose pose;
  pose.orientation.w = 1.0;

  pose.position.x = 0.62;
  pose.position.y = -0.30;
  pose.position.z = work_surface_z - table_height / 2.0;
  objects.push_back(make_box("source_table", pose, {table_size_x, table_size_y, table_height}));

  pose.position.x = 0.62;
  pose.position.y = 0.36;
  pose.position.z = work_surface_z - table_height / 2.0;
  objects.push_back(make_box("target_table", pose, {table_size_x, table_size_y, table_height}));

  pose.position.x = 0.62;
  pose.position.y = 0.19;
  pose.position.z = work_surface_z + wall_height / 2.0;
  objects.push_back(make_box("target_sector_front_wall", pose, {0.32, wall_thickness, wall_height}));

  pose.position.x = 0.62;
  pose.position.y = 0.53;
  pose.position.z = work_surface_z + wall_height / 2.0;
  objects.push_back(make_box("target_sector_back_wall", pose, {0.32, wall_thickness, wall_height}));

  pose.position.x = 0.47;
  pose.position.y = 0.36;
  pose.position.z = work_surface_z + wall_height / 2.0;
  objects.push_back(make_box("target_sector_left_wall", pose, {wall_thickness, 0.34, wall_height}));

  pose.position.x = 0.77;
  pose.position.y = 0.36;
  pose.position.z = work_surface_z + wall_height / 2.0;
  objects.push_back(make_box("target_sector_right_wall", pose, {wall_thickness, 0.34, wall_height}));

  pose.position.x = 0.62;
  pose.position.y = 0.36;
  pose.position.z = work_surface_z + wall_height / 2.0;
  objects.push_back(make_box("target_sector_midline_x", pose, {wall_thickness, 0.34, wall_height}));

  pose.position.x = 0.62;
  pose.position.y = 0.36;
  pose.position.z = work_surface_z + wall_height / 2.0;
  objects.push_back(make_box("target_sector_midline_y", pose, {0.32, wall_thickness, wall_height}));

  for (const auto& object : config.objects)
  {
    objects.push_back(make_box(object.object_id, object.initial_pose, object.collision_dimensions));
  }

  return objects;
}

std::vector<std::string> build_vision_scene_ids(const VisionManagerConfig& config)
{
  std::vector<std::string> ids{
    "source_table",
    "target_table",
    "target_sector_front_wall",
    "target_sector_back_wall",
    "target_sector_left_wall",
    "target_sector_right_wall",
    "target_sector_midline_x",
    "target_sector_midline_y",
  };
  for (const auto& object : config.objects)
  {
    ids.push_back(object.object_id);
  }
  return ids;
}

void initialize_gazebo_interface_for_objects(
  const std::shared_ptr<rclcpp::Node>& node,
  sort_demo::GazeboDetachInterface& gazebo_detach_interface,
  const std::vector<std::string>& object_ids)
{
  for (const auto& object_id : object_ids)
  {
    gazebo_detach_interface.attach_publishers[object_id] =
      node->create_publisher<std_msgs::msg::Empty>("/" + object_id + "/attach", 10);
    gazebo_detach_interface.detach_publishers[object_id] =
      node->create_publisher<std_msgs::msg::Empty>("/" + object_id + "/detach", 10);
    gazebo_detach_interface.state_subscribers[object_id] =
      node->create_subscription<std_msgs::msg::Bool>(
        "/" + object_id + "/state", 10,
        [&gazebo_detach_interface, object_id](const std_msgs::msg::Bool::SharedPtr message) {
          std::scoped_lock lock(gazebo_detach_interface.state_mutex);
          gazebo_detach_interface.attached_states[object_id] = message->data;
          gazebo_detach_interface.state_received[object_id] = true;
        });
  }
}

bool reset_objects_at_startup(
  const rclcpp::Logger& logger,
  sort_demo::GazeboDetachInterface& gazebo_detach_interface,
  const VisionManagerConfig& config)
{
  if (!config.reset_objects_on_startup)
  {
    return true;
  }

  for (const auto& object : config.objects)
  {
    if (!sort_demo::request_cube_attach_state(
          logger, gazebo_detach_interface, object.object_id, false, 2, false))
    {
      return false;
    }
    std::this_thread::sleep_for(200ms);

    if (!gazebo_detach_interface.set_entity_state_client)
    {
      continue;
    }

    auto request = std::make_shared<sort_demo::SetEntityState::Request>();
    request->entity = object.object_id;
    request->state.header.frame_id = "world";
    request->state.pose = object.initial_pose;

    auto future = gazebo_detach_interface.set_entity_state_client->async_send_request(request);
    if (future.wait_for(5s) != std::future_status::ready)
    {
      RCLCPP_ERROR(logger, "Timed out resetting Gazebo pose for '%s'.", object.object_id.c_str());
      return false;
    }

    const auto response = future.get();
    if (!response)
    {
      RCLCPP_ERROR(logger, "No response received while resetting Gazebo pose for '%s'.", object.object_id.c_str());
      return false;
    }
  }

  return true;
}

}  // namespace manip_sort_pipeline::vision
