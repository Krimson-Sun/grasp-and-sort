#include "manip_sort_pipeline/sort_demo/scene.hpp"

#include <algorithm>
#include <chrono>
#include <thread>

#include <moveit_msgs/msg/planning_scene.hpp>
#include <moveit_msgs/msg/planning_scene_components.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

using namespace std::chrono_literals;

namespace manip_sort_pipeline::sort_demo
{
namespace
{

moveit_msgs::msg::CollisionObject make_box(
  const std::string& id,
  double x,
  double y,
  double z,
  const BoxDimensions& dimensions,
  const std::string& frame_id = "base_link")
{
  moveit_msgs::msg::CollisionObject object;
  object.id = id;
  object.header.frame_id = frame_id;

  shape_msgs::msg::SolidPrimitive primitive;
  primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
  primitive.dimensions = {dimensions.x, dimensions.y, dimensions.z};

  geometry_msgs::msg::Pose pose;
  pose.orientation.w = 1.0;
  pose.position.x = x;
  pose.position.y = y;
  pose.position.z = z;

  object.primitives.push_back(primitive);
  object.primitive_poses.push_back(pose);
  object.operation = moveit_msgs::msg::CollisionObject::ADD;
  return object;
}

bool fetch_allowed_collision_matrix(
  const rclcpp::Logger& logger,
  const rclcpp::Client<GetPlanningScene>::SharedPtr& get_planning_scene_client,
  moveit_msgs::msg::AllowedCollisionMatrix& acm)
{
  if (!get_planning_scene_client)
  {
    RCLCPP_ERROR(logger, "GetPlanningScene client is not available.");
    return false;
  }

  if (!get_planning_scene_client->wait_for_service(2s))
  {
    RCLCPP_ERROR(logger, "GetPlanningScene service is not available.");
    return false;
  }

  auto request = std::make_shared<GetPlanningScene::Request>();
  request->components.components =
    moveit_msgs::msg::PlanningSceneComponents::ALLOWED_COLLISION_MATRIX;

  auto future = get_planning_scene_client->async_send_request(request);
  if (future.wait_for(2s) != std::future_status::ready)
  {
    RCLCPP_ERROR(logger, "Timed out while requesting the current allowed collision matrix.");
    return false;
  }

  const auto response = future.get();
  acm = response->scene.allowed_collision_matrix;
  return true;
}

std::size_t ensure_acm_entry(
  moveit_msgs::msg::AllowedCollisionMatrix& acm,
  const std::string& name)
{
  const auto existing = std::find(acm.entry_names.begin(), acm.entry_names.end(), name);
  if (existing != acm.entry_names.end())
  {
    return static_cast<std::size_t>(std::distance(acm.entry_names.begin(), existing));
  }

  const std::size_t previous_size = acm.entry_names.size();
  acm.entry_names.push_back(name);

  for (auto& entry : acm.entry_values)
  {
    entry.enabled.push_back(false);
  }

  moveit_msgs::msg::AllowedCollisionEntry new_entry;
  new_entry.enabled.resize(previous_size + 1, false);
  acm.entry_values.push_back(new_entry);

  return previous_size;
}

void set_acm_pair(
  moveit_msgs::msg::AllowedCollisionMatrix& acm,
  const std::string& first,
  const std::string& second,
  bool allowed)
{
  const auto first_index = ensure_acm_entry(acm, first);
  const auto second_index = ensure_acm_entry(acm, second);

  acm.entry_values[first_index].enabled[second_index] = allowed;
  acm.entry_values[second_index].enabled[first_index] = allowed;
}

}  // namespace

geometry_msgs::msg::Pose make_pose(
  double x,
  double y,
  double z,
  const geometry_msgs::msg::Quaternion& orientation)
{
  geometry_msgs::msg::Pose pose;
  pose.position.x = x;
  pose.position.y = y;
  pose.position.z = z;
  pose.orientation = orientation;
  return pose;
}

geometry_msgs::msg::Quaternion make_quaternion_from_axes(
  const Eigen::Vector3d& x_axis,
  const Eigen::Vector3d& y_axis,
  const Eigen::Vector3d& z_axis)
{
  Eigen::Matrix3d rotation;
  rotation.col(0) = x_axis.normalized();
  rotation.col(1) = y_axis.normalized();
  rotation.col(2) = z_axis.normalized();

  const Eigen::Quaterniond quaternion(rotation);
  geometry_msgs::msg::Quaternion result;
  result.x = quaternion.x();
  result.y = quaternion.y();
  result.z = quaternion.z();
  result.w = quaternion.w();
  return result;
}

std::vector<std::string> get_touch_links()
{
  return {
    "gripper_tcp",
    "robotiq_85_base_link",
    "robotiq_85_left_knuckle_link",
    "robotiq_85_right_knuckle_link",
    "robotiq_85_left_finger_link",
    "robotiq_85_right_finger_link",
    "robotiq_85_left_inner_knuckle_link",
    "robotiq_85_right_inner_knuckle_link",
    "robotiq_85_left_finger_tip_link",
    "robotiq_85_right_finger_tip_link"};
}

std::map<std::string, geometry_msgs::msg::Pose> get_initial_cube_poses()
{
  constexpr double work_surface_z = 0.0;
  geometry_msgs::msg::Quaternion identity_orientation;
  identity_orientation.w = 1.0;

  return {
    {"cube_small", make_pose(0.52, -0.40, work_surface_z + 0.04 / 2.0, identity_orientation)},
    {"cube_medium", make_pose(0.62, -0.30, work_surface_z + 0.05 / 2.0, identity_orientation)},
    {"cube_large", make_pose(0.72, -0.20, work_surface_z + 0.06 / 2.0, identity_orientation)},
  };
}

std::vector<geometry_msgs::msg::Quaternion> get_vertical_grasp_orientations()
{
  return {
    make_quaternion_from_axes(
      Eigen::Vector3d(0.0, 0.0, -1.0), Eigen::Vector3d(0.0, -1.0, 0.0),
      Eigen::Vector3d(-1.0, 0.0, 0.0)),
    make_quaternion_from_axes(
      Eigen::Vector3d(0.0, 0.0, -1.0), Eigen::Vector3d(0.0, 1.0, 0.0),
      Eigen::Vector3d(1.0, 0.0, 0.0)),
    make_quaternion_from_axes(
      Eigen::Vector3d(0.0, 0.0, -1.0), Eigen::Vector3d(1.0, 0.0, 0.0),
      Eigen::Vector3d(0.0, -1.0, 0.0)),
    make_quaternion_from_axes(
      Eigen::Vector3d(0.0, 0.0, -1.0), Eigen::Vector3d(-1.0, 0.0, 0.0),
      Eigen::Vector3d(0.0, 1.0, 0.0)),
  };
}

bool wait_for_objects(
  const rclcpp::Logger& logger,
  moveit::planning_interface::PlanningSceneInterface& planning_scene_interface,
  const std::vector<std::string>& object_ids)
{
  for (int attempt = 1; attempt <= 20; ++attempt)
  {
    const auto objects = planning_scene_interface.getObjects(object_ids);
    if (objects.size() == object_ids.size())
    {
      RCLCPP_INFO(logger, "Planning scene contains all %zu expected objects.", object_ids.size());
      return true;
    }

    RCLCPP_INFO(logger, "Waiting for planning scene objects... (%d/20)", attempt);
    std::this_thread::sleep_for(500ms);
  }

  return false;
}

bool set_grasp_object_collisions(
  const rclcpp::Logger& logger,
  moveit::planning_interface::PlanningSceneInterface& planning_scene_interface,
  const rclcpp::Client<GetPlanningScene>::SharedPtr& get_planning_scene_client,
  const std::string& object_id,
  const std::vector<std::string>& touch_links,
  bool allowed)
{
  moveit_msgs::msg::PlanningScene planning_scene;
  planning_scene.is_diff = true;
  if (!fetch_allowed_collision_matrix(
        logger, get_planning_scene_client, planning_scene.allowed_collision_matrix))
  {
    return false;
  }

  for (const auto& link : touch_links)
  {
    set_acm_pair(planning_scene.allowed_collision_matrix, object_id, link, allowed);
  }

  if (!planning_scene_interface.applyPlanningScene(planning_scene))
  {
    RCLCPP_ERROR(
      logger, "Failed to update allowed collisions for '%s' (allowed=%s).", object_id.c_str(),
      allowed ? "true" : "false");
    return false;
  }

  RCLCPP_INFO(
    logger, "Allowed collisions for '%s' with gripper links set to %s.", object_id.c_str(),
    allowed ? "true" : "false");
  return true;
}

bool set_surface_collisions(
  const rclcpp::Logger& logger,
  moveit::planning_interface::PlanningSceneInterface& planning_scene_interface,
  const rclcpp::Client<GetPlanningScene>::SharedPtr& get_planning_scene_client,
  const std::string& surface_id,
  const std::vector<std::string>& touch_links,
  bool allowed)
{
  moveit_msgs::msg::PlanningScene planning_scene;
  planning_scene.is_diff = true;
  if (!fetch_allowed_collision_matrix(
        logger, get_planning_scene_client, planning_scene.allowed_collision_matrix))
  {
    return false;
  }

  for (const auto& link : touch_links)
  {
    set_acm_pair(planning_scene.allowed_collision_matrix, surface_id, link, allowed);
  }

  if (!planning_scene_interface.applyPlanningScene(planning_scene))
  {
    RCLCPP_ERROR(
      logger, "Failed to update allowed collisions for surface '%s' (allowed=%s).",
      surface_id.c_str(), allowed ? "true" : "false");
    return false;
  }

  RCLCPP_INFO(
    logger, "Allowed collisions for surface '%s' with gripper links set to %s.",
    surface_id.c_str(), allowed ? "true" : "false");
  return true;
}

bool set_collision_pair(
  const rclcpp::Logger& logger,
  moveit::planning_interface::PlanningSceneInterface& planning_scene_interface,
  const rclcpp::Client<GetPlanningScene>::SharedPtr& get_planning_scene_client,
  const std::string& first_id,
  const std::string& second_id,
  bool allowed)
{
  moveit_msgs::msg::PlanningScene planning_scene;
  planning_scene.is_diff = true;
  if (!fetch_allowed_collision_matrix(
        logger, get_planning_scene_client, planning_scene.allowed_collision_matrix))
  {
    return false;
  }

  set_acm_pair(planning_scene.allowed_collision_matrix, first_id, second_id, allowed);

  if (!planning_scene_interface.applyPlanningScene(planning_scene))
  {
    RCLCPP_ERROR(
      logger, "Failed to update allowed collision pair '%s' <-> '%s' (allowed=%s).",
      first_id.c_str(), second_id.c_str(), allowed ? "true" : "false");
    return false;
  }

  RCLCPP_INFO(
    logger, "Allowed collision pair '%s' <-> '%s' set to %s.", first_id.c_str(),
    second_id.c_str(), allowed ? "true" : "false");
  return true;
}

std::vector<moveit_msgs::msg::CollisionObject> build_scene_objects()
{
  constexpr double table_height = 0.76;
  constexpr double work_surface_z = -0.002;
  constexpr double table_size_x = 0.60;
  constexpr double table_size_y = 0.50;
  constexpr double wall_height = 0.10;
  constexpr double wall_thickness = 0.02;

  std::vector<moveit_msgs::msg::CollisionObject> objects;
  objects.reserve(11);

  objects.push_back(make_box(
    "source_table", 0.62, -0.30, work_surface_z - table_height / 2.0,
    {table_size_x, table_size_y, table_height}));
  objects.push_back(make_box(
    "target_table", 0.62, 0.36, work_surface_z - table_height / 2.0,
    {table_size_x, table_size_y, table_height}));

  objects.push_back(make_box(
    "target_sector_front_wall", 0.62, 0.19, work_surface_z + wall_height / 2.0,
    {0.32, wall_thickness, wall_height}));
  objects.push_back(make_box(
    "target_sector_back_wall", 0.62, 0.53, work_surface_z + wall_height / 2.0,
    {0.32, wall_thickness, wall_height}));
  objects.push_back(make_box(
    "target_sector_left_wall", 0.47, 0.36, work_surface_z + wall_height / 2.0,
    {wall_thickness, 0.34, wall_height}));
  objects.push_back(make_box(
    "target_sector_right_wall", 0.77, 0.36, work_surface_z + wall_height / 2.0,
    {wall_thickness, 0.34, wall_height}));
  objects.push_back(make_box(
    "target_sector_divider_1", 0.62, 0.30, work_surface_z + wall_height / 2.0,
    {0.32, wall_thickness, wall_height}));
  objects.push_back(make_box(
    "target_sector_divider_2", 0.62, 0.42, work_surface_z + wall_height / 2.0,
    {0.32, wall_thickness, wall_height}));

  objects.push_back(make_box("cube_small", 0.52, -0.40, work_surface_z + 0.04 / 2.0, {0.04, 0.04, 0.04}));
  objects.push_back(make_box("cube_medium", 0.62, -0.30, work_surface_z + 0.05 / 2.0, {0.05, 0.05, 0.05}));
  objects.push_back(make_box("cube_large", 0.72, -0.20, work_surface_z + 0.06 / 2.0, {0.06, 0.06, 0.06}));

  return objects;
}

std::vector<std::string> build_scene_ids()
{
  return {
    "source_table",
    "target_table",
    "target_sector_front_wall",
    "target_sector_back_wall",
    "target_sector_left_wall",
    "target_sector_right_wall",
    "target_sector_divider_1",
    "target_sector_divider_2",
    "cube_small",
    "cube_medium",
    "cube_large"};
}

}  // namespace manip_sort_pipeline::sort_demo
