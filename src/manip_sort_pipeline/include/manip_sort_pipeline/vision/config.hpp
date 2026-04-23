#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <moveit_msgs/msg/collision_object.hpp>
#include <rclcpp/rclcpp.hpp>

#include "manip_sort_pipeline/vision/types.hpp"

namespace manip_sort_pipeline::vision
{

VisionManagerConfig load_vision_manager_config(rclcpp::Node& node);
std::vector<std::string> get_object_ids(const VisionManagerConfig& config);
std::map<std::string, VisionObjectConfig> build_object_lookup(const VisionManagerConfig& config);
std::vector<moveit_msgs::msg::CollisionObject> build_vision_scene_objects(const VisionManagerConfig& config);
std::vector<std::string> build_vision_scene_ids(const VisionManagerConfig& config);
void initialize_gazebo_interface_for_objects(
  const std::shared_ptr<rclcpp::Node>& node,
  sort_demo::GazeboDetachInterface& gazebo_detach_interface,
  const std::vector<std::string>& object_ids);
bool reset_objects_at_startup(
  const rclcpp::Logger& logger,
  sort_demo::GazeboDetachInterface& gazebo_detach_interface,
  const VisionManagerConfig& config);

}  // namespace manip_sort_pipeline::vision
