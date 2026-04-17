#pragma once

#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "manip_sort_pipeline/sort_demo/types.hpp"

namespace manip_sort_pipeline::sort_demo
{

bool wait_for_gazebo_detach_interface(
  const std::shared_ptr<rclcpp::Node>& node,
  const rclcpp::Logger& logger,
  GazeboDetachInterface& gazebo_detach_interface,
  const std::vector<std::string>& cube_ids);

bool request_cube_attach_state(
  const rclcpp::Logger& logger,
  GazeboDetachInterface& gazebo_detach_interface,
  const std::string& cube_id,
  bool attach,
  int max_attempts = 5,
  bool require_confirmation = true);

bool reset_detachable_cubes_at_startup(
  const rclcpp::Logger& logger,
  GazeboDetachInterface& gazebo_detach_interface,
  const std::vector<std::string>& cube_ids);

}  // namespace manip_sort_pipeline::sort_demo
