#pragma once

#include <memory>

#include <rclcpp/rclcpp.hpp>

namespace manip_sort_pipeline::vision
{

bool run_vision_sort_manager(const std::shared_ptr<rclcpp::Node>& node);

}  // namespace manip_sort_pipeline::vision
