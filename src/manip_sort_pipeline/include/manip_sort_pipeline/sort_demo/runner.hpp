#pragma once

#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "manip_sort_pipeline/sort_demo/types.hpp"

namespace manip_sort_pipeline::sort_demo
{

SortDemoConfig load_sort_demo_config(rclcpp::Node& node);
bool run_sort_scene_demo(const std::shared_ptr<rclcpp::Node>& node);

}  // namespace manip_sort_pipeline::sort_demo
