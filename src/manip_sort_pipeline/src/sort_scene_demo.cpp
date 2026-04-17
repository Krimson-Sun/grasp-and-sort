#include <rclcpp/rclcpp.hpp>

#include "manip_sort_pipeline/sort_demo/runner.hpp"

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<rclcpp::Node>("sort_scene_demo");
  const bool success = manip_sort_pipeline::sort_demo::run_sort_scene_demo(node);
  rclcpp::shutdown();
  return success ? 0 : 1;
}
