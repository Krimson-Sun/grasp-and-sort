#include <rclcpp/rclcpp.hpp>

#include "manip_sort_pipeline/vision/manager.hpp"

int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<rclcpp::Node>("sort_manager");
  const bool success = manip_sort_pipeline::vision::run_vision_sort_manager(node);
  rclcpp::shutdown();
  return success ? 0 : 1;
}
