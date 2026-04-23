#include "manip_sort_pipeline/sort_demo/gazebo.hpp"

#include <algorithm>
#include <chrono>
#include <optional>
#include <thread>

#include <simulation_interfaces/msg/result.hpp>

#include "manip_sort_pipeline/sort_demo/scene.hpp"

using namespace std::chrono_literals;

namespace manip_sort_pipeline::sort_demo
{
namespace
{

std::optional<std::string> find_service_name_by_type(
  const std::shared_ptr<rclcpp::Node>& node,
  const std::string& service_type)
{
  const auto service_map = node->get_service_names_and_types();
  for (const auto& [service_name, service_types] : service_map)
  {
    if (std::find(service_types.begin(), service_types.end(), service_type) != service_types.end())
    {
      return service_name;
    }
  }

  return std::nullopt;
}

bool wait_for_bridge_subscribers(
  const rclcpp::Logger& logger,
  const EmptyPublisher::SharedPtr& publisher,
  const std::string& topic_name)
{
  constexpr int max_attempts = 30;
  for (int attempt = 1; attempt <= max_attempts; ++attempt)
  {
    if (publisher->get_subscription_count() > 0)
    {
      RCLCPP_INFO(logger, "Bridge subscriber is ready for '%s'.", topic_name.c_str());
      return true;
    }

    RCLCPP_INFO(
      logger, "Waiting for bridge subscriber on '%s'... (%d/%d)", topic_name.c_str(), attempt,
      max_attempts);
    std::this_thread::sleep_for(500ms);
  }

  return false;
}

bool publish_detach_command(
  const rclcpp::Logger& logger,
  const EmptyPublisher::SharedPtr& publisher,
  const std::string& topic_name,
  const std::string& action)
{
  if (!publisher)
  {
    RCLCPP_ERROR(logger, "Publisher for '%s' is not available.", topic_name.c_str());
    return false;
  }

  std_msgs::msg::Empty message;
  for (int i = 0; i < 3; ++i)
  {
    publisher->publish(message);
    std::this_thread::sleep_for(100ms);
  }

  RCLCPP_INFO(logger, "Published Gazebo %s command on '%s'.", action.c_str(), topic_name.c_str());
  return true;
}

bool publish_cube_attach_command(
  const rclcpp::Logger& logger,
  GazeboDetachInterface& gazebo_detach_interface,
  const std::string& cube_id,
  bool attach)
{
  const auto& publishers =
    attach ? gazebo_detach_interface.attach_publishers : gazebo_detach_interface.detach_publishers;
  const auto publisher_it = publishers.find(cube_id);
  if (publisher_it == publishers.end())
  {
    RCLCPP_ERROR(
      logger, "Gazebo detachable joint publisher for '%s' (%s) was not found.", cube_id.c_str(),
      attach ? "attach" : "detach");
    return false;
  }

  return publish_detach_command(
    logger, publisher_it->second, "/" + cube_id + (attach ? "/attach" : "/detach"),
    attach ? "attach" : "detach");
}

bool wait_for_cube_attach_state(
  const rclcpp::Logger& logger,
  GazeboDetachInterface& gazebo_detach_interface,
  const std::string& cube_id,
  bool expected_attached,
  std::chrono::milliseconds timeout = 2500ms)
{
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline)
  {
    {
      std::scoped_lock lock(gazebo_detach_interface.state_mutex);
      const auto received_it = gazebo_detach_interface.state_received.find(cube_id);
      const auto attached_it = gazebo_detach_interface.attached_states.find(cube_id);
      if (
        received_it != gazebo_detach_interface.state_received.end() && received_it->second &&
        attached_it != gazebo_detach_interface.attached_states.end() &&
        attached_it->second == expected_attached)
      {
        RCLCPP_INFO(
          logger, "Gazebo detachable state for '%s' confirmed: attached=%s.", cube_id.c_str(),
          expected_attached ? "true" : "false");
        return true;
      }
    }

    std::this_thread::sleep_for(100ms);
  }

  RCLCPP_WARN(
    logger, "Timed out waiting for Gazebo detachable state for '%s' to become attached=%s.",
    cube_id.c_str(), expected_attached ? "true" : "false");
  return false;
}

bool set_cube_world_pose(
  const rclcpp::Logger& logger,
  GazeboDetachInterface& gazebo_detach_interface,
  const std::string& cube_id,
  const geometry_msgs::msg::Pose& pose)
{
  if (!gazebo_detach_interface.set_entity_state_client)
  {
    RCLCPP_WARN(
      logger, "Skipping startup pose reset for '%s' because SetEntityState service is unavailable.",
      cube_id.c_str());
    return true;
  }

  auto request = std::make_shared<SetEntityState::Request>();
  request->entity = cube_id;
  request->state.header.frame_id = "world";
  request->state.pose = pose;

  auto future = gazebo_detach_interface.set_entity_state_client->async_send_request(request);
  if (future.wait_for(5s) != std::future_status::ready)
  {
    RCLCPP_ERROR(logger, "Timed out resetting Gazebo pose for '%s'.", cube_id.c_str());
    return false;
  }

  const auto response = future.get();
  if (!response)
  {
    RCLCPP_ERROR(logger, "No response received while resetting Gazebo pose for '%s'.", cube_id.c_str());
    return false;
  }

  if (response->result.result != simulation_interfaces::msg::Result::RESULT_OK)
  {
    RCLCPP_ERROR(
      logger, "Failed to reset Gazebo pose for '%s': %s", cube_id.c_str(),
      response->result.error_message.c_str());
    return false;
  }

  RCLCPP_INFO(
    logger, "Reset Gazebo pose for '%s' to x=%.3f y=%.3f z=%.3f.", cube_id.c_str(),
    pose.position.x, pose.position.y, pose.position.z);
  return true;
}

}  // namespace

bool wait_for_gazebo_detach_interface(
  const std::shared_ptr<rclcpp::Node>& node,
  const rclcpp::Logger& logger,
  GazeboDetachInterface& gazebo_detach_interface,
  const std::vector<std::string>& cube_ids)
{
  for (const auto& cube_id : cube_ids)
  {
    const auto attach_it = gazebo_detach_interface.attach_publishers.find(cube_id);
    const auto detach_it = gazebo_detach_interface.detach_publishers.find(cube_id);
    if (
      attach_it == gazebo_detach_interface.attach_publishers.end() ||
      detach_it == gazebo_detach_interface.detach_publishers.end())
    {
      RCLCPP_ERROR(logger, "Gazebo detachable joint publishers are missing for '%s'.", cube_id.c_str());
      return false;
    }

    if (!wait_for_bridge_subscribers(logger, attach_it->second, "/" + cube_id + "/attach"))
    {
      return false;
    }
    if (!wait_for_bridge_subscribers(logger, detach_it->second, "/" + cube_id + "/detach"))
    {
      return false;
    }
  }

  constexpr int max_attempts = 30;
  for (int attempt = 1; attempt <= max_attempts; ++attempt)
  {
    const auto service_name = find_service_name_by_type(node, "simulation_interfaces/srv/SetEntityState");
    if (service_name)
    {
      gazebo_detach_interface.set_entity_state_service_name = *service_name;
      gazebo_detach_interface.set_entity_state_client = node->create_client<SetEntityState>(*service_name);
      if (gazebo_detach_interface.set_entity_state_client->wait_for_service(1s))
      {
        RCLCPP_INFO(logger, "Gazebo entity state service is ready at '%s'.", service_name->c_str());
        return true;
      }
    }

    RCLCPP_INFO(
      logger, "Waiting for Gazebo SetEntityState service... (%d/%d)", attempt, max_attempts);
    std::this_thread::sleep_for(500ms);
  }

  RCLCPP_WARN(
    logger,
    "Gazebo SetEntityState service was not found. Continuing without startup pose reset.");
  return true;
}

bool request_cube_attach_state(
  const rclcpp::Logger& logger,
  GazeboDetachInterface& gazebo_detach_interface,
  const std::string& cube_id,
  bool attach,
  int max_attempts,
  bool require_confirmation)
{
  {
    std::scoped_lock lock(gazebo_detach_interface.state_mutex);
    gazebo_detach_interface.state_received[cube_id] = false;
  }

  for (int attempt = 1; attempt <= max_attempts; ++attempt)
  {
    if (!publish_cube_attach_command(logger, gazebo_detach_interface, cube_id, attach))
    {
      return false;
    }

    if (wait_for_cube_attach_state(logger, gazebo_detach_interface, cube_id, attach))
    {
      return true;
    }

    if (!require_confirmation)
    {
      RCLCPP_WARN(
        logger,
        "Continuing after Gazebo %s for '%s' without state confirmation.",
        attach ? "attach" : "detach", cube_id.c_str());
      return true;
    }

    RCLCPP_WARN(
      logger, "Retrying Gazebo %s for '%s' (%d/%d).", attach ? "attach" : "detach",
      cube_id.c_str(), attempt, max_attempts);
  }

  return false;
}

bool reset_detachable_cubes_at_startup(
  const rclcpp::Logger& logger,
  GazeboDetachInterface& gazebo_detach_interface,
  const std::vector<std::string>& cube_ids)
{
  const auto initial_cube_poses = get_initial_cube_poses();

  for (const auto& cube_id : cube_ids)
  {
    const auto pose_it = initial_cube_poses.find(cube_id);
    if (pose_it == initial_cube_poses.end())
    {
      continue;
    }

    if (!request_cube_attach_state(logger, gazebo_detach_interface, cube_id, false, 2, false))
    {
      return false;
    }
    std::this_thread::sleep_for(200ms);

    if (!set_cube_world_pose(logger, gazebo_detach_interface, cube_id, pose_it->second))
    {
      return false;
    }
  }

  return true;
}

}  // namespace manip_sort_pipeline::sort_demo
