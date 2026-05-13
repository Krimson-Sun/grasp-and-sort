#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <moveit_msgs/srv/get_planning_scene.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <simulation_interfaces/srv/set_entity_state.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/empty.hpp>

namespace manip_sort_pipeline::sort_demo
{

struct BoxDimensions
{
  double x;
  double y;
  double z;
};

struct SortingTask
{
  std::string cube_id;
  geometry_msgs::msg::Pose pick_pose;
  geometry_msgs::msg::Pose place_pose;
  std::vector<geometry_msgs::msg::Pose> transfer_waypoints;
};

using GripperActionClient = rclcpp_action::Client<control_msgs::action::FollowJointTrajectory>;
using EmptyPublisher = rclcpp::Publisher<std_msgs::msg::Empty>;
using SetEntityState = simulation_interfaces::srv::SetEntityState;
using GetPlanningScene = moveit_msgs::srv::GetPlanningScene;

struct GazeboDetachInterface
{
  std::map<std::string, EmptyPublisher::SharedPtr> attach_publishers;
  std::map<std::string, EmptyPublisher::SharedPtr> detach_publishers;
  std::map<std::string, rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr> state_subscribers;
  std::map<std::string, bool> attached_states;
  std::map<std::string, bool> state_received;
  std::mutex state_mutex;
  rclcpp::Client<SetEntityState>::SharedPtr set_entity_state_client;
  std::string set_entity_state_service_name;
};

struct SortExecutionOptions
{
  double approach_z = 0.28;
  double dwell_seconds = 0.8;
  bool cartesian_avoid_collisions = true;
  double pick_retry_step = 0.005;
  int pick_retry_count = 4;
};

struct PlannerCandidate
{
  std::string pipeline_id;
  std::string planner_id;
  int attempts = 1;
};

struct TransferScoreWeights
{
  double detour_ratio = 3.0;
  double joint_path_length = 0.4;
  double duration = 0.2;
  double z_overshoot = 2.0;
  double max_detour_ratio = 3.0;
  double max_z_overshoot = 0.35;
  double max_duration = 0.0;
};

struct SortDemoConfig
{
  std::string arm_group_name = "ur_manipulator";
  std::string gripper_group_name = "gripper";
  std::string end_effector_link = "robotiq_85_base_link";
  std::string pose_reference_frame = "base_link";
  std::string gripper_action_name = "/gripper_controller/follow_joint_trajectory";
  double planning_time = 12.0;
  int planning_attempts = 10;
  double arm_velocity_scaling = 0.12;
  double arm_acceleration_scaling = 0.12;
  double gripper_velocity_scaling = 1.0;
  double gripper_acceleration_scaling = 1.0;
  double goal_position_tolerance = 0.005;
  double goal_orientation_tolerance = 0.03;
  double gripper_joint_tolerance = 0.01;
  double gripper_base_to_grasp_plane = 0.104;
  double grasp_surface_clearance = 0.01;
  SortExecutionOptions execution_options;
};

}  // namespace manip_sort_pipeline::sort_demo
