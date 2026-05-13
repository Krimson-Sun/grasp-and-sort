#pragma once

#include <map>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose.hpp>

#include "manip_sort_pipeline/sort_demo/types.hpp"

namespace manip_sort_pipeline::vision
{

struct VisionObjectConfig
{
  std::string object_id;
  std::string class_name;
  std::string color_name;
  int bin_id = 0;
  std::string bin_name;
  geometry_msgs::msg::Pose initial_pose;
  sort_demo::BoxDimensions collision_dimensions{0.05, 0.05, 0.02};
};

struct VisionExecutionTask
{
  std::string object_id;
  geometry_msgs::msg::Pose grasp_pose;
  geometry_msgs::msg::Pose place_pose;
};

struct VisionManagerConfig
{
  sort_demo::SortDemoConfig sort_demo_config;
  std::string scan_named_target = "up";
  std::string perception_topic = "/detected_objects";
  double scan_timeout = 2.0;
  double perception_stale_after = 1.0;
  double pregrasp_z_offset = 0.14;
  double lift_z_offset = 0.16;
  double place_z_offset = 0.12;
  double retreat_z_offset = 0.16;
  bool dry_run = false;
  bool reset_objects_on_startup = true;
  bool capture_decision_frames = true;
  bool return_to_scan_after_success = false;
  bool return_to_scan_after_candidate_failure = false;
  bool recover_to_scan_after_skipped_success_timeout = false;
  std::string decision_capture_topic = "/debug/capture_decision_frame";
  std::vector<sort_demo::PlannerCandidate> pregrasp_planners{
    {"pilz_industrial_motion_planner", "PTP", 1},
    {"ompl", "", 4},
  };
  sort_demo::TransferScoreWeights pregrasp_score_weights{
    1.0,
    0.8,
    0.2,
    0.2,
    4.0,
    0.60,
    0.0,
  };
  std::vector<sort_demo::PlannerCandidate> transfer_planners{
    {"pilz_industrial_motion_planner", "LIN", 1},
    {"pilz_industrial_motion_planner", "PTP", 1},
    {"ompl", "", 4},
  };
  sort_demo::TransferScoreWeights transfer_score_weights;
  std::map<std::string, geometry_msgs::msg::Pose> bin_place_poses;
  std::vector<VisionObjectConfig> objects;
};

}  // namespace manip_sort_pipeline::vision
