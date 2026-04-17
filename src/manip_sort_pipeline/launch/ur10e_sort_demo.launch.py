import os

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    gazebo_gui = LaunchConfiguration("gazebo_gui")
    launch_rviz = LaunchConfiguration("launch_rviz")
    launch_servo = LaunchConfiguration("launch_servo")
    world_file = LaunchConfiguration("world_file")
    demo_delay_sec = LaunchConfiguration("demo_delay_sec")
    gz_partition = LaunchConfiguration("gz_partition")
    ur_type = LaunchConfiguration("ur_type")
    controllers_file = LaunchConfiguration("controllers_file")

    moveit_config = (
        MoveItConfigsBuilder(robot_name="ur", package_name="manip_sort_description")
        .robot_description(
            file_path=Path("urdf") / "ur10e_parallel_gripper.urdf.xacro",
            mappings={
                "name": "ur",
                "ur_type": ur_type,
                "tf_prefix": "",
                "simulation_controllers": controllers_file,
                "safety_limits": "true",
                "safety_pos_margin": "0.15",
                "safety_k_position": "20",
            },
        )
        .robot_description_semantic(
            Path("srdf") / "ur_with_gripper.srdf.xacro", {"name": "ur"}
        )
        .robot_description_kinematics(Path("config") / "kinematics.yaml")
        .joint_limits(Path("config") / "joint_limits.yaml")
        .planning_pipelines(
            pipelines=["ompl", "chomp", "pilz_industrial_motion_planner", "stomp"]
        )
        .trajectory_execution(Path("config") / "moveit_controllers.yaml")
        .to_moveit_configs()
    )

    bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("manip_sort_bringup"), "launch", "ur10e_sim_moveit.launch.py"]
            )
        ),
        launch_arguments={
            "gazebo_gui": gazebo_gui,
            "launch_rviz": launch_rviz,
            "launch_servo": launch_servo,
            "world_file": world_file,
            "gz_partition": gz_partition,
            "ur_type": ur_type,
            "controllers_file": controllers_file,
        }.items(),
    )

    grasp_bridge_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("ros_gz_bridge"), "launch", "ros_gz_bridge.launch.py"]
            )
        ),
        launch_arguments={
            "bridge_name": "grasp_attach_bridge",
            "config_file": PathJoinSubstitution(
                [FindPackageShare("manip_sort_pipeline"), "config", "gz_attach_bridge.yaml"]
            ),
            "use_composition": "False",
            "create_own_container": "False",
            "log_level": "info",
        }.items(),
    )

    demo_node = Node(
        package="manip_sort_pipeline",
        executable="sort_scene_demo",
        output="screen",
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
            moveit_config.planning_pipelines,
            moveit_config.trajectory_execution,
            {
                "use_sim_time": True,
                "end_effector_link": "robotiq_85_base_link",
                "approach_z": 0.28,
                "gripper_base_to_grasp_plane": 0.104,
                "grasp_surface_clearance": 0.01,
                "cartesian_avoid_collisions": True,
                "pick_retry_step": 0.005,
                "pick_retry_count": 4,
            },
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("gazebo_gui", default_value="true"),
            DeclareLaunchArgument("launch_rviz", default_value="true"),
            DeclareLaunchArgument("launch_servo", default_value="false"),
            DeclareLaunchArgument("ur_type", default_value="ur10e"),
            DeclareLaunchArgument(
                "controllers_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("manip_sort_description"),
                        "config",
                        "ur10e_gripper_controllers.yaml",
                    ]
                ),
            ),
            DeclareLaunchArgument(
                "world_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("manip_sort_pipeline"), "worlds", "hardcoded_sort_scene.sdf"]
                ),
            ),
            DeclareLaunchArgument("gz_partition", default_value="manip_sort_ur10e_sort_demo"),
            DeclareLaunchArgument("demo_delay_sec", default_value="30.0"),
            bringup_launch,
            grasp_bridge_launch,
            TimerAction(period=demo_delay_sec, actions=[demo_node]),
        ]
    )
