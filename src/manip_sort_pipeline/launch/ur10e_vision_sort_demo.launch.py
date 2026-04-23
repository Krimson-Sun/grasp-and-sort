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
    gz_partition = LaunchConfiguration("gz_partition")
    ur_type = LaunchConfiguration("ur_type")
    controllers_file = LaunchConfiguration("controllers_file")
    startup_delay_sec = LaunchConfiguration("startup_delay_sec")
    vision_description_file = [
        PathJoinSubstitution(
            [
                FindPackageShare("manip_sort_description"),
                "urdf",
                "ur10e_parallel_gripper.urdf.xacro",
            ]
        ),
        " enable_vision_detachables:=true",
    ]

    moveit_config = (
        MoveItConfigsBuilder(robot_name="ur", package_name="manip_sort_description")
        .robot_description(
            file_path=Path("urdf") / "ur10e_parallel_gripper.urdf.xacro",
            mappings={
                "name": "ur",
                "ur_type": ur_type,
                "tf_prefix": "",
                "simulation_controllers": controllers_file,
                "enable_vision_detachables": "true",
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
            "description_file": vision_description_file,
        }.items(),
    )

    bridge_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("ros_gz_bridge"), "launch", "ros_gz_bridge.launch.py"]
            )
        ),
        launch_arguments={
            "bridge_name": "vision_sort_bridge",
            "config_file": PathJoinSubstitution(
                [FindPackageShare("manip_sort_pipeline"), "config", "gz_vision_bridge.yaml"]
            ),
            "use_composition": "False",
            "create_own_container": "False",
            "log_level": "info",
        }.items(),
    )

    static_camera_link_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "--x",
            "0.62",
            "--y",
            "-0.30",
            "--z",
            "1.05",
            "--roll",
            "0.0",
            "--pitch",
            "1.57079632679",
            "--yaw",
            "0.0",
            "--frame-id",
            "world",
            "--child-frame-id",
            "camera_link",
        ],
        output="screen",
    )

    static_camera_optical_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            "--x",
            "0.0",
            "--y",
            "0.0",
            "--z",
            "0.0",
            "--roll",
            "-1.57079632679",
            "--pitch",
            "0.0",
            "--yaw",
            "-1.57079632679",
            "--frame-id",
            "camera_link",
            "--child-frame-id",
            "camera_optical_frame",
        ],
        output="screen",
    )

    perception_node = Node(
        package="manip_sort_perception",
        executable="perception_grasp_node",
        output="screen",
        parameters=[
            PathJoinSubstitution(
                [FindPackageShare("manip_sort_perception"), "config", "perception.yaml"]
            ),
            {"use_sim_time": True},
        ],
    )

    manager_node = Node(
        package="manip_sort_pipeline",
        executable="vision_sort_manager",
        output="screen",
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
            moveit_config.planning_pipelines,
            moveit_config.trajectory_execution,
            PathJoinSubstitution(
                [FindPackageShare("manip_sort_pipeline"), "config", "scene_sorting.yaml"]
            ),
            PathJoinSubstitution(
                [FindPackageShare("manip_sort_pipeline"), "config", "bins.yaml"]
            ),
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
                    [FindPackageShare("manip_sort_pipeline"), "worlds", "vision_sort_scene.sdf"]
                ),
            ),
            DeclareLaunchArgument("gz_partition", default_value="manip_sort_ur10e_vision_sort_demo"),
            DeclareLaunchArgument("startup_delay_sec", default_value="25.0"),
            bringup_launch,
            bridge_launch,
            static_camera_link_tf,
            static_camera_optical_tf,
            TimerAction(period=startup_delay_sec, actions=[perception_node, manager_node]),
        ]
    )
