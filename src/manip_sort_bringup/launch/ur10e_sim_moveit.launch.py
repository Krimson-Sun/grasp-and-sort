from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    gazebo_gui = LaunchConfiguration("gazebo_gui")
    launch_rviz = LaunchConfiguration("launch_rviz")
    launch_servo = LaunchConfiguration("launch_servo")
    world_file = LaunchConfiguration("world_file")
    initial_joint_controller = LaunchConfiguration("initial_joint_controller")
    activate_joint_controller = LaunchConfiguration("activate_joint_controller")
    description_file = LaunchConfiguration("description_file")
    controllers_file = LaunchConfiguration("controllers_file")
    ur_type = LaunchConfiguration("ur_type")
    gz_partition = LaunchConfiguration("gz_partition")

    ur_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("ur_simulation_gz"), "launch", "ur_sim_control.launch.py"]
            )
        ),
        launch_arguments={
            "ur_type": ur_type,
            "gazebo_gui": gazebo_gui,
            "launch_rviz": "false",
            "world_file": world_file,
            "activate_joint_controller": activate_joint_controller,
            "initial_joint_controller": initial_joint_controller,
            "description_file": description_file,
            "controllers_file": controllers_file,
        }.items(),
    )

    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("manip_sort_bringup"), "launch", "ur10e_moveit.launch.py"]
            )
        ),
        launch_arguments={
            "launch_rviz": launch_rviz,
            "launch_servo": launch_servo,
            "use_sim_time": "true",
            "description_file": description_file,
            "controllers_file": controllers_file,
            "ur_type": ur_type,
        }.items(),
    )

    gripper_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "gripper_controller",
            "--controller-manager",
            "/controller_manager",
            "--controller-manager-timeout",
            "120",
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("gazebo_gui", default_value="true"),
            DeclareLaunchArgument("launch_rviz", default_value="true"),
            DeclareLaunchArgument("launch_servo", default_value="false"),
            DeclareLaunchArgument("ur_type", default_value="ur10e"),
            DeclareLaunchArgument("gz_partition", default_value="manip_sort_ur10e"),
            DeclareLaunchArgument("activate_joint_controller", default_value="true"),
            DeclareLaunchArgument(
                "initial_joint_controller",
                default_value="scaled_joint_trajectory_controller",
            ),
            DeclareLaunchArgument(
                "description_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("manip_sort_description"),
                        "urdf",
                        "ur10e_parallel_gripper.urdf.xacro",
                    ]
                ),
            ),
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
            DeclareLaunchArgument("world_file", default_value="empty.sdf"),
            SetEnvironmentVariable("GZ_PARTITION", gz_partition),
            ur_control_launch,
            moveit_launch,
            TimerAction(period=15.0, actions=[gripper_controller_spawner]),
        ]
    )
