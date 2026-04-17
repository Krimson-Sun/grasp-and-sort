from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    gazebo_gui = LaunchConfiguration("gazebo_gui")
    launch_rviz = LaunchConfiguration("launch_rviz")
    launch_servo = LaunchConfiguration("launch_servo")
    world_file = LaunchConfiguration("world_file")
    demo_delay_sec = LaunchConfiguration("demo_delay_sec")
    gz_partition = LaunchConfiguration("gz_partition")

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
        }.items(),
    )

    demo_node = Node(
        package="manip_sort_pipeline",
        executable="five_point_moveit_demo",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("gazebo_gui", default_value="true"),
            DeclareLaunchArgument("launch_rviz", default_value="true"),
            DeclareLaunchArgument("launch_servo", default_value="false"),
            DeclareLaunchArgument("world_file", default_value="empty.sdf"),
            DeclareLaunchArgument("gz_partition", default_value="manip_sort_ur10e_demo"),
            DeclareLaunchArgument("demo_delay_sec", default_value="15.0"),
            bringup_launch,
            TimerAction(period=demo_delay_sec, actions=[demo_node]),
        ]
    )
