import os
import xacro

from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch import LaunchDescription

def generate_launch_description():
    robot_description_pkg = get_package_share_directory('robot_description')
    gazebo_ros_pkg = get_package_share_directory('gazebo_ros')

    urdf_file_path = os.path.join(robot_description_pkg, 'urdf', 'hexapod.urdf.xacro')
    with open(urdf_file_path, 'r') as f:
        doc = xacro.parse(f)
        xacro.process_doc(doc)
        params = {'robot_description': doc.toxml()}
    
    world_file_path = os.path.join(robot_description_pkg, 'worlds', 'empty.world')
    gazebo_server_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gazebo_ros_pkg, "launch", "gzserver.launch.py")),
        launch_arguments={"pause": "true", "verbose": "true", "world": world_file_path}.items()
    )

    gazebo_client_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gazebo_ros_pkg, "launch", "gzclient.launch.py"))
    )

    spawn_entity_cmd = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=["-entity", "hexapod", "-topic", "robot_description"],
        parameters=[params]
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[params],
        arguments=[urdf_file_path]
    )

    return LaunchDescription([
        gazebo_server_cmd,
        gazebo_client_cmd,
        robot_state_publisher,
        spawn_entity_cmd
    ])
    