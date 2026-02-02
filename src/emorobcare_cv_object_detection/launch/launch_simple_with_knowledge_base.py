import os

from ament_index_python.packages import get_package_share_directory


from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, ExecuteProcess, Shutdown
from launch_ros.actions import Node, SetRemap
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource

# ros2 launch emorobcare_cv_object_detection launch.py

# force colorized output for all the nodes
os.environ["RCUTILS_COLORIZED_OUTPUT"] = "1"

def generate_launch_description():

    ld = LaunchDescription()

    #  Remap
    ld.add_action(SetRemap(src='image', dst='/camera/image_raw'))
    ld.add_action(SetRemap(src='camera_info', dst='/camera/camera_info'))
    
    # Object detection node
    ld.add_action(Node(
        package='emorobcare_cv_object_detection',
        executable='object_detector_node', 
        output='screen',
        emulate_tty=True,
    ))
    
    # Knoweledge core
    ld.add_action(IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('knowledge_core'), 'launch'), '/knowledge_core.launch.py'])
    ))
    
    ld.add_action(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0.1', '-0.5', '0.5',
                   '-0.5', '0.5', 'sellion_link', 'camera'],
    ))

    ld.add_action(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0.20', '0', '0',
                   '0', '1', 'base_link', 'sellion_link'],
    ))
    
    return ld
