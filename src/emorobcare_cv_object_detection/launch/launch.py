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

    # launch arguments, webcam or phone
    ld.add_action(DeclareLaunchArgument(
        'video_source',
        default_value='webcam',
        description='Webcam or phone'
    ))
    video_source = LaunchConfiguration('video_source')
    # phone_url = LaunchConfiguration('phone_url')
    
    #  remap
    ld.add_action(SetRemap(src='image', dst='/camera/image_raw'))
    ld.add_action(SetRemap(src='camera_info', dst='/camera/camera_info'))


    # Webcam group
    ld.add_action(GroupAction([
        Node(
            package='gscam',
            executable='gscam_node',
            condition=IfCondition(PythonExpression(["'", video_source, "' == 'webcam'"])),
            parameters=[{
                'gscam_config': 'v4l2src device=/dev/video0 ! image/jpeg,width=640,height=480,framerate=30/1 ! jpegdec ! videoconvert',
                'use_sensor_data_qos': True,
                'camera_name': 'camera',
                'camera_info_url': 'package://interaction_sim/config/camera_info.yaml',
                'frame_id': 'camera'
            }]
        )
    ]))
    
    # Object detection node
    ld.add_action(Node(
        package='emorobcare_cv_object_detection',
        executable='object_detector_node', 
        output='screen',
        emulate_tty=True,
    ))
    
    
    # knowelge core
    
    ld.add_action(IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('knowledge_core'), 'launch'), '/knowledge_core.launch.py'])
    ))
    
    # modified visualisation
    ld.add_action(IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('hri_visualization'), 'launch'), '/hri_visualization.launch.py'])
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
    rqt_cmd = ['rqt',             
           '--perspective-file', 
           os.path.join(
               get_package_share_directory('interaction_sim'),
               'config/simulator.perspective')]

    rqt = ExecuteProcess(
        cmd=rqt_cmd,
        output='log',
        shell=False,
        on_exit=Shutdown()
    )

    ld.add_action(rqt)
    
    return ld
