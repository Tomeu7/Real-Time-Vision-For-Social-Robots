"""
Unified launcher for testing real-time performance of all HRI components.

Usage:
    ros2 launch emorobcare_cv_pointing_inference full_hri_test.launch.py
    ros2 launch emorobcare_cv_pointing_inference full_hri_test.launch.py yesno:=true
    ros2 launch emorobcare_cv_pointing_inference full_hri_test.launch.py use_id_launcher:=true

Components included:
    - hri_person_manager (always)
    - hri_face_detect (always)
    - hri_body_detect (always)
    - hri_face_identification (always)
    - hri_engagement (always, with 5s delay)
    - emorobcare_cv_gesture_detection (always)
    - emorobcare_cv_pointing_inference (always)
    - emorobcare_cv_object_detection (always)
    - emorobcare_cv_hri_yesno_recognition (optional, yesno:=true)

Arguments:
    use_id_launcher: Use persistent face database with identify_all_faces=true (default: false)
    yesno: Enable yes/no head gesture recognition (default: false)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch_ros.actions import Node, SetRemap
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, LaunchConfigurationEquals
from launch.launch_description_sources import PythonLaunchDescriptionSource

# Force colorized output for all the nodes
os.environ["RCUTILS_COLORIZED_OUTPUT"] = "1"


def generate_launch_description():

    ld = LaunchDescription()

    # =========================================================================
    # Launch Arguments
    # =========================================================================
    ld.add_action(DeclareLaunchArgument(
        'use_id_launcher',
        default_value='false',
        description='Use persistent face database (identify_all_faces=true, can_learn_new_faces=false)'
    ))

    ld.add_action(DeclareLaunchArgument(
        'yesno',
        default_value='true',
        description='Enable yes/no head gesture recognition node'
    ))

    use_id_launcher = LaunchConfiguration('use_id_launcher')
    yesno = LaunchConfiguration('yesno')

    ld.add_action(DeclareLaunchArgument(
        'multiuser_engagement',
        default_value='false',
        description='Use multi-user engagement instead of hri_engagement'
    ))

    multiuser_engagement = LaunchConfiguration('multiuser_engagement')

    # =========================================================================
    # Camera Remapping (applies to all nodes)
    # =========================================================================
    ld.add_action(SetRemap(src='image', dst='/camera/image_raw'))
    ld.add_action(SetRemap(src='camera_info', dst='/camera/camera_info'))

    # =========================================================================
    # TF Static Transform Publishers
    # =========================================================================
    ld.add_action(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0.1', '-0.5', '0.5', '-0.5', '0.5', 'sellion_link', 'camera'],
    ))

    ld.add_action(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0.20', '0', '0', '0', '1', 'base_link', 'sellion_link'],
    ))

    ld.add_action(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '0.20', '0', '-0.25',                # translation: x, y, z
            '0', '0.5', '0', '0.8660',        # rotation quaternion (40° upward tilt)
            'sellion_link', 'tablet_link'        # parent → child
        ],
        name='sellion_to_tablet_tf',
    ))

    # =========================================================================
    # 1. Face Detection (hri_face_detect)
    # =========================================================================
    ld.add_action(IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('hri_face_detect'), 'launch'),
            '/face_detect.launch.py'
        ])
    ))
    

    # =========================================================================
    # 2. Person Manager (hri_person_manager)
    # =========================================================================
    ld.add_action(IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('hri_person_manager'), 'launch'),
            '/person_manager.launch.py'
        ]),
        launch_arguments={
            'reference_frame': 'camera',
            'robot_reference_frame': 'sellion_link'
        }.items(),
    ))

    # =========================================================================
    # 3. Body Detection (hri_body_detect)
    # =========================================================================
    
    ld.add_action(IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('hri_body_detect'), 'launch'),
            '/hri_body_detect.launch.py'
        ])
    ))
    

    # =========================================================================
    # 4. Face Identification (hri_face_identification)
    #    - Default mode: can_learn_new_faces=true
    #    - ID mode (use_id_launcher): can_learn_new_faces=false, identify_all_faces=true
    # =========================================================================
    # Default face identification (when use_id_launcher is false)
    ld.add_action(IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('hri_face_identification'), 'launch'),
            '/hri_face_identification.launch.py'
        ]),
        launch_arguments={
            'processing_rate': '5.0',
            'can_learn_new_faces': 'true',
            'match_distance_threshold': '0.6'
        }.items(),
        condition=LaunchConfigurationEquals('use_id_launcher', 'false')
    ))

    # ID launcher face identification (when use_id_launcher is true)
    ld.add_action(IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('hri_face_identification'), 'launch'),
            '/hri_face_identification.launch.py'
        ]),
        launch_arguments={
            'processing_rate': '5.0',
            'can_learn_new_faces': 'false',
            'match_distance_threshold': '0.6',
            'identify_all_faces': 'true',
            'persistent_face_database_path': '/home/user/vision/faces_tomeu_sara_db.json'
        }.items(),
        condition=LaunchConfigurationEquals('use_id_launcher', 'true')
    ))

    # =========================================================================
    # 5. Engagement (hri_engagement) - Delayed by 5 seconds
    # =========================================================================
    ld.add_action(TimerAction(
        period=5.0,
        actions=[
            # Single-user engagement (default)
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                    os.path.join(get_package_share_directory('hri_engagement'), 'launch'),
                    '/hri_engagement.launch.py'
                ]),
                launch_arguments={
                    'log_level': 'DEBUG',
                    'auto_activate': 'false'
                }.items(),
                condition=LaunchConfigurationEquals('multiuser_engagement', 'false')
            ),

            # Multi-user engagement
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                    os.path.join(get_package_share_directory('emorobcare_cv_hri_multiuser_engagement'), 'launch'),
                    '/emorobcare_cv_hri_multiuser_engagement.launch.py'
                ]),
                launch_arguments={
                    'log_level': 'DEBUG',
                    'auto_activate': 'false'
                }.items(),
                condition=LaunchConfigurationEquals('multiuser_engagement', 'true')
            ),
        ]
    ))


    # =========================================================================
    # 6. Gesture Detection (emorobcare_cv_gesture_detection)
    # =========================================================================
    
    ld.add_action(Node(
        package='emorobcare_cv_gesture_detection',
        executable='gesture_detector_node',
        output='screen',
        emulate_tty=True,
    ))
    
    # =========================================================================
    # 7. Pointing Inference (emorobcare_cv_pointing_inference)
    # =========================================================================
    ld.add_action(Node(
        package='emorobcare_cv_pointing_inference',
        executable='pointing_inference_node',
        output='screen',
        emulate_tty=True,
    ))

    # =========================================================================
    # 8. Object Detection (emorobcare_cv_object_detection)
    # =========================================================================
    ld.add_action(Node(
        package='emorobcare_cv_object_detection',
        executable='object_detector_node',
        output='screen',
        emulate_tty=True,
    ))

    # =========================================================================
    # 9. Yes/No Recognition (optional - emorobcare_cv_hri_yesno_recognition)
    # =========================================================================
    
    ld.add_action(Node(
        package='emorobcare_cv_hri_yesno_recognition',
        executable='hri_yesno_recognition_node',
        output='screen',
        emulate_tty=True,
        condition=IfCondition(yesno)
    ))

    return ld
