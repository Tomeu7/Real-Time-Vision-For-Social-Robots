#!/usr/bin/env python3
"""
Launch file for multi-user engagement with debug visualizer

Launches:
- Multi-user engagement detection node
- Multi-user engagement debug visualizer
- Static TF transforms (sellion_link and camera)
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, ExecuteProcess, Shutdown
from launch_ros.actions import Node, SetRemap
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource

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

    ld.add_action(IncludeLaunchDescription(PythonLaunchDescriptionSource([os.path.join(
        get_package_share_directory('hri_face_detect'), 'launch'), '/face_detect.launch.py'])
    ))


    ld.add_action(IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('hri_person_manager'), 'launch'), '/person_manager.launch.py']),
        launch_arguments={"reference_frame": "camera", "robot_reference_frame": "sellion_link"}.items(),
    ))
    

     # face recognition node
    
    ld.add_action(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(
                    get_package_share_directory('hri_face_identification'),
                    'launch',
                    'hri_face_identification.launch.py'
                )
            ]),
            launch_arguments={
                'processing_rate': '2.0',
                'can_learn_new_faces': 'true',
                'match_distance_threshold': '0.5'
            }.items()
        ))

    # Launch multi-user engagement detection node
    ld.add_action(IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('emorobcare_cv_hri_multiuser_engagement'), 'launch'), '/emorobcare_cv_hri_multiuser_engagement.launch.py']),
        launch_arguments={'log_level': 'DEBUG'}.items()
    ))

    """
                 ┌──────────────────────────────┐
                 │          map (world)         │
                 └──────────────┬───────────────┘
                                │
                                ▼
                      base_link  (robot base)
                          │
          z=+0.20 m ↑     │
                          ▼
                    sellion_link  (robot face / head)
                     ├──────────────┬──────────────┐
                     │              │              │
                     ▼              ▼              ▼
              camera (optical)   tablet_link   (other sensors)
                (0,0,0.10 m)    (0.30,0,-0.25 m)
                rotated 90°     pitched up 40°
                (x,y,z)          (x,y,z)

    """
    # Link: phyisical piece of hardware, part of robot structure
    # Frame: coordenate syste
    # Static transform: sellion_link -> camera
    # Quaternion: x=-0.5, y=0.5, z=-0.5, w=0.5 (90° rotations for optical frame)
    ld.add_action(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '-0.1', '-0.5', '0.5', # 10 cm above camera
                   '-0.5', '0.5', 'sellion_link', 'camera'],
        name='sellion_to_camera_tf',
    ))
    # This part is a bit counterintuitive — but it’s purely convention.
    # ROS defines a standard “optical frame” orientation for all cameras, according to REP 103 & REP 105

    # Static transform: base_link -> sellion_link
    # Translation: z=0.20m (robot head height)
    ld.add_action(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0.20', '0', '0', # 20 cm above base_link
                   '0', '1', 'base_link', 'sellion_link'],
        name='base_to_sellion_tf',
    ))

    # ─────────────────────────────────────────────
    # Static transform: sellion_link → tablet_link
    # ─────────────────────────────────────────────
    # Defines the pose of the tablet relative to the robot's face (sellion_link)
    # Translation (x, y, z) in meters:
    #   x = +0.30  → tablet is 30 cm in front of the face
    #   y =  0.00  → centered (no lateral offset)
    #   z = -0.25  → 25 cm below the face
    #
    # Rotation (quaternion: x, y, z, w):
    #   The tablet needs to "face" back toward the robot/person for engagement detection
    #   Combined rotation: 180° yaw (turn around) + 40° pitch down
    #   Result: qx=0, qy=-0.3420, qz=1, qw=0
    #   This makes the tablet's +X axis point backward toward the person
    #
    # This means the tablet lies below and in front of the camera/sellion area,
    # tilted to face back toward the robot (enabling mutual gaze detection).
    #
    # Parent frame: sellion_link
    # Child frame:  tablet_link
    # ─────────────────────────────────────────────
    # from
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

    ld.add_action(IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('knowledge_core'), 'launch'), '/knowledge_core.launch.py'])
    ))

    # Modified visualisation
    ld.add_action(IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('hri_visualization'), 'launch'), '/hri_visualization.launch.py'])
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

