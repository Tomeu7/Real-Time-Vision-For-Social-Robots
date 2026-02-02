from setuptools import find_packages, setup

package_name = 'emorobcare_cv_gesture_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/config.yaml']),
        ('share/' + package_name + '/models', ['models/keypoint_classifier.tflite']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='tom_93_mot@hotmail.com',
    description='ROS2 node for real-time hand gesture detection using MediaPipe and TFLite',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gesture_detector_node = emorobcare_cv_gesture_detection.node_gesture_detection:main'
        ],
    },
)
