from setuptools import find_packages, setup
import glob
import os

package_name = 'emorobcare_cv_object_detection'

model_files = glob.glob('models/**/*.pt', recursive=True) + glob.glob('models/**/*.tflite', recursive=True)

data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/config.yaml']),
        ('share/' + package_name + '/launch', [
        'launch/launch.py',
        'launch/launch_simple_with_knowledge_base.py',
        'launch/emorobot_object_detector.launch.py'
    ])
    ]

for filepath in model_files:
    target_dir = os.path.join('share', package_name, os.path.dirname(filepath))
    data_files.append((target_dir, [filepath]))

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='tom_93_mot@hotmail.com',
    description='ROS2 node for real-time object detection using YOLO',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'object_detector_node = emorobcare_cv_object_detection.node_object_detection:main'
        ],
    },
)
