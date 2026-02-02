from setuptools import find_packages, setup

package_name = 'emorobcare_cv_pointing_inference'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/config.yaml']),
        ('share/' + package_name + '/launch', [
        'launch/full_hri_test.launch.py'
    ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='tom_93_mot@hotmail.com',
    description='ROS2 node for pointing gesture inference and target estimation',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pointing_inference_node = emorobcare_cv_pointing_inference.node_pointing_inference:main',
        ],
    },
)
