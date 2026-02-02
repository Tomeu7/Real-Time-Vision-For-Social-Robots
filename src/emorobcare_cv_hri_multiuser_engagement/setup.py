# Copyright (c) 2023 PAL Robotics S.L. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup

package_name = 'emorobcare_cv_hri_multiuser_engagement'

setup(
    name=package_name,
    version='2.0.2',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/ament_index/resource_index/pal_system_module',
         ['module/' + package_name]),
        (f'share/ament_index/resource_index/pal_configuration.{package_name}',
         [f'pal_configuration/{package_name}']),
        ('share/' + package_name + '/launch',
         ['launch/emorobcare_cv_hri_multiuser_engagement.launch.py',
          'launch/main.launch.py']),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/module',
         ['module/emorobcare_cv_hri_multiuser_engagement.yaml']),
        ('share/' + package_name + '/config', ['config/00-default.yaml']),
        ('share/' + package_name + '/config', ['config/' + package_name + '_analyzers.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Séverin Lemaignan',
    maintainer_email='severin.lemaignan@pal-robotics.com',
    description='Engagement estimation for the ROS4HRI framework',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
                'emorobcare_cv_hri_multiuser_engagement = emorobcare_cv_hri_multiuser_engagement.emorobcare_cv_hri_multiuser_engagement:main'
        ],
    },
)
