from setuptools import find_packages, setup

package_name = 'emorobcare_cv_hri_yesno_recognition'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/' + package_name + '/config', ['config/config.yaml']),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='ROS2 node for yes/no head gesture recognition',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hri_yesno_recognition_node = emorobcare_cv_hri_yesno_recognition.node_hri_yesno_recognition:main'
        ],
    },
)
