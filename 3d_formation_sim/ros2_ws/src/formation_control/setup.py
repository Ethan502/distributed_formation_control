import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'formation_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'numpy', 'scipy'],
    zip_safe=True,
    maintainer='ethan',
    maintainer_email='esmit502@byu.edu',
    description='3D distributed formation control with PX4 quadrotors',
    license='MIT',
    entry_points={
        'console_scripts': [
            'drone_node = formation_control.drone_node:main',
            'coordinator_node = formation_control.coordinator_node:main',
        ],
    },
)
