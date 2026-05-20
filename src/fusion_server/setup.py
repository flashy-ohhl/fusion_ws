from setuptools import setup

package_name = 'fusion_server'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zjj',
    maintainer_email='zjj@todo.todo',
    description='YOLOv11 + AclLite + LiDAR-Camera fusion server',
    license='Apache-2.0',
    # tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fusion_server_node = fusion_server.fusion_server_node:main',
        ],
    },
)

