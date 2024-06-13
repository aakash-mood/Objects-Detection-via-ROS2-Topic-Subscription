from setuptools import setup

package_name = 'hw3'

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
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'publisher = hw3.publisher:main',
            'subscriber = hw3.subscriber:main',
            'subscriber_vanilla = hw3.subscriber_vanilla_resnet:main',
            'subscriber_bag = hw3.subscriber_rosbag:main',
            'subscriber_bag_vanilla = hw3.subscriber_rosbag_vanilla_resnet:main',
        ],
    },
)
