from setuptools import setup
package_name = 'headtraking'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # ↓↓↓ 이 줄이 중요! (런치 파일 설치)
        ('share/' + package_name + '/launch', ['launch/system_bringup.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubp',
    maintainer_email='dev@ubp.local',
    description='Head tracking PID that turns vision target into head commands',
    license='MIT',
    entry_points={
        'console_scripts': [
            'head_tracking_node = headtraking.head_tracking_node:main',
        ],
    },
)
