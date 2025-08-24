from launch import LaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    # realsense2_camera 패키지의 기본 launch 파일 경로 찾기
    realsense_launch_file = os.path.join(
        get_package_share_directory('realsense2_camera'),
        'launch',
        'rs_launch.py'
    )

    return LaunchDescription([

        # 1. dxl_operator
        Node(
            package='dxl_operator',
            executable='dxl_operator_node',
            name='dxl_operator',
            output='screen'
        ),

        # 2. vision - person_tracker_node.py
        Node(
            package='vision',
            executable='person_tracker_node',
            name='person_tracker',
            output='screen'
        ),

        # 3. headtraking - head_tracking_node
        Node(
            package='headtraking',
            executable='head_tracking_node',
            name='head_tracking',
            output='screen'
        ),

        # 4. realsense2_camera - rs_launch.py 포함
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(realsense_launch_file),
            launch_arguments={
                'align_depth': 'true',
                'enable_gyro': 'true',
                'enable_accel': 'true'
            }.items()
        )
    ])
