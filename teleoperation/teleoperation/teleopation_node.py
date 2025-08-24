#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from sensor_msgs.msg import JointState
import threading
from typing import Dict, List, Optional

from dynamixel_sdk import (
    PortHandler, PacketHandler, GroupSyncWrite, GroupSyncRead
)


class TeleoperationSuitController(Node):
    """
    ROS2 Teleoperation Suit Controller Node
    - 다이나믹셀 슈트 (ID 1~14)를 제어
    - 초기화: 토크 온 -> 원점(2048) 이동 -> 토크 오프
    - 엔코더 값을 지속적으로 퍼블리시
    """

    def __init__(self):
        super().__init__('teleoperation')
        
        # ----------------------------
        # Parameters
        # ----------------------------
        self.declare_parameter('suit_port', '/dev/ttyUSB1')  # 슈트용 별도 포트
        self.declare_parameter('suit_baudrate', 1000000)
        self.declare_parameter('protocol_version', 2.0)
        self.declare_parameter('read_frequency', 50.0)  # 엔코더 읽기 주파수
        
        # 슈트 모터 ID들 (1~14)
        self.declare_parameter('suit_motor_ids', list(range(1, 15)))
        
        # 초기화 관련 파라미터
        self.declare_parameter('init_profile_velocity', 50)  # 초기화 시 움직임 속도
        self.declare_parameter('init_profile_acceleration', 10)
        self.declare_parameter('init_timeout_sec', 10.0)  # 초기화 타임아웃
        self.declare_parameter('position_threshold', 50)  # 위치 도달 임계값 (틱)

        # Get parameters
        self.suit_port = self.get_parameter('suit_port').value
        self.suit_baudrate = self.get_parameter('suit_baudrate').value
        self.protocol_version = self.get_parameter('protocol_version').value
        self.read_frequency = self.get_parameter('read_frequency').value
        
        self.suit_motor_ids = self.get_parameter('suit_motor_ids').value
        
        self.init_profile_velocity = self.get_parameter('init_profile_velocity').value
        self.init_profile_acceleration = self.get_parameter('init_profile_acceleration').value
        self.init_timeout_sec = self.get_parameter('init_timeout_sec').value
        self.position_threshold = self.get_parameter('position_threshold').value

        # ----------------------------
        # Dynamixel Constants
        # ----------------------------
        self.ADDR_TORQUE_ENABLE = 64
        self.ADDR_PROFILE_ACCELERATION = 108
        self.ADDR_PROFILE_VELOCITY = 112
        self.ADDR_GOAL_POSITION = 116
        self.ADDR_PRESENT_POSITION = 132
        self.LEN_GOAL_POSITION = 4
        self.LEN_PRESENT_POSITION = 4
        
        self.TICKS_PER_REV = 4095.0
        self.CENTER_TICK = 2048
        self.DEG_TO_TICK = self.TICKS_PER_REV / 360.0

        # ----------------------------
        # State Variables
        # ----------------------------
        self.current_positions = {motor_id: 0 for motor_id in self.suit_motor_ids}
        self.initialization_complete = False
        self.torque_enabled = False
        
        self.read_lock = threading.Lock()
        
        # Initialize Dynamixel
        self.setup_dynamixel()
        
        # ----------------------------
        # ROS2 Setup
        # ----------------------------
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Publishers
        self.body_cmd_pub = self.create_publisher(
            Float64MultiArray,
            '/ubp/body/cmd',
            qos_profile
        )
        
        self.suit_joint_state_pub = self.create_publisher(
            JointState,
            '/ubp/teleop/suit/joint_states',
            qos_profile
        )
        
        # Status publisher
        self.suit_status_pub = self.create_publisher(
            Float64MultiArray,
            '/ubp/teleop/suit/status',
            qos_profile
        )
        
        # Encoder reading timer
        self.read_timer = self.create_timer(
            1.0 / self.read_frequency,
            self.read_and_publish_encoders
        )
        
        # Status publishing timer
        self.status_timer = self.create_timer(
            0.5,  # 2Hz
            self.publish_status
        )
        
        self.get_logger().info(f'Teleoperation Suit Controller initialized')
        self.get_logger().info(f'Suit motors: {self.suit_motor_ids}')
        self.get_logger().info(f'Port: {self.suit_port}')
        
        # Start initialization process
        self.initialize_suit()

    def setup_dynamixel(self):
        """Initialize Dynamixel hardware for suit"""
        try:
            # Port setup
            self.port_handler = PortHandler(self.suit_port)
            if not self.port_handler.openPort():
                raise RuntimeError(f"Failed to open suit port: {self.suit_port}")
            
            if not self.port_handler.setBaudRate(self.suit_baudrate):
                raise RuntimeError(f"Failed to set suit baudrate: {self.suit_baudrate}")
            
            self.packet_handler = PacketHandler(self.protocol_version)
            
            # Group sync write/read setup
            self.group_sync_write = GroupSyncWrite(
                self.port_handler, self.packet_handler, 
                self.ADDR_GOAL_POSITION, self.LEN_GOAL_POSITION
            )
            
            self.group_sync_read = GroupSyncRead(
                self.port_handler, self.packet_handler,
                self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION
            )
            
            # Add all suit motors to sync read
            for motor_id in self.suit_motor_ids:
                self.group_sync_read.addParam(motor_id)
            
            self.get_logger().info('Dynamixel suit setup completed successfully')
            
        except Exception as e:
            self.get_logger().error(f'Dynamixel suit setup failed: {str(e)}')
            raise

    def initialize_suit(self):
        """Initialize suit: Enable torque -> Move to center -> Disable torque"""
        self.get_logger().info('Starting suit initialization...')
        
        try:
            # Step 1: Enable torque for all motors
            self.get_logger().info('Step 1: Enabling torque for all suit motors')
            for motor_id in self.suit_motor_ids:
                dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
                    self.port_handler, motor_id, self.ADDR_TORQUE_ENABLE, 1
                )
                if dxl_comm_result != 0 or dxl_error != 0:
                    self.get_logger().warn(f'Failed to enable torque for suit motor {motor_id}')
                else:
                    self.get_logger().debug(f'Torque enabled for suit motor {motor_id}')
                
                # Set profile parameters
                self.packet_handler.write4ByteTxRx(
                    self.port_handler, motor_id, self.ADDR_PROFILE_VELOCITY, 
                    self.init_profile_velocity
                )
                self.packet_handler.write4ByteTxRx(
                    self.port_handler, motor_id, self.ADDR_PROFILE_ACCELERATION, 
                    self.init_profile_acceleration
                )
            
            self.torque_enabled = True
            self.get_logger().info('Torque enabled for all suit motors')
            
            # Step 2: Move all motors to center position (2048)
            self.get_logger().info('Step 2: Moving all motors to center position (2048)')
            center_positions = {motor_id: self.CENTER_TICK for motor_id in self.suit_motor_ids}
            self.move_to_positions_sync(center_positions)
            
            # Step 3: Wait for all motors to reach center position
            self.get_logger().info('Step 3: Waiting for motors to reach center position...')
            start_time = time.time()
            all_reached = False
            
            while not all_reached and (time.time() - start_time) < self.init_timeout_sec:
                self.read_present_positions()
                all_reached = True
                
                for motor_id in self.suit_motor_ids:
                    current_pos = self.current_positions[motor_id]
                    error = abs(current_pos - self.CENTER_TICK)
                    if error > self.position_threshold:
                        all_reached = False
                        break
                
                if not all_reached:
                    time.sleep(0.1)  # Wait 100ms before checking again
            
            if all_reached:
                self.get_logger().info('All motors reached center position')
            else:
                self.get_logger().warn('Initialization timeout - some motors may not have reached center')
            
            # Step 4: Disable torque for all motors
            self.get_logger().info('Step 4: Disabling torque for all suit motors')
            for motor_id in self.suit_motor_ids:
                dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
                    self.port_handler, motor_id, self.ADDR_TORQUE_ENABLE, 0
                )
                if dxl_comm_result != 0 or dxl_error != 0:
                    self.get_logger().warn(f'Failed to disable torque for suit motor {motor_id}')
                else:
                    self.get_logger().debug(f'Torque disabled for suit motor {motor_id}')
            
            self.torque_enabled = False
            self.initialization_complete = True
            self.get_logger().info('Suit initialization completed successfully!')
            self.get_logger().info('Suit is now ready for teleoperation - encoders will be published')
            
        except Exception as e:
            self.get_logger().error(f'Suit initialization failed: {str(e)}')
            self.initialization_complete = False

    def move_to_positions_sync(self, positions: Dict[int, int]):
        """Move multiple motors to specified positions synchronously"""
        try:
            self.group_sync_write.clearParam()
            
            for motor_id, position in positions.items():
                # Convert to byte array
                param_goal = [
                    (position >> 0) & 0xFF,
                    (position >> 8) & 0xFF,
                    (position >> 16) & 0xFF,
                    (position >> 24) & 0xFF
                ]
                
                self.group_sync_write.addParam(motor_id, param_goal)
            
            dxl_comm_result = self.group_sync_write.txPacket()
            self.group_sync_write.clearParam()
            
            if dxl_comm_result != 0:
                self.get_logger().debug(f'GroupSyncWrite failed: {dxl_comm_result}')
                
        except Exception as e:
            self.get_logger().warn(f'Failed to move suit motors: {str(e)}')

    def read_present_positions(self):
        """Read current positions of all suit motors"""
        try:
            dxl_comm_result = self.group_sync_read.txRxPacket()
            if dxl_comm_result != 0:
                return False
            
            with self.read_lock:
                for motor_id in self.suit_motor_ids:
                    if self.group_sync_read.isAvailable(motor_id, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION):
                        present_pos = self.group_sync_read.getData(motor_id, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION)
                        self.current_positions[motor_id] = present_pos
            
            return True
            
        except Exception as e:
            self.get_logger().debug(f'Failed to read present positions: {str(e)}')
            return False

    def read_and_publish_encoders(self):
        """Read encoder values and publish them"""
        if not self.initialization_complete:
            return
        
        if self.read_present_positions():
            # Publish encoder values as raw ticks
            encoder_msg = Float64MultiArray()
            
            with self.read_lock:
                # Sort by motor ID for consistent ordering
                sorted_ids = sorted(self.suit_motor_ids)
                encoder_msg.data = [float(self.current_positions[motor_id]) for motor_id in sorted_ids]
            
            # Add dimension info
            dim = MultiArrayDimension()
            dim.label = "suit_encoders"
            dim.size = len(self.suit_motor_ids)
            dim.stride = len(self.suit_motor_ids)
            encoder_msg.layout.dim.append(dim)
            
            self.body_cmd_pub.publish(encoder_msg)
            
            # Also publish as joint states (in degrees)
            self.publish_joint_states()

    def publish_joint_states(self):
        """Publish current joint states"""
        if not self.initialization_complete:
            return
        
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        with self.read_lock:
            # Add joint names and positions (sorted by ID)
            sorted_ids = sorted(self.suit_motor_ids)
            for motor_id in sorted_ids:
                joint_name = f'suit_joint_{motor_id:02d}'  # e.g., suit_joint_01
                
                msg.name.append(joint_name)
                msg.position.append(self.tick_to_deg(self.current_positions[motor_id]))
                msg.velocity.append(0.0)  # Not calculated in this implementation
                msg.effort.append(0.0)    # Not measured in this implementation
        
        self.suit_joint_state_pub.publish(msg)

    def publish_status(self):
        """Publish controller status"""
        status_msg = Float64MultiArray()
        status_msg.data = [
            1.0 if self.initialization_complete else 0.0,  # Initialization status
            1.0 if self.torque_enabled else 0.0,           # Torque status
            float(len(self.suit_motor_ids)),                # Number of motors
            time.time()                                     # Timestamp
        ]
        self.suit_status_pub.publish(status_msg)

    def tick_to_deg(self, tick: int) -> float:
        """Convert motor ticks to degrees (center = 0°)"""
        return (float(tick) - self.CENTER_TICK) / self.DEG_TO_TICK

    def deg_to_tick(self, deg: float) -> int:
        """Convert degrees to motor ticks (0° = center)"""
        return int(round(self.CENTER_TICK + (self.DEG_TO_TICK * float(deg))))

    def destroy_node(self):
        """Clean shutdown"""
        try:
            # If torque is still enabled, disable it
            if self.torque_enabled:
                self.get_logger().info('Disabling torque for all suit motors during shutdown...')
                for motor_id in self.suit_motor_ids:
                    self.packet_handler.write1ByteTxRx(
                        self.port_handler, motor_id, self.ADDR_TORQUE_ENABLE, 0
                    )
            
            self.port_handler.closePort()
            self.get_logger().info('Teleoperation suit controller shutdown completed')
        except Exception as e:
            self.get_logger().error(f'Error during shutdown: {str(e)}')
        
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    try:
        controller = TeleoperationSuitController()
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        try:
            controller.destroy_node()
        except:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()