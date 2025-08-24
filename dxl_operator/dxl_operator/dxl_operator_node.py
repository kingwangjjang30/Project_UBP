#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from sensor_msgs.msg import JointState
import threading
from typing import Dict, List, Optional, Tuple

from dynamixel_sdk import (
    PortHandler, PacketHandler, GroupSyncWrite, GroupSyncRead
)


class DynamixelController(Node):
    """
    ROS2 Dynamixel Controller Node
    - 토크 온 후 대기 상태에서 목표 위치 구독
    - PID 제어로 부드러운 움직임
    - 확장 가능한 모터 개수 (head 2개 + 추후 body 모터들)
    """

    def __init__(self):
        super().__init__('dxl_operator')
        
        # ----------------------------
        # Parameters
        # ----------------------------
        self.declare_parameter('port', '/dev/ttyUSB0')
        self.declare_parameter('baudrate', 1000000)
        self.declare_parameter('protocol_version', 2.0)
        self.declare_parameter('control_frequency', 50.0)
        
        # Motor configurations - 추후 확장 가능
        self.declare_parameter('head_yaw_id', 15)
        self.declare_parameter('head_pitch_id', 16)
        self.declare_parameter('body_ids', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])  # 추후 body 모터들
        
        # PID gains
        self.declare_parameter('pid_kp', 0.8)
        self.declare_parameter('pid_ki', 0.01)
        self.declare_parameter('pid_kd', 0.05)
        
        # Movement limits
        self.declare_parameter('max_angle_deg', 60.0)
        self.declare_parameter('profile_velocity', 30)
        self.declare_parameter('profile_acceleration', 5)
        
        # Timeout for return to home
        self.declare_parameter('command_timeout_sec', 3.0)

        # Get parameters
        self.port = self.get_parameter('port').value
        self.baudrate = self.get_parameter('baudrate').value
        self.protocol_version = self.get_parameter('protocol_version').value
        self.control_frequency = self.get_parameter('control_frequency').value
        
        self.head_yaw_id = self.get_parameter('head_yaw_id').value
        self.head_pitch_id = self.get_parameter('head_pitch_id').value
        self.body_ids = self.get_parameter('body_ids').value
        
        self.kp = self.get_parameter('pid_kp').value
        self.ki = self.get_parameter('pid_ki').value
        self.kd = self.get_parameter('pid_kd').value
        
        self.max_angle_deg = self.get_parameter('max_angle_deg').value
        self.profile_velocity = self.get_parameter('profile_velocity').value
        self.profile_acceleration = self.get_parameter('profile_acceleration').value
        self.command_timeout_sec = self.get_parameter('command_timeout_sec').value

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
        self.MAX_TICK_OFFSET = int(self.DEG_TO_TICK * self.max_angle_deg)

        # All motor IDs (head + body)
        self.all_motor_ids = [self.head_yaw_id, self.head_pitch_id] + self.body_ids
        self.head_ids = [self.head_yaw_id, self.head_pitch_id]
        
        # ----------------------------
        # Control Variables (Initialize BEFORE setup_dynamixel)
        # ----------------------------
        self.current_positions = {motor_id: self.CENTER_TICK for motor_id in self.all_motor_ids}
        self.goal_positions = {motor_id: self.CENTER_TICK for motor_id in self.all_motor_ids}
        self.target_positions = {motor_id: self.CENTER_TICK for motor_id in self.all_motor_ids}
        
        # Command timeout tracking
        self.last_head_cmd_time = 0.0
        self.last_body_cmd_time = 0.0
        self.head_at_home = True
        self.body_at_home = True
        
        # PID controllers for each motor
        self.pid_controllers = {}
        for motor_id in self.all_motor_ids:
            self.pid_controllers[motor_id] = PIDController(self.kp, self.ki, self.kd)
        
        self.control_lock = threading.Lock()
        self.last_control_time = time.time()
        
        # Initialize Dynamixel AFTER control variables are set
        self.setup_dynamixel()
        
        # ----------------------------
        # ROS2 Setup
        # ----------------------------
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        
        # Subscribers - 기존 토픽 구조에 맞춤
        self.head_cmd_sub = self.create_subscription(
            Float64MultiArray,
            '/ubp/head/cmd',
            self.head_cmd_callback,
            qos_profile
        )
        
        self.body_cmd_sub = self.create_subscription(
            Float64MultiArray,
            '/ubp/body/cmd', 
            self.body_cmd_callback,
            qos_profile
        )
        
        # Publishers
        self.joint_state_pub = self.create_publisher(
            JointState,
            'joint_states',
            qos_profile
        )
        
        # Control timer
        self.control_timer = self.create_timer(
            1.0 / self.control_frequency,
            self.control_loop
        )
        
        # Status timer
        self.status_timer = self.create_timer(
            0.1,  # 10Hz
            self.publish_joint_states
        )
        
        self.get_logger().info(f'Dynamixel Controller initialized with {len(self.all_motor_ids)} motors')
        self.get_logger().info(f'Head motors: {self.head_ids}')
        if self.body_ids:
            self.get_logger().info(f'Body motors: {self.body_ids}')

    def setup_dynamixel(self):
        """Initialize Dynamixel hardware"""
        try:
            # Port setup
            self.port_handler = PortHandler(self.port)
            if not self.port_handler.openPort():
                raise RuntimeError(f"Failed to open port: {self.port}")
            
            if not self.port_handler.setBaudRate(self.baudrate):
                raise RuntimeError(f"Failed to set baudrate: {self.baudrate}")
            
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
            
            # Add all motors to sync read
            for motor_id in self.all_motor_ids:
                self.group_sync_read.addParam(motor_id)
            
            # Enable torque and set profile for all motors
            for motor_id in self.all_motor_ids:
                # Enable torque
                dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
                    self.port_handler, motor_id, self.ADDR_TORQUE_ENABLE, 1
                )
                if dxl_comm_result != 0 or dxl_error != 0:
                    self.get_logger().warn(f'Failed to enable torque for motor {motor_id}')
                
                # Set profile velocity
                dxl_comm_result, dxl_error = self.packet_handler.write4ByteTxRx(
                    self.port_handler, motor_id, self.ADDR_PROFILE_VELOCITY, self.profile_velocity
                )
                
                # Set profile acceleration  
                dxl_comm_result, dxl_error = self.packet_handler.write4ByteTxRx(
                    self.port_handler, motor_id, self.ADDR_PROFILE_ACCELERATION, self.profile_acceleration
                )
            
            # Move all motors to center position initially
            self.move_to_positions_sync(self.current_positions)
            time.sleep(2.0)  # Wait for initial positioning
            
            self.get_logger().info('Dynamixel setup completed successfully')
            
        except Exception as e:
            self.get_logger().error(f'Dynamixel setup failed: {str(e)}')
            raise

    def head_cmd_callback(self, msg: Float64MultiArray):
        """Handle head command from head tracking node (yaw, pitch in degrees)"""
        if len(msg.data) != 2:
            self.get_logger().warn('Head cmd message should contain exactly 2 values (yaw, pitch)')
            return
        
        yaw_deg, pitch_deg = msg.data
        
        # Clamp to limits
        yaw_deg = max(-self.max_angle_deg, min(self.max_angle_deg, yaw_deg))
        pitch_deg = max(-self.max_angle_deg, min(self.max_angle_deg, pitch_deg))
        
        # Convert to ticks
        yaw_tick = self.deg_to_tick(yaw_deg)
        pitch_tick = self.deg_to_tick(pitch_deg)
        
        with self.control_lock:
            self.target_positions[self.head_yaw_id] = yaw_tick
            self.target_positions[self.head_pitch_id] = pitch_tick
            self.last_head_cmd_time = time.time()
            self.head_at_home = False
        
        self.get_logger().debug(f'Head cmd received: yaw={yaw_deg:.1f}°, pitch={pitch_deg:.1f}°')

    def body_cmd_callback(self, msg: Float64MultiArray):
        """Handle body command (degrees for each body motor)"""
        if len(msg.data) != len(self.body_ids):
            self.get_logger().warn(f'Body cmd message should contain {len(self.body_ids)} values')
            return
        
        with self.control_lock:
            for i, motor_id in enumerate(self.body_ids):
                angle_deg = msg.data[i]
                # Apply limits if needed (could be motor-specific)
                angle_deg = max(-self.max_angle_deg, min(self.max_angle_deg, angle_deg))
                self.target_positions[motor_id] = self.deg_to_tick(angle_deg)
            self.last_body_cmd_time = time.time()
            self.body_at_home = False
        
        self.get_logger().debug(f'Body cmd received for {len(self.body_ids)} motors')

    def control_loop(self):
        """Main control loop with PID"""
        current_time = time.time()
        dt = current_time - self.last_control_time
        self.last_control_time = current_time
        
        # Check for command timeouts and return to home position
        self.check_command_timeout(current_time)
        
        # Read current positions
        self.read_present_positions()
        
        # Calculate PID outputs and update goal positions
        new_goals = {}
        
        with self.control_lock:
            for motor_id in self.all_motor_ids:
                current_pos = self.current_positions[motor_id]
                target_pos = self.target_positions[motor_id]
                
                # PID control
                error = target_pos - current_pos
                pid_output = self.pid_controllers[motor_id].compute(error, dt)
                
                # Update goal position with PID output
                new_goal = current_pos + int(pid_output)
                
                # Clamp to valid range
                new_goal = max(0, min(int(self.TICKS_PER_REV), new_goal))
                
                new_goals[motor_id] = new_goal
                self.goal_positions[motor_id] = new_goal
        
        # Send new goals to motors
        self.move_to_positions_sync(new_goals)

    def read_present_positions(self):
        """Read current positions of all motors"""
        try:
            dxl_comm_result = self.group_sync_read.txRxPacket()
            if dxl_comm_result != 0:
                return
            
            for motor_id in self.all_motor_ids:
                if self.group_sync_read.isAvailable(motor_id, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION):
                    present_pos = self.group_sync_read.getData(motor_id, self.ADDR_PRESENT_POSITION, self.LEN_PRESENT_POSITION)
                    self.current_positions[motor_id] = present_pos
        except Exception as e:
            self.get_logger().debug(f'Failed to read present positions: {str(e)}')

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
            self.get_logger().warn(f'Failed to move motors: {str(e)}')

    def check_command_timeout(self, current_time: float):
        """Check for command timeouts and return to home position if needed"""
        with self.control_lock:
            # Check head command timeout
            if not self.head_at_home and (current_time - self.last_head_cmd_time) > self.command_timeout_sec:
                self.get_logger().info('Head command timeout - returning to home position')
                self.target_positions[self.head_yaw_id] = self.CENTER_TICK
                self.target_positions[self.head_pitch_id] = self.CENTER_TICK
                self.head_at_home = True
                
                # Reset PID controllers for head motors
                self.pid_controllers[self.head_yaw_id].reset()
                self.pid_controllers[self.head_pitch_id].reset()
            
            # Check body command timeout (only if there are body motors)
            if self.body_ids and not self.body_at_home and (current_time - self.last_body_cmd_time) > self.command_timeout_sec:
                self.get_logger().info('Body command timeout - returning to home position')
                for motor_id in self.body_ids:
                    self.target_positions[motor_id] = self.CENTER_TICK
                    self.pid_controllers[motor_id].reset()
                self.body_at_home = True

    def publish_joint_states(self):
        """Publish current joint states"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # Add joint names and positions
        for motor_id in self.all_motor_ids:
            if motor_id == self.head_yaw_id:
                joint_name = 'head_yaw'
            elif motor_id == self.head_pitch_id:
                joint_name = 'head_pitch'
            else:
                joint_name = f'body_joint_{motor_id}'
            
            msg.name.append(joint_name)
            msg.position.append(self.tick_to_deg(self.current_positions[motor_id]))
            msg.velocity.append(0.0)  # Could be calculated if needed
            msg.effort.append(0.0)    # Could be read if needed
        
        self.joint_state_pub.publish(msg)

    def deg_to_tick(self, deg: float) -> int:
        """Convert degrees to motor ticks (0° = center)"""
        return int(round(self.CENTER_TICK + (self.DEG_TO_TICK * float(deg))))

    def tick_to_deg(self, tick: int) -> float:
        """Convert motor ticks to degrees (center = 0°)"""
        return (float(tick) - self.CENTER_TICK) / self.DEG_TO_TICK

    def destroy_node(self):
        """Clean shutdown"""
        try:
            # Disable torque for all motors
            for motor_id in self.all_motor_ids:
                self.packet_handler.write1ByteTxRx(
                    self.port_handler, motor_id, self.ADDR_TORQUE_ENABLE, 0
                )
            
            self.port_handler.closePort()
            self.get_logger().info('Dynamixel controller shutdown completed')
        except Exception as e:
            self.get_logger().error(f'Error during shutdown: {str(e)}')
        
        super().destroy_node()


class PIDController:
    """Simple PID controller"""
    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki  
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0
        self.max_integral = 1000.0  # Anti-windup

    def compute(self, error: float, dt: float) -> float:
        if dt <= 0:
            return 0.0
        
        # Integral term with anti-windup
        self.integral += error * dt
        self.integral = max(-self.max_integral, min(self.max_integral, self.integral))
        
        # Derivative term
        derivative = (error - self.prev_error) / dt
        
        # PID output
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        self.prev_error = error
        return output

    def reset(self):
        """Reset PID state"""
        self.prev_error = 0.0
        self.integral = 0.0


def main(args=None):
    rclpy.init(args=args)
    
    try:
        controller = DynamixelController()
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