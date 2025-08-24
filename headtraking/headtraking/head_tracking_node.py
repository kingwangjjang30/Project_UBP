#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

def _is_finite(x):
    return x is not None and math.isfinite(x)

class HeadTracking(Node):
    def __init__(self):
        super().__init__('head_tracking_node')
        # --- Parameters ---
        self.declare_parameter('kp_yaw', 1.0)
        self.declare_parameter('kp_pitch', 1.0)
        self.declare_parameter('max_speed_deg_s', 120.0)
        self.declare_parameter('limit_yaw_deg', 60.0)
        self.declare_parameter('limit_pitch_deg', 30.0)
        self.declare_parameter('update_hz', 30.0)
        self.declare_parameter('target_timeout_sec', 0.5)   # ← 미수신 복귀용
        self.declare_parameter('return_speed_deg_s', 60.0)  # ← 복귀 속도

        # --- State ---
        self.target = [0.0, 0.0]   # [yaw_deg, pitch_deg] (화면 오프셋 각도)
        self.cmd = [0.0, 0.0]      # 누적 명령 (절대 각도(deg) 기준 가정)
        self.last_target_ts = time.monotonic()

        # --- I/O ---
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.sub = self.create_subscription(Float64MultiArray, '/ubp/vision/target', self.on_target, qos)
        self.pub = self.create_publisher(Float64MultiArray, '/ubp/head/cmd', qos.depth)

        # --- Timer ---
        self.prev_tick = time.monotonic()
        hz = float(self.get_parameter('update_hz').value)
        self.timer = self.create_timer(max(1e-3, 1.0 / max(1.0, hz)), self.on_timer)

    def on_target(self, msg: Float64MultiArray):
        if len(msg.data) >= 2:
            y, p = msg.data[0], msg.data[1]
            # NaN/Inf 방어
            if not _is_finite(y) or not _is_finite(p):
                self.get_logger().warn('Received non-finite target; ignoring.', throttle_duration_sec=2.0)
                return
            # 현실적인 각도 클램프(입력 단계) — 필요 시 범위 조정
            y = max(-180.0, min(180.0, float(y)))
            p = max(-90.0,  min(90.0,  float(p)))
            self.target = [y, p]
            self.last_target_ts = time.monotonic()

    def on_timer(self):
        now = time.monotonic()
        dt = now - self.prev_tick
        self.prev_tick = now
        if dt <= 0:
            return

        # --- 파라미터 읽기 (런타임 수정 반영) ---
        kp_y = float(self.get_parameter('kp_yaw').value)
        kp_p = float(self.get_parameter('kp_pitch').value)
        max_speed = float(self.get_parameter('max_speed_deg_s').value)     # deg/s
        ly = float(self.get_parameter('limit_yaw_deg').value)
        lp = float(self.get_parameter('limit_pitch_deg').value)
        timeout = float(self.get_parameter('target_timeout_sec').value)
        ret_speed = float(self.get_parameter('return_speed_deg_s').value)  # deg/s

        # --- 타깃 유효성 & 타임아웃 ---
        stale = (now - self.last_target_ts) > timeout

        if stale:
            # 타깃이 오래 안 왔으면 0,0으로 점진 복귀 (속도 제한 ret_speed)
            for i in (0, 1):
                err_to_zero = -self.cmd[i]
                step = max(-ret_speed * dt, min(ret_speed * dt, err_to_zero))
                self.cmd[i] += step
        else:
            # PI 중 I만(적분형) — 화면 중심을 0으로 수렴
            dy = -kp_y * self.target[0]    # deg/s 개념으로 보고 dt로 스케일
            dp = kp_p * self.target[1]

            # per-dt 속도 제한
            dy = max(-max_speed, min(max_speed, dy)) * dt
            dp = max(-max_speed, min(max_speed, dp)) * dt

            # 적분 & anti-windup: 포화되면 그 방향 적분 억제
            # yaw
            new_yaw = self.cmd[0] + dy
            if new_yaw > ly:
                new_yaw = ly
                # 포화 + 증가 방향이면 적분 억제
            elif new_yaw < -ly:
                new_yaw = -ly
            self.cmd[0] = new_yaw

            # pitch
            new_pitch = self.cmd[1] + dp
            if new_pitch > lp:
                new_pitch = lp
            elif new_pitch < -lp:
                new_pitch = -lp
            self.cmd[1] = new_pitch

        # --- 퍼블리시 ---
        out = Float64MultiArray()
        out.data = [float(self.cmd[0]), float(self.cmd[1])]
        self.pub.publish(out)

def main():
    rclpy.init()
    node = HeadTracking()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
