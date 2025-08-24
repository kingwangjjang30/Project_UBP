#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy, math, time
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge
import cv2
import os

# YOLO (얼굴)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False


class CameraViewer(Node):
    def __init__(self):
        super().__init__('camera_viewer_node')

        # === 파라미터 ===
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('publish_hz', 15.0)
        self.declare_parameter('face_conf_threshold', 0.40)
        self.declare_parameter('show_debug_text', True)
        self.declare_parameter('face_model', 'yolov8n-face-lindevs.pt')

        self.color_topic = self.get_parameter('color_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.min_period = 1.0 / max(1.0, float(self.get_parameter('publish_hz').value))
        self.face_conf_th = float(self.get_parameter('face_conf_threshold').value)
        self.show_debug_text = bool(self.get_parameter('show_debug_text').value)
        self.face_model_name = self.get_parameter('face_model').value

        self.get_logger().info('=== Camera Viewer Started ===')
        self.get_logger().info(f'Color topic: {self.color_topic}')
        self.get_logger().info(f'CameraInfo topic: {self.camera_info_topic}')

        pkg_share = get_package_share_directory('vision')
        model_path = os.path.join(pkg_share, 'models', self.face_model_name)

        self.bridge = CvBridge()
        self.frame_count = 0
        self.last_frame = None
        self.last_pub_ts = 0.0

        # 내참 (fx, fy, cx, cy)
        self.fx = self.fy = None
        self.cx = self.cy = None

        # 퍼블리셔
        self.target_pub = self.create_publisher(Float64MultiArray, '/ubp/vision/target', 10)

        # QoS (센서 데이터용)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # 구독
        self.info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_cb, sensor_qos)
        self.color_sub = self.create_subscription(Image, self.color_topic, self.color_callback, sensor_qos)

        # YOLO 얼굴 모델 로드
        self.face_detector = None
        if YOLO_AVAILABLE:
            try:
                # 먼저 지정된 모델 시도
                if os.path.exists(model_path):
                    self.face_detector = YOLO(model_path)
                    self.get_logger().info(f'YOLO(face) loaded from: {model_path}')
                else:
                    # 모델 파일이 없으면 일반 YOLOv8로 대체 (person 클래스 사용)
                    self.get_logger().warn(f'Face model not found at: {model_path}')
                    self.get_logger().info('Trying to use general YOLOv8 model for person detection...')
                    self.face_detector = YOLO('yolov8n.pt')  # 자동 다운로드됨
                    self.use_person_detection = True
                    self.get_logger().info('Using YOLOv8 person detection as fallback')
            except Exception as e:
                self.get_logger().warn(f'Failed to load face model "{self.face_model_name}": {e}')
                try:
                    # 최후 수단: 일반 YOLOv8 사용
                    self.face_detector = YOLO('yolov8n.pt')
                    self.use_person_detection = True
                    self.get_logger().info('Using YOLOv8 person detection as emergency fallback')
                except Exception as e2:
                    self.get_logger().error(f'All model loading failed: {e2}')
        else:
            self.get_logger().warn('Ultralytics not available. Face detection disabled.')
        
        # 사람 감지 모드 플래그
        self.use_person_detection = getattr(self, 'use_person_detection', False)

        # 창 준비
        cv2.namedWindow('Camera Viewer', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Camera Viewer', 800, 600)
        self.get_logger().info('Ready! Press ESC or Q to quit')

    # === 콜백들 ===
    def camera_info_cb(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.get_logger().info(
            f'CameraInfo: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}',
            throttle_duration_sec=5.0
        )

    def color_callback(self, msg: Image):
        try:
            self.frame_count += 1
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.last_frame = cv_image
            self._detect_and_publish(cv_image)
            self.get_logger().info(f'Frame #{self.frame_count}', throttle_duration_sec=1.0)
        except Exception as e:
            self.get_logger().error(f'Error: {str(e)}')

    # === 얼굴 감지 & 퍼블리시 ===
    def _detect_and_publish(self, frame):
        if self.face_detector is None:
            return

        try:
            res = self.face_detector(frame, verbose=False)[0]
        except Exception as e:
            self.get_logger().warn(f'YOLO inference error: {e}', throttle_duration_sec=5.0)
            return

        # 감지 결과 처리
        best = None
        best_area = 0.0
        
        if self.use_person_detection:
            # 사람 감지 모드 (클래스 0 = person)
            for b in res.boxes:
                if b.cls is None or int(b.cls[0]) != 0:  # person class만 필터
                    continue
                conf = float(b.conf[0].item()) if b.conf is not None else 0.0
                if conf < self.face_conf_th:
                    continue
                x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                if area > best_area:
                    best_area, best = area, (x1, y1, x2, y2, conf, 'person')
        else:
            # 얼굴 감지 모드
            for b in res.boxes:
                conf = float(b.conf[0].item()) if b.conf is not None else 0.0
                if conf < self.face_conf_th:
                    continue
                x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                if area > best_area:
                    best_area, best = area, (x1, y1, x2, y2, conf, 'face')

        if best is None:
            return

        x1, y1, x2, y2, conf, label = best
        cx_t = 0.5 * (x1 + x2)
        cy_t = 0.5 * (y1 + y2)

        # 박스 그리기
        color = (255, 128, 0) if label == 'face' else (0, 255, 128)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f'{label} {conf:.2f}', (int(x1), max(0, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        if self.fx is None:
            self.get_logger().warn('No CameraInfo yet; skip publish', throttle_duration_sec=2.0)
            return

        yaw_rad = math.atan2((cx_t - self.cx) / self.fx, 1.0)
        pitch_rad = -math.atan2((cy_t - self.cy) / self.fy, 1.0)
        yaw_deg = math.degrees(yaw_rad)
        pitch_deg = math.degrees(pitch_rad)

        now = time.time()
        if now - self.last_pub_ts >= self.min_period:
            msg_out = Float64MultiArray()
            msg_out.data = [yaw_deg, pitch_deg]
            self.target_pub.publish(msg_out)
            self.last_pub_ts = now

        cv2.circle(frame, (int(cx_t), int(cy_t)), 4, (0, 255, 255), -1)
        cv2.circle(frame, (int(self.cx), int(self.cy)), 4, (255, 0, 255), -1)
        cv2.line(frame, (int(self.cx), int(self.cy)), (int(cx_t), int(cy_t)), (0, 255, 255), 1)
        if self.show_debug_text:
            cv2.putText(frame, f"yaw={yaw_deg:.2f} deg  pitch={pitch_deg:.2f} deg",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)


def main(args=None):
    rclpy.init(args=args)
    node = CameraViewer()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)

            if node.last_frame is not None:
                frame = node.last_frame
                h, w = frame.shape[:2]
                cv2.putText(frame, f'Frame: {node.frame_count}', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f'Size: {w}x{h}', (10, 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow('Camera Viewer', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                node.get_logger().info('Exit key pressed')
                break

    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()