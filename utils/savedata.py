#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import cv2
import numpy as np
import struct
import time
import os
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class SyncSaver(Node):
    def __init__(self):
        super().__init__('sync_data_saver')

        # RTSP 设置 - 加强版
        self.rtsp_url = "rtsp://admin:Aa805401@192.168.1.208:554/Streaming/Channels/101"
        
        # 强制 TCP + FFMPEG 后端
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

        self.cap = None
        max_retries = 15
        for attempt in range(max_retries):
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            if self.cap.isOpened():
                self.get_logger().info(f"RTSP 连接成功！URL: {self.rtsp_url}")
                # 可选：设置缓冲区大小，减少延迟
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                break
            self.get_logger().warn(f"RTSP 尝试 {attempt+1}/{max_retries} 失败...")
            time.sleep(2)

        if not self.cap.isOpened():
            self.get_logger().fatal("无法连接 RTSP 摄像头，请检查网络/账号/URL/相机状态")
            raise RuntimeError("RTSP connection failed after retries")

        # 订阅雷达点云
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )

        self.subscription = self.create_subscription(
            PointCloud2,
            '/topic200',  # ← 确认这个话题名是否正确（用 ros2 topic list 查看）
            self.callback,
            qos
        )

        self.get_logger().info("同步保存程序启动，等待雷达数据...")

    def callback(self, msg: PointCloud2):
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # 抓取当前最新帧（近似同步）
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("抓取 RTSP 帧失败，跳过本次保存")
            return

        # 保存 RGB
        img_filename = f"{timestamp:.6f}_rgb.png"
        cv2.imwrite(img_filename, frame)
        self.get_logger().info(f"保存图像: {img_filename}")

        # 提取点云（只取 x y z）
        points = []
        for i in range(0, len(msg.data), msg.point_step):
            try:
                x = struct.unpack_from('f', msg.data, i + 0)[0]
                y = struct.unpack_from('f', msg.data, i + 4)[0]
                z = struct.unpack_from('f', msg.data, i + 8)[0]
                points.append([x, y, z])
            except struct.error:
                self.get_logger().warn("点云数据解析异常，跳过部分点")
                continue

        if not points:
            self.get_logger().warn("点云为空，跳过保存 PCD")
            return

        points = np.array(points)

        # 保存 PCD
        pcd_filename = f"{timestamp:.6f}_radar.pcd"
        self.save_pcd(points, pcd_filename)
        self.get_logger().info(f"保存 PCD: {pcd_filename}")

    def save_pcd(self, points, filename):
        try:
            with open(filename, 'w') as f:
                f.write("VERSION .7\n")
                f.write("FIELDS x y z\n")
                f.write("SIZE 4 4 4\n")
                f.write("TYPE F F F\n")
                f.write("COUNT 1 1 1\n")
                f.write(f"WIDTH {len(points)}\n")
                f.write("HEIGHT 1\n")
                f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
                f.write(f"POINTS {len(points)}\n")
                f.write("DATA ascii\n")
                for p in points:
                    f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        except Exception as e:
            self.get_logger().error(f"保存 PCD 失败: {e}")

    def destroy_node(self):
        if self.cap is not None:
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SyncSaver()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"程序异常退出: {e}")
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()