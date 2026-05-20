import rclpy
from rclpy.node import Node
import time
import cv2
import os
import numpy as np
from cv_bridge import CvBridge

from fusion_msgs.srv import FusionCommand
from fusion_msgs.msg import FusionResult

# ================= 配置区域 =================
TEST_RTSP = "rtsp://admin:Aa805401@192.168.1.208:554/Streaming/Channels/101"
TEST_PC_TOPIC = "/topic200"        # 对应服务端 pointcloud_topic
TEST_PUB_TOPIC_VAL = "/topic_r31_201"  # 对应服务端 topic_r31
SAVE_DIR = "output_images"
# ===========================================

class FusionTestClient(Node):
    def __init__(self):
        super().__init__('fusion_test_client')
        
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)

        self.cli = self.create_client(FusionCommand, '/fusion_command')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('等待服务端 /fusion_command 上线...')
            
        self.sub = self.create_subscription(
            FusionResult,
            TEST_PUB_TOPIC_VAL,
            self.listener_callback,
            10
        )
        
        self.bridge = CvBridge()
        self.last_time = time.time()
        self.frame_count = 0

    def send_start(self):
        req = FusionCommand.Request()
        # ⭐ 修改点 1: 字段名改为 enable_mode，值为 1 (启动)
        req.enable_mode = 1 
        req.rtsp = TEST_RTSP
        req.pointcloud_topic = TEST_PC_TOPIC
        req.topic_r31 = TEST_PUB_TOPIC_VAL 

        self.get_logger().info("📤 发送 Start (enable_mode=1) 指令...")
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        try:
            res = future.result()
            if res.success:
                self.get_logger().info(f"✅ 启动成功: {res.message}")
            else:
                self.get_logger().error(f"❌ 启动失败: {res.message}")
        except Exception as e:
            self.get_logger().error(f"❌ 调用异常: {e}")

    def listener_callback(self, msg: FusionResult):
        now = time.time()
        interval = now - self.last_time
        self.last_time = now
        self.frame_count += 1

        print("\n" + "=" * 60)
        self.get_logger().info(f"📥 收到第 {self.frame_count} 帧 | 间隔: {interval:.2f}s")
        
        # --- 1. 打印详细数据 ---
        print(f"\n[📦 2D 检测框列表] (共 {len(msg.box_2d)} 个)")
        for i, b in enumerate(msg.box_2d):
            print(f"  [{i}] {b.class_name:<6} score={b.score:.2f} | xywh=[{int(b.cx)},{int(b.cy)},{int(b.width)},{int(b.height)}]")

        print(f"\n[📦 3D 融合框列表] (共 {len(msg.box_3d)} 个)")
        for i, b in enumerate(msg.box_3d):
            # center_lidar 是 geometry_msgs/Point
            print(f"  [{i}] {b.class_name:<6} score={b.score:.2f} | Pos=({b.center_lidar.x:.2f}, {b.center_lidar.y:.2f}, {b.center_lidar.z:.2f}) | Pts={b.point_count}")
        
        # 打印 point 列表信息
        print(f"\n[📍 Point 列表信息] (共 {len(msg.point)} 个点)")
        for i, p in enumerate(msg.point):
            print(f"  Point[{i}]: x={p.x:.2f}, y={p.y:.2f}, z={p.z:.2f}")
            
        # --- 2. 保存图像 ---
        # 保存 Image1 (相机原图)
        if msg.image1.height > 0:
            try:
                img1 = self.bridge.imgmsg_to_cv2(msg.image1, desired_encoding='bgr8')
                name1 = os.path.join(SAVE_DIR, f"frame_{self.frame_count:04d}_Raw.jpg")
                cv2.imwrite(name1, img1)
                print(f"  💾 保存原图: {name1}")
            except Exception as e:
                print(f"  ❌ 保存原图失败: {e}")

        # 保存 Image2 (含2D框图)
        if msg.image2.height > 0:
            try:
                img2 = self.bridge.imgmsg_to_cv2(msg.image2, desired_encoding='bgr8')
                name2 = os.path.join(SAVE_DIR, f"frame_{self.frame_count:04d}_2D.jpg")
                cv2.imwrite(name2, img2)
                print(f"  💾 保存2D框图: {name2}")
            except Exception as e:
                print(f"  ❌ 保存2D框图失败: {e}")

def main(args=None):
    rclpy.init(args=args)
    client_node = FusionTestClient()

    try:
        client_node.send_start()
        print("\n🚀 监听中... 按 Ctrl+C 退出并停止服务端\n")
        rclpy.spin(client_node)
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户终止操作")
    finally:
        if rclpy.ok():
            print("正在请求停止服务端...")
            try:
                req = FusionCommand.Request()
                # ⭐ 修改点 2: 字段名改为 enable_mode，值为 0 (停止)
                req.enable_mode = 0 
                client_node.cli.call_async(req)
                time.sleep(0.5) 
            except Exception:
                pass
        
        client_node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    main()