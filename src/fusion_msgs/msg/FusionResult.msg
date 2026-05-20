# 新增的透传 ID 字段 (对应要求 3)
string camera1_id
fusion_msgs/Box2D[] box_1    # 原 box_2d
sensor_msgs/Image image1_1  # 原 image1
sensor_msgs/Image image1_2  # 原 image2

string camera2_id
fusion_msgs/Box2D[] box_2    # 原 box_2d
sensor_msgs/Image image2_1  # 原 image1
sensor_msgs/Image image2_2  # 原 image2

string lidar_id
# 检测框数据 (对应要求 4)
fusion_msgs/Box3D[] box_3d  # 原 box_3d
sensor_msgs/Image image3    # 保持不变

# 挡墙顶点（从 wall_info_topic 透传过来，并入主消息）
geometry_msgs/Point[] height_points
# 挡墙运动方向告警，每辆 car 一项；按当前帧 car 检测顺序排列
fusion_msgs/AlarmTTJ[] alarm_ttj