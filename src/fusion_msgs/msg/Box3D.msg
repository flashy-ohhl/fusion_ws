int32 class_id              # 目标的类别序号
string class_name           # 目标的类别名称
float32 score               # 目标的检测置信度得分

geometry_msgs/Point center_lidar        # 目标 3D 几何中心在 LiDAR（激光雷达）坐标系 下的 (x, y, z) 坐标

float32 length              # 3D 边界框的长度
float32 width               # 3D 边界框的宽度
float32 height              # 3D 边界框的高度

float32 yaw_lidar_deg       # 目标在 LiDAR 坐标系下绕 Z 轴旋转的偏航角
float32 yaw_lidar_rad       # 目标在 LiDAR 坐标系下绕 Z 轴旋转的偏航角 弧度制
int32 point_count           # 属于该目标的 LiDAR 点云数量