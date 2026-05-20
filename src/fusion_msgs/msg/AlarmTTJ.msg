# 单辆 car 的"挡墙运动方向告警"判定结果
bool status                 # true=违规, false=正常/历史不足
float32 degree              # 运动方向 a 与最近墙段 b 的夹角 (度)，[0, 90]；历史不足时 = -1.0
sensor_msgs/Image image4    # 告警 BEV；仅在 status=true 时画详细，false 时为占位空图
