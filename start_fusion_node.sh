#!/bin/bash

# 等待系统完全启动
sleep 10

# 设置环境
source /opt/ros/humble/setup.bash  # 根据你的ROS2版本修改（humble/foxy/galactic等）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /root/zjj/fusion_ws/install/setup.bash
export PYTHONPATH=$PYTHONPATH:/root/zjj/samples/python/common/acllite
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

export ROS_DOMAIN_ID=66

export ROS_LOCALHOST_ONLY=1

export CYCLONEDDS_URI=file:///root/camera_lidar/cyclonedds_config.xml

# 切换到工作目录（模型路径是相对路径，必须切换）
cd /root/zjj/fusion_ws

# 启动节点
exec ros2 run fusion_server fusion_server_node \
    --ros-args \
    -p model_path:=./model/yolov11s.om \
    -p setup_yaml:=./config/setup_config.yaml \
    -p camchain_yaml:=./config/camchain.yaml