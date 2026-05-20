cmake_minimum_required(VERSION 3.8)
project(fusion_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 寻找 ROS 2 依赖
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# 声明需要编译的 msg 和 srv
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Box2D.msg"
  "msg/Box3D.msg"
  "msg/AlarmTTJ.msg"
  "msg/FusionResult.msg"
  "srv/FusionCommand.srv"
  DEPENDENCIES geometry_msgs std_msgs sensor_msgs
)

# 极其关键：生成 ROS 2 环境钩子并注册包
ament_package()