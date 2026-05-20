# 修改3D框拟合算法
## 是不是要根据不同类别，拟合不同的3D框
##更改输入输出接口
4.25 删除图片保存功能
4.28 尝试训练加速
5.4 尝试绕过DVPP处理
5.5
改动说明
acllite_safe_engine.py：

新增 bgr_to_nv12() 工具函数，BGR图像 → 模型需要的NV12格式numpy数组
新增 infer_from_bgr() 方法，直接传numpy，绕过JPEG文件IO和DVPP
原有 infer_from_jpeg_path() 等方法保留，万一新方法有问题可以一键回退

fusion_server_node.py：

唯一改动：_infer_yolo_from_frame() 改成调用 engine.infer_from_bgr(frame)
删除了 tempfile 相关的所有代码（不再需要写文件）
加了 cam1/cam2 各自的推理耗时打印，方便你看效果


# 2026.5.16 整合了输出的数据结构，加了挡墙离群点的剔除
#           但是现在依旧向本地保存图片用于调试，这样就会造成一轮计算时间有点长