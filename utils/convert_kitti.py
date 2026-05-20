import open3d as o3d
import numpy as np

def transform_point_cloud(input_pcd_path, output_npy_path):
    # 读取PCD文件
    pcd = o3d.io.read_point_cloud(input_pcd_path)
    
    # 获取点云数据
    points = np.asarray(pcd.points)
    
    # 假设原始点云的坐标系是 x轴向右为正，Y轴向前为正，Z轴向上为正
    # 我们需要将其转换为 OpenPCDet 的坐标系：x指向前方，y指向左侧，z指向上方
    
    # 进行坐标变换
    transformed_points = np.zeros_like(points)
    transformed_points[:, 0] = -points[:, 1]  # x轴指向前方
    transformed_points[:, 1] = -points[:, 0]  # y轴指向左侧
    transformed_points[:, 2] = points[:, 2]   # z轴指向上方
    
    # 设置强度信息（如果没有强度信息，设置为0）
    # 如果有强度信息，可以在这里进行处理
    # 例如，假设强度信息存储在第四列
    # intensity = points[:, 3]
    # normalized_intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())
    # transformed_points = np.hstack((transformed_points, normalized_intensity[:, None]))
    
    # 如果没有强度信息，设置为0
    transformed_points = np.hstack((transformed_points, np.zeros((transformed_points.shape[0], 1))))
    
    # 保存转换后的点云数据到Numpy文件
    np.save(output_npy_path, transformed_points)
    print(f"Transformed point cloud saved to {output_npy_path}")

if __name__ == "__main__":
    input_pcd_path = "/root/zjj/fusion_ws_2_1/shujuji/1.pcd"  # 替换为您的输入PCD文件路径
    output_npy_path = "/root/zjj/fusion_ws_2_1/shujuji/output.npy"  # 替换为您的输出Numpy文件路径
    transform_point_cloud(input_pcd_path, output_npy_path)