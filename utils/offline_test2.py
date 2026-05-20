#基础调试代码 添加了单独检测物体的功能
#同时解决了单独检测物体时检测框ID为0的bug
import os
import time
import yaml
import tempfile
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import cv2
import open3d as o3d

# 模拟 ROS 的 SafeAclLiteEngine 导入
try:
    from acllite_safe_engine import SafeAclLiteEngine
except ImportError:
    print("❌ 错误: 找不到 acllite_safe_engine.py，请确保它在当前目录或 PYTHONPATH 中")
    exit(1)

# ======================= ⭐ 调试配置区域 =======================
DEBUG_SELECT_ID = None  # 设置为整数可单独调试某一个ID

# ======================= 文件路径配置 =======================
IMAGE_PATH = "/root/zjj/fusion_ws_2_1/shujuji/1_baseline.jpg"
PCD_PATH = "/root/zjj/fusion_ws_2_1/shujuji/1.pcd"

MODEL_PATH = "/root/zjj/sampleYOLOV11/model/yolov11s.om"
SETUP_YAML = "/root/zjj/fusion_ws_2_1/config/setup_config.yaml"
CAMCHAIN_YAML = "/root/zjj/fusion_ws_2_1/config/camchain.yaml"

OUTPUT_DIR = "./offline_results"
PCD_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "pcd_objects")

# ======================= 全局常量 =======================
TARGET_CLASSES = {
    0: ('person', (255, 0, 0)),
    2: ('car', (0, 255, 0)),
    5: ('bus', (0, 0, 255)),
    7: ('truck', (255, 255, 0))
}

FONT_SCALE = 0.6
DBSCAN_EPS = 0.8
DBSCAN_MIN_SAMPLES = 15
MIN_POINTS_FOR_BOX = 40
DEFAULT_Z_MIN = -5.0
DEFAULT_Z_MAX = 1.5
DEFAULT_XY_MAX_DIST = 120.0

BEV_WIDTH = 800
BEV_HEIGHT = 800
BEV_X_RANGE = (-30, 30)
BEV_Y_RANGE = (0, 60)

# =================== 辅助函数：颜色生成 ===================
def get_color_by_id(box_id):
    """根据ID生成固定的亮色"""
    random.seed(box_id)
    return [random.random(), random.random(), random.random()]

# =================== 辅助函数：生成 3D 框的点云 ===================
def create_box_points(center, size, yaw, color=[0, 1, 0], density=0.1):
    """
    将 3D 框的 12 条边转换成密集的点，以便在 PCD 文件中显示。
    """
    cx, cy, cz = center
    l, w, h = size
    
    # 1. 计算 8 个角点 (局部坐标)
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    
    corners = np.vstack([x_corners, y_corners, z_corners]).T # (8, 3)
    
    # 2. 旋转
    c = np.cos(yaw)
    s = np.sin(yaw)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    corners_world = (rot @ corners.T).T + np.array([cx, cy, cz])
    
    # 3. 定义 12 条边的连接关系 (索引)
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0], # 顶面
        [4, 5], [5, 6], [6, 7], [7, 4], # 底面
        [0, 4], [1, 5], [2, 6], [3, 7]  # 侧棱
    ]
    
    box_points = []
    
    # 4. 在每条边上插值生成点
    for start_idx, end_idx in lines:
        p_start = corners_world[start_idx]
        p_end = corners_world[end_idx]
        dist = np.linalg.norm(p_end - p_start)
        num_points = int(dist / density) # 每 density 米生成一个点
        if num_points < 2: num_points = 2
        
        # 线性插值
        for i in range(num_points):
            alpha = i / (num_points - 1)
            p = p_start * (1 - alpha) + p_end * alpha
            box_points.append(p)
            
    # 转为 Open3D 格式
    pcd_box = o3d.geometry.PointCloud()
    pcd_box.points = o3d.utility.Vector3dVector(np.array(box_points))
    # 涂成绿色 (框颜色)
    pcd_box.paint_uniform_color(color) 
    return pcd_box

# =================== 数据类 ===================
@dataclass
class BoundBox:
    x: float; y: float; width: float; height: float
    score: float; class_index: int; index: int
    box_id: int = -1

# =================== 工具函数 ===================
def nms_per_class(boxes: List[BoundBox], iou_thr: float = 0.45) -> List[BoundBox]:
    if not boxes: return []
    groups = {}
    for b in boxes: groups.setdefault(b.class_index, []).append(b)
    kept = []
    for _, arr in groups.items():
        arr.sort(key=lambda b: b.score, reverse=True)
        while arr:
            m = arr.pop(0)
            kept.append(m)
            mx1, my1 = m.x - m.width/2, m.y - m.height/2
            mx2, my2 = m.x + m.width/2, m.y + m.height/2
            m_area = m.width * m.height
            remain = []
            for g in arr:
                gx1, gy1 = g.x - g.width/2, g.y - g.height/2
                gx2, gy2 = g.x + g.width/2, g.y + g.height/2
                ix1, iy1 = max(mx1, gx1), max(my1, gy1)
                ix2, iy2 = min(mx2, gx2), min(my2, gy2)
                iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
                inter = iw * ih
                union = m_area + g.width * g.height - inter
                iou = inter / union if union > 0 else 0.0
                if iou <= iou_thr: remain.append(g)
            arr = remain
    return kept

def visualize_detections(image: np.ndarray, detections: List[BoundBox]) -> np.ndarray:
    vis = image.copy()
    for det in detections:
        x1, y1 = int(det.x - det.width / 2), int(det.y - det.height / 2)
        x2, y2 = int(det.x + det.width / 2), int(det.y + det.height / 2)
        class_name, color = TARGET_CLASSES.get(det.class_index, (f"{det.class_index}", (0, 255, 255)))
        label = f"ID:{det.box_id} {class_name} {det.score:.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return vis

def compute_3d_bbox_from_points(points: np.ndarray) -> Optional[Dict[str, Any]]:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] < MIN_POINTS_FOR_BOX: return None
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov_xy = np.cov(centered[:, :2].T)
    eigvals, eigvecs = np.linalg.eigh(cov_xy)
    main_vec = eigvecs[:, np.argmax(eigvals)]
    yaw = float(np.arctan2(main_vec[1], main_vec[0]))
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    rot = np.array([[cos_y, -sin_y, 0], [sin_y, cos_y, 0], [0, 0, 1]], dtype=np.float32)
    rotated = centered @ rot.T
    size = rotated.max(axis=0) - rotated.min(axis=0)
    return {"center": centroid.tolist(), "size": size.tolist(), "yaw": yaw}

def draw_lidar_projection_on_image(image, points, T_lidar_to_cam, fx, fy, cx, cy):
    vis = image.copy()
    H, W = image.shape[:2]
    homo = np.ones((points.shape[0], 4), dtype=np.float32)
    homo[:, :3] = points
    cam_pts = (T_lidar_to_cam @ homo.T).T
    valid = cam_pts[:, 2] > 0.5
    cam_pts = cam_pts[valid]
    depths = cam_pts[:, 2]
    if len(cam_pts) == 0: return vis
    u = (fx * cam_pts[:, 0] / depths + cx).astype(np.int32)
    v = (fy * cam_pts[:, 1] / depths + cy).astype(np.int32)
    mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, depths = u[mask], v[mask], depths[mask]
    norm_depth = np.clip(depths / 60.0 * 255, 0, 255).astype(np.uint8)
    colors = cv2.applyColorMap(norm_depth, cv2.COLORMAP_JET)
    for i in range(len(u)):
        c = (int(colors[i][0][0]), int(colors[i][0][1]), int(colors[i][0][2]))
        cv2.circle(vis, (u[i], v[i]), 1, c, -1)
    return vis

def draw_bev_map(points, boxes_3d_info):
    bev = np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8)
    def lidar_to_bev(x, y):
        u = int((x - BEV_X_RANGE[0]) / (BEV_X_RANGE[1] - BEV_X_RANGE[0]) * BEV_WIDTH)
        v = int(BEV_HEIGHT - (y - BEV_Y_RANGE[0]) / (BEV_Y_RANGE[1] - BEV_Y_RANGE[0]) * BEV_HEIGHT)
        return u, v
    
    for p in points:
        if BEV_X_RANGE[0] < p[0] < BEV_X_RANGE[1] and BEV_Y_RANGE[0] < p[1] < BEV_Y_RANGE[1]:
            u, v = lidar_to_bev(p[0], p[1])
            cv2.circle(bev, (u, v), 1, (200, 200, 200), -1)
            
    for box in boxes_3d_info:
        if box is None: continue
        cx, cy = box["center_lidar"]["x"], box["center_lidar"]["y"]
        l, w = box["size_3d"]["length"], box["size_3d"]["width"]
        yaw = box["yaw_lidar_rad"]
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -s], [s, c]])
        corners = np.array([[l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2]])
        corners_world = (rot @ corners.T).T + np.array([cx, cy])
        pts_poly = np.array([lidar_to_bev(pt[0], pt[1]) for pt in corners_world], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(bev, [pts_poly], True, (0, 255, 0), 2)
        head_x, head_y = cx + (l/2)*c, cy + (l/2)*s
        cv2.arrowedLine(bev, lidar_to_bev(cx, cy), lidar_to_bev(head_x, head_y), (0, 0, 255), 1)
        
        if "box_id" in box:
            u_text, v_text = lidar_to_bev(cx, cy)
            cv2.putText(bev, f"ID:{box['box_id']}", (u_text+5, v_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return bev

# =================== 核心融合类 ===================
class LidarCameraFusion:
    def __init__(self, model_path, setup_yaml=None, camchain_yaml=None, model_w=640, model_h=640):
        self.model_path = model_path
        self.model_w, self.model_h = int(model_w), int(model_h)
        self.conf_thres, self.nms_thres = 0.25, 0.45
        self.T_lidar_to_cam = np.eye(4, dtype=np.float32)
        self.fx = self.fy = self.cx = self.cy = None
        self.depth_range_min, self.depth_range_max = -float('inf'), float('inf')
        self.z_min_filter, self.z_max_filter = DEFAULT_Z_MIN, DEFAULT_Z_MAX
        self.xy_max_dist = DEFAULT_XY_MAX_DIST
        
        self.current_image = None
        self.image_width = None
        self.image_height = None
        
        if setup_yaml: self._load_setup_yaml(setup_yaml)
        if camchain_yaml: self._load_camchain_yaml(camchain_yaml)
        print("外参矩阵 T_lidar_to_cam:\n", self.T_lidar_to_cam)
        
        print("🚀 初始化 NPU 引擎...")
        self.engine = SafeAclLiteEngine(self.model_path, self.model_w, self.model_h, device_id=0)

    def _load_setup_yaml(self, path):
        try:
            with open(path, 'r') as f: y = yaml.safe_load(f)
            if 'transformation_matrix' in y: self.T_lidar_to_cam = np.array(y['transformation_matrix'], dtype=np.float32)
            if 'depth_range' in y:
                self.depth_range_min = y['depth_range'].get('min', -float('inf'))
                self.depth_range_max = y['depth_range'].get('max', float('inf'))
            if 'z_filter' in y:
                self.z_min_filter = y['z_filter'].get('min', DEFAULT_Z_MIN)
                self.z_max_filter = y['z_filter'].get('max', DEFAULT_Z_MAX)
            if 'xy_max_dist' in y: self.xy_max_dist = float(y['xy_max_dist'])
        except Exception as e: print(f"Yaml Load Error: {e}")

    def _load_camchain_yaml(self, path):
        try:
            with open(path, 'r') as f: cam = yaml.safe_load(f).get('cam0', None)
            if cam and 'intrinsics' in cam: self.fx, self.fy, self.cx, self.cy = map(float, cam['intrinsics'][:4])
        except Exception: pass

    def _infer_yolo_from_frame(self, frame):
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            cv2.imwrite(tmp_path, frame)
            return self.engine.infer_from_jpeg_path(tmp_path)
        except Exception as e:
            print(f"Inference Error: {e}")
            return None
        finally:
            if tmp_path and os.path.exists(tmp_path): os.remove(tmp_path)

    def process_image_array(self, frame):
        self.current_image = frame
        self.image_height, self.image_width = frame.shape[:2]
        if self.cx is None: self.cx, self.cy = self.image_width/2, self.image_height/2
        
        outputs = self._infer_yolo_from_frame(frame)
        if outputs is None: return [], frame.copy()
        
        out = outputs[0]
        if out.ndim == 3: out = out[0]
        elif out.ndim == 1:
            n = out.size // (4 + 80)
            out = out.reshape(4 + 80, n)
        
        xywh = out[0:4, :]
        cls_scores = out[4:, :]
        cls_idx = np.argmax(cls_scores, axis=0)
        scores = cls_scores[cls_idx, np.arange(out.shape[1])]
        keep = scores > self.conf_thres
        xywh, cls_idx, scores = xywh[:, keep], cls_idx[keep], scores[keep]
        indices = np.where(keep)[0]
        
        xs = xywh[0] * self.image_width / self.model_w
        ys = xywh[1] * self.image_height / self.model_h
        ws = xywh[2] * self.image_width / self.model_w
        hs = xywh[3] * self.image_height / self.model_h
        
        dets = []
        for i in range(len(xs)):
            c = int(cls_idx[i])
            if c in TARGET_CLASSES:
                dets.append(BoundBox(float(xs[i]), float(ys[i]), float(ws[i]), float(hs[i]), float(scores[i]), c, int(indices[i])))
        dets = nms_per_class(dets, self.nms_thres)
        
        for i, d in enumerate(dets):
            d.box_id = i
            
        return dets, None 

    def fuse_with_pointcloud(self, points, dets):
        if self.current_image is None: return None, None, [None]*len(dets)
        pts = np.asarray(points, dtype=np.float32)
        if pts.size == 0: return None, None, [None]*len(dets)

        img_proj = draw_lidar_projection_on_image(self.current_image, pts, self.T_lidar_to_cam, self.fx, self.fy, self.cx, self.cy)

        # 滤波
        mask = np.all(np.isfinite(pts), axis=1)
        mask &= (pts[:, 1] > self.depth_range_min) & (pts[:, 1] < self.depth_range_max)
        mask &= (pts[:, 2] >= self.z_min_filter) & (pts[:, 2] <= self.z_max_filter)
        if self.xy_max_dist: mask &= np.hypot(pts[:, 0], pts[:, 1]) <= self.xy_max_dist
        pts_filtered = pts[mask]
        
        # ⭐ 初始化主点云 (背景设为灰色)
        pcd_main = o3d.geometry.PointCloud()
        pcd_main.points = o3d.utility.Vector3dVector(pts_filtered)
        pcd_main.paint_uniform_color([0.5, 0.5, 0.5]) # 灰色背景
        
        # 记录所有要合并的几何体
        geometries_to_merge = [pcd_main]

        homo = np.ones((len(pts_filtered), 4), dtype=np.float32)
        homo[:, :3] = pts_filtered
        cam_pts = (self.T_lidar_to_cam @ homo.T).T
        valid = cam_pts[:, 2] > 1e-3
        cam_pts = cam_pts[valid]
        pts_valid = pts_filtered[valid]
        
        u = self.fx * cam_pts[:, 0] / cam_pts[:, 2] + self.cx
        v = self.fy * cam_pts[:, 1] / cam_pts[:, 2] + self.cy
        in_img = (u >= 0) & (u < self.image_width) & (v >= 0) & (v < self.image_height)
        u, v, pts_valid = u[in_img], v[in_img], pts_valid[in_img]

        enhanced_results = []
        if not os.path.exists(PCD_OUTPUT_DIR): os.makedirs(PCD_OUTPUT_DIR)

        for idx, det in enumerate(dets):
            res = None
            margin = 10
            roi_mask = (u >= det.x - det.width/2 - margin) & (u <= det.x + det.width/2 + margin) & \
                       (v >= det.y - det.height/2 - margin) & (v <= det.y + det.height/2 + margin)
            roi_pts = pts_valid[roi_mask]
            
            # 为当前物体生成颜色
            obj_color = get_color_by_id(det.box_id)
            
            # ⭐ 功能 1: 将检测到的点云染色并叠加
            if len(roi_pts) > 0:
                pcd_obj = o3d.geometry.PointCloud()
                pcd_obj.points = o3d.utility.Vector3dVector(roi_pts)
                pcd_obj.paint_uniform_color(obj_color) # 染色
                geometries_to_merge.append(pcd_obj)
                
                # 保存单独文件
                o3d.io.write_point_cloud(os.path.join(PCD_OUTPUT_DIR, f"obj_ID{det.box_id}_roi.pcd"), pcd_obj)
            
            if len(roi_pts) >= MIN_POINTS_FOR_BOX:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(roi_pts)
                pcd = pcd.voxel_down_sample(0.15)
                ds_pts = np.asarray(pcd.points, dtype=np.float32)
                if len(ds_pts) >= MIN_POINTS_FOR_BOX:
                    labels = np.array(pcd.cluster_dbscan(eps=DBSCAN_EPS, min_points=DBSCAN_MIN_SAMPLES, print_progress=False))
                    if labels.size > 0:
                        lbl_pts = ds_pts if labels.max() < 0 else ds_pts[labels == np.argmax(np.bincount(labels[labels>=0]))]
                        if len(lbl_pts) >= MIN_POINTS_FOR_BOX:
                            box = compute_3d_bbox_from_points(lbl_pts)
                            if box:
                                res = {
                                    "center_lidar": {"x": box["center"][0], "y": box["center"][1], "z": box["center"][2]},
                                    "size_3d": {"length": box["size"][0], "width": box["size"][1], "height": box["size"][2]},
                                    "yaw_lidar_rad": box["yaw"],
                                    "box_id": det.box_id
                                }
                                
                                # ⭐ 功能 2: 生成 3D 框的点云线段并叠加
                                pcd_box = create_box_points(box["center"], box["size"], box["yaw"], color=[0, 1, 0]) # 绿色框
                                geometries_to_merge.append(pcd_box)
                                
            enhanced_results.append(res)

        # ⭐ 合并所有点云 (背景 + 各个物体 + 框)
        final_pcd = geometries_to_merge[0]
        for i in range(1, len(geometries_to_merge)):
            final_pcd += geometries_to_merge[i]
            
        # 保存最终的全场景标注点云
        o3d.io.write_point_cloud(os.path.join(OUTPUT_DIR, "final_scene_labeled.pcd"), final_pcd)

        img_bev = draw_bev_map(pts_filtered, enhanced_results)
        return img_proj, img_bev, enhanced_results

# =================== 主程序 ===================
def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    print(f"📂 读取图像: {IMAGE_PATH}")
    if not os.path.exists(IMAGE_PATH): return
    frame = cv2.imread(IMAGE_PATH)

    print(f"📂 读取点云: {PCD_PATH}")
    if not os.path.exists(PCD_PATH): return
    
    pcd_load = o3d.io.read_point_cloud(PCD_PATH)
    points = np.asarray(pcd_load.points, dtype=np.float32)
    print(f"✅ 点云加载完成，点数: {points.shape[0]}")

    fusion = LidarCameraFusion(MODEL_PATH, SETUP_YAML, CAMCHAIN_YAML)

    t0 = time.time()
    dets, _ = fusion.process_image_array(frame)
    print(f"🧠 YOLO 2D 检测完成: {len(dets)} 个目标")

    # ID 筛选逻辑
    if DEBUG_SELECT_ID is not None:
        if 0 <= DEBUG_SELECT_ID < len(dets):
            print(f"🔍 [调试模式] 仅保留检测框 ID: {DEBUG_SELECT_ID}")
            dets = [dets[DEBUG_SELECT_ID]]
        else:
            print(f"⚠️ 警告: DEBUG_SELECT_ID {DEBUG_SELECT_ID} 超出范围")

    vis2d = visualize_detections(frame, dets)
    img_proj, img_bev, results = fusion.fuse_with_pointcloud(points, dets)
    print(f"📦 3D 融合完成，耗时 {time.time()-t0:.3f}s")

    # 打印结果
    print("\n" + "="*30 + " 检测结果 " + "="*30)
    for i, res in enumerate(results):
        det = dets[i]
        cname = TARGET_CLASSES.get(det.class_index, ("u",None))[0]
        if res:
            print(f"[ID:{det.box_id}] {cname} (score={det.score:.2f}) | 3D Pos: ({res['center_lidar']['x']:.2f}, {res['center_lidar']['y']:.2f}, {res['center_lidar']['z']:.2f})")
        else:
            print(f"[ID:{det.box_id}] {cname} (score={det.score:.2f}) | 3D: 未匹配到点云")

    cv2.imwrite(os.path.join(OUTPUT_DIR, "result_01_raw.jpg"), frame)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "result_02_2d_box.jpg"), vis2d)
    
    if img_proj is not None: cv2.imwrite(os.path.join(OUTPUT_DIR, "result_03_projection.jpg"), img_proj)
    if img_bev is not None: cv2.imwrite(os.path.join(OUTPUT_DIR, "result_04_bev.jpg"), img_bev)
        
    print(f"💾 全场景标注PCD已保存: {os.path.join(OUTPUT_DIR, 'final_scene_labeled.pcd')}")
    print(f"💾 物体独立PCD文件已保存至: {PCD_OUTPUT_DIR}")
    print("\n✅ 所有处理完成。")

if __name__ == "__main__":
    main()