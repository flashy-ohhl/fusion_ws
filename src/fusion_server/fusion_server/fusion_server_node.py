import os
import time
import yaml
import threading
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import cv2
import open3d as o3d

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs_py import point_cloud2 as pc2
from cv_bridge import CvBridge
from geometry_msgs.msg import Point

from fusion_msgs.msg import FusionResult, Box2D, Box3D, AlarmTTJ
from fusion_msgs.srv import FusionCommand

from .acllite_safe_engine import SafeAclLiteEngine

# ======================= 全局配置 =======================
TARGET_CLASSES = {
    0: ('person', (255, 0, 0)),
    2: ('car', (0, 255, 0)),
    7: ('truck', (255, 255, 0))
}

DBSCAN_MIN_SAMPLES = 15
MIN_POINTS_FOR_BOX = 40

DEFAULT_Z_MIN = -5.0
DEFAULT_Z_MAX = 1.5
DEFAULT_XY_MAX_DIST = 60.0

FUSION_PERIOD_SEC = 2
WARMUP_SECONDS = 3.0

BEV_WIDTH = 1000
BEV_HEIGHT = 1000
BEV_X_RANGE = (-35, 35)
BEV_Y_RANGE = (-5, 65)

# === BEV 投影参数（image3 / image4 共用，纯垂直俯视，完全忽略 z）===
# 视野覆盖：左右 ±BEV_WIDTH/2/BEV_SCALE_X ≈ ±33m，前方 BEV_ORIGIN_V/BEV_SCALE_Y ≈ 60m，后方少量
BEV_SCALE_X = 15.0                       # LiDAR x 方向 → 像素 / 米
BEV_SCALE_Y = 15.0                       # LiDAR y 方向 → 像素 / 米
BEV_ORIGIN_U = BEV_WIDTH // 2            # LiDAR 原点对应 BEV 的 u 像素（水平居中）
BEV_ORIGIN_V = BEV_HEIGHT - 100          # LiDAR 原点对应 BEV 的 v 像素（靠底部，给前方留空间）

WALL_CURVE_COLOR = (0, 165, 255)
WALL_CURVE_THICKNESS = 3
WALL_POINT_COLOR = (0, 100, 255)
WALL_POINT_RADIUS = 5

# === 挡墙运动方向告警 (alarm_ttj) 配置 ===
ALARM_TTJ_NUM_SEGMENTS = 5            # 挡墙均分段数
ALARM_TTJ_HISTORY_LEN = 30            # 轨迹直线 a 需要的历史帧数
ALARM_TTJ_ANGLE_THRESHOLD_DEG = 20.0  # a 与 b 夹角小于此值 → 违规
ALARM_TTJ_TRACK_MATCH_DIST = 5.0      # car 跨帧关联的最近邻距离阈值 (米)
ALARM_TTJ_TRACK_MAX_MISSED = 10       # 连续未匹配多少次后丢弃轨迹

# === 调试: 把每次 _process 的所有图像保存到 output/<frame_xxx_ts>/ ===
DEBUG_SAVE_OUTPUT = False                                     # 总开关，关掉就完全跳过 IO
DEBUG_OUTPUT_DIR = os.path.join(os.getcwd(), 'output')       # 启动时 cwd 是工作空间根
DEBUG_SAVE_EXT = '.jpg'                                      # 想看无损用 '.png'


@dataclass
class BoundBox:
    x: float; y: float; width: float; height: float
    score: float; class_index: int; index: int


# =================== 工具函数 ===================
def make_debug_frame_dir(frame_idx: int) -> Optional[str]:
    """为本次 _process 创建一个子文件夹，返回路径。开关关或失败时返回 None。"""
    if not DEBUG_SAVE_OUTPUT:
        return None
    try:
        ts = time.strftime('%Y%m%d_%H%M%S')
        sub = os.path.join(DEBUG_OUTPUT_DIR, f"frame_{frame_idx:06d}_{ts}")
        os.makedirs(sub, exist_ok=True)
        return sub
    except Exception as e:
        print(f"[debug] make_debug_frame_dir failed: {e}")
        return None


def save_debug_img(frame_dir: Optional[str], name: str, img: Optional[np.ndarray]) -> None:
    """保存一张 BGR 图像到 frame_dir/name。frame_dir 或 img 为空时直接跳过。"""
    if frame_dir is None or img is None:
        return
    try:
        cv2.imwrite(os.path.join(frame_dir, name + DEBUG_SAVE_EXT), img)
    except Exception as e:
        print(f"[debug] save_debug_img {name} failed: {e}")


def nms_per_class(boxes: List[BoundBox], iou_thr: float = 0.45) -> List[BoundBox]:
    if not boxes: return []
    groups: Dict[int, List[BoundBox]] = {}
    for b in boxes:
        groups.setdefault(b.class_index, []).append(b)
    kept: List[BoundBox] = []
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
        class_name, color = TARGET_CLASSES.get(det.class_index, (f"class_{det.class_index}", (0, 255, 255)))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"{class_name}:{det.score:.2f}", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return vis


def robust_pointcloud_conversion(msg: PointCloud2) -> Optional[np.ndarray]:
    """
    极速解析点云：利用 numpy 的 frombuffer 直接读取内存，耗时通常在 5~20 毫秒
    """
    if not msg.data:
        return None
    
    try:
        # 1. 解析字段偏移量，找到 x, y, z 在每一个点的 byte 里的位置
        offsets = {f.name: f.offset for f in msg.fields if f.name in ['x', 'y', 'z']}
        if len(offsets) < 3:
            return None
        
        # 2. 将一维的 bytes 直接映射成二维的 uint8 numpy 数组
        # 每一行是一个 point (长度为 msg.point_step)
        cloud_data = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
        
        # 3. 利用视图(view)和切片快速提取 float32 数据 (假设 xyz 都是 32 位浮点数)
        # 这样做完全避免了 Python 层的 for 循环
        x = cloud_data[:, offsets['x']:offsets['x']+4].view(np.float32)
        y = cloud_data[:, offsets['y']:offsets['y']+4].view(np.float32)
        z = cloud_data[:, offsets['z']:offsets['z']+4].view(np.float32)
        
        # 4. 拼接并过滤 NaN
        pts = np.hstack((x, y, z))
        mask = np.isfinite(pts).all(axis=1)
        
        return pts[mask]
    except Exception as e:
        print(f"Fast PC conversion failed: {e}")
        return None


def compute_3d_bbox_from_points(points: np.ndarray, class_index: int = -1) -> Optional[Dict[str, Any]]:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] < MIN_POINTS_FOR_BOX: return None
    z_min, z_max = pts[:, 2].min(), pts[:, 2].max()
    z_min -= 0.1; z_max += 0.1
    height = max(z_max - z_min, 0.5)
    pts_2d_cv = pts[:, :2].reshape(-1, 1, 2).astype(np.float32)
    rect = cv2.minAreaRect(pts_2d_cv)
    (cx, cy), (w, l), angle = rect
    yaw = np.radians(angle)
    if w > l: w, l = l, w; yaw += np.pi / 2.0
    # 此刻 l ≥ w，yaw 指向"几何长边"方向

    # === 车/卡车朝向启发式纠正（仅用当前帧点云几何，不用历史） ===
    # 单视角雷达只能看到面向自己的那一面：
    #   - 车端面 (前/后) 朝雷达：点云在垂直于视线方向上延展（带宽 ≈ 车宽）
    #     → 几何长边 ⊥ 视线方向 → 几何长边其实是"车宽"方向，车长方向应等于视线方向
    #   - 车侧面朝雷达：点云沿平行视线方向延展（带长 ≈ 车长）
    #     → 几何长边 ∥ 视线方向 → 几何长边就是车长方向
    # 通过几何长边与视线方向的夹角判断属于哪种情况。
    if class_index in (2, 7):
        sight_yaw = np.arctan2(cy, cx)                              # 从雷达指向车中心
        diff = abs(((yaw - sight_yaw + np.pi) % (2 * np.pi)) - np.pi)
        diff = min(diff, np.pi - diff)                              # 折叠到 [0, π/2]
        if diff > np.radians(60):
            # 几何长边几乎 ⊥ 视线 → 点云是车端面，需要把"长"和"宽"互换、yaw 改为视线方向
            yaw = sight_yaw
            w, l = l, w

    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
    PADDING = 0.35
    w += PADDING; l += PADDING
    if class_index == 2:
        w = max(w, 1.7); l = max(l, 4.0); height = max(height, 1.4)
    elif class_index == 7:
        w = max(w, 2.2); l = max(l, 5.5); height = max(height, 2.0)
    elif class_index == 0:
        w = max(w, 0.6); l = max(l, 0.6)
    if l > 20.0 or w > 6.0: return None
    return {
        "center": [float(cx), float(cy), float(z_min + height / 2.0)],
        "size": [float(l), float(w), float(height)],
        "yaw": float(yaw)
    }


def draw_lidar_projection_on_image(image, points, T_lidar_to_cam, fx, fy, cx, cy):
    vis = image.copy()
    H, W = image.shape[:2]
    homo = np.ones((points.shape[0], 4), dtype=np.float32)
    homo[:, :3] = points
    cam_pts = (T_lidar_to_cam @ homo.T).T
    valid = cam_pts[:, 2] > 0.5
    cam_pts = cam_pts[valid]
    if len(cam_pts) == 0: return vis
    depths = cam_pts[:, 2]
    u = (fx * cam_pts[:, 0] / depths + cx).astype(np.int32)
    v = (fy * cam_pts[:, 1] / depths + cy).astype(np.int32)
    mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, depths = u[mask], v[mask], depths[mask]
    norm_depth = np.clip(depths / 60.0 * 255, 0, 255).astype(np.uint8)
    colors = cv2.applyColorMap(norm_depth, cv2.COLORMAP_JET).reshape(-1, 3)
    vis[v, u] = colors
    vis[np.clip(v+1, 0, H-1), u] = colors
    vis[v, np.clip(u+1, 0, W-1)] = colors
    vis[np.clip(v+1, 0, H-1), np.clip(u+1, 0, W-1)] = colors
    cv2.putText(vis, "Calibration Check View", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return vis


WALL_OUTLIER_MAD_K = 4.0       # 残差 > k * MAD 视为离群 (MAD = median absolute deviation)
WALL_OUTLIER_MIN_PIX = 12      # 像素绝对下限：残差再小不剔（避免点云本身紧凑时误伤），约 0.8 米
WALL_OUTLIER_MIN_KEEP = 4      # 剔后剩余点不能少于此数
WALL_OUTLIER_POINT_COLOR = (60, 60, 200)  # 离群点高亮颜色 (BGR 红)


def draw_wall_curve_on_bev(img, height_points, bev_cx, bev_cy, scale_x, scale_y):
    """
    用挡墙顶点拟一条多项式曲线画在 BEV 上。
    离群点剔除策略：一次诊断 + 一次重拟，共 2 次 polyfit。
      1) 用全部点拟合，算 |残差|
      2) 取 |残差| 的 MAD（中位绝对偏差，比 std 更稳健）作为尺度
      3) 阈值 = max(MAD_K * MAD, MIN_PIX)，超过即剔
      4) 用保留点再拟一次得到最终曲线
    保留点画橙色，被剔除的点画红色，方便调试观察。
    """
    if not height_points or len(height_points) < 2:
        return img
    pts = np.array([[p.x, p.y] for p in height_points], dtype=np.float64)
    pts = pts[np.argsort(pts[:, 0])]
    u_pts = (bev_cx + pts[:, 0] * scale_x).astype(np.int32)
    v_pts = (bev_cy - pts[:, 1] * scale_y).astype(np.int32)
    valid = (u_pts >= 0) & (u_pts < BEV_WIDTH) & (v_pts >= 0) & (v_pts < BEV_HEIGHT)
    u_valid = u_pts[valid].astype(np.float64)
    v_valid = v_pts[valid].astype(np.float64)
    if len(u_valid) < 2:
        return img

    keep_mask = np.ones(len(u_valid), dtype=bool)
    coef = None
    x_axis_is_u = True
    try:
        x_span = float(u_valid.max() - u_valid.min())
        y_span = float(v_valid.max() - v_valid.min())
        x_axis_is_u = x_span >= y_span
        degree = min(3, len(u_valid) - 1)

        # === 第 1 次 polyfit：诊断残差 ===
        if x_axis_is_u:
            xs_all, ys_all = u_valid, v_valid
        else:
            xs_all, ys_all = v_valid, u_valid
        coef_diag = np.polyfit(xs_all, ys_all, degree)
        resid = np.abs(ys_all - np.poly1d(coef_diag)(xs_all))

        # MAD（中位绝对偏差），不易被少数离群点污染
        mad = float(np.median(resid))
        threshold = max(WALL_OUTLIER_MAD_K * mad, float(WALL_OUTLIER_MIN_PIX))
        candidate_keep = resid < threshold

        # 剩余点够多再启用剔除，否则保持全保留
        if candidate_keep.sum() >= max(WALL_OUTLIER_MIN_KEEP, degree + 1):
            keep_mask = candidate_keep

        # === 第 2 次 polyfit：用 keep 点出最终曲线 ===
        xs_keep = xs_all[keep_mask]
        ys_keep = ys_all[keep_mask]
        cur_degree = min(degree, len(xs_keep) - 1)
        if cur_degree >= 1:
            coef = np.polyfit(xs_keep, ys_keep, cur_degree)

        # 画点：保留橙色，离群红色
        for pu, pv, keep in zip(u_valid, v_valid, keep_mask):
            color = WALL_POINT_COLOR if keep else WALL_OUTLIER_POINT_COLOR
            cv2.circle(img, (int(pu), int(pv)), WALL_POINT_RADIUS, color, -1)

        # 画拟合曲线
        if coef is not None and keep_mask.sum() >= 2:
            u_in = u_valid[keep_mask]
            v_in = v_valid[keep_mask]
            poly = np.poly1d(coef)
            if x_axis_is_u:
                u_line = np.linspace(u_in.min(), u_in.max(), 400)
                v_line = poly(u_line)
            else:
                v_line = np.linspace(v_in.min(), v_in.max(), 400)
                u_line = poly(v_line)
            u_line = u_line.astype(np.int32)
            v_line = v_line.astype(np.int32)
            cv_mask = (u_line >= 0) & (u_line < BEV_WIDTH) & (v_line >= 0) & (v_line < BEV_HEIGHT)
            u_line, v_line = u_line[cv_mask], v_line[cv_mask]
            if len(u_line) >= 2:
                curve_pts = np.column_stack([u_line, v_line]).reshape(-1, 1, 2).astype(np.int32)
                cv2.polylines(img, [curve_pts], isClosed=False,
                              color=WALL_CURVE_COLOR, thickness=WALL_CURVE_THICKNESS,
                              lineType=cv2.LINE_AA)
    except Exception:
        # 拟合失败：原样画点 + 顺序折线兜底
        for pu, pv in zip(u_valid, v_valid):
            cv2.circle(img, (int(pu), int(pv)), WALL_POINT_RADIUS, WALL_POINT_COLOR, -1)
        for i in range(len(u_valid) - 1):
            cv2.line(img, (int(u_valid[i]), int(v_valid[i])),
                     (int(u_valid[i+1]), int(v_valid[i+1])),
                     WALL_CURVE_COLOR, WALL_CURVE_THICKNESS)

    n_kept = int(keep_mask.sum())
    n_total = int(len(u_valid))
    cv2.putText(img, f"Wall Curve ({n_kept}/{n_total} kept)", (10, BEV_HEIGHT - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WALL_CURVE_COLOR, 2)
    return img


def draw_bev_map(points, boxes_3d_info, height_points=None):
    """
    纯垂直俯视 BEV (image3)。投影完全忽略 z，与 image4 共享同一套像素坐标系。
    - 点云：灰色俯视底图
    - 检测框：底面 4 边俯视矩形（绿色） + 车头方向短线（红色）
    - 挡墙：height_points 在 xy 平面的拟合曲线
    """
    img = np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8)
    scale_x, scale_y = BEV_SCALE_X, BEV_SCALE_Y
    cx, cy = BEV_ORIGIN_U, BEV_ORIGIN_V

    mask = ((points[:, 0] > BEV_X_RANGE[0]) & (points[:, 0] < BEV_X_RANGE[1]) &
            (points[:, 1] > BEV_Y_RANGE[0]) & (points[:, 1] < BEV_Y_RANGE[1]))
    valid_pts = points[mask]
    if len(valid_pts) > 0:
        u = (cx + valid_pts[:, 0] * scale_x).astype(np.int32)
        v = (cy - valid_pts[:, 1] * scale_y).astype(np.int32)
        valid_uv = (u >= 0) & (u < BEV_WIDTH) & (v >= 0) & (v < BEV_HEIGHT)
        img[v[valid_uv], u[valid_uv]] = (180, 180, 180)

    def project_to_2d(x, y):
        return int(cx + x * scale_x), int(cy - y * scale_y)

    color_palette = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 165, 255), (130, 0, 250)]
    for idx, box in enumerate(boxes_3d_info):
        if box is None: continue
        obj_color = color_palette[idx % len(color_palette)]
        cluster_pts = box.get("cluster_pts", None)
        if cluster_pts is not None:
            c_u = (cx + cluster_pts[:, 0] * scale_x).astype(np.int32)
            c_v = (cy - cluster_pts[:, 1] * scale_y).astype(np.int32)
            c_valid = (c_u >= 0) & (c_u < BEV_WIDTH) & (c_v >= 0) & (c_v < BEV_HEIGHT)
            for px, py in zip(c_u[c_valid], c_v[c_valid]):
                cv2.circle(img, (px, py), 2, obj_color, -1)
        cx_b, cy_b = box["center_lidar"]["x"], box["center_lidar"]["y"]
        l, w = box["size_3d"]["length"], box["size_3d"]["width"]
        yaw = box["yaw_lidar_rad"]
        c_a, s_a = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c_a, -s_a], [s_a, c_a]])
        # 俯视下只需要 4 个底角（上下面投影重合）
        corners_local = np.array([
            [ l/2,  w/2],   # 0: 右前
            [ l/2, -w/2],   # 1: 左前
            [-l/2, -w/2],   # 2: 左后
            [-l/2,  w/2],   # 3: 右后
        ])
        corners_world = (rot @ corners_local.T).T + np.array([cx_b, cy_b])
        pts_2d = [project_to_2d(p[0], p[1]) for p in corners_world]
        for i in range(4):
            cv2.line(img, pts_2d[i], pts_2d[(i + 1) % 4], (0, 255, 0), 2)
        # 车头方向：从中心指向前侧（0/1 中点），红色短线
        front_mid = ((pts_2d[0][0] + pts_2d[1][0]) // 2,
                     (pts_2d[0][1] + pts_2d[1][1]) // 2)
        center_uv = project_to_2d(cx_b, cy_b)
        cv2.line(img, center_uv, front_mid, (0, 0, 255), 2)
    if height_points and len(height_points) >= 2:
        img = draw_wall_curve_on_bev(img, height_points, cx, cy, scale_x, scale_y)
    cv2.putText(img, "Top-Down BEV (xy plane)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img


# =================== alarm_ttj: PCA 拟合 / 跟踪 / 告警 / 绘图 ===================
def fit_line_pca_2d(pts_xy: np.ndarray):
    """
    对二维点做 PCA 拟合直线。
    返回 (mu, direction) — mu 为质心 (2,)，direction 为单位主方向向量 (2,)。
    """
    pts = np.asarray(pts_xy, dtype=np.float64)
    mu = pts.mean(axis=0)
    centered = pts - mu
    cov = np.cov(centered.T)
    if np.ndim(cov) == 0:
        return mu, np.array([1.0, 0.0])
    eigvals, eigvecs = np.linalg.eigh(cov)
    d = eigvecs[:, -1]
    n = np.linalg.norm(d)
    if n < 1e-12:
        return mu, np.array([1.0, 0.0])
    return mu, d / n


def point_to_line_distance_2d(pt, mu, direction):
    v = np.asarray(pt, dtype=np.float64) - np.asarray(mu, dtype=np.float64)
    proj = float(np.dot(v, direction))
    perp = v - proj * np.asarray(direction, dtype=np.float64)
    return float(np.linalg.norm(perp))


def line_segment_endpoints(pts_xy, mu, direction):
    """段内点投影到主方向上，取投影最小/最大对应位置作为线段两端。"""
    v = np.asarray(pts_xy, dtype=np.float64) - mu
    t = v @ direction
    return mu + t.min() * direction, mu + t.max() * direction


def fit_wall_segments(height_points, num_segments=ALARM_TTJ_NUM_SEGMENTS):
    """
    把 wall_info 的 height_point (按 x 从左到右) 平均分成 num_segments 段，
    每段在 (x, y) 平面用 PCA 拟一条直线。
    返回 list[dict]: {"mu", "dir", "ep1", "ep2", "pts"}
    """
    if not height_points or len(height_points) < num_segments * 2:
        return []
    pts_xy = np.array([[p.x, p.y] for p in height_points], dtype=np.float64)
    pts_xy = pts_xy[np.argsort(pts_xy[:, 0])]
    indices = np.array_split(np.arange(len(pts_xy)), num_segments)
    segments = []
    for idx in indices:
        if len(idx) < 2:
            continue
        seg_pts = pts_xy[idx]
        mu, d = fit_line_pca_2d(seg_pts)
        ep1, ep2 = line_segment_endpoints(seg_pts, mu, d)
        segments.append({"mu": mu, "dir": d, "ep1": ep1, "ep2": ep2, "pts": seg_pts})
    return segments


class CarTrack:
    __slots__ = ("track_id", "history", "missed")
    def __init__(self, track_id, init_center, max_history=ALARM_TTJ_HISTORY_LEN):
        self.track_id = track_id
        # deque(maxlen=...) 自动丢弃最旧元素，避免手动切片重建 list
        self.history: deque = deque(
            [np.asarray(init_center, dtype=np.float64)], maxlen=max_history)
        self.missed = 0


class CarTracker:
    """简易最近邻跟踪：每帧把当前 car 中心点贪心匹配到上一帧 track，未匹配则起新轨迹。"""
    def __init__(self,
                 max_history=ALARM_TTJ_HISTORY_LEN,
                 match_dist=ALARM_TTJ_TRACK_MATCH_DIST,
                 max_missed=ALARM_TTJ_TRACK_MAX_MISSED):
        self.tracks: List[CarTrack] = []
        self.next_id = 0
        self.max_history = max_history
        self.match_dist = match_dist
        self.max_missed = max_missed

    def reset(self):
        self.tracks = []
        self.next_id = 0

    def update(self, current_centers):
        """
        current_centers: list of (x, y)
        返回 list[CarTrack]，与 current_centers 一一对应。
        """
        assigned: List[Optional[CarTrack]] = [None] * len(current_centers)
        used_track_idx = set()

        # 计算所有 (cur, track) 距离对，按距离升序贪心匹配
        pairs = []
        for ci, c in enumerate(current_centers):
            cn = np.asarray(c, dtype=np.float64)
            for ti, tr in enumerate(self.tracks):
                d = float(np.linalg.norm(cn - tr.history[-1]))
                if d <= self.match_dist:
                    pairs.append((d, ci, ti))
        pairs.sort(key=lambda x: x[0])

        used_ci = set()
        for d, ci, ti in pairs:
            if ci in used_ci or ti in used_track_idx:
                continue
            tr = self.tracks[ti]
            tr.history.append(np.asarray(current_centers[ci], dtype=np.float64))
            tr.missed = 0
            assigned[ci] = tr
            used_ci.add(ci)
            used_track_idx.add(ti)

        # 未匹配的当前 car 起新轨迹
        for ci, c in enumerate(current_centers):
            if assigned[ci] is None:
                tr = CarTrack(self.next_id, c, max_history=self.max_history)
                self.next_id += 1
                self.tracks.append(tr)
                assigned[ci] = tr

        # 未参与匹配的旧 track 计 missed，超阈值丢弃
        for ti, tr in enumerate(self.tracks):
            if ti not in used_track_idx and tr not in assigned:
                tr.missed += 1
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return assigned


def check_alarm_ttj(track: Optional[CarTrack], wall_segments):
    """
    历史帧 < 30 或 没有可用墙段 → 返回 None（视为不告警）。
    否则返回 dict: {status, angle_deg, best_seg, traj_mu, traj_dir, min_wall_dist}
    """
    if track is None or len(track.history) < ALARM_TTJ_HISTORY_LEN:
        return None
    if not wall_segments:
        return None

    c_now = track.history[-1]
    best_seg, best_dist = None, float('inf')
    for seg in wall_segments:
        d = point_to_line_distance_2d(c_now, seg["mu"], seg["dir"])
        if d < best_dist:
            best_dist = d
            best_seg = seg
    if best_seg is None:
        return None

    traj_pts = np.stack(track.history, axis=0)
    mu_a, d_a = fit_line_pca_2d(traj_pts)
    d_b = best_seg["dir"]
    cos_theta = float(abs(np.dot(d_a, d_b)))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    angle_deg = float(np.degrees(np.arccos(cos_theta)))
    status = 1 if angle_deg < ALARM_TTJ_ANGLE_THRESHOLD_DEG else 0
    return {
        "status": status,
        "angle_deg": angle_deg,
        "best_seg": best_seg,
        "traj_mu": mu_a,
        "traj_dir": d_a,
        "min_wall_dist": float(best_dist),
    }


# 跨相机去重：同类别 + 3D 中心点距离 < 阈值 视为同一物体，保留 score 最高
DEDUP_DIST_BY_CLASS = {
    0: 0.8,   # person
    2: 2.5,   # car
    7: 2.5,   # truck
}
DEDUP_DEFAULT_DIST = 1.5


def deduplicate_enhanced_results(results):
    """对所有相机融合后的 enhanced 结果做跨相机去重，保留 YOLO score 最高的那个。"""
    if not results:
        return []
    sorted_results = sorted(results, key=lambda r: -float(r.get('score', 0.0)))
    kept = []
    for cur in sorted_results:
        cur_cls = cur.get('class_index')
        cur_c = np.array([cur['center_lidar']['x'],
                          cur['center_lidar']['y'],
                          cur['center_lidar']['z']], dtype=np.float64)
        thresh = DEDUP_DIST_BY_CLASS.get(cur_cls, DEDUP_DEFAULT_DIST)
        is_dup = False
        for k in kept:
            if k.get('class_index') != cur_cls:
                continue
            k_c = np.array([k['center_lidar']['x'],
                            k['center_lidar']['y'],
                            k['center_lidar']['z']], dtype=np.float64)
            if float(np.linalg.norm(cur_c - k_c)) < thresh:
                is_dup = True
                break
        if not is_dup:
            kept.append(cur)
    return kept


def draw_alarm_bev(pts, car_obj, track, wall_segments, alarm_info):
    """
    画告警 BEV (image4) — 与 draw_bev_map 用同一套投影参数（忽略 z）。
    - 5 条墙线：最近的 best_seg 红色，其余绿色
    - car 检测框：红色（鸟瞰矩形）
    - 轨迹拟合直线 a：从最早点投影到最新点投影，带箭头
    """
    img = np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8)
    scale_x, scale_y = BEV_SCALE_X, BEV_SCALE_Y
    bev_cx, bev_cy = BEV_ORIGIN_U, BEV_ORIGIN_V

    def proj(x, y):
        return int(bev_cx + x * scale_x), int(bev_cy - y * scale_y)

    # 底图：稀点云（灰）
    if pts is not None and len(pts) > 0:
        mask = ((pts[:, 0] > BEV_X_RANGE[0]) & (pts[:, 0] < BEV_X_RANGE[1]) &
                (pts[:, 1] > BEV_Y_RANGE[0]) & (pts[:, 1] < BEV_Y_RANGE[1]))
        valid = pts[mask]
        if len(valid) > 0:
            u = (bev_cx + valid[:, 0] * scale_x).astype(np.int32)
            v = (bev_cy - valid[:, 1] * scale_y).astype(np.int32)
            ok = (u >= 0) & (u < BEV_WIDTH) & (v >= 0) & (v < BEV_HEIGHT)
            img[v[ok], u[ok]] = (90, 90, 90)

    # 5 段拟合直线
    best_seg = alarm_info["best_seg"] if alarm_info else None
    for seg in wall_segments:
        is_best = (best_seg is not None) and (seg is best_seg)
        color = (0, 0, 255) if is_best else (0, 255, 0)
        p1 = proj(seg["ep1"][0], seg["ep1"][1])
        p2 = proj(seg["ep2"][0], seg["ep2"][1])
        cv2.line(img, p1, p2, color, 3, cv2.LINE_AA)

    # car 检测框（鸟瞰矩形，红色）
    cx_b = float(car_obj['center_lidar']['x'])
    cy_b = float(car_obj['center_lidar']['y'])
    l = float(car_obj['size_3d']['length'])
    w = float(car_obj['size_3d']['width'])
    yaw = float(car_obj['yaw_lidar_rad'])
    c_a, s_a = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c_a, -s_a], [s_a, c_a]])
    corners_local = np.array([[ l/2,  w/2],
                              [ l/2, -w/2],
                              [-l/2, -w/2],
                              [-l/2,  w/2]])
    corners_world = (rot @ corners_local.T).T + np.array([cx_b, cy_b])
    corners_uv = [proj(p[0], p[1]) for p in corners_world]
    for i in range(4):
        cv2.line(img, corners_uv[i], corners_uv[(i + 1) % 4], (0, 0, 255), 2, cv2.LINE_AA)

    # 轨迹箭头：从最早点投影到最新点投影
    if track is not None and len(track.history) >= 2:
        traj = np.stack(track.history, axis=0)
        mu_a, d_a = fit_line_pca_2d(traj)
        proj_t = (traj - mu_a) @ d_a
        start = mu_a + float(proj_t[0]) * d_a
        end = mu_a + float(proj_t[-1]) * d_a
        cv2.arrowedLine(img,
                        proj(start[0], start[1]),
                        proj(end[0], end[1]),
                        (255, 200, 0), 2, line_type=cv2.LINE_AA, tipLength=0.15)
        # 历史点散点
        for p in traj:
            u, v = proj(p[0], p[1])
            if 0 <= u < BEV_WIDTH and 0 <= v < BEV_HEIGHT:
                cv2.circle(img, (u, v), 2, (255, 200, 0), -1)

    # 文字
    hist_len = len(track.history) if track is not None else 0
    if alarm_info:
        text = (f"Angle a-b: {alarm_info['angle_deg']:.1f}deg | "
                f"WallDist: {alarm_info['min_wall_dist']:.2f}m | "
                f"{'ALARM' if alarm_info['status'] == 1 else 'OK'}")
        color = (0, 0, 255) if alarm_info["status"] == 1 else (0, 255, 0)
    else:
        text = f"Track len: {hist_len}/{ALARM_TTJ_HISTORY_LEN} (need {ALARM_TTJ_HISTORY_LEN})"
        color = (200, 200, 200)
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(img, "alarm_ttj BEV", (10, BEV_HEIGHT - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


# =================== 融合核心类 ===================
class LidarCameraFusion:
    def __init__(self, model_path, setup_yaml=None, camchain_yaml=None, model_w=640, model_h=640):
        self.model_path = model_path
        self.model_w, self.model_h = int(model_w), int(model_h)
        self.conf_thres, self.nms_thres = 0.25, 0.45

        self.T_lidar_to_cam = {}
        self.intrinsics = {}
        self.depth_range_min = -float('inf')
        self.depth_range_max = float('inf')
        self.z_min_filter = DEFAULT_Z_MIN
        self.z_max_filter = DEFAULT_Z_MAX
        self.xy_max_dist = DEFAULT_XY_MAX_DIST

        if setup_yaml: self._load_setup_yaml(setup_yaml)
        if camchain_yaml: self._load_camchain_yaml(camchain_yaml)
        print("已加载的外参矩阵:\n", self.T_lidar_to_cam)
        print("已加载的相机内参:\n", self.intrinsics)
        self.engine = SafeAclLiteEngine(self.model_path, self.model_w, self.model_h, device_id=0)

    def _load_setup_yaml(self, path):
        try:
            with open(path, 'r') as f: y = yaml.safe_load(f)
            if 'transformation_matrix1' in y:
                self.T_lidar_to_cam['cam1'] = np.array(y['transformation_matrix1'], dtype=np.float32)
            if 'transformation_matrix2' in y:
                self.T_lidar_to_cam['cam2'] = np.array(y['transformation_matrix2'], dtype=np.float32)
            if 'depth_range' in y:
                self.depth_range_min = y['depth_range'].get('min', self.depth_range_min)
                self.depth_range_max = y['depth_range'].get('max', self.depth_range_max)
            if 'z_filter' in y:
                self.z_min_filter = y['z_filter'].get('min', self.z_min_filter)
                self.z_max_filter = y['z_filter'].get('max', self.z_max_filter)
            if 'xy_max_dist' in y: self.xy_max_dist = float(y['xy_max_dist'])
        except Exception as e:
            print(f"Error loading setup YAML: {e}")

    def _load_camchain_yaml(self, path):
        try:
            with open(path, 'r') as f: cam = yaml.safe_load(f)
            if 'cam1' in cam and 'intrinsics' in cam['cam1']:
                self.intrinsics['cam1'] = list(map(float, cam['cam1']['intrinsics'][:4]))
            if 'cam2' in cam and 'intrinsics' in cam['cam2']:
                self.intrinsics['cam2'] = list(map(float, cam['cam2']['intrinsics'][:4]))
        except Exception as e:
            print(f"Error loading camchain YAML: {e}")

    def _infer_yolo_from_frame(self, frame):
        """
        ✅ 新方式：BGR → NV12 → 直接送NPU，绕过JPEG文件IO和DVPP
        """
        try:
            return self.engine.infer_from_bgr(frame)
        except Exception as e:
            print(f"❌ 推理失败: {e}")
            return None

    def process_image_array(self, frame):
        if frame is None: raise ValueError("Empty Frame")
        image_height, image_width = frame.shape[:2]

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

        xs = xywh[0] * image_width / self.model_w
        ys = xywh[1] * image_height / self.model_h
        ws = xywh[2] * image_width / self.model_w
        hs = xywh[3] * image_height / self.model_h

        dets = []
        for i in range(len(xs)):
            c = int(cls_idx[i])
            if c in TARGET_CLASSES:
                dets.append(BoundBox(float(xs[i]), float(ys[i]), float(ws[i]), float(hs[i]),
                                     float(scores[i]), c, int(indices[i])))
        dets = nms_per_class(dets, self.nms_thres)
        vis = visualize_detections(frame, dets)
        return dets, vis

    def fuse_with_pointcloud(self, points, dets, image, cam_id='cam1'):
        if cam_id not in self.intrinsics or cam_id not in self.T_lidar_to_cam:
            return None, [None]*len(dets)

        fx, fy, cx, cy = self.intrinsics[cam_id]
        T_ext = self.T_lidar_to_cam[cam_id]
        pts = np.asarray(points, dtype=np.float32)
        if pts.size == 0: return image.copy(), [None]*len(dets)

        t_proj = time.time()
        
        # 【核心优化】：相机外参投影全函数只做一次！
        homo = np.ones((len(pts), 4), dtype=np.float32)
        homo[:, :3] = pts
        cam_pts = (T_ext @ homo.T).T
        
        # 仅保留在相机前方的点
        valid = cam_pts[:, 2] > 1e-3 
        cam_pts = cam_pts[valid]
        pts = pts[valid]

        if pts.size == 0: return image.copy(), [None]*len(dets)

        # 统一计算图像上的像素坐标 u, v
        u = fx * cam_pts[:, 0] / cam_pts[:, 2] + cx
        v = fy * cam_pts[:, 1] / cam_pts[:, 2] + cy
        t_filter = time.time()

        # 【核心优化】：复用已经算好的 u, v 和 cam_pts，直接在原图上绘制激光点投影
        vis = image.copy()
        H, W = image.shape[:2]
        # 筛选出落在图像视野内、且深度 > 0.5 的点用于绘制
        draw_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (cam_pts[:, 2] > 0.5)
        
        u_draw = u[draw_mask].astype(np.int32)
        v_draw = v[draw_mask].astype(np.int32)
        depths_draw = cam_pts[:, 2][draw_mask]
        
        if len(depths_draw) > 0:
            norm_depth = np.clip(depths_draw / 60.0 * 255, 0, 255).astype(np.uint8)
            colors = cv2.applyColorMap(norm_depth, cv2.COLORMAP_JET).reshape(-1, 3)
            # 批量绘制（加粗像素点）
            vis[v_draw, u_draw] = colors
            vis[np.clip(v_draw+1, 0, H-1), u_draw] = colors
            vis[v_draw, np.clip(u_draw+1, 0, W-1)] = colors
            vis[np.clip(v_draw+1, 0, H-1), np.clip(u_draw+1, 0, W-1)] = colors
            cv2.putText(vis, "Calibration Check View", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            
        t_uv = time.time()

        enhanced_results = []
        t_cluster_total = 0.0

        for det in dets:
            t_box_start = time.time()
            result = None
            core_w, core_h = det.width * 0.40, det.height * 0.40
            x1, y1 = det.x - core_w, det.y - core_h
            x2, y2 = det.x + core_w, det.y + core_h
            
            # ROI 截取
            roi_pts = pts[(u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)]

            if len(roi_pts) >= MIN_POINTS_FOR_BOX:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(roi_pts)
                pcd = pcd.voxel_down_sample(0.2)
                ds_pts = np.asarray(pcd.points, dtype=np.float32)

                if len(ds_pts) >= DBSCAN_MIN_SAMPLES:
                    labels = np.array(pcd.cluster_dbscan(eps=0.4, min_points=8, print_progress=False))
                    if labels.size > 0 and labels.max() >= 0:
                        unique_labels = np.unique(labels[labels >= 0])
                        best_cluster, min_dist = None, float('inf')
                        for lbl in unique_labels:
                            cluster_pts = ds_pts[labels == lbl]
                            if len(cluster_pts) < DBSCAN_MIN_SAMPLES: continue
                            dist = np.mean(np.linalg.norm(cluster_pts[:, :2], axis=1))
                            if dist < min_dist:
                                min_dist = dist; best_cluster = cluster_pts
                        if best_cluster is not None:
                            box = compute_3d_bbox_from_points(best_cluster, det.class_index)
                            if box:
                                result = {
                                    "class_index": det.class_index,
                                    "class_name": TARGET_CLASSES.get(det.class_index, ("unk", None))[0],
                                    "score": det.score,
                                    "center_lidar": {"x": box["center"][0], "y": box["center"][1], "z": box["center"][2]},
                                    "size_3d": {"length": box["size"][0], "width": box["size"][1], "height": box["size"][2]},
                                    "yaw_lidar_deg": np.degrees(box["yaw"]),
                                    "yaw_lidar_rad": box["yaw"],
                                    "point_count": int(len(best_cluster)),
                                    "cluster_pts": best_cluster
                                }
                # 显式释放 Open3D C++ 对象，避免 GC 不及时导致内存累积
                try:
                    del pcd
                except Exception:
                    pass
            enhanced_results.append(result)
            t_cluster_total += (time.time() - t_box_start)

        print(f"    [{cam_id}细节] 矩阵投影={t_filter-t_proj:.3f}s, 图像绘制={t_uv-t_filter:.3f}s, 聚类循环({len(dets)}次)={t_cluster_total:.3f}s")
        return vis, enhanced_results


# =================== ROS2 节点 ===================
class FusionServerNode(Node):
    def __init__(self):
        super().__init__('fusion_server_node')
        self.declare_parameter('model_path', '')
        self.declare_parameter('setup_yaml', '')
        self.declare_parameter('camchain_yaml', '')

        mp = self.get_parameter('model_path').value
        sy = self.get_parameter('setup_yaml').value
        cy = self.get_parameter('camchain_yaml').value

        self.fusion = LidarCameraFusion(mp, sy, cy)
        self.bridge = CvBridge()
        self.running = False

        self.camera1_topic = ""
        self.camera2_topic = ""
        self.pc_topic = ""
        self.pub_topic = ""
        self.wall_info_topic = ""
        self.start_ts = 0.0
        self._last_ts = 0.0

        self.sub_pc = None
        self.sub_img1 = None
        self.sub_img2 = None
        self.sub_wall = None
        self.pub = None

        self.latest_cloud = None
        self.latest_image1 = None
        self.latest_image2 = None
        self.latest_height_points = []
        self.data_lock = threading.Lock()

        self.lidar_id = "default_lidar"
        self.camera1_id = "default_cam1"
        self.camera2_id = "default_cam2"

        # 跨帧 car 跟踪器（用于 alarm_ttj 轨迹拟合）
        self.car_tracker = CarTracker()

        # 调试: 每次 _process 落盘所有图像
        self.frame_idx = 0
        if DEBUG_SAVE_OUTPUT:
            try:
                os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)
                self.get_logger().info(f"📁 调试图像将保存到: {DEBUG_OUTPUT_DIR}")
            except Exception as e:
                self.get_logger().warn(f"创建 DEBUG_OUTPUT_DIR 失败: {e}")

        self.srv = self.create_service(FusionCommand, 'fusion_command', self.handle_cmd)
        self.timer = self.create_timer(1.0, self.timer_cb)
        self.get_logger().info("✅ Node Ready. Waiting for Topic-based command...")

    def _setup(self, t_pc, t_img1, t_img2, t_pub, t_wall):
        qos_sensor = qos_profile_sensor_data
        if self.sub_pc: self.destroy_subscription(self.sub_pc)
        self.sub_pc = self.create_subscription(PointCloud2, t_pc, self.cloud_cb, qos_sensor)
        if self.sub_img1: self.destroy_subscription(self.sub_img1)
        if t_img1:
            self.sub_img1 = self.create_subscription(Image, t_img1, self.image1_cb, qos_sensor)
        if self.sub_img2: self.destroy_subscription(self.sub_img2)
        if t_img2:
            self.sub_img2 = self.create_subscription(Image, t_img2, self.image2_cb, qos_sensor)

        if self.sub_wall:
            self.destroy_subscription(self.sub_wall)
            self.sub_wall = None
        if t_wall:
            try:
                from patchworkpp_msgs.msg import WallInfo
                self.sub_wall = self.create_subscription(
                    WallInfo, t_wall, self.wall_info_cb, qos_sensor)
                self.get_logger().info(f"✅ WallInfo订阅成功: {t_wall}")
            except ImportError:
                self.get_logger().warn(
                    f"⚠️ patchworkpp_msgs包未找到，跳过WallInfo订阅({t_wall})。")

        if self.pub: self.destroy_publisher(self.pub)
        self.pub = self.create_publisher(FusionResult, t_pub, 10)

        self.get_logger().info(
            f"✅ Setup OK: PointCloud={t_pc}, Cam1={t_img1}, "
            f"Cam2={t_img2}, WallInfo={t_wall or '未指定'}")

    def _teardown(self):
        if self.sub_pc: self.destroy_subscription(self.sub_pc); self.sub_pc = None
        if self.sub_img1: self.destroy_subscription(self.sub_img1); self.sub_img1 = None
        if self.sub_img2: self.destroy_subscription(self.sub_img2); self.sub_img2 = None
        if self.sub_wall: self.destroy_subscription(self.sub_wall); self.sub_wall = None
        if self.pub: self.destroy_publisher(self.pub); self.pub = None
        self.latest_cloud = None
        self.latest_image1 = None
        self.latest_image2 = None
        self.latest_height_points = []
        self.car_tracker.reset()
        # 显式释放 NPU 引擎，避免 ACL 内存不归还
        try:
            if self.fusion and self.fusion.engine:
                self.fusion.engine.release()
        except Exception as e:
            self.get_logger().warn(f"NPU 引擎释放异常: {e}")
        self.get_logger().info("❌ Teardown done.")

    def destroy_node(self):
        """重写销毁逻辑，确保节点退出时也能释放 NPU 资源。"""
        self._teardown()
        super().destroy_node()

    def cloud_cb(self, msg):
        with self.data_lock: self.latest_cloud = msg

    def image1_cb(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.data_lock: self.latest_image1 = cv_img
        except Exception: pass

    def image2_cb(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.data_lock: self.latest_image2 = cv_img
        except Exception: pass

    def wall_info_cb(self, msg):
        try:
            points = list(msg.height_point)
            with self.data_lock:
                self.latest_height_points = points
        except Exception as e:
            self.get_logger().warn(f"WallInfo回调异常: {e}")

    def handle_cmd(self, req, res):
        if req.enable_mode == 1:
            if self.running: self._teardown()
            self.camera1_topic = getattr(req, 'camera1_topic', '') or ""
            self.camera2_topic = getattr(req, 'camera2_topic', '') or ""
            self.lidar_id = getattr(req, 'lidar_id', '') or "default_lidar"
            self.camera1_id = getattr(req, 'camera1_id', '') or "default_cam1"
            self.camera2_id = getattr(req, 'camera2_id', '') or "default_cam2"
            self.pc_topic = getattr(req, 'pointcloud_topic', '')
            self.pub_topic = getattr(req, 'pointcloud_topic_r', '')
            self.wall_info_topic = getattr(req, 'wall_info_topic', '') or ""
            try:
                self._setup(self.pc_topic, self.camera1_topic, self.camera2_topic,
                            self.pub_topic, self.wall_info_topic)
                self.running = True
                self.start_ts = time.time()
                res.success = True
                res.message = f"Started Dual Camera Mode. Waiting {WARMUP_SECONDS}s..."
            except Exception as e:
                res.success = False; res.message = str(e)
        elif req.enable_mode == 0:
            self.running = False; self._teardown()
            res.success = True; res.message = "Stopped"
        return res

    def timer_cb(self):
        if not self.running: return
        now = time.time()
        if now - self.start_ts < WARMUP_SECONDS: return
        if now - self._last_ts < FUSION_PERIOD_SEC: return
        self._process()
        self._last_ts = time.time()

    def _process(self):
        t_start = time.time()
        with self.data_lock:
            c = self.latest_cloud
            frame1 = self.latest_image1
            frame2 = self.latest_image2
            height_points = list(self.latest_height_points)

        if c is None:
            self.get_logger().warn("等待数据中: 尚未收到雷达点云，跳过...")
            return
        if frame1 is None and frame2 is None:
            self.get_logger().warn("等待数据中: 没有任何相机图像...")
            return

        res = FusionResult()
        if hasattr(res, 'camera1_id'): res.camera1_id = self.camera1_id
        if hasattr(res, 'camera2_id'): res.camera2_id = self.camera2_id
        if hasattr(res, 'lidar_id'):   res.lidar_id = self.lidar_id

        # --- [调试] 本次 _process 的图像落盘子文件夹 ---
        frame_dir = make_debug_frame_dir(self.frame_idx)

        # --- [极速点云解析: 100% 无损] ---
        t_pc_start = time.time()
        raw_pts = robust_pointcloud_conversion(c)
        t_pc_end = time.time()

        # --- [空间预过滤: 100% 无损，剔除视野外无效点] ---
        pts = None
        t_downsample_start = time.time()
        if raw_pts is not None and len(raw_pts) > 0:
            # 仅做 Z 轴和最大距离的物理裁剪，一个有效目标点都不抽稀！
            mask = (raw_pts[:, 1] > self.fusion.depth_range_min) & (raw_pts[:, 1] < self.fusion.depth_range_max)
            mask &= (raw_pts[:, 2] >= self.fusion.z_min_filter) & (raw_pts[:, 2] <= self.fusion.z_max_filter)
            if self.fusion.xy_max_dist: 
                # 使用平方距离比较，省去昂贵的 np.hypot 开方运算
                dist_sq = raw_pts[:, 0]**2 + raw_pts[:, 1]**2
                mask &= (dist_sq <= self.fusion.xy_max_dist**2)
            pts = raw_pts[mask]
            pts = np.ascontiguousarray(pts) # 保证内存连续，加速矩阵乘法
        t_downsample_end = time.time()

        all_enhanced_results = []
        res.box_1 = []
        res.box_2 = []
        res.box_3d = []
        empty_img = np.zeros((1, 1, 3), dtype=np.uint8)

        # ================= 相机 1 处理 =================
        if frame1 is not None and self.camera1_topic:
            dets1, vis1 = self.fusion.process_image_array(frame1)
            res.image1_1 = self.bridge.cv2_to_imgmsg(frame1, encoding='bgr8')
            res.image1_2 = self.bridge.cv2_to_imgmsg(vis1, encoding='bgr8')
            save_debug_img(frame_dir, "image1_1_cam1_raw", frame1)
            save_debug_img(frame_dir, "image1_2_cam1_det", vis1)
            for det in dets1:
                b2 = Box2D()
                b2.class_id, b2.class_name = det.class_index, str(TARGET_CLASSES.get(det.class_index, ("u", None))[0])
                b2.score, b2.cx, b2.cy = det.score, det.x, det.y
                b2.width, b2.height = det.width, det.height
                res.box_1.append(b2)
            
            if pts is not None:
                img_proj1, enh1 = self.fusion.fuse_with_pointcloud(pts, dets1, frame1, cam_id='cam1')
                all_enhanced_results.extend([e for e in enh1 if e is not None])
        else:
            if hasattr(res, 'image1_1'): res.image1_1 = self.bridge.cv2_to_imgmsg(empty_img, encoding='bgr8')
            if hasattr(res, 'image1_2'): res.image1_2 = self.bridge.cv2_to_imgmsg(empty_img, encoding='bgr8')

        # ================= 相机 2 处理 =================
        if frame2 is not None and self.camera2_topic:
            dets2, vis2 = self.fusion.process_image_array(frame2)
            if hasattr(res, 'image2_1'): res.image2_1 = self.bridge.cv2_to_imgmsg(frame2, encoding='bgr8')
            if hasattr(res, 'image2_2'): res.image2_2 = self.bridge.cv2_to_imgmsg(vis2, encoding='bgr8')
            save_debug_img(frame_dir, "image2_1_cam2_raw", frame2)
            save_debug_img(frame_dir, "image2_2_cam2_det", vis2)
            for det in dets2:
                b2 = Box2D()
                b2.class_id, b2.class_name = det.class_index, str(TARGET_CLASSES.get(det.class_index, ("u", None))[0])
                b2.score, b2.cx, b2.cy = det.score, det.x, det.y
                b2.width, b2.height = det.width, det.height
                res.box_2.append(b2)
            
            if pts is not None:
                img_proj2, enh2 = self.fusion.fuse_with_pointcloud(pts, dets2, frame2, cam_id='cam2')
                all_enhanced_results.extend([e for e in enh2 if e is not None])
        else:
            if hasattr(res, 'image2_1'): res.image2_1 = self.bridge.cv2_to_imgmsg(empty_img, encoding='bgr8')
            if hasattr(res, 'image2_2'): res.image2_2 = self.bridge.cv2_to_imgmsg(empty_img, encoding='bgr8')

        # === 跨相机去重：同物体可能被 cam1 和 cam2 同时检测到 ===
        before_dedup = len(all_enhanced_results)
        all_enhanced_results = deduplicate_enhanced_results(all_enhanced_results)
        if before_dedup != len(all_enhanced_results):
            self.get_logger().info(
                f"  - 跨相机去重: {before_dedup} → {len(all_enhanced_results)}"
            )

        # === 跑 car tracker (供告警阶段使用，box 的 yaw 不依赖历史) ===
        car_results = [obj for obj in all_enhanced_results
                       if obj is not None and obj.get('class_index') == 2]
        car_centers = [(float(o['center_lidar']['x']), float(o['center_lidar']['y']))
                       for o in car_results]
        tracks_for_cars = self.car_tracker.update(car_centers)

        # 3D Boxes 装填
        for obj in all_enhanced_results:
            b3 = Box3D()
            b3.class_id, b3.class_name, b3.score = int(obj['class_index']), str(obj['class_name']), float(obj['score'])
            b3.center_lidar = Point(x=float(obj['center_lidar']['x']), y=float(obj['center_lidar']['y']), z=float(obj['center_lidar']['z']))
            b3.length, b3.width, b3.height = float(obj['size_3d']['length']), float(obj['size_3d']['width']), float(obj['size_3d']['height'])
            b3.yaw_lidar_rad, b3.point_count = float(obj['yaw_lidar_rad']), int(obj['point_count'])
            res.box_3d.append(b3)

        # BEV 绘制
        t_bev_start = time.time()
        if pts is not None:
            img_bev = draw_bev_map(pts, all_enhanced_results, height_points)
            res.image3 = self.bridge.cv2_to_imgmsg(img_bev, encoding='bgr8')
            save_debug_img(frame_dir, "image3_bev", img_bev)
        else:
            err_img = np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8)
            cv2.putText(err_img, "No Valid Lidar", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            res.image3 = self.bridge.cv2_to_imgmsg(err_img, encoding='bgr8')
            save_debug_img(frame_dir, "image3_bev", err_img)
        t_bev_end = time.time()

        # ================= alarm_ttj: 挡墙运动方向告警 =================
        # car_results / tracks_for_cars 已在装填 box_3d 之前提前算好，这里直接用。
        # 把挡墙 height_points 均分5段并各自 PCA 拟合
        wall_segments = fit_wall_segments(height_points, ALARM_TTJ_NUM_SEGMENTS)

        # 把挡墙顶点也透传到 FusionResult.height_points (合并发布，不再单独话题)
        if hasattr(res, 'height_points') and height_points:
            try:
                res.height_points = [
                    Point(x=float(p.x), y=float(p.y), z=float(p.z))
                    for p in height_points
                ]
            except Exception as e:
                self.get_logger().warn(f"height_points 透传失败: {e}")

        # 每辆 car 一个 AlarmTTJ：{status: bool, degree: float, image4: Image}
        alarm_list: List[AlarmTTJ] = []
        for i, (car_obj, tr) in enumerate(zip(car_results, tracks_for_cars)):
            alarm_info = check_alarm_ttj(tr, wall_segments)
            if alarm_info is not None:
                status = bool(alarm_info["status"])
                degree = float(alarm_info["angle_deg"])
            else:
                # 历史不足 30 帧 或 没有可用墙段 → 无法判定
                status = False
                degree = -1.0
            # 只在违规时画详细告警 BEV，否则用占位空黑图保持长度对齐
            if status:
                img4 = draw_alarm_bev(pts, car_obj, tr, wall_segments, alarm_info)
            else:
                img4 = np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8)
            a = AlarmTTJ()
            a.status = status
            a.degree = degree
            a.image4 = self.bridge.cv2_to_imgmsg(img4, encoding='bgr8')
            alarm_list.append(a)
            save_debug_img(
                frame_dir,
                f"alarm_car{i}_status{int(status)}_deg{degree:.1f}",
                img4
            )

        if hasattr(res, 'alarm_ttj'):
            res.alarm_ttj = alarm_list

        if self.pub:
            self.pub.publish(res)
            t_total_end = time.time()
            n_alarm = sum(1 for a in alarm_list if a.status)
            self.get_logger().info(
                f"✅ Pub结果:\n"
                f"  - 点云解析: {t_pc_end-t_pc_start:.3f}s\n"
                f"  - 空间裁剪(无损): {t_downsample_end-t_downsample_start:.3f}s\n"
                f"  - 发布开销: {t_total_end-t_bev_end:.3f}s\n"
                f"  👉 总计总耗时: {t_total_end-t_start:.3f}s | 2D: {len(res.box_1)}+{len(res.box_2)}, "
                f"3D: {len(all_enhanced_results)} | wall_pts={len(height_points)}, "
                f"wall_segs={len(wall_segments)} | cars={len(car_results)}, alarm={n_alarm}"
                + (f"\n  - 调试落盘: {frame_dir}" if frame_dir else "")
            )

        # 自增帧号（无论是否成功 publish）
        self.frame_idx += 1


def main(args=None):
    rclpy.init(args=args)
    node = FusionServerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()