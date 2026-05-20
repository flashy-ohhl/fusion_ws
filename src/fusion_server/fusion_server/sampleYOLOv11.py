# python3 sampleYOLOv11.py /path/to/yolov11s.om /path/to/image.jpg

import os
import sys
import time
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

# AclLite (全小写模块名，与你当前路径一致)
from acllite_logger import log_info
from acllite_resource import AclLiteResource
from acllite_imageproc import AclLiteImage, AclLiteImageProc
from acllite_model import AclLiteModel
from .sampleYOLOv11 import SampleYOLOV11, BoundBox

# 兼容 label.py 的不同变量名
try:
    from label import label as LABELS
except Exception:
    try:
        from label import labels as LABELS
    except Exception:
        LABELS = [str(i) for i in range(80)]




class LidarCameraFusion:
    def __init__(self, model_path, setup_yaml=None, camchain_yaml=None):
        self.yolo = SampleYOLOV11(
            model_path,
            model_w=640,
            model_h=640,
            conf_thres=0.25,
            nms_thres=0.45,
            num_classes=80
        )
        self.yolo.init()

        self.current_image = None


@dataclass
class BoundBox:
    x: float
    y: float
    width: float
    height: float
    score: float
    class_index: int
    index: int

def nms_per_class(boxes: List[BoundBox], iou_thr: float = 0.45) -> List[BoundBox]:
    """与 C++ 等价：按类别分别 NMS"""
    if not boxes:
        return []
    groups = {}
    for b in boxes:
        groups.setdefault(b.class_index, []).append(b)

    kept: List[BoundBox] = []
    for cls, arr in groups.items():
        arr.sort(key=lambda b: b.score, reverse=True)
        selected = []
        while arr:
            m = arr.pop(0)
            selected.append(m)
            remain = []
            mx1 = m.x - m.width / 2
            my1 = m.y - m.height / 2
            mx2 = m.x + m.width / 2
            my2 = m.y + m.height / 2
            m_area = m.width * m.height
            for g in arr:
                gx1 = g.x - g.width / 2
                gy1 = g.y - g.height / 2
                gx2 = g.x + g.width / 2
                gy2 = g.y + g.height / 2
                ix1, iy1 = max(mx1, gx1), max(my1, gy1)
                ix2, iy2 = min(mx2, gx2), min(my2, gy2)
                iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
                inter = iw * ih
                union = m_area + g.width * g.height - inter
                iou = inter / union if union > 0 else 0.0
                if iou <= iou_thr:
                    remain.append(g)
            arr = remain
        kept.extend(selected)
    return kept


class SampleYOLOV11:
    def __init__(self, model_path: str, model_w: int = 640, model_h: int = 640,
                 conf_thres: float = 0.25, nms_thres: float = 0.45, num_classes: int = 80):
        self.model_path = model_path
        self.model_w = model_w
        self.model_h = model_h
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.num_classes = num_classes

        self.resource = None
        self.dvpp = None
        self.model = None

        # 缓存
        self._src_acl_img = None
        self._dvpp_resized = None

    # ================== 初始化/释放 ==================
    def init(self):
        log_info("init resource stage:")
        self.resource = AclLiteResource()
        self.resource.init()
        self.dvpp = AclLiteImageProc(self.resource)

        log_info("Init model resource start...")
        self.model = AclLiteModel(self.model_path)
        log_info("Init model resource success")

    def release(self):
        try:
            if self.dvpp is not None:
                self.dvpp.__del__()
        except Exception:
            pass
        try:
            if self.model is not None:
                self.model.__del__()
        except Exception:
            pass
        try:
            if self.resource is not None:
                self.resource.__del__()
        except Exception:
            pass

    # ================== 预处理 ==================
    def process_input(self, image_path: str):
        # 读入并拷贝到 DVPP
        self._src_acl_img = AclLiteImage(image_path)
        dvpp_in = self._src_acl_img.copy_to_dvpp()

        # JPEGD -> YUV(NV12)
        yuv = self.dvpp.jpegd(dvpp_in)

        # 直接拉伸到模型输入尺寸（与 C++ 的 Resize 一致，不做 letterbox）
        self._dvpp_resized = self.dvpp.resize(yuv, self.model_w, self.model_h)

    # ================== 推理 ==================
    def inference(self) -> List[np.ndarray]:
        # AclLiteModel.execute 接受 DVPP buffer 列表作为输入
        results = self.model.execute([self._dvpp_resized])
        # 一般返回的是 host 侧的 numpy 数组列表
        return results

    # ================== 后处理 ==================
    def postprocess(self, outputs: List[np.ndarray], src_w: int, src_h: int) -> List[BoundBox]:
        """
        解析 YOLOv11 输出：
        YOLOv11 输出格式通常是 [1, 84, 8400] 或类似格式
        布局：前4个为(cx,cy,w,h)，后续80个为分类分数
        """
        if not outputs:
            return []

        out = outputs[0]
        
        # 调试输出形状信息
        log_info(f"Output shape: {out.shape}, ndim: {out.ndim}")
        
        # 处理不同的输出形状
        if out.ndim == 3:  # (1,84,8400) 或 (1,4+num_classes,8400)
            out = out[0]  # 去除batch维度
        elif out.ndim == 2:  # (84,8400)
            pass
        elif out.ndim == 1:
            # 扁平化输出，需要reshape
            if out.size % (4 + self.num_classes) != 0:
                raise ValueError(f"Unexpected flat output size: {out.size}, expected multiple of {4 + self.num_classes}")
            num_boxes = out.size // (4 + self.num_classes)
            out = out.reshape(4 + self.num_classes, num_boxes)
        else:
            raise ValueError(f"Unexpected output ndim: {out.ndim}")

        # 验证通道维度
        if out.shape[0] != 4 + self.num_classes:
            log_info(f"Warning: Channel dim {out.shape[0]}, expected {4 + self.num_classes}")
            # 尝试自动调整num_classes
            actual_num_classes = out.shape[0] - 4
            if actual_num_classes > 0:
                self.num_classes = actual_num_classes
                log_info(f"Auto-adjusted num_classes to: {self.num_classes}")

        num_boxes = out.shape[1]
        xywh = out[0:4, :]  # (4, num_boxes)
        cls_scores = out[4:, :]  # (num_classes, num_boxes)

        # 取每个锚点的最大类别与分数
        cls_idx = np.argmax(cls_scores, axis=0)  # (num_boxes,)
        scores = cls_scores[cls_idx, np.arange(num_boxes)]  # (num_boxes,)

        # 过滤低置信度检测
        keep = scores > self.conf_thres
        if not np.any(keep):
            return []

        xywh = xywh[:, keep]
        cls_idx = cls_idx[keep]
        scores = scores[keep]
        indices = np.where(keep)[0]

        # 坐标从模型分辨率映射回原图（直接线性缩放）
        xs = xywh[0] * src_w / self.model_w
        ys = xywh[1] * src_h / self.model_h
        ws = xywh[2] * src_w / self.model_w
        hs = xywh[3] * src_h / self.model_h

        boxes: List[BoundBox] = []
        for i in range(xs.shape[0]):
            boxes.append(
                BoundBox(
                    x=float(xs[i]),
                    y=float(ys[i]),
                    width=float(ws[i]),
                    height=float(hs[i]),
                    score=float(scores[i]),
                    class_index=int(cls_idx[i]),
                    index=int(indices[i]),
                )
            )

        # NMS
        boxes = nms_per_class(boxes, self.nms_thres)
        return boxes

    # ================== 绘制保存 ==================
    def draw_and_save(self, image_path: str, boxes: List[BoundBox], save_path: str):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(image_path)

        for i, b in enumerate(boxes):
            x1 = int(b.x - b.width / 2)
            y1 = int(b.y - b.height / 2)
            x2 = int(b.x + b.width / 2)
            y2 = int(b.y + b.height / 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            name = LABELS[b.class_index] if 0 <= b.class_index < len(LABELS) else str(b.class_index)
            cv2.putText(img, f"{name} {b.score:.2f}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img)


def is_image_file(name: str) -> bool:
    ext = os.path.splitext(name)[1].lower()
    return ext in {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    """
    用法：
      1) 单图：
         python3 sampleYOLOv11.py /path/to/yolov11s.om /path/to/image.jpg
      2) 目录：
         python3 sampleYOLOv11.py /path/to/yolov11s.om /path/to/dir
    """
    if len(sys.argv) != 3:
        print("Usage: python3 sampleYOLOv11.py <model.om> <image_path_or_dir>")
        sys.exit(1)

    model_path = os.path.abspath(sys.argv[1])
    inp_path = os.path.abspath(sys.argv[2])

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型不存在: {model_path}")
    if not os.path.exists(inp_path):
        raise FileNotFoundError(f"输入路径不存在: {inp_path}")

    # 使用YOLOv11的默认参数
    yolo = SampleYOLOV11(model_path, model_w=640, model_h=640,
                        conf_thres=0.25, nms_thres=0.45, num_classes=80)
    yolo.init()

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../out"))
    os.makedirs(out_dir, exist_ok=True)

    paths: List[str] = []
    if os.path.isdir(inp_path):
        for name in os.listdir(inp_path):
            if name in {".", "..", ".keep"}:
                continue
            p = os.path.join(inp_path, name)
            if os.path.isfile(p) and is_image_file(p):
                paths.append(p)
    else:
        if is_image_file(inp_path):
            paths = [inp_path]
        else:
            raise ValueError(f"不是图片文件: {inp_path}")

    if not paths:
        raise RuntimeError("目录中没有图片（或都被过滤）")

    try:
        for i, p in enumerate(paths):
            t0 = time.time()
            yolo.process_input(p)
            results = yolo.inference()
            
            # 获取原图尺寸
            img = cv2.imread(p)
            if img is None:
                raise FileNotFoundError(p)
            h, w = img.shape[:2]
            
            boxes = yolo.postprocess(results, w, h)
            save_path = os.path.join(out_dir, f"out_{i}.jpg")
            yolo.draw_and_save(p, boxes, save_path)
            
            dt = time.time() - t0
            fps = 1.0 / dt if dt > 0 else float("inf")
            log_info(f"Inference elapsed time: {dt:.4f}s, fps: {fps:.2f} | {os.path.basename(p)} -> {save_path}")
            log_info(f"Detected {len(boxes)} objects")
            
    finally:
        yolo.release()
        log_info("YOLOv11 inference completed successfully")


if __name__ == "__main__":
    main()