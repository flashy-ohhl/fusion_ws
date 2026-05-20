import os
import tempfile
from typing import List, Optional, Union

import numpy as np
import cv2

from acllite_resource import AclLiteResource
from acllite_model import AclLiteModel
from acllite_image import AclLiteImage
from acllite_imageproc import AclLiteImageProc


# =====================================================================
# BGR → NV12 转换工具函数
# =====================================================================
def bgr_to_nv12(bgr_image: np.ndarray, target_w: int = 640, target_h: int = 640) -> np.ndarray:
    """
    BGR图像 → NV12格式numpy数组，可直接喂给om模型。

    NV12布局：
      [Y plane: target_h × target_w]
      [UV plane: (target_h/2) × target_w，UVUV交织]
    总字节数 = target_w × target_h × 1.5

    返回: shape=(target_h*3//2, target_w) 的 uint8 数组（C连续内存）
    """
    h, w = target_h, target_w

    # Step1: resize到模型输入尺寸
    resized = cv2.resize(bgr_image, (w, h))

    # Step2: BGR → I420 (YYYY...UUUU...VVVV)
    # OpenCV 的 BGR2YUV_I420 输出形状是 (h*3//2, w)
    yuv_i420 = cv2.cvtColor(resized, cv2.COLOR_BGR2YUV_I420)

    # I420 布局解析：
    #   yuv_i420[0:h, :]            → Y plane (h, w)
    #   yuv_i420[h : h+h//4, :]     → U plane 数据，需要 reshape 成 (h/2, w/2)
    #   yuv_i420[h+h//4 : h*3//2, :] → V plane 数据，reshape 同上
    uv_h = h // 2
    uv_w = w // 2

    y_plane = yuv_i420[:h, :]
    u_plane = yuv_i420[h:h + h // 4, :].reshape(uv_h, uv_w)
    v_plane = yuv_i420[h + h // 4:h * 3 // 2, :].reshape(uv_h, uv_w)

    # Step3: U/V 交织成 NV12 的 UV plane
    uv_plane = np.empty((uv_h, w), dtype=np.uint8)
    uv_plane[:, 0::2] = u_plane   # 偶数列放 U
    uv_plane[:, 1::2] = v_plane   # 奇数列放 V

    # Step4: 拼接 Y + UV
    nv12 = np.empty((h * 3 // 2, w), dtype=np.uint8)
    nv12[:h, :] = y_plane
    nv12[h:, :] = uv_plane

    return np.ascontiguousarray(nv12)


# =====================================================================
# 推理引擎（保留原有接口，新增 infer_from_bgr）
# =====================================================================
class SafeAclLiteEngine:
    """
    DVPP 推理引擎（保留原JPEG路径），新增 BGR→NV12 直送路径。

    使用方式：
      - infer_from_bgr(bgr_array)  ← 推荐，绕过JPEG和DVPP，最快
      - infer_from_jpeg_path(path) ← 原始方式，保留作为回退
      - infer_from_jpeg_bytes(bytes)
    """

    def __init__(self, model_path: str, model_w: int = 640,
                 model_h: int = 640, device_id: int = 0):
        self.model_path = model_path
        self.model_w = int(model_w)
        self.model_h = int(model_h)
        self.device_id = int(device_id)

        self.resource: Optional[AclLiteResource] = None
        self.image_proc: Optional[AclLiteImageProc] = None
        self.model: Optional[AclLiteModel] = None

        self._init_acl()

    def _init_acl(self):
        print("✅ 初始化 AclLiteResource")
        self.resource = AclLiteResource(self.device_id)
        self.resource.init()
        # ❌ 禁用 DVPP：infer_from_bgr 完全走 CPU 预处理，不需要 AclLiteImageProc
        # 避免 Atlas 500 A2 上 DVPP 预分配数 GB NPU 内存
        self.image_proc = None
        print("✅ 加载模型:", self.model_path)
        self.model = AclLiteModel(self.model_path)
        print("✅✅✅ 推理引擎初始化完成（支持BGR直传NV12，DVPP已禁用）")

    # ---------------- 新接口：BGR 直接转 NV12 推理 ----------------
    def infer_from_bgr(self, bgr_image: np.ndarray) -> Optional[List[np.ndarray]]:
        """
        ✅ 推荐使用：BGR numpy → NV12 → NPU推理。
        完全绕过 JPEG 编码、文件IO、DVPP 解码、DVPP resize。
        """
        if bgr_image is None:
            return None
        try:
            nv12 = bgr_to_nv12(bgr_image, self.model_w, self.model_h)
            # 直接送进模型，AclLiteModel 支持 numpy 输入
            return self.model.execute([nv12])
        except Exception as e:
            print(f"❌ infer_from_bgr 失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    # ---------------- 原始接口：JPEG → DVPP（已禁用，如需使用请先开启 DVPP） ----------------
    def infer_from_jpeg_path(self, jpeg_path: str) -> Optional[List[np.ndarray]]:
        print("⚠️ infer_from_jpeg_path 已禁用（DVPP 未初始化），请使用 infer_from_bgr")
        return None

    def infer_from_jpeg_bytes(self, jpeg_bytes: Union[bytes, bytearray]) -> Optional[List[np.ndarray]]:
        print("⚠️ infer_from_jpeg_bytes 已禁用（DVPP 未初始化），请使用 infer_from_bgr")
        return None

    def release(self):
        try:
            if self.model is not None:
                self.model.__del__()
            # DVPP 已禁用，仅在有实例时才释放
            if self.image_proc is not None:
                try:
                    self.image_proc.__del__()
                except Exception:
                    pass
            if self.resource is not None:
                if hasattr(self.resource, "release"):
                    self.resource.release()
                else:
                    self.resource.__del__()
            print("✅ AclLite 资源已释放")
        except Exception as e:
            print("⚠️ AclLite 释放异常:", e)

    def __del__(self):
        self.release()