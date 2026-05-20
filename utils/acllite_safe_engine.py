# -*- coding: utf-8 -*-
"""
SafeAclLiteEngine (DVPP-only)

输入：
- JPEG 文件路径 infer_from_jpeg_path()
- JPEG bytes infer_from_jpeg_bytes()

流程：
JPEG -> AclLiteImage -> copy_to_dvpp -> jpegd -> resize -> model.execute([dvpp_buf])
"""

import os
import tempfile
from typing import List, Optional, Union

import numpy as np

from acllite_resource import AclLiteResource
from acllite_model import AclLiteModel
from acllite_image import AclLiteImage
from acllite_imageproc import AclLiteImageProc


class SafeAclLiteEngine:
    """
    ✅ DVPP-only 推理引擎
    - 不做 CPU BGR 预处理，不做 HWC/CHW，不做 /255
    - 输入必须是 JPEG（路径或 bytes）
    """

    def __init__(self, model_path: str, model_w: int = 640, model_h: int = 640, device_id: int = 0):
        self.model_path = model_path
        self.model_w = int(model_w)
        self.model_h = int(model_h)
        self.device_id = int(device_id)

        self.resource: Optional[AclLiteResource] = None
        self.image_proc: Optional[AclLiteImageProc] = None
        self.model: Optional[AclLiteModel] = None

        self._init_acl()

    def _init_acl(self):
        print("✅ 初始化 AclLiteResource（DVPP-only）")
        self.resource = AclLiteResource(self.device_id)
        self.resource.init()

        self.image_proc = AclLiteImageProc(self.resource)

        print("✅ 加载模型:", self.model_path)
        self.model = AclLiteModel(self.model_path)

        print("✅✅✅ AclLite DVPP-only 推理引擎初始化完成")

    def infer_from_jpeg_path(self, jpeg_path: str) -> Optional[List[np.ndarray]]:
        """
        输入 JPEG 文件路径，走 DVPP 解码 + resize
        """
        if not jpeg_path or (not os.path.exists(jpeg_path)):
            print(f"❌ infer_from_jpeg_path: 文件不存在: {jpeg_path}")
            return None

        try:
            acl_img = AclLiteImage(jpeg_path)
            dvpp_in = acl_img.copy_to_dvpp()

            yuv = self.image_proc.jpegd(dvpp_in)
            resized = self.image_proc.resize(yuv, self.model_w, self.model_h)

            outputs = self.model.execute([resized])
            return outputs
        except Exception as e:
            print(f"❌ DVPP 推理失败 (path): {e}")
            return None

    def infer_from_jpeg_bytes(self, jpeg_bytes: Union[bytes, bytearray]) -> Optional[List[np.ndarray]]:
        """
        输入 JPEG bytes。
        AclLiteImage 原生通常接 path，所以这里用临时文件落地一次，保证兼容最稳。
        """
        if jpeg_bytes is None or len(jpeg_bytes) == 0:
            print("❌ infer_from_jpeg_bytes: 输入为空")
            return None

        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            with open(tmp_path, "wb") as f:
                f.write(jpeg_bytes)
            return self.infer_from_jpeg_path(tmp_path)
        except Exception as e:
            print(f"❌ DVPP 推理失败 (bytes): {e}")
            return None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def release(self):
        try:
            if self.model is not None:
                self.model.__del__()
            if self.image_proc is not None:
                self.image_proc.__del__()
            if self.resource is not None:
                # 有的版本是 release，有的版本是 __del__
                if hasattr(self.resource, "release"):
                    self.resource.release()
                else:
                    self.resource.__del__()
            print("✅ AclLite DVPP 资源已释放")
        except Exception as e:
            print("⚠️ AclLite 释放异常:", e)

    def __del__(self):
        self.release()
