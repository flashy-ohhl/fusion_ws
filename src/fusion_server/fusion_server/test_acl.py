import cv2
from acllite_resource import AclLiteResource
from acllite_model import AclLiteModel
from acllite_image import AclLiteImage
from acllite_imageproc import AclLiteImageProc

MODEL = "/data/zjj/sampleYOLOV11/model/yolov11s.om"
IMG   = "/data/zjj/sampleYOLOV11/shujuji/ros2_recorded_data/20250927_100604/camera_20250927_100604_20250927_100606_039.jpg"

print("✅ init resource")
res = AclLiteResource()
res.init()

print("✅ init dvpp")
proc = AclLiteImageProc(res)

print("✅ load model")
model = AclLiteModel(MODEL)

print("✅ load image")
acl_img = AclLiteImage(IMG)
dvpp_img = acl_img.copy_to_dvpp()

print("✅ jpeg decode")
yuv = proc.jpegd(dvpp_img)

print("✅ resize")
resized = proc.resize(yuv, 640, 640)

print("✅ execute model")
out = model.execute([resized])

print("✅ inference success, output len =", len(out))
