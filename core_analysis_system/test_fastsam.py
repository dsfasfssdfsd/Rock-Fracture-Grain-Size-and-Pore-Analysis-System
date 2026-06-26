"""
测试FastSAM模型
"""
import cv2
from ultralytics import FastSAM

model_path = r'C:\Users\ZhuanZ1\Desktop\code1\code1\FastSAM-x.pt'
print(f"加载模型: {model_path}")

# 加载模型
model = FastSAM(model_path)
print("模型加载成功!")

# 测试图像
test_img = r'C:\Users\ZhuanZ1\Desktop\code1\code1\picture\S0110.bmp'
print(f"\n测试图像: {test_img}")
img = cv2.imread(test_img)
if img is not None:
    print(f"图像大小: {img.shape}")

    # 推理
    results = model(test_img, imgsz=640, conf=0.5, iou=0.7)
    print(f"检测到 {len(results[0].masks)} 个对象")

    # 查看类别
    if hasattr(results[0], 'boxes') and results[0].boxes is not None:
        print(f"类别: {results[0].boxes.cls}")
else:
    print("图像不存在")

print("\nFastSAM 可以正常使用!")
