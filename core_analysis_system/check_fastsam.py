"""
检查FastSAM模型是否可用
"""
import torch
import os

model_path = r'C:\Users\ZhuanZ1\Desktop\code1\code1\FastSAM-x.pt'
print(f"模型文件存在: {os.path.exists(model_path)}")
print(f"模型文件大小: {os.path.getsize(model_path) / 1024 / 1024:.1f} MB")

try:
    checkpoint = torch.load(model_path, map_location='cpu')
    print(f"模型 keys: {list(checkpoint.keys())[:10] if isinstance(checkpoint, dict) else 'Not a dict'}")
    print("模型可以加载!")
except Exception as e:
    print(f"加载失败: {e}")
