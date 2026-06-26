"""
U-Net裂缝检测推理模块
使用训练好的LightUNet模型进行精确裂缝分割
"""
import os
import cv2
import numpy as np

# 修复PyTorch DLL加载问题
try:
    import torch
    torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
    if os.path.exists(torch_lib):
        os.add_dll_directory(torch_lib)
except Exception:
    pass


class UNetInference:
    """U-Net裂缝检测推理"""

    def __init__(self, model_path=None):
        self.model = None
        self.device = 'cpu'
        self._model_path = model_path or r'C:\Users\ZhuanZ1\Desktop\code1\code1\core_analysis_system\unet_fracture.pth'
        self._loaded = False
        self.input_size = 512  # 推理输入尺寸，从256增大到512提升精度

    def is_available(self):
        """模型文件是否存在"""
        return os.path.exists(self._model_path)

    def load_model(self):
        """加载模型，成功返回模型，失败返回None"""
        if self._loaded and self.model is not None:
            return self.model

        if not os.path.exists(self._model_path):
            self._loaded = True
            return None

        try:
            import torch
            import torch.nn as nn

            class ConvBlock(nn.Module):
                def __init__(self, in_ch, out_ch):
                    super().__init__()
                    self.conv = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 3, padding=1),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_ch, out_ch, 3, padding=1),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(inplace=True)
                    )

                def forward(self, x):
                    return self.conv(x)

            class LightUNet(nn.Module):
                def __init__(self, in_channels=3, out_channels=1):
                    super().__init__()
                    self.enc1 = ConvBlock(in_channels, 32)
                    self.enc2 = ConvBlock(32, 64)
                    self.enc3 = ConvBlock(64, 128)
                    self.enc4 = ConvBlock(128, 256)
                    self.pool = nn.MaxPool2d(2)
                    self.bottleneck = ConvBlock(256, 512)
                    self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
                    self.dec4 = ConvBlock(512, 256)
                    self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
                    self.dec3 = ConvBlock(256, 128)
                    self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
                    self.dec2 = ConvBlock(128, 64)
                    self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
                    self.dec1 = ConvBlock(64, 32)
                    self.final = nn.Conv2d(32, out_channels, 1)

                def forward(self, x):
                    e1 = self.enc1(x)
                    e2 = self.enc2(self.pool(e1))
                    e3 = self.enc3(self.pool(e2))
                    e4 = self.enc4(self.pool(e3))
                    b = self.bottleneck(self.pool(e4))
                    d4 = self.up4(b)
                    d4 = torch.cat([d4, e4], dim=1)
                    d4 = self.dec4(d4)
                    d3 = self.up3(d4)
                    d3 = torch.cat([d3, e3], dim=1)
                    d3 = self.dec3(d3)
                    d2 = self.up2(d3)
                    d2 = torch.cat([d2, e2], dim=1)
                    d2 = self.dec2(d2)
                    d1 = self.up1(d2)
                    d1 = torch.cat([d1, e1], dim=1)
                    d1 = self.dec1(d1)
                    return torch.sigmoid(self.final(d1))

            self.model = LightUNet()
            checkpoint = torch.load(self._model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            return self.model
        except Exception:
            self._loaded = True
            self.model = None
            return None

    def is_model_loaded(self):
        """模型是否真正加载成功"""
        return self.model is not None

    def _preprocess(self, img):
        """预处理图像"""
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img_r = cv2.resize(img, (256, 256))
        img_r = img_r.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_r = (img_r - mean) / std
        img_r = np.transpose(img_r, (2, 0, 1))
        return img_r

    def predict(self, img, threshold=0.5):
        """预测裂缝mask"""
        if self.model is None:
            self.load_model()
        if self.model is None:
            return np.zeros(img.shape[:2], dtype=np.uint8)

        import torch

        img_tensor = torch.from_numpy(self._preprocess(img)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(img_tensor)
            pred_np = pred.squeeze().cpu().numpy()

        pred_np = cv2.resize(pred_np, (img.shape[1], img.shape[0]))
        mask = (pred_np > threshold).astype(np.uint8) * 255

        return mask

    def predict_and_postprocess(self, img, threshold=0.25, min_area=20, max_area_ratio=0.30):
        """预测并进行后处理，得到精确的裂缝线条"""
        h, w = img.shape[:2]
        total_area = h * w

        # 基础预测
        mask = self.predict(img, threshold)

        if mask.sum() == 0:
            return mask

        # 连通域过滤
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = np.zeros_like(mask)

        for i in range(1, n):
            x, y, ww, hh, area = stats[i]

            # 面积过滤
            if area < min_area or area / total_area > max_area_ratio:
                continue

            # 长宽比过滤：裂缝是细长结构
            aspect = max(ww, hh) / max(min(ww, hh), 1)
            # 面积越小，要求的长宽比越低
            if area < 50:
                # 小区域允许较低的长宽比
                if aspect < 1.2:
                    continue
            else:
                if aspect < 1.5:
                    continue

            out[labels == i] = 255

        # 闭运算连接断裂的裂缝
        kernel = np.ones((3, 3), np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=3)

        # 再次过滤（去除闭运算后产生的大块区域）
        n2, labels2, stats2, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
        final = np.zeros_like(out)
        for i in range(1, n2):
            area = stats2[i][4]
            if area < min_area * 2 or area / total_area > max_area_ratio:
                continue
            final[labels2 == i] = 255

        # 细化为中心线
        final = self._extract_center_line(final)

        return final

    def _extract_center_line(self, mask):
        """提取中心线（距离变换方法）"""
        if mask.sum() == 0:
            return mask

        # 距离变换
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)

        # 取中心线
        max_dist = dist.max()
        if max_dist > 0:
            # 保留距离最大的50%作为中心线
            center_line = (dist > max_dist * 0.4).astype(np.uint8) * 255
        else:
            center_line = mask

        # 小膨胀使其更可见
        kernel = np.ones((2, 2), np.uint8)
        center_line = cv2.dilate(center_line, kernel, iterations=1)

        return center_line