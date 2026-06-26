# -*- coding: utf-8 -*-
"""UNet 推理子进程脚本（避免 DLL 冲突）"""
import sys, os
import cv2
import numpy as np

model_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'unet_fracture.pth'))

def predict(img_path, out_path, threshold=0.3):
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
                nn.ReLU(inplace=True))
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

    device = 'cpu'
    model = LightUNet()
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f'Cannot read {img_path}')

    h, w = img.shape[:2]

    # 多尺度滑窗推理：用512尺寸的窗口滑过整张图，提升大图像精度
    tile_size = 512
    overlap = 128

    if h <= tile_size and w <= tile_size:
        # 小图直接推理，缩放到tile_size
        img_r = cv2.resize(img, (tile_size, tile_size))
        img_r = img_r.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_r = (img_r - mean) / std
        img_r = np.transpose(img_r, (2, 0, 1))

        with torch.no_grad():
            pred = model(torch.from_numpy(img_r).unsqueeze(0).to(device))
            pred_np = pred.squeeze().cpu().numpy()

        pred_full = cv2.resize(pred_np, (w, h))
    else:
        # 大图：滑窗推理
        stride = tile_size - overlap
        pred_full = np.zeros((h, w), dtype=np.float32)
        count_full = np.zeros((h, w), dtype=np.float32)

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        y = 0
        while y < h:
            x = 0
            y_end = min(y + tile_size, h)
            y_start = y_end - tile_size
            if y_start < 0:
                y_start = 0
                y_end = tile_size
            while x < w:
                x_end = min(x + tile_size, w)
                x_start = x_end - tile_size
                if x_start < 0:
                    x_start = 0
                    x_end = tile_size

                tile = img[y_start:y_end, x_start:x_end]
                tile_r = cv2.resize(tile, (tile_size, tile_size))
                tile_r = tile_r.astype(np.float32) / 255.0
                tile_r = (tile_r - mean) / std
                tile_r = np.transpose(tile_r, (2, 0, 1))

                with torch.no_grad():
                    pred_tile = model(torch.from_numpy(tile_r).unsqueeze(0).to(device))
                    pred_np = pred_tile.squeeze().cpu().numpy()

                pred_orig = cv2.resize(pred_np, (x_end - x_start, y_end - y_start))
                pred_full[y_start:y_end, x_start:x_end] += pred_orig
                count_full[y_start:y_end, x_start:x_end] += 1.0

                x += stride
            y += stride

        count_full[count_full == 0] = 1.0
        pred_full = pred_full / count_full

    mask = (pred_full > threshold).astype(np.uint8) * 255

    # 后处理：形态学闭运算连接断裂的裂缝
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    cv2.imwrite(out_path, mask)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: unet_infer_subprocess.py <input_image> <output_mask> [threshold]')
        sys.exit(1)
    th = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    predict(sys.argv[1], sys.argv[2], th)
