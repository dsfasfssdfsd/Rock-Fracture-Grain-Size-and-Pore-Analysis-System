"""
U-Net裂缝检测推理 - 加载训练好的模型进行预测
"""
import os, cv2, numpy as np, torch, torch.nn as nn

MODEL_PATH = r'C:\Users\ZhuanZ1\Desktop\code1\code1\core_analysis_system\unet_fracture.pth'
DEVICE = 'cpu'

# ============ 模型定义（与训练时一致）============

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


# ============ 推理 ============

def load_model():
    """加载模型"""
    model = LightUNet(in_channels=3, out_channels=1).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print(f"模型已加载: {MODEL_PATH}")
    else:
        print(f"模型文件不存在: {MODEL_PATH}")
    return model


def predict_image(model, image, threshold=0.5, target_size=256):
    """对单张图像进行裂缝检测预测"""
    orig_h, orig_w = image.shape[:2]

    # 预处理
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_r = cv2.resize(img, (target_size, target_size))
    img_r = img_r.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_r = (img_r - mean) / std
    img_r = np.transpose(img_r, (2, 0, 1))
    img_t = torch.from_numpy(img_r).unsqueeze(0).to(DEVICE)

    # 推理
    with torch.no_grad():
        output = model(img_t)
        prob = output.squeeze().cpu().numpy()

    # 后处理
    mask = cv2.resize(prob, (orig_w, orig_h))
    mask_binary = (mask > threshold).astype(np.uint8) * 255

    # 形态学清理
    kernel = np.ones((3, 3), np.uint8)
    mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
    mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)

    return mask_binary


def predict_gray(model, gray, threshold=0.5, target_size=256):
    """对灰度图进行预测"""
    orig_h, orig_w = gray.shape

    # 灰度转RGB
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img_r = cv2.resize(img, (target_size, target_size))
    img_r = img_r.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_r = (img_r - mean) / std
    img_r = np.transpose(img_r, (2, 0, 1))
    img_t = torch.from_numpy(img_r).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_t)
        prob = output.squeeze().cpu().numpy()

    mask = cv2.resize(prob, (orig_w, orig_h))
    mask_binary = (mask > threshold).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
    mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)

    return mask_binary


if __name__ == '__main__':
    model = load_model()
    if model is None:
        print("模型未加载，请先训练模型")
    else:
        # 测试
        test_img = r'C:\Users\ZhuanZ1\Desktop\code1\code1\picture\S0110.bmp'
        if os.path.exists(test_img):
            img = cv2.imread(test_img)
            mask = predict_image(model, img)
            print(f"检测到 {np.sum(mask > 0)} 个裂缝像素")
