"""
深度学习裂缝检测 - 轻量U-Net训练脚本（无预训练依赖）
"""
import os, cv2, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time, random, math, sys
from sklearn.model_selection import train_test_split

# 打开日志
LOG_FILE = r'C:\Users\ZhuanZ1\Desktop\code1\code1\core_analysis_system\train_log.txt'
log_f = open(LOG_FILE, 'w')

def log(msg):
    print(msg, flush=True)
    log_f.write(msg + '\n')
    log_f.flush()

# ============ 配置 ============
MCD_DIR = r'C:\Users\ZhuanZ1\Desktop\code1\code1\MCD\MCD'
IMG_DIR = os.path.join(MCD_DIR, 'JPEGImages')
ANN_DIR = os.path.join(MCD_DIR, 'Annotations')
OUTPUT_DIR = r'C:\Users\ZhuanZ1\Desktop\code1\code1\core_analysis_system'

PATCH_SIZE = 256   # 匹配推理尺寸
BATCH_SIZE = 2     # 256patch内存更大，减小batch
NUM_EPOCHS = 80    # 配合早停，给足够收敛空间
LEARNING_RATE = 3e-4
NUM_WORKERS = 0
DEVICE = 'cpu'
TRAIN_SPLIT = 0.8
EARLY_STOP_PATIENCE = 20  # 验证F1连续20epoch不提升则停止

log(f"使用设备: {DEVICE}")

# ============ 模型 ============

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
    """轻量U-Net"""
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


# ============ 数据集 ============

class FractureDataset(Dataset):
    def __init__(self, image_paths, mask_paths, patch_size=256, augment=True, max_patches=12):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.patch_size = patch_size
        self.augment = augment
        self.max_patches = max_patches

    def __len__(self):
        return len(self.image_paths) * self.max_patches

    def __getitem__(self, idx):
        img_idx = idx // self.max_patches
        img = cv2.imread(self.image_paths[img_idx])
        mask = cv2.imread(self.mask_paths[img_idx], 0)
        if img is None:
            return torch.zeros(3, self.patch_size, self.patch_size), torch.zeros(1, self.patch_size, self.patch_size)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = (mask > 0).astype(np.uint8)
        h, w = mask.shape

        has_fracture = mask.sum() > 0
        for _ in range(5):
            if has_fracture and random.random() < 0.5:
                pts = np.argwhere(mask > 0)
                if len(pts) > 0:
                    ry, rx = random.choice(pts[:, 0]), random.choice(pts[:, 1])
                else:
                    ry, rx = random.randint(0, h), random.randint(0, w)
            else:
                ry, rx = random.randint(0, h - 1), random.randint(0, w - 1)

            y0 = max(0, ry - self.patch_size // 2)
            x0 = max(0, rx - self.patch_size // 2)
            y1 = min(h, y0 + self.patch_size)
            x1 = min(w, x0 + self.patch_size)
            patch = img[y0:y1, x0:x1]
            pmask = mask[y0:y1, x0:x1]
            if patch.shape[0] < self.patch_size or patch.shape[1] < self.patch_size:
                patch = cv2.resize(patch, (self.patch_size, self.patch_size))
                pmask = cv2.resize(pmask, (self.patch_size, self.patch_size), interpolation=cv2.INTER_NEAREST)
            break

        if self.augment:
            if random.random() < 0.5:
                patch = np.fliplr(patch).copy()
                pmask = np.fliplr(pmask).copy()
            if random.random() < 0.5:
                patch = np.flipud(patch).copy()
                pmask = np.flipud(pmask).copy()
            k = random.randint(0, 3)
            patch = np.rot90(patch, k).copy()
            pmask = np.rot90(pmask, k).copy()
            if random.random() < 0.5:
                alpha = random.uniform(0.8, 1.2)
                beta = random.uniform(-15, 15)
                patch = np.clip(patch * alpha + beta, 0, 255).astype(np.uint8)

        patch = patch.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        patch = (patch - mean) / std
        patch = np.transpose(patch, (2, 0, 1))
        return torch.from_numpy(patch), torch.from_numpy(pmask.astype(np.float32)).unsqueeze(0)


class ValDataset(Dataset):
    def __init__(self, image_paths, mask_paths, target_size=256):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        mask = cv2.imread(self.mask_paths[idx], 0)
        orig_h, orig_w = mask.shape[:2]
        if img is None:
            return torch.zeros(3, self.target_size, self.target_size), torch.zeros(1, self.target_size, self.target_size), orig_h, orig_w

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = (mask > 0).astype(np.float32)
        img_r = cv2.resize(img, (self.target_size, self.target_size))
        mask_r = cv2.resize(mask, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)
        img_r = img_r.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_r = (img_r - mean) / std
        img_r = np.transpose(img_r, (2, 0, 1))
        return torch.from_numpy(img_r), torch.from_numpy(mask_r).unsqueeze(0), orig_h, orig_w


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.5, smooth=1e-6):
        super().__init__()
        self.bce = nn.BCELoss()
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        dice_loss = 1 - dice
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


def train_unet():
    img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    img_paths = [os.path.join(IMG_DIR, f) for f in img_files]
    ann_paths = [os.path.join(ANN_DIR, os.path.splitext(f)[0] + '.png') for f in img_files]
    valid = [(ip, ap) for ip, ap in zip(img_paths, ann_paths) if os.path.exists(ip) and os.path.exists(ap)]
    img_paths = [v[0] for v in valid]
    ann_paths = [v[1] for v in valid]
    log(f"有效数据: {len(img_paths)} 张")

    train_imgs, val_imgs, train_anns, val_anns = train_test_split(
        img_paths, ann_paths, test_size=1-TRAIN_SPLIT, random_state=42
    )
    log(f"训练集: {len(train_imgs)} 张, 验证集: {len(val_imgs)} 张")

    train_dataset = FractureDataset(train_imgs, train_anns, PATCH_SIZE, augment=True, max_patches=24)
    val_dataset = ValDataset(val_imgs, val_anns, target_size=PATCH_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=NUM_WORKERS)

    model = LightUNet(in_channels=3, out_channels=1).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    log(f"模型参数: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    criterion = DiceBCELoss(bce_weight=0.5)

    best_f1 = 0
    best_epoch = 0
    model_path = os.path.join(OUTPUT_DIR, 'unet_fracture.pth')

    log(f"\n开始训练，共 {NUM_EPOCHS} epochs (CPU模式)")
    log("-" * 70)

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        model.train()
        train_loss = 0
        for imgs, masks in train_loader:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)
        scheduler.step()

        model.eval()
        val_f1s = []
        with torch.no_grad():
            for imgs, masks, _, _, in val_loader:
                imgs = imgs.to(DEVICE)
                outputs = model(imgs)
                for i in range(outputs.shape[0]):
                    p = outputs[i].cpu().numpy().squeeze()
                    m = masks[i].numpy().squeeze()
                    tp = (p > 0.5) & (m > 0.5)
                    fp = (p > 0.5) & (m <= 0.5)
                    fn = (p <= 0.5) & (m > 0.5)
                    f1 = 2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum() + 1e-6)
                    val_f1s.append(f1)

        val_f1 = np.mean(val_f1s) if val_f1s else 0
        elapsed = time.time() - t0
        marker = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)
            marker = " *"
        lr = optimizer.param_groups[0]['lr']
        log(f"Epoch {epoch+1:3d}/{NUM_EPOCHS}: loss={train_loss:.4f} val_f1={val_f1:.4f} best={best_f1:.4f}(ep{best_epoch+1}) lr={lr:.2e} [{elapsed:.0f}s]{marker}")

        if epoch - best_epoch >= EARLY_STOP_PATIENCE and epoch > 20:
            log(f"\n早停! (验证F1连续{EARLY_STOP_PATIENCE}轮未提升)")
            break

    log(f"\n训练完成！最佳验证F1: {best_f1:.4f} (epoch {best_epoch+1})")
    log(f"模型: {model_path}")
    log_f.close()
    return model_path


if __name__ == '__main__':
    train_unet()
