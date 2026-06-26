"""
深度学习裂缝检测 - 继续训练U-Net（更高精度配置）
"""
import os, cv2, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time, random, math, sys
from sklearn.model_selection import train_test_split

LOG_FILE = r'C:\Users\ZhuanZ1\Desktop\code1\code1\core_analysis_system\train_log_v2.txt'
log_f = open(LOG_FILE, 'w')

def log(msg):
    print(msg, flush=True)
    log_f.write(msg + '\n')
    log_f.flush()

MCD_DIR = r'C:\Users\ZhuanZ1\Desktop\code1\code1\MCD\MCD'
IMG_DIR = os.path.join(MCD_DIR, 'JPEGImages')
ANN_DIR = os.path.join(MCD_DIR, 'Annotations')
OUTPUT_DIR = r'C:\Users\ZhuanZ1\Desktop\code1\code1\core_analysis_system'

PATCH_SIZE = 256
BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_WORKERS = 0
DEVICE = 'cpu'
TRAIN_SPLIT = 0.8

log(f"使用设备: {DEVICE}")

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
            if has_fracture and random.random() < 0.6:
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
                beta = random.uniform(-20, 20)
                patch = np.clip(patch * alpha + beta, 0, 255).astype(np.uint8)
            if random.random() < 0.3:
                sigma = random.uniform(0.5, 2.0)
                patch = cv2.GaussianBlur(patch, (3, 3), sigma)

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
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * (1 - dice)

def compute_f1(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    target_bin = target.float()
    intersection = (pred_bin * target_bin).sum().item()
    precision = intersection / (pred_bin.sum().item() + 1e-6)
    recall = intersection / (target_bin.sum().item() + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return f1, precision, recall

def train():
    img_files = sorted([os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR) if f.endswith('.jpg')])
    ann_files = [os.path.join(ANN_DIR, os.path.basename(f).replace('.jpg', '.png')) for f in img_files]

    valid_files = []
    valid_anns = []
    for img, ann in zip(img_files, ann_files):
        if os.path.exists(img) and os.path.exists(ann):
            valid_files.append(img)
            valid_anns.append(ann)

    log(f"有效样本: {len(valid_files)}")

    train_imgs, val_imgs, train_anns, val_anns = train_test_split(
        valid_files, valid_anns, test_size=1-TRAIN_SPLIT, random_state=42)

    log(f"训练集: {len(train_imgs)}, 验证集: {len(val_imgs)}")

    train_dataset = FractureDataset(train_imgs, train_anns, patch_size=PATCH_SIZE, max_patches=12)
    val_dataset = ValDataset(val_imgs, val_anns, target_size=256)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    model = LightUNet().to(DEVICE)

    try:
        checkpoint = torch.load(os.path.join(OUTPUT_DIR, 'unet_fracture.pth'), map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint)
        log("已加载预训练模型，继续训练")
    except:
        log("无预训练模型，从头训练")

    criterion = DiceBCELoss(bce_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_f1 = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_f1 = 0.0
        start_time = time.time()

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            batch_f1, _, _ = compute_f1(outputs.detach(), masks.detach())
            train_f1 += batch_f1

        train_loss /= len(train_loader)
        train_f1 /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_f1 = 0.0
        val_p = 0.0
        val_r = 0.0

        with torch.no_grad():
            for images, masks, _, _ in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                f1, p, r = compute_f1(outputs, masks)
                val_f1 += f1
                val_p += p
                val_r += r

        val_loss /= len(val_loader)
        val_f1 /= len(val_loader)
        val_p /= len(val_loader)
        val_r /= len(val_loader)

        scheduler.step()

        elapsed = time.time() - start_time
        log(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {train_loss:.4f}/{val_loss:.4f} | F1: {train_f1:.4f}/{val_f1:.4f} | P/R: {val_p:.4f}/{val_r:.4f} | Time: {elapsed:.1f}s")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'unet_fracture.pth'))
            log(f"  保存最佳模型 (F1={best_f1:.4f})")

    log(f"训练完成! 最佳F1: {best_f1:.4f}")
    log_f.close()

if __name__ == "__main__":
    train()