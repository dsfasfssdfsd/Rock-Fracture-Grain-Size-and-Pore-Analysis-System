"""UNet裂缝检测常驻子进程模块
通过常驻子进程避免每次启动Python+加载模型的开销
"""
import os
import sys
import tempfile
import subprocess
import threading
import time
import cv2
import numpy as np


UNET_MODEL_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', 'unet_fracture.pth'))


class UNetDaemon:
    """UNet常驻子进程管理器"""

    def __init__(self):
        self._process = None
        self._tmp_dir = None
        self._lock = threading.Lock()
        self._ready = False

    def _ensure_started(self):
        """确保子进程已启动"""
        if self._process is not None and self._process.poll() is None and self._ready:
            return True

        self._stop()

        self._tmp_dir = tempfile.mkdtemp(prefix='unet_daemon_')
        input_path = os.path.join(self._tmp_dir, 'input.png')
        output_path = os.path.join(self._tmp_dir, 'output.png')
        signal_path = os.path.join(self._tmp_dir, 'signal.txt')

        script = f'''
import sys
import os
import cv2
import numpy as np
import time

model_path = r"{UNET_MODEL_PATH}"
input_path = r"{input_path}"
output_path = r"{output_path}"
signal_path = r"{signal_path}"

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

    device = "cpu"
    model = LightUNet()
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    with open(signal_path, "w") as f:
        f.write("READY")
except Exception as e:
    with open(signal_path, "w") as f:
        f.write(f"ERROR:{{str(e)}}")
    sys.exit(1)

# 主循环
last_mtime = 0
while True:
    try:
        if not os.path.exists(input_path):
            time.sleep(0.05)
            continue
        mtime = os.path.getmtime(input_path)
        if mtime == last_mtime:
            time.sleep(0.05)
            continue
        last_mtime = mtime
        time.sleep(0.05)

        img = cv2.imread(input_path)
        if img is None:
            continue

        # 读取参数
        threshold = 0.3
        tile_size = 512
        params_path = os.path.join(r"{self._tmp_dir}", "params.txt")
        if os.path.exists(params_path):
            try:
                with open(params_path, "r") as f:
                    for line in f:
                        k, v = line.strip().split("=")
                        if k == "threshold":
                            threshold = float(v)
                        elif k == "tile_size":
                            tile_size = int(v)
            except Exception:
                pass

        h, w = img.shape[:2]
        overlap = 128

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        if h <= tile_size and w <= tile_size:
            img_r = cv2.resize(img, (tile_size, tile_size))
            img_r = img_r.astype(np.float32) / 255.0
            img_r = (img_r - mean) / std
            img_r = np.transpose(img_r, (2, 0, 1))
            with torch.no_grad():
                pred = model(torch.from_numpy(img_r).unsqueeze(0).to(device))
                pred_np = pred.squeeze().cpu().numpy()
            pred_full = cv2.resize(pred_np, (w, h))
        else:
            stride = tile_size - overlap
            pred_full = np.zeros((h, w), dtype=np.float32)
            count_full = np.zeros((h, w), dtype=np.float32)
            for y in range(0, h, stride):
                for x in range(0, w, stride):
                    y2 = min(y + tile_size, h)
                    x2 = min(x + tile_size, w)
                    y1 = max(0, y2 - tile_size)
                    x1 = max(0, x2 - tile_size)
                    patch = img[y1:y2, x1:x2]
                    patch_r = cv2.resize(patch, (tile_size, tile_size))
                    patch_r = patch_r.astype(np.float32) / 255.0
                    patch_r = (patch_r - mean) / std
                    patch_r = np.transpose(patch_r, (2, 0, 1))
                    with torch.no_grad():
                        pred_p = model(torch.from_numpy(patch_r).unsqueeze(0).to(device))
                        pred_p = pred_p.squeeze().cpu().numpy()
                    pred_full[y1:y2, x1:x2] += cv2.resize(pred_p, (x2 - x1, y2 - y1))
                    count_full[y1:y2, x1:x2] += 1.0
            count_full[count_full < 1] = 1
            pred_full /= count_full

        # 后处理：阈值+闭运算
        mask = (pred_full > threshold).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        cv2.imwrite(output_path, mask)

        done_path = os.path.join(r"{self._tmp_dir}", "done.txt")
        with open(done_path, "w") as f:
            f.write(str(int(time.time())))

    except Exception as e:
        try:
            with open(signal_path, "w") as f:
                f.write(f"ERROR:{{str(e)}}")
        except Exception:
            pass
'''

        try:
            self._process = subprocess.Popen(
                [sys.executable, '-c', script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self._tmp_dir
            )
        except Exception as e:
            print(f"启动UNet子进程失败: {e}")
            return False

        signal_file = os.path.join(self._tmp_dir, 'signal.txt')
        for _ in range(1200):
            if os.path.exists(signal_file):
                try:
                    with open(signal_file, 'r') as f:
                        status = f.read().strip()
                    if status == 'READY':
                        self._ready = True
                        return True
                    elif status.startswith('ERROR'):
                        print(f"UNet子进程错误: {status}")
                        self._stop()
                        return False
                except Exception:
                    pass
            if self._process.poll() is not None:
                print("UNet子进程已退出")
                return False
            time.sleep(0.05)

        print("UNet子进程启动超时")
        self._stop()
        return False

    def predict(self, img, threshold=0.3, tile_size=512):
        """调用常驻子进程进行UNet推理"""
        with self._lock:
            if not self._ensure_started():
                return None

            input_path = os.path.join(self._tmp_dir, 'input.png')
            output_path = os.path.join(self._tmp_dir, 'output.png')
            done_path = os.path.join(self._tmp_dir, 'done.txt')
            params_path = os.path.join(self._tmp_dir, 'params.txt')

            for p in [input_path, output_path, done_path, params_path]:
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass

            try:
                with open(params_path, 'w') as f:
                    f.write(f"threshold={threshold}\n")
                    f.write(f"tile_size={tile_size}\n")
            except Exception:
                pass

            cv2.imwrite(input_path, img)

            for _ in range(600):
                if os.path.exists(done_path):
                    break
                if self._process.poll() is not None:
                    print("UNet子进程异常退出")
                    self._stop()
                    return None
                time.sleep(0.05)
            else:
                print("UNet检测超时")
                return None

            if not os.path.exists(output_path):
                return None

            try:
                mask = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
                return mask
            except Exception as e:
                print(f"读取UNet结果失败: {e}")
                return None

    def _stop(self):
        """停止子进程"""
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None
        self._ready = False
        if self._tmp_dir:
            try:
                import shutil
                shutil.rmtree(self._tmp_dir, ignore_errors=True)
            except Exception:
                pass
            self._tmp_dir = None

    def __del__(self):
        self._stop()


_unet_daemon = None
_daemon_lock = threading.Lock()


def get_unet_daemon():
    """获取全局UNet常驻子进程实例"""
    global _unet_daemon
    with _daemon_lock:
        if _unet_daemon is None:
            _unet_daemon = UNetDaemon()
    return _unet_daemon


def run_unet_daemon(img, threshold=0.3, tile_size=512):
    """使用常驻子进程运行UNet裂缝检测
    
    Args:
        img: 输入图像 (BGR)
        threshold: 二值化阈值
        tile_size: 滑窗尺寸
    
    Returns:
        mask: 裂缝掩码，失败返回None
    """
    daemon = get_unet_daemon()
    return daemon.predict(img, threshold=threshold, tile_size=tile_size)
