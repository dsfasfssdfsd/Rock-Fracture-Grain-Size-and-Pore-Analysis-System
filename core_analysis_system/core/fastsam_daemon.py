"""FastSAM常驻子进程模块
通过常驻子进程避免每次启动Python+加载模型的开销
子进程启动后加载模型，通过stdin/stdout接收任务
"""
import os
import sys
import tempfile
import subprocess
import threading
import time
import cv2
import numpy as np


FASTSAM_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'FastSAM-x.pt')


class FastSAMDaemon:
    """FastSAM常驻子进程管理器"""

    def __init__(self):
        self._process = None
        self._tmp_dir = None
        self._lock = threading.Lock()
        self._ready = False

    def _ensure_started(self):
        """确保子进程已启动"""
        if self._process is not None and self._process.poll() is None and self._ready:
            return True

        # 清理旧进程
        self._stop()

        # 创建临时目录
        self._tmp_dir = tempfile.mkdtemp(prefix='fastsam_daemon_')
        input_path = os.path.join(self._tmp_dir, 'input.png')
        output_path = os.path.join(self._tmp_dir, 'output.npz')
        signal_path = os.path.join(self._tmp_dir, 'signal.txt')

        # 子进程脚本
        script = f'''
import sys
import os
import cv2
import numpy as np
import time

try:
    from ultralytics import FastSAM
except Exception as e:
    with open(r"{signal_path}", "w") as f:
        f.write(f"ERROR:{{str(e)}}")
    sys.exit(1)

model_path = r"{FASTSAM_MODEL_PATH}"
input_path = r"{input_path}"
output_path = r"{output_path}"
signal_path = r"{signal_path}"

try:
    model = FastSAM(model_path)
    with open(signal_path, "w") as f:
        f.write("READY")
except Exception as e:
    with open(signal_path, "w") as f:
        f.write(f"ERROR:{{str(e)}}")
    sys.exit(1)

# 主循环：等待输入文件，处理，写输出
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
        time.sleep(0.05)  # 等待写入完成

        img = cv2.imread(input_path)
        if img is None:
            continue

        # 读取参数
        params = {{}}
        params_path = os.path.join(r"{self._tmp_dir}", "params.txt")
        if os.path.exists(params_path):
            try:
                with open(params_path, "r") as f:
                    for line in f:
                        k, v = line.strip().split("=")
                        params[k] = float(v) if "." in v else int(v)
            except Exception:
                pass

        conf = params.get("conf", 0.05)
        iou = params.get("iou", 0.5)
        imgsz = params.get("imgsz", 640)

        h, w = img.shape[:2]
        results = model(img, imgsz=imgsz, conf=conf, iou=iou, device="cpu", retina_masks=True)

        contours_list = []
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            for m in masks:
                m_r = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                cnts, _ = cv2.findContours(m_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    cnt = max(cnts, key=cv2.contourArea)
                    contours_list.append(cnt)

        if contours_list:
            np.savez(output_path, *contours_list)
        else:
            np.savez(output_path, empty=np.array([]))

        # 写完成信号
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

        # 启动子进程
        try:
            self._process = subprocess.Popen(
                [sys.executable, '-c', script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self._tmp_dir
            )
        except Exception as e:
            print(f"启动FastSAM子进程失败: {e}")
            return False

        # 等待READY信号（最多60秒）
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
                        print(f"FastSAM子进程错误: {status}")
                        self._stop()
                        return False
                except Exception:
                    pass
            if self._process.poll() is not None:
                print("FastSAM子进程已退出")
                return False
            time.sleep(0.05)

        print("FastSAM子进程启动超时")
        self._stop()
        return False

    def detect(self, img, conf=0.05, iou=0.5, imgsz=640):
        """调用常驻子进程进行检测"""
        with self._lock:
            if not self._ensure_started():
                return None

            input_path = os.path.join(self._tmp_dir, 'input.png')
            output_path = os.path.join(self._tmp_dir, 'output.npz')
            done_path = os.path.join(self._tmp_dir, 'done.txt')
            params_path = os.path.join(self._tmp_dir, 'params.txt')

            # 清理旧文件
            for p in [input_path, output_path, done_path, params_path]:
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass

            # 写参数
            try:
                with open(params_path, 'w') as f:
                    f.write(f"conf={conf}\n")
                    f.write(f"iou={iou}\n")
                    f.write(f"imgsz={imgsz}\n")
            except Exception:
                pass

            # 写输入图像（触发子进程处理）
            cv2.imwrite(input_path, img)

            # 等待完成（最多30秒）
            for _ in range(600):
                if os.path.exists(done_path):
                    break
                if self._process.poll() is not None:
                    print("FastSAM子进程异常退出")
                    self._stop()
                    return None
                time.sleep(0.05)
            else:
                print("FastSAM检测超时")
                return None

            # 读取结果
            if not os.path.exists(output_path):
                return []

            try:
                data = np.load(output_path, allow_pickle=True)
                if 'empty' in data:
                    return []
                contours = [data[k] for k in sorted(data.keys()) if k != 'empty']
                return contours
            except Exception as e:
                print(f"读取FastSAM结果失败: {e}")
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


# 全局单例
_fastsam_daemon = None
_daemon_lock = threading.Lock()


def get_fastsam_daemon():
    """获取全局FastSAM常驻子进程实例"""
    global _fastsam_daemon
    with _daemon_lock:
        if _fastsam_daemon is None:
            _fastsam_daemon = FastSAMDaemon()
    return _fastsam_daemon


def run_fastsam_daemon(img, conf=0.05, iou=0.5, imgsz=640):
    """使用常驻子进程运行FastSAM
    
    Args:
        img: 输入图像
        conf: 置信度阈值
        iou: IOU阈值
        imgsz: 推理尺寸
    
    Returns:
        contours: 轮廓列表，失败返回None
    """
    daemon = get_fastsam_daemon()
    return daemon.detect(img, conf=conf, iou=iou, imgsz=imgsz)
