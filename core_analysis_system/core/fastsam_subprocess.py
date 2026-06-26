"""FastSAM子进程推理模块
解决GUI进程中PyTorch DLL加载失败的问题：
启动独立Python子进程运行FastSAM，通过临时文件传递输入输出
"""
import os
import sys
import tempfile
import subprocess
import cv2
import numpy as np


FASTSAM_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'FastSAM-x.pt')


def run_fastsam_subprocess(img, conf=0.05, iou=0.5, imgsz=1024):
    """在子进程中运行FastSAM，返回轮廓列表
    
    Args:
        img: 输入图像 (BGR)
        conf: 置信度阈值
        iou: IOU阈值
        imgsz: 推理尺寸
    
    Returns:
        contours: 轮廓列表 (numpy数组列表)
        失败返回None
    """
    # 创建临时文件
    tmp_dir = tempfile.mkdtemp(prefix='fastsam_')
    input_path = os.path.join(tmp_dir, 'input.png')
    output_path = os.path.join(tmp_dir, 'output.npz')

    try:
        # 保存输入图像
        cv2.imwrite(input_path, img)

        # 构造子进程脚本
        script = f'''
import sys
import cv2
import numpy as np

try:
    from ultralytics import FastSAM
    
    model_path = r"{FASTSAM_MODEL_PATH}"
    input_path = r"{input_path}"
    output_path = r"{output_path}"
    conf = {conf}
    iou = {iou}
    imgsz = {imgsz}
    
    img = cv2.imread(input_path)
    if img is None:
        print("ERROR: cannot read input", flush=True)
        sys.exit(1)
    
    h, w = img.shape[:2]
    model = FastSAM(model_path)
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
    
    # 保存轮廓为npz
    if contours_list:
        np.savez(output_path, *contours_list)
    else:
        np.savez(output_path, empty=np.array([]))
    
    print(f"OK:{{len(contours_list)}}", flush=True)
    
except Exception as e:
    import traceback
    print(f"ERROR:{{str(e)}}", flush=True)
    traceback.print_exc()
    sys.exit(1)
'''

        # 启动子进程
        result = subprocess.run(
            [sys.executable, '-c', script],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=tmp_dir
        )

        if result.returncode != 0:
            print(f"FastSAM子进程失败: {result.stderr}")
            return None

        # 检查输出
        if not os.path.exists(output_path):
            print("FastSAM子进程无输出文件")
            return None

        # 加载轮廓
        data = np.load(output_path, allow_pickle=True)
        if 'empty' in data:
            return []

        contours = [data[k] for k in sorted(data.keys())
                    if k not in ('empty',)]
        return contours

    except Exception as e:
        print(f"FastSAM子进程调用异常: {e}")
        return None
    finally:
        # 清理临时文件
        try:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass
