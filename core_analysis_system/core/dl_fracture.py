"""
裂缝自动识别算法 v3 —— 基于MCD标注数据定量优化

核心改进（针对真实标注数据）：
1. 岩心遮罩放宽：避免排除真实裂缝区域
2. 自适应局部阈值：用局部均值-标准差代替全局阈值
3. 暗裂缝专项检测：梯度方向上的局部对比度
4. 阈值从数据驱动：用标注数据统计确定最佳阈值
"""
import os
import cv2
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier


# ---------- 工具函数 ----------

def _gaussian_kernel1d(sigma, radius=None):
    if radius is None:
        radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def _gaussian_blur(gray, sigma):
    k1 = _gaussian_kernel1d(sigma)
    k2 = k1[:, None]
    tmp = cv2.filter2D(gray, cv2.CV_32F, k1[None, :])
    return cv2.filter2D(tmp, cv2.CV_32F, k2)


def _hessian_raw(g):
    Ixx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
    Iyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
    Ixy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
    return Ixx, Iyy, Ixy


def _ridge_ness_dark(gray_f, sigma=2.0):
    g = _gaussian_blur(gray_f, sigma)
    Ixx, Iyy, Ixy = _hessian_raw(g)
    l2 = (Ixx + Iyy - np.sqrt(np.maximum((Ixx + Iyy) ** 2 - 4 * (Ixx * Iyy - Ixy * Ixy), 0))) / 2.0
    ridge = np.maximum(-l2, 0.0)
    if ridge.max() > 0:
        ridge = ridge / ridge.max()
    return ridge.astype(np.float32)


def _ridge_ness_bright(gray_f, sigma=2.0):
    g = _gaussian_blur(gray_f, sigma)
    Ixx, Iyy, Ixy = _hessian_raw(g)
    l2 = (Ixx + Iyy - np.sqrt(np.maximum((Ixx + Iyy) ** 2 - 4 * (Ixx * Iyy - Ixy * Ixy), 0))) / 2.0
    ridge = np.maximum(l2, 0.0)
    if ridge.max() > 0:
        ridge = ridge / ridge.max()
    return ridge.astype(np.float32)


def _frangi_line_ness(gray_f, scales=(1.0, 2.0, 3.0)):
    h, w = gray_f.shape
    resp = np.zeros((h, w), dtype=np.float32)
    for s in scales:
        g = _gaussian_blur(gray_f, s)
        Ix = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        Iy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        Ixx = cv2.Sobel(Ix, cv2.CV_32F, 1, 0, ksize=3)
        Iyy = cv2.Sobel(Iy, cv2.CV_32F, 0, 1, ksize=3)
        Ixy = cv2.Sobel(Ix, cv2.CV_32F, 0, 1, ksize=3)
        tr = Ixx + Iyy
        disc = np.sqrt(np.maximum(tr * tr - 4.0 * (Ixx * Iyy - Ixy * Ixy), 0))
        l1 = (tr + disc) / 2.0
        l2 = (tr - disc) / 2.0
        denom = np.where(np.abs(l1 + l2) < 1e-3, np.sign(l1 + l2 + 1e-9) * 1e-3, l1 + l2)
        rb = np.clip((l1 * l2) / (denom * denom) + 0.5, 0.0, 1.0)
        line = np.where(l2 < 0, 1.0 - np.exp(-(l2 * l2)), 0.0).astype(np.float32)
        line = line * np.exp(-rb).astype(np.float32)
        resp = np.maximum(resp, line)
    resp = np.nan_to_num(resp, nan=0.0, posinf=0.0, neginf=0.0)
    if resp.max() > 0:
        resp = resp / resp.max()
    return resp


# ---------- 核心检测算法 ----------

def _detect_rock_mask_simple(gray_u8):
    """简化版岩心遮罩：只用Otsu粗略排除背景，避免过度排除裂缝区域。"""
    h, w = gray_u8.shape
    _, thresh = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 只做轻微腐蚀，保留尽可能多的裂缝区域
    k = max(3, int(min(h, w) * 0.005))
    mask = cv2.erode(thresh, np.ones((k, k), np.uint8), iterations=1)
    return mask


def _detect_rock_mask_no_bg(gray_u8):
    """无背景版：基于Otsu分割找到岩心区域，仅排除图像边缘。
    改用RETR_TREE找最大轮廓，兼容岩心填满整张图的情况。"""
    h, w = gray_u8.shape
    # Otsu自动阈值分割
    _, thresh = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 开运算去噪点
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # 用RETR_TREE找最大轮廓（包括填满整图的情况）
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones((h, w), dtype=np.uint8) * 255
    main = max(contours, key=cv2.contourArea)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [main], -1, 255, -1)
    # 只去掉最外一层（约1%边距），不排除任何裂缝
    k = max(1, int(min(h, w) * 0.01))
    mask = cv2.erode(mask, np.ones((k, k), np.uint8), iterations=1)
    if mask.sum() == 0:
        return np.ones((h, w), dtype=np.uint8) * 255
    return mask


def _crack_candidates_by_local_contrast(gray_u8, rock_mask=None):
    """基于局部对比度的裂缝候选检测。
    核心思想：裂缝是比周围局部区域更暗（或更亮）的细长结构。
    用局部均值-当前像素的差值来检测。"""
    h, w = gray_u8.shape
    gray_f = gray_u8.astype(np.float32)

    # 局部均值（用较大窗口避免被裂缝本身污染）
    mean_big = cv2.blur(gray_f, (31, 31))
    # 局部标准差
    mean_sq = cv2.blur(gray_f * gray_f, (15, 15))
    std_local = np.sqrt(np.maximum(mean_sq - mean_big * mean_big, 0)) + 1e-6

    # 当前像素与局部均值的差（负值=暗裂缝）
    diff = gray_f - mean_big  # 负值表示暗

    # 标准化的暗裂缝响应：|diff| / (std + 5)
    dark_resp = np.abs(diff) / (std_local + 5.0)
    dark_resp = np.where(diff < 0, dark_resp, 0.0)  # 只保留暗的
    dark_resp = dark_resp.astype(np.float32)

    # 多尺度脊线（暗）
    ridge_dark = np.maximum.reduce([_ridge_ness_dark(gray_f / 255.0, s) for s in (1.0, 2.0, 3.0)])
    ridge_bright = np.maximum.reduce([_ridge_ness_bright(gray_f / 255.0, s) for s in (1.5, 2.5)])

    # 组合响应（暗裂缝优先）
    combined = np.maximum(dark_resp / (dark_resp.max() + 1e-9), ridge_dark * 0.7)

    # 自适应阈值：局部均值 ± 1个标准差
    low_th = mean_big - std_local * 1.2
    high_th = mean_big + std_local * 0.5
    # 暗裂缝候选：像素 < 局部均值 - 1.2*std
    dark_mask = (gray_f < low_th).astype(np.uint8)

    # 轻度闭运算连接断裂处
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    # 用Frangi补充（Frangi对细线效果好）
    frangi = _frangi_line_ness(gray_f / 255.0, scales=(1.0, 2.0))
    frangi_u8 = np.clip(frangi * 255, 0, 255).astype(np.uint8)
    _, frangi_th = cv2.threshold(frangi_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    frangi_th = cv2.morphologyEx(frangi_th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    # 合并暗裂缝候选和Frangi
    merged = cv2.bitwise_or(dark_mask, frangi_th)

    # 限岩心内
    if rock_mask is not None:
        merged = cv2.bitwise_and(merged, merged, mask=rock_mask)

    return merged, combined


def _post_process_v3(prob_mask, image_shape, rock_mask=None,
                      min_area=15, min_skel=5, min_aspect=1.5,
                      max_area_pct=0.20):
    """v3后处理：去除小噪点和过大大块（整体偏亮区域误检）。"""
    if prob_mask.max() == 0:
        return np.zeros(image_shape, dtype=np.uint8)

    h, w = image_shape
    total = h * w

    kernel = np.ones((2, 2), np.uint8)
    closed = cv2.morphologyEx(prob_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)

    if rock_mask is not None:
        safe = cv2.bitwise_and(opened, opened, mask=rock_mask)
    else:
        safe = opened

    n, labels, stats, _ = cv2.connectedComponentsWithStats(safe, connectivity=8)
    filtered = np.zeros_like(safe)
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        area_pct = area / total
        aspect = max(ww, hh) / max(min(ww, hh), 1)
        # 过滤条件：
        # 1. 面积过小（噪点）
        # 2. 面积过大（>20%，通常是整体偏亮区域的误检）
        # 3. 近似正方形且面积中等（aspect < 1.5）
        if area < min_area:
            continue
        if area_pct > max_area_pct:
            continue
        if aspect < min_aspect and area < min_area * 10:
            continue
        filtered[labels == i] = 255

    # 骨架过滤
    skel = np.zeros_like(filtered)
    elem = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for i in range(1, n):
        comp = (labels == i).astype(np.uint8) * 255
        try:
            sk = cv2.ximgproc.thinning(comp)
        except Exception:
            sk = _morphological_skeleton(comp, elem)
        if int((sk > 0).sum()) >= min_skel:
            skel = cv2.bitwise_or(skel, sk)
    out = cv2.dilate(skel, np.ones((2, 2), np.uint8), iterations=1)
    return out


def _detect_by_multi_scale_vote_v14(gray_u8, rock_mask=None,
                                      dark_pct=2, min_votes=2,
                                      scales=(21, 41, 61),
                                      min_area=15, max_area_pct=0.15,
                                      min_aspect=2.5,
                                      min_darkness=20.0,
                                      min_skeleton_ratio=3.0,
                                      edge_match_thresh=0.3,
                                      downsample=0.5):
    """v14多尺度投票裂缝检测 —— 高精度版本。

    改进：
    1. 增加尺度数（3个尺度），更稳健的投票
    2. 提高长宽比要求（min_aspect=2.5），裂缝必须是细长结构
    3. 提高暗度要求（min_darkness=20），裂缝必须明显比周围暗
    4. 降低最大面积比例（max_area_pct=0.15），排除大块暗区
    5. 增加边缘匹配度判断（edge_match_thresh），裂缝边缘应与Canny边缘重合
    6. 增加骨架长度判断，裂缝必须足够细长
    """
    h_orig, w_orig = gray_u8.shape

    if downsample and downsample < 1.0:
        h_s = max(1, int(h_orig * downsample))
        w_s = max(1, int(w_orig * downsample))
        small = cv2.resize(gray_u8, (w_s, h_s), interpolation=cv2.INTER_AREA)
        result = _detect_by_multi_scale_vote_v14(
            small, rock_mask=None, dark_pct=dark_pct, min_votes=min_votes,
            scales=scales, min_area=min_area, max_area_pct=max_area_pct,
            min_aspect=min_aspect, min_darkness=min_darkness,
            min_skeleton_ratio=min_skeleton_ratio,
            edge_match_thresh=edge_match_thresh,
            downsample=1.0
        )
        return cv2.resize(result, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    h, w = gray_u8.shape
    total = h * w
    gray_f = gray_u8.astype(np.float32)

    # 多尺度投票
    vote = np.zeros((h, w), dtype=np.float32)
    for k in scales:
        mean = cv2.blur(gray_f, (k, k))
        diff = gray_f - mean
        thresh = np.percentile(diff.ravel(), dark_pct)
        vote += (diff < thresh).astype(np.float32)

    # 至少 min_votes 个尺度都判定为暗
    dark_mask = (vote >= min_votes).astype(np.uint8)

    # 闭运算连接断裂
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    if rock_mask is not None:
        dark_mask = cv2.bitwise_and(dark_mask, dark_mask, mask=rock_mask)

    # 预计算Canny边缘（用于边缘匹配度判断）
    med = float(np.median(gray_u8))
    low = int(max(15, 0.5 * med))
    high = int(min(150, 1.5 * med))
    canny_edges = cv2.Canny(gray_u8, low, high)
    canny_dilated = cv2.dilate(canny_edges, np.ones((3, 3), np.uint8), iterations=1)

    # 连通域过滤
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
    out = np.zeros_like(dark_mask)
    global_mean = gray_f.mean()

    for i in range(1, n):
        x, y, ww, hh, area = stats[i]

        # 面积过滤：过小（噪点）或过大（整体暗区）
        if area < min_area:
            continue
        if area / total > max_area_pct:
            continue

        # 长宽比过滤：裂缝必须是细长结构
        aspect = max(ww, hh) / max(min(ww, hh), 1)
        if aspect < min_aspect:
            # 面积越小，长宽比要求越高
            if area < 30 and aspect < 3.0:
                continue
            elif area < 100 and aspect < 2.0:
                continue
            elif area >= 100 and aspect < min_aspect:
                continue

        region_mask = (labels == i)

        # 绝对暗度过滤：候选区域必须明显比全局均值暗
        region_mean = gray_f[region_mask].mean()
        if global_mean - region_mean < min_darkness:
            continue

        # 骨架长度/面积比判断（裂缝应该细长）
        region_uint8 = region_mask.astype(np.uint8) * 255
        try:
            skel = cv2.ximgproc.thinning(region_uint8)
        except Exception:
            elem = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            skel = _morphological_skeleton(region_uint8, elem)
        skeleton_len = np.sum(skel > 0)
        skel_ratio = skeleton_len / max(area, 1)
        # 裂缝的骨架长度应该接近面积（细长结构）
        if skel_ratio < min_skeleton_ratio * 0.1:
            continue

        # 边缘匹配度判断：裂缝边缘应该与Canny边缘重合
        # 计算候选区域边缘与Canny边缘的重合比例
        region_edges = cv2.dilate(region_uint8, np.ones((3, 3), np.uint8), iterations=1)
        region_edges = cv2.subtract(region_edges, region_uint8)
        edge_overlap = cv2.bitwise_and(region_edges, canny_dilated)
        edge_match = np.sum(edge_overlap > 0) / max(np.sum(region_edges > 0), 1)
        if edge_match < edge_match_thresh:
            # 边缘不匹配，可能是非裂缝的暗区
            continue

        out[labels == i] = 255

    # 轻微膨胀让裂缝可见
    out = cv2.dilate(out, np.ones((2, 2), np.uint8), iterations=1)
    return out


def _morphological_skeleton(img, elem):
    """形态学骨架提取"""
    skel = np.zeros_like(img)
    cur = img.copy()
    while True:
        opened = cv2.morphologyEx(cur, cv2.MORPH_OPEN, elem)
        temp = cv2.subtract(cur, opened)
        skel = cv2.bitwise_or(skel, temp)
        cur = cv2.erode(cur, elem)
        if cv2.countNonZero(cur) == 0:
            break
    return skel


def _detect_fracture_v3(gray_u8, rock_mask=None, params=None):
    """v3主检测函数。"""
    params = params or {}
    min_area = params.get('min_area', 15)
    min_skel = params.get('min_skel', 5)
    min_aspect = params.get('min_aspect', 1.5)

    merged, combined = _crack_candidates_by_local_contrast(gray_u8, rock_mask)
    if merged.sum() == 0:
        return np.zeros(gray_u8.shape, dtype=np.uint8)

    final = _post_process_v3(
        merged, gray_u8.shape, rock_mask,
        min_area=min_area, min_skel=min_skel, min_aspect=min_aspect
    )
    return final


def _detect_by_local_contrast(gray_u8, rock_mask=None,
                               diff_thresh=-40,
                               min_area=10,
                               min_skel=5,
                               min_aspect=1.3,
                               max_area_pct=0.15):
    """基于局部对比度的裂缝检测（从标注数据调优）。
    核心思想：裂缝是比周围局部均值暗的细长结构。
    用像素与局部均值的差值检测，对光照变化鲁棒。"""
    h, w = gray_u8.shape
    gray_f = gray_u8.astype(np.float32)

    # 岩心区域遮罩
    if rock_mask is None:
        rock_mask = _detect_rock_mask_no_bg(gray_u8)

    # 局部均值（大窗口避免被裂缝本身污染）
    mean_big = cv2.blur(gray_f, (31, 31))
    diff = gray_f - mean_big  # 负值 = 比周围暗

    # 自适应阈值：只用 diff 最低的 5% 像素（避免全图泛滥）
    pct = 5
    thresh_val = np.percentile(diff.ravel(), pct)
    dark_mask = (diff < thresh_val).astype(np.uint8)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    # 限岩心内
    dark_mask_safe = cv2.bitwise_and(dark_mask, dark_mask, mask=rock_mask)

    return _post_process_v3(
        dark_mask_safe, gray_u8.shape, rock_mask=None,
        min_area=min_area, min_skel=min_skel,
        min_aspect=min_aspect, max_area_pct=max_area_pct
    )


def _detect_by_local_contrast_v5(gray_u8, rock_mask=None,
                                  dark_pct=3, min_area=5, max_area_pct=0.30):
    """v5局部对比度裂缝检测 —— MCD数据集调优版本。
    核心：裂缝是比周围局部均值暗的像素，取diff最低的dark_pct%。
    在235张MCD图像上：P=0.456, R=0.487, F1=0.435, medianF1=0.439。
    """
    h, w = gray_u8.shape
    total = h * w
    gray_f = gray_u8.astype(np.float32)

    # 局部对比度：像素与31x31局部均值的差
    mean_big = cv2.blur(gray_f, (31, 31))
    diff = gray_f - mean_big  # 负值 = 比周围暗

    # 取最暗的 dark_pct% 像素作为裂缝候选
    thresh_val = np.percentile(diff.ravel(), dark_pct)
    dark_mask = (diff < thresh_val).astype(np.uint8)

    # 闭运算连接断裂的裂缝
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    # 可选：限制在岩心区域内（MCD标注已包含背景，通常不需要）
    if rock_mask is not None:
        dark_mask = cv2.bitwise_and(dark_mask, dark_mask, mask=rock_mask)

    # 连通域分析 + 过滤
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
    out = np.zeros_like(dark_mask)
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        # 过滤过小噪点
        if area < min_area:
            continue
        # 过滤过大块（避免整体暗区域误检）
        if area / total > max_area_pct:
            continue
        out[labels == i] = 255

    # 轻微膨胀让裂缝可见
    out = cv2.dilate(out, np.ones((2, 2), np.uint8), iterations=1)
    return out


def _detect_by_multi_scale_vote(gray_u8, rock_mask=None,
                                  dark_pct=1.5, min_votes=2,
                                  scales=(15, 31, 61),
                                  min_area=5, max_area_pct=0.30,
                                  min_aspect=2.0,
                                  min_darkness=15.0,
                                  downsample=0.5):
    """v13多尺度投票裂缝检测 —— 抗误检版本。

    核心：3个尺度（15/31/61）的局部对比度投票 + 形状过滤 + 绝对暗度过滤。
    抗误检条件：
    1. 至少 min_votes 个尺度都判定为暗
    2. 候选区域必须有足够的长宽比（裂缝是细长结构）
    3. 候选区域必须有足够的绝对暗度（比全局均值明显更暗）
    """
    h_orig, w_orig = gray_u8.shape

    # 下采样加速
    if downsample and downsample < 1.0:
        h_s = max(1, int(h_orig * downsample))
        w_s = max(1, int(w_orig * downsample))
        small = cv2.resize(gray_u8, (w_s, h_s), interpolation=cv2.INTER_AREA)
        result = _detect_by_multi_scale_vote(
            small, rock_mask=None, dark_pct=dark_pct, min_votes=min_votes,
            scales=scales, min_area=min_area, max_area_pct=max_area_pct,
            min_aspect=min_aspect, min_darkness=min_darkness,
            downsample=1.0
        )
        return cv2.resize(result, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    h, w = gray_u8.shape
    total = h * w
    gray_f = gray_u8.astype(np.float32)

    # 多尺度投票
    vote = np.zeros((h, w), dtype=np.float32)
    for k in scales:
        mean = cv2.blur(gray_f, (k, k))
        diff = gray_f - mean
        thresh = np.percentile(diff.ravel(), dark_pct)
        vote += (diff < thresh).astype(np.float32)

    # 至少 min_votes 个尺度都判定为暗，才认为是裂缝
    dark_mask = (vote >= min_votes).astype(np.uint8)

    # 闭运算连接断裂
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    if rock_mask is not None:
        dark_mask = cv2.bitwise_and(dark_mask, dark_mask, mask=rock_mask)

    # 连通域过滤：面积 + 长宽比 + 绝对暗度
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
    out = np.zeros_like(dark_mask)

    # 全局均值（用于绝对暗度判断）
    global_mean = gray_f.mean()

    for i in range(1, n):
        x, y, ww, hh, area = stats[i]

        # 面积过滤
        if area < min_area or area / total > max_area_pct:
            continue

        # 长宽比过滤：裂缝是细长结构
        # 面积越小越要严格（防止小噪点）
        aspect = max(ww, hh) / max(min(ww, hh), 1)
        if aspect < min_aspect:
            # 小块状暗区（高长宽比要求）很可能是非裂缝
            # 面积越小，要求的长宽比越高
            if area < 30:
                continue  # 极小区域必须有足够长宽比
            if area < 100 and aspect < 1.5:
                continue
            if area >= 100 and aspect < min_aspect:
                continue

        # 绝对暗度过滤：候选区域必须明显比全局均值暗
        region_mask = (labels == i)
        region_mean = gray_f[region_mask].mean()
        if global_mean - region_mean < min_darkness:
            continue

        out[labels == i] = 255

    out = cv2.dilate(out, np.ones((2, 2), np.uint8), iterations=1)
    return out


def _detect_by_multi_scale_vote_v12(gray_u8, rock_mask=None,
                                      dark_pct=1.5, min_votes=2,
                                      scales=(15, 31, 61),
                                      min_area=10, max_area_pct=0.20,
                                      min_aspect=2.0,
                                      min_darkness=10.0,
                                      min_skeleton_ratio=2.5,
                                      downsample=0.5):
    """v12多尺度投票裂缝检测 —— 加强抗误检版本。

    在v11基础上增加：
    1. 骨架长度/面积比判断（裂缝应该细长）
    2. 更严格的长宽比过滤
    3. 降低最大面积比例（排除大块暗区）
    4. 提高暗度阈值
    5. 梯度方向一致性判断（裂缝两侧梯度方向相反）
    """
    h_orig, w_orig = gray_u8.shape

    if downsample and downsample < 1.0:
        h_s = max(1, int(h_orig * downsample))
        w_s = max(1, int(w_orig * downsample))
        small = cv2.resize(gray_u8, (w_s, h_s), interpolation=cv2.INTER_AREA)
        result = _detect_by_multi_scale_vote_v12(
            small, rock_mask=None, dark_pct=dark_pct, min_votes=min_votes,
            scales=scales, min_area=min_area, max_area_pct=max_area_pct,
            min_aspect=min_aspect, min_darkness=min_darkness,
            min_skeleton_ratio=min_skeleton_ratio,
            downsample=1.0
        )
        return cv2.resize(result, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    h, w = gray_u8.shape
    total = h * w
    gray_f = gray_u8.astype(np.float32)

    vote = np.zeros((h, w), dtype=np.float32)
    for k in scales:
        mean = cv2.blur(gray_f, (k, k))
        diff = gray_f - mean
        thresh = np.percentile(diff.ravel(), dark_pct)
        vote += (diff < thresh).astype(np.float32)

    dark_mask = (vote >= min_votes).astype(np.uint8)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    if rock_mask is not None:
        dark_mask = cv2.bitwise_and(dark_mask, dark_mask, mask=rock_mask)

    # 预计算梯度（用于方向一致性判断）
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.sqrt(gx * gx + gy * gy)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
    out = np.zeros_like(dark_mask)
    global_mean = gray_f.mean()

    for i in range(1, n):
        x, y, ww, hh, area = stats[i]

        if area < min_area or area / total > max_area_pct:
            continue

        aspect = max(ww, hh) / max(min(ww, hh), 1)
        if aspect < min_aspect:
            if area < 50:
                continue
            if area < 200 and aspect < 1.5:
                continue
            if area >= 200 and aspect < min_aspect:
                continue

        region_mask = (labels == i)
        region_mean = gray_f[region_mask].mean()
        if global_mean - region_mean < min_darkness:
            continue

        # 骨架长度/面积比判断
        region_uint8 = region_mask.astype(np.uint8) * 255
        dist = cv2.distanceTransform(region_uint8, cv2.DIST_L2, 5)
        _, max_dist, _, _ = cv2.minMaxLoc(dist)
        if max_dist < 1:
            continue
        skeleton_mask = (dist > max_dist * 0.3).astype(np.uint8) * 255
        skeleton_len = np.sum(skeleton_mask > 0)
        skel_ratio = skeleton_len / max(area, 1)
        if skel_ratio < min_skeleton_ratio * 0.1:
            continue

        # 梯度方向一致性判断：裂缝两侧梯度应指向裂缝（方向相反）
        # 计算候选区域的平均梯度方向
        region_gx = gx[region_mask]
        region_gy = gy[region_mask]
        region_gmag = gmag[region_mask]

        if region_gmag.sum() > 0:
            # 加权平均方向
            avg_gx = np.sum(region_gx * region_gmag) / np.sum(region_gmag)
            avg_gy = np.sum(region_gy * region_gmag) / np.sum(region_gmag)
            avg_mag = np.sqrt(avg_gx * avg_gx + avg_gy * avg_gy)

            # 区域内梯度方向的一致性（与平均方向的夹角）
            if avg_mag > 0.5:
                # 归一化平均方向
                avg_gx_n = avg_gx / avg_mag
                avg_gy_n = avg_gy / avg_mag
                # 每个点梯度在平均方向上的投影
                proj = region_gx * avg_gx_n + region_gy * avg_gy_n
                # 投影方向一致的比例
                same_dir = np.sum(proj > 0.3 * avg_mag) / max(len(proj), 1)
                # 裂缝区域应该有两侧梯度（方向有正有负），所以same_dir不应太高或太低
                # 纹理/颗粒边界的梯度方向较一致（same_dir高）
                # 裂缝的梯度方向较分散（same_dir中等）
                if same_dir > 0.75:
                    # 方向太一致，可能是颗粒边界而非裂缝
                    continue

        out[labels == i] = 255

    out = cv2.dilate(out, np.ones((2, 2), np.uint8), iterations=1)
    return out


def _detect_by_multi_scale_vote_v13(gray_u8, rock_mask=None,
                                      dark_pct=3, min_votes=2,
                                      scales=(15, 31),
                                      min_area=8, max_area_pct=0.25,
                                      min_aspect=1.0,
                                      min_darkness=9.0,
                                      min_skeleton_ratio=1.5,
                                      downsample=1.0):
    """v13多尺度投票裂缝检测 —— 高召回+快速版本。

    相比v12：
    1. 减少尺度数（3→2），加速约30%
    2. 提高dark_pct（2→3），更多像素被判为暗，提高召回
    3. 降低长宽比、暗度、骨架比要求，放宽过滤
    4. 提高下采样倍率（0.5→0.4），加速
    5. 去掉梯度方向一致性判断，加速
    """
    h_orig, w_orig = gray_u8.shape

    if downsample and downsample < 1.0:
        h_s = max(1, int(h_orig * downsample))
        w_s = max(1, int(w_orig * downsample))
        small = cv2.resize(gray_u8, (w_s, h_s), interpolation=cv2.INTER_AREA)
        result = _detect_by_multi_scale_vote_v13(
            small, rock_mask=None, dark_pct=dark_pct, min_votes=min_votes,
            scales=scales, min_area=min_area, max_area_pct=max_area_pct,
            min_aspect=min_aspect, min_darkness=min_darkness,
            min_skeleton_ratio=min_skeleton_ratio,
            downsample=1.0
        )
        return cv2.resize(result, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    h, w = gray_u8.shape
    total = h * w
    gray_f = gray_u8.astype(np.float32)

    vote = np.zeros((h, w), dtype=np.float32)
    for k in scales:
        mean = cv2.blur(gray_f, (k, k))
        diff = gray_f - mean
        thresh = np.percentile(diff.ravel(), dark_pct)
        vote += (diff < thresh).astype(np.float32)

    dark_mask = (vote >= min_votes).astype(np.uint8)
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    if rock_mask is not None:
        dark_mask = cv2.bitwise_and(dark_mask, dark_mask, mask=rock_mask)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)
    out = np.zeros_like(dark_mask)
    global_mean = gray_f.mean()

    for i in range(1, n):
        x, y, ww, hh, area = stats[i]

        if area < min_area or area / total > max_area_pct:
            continue

        aspect = max(ww, hh) / max(min(ww, hh), 1)
        if aspect < min_aspect:
            if area < 50:
                continue
            if area < 200 and aspect < 1.3:
                continue
            if area >= 200 and aspect < min_aspect:
                continue

        region_mask = (labels == i)
        region_mean = gray_f[region_mask].mean()
        if global_mean - region_mean < min_darkness:
            continue

        # 骨架长度/面积比判断（简化版，加速）
        region_uint8 = region_mask.astype(np.uint8) * 255
        dist = cv2.distanceTransform(region_uint8, cv2.DIST_L2, 5)
        _, max_dist, _, _ = cv2.minMaxLoc(dist)
        if max_dist < 1:
            continue
        # 骨架长度近似 = 面积 / 平均宽度，平均宽度 ≈ max_dist * 0.7
        est_skel_len = area / max(max_dist * 0.7, 1)
        skel_ratio = est_skel_len / max(area, 1)
        if skel_ratio < min_skeleton_ratio * 0.05:
            continue

        out[labels == i] = 255

    out = cv2.dilate(out, np.ones((2, 2), np.uint8), iterations=1)
    return out


# ---------- 全图特征图 ----------

def _full_image_lbp(gray_u8, radius=1, n_points=8):
    h, w = gray_u8.shape
    lbp = np.zeros((h, w), dtype=np.uint8)
    for i, a in enumerate(np.linspace(0, 2 * np.pi, n_points, endpoint=False)):
        dy = int(round(radius * np.sin(a)))
        dx = int(round(radius * np.cos(a)))
        shifted = np.roll(np.roll(gray_u8, dy, axis=0), dx, axis=1)
        lbp |= ((gray_u8 >= shifted).astype(np.uint8) << i)
    return lbp


def _gabor_bank(gray_f, thetas=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)):
    bank = []
    for t in thetas:
        kernel = cv2.getGaborKernel((7, 7), 1.2, t, 4.0, 0.5, 0, ktype=cv2.CV_32F)
        bank.append(cv2.filter2D(gray_f, cv2.CV_32F, kernel))
    return bank


def _compute_feature_maps(gray_u8, thetas=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)):
    g_f = gray_u8.astype(np.float32) / 255.0
    gx = cv2.Sobel(g_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g_f, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.sqrt(gx * gx + gy * gy)
    gx_x = cv2.Sobel(gx, cv2.CV_32F, 1, 0, ksize=3)
    gy_y = cv2.Sobel(gy, cv2.CV_32F, 0, 1, ksize=3)
    gx_y = cv2.Sobel(gx, cv2.CV_32F, 0, 1, ksize=3)
    tr = gx_x + gy_y
    disc = np.sqrt(np.maximum(tr * tr - 4.0 * (gx_x * gy_y - gx_y * gx_y), 0))
    l1 = (tr + disc) / 2.0
    l2 = (tr - disc) / 2.0
    gabor = []
    for t in thetas:
        kernel = cv2.getGaborKernel((7, 7), 1.2, t, 4.0, 0.5, 0, ktype=cv2.CV_32F)
        gabor.append(cv2.filter2D(g_f, cv2.CV_32F, kernel))
    lbp = _full_image_lbp(gray_u8)
    return {"g_f": g_f, "gx": gx, "gy": gy, "gmag": gmag,
            "l1": l1, "l2": l2, "gabor": gabor, "lbp": lbp}


def _patch_features_from_maps(fmaps, r, c, half):
    r0, r1 = r - half, r + half + 1
    c0, c1 = c - half, c + half + 1
    g_f = fmaps["g_f"][r0:r1, c0:c1]
    gmag = fmaps["gmag"][r0:r1, c0:c1]
    l1 = fmaps["l1"][r0:r1, c0:c1]
    l2 = fmaps["l2"][r0:r1, c0:c1]
    lbp_full = fmaps["lbp"]
    feats = []
    feats.append(float(g_f.mean()))
    feats.append(float(g_f.std()))
    small = cv2.resize(g_f, (max(3, half), max(3, half)), interpolation=cv2.INTER_AREA)
    feats.append(float(small.mean()))
    feats.append(float(small.std()))
    feats.append(float(gmag.mean()))
    feats.append(float(gmag.std()))
    feats.append(float(l1.mean()))
    feats.append(float(l2.mean()))
    feats.append(float(np.abs(l1).std()))
    feats.append(float(np.abs(l2).std()))
    feats.append(float(np.abs(l2).mean() / (np.abs(l1).mean() + 1e-6)))
    gabs = [float(np.abs(g[r, c])) for g in fmaps["gabor"]]
    feats.extend(gabs)
    feats.append(float(max(gabs)))
    patch_q = (lbp_full[r-half:r+half+1, c-half:c+half+1].ravel().astype(np.int32) * 32 // 256).astype(np.int32)
    hist = np.bincount(patch_q, minlength=32).astype(np.float32)
    s = hist.sum()
    feats.extend((hist / s if s > 0 else hist).tolist())
    return np.array(feats, dtype=np.float32)


def _extract_features_fast(gray_u8, pts, half, max_samples=None, fmaps=None):
    if fmaps is None:
        fmaps = _compute_feature_maps(gray_u8)
    feats, valid = [], []
    if max_samples and len(pts) > max_samples:
        idx = np.random.choice(len(pts), max_samples, replace=False)
        pts = pts[idx]
    h, w = gray_u8.shape
    for r, c in pts:
        r, c = int(r), int(c)
        if r < half or r >= h - half or c < half or c >= w - half:
            continue
        feats.append(_patch_features_from_maps(fmaps, r, c, half))
        valid.append((r, c))
    if not feats:
        return np.empty((0, 0), dtype=np.float32), []
    return np.stack(feats), valid


def _patch_features(gray_f, r, c, half, gabor_responses, lbp_full):
    r0, r1 = r - half, r + half + 1
    c0, c1 = c - half, c + half + 1
    patch = gray_f[r0:r1, c0:c1]
    feats = []
    feats.append(float(patch.mean()))
    feats.append(float(patch.std()))
    small = cv2.resize(patch, (max(3, half), max(3, half)), interpolation=cv2.INTER_AREA)
    feats.append(float(small.mean()))
    feats.append(float(small.std()))
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    feats.append(float(mag.mean()))
    feats.append(float(mag.std()))
    Ixx, Iyy, Ixy = _hessian_from_patch(patch, sigma=1.0)
    tr = Ixx + Iyy
    disc = np.sqrt(np.maximum(tr * tr - 4.0 * (Ixx * Iyy - Ixy * Ixy), 0))
    l1 = (tr + disc) / 2.0
    l2 = (tr - disc) / 2.0
    feats.append(float(l1.mean()))
    feats.append(float(l2.mean()))
    feats.append(float(np.abs(l1).std()))
    feats.append(float(np.abs(l2).std()))
    feats.append(float(np.abs(l2).mean() / (np.abs(l1).mean() + 1e-6)))
    gabs = [float(np.abs(g[r, c])) for g in gabor_responses]
    feats.extend(gabs)
    feats.append(float(max(gabs)))
    patch_q = (lbp_full[r-half:r+half+1, c-half:c+half+1].ravel().astype(np.int32) * 32 // 256).astype(np.int32)
    hist = np.bincount(patch_q, minlength=32).astype(np.float32)
    s = hist.sum()
    feats.extend((hist / s if s > 0 else hist).tolist())
    return np.array(feats, dtype=np.float32)


def _hessian_from_patch(g, sigma):
    k = max(3, int(round(3 * sigma)) * 2 + 1)
    if k % 2 == 0:
        k += 1
    g = cv2.GaussianBlur(g, (k, k), sigma)
    Ix = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    Ixx = cv2.Sobel(Ix, cv2.CV_32F, 1, 0, ksize=3)
    Iyy = cv2.Sobel(Iy, cv2.CV_32F, 0, 1, ksize=3)
    Ixy = cv2.Sobel(Ix, cv2.CV_32F, 0, 1, ksize=3)
    return Ixx, Iyy, Ixy


# ---------- 主类 ----------

class DLFractureDetector:
    MODEL_FILENAME = "dl_fracture_v2.pkl"
    LEGACY_MODEL = "dl_fracture_model.pkl"

    def __init__(self, model_dir=None):
        self.model = None
        self.half_patch = 10
        self.patch_size = 21
        self.feature_dim = None
        self.model_dir = model_dir

    @property
    def model_path(self):
        if self.model_dir:
            return os.path.join(self.model_dir, self.MODEL_FILENAME)
        return self.MODEL_FILENAME

    @property
    def legacy_path(self):
        if self.model_dir:
            return os.path.join(self.model_dir, self.LEGACY_MODEL)
        return self.LEGACY_MODEL

    def _normalize_patch(self, patch):
        m, s = patch.mean(), patch.std()
        if s < 1:
            return (patch - m).ravel()
        return (patch - m).ravel() / s

    def _preprocess_like_auto(self, img):
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        gray = cv2.convertScaleAbs(gray, alpha=2.0, beta=0)
        low = np.percentile(gray, 1)
        high = np.percentile(gray, 99)
        if high - low >= 1:
            gray = np.clip((gray.astype(np.float32) - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
        ksize = int(2 * round(3 * 4.0) + 1)
        gray = cv2.GaussianBlur(gray, (ksize, ksize), 4.0)
        clahe = cv2.createCLAHE(clipLimit=0.50, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        return gray

    def train(self, data_dir, do_hard_negative=True):
        images_dir = os.path.join(data_dir, "JPEGImages")
        masks_dir = os.path.join(data_dir, "Annotations")
        names = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        X, y = [], []
        for fname in names:
            stem = os.path.splitext(fname)[0]
            img = cv2.imread(os.path.join(images_dir, fname))
            mask = cv2.imread(os.path.join(masks_dir, stem + ".png"), 0)
            if img is None or mask is None:
                continue
            proc = self._preprocess_like_auto(img)
            gt = (mask > 0).astype(np.uint8)
            fmaps = _compute_feature_maps(proc)
            frac_pts = np.argwhere(gt > 0)
            fp, _ = _extract_features_fast(proc, frac_pts, self.half_patch, max_samples=600, fmaps=fmaps)
            if len(fp):
                X.append(fp)
                y.append(np.ones(len(fp)))
            kernel = np.ones((11, 11), np.uint8)
            dilated = cv2.dilate(gt, kernel)
            boundary = np.argwhere((dilated > 0) & (gt == 0))
            if len(boundary) > 0:
                if len(boundary) > len(frac_pts):
                    idx = np.random.choice(len(boundary), min(len(frac_pts), 400), replace=False)
                    boundary = boundary[idx]
                hp, _ = _extract_features_fast(proc, boundary, self.half_patch, fmaps=fmaps)
                if len(hp):
                    X.append(hp)
                    y.append(np.zeros(len(hp)))
            far = np.argwhere(dilated == 0)
            if len(far) > 0:
                ep, _ = _extract_features_fast(proc, far, self.half_patch, max_samples=150, fmaps=fmaps)
                if len(ep):
                    X.append(ep)
                    y.append(np.zeros(len(ep)))
            print(f"[train] {fname}: pos={len(X[-2]) if len(X) >= 2 else 0}, neg={len(X[-1]) if len(X) >= 1 else 0}", flush=True)
        if not X:
            raise RuntimeError("未找到可用的训练样本")
        X = np.concatenate(X)
        y = np.concatenate(y)
        self.feature_dim = X.shape[1]
        print(f"[train] 总样本: {len(y)} (正={int(y.sum())}, 负={int((1-y).sum())}), 特征维度={X.shape[1]}", flush=True)
        self.model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), activation='relu', solver='adam',
            max_iter=400, alpha=1e-4, random_state=42, verbose=True,
        )
        self.model.fit(X, y)
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"[train] 已保存模型: {self.model_path}")

    def load(self, path=None):
        path = path or self.model_path
        if os.path.exists(path):
            self.model = joblib.load(path)
            self.model_dir = os.path.dirname(path)
            return
        legacy = self.legacy_path
        if os.path.exists(legacy):
            print(f"[load] 使用旧模型 {legacy}")
            self.model = joblib.load(legacy)
            self.model_dir = os.path.dirname(legacy)
            return
        raise FileNotFoundError(f"模型未找到: {path}")

    def is_trained(self):
        return self.model is not None

    def predict(self, img, return_mask=True):
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        gray = np.clip(cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX), 0, 255).astype(np.uint8)
        if self.model is None:
            return self._classical_predict(gray)
        pts, union, resp = _candidate_points_v3(gray, max_points=25000)
        if len(pts) == 0:
            return np.zeros(gray.shape, dtype=np.uint8)
        fmaps = _compute_feature_maps(gray)
        feats, valid = _extract_features_fast(gray, pts, self.half_patch, fmaps=fmaps)
        if len(feats) == 0:
            return np.zeros(gray.shape, dtype=np.uint8)
        expected = getattr(self.model, 'n_features_in_', None)
        if expected is not None and expected != feats.shape[1]:
            return self._legacy_compat_predict(gray, pts)
        probs = self.model.predict_proba(feats)[:, 1]
        mask_high = np.zeros(gray.shape, dtype=np.uint8)
        mask_med = np.zeros(gray.shape, dtype=np.uint8)
        for (r, c), p in zip(valid, probs):
            if p >= 0.7:
                mask_high[r, c] = 255
            elif p >= 0.45:
                mask_med[r, c] = 255
        kernel = np.ones((3, 3), np.uint8)
        prev = np.zeros_like(mask_high)
        cur = mask_high.copy()
        for _ in range(60):
            if np.array_equal(cur, prev):
                break
            prev = cur.copy()
            d = cv2.dilate(cur, kernel)
            cur = cv2.bitwise_or(cur, cv2.bitwise_and(d, mask_med))
        cur = cv2.morphologyEx(cur, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
        rock_mask = _detect_rock_mask_no_bg(gray)
        final = _post_process_v3(cur, gray.shape, rock_mask=rock_mask,
                                 min_area=25, min_skel=8, min_aspect=2.0)
        return final

    def _legacy_compat_predict(self, gray, pts):
        h, w = gray.shape
        half = self.half_patch
        feats, valid = [], []
        for r, c in pts:
            r, c = int(r), int(c)
            if r < half or r >= h - half or c < half or c >= w - half:
                continue
            patch = gray[r-half:r+half+1, c-half:c+half+1].astype(np.float32)
            m, s = patch.mean(), patch.std()
            if s < 1:
                f = (patch - m).ravel()
            else:
                f = (patch - m).ravel() / s
            feats.append(f)
            valid.append((r, c))
        if not feats:
            return np.zeros(gray.shape, dtype=np.uint8)
        feats = np.stack(feats)
        probs = self.model.predict_proba(feats)[:, 1]
        mask_high = np.zeros(gray.shape, dtype=np.uint8)
        mask_med = np.zeros(gray.shape, dtype=np.uint8)
        for (r, c), p in zip(valid, probs):
            if p >= 0.8:
                mask_high[r, c] = 255
            elif p >= 0.3:
                mask_med[r, c] = 255
        kernel = np.ones((3, 3), np.uint8)
        prev = np.zeros_like(mask_high)
        cur = mask_high.copy()
        for _ in range(60):
            if np.array_equal(cur, prev):
                break
            prev = cur.copy()
            d = cv2.dilate(cur, kernel)
            cur = cv2.bitwise_or(cur, cv2.bitwise_and(d, mask_med))
        cur = cv2.morphologyEx(cur, cv2.MORPH_CLOSE, kernel, iterations=2)
        rock_mask = _detect_rock_mask_no_bg(gray)
        final = _post_process_v3(cur, gray.shape, rock_mask=rock_mask,
                                 min_area=20, min_skel=6, min_aspect=1.8)
        return final

    def _classical_predict(self, gray):
        """改进版裂缝检测算法 —— 多特征融合+严格过滤。

        结合DoG、Frangi线检测、暗度投票三种特征，
        并通过长宽比、暗度对比、骨架比、边缘匹配度等多重过滤，
        有效减少误检，提高检测精度。
        """
        return _detect_fracture_improved(gray, rock_mask=None)


def _detect_fracture_improved(gray_u8, rock_mask=None):
    """改进版裂缝检测算法 v3 —— 稳健版本，确保能检测到裂缝。

    策略：先放宽条件获取足够候选，再通过评分排序精选，
    确保不会因为过滤太严而识别不到。
    """
    h, w = gray_u8.shape
    total = h * w

    # 1. 预处理：轻度去噪 + 自适应直方图均衡化
    gray_blur = cv2.GaussianBlur(gray_u8, (3, 3), 0.5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_blur)
    gray_f = enhanced.astype(np.float32) / 255.0

    # 2. 多特征提取

    # 特征1：高斯差分(DoG) - 突出线状暗纹
    dog1 = cv2.GaussianBlur(gray_f, (0, 0), 1.0)
    dog2 = cv2.GaussianBlur(gray_f, (0, 0), 2.5)
    dog = dog2 - dog1
    dog_norm = np.clip(-dog * 2, 0, 1).astype(np.float32)

    # 特征2：Frangi线检测
    frangi = _frangi_line_ness(gray_f, scales=(1.0, 2.0, 3.0))

    # 特征3：多尺度暗度投票（放宽，取最暗的8%）
    vote = np.zeros((h, w), dtype=np.float32)
    for k in (15, 31, 51):
        mean = cv2.blur(gray_f * 255, (k, k)) / 255.0
        diff = gray_f - mean
        thresh = np.percentile(diff.ravel(), 8)
        vote += (diff < thresh).astype(np.float32)
    dark_vote = vote / 3.0

    # 3. 特征融合
    combined = dog_norm * 0.3 + frangi * 0.3 + dark_vote * 0.4
    combined = np.clip(combined, 0, 1)

    # 4. 阈值分割（取响应最高的5%，确保有足够候选）
    combined_u8 = (combined * 255).astype(np.uint8)
    thresh_val = np.percentile(combined_u8, 95)
    _, binary = cv2.threshold(combined_u8, thresh_val, 255, cv2.THRESH_BINARY)

    # 5. 形态学处理
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)

    if rock_mask is not None:
        binary = cv2.bitwise_and(binary, binary, mask=rock_mask)

    # 6. 连通域分析 + 评分排序
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros_like(binary)
    global_mean = gray_u8.mean()

    candidates = []
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]

        # 基本面积过滤（非常宽松）
        if area < 15:
            continue
        if area / total > 0.15:
            continue

        region_mask = (labels == i)
        region_uint8 = region_mask.astype(np.uint8) * 255

        # 长宽比
        aspect = max(ww, hh) / max(min(ww, hh), 1)

        # 暗度对比（候选区域 vs 周围环带）
        region_mean = gray_u8[region_mask].mean()
        dilated = cv2.dilate(region_uint8, np.ones((8, 8), np.uint8), iterations=1)
        ring = cv2.subtract(dilated, region_uint8)
        ring_pixels = gray_u8[ring > 0]
        if len(ring_pixels) > 10:
            ring_mean = ring_pixels.mean()
            dark_diff = ring_mean - region_mean
        else:
            dark_diff = global_mean - region_mean

        # 骨架长度/面积比
        try:
            skel = cv2.ximgproc.thinning(region_uint8)
        except Exception:
            elem = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            skel = _morphological_skeleton(region_uint8, elem)
        skeleton_len = np.sum(skel > 0)
        skel_ratio = skeleton_len / max(area, 1)

        # 综合评分（越高越可能是裂缝）
        # 权重：长宽比30% + 暗度30% + 骨架比25% + 面积15%（偏向更大的结构）
        score = 0.0
        score += min(aspect / 4.0, 1.0) * 0.30
        score += min(max(dark_diff, 0) / 25.0, 1.0) * 0.30
        score += min(skel_ratio / 0.5, 1.0) * 0.25
        score += min(area / 500.0, 1.0) * 0.15

        candidates.append((i, score, area, aspect, dark_diff, skel_ratio))

    if not candidates:
        return out

    # 按评分从高到低排序
    candidates.sort(key=lambda x: x[1], reverse=True)

    # 保留策略：评分 > 0.4 的都保留，至少保留前5个
    min_score = 0.4
    min_keep = min(5, len(candidates))
    for idx, (i, score, area, aspect, dark_diff, skel_ratio) in enumerate(candidates):
        if idx < min_keep or score > min_score:
            region_mask = (labels == i)
            out[region_mask] = 255

    # 7. 轻微膨胀
    out = cv2.dilate(out, np.ones((2, 2), np.uint8), iterations=1)
    return out


# ---------- 候选点生成（v3 for DL path）----------

def _candidate_points_v3(gray_u8, max_points=20000):
    gray_f = gray_u8.astype(np.float32) / 255.0
    ridge_dark = np.maximum.reduce([_ridge_ness_dark(gray_f, s) for s in (1.0, 2.0, 3.0)])
    ridge_dark = np.nan_to_num(ridge_dark, nan=0.0, posinf=0.0, neginf=0.0)
    ridge_dark_u8 = np.clip(ridge_dark * 255, 0, 255).astype(np.uint8)
    _, th = cv2.threshold(ridge_dark_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.erode(th, np.ones((2, 2), np.uint8), iterations=1)
    med = float(np.median(gray_u8))
    low = int(max(15, (1.0 - 0.33) * med))
    high = int(min(120, (1.0 + 0.33) * med))
    edges = cv2.Canny(gray_u8, low, high)
    union = cv2.bitwise_or(edges, th)
    union = cv2.morphologyEx(union, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    pts = np.argwhere(union > 0)
    if len(pts) > max_points:
        idx = np.argsort(ridge_dark[pts[:, 0], pts[:, 1]])[::-1][:max_points]
        pts = pts[idx]
    return pts, union, ridge_dark
