"""
线性体(断裂/线性构造)自动提取

把"断裂解译"从可视化(人看图)推进到机器可读数据。核心流程:
  多方位山体阴影(光照无关) → Canny 边缘 → 坡度门控 → 骨架化
  → 概率霍夫提线 → 长度过滤 → 矢量/密度/距断裂距离 + 玫瑰图统计

输出供下游(geo-analyser 地形归一化与构造加权、geo-exploration 构造控矿因子、
geo-reporter 构造实证)直接消费。

诚实边界:自动提取天然含噪(水系、山脊会冒充断裂),已用坡度门控 + 长度过滤抑制,
但结果是**决策支持层**,非已验证构造地质结论。
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.ndimage import uniform_filter, distance_transform_edt
from loguru import logger

try:
    from skimage.feature import canny
    from skimage.morphology import skeletonize
    from skimage.transform import probabilistic_hough_line
    _SKIMAGE_OK = True
except Exception:  # 依赖缺失时优雅降级
    _SKIMAGE_OK = False


def _segment_strike_deg(p0, p1, pixel_size_m: Tuple[float, float]) -> float:
    """线段走向(0–180°,正北为0,顺时针)。p=(col,row),row 向南增。"""
    (x0, y0), (x1, y1) = p0, p1
    d_east = (x1 - x0) * pixel_size_m[0]
    d_north = -(y1 - y0) * pixel_size_m[1]
    strike = np.degrees(np.arctan2(d_east, d_north)) % 180.0
    return float(strike)


def _segment_length_m(p0, p1, pixel_size_m: Tuple[float, float]) -> float:
    (x0, y0), (x1, y1) = p0, p1
    dx = (x1 - x0) * pixel_size_m[0]
    dy = (y1 - y0) * pixel_size_m[1]
    return float(np.hypot(dx, dy))


def _pixel_to_lonlat(col: float, row: float, transform) -> Tuple[float, float]:
    """像素中心 (col,row) → (lon,lat),用 Affine 变换。"""
    lon, lat = transform * (col + 0.5, row + 0.5)
    return float(lon), float(lat)


def _score_segment(p0, p1, img, skel, slope, curvature, valid_mask, pixel_size_m):
    """
    为单条线段计算置信度 (0–1)。

    综合四个维度:
      1. 边缘强度:沿线段采样拉伸后图像的梯度幅值均值(高=边缘清晰)
      2. 骨架连续性:线段像元落在骨架上的比例(高=连贯的线性特征)
      3. 坡度信号:沿线段坡度均值(适度坡度更可能是构造而非平坦区噪声)
      4. 曲率一致性(可选):线段穿越区域曲率方向一致(正值/脊 或 负值/谷)时加分

    Returns:
        float: 置信度 [0, 1]
    """
    H, W = img.shape
    length_px = max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1]))
    n_samp = max(3, int(length_px))
    ts = np.linspace(0, 1, n_samp)

    edge_vals, skel_vals, slope_vals, curv_vals = [], [], [], []
    for t in ts:
        cc = int(round(p0[0] + t * (p1[0] - p0[0])))
        rr = int(round(p0[1] + t * (p1[1] - p0[1])))
        if 0 <= rr < H and 0 <= cc < W:
            # 边缘强度:取邻域梯度(3x3)的最大值
            r_lo, r_hi = max(0, rr - 1), min(H, rr + 2)
            c_lo, c_hi = max(0, cc - 1), min(W, cc + 2)
            patch = img[r_lo:r_hi, c_lo:c_hi]
            if patch.size >= 4:
                gy, gx = np.gradient(patch)
                edge_vals.append(float(np.hypot(gx, gy).max()))
            skel_vals.append(float(skel[rr, cc]))
            slope_vals.append(float(slope[rr, cc]))
            if curvature is not None:
                curv_vals.append(float(curvature[rr, cc]))

    if not edge_vals:
        return 0.5  # 无采样点,给默认值

    # 1. 边缘强度 (0–1):归一化到 [0,1],越高越好
    mean_edge = float(np.mean(edge_vals))
    edge_score = min(1.0, mean_edge / 0.5)  # 0.5 为典型强边缘梯度

    # 2. 骨架连续性 (0–1):线段经过的骨架像元比例
    skel_ratio = float(np.mean(skel_vals)) if skel_vals else 0.0
    skel_score = skel_ratio

    # 3. 坡度信号 (0–1):适度坡度(5–30°)加分,太平坦(<2°)扣分
    mean_slope = float(np.mean(slope_vals)) if slope_vals else 0.0
    if mean_slope < 2.0:
        slope_score = 0.3
    elif mean_slope < 30.0:
        slope_score = min(1.0, mean_slope / 15.0)
    else:
        slope_score = 0.7  # 极陡可能是滑坡而非构造

    # 4. 曲率一致性(可选):同号曲率占多数 → 一致性高
    curv_score = 0.5  # 默认
    if curv_vals:
        signs = [1 if v > 0 else (-1 if v < 0 else 0) for v in curv_vals]
        nonzero = [s for s in signs if s != 0]
        if len(nonzero) >= 3:
            same_sign_ratio = max(nonzero.count(1), nonzero.count(-1)) / len(nonzero)
            curv_score = 0.3 + 0.7 * same_sign_ratio
        elif nonzero:
            curv_score = 0.4

    # 加权综合 (边缘+骨架为主,坡度+曲率为辅)
    confidence = (0.30 * edge_score +
                  0.30 * skel_score +
                  0.20 * slope_score +
                  0.20 * curv_score)
    return float(np.clip(confidence, 0.0, 1.0))


def extract_lineaments(
    multidir_hillshade: np.ndarray,
    slope: np.ndarray,
    pixel_size_m: Tuple[float, float],
    transform,
    valid_mask: Optional[np.ndarray] = None,
    canny_sigma: float = 1.5,
    slope_gate_deg: float = 1.0,
    slope_gate_pct: float = 30.0,
    slope_gate_cap: float = 8.0,
    min_length_m: float = 300.0,
    density_window_m: float = 1000.0,
    rng_seed: Optional[int] = None,
) -> Dict:
    """
    从多方位山体阴影提取线性体。

    Args:
        multidir_hillshade: 多方位融合山体阴影 (H,W),值域[0,1](光照无关,消除方位盲区)
        slope: 坡度(度) (H,W),用于门控:只在有坡度处保留边缘,压制平坦区/水域噪声
        pixel_size_m: (x,y) 像元地面尺寸(米)
        transform: 裁剪后 DEM 的 Affine(像素→经纬度)
        valid_mask: 有效像元(True=有效);None 则按 isfinite 推断
        canny_sigma: Canny 高斯尺度
        slope_gate_deg: 坡度门控的*下限*(度);实际门控取该下限与坡度分位数的较大者
        slope_gate_pct: 坡度门控自适应分位(对地形起伏自适应:陡区只留强边缘,缓区放宽)
        min_length_m: 线段最小长度(米),短于此的剔除(抑制噪声)
        density_window_m: 断裂密度滑窗边长(米)
        rng_seed: 概率霍夫提线的随机种子;None=每次随机(历史行为),给定整数则结果可复现

    Returns:
        dict: {
          'segments': [{'p0':(lon,lat),'p1':(lon,lat),'strike_deg':float,'length_m':float}, ...],
          'mask': 线性体二值栅格 (H,W) bool,
          'distance_m': 距最近线性体距离栅格 (H,W) float32(米),
          'density': 断裂密度栅格 (H,W) float32(滑窗内线性体像元占比),
          'stats': {n_lineaments,total_length_km,density_mean,dominant_strikes_deg},
        }
    """
    H, W = multidir_hillshade.shape
    empty = {
        'segments': [], 'mask': np.zeros((H, W), bool),
        'distance_m': np.full((H, W), np.nan, np.float32),
        'density': np.zeros((H, W), np.float32),
        'stats': {'n_lineaments': 0, 'total_length_km': 0.0,
                  'density_mean': 0.0, 'dominant_strikes_deg': []},
    }
    if not _SKIMAGE_OK:
        logger.warning("scikit-image 不可用,跳过线性体提取(仅产出地形量算)")
        return empty

    if valid_mask is None:
        valid_mask = np.isfinite(multidir_hillshade) & np.isfinite(slope)

    # 多方位阴影动态范围很窄(常 0.6~0.7),需先拉伸到 [0,1],否则 Canny 的
    # 绝对阈值会过滤掉所有边缘。配合 use_quantiles 用梯度分位阈值,适配任意尺度。
    finite = multidir_hillshade[valid_mask & np.isfinite(multidir_hillshade)]
    if finite.size:
        lo, hi = np.percentile(finite, 2), np.percentile(finite, 98)
    else:
        lo, hi = 0.0, 1.0
    img = np.clip((np.nan_to_num(multidir_hillshade, nan=lo) - lo) / (hi - lo + 1e-9), 0, 1)
    # Canny 边缘检测(在光照无关的多方位阴影上),分位阈值
    edges = canny(img.astype(np.float64), sigma=canny_sigma, mask=valid_mask,
                  use_quantiles=True, low_threshold=0.8, high_threshold=0.92)
    # 坡度门控:断裂为地形突变,只在有坡度处保留边缘以压制平坦区噪声。
    # 阈值需双向自适应且**设上限**:
    #   - 缓区(如小而平 AOI,坡度<阈值):用低分位下探,避免一刀切掉所有边缘;
    #   - 陡区(如山区,坡度普遍很大):必须封顶,否则高分位会把真实断裂边缘也滤掉(0 条)。
    svals = slope[valid_mask & np.isfinite(slope)]
    if svals.size:
        gate = min(slope_gate_cap, max(slope_gate_deg, float(np.percentile(svals, slope_gate_pct))))
    else:
        gate = slope_gate_deg
    edges &= (np.nan_to_num(slope, nan=0.0) >= gate)
    if not edges.any():
        return empty

    skel = skeletonize(edges)

    # 概率霍夫提线(线段端点为像素坐标 (col,row));阈值随影像尺寸自适应,
    # 小 AOI(像素少)用更低的投票阈值,否则短而真实的线性体会被漏检。
    min_len_px = max(3, int(min_length_m / max(pixel_size_m[0], pixel_size_m[1])))
    hough_thr = int(np.clip(min(H, W) // 4, 5, 10))
    # rng=None 时 skimage 用全新熵源(np.random.seed 也无法固定),故须显式传 Generator
    _rng = np.random.default_rng(rng_seed) if rng_seed is not None else None
    lines = probabilistic_hough_line(
        skel, threshold=hough_thr, line_length=min_len_px, line_gap=3, rng=_rng,
    )

    # 曲率(用于水系假阳性判别:河谷=负曲率/凹形)
    curvature = None
    if valid_mask is not None:
        try:
            from core.terrain_utils import TerrainProcessor
            # 从 slope 估算曲率方向(简化:用坡度的拉普拉斯近似)
            # 负值=凹(河谷),正值=凸(山脊)
            slope_filled = np.nan_to_num(slope, nan=0.0)
            laplacian = np.zeros_like(slope_filled)
            laplacian[1:-1, 1:-1] = (
                slope_filled[:-2, 1:-1] + slope_filled[2:, 1:-1] +
                slope_filled[1:-1, :-2] + slope_filled[1:-1, 2:] -
                4 * slope_filled[1:-1, 1:-1]
            )
            curvature = laplacian
        except Exception:
            pass

    segments = []
    mask = np.zeros((H, W), bool)
    for (p0, p1) in lines:
        length_m = _segment_length_m(p0, p1, pixel_size_m)
        if length_m < min_length_m:
            continue
        strike = _segment_strike_deg(p0, p1, pixel_size_m)
        lon0, lat0 = _pixel_to_lonlat(p0[0], p0[1], transform)
        lon1, lat1 = _pixel_to_lonlat(p1[0], p1[1], transform)
        # ---- 置信度评分 (0-1) ----
        confidence = _score_segment(p0, p1, img, skel, slope, curvature,
                                    valid_mask, pixel_size_m)

        # ---- 水系假阳性过滤 ----
        is_valley = False
        if curvature is not None:
            n_samp = max(3, int(length_m / max(pixel_size_m) / 5))
            ts = np.linspace(0.1, 0.9, n_samp)
            curv_vals, slope_vals = [], []
            for t in ts:
                cc = int(round(p0[0] + t * (p1[0] - p0[0])))
                rr = int(round(p0[1] + t * (p1[1] - p0[1])))
                if 0 <= rr < H and 0 <= cc < W:
                    curv_vals.append(curvature[rr, cc])
                    slope_vals.append(slope[rr, cc])
            if curv_vals:
                mean_curv = float(np.mean(curv_vals))
                mean_slope = float(np.mean(slope_vals))
                # 河谷:显著凹曲率 + 低-中坡度
                if mean_curv < -0.3 and mean_slope < 8.0:
                    is_valley = True
                    confidence *= 0.4

        segments.append({'p0': (lon0, lat0), 'p1': (lon1, lat1),
                         'strike_deg': strike, 'length_m': length_m,
                         'confidence': round(float(confidence), 3),
                         'is_valley_candidate': is_valley})
        # 在 mask 上栅格化该线段(Bresenham 近似)
        n = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1]))) + 1
        cols = np.linspace(p0[0], p1[0], n).astype(int).clip(0, W - 1)
        rows = np.linspace(p0[1], p1[1], n).astype(int).clip(0, H - 1)
        mask[rows, cols] = True

    if not segments:
        return empty

    # 距断裂距离(米):各向异性 EDT,sampling 用像元米尺寸
    distance_m = distance_transform_edt(
        ~mask, sampling=(pixel_size_m[1], pixel_size_m[0]),
    ).astype(np.float32)
    distance_m[~valid_mask] = np.nan

    # 断裂密度:滑窗内线性体像元占比
    win_px = max(3, int(density_window_m / max(pixel_size_m[0], pixel_size_m[1])))
    density = uniform_filter(mask.astype(np.float32), size=win_px).astype(np.float32)
    density[~valid_mask] = 0.0

    # 主构造方向:按长度加权的走向直方图,取峰值
    strikes = np.array([s['strike_deg'] for s in segments])
    lengths = np.array([s['length_m'] for s in segments])
    hist, edges_b = np.histogram(strikes, bins=18, range=(0, 180), weights=lengths)
    order = np.argsort(hist)[::-1]
    centers = (edges_b[:-1] + edges_b[1:]) / 2
    dominant = [float(centers[i]) for i in order[:3] if hist[i] > 0]

    stats = {
        'n_lineaments': len(segments),
        'total_length_km': float(lengths.sum() / 1000.0),
        'density_mean': float(np.nanmean(density[valid_mask])) if valid_mask.any() else 0.0,
        'dominant_strikes_deg': dominant,
    }
    logger.info(f"线性体提取: {len(segments)} 条, 总长 {stats['total_length_km']:.1f} km, "
                f"主方向 {['%.0f°'%d for d in dominant]}")
    return {'segments': segments, 'mask': mask, 'distance_m': distance_m,
            'density': density, 'stats': stats}


def write_lineaments_geojson(segments: List[Dict], path: str, crs: str = "EPSG:4326"):
    """线段 → GeoJSON(用 shapely,不引入 geopandas)。"""
    features = []
    for i, s in enumerate(segments):
        features.append({
            'type': 'Feature',
            'geometry': {'type': 'LineString', 'coordinates': [list(s['p0']), list(s['p1'])]},
            'properties': {'id': i, 'strike_deg': round(s['strike_deg'], 1),
                           'length_m': round(s['length_m'], 1)},
        })
    fc = {'type': 'FeatureCollection',
          'crs': {'type': 'name', 'properties': {'name': crs}},
          'features': features}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(fc, f, ensure_ascii=False)


def plot_rose_diagram(segments: List[Dict], path: str, title: str = "构造方向玫瑰图"):
    """按长度加权的走向玫瑰图(双向对称,0-180°→镜像到0-360°)。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not segments:
        return
    strikes = np.array([s['strike_deg'] for s in segments])
    lengths = np.array([s['length_m'] for s in segments])
    nb = 36
    bins = np.linspace(0, 180, nb // 2 + 1)
    h, _ = np.histogram(strikes, bins=bins, weights=lengths)
    h2 = np.concatenate([h, h])  # 双向对称
    theta = np.deg2rad(np.linspace(0, 360, nb, endpoint=False) + (360 / nb) / 2)
    width = np.deg2rad(360 / nb)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.bar(theta, h2, width=width, color='#c0392b', edgecolor='k', alpha=0.7)
    ax.set_yticklabels([])
    ax.set_title(title, fontsize=11, pad=12)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
