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
    lines = probabilistic_hough_line(
        skel, threshold=hough_thr, line_length=min_len_px, line_gap=3,
    )

    segments = []
    mask = np.zeros((H, W), bool)
    for (p0, p1) in lines:
        length_m = _segment_length_m(p0, p1, pixel_size_m)
        if length_m < min_length_m:
            continue
        strike = _segment_strike_deg(p0, p1, pixel_size_m)
        lon0, lat0 = _pixel_to_lonlat(p0[0], p0[1], transform)
        lon1, lat1 = _pixel_to_lonlat(p1[0], p1[1], transform)
        segments.append({'p0': (lon0, lat0), 'p1': (lon1, lat1),
                         'strike_deg': strike, 'length_m': length_m})
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
