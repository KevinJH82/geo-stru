"""
insar_fusion.py — InSAR 形变 × 遥感地质构造解译 融合

支持两种 InSAR 数据源:
  1. MintPy SBAS (h5 格式) — load_mintpy()
  2. geo-insar SBAS (TIF+npy+JSON 格式) — load_geo_insar()

MintPy / geo-insar → 标准化 GeoTIFF + 掩膜
  → A 形变线性体(LOS 速率梯度)
  → A 地形线性体活动性打标(优先用垂直速率)
  → B 沉降连通域(优先用垂直速率)
  → C 东西向形变线性体(ew_velocity 梯度, 需升降双轨 2D 分解)
  → 落盘 GeoJSON / GeoTIFF / 图件 + 遵循 schema 的 metadata.json
"""

import os
import json
import importlib.util as _ilu
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import reproject, transform_bounds, Resampling
from scipy import ndimage
from loguru import logger

from config import __version__
from core.terrain_utils import TerrainProcessor
from core import lineament

SOURCE = "geo-stru-insar-fusion"
VERSION = __version__
DEFAULT_SEED = 42
DEFAULT_COH_THR = 0.7

MIN_ACQUISITIONS = 10
MIN_TIMESPAN_DAYS = 180


# ---------------------------------------------------------------------------
# commons 按文件加载
# ---------------------------------------------------------------------------
def _load_commons_insar_utils():
    p = Path(__file__).resolve().parents[2] / "commons" / "insar_utils.py"
    if not p.exists():
        return None
    try:
        spec = _ilu.spec_from_file_location("geostru_commons_insar_utils", p)
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


_iu = _load_commons_insar_utils()


def _los_to_vertical(disp_los, inc_deg):
    if _iu is not None:
        return _iu.los_to_vertical(disp_los, inc_deg)
    return disp_los / np.cos(np.radians(inc_deg))


def _coherence_mask(coh, thr):
    if _iu is not None:
        return _iu.coherence_mask(coh, threshold=thr)
    return (coh >= thr) & ~np.isnan(coh)


# ---------------------------------------------------------------------------
# 1a. MintPy 加载器 (原有)
# ---------------------------------------------------------------------------
def load_mintpy(mintpy_dir: str) -> Dict:
    d = Path(mintpy_dir)

    def rd(fn, ds):
        with h5py.File(d / fn, "r") as f:
            return f[ds][:]

    import h5py
    with h5py.File(d / "velocity.h5", "r") as f:
        a = dict(f.attrs)
        vel = f["velocity"][:].astype(np.float64) * 1000.0

    coh = rd("temporalCoherence.h5", "temporalCoherence").astype(np.float64)
    with h5py.File(d / "inputs" / "geometryGeo.h5", "r") as f:
        dem = f["height"][:].astype(np.float64)
        inc = f["incidenceAngle"][:].astype(np.float64)
        wmask = f["waterMask"][:].astype(bool) if "waterMask" in f else np.ones_like(dem, bool)

    ts, dates = None, []
    ts_path = d / "timeseries.h5"
    if ts_path.exists():
        with h5py.File(ts_path, "r") as f:
            ts = f["timeseries"][:].astype(np.float64) * 1000.0
            dates = [x.decode() if isinstance(x, bytes) else str(x) for x in f["date"][:]]

    H, W = vel.shape
    X0, Y0 = float(a["X_FIRST"]), float(a["Y_FIRST"])
    xs, ys = float(a["X_STEP"]), float(a["Y_STEP"])
    transform = Affine.translation(X0, Y0) * Affine.scale(xs, ys)
    epsg = int(a.get("EPSG", 4326))

    return {
        "vel": vel, "coh": coh, "dem": dem, "inc": inc, "wmask": wmask,
        "ts": ts, "dates": dates, "vertical": None, "ew": None,
        "transform": transform, "epsg": epsg, "shape": (H, W),
        "pixel_m": (abs(xs), abs(ys)),
        "inc_mean": float(np.nanmean(inc)),
        "wavelength_m": float(a.get("WAVELENGTH", np.nan)),
        "orbit": a.get("ORBIT_DIRECTION"), "ref_xy": (int(a.get("REF_X", -1)), int(a.get("REF_Y", -1))),
        "start_date": str(a.get("START_DATE", "")), "end_date": str(a.get("END_DATE", "")),
        "source": "mintpy_sbas", "attrs": a,
    }


# ---------------------------------------------------------------------------
# 1b. geo-insar 加载器 (新增)
# ---------------------------------------------------------------------------
def load_geo_insar(insar_dir: str) -> Dict:
    """
    读取 geo-insar 的 SBAS 产物 (velocity_mm_per_year.tif + cumulative_displacement.npy
    + dates.json + summary.json), 以及 2D 分解结果 (vertical_velocity.tif / ew_velocity.tif)。

    如果存在升降双轨,用各自的 dominant burst;如果只有单轨,也能用。
    """
    d = Path(insar_dir)

    # 找 dominant burst(s)
    sbas_dir = d / "sbas"
    bursts = sorted([b for b in sbas_dir.iterdir()
                     if b.is_dir() and (b / "velocity_mm_per_year.tif").exists()])
    if not bursts:
        raise FileNotFoundError(f"sbas/ 下无 velocity_mm_per_year.tif: {sbas_dir}")

    # 读第一个 burst 作为主 LOS (升轨优先)
    asc_burst = None
    desc_burst = None
    for b in bursts:
        summ = json.load(open(b / "summary.json"))
        orbit = summ.get("orbit_direction", "")
        if "ASCENDING" in orbit and asc_burst is None:
            asc_burst = b
        elif "DESCENDING" in orbit and desc_burst is None:
            desc_burst = b

    # 如果没有 orbit 信息,用第一个 burst
    primary = asc_burst or bursts[0]
    summ = json.load(open(primary / "summary.json"))

    # LOS 速率
    with rasterio.open(primary / "velocity_mm_per_year.tif") as src:
        vel = src.read(1).astype(np.float64)
        transform = src.transform
        epsg = src.crs.to_epsg() if src.crs else 4326
        H, W = vel.shape
        pixel_m = (abs(transform.a), abs(transform.e))

    # 时序
    ts_path = primary / "cumulative_displacement.npy"
    ts = np.load(ts_path).astype(np.float64) if ts_path.exists() else None
    dates_path = primary / "dates.json"
    dates = json.load(open(dates_path)) if dates_path.exists() else []

    # DEM (从第一个干涉对目录读取)
    dem = None
    sent_dir = d / "sentinel1_insar"
    if sent_dir.exists():
        for pair_dir in sorted(sent_dir.iterdir()):
            dem_path = pair_dir / "dem.tif"
            if dem_path.exists():
                with rasterio.open(dem_path) as s:
                    dem = s.read(1).astype(np.float64)
                break

    # 2D 分解 (如果存在)
    vertical, ew = None, None
    vert_path = d / "vertical_velocity.tif"
    ew_path = d / "ew_velocity.tif"
    if vert_path.exists():
        with rasterio.open(vert_path) as s:
            vertical = s.read(1).astype(np.float64)
        with rasterio.open(ew_path) as s:
            ew = s.read(1).astype(np.float64)
        logger.info(f"[insar_fusion] 2D 分解结果已加载 (vertical+ew)")

    return {
        "vel": vel, "coh": None, "dem": dem, "inc": None, "wmask": None,
        "ts": ts, "dates": dates, "vertical": vertical, "ew": ew,
        "transform": transform, "epsg": epsg, "shape": (H, W),
        "pixel_m": pixel_m,
        "inc_mean": summ.get("incidence_angle_mean", 38.0),
        "wavelength_m": 0.0554658,
        "orbit": summ.get("orbit_direction"),
        "ref_xy": (-1, -1),
        "start_date": summ.get("date_range", ["", ""])[0] if summ.get("date_range") else "",
        "end_date": summ.get("date_range", ["", ""])[1] if summ.get("date_range") else "",
        "source": "geo_insar_sbas",
        "attrs": summ,
        # geo-insar 特有
        "asc_burst": str(asc_burst) if asc_burst else None,
        "desc_burst": str(desc_burst) if desc_burst else None,
        "has_2d": vertical is not None,
        "valid_pixel_pct": summ.get("valid_pixel_pct"),
    }


# ---------------------------------------------------------------------------
# 2. 掩膜 / 梯度
# ---------------------------------------------------------------------------
def build_mask(data: Dict, coh_thr: float = DEFAULT_COH_THR) -> np.ndarray:
    if data.get("coh") is not None:
        return _coherence_mask(data["coh"], coh_thr) & data.get("wmask", np.ones(data["vel"].shape, bool)) & np.isfinite(data["vel"])
    return np.isfinite(data["vel"])


def velocity_gradient(velm, valid):
    fill = np.nanmean(velm[valid]) if valid.any() else 0.0
    gy, gx = np.gradient(np.where(np.isfinite(velm), velm, fill))
    grad = np.hypot(gx, gy)
    grad[~valid] = np.nan
    return grad


# ---------------------------------------------------------------------------
# 3. 线性体提取
# ---------------------------------------------------------------------------
def deformation_lineaments(velm, valid, grad, transform, pixel_m, seed=DEFAULT_SEED):
    fin = velm[valid]
    lo, hi = np.nanpercentile(fin, 2), np.nanpercentile(fin, 98)
    vnorm = np.clip((np.nan_to_num(velm, nan=lo) - lo) / (hi - lo + 1e-9), 0, 1)
    return lineament.extract_lineaments(
        multidir_hillshade=vnorm, slope=np.nan_to_num(grad, nan=0.0),
        pixel_size_m=pixel_m, transform=transform, valid_mask=valid,
        canny_sigma=1.2, slope_gate_deg=0.0, slope_gate_pct=70.0, slope_gate_cap=1e9,
        min_length_m=120.0, density_window_m=800.0, rng_seed=seed,
    )


def topographic_lineaments(dem, transform, pixel_m, seed=DEFAULT_SEED):
    multidir = TerrainProcessor.compute_multidirectional_hillshade(dem, pixel_m)
    slope_deg = TerrainProcessor.compute_slope(dem, pixel_m)
    lin = lineament.extract_lineaments(
        multidir_hillshade=multidir, slope=slope_deg,
        pixel_size_m=pixel_m, transform=transform, valid_mask=np.isfinite(dem),
        min_length_m=120.0, density_window_m=800.0, rng_seed=seed,
    )
    lin["_multidir"] = multidir
    return lin


def ew_deformation_lineaments(ew, valid, transform, pixel_m, seed=DEFAULT_SEED):
    """在东西向速率场上提取形变线性体 (需 2D 分解)。"""
    grad = velocity_gradient(ew, valid)
    return deformation_lineaments(ew, valid, grad, transform, pixel_m, seed), grad


# ---------------------------------------------------------------------------
# 4. 线性体活动性打标
# ---------------------------------------------------------------------------
def score_activity(segments, velm, coh, valid, transform, grad,
                   half_px=2, n_stations=5):
    inv = ~transform
    H, W = velm.shape
    dv_ref = float(np.nanpercentile(grad[valid], 75)) * 2 if valid.any() else 1.0

    def sample(seg):
        c0, r0 = inv * seg["p0"]
        c1, r1 = inv * seg["p1"]
        dc, dr = c1 - c0, r1 - r0
        L = np.hypot(dc, dr)
        if L < 1:
            return np.nan, 0, np.nan
        nx, ny = -dr / L, dc / L
        A, B, C = [], [], []
        for t in np.linspace(0.2, 0.8, n_stations):
            cc, rr = c0 + t * dc, r0 + t * dr
            for s in range(1, half_px + 1):
                for sign, bucket in ((+1, A), (-1, B)):
                    px = int(round(cc + sign * s * nx))
                    py = int(round(rr + sign * s * ny))
                    if 0 <= py < H and 0 <= px < W and valid[py, px]:
                        bucket.append(velm[py, px])
                        C.append(coh[py, px] if coh is not None else np.nan)
        if len(A) < 2 or len(B) < 2:
            return np.nan, len(A) + len(B), np.nan
        return float(abs(np.nanmean(A) - np.nanmean(B))), len(A) + len(B), float(np.nanmean(C))

    out = []
    for i, seg in enumerate(segments):
        dv, nv, cm = sample(seg)
        if not np.isfinite(dv) or nv < 4:
            cls, score = "无数据", 0.0
        else:
            score = float(dv / (dv_ref + 1e-6))
            cls = "形变一致(活动?)" if dv >= dv_ref else "仅地形(古/锁定?)"
        out.append({
            "id": i, "strike_deg": round(seg["strike_deg"], 1),
            "length_m": round(seg["length_m"], 1),
            "dv_mm_yr": None if not np.isfinite(dv) else round(dv, 2),
            "n_valid": nv, "coh": None if not np.isfinite(cm) else round(cm, 2),
            "activity_class": cls, "activity_score": round(score, 2),
            "p0": list(seg["p0"]), "p1": list(seg["p1"]),
        })
    return out


# ---------------------------------------------------------------------------
# 5. 沉降探测
# ---------------------------------------------------------------------------
def detect_subsidence(velm, valid, transform, pixel_m, k_sigma=1.5,
                      min_area_px=5):
    fin = velm[valid]
    thr = float(np.nanmean(fin) - k_sigma * np.nanstd(fin))
    sub = valid & (velm < thr)
    lbl, n = ndimage.label(sub)
    clusters = []
    for kk in range(1, n + 1):
        m = lbl == kk
        area_px = int(m.sum())
        if area_px < min_area_px:
            lbl[m] = 0
            continue
        ys_, xs_ = np.where(m)
        cx, cy = transform * (xs_.mean() + 0.5, ys_.mean() + 0.5)
        clusters.append({
            "id": len(clusters), "label": kk, "area_px": area_px,
            "area_m2": round(area_px * pixel_m[0] * pixel_m[1], 1),
            "min_vel_mm_yr": round(float(np.nanmin(velm[m])), 2),
            "mean_vel_mm_yr": round(float(np.nanmean(velm[m])), 2),
            "centroid": [round(cx, 1), round(cy, 1)],
        })
    return clusters, lbl, thr


# ---------------------------------------------------------------------------
# 5b. B3 沉降漏斗多边形圈定 + 长轴分析
# ---------------------------------------------------------------------------
def delineate_goaf_polygons(clusters, lbl, velm, transform, pixel_m,
                            topo_strikes=None):
    """
    将每个沉降簇转为凸包多边形,并计算长轴方向(PCA)。

    Args:
        clusters: detect_subsidence 返回的 cluster 列表(会被原地更新)
        lbl: 连通分量标注栅格
        velm: 速率栅格
        transform: Affine
        pixel_m: (x, y) 像元尺寸
        topo_strikes: 地形线性体走向列表,用于长轴对比

    Returns:
        更新后的 clusters 列表(每条新增 boundary, long_axis_deg, strike_diff_deg)
    """
    H, W = lbl.shape
    for cl in clusters:
        cid = cl["label"]  # 原始连通分量标签
        m = lbl == cid
        if not m.any():
            cl["boundary"] = None
            cl["long_axis_deg"] = None
            cl["strike_diff_deg"] = None
            continue

        ys, xs = np.where(m)
        # 像元→地理坐标
        coords = []
        for r, c in zip(ys, xs):
            lon, lat = transform * (c + 0.5, r + 0.5)
            coords.append([lon, lat])
        coords = np.array(coords)

        # 凸包 (Graham scan via scipy)
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(coords)
            hull_coords = coords[hull.vertices].tolist()
            # 闭合
            hull_coords.append(hull_coords[0])
        except Exception:
            # 退化(共线/点数<3)→用 bbox
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            hull_coords = [[x_min, y_min], [x_max, y_min],
                           [x_max, y_max], [x_min, y_max], [x_min, y_min]]

        cl["boundary"] = hull_coords

        # PCA 长轴方向 (取第一主成分)
        if len(coords) >= 2:
            centered = coords - coords.mean(axis=0)
            # 对坐标做各向异性校正 (经度方向需要 ×cos(lat) 近似)
            mean_lat = coords[:, 1].mean()
            cos_lat = np.cos(np.radians(mean_lat))
            scaled = centered.copy()
            scaled[:, 0] *= cos_lat  # 把经度差近似换算为米
            scaled[:, 0] *= 111320   # 度→米
            scaled[:, 1] *= 110540   # 度→米
            cov = np.cov(scaled.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            # 最大特征值对应的特征向量
            principal = eigvecs[:, np.argmax(eigvals)]
            # 长轴方位角 (0=N, 顺时针, 0–180°)
            dx, dy = principal[0], principal[1]  # 东, 北
            axis_deg = np.degrees(np.arctan2(dx, dy)) % 180.0
            cl["long_axis_deg"] = round(float(axis_deg), 1)
            # 长轴 vs 最近断裂走向的最小差
            if topo_strikes:
                diffs = [min(abs(axis_deg - s), 180 - abs(axis_deg - s)) for s in topo_strikes]
                cl["strike_diff_deg"] = round(float(min(diffs)), 1)
            else:
                cl["strike_diff_deg"] = None
        else:
            cl["long_axis_deg"] = None
            cl["strike_diff_deg"] = None

    return clusters


# ---------------------------------------------------------------------------
# 5c. B4 沉降时序分类
# ---------------------------------------------------------------------------
def classify_subsidence_timeseries(clusters, lbl, ts_array, dates):
    """
    对每个沉降簇提取均值累计形变时序,拟合线性/二次模型,分类为:
      - "linear": 线性沉降(匀速) → 持续活动采空
      - "accelerating": 加速沉降(二次项显著为负) → 活动加剧
      - "decelerating": 减速沉降(二次项显著为正) → 趋于稳定
      - "stable": 无显著趋势
      - "no_data": 时序不可用

    Args:
        clusters: cluster 列表(会被原地更新)
        lbl: 连通分量标注栅格
        ts_array: (N_dates, H, W) 累计形变数组(mm), 或 None
        dates: 日期字符串列表

    Returns:
        更新后的 clusters 列表(每条新增 ts_class, ts_r2, ts_rate_mm_yr)
    """
    if ts_array is None or len(dates) < 3:
        for cl in clusters:
            cl["ts_class"] = "no_data"
            cl["ts_r2"] = None
            cl["ts_rate_mm_yr"] = None
        return clusters

    n_dates = ts_array.shape[0]
    H, W = ts_array.shape[1:]

    # 时间轴 (天,相对于首日)
    try:
        from datetime import datetime as _dt
        t_days = np.array([
            (_dt.strptime(d.replace("-", ""), "%Y%m%d") -
             _dt.strptime(dates[0].replace("-", ""), "%Y%m%d")).days
            for d in dates
        ], dtype=np.float64)
    except Exception:
        t_days = np.arange(n_dates, dtype=np.float64)

    if t_days[-1] < 1:
        t_days = np.arange(n_dates, dtype=np.float64)

    for cl in clusters:
        cid = cl["label"]  # 原始连通分量标签
        m = lbl == cid
        if not m.any() or m.sum() < 3:
            cl["ts_class"] = "no_data"
            cl["ts_r2"] = None
            cl["ts_rate_mm_yr"] = None
            continue

        # 提取簇内均值时序
        mean_ts = np.array([
            float(np.nanmean(ts_array[t][m])) if np.any(np.isfinite(ts_array[t][m]))
            else np.nan
            for t in range(n_dates)
        ])
        valid_ts = np.isfinite(mean_ts)
        if valid_ts.sum() < 3:
            cl["ts_class"] = "no_data"
            cl["ts_r2"] = None
            cl["ts_rate_mm_yr"] = None
            continue

        y = mean_ts[valid_ts] - mean_ts[valid_ts][0]  # 去起始偏移
        t = t_days[valid_ts]

        # 线性拟合: y = a*t + b
        coeffs_lin = np.polyfit(t, y, 1)
        y_lin = np.polyval(coeffs_lin, t)
        ss_res_lin = np.sum((y - y_lin) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
        r2_lin = 1 - ss_res_lin / ss_tot

        # 二次拟合: y = a*t^2 + b*t + c
        coeffs_quad = np.polyfit(t, y, 2)
        y_quad = np.polyval(coeffs_quad, t)
        ss_res_quad = np.sum((y - y_quad) ** 2)
        r2_quad = 1 - ss_res_quad / ss_tot

        rate_mm_yr = round(float(coeffs_lin[0]) * 365.25, 2)

        # 分类规则
        quad_a = coeffs_quad[0]  # 二次项系数
        if ss_tot < 1e-6:
            cls = "stable"
        elif r2_lin > 0.8:
            # 线性拟合已经很好
            if abs(rate_mm_yr) < 1.0:
                cls = "stable"
            else:
                cls = "linear"
                # 检查是否有加速/减速信号
                if r2_quad - r2_lin > 0.1 and abs(quad_a) > 1e-4:
                    cls = "accelerating" if quad_a < 0 else "decelerating"
        elif r2_quad > 0.7:
            cls = "accelerating" if quad_a < 0 else "decelerating"
        else:
            cls = "stable"  # 拟合不好→保守判

        cl["ts_class"] = cls
        cl["ts_r2"] = round(float(max(r2_lin, r2_quad)), 3)
        cl["ts_rate_mm_yr"] = rate_mm_yr

    return clusters


# ---------------------------------------------------------------------------
# 5d. C线 形变归因 (断裂蠕动 / 滑坡 / 采空沉降 / 未确定)
# ---------------------------------------------------------------------------
def attribute_deformation(clusters, lbl, velm, valid, transform, pixel_m,
                          defm_segments=None, slope=None, aspect=None,
                          distance_to_lineament=None):
    """
    对每个沉降簇做形变归因分类。

    规则(按优先级):
      1. **采空沉降** (goaf): 圆/椭圆形态 + 时序分类为 linear/accelerating + 负速率
      2. **滑坡** (landslide): 坡度大(>15°) + 沉降方向与坡向一致
      3. **断裂蠕动** (fault_creep): 距断裂近(<3 像元) + 漏斗长轴与断裂走向一致(<30°)
      4. **未确定** (undetermined): 不满足以上规则

    Args:
        clusters: 含 B3/B4 属性的 cluster 列表
        lbl: 连通分量标注栅格
        velm: 速率栅格
        valid: 有效掩膜
        transform: Affine
        pixel_m: 像元尺寸
        defm_segments: 形变线性体段列表(走向用于交叉验证)
        slope: 坡度栅格(度), 可选
        aspect: 坡向栅格(度, -1=平), 可选
        distance_to_lineament: 距断裂距离栅格(米), 可选

    Returns:
        attribution_raster: (H,W) int8 栅格 (0=无, 1=采空, 2=滑坡, 3=断裂蠕动, 9=未确定)
        clusters: 更新后的列表(每条新增 attribution_class, attribution_confidence)
    """
    H, W = lbl.shape
    attr_raster = np.zeros((H, W), dtype=np.int8)

    for cl in clusters:
        cid = cl["label"]
        m = lbl == cid
        if not m.any():
            cl["attribution_class"] = "undetermined"
            cl["attribution_confidence"] = 0.0
            continue

        scores = {
            "goaf": 0.0,
            "landslide": 0.0,
            "fault_creep": 0.0,
        }

        # ---- 1. 采空沉降信号 ----
        ts_class = cl.get("ts_class", "no_data")
        mean_vel = cl.get("mean_vel_mm_yr", 0)
        area_m2 = cl.get("area_m2", 0)

        # 负速率 (越负越可信)
        if mean_vel < -2.0:
            scores["goaf"] += 0.3
        if mean_vel < -5.0:
            scores["goaf"] += 0.2

        # 时序分类: 持续/加速沉降 → 活动采空
        if ts_class in ("linear", "accelerating"):
            scores["goaf"] += 0.3
        if ts_class == "accelerating":
            scores["goaf"] += 0.1

        # 面积/形态: 面积适中(100–100000 m²) → 典型采空漏斗
        if 100 <= area_m2 <= 100000:
            scores["goaf"] += 0.1

        # 圆形度: 面积/周长² 比值接近圆 → 支持采空
        if cl.get("boundary"):
            boundary = cl["boundary"]
            n_pts = len(boundary) - 1  # 去掉闭合点
            if n_pts >= 3:
                perimeter = sum(
                    np.hypot(boundary[i+1][0] - boundary[i][0],
                             boundary[i+1][1] - boundary[i][1])
                    for i in range(n_pts)
                )
                if perimeter > 0:
                    circularity = 4 * np.pi * area_m2 / (perimeter ** 2)
                    # 圆形度 0–1, 越接近 1 越圆
                    if circularity > 0.5:
                        scores["goaf"] += 0.1

        # ---- 2. 滑坡信号 ----
        if slope is not None:
            mean_slope = float(np.nanmean(slope[m]))
            if mean_slope > 15.0:
                scores["landslide"] += 0.3
            elif mean_slope > 8.0:
                scores["landslide"] += 0.15

            # 坡向一致性: 沉降区坡向集中 → 滑坡
            if aspect is not None:
                aspect_vals = aspect[m]
                aspect_valid = aspect_vals[aspect_vals >= 0]
                if len(aspect_valid) > 5:
                    # 用圆形统计: mean resultant length
                    rad = np.deg2rad(aspect_valid)
                    R = np.hypot(np.mean(np.sin(rad)), np.mean(np.cos(rad)))
                    if R > 0.5:  # 坡向集中
                        scores["landslide"] += 0.2

        # ---- 3. 断裂蠕动信号 ----
        if distance_to_lineament is not None:
            mean_dist = float(np.nanmean(distance_to_lineament[m]))
            px_m = max(pixel_m)
            # 距断裂近(平均距离 < 5 像元)
            if mean_dist < 5 * px_m:
                scores["fault_creep"] += 0.3
            if mean_dist < 2 * px_m:
                scores["fault_creep"] += 0.2

        # 长轴方向与断裂走向一致
        long_axis = cl.get("long_axis_deg")
        strike_diff = cl.get("strike_diff_deg")
        if strike_diff is not None:
            if strike_diff < 30:
                scores["fault_creep"] += 0.3
            elif strike_diff < 45:
                scores["fault_creep"] += 0.15

        # 形变线性体走向与沉降簇重叠(交叉验证)
        if defm_segments and cl.get("boundary"):
            # 简化: 检查是否有形变线性体经过沉降簇附近
            inv = ~transform
            n_nearby = 0
            nearby_strikes = []
            for seg in defm_segments:
                c0, r0 = inv * seg["p0"]
                c1, r1 = inv * seg["p1"]
                mid_c, mid_r = (c0 + c1) / 2, (r0 + r1) / 2
                if m[int(np.clip(mid_r, 0, H-1)), int(np.clip(mid_c, 0, W-1))]:
                    n_nearby += 1
                    nearby_strikes.append(seg.get("strike_deg", 0))
            if n_nearby > 0:
                scores["fault_creep"] += 0.2
                # 形变线性体走向与长轴一致
                if long_axis is not None and nearby_strikes:
                    for s in nearby_strikes:
                        diff = min(abs(long_axis - s), 180 - abs(long_axis - s))
                        if diff < 30:
                            scores["fault_creep"] += 0.1
                            break

        # ---- 归因决策 ----
        best_class = max(scores, key=scores.get)
        best_score = scores[best_class]

        if best_score < 0.3:
            attribution = "undetermined"
            confidence = best_score / 0.3  # 归一化到 [0,1)
        else:
            attribution = best_class
            confidence = min(1.0, best_score)

        cl["attribution_class"] = attribution
        cl["attribution_confidence"] = round(float(confidence), 3)
        cl["attribution_scores"] = {k: round(v, 3) for k, v in scores.items()}

        # 写入栅格
        attr_code = {"goaf": 1, "landslide": 2, "fault_creep": 3}.get(attribution, 9)
        attr_raster[m] = attr_code

    return attr_raster, clusters


# ---------------------------------------------------------------------------
# 6. 落盘
# ---------------------------------------------------------------------------
def _write_gtiff(path, arr, transform, epsg, nodata=np.nan):
    arr = arr.astype("float32")
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype="float32", crs=CRS.from_epsg(epsg),
                       transform=transform, nodata=nodata, compress="deflate") as dst:
        dst.write(arr, 1)


def _write_line_geojson(records, path, crs_str):
    feats = []
    for r in records:
        props = {k: v for k, v in r.items() if k not in ("p0", "p1")}
        feats.append({"type": "Feature",
                      "geometry": {"type": "LineString", "coordinates": [r["p0"], r["p1"]]},
                      "properties": props})
    fc = {"type": "FeatureCollection",
          "crs": {"type": "name", "properties": {"name": crs_str}}, "features": feats}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False)


def _write_point_geojson(clusters, path, crs_str):
    feats = [{"type": "Feature",
              "geometry": {"type": "Point", "coordinates": c["centroid"]},
              "properties": {k: v for k, v in c.items() if k not in ("centroid", "boundary")}}
             for c in clusters]
    fc = {"type": "FeatureCollection",
          "crs": {"type": "name", "properties": {"name": crs_str}}, "features": feats}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False)


def _write_polygon_geojson(clusters, path, crs_str):
    """沉降漏斗多边形(凸包)→ GeoJSON。"""
    feats = []
    for c in clusters:
        boundary = c.get("boundary")
        if boundary is None:
            continue
        props = {k: v for k, v in c.items()
                 if k not in ("centroid", "boundary")}
        feats.append({"type": "Feature",
                      "geometry": {"type": "Polygon", "coordinates": [boundary]},
                      "properties": props})
    fc = {"type": "FeatureCollection",
          "crs": {"type": "name", "properties": {"name": crs_str}}, "features": feats}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 7. 图件
# ---------------------------------------------------------------------------
def _setup_cjk_font():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass
    return plt


def render_overlay(out_dir, data, valid, velm, grad, topo, defm, sub_lbl, multidir,
                   ew_data=None, ew_defm=None, ew_grad=None, structural_dir=None):
    plt = _setup_cjk_font()
    inv = ~data["transform"]
    has_ew = ew_data is not None
    ncols = 4 if has_ew else 3

    def draw(ax, segs, color, lw=1.3, label=None):
        first = True
        for s in segs:
            c0, r0 = inv * s["p0"]; c1, r1 = inv * s["p1"]
            ax.plot([c0, c1], [r0, r1], color=color, lw=lw, label=label if first else None)
            first = False

    # 加载 geo-stru 地形线性体叠加
    topo_segs_overlay = None
    if structural_dir:
        gj_path = Path(structural_dir) / "lineaments.geojson"
        if gj_path.exists():
            fc = json.load(open(gj_path))
            topo_segs_overlay = [(f["geometry"]["coordinates"][0], f["geometry"]["coordinates"][1])
                                 for f in fc["features"]]

    fig, axs = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    vmin, vmax = -40, 40

    # Panel 1: LOS 速率 + 线性体
    ax = axs[0]
    im = ax.imshow(velm, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    draw(ax, topo["segments"], "k", 1.3, "terrain")
    draw(ax, defm["segments"], "lime", 1.1, "deformation")
    ax.contour(sub_lbl > 0, levels=[0.5], colors="magenta", linewidths=1.5)
    ax.set_title("LOS velocity (mm/yr)")
    ax.legend(loc="lower right", fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Panel 2: DEM + 地形线性体
    ax = axs[1]; ax.imshow(multidir, cmap="gray")
    draw(ax, topo["segments"], "red", 1.3)
    if topo_segs_overlay:
        for p0, p1 in topo_segs_overlay:
            c0, r0 = inv * p0; c1, r1 = inv * p1
            ax.plot([c0, c1], [r0, r1], "cyan", lw=1.0)
    ax.set_title("DEM multidirectional hillshade")

    # Panel 3: gradient + deformation lineaments
    ax = axs[2]
    im = ax.imshow(grad, cmap="magma", vmax=np.nanpercentile(grad, 98))
    draw(ax, defm["segments"], "cyan", 1.1)
    ax.set_title("|grad(LOS)| + def. lineaments")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Panel 4: EW velocity + EW lineaments (if available)
    if has_ew:
        ax = axs[3]
        im = ax.imshow(ew_data, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        draw(ax, topo["segments"], "k", 1.0, "terrain")
        if ew_defm and ew_defm.get("segments"):
            draw(ax, ew_defm["segments"], "lime", 1.1, "EW deformation")
        ax.set_title("EW velocity (mm/yr) + EW lineaments")
        ax.legend(loc="lower right", fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    p = os.path.join(out_dir, "fusion_overlay.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return "fusion_overlay.png"


def render_timeseries(out_dir, data, velm, sub_lbl):
    if data["ts"] is None or not data["dates"]:
        return None
    plt = _setup_cjk_font()
    ts_shape = data["ts"].shape[1:]  # 时序栅格的空间尺寸
    # 用时序栅格自身尺寸来找最大形变点 (sub_lbl 可能在不同网格上)
    ts_first = data["ts"][0]
    ts_valid = np.isfinite(ts_first)
    if ts_valid.any():
        py, px = np.unravel_index(np.nanargmin(ts_first), ts_shape)
    else:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    series = data["ts"][:, py, px] - data["ts"][0, py, px]
    ax.plot(range(len(data["dates"])), series, "o-", markersize=3)
    ax.set_xticks(range(len(data["dates"])))
    ax.set_xticklabels(data["dates"], rotation=45, fontsize=8)
    ax.set_ylabel("Cumulative LOS disp. (mm)")
    ax.set_title(f"Max deformation point (px={px},{py})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "timeseries_maxpoint.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return "timeseries_maxpoint.png"


# ---------------------------------------------------------------------------
# 8. 主编排
# ---------------------------------------------------------------------------
def run_fusion(insar_dir: str, out_dir: str, aoi_name: Optional[str] = None,
               seed: int = DEFAULT_SEED, coh_thr: float = DEFAULT_COH_THR,
               make_plots: bool = True, created_at: Optional[str] = None,
               structural_dir: Optional[str] = None,
               task_code: Optional[str] = None) -> Dict:
    """
    端到端融合。自动检测数据格式 (MintPy h5 vs geo-insar TIF+npy)。
    如果有 2D 分解结果,用垂直速率做活动性/沉降,新增东西向线性体。
    """
    os.makedirs(out_dir, exist_ok=True)

    # 自动检测格式
    insar_path = Path(insar_dir)
    if (insar_path / "velocity.h5").exists():
        data = load_mintpy(insar_dir)
        logger.info(f"[insar_fusion] MintPy 格式")
    elif (insar_path / "sbas").exists():
        data = load_geo_insar(insar_dir)
        logger.info(f"[insar_fusion] geo-insar 格式 (source={data['source']})")
    else:
        raise FileNotFoundError(f"无法识别 InSAR 数据格式: {insar_dir}")

    H, W = data["shape"]
    transform, epsg, pixel_m = data["transform"], data["epsg"], data["pixel_m"]
    crs_str = f"EPSG:{epsg}"

    valid = build_mask(data, coh_thr)
    velm = np.where(valid, data["vel"], np.nan)
    grad = velocity_gradient(velm, valid)

    # 垂直/东西向 (如果有 2D 分解)
    vert, ew = data.get("vertical"), data.get("ew")
    has_2d = vert is not None

    fin = velm[valid]
    logger.info(f"[insar_fusion] valid {valid.sum()}/{H*W} ({100*valid.sum()/(H*W):.1f}%) "
                f"LOS {fin.min():.1f}~{fin.max():.1f} mm/yr std {fin.std():.2f}"
                f"{' [2D decomposed]' if has_2d else ''}")

    # DEM (可能没有)
    dem = data.get("dem")
    multidir = None

    # 地形线性体 (需要 DEM)
    topo = {"segments": [], "mask": np.zeros((H, W), bool),
            "distance_m": np.full((H, W), np.nan, np.float32),
            "density": np.zeros((H, W), np.float32),
            "stats": {"n_lineaments": 0, "total_length_km": 0.0,
                      "density_mean": 0.0, "dominant_strikes_deg": []}}
    if dem is not None:
        # 对齐 DEM 到 velocity grid (如果尺寸不同)
        if dem.shape != (H, W):
            dem_aligned = np.full((H, W), np.nan, dtype=np.float64)
            dem_transform = data["transform"]  # velocity grid transform
            # DEM 可能有自己的 transform; 简单处理: resize
            from scipy.ndimage import zoom
            zh, zw = H / dem.shape[0], W / dem.shape[1]
            dem = zoom(np.nan_to_num(dem, nan=0), (zh, zw), order=1)
            dem = dem[:H, :W]
        topo = topographic_lineaments(dem, transform, pixel_m, seed)
        multidir = topo.pop("_multidir")
    elif structural_dir:
        # 从 geo-stru 读取已有的线性体
        gj_path = Path(structural_dir) / "lineaments.geojson"
        if gj_path.exists():
            fc = json.load(open(gj_path))
            topo["segments"] = [
                {"p0": tuple(f["geometry"]["coordinates"][0]),
                 "p1": tuple(f["geometry"]["coordinates"][1]),
                 "strike_deg": f["properties"].get("strike_deg", 0),
                 "length_m": f["properties"].get("length_m", 0)}
                for f in fc["features"]
            ]
            topo["stats"]["n_lineaments"] = len(topo["segments"])
            topo["stats"]["dominant_strikes_deg"] = [s["strike_deg"] for s in topo["segments"][:3]]
            logger.info(f"[insar_fusion] 从 geo-stru 加载 {len(topo['segments'])} 条地形线性体")

    # A1 LOS 形变线性体
    defm = deformation_lineaments(velm, valid, grad, transform, pixel_m, seed)

    # A2 活动性打标: 优先用垂直速率
    if has_2d:
        vert_valid = np.isfinite(vert)
        vert_grad = velocity_gradient(np.where(vert_valid, vert, np.nan), vert_valid)
        activity = score_activity(topo["segments"], np.where(vert_valid, vert, np.nan),
                                  None, vert_valid, transform, vert_grad)
        activity_field = "vertical"
    else:
        activity = score_activity(topo["segments"], velm, data.get("coh"), valid, transform, grad)
        activity_field = "los"
    n_active = sum(1 for x in activity if x["activity_class"].startswith("形变一致"))

    # B 沉降: 优先用垂直速率
    if has_2d:
        vert_valid = np.isfinite(vert)
        vert_fill = np.where(vert_valid, vert, np.nan)
        clusters, sub_lbl, sub_thr = detect_subsidence(vert_fill, vert_valid, transform, pixel_m)
        subsidence_field = "vertical"
    else:
        clusters, sub_lbl, sub_thr = detect_subsidence(velm, valid, transform, pixel_m)
        subsidence_field = "los"

    # B3 沉降漏斗多边形圈定 + 长轴分析
    topo_strikes = topo["stats"].get("dominant_strikes_deg", [])
    if clusters:
        clusters = delineate_goaf_polygons(
            clusters, sub_lbl, velm, transform, pixel_m, topo_strikes=topo_strikes)
        logger.info(f"[insar_fusion] B3: {sum(1 for c in clusters if c.get('boundary'))} 个多边形圈定")
        for cl in clusters:
            if cl.get("long_axis_deg") is not None:
                logger.info(f"  簇{cl['id']}: 长轴 {cl['long_axis_deg']}°, "
                            f"与断裂差 {cl.get('strike_diff_deg', 'N/A')}°, "
                            f"面积 {cl['area_m2']:.0f} m²")

    # B4 沉降时序分类
    ts_array = data.get("ts")
    ts_dates = data.get("dates", [])
    if clusters:
        clusters = classify_subsidence_timeseries(clusters, sub_lbl, ts_array, ts_dates)
        ts_summary = {}
        for cl in clusters:
            cls = cl.get("ts_class", "no_data")
            ts_summary[cls] = ts_summary.get(cls, 0) + 1
        logger.info(f"[insar_fusion] B4: 时序分类 {ts_summary}")

    # C线 形变归因
    attr_raster = np.zeros((H, W), dtype=np.int8)
    if clusters:
        # 准备地形辅助数据
        slope_for_attr = None
        aspect_for_attr = None
        dist_for_attr = None
        if dem is not None:
            try:
                slope_for_attr = TerrainProcessor.compute_slope(dem, pixel_m)
                aspect_for_attr = TerrainProcessor.compute_aspect(dem, pixel_m)
            except Exception:
                pass
        if topo.get("distance_m") is not None and not np.all(np.isnan(topo["distance_m"])):
            dist_for_attr = topo["distance_m"]
        elif structural_dir:
            # 尝试从 geo-stru 产物读取距离栅格
            dist_path = os.path.join(structural_dir, "distance_to_lineament.tif")
            if os.path.exists(dist_path):
                try:
                    with rasterio.open(dist_path) as src:
                        dist_for_attr = src.read(1).astype(np.float64)
                        # 尺寸对齐
                        if dist_for_attr.shape != (H, W):
                            from scipy.ndimage import zoom
                            zh, zw = H / dist_for_attr.shape[0], W / dist_for_attr.shape[1]
                            dist_for_attr = zoom(np.nan_to_num(dist_for_attr, nan=1e6), (zh, zw))
                            dist_for_attr = dist_for_attr[:H, :W]
                except Exception:
                    pass

        attr_raster, clusters = attribute_deformation(
            clusters, sub_lbl, velm, valid, transform, pixel_m,
            defm_segments=defm["segments"],
            slope=slope_for_attr, aspect=aspect_for_attr,
            distance_to_lineament=dist_for_attr)
        attr_summary = {}
        for cl in clusters:
            cls = cl.get("attribution_class", "undetermined")
            attr_summary[cls] = attr_summary.get(cls, 0) + 1
        logger.info(f"[insar_fusion] C线 归因: {attr_summary}")

    # C 东西向形变线性体 (如果有 2D 分解)
    ew_defm, ew_grad = None, None
    if has_ew := (ew is not None):
        ew_valid = np.isfinite(ew)
        ew_defm, ew_grad = ew_deformation_lineaments(
            np.where(ew_valid, ew, np.nan), ew_valid, transform, pixel_m, seed)

    # ---- 落盘栅格 ----
    _write_gtiff(os.path.join(out_dir, "los_velocity_mm_yr.tif"), velm, transform, epsg)
    if has_2d:
        _write_gtiff(os.path.join(out_dir, "vertical_velocity_mm_yr.tif"),
                     np.where(np.isfinite(vert), vert, np.nan), transform, epsg)
        _write_gtiff(os.path.join(out_dir, "ew_velocity_mm_yr.tif"),
                     np.where(np.isfinite(ew), ew, np.nan), transform, epsg)
    _write_gtiff(os.path.join(out_dir, "velocity_gradient.tif"), grad, transform, epsg)
    # C线: 归因栅格 (始终落盘,无簇时为全零)
    _write_gtiff(os.path.join(out_dir, "deformation_attribution.tif"),
                 attr_raster.astype("float32"), transform, epsg, nodata=0)

    # ---- 落盘矢量 ----
    lineament.write_lineaments_geojson(defm["segments"],
                                       os.path.join(out_dir, "deformation_lineaments.geojson"), crs=crs_str)
    if topo["segments"]:
        lineament.write_lineaments_geojson(topo["segments"],
                                           os.path.join(out_dir, "topographic_lineaments.geojson"), crs=crs_str)
    _write_line_geojson(activity, os.path.join(out_dir, "lineaments_activity.geojson"), crs_str)
    _write_point_geojson(clusters, os.path.join(out_dir, "subsidence_clusters.geojson"), crs_str)
    # B3: goaf 多边形
    _write_polygon_geojson(clusters, os.path.join(out_dir, "goaf_polygons.geojson"), crs_str)
    # C线: 归因矢量(每个簇含 attribution_class + confidence; 无簇时空 FC)
    _write_point_geojson(clusters, os.path.join(out_dir, "deformation_attribution.geojson"), crs_str)
    lineament.plot_rose_diagram(topo["segments"], os.path.join(out_dir, "rose_topographic.png"), "Topographic strikes")
    lineament.plot_rose_diagram(defm["segments"], os.path.join(out_dir, "rose_deformation.png"), "Deformation strikes")
    if ew_defm and ew_defm.get("segments"):
        lineament.write_lineaments_geojson(ew_defm["segments"],
                                           os.path.join(out_dir, "ew_deformation_lineaments.geojson"), crs=crs_str)
        lineament.plot_rose_diagram(ew_defm["segments"], os.path.join(out_dir, "rose_ew_deformation.png"),
                                    "EW deformation strikes")

    products = {
        "los_velocity_mm_yr": "los_velocity_mm_yr.tif",
        "velocity_gradient": "velocity_gradient.tif",
        "deformation_lineaments_geojson": "deformation_lineaments.geojson",
        "topographic_lineaments_geojson": "topographic_lineaments.geojson",
        "lineaments_activity_geojson": "lineaments_activity.geojson",
        "subsidence_clusters_geojson": "subsidence_clusters.geojson",
        "goaf_polygons_geojson": "goaf_polygons.geojson",
        "deformation_attribution_tif": "deformation_attribution.tif",
        "deformation_attribution_geojson": "deformation_attribution.geojson",
        "rose_topographic_png": "rose_topographic.png",
        "rose_deformation_png": "rose_deformation.png",
    }
    if has_2d:
        products["vertical_velocity_mm_yr"] = "vertical_velocity_mm_yr.tif"
        products["ew_velocity_mm_yr"] = "ew_velocity_mm_yr.tif"
    if ew_defm and ew_defm.get("segments"):
        products["ew_deformation_lineaments_geojson"] = "ew_deformation_lineaments.geojson"
        products["rose_ew_deformation_png"] = "rose_ew_deformation.png"

    if make_plots:
        products["overlay_png"] = render_overlay(
            out_dir, data, valid, velm, grad, topo, defm, sub_lbl, multidir,
            ew_data=ew if has_2d else None, ew_defm=ew_defm, ew_grad=ew_grad,
            structural_dir=structural_dir)
        tsp = render_timeseries(out_dir, data, velm, sub_lbl)
        if tsp:
            products["timeseries_png"] = tsp

    # ---- signal quality ----
    n_acq = len(data["dates"]) or 0
    span_days = _span_days(data["start_date"], data["end_date"])
    sufficient = (n_acq >= MIN_ACQUISITIONS) and (span_days >= MIN_TIMESPAN_DAYS)
    signal_quality = "ok" if sufficient else "insufficient"
    if has_2d:
        signal_quality += "+2d"

    # ---- bbox (lon/lat) ----
    left, top = transform * (0, 0)
    right, bottom = transform * (W, H)
    bbox_ll = list(transform_bounds(CRS.from_epsg(epsg), CRS.from_epsg(4326),
                                    left, bottom, right, top))

    metadata = {
        "source": SOURCE, "source_version": VERSION,
        "run_id": os.path.basename(os.path.normpath(out_dir)),
        "task_code": task_code or "",
        "aoi_name": aoi_name or Path(insar_dir).name,
        "aoi_bbox": [round(x, 6) for x in bbox_ll],
        "crs": crs_str, "grid": [H, W], "pixel_size_m": list(pixel_m),
        "seed": seed,
        "insar_provenance": {
            "source_insar": data["source"],
            "orbit_direction": data.get("orbit"),
            "incidence_angle_mean": data.get("inc_mean"),
            "date_range": [data["start_date"], data["end_date"]],
            "n_acquisitions": n_acq, "timespan_days": span_days,
            "has_2d_decomposition": has_2d,
        },
        "products": products,
        "fusion_stats": {
            "valid_ratio": round(float(valid.sum() / (H * W)), 3),
            "los_velocity_mm_yr": {
                "min": round(float(fin.min()), 1), "max": round(float(fin.max()), 1),
                "mean": round(float(fin.mean()), 2), "std": round(float(fin.std()), 2)},
            "n_topographic_lineaments": topo["stats"]["n_lineaments"],
            "n_deformation_lineaments": defm["stats"]["n_lineaments"],
            "topographic_dominant_strikes_deg": topo["stats"]["dominant_strikes_deg"],
            "deformation_dominant_strikes_deg": defm["stats"]["dominant_strikes_deg"],
            "n_active_consistent_lineaments": n_active,
            "activity_field": activity_field,
            "n_subsidence_clusters": len(clusters),
            "subsidence_threshold_mm_yr": round(sub_thr, 2),
            "subsidence_field": subsidence_field,
            "subsidence_details": [
                {
                    "id": c["id"],
                    "area_m2": c.get("area_m2"),
                    "mean_vel_mm_yr": c.get("mean_vel_mm_yr"),
                    "long_axis_deg": c.get("long_axis_deg"),
                    "strike_diff_deg": c.get("strike_diff_deg"),
                    "ts_class": c.get("ts_class", "no_data"),
                    "ts_rate_mm_yr": c.get("ts_rate_mm_yr"),
                }
                for c in clusters
            ] if clusters else [],
            "n_ew_deformation_lineaments": len(ew_defm["segments"]) if ew_defm else 0,
            "ew_dominant_strikes_deg": ew_defm["stats"]["dominant_strikes_deg"] if ew_defm else [],
            "attribution_summary": {
                cl.get("attribution_class", "undetermined"): sum(
                    1 for c in clusters if c.get("attribution_class") == cl.get("attribution_class")
                )
                for cl in clusters
            } if clusters else {},
            "attribution_details": [
                {
                    "id": c["id"],
                    "attribution_class": c.get("attribution_class", "undetermined"),
                    "attribution_confidence": c.get("attribution_confidence", 0),
                    "attribution_scores": c.get("attribution_scores", {}),
                }
                for c in clusters
            ] if clusters else [],
            "signal_quality": signal_quality,
        },
        "data_caveat": (
            "" if sufficient else
            f"Insufficient temporal coverage ({n_acq} dates, {span_days}d). "
            "Results for pipeline verification only."
        ),
        "created_at": created_at or datetime.now().isoformat(timespec="seconds"),
    }

    # ---- 矿床类型构造推理 ----
    try:
        from core.deposit_inference import infer_deposit_type
        # 构造 structural_stats
        fusion_structural_stats = {
            "n_lineaments": topo["stats"]["n_lineaments"],
            "lineament_density_mean": topo["stats"]["density_mean"],
            "dominant_strikes_deg": topo["stats"]["dominant_strikes_deg"],
            "elevation_range_m": [float(np.nanmin(dem)), float(np.nanmax(dem))]
            if dem is not None else [0, 9999],
        }
        # 归因统计
        attr_stats = {}
        for cl in clusters:
            cls = cl.get("attribution_class", "undetermined")
            attr_stats[cls] = attr_stats.get(cls, 0) + 1
        metadata["deposit_inference"] = infer_deposit_type(
            structural_stats=fusion_structural_stats,
            attribution_stats=attr_stats,
            lineament_details=topo.get("segments"),
        )
    except Exception as e:
        logger.warning(f"矿床类型推理失败(非致命): {e}")

    # 校验 metadata 字段完整性
    try:
        from core.structural_engine import _validate_metadata
        _validate_metadata(metadata)
    except Exception as e:
        logger.warning(f"metadata schema 校验失败(非致命): {e}")

    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info(f"[insar_fusion] Done -> {out_dir}  quality={signal_quality}  "
                f"topo={topo['stats']['n_lineaments']} def={defm['stats']['n_lineaments']} "
                f"active={n_active} subsidence={len(clusters)} "
                f"ew_def={len(ew_defm['segments']) if ew_defm else 0}")
    return metadata


def _span_days(start, end):
    try:
        s = datetime.strptime(start.replace("-", ""), "%Y%m%d")
        e = datetime.strptime(end.replace("-", ""), "%Y%m%d")
        return int((e - s).days)
    except Exception:
        return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="InSAR x structural fusion")
    ap.add_argument("insar_dir", help="MintPy dir OR geo-insar AOI dir")
    ap.add_argument("out_dir", help="Output dir")
    ap.add_argument("--aoi-name", default=None)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--structural-dir", default=None, help="geo-stru structural output dir (with lineaments.geojson)")
    args = ap.parse_args()
    md = run_fusion(args.insar_dir, args.out_dir, aoi_name=args.aoi_name,
                    seed=args.seed, make_plots=not args.no_plots,
                    structural_dir=args.structural_dir)
    print(json.dumps(md["fusion_stats"], ensure_ascii=False, indent=2))
