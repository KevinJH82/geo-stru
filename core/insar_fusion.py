"""
insar_fusion.py — InSAR 形变 × 遥感地质构造解译 融合

把 MintPy SBAS 输出(velocity/timeseries/temporalCoherence/geometry h5)接入构造解译:
  MintPy h5 → 标准化 GeoTIFF + 掩膜
            → A 形变线性体(LOS 速率梯度上找不连续)
            → A 地形线性体活动性打标(跨线剖面 LOS 速率差)
            → B 沉降连通域(相干负速率,圈采空/沉降)
            → 落盘 GeoJSON / GeoTIFF / 图件 + 遵循 schema 的 metadata.json

设计取舍:
  - 固定随机种子(默认 42),概率霍夫提线结果可复现(见 lineament.extract_lineaments rng_seed)。
  - 复用 core.terrain_utils(山体阴影/坡度)、core.lineament(线性体提取)、
    commons/insar_utils(LOS→垂直 / 相干掩膜)。不重造轮子。
  - commons 用 importlib 按文件加载,**不**污染 sys.path(与 structural_engine 同策略,
    避免 Flask 重载器监视整个 monorepo)。
  - metadata.json:source="geo-stru-insar-fusion",兼顾 structural_schema 的发现字段
    (aoi_bbox/crs/products) 与 insar_schema 的溯源字段(orbit/incidence/date_range)。

诚实边界:本模块只做**决策支持层**。短时序(<6 个月 / <10 景)下 velocity 以噪声为主,
fusion_stats.signal_quality 会标记 "insufficient",产物仅供管线验证与定性套合,不作定量构造结论。
"""

import os
import json
import importlib.util as _ilu
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import h5py
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import transform_bounds
from scipy import ndimage
from loguru import logger

from core.terrain_utils import TerrainProcessor
from core import lineament

SOURCE = "geo-stru-insar-fusion"
VERSION = "0.1.0"
DEFAULT_SEED = 42
DEFAULT_COH_THR = 0.7

# 可靠定量的最低门槛(低于此只作 POC/定性);来自 docs/InSAR融合方案.md 第八节
MIN_ACQUISITIONS = 10
MIN_TIMESPAN_DAYS = 180


# ---------------------------------------------------------------------------
# commons/insar_utils 按文件加载(零 sys.path 污染,镜像 structural_engine 的做法)
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
    except Exception as e:  # 优雅降级:本地实现兜底
        logger.warning(f"commons/insar_utils 加载失败({e}),用内置兜底")
        return None


_iu = _load_commons_insar_utils()


def _los_to_vertical(disp_los: np.ndarray, inc_deg: float) -> np.ndarray:
    if _iu is not None:
        return _iu.los_to_vertical(disp_los, inc_deg)
    return disp_los / np.cos(np.radians(inc_deg))


def _coherence_mask(coh: np.ndarray, thr: float) -> np.ndarray:
    if _iu is not None:
        return _iu.coherence_mask(coh, threshold=thr)
    return (coh >= thr) & ~np.isnan(coh)


# ---------------------------------------------------------------------------
# 1. 加载 MintPy 输出
# ---------------------------------------------------------------------------
def load_mintpy(mintpy_dir: str) -> Dict:
    """
    读取 MintPy SBAS 目录,返回统一的栅格 + 地理参考 + 溯源信息。

    需要:velocity.h5, temporalCoherence.h5, inputs/geometryGeo.h5
    可选:timeseries.h5(用于时序判据)、waterMask.h5
    """
    d = Path(mintpy_dir)

    def rd(fn, ds):
        with h5py.File(d / fn, "r") as f:
            return f[ds][:]

    with h5py.File(d / "velocity.h5", "r") as f:
        a = dict(f.attrs)
        vel = f["velocity"][:].astype(np.float64) * 1000.0  # m/yr -> mm/yr

    coh = rd("temporalCoherence.h5", "temporalCoherence").astype(np.float64)

    geom = d / "inputs" / "geometryGeo.h5"
    with h5py.File(geom, "r") as f:
        dem = f["height"][:].astype(np.float64)
        inc = f["incidenceAngle"][:].astype(np.float64)
        wmask = f["waterMask"][:].astype(bool) if "waterMask" in f else np.ones_like(dem, bool)

    ts, dates = None, []
    ts_path = d / "timeseries.h5"
    if ts_path.exists():
        with h5py.File(ts_path, "r") as f:
            ts = f["timeseries"][:].astype(np.float64) * 1000.0  # mm
            dates = [x.decode() if isinstance(x, bytes) else str(x) for x in f["date"][:]]

    H, W = vel.shape
    X0, Y0 = float(a["X_FIRST"]), float(a["Y_FIRST"])
    xs, ys = float(a["X_STEP"]), float(a["Y_STEP"])
    transform = Affine.translation(X0, Y0) * Affine.scale(xs, ys)
    epsg = int(a.get("EPSG", 4326))

    return {
        "vel": vel, "coh": coh, "dem": dem, "inc": inc, "wmask": wmask,
        "ts": ts, "dates": dates,
        "transform": transform, "epsg": epsg, "shape": (H, W),
        "pixel_m": (abs(xs), abs(ys)),
        "inc_mean": float(np.nanmean(inc)),
        "wavelength_m": float(a.get("WAVELENGTH", np.nan)),
        "orbit": a.get("ORBIT_DIRECTION"),
        "ref_xy": (int(a.get("REF_X", -1)), int(a.get("REF_Y", -1))),
        "start_date": str(a.get("START_DATE", "")),
        "end_date": str(a.get("END_DATE", "")),
        "attrs": a,
    }


# ---------------------------------------------------------------------------
# 2. 掩膜 / 梯度
# ---------------------------------------------------------------------------
def build_mask(data: Dict, coh_thr: float = DEFAULT_COH_THR) -> np.ndarray:
    """有效像元 = 相干性达标 & 非水 & 速率有限。复用 insar_utils.coherence_mask。"""
    return _coherence_mask(data["coh"], coh_thr) & data["wmask"] & np.isfinite(data["vel"])


def velocity_gradient(velm: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """LOS 速率梯度幅值 |∇v|(mm/yr/像元),无效处 NaN。"""
    fill = np.nanmean(velm[valid]) if valid.any() else 0.0
    gy, gx = np.gradient(np.where(np.isfinite(velm), velm, fill))
    grad = np.hypot(gx, gy)
    grad[~valid] = np.nan
    return grad


# ---------------------------------------------------------------------------
# 3. 线性体提取(A 形变 / 地形)
# ---------------------------------------------------------------------------
def deformation_lineaments(velm, valid, grad, transform, pixel_m, seed=DEFAULT_SEED) -> Dict:
    """在 LOS 速率场上找形变不连续(梯度门控),提取“形变线性体”。"""
    fin = velm[valid]
    lo, hi = np.nanpercentile(fin, 2), np.nanpercentile(fin, 98)
    vnorm = np.clip((np.nan_to_num(velm, nan=lo) - lo) / (hi - lo + 1e-9), 0, 1)
    return lineament.extract_lineaments(
        multidir_hillshade=vnorm, slope=np.nan_to_num(grad, nan=0.0),
        pixel_size_m=pixel_m, transform=transform, valid_mask=valid,
        canny_sigma=1.2, slope_gate_deg=0.0, slope_gate_pct=70.0, slope_gate_cap=1e9,
        min_length_m=120.0, density_window_m=800.0, rng_seed=seed,
    )


def topographic_lineaments(dem, transform, pixel_m, seed=DEFAULT_SEED) -> Dict:
    """DEM 多方位阴影 → 地形线性体(复用 terrain_utils)。"""
    multidir = TerrainProcessor.compute_multidirectional_hillshade(dem, pixel_m)
    slope_deg = TerrainProcessor.compute_slope(dem, pixel_m)
    lin = lineament.extract_lineaments(
        multidir_hillshade=multidir, slope=slope_deg,
        pixel_size_m=pixel_m, transform=transform, valid_mask=np.isfinite(dem),
        min_length_m=120.0, density_window_m=800.0, rng_seed=seed,
    )
    lin["_multidir"] = multidir
    return lin


# ---------------------------------------------------------------------------
# 4. 线性体活动性打标(A2)
# ---------------------------------------------------------------------------
def score_activity(segments, velm, coh, valid, transform, grad,
                   half_px: int = 2, n_stations: int = 5) -> List[Dict]:
    """
    对每条(地形)线性体作法向剖面,采两侧 LOS 速率差 ΔV_LOS。
    分类:dv>=参考尺度→“形变一致(活动?)”;否则→“仅地形(古/锁定?)”;样本不足→“无数据”。
    """
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
        nx, ny = -dr / L, dc / L  # 像素空间单位法向
        A, B, C = [], [], []
        for t in np.linspace(0.2, 0.8, n_stations):
            cc, rr = c0 + t * dc, r0 + t * dr
            for s in range(1, half_px + 1):
                for sign, bucket in ((+1, A), (-1, B)):
                    px = int(round(cc + sign * s * nx))
                    py = int(round(rr + sign * s * ny))
                    if 0 <= py < H and 0 <= px < W and valid[py, px]:
                        bucket.append(velm[py, px]); C.append(coh[py, px])
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
            "dv_los_mm_yr": None if not np.isfinite(dv) else round(dv, 2),
            "n_valid": nv, "coh": None if not np.isfinite(cm) else round(cm, 2),
            "activity_class": cls, "activity_score": round(score, 2),
            "p0": list(seg["p0"]), "p1": list(seg["p1"]),
        })
    return out


# ---------------------------------------------------------------------------
# 5. 沉降探测(B)
# ---------------------------------------------------------------------------
def detect_subsidence(velm, valid, transform, pixel_m, k_sigma: float = 1.5,
                      min_area_px: int = 5) -> Tuple[List[Dict], np.ndarray]:
    """相干负速率连通域 → 沉降簇(圈采空/沉降漏斗候选)。返回 (簇列表, 标签栅格)。"""
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
            "id": len(clusters), "area_px": area_px,
            "area_m2": round(area_px * pixel_m[0] * pixel_m[1], 1),
            "min_vel_mm_yr": round(float(np.nanmin(velm[m])), 2),
            "mean_vel_mm_yr": round(float(np.nanmean(velm[m])), 2),
            "centroid": [round(cx, 1), round(cy, 1)],
        })
    return clusters, lbl, thr


# ---------------------------------------------------------------------------
# 6. 落盘:GeoTIFF / GeoJSON
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
              "properties": {k: v for k, v in c.items() if k != "centroid"}}
             for c in clusters]
    fc = {"type": "FeatureCollection",
          "crs": {"type": "name", "properties": {"name": crs_str}}, "features": feats}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 7. 图件(可选)
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


def render_overlay(out_dir, data, valid, velm, grad, topo, defm, sub_lbl, multidir):
    plt = _setup_cjk_font()
    inv = ~data["transform"]

    def draw(ax, segs, color, lw=1.3, label=None):
        first = True
        for s in segs:
            c0, r0 = inv * s["p0"]; c1, r1 = inv * s["p1"]
            ax.plot([c0, c1], [r0, r1], color=color, lw=lw, label=label if first else None)
            first = False

    fig, axs = plt.subplots(1, 3, figsize=(17, 6))
    ax = axs[0]
    im = ax.imshow(velm, cmap="RdBu_r", vmin=-15, vmax=15)
    draw(ax, topo["segments"], "k", 1.3, "地形线性体")
    draw(ax, defm["segments"], "lime", 1.1, "形变线性体")
    ax.contour(sub_lbl > 0, levels=[0.5], colors="magenta", linewidths=1.5)
    rx, ry = data["ref_xy"]
    if rx >= 0:
        ax.plot(rx, ry, "ks", ms=6); ax.text(rx + 1, ry, "ref", fontsize=8)
    ax.set_title("LOS速率(mm/yr)+地形/形变线性体+沉降")
    ax.legend(loc="lower right", fontsize=8); plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axs[1]; ax.imshow(multidir, cmap="gray")
    draw(ax, topo["segments"], "red", 1.3); ax.set_title("DEM多方位阴影 + 地形线性体")

    ax = axs[2]
    im = ax.imshow(grad, cmap="magma", vmax=np.nanpercentile(grad, 98))
    draw(ax, defm["segments"], "cyan", 1.1)
    ax.set_title("LOS速率梯度|∇v| + 形变线性体"); plt.colorbar(im, ax=ax, fraction=0.046)

    fig.tight_layout()
    p = os.path.join(out_dir, "fusion_overlay.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    return "fusion_overlay.png"


def render_timeseries(out_dir, data, velm, sub_lbl):
    if data["ts"] is None or not data["dates"]:
        return None
    plt = _setup_cjk_font()
    if (sub_lbl > 0).any():
        m = sub_lbl == 1
        ys_, xs_ = np.where(m)
        j = int(np.argmin(velm[m])); py, px = ys_[j], xs_[j]
    else:
        py, px = np.unravel_index(np.nanargmin(velm), velm.shape)
    fig, ax = plt.subplots(figsize=(7, 4))
    series = data["ts"][:, py, px] - data["ts"][0, py, px]
    ax.plot(range(len(data["dates"])), series, "o-")
    ax.set_xticks(range(len(data["dates"]))); ax.set_xticklabels(data["dates"], rotation=45, fontsize=8)
    ax.set_ylabel("累计LOS形变 (mm)"); ax.set_title(f"最大形变点时序 (px={px},{py})")
    ax.grid(alpha=0.3); fig.tight_layout()
    p = os.path.join(out_dir, "timeseries_maxpoint.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    return "timeseries_maxpoint.png"


# ---------------------------------------------------------------------------
# 8. 主编排
# ---------------------------------------------------------------------------
def run_fusion(mintpy_dir: str, out_dir: str, aoi_name: Optional[str] = None,
               seed: int = DEFAULT_SEED, coh_thr: float = DEFAULT_COH_THR,
               make_plots: bool = True, created_at: Optional[str] = None) -> Dict:
    """
    端到端融合。读取 MintPy 目录,产出 GeoTIFF/GeoJSON/图件 + metadata.json。
    返回 metadata dict。
    """
    os.makedirs(out_dir, exist_ok=True)
    data = load_mintpy(mintpy_dir)
    H, W = data["shape"]
    transform, epsg, pixel_m = data["transform"], data["epsg"], data["pixel_m"]
    crs_str = f"EPSG:{epsg}"

    valid = build_mask(data, coh_thr)
    velm = np.where(valid, data["vel"], np.nan)
    grad = velocity_gradient(velm, valid)
    vert = np.where(valid, _los_to_vertical(velm, data["inc_mean"]), np.nan)

    fin = velm[valid]
    logger.info(f"[insar_fusion] 有效 {valid.sum()}/{H*W} ({100*valid.sum()/(H*W):.1f}%) "
                f"LOS {fin.min():.1f}~{fin.max():.1f} mm/yr std {fin.std():.2f}")

    # A 线性体
    defm = deformation_lineaments(velm, valid, grad, transform, pixel_m, seed)
    topo = topographic_lineaments(data["dem"], transform, pixel_m, seed)
    multidir = topo.pop("_multidir")
    # A2 活动性
    activity = score_activity(topo["segments"], velm, data["coh"], valid, transform, grad)
    n_active = sum(1 for x in activity if x["activity_class"].startswith("形变一致"))
    # B 沉降
    clusters, sub_lbl, sub_thr = detect_subsidence(velm, valid, transform, pixel_m)

    # ---- 落盘栅格 ----
    _write_gtiff(os.path.join(out_dir, "los_velocity_mm_yr.tif"), velm, transform, epsg)
    _write_gtiff(os.path.join(out_dir, "vertical_velocity_mm_yr.tif"), vert, transform, epsg)
    _write_gtiff(os.path.join(out_dir, "velocity_gradient.tif"), grad, transform, epsg)
    _write_gtiff(os.path.join(out_dir, "temporal_coherence.tif"), data["coh"], transform, epsg)

    # ---- 落盘矢量 ----
    lineament.write_lineaments_geojson(defm["segments"],
                                       os.path.join(out_dir, "deformation_lineaments.geojson"), crs=crs_str)
    lineament.write_lineaments_geojson(topo["segments"],
                                       os.path.join(out_dir, "topographic_lineaments.geojson"), crs=crs_str)
    _write_line_geojson(activity, os.path.join(out_dir, "lineaments_activity.geojson"), crs_str)
    _write_point_geojson(clusters, os.path.join(out_dir, "subsidence_clusters.geojson"), crs_str)
    lineament.plot_rose_diagram(topo["segments"], os.path.join(out_dir, "rose_topographic.png"), "地形线性体走向")
    lineament.plot_rose_diagram(defm["segments"], os.path.join(out_dir, "rose_deformation.png"), "形变线性体走向")

    products = {
        "los_velocity_mm_yr": "los_velocity_mm_yr.tif",
        "vertical_velocity_mm_yr": "vertical_velocity_mm_yr.tif",
        "velocity_gradient": "velocity_gradient.tif",
        "temporal_coherence": "temporal_coherence.tif",
        "deformation_lineaments_geojson": "deformation_lineaments.geojson",
        "topographic_lineaments_geojson": "topographic_lineaments.geojson",
        "lineaments_activity_geojson": "lineaments_activity.geojson",
        "subsidence_clusters_geojson": "subsidence_clusters.geojson",
        "rose_topographic_png": "rose_topographic.png",
        "rose_deformation_png": "rose_deformation.png",
    }
    if make_plots:
        products["overlay_png"] = render_overlay(out_dir, data, valid, velm, grad, topo, defm, sub_lbl, multidir)
        tsp = render_timeseries(out_dir, data, velm, sub_lbl)
        if tsp:
            products["timeseries_png"] = tsp

    # ---- 信号质量门(诚实边界) ----
    n_acq = len(data["dates"]) or 0
    span_days = _span_days(data["start_date"], data["end_date"])
    sufficient = (n_acq >= MIN_ACQUISITIONS) and (span_days >= MIN_TIMESPAN_DAYS)
    signal_quality = "ok" if sufficient else "insufficient"

    # ---- bbox(lon/lat,供 broker 按研究区相交发现) ----
    left, top = transform * (0, 0)
    right, bottom = transform * (W, H)
    bbox_ll = list(transform_bounds(CRS.from_epsg(epsg), CRS.from_epsg(4326),
                                    left, bottom, right, top))

    metadata = {
        "source": SOURCE,
        "source_version": VERSION,
        "run_id": os.path.basename(os.path.normpath(out_dir)),
        "aoi_name": aoi_name or Path(mintpy_dir).name,
        "aoi_bbox": [round(x, 6) for x in bbox_ll],
        "crs": crs_str,
        "grid": [H, W],
        "pixel_size_m": list(pixel_m),
        "seed": seed,
        "insar_provenance": {
            "source_insar": "mintpy_sbas",
            "orbit_direction": data["orbit"],
            "incidence_angle_mean": round(data["inc_mean"], 3),
            "wavelength_m": data["wavelength_m"],
            "reference_pixel": list(data["ref_xy"]),
            "date_range": [data["start_date"], data["end_date"]],
            "n_acquisitions": n_acq,
            "timespan_days": span_days,
        },
        "products": products,
        "fusion_stats": {
            "valid_ratio": round(float(valid.sum() / (H * W)), 3),
            "coherence_mean_valid": round(float(data["coh"][valid].mean()), 3),
            "los_velocity_mm_yr": {
                "min": round(float(fin.min()), 1), "max": round(float(fin.max()), 1),
                "mean": round(float(fin.mean()), 2), "std": round(float(fin.std()), 2)},
            "n_topographic_lineaments": topo["stats"]["n_lineaments"],
            "n_deformation_lineaments": defm["stats"]["n_lineaments"],
            "topographic_dominant_strikes_deg": topo["stats"]["dominant_strikes_deg"],
            "deformation_dominant_strikes_deg": defm["stats"]["dominant_strikes_deg"],
            "n_active_consistent_lineaments": n_active,
            "n_subsidence_clusters": len(clusters),
            "subsidence_threshold_mm_yr": round(sub_thr, 2),
            "signal_quality": signal_quality,
        },
        "data_caveat": (
            "" if sufficient else
            f"时序不足(景数 {n_acq}<{MIN_ACQUISITIONS} 或 跨度 {span_days}d<{MIN_TIMESPAN_DAYS}d):"
            "velocity 以噪声为主,本结果仅供管线验证与定性套合,不作定量构造结论。"
            "建议扩展至 ≥1 年、≥20–30 景,最好升+降双轨。"
        ),
        "created_at": created_at or datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info(f"[insar_fusion] 完成 → {out_dir}  signal_quality={signal_quality}  "
                f"地形线性体 {topo['stats']['n_lineaments']} / 形变线性体 {defm['stats']['n_lineaments']} / "
                f"活动 {n_active} / 沉降簇 {len(clusters)}")
    return metadata


def _span_days(start: str, end: str) -> int:
    try:
        s = datetime.strptime(start, "%Y%m%d")
        e = datetime.strptime(end, "%Y%m%d")
        return int((e - s).days)
    except Exception:
        return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="InSAR×构造解译 融合(MintPy 目录 → 融合产物)")
    ap.add_argument("mintpy_dir", help="MintPy SBAS 输出目录(含 velocity.h5 等)")
    ap.add_argument("out_dir", help="融合产物输出目录")
    ap.add_argument("--aoi-name", default=None)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--coh-thr", type=float, default=DEFAULT_COH_THR)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    md = run_fusion(args.mintpy_dir, args.out_dir, aoi_name=args.aoi_name,
                    seed=args.seed, coh_thr=args.coh_thr, make_plots=not args.no_plots)
    print(json.dumps(md["fusion_stats"], ensure_ascii=False, indent=2))
