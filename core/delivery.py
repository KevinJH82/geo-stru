"""
delivery.py — 从交付数据库(冬季)按 ROI/项目定位 geo-stru 所需卫星数据

替代"上传卫星数据 ZIP"流程:大 ZIP 上传解压太慢。改为复用 geo-analyser 的
delivery_project(交付库 + 冬季子目录约定),按已上传的 ROI 文件名定位项目,
直接取冬季交付里的 DEM.tif(地形分析必需)与 Landsat/S2 波段目录(2-1C 可选叠加)。

零 sys.path 污染:用 importlib 从绝对路径加载 geo-analyser/delivery_project.py
(与 commons/aoi 复用 geo-downloader 解析器同一思路)。
"""

import importlib.util
import re
from pathlib import Path
from typing import Dict, List, Optional

_DP_PATH = Path("/opt/deepexplor-services/geo-analyser/delivery_project.py")
_dp = None


def _load():
    """从绝对路径加载 geo-analyser 的 delivery_project,缓存。"""
    global _dp
    if _dp is not None:
        return _dp
    if not _DP_PATH.exists():
        raise ImportError(f"找不到 {_DP_PATH}(geo-analyser 缺失,无法复用交付库定位)")
    spec = importlib.util.spec_from_file_location("geo_analyser_delivery_project", str(_DP_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _dp = module
    return module


def list_projects() -> List[Dict[str, str]]:
    """列出交付根目录下所有项目(供前端下拉)。失败/未挂载返回空。"""
    try:
        return _load().list_projects()
    except Exception:
        return []


def resolve_project_dir(name_or_filename: str, roi_geojson=None) -> Optional[Path]:
    """
    定位交付项目目录,兼容两种入参:
      - 项目名(下拉框传来,可能含小数点如 "...6.82km2_..."):优先按精确目录名匹配,
        避免 Path().stem 把 ".82km2_..." 误当扩展名;
      - ROI 文件名(如 "X.ovkml"):回退到 delivery_project 的取主名匹配。
    roi_geojson(可选):给定则名字匹配失败时按几何覆盖兜底定位(KML 改名也能命中)。
    """
    if not name_or_filename:
        return None
    try:
        dp = _load()
        cand = dp.DELIVERY_ROOT / name_or_filename
        if cand.is_dir():
            return cand
        return dp.resolve_project_dir(name_or_filename, roi_geojson)
    except Exception:
        return None


def resolve_project_dir_verbose(name_or_filename: str, roi_geojson=None, delivery_id: str = "") -> dict:
    """返回 {dir, method, candidates, delivery_id},供未命中时列候选友好报错。

    delivery_id(门户绑定):给定则优先按 ID 定位。
    """
    try:
        dp = _load()
        if not delivery_id:
            cand = dp.DELIVERY_ROOT / (name_or_filename or "")
            if cand.is_dir():
                return {"dir": cand, "method": "exact", "candidates": []}
        return dp.resolve_project_dir_verbose(name_or_filename, roi_geojson, delivery_id)
    except Exception:
        return {"dir": None, "method": "none", "candidates": []}


def _find_dem(directory: Path) -> Optional[Path]:
    """
    在目录下模糊匹配 DEM 文件(不区分大小写,支持多种命名)。

    匹配规则(按优先级):
      1. 精确: DEM.tif
      2. 前缀: dem*.tif / DEM*.tif
      3. 任意位置含 dem: *dem*.tif
    """
    if not directory.is_dir():
        return None
    # 优先精确匹配
    exact = directory / "DEM.tif"
    if exact.exists():
        return exact
    # 模糊匹配(不区分大小写)
    candidates = []
    for f in directory.iterdir():
        if not f.is_file():
            continue
        name_lower = f.name.lower()
        if not name_lower.endswith(('.tif', '.tiff')):
            continue
        if name_lower == 'dem.tif':
            return f  # 优先精确
        if name_lower.startswith('dem'):
            candidates.append(f)
        elif 'dem' in name_lower:
            candidates.append(f)
    if candidates:
        # 按文件名长度排序(短名优先,如 dem.tif > dem_12m.tif > dem_srtm_30m.tif)
        candidates.sort(key=lambda p: len(p.name))
        return candidates[0]
    return None


def _find_sensor_dir(winter_dir: Path, sensor_patterns: List[str]) -> Optional[Path]:
    """
    在冬季目录下查找传感器子目录(支持模糊匹配)。

    Args:
        winter_dir: 冬季子目录
        sensor_patterns: 要匹配的模式列表(如 ["Landsat8", "Landsat9"])
    """
    # 优先用 delivery_project 的精确匹配
    try:
        dp = _load()
        for pat in sensor_patterns:
            sub = dp._find_sensor_subdir(winter_dir, pat)
            if sub:
                return sub
    except Exception:
        pass
    # 回退:目录名模糊匹配
    if winter_dir.is_dir():
        for child in sorted(winter_dir.iterdir()):
            if not child.is_dir():
                continue
            name_lower = child.name.lower()
            for pat in sensor_patterns:
                if pat.lower() in name_lower:
                    return child
    return None


def locate_winter_data(project_dir) -> Dict[str, Optional[str]]:
    """
    在项目的冬季子目录定位 geo-stru 所需数据。

    Returns
    -------
    {winter, dem, landsat_dir, landsat_sensor} —— dem/landsat_dir 不存在则为 None。
    DEM 为地形分析必需;Landsat/S2 仅 2-1C 叠加可选。
    """
    out = {"winter": None, "dem": None, "landsat_dir": None, "landsat_sensor": None}
    try:
        dp = _load()
        pd = Path(project_dir)
        winter = dp._winter_dir(pd)
        if not winter:
            return out
        out["winter"] = str(winter)
        # DEM:模糊匹配
        dem = _find_dem(winter)
        if dem:
            out["dem"] = str(dem)
        # 传感器: Landsat8/9 → Sentinel-2 回退
        landsat_dir = _find_sensor_dir(winter, ["Landsat8", "Landsat9"])
        if landsat_dir:
            out["landsat_dir"] = str(landsat_dir)
            out["landsat_sensor"] = "Landsat"
        else:
            # 回退 Sentinel-2
            s2_dir = _find_sensor_dir(winter, ["Sentinel2", "Sentinel-2", "S2"])
            if s2_dir:
                out["landsat_dir"] = str(s2_dir)
                out["landsat_sensor"] = "Sentinel2"
    except Exception:
        pass
    return out
