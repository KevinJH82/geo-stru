"""
delivery.py — 从交付数据库(冬季)按 ROI/项目定位 geo-stru 所需卫星数据

替代"上传卫星数据 ZIP"流程:大 ZIP 上传解压太慢。改为复用 geo-analyser 的
delivery_project(交付库 + 冬季子目录约定),按已上传的 ROI 文件名定位项目,
直接取冬季交付里的 DEM.tif(地形分析必需)与 Landsat 波段目录(2-1C 可选叠加)。

零 sys.path 污染:用 importlib 从绝对路径加载 geo-analyser/delivery_project.py
(与 commons/aoi 复用 geo-downloader 解析器同一思路)。
"""

import importlib.util
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


def resolve_project_dir(name_or_filename: str) -> Optional[Path]:
    """
    定位交付项目目录,兼容两种入参:
      - 项目名(下拉框传来,可能含小数点如 "...6.82km2_..."):优先按精确目录名匹配,
        避免 Path().stem 把 ".82km2_..." 误当扩展名;
      - ROI 文件名(如 "X.ovkml"):回退到 delivery_project 的取主名匹配。
    """
    if not name_or_filename:
        return None
    try:
        dp = _load()
        cand = dp.DELIVERY_ROOT / name_or_filename
        if cand.is_dir():
            return cand
        return dp.resolve_project_dir(name_or_filename)
    except Exception:
        return None


def locate_winter_data(project_dir) -> Dict[str, Optional[str]]:
    """
    在项目的冬季子目录定位 geo-stru 所需数据。

    Returns
    -------
    {winter, dem, landsat_dir, landsat_sensor} —— dem/landsat_dir 不存在则为 None。
    DEM 为地形分析必需;Landsat 仅 2-1C 叠加可选。
    """
    out = {"winter": None, "dem": None, "landsat_dir": None, "landsat_sensor": None}
    try:
        dp = _load()
        pd = Path(project_dir)
        winter = dp._winter_dir(pd)
        if not winter:
            return out
        out["winter"] = str(winter)
        dem = winter / "DEM.tif"
        if dem.exists():
            out["dem"] = str(dem)
        for sk in ("Landsat8", "Landsat9"):
            sub = dp._find_sensor_subdir(winter, sk)
            if sub:
                out["landsat_dir"] = str(sub)
                out["landsat_sensor"] = sk
                break
    except Exception:
        pass
    return out
