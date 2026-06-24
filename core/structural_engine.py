"""
遥感地质构造解译图生成引擎

解析KML/ROI文件，调用TerrainProcessor生成三类构造解译图
"""

import os
import re
import json
import importlib
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import zipfile
import tempfile
import rasterio
from datetime import datetime
from loguru import logger
from rasterio.transform import Affine
from typing import List, Tuple, Optional, Dict
from pathlib import Path

from config import __version__
from core.terrain_utils import TerrainProcessor
from core.structural_map_viz import StructuralMapVisualizer
from core import lineament


# ---------------------------------------------------------------------------
# metadata.json schema 校验
# ---------------------------------------------------------------------------
_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "commons" / "structural_schema.json"


def _validate_metadata(metadata: dict) -> None:
    """
    用 structural_schema.json 校验 metadata 字段完整性。
    校验失败只打警告不阻断——避免阻塞产物落盘。
    """
    try:
        import jsonschema
        if not _SCHEMA_PATH.exists():
            logger.warning(f"schema 文件不存在,跳过校验: {_SCHEMA_PATH}")
            return
        with open(_SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema = json.load(f)
        jsonschema.validate(instance=metadata, schema=schema)
        logger.debug("metadata.json schema 校验通过")
    except ImportError:
        pass  # jsonschema 不可用,跳过
    except Exception as e:
        logger.warning(f"metadata.json schema 校验失败(非致命): {e}")

# 复用平台共享的 AOI 解析:按文件路径 importlib 加载 commons/aoi.py,真正零污染。
# 关键:*不*把仓库根插入 sys.path —— 否则 Flask 调试重载器(StatReloader 会
# 遍历整个 sys.path 目录树)会监视整个 monorepo,任何兄弟子系统(geo-analyser 等)
# 的改动都会重启本服务并杀掉正在跑的后台任务。commons/aoi.py 仅依赖标准库,可独立加载。
import importlib.util as _importlib_util


def _load_commons_parse_aoi():
    aoi_path = Path(__file__).resolve().parents[2] / "commons" / "aoi.py"
    if not aoi_path.exists():
        return None
    try:
        spec = _importlib_util.spec_from_file_location("geostru_commons_aoi", aoi_path)
        mod = _importlib_util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, "parse_aoi", None)
    except Exception:  # commons 不可用时优雅降级到文件名
        return None


_commons_parse_aoi = _load_commons_parse_aoi()


def elem_tag(elem):
    """获取XML元素的本地标签名（去掉命名空间）"""
    tag = elem.tag
    if '}' in tag:
        tag = tag.split('}', 1)[1]
    return tag


class StructuralEngine:

    @staticmethod
    def parse_kml_polygon(kml_path: str) -> List[Tuple[float, float]]:
        """从KML/KMZ文件中提取多边形坐标"""
        coords = []

        if kml_path.lower().endswith('.kmz'):
            with zipfile.ZipFile(kml_path, 'r') as z:
                kml_files = [f for f in z.namelist() if f.endswith('.kml')]
                if not kml_files:
                    raise ValueError("KMZ中未找到KML文件")
                with tempfile.NamedTemporaryFile(suffix='.kml', delete=False, mode='w') as tmp:
                    tmp.write(z.read(kml_files[0]))
                    kml_path = tmp.name

        with open(kml_path, 'r', encoding='utf-8') as f:
            content = f.read()

        content = re.sub(r'\sxmlns[^"]*"[^"]*"', '', content)
        content = content.replace('kml:', '').replace('gx:', '')

        root = ET.fromstring(content)

        def extract_coords(elem):
            result = []
            if elem.text:
                raw = elem.text.strip()
                for pair in raw.split():
                    parts = pair.split(',')
                    if len(parts) >= 2:
                        try:
                            lon, lat = float(parts[0]), float(parts[1])
                            result.append((lon, lat))
                        except ValueError:
                            continue
            if len(result) > 1 and result[0] == result[-1]:
                result = result[:-1]
            return result

        for coord_elem in root.iter():
            tag = elem_tag(coord_elem)
            if tag == 'polygon' or tag == 'LinearRing':
                for child in coord_elem.iter():
                    if elem_tag(child) == 'coordinates':
                        c = extract_coords(child)
                        if len(c) >= 3:
                            return c

        for coord_elem in root.iter():
            if elem_tag(coord_elem) == 'LineString':
                for child in coord_elem.iter():
                    if elem_tag(child) == 'coordinates':
                        c = extract_coords(child)
                        if len(c) >= 3:
                            return c

        all_points = []
        for coord_elem in root.iter():
            if elem_tag(coord_elem) == 'coordinates':
                c = extract_coords(coord_elem)
                all_points.extend(c)
        if len(all_points) >= 3:
            return all_points

        raise ValueError("KML文件中未找到足够的多边形坐标（至少需要3个点）")

    @staticmethod
    def parse_roi_polygon(roi_path: str) -> List[Tuple[float, float]]:
        """从ROI文件(Excel/CSV)中提取坐标构成多边形"""
        ext = Path(roi_path).suffix.lower()

        if ext in ('.xlsx', '.xls'):
            df = pd.read_excel(roi_path)
        elif ext == '.csv':
            df = pd.read_csv(roi_path)
        else:
            raise ValueError(f"不支持的ROI文件格式: {ext}")

        lon_col = lat_col = None
        for col in df.columns:
            cl = str(col).lower()
            if any(k in cl for k in ['lon', 'lng', 'longitude', '经度', 'x']):
                lon_col = col
            elif any(k in cl for k in ['lat', 'latitude', '纬度', 'y']):
                lat_col = col

        if lon_col is None or lat_col is None:
            lon_col, lat_col = df.columns[0], df.columns[1]

        lons = pd.to_numeric(df[lon_col], errors='coerce').dropna().values
        lats = pd.to_numeric(df[lat_col], errors='coerce').dropna().values

        if len(lons) < 3:
            raise ValueError("ROI坐标点数不足3个，无法构成多边形")

        coords = [(float(lo), float(la)) for lo, la in zip(lons, lats)]
        return coords

    @staticmethod
    def detect_file_type(file_path: str) -> str:
        """自动判断文件类型"""
        ext = Path(file_path).suffix.lower()
        if ext in ('.kml', '.kmz', '.ovkml'):
            return 'kml'
        elif ext in ('.xlsx', '.xls', '.csv'):
            return 'roi'
        raise ValueError(f"不支持的文件类型: {ext}，请上传 KML/KMZ/OVKML 或 Excel/CSV 文件")

    @staticmethod
    def parse_polygon(file_path: str) -> List[Tuple[float, float]]:
        """根据文件类型自动解析多边形坐标"""
        ftype = StructuralEngine.detect_file_type(file_path)
        if ftype == 'kml':
            return StructuralEngine.parse_kml_polygon(file_path)
        else:
            return StructuralEngine.parse_roi_polygon(file_path)

    @staticmethod
    def get_aoi_name(file_path: str) -> str:
        """
        提取 AOI 名称,用于按区域命名输出目录(与平台其它子系统一致)。
        优先复用 commons.aoi.parse_aoi(读 KML <name>),失败则回退到文件名。
        """
        if _commons_parse_aoi is not None:
            try:
                _bbox, _geom, area_name = _commons_parse_aoi(file_path)
                if area_name:
                    return str(area_name)
            except Exception:
                pass
        return Path(file_path).stem

    @staticmethod
    def generate_maps(
        dem_path: str,
        polygon_coords: List[Tuple[float, float]],
        output_dir: str,
        landsat_dir: Optional[str] = None,
        azimuth: float = 315,
        altitude: float = 45,
        use_landsat: bool = True,
        log_callback=None,
        aoi_name: Optional[str] = None,
        created_at: Optional[str] = None,
        mineral_hint: Optional[str] = None,
        trace_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        task_code: Optional[str] = None,
    ) -> Dict:
        """
        生成三类遥感地质构造解译图

        Returns:
            dict: {
                'output_files': {
                    'hillshade': '图2-1A_....png',
                    'aspect': '图2-1B_....png',
                    'terrain': '图2-1C_....png',
                },
                'result_dir': str,
                'polygon_coords': list,
            }
        """
        def log(msg, level='INFO'):
            if log_callback:
                log_callback(msg, level)

        os.makedirs(output_dir, exist_ok=True)

        log("加载DEM数据...")
        dem_info = TerrainProcessor.load_dem(dem_path)
        dem_data = dem_info['data']
        dem_transform = dem_info['transform']
        dem_crs = dem_info['crs']
        pixel_size_m = dem_info['pixel_size_m']

        log(f"裁剪到工作区 ({len(polygon_coords)} 个顶点)...")
        dem_clipped, transform_clipped = TerrainProcessor.clip_to_workarea(
            dem_data, dem_transform, polygon_coords,
        )
        log(f"裁剪后DEM: {dem_clipped.shape}, 高程 {np.nanmin(dem_clipped):.1f}-{np.nanmax(dem_clipped):.1f}m")

        # 垂直夸张仅用于制图(PNG)增强地形可读性;落盘的分析栅格用 z=1(保留真实坡度量值)。
        z_display = 5
        log(f"计算山体阴影 (方位角={azimuth}°, 高度角={altitude}°, 制图夸张={z_display}x)...")

        hillshade = TerrainProcessor.compute_hillshade(  # 制图用(夸张)
            dem_clipped, pixel_size_m, azimuth=azimuth, altitude=altitude,
            z_factor=z_display,
        )
        hillshade_analytic = TerrainProcessor.compute_hillshade(  # 落盘用(无夸张)
            dem_clipped, pixel_size_m, azimuth=azimuth, altitude=altitude,
            z_factor=1,
        )

        log("计算坡向...")
        aspect = TerrainProcessor.compute_aspect(dem_clipped, pixel_size_m)

        log("计算坡度...")
        slope = TerrainProcessor.compute_slope(dem_clipped, pixel_size_m)

        log("计算光照无关地形增强(天空视域因子/开度/曲率)...")
        svf, openness = TerrainProcessor.compute_skyview_openness(dem_clipped, pixel_size_m)
        curvature = TerrainProcessor.compute_curvature(dem_clipped, pixel_size_m)

        log("提取线性体(断裂/线性构造)...")
        valid_mask = np.isfinite(dem_clipped)
        multidir = TerrainProcessor.compute_multidirectional_hillshade(
            dem_clipped, pixel_size_m)
        multidir = np.where(valid_mask, multidir, np.nan)
        # 最短线性体长度随 AOI 尺度自适应:小 AOI(如 1km²)用更短阈值,否则会被全部滤掉;
        # 下限 120m(防噪声),上限沿用 300m。
        _short_extent_m = min(dem_clipped.shape[0] * pixel_size_m[1],
                              dem_clipped.shape[1] * pixel_size_m[0])
        _min_len = max(120.0, min(300.0, 0.12 * _short_extent_m))
        lin = lineament.extract_lineaments(
            multidir, slope, pixel_size_m, transform_clipped, valid_mask=valid_mask,
            min_length_m=_min_len)
        log(f"线性体: {lin['stats']['n_lineaments']} 条, "
            f"主方向 {lin['stats']['dominant_strikes_deg']}")

        landsat_rgb = None
        if use_landsat and landsat_dir and os.path.isdir(landsat_dir):
            # 波段发现:兼容多种文件命名(B7.tif / B07.tif / SR_B7.TIF / B7_30m.tif 等)
            def _resolve_band(band_name, directory):
                """模糊匹配波段文件(不区分大小写,支持前缀/后缀变体)。"""
                bn = band_name.lower()
                for f in sorted(os.listdir(directory)):
                    fl = f.lower()
                    if not fl.endswith(('.tif', '.tiff')):
                        continue
                    # 精确: B7.tif
                    if fl == f'{bn}.tif' or fl == f'{bn}.tiff':
                        return os.path.join(directory, f)
                    # 前缀: B07.tif, SR_B7.tif
                    stem = fl.replace('.tif', '').replace('.tiff', '')
                    if stem == bn or stem.endswith(bn) or stem.startswith(bn):
                        return os.path.join(directory, f)
                    # 数字匹配: B07 → B7
                    try:
                        num = bn.lstrip('b')
                        if num and (stem == f'b{int(num):02d}' or stem.endswith(f'b{int(num):02d}')):
                            return os.path.join(directory, f)
                    except ValueError:
                        pass
                return None

            # 组合优先级: SWIR地质 > 真彩色; Landsat > S2
            combos = [
                (['B7', 'B6', 'B4'], 'Landsat SWIR 7-6-4'),
                (['B4', 'B3', 'B2'], 'Landsat 真彩色 4-3-2'),
                (['B12', 'B11', 'B8A'], 'Sentinel-2 SWIR 12-11-8A'),
                (['B11', 'B8A', 'B04'], 'Sentinel-2 短波 11-8A-4'),
            ]
            for combo, label in combos:
                paths = [_resolve_band(b, landsat_dir) for b in combo]
                if not all(paths):
                    continue
                tag = '-'.join(b[1:] for b in combo)
                log(f"加载波段组合 {label} 并重投影...")
                try:
                    rgb_bands = [
                        TerrainProcessor.percent_stretch(
                            TerrainProcessor.resample_to_dem_grid(
                                p, transform_clipped, dem_crs, dem_clipped.shape), 2)
                        for p in paths
                    ]
                    landsat_rgb = np.stack(rgb_bands, axis=-1)
                except Exception as e:
                    log(f"波段加载失败: {e}，尝试下一组合", 'WARNING')
                break
            else:
                log("未找到可用的 Landsat/S2 波段组合,使用纯地形渲染", 'WARNING')

        log("计算地形渲染...")
        terrain_render = TerrainProcessor.compute_terrain_render(
            dem_clipped, pixel_size_m, landsat_rgb=landsat_rgb,
        )

        log("生成图2-1A 山体阴影遥感地质构造解译图...")
        # 图幅纵横比随 AOI 地面纵横比自适应,使地图主体填满画布(小/方形 AOI 不再被
        # 固定 18×14 画布"信封化"成中间一小块)。
        _h, _w = dem_clipped.shape
        _gw = _w * pixel_size_m[0]
        _gh = _h * pixel_size_m[1]
        _ratio = (_gw / _gh) if _gh > 0 else 1.0
        _L = 14.0
        if _ratio >= 1:
            _figsize = (_L, max(9.0, _L / _ratio))
        else:
            _figsize = (max(9.0, _L * _ratio), _L)
        viz = StructuralMapVisualizer(figsize=_figsize, dpi=300)

        file_a = "图2-1A_山体阴影遥感地质构造解译图.png"
        viz.plot_hillshade_map(
            hillshade, dem_clipped, transform_clipped,
            output_path=os.path.join(output_dir, file_a),
            title=f"图2-1A  1:5万山体阴影（{int(azimuth)}°方向）遥感地质构造解译图",
            work_area=polygon_coords,
            lineaments=lin['segments'],
        )

        log("生成图2-1B 坡向分析遥感地质构造解译图...")
        file_b = "图2-1B_坡向分析遥感地质构造解译图.png"
        viz.plot_aspect_map(
            aspect, dem_clipped, transform_clipped,
            output_path=os.path.join(output_dir, file_b),
            work_area=polygon_coords,
            hillshade=hillshade,
            slope=slope,
        )

        log("生成图2-1C 地形渲染遥感地质构造解译图...")
        file_c = "图2-1C_地形渲染遥感地质构造解译图.png"
        viz.plot_terrain_render_map(
            terrain_render, dem_clipped, transform_clipped,
            output_path=os.path.join(output_dir, file_c),
            work_area=polygon_coords,
        )

        # 落盘分析级 GeoTIFF:山体阴影用无夸张版本(z=1),量值才有物理意义。
        for name, data in [('hillshade_315.tif', hillshade_analytic),
                           ('aspect.tif', aspect.astype(np.float32)),
                           ('slope.tif', slope.astype(np.float32)),
                           ('svf.tif', svf.astype(np.float32)),
                           ('openness.tif', openness.astype(np.float32)),
                           ('curvature.tif', curvature.astype(np.float32))]:
            path = os.path.join(output_dir, name)
            profile = dem_info['profile'].copy()
            # nodata 须与目标 dtype 兼容：源DEM的 nan 仅对浮点有效，
            # 整型波段(如 uint8 山体阴影)需清掉 nan，否则 rasterio 报错。
            nodata = np.nan if np.issubdtype(data.dtype, np.floating) else None
            profile.update(width=data.shape[1], height=data.shape[0],
                           transform=transform_clipped, dtype=str(data.dtype),
                           nodata=nodata)
            with rasterio.open(path, 'w', **profile) as dst:
                dst.write(data, 1)

        # ---- 阶段2 构造提取:落盘下游可消费的线性体产物 ----
        lin_products = {}
        if lin['stats']['n_lineaments'] > 0:
            for name, data in [('distance_to_lineament.tif', lin['distance_m']),
                               ('lineament_density.tif', lin['density'].astype(np.float32))]:
                path = os.path.join(output_dir, name)
                profile = dem_info['profile'].copy()
                profile.update(width=data.shape[1], height=data.shape[0],
                               transform=transform_clipped, dtype='float32', nodata=np.nan)
                with rasterio.open(path, 'w', **profile) as dst:
                    dst.write(data.astype(np.float32), 1)
            crs_for_geojson = dem_crs.to_string() if dem_crs is not None else "EPSG:4326"
            lineament.write_lineaments_geojson(
                lin['segments'], os.path.join(output_dir, 'lineaments.geojson'),
                crs=crs_for_geojson)
            lineament.plot_rose_diagram(
                lin['segments'], os.path.join(output_dir, 'rose_diagram.png'))
            lin_products = {
                'distance_to_lineament': 'distance_to_lineament.tif',
                'lineament_density': 'lineament_density.tif',
                'lineaments_geojson': 'lineaments.geojson',
                'rose_diagram': 'rose_diagram.png',
            }

        # ---- 阶段0 契约层:落盘 metadata.json,使产物可被下游(bbox 相交 + AOI 目录)发现 ----
        lons = [c[0] for c in polygon_coords]
        lats = [c[1] for c in polygon_coords]
        aoi_bbox = [float(min(lons)), float(min(lats)),
                    float(max(lons)), float(max(lats))]
        crs_str = dem_crs.to_string() if dem_crs is not None else "EPSG:4326"
        metadata = {
            'source': 'geo-stru',
            'source_version': __version__,
            'run_id': os.path.basename(os.path.normpath(output_dir)),
            'task_code': task_code or '',
            'aoi_name': aoi_name or '',
            'aoi_bbox': aoi_bbox,
            'crs': crs_str,
            'dem_shape': [int(dem_clipped.shape[0]), int(dem_clipped.shape[1])],
            'pixel_size_m': [float(pixel_size_m[0]), float(pixel_size_m[1])],
            'products': {
                'hillshade': 'hillshade_315.tif',
                'aspect': 'aspect.tif',
                'slope': 'slope.tif',
                'svf': 'svf.tif',
                'openness': 'openness.tif',
                'curvature': 'curvature.tif',
                'map_hillshade_png': file_a,
                'map_aspect_png': file_b,
                'map_terrain_png': file_c,
                **lin_products,
            },
            'structural_stats': {
                'elevation_range_m': [float(np.nanmin(dem_clipped)),
                                      float(np.nanmax(dem_clipped))],
                'n_lineaments': lin['stats']['n_lineaments'],
                'total_lineament_length_km': lin['stats']['total_length_km'],
                'lineament_density_mean': lin['stats']['density_mean'],
                'dominant_strikes_deg': lin['stats']['dominant_strikes_deg'],
            },
            'created_at': created_at or datetime.now().isoformat(timespec='seconds'),
        }

        # ---- 矿床类型构造推理 ----
        try:
            from core.deposit_inference import infer_deposit_type, _extract_terrain_stats
            terrain_stats = _extract_terrain_stats(dem_clipped, slope, svf, curvature)
            deposit_result = infer_deposit_type(
                structural_stats=metadata['structural_stats'],
                terrain_stats=terrain_stats,
                lineament_details=lin.get('segments'),
                mineral_hint=mineral_hint,
            )
            metadata['deposit_inference'] = deposit_result
            primary = deposit_result.get('primary_model', '未确定')
            conf = deposit_result.get('primary_confidence', 0)
            log(f"矿床类型推理: {primary} (置信度 {conf:.2f})")
        except Exception as e:
            log(f"矿床类型推理失败(非致命): {e}", 'WARNING')

        # 校验 metadata 字段完整性（先校验原始字段，再注入轨迹键，避免触动 schema 校验）
        _validate_metadata(metadata)

        # 决策轨迹血缘三键（容错，不影响产物）：显式 trace_id 优先 → 自生成（stru 多为叶子证据源）
        try:
            from commons.trace import stamp_metadata
            stamp_metadata(metadata, explicit_trace_id=trace_id, tenant_id=tenant_id)
        except Exception:
            pass

        metadata_path = os.path.join(output_dir, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        log("全部生成完成!")

        return {
            'output_files': {
                'hillshade': file_a,
                'aspect': file_b,
                'terrain': file_c,
            },
            'result_dir': output_dir,
            'metadata_path': metadata_path,
            'products': metadata['products'],
            'structural_stats': metadata['structural_stats'],
            'deposit_inference': metadata.get('deposit_inference'),
            'aoi_name': aoi_name or '',
            'aoi_bbox': aoi_bbox,
            'polygon_coords': [(float(lo), float(la)) for lo, la in polygon_coords],
            'dem_shape': list(dem_clipped.shape),
            'elevation_range': [float(np.nanmin(dem_clipped)),
                                float(np.nanmax(dem_clipped))],
        }
