"""
地形处理工具

提供DEM地形分析功能：山体阴影、坡向、坡度、地形渲染
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine
from scipy.ndimage import uniform_filter, gaussian_filter
from typing import Dict, Tuple, Optional
from loguru import logger


class TerrainProcessor:
    """DEM地形处理类"""

    # 梯度计算前对DEM做的轻度高斯平滑标准差(像素)。
    # 抑制DEM量化导致的阶梯状噪声，使山体阴影/坡向/地形渲染更顺滑；0=关闭。
    SMOOTH_SIGMA = 1.0

    @staticmethod
    def load_dem(file_path: str) -> Dict:
        """
        加载DEM并返回完整信息

        Returns:
            dict: {
                'data': np.ndarray, 'transform': Affine,
                'crs': CRS, 'bounds': BoundingBox, 'profile': dict,
                'pixel_size_m': (float, float)
            }
        """
        with rasterio.open(file_path) as src:
            data = src.read(1).astype(np.float32)
            transform = src.transform
            crs = src.crs
            bounds = src.bounds
            profile = src.profile.copy()

        # 计算像素尺寸（米）—— 必须区分坐标系,否则会静默产出错误量值。
        # 地理坐标(经纬度,度):用中心纬度做 度→米 换算。
        # 投影坐标(UTM 等,本就是米):直接取 transform 像元边长,*不能*再乘 111320。
        center_lat = (bounds.top + bounds.bottom) / 2
        if crs is not None and crs.is_projected:
            pixel_size_x = abs(transform[0])
            pixel_size_y = abs(transform[4])
            logger.info(f"DEM 为投影坐标系({crs.to_string()}),像元尺寸取原生米值")
        else:
            # 地理坐标,或 CRS 缺失(沿用按经纬度近似换算的历史行为)
            pixel_size_x = abs(transform[0]) * 111320 * np.cos(np.radians(center_lat))
            pixel_size_y = abs(transform[4]) * 110540
            if crs is None:
                logger.warning("DEM 无 CRS,按地理坐标(经纬度)近似换算像元尺寸")

        logger.info(
            f"DEM加载完成: {data.shape}, "
            f"像素尺寸: {pixel_size_x:.1f}m x {pixel_size_y:.1f}m, "
            f"高程范围: {np.nanmin(data):.1f}m - {np.nanmax(data):.1f}m"
        )

        return {
            'data': data,
            'transform': transform,
            'crs': crs,
            'bounds': bounds,
            'profile': profile,
            'pixel_size_m': (pixel_size_x, pixel_size_y),
        }

    @staticmethod
    def compute_hillshade(
        dem: np.ndarray,
        pixel_size_m: Tuple[float, float],
        azimuth: float = 315,
        altitude: float = 45,
        z_factor: float = 1,
    ) -> np.ndarray:
        """标准单方向山体阴影"""
        azimuth_rad = np.radians(360 - azimuth + 90)
        altitude_rad = np.radians(altitude)

        dz_dx, dz_dy = TerrainProcessor._compute_gradients(dem, pixel_size_m, z_factor)

        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        aspect_rad = np.arctan2(dz_dy, -dz_dx)

        hillshade = (
            np.sin(altitude_rad) * np.cos(slope_rad)
            + np.cos(altitude_rad) * np.sin(slope_rad) * np.cos(azimuth_rad - aspect_rad)
        )

        # DEM含nodata(nan)时，先填充再转uint8，避免无效值转换警告/黑斑
        hillshade = np.clip(255 * hillshade, 0, 255)
        hillshade = np.where(np.isfinite(hillshade), hillshade, 0).astype(np.uint8)
        return hillshade

    @staticmethod
    def compute_aspect(
        dem: np.ndarray,
        pixel_size_m: Tuple[float, float],
    ) -> np.ndarray:
        """
        计算坡向（0-360度）

        0/360=北, 90=东, 180=南, 270=西, -1=平坦
        """
        dz_dx, dz_dy = TerrainProcessor._compute_gradients(dem, pixel_size_m, 1)

        aspect = 180 - np.degrees(np.arctan2(dz_dy, -dz_dx))
        aspect = aspect % 360

        # 标记平坦区域:用坡度角阈值(<1.5°)而非"严格水平",
        # 否则近平坦区(微小非零坡度)会产生乱跳的坡向噪声写入 aspect.tif。
        slope_deg = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
        flat_mask = slope_deg < 1.5
        aspect[flat_mask] = -1

        return aspect

    @staticmethod
    def compute_slope(
        dem: np.ndarray,
        pixel_size_m: Tuple[float, float],
    ) -> np.ndarray:
        """计算坡度（度）"""
        dz_dx, dz_dy = TerrainProcessor._compute_gradients(dem, pixel_size_m, 1)
        slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))
        return slope

    @staticmethod
    def _shift(arr: np.ndarray, dr: int, dc: int) -> np.ndarray:
        """整体平移数组,露出的边缘填 NaN(用于地平线扫描)。"""
        out = np.full_like(arr, np.nan, dtype=np.float64)
        H, W = arr.shape
        r0s, r0d = (dr, 0) if dr >= 0 else (0, -dr)
        c0s, c0d = (dc, 0) if dc >= 0 else (0, -dc)
        rh = H - abs(dr); cw = W - abs(dc)
        if rh > 0 and cw > 0:
            out[r0d:r0d + rh, c0d:c0d + cw] = arr[r0s:r0s + rh, c0s:c0s + cw]
        return out

    @staticmethod
    def compute_skyview_openness(
        dem: np.ndarray,
        pixel_size_m: Tuple[float, float],
        n_dir: int = 16,
        max_radius_px: int = 12,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        光照无关地形增强:天空视域因子(SVF)与正地形开度(Positive Openness)。

        基于地平线角扫描(Yokoyama 等):沿 n_dir 个方位、到 max_radius_px 半径,
        求每个方向的最大仰角(地平线角)。
          - SVF = 1 - mean_dir( sin(max(0, 地平线角)) )  ∈[0,1],谷/凹陷低、脊/开阔高
          - 正开度 = mean_dir( 90° - 地平线角 )         (度),脊/凸起大、谷小

        二者都与光源方向无关,揭示线性构造/断裂显著优于单方位山体阴影。
        纯 numpy 实现,不依赖 rvt-py。

        Returns
        -------
        (svf[0,1], openness_deg)
        """
        d = dem.astype(np.float64)
        cell = (pixel_size_m[0] + pixel_size_m[1]) / 2.0
        svf_acc = np.zeros(d.shape, dtype=np.float64)
        open_acc = np.zeros(d.shape, dtype=np.float64)
        for k in range(n_dir):
            az = 2.0 * np.pi * k / n_dir
            ux, uy = np.cos(az), np.sin(az)
            best = np.full(d.shape, -np.inf, dtype=np.float64)
            for step in range(1, max_radius_px + 1):
                dr = int(round(step * uy))
                dc = int(round(step * ux))
                if dr == 0 and dc == 0:
                    continue
                dz = TerrainProcessor._shift(d, dr, dc) - d
                ang = np.arctan2(dz, step * cell)  # 仰角(弧度)
                best = np.fmax(best, ang)           # fmax 忽略 NaN
            best[~np.isfinite(best)] = 0.0          # 无有效地平线→视为水平
            svf_acc += np.sin(np.clip(best, 0.0, np.pi / 2))
            open_acc += (np.pi / 2 - best)
        svf = 1.0 - svf_acc / n_dir
        openness_deg = np.degrees(open_acc / n_dir)
        nod = ~np.isfinite(dem)
        svf[nod] = np.nan
        openness_deg[nod] = np.nan
        return svf.astype(np.float32), openness_deg.astype(np.float32)

    @staticmethod
    def compute_curvature(
        dem: np.ndarray,
        pixel_size_m: Tuple[float, float],
    ) -> np.ndarray:
        """
        地形总曲率(剖面+平面,∝ 高程二阶导)。脊线/背斜枢纽为正,谷线/向斜为负,
        为褶皱/线性构造判读提供比"坡向上色"更直接的二阶地貌依据。

        基于轻度平滑后的有限差分拉普拉斯(按 x/y 米尺度归一)。单位:1/100m(放大便于成图)。
        """
        cx, cy = pixel_size_m
        z = gaussian_filter(np.nan_to_num(dem.astype(np.float64),
                                          nan=float(np.nanmean(dem))),
                            sigma=max(TerrainProcessor.SMOOTH_SIGMA, 1.0))
        zxx = (np.roll(z, -1, 1) - 2 * z + np.roll(z, 1, 1)) / (cx ** 2)
        zyy = (np.roll(z, -1, 0) - 2 * z + np.roll(z, 1, 0)) / (cy ** 2)
        curv = -(zxx + zyy) * 100.0  # ∝ 凸正凹负;×100 提升量纲可读性
        curv[~np.isfinite(dem)] = np.nan
        return curv.astype(np.float32)

    @staticmethod
    def compute_multidirectional_hillshade(
        dem: np.ndarray,
        pixel_size_m: Tuple[float, float],
        azimuths=(315, 45, 135, 225),
        weights=(0.3, 0.3, 0.2, 0.2),
        z_factor: float = 1,
    ) -> np.ndarray:
        """
        多方位融合山体阴影,值域[0,1](float64)。

        多方位融合可消除单一光照下"与光照平行的线性体不可见"的方位盲区,
        是地形渲染与线性体(断裂)提取共用的光照无关基底。
        """
        combined = np.zeros(dem.shape, dtype=np.float64)
        for az, w in zip(azimuths, weights):
            shade = TerrainProcessor.compute_hillshade(
                dem, pixel_size_m, azimuth=az, z_factor=z_factor)
            combined += (shade.astype(np.float64) / 255.0) * w
        return combined

    @staticmethod
    def compute_terrain_render(
        dem: np.ndarray,
        pixel_size_m: Tuple[float, float],
        landsat_rgb: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        多方向地形渲染

        融合多个方位角的山体阴影 + 坡度着色 + 可选Landsat叠加

        Args:
            dem: DEM数据
            pixel_size_m: 像素尺寸(米)
            landsat_rgb: 可选Landsat RGB数组 (H, W, 3)，值域[0,1]

        Returns:
            RGB数组 (H, W, 3)，float32，值域[0,1]
        """
        # 多方向山体阴影(复用共用实现)
        combined_shade = TerrainProcessor.compute_multidirectional_hillshade(
            dem, pixel_size_m)

        # 坡度着色
        slope = TerrainProcessor.compute_slope(dem, pixel_size_m)
        slope_norm = np.clip(slope / 30.0, 0, 1)
        slope_shade = 1.0 - 0.5 * slope_norm

        # 融合阴影和坡度
        intensity = np.clip(combined_shade * slope_shade, 0, 1)

        if landsat_rgb is not None:
            # IHS 融合:把影像转 HSV(IHS 代理),用地形阴影替换明度(I),保留色度(H/S)。
            # 相比固定 α 加权,这样既保住岩性/光谱色彩,又叠加清晰的地形立体感。
            import matplotlib.colors as _mc
            hsv = _mc.rgb_to_hsv(np.clip(np.nan_to_num(landsat_rgb), 0, 1))
            hsv[:, :, 2] = np.clip(0.15 + 0.85 * intensity, 0, 1)  # 明度=地形(留底避免全黑)
            rgb = np.clip(_mc.hsv_to_rgb(hsv), 0, 1).astype(np.float32)
        else:
            # 纯地形渲染（暖色调）
            rgb = np.zeros((*dem.shape, 3), dtype=np.float32)
            rgb[:, :, 0] = np.clip(intensity * 0.85 + slope_norm * 0.15, 0, 1)
            rgb[:, :, 1] = np.clip(intensity * 0.78, 0, 1)
            rgb[:, :, 2] = np.clip(intensity * 0.65, 0, 1)

        return rgb.astype(np.float32)

    @staticmethod
    def resample_to_dem_grid(
        src_path: str,
        dem_transform: Affine,
        dem_crs,
        dem_shape: Tuple[int, int],
        src_band: int = 1,
    ) -> np.ndarray:
        """
        将外部栅格重投影/重采样到DEM网格

        Args:
            src_path: 源栅格文件路径
            dem_transform: DEM的Affine变换
            dem_crs: DEM的CRS
            dem_shape: DEM的(height, width)
            src_band: 要读取的波段号
        """
        dst_data = np.zeros(dem_shape, dtype=np.float32)

        with rasterio.open(src_path) as src:
            reproject(
                source=rasterio.band(src, src_band),
                destination=dst_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dem_transform,
                dst_crs=dem_crs,
                resampling=Resampling.bilinear,
            )

        return dst_data

    @staticmethod
    def clip_to_workarea(
        data: np.ndarray,
        transform: Affine,
        corner_coords: list,
    ) -> Tuple[np.ndarray, Affine]:
        """
        按坐标裁剪到工作区

        Args:
            data: 栅格数据 (H, W)
            transform: 原始Affine变换
            corner_coords: [(lon, lat), ...] 工作区角点列表

        Returns:
            (裁剪后数据, 新的Affine变换)
        """
        lons = [c[0] for c in corner_coords]
        lats = [c[1] for c in corner_coords]

        lon_min, lon_max = min(lons), max(lons)
        lat_min, lat_max = min(lats), max(lats)

        # 像素坐标范围
        col_min = int((lon_min - transform[2]) / transform[0])
        col_max = int((lon_max - transform[2]) / transform[0])
        row_min = int((transform[5] - lat_max) / (-transform[4]))
        row_max = int((transform[5] - lat_min) / (-transform[4]))

        # 边界检查
        col_min = max(0, col_min)
        col_max = min(data.shape[1], col_max)
        row_min = max(0, row_min)
        row_max = min(data.shape[0], row_max)

        clipped = data[row_min:row_max, col_min:col_max]

        new_transform = Affine(
            transform[0], transform[1], transform[2] + col_min * transform[0],
            transform[3], transform[4], transform[5] + row_min * transform[4],
        )

        # 按真实多边形裁剪:bbox 之内、多边形之外的像素置为 nodata(NaN),
        # 避免非矩形 AOI 时把红框外的数据当作工作区参与分析与展示。
        if len(corner_coords) >= 3:
            try:
                from rasterio.features import geometry_mask
                ring = [(float(lo), float(la)) for lo, la in corner_coords]
                if ring[0] != ring[-1]:
                    ring.append(ring[0])  # 闭合环
                geom = {'type': 'Polygon', 'coordinates': [ring]}
                outside = geometry_mask(
                    [geom], out_shape=clipped.shape,
                    transform=new_transform, invert=False,
                )  # invert=False: 多边形外=True
                clipped = clipped.astype(np.float32, copy=True)
                clipped[outside] = np.nan
            except Exception as e:
                logger.warning(f"多边形裁剪失败,回退到外接矩形: {e}")

        return clipped, new_transform

    @staticmethod
    def percent_stretch(band: np.ndarray, percent: float = 2) -> np.ndarray:
        """百分位线性拉伸到[0,1]"""
        valid = band[~np.isnan(band)]
        if len(valid) == 0:
            return np.zeros_like(band)
        lo = np.percentile(valid, percent)
        hi = np.percentile(valid, 100 - percent)
        if hi - lo < 1e-10:
            return np.zeros_like(band)
        return np.clip((band - lo) / (hi - lo), 0, 1).astype(np.float32)

    @staticmethod
    def _compute_gradients(
        dem: np.ndarray,
        pixel_size_m: Tuple[float, float],
        z_factor: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """3x3 Horn法计算梯度（先做轻度高斯平滑抑制阶梯状噪声）"""
        cell_x, cell_y = pixel_size_m
        if TerrainProcessor.SMOOTH_SIGMA > 0:
            dem = gaussian_filter(dem.astype(np.float64), sigma=TerrainProcessor.SMOOTH_SIGMA)
        padded = np.pad(dem, 1, mode='reflect') * z_factor

        # Horn 3x3 kernel
        dz_dx = (
            (padded[0:-2, 2:] + 2 * padded[1:-1, 2:] + padded[2:, 2:])
            - (padded[0:-2, 0:-2] + 2 * padded[1:-1, 0:-2] + padded[2:, 0:-2])
        ) / (8.0 * cell_x)

        dz_dy = (
            (padded[2:, 0:-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:])
            - (padded[0:-2, 0:-2] + 2 * padded[0:-2, 1:-1] + padded[0:-2, 2:])
        ) / (8.0 * cell_y)

        return dz_dx, dz_dy
