"""
遥感地质构造解译图可视化

生成图2-1A/B/C三类遥感地质构造解译图
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon as MPLPolygon
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from rasterio.transform import Affine
from typing import Dict, Optional, Tuple, List
from pathlib import Path
from loguru import logger


plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'PingFang SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def _deg_to_dm(deg: float, is_lat: bool) -> str:
    sign = ''
    if is_lat:
        sign = 'N' if deg >= 0 else 'S'
    else:
        sign = 'E' if deg >= 0 else 'W'
    deg_abs = abs(deg)
    d = int(deg_abs)
    m = (deg_abs - d) * 60
    return f"{d}°{m:04.1f}'{sign}"


class StructuralMapVisualizer:

    WORK_AREA_CORNERS = [
        (-4.0987, 12.8305),
        (-3.5004, 12.8346),
        (-3.4996, 12.4177),
        (-4.1013, 12.4246),
    ]

    def __init__(self, figsize: Tuple[int, int] = (18, 14), dpi: int = 300):
        self.figsize = figsize
        self.dpi = dpi

    def plot_hillshade_map(
        self,
        hillshade: np.ndarray,
        dem: np.ndarray,
        transform: Affine,
        output_path: str,
        title: str = '图2-1A  1:5万山体阴影（315°方向）遥感地质构造解译图',
        work_area: Optional[List[Tuple[float, float]]] = None,
        lineaments: Optional[List[Dict]] = None,
    ):
        """图2-1A：灰度山体阴影 + 自动提取的线性体(断裂)叠加"""
        if work_area is None:
            work_area = self.WORK_AREA_CORNERS

        fig, ax = plt.subplots(figsize=self.figsize)
        bounds = self._get_spatial_bounds(hillshade.shape, transform)
        ext = [bounds['left'], bounds['right'], bounds['bottom'], bounds['top']]

        # 无数据(nodata)区域屏蔽为留白，避免黑斑误读为深阴影
        hs = np.ma.masked_array(hillshade.astype(np.float32),
                                mask=~np.isfinite(dem))
        ax.imshow(hs, cmap='gray', origin='upper', extent=ext,
                  vmin=0, vmax=255, aspect='equal', interpolation='bilinear')

        self._add_contours(ax, dem, transform)
        self._add_lineaments(ax, lineaments)
        self._add_work_area_boundary(ax, work_area)
        self._add_coordinate_grid(ax, bounds)
        self._add_scale_bar(ax, bounds)
        self._add_north_arrow(ax, bounds)
        self._add_structural_legend(ax, bounds, has_lineaments=bool(lineaments))

        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
        plt.tight_layout()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        logger.info(f"图2-1A已保存: {output_path}")

    def plot_aspect_map(
        self,
        aspect: np.ndarray,
        dem: np.ndarray,
        transform: Affine,
        output_path: str,
        title: str = '图2-1B  1:5万坡向分析遥感地质构造解译图',
        work_area: Optional[List[Tuple[float, float]]] = None,
        hillshade: Optional[np.ndarray] = None,
        slope: Optional[np.ndarray] = None,
    ):
        """图2-1B：坡向(色相)+山体阴影(明度)+坡度(饱和度) HSV合成，色彩与地形浑然一体"""
        if work_area is None:
            work_area = self.WORK_AREA_CORNERS

        fig, ax = plt.subplots(figsize=self.figsize)
        bounds = self._get_spatial_bounds(aspect.shape, transform)
        ext = [bounds['left'], bounds['right'], bounds['bottom'], bounds['top']]

        # HSV合成：H=坡向(朝向), V=山体阴影(立体), S=坡度(平坦区去色变灰)
        nodata = ~np.isfinite(dem)
        asp = aspect.astype(np.float32)
        flat = asp < 0
        hue = (np.where(flat, 0.0, asp) % 360) / 360.0

        if slope is not None:
            # 坡度越大颜色越饱和；缓坡保留少量色彩，陡坡(≥12°)满饱和
            sat = np.clip(slope.astype(np.float32) / 12.0, 0.0, 1.0)
            sat = 0.25 + 0.75 * sat
        else:
            sat = np.full_like(hue, 0.9)
        sat[flat] = 0.0  # 平坦区→灰

        if hillshade is not None:
            # 明度由山体阴影驱动呈现立体感，压到[0.4,1]避免阴影区丢失色相
            val = 0.4 + 0.6 * (hillshade.astype(np.float32) / 255.0)
        else:
            val = np.full_like(hue, 0.85)

        # nodata处的nan会让hsv_to_rgb内部cast报警，先清零再统一覆盖为白
        for ch in (hue, sat, val):
            ch[~np.isfinite(ch)] = 0.0
        rgb = mcolors.hsv_to_rgb(np.stack([hue, sat, val], axis=-1))
        rgb[nodata] = 1.0  # nodata区留白
        ax.imshow(rgb, origin='upper', extent=ext, aspect='equal',
                  interpolation='bilinear')

        # 指南针色盘图例：直接用颜色读朝向
        self._add_aspect_colorwheel(ax)

        self._add_contours(ax, dem, transform)
        self._add_work_area_boundary(ax, work_area)
        self._add_coordinate_grid(ax, bounds)
        self._add_scale_bar(ax, bounds)
        self._add_north_arrow(ax, bounds)
        self._add_structural_legend(ax, bounds)

        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
        plt.tight_layout()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        logger.info(f"图2-1B已保存: {output_path}")

    def plot_terrain_render_map(
        self,
        terrain_rgb: np.ndarray,
        dem: np.ndarray,
        transform: Affine,
        output_path: str,
        title: str = '图2-1C  1:5万地形渲染遥感地质构造解译图',
        work_area: Optional[List[Tuple[float, float]]] = None,
    ):
        """图2-1C：地形渲染遥感地质构造解译图"""
        if work_area is None:
            work_area = self.WORK_AREA_CORNERS

        fig, ax = plt.subplots(figsize=self.figsize)
        bounds = self._get_spatial_bounds(terrain_rgb.shape[:2], transform)

        trgb = np.array(terrain_rgb, dtype=np.float32, copy=True)
        trgb[~np.isfinite(dem)] = 1.0  # nodata区留白
        ax.imshow(trgb, origin='upper',
                  extent=[bounds['left'], bounds['right'], bounds['bottom'], bounds['top']],
                  aspect='equal', interpolation='bilinear')

        self._add_contours(ax, dem, transform, color='k', alpha=0.3)
        self._add_work_area_boundary(ax, work_area)
        self._add_coordinate_grid(ax, bounds)
        self._add_scale_bar(ax, bounds)
        self._add_north_arrow(ax, bounds)
        self._add_structural_legend(ax, bounds)

        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
        plt.tight_layout()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        logger.info(f"图2-1C已保存: {output_path}")

    # ------------------------------------------------------------------
    # Map elements
    # ------------------------------------------------------------------

    def _get_spatial_bounds(self, shape: tuple, transform: Affine) -> Dict[str, float]:
        rows, cols = shape[:2]
        left = transform[2]
        top = transform[5]
        right = left + cols * transform[0]
        bottom = top + rows * transform[4]
        return {'left': left, 'right': right, 'top': top, 'bottom': bottom}

    def _add_coordinate_grid(self, ax, bounds: Dict[str, float]):
        interval = 1 / 12
        lon_ticks = np.arange(
            np.floor(bounds['left'] / interval) * interval,
            np.ceil(bounds['right'] / interval) * interval + interval / 2,
            interval,
        )
        lat_ticks = np.arange(
            np.floor(bounds['bottom'] / interval) * interval,
            np.ceil(bounds['top'] / interval) * interval + interval / 2,
            interval,
        )
        ax.set_xticks(lon_ticks)
        ax.set_yticks(lat_ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: _deg_to_dm(x, False)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: _deg_to_dm(y, True)))
        ax.grid(True, linestyle='--', alpha=0.4, color='gray', linewidth=0.5)
        ax.set_xlim(bounds['left'], bounds['right'])
        ax.set_ylim(bounds['bottom'], bounds['top'])

    @staticmethod
    def _nice_scalebar_km(width_km: float) -> float:
        """按图幅宽度选"美观"比例尺长度(约 1/4 图宽)。小 AOI 不再硬用 5km。"""
        target = max(width_km * 0.25, 1e-9)
        for v in (0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 50, 100):
            if target <= v:
                return float(v)
        return 200.0

    def _add_scale_bar(self, ax, bounds: Dict[str, float]):
        center_lat = (bounds['top'] + bounds['bottom']) / 2
        m_per_deg_lon = 111320 * np.cos(np.radians(center_lat))
        width_km = (bounds['right'] - bounds['left']) * m_per_deg_lon / 1000
        bar_length_km = self._nice_scalebar_km(width_km)
        bar_length_deg = bar_length_km * 1000 / m_per_deg_lon
        # 刻度高度/文字偏移按*图幅高度*取(而非按条长),避免小 AOI 时比例尺furniture撑爆图幅
        tick = (bounds['top'] - bounds['bottom']) * 0.012
        x_start = bounds['left'] + (bounds['right'] - bounds['left']) * 0.06
        y_pos = bounds['bottom'] + (bounds['top'] - bounds['bottom']) * 0.05
        ax.plot([x_start, x_start + bar_length_deg], [y_pos, y_pos], 'k-', linewidth=3)
        for xx in (x_start, x_start + bar_length_deg):
            ax.plot([xx, xx], [y_pos - tick, y_pos + tick], 'k-', linewidth=2)
        label = f'{bar_length_km:g} km' if bar_length_km >= 1 else f'{int(round(bar_length_km * 1000))} m'
        ax.text(x_start + bar_length_deg / 2, y_pos + tick * 1.5, label,
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    def _add_north_arrow(self, ax, bounds: Dict[str, float]):
        x = bounds['right'] - (bounds['right'] - bounds['left']) * 0.04
        y = bounds['top'] - (bounds['top'] - bounds['bottom']) * 0.06
        dy = (bounds['top'] - bounds['bottom']) * 0.04
        ax.annotate('N', xy=(x, y + dy), fontsize=14, fontweight='bold',
                    ha='center', va='bottom', color='black',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.8))
        ax.annotate('', xy=(x, y + dy), xytext=(x, y),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))

    def _add_aspect_colorwheel(self, ax):
        """在坡向图右上角绘制指南针色盘图例(N上, 顺时针, 色相对应朝向)"""
        try:
            wax = ax.inset_axes([0.015, 0.83, 0.14, 0.14], projection='polar')
        except Exception:
            return  # 旧版matplotlib不支持inset极坐标时静默跳过
        wax.set_theta_zero_location('N')
        wax.set_theta_direction(-1)  # 顺时针，与罗盘一致

        n = 180
        theta = np.linspace(0, 2 * np.pi, n + 1)
        radii = np.array([0.55, 1.0])
        colors = (np.degrees(theta[:-1]) / 360.0).reshape(1, n)  # 色相=罗盘角/360
        wax.pcolormesh(theta, radii, colors, cmap='hsv', vmin=0, vmax=1,
                       shading='flat')

        wax.set_ylim(0, 1)
        wax.set_yticks([])
        wax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        wax.set_xticklabels(['N', 'E', 'S', 'W'], fontsize=8, fontweight='bold')
        wax.tick_params(pad=-2)
        wax.spines['polar'].set_visible(False)
        wax.set_title('坡向', fontsize=9, pad=2)
        wax.patch.set_alpha(0.0)

    def _add_lineaments(self, ax, lineaments: Optional[List[Dict]]):
        """绘制自动提取的线性体(断裂),让图面与"构造解译"名实相符。"""
        if not lineaments:
            return
        for s in lineaments:
            (lon0, lat0), (lon1, lat1) = s['p0'], s['p1']
            ax.plot([lon0, lon1], [lat0, lat1], color='#ff2d2d', linewidth=1.8,
                    alpha=0.95, zorder=8, solid_capstyle='round')

    def _add_work_area_boundary(self, ax, corners: List[Tuple[float, float]]):
        polygon = MPLPolygon(corners, closed=True, edgecolor='red', facecolor='none',
                             linewidth=2.0, linestyle='-', zorder=10)
        ax.add_patch(polygon)
        labels = ['A', 'B', 'C', 'D']
        for (lon, lat), label in zip(corners, labels):
            ax.plot(lon, lat, 'r^', markersize=8, zorder=11)
            ax.text(lon, lat, f' {label}', fontsize=10, color='red',
                    fontweight='bold', va='bottom', ha='left', zorder=11)

    @staticmethod
    def _nice_step(raw: float) -> float:
        """取接近raw的"美观"步长(1/2/2.5/5×10^n)"""
        if raw <= 0:
            return 1.0
        mag = 10 ** np.floor(np.log10(raw))
        for m in (1, 2, 2.5, 5, 10):
            if raw <= m * mag:
                return float(m * mag)
        return float(10 * mag)

    def _add_contours(self, ax, dem: np.ndarray, transform: Affine,
                      levels=None, color='k', alpha=0.4, n_levels: int = 12):
        rows, cols = dem.shape
        # 平滑DEM，使等高线连续顺滑而非锯齿状
        try:
            from scipy.ndimage import gaussian_filter
            dem_s = gaussian_filter(dem.astype(np.float64), sigma=1.5)
        except Exception:
            dem_s = dem.astype(np.float64)

        if levels is None:
            vmin, vmax = float(np.nanmin(dem_s)), float(np.nanmax(dem_s))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or (vmax - vmin) < 1e-3:
                return
            step = self._nice_step((vmax - vmin) / n_levels)
            start = np.ceil(vmin / step) * step
            levels = np.arange(start, vmax, step)
            if len(levels) < 2:
                return

        x_coords = np.linspace(transform[2], transform[2] + cols * transform[0], cols)
        y_coords = np.linspace(transform[5], transform[5] + rows * transform[4], rows)
        try:
            cs = ax.contour(x_coords, y_coords, dem_s, levels=levels,
                            colors=color, alpha=alpha, linewidths=0.6, zorder=5)
            ax.clabel(cs, inline=True, fontsize=6, fmt='%d m')
        except Exception:
            pass

    def _add_structural_legend(self, ax, bounds: Dict[str, float], has_lineaments: bool = False):
        # 仅列出图面真实绘制的要素,避免"图例承诺了算法从不绘制的要素"。
        if has_lineaments:
            legend_elements = [
                Line2D([0], [0], color='red', linewidth=1.5, label='自动提取线性体(断裂)'),
            ]
        else:
            legend_elements = [
                Line2D([0], [0], color='red', linewidth=2, label='断层'),
                Line2D([0], [0], color='blue', linewidth=1.5, linestyle='--', label='褶皱'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                       markeredgecolor='green', markersize=10, linewidth=0, label='环形构造'),
                Line2D([0], [0], color='orange', linewidth=1, label='线性构造'),
            ]
        legend = ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
                           framealpha=0.8, edgecolor='gray',
                           title='构造解译要素', title_fontsize=10)
        legend.get_frame().set_facecolor('white')
