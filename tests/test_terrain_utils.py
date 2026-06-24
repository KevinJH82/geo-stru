"""
test_terrain_utils.py — 地形量算核心函数测试

用已知几何特征的合成 DEM 验证 slope / aspect / hillshade 输出正确性。
"""
import numpy as np
import pytest
from rasterio.transform import Affine
from core.terrain_utils import TerrainProcessor


@pytest.fixture
def flat_dem():
    """平坦 DEM (所有值 = 100m),像元 30m。"""
    return np.full((50, 50), 100.0), (30.0, 30.0)


@pytest.fixture
def slope_dem():
    """
    向东均匀倾斜的 DEM:
    每列增加 1m (col 0=0m, col 49=49m), 像元 10m。
    理论坡度 = arctan(1/10) ≈ 5.71°, 坡向 = 90° (向东)。
    """
    H, W = 50, 50
    px = 10.0
    dem = np.arange(W, dtype=np.float64)[np.newaxis, :] * np.ones((H, 1))
    return dem, (px, px)


@pytest.fixture
def cone_dem():
    """锥形 DEM (中心最高),像元 30m。"""
    H, W = 60, 60
    px = 30.0
    cy, cx = H / 2, W / 2
    y, x = np.mgrid[0:H, 0:W]
    dem = 100.0 - np.sqrt((y - cy) ** 2 + (x - cx) ** 2) * 2
    return dem, (px, px)


class TestSlope:
    def test_flat_slope_near_zero(self, flat_dem):
        dem, px = flat_dem
        slope = TerrainProcessor.compute_slope(dem, px)
        assert slope.shape == dem.shape
        # 平坦区域坡度应接近 0
        assert np.all(slope < 0.1)

    def test_uniform_slope_value(self, slope_dem):
        dem, px = slope_dem
        slope = TerrainProcessor.compute_slope(dem, px)
        # 内部区域 (去掉边缘) 坡度应 ≈ arctan(1/10)
        interior = slope[5:-5, 5:-5]
        expected = np.degrees(np.arctan(1.0 / 10.0))
        np.testing.assert_allclose(interior.mean(), expected, atol=0.5)

    def test_slope_range(self, cone_dem):
        dem, px = cone_dem
        slope = TerrainProcessor.compute_slope(dem, px)
        assert slope.min() >= 0
        assert slope.max() <= 90


class TestAspect:
    def test_flat_aspect_minus_one(self, flat_dem):
        dem, px = flat_dem
        aspect = TerrainProcessor.compute_aspect(dem, px)
        # 平坦区域应标记为 -1
        assert np.all(aspect == -1)

    def test_east_slope_aspect_consistent(self, slope_dem):
        dem, px = slope_dem
        aspect = TerrainProcessor.compute_aspect(dem, px)
        interior = aspect[5:-5, 5:-5]
        # 向东均匀倾斜 → 所有内部像元坡向应一致 (不为 -1)
        assert np.all(interior >= 0)
        # 坡向标准差应极小 (均匀坡)
        assert interior.std() < 1.0


class TestHillshade:
    def test_hillshade_uint8(self, slope_dem):
        dem, px = slope_dem
        hs = TerrainProcessor.compute_hillshade(dem, px, azimuth=315, altitude=45)
        assert hs.dtype == np.uint8
        assert hs.min() >= 0
        assert hs.max() <= 255

    def test_hillshade_nonzero(self, cone_dem):
        dem, px = cone_dem
        hs = TerrainProcessor.compute_hillshade(dem, px, azimuth=315, altitude=45)
        # 锥形 DEM 不应全黑
        assert hs.mean() > 0


class TestMultidirectional:
    def test_output_range(self, cone_dem):
        dem, px = cone_dem
        md = TerrainProcessor.compute_multidirectional_hillshade(dem, px)
        assert md.min() >= 0.0
        assert md.max() <= 1.0

    def test_no_nan(self, cone_dem):
        dem, px = cone_dem
        md = TerrainProcessor.compute_multidirectional_hillshade(dem, px)
        assert np.all(np.isfinite(md))


class TestLoadDem:
    def test_load_dem(self, tmp_path):
        """写入一个简单 GeoTIFF DEM 并验证 load_dem 读取。"""
        H, W = 20, 30
        data = np.random.rand(H, W).astype(np.float32) * 500 + 100
        transform = Affine.translation(120.0, 37.0) * Affine.scale(0.0003, -0.0003)
        import rasterio
        path = str(tmp_path / "test_dem.tif")
        with rasterio.open(path, "w", driver="GTiff", height=H, width=W,
                           count=1, dtype="float32", crs="EPSG:4326",
                           transform=transform, nodata=np.nan) as dst:
            dst.write(data, 1)
        info = TerrainProcessor.load_dem(path)
        assert info["data"].shape == (H, W)
        assert info["crs"].to_epsg() == 4326
        assert len(info["pixel_size_m"]) == 2
        assert all(p > 0 for p in info["pixel_size_m"])
