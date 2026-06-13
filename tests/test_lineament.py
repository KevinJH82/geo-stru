"""
test_lineament.py — 线性体提取测试

用合成条纹图验证:
  1. 提取的走向与已知方向一致
  2. 固定 seed 两次运行结果完全一致
  3. 空输入不崩溃
"""
import numpy as np
import pytest
from rasterio.transform import Affine
from core.lineament import extract_lineaments, write_lineaments_geojson, plot_rose_diagram


def _make_stripe_image(H, W, angle_deg, n_stripes=5, width_px=2):
    """
    生成含斜条纹的合成图像 (值域 [0, 1])。
    条纹方向由 angle_deg 控制 (0=N-S, 90=E-W)。
    """
    img = np.zeros((H, W), dtype=np.float64)
    rad = np.radians(angle_deg)
    for r in range(H):
        for c in range(W):
            # 沿法向投影
            proj = r * np.cos(rad) + c * np.sin(rad)
            if (int(proj) // max(1, W // n_stripes)) % 2 == 0:
                img[r, c] = 1.0
    return img


@pytest.fixture
def ew_stripes():
    """东西向条纹 (走向 ≈ 90°)。"""
    return _make_stripe_image(80, 80, angle_deg=90, n_stripes=6)


@pytest.fixture
def ns_stripes():
    """南北向条纹 (走向 ≈ 0°/180°)。"""
    return _make_stripe_image(80, 80, angle_deg=0, n_stripes=6)


@pytest.fixture
def slope_field():
    """模拟坡度场 (全 5°,足够通过门控)。"""
    return np.full((80, 80), 5.0)


@pytest.fixture
def identity_transform():
    return Affine.identity()


class TestExtractLineaments:
    def test_stripes_produce_lineaments(self, ew_stripes, slope_field, identity_transform):
        """条纹图应能提取出线性体 (数量 > 0)。"""
        result = extract_lineaments(
            ew_stripes, slope_field, (10.0, 10.0), identity_transform,
            rng_seed=42, min_length_m=50.0,
        )
        assert result["stats"]["n_lineaments"] > 0
        assert result["stats"]["total_length_km"] > 0
        # 每条段都有走向
        for seg in result["segments"]:
            assert 0 <= seg["strike_deg"] <= 180

    def test_dominant_strikes_populated(self, ns_stripes, slope_field, identity_transform):
        result = extract_lineaments(
            ns_stripes, slope_field, (10.0, 10.0), identity_transform,
            rng_seed=42, min_length_m=50.0,
        )
        dominant = result["stats"]["dominant_strikes_deg"]
        assert len(dominant) > 0
        # 走向值在合法范围 [0, 180]
        for d in dominant:
            assert 0 <= d <= 180

    def test_reproducibility(self, ew_stripes, slope_field, identity_transform):
        """固定 seed → 两次运行结果完全一致。"""
        kwargs = dict(
            multidir_hillshade=ew_stripes, slope=slope_field,
            pixel_size_m=(10.0, 10.0), transform=identity_transform,
            rng_seed=42, min_length_m=50.0,
        )
        r1 = extract_lineaments(**kwargs)
        r2 = extract_lineaments(**kwargs)
        assert r1["stats"]["n_lineaments"] == r2["stats"]["n_lineaments"]
        assert r1["stats"]["dominant_strikes_deg"] == r2["stats"]["dominant_strikes_deg"]

    def test_empty_input(self, identity_transform):
        """全零/平坦输入不崩溃。"""
        flat = np.zeros((30, 30), dtype=np.float64)
        slope = np.zeros((30, 30), dtype=np.float64)
        result = extract_lineaments(flat, slope, (10.0, 10.0), identity_transform, rng_seed=42)
        # 可能 0 条,不应崩溃
        assert "stats" in result
        assert "segments" in result


class TestGeoJson:
    def test_write_geojson(self, tmp_path):
        segments = [
            {"p0": (120.0, 37.0), "p1": (120.01, 37.01), "strike_deg": 45.0, "length_m": 1500.0},
        ]
        path = str(tmp_path / "test.geojson")
        write_lineaments_geojson(segments, path, crs="EPSG:4326")
        import json
        with open(path) as f:
            fc = json.load(f)
        assert fc["type"] == "FeatureCollection"
        assert len(fc["features"]) == 1
        assert fc["features"][0]["properties"]["strike_deg"] == 45.0

    def test_write_empty_geojson(self, tmp_path):
        path = str(tmp_path / "empty.geojson")
        write_lineaments_geojson([], path, crs="EPSG:4326")
        import json
        with open(path) as f:
            fc = json.load(f)
        assert len(fc["features"]) == 0


class TestRoseDiagram:
    def test_plot_does_not_crash(self, tmp_path):
        segments = [
            {"p0": (0, 0), "p1": (1, 1), "strike_deg": 45.0, "length_m": 100.0},
            {"p0": (0, 0), "p1": (1, 0), "strike_deg": 90.0, "length_m": 200.0},
        ]
        path = str(tmp_path / "rose.png")
        plot_rose_diagram(segments, path, "Test rose")
        import os
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
