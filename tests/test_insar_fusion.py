"""
test_insar_fusion.py — InSAR 融合模块测试

用 geo-insar 格式的最小合成数据验证:
  1. load_geo_insar() 正确加载
  2. run_fusion() 端到端跑通
  3. 产物落盘 (GeoTIFF / GeoJSON / metadata.json)
  4. metadata.json 包含 signal_quality 字段
"""
import json
import os
import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine


@pytest.fixture
def mock_geo_insar_dir(tmp_path):
    """创建一个最小化的 geo-insar AOI 目录结构。"""
    aoi = tmp_path / "mock_aoi"
    sbas = aoi / "sbas" / "067367_IW2"
    sbas.mkdir(parents=True)

    # velocity_mm_per_year.tif (UTM, 20x20, 简单信号)
    H, W = 20, 20
    vel = np.random.RandomState(42).randn(H, W).astype(np.float32) * 5
    transform = Affine.translation(409000, 5477000) * Affine.scale(80, -80)
    with rasterio.open(str(sbas / "velocity_mm_per_year.tif"), "w",
                       driver="GTiff", height=H, width=W, count=1,
                       dtype="float32", crs="EPSG:32652",
                       transform=transform, nodata=np.nan) as dst:
        dst.write(vel, 1)

    # dates.json
    dates = ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01",
             "2024-05-01", "2024-06-01", "2024-07-01", "2024-08-01",
             "2024-09-01", "2024-10-01", "2024-11-01"]
    with open(sbas / "dates.json", "w") as f:
        json.dump(dates, f)

    # cumulative_displacement.npy
    ts = np.random.RandomState(42).randn(len(dates), H, W).astype(np.float64) * 2
    np.save(str(sbas / "cumulative_displacement.npy"), ts)

    # summary.json
    summary = {
        "burst": "067367_IW2", "n_dates": len(dates), "n_pairs": 10,
        "date_range": [dates[0], dates[-1]],
        "orbit_direction": "ASCENDING",
        "valid_pixel_pct": 85.0,
        "velocity_mm_per_year": {"min": -15, "max": 15, "mean": 0, "std": 5},
    }
    with open(sbas / "summary.json", "w") as f:
        json.dump(summary, f)

    # 2D decomposition (vertical + ew, EPSG:4326)
    H2, W2 = 15, 15
    vert = np.random.RandomState(42).randn(H2, W2).astype(np.float32) * 3
    ew = np.random.RandomState(43).randn(H2, W2).astype(np.float32) * 4
    t4326 = Affine.translation(120.0, 37.0) * Affine.scale(0.001, -0.001)
    for name, data in [("vertical_velocity.tif", vert), ("ew_velocity.tif", ew)]:
        with rasterio.open(str(aoi / name), "w", driver="GTiff",
                           height=H2, width=W2, count=1, dtype="float32",
                           crs="EPSG:4326", transform=t4326, nodata=np.nan) as dst:
            dst.write(data, 1)

    return aoi


class TestLoadGeoInsar:
    def test_load_basic(self, mock_geo_insar_dir):
        from core.insar_fusion import load_geo_insar
        data = load_geo_insar(str(mock_geo_insar_dir))
        assert data["source"] == "geo_insar_sbas"
        assert data["shape"] == (20, 20)
        assert len(data["dates"]) == 11
        assert data["vertical"] is not None
        assert data["ew"] is not None
        assert data["has_2d"] is True
        assert data["epsg"] == 32652

    def test_load_missing_dir(self, tmp_path):
        from core.insar_fusion import load_geo_insar
        with pytest.raises(FileNotFoundError):
            load_geo_insar(str(tmp_path / "nonexistent"))


class TestRunFusion:
    def test_end_to_end(self, mock_geo_insar_dir, tmp_path):
        from core.insar_fusion import run_fusion
        out_dir = str(tmp_path / "fusion_output")
        md = run_fusion(
            insar_dir=str(mock_geo_insar_dir),
            out_dir=out_dir,
            aoi_name="test_aoi",
            seed=42,
            make_plots=False,
        )
        assert md["source"] == "geo-stru-insar-fusion"
        assert "ok+2d" in md["fusion_stats"]["signal_quality"]
        assert md["fusion_stats"]["n_deformation_lineaments"] >= 0
        assert md["fusion_stats"]["n_ew_deformation_lineaments"] >= 0

    def test_products_on_disk(self, mock_geo_insar_dir, tmp_path):
        from core.insar_fusion import run_fusion
        out_dir = str(tmp_path / "fusion_output")
        run_fusion(str(mock_geo_insar_dir), out_dir, seed=42, make_plots=False)
        # 必须存在的产物
        for f in ["los_velocity_mm_yr.tif", "velocity_gradient.tif",
                   "deformation_lineaments.geojson", "lineaments_activity.geojson",
                   "subsidence_clusters.geojson", "metadata.json"]:
            assert os.path.exists(os.path.join(out_dir, f)), f"Missing: {f}"
        # 2D 分解产物
        for f in ["vertical_velocity_mm_yr.tif", "ew_velocity_mm_yr.tif"]:
            assert os.path.exists(os.path.join(out_dir, f)), f"Missing: {f}"

    def test_metadata_valid(self, mock_geo_insar_dir, tmp_path):
        from core.insar_fusion import run_fusion
        out_dir = str(tmp_path / "fusion_output")
        md = run_fusion(str(mock_geo_insar_dir), out_dir, seed=42, make_plots=False)
        # metadata.json 也落盘了
        with open(os.path.join(out_dir, "metadata.json")) as f:
            disk_md = json.load(f)
        assert disk_md["source"] == md["source"]
        assert disk_md["fusion_stats"]["signal_quality"] == md["fusion_stats"]["signal_quality"]
        assert "aoi_bbox" in disk_md
        assert len(disk_md["aoi_bbox"]) == 4
