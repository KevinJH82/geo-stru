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
                   "subsidence_clusters.geojson", "goaf_polygons.geojson",
                   "deformation_attribution.tif", "deformation_attribution.geojson",
                   "metadata.json"]:
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


# ---------------------------------------------------------------------------
# P2: 沉降多边形 + 时序分类 专项测试
# ---------------------------------------------------------------------------

def _make_subsidence_velocity(H, W, transform, n_clusters=2, radius_px=4):
    """创建含明显负速率漏斗的速度场 + 标注栅格。"""
    rng = np.random.RandomState(123)
    vel = rng.randn(H, W).astype(np.float64) * 2  # 背景噪声
    valid = np.ones((H, W), bool)
    # 植入负值漏斗
    centers = [(H // 4, W // 4), (3 * H // 4, 3 * W // 4)][:n_clusters]
    for cy, cx in centers:
        for r in range(max(0, cy - radius_px), min(H, cy + radius_px + 1)):
            for c in range(max(0, cx - radius_px), min(W, cx + radius_px + 1)):
                dist = np.hypot(r - cy, c - cx)
                if dist <= radius_px:
                    vel[r, c] = -15 - 5 * (1 - dist / radius_px)  # 强负值
    return vel, valid, centers


class TestGoafPolygons:
    def test_polygon_delineation(self, tmp_path):
        """检测到的沉降簇应能生成凸包多边形。"""
        from core.insar_fusion import detect_subsidence, delineate_goaf_polygons
        H, W = 30, 30
        transform = Affine.translation(120.0, 37.0) * Affine.scale(0.001, -0.001)
        pixel_m = (111, 111)
        vel, valid, _ = _make_subsidence_velocity(H, W, transform, n_clusters=1)
        clusters, lbl, thr = detect_subsidence(vel, valid, transform, pixel_m,
                                                k_sigma=1.0, min_area_px=3)
        assert len(clusters) >= 1, "应至少检测到 1 个沉降簇"
        clusters = delineate_goaf_polygons(clusters, lbl, vel, transform, pixel_m)
        cl = clusters[0]
        assert cl["boundary"] is not None, "应有凸包多边形"
        assert len(cl["boundary"]) >= 4, "多边形至少 4 个点(含闭合点)"
        assert cl["long_axis_deg"] is not None
        assert 0 <= cl["long_axis_deg"] <= 180

    def test_long_axis_vs_strike(self, tmp_path):
        """长轴方向应能计算与断裂走向的差异。"""
        from core.insar_fusion import detect_subsidence, delineate_goaf_polygons
        H, W = 30, 30
        transform = Affine.translation(120.0, 37.0) * Affine.scale(0.001, -0.001)
        vel, valid, _ = _make_subsidence_velocity(H, W, transform, n_clusters=1)
        clusters, lbl, thr = detect_subsidence(vel, valid, transform, pixel_m=(111, 111),
                                                k_sigma=1.0, min_area_px=3)
        topo_strikes = [45.0, 135.0]
        clusters = delineate_goaf_polygons(clusters, lbl, vel, transform,
                                            pixel_m=(111, 111), topo_strikes=topo_strikes)
        cl = clusters[0]
        assert cl["strike_diff_deg"] is not None
        assert 0 <= cl["strike_diff_deg"] <= 90

    def test_write_polygon_geojson(self, tmp_path):
        """多边形 GeoJSON 可正确落盘。"""
        from core.insar_fusion import detect_subsidence, delineate_goaf_polygons, _write_polygon_geojson
        H, W = 30, 30
        transform = Affine.translation(120.0, 37.0) * Affine.scale(0.001, -0.001)
        vel, valid, _ = _make_subsidence_velocity(H, W, transform, n_clusters=1)
        clusters, lbl, thr = detect_subsidence(vel, valid, transform, pixel_m=(111, 111),
                                                k_sigma=1.0, min_area_px=3)
        clusters = delineate_goaf_polygons(clusters, lbl, vel, transform, pixel_m=(111, 111))
        path = str(tmp_path / "goaf.geojson")
        _write_polygon_geojson(clusters, path, "EPSG:4326")
        assert os.path.exists(path)
        fc = json.load(open(path))
        assert fc["type"] == "FeatureCollection"
        assert len(fc["features"]) >= 1
        assert fc["features"][0]["geometry"]["type"] == "Polygon"


class TestSubsidenceTimeseries:
    def _make_ts(self, n_dates, H, W, clusters_lbl, mode="linear"):
        """创建模拟时序:每个簇内按模式生成。"""
        rng = np.random.RandomState(42)
        ts = rng.randn(n_dates, H, W).astype(np.float64) * 0.5
        cids = sorted(set(clusters_lbl.flat) - {0})
        for cid in cids:
            m = clusters_lbl == cid
            for t in range(n_dates):
                base = -2.0 * t if mode == "linear" else -0.5 * t ** 1.3
                ts[t][m] = base + rng.randn(m.sum()) * 0.3
        return ts

    def test_linear_classification(self):
        """线性沉降应被正确分类。"""
        from core.insar_fusion import detect_subsidence, classify_subsidence_timeseries
        H, W = 20, 20
        transform = Affine.translation(120.0, 37.0) * Affine.scale(0.001, -0.001)
        vel, valid, _ = _make_subsidence_velocity(H, W, transform, n_clusters=1)
        clusters, lbl, thr = detect_subsidence(vel, valid, transform, pixel_m=(111, 111),
                                                k_sigma=1.0, min_area_px=3)
        assert len(clusters) >= 1
        dates = [f"2024-{m:02d}-01" for m in range(1, 13)]
        ts = self._make_ts(12, H, W, lbl, mode="linear")
        clusters = classify_subsidence_timeseries(clusters, lbl, ts, dates)
        cl = clusters[0]
        assert cl["ts_class"] in ("linear", "accelerating", "stable")
        assert cl["ts_r2"] is not None
        assert cl["ts_rate_mm_yr"] is not None

    def test_no_ts_data(self):
        """无时序数据 → no_data 分类。"""
        from core.insar_fusion import detect_subsidence, classify_subsidence_timeseries
        H, W = 20, 20
        transform = Affine.translation(120.0, 37.0) * Affine.scale(0.001, -0.001)
        vel, valid, _ = _make_subsidence_velocity(H, W, transform, n_clusters=1)
        clusters, lbl, thr = detect_subsidence(vel, valid, transform, pixel_m=(111, 111),
                                                k_sigma=1.0, min_area_px=3)
        clusters = classify_subsidence_timeseries(clusters, lbl, None, [])
        assert clusters[0]["ts_class"] == "no_data"

    def test_fusion_includes_subsidence_details(self, mock_geo_insar_dir, tmp_path):
        """端到端融合应包含 subsidence_details。"""
        from core.insar_fusion import run_fusion
        out_dir = str(tmp_path / "fusion_p2")
        md = run_fusion(str(mock_geo_insar_dir), out_dir, seed=42, make_plots=False)
        assert "subsidence_details" in md["fusion_stats"]
        assert isinstance(md["fusion_stats"]["subsidence_details"], list)


# ---------------------------------------------------------------------------
# P3: 形变归因 专项测试
# ---------------------------------------------------------------------------
class TestDeformationAttribution:
    def _make_clusters_with_props(self, H=30, W=30, n_clusters=2):
        """创建含 B3/B4 属性的模拟 cluster。"""
        from core.insar_fusion import detect_subsidence, delineate_goaf_polygons
        transform = Affine.translation(120.0, 37.0) * Affine.scale(0.001, -0.001)
        vel, valid, centers = _make_subsidence_velocity(H, W, transform, n_clusters=n_clusters)
        clusters, lbl, thr = detect_subsidence(vel, valid, transform, pixel_m=(111, 111),
                                                k_sigma=1.0, min_area_px=3)
        # 添加 B4 时序属性
        for cl in clusters:
            cl["ts_class"] = "linear"
            cl["ts_rate_mm_yr"] = -10.0
            cl["ts_r2"] = 0.9
        # B3 多边形
        clusters = delineate_goaf_polygons(clusters, lbl, vel, transform, pixel_m=(111, 111))
        return clusters, lbl, vel, valid, transform

    def test_goaf_attribution(self):
        """负速率 + 线性时序 + 圆形 → 采空沉降。"""
        from core.insar_fusion import attribute_deformation
        clusters, lbl, vel, valid, transform = self._make_clusters_with_props()
        attr_raster, clusters = attribute_deformation(
            clusters, lbl, vel, valid, transform, (111, 111))
        # 至少有一个簇被归因
        classes = [c["attribution_class"] for c in clusters]
        assert len(classes) > 0
        for cl in clusters:
            assert "attribution_confidence" in cl
            assert "attribution_scores" in cl
            assert 0 <= cl["attribution_confidence"] <= 1.0

    def test_fault_creep_attribution(self):
        """距断裂近 + 长轴与走向一致 → 断裂蠕动。"""
        from core.insar_fusion import attribute_deformation
        clusters, lbl, vel, valid, transform = self._make_clusters_with_props(n_clusters=1)
        # 模拟距断裂距离栅格: 簇内距离很近
        H, W = lbl.shape
        dist = np.full((H, W), 1000.0)  # 默认远
        # 第一个簇内设近
        cid = clusters[0]["label"]
        m = lbl == cid
        dist[m] = 50.0  # 很近
        # 设长轴与断裂走向差异小
        clusters[0]["strike_diff_deg"] = 10.0
        attr_raster, clusters = attribute_deformation(
            clusters, lbl, vel, valid, transform, (111, 111),
            distance_to_lineament=dist)
        # 应有断裂蠕动分数提升
        assert clusters[0]["attribution_scores"]["fault_creep"] >= 0.3

    def test_landslide_attribution(self):
        """坡度大 + 坡向集中 → 滑坡。"""
        from core.insar_fusion import attribute_deformation
        clusters, lbl, vel, valid, transform = self._make_clusters_with_props(n_clusters=1)
        H, W = lbl.shape
        # 模拟陡坡
        slope = np.full((H, W), 5.0)
        cid = clusters[0]["label"]
        slope[lbl == cid] = 25.0
        # 模拟坡向集中
        aspect = np.full((H, W), -1.0)
        aspect[lbl == cid] = 90.0  # 全部朝东
        attr_raster, clusters = attribute_deformation(
            clusters, lbl, vel, valid, transform, (111, 111),
            slope=slope, aspect=aspect)
        assert clusters[0]["attribution_scores"]["landslide"] >= 0.3

    def test_raster_output(self):
        """归因栅格值正确编码。"""
        from core.insar_fusion import attribute_deformation
        clusters, lbl, vel, valid, transform = self._make_clusters_with_props(n_clusters=1)
        attr_raster, clusters = attribute_deformation(
            clusters, lbl, vel, valid, transform, (111, 111))
        # 归因区域值在 {1, 2, 3, 9} 中
        vals = set(attr_raster[attr_raster > 0])
        assert vals.issubset({1, 2, 3, 9})

    def test_fusion_includes_attribution(self, mock_geo_insar_dir, tmp_path):
        """端到端融合应包含归因产物。"""
        from core.insar_fusion import run_fusion
        out_dir = str(tmp_path / "fusion_p3")
        md = run_fusion(str(mock_geo_insar_dir), out_dir, seed=42, make_plots=False)
        # 归因产物存在
        assert os.path.exists(os.path.join(out_dir, "deformation_attribution.tif"))
        assert os.path.exists(os.path.join(out_dir, "deformation_attribution.geojson"))
        # metadata 含归因统计
        assert "attribution_summary" in md["fusion_stats"]
        assert "attribution_details" in md["fusion_stats"]
