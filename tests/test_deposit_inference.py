"""
test_deposit_inference.py — 矿床类型构造推理引擎测试

用合成构造/地形/归因数据验证:
  1. 胶东金矿(NE走向 + 高密度) → 蚀变岩型金矿
  2. 斑岩(方向分散 + 中高海拔) → 斑岩型铜钼矿
  3. 沉积(低密度 + 缓坡 + 采空) → 沉积型矿产
  4. 输出格式正确(字段齐全, 置信度在 [0,1])
  5. insar_fusion 端到端包含 deposit_inference
"""
import pytest
from core.deposit_inference import infer_deposit_type, _strike_diff, _extract_terrain_stats


class TestHelpers:
    def test_strike_diff_same(self):
        assert _strike_diff(0, 0) == 0
        assert _strike_diff(90, 90) == 0

    def test_strike_diff_complement(self):
        """0° 和 180° 是同一走向(走向的双向性)。"""
        assert _strike_diff(0, 180) == 0
        assert _strike_diff(10, 170) == 20  # 10° vs 170° 差 20°,不是同向

    def test_strike_diff_orthogonal(self):
        assert _strike_diff(0, 90) == 90

    def test_strike_diff_near(self):
        assert _strike_diff(30, 45) == 15


class TestExtractTerrainStats:
    def test_basic(self):
        import numpy as np
        slope = np.full((10, 10), 5.0)
        svf = np.full((10, 10), 0.8)
        curvature = np.random.RandomState(42).randn(10, 10).astype(np.float32)
        dem = np.full((10, 10), 500.0)
        stats = _extract_terrain_stats(dem, slope, svf, curvature)
        assert abs(stats["slope_mean"] - 5.0) < 0.01
        assert abs(stats["svf_mean"] - 0.8) < 0.01
        assert "curvature_valley_ratio" in stats
        assert stats["elevation_min"] == 500.0

    def test_none_inputs(self):
        stats = _extract_terrain_stats(None, None, None, None)
        assert stats == {}


class TestGoldDeposit:
    """胶东金矿场景: NE走向 + 高密度 + 中低山 → 蚀变岩型金矿。"""

    def test_gold_ne_strike(self):
        result = infer_deposit_type(
            structural_stats={
                "dominant_strikes_deg": [35, 45, 110],
                "n_lineaments": 12,
                "lineament_density_mean": 0.015,
                "elevation_range_m": [200, 800],
                "total_lineament_length_km": 8.5,
            },
        )
        # 金矿(NE走向+高密度+共轭)应在 top 3
        top3 = [c["deposit_type"] for c in result["candidates"][:3]]
        assert "蚀变岩型金矿(破碎带)" in top3
        assert result["primary_confidence"] >= 0.3
        # 验证输出字段
        assert len(result["candidates"]) == 10  # 10 种矿床类型
        for c in result["candidates"]:
            assert "deposit_type" in c
            assert "confidence" in c
            assert "evidence" in c
            assert "control_model" in c
            assert 0 <= c["confidence"] <= 1.0

    def test_gold_with_goaf(self):
        """采空佐证应提升金矿置信度。"""
        result_base = infer_deposit_type(
            structural_stats={
                "dominant_strikes_deg": [35],
                "n_lineaments": 10,
                "lineament_density_mean": 0.012,
                "elevation_range_m": [300, 700],
            },
        )
        result_goaf = infer_deposit_type(
            structural_stats={
                "dominant_strikes_deg": [35],
                "n_lineaments": 10,
                "lineament_density_mean": 0.012,
                "elevation_range_m": [300, 700],
            },
            attribution_stats={"goaf": 2},
        )
        # 有采空→金矿置信度应更高
        gold_base = [c for c in result_base["candidates"]
                     if c["deposit_type"] == "蚀变岩型金矿(破碎带)"][0]["confidence"]
        gold_goaf = [c for c in result_goaf["candidates"]
                     if c["deposit_type"] == "蚀变岩型金矿(破碎带)"][0]["confidence"]
        assert gold_goaf >= gold_base


class TestPorphyry:
    """斑岩场景: 方向分散 + 中高海拔 → 斑岩型铜钼矿。"""

    def test_porphyry_dispersion(self):
        result = infer_deposit_type(
            structural_stats={
                "dominant_strikes_deg": [10, 70, 130, 160],  # 多方向分散
                "n_lineaments": 15,
                "lineament_density_mean": 0.008,
                "elevation_range_m": [500, 3000],
            },
        )
        # 斑岩应在前列
        top3 = [c["deposit_type"] for c in result["candidates"][:3]]
        assert "斑岩型铜钼矿" in top3


class TestSedimentary:
    """沉积场景: 低密度 + 缓坡 + 采空 → 沉积型矿产。"""

    def test_sedimentary_flat(self):
        result = infer_deposit_type(
            structural_stats={
                "dominant_strikes_deg": [90],
                "n_lineaments": 1,
                "lineament_density_mean": 0.001,
                "elevation_range_m": [50, 300],
            },
            terrain_stats={"slope_mean": 3.0},
            attribution_stats={"goaf": 3},
        )
        top2 = [c["deposit_type"] for c in result["candidates"][:2]]
        assert "沉积型矿产(煤/铝土/盐类)" in top2


class TestSkarn:
    """矽卡岩场景: NE-EW走向 + 中等密度 → 矽卡岩型。"""

    def test_skarn_strikes(self):
        result = infer_deposit_type(
            structural_stats={
                "dominant_strikes_deg": [60, 85],
                "n_lineaments": 8,
                "lineament_density_mean": 0.005,
                "elevation_range_m": [200, 1800],
            },
            terrain_stats={"curvature_valley_ratio": 0.45},
        )
        top3 = [c["deposit_type"] for c in result["candidates"][:3]]
        assert "矽卡岩型铁矿/铜矿" in top3


class TestOutputFormat:
    def test_summary_present(self):
        result = infer_deposit_type(
            structural_stats={
                "dominant_strikes_deg": [45],
                "n_lineaments": 5,
                "lineament_density_mean": 0.008,
                "elevation_range_m": [100, 1000],
            },
        )
        assert "structural_control_summary" in result
        assert len(result["structural_control_summary"]) > 0
        assert "primary_model" in result
        assert "primary_confidence" in result

    def test_empty_input(self):
        """无数据输入不崩溃。"""
        result = infer_deposit_type(structural_stats={})
        assert result["primary_model"] is not None
        assert len(result["candidates"]) == 10

    def test_insar_fusion_includes_deposit_inference(self, tmp_path):
        """insar_fusion 端到端应包含 deposit_inference。"""
        import json
        import os
        import numpy as np
        from rasterio.transform import Affine
        import rasterio

        # 创建最小 mock data
        aoi = tmp_path / "mock_aoi"
        sbas = aoi / "sbas" / "067367_IW2"
        sbas.mkdir(parents=True)
        H, W = 20, 20
        vel = np.random.RandomState(42).randn(H, W).astype(np.float32) * 5
        transform = Affine.translation(409000, 5477000) * Affine.scale(80, -80)
        with rasterio.open(str(sbas / "velocity_mm_per_year.tif"), "w",
                           driver="GTiff", height=H, width=W, count=1,
                           dtype="float32", crs="EPSG:32652",
                           transform=transform, nodata=np.nan) as dst:
            dst.write(vel, 1)
        dates = [f"2024-{m:02d}-01" for m in range(1, 13)]
        with open(sbas / "dates.json", "w") as f:
            json.dump(dates, f)
        ts = np.random.RandomState(42).randn(12, H, W).astype(np.float64) * 2
        np.save(str(sbas / "cumulative_displacement.npy"), ts)
        summary = {
            "burst": "067367_IW2", "n_dates": 12, "n_pairs": 10,
            "date_range": [dates[0], dates[-1]], "orbit_direction": "ASCENDING",
            "valid_pixel_pct": 85.0,
            "velocity_mm_per_year": {"min": -15, "max": 15, "mean": 0, "std": 5},
        }
        with open(sbas / "summary.json", "w") as f:
            json.dump(summary, f)

        from core.insar_fusion import run_fusion
        out_dir = str(tmp_path / "fusion_deposit")
        md = run_fusion(str(aoi), out_dir, seed=42, make_plots=False)
        assert "deposit_inference" in md
        assert md["deposit_inference"]["primary_model"] is not None


class TestMineralHint:
    """矿种方向参数测试。"""

    def _base_stats(self, strikes=[45], density=0.008, elev=[200, 800]):
        return {
            "dominant_strikes_deg": strikes,
            "n_lineaments": 8,
            "lineament_density_mean": density,
            "elevation_range_m": elev,
            "total_lineament_length_km": 5.0,
        }

    def test_petroleum_hint_boosts_oil_gas(self):
        """petroleum hint 应让油气类型排名上升。"""
        result_no_hint = infer_deposit_type(structural_stats=self._base_stats())
        result_hinted = infer_deposit_type(
            structural_stats=self._base_stats(), mineral_hint="petroleum")

        # 无hint时油气类型的排名
        no_hint_pos = {c["deposit_type"]: i for i, c in enumerate(result_no_hint["candidates"])}
        hinted_pos = {c["deposit_type"]: i for i, c in enumerate(result_hinted["candidates"])}

        oil_types = ["常规油气藏(微渗漏)", "致密油气/页岩气"]
        for ot in oil_types:
            if ot in no_hint_pos and ot in hinted_pos:
                assert hinted_pos[ot] <= no_hint_pos[ot], \
                    f"{ot} should rank equal or higher with petroleum hint"

        # hinted 版本应记录 mineral_hint
        assert result_hinted["mineral_hint"] == "petroleum"

    def test_gold_hint_boosts_gold(self):
        """gold hint 应让金矿类型排名第一。"""
        result = infer_deposit_type(
            structural_stats=self._base_stats(strikes=[35]),
            mineral_hint="gold",
        )
        gold_types = {"蚀变岩型金矿(破碎带)", "石英脉型金矿"}
        top2 = {c["deposit_type"] for c in result["candidates"][:2]}
        assert gold_types & top2, "金矿类型应在 top 2"

    def test_iron_hint(self):
        """iron hint 应让铁矿类型排名上升。"""
        result = infer_deposit_type(
            structural_stats=self._base_stats(strikes=[60]),
            mineral_hint="iron",
        )
        top3 = [c["deposit_type"] for c in result["candidates"][:3]]
        assert any("矽卡岩" in t or "BIF" in t for t in top3)

    def test_no_hint_unchanged(self):
        """无 hint 时 mineral_hint 字段为 None。"""
        result = infer_deposit_type(structural_stats=self._base_stats())
        assert result["mineral_hint"] is None

    def test_unknown_hint_no_crash(self):
        """未知矿种不崩溃,只是不加分。"""
        result = infer_deposit_type(
            structural_stats=self._base_stats(),
            mineral_hint="uranium",
        )
        assert result["primary_model"] is not None
        assert result["mineral_hint"] == "uranium"


class TestLoessPlateauOilGas:
    """
    黄土高原(鄂尔多斯盆地)油气场景回归测试。

    ROI: 甘肃庆阳华池县,长庆油田区域。
    高程 ~1000-1600m,黄土沟壑坡度 5-15°,断裂密度偏低。
    选择 mineral_hint="petroleum" 时,油气类型应为首选,
    不应被"斑岩型铜钼矿"压制。
    """

    def _loess_oil_stats(self):
        """庆阳华池典型构造/地形参数。"""
        return {
            "dominant_strikes_deg": [30, 70, 150],
            "n_lineaments": 5,
            "lineament_density_mean": 0.003,
            "elevation_range_m": [1100, 1550],
            "total_lineament_length_km": 3.2,
        }

    def _loess_terrain(self):
        return {"slope_mean": 9.5, "svf_mean": 0.85, "curvature_valley_ratio": 0.45}

    def test_petroleum_wins_over_porphyry(self):
        """选了 petroleum hint,首选应为油气类型而非斑岩铜钼。"""
        result = infer_deposit_type(
            structural_stats=self._loess_oil_stats(),
            terrain_stats=self._loess_terrain(),
            mineral_hint="petroleum",
        )
        oil_gas_types = {"常规油气藏(微渗漏)", "致密油气/页岩气"}
        assert result["primary_model"] in oil_gas_types, (
            f"期望首选为油气类型,实际为 {result['primary_model']} "
            f"(置信度 {result['primary_confidence']:.3f})"
        )

    def test_oil_gas_base_score_reasonable(self):
        """无 hint 时油气类型基础分不低于 0.25(高程/坡度不应严重扣分)。"""
        result = infer_deposit_type(
            structural_stats=self._loess_oil_stats(),
            terrain_stats=self._loess_terrain(),
        )
        for c in result["candidates"]:
            if c["deposit_type"] in ("常规油气藏(微渗漏)", "致密油气/页岩气"):
                assert c["confidence"] >= 0.25, (
                    f"{c['deposit_type']} 基础分 {c['confidence']:.3f} < 0.25, "
                    "高程/坡度参数可能仍过窄"
                )
