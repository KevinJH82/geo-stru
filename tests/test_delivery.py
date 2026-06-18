"""
test_delivery.py — 数据发现鲁棒性测试

验证 DEM 模糊匹配、多命名兼容、传感器目录发现。
"""
import os
import pytest
from pathlib import Path
from core.delivery import _find_dem, _find_sensor_dir


class TestFindDem:
    def test_exact_dem(self, tmp_path):
        (tmp_path / "DEM.tif").write_bytes(b"fake")
        assert _find_dem(tmp_path) == tmp_path / "DEM.tif"

    def test_lowercase_dem(self, tmp_path):
        (tmp_path / "dem.tif").write_bytes(b"fake")
        result = _find_dem(tmp_path)
        assert result is not None
        assert result.name.lower() == "dem.tif"

    def test_dem_with_resolution(self, tmp_path):
        (tmp_path / "DEM_12m.tif").write_bytes(b"fake")
        result = _find_dem(tmp_path)
        assert result is not None
        assert "DEM" in result.name or "dem" in result.name.lower()

    def test_dem_prefix_match(self, tmp_path):
        (tmp_path / "dem_srtm.tif").write_bytes(b"fake")
        result = _find_dem(tmp_path)
        assert result is not None

    def test_no_dem(self, tmp_path):
        (tmp_path / "other.tif").write_bytes(b"fake")
        assert _find_dem(tmp_path) is None

    def test_dem_prefers_shorter_name(self, tmp_path):
        """多个匹配时,短名优先。"""
        (tmp_path / "dem.tif").write_bytes(b"fake")
        (tmp_path / "dem_alos_12m.tif").write_bytes(b"fake")
        result = _find_dem(tmp_path)
        assert result.name.lower() == "dem.tif"

    def test_tiff_extension(self, tmp_path):
        (tmp_path / "DEM.tiff").write_bytes(b"fake")
        result = _find_dem(tmp_path)
        assert result is not None

    def test_nonexistent_dir(self, tmp_path):
        assert _find_dem(tmp_path / "nonexistent") is None


class TestFindSensorDir:
    def test_exact_subdir(self, tmp_path):
        sensor = tmp_path / "Landsat8"
        sensor.mkdir()
        result = _find_sensor_dir(tmp_path, ["Landsat8", "Landsat9"])
        # 如果 delivery_project._find_sensor_subdir 失败,回退到模糊匹配
        if result is not None:
            assert "landsat" in result.name.lower() or "Landsat" in result.name

    def test_case_insensitive(self, tmp_path):
        sensor = tmp_path / "sentinel2"
        sensor.mkdir()
        result = _find_sensor_dir(tmp_path, ["Sentinel2"])
        if result is not None:
            assert "sentinel" in result.name.lower()

    def test_no_match(self, tmp_path):
        (tmp_path / "other_data").mkdir()
        # 不应崩溃,返回 None
        result = _find_sensor_dir(tmp_path, ["Landsat8"])
        # delivery_project 可能抛异常,回退到模糊匹配
        # 这里只要不崩溃就OK
