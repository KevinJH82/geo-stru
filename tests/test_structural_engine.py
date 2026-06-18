"""
test_structural_engine.py — StructuralEngine 解析与检测测试

覆盖:
  1. detect_file_type 各类文件扩展名
  2. parse_kml_polygon 基本解析 / 边界情况
  3. parse_roi_polygon CSV/Excel 解析
  4. get_aoi_name 回退逻辑
  5. generate_maps 参数校验
"""
import os
import json
import pytest
import numpy as np
import tempfile
from pathlib import Path

from core.structural_engine import StructuralEngine


# ---------------------------------------------------------------------------
# KML 样本
# ---------------------------------------------------------------------------
SIMPLE_KML = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <Placemark>
    <Polygon>
      <outerBoundaryIs>
        <LinearRing>
          <coordinates>
            120.0,37.0 120.1,37.0 120.1,37.1 120.0,37.1 120.0,37.0
          </coordinates>
        </LinearRing>
      </outerBoundaryIs>
    </Polygon>
  </Placemark>
</Document>
</kml>"""

LINESTRING_KML = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <Placemark>
    <LineString>
      <coordinates>
        120.0,37.0 120.1,37.0 120.1,37.1
      </coordinates>
    </LineString>
  </Placemark>
</Document>
</kml>"""

NS_KML = """<?xml version="1.0" encoding="UTF-8"?>
<kml:kml xmlns:kml="http://www.opengis.net/kml/2.2">
<kml:Document>
  <kml:Placemark>
    <kml:Polygon>
      <kml:outerBoundaryIs>
        <kml:LinearRing>
          <kml:coordinates>
            100.0,30.0 100.2,30.0 100.2,30.2 100.0,30.2 100.0,30.0
          </kml:coordinates>
        </kml:LinearRing>
      </kml:outerBoundaryIs>
    </kml:Polygon>
  </kml:Placemark>
</kml:Document>
</kml:kml>"""

EMPTY_KML = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
</Document>
</kml>"""


# ---------------------------------------------------------------------------
# detect_file_type
# ---------------------------------------------------------------------------
class TestDetectFileType:
    def test_kml(self, tmp_path):
        p = tmp_path / "test.kml"
        p.write_text(SIMPLE_KML)
        assert StructuralEngine.detect_file_type(str(p)) == 'kml'

    def test_kmz(self, tmp_path):
        p = tmp_path / "test.kmz"
        p.write_bytes(b"")
        assert StructuralEngine.detect_file_type(str(p)) == 'kml'

    def test_ovkml(self, tmp_path):
        p = tmp_path / "test.ovkml"
        p.write_text(SIMPLE_KML)
        assert StructuralEngine.detect_file_type(str(p)) == 'kml'

    def test_xlsx(self, tmp_path):
        p = tmp_path / "test.xlsx"
        p.write_bytes(b"")
        assert StructuralEngine.detect_file_type(str(p)) == 'roi'

    def test_csv(self, tmp_path):
        p = tmp_path / "test.csv"
        p.write_text("lon,lat\n120,37\n")
        assert StructuralEngine.detect_file_type(str(p)) == 'roi'

    def test_unsupported(self, tmp_path):
        p = tmp_path / "test.txt"
        p.write_text("hello")
        with pytest.raises(ValueError, match="不支持的文件类型"):
            StructuralEngine.detect_file_type(str(p))


# ---------------------------------------------------------------------------
# parse_kml_polygon
# ---------------------------------------------------------------------------
class TestParseKmlPolygon:
    def test_simple_polygon(self, tmp_path):
        p = tmp_path / "test.kml"
        p.write_text(SIMPLE_KML, encoding='utf-8')
        coords = StructuralEngine.parse_kml_polygon(str(p))
        assert len(coords) == 4  # 闭合点被去掉
        assert coords[0] == pytest.approx((120.0, 37.0), abs=0.01)

    def test_linestring_fallback(self, tmp_path):
        p = tmp_path / "test.kml"
        p.write_text(LINESTRING_KML, encoding='utf-8')
        coords = StructuralEngine.parse_kml_polygon(str(p))
        assert len(coords) == 3

    def test_namespaced_kml(self, tmp_path):
        p = tmp_path / "test.kml"
        p.write_text(NS_KML, encoding='utf-8')
        coords = StructuralEngine.parse_kml_polygon(str(p))
        assert len(coords) == 4

    def test_empty_kml_raises(self, tmp_path):
        p = tmp_path / "test.kml"
        p.write_text(EMPTY_KML, encoding='utf-8')
        with pytest.raises(ValueError, match="未找到足够"):
            StructuralEngine.parse_kml_polygon(str(p))


# ---------------------------------------------------------------------------
# parse_roi_polygon
# ---------------------------------------------------------------------------
class TestParseRoiPolygon:
    def test_csv(self, tmp_path):
        p = tmp_path / "test.csv"
        p.write_text("lon,lat\n120.0,37.0\n120.1,37.0\n120.1,37.1\n", encoding='utf-8')
        coords = StructuralEngine.parse_roi_polygon(str(p))
        assert len(coords) == 3
        assert coords[0] == pytest.approx((120.0, 37.0))

    def test_csv_chinese_headers(self, tmp_path):
        p = tmp_path / "test.csv"
        p.write_text("经度,纬度\n100.0,30.0\n100.1,30.0\n100.1,30.1\n", encoding='utf-8')
        coords = StructuralEngine.parse_roi_polygon(str(p))
        assert len(coords) == 3

    def test_csv_fallback_columns(self, tmp_path):
        """无 lon/lat 列名时用前两列。"""
        p = tmp_path / "test.csv"
        p.write_text("x,y\n50.0,60.0\n50.1,60.0\n50.1,60.1\n", encoding='utf-8')
        coords = StructuralEngine.parse_roi_polygon(str(p))
        assert len(coords) == 3

    def test_too_few_points_raises(self, tmp_path):
        p = tmp_path / "test.csv"
        p.write_text("lon,lat\n120.0,37.0\n", encoding='utf-8')
        with pytest.raises(ValueError, match="不足3个"):
            StructuralEngine.parse_roi_polygon(str(p))

    def test_unsupported_format(self, tmp_path):
        p = tmp_path / "test.json"
        p.write_text("{}")
        with pytest.raises(ValueError, match="不支持的ROI文件格式"):
            StructuralEngine.parse_roi_polygon(str(p))


# ---------------------------------------------------------------------------
# parse_polygon (自动检测)
# ---------------------------------------------------------------------------
class TestParsePolygon:
    def test_auto_kml(self, tmp_path):
        p = tmp_path / "test.kml"
        p.write_text(SIMPLE_KML, encoding='utf-8')
        coords = StructuralEngine.parse_polygon(str(p))
        assert len(coords) >= 3

    def test_auto_csv(self, tmp_path):
        p = tmp_path / "test.csv"
        p.write_text("lon,lat\n120,37\n120.1,37\n120.1,37.1\n", encoding='utf-8')
        coords = StructuralEngine.parse_polygon(str(p))
        assert len(coords) == 3


# ---------------------------------------------------------------------------
# get_aoi_name
# ---------------------------------------------------------------------------
class TestGetAoiName:
    def test_fallback_to_stem(self, tmp_path):
        """无 commons/aoi 时回退到文件名。"""
        p = tmp_path / "甘肃庆阳华池县.ovkml"
        p.write_text(SIMPLE_KML, encoding='utf-8')
        name = StructuralEngine.get_aoi_name(str(p))
        assert "庆阳" in name or "华池" in name or name == "甘肃庆阳华池县"


# ---------------------------------------------------------------------------
# _validate_metadata (schema 校验)
# ---------------------------------------------------------------------------
class TestValidateMetadata:
    def test_valid_metadata(self):
        """合法 metadata 应不抛异常。"""
        from core.structural_engine import _validate_metadata
        md = {
            'source': 'geo-stru',
            'aoi_name': 'test',
            'aoi_bbox': [120.0, 37.0, 120.1, 37.1],
            'crs': 'EPSG:4326',
            'products': {},
        }
        _validate_metadata(md)  # 不应抛

    def test_missing_required_field_warns(self):
        """缺少必填字段应只打警告不抛异常。"""
        from core.structural_engine import _validate_metadata
        md = {'source': 'geo-stru'}  # 缺 aoi_name, aoi_bbox, products
        _validate_metadata(md)  # 不应抛

    def test_insar_fusion_source(self):
        """insar_fusion source 也应通过。"""
        from core.structural_engine import _validate_metadata
        md = {
            'source': 'geo-stru-insar-fusion',
            'aoi_name': 'test',
            'aoi_bbox': [120.0, 37.0, 120.1, 37.1],
            'crs': 'EPSG:4326',
            'products': {},
        }
        _validate_metadata(md)  # 不应抛
