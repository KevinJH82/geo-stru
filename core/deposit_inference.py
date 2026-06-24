"""
deposit_inference.py — 基于构造特征的矿床类型推理引擎

仅依赖 geo-stru 自身产出(断裂走向/密度/地形/曲率/形变归因),不依赖外部数据源。
基于地质学公认的构造控矿模式规则,推理 ROI 可能的矿床类型组合与构造控矿模式。

规则配置外置为 config/deposit_rules.yaml,修改后无需改代码即可生效。
若 YAML 文件不存在或解析失败,回退到内置硬编码默认值。

诚实边界:这是基于构造特征的**间接推理**,非直接矿床识别。结果为决策支持层,
需蚀变/地球化学/钻探等多源信息验证。
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. 规则加载: YAML 配置文件 → DEPOSIT_RULES + COMMODITY_MAP
# ---------------------------------------------------------------------------
_YAML_PATH = Path(__file__).resolve().parents[1] / "config" / "deposit_rules.yaml"


def _load_rules_from_yaml() -> tuple:
    """
    从 config/deposit_rules.yaml 加载规则。
    返回 (DEPOSIT_RULES, COMMODITY_MAP)。
    失败时返回 (None, None) 触发回退。
    """
    if not _YAML_PATH.exists():
        return None, None
    try:
        import yaml
        with open(_YAML_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        rules = data.get("deposit_rules", {})
        cmap = data.get("commodity_map", {})
        if not rules:
            return None, None
        return rules, cmap
    except Exception:
        return None, None


def _get_rules_and_map():
    """获取规则和映射,优先 YAML,回退硬编码。"""
    rules, cmap = _load_rules_from_yaml()
    if rules is not None:
        return rules, cmap or {}
    # 回退:硬编码默认值 (保证无 YAML 文件时也能工作)
    return _HARDCODED_RULES, _HARDCODED_MAP


# ---- 硬编码默认值 (仅当 YAML 文件不可用时使用) ----
_HARDCODED_RULES = {
    "蚀变岩型金矿(破碎带)": {
        "typical_strikes": [25, 45], "strike_tolerance": 30, "min_density": 0.005,
        "conjugate_bonus": True, "conjugate_pair": [80, 120],
        "elevation_range": [100, 1500], "curvature_valley_ratio": None,
        "slope_mean_max": None, "subsidence_hint": "goaf",
        "description": "构造破碎带控矿,NE/NNE走向断裂为主",
        "control_model": "破碎带蚀变岩型", "commodities": ["gold", "copper_gold"],
    },
    "斑岩型铜钼矿": {
        "typical_strikes": None, "strike_tolerance": 90, "min_density": 0.003,
        "conjugate_bonus": False, "conjugate_pair": None,
        "elevation_range": [200, 4000], "curvature_valley_ratio": None,
        "slope_mean_max": None, "strike_dispersion_bonus": True,
        "description": "多方向断裂+环形构造,中高海拔岩体侵入环境",
        "control_model": "岩体侵位构造", "commodities": ["copper", "molybdenum", "copper_gold"],
    },
    "矽卡岩型铁矿/铜矿": {
        "typical_strikes": [45, 90], "strike_tolerance": 45, "min_density": 0.002,
        "conjugate_bonus": True, "conjugate_pair": [30, 60],
        "elevation_range": [100, 2500], "curvature_valley_ratio": [0.3, 0.7],
        "slope_mean_max": None, "description": "接触带构造,NE-EW走向,脊谷交替",
        "control_model": "接触带控矿", "commodities": ["iron", "copper", "tungsten"],
    },
    "VMS型多金属矿": {
        "typical_strikes": None, "strike_tolerance": 90, "min_density": 0.001,
        "conjugate_bonus": False, "conjugate_pair": None,
        "elevation_range": [50, 2000], "curvature_valley_ratio": [0.4, 0.6],
        "slope_mean_max": None, "description": "火山-沉积建造,断裂发育程度中等",
        "control_model": "火山-沉积建造", "commodities": ["polymetallic", "copper", "gold", "zinc", "lead"],
    },
    "石英脉型金矿": {
        "typical_strikes": [0, 30], "strike_tolerance": 25, "min_density": 0.004,
        "conjugate_bonus": True, "conjugate_pair": [60, 100],
        "elevation_range": [50, 1500], "curvature_valley_ratio": None,
        "slope_mean_max": None, "strike_concentration_bonus": True,
        "description": "石英脉沿断裂充填,走向集中",
        "control_model": "断裂充填脉型", "commodities": ["gold"],
    },
    "沉积型矿产(煤/铝土/盐类)": {
        "typical_strikes": None, "strike_tolerance": 90, "min_density": 0.0,
        "conjugate_bonus": False, "conjugate_pair": None,
        "elevation_range": [0, 800], "curvature_valley_ratio": None,
        "slope_mean_max": 10.0, "subsidence_hint": "goaf",
        "description": "沉积盆地/缓坡区,断裂不发育",
        "control_model": "沉积盆地", "commodities": ["coal", "bauxite", "salt"],
    },
    "常规油气藏(微渗漏)": {
        "typical_strikes": None, "strike_tolerance": 90, "min_density": 0.0,
        "conjugate_bonus": False, "conjugate_pair": None,
        "elevation_range": [0, 2000], "curvature_valley_ratio": None,
        "slope_mean_max": 20.0,
        "description": "沉积盆地油气藏,断裂不发育;覆盖平原盆地与黄土高原含油气区",
        "control_model": "构造-地层圈闭", "commodities": ["petroleum", "gas", "oil"],
    },
    "致密油气/页岩气": {
        "typical_strikes": [30, 60], "strike_tolerance": 45, "min_density": 0.002,
        "conjugate_bonus": False, "conjugate_pair": None,
        "elevation_range": [100, 2500], "curvature_valley_ratio": None,
        "slope_mean_max": 25.0,
        "description": "致密储层受区域构造/裂缝控制;覆盖鄂尔多斯致密油气/四川页岩气等",
        "control_model": "裂缝型致密储层", "commodities": ["petroleum", "gas", "coalbed_gas", "shale_gas"],
    },
    "煤层气/煤炭": {
        "typical_strikes": None, "strike_tolerance": 90, "min_density": 0.0,
        "conjugate_bonus": False, "conjugate_pair": None,
        "elevation_range": [0, 1800], "curvature_valley_ratio": None,
        "slope_mean_max": 15.0, "subsidence_hint": "goaf",
        "description": "含煤盆地/黄土高原矿区,缓坡-中坡,采空佐证",
        "control_model": "沉积盆地", "commodities": ["coal", "coalbed_gas"],
    },
    "BIF型铁矿": {
        "typical_strikes": None, "strike_tolerance": 90, "min_density": 0.001,
        "conjugate_bonus": False, "conjugate_pair": None,
        "elevation_range": [100, 2000], "curvature_valley_ratio": None,
        "slope_mean_max": None, "description": "条带状铁建造,受地层走向控制",
        "control_model": "沉积-变质型", "commodities": ["iron"],
    },
}

# 拼写修正:上面变量名打错了,此处保留正确引用
_HARDCODED_MAP = {
    "gold": ["蚀变岩型金矿(破碎带)", "石英脉型金矿"],
    "copper": ["斑岩型铜钼矿", "矽卡岩型铁矿/铜矿"],
    "molybdenum": ["斑岩型铜钼矿"],
    "iron": ["矽卡岩型铁矿/铜矿", "BIF型铁矿"],
    "petroleum": ["常规油气藏(微渗漏)", "致密油气/页岩气"],
    "oil": ["常规油气藏(微渗漏)"],
    "gas": ["常规油气藏(微渗漏)", "致密油气/页岩气"],
    "coalbed_gas": ["致密油气/页岩气", "煤层气/煤炭"],
    "coal": ["煤层气/煤炭"],
    "polymetallic": ["VMS型多金属矿", "矽卡岩型铁矿/铜矿"],
    "copper_gold": ["斑岩型铜钼矿", "蚀变岩型金矿(破碎带)"],
    "shale_gas": ["致密油气/页岩气"],
    "tungsten": ["矽卡岩型铁矿/铜矿"],
    "zinc": ["VMS型多金属矿"],
    "lead": ["VMS型多金属矿"],
    "bauxite": ["沉积型矿产(煤/铝土/盐类)"],
    "salt": ["沉积型矿产(煤/铝土/盐类)"],
}

# 保持向后兼容:模块级属性仍可被外部代码直接引用
DEPOSIT_RULES, COMMODITY_MAP = _get_rules_and_map()


# 项目名/AOI 名关键词 → 矿种代码(commodity code,与 COMMODITY_MAP 键一致)。
# 按"先具体后宽泛"顺序匹配,首个命中即返回;避免 "多金属" 被 "金" 误匹配等问题。
NAME_HINT_KEYWORDS = [
    ("页岩气", "shale_gas"),
    ("煤层气", "coalbed_gas"),
    ("多金属", "polymetallic"),
    ("金属", "polymetallic"),
    ("石油", "petroleum"),
    ("油气", "petroleum"),
    ("天然气", "gas"),
    ("煤", "coal"),
    ("铜", "copper"),
    ("铁", "iron"),
    ("钼", "molybdenum"),
    ("钨", "tungsten"),
    ("铅", "lead"),
    ("锌", "zinc"),
    ("铝土", "bauxite"),
    ("盐", "salt"),
    ("金", "gold"),
]


def infer_mineral_hint_from_name(name: Optional[str]) -> Optional[str]:
    """从项目名/AOI 名按关键词推断矿种代码,无命中返回 None。

    用于用户未显式选择矿种方向时的兜底引导:如项目名"测试油气"→"petroleum"。
    返回值为 COMMODITY_MAP 键(英文代码),可直接作为 mineral_hint 传入 infer_deposit_type。
    """
    if not name:
        return None
    text = str(name).lower()
    for kw, code in NAME_HINT_KEYWORDS:
        if kw in name or kw in text:
            return code
    return None


# ---------------------------------------------------------------------------
# 2. 辅助函数
# ---------------------------------------------------------------------------
def _strike_diff(a: float, b: float) -> float:
    """走向最小差值 [0, 90]。"""
    d = abs(a - b) % 180
    return min(d, 180 - d)


def _check_conjugate(strikes: List[float], pair_range: List[float]) -> bool:
    """检查走向列表中是否有与 pair_range 相近的第二组走向(共轭)。"""
    if not strikes or not pair_range:
        return False
    for s in strikes:
        lo, hi = pair_range
        if lo <= s <= hi or lo <= (s + 180) % 180 <= hi:
            return True
    return False


def _extract_terrain_stats(dem: Optional[np.ndarray], slope: Optional[np.ndarray],
                           svf: Optional[np.ndarray],
                           curvature: Optional[np.ndarray]) -> Dict:
    """从地形栅格提取统计特征。"""
    stats = {}
    if slope is not None:
        valid = np.isfinite(slope)
        if valid.any():
            stats["slope_mean"] = float(np.nanmean(slope[valid]))
            stats["slope_max"] = float(np.nanmax(slope[valid]))
    if svf is not None:
        valid = np.isfinite(svf)
        if valid.any():
            stats["svf_mean"] = float(np.nanmean(svf[valid]))
    if curvature is not None:
        valid = np.isfinite(curvature)
        if valid.any():
            pos = float((curvature[valid] > 0).sum())
            neg = float((curvature[valid] < 0).sum())
            total = pos + neg + 1e-9
            stats["curvature_valley_ratio"] = round(neg / total, 3)
    if dem is not None:
        valid = np.isfinite(dem)
        if valid.any():
            stats["elevation_min"] = float(np.nanmin(dem[valid]))
            stats["elevation_max"] = float(np.nanmax(dem[valid]))
    return stats


# ---------------------------------------------------------------------------
# 3. 核心推理函数
# ---------------------------------------------------------------------------
def infer_deposit_type(
    structural_stats: Dict,
    terrain_stats: Optional[Dict] = None,
    attribution_stats: Optional[Dict] = None,
    lineament_details: Optional[List[Dict]] = None,
    mineral_hint: Optional[str] = None,
) -> Dict:
    """
    基于 geo-stru 纯构造特征推理矿床类型。

    Args:
        structural_stats: metadata 的 structural_stats 字典
            (dominant_strikes_deg, n_lineaments, lineament_density_mean, elevation_range_m, ...)
        terrain_stats: {slope_mean, svf_mean, curvature_valley_ratio, ...} (可选)
        attribution_stats: {goaf: N, landslide: N, fault_creep: N, ...} (可选)
        lineament_details: 线性体段列表含 is_valley_candidate (可选)
        mineral_hint: 矿种方向(如 "gold"/"petroleum"/"copper"/"iron" 等, 可选)

    Returns:
        {
            "candidates": [{deposit_type, confidence, evidence, control_model}, ...],
            "structural_control_summary": str,
            "primary_model": str,
            "primary_confidence": float,
        }
    """
    terrain_stats = terrain_stats or {}
    attribution_stats = attribution_stats or {}

    # 提取输入特征
    strikes = structural_stats.get("dominant_strikes_deg") or []
    density = structural_stats.get("lineament_density_mean", 0.0)
    n_lin = structural_stats.get("n_lineaments", 0)
    elev_range = structural_stats.get("elevation_range_m") or [0, 9999]

    # 走向集中度: 用 dominant_strikes 间差值度量
    strike_concentration = 0.0
    if len(strikes) >= 2:
        diffs = [_strike_diff(strikes[i], strikes[i + 1]) for i in range(len(strikes) - 1)]
        mean_diff = float(np.mean(diffs))
        # 差值小→集中,差值大→分散; 映射到 [0,1]: 0°=最集中, 90°=最分散
        strike_concentration = 1.0 - min(mean_diff / 90.0, 1.0)

    # 河谷候选比例
    valley_ratio = 0.0
    if lineament_details:
        n_valley = sum(1 for s in lineament_details if s.get("is_valley_candidate"))
        valley_ratio = n_valley / max(len(lineament_details), 1)

    # 归因统计
    n_goaf = attribution_stats.get("goaf", 0)
    n_fault_creep = attribution_stats.get("fault_creep", 0)
    n_landslide = attribution_stats.get("landslide", 0)

    candidates = []
    for dtype, rule in DEPOSIT_RULES.items():
        score = 0.0
        evidence = []

        # ---- 1. 走向匹配 (权重 0.30) ----
        if rule["typical_strikes"] is not None and strikes:
            best_match = min(
                (_strike_diff(strikes[0], ts) for ts in rule["typical_strikes"]),
                default=90
            )
            tol = rule["strike_tolerance"]
            if best_match <= tol:
                # 线性衰减: 0°差→满分1, tol°差→0
                s = 1.0 - best_match / tol
                score += 0.30 * s
                evidence.append(f"主走向{strikes[0]:.0f}°匹配{dtype}({best_match:.0f}°差)")
            else:
                score += 0.0
                evidence.append(f"主走向{strikes[0]:.0f}°偏离{dtype}({best_match:.0f}°差)")
        elif rule["typical_strikes"] is None:
            # 走向不敏感→给半分
            score += 0.15
            evidence.append("走向不敏感类型")
        else:
            # 无走向数据
            score += 0.10

        # ---- 2. 断裂密度 (权重 0.20) ----
        min_d = rule["min_density"]
        if density >= min_d:
            ratio = min(density / max(min_d, 1e-9), 3.0)
            score += 0.20 * min(ratio, 1.0)
            evidence.append(f"断裂密度{density:.4f}(≥{min_d})")
        else:
            # 低于下限→衰减
            if min_d > 0:
                score += 0.20 * (density / min_d) * 0.3
            evidence.append(f"断裂密度低{density:.4f}(<{min_d})")

        # ---- 3. 共轭性 (权重 0.15) ----
        if rule["conjugate_bonus"] and rule["conjugate_pair"]:
            if _check_conjugate(strikes[1:] if len(strikes) > 1 else [], rule["conjugate_pair"]):
                score += 0.15
                evidence.append("存在共轭走向")
            else:
                score += 0.0
                evidence.append("未检出共轭走向")
        elif rule["conjugate_bonus"] and len(strikes) >= 2:
            # 有共轭加分但未指定 pair,检查任意两组差是否 >40°
            if _strike_diff(strikes[0], strikes[1]) > 40:
                score += 0.10
                evidence.append("多组走向(潜在共轭)")
        else:
            score += 0.05  # 不要求共轭→给少量分

        # ---- 4. 地形匹配 (权重 0.15) ----
        elev_lo, elev_hi = rule["elevation_range"]
        elev_mid = (elev_range[0] + elev_range[1]) / 2 if len(elev_range) == 2 else 500
        if elev_lo <= elev_mid <= elev_hi:
            score += 0.08
            evidence.append(f"高程{elev_mid:.0f}m在{dtype}范围内")
        else:
            score += 0.02

        slope_max_cond = rule.get("slope_mean_max")
        slope_mean = terrain_stats.get("slope_mean")
        if slope_max_cond is not None and slope_mean is not None:
            if slope_mean <= slope_max_cond:
                score += 0.07
                evidence.append(f"平均坡度{slope_mean:.1f}°(缓坡)")
            else:
                score += 0.01
                evidence.append(f"平均坡度{slope_mean:.1f}°(偏陡)")
        else:
            score += 0.04

        cvr = rule.get("curvature_valley_ratio")
        actual_cvr = terrain_stats.get("curvature_valley_ratio")
        if cvr is not None and actual_cvr is not None:
            if cvr[0] <= actual_cvr <= cvr[1]:
                score += 0.05
                evidence.append(f"脊谷比{actual_cvr:.2f}匹配")
        else:
            score += 0.02

        # ---- 5. 形变佐证 (权重 0.10) ----
        hint = rule.get("subsidence_hint")
        if hint == "goaf" and n_goaf > 0:
            score += 0.10
            evidence.append(f"检出采空沉降({n_goaf}处)")
        elif hint == "goaf" and n_goaf == 0:
            score += 0.0
        elif n_fault_creep > 0 and rule.get("conjugate_bonus"):
            score += 0.05
            evidence.append("活动断裂佐证")
        else:
            score += 0.02

        # ---- 6. 走向集中/分散特殊加分 (权重 0.10) ----
        if rule.get("strike_concentration_bonus") and strike_concentration > 0.6:
            score += 0.10
            evidence.append(f"走向高度集中({strike_concentration:.2f})")
        elif rule.get("strike_dispersion_bonus") and strike_concentration < 0.3:
            score += 0.10
            evidence.append("走向分散(环形构造特征)")
        else:
            score += 0.03

        confidence = round(float(np.clip(score, 0.0, 1.0)), 3)
        candidates.append({
            "deposit_type": dtype,
            "confidence": confidence,
            "evidence": evidence,
            "control_model": rule["control_model"],
            "description": rule["description"],
            "related_commodities": rule.get("commodities", []),
        })

    # ---- mineral_hint 加分 ----
    hinted_types = set()
    if mineral_hint:
        hint_lower = mineral_hint.lower().strip()
        # 直接查 COMMODITY_MAP
        if hint_lower in COMMODITY_MAP:
            hinted_types = set(COMMODITY_MAP[hint_lower])
        else:
            # 模糊匹配: 检查 commodities 字段
            for c in candidates:
                commodities = c.get("related_commodities", [])
                if any(hint_lower in (com or "").lower() for com in commodities):
                    hinted_types.add(c["deposit_type"])
        # 对相关类型加 0.15 bias
        if hinted_types:
            for c in candidates:
                if c["deposit_type"] in hinted_types:
                    c["confidence"] = round(min(c["confidence"] + 0.15, 1.0), 3)
                    c["evidence"].append(f"矿种方向({mineral_hint})匹配加分")

    # 按置信度排序
    candidates.sort(key=lambda c: c["confidence"], reverse=True)

    # 构造控矿综合描述
    primary = candidates[0] if candidates else None
    summary = _generate_summary(strikes, density, n_lin, primary, attribution_stats)

    return {
        "candidates": candidates,
        "structural_control_summary": summary,
        "primary_model": primary["deposit_type"] if primary else "未确定",
        "primary_confidence": primary["confidence"] if primary else 0.0,
        "mineral_hint": mineral_hint or None,
    }


def _generate_summary(strikes, density, n_lin, primary, attribution_stats) -> str:
    """生成构造控矿综合描述(中文)。"""
    parts = []

    # 走向
    if strikes:
        dirs = []
        for s in strikes[:3]:
            dirs.append(f"{s:.0f}°")
        parts.append(f"主构造走向{', '.join(dirs)}")

    # 密度
    if density > 0.01:
        parts.append("断裂密集发育")
    elif density > 0.003:
        parts.append("断裂中等发育")
    elif n_lin > 0:
        parts.append("断裂稀疏")
    else:
        parts.append("未检出线性构造")

    # 控矿程度
    if primary and primary["confidence"] >= 0.5:
        parts.append(f"推断{primary['control_model']}控矿模式")
    elif primary and primary["confidence"] >= 0.3:
        parts.append("构造控矿程度中等")
    else:
        parts.append("构造控矿证据不足")

    # 形变
    n_goaf = attribution_stats.get("goaf", 0)
    n_fc = attribution_stats.get("fault_creep", 0)
    if n_goaf > 0:
        parts.append(f"检出采空沉降{n_goaf}处(在采矿区佐证)")
    if n_fc > 0:
        parts.append(f"检出断裂蠕动{n_fc}处(活动构造)")

    return "；".join(parts) + "。"
