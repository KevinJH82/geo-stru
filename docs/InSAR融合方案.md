# 招远庙山金矿 InSAR × 遥感地质构造解译 融合方案（方法文档）

## Context（为什么做这件事）

针对一套位于**山东招远庙山金矿**的 InSAR 数据，回答：这对**地质构造解译**有没有帮助？如果有，给出一套融合方案。

数据与代码现状：

- **数据**（`…/INSAR数据/招远`）：MintPy SBAS 结果（HyP3 Sentinel‑1 → `smallbaselineApp`）。AOI 极小（37.1774–37.2003°N, 120.4293–120.4562°E，约 2.5×2.4 km，61×65 像元，UTM 50N）。**时序很短：5 景、48 天（2025‑11‑12→12‑30）、4 个干涉对单链、单轨**；`deramp=linear`、ERA5 对流层。相干性极高（temporalCoherence≈1.0，冬季少植被、基岩/矿区地表，非常适合 InSAR）。velocity（±20 mm/yr）当前以噪声为主，未见清晰的断块差异或沉降漏斗。
- **代码**（`/opt/Project/deepexplor-services`）：`geo-stru` 正是遥感地质构造解译系统（DEM→线性体→`distance_to_lineament`/`density`/`lineaments.geojson`/玫瑰图）；`geo-insar` 产出 LOS 形变/速率但**不消费构造数据**；**目前没有任何模块把 InSAR 形变反哺到构造解译**——这正是问题指向的空白。可复用件：`geo-stru/core/lineament.py:extract_lineaments`、`commons/insar_utils.py`（`los_to_vertical`/`coherence_mask`）、`commons/structural_broker.py`、`commons/insar_schema.json` + `structural_schema.json`。

**交付物**：纯方法/设计文档（不写代码）。**方向**：四条线全做、双向闭环。

**诚实结论（先行）**：站点与方法**适合**用 InSAR 支撑构造解译；但**这套具体数据时序太短（48 天）**，只能作方法验证/定性 POC，**可靠定量结论需扩展时序**。本方案先把融合框架与方法定清楚，并明确数据要求。

---

## 一、InSAR 对地质构造解译的价值（机理，结合胶东金矿背景）

招远属胶东金矿省，**蚀变岩型（破碎带）金矿、构造控矿**，矿体受 NE/NNE 向断裂（招平断裂系）控制。InSAR 形变场可从五个机理支撑构造解译：

1. **形变梯度 / 不连续 → 活动与隐伏断裂定位**：跨断裂的差异 LOS 运动表现为速率突变带，可定位地表/近地表活动断裂；地形无表达的隐伏断裂也能被形变梯度揭示。
2. **相干性纹理 → 构造/岩性边界**：相干性的线状突变常沿断裂破碎带或岩性界面分布，是地形解译的独立佐证。
3. **沉降漏斗 → 采空区（goaf）圈定**：地下开采引起的相干负速率漏斗，其外包络勾勒采空边界；漏斗长轴常平行矿体/控矿断裂走向，反演构造产状。
4. **LOS 时序 → 断裂活动性 / 蠕动 vs 锁定**：累计形变曲线区分活动蠕动段与锁定段。
5. **空间一致性交叉验证**：地形提取的“线性体”与形变提取的“形变线性体”叠合，可剔除水系/人工地物造成的假线性体，提高构造解译置信度。

## 二、当前数据体检（诚实评估）

- **优点**：相干性极高、AOI 精准聚焦矿区、已做 ERA5 对流层与 demErr 改正、管线规整。
- **局限**：
  - 时间跨度 < 2 个月 → 年速率信噪比低，难分离构造信号与大气/季节噪声；
  - 单一几何（单轨）→ 仅 1 维 LOS，**无法分解垂直/水平**，得不到断层滑动矢量；
  - `deramp=linear` 去长波 → 区域构造梯度可能被部分削弱（2.5 km 尺度影响有限，主要去轨道/大气坡面）；
  - AOI 过小 → 招平等主干断裂延伸出界，看不到完整断裂系。
- **判定**：当前数据**适合做“方法验证 + 定性套合”**，活动性/沉降速率的**定量结论需扩展时序**（见第八节）。

## 三、融合总体架构（双向）

```
  ┌────────────────────┐        ┌────────────────────┐
  │  geo-stru（构造）   │        │  geo-insar（形变）  │
  │ lineaments / 距断裂 │        │ LOS 速率/时序/相干  │
  └─────────┬──────────┘        └──────────┬─────────┘
            │  structural_broker            │  insar_schema
            ▼                               ▼
        ┌───────────────────────────────────────┐
        │        融合层（统一栅格 + 四条线）       │
        │  A 反哺构造  B 采空圈定  C 归因  D 闭环  │
        └───────────────────────────────────────┘
```
统一工作栅格：以 geo-stru 的 DEM/AOI 栅格为基准（EPSG:4326），MintPy（UTM 50N）经 `rasterio.warp.reproject` 重投影/重采样对齐。

## 四、数据准备与对齐（预处理）

1. **MintPy h5 → GeoTIFF 标准化**（这恰好补齐 geo-insar 当前缺失的 MintPy ingestion）：
   - `velocity.h5` → `los_velocity_mm_yr.tif`；`timeseries(_ramp_demErr).h5` → 逐期/累计 `los_cumdisp_*.tif`；`temporalCoherence.h5` → `temporal_coh.tif`；`inputs/geometryGeo.h5` → `incidence_angle.tif`/`azimuth_angle.tif`。
   - 工具：MintPy `save_gdal.py` 或 h5py 自写导出（数据已地理编码，仅需投影/单位透传）。
2. **掩膜**：`temporalCoherence > 0.7` 且 `waterMask` 有效；复用 `insar_utils.coherence_mask`。
3. **对齐**：统一投影、重采样到 geo-stru AOI 栅格；裁剪到公共 ROI。
4. **元数据**：遵循 `insar_schema.json`（`source`、`incidence_angle_mean≈38.8°`、`date_range`、`orbit_direction`、`stats`），便于 broker 发现与下游消费。

## 五、四条融合分析线

### A. InSAR 反哺构造解译（落点 geo-stru）
- **A1 形变线性体提取**：算 LOS 速率梯度幅值 `|∇v|`；**复用 `extract_lineaments()`**（把梯度场当作“多方位阴影”输入）提取“形变线性体”，输出 `deformation_lineaments.geojson` + 玫瑰图。
- **A2 线性体活动性打标**：对 `lineaments.geojson` 每条线做**垂直剖面**，采样两侧 LOS 速率差 `ΔV_LOS`，结合相干性/有效样本数→活动性评分。分类：
  - *地形+形变一致* → **活动断裂**；*仅地形* → 古/锁定；*仅形变* → **隐伏断裂**（地形无表达）。
  - 输出 `lineaments_activity.geojson`，新增属性 `dv_los_mm_yr / coh / n_valid / activity_class / activity_score`。
- **A3 走向一致性**：地形线性体主方向 vs 形变线性体主方向玫瑰叠加，并与区域 NE/NNE 招平系对比。

### B. 采空区 / 沉降圈定（金矿重点）
- **B1 沉降探测**：在 LOS 速率上检测**相干负值连通域**（连通分量标注，参照 geo-analyser `los_velocity_clustering` 思路），阈值（如 `< −X mm/yr`）+ 面积/形态过滤。
- **B2 LOS→垂直**：复用 `insar_utils.los_to_vertical`（入射角≈38.8°，**明确标注“近垂直运动”假设**）估垂直沉降。
- **B3 goaf 圈定**：沉降漏斗外包络→采空边界多边形 `goaf_polygons.geojson`；漏斗长轴方向 vs 控矿断裂走向对比（矿体多沿断裂带，沉降椭圆长轴常平行矿体走向）。
- **B4 时序判据**：用 `timeseries` 判沉降是否线性/加速 → 活动采空 vs 稳定。

### C. 形变归因（落点 geo-insar，消费 structural_broker）
- 经 `structural_broker` 拉取 geo-stru 的 `distance_to_lineament`/`density`；对每个形变连通域按规则归类：
  - 距断裂近 + 沿走向线状 → **断裂蠕动**；坡向一致 + 坡度大 → **滑坡**；圆/椭圆漏斗 + 矿区 → **采空沉降**。
  - 输出 `deformation_attribution.{tif,geojson}`（类别 + 置信度），写回 broker 供 `geo-reporter` 消费。

### D. 双向闭环
- geo-stru 活动性标签 → 作为 geo-insar 归因先验；geo-insar 沉降/活动结果 → 反向精化 geo-stru 线性体置信度。可迭代一轮。

## 六、产品清单

| 产品 | 格式 | 内容 | 消费方 |
|---|---|---|---|
| `los_velocity_mm_yr.tif` / `vertical_velocity_mm_yr.tif` | GeoTIFF | LOS/垂直速率 | 融合层、reporter |
| `temporal_coh.tif` | GeoTIFF | 可靠性掩膜 | 全部 |
| `deformation_lineaments.geojson` | GeoJSON | 形变线性体 + 走向 | geo-stru |
| `lineaments_activity.geojson` | GeoJSON | 线性体活动性标签 | geo-stru/exploration |
| `goaf_polygons.geojson` / `subsidence_funnels.geojson` | GeoJSON | 采空边界/沉降漏斗 | reporter/矿方 |
| `deformation_attribution.{tif,geojson}` | GeoTIFF/GeoJSON | 形变类别 | geo-insar/reporter |
| 图件 | PNG | ①阴影+线性体+LOS速率叠加 ②跨断层 LOS 剖面 ③地形vs形变玫瑰对比 ④沉降漏斗与控矿构造叠合 ⑤时序曲线 | reporter |
| `fusion_metadata.json` | JSON | 遵循双 schema，链接两侧 run_id | broker |

## 七、与现有架构的衔接（最大化复用，不重造轮子）

- **复用**：`core/lineament.py`（`extract_lineaments` / `write_lineaments_geojson` / `plot_rose_diagram`）；`commons/insar_utils.py`（`los_to_vertical` / `coherence_mask` / `read/validate_metadata`）；`commons/structural_broker.py`；`commons/insar_schema.json` + `structural_schema.json`。
- **将来实施落点**（本次仅文档）：A/B → `geo-stru/core/insar_fusion.py`；C → geo-insar 侧 attribution 步骤；可选 `commons/fusion_broker.py` 串联。
- **顺带补缺**：geo-insar 目前无 MintPy ingestion，本方案第四节的“MintPy→GeoTIFF 标准化”正好补上。

## 八、可信度与数据要求（诚实边界）

- 当前数据：**仅 POC / 定性**。
- 要做可靠定量，建议：
  - 时序 **≥ 20–30 景、≥ 1 年**（最好 2 年），覆盖矿山生产周期；
  - **升 + 降双轨 → 2D 分解**（垂直 + 东西向），方能区分沉降 vs 水平断裂运动；
  - 验证 ERA5 大气改正效果，参考点选稳定基岩；矿区建筑/角反射器作 PS 提精度；
  - 与已知矿体/巷道/地质图/钻孔套合验证。
- **风险**：单点采空等短波形变可能被 `deramp`/参考点吸收；解缠误差；形变线性体可能混入水系/人工地物假象（与地形线性体同源局限，已用坡度门控+长度过滤缓解）。

## 九、实施阶段建议（roadmap）

P0 数据标准化（MintPy→GeoTIFF+mask）→ P1 A 线（活动性）→ P2 B 线（采空）→ P3 C 线（归因）→ P4 双向闭环 + 报告。

## 验证（方法层面如何检验）

用本 AOI 跑一次 POC：检验 A2 是否给出 ≥1 条“形变一致”的线性体；B1 是否检出已知采空沉降；与矿方已知采空区/矿体走向对照。**因数据短，主要验证“管线跑通 + 空间套合合理”，不下定量结论。**
