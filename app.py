#!/usr/bin/env python3
"""遥感地质构造解译系统 - 独立 Flask 应用"""

import os
import re
import time
import json
import threading
import zipfile
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_file, Response
from loguru import logger

from config.config import Config
from core.structural_engine import StructuralEngine
from core.deposit_inference import infer_mineral_hint_from_name
from core import delivery
from utils.file_utils import get_file_size
from utils.ids import new_task_code

app = Flask(__name__)
# ── 内部鉴权:拒绝绕过 BFF 的直连(PORTAL_INTERNAL_KEY 配置后生效) ──
try:
    import sys as _ia_sys
    if '/opt/deepexplor-services' not in _ia_sys.path:
        _ia_sys.path.insert(0, '/opt/deepexplor-services')
    from commons.internal_auth import init_internal_auth as _init_internal_auth
    _init_internal_auth(app)
except Exception as _ia_e:
    print(f'[internal_auth] 跳过接入: {_ia_e}')
app.secret_key = Config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH

# loguru 默认输出到 stderr; 如需文件日志,按 Config.LOG_FILE 配置 sink。
# extra[task_code] 默认 "-"，分析任务在后台线程内用 logger.contextualize 注入具体编码,
# 使该任务的所有日志行都带上编码,便于按编码 grep 回溯。
_LOG_FORMAT = ("{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {extra[task_code]} | "
               "{name}:{function}:{line} - {message}")
logger.configure(
    handlers=[{"sink": __import__('sys').stderr, "level": Config.LOG_LEVEL,
               "format": _LOG_FORMAT}],
    extra={"task_code": "-"},
)
if Config.LOG_FILE:
    os.makedirs(os.path.dirname(Config.LOG_FILE), exist_ok=True)
    logger.add(Config.LOG_FILE, level=Config.LOG_LEVEL,
               rotation="10 MB", retention="7 days",
               format=_LOG_FORMAT)

task_counter = 0
analysis_tasks = {}
_tasks_lock = threading.Lock()

# 任务状态清理: 完成的任务超过此阈值后,清理最早的,防止内存无限增长。
MAX_COMPLETED_TASKS = 50

STRUCTURAL_EXTENSIONS = {'kml', 'kmz', 'ovkml', 'xlsx', 'xls', 'csv'}


def _cleanup_old_tasks():
    """当已完成/失败的任务超过阈值时,删除最早的条目以释放内存。"""
    global analysis_tasks
    with _tasks_lock:
        finished = [(tid, t) for tid, t in analysis_tasks.items()
                    if t.get('status') in ('completed', 'failed')]
        if len(finished) <= MAX_COMPLETED_TASKS:
            return
        # 按创建时间排序,删除最早的
        finished.sort(key=lambda x: x[1].get('start_time', ''))
        to_remove = finished[:len(finished) - MAX_COMPLETED_TASKS]
        for tid, _ in to_remove:
            del analysis_tasks[tid]
        logger.info(f"已清理 {len(to_remove)} 个历史任务, 当前保留 {len(analysis_tasks)} 个")


@app.route('/')
def index():
    return render_template('structural.html')


_REPORT_MD = Path(__file__).parent / 'docs' / '图件产物参数说明与分析.md'


@app.route('/docs/image-parameters')
def docs_image_parameters():
    """在线查看《图件产物参数说明与分析》——md 用 marked.js 渲染。"""
    try:
        md_text = _REPORT_MD.read_text(encoding='utf-8')
    except Exception as e:
        return Response(f"报告读取失败: {e}", status=500, mimetype='text/plain; charset=utf-8')
    md_js = json.dumps(md_text)
    html = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>图件产物参数说明与分析 · geo-stru</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", sans-serif; color: #1e293b; line-height: 1.7; max-width: 1100px; margin: 0 auto; padding: 32px 40px; background: #fafbfc; }
  h1 { font-size: 28px; margin: 0 0 16px; padding-bottom: 10px; border-bottom: 3px solid #2563eb; }
  h2 { font-size: 20px; margin: 32px 0 12px; padding-bottom: 6px; border-bottom: 2px solid #2563eb; color: #1d4ed8; }
  h3 { font-size: 16px; margin: 20px 0 8px; color: #334155; }
  h4 { font-size: 14px; margin: 14px 0 6px; color: #475569; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; margin: 10px 0; background: white; }
  th, td { padding: 7px 10px; border: 1px solid #e2e8f0; text-align: left; vertical-align: top; }
  th { background: #f1f5f9; font-weight: 600; }
  code { background: #f1f5f9; padding: 1px 6px; border-radius: 3px; font-family: "Menlo", "Consolas", monospace; font-size: 0.9em; color: #be123c; }
  pre code { display: block; padding: 12px 16px; background: #1e293b; color: #f1f5f9; border-radius: 6px; overflow-x: auto; }
  blockquote { border-left: 3px solid #94a3b8; padding-left: 14px; color: #475569; margin: 12px 0; background: #f8fafc; padding: 8px 14px; border-radius: 0 4px 4px 0; }
  hr { border: none; border-top: 1px solid #cbd5e1; margin: 24px 0; }
  a { color: #2563eb; text-decoration: none; }
  a:hover { text-decoration: underline; }
  ul, ol { padding-left: 26px; }
  li { margin: 3px 0; }
  .toolbar { position: sticky; top: 0; background: rgba(250,251,252,0.95); backdrop-filter: blur(6px); padding: 12px 0; margin: -12px 0 8px; border-bottom: 1px solid #e2e8f0; z-index: 10; }
  .toolbar a { font-size: 12px; padding: 6px 12px; background: white; border: 1px solid #cbd5e1; border-radius: 6px; margin-right: 8px; display: inline-block; }
  .toolbar a:hover { background: #f1f5f9; text-decoration: none; }
</style></head>
<body>
<div class="toolbar">
  <a href="javascript:window.print();">🖨 打印 / 存 PDF</a>
  <a href="/docs/image-parameters.md" download="geo-stru_图件产物参数说明与分析.md">📥 下载 Markdown</a>
  <a href="/">↩ 返回</a>
</div>
<div id="content"></div>
<script>
  const md = __MD_TEXT__;
  document.getElementById('content').innerHTML = marked.parse(md, { gfm: true, breaks: false });
</script>
</body></html>"""
    return Response(html.replace("__MD_TEXT__", md_js), mimetype='text/html; charset=utf-8')


@app.route('/docs/image-parameters.md')
def docs_image_parameters_md():
    """下载报告 Markdown 原文。"""
    try:
        md_text = _REPORT_MD.read_text(encoding='utf-8')
    except Exception as e:
        return Response(f"报告读取失败: {e}", status=500, mimetype='text/plain; charset=utf-8')
    return Response(md_text, mimetype='text/markdown; charset=utf-8')


@app.route('/api/upload_data', methods=['POST'])
def upload_data():
    """上传并解压卫星数据ZIP包"""
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'success': False, 'message': '未选择文件'})

        ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
        if ext != 'zip':
            return jsonify({'success': False, 'message': '请上传 ZIP 格式的卫星数据包'})

        extract_dir = os.path.join(Config.UPLOAD_FOLDER, 'structural', f'data_{int(time.time())}')
        os.makedirs(extract_dir, exist_ok=True)

        zip_path = os.path.join(extract_dir, 'data.zip')
        file.save(zip_path)

        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
        os.remove(zip_path)

        checks = {'DEM': False, 'Landsat': False, 'ASTER': False}
        for root, dirs, files in os.walk(extract_dir):
            for f in files:
                fl = f.lower()
                if fl.startswith('dem') and fl.endswith('.tif'):
                    checks['DEM'] = True
                if 'landsat' in root.lower() or 'landsat' in fl:
                    checks['Landsat'] = True
                if fl.startswith('b') and fl.endswith('.tif') and not checks['Landsat']:
                    try:
                        bn = int(fl[1:].replace('.tif', '').rstrip('n'))
                        if 1 <= bn <= 11:
                            checks['Landsat'] = True
                    except ValueError:
                        pass
                if 'aster' in root.lower() or 'aster' in fl:
                    checks['ASTER'] = True

        return jsonify({
            'success': True,
            'data_dir': extract_dir,
            'file_size': get_file_size(extract_dir),
            'checks': checks,
        })
    except Exception as e:
        logger.error(f"卫星数据上传失败: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/upload_area', methods=['POST'])
def upload_area():
    """上传并解析KML/ROI区域文件"""
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'success': False, 'message': '未选择文件'})

        ext = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
        if ext not in STRUCTURAL_EXTENSIONS:
            return jsonify({'success': False, 'message': f'不支持的文件类型: .{ext}，请上传 KML/KMZ/OVKML 或 Excel/CSV 文件'})

        save_dir = os.path.join(Config.UPLOAD_FOLDER, 'structural')
        os.makedirs(save_dir, exist_ok=True)

        safe_name = f"area_{int(time.time())}.{ext}"
        file_path = os.path.join(save_dir, safe_name)
        file.save(file_path)

        coords = StructuralEngine.parse_polygon(file_path)

        # 按 ROI 定位交付项目:名字优先,失败时按几何覆盖兜底(KML 改名也能命中)。
        # 探测冬季数据(DEM 必需 / Landsat 可选),替代上传卫星数据 ZIP 的方式。
        roi_geojson = None
        if coords:
            ring = [list(p) for p in coords]
            roi_geojson = {"type": "Polygon", "coordinates": [ring]}
        res = delivery.resolve_project_dir_verbose(
            file.filename, roi_geojson=roi_geojson,
            delivery_id=request.headers.get("X-Delivery-Id", ""))
        project_dir = res.get("dir")
        resolved = {}
        if project_dir:
            wd = delivery.locate_winter_data(project_dir)
            resolved = {
                'project_name': os.path.basename(str(project_dir)),
                'resolved_by': res.get("method"),   # exact / normalized / spatial
                'dem_available': bool(wd.get('dem')),
                'landsat_available': bool(wd.get('landsat_dir')),
                'landsat_sensor': wd.get('landsat_sensor'),
            }
        else:
            # 未命中:回传候选交付,供前端提示"最接近的是哪几个"而非干报错
            resolved = {'project_name': None, 'candidates': res.get("candidates") or []}

        return jsonify({
            'success': True,
            'file_path': file_path,
            'filename': file.filename,
            'file_size': get_file_size(file_path),
            'polygon_coords': coords,
            'resolved': resolved,   # 自动定位到的交付项目与冬季数据可用性(可能为空)
        })
    except Exception as e:
        logger.error(f"区域文件解析失败: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/list_projects')
def api_list_projects():
    """列出交付库项目(供前端下拉,在 ROI 文件名无法自动匹配时手动选择)。"""
    return jsonify({'success': True, 'projects': delivery.list_projects()})


@app.route('/api/project_data')
def api_project_data():
    """探测某交付项目冬季的 DEM / Landsat 可用性。"""
    name = request.args.get('project', '')
    pd = delivery.resolve_project_dir(name)
    if not pd:
        return jsonify({'success': False, 'message': '未在交付库找到该项目'})
    wd = delivery.locate_winter_data(pd)
    return jsonify({
        'success': True,
        'project_name': os.path.basename(str(pd)),
        'dem_available': bool(wd.get('dem')),
        'landsat_available': bool(wd.get('landsat_dir')),
        'landsat_sensor': wd.get('landsat_sensor'),
    })


@app.route('/api/start', methods=['POST'])
def start_generation():
    """启动遥感地质构造解译图生成任务"""
    global task_counter

    try:
        params = request.json
        file_path = params.get('file_path')
        project_name = params.get('project_name')

        if not file_path or not os.path.exists(file_path):
            return jsonify({'success': False, 'message': '区域文件不存在，请重新上传'})

        # 从交付库(冬季)定位卫星数据,替代上传 ZIP:DEM 必需,Landsat 可选。
        project_dir = delivery.resolve_project_dir(project_name) if project_name else None
        if project_dir is None:
            project_dir = delivery.resolve_project_dir(os.path.basename(file_path))
        if project_dir is None:
            return jsonify({'success': False, 'message': '未在交付库定位到对应项目,请在下拉框选择项目'})

        wd = delivery.locate_winter_data(project_dir)
        dem_path = wd.get('dem')
        landsat_dir = wd.get('landsat_dir')
        if not dem_path:
            return jsonify({'success': False,
                            'message': f'交付项目「{os.path.basename(str(project_dir))}」冬季子目录无 DEM.tif,无法生成构造解译图'})

        # 租户上下文:BFF 经 /svc 注入 X-Tenant-Id;线程内无 request 上下文,先在此捕获
        tenant_id = request.headers.get('X-Tenant-Id')

        task_id = f"struct_{task_counter:04d}"
        task_counter += 1
        # 全局唯一任务编码:写入 metadata / 嵌入目录名 / 绑定日志 / 返回前端,凭此回溯。
        task_code = new_task_code()

        # 用交付项目名作为 AOI 名(规范、与平台一致、可被下游 broker 发现),
        # 而非上传时改写的 area_<ts> 文件名。每次分析存为独立 run 子目录,不覆盖历史。
        aoi_name = os.path.basename(str(project_dir))
        safe_aoi = re.sub(r'[\\/:*?"<>|]+', '_', aoi_name).strip() or task_id

        # 矿种方向:用户显式选择优先;漏选时按项目名/AOI 名关键词兜底推断
        # (如"测试油气"→petroleum),避免无引导时构造推理偏向金属矿。
        raw_hint = params.get('mineral_hint')
        mineral_hint = (raw_hint or infer_mineral_hint_from_name(aoi_name)
                        or infer_mineral_hint_from_name(project_name))
        hint_auto = bool(mineral_hint) and not raw_hint

        run_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + task_code
        output_dir = os.path.join(Config.RESULTS_FOLDER, safe_aoi, 'structural', run_id)
        analysis_tasks[task_id] = {
            'id': task_id,
            'task_code': task_code,
            'aoi_name': aoi_name,
            'status': 'running',
            'progress': 0,
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'logs': [],
            'results': None,
        }

        def run_task():
            with logger.contextualize(task_code=task_code):
                try:
                    coords = StructuralEngine.parse_polygon(file_path)

                    def on_log(msg, level='INFO'):
                        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        analysis_tasks[task_id]['logs'].append(f"[{ts}] [{level}] [{task_code}] {msg}")
                        if len(analysis_tasks[task_id]['logs']) > 100:
                            analysis_tasks[task_id]['logs'] = analysis_tasks[task_id]['logs'][-100:]

                    if hint_auto:
                        on_log(f"未指定矿种方向,已按项目名「{aoi_name}」自动推断为: {mineral_hint}", 'INFO')

                    analysis_tasks[task_id]['progress'] = 10

                    results = StructuralEngine.generate_maps(
                        dem_path=dem_path,
                        polygon_coords=coords,
                        output_dir=output_dir,
                        landsat_dir=landsat_dir if params.get('use_landsat', True) else None,
                        azimuth=params.get('azimuth', 315),
                        altitude=params.get('altitude', 30),
                        use_landsat=params.get('use_landsat', True),
                        log_callback=on_log,
                        aoi_name=aoi_name,
                        mineral_hint=mineral_hint,
                        tenant_id=tenant_id,
                        task_code=task_code,
                    )

                    analysis_tasks[task_id]['status'] = 'completed'
                    analysis_tasks[task_id]['progress'] = 100
                    analysis_tasks[task_id]['results'] = results
                    analysis_tasks[task_id]['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                except Exception as e:
                    import traceback
                    err = traceback.format_exc()
                    logger.error(f"构造解译图生成失败: {err}")
                    analysis_tasks[task_id]['status'] = 'failed'
                    analysis_tasks[task_id]['error'] = str(e)
                    analysis_tasks[task_id]['progress'] = 0
                    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    analysis_tasks[task_id]['logs'].append(f"[{ts}] [ERROR] [{task_code}] {str(e)}")
                finally:
                    _cleanup_old_tasks()

        t = threading.Thread(target=run_task, daemon=True)
        t.start()

        return jsonify({'success': True, 'task_id': task_id, 'task_code': task_code})

    except Exception as e:
        logger.error(f"启动失败: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/status/<task_id>')
def task_status(task_id):
    """查询任务状态"""
    if task_id not in analysis_tasks:
        return jsonify({'success': False, 'message': '任务不存在'})

    task = analysis_tasks[task_id]
    if task['status'] == 'running' and task['progress'] < 90:
        task['progress'] = min(task['progress'] + 8, 90)

    safe = {}
    for k, v in task.items():
        if k == 'results' and isinstance(v, dict):
            safe[k] = v
        else:
            safe[k] = v

    return jsonify({'success': True, 'task': safe})


# ---------------------------------------------------------------------------
# 历史分析记录
# ---------------------------------------------------------------------------
@app.route('/api/history')
def api_history():
    """扫描 results/ 目录,返回所有已完成的构造解译/InSAR 融合分析记录。"""
    results_root = Path(Config.RESULTS_FOLDER)
    if not results_root.exists():
        return jsonify({'success': True, 'records': []})

    records = []
    for aoi_dir in sorted(results_root.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True):
        if not aoi_dir.is_dir() or aoi_dir.name.startswith('_'):
            continue
        # 查找所有 run (structural 和 insar_fusion)
        for category in ('structural', 'insar_fusion'):
            cat_dir = aoi_dir / category
            if not cat_dir.is_dir():
                continue
            # 版本化布局: run 子目录
            for run_dir in sorted(cat_dir.iterdir(), key=lambda d: d.name, reverse=True):
                md_path = run_dir / 'metadata.json'
                if not md_path.exists():
                    continue
                try:
                    with open(md_path, 'r', encoding='utf-8') as f:
                        md = json.load(f)
                    records.append({
                        'aoi_name': md.get('aoi_name') or aoi_dir.name,
                        'source': md.get('source', ''),
                        'run_id': md.get('run_id') or run_dir.name,
                        'task_code': md.get('task_code', ''),
                        'category': category,
                        'created_at': md.get('created_at', ''),
                        'aoi_bbox': md.get('aoi_bbox'),
                        'result_dir': str(run_dir),
                        'metadata_path': str(md_path),
                        'products': md.get('products', {}),
                        'structural_stats': md.get('structural_stats') or md.get('fusion_stats', {}).get('topographic_dominant_strikes_deg'),
                        'deposit_inference': md.get('deposit_inference'),
                        'n_products': len(md.get('products', {})),
                    })
                except Exception:
                    continue
    return jsonify({'success': True, 'records': records[:50]})  # 最多返回50条


@app.route('/api/history_result')
def api_history_result():
    """从磁盘加载历史分析结果(供前端历史面板回顾)。"""
    metadata_path = request.args.get('metadata_path', '')
    result_dir = request.args.get('result_dir', '')

    if not metadata_path or not os.path.exists(metadata_path):
        return jsonify({'success': False, 'message': 'metadata 不存在'}), 404
    # 路径安全检查
    if '..' in metadata_path or not os.path.abspath(metadata_path).startswith(
            os.path.abspath(Config.RESULTS_FOLDER)):
        return jsonify({'success': False, 'message': '非法路径'}), 400

    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            md = json.load(f)
    except Exception as e:
        return jsonify({'success': False, 'message': f'读取 metadata 失败: {e}'}), 500

    products = md.get('products', {})
    # 构造与 showResults 兼容的结果对象
    results = {
        'result_dir': result_dir,
        'output_files': {
            'hillshade': products.get('map_hillshade_png', ''),
            'aspect': products.get('map_aspect_png', ''),
            'terrain': products.get('map_terrain_png', ''),
        },
        'products': products,
        'structural_stats': md.get('structural_stats', {}),
        'deposit_inference': md.get('deposit_inference'),
        'aoi_name': md.get('aoi_name', ''),
        'aoi_bbox': md.get('aoi_bbox', []),
        'elevation_range': md.get('structural_stats', {}).get('elevation_range_m', []),
    }
    # 注册到内存以便图片加载
    global task_counter
    hist_id = f"hist_{task_counter:04d}"
    task_counter += 1
    analysis_tasks[hist_id] = {
        'id': hist_id, 'status': 'completed', 'progress': 100,
        'results': results, 'logs': [],
    }
    results['task_id'] = hist_id
    return jsonify({'success': True, 'results': results})


@app.route('/api/result/<task_id>/<filename>')
def result_file(task_id, filename):
    """获取生成的图片文件"""
    # 路径穿越防护:禁止斜杠和 ..
    if '/' in filename or '\\' in filename or '..' in filename:
        return jsonify({'success': False, 'message': '非法文件名'}), 400

    if task_id not in analysis_tasks:
        return jsonify({'success': False, 'message': '任务不存在'}), 404

    task = analysis_tasks[task_id]
    if task['status'] != 'completed' or not task.get('results'):
        return jsonify({'success': False, 'message': '任务未完成'}), 400

    result_dir = task['results'].get('result_dir')
    if not result_dir:
        return jsonify({'success': False, 'message': '结果目录不存在'}), 404

    file_path = os.path.join(result_dir, filename)
    # 二次确认:解析后路径仍在 result_dir 内
    if not os.path.abspath(file_path).startswith(os.path.abspath(result_dir)):
        return jsonify({'success': False, 'message': '非法路径'}), 400
    if not os.path.exists(file_path):
        return jsonify({'success': False, 'message': '文件不存在'}), 404

    return send_file(file_path)


# 衍生栅格(GeoTIFF)无法在浏览器直接显示,这里按产品类型套色渲染成 PNG 预览。
_PREVIEW_CMAP = {
    'svf.tif': 'cividis', 'openness.tif': 'cividis', 'slope.tif': 'YlOrRd',
    'aspect.tif': 'hsv', 'hillshade_315.tif': 'gray',
    'curvature.tif': 'RdBu_r',                       # 双向:脊正谷负
    'distance_to_lineament.tif': 'viridis_r',         # 近断裂亮
    'lineament_density.tif': 'hot',
}


@app.route('/api/preview/<task_id>/<filename>')
def preview_raster(task_id, filename):
    """把结果目录下的 GeoTIFF 渲染成套色 PNG 预览(仅 .tif)。"""
    if task_id not in analysis_tasks:
        return jsonify({'success': False, 'message': '任务不存在'}), 404
    task = analysis_tasks[task_id]
    if task['status'] != 'completed' or not task.get('results'):
        return jsonify({'success': False, 'message': '任务未完成'}), 400
    result_dir = task['results'].get('result_dir')
    if not filename.lower().endswith('.tif') or '/' in filename or '\\' in filename or '..' in filename:
        return jsonify({'success': False, 'message': '仅支持结果目录内的 .tif 预览'}), 400
    file_path = os.path.join(result_dir or '', filename)
    if not result_dir:
        return jsonify({'success': False, 'message': '结果目录不存在'}), 404
    # 二次确认:解析后路径仍在 result_dir 内
    if not os.path.abspath(file_path).startswith(os.path.abspath(result_dir)):
        return jsonify({'success': False, 'message': '非法路径'}), 400
    if not os.path.exists(file_path):
        return jsonify({'success': False, 'message': '文件不存在'}), 404

    try:
        import io
        import numpy as np
        import rasterio
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        with rasterio.open(file_path) as src:
            arr = src.read(1).astype('float32')
        nod = ~np.isfinite(arr)
        valid = arr[~nod]
        cmap = _PREVIEW_CMAP.get(filename, 'viridis')
        if valid.size == 0:
            vmin, vmax = 0.0, 1.0
        elif filename == 'curvature.tif':
            m = float(np.nanpercentile(np.abs(valid), 98)) or 1.0
            vmin, vmax = -m, m                      # 对称,0=平
        else:
            vmin = float(np.nanpercentile(valid, 2))
            vmax = float(np.nanpercentile(valid, 98))
            if vmax - vmin < 1e-9:
                vmax = vmin + 1e-9
        disp = np.ma.masked_array(arr, mask=nod)
        fig, ax = plt.subplots(figsize=(6, 6 * arr.shape[0] / max(1, arr.shape[1])))
        ax.imshow(disp, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axis('off')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=120,
                    facecolor='white')
        plt.close(fig)
        buf.seek(0)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        logger.error(f"预览渲染失败 {filename}: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500


# ---------------------------------------------------------------------------
# InSAR × 构造解译 融合端点
# ---------------------------------------------------------------------------
@app.route('/api/insar_fusion', methods=['POST'])
def api_insar_fusion():
    """
    触发 InSAR 形变 × 构造解译融合。

    接收 geo-insar AOI 目录路径 + 可选的 geo-stru structural 目录路径,
    在后台线程中调用 core.insar_fusion.run_fusion(), 产物落盘到
    results/<AOI>/insar_fusion/<run_id>/。

    自动检测数据格式(MintPy h5 / geo-insar TIF+npy), 如有 2D 分解结果
    则用垂直速率做活动性打标 + 沉降探测, 并提取东西向形变线性体。
    """
    params = request.json or {}
    insar_dir = params.get('insar_dir')
    structural_dir = params.get('structural_dir')
    aoi_name = params.get('aoi_name')
    seed = params.get('seed', 42)

    if not insar_dir or not os.path.isdir(insar_dir):
        return jsonify({'success': False,
                        'message': 'insar_dir 不存在,请提供 geo-insar AOI 目录路径'}), 400

    if structural_dir and not os.path.isdir(structural_dir):
        return jsonify({'success': False,
                        'message': 'structural_dir 不存在'}), 400

    # 全局唯一任务编码:写入 metadata / 嵌入目录名 / 绑定日志 / 返回前端,凭此回溯。
    task_code = new_task_code()

    # 输出目录
    safe_name = re.sub(r'[\\/:*?"<>|]+', '_', aoi_name or Path(insar_dir).name).strip()
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + task_code
    output_dir = os.path.join(Config.RESULTS_FOLDER, safe_name, 'insar_fusion', run_id)

    task_id = f"insar_{task_counter:04d}"
    task_counter += 1
    analysis_tasks[task_id] = {
        'id': task_id, 'task_code': task_code, 'aoi_name': aoi_name or safe_name,
        'status': 'running', 'progress': 0,
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'logs': [], 'results': None,
    }

    def run_fusion_task():
        with logger.contextualize(task_code=task_code):
            try:
                from core.insar_fusion import run_fusion
                md = run_fusion(
                    insar_dir=insar_dir, out_dir=output_dir,
                    aoi_name=aoi_name, seed=seed,
                    structural_dir=structural_dir, make_plots=True,
                    task_code=task_code,
                )
                analysis_tasks[task_id]['status'] = 'completed'
                analysis_tasks[task_id]['progress'] = 100
                analysis_tasks[task_id]['results'] = {
                    'result_dir': output_dir,
                    'metadata': md,
                    'output_files': md.get('products', {}),
                }
                analysis_tasks[task_id]['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            except Exception as e:
                import traceback
                err = traceback.format_exc()
                logger.error(f"InSAR 融合失败: {err}")
                analysis_tasks[task_id]['status'] = 'failed'
                analysis_tasks[task_id]['error'] = str(e)
            finally:
                _cleanup_old_tasks()

    t = threading.Thread(target=run_fusion_task, daemon=True)
    t.start()

    return jsonify({
        'success': True, 'task_id': task_id, 'task_code': task_code,
        'output_dir': output_dir,
    })


# ---------------------------------------------------------------------------
# 矿床类型构造推理端点
# ---------------------------------------------------------------------------
@app.route('/api/deposit_inference', methods=['POST'])
def api_deposit_inference():
    """
    基于 geo-stru 已有分析结果的构造特征,推理 ROI 可能的矿床类型。

    接受 task_id (构造分析任务) 或 metadata_path (metadata.json 路径),
    读取 structural_stats / 地形统计, 调用推理引擎返回候选矿床类型。

    纯构造特征推理,不依赖蚀变/地球化学/已知矿点等外部数据。
    """
    params = request.json or {}
    task_id = params.get('task_id')
    metadata_path = params.get('metadata_path')

    # 从 task_id 定位 metadata
    md = None
    if task_id and task_id in analysis_tasks:
        task = analysis_tasks[task_id]
        if task.get('status') != 'completed':
            return jsonify({'success': False,
                            'message': f'任务 {task_id} 状态为 {task.get("status")}, 尚未完成'}), 400
        result_dir = task.get('results', {}).get('result_dir')
        if result_dir:
            mp = os.path.join(result_dir, 'metadata.json')
            if os.path.exists(mp):
                metadata_path = mp
    elif metadata_path and not os.path.exists(metadata_path):
        return jsonify({'success': False, 'message': f'metadata_path 不存在: {metadata_path}'}), 400

    if not metadata_path or not os.path.exists(metadata_path):
        return jsonify({'success': False,
                        'message': '请提供 task_id 或 metadata_path'}), 400

    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            md = json.load(f)
    except Exception as e:
        return jsonify({'success': False, 'message': f'读取 metadata 失败: {e}'}), 500

    # 如果 metadata 已有 deposit_inference,直接返回
    existing = md.get('deposit_inference')
    if existing and existing.get('primary_model'):
        return jsonify({'success': True, 'deposit_inference': existing,
                        'task_code': md.get('task_code', ''),
                        'source': 'cached'})

    # 否则重新推理
    structural_stats = md.get('structural_stats') or md.get('fusion_stats', {})
    if not structural_stats:
        return jsonify({'success': False,
                        'message': 'metadata 中无 structural_stats,请先运行构造分析'}), 400

    try:
        from core.deposit_inference import infer_deposit_type

        # 提取归因统计(如果有)
        attribution_stats = {}
        attr_details = md.get('fusion_stats', {}).get('attribution_details', [])
        if not attr_details:
            # 尝试从 subsidence_details 提取
            sub_details = md.get('fusion_stats', {}).get('subsidence_details', [])
            for sd in sub_details:
                ts = sd.get('ts_class', 'no_data')
                if ts in ('linear', 'accelerating'):
                    attribution_stats['goaf'] = attribution_stats.get('goaf', 0) + 1
        else:
            for ad in attr_details:
                cls = ad.get('attribution_class', 'undetermined')
                attribution_stats[cls] = attribution_stats.get(cls, 0) + 1

        result = infer_deposit_type(
            structural_stats=structural_stats,
            attribution_stats=attribution_stats,
            mineral_hint=params.get('mineral_hint'),
        )
        return jsonify({'success': True, 'deposit_inference': result,
                        'task_code': md.get('task_code', ''), 'source': 'computed'})
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'message': f'推理失败: {traceback.format_exc()}'}), 500
