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
from config.config import Config
from core.structural_engine import StructuralEngine
from core import delivery
from utils.file_utils import get_file_size
from utils.logger import get_logger

app = Flask(__name__)
app.secret_key = Config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH

logger = get_logger(__name__, Config.LOG_FILE)

task_counter = 0
analysis_tasks = {}

STRUCTURAL_EXTENSIONS = {'kml', 'kmz', 'ovkml', 'xlsx', 'xls', 'csv'}


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

        # 按 ROI 原始文件名定位交付项目,探测冬季数据(DEM 必需 / Landsat 可选),
        # 替代上传卫星数据 ZIP 的方式。
        resolved = {}
        project_dir = delivery.resolve_project_dir(file.filename)
        if project_dir:
            wd = delivery.locate_winter_data(project_dir)
            resolved = {
                'project_name': os.path.basename(str(project_dir)),
                'dem_available': bool(wd.get('dem')),
                'landsat_available': bool(wd.get('landsat_dir')),
                'landsat_sensor': wd.get('landsat_sensor'),
            }

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

        task_id = f"struct_{task_counter:04d}"
        task_counter += 1

        # 用交付项目名作为 AOI 名(规范、与平台一致、可被下游 broker 发现),
        # 而非上传时改写的 area_<ts> 文件名。每次分析存为独立 run 子目录,不覆盖历史。
        aoi_name = os.path.basename(str(project_dir))
        safe_aoi = re.sub(r'[\\/:*?"<>|]+', '_', aoi_name).strip() or task_id
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + task_id
        output_dir = os.path.join(Config.RESULTS_FOLDER, safe_aoi, 'structural', run_id)
        analysis_tasks[task_id] = {
            'id': task_id,
            'aoi_name': aoi_name,
            'status': 'running',
            'progress': 0,
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'logs': [],
            'results': None,
        }

        def run_task():
            try:
                coords = StructuralEngine.parse_polygon(file_path)

                def on_log(msg, level='INFO'):
                    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    analysis_tasks[task_id]['logs'].append(f"[{ts}] [{level}] {msg}")
                    if len(analysis_tasks[task_id]['logs']) > 100:
                        analysis_tasks[task_id]['logs'] = analysis_tasks[task_id]['logs'][-100:]

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
                analysis_tasks[task_id]['logs'].append(f"[{ts}] [ERROR] {str(e)}")

        t = threading.Thread(target=run_task, daemon=True)
        t.start()

        return jsonify({'success': True, 'task_id': task_id})

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


@app.route('/api/result/<task_id>/<filename>')
def result_file(task_id, filename):
    """获取生成的图片文件"""
    if task_id not in analysis_tasks:
        return jsonify({'success': False, 'message': '任务不存在'}), 404

    task = analysis_tasks[task_id]
    if task['status'] != 'completed' or not task.get('results'):
        return jsonify({'success': False, 'message': '任务未完成'}), 400

    result_dir = task['results'].get('result_dir')
    if not result_dir:
        return jsonify({'success': False, 'message': '结果目录不存在'}), 404

    file_path = os.path.join(result_dir, filename)
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
    if not filename.lower().endswith('.tif') or '/' in filename or '..' in filename:
        return jsonify({'success': False, 'message': '仅支持结果目录内的 .tif 预览'}), 400
    file_path = os.path.join(result_dir or '', filename)
    if not result_dir or not os.path.exists(file_path):
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

    # 输出目录
    safe_name = re.sub(r'[\\/:*?"<>|]+', '_', aoi_name or Path(insar_dir).name).strip()
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_insar'
    output_dir = os.path.join(Config.RESULTS_FOLDER, safe_name, 'insar_fusion', run_id)

    task_id = f"insar_{task_counter:04d}"
    task_counter += 1
    analysis_tasks[task_id] = {
        'id': task_id, 'aoi_name': aoi_name or safe_name,
        'status': 'running', 'progress': 0,
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'logs': [], 'results': None,
    }

    def run_fusion_task():
        try:
            from core.insar_fusion import run_fusion
            md = run_fusion(
                insar_dir=insar_dir, out_dir=output_dir,
                aoi_name=aoi_name, seed=seed,
                structural_dir=structural_dir, make_plots=True,
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

    t = threading.Thread(target=run_fusion_task, daemon=True)
    t.start()

    return jsonify({
        'success': True, 'task_id': task_id,
        'output_dir': output_dir,
    })
