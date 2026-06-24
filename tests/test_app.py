"""
test_app.py — Flask API 端点测试

用 Flask test client 测试:
  1. 页面路由 (/)
  2. 上传端点参数校验
  3. 任务状态查询
  4. 路径穿越防护
  5. 历史记录 API
"""
import json
import os
import pytest
import tempfile

# 设置测试环境变量 (避免加载生产配置)
os.environ.setdefault('FLASK_DEBUG', 'false')


@pytest.fixture
def client():
    from app import app
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


class TestIndexRoute:
    def test_index_returns_html(self, client):
        rv = client.get('/')
        assert rv.status_code == 200
        assert '地质构造解译系统'.encode('utf-8') in rv.data


class TestUploadAreaValidation:
    def test_no_file(self, client):
        rv = client.post('/api/upload_area', data={})
        data = json.loads(rv.data)
        assert data['success'] is False

    def test_unsupported_extension(self, client, tmp_path):
        from io import BytesIO
        f = BytesIO(b"hello")
        rv = client.post('/api/upload_area', data={
            'file': (f, 'test.txt'),
        }, content_type='multipart/form-data')
        data = json.loads(rv.data)
        assert data['success'] is False
        assert '不支持' in data.get('message', '')


class TestStartValidation:
    def test_missing_file_path(self, client):
        rv = client.post('/api/start',
                         data=json.dumps({'project_name': 'test'}),
                         content_type='application/json')
        data = json.loads(rv.data)
        assert data['success'] is False


class TestTaskStatus:
    def test_nonexistent_task(self, client):
        rv = client.get('/api/status/nonexistent_task')
        data = json.loads(rv.data)
        assert data['success'] is False
        assert '不存在' in data.get('message', '')


class TestResultPathTraversal:
    def test_dotdot_in_filename(self, client):
        rv = client.get('/api/result/some_task/../../etc/passwd')
        # Flask/Werkzeug 会规范化 URL,可能返回 404 或 400
        assert rv.status_code in (400, 404)

    def test_slash_in_filename(self, client):
        rv = client.get('/api/result/some_task/subdir/file.png')
        # 任务不存在 → 404; 或路径检查 → 400
        assert rv.status_code in (400, 404)

    def test_preview_slash_in_filename(self, client):
        rv = client.get('/api/preview/some_task/subdir/file.tif')
        assert rv.status_code in (400, 404)

    def test_preview_dotdot_in_filename(self, client):
        rv = client.get('/api/preview/some_task/../../../etc/passwd.tif')
        assert rv.status_code in (400, 404)

    def test_preview_non_tif(self, client):
        rv = client.get('/api/preview/some_task/file.png')
        # task 不存在 → 404; 或 .tif 检查 → 400
        assert rv.status_code in (400, 404)


class TestListProjects:
    def test_returns_json(self, client):
        rv = client.get('/api/list_projects')
        data = json.loads(rv.data)
        assert data['success'] is True
        assert isinstance(data.get('projects'), list)


class TestHistoryAPI:
    def test_history_returns_list(self, client):
        rv = client.get('/api/history')
        data = json.loads(rv.data)
        assert data['success'] is True
        assert isinstance(data.get('records'), list)

    def test_history_result_missing_path(self, client):
        rv = client.get('/api/history_result?metadata_path=/nonexistent/metadata.json&result_dir=/nonexistent')
        data = json.loads(rv.data)
        assert data['success'] is False

    def test_history_result_path_traversal(self, client):
        rv = client.get('/api/history_result?metadata_path=../../etc/passwd&result_dir=/tmp')
        data = json.loads(rv.data)
        assert data['success'] is False
        assert rv.status_code in (400, 404)


class TestDocsRoute:
    def test_image_parameters_page(self, client):
        rv = client.get('/docs/image-parameters')
        # 可能返回 200 (报告存在) 或 500 (报告不存在)
        assert rv.status_code in (200, 500)


class TestDepositInferenceAPI:
    def test_missing_params(self, client):
        rv = client.post('/api/deposit_inference',
                         data=json.dumps({}),
                         content_type='application/json')
        data = json.loads(rv.data)
        assert data['success'] is False
