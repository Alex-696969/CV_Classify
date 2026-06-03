"""
Dsc:启动服务
"""

if __name__ == '__main__':
    from server import start
    model_path = r'D:\MySoftWare\Pycharm_20250301\PythonProject\CV_Project\projects\cv_classify\src\output\0.95000.onnx'

    start(
        host='http://127.0.0.1',
        port=5000,
        model_file=model_path,
        model_type='onnx'
    )