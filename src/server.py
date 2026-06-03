"""
Desc:服务端
"""
from io import BytesIO
from flask import Flask, request, jsonify
import base64
from PIL import Image

from precdictor import Predictor
from typing import Literal

app = Flask(__name__)
predictor = None



def base64_2_pilimg(img_base64):
    img_bytes = base64.b64decode(img_base64)
    # PIL不能读内存，需要用BytesIO把内存中的数据封装成一个文件对象。
    img = Image.open(BytesIO(img_bytes))
    return img

@app.get('/')
def index():
    return '图像分类后端接口'

@app.route('/predict', methods=['POST'])
def predict():

    _args = request.form
    img_b64 = _args.get('img')
    top_k = _args.get('top_k', default=1)
    top_k = int(top_k)
    img = base64_2_pilimg(img_b64)
    print(type(img))

    # 推理
    result = predictor.predict(
        img=img,
        top_k=top_k
    )
    return jsonify({
        'code': 0,
        'data': result,
        'msg': '成功！'
    })


def start(host, port, model_file, model_type:Literal['torchscript', 'onnx']):
    global predictor
    predictor = Predictor(
        model_path=model_file,
        model_type=model_type
    )
    app.run()





if __name__ == '__main__':
    start(
        host=r'http://127.0.0.1',
        port=5000,
        model_file=r'D:\MySoftWare\Pycharm_20250301\PythonProject\CV_Project\projects\cv_classify\src\output\0.95000.onnx',
        model_type='onnx'
          )
