"""
Desc：预测器，可选用pt或onnxruntime进行推理
- init方法：模型初始化
- predict方法：预测，以json格式返回
"""
import json
import os.path
from typing import Literal, Union

import torch.jit
from PIL import Image
from datas.datas import load_transform
import onnxruntime


class Predictor:
    def __init__(self, model_path:str, model_type:Literal['torchscript', 'onnx']='torchscript'):
        self.model_path = model_path
        self.model_type = model_type
        if self.model_type == "torchscript":
            extra_files = {'class_name':'', 'best_acc': ''}
            self.model = torch.jit.load(f=self.model_path, map_location='cpu', _extra_files=extra_files)
            self.class_name = json.loads(extra_files['class_name'])
            self.best_acc = json.loads(extra_files['best_acc'])
            extra_files['class_name'] = self.class_name
            extra_files['best_acc'] = self.best_acc
            print(f'torchscript模型导入成功，附带参数{extra_files}')
        elif self.model_type == 'onnx':
            json_path = os.path.splitext(self.model_path)[0] + '.json'
            if os.path.exists(json_path):
                with open(json_path, mode='r', encoding='utf-8') as file:
                    extra_files = json.load(file)
                self.class_name = extra_files['class_name']
                self.best_acc = extra_files['best_acc']
            else:
                print(f'json文件的路径不存在，请检查路径：{json_path}')
            self.session = onnxruntime.InferenceSession(path_or_bytes=self.model_path)




    def predict(self, img:Union[str, Image.Image], top_k:int = 5):
        my_transform = load_transform(train=False)  # shape[c, h, w]
        if self.model_type == 'torchscript':
            # img为路径，在本地测试用
            if isinstance(img, str):
                # 如果传的是str，则需要用Pil读取，并用和训练时相同的预处理方式
                img = Image.open(fp=img)
            img = my_transform(img)
            img = torch.unsqueeze(img, dim=0)
            prob = self.model(img)
        elif self.model_type == 'onnx':
            if isinstance(img, str):
            # 如果传的是str，则需要用Pil读取，并用和训练时相同的预处理方式
                img = Image.open(fp=img)
                img = my_transform(img)
                img = torch.unsqueeze(img, dim=0)
                img = img.numpy()
                prob = self.session.run(None, input_feed={self.session.get_inputs()[0].name: img})[0]
                # print(self.session.get_inputs()[0])
                # print(self.session.get_outputs()[0])
                prob = torch.asarray(prob)

        # 对输出进行后处理，返回格式为[{'类别'：概率}...]
        values, indices = torch.topk(prob, k=top_k, dim=-1)
        values_list = values[0].tolist()
        indices_list = indices[0].tolist()
        result = []
        for i in range(top_k):
            dic = {}
            index = indices_list[i]
            _class_name = self.class_name[index]
            _prob = round(values_list[i], 5)
            dic[_class_name] = _prob
            result.append(dic)
        return result



if __name__ == '__main__':
    predictor = Predictor(
        r'D:\MySoftWare\Pycharm_20250301\PythonProject\CV_Project\projects\cv_classify\src\output\0.95000.onnx',
        'onnx'
    )
    result = predictor.predict(img=r"D:\Road_AI\06 Deep Learning\CV\20260412\17flowers -2\all\c1\image_0001.jpg", top_k=3)
    print(result)
