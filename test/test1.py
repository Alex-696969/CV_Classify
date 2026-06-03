import os

import json



json_path = r'D:\MySoftWare\Pycharm_20250301\PythonProject\CV_Project\projects\cv_classify\src\output\0.95000.json'
with open(file=json_path, mode='r', encoding='utf-8') as file:
    data = json.load(fp=file)
print(type(data))