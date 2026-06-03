"""
Desc:模拟客户端测试
"""

import requests
import base64

from PIL import Image


def img2base64(img_path):
    with open(img_path, mode='rb') as file:
        img_b = file.read()
        img_base64 = base64.b64encode(img_b)
        return img_base64



if __name__ == '__main__':
    url = r'http://127.0.0.1:5000/predict'  # 需要换成自己的flask地址
    img_path = r'D:\Road_AI\06 Deep Learning\CV\20260412\17flowers -2\all\c1\image_0001.jpg'
    img_base64 = img2base64(img_path)
    response = requests.post(
        url=url,
        data={
            'img': img_base64,
            'top_k': 5
        }
    )
    if response.status_code == 200:
        print('访问服务器成功！')
        print(f'{response.json()}')
    else:
        print(f'访问服务器失败{response.status_code}')

