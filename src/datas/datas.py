"""
DESC：返回数据加载器，训练时返回训练集，测试集；推理时返回样本集。
-方法一：从数据集中拆分出train,val文件夹
-方法二：不拆分train,val文件夹，加载时分隔成train,val
"""
import os
import shutil

import PIL
import cv2
import torch
from torchvision.datasets import DatasetFolder
from torch.utils.data import dataloader, random_split
from PIL import Image
import torch.nn as nn
from torchvision.transforms import v2 as transforms
import numpy as np








# 按长边缩放，短边补0
class Letterbox:
    def __init__(self, size:int = 224):
        self.size = size

    def __call__(self, img:torch.Tensor = None):
        self.img = img
        # 按长边缩放到size
        _, h, w = self.img.shape
        max_size = max(h, w)

        ratio = 224 / max_size
        new_h = round(h * ratio)
        new_w = round(w * ratio)
        # 得到按长边缩放到size后的新图像
        self.img = transforms.Resize(size=(new_h, new_w))(self.img)

        # 计算填充
        pad_w = self.size - new_w
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_h = self.size - new_h
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        padding = [pad_left, pad_top, pad_right, pad_bottom]

        # 填充短边为0
        self.img = transforms.Pad(padding=padding, fill=0, padding_mode="constant")(self.img)

        return self.img











def load_transform(train=True):
    if train:
        my_transform = transforms.Compose([
            transforms.ToImage(),
            Letterbox(size=224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=30,fill=0),
            transforms.ToDtype(dtype=torch.float32, scale=True)
        ])
    else:
        my_transform = transforms.Compose([
            transforms.ToImage(),
            Letterbox(size=224),
            transforms.ToDtype(dtype=torch.float32, scale=True)
        ]) # shape [c, h, w]
    return my_transform



class Get_dataloaer:
    def __init__(self, dataset_path, split_train_test_dir: bool = True, batch_size:int = 16):
        self.dataset_path = dataset_path
        self.split_train_test_dir = split_train_test_dir
        self.train_dataset_path = None
        self.test_dataset_path = None
        self.batch_size = batch_size
        self.class_num = None


    def __call__(self):
        if self.split_train_test_dir:
            self.train_dataset_path, self.test_dataset_path = self._train_test_split()
            train_dataset = DatasetFolder(
                root=self.train_dataset_path,
                loader=Image.open,
                transform=load_transform(),
                extensions=('jpg',)
            )
            test_dataset = DatasetFolder(
                root=self.test_dataset_path,
                loader=Image.open,
                transform=load_transform(),
                extensions=('jpg',)
            )
            self.class_num = test_dataset.classes

        else:
            datasets = DatasetFolder(
                root=self.dataset_path,
                loader=Image.open,
                transform=load_transform(),
                extensions=('jpg',)
            )
            self.class_num = datasets.classes
            data_num = len(datasets)
            train_num = int(0.8 * data_num)
            test_num = data_num - train_num
            train_dataset, test_dataset= random_split(
                dataset=datasets,
                lengths=[train_num, test_num],
                generator=torch.Generator().manual_seed(42)
            )


        print(f'训练集数据量为：{len(train_dataset)}，测试集数据量为{len(test_dataset)}')
        train_dataloader = dataloader.DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(42)
        )
        test_dataloader = dataloader.DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(42)
        )


        return train_dataloader, test_dataloader, self.class_num


    def _train_test_split(self):



        """方法一：先将文件夹划分为train,val，再导入"""
        train_dataset_path = os.path.join(os.path.dirname(self.dataset_path), 'train')
        test_dataset_path = os.path.join(os.path.dirname(self.dataset_path), 'val')
        os.makedirs(train_dataset_path, exist_ok=True)
        os.makedirs(test_dataset_path, exist_ok=True)
        # 判断一下，如果train,val文件夹下有内容，则全部删除
        if len(os.listdir(train_dataset_path)) > 0:
            print(f'train和val文件夹中已有文件，如需重新生成，请先清空train文件夹和val文件夹，路径为：{os.path.dirname(self.dataset_path)}')
            return train_dataset_path, test_dataset_path

        # 开始按比例移动图片至训练集文件夹和验证集文件夹，这里的逻辑是先构建一个装有train和val的列表，最后将图片复制到对应的文件夹中
        val_ratio = 0.2  # 验证集占比0.2
        val_cnt, train_cnt = 0, 0
        for class_name in os.listdir(self.dataset_path):
            for img_name in os.listdir(os.path.join(self.dataset_path, class_name)):
                img_path_src = os.path.join(self.dataset_path, class_name, img_name)
                if torch.rand(1) < val_ratio:
                    img_path_dst = os.path.join(test_dataset_path, class_name, img_name)
                    val_cnt += 1
                else:
                    img_path_dst = os.path.join(train_dataset_path, class_name, img_name)
                    train_cnt += 1
                os.makedirs(os.path.dirname(img_path_dst), exist_ok=True)
                shutil.copy(img_path_src, img_path_dst)

        print(
            f'数据集划分完成：\n训练集有图片{train_cnt}张，路径为：{train_dataset_path}\n测试集有图片{val_cnt}张，路径为：{test_dataset_path}')
        return train_dataset_path, test_dataset_path



# 测试图像转换是否成功
def interface01():
    img = Image.open(r"D:\Road_AI\06 Deep Learning\CV\20260412\17flowers -2\train\c1\image_0001.jpg")
    my_transform = load_transform()
    img = my_transform(img)
    img_numpy = img.numpy()
    print(img_numpy.shape)
    img_numpy = np.transpose(img_numpy, axes=(1, 2, 0)) # c,h,w->h,w,c
    bgr_img_numpy = img_numpy[:, :, ::-1]
    cv2.imshow('1', bgr_img_numpy)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()




if __name__ == '__main__':
    train_loader, test_loader, class_num = Get_dataloaer(
        dataset_path=r"D:\Road_AI\06 Deep Learning\CV\20260412\17flowers -2\all",
        split_train_test_dir=False)()













