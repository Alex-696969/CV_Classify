"""

Desc：模型训练
_train_epoch
_val_epoch

"""
import os.path
from typing import Union, Literal

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torch.optim import SGD

from datas.datas import Get_dataloaer
from net.common import build_network
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
import onnx


class Trainer:
    def __init__(self,
                 class_num:int = 17,
                 dataset_path:str = None,
                 epoch:int = 100,
                 batch_size:int = 16,
                 freeze:Union[bool, int] = False,
                 split_train_test_dir = True # 为true时会把all文件夹分成train,val两个文件夹
                 ):

        self.train_dataloader, self.test_dataloader, self.class_name = Get_dataloaer(
            dataset_path=dataset_path,
            split_train_test_dir=split_train_test_dir,
            batch_size=batch_size
        )()
        self.class_num = class_num
        self.module = build_network('vgg', num_classes=self.class_num, freeze=False)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.loss_fn = self._build_loss()
        self.loss = []
        self.opt = self._opt_lr()

        self.epoch = epoch
        self.best_acc = 0
        self._epoch = 0
        self.train_batch = 1
        self.train_batch_num = len(self.train_dataloader)

        self.ouput_dir = os.path.join(os.path.dirname(__file__), 'output')


    def _build_loss(self):
        return nn.CrossEntropyLoss()
    def _opt_lr(self):
        return SGD(params=self.module.parameters(), lr=0.001)

    def _train_epoch(self):
        self.module.to(device=self.device)
        self.module.train()
        k = 0
        for x_batch_train, y_batch_train in self.train_dataloader:
            x_batch_train = x_batch_train.to(device=self.device)
            y_batch_train = y_batch_train.to(device=self.device)
            if k == 10:
                break
            else:
                k += 1


            # 前向执行算损失
            score_batch = self.module(x_batch_train)
            loss_batch  = self.loss_fn(score_batch, y_batch_train)
            self.loss.append(loss_batch.item())
            print(f'第{self._epoch + 1}/{self.epoch}轮，第{self.train_batch}/{self.train_batch_num}个批次，loss:{loss_batch:.5f}')

            # 反向
            # 重置梯度
            self.opt.zero_grad()
            # 损失回传
            loss_batch.backward()
            # 梯度更新
            self.opt.step()
            # 记录训练批次
            self.train_batch += 1


    def _metric(self):
        with torch.no_grad():
            self.module.eval()
            y_true_all, y_pre_all = [], []
            for x_batch_test, y_batch_test in self.test_dataloader:
                x_batch_test = x_batch_test.to(device=self.device)
                # 前向算预测值
                score_test = self.module(x_batch_test)
                y_true_all.extend(y_batch_test.numpy())
                y_pre_all.extend(score_test.cpu().numpy())
            y_pre_all = np.argmax(y_pre_all, axis=1)
            _acc = accuracy_score(y_true_all, y_pre_all)
            print(f'第{self._epoch + 1}/{self.epoch}轮，验证集准确率为：{_acc}')
            return _acc


    def _save(self, _acc):
        os.makedirs(self.ouput_dir, exist_ok=True)

        if _acc > self.best_acc:
            # 删除上一个pkl
            last_model_path = os.path.join(self.ouput_dir, f'{self.best_acc:.5f}.pkl')
            if os.path.exists(last_model_path):
                os.remove(last_model_path)

            self.best_acc = _acc
            ckpt = {
                'state_dict': self.module.state_dict(),
                'best_acc': self.best_acc,
                'class_name': self.class_name,
                '_epoch': self._epoch,
            }

            torch.save(ckpt, os.path.join(self.ouput_dir, f'{self.best_acc:.5f}.pkl'))
            print(f'检查点保存成功，路径：'+ os.path.join(self.ouput_dir, f'{self.best_acc}.pkl'))

    def fit(self):
        for _epoch in range(self.epoch):
            self._epoch = _epoch
            self._train_epoch()
            # 每训练完一轮，算一次准确率
            _acc = self._metric()
            # 保存当前训练轮数

            # 每训练完一轮保存一个检查点
            self._save(_acc=_acc)

    # 从检查点导出模型文件
    def export(self, ckpt_path:str, format: Literal['torchscript', 'onnx'] = 'torchscript'):
        # 导入检查点模型
        # 方式一
        ckpt = torch.load(
            f=ckpt_path,
            map_location='cpu',
            weights_only=False
        )
        print(ckpt.keys())
        best_acc = ckpt['best_acc']
        class_name = ckpt['class_name']
        net = build_network('vgg', num_classes=len(class_name), freeze=False)
        net.load_state_dict(state_dict=ckpt['state_dict'], strict=True)
        print(f'检查点导入成功，参数恢复成功！...')
        net.eval()
        # 判断导出格式
        if format == 'torchscript':

            ts = torch.jit.trace_module(net, inputs={'forward': torch.rand(1, 3, 224, 224)})
            f = os.path.join(os.path.dirname(ckpt_path), f'{best_acc:.5f}.pt')
            torch.jit.save(
                ts,
                f=f
            )
        elif format == 'onnx':
            f = os.path.join(os.path.dirname(ckpt_path), f'{best_acc:.5f}.onnx')
            torch.onnx.export(
                model=net,
                args=(torch.rand(1, 3, 224, 224),),
                f=f
            )
        print(f'{format}模型导出成功，path:{f}')
        print('开启模型检查...')
        onnx_model = onnx.load(f)
        onnx.checker.check_model(onnx_model)
        print('模型检查通过...')








if __name__ == '__main__':
    trainer = Trainer(
        dataset_path=r"D:\Road_AI\06 Deep Learning\CV\20260412\17flowers -2\all",
        epoch=100,
        batch_size=16,
        freeze=False,
        split_train_test_dir=True
    )
    trainer.export(
        ckpt_path=r'D:\MySoftWare\Pycharm_20250301\PythonProject\CV_Project\projects\cv_classify\src\output\0.24643.pkl',
        format='onnx'
    )

