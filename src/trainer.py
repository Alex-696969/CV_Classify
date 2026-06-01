"""

Desc：模型训练
_train_epoch
_val_epoch

"""
import os.path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torch.optim import SGD

from datas.datas import Get_dataloaer
from net.common import build_network
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score


class Trainer:
    def __init__(self,
                 class_num:int = 17,
                 dataset_path:str = None,
                 epoch:int = 100,
                 batch_size:int = 16,
                 freeze:Union[bool, int] = False,
                 split_train_test_dir = True # 为true时会把all文件夹分成train,val两个文件夹
                 ):

        self.train_dataloader, self.test_dataloader, self.class_num = Get_dataloaer(
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
            print(f'损失为{loss_batch:.5f}')
            # 反向
            # 重置梯度
            self.opt.zero_grad()
            # 损失回传
            loss_batch.backward()
            # 梯度更新
            self.opt.step()


    def _metric(self, _epoch:int, ):
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
            print(f'第{_epoch + 1}/{self.epoch}轮，验证集准确率为：{_acc}')
            return _acc
    def fit(self):
        for _epoch in range(self.epoch):
            self._train_epoch()
            # 每训练完一轮，算一次准确率
            _acc = self._metric(_epoch=_epoch)
            # 每训练完一轮保存一个检查点
            self._save(_acc=_acc)

    def _save(self, _acc):
        dir_name = os.path.dirname(__file__)
        ouput_dir = os.path.join(dir_name, 'output')
        os.makedirs(ouput_dir, exist_ok=True)

        if _acc > self.best_acc:
            # 删除上一个pkl
            last_model_path = os.path.join(ouput_dir, f'{self.best_acc:.5f}.pkl')
            if os.path.exists(last_model_path):
                os.remove(last_model_path)

            self.best_acc = _acc
            torch.save(self.module, os.path.join(ouput_dir, f'{self.best_acc:.5f}.pkl'))
            print(f'检查点保存成功，路径：'+ os.path.join(ouput_dir, f'{self.best_acc}.pkl'))





if __name__ == '__main__':
    trainer = Trainer(
        dataset_path=r"D:\Road_AI\06 Deep Learning\CV\20260412\17flowers -2\all",
        epoch=100,
        batch_size=16,
        freeze=False,
        split_train_test_dir=True
    )
    trainer.fit()