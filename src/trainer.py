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
from torch import optim

from datas.datas import Get_dataloaer
from net.common import build_network

from sklearn.metrics import accuracy_score
import onnx
from utils.log import Logger


__all__ = {

}




class Trainer:
    def __init__(self,
                 class_num:int = 17,
                 dataset_path:str = None,
                 epoch:int = 100,
                 batch_size:int = 16,
                 freeze:Union[bool, int] = False,
                 split_train_test_dir = True, # 为true时会把all文件夹分成train,val两个文件夹
                 lr:float = 0.01
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
        self.epoch = epoch
        self.lr = lr


        self.best_acc = 0
        self._epoch = 0
        self.train_batch = 0
        self.train_batch_num = len(self.train_dataloader)

        self.ouput_dir = os.path.join(os.path.dirname(__file__), 'output')
        self.writer = Logger(output_dir=self.ouput_dir).build_logger()
        # 记录训练集损失，验证集的准确率，学习率变化
        self.opt, self.lr_scheduler = self._opt_lr()
        self.writer.add_graph(model=self.module, input_to_model=torch.rand(1, 3, 224, 224), verbose=True)



    def _build_loss(self):
        return nn.CrossEntropyLoss()
    def _opt_lr(self):
        migrate, origin = [], []
        for para in self.module.parameters():
            # 如果该层没有冻结，且是迁移过来的参数，采用小的学习率
            if para.requires_grad and (not para.is_init_rand) :
                migrate.append(para)
            # 如果该层没有冻结，且是初始化的参数，采用大的学习率
            elif para.requires_grad and para.is_init_rand:
                origin.append(para)
        # 组内写了，用组内的，组内没写用外面的。
        opter = optim.SGD([
            {"params": migrate, "lr": self.lr * 0.1},
            {"params": origin, "lr": self.lr}
        ], lr=self.lr)
        lr_scheduler = optim.lr_scheduler.LinearLR(optimizer=opter, start_factor=1, end_factor=0.1, total_iters=self.epoch)
        self.writer.add_scalar('lr1_migrate', scalar_value=lr_scheduler.get_last_lr()[0], global_step=self._epoch)
        self.writer.add_scalar('lr2_origin', scalar_value=lr_scheduler.get_last_lr()[1], global_step=self._epoch)
        return opter, lr_scheduler

    def _train_epoch(self):
        self.module.to(device=self.device)
        self.module.train()
        self.train_batch = 1
        k = 0
        for x_batch_train, y_batch_train in self.train_dataloader:
            x_batch_train = x_batch_train.to(device=self.device)
            y_batch_train = y_batch_train.to(device=self.device)
            # 测试用
            if k == 10:
                break
            else:
                k += 1
            # 前向执行算损失
            score_batch = self.module(x_batch_train)
            loss_batch  = self.loss_fn(score_batch, y_batch_train)
            self.loss.append(loss_batch.item())
            print(f'第{self._epoch + 1}/{self.epoch}轮，第{self.train_batch}/{self.train_batch_num}个批次，loss:{loss_batch:.5f}')
            # 每个批次记录一次损失
            self.writer.add_scalar('loss_batch', scalar_value=loss_batch.item(), global_step=self.train_batch + self._epoch * self.train_batch_num)
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
            print(f'第{self._epoch}/{self.epoch}轮，验证集准确率为：{_acc}')
            self.writer.add_scalar('acc', scalar_value=_acc, global_step=self._epoch)
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
            # 保存当前训练轮数
            self._epoch = _epoch + 1
            self._train_epoch()
            # 每训练完一轮，算一次准确率
            _acc = self._metric()
            # 学习率更新

            self.lr_scheduler.step()
            self.writer.add_scalar('lr1_migrate', scalar_value=self.lr_scheduler.get_last_lr()[0], global_step=self._epoch)
            self.writer.add_scalar('lr2_origin', scalar_value=self.lr_scheduler.get_last_lr()[1], global_step=self._epoch)
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
        split_train_test_dir=True,
        lr=0.001
    )
    trainer.fit()
    #
    # trainer.export(
    #     ckpt_path=r'D:\MySoftWare\Pycharm_20250301\PythonProject\CV_Project\projects\cv_classify\src\output\0.24643.pkl',
    #     format='onnx'
    # )

