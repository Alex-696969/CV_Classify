
"""
Desc：网络结构
- 每个模型类中写：1.初使化方法，2.forward方法
"""
from typing import Union

import torch
import torch.nn as nn
from torchvision import models


def load_default_config():

    return [
        # ( 模型缩写， 输出通道数）
        ('CBR', 32),
        ('MP',),
        ('CBR', 64),
        ('MP',),
        ('CBR', 128),
        ('MP',),
        ('CBR', 128),
        ('CBR', 128),
        ('MP',),
        ('CBR', 256),
        ('CBR', 256),
        ('MP',),
    ]

class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))



class ClassifyNetwork_custom(nn.Module):
    def __init__(self, num_classes:int = 10, cfg=None, train=True):
        super().__init__()

        in_channels = 3
        layers_all = []
        if cfg == None:
            cfg = load_default_config()
        for layer in cfg:
            if layer[0] == 'CBR':
                out_channels = layer[1]
                layers_all.append(ConvBnRelu(in_channels=in_channels, out_channels=out_channels))
                in_channels = out_channels
            elif layer[0] == 'MP':
                layers_all.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.features = nn.Sequential(*layers_all)
        self.adaavgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Linear(in_features=in_channels, out_features=num_classes)
        self.train = train

    def forward(self, x):
        x = self.features(x)
        x = self.adaavgpool(x)
        batch = x.shape[0]
        x = torch.reshape(x, shape=(batch, -1))
        if self.train:
            return self.classifier(x)
        else:
            return torch.softmax(self.classifier(x), dim=-1)




class ClassifyNetwork_Vgg16(nn.Module):
    export_features_net: bool = False
    
    def __init__(self, num_classes, freeze:Union[int, bool] = None):
        super().__init__()
        vgg_bn = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
        # 对迁移过来的模型做一个标志位，用于模型训练的时候，给这部分用小的学习率
        for para in vgg_bn.parameters():
            para.is_init_rand = False
        # 写冻结结构：
        # 1.为bool值时，为True，全部冻结
        # 2.为数字时，表示冻结前int层
        freeze_layers = []
        if isinstance(freeze, bool):
            if freeze == True:
                freeze_layers.extend(['features', 'avgpool', 'classifier'])
        elif isinstance(freeze, int):
            # 如果freeze>44层，即features层
            if freeze > len(vgg_bn.features):
                for i in range(len(vgg_bn.features)):
                    freeze_layers.append(f'features.{i}')
                for i in range(freeze - len(vgg_bn.features) - 1):
                    freeze_layers.append(f'classifier.{i}')
            else:
                for i in range(freeze):
                    freeze_layers.append(f'features.{i}')
        if len(freeze_layers) > 0:
            for name, param in vgg_bn.named_parameters():
                if name.startswith(tuple(freeze_layers)):
                    param.requires_grad = False
                    print(f'参数{name}冻结成功...')

        # 分类的最后一层替换成我们需要的类别数
        vgg_bn.classifier[-1] = nn.Linear(in_features=4096, out_features=num_classes, bias=True)
        for para in vgg_bn.classifier[-1].parameters():
            para.is_init_rand = True
        self.vgg_bn = vgg_bn
        print(self.vgg_bn)

    def forward(self, input):
        score = self.vgg_bn(input)

        if self.vgg_bn.training:
            return score
        else:
            if self.export_features_net:
                return score
            else:
                return torch.softmax(score, dim=-1)




def build_network(model_name:str, num_classes, cfg=None, freeze:Union[int, bool]=None):
    if model_name == 'vgg':
        return ClassifyNetwork_Vgg16(num_classes=num_classes, freeze= freeze)
    else:
        return ClassifyNetwork_custom(num_classes=num_classes, cfg=cfg)


if __name__ == '__main__':
    net = build_network('vgg', num_classes=10)
    y = net(torch.rand(3, 3, 224, 224))
    print(net)
    print(y)
