

import torch.nn as nn
from torchvision import models

net = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)

for name, para in net.named_parameters():
    print(type(name), type(para))
    print(para)