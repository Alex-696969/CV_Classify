from pyexpat import features
from torchvision import models

net = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)

for name, param in net.named_parameters():
    print(name)
    if name.startswith("features"):
        param.requires_grad = False


for param in net.parameters():
    print(param.requires_grad)


print(net)
