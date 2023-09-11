import torch
import torch.nn as nn

BN_MOMENTUM = 0.1

class ResNetBlock(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        self.block = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=3, stirde=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernal_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
        )
        self.downsample = downsample
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        out = self.activation(out)
        return out
    

class Encoder(nn.Modele):
    def __init__(self, in_planes=64):
        super(Encoder, self).__init__()
        self.in_planes = in_planes 
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.in_planes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # building a ResNet-18 here
        self.layer1 = self._make_layer(planes=64, n_blocks=2) # no downsample layer in the first block
        self.layer2 = self._make_layer(planes=128, n_blocks=2, stride=2)
        self.layer3 = self._make_layer(planes=256, n_blocks=2, stride=2)
        self.layer4 = self._make_layer(planes=512, n_blocks=2, stride=2)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


    def _make_layer(self, planes, n_blocks=1, stride=1):
        assert n_blocks > 0, "n_blocks must be greater than 0" 
        downsample = None
        if stride != 1 or self.in_planes != planes * ResNetBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * ResNetBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * ResNetBlock.expansion, momentum=BN_MOMENTUM),
            )

        layers = [ResNetBlock(self.in_planes, planes, stride, downsample)]
        for _ in range(n_blocks-1):
            layers.append(ResNetBlock(self.in_planes, planes))
        
        return nn.Sequential(*layers)


class FCNBottleNeck(nn.Module):
    def __init__(self, d, planes=512):
        # an fully convolutional hour glass network converging to a 1x1xd feature map and back.
        # d is the design parameter here 
        super(FCNBottleNeck, self).__init__()
        self.d = d
        self.planes = planes
        self.block = nn.Sequential(
            # nn.Conv2d(planes, 512, kernel_size=3, stride=1, padding=1, bias=False),
        )

        pass
    
    def forward(self, x):
        pass


class Decoder(nn.Module):
    def __init__(self):
        self.layer4 = self._make_deconv_layer(512)
        self.layer3 = self._make_deconv_layer(256)
        self.layer2 = self._make_deconv_layer(128)
        self.layer1 = self._make_deconv_layer(64)
        self.layer0 = self._make_deconv_layer(3)

    def _make_deconv_layer(self, planes):

