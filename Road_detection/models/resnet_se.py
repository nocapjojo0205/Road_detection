import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Callable


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SEBasicBlock(nn.Module):
    expansion = 1
    """改进后的残差块，集成SE模块"""
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ):
        super(SEBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.se = SEBlock(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

# ========================
# 六类分类模型定义
# ========================
class RoadClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.backbone = models.resnet18(pretrained=False)  # 先不加载预训练权重

        # 获取原始ResNet-18的预训练权重
        pretrained_dict = models.resnet18(pretrained=True).state_dict()

        # 修改fc层适配当前任务
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

        # 获取当前模型的状态字典
        model_dict = self.backbone.state_dict()

        # 过滤掉不匹配的fc层权重
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict and model_dict[k].shape == v.shape}

        # 更新并加载权重
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict)

    def forward(self, x):
        return self.backbone(x)