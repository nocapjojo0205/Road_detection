import torch
import torch.nn as nn
from torchvision import models
from torch.hub import load_state_dict_from_url


class ResNet18(nn.Module):
    def __init__(self, num_classes=6):  # 这里的 num_classes 默认为6，根据你的任务进行修改
        super(ResNet18, self).__init__()

        self.backbone = models.resnet18(pretrained=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAMBlock(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CBAMBlock, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class ResNet34_CBAM(nn.Module):
    def __init__(self, num_classes=6):
        super(ResNet34_CBAM, self).__init__()
        self.resnet34 = models.resnet34(pretrained=False)
        self.resnet34.load_state_dict(load_state_dict_from_url(
            'https://download.pytorch.org/models/resnet34-333f7ec4.pth', progress=True))
        self.resnet34.layer1[0].conv2 = nn.Sequential(
            self.resnet34.layer1[0].conv2,
            CBAMBlock(64)
        )
        self.resnet34.layer2[0].conv2 = nn.Sequential(
            self.resnet34.layer2[0].conv2,
            CBAMBlock(128)
        )
        self.resnet34.layer3[0].conv2 = nn.Sequential(
            self.resnet34.layer3[0].conv2,
            CBAMBlock(256)
        )
        self.resnet34.layer4[0].conv2 = nn.Sequential(
            self.resnet34.layer4[0].conv2,
            CBAMBlock(512)
        )
        self.resnet34.fc = nn.Linear(self.resnet34.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet34(x)


def MobileNetv2(num_classes=6, pretrained=False):
    # 加载 MobileNetV2 模型
    model = models.mobilenet_v2(weights='IMAGENET1K_V2' if pretrained else None)

    # 修改最后的全连接层以适应新的类别数
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    return model


# 用于测试模型定义
if __name__ == "__main__":
    # 创建模型对象
    model = ResNet34_CBAM(num_classes=6)  # 如果有其他类别，可以修改 num_classes
    # 输出模型架构
    print(model)
