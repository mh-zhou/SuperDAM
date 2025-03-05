import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model
# from timm.models.factory import create_model
from timm.models import create_model, list_models


# 定义加载预训练权重的函数
def load_pretrained_weights(model, url, map_location="cpu"):
    try:
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location=map_location, check_hash=True)
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
            print(f"Successfully loaded pretrained weights from {url}")
        else:
            print(f"Checkpoint from {url} does not contain 'model' key.")
    except Exception as e:
        print(f"Error loading pretrained weights from {url}: {e}")

# 假设的 model_urls 字典
model_urls = {
    'super_dam_pretrained': '/home/bygpu/med/newmodel_weights.pth'  # 替换为实际的预训练权重 URL
}

# 定义 DGB 模块
class DGB(nn.Module):
    def __init__(self, in_channels, reduction=4, groups=4):
        super(DGB, self).__init__()
        self.groups = groups
        mid_channels = max(1, in_channels // reduction)

        self.weight_layer = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, groups=self.groups, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1, groups=self.groups, bias=False),
            nn.Sigmoid()
        )
        self.pointwise_groups = nn.Conv2d(in_channels, in_channels, 1, groups=self.groups, bias=False)

    def forward(self, x):
        weights = self.weight_layer(x)
        x = x * weights
        x = self.pointwise_groups(x)
        return x


# 定义全局注意力模块（GAM）
class GAM_Attention(nn.Module):
    def __init__(self, in_channels):
        super(GAM_Attention, self).__init__()

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 16, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        x_global = self.global_avgpool(x).view(b, c)
        x_channel_att = self.channel_attention(x_global).view(b, c, 1, 1)
        x = x * x_channel_att
        return x


# 定义 LinearBottleNeck_1 模块
class LinearBottleNeck_1(nn.Module):
    def __init__(self, in_c, out_c, s, t):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_c, in_c * t, 1),
            nn.BatchNorm2d(in_c * t),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_c * t, in_c * t, 3, stride=s, padding=1, groups=in_c * t),
            nn.BatchNorm2d(in_c * t),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_c * t, in_c * t, 1, stride=1, padding=0, groups=1),
            nn.BatchNorm2d(in_c * t),
            nn.Conv2d(in_c * t, out_c, 1),
            nn.BatchNorm2d(out_c)
        )

        self.stride = s
        self.in_channels = in_c
        self.out_channels = out_c
        self.attention = GAM_Attention(out_c)

    def forward(self, x):
        residual = self.residual(x)
        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x
        residual = self.attention(residual)
        return residual


# 定义 LinearBottleNeck_2 模块
class LinearBottleNeck_2(nn.Module):
    def __init__(self, in_c, out_c, s, t):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_c, in_c * t, 1),
            nn.BatchNorm2d(in_c * t),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_c * t, in_c * t, 3, stride=s, padding=1, groups=in_c * t),
            nn.BatchNorm2d(in_c * t),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_c * t, in_c * t, 1, stride=1, padding=0, groups=1),
            nn.BatchNorm2d(in_c * t),
            nn.Conv2d(in_c * t, out_c, 1),
            nn.BatchNorm2d(out_c)
        )

        self.residual_1 = nn.Sequential(
            nn.Conv2d(in_c, in_c * t, 1),
            nn.BatchNorm2d(in_c * t),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_c * t, in_c * t, 5, stride=s, padding=2, groups=in_c * t),
            nn.BatchNorm2d(in_c * t),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_c * t, in_c * t, 1, stride=1, padding=0, groups=1),
            nn.BatchNorm2d(in_c * t),
            nn.Conv2d(in_c * t, out_c, 1),
            nn.BatchNorm2d(out_c)
        )

        self.residual_2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, stride=2),
            nn.BatchNorm2d(out_c)
        )

        self.stride = s
        self.in_channels = in_c
        self.out_channels = out_c

    def forward(self, x):
        residual = self.residual(x)
        residual_1 = self.residual_1(x)
        residual_2 = self.residual_2(x)
        out_feature = residual_1 + residual + residual_2
        return out_feature


# 定义 DAMNet
class DAMNetWithDGB(nn.Module):
    def __init__(self, class_num=2,img_size=None):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        self.stage1 = nn.Sequential(
            LinearBottleNeck_1(32, 16, 1, 1),
            DGB(16)
        )
        self.stage2 = self.make_stage(2, 16, 24, 2, 6)
        self.stage3 = self.make_stage(3, 24, 32, 2, 6)
        self.stage4 = nn.Sequential(
            self.make_stage(4, 32, 64, 2, 6),
            DGB(64)
        )
        self.stage5 = self.make_stage(3, 64, 96, 1, 6)
        self.stage6 = nn.Sequential(
            self.make_stage(3, 96, 160, 2, 6),
            DGB(160)
        )
        self.stage7 = LinearBottleNeck_1(160, 320, 1, 6)

        self.conv1 = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
            DGB(1280)
        )
        self.conv2 = nn.Conv2d(1280, class_num, 1)

    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return x

    def make_stage(self, repeat, in_c, out_c, s, t):
        layers = []
        if s == 1:
            layers.append(LinearBottleNeck_1(in_c, out_c, s, t))
        else:
            layers.append(LinearBottleNeck_2(in_c, out_c, s, t))

        while repeat - 1:
            layers.append(LinearBottleNeck_1(out_c, out_c, 1, t))
            repeat -= 1

        return nn.Sequential(*layers)


# 定义 SuperDAM 函数
def SuperDAM(class_num=3, img_size=None, **kwargs):
    model = DAMNetWithDGB(class_num=class_num, img_size=img_size)
    return model

# 注册 SuperDAM 到 timm 模型注册表，并支持预训练权重加载
@register_model
def super_dam(pretrained=False, **kwargs):
    model = SuperDAM(**kwargs)
    if pretrained:
        url = model_urls.get('super_dam_pretrained')
        if url:
            load_pretrained_weights(model, url)
        else:
            print("No pretrained weights URL found for super_dam.")
    return model

# 使用 create_model 函数创建 SuperDAM 模型
if __name__ == "__main__":
    model = create_model('super_dam', pretrained=False, class_num=3)

    print(list_models())
    print(model)