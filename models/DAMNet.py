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
    'dam_pretrained': '/home/bygpu/med/newmodel_weights.pth'  # 替换为实际的预训练权重 URL
}

# 定义全局注意力模块（GAM）
class GAM_Attention(nn.Module):
    def __init__(self, in_channels):
        super(GAM_Attention, self).__init__()

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化层
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16),  # 线性变换层，减小通道数
            nn.ReLU(inplace=True),  # ReLU 激活函数
            nn.Linear(in_channels // 16, in_channels),  # 线性变换层，恢复通道数
            nn.Sigmoid()  # Sigmoid 激活函数，产生通道注意力权重
        )

    def forward(self, x):
        b, c, h, w = x.size()

        # 全局平均池化，将特征图变成全局平均值
        x_global = self.global_avgpool(x).view(b, c)

        # 通道注意力：通过线性变换和 Sigmoid 操作产生通道权重
        x_channel_att = self.channel_attention(x_global).view(b, c, 1, 1)

        # 将输入特征图按通道加权
        x = x * x_channel_att

        return x


# 定义 LinearBottleNeck_1 模块
class LinearBottleNeck_1(nn.Module):
    def __init__(self, in_c, out_c, s, t):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_c, in_c * t, 1),  # 1x1 卷积层，升维操作
            nn.BatchNorm2d(in_c * t),  # 批归一化
            nn.ReLU6(inplace=True),  # ReLU6 激活函数

            nn.Conv2d(in_c * t, in_c * t, 3, stride=s, padding=1, groups=in_c * t),  # 3x3 深度可分离卷积
            nn.BatchNorm2d(in_c * t),  # 批归一化
            nn.ReLU6(inplace=True),  # ReLU6 激活函数

            nn.Conv2d(in_c * t, in_c * t, 1, stride=1, padding=0, groups=1),  # 1x1 卷积层
            nn.BatchNorm2d(in_c * t),  # 批归一化

            nn.Conv2d(in_c * t, out_c, 1),  # 1x1 卷积层，降维操作
            nn.BatchNorm2d(out_c)  # 批归一化
        )

        self.stride = s  # 步长
        self.in_channels = in_c  # 输入通道数
        self.out_channels = out_c  # 输出通道数

        # 添加全局注意力模块
        self.attention = GAM_Attention(out_c)  # 使用定义的全局注意力模块

    def forward(self, x):
        residual = self.residual(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x  # 恒等映射，如果步长为1且通道数不变，则加上原始输入

        # 应用全局注意力
        residual = self.attention(residual)

        return residual


# 定义 LinearBottleNeck_2 模块
class LinearBottleNeck_2(nn.Module):
    def __init__(self, in_c, out_c, s, t):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_c, in_c * t, 1),  # 1x1 卷积层，升维操作
            nn.BatchNorm2d(in_c * t),  # 批归一化
            nn.ReLU6(inplace=True),  # ReLU6 激活函数

            nn.Conv2d(in_c * t, in_c * t, 3, stride=s, padding=1, groups=in_c * t),  # 3x3 深度可分离卷积
            nn.BatchNorm2d(in_c * t),  # 批归一化
            nn.ReLU6(inplace=True),  # ReLU6 激活函数

            nn.Conv2d(in_c * t, in_c * t, 1, stride=1, padding=0, groups=1),  # 1x1 卷积层
            nn.BatchNorm2d(in_c * t),  # 批归一化

            nn.Conv2d(in_c * t, out_c, 1),  # 1x1 卷积层，降维操作
            nn.BatchNorm2d(out_c)  # 批归一化
        )

        self.residual_1 = nn.Sequential(
            nn.Conv2d(in_c, in_c * t, 1),  # 1x1 卷积层，升维操作
            nn.BatchNorm2d(in_c * t),  # 批归一化
            nn.ReLU6(inplace=True),  # ReLU6 激活函数

            nn.Conv2d(in_c * t, in_c * t, 5, stride=s, padding=2, groups=in_c * t),  # 5x5 深度可分离卷积
            nn.BatchNorm2d(in_c * t),  # 批归一化
            nn.ReLU6(inplace=True),  # ReLU6 激活函数

            nn.Conv2d(in_c * t, in_c * t, 1, stride=1, padding=0, groups=1),  # 1x1 卷积层
            nn.BatchNorm2d(in_c * t),  # 批归一化

            nn.Conv2d(in_c * t, out_c, 1),  # 1x1 卷积层，降维操作
            nn.BatchNorm2d(out_c)  # 批归一化
        )

        self.residual_2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, stride=2),  # 1x1 卷积层，步长为2，降维操作
            nn.BatchNorm2d(out_c)  # 批归一化
        )

        self.stride = s  # 步长
        self.in_channels = in_c  # 输入通道数
        self.out_channels = out_c  # 输出通道数

    def forward(self, x):
        residual = self.residual(x)
        residual_1 = self.residual_1(x)
        residual_2 = self.residual_2(x)

        # 多尺度特征融合
        out_feature = residual_1 + residual + residual_2

        return out_feature

# 定义 DAMNet

class DAMNet(nn.Module):
    def __init__(self, class_num=2,img_size=None):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 第一层卷积
            nn.BatchNorm2d(32),  # 批归一化
            nn.ReLU6(inplace=True),  # ReLU6 激活函数
        )

        self.stage1 = LinearBottleNeck_1(32, 16, 1, 1)  # 第一个模块
        self.stage2 = self.make_stage(2, 16, 24, 2, 6)  # 第二个模块
        self.stage3 = self.make_stage(3, 24, 32, 2, 6)  # 第三个模块
        self.stage4 = self.make_stage(4, 32, 64, 2, 6)  # 第四个模块
        self.stage5 = self.make_stage(3, 64, 96, 1, 6)  # 第五个模块
        self.stage6 = self.make_stage(3, 96, 160, 2, 6)  # 第六个模块
        self.stage7 = LinearBottleNeck_1(160, 320, 1, 6)  # 第七个模块

        self.conv1 = nn.Sequential(
            nn.Conv2d(320, 1280, 1),  # 1x1 卷积层
            nn.BatchNorm2d(1280),  # 批归一化
            nn.ReLU6(inplace=True)  # ReLU6 激活函数
        )

        self.conv2 = nn.Conv2d(1280, class_num, 1)  # 输出分类结果

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
def damnet(class_num=3, img_size=None, **kwargs):
    model = DAMNet(class_num=class_num, img_size=img_size)
    return model

# 注册 SuperDAM 到 timm 模型注册表，并支持预训练权重加载
@register_model
def dam(pretrained=False, **kwargs):
    model = damnet(**kwargs)
    if pretrained:
        url = model_urls.get('dam_pretrained')
        if url:
            load_pretrained_weights(model, url)
        else:
            print("No pretrained weights URL found for dam.")
    return model

# 使用 create_model 函数创建 SuperDAM 模型
if __name__ == "__main__":
    model = create_model('dam', pretrained=False, class_num=3)

    print(list_models())
    print(model)