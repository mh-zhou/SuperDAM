import torch
import torch.nn as nn
import torchvision.models as models
from timm.models.registry import register_model
from timm.models import create_model, list_models

# 定义加载预训练权重的函数
def load_pretrained_weights(model, url, map_location="cpu"):
    try:
        checkpoint = torch.load(url, map_location=map_location)
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
            print(f"Successfully loaded pretrained weights from {url}")
        else:
            print(f"Checkpoint from {url} does not contain 'model' key.")
    except Exception as e:
        print(f"Error loading pretrained weights from {url}: {e}")

# 假设的 model_urls 字典
model_urls = {
    'dva_pretrained': '/home/bygpu/med/newmodel_weights.pth'  # 替换为实际的预训练权重 URL
}


class att(nn.Module):
    def __init__(self, input_channel):
        "the soft attention module"
        super(att, self).__init__()
        self.channel_in = input_channel

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channel,
                out_channels=512,
                kernel_size=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=256,
                kernel_size=1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=64,
                kernel_size=1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=1,
                kernel_size=1),
            nn.Softmax(dim=2)
        )

    def forward(self, x):
        mask = x
        mask = self.conv1(mask)
        mask = self.conv2(mask)
        mask = self.conv3(mask)
        att = self.conv4(mask)
        output = torch.mul(x, att)
        return output


class CNN(nn.Module):
    def __init__(self,
                 num_classes=2,
                 feature='Vgg11',
                 feature_shape=(512, 7, 7),
                 pretrained=True,
                 requires_grad=True,
                 img_size=None):
        super(CNN, self).__init__()

        # Feature Extraction
        if feature == 'Alex':
            self.ft_ext = models.alexnet(weights=models.AlexNet_Weights.DEFAULT if pretrained else None)
            self.ft_ext_modules = list(list(self.ft_ext.children())[:-2][0][:9])

        elif feature == 'Res34':
            self.ft_ext = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
            self.ft_ext_modules = list(self.ft_ext.children())[0:3] + list(self.ft_ext.children())[4:-2]

        elif feature == 'Res18':
            self.ft_ext = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            self.ft_ext_modules = list(self.ft_ext.children())[0:3] + list(self.ft_ext.children())[4:-2]

        elif feature == 'Vgg16':
            self.ft_ext = models.vgg16(weights=models.VGG16_Weights.DEFAULT if pretrained else None)
            self.ft_ext_modules = list(self.ft_ext.children())[0][:30]

        elif feature == 'Vgg11':
            self.ft_ext = models.vgg11(weights=models.VGG11_Weights.DEFAULT if pretrained else None)
            self.ft_ext_modules = list(self.ft_ext.children())[0][:19]

        elif feature == 'Mobile':
            self.ft_ext = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
            self.ft_ext_modules = list(self.ft_ext.children())[0]

        self.ft_ext = nn.Sequential(*self.ft_ext_modules)
        for p in self.ft_ext.parameters():
            p.requires_grad = requires_grad

        # 模拟输入数据，计算实际的特征提取输出形状
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            dummy_output = self.ft_ext(dummy_input)
        actual_feature_shape = dummy_output.shape[1:]
        conv1_output_features = int(actual_feature_shape[0])

        fc1_input_features = int(conv1_output_features * actual_feature_shape[1] * actual_feature_shape[2])
        fc1_output_features = int(conv1_output_features * 2)
        fc2_output_features = int(fc1_output_features / 4)

        self.attn = att(conv1_output_features)

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=actual_feature_shape[0],
                out_channels=conv1_output_features,
                kernel_size=1,
            ),
            nn.BatchNorm2d(conv1_output_features),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(fc1_input_features, fc1_output_features),
            nn.BatchNorm1d(fc1_output_features),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(fc1_output_features, fc2_output_features),
            nn.BatchNorm1d(fc2_output_features),
            nn.ReLU()
        )

        self.out = nn.Linear(fc2_output_features, num_classes)

    def forward(self, x, drop_prob=0.5):
        x = self.ft_ext(x)
        # print(f"Feature extraction output shape: {x.shape}")
        x = self.attn(x)
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        # print(f"Flattened output shape: {x.shape}")
        x = self.fc1(x)
        x = nn.Dropout(drop_prob)(x)
        x = self.fc2(x)
        x = nn.Dropout(drop_prob)(x)
        prob = self.out(x)

        return prob

# 定义 SuperDAM 函数
def dvtt(num_classes=3, img_size=None, **kwargs):
    model = CNN(num_classes=num_classes, img_size=img_size)
    return model

# 注册 SuperDAM 到 timm 模型注册表，并支持预训练权重加载
@register_model
def dva(pretrained=False, **kwargs):
    model = dvtt(**kwargs)
    if pretrained:
        url = model_urls.get('dva_pretrained')
        if url:
            load_pretrained_weights(model, url)
        else:
            print("No pretrained weights URL found for dam.")
    return model

# 使用 create_model 函数创建 SuperDAM 模型
if __name__ == "__main__":
    model = create_model('dva', pretrained=False, num_classes=3)

    print(list_models())
    print(model)
    # 模拟输入数据
    input_data = torch.randn(64, 3, 224, 224)
    output = model(input_data)
    print(f"Output shape: {output.shape}")