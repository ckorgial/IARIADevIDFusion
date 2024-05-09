import torch.nn as nn
import torchvision.models as models


class SqueezeNet1_1(nn.Module):
    # https://github.com/culv/SqueezeTune/blob/master/finetune.py
    def __init__(self, num_classes, weights=models.MobileNet_V3_Small_Weights.DEFAULT, freeze=False):
        super(SqueezeNet1_1, self).__init__()
        self.model = models.squeezenet1_1(weights=weights)
        # Reshape classification layer to have 'num_classes' outputs
        self.model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        self.model.num_classes = num_classes

        # Replace dropout layer in classifier with batch normalization
        self.model.classifier[0] = nn.BatchNorm2d(512)

        if freeze:
            # Freeze all parameters
            for p in self.model.parameters():
                p.requires_grad = False

            for p in self.model.classifier.parameters():
                p.requires_grad = True

    def forward(self, x):
        output = self.model(x)
        return output


class MobileNetV3Large(nn.Module):
    def __init__(self, num_classes, weights=models.MobileNet_V3_Small_Weights.DEFAULT, freeze_params=False):
        super(MobileNetV3Large, self).__init__()
        self.model = models.mobilenet_v3_large(weights=weights)
        self.model.classifier = nn.Sequential(*[nn.Linear(960, 350),
                                                nn.Linear(350, num_classes)])
        # self.model.classifier = nn.Linear(960, num_classes)

        if freeze_params:
            # Freeze all parameters
            for p in self.model.parameters():
                p.requires_grad = False

            for p in self.model.classifier.parameters():
                p.requires_grad = True

    def forward(self, x):
        output = self.model(x)
        return output


class MobileNetV3Small(nn.Module):
    def __init__(self, num_classes, weights=models.MobileNet_V3_Small_Weights.DEFAULT):
        super(MobileNetV3Small, self).__init__()
        self.model = models.mobilenet_v3_small(weights=weights)
        self.model.classifier = nn.Linear(576, num_classes)

    def forward(self, x):
        output = self.model(x)
        return output


class DenseNet201(nn.Module):
    def __init__(self, num_classes, weights=models.DenseNet201_Weights.DEFAULT):
        super(DenseNet201, self).__init__()
        self.model = models.densenet201(weights=weights)
        self.model.classifier = nn.Linear(1920, num_classes)

    def forward(self, x):
        output = self.model(x)
        return output


class ResNet50(nn.Module):
    def __init__(self, num_classes, weights=models.ResNet50_Weights.DEFAULT):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(weights=weights)
        self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        output = self.model(x)
        return output


class ResNet18(nn.Module):
    def __init__(self, num_classes, weights=models.ResNet50_Weights.DEFAULT):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(weights=weights)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        output = self.model(x)
        return output


class InceptionV3(nn.Module):
    def __init__(self, num_classes, weights=True):
        super(InceptionV3, self).__init__()
        self.model = models.inception_v3(weights=weights, aux_logits=False)
        self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        output = self.model(x)
        return output
