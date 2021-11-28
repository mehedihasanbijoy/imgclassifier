import torch, torchvision


def alexnet(num_classes):
    pass


def vgg11(num_classes):
    pass


def vgg13(num_classes):
    pass


def vgg16(num_classes):
    pass


def vgg19(num_classes):
    pass


def resnet18(num_classes):
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes
    )
    return model


def resnet34(num_classes):
    pass


def resnet50(num_classes):
    pass


def resnet101(num_classes):
    pass


def resnet152(num_classes):
    pass


def densenet121(num_classes):
    pass


def densenet161(num_classes):
    pass


def densenet169(num_classes):
    pass


def densenet201(num_classes):
    pass


def mobilenet_v2(num_classes):
    pass


def mobilenet_v3_large(num_classes):
    pass


def mobilenet_v3_small(num_classes):
    pass


def mnasnet_1(num_classes):
    pass


def mnasnet_p5(num_classes):
    pass


def shufflenet_v2_1(num_classes):
    pass


def shufflenet_v2_p5(num_classes):
    pass


def resnext50_4d(num_classes):
    pass


def resnext101_8d(num_classes):
    pass