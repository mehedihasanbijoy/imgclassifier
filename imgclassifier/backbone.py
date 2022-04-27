import torch, torchvision


def alexnet(num_classes):
    model = torchvision.models.alexnet(pretrained=True)
    model.classifier[6] = torch.nn.Linear(
        in_features=model.classifier[6].in_features,
        out_features=num_classes
    )
    return model


def vgg11(num_classes):
    model = torchvision.models.vgg11(pretrained=True)
    model.classifier[6] = torch.nn.Linear(
        in_features=model.classifier[6].in_features,
        out_features=num_classes
    )
    return model


def vgg13(num_classes):
    model = torchvision.models.vgg13(pretrained=True)
    model.classifier[6] = torch.nn.Linear(
        in_features=model.classifier[6].in_features,
        out_features=num_classes
    )
    return model


def vgg16(num_classes):
    model = torchvision.models.vgg16(pretrained=True)
    model.classifier[6] = torch.nn.Linear(
        in_features=model.classifier[6].in_features,
        out_features=num_classes
    )
    return model


def vgg19(num_classes):
    model = torchvision.models.vgg19(pretrained=True)
    model.classifier[6] = torch.nn.Linear(
        in_features=model.classifier[6].in_features,
        out_features=num_classes
    )
    return model


def resnet18(num_classes):
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes
    )
    return model


def resnet34(num_classes):
    model = torchvision.models.resnet34(pretrained=True)
    model.fc = torch.nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes
    )
    return model


def resnet50(num_classes):
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes
    )
    return model


def resnet101(num_classes):
    model = torchvision.models.resnet101(pretrained=True)
    model.fc = torch.nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes
    )
    return model


def resnet152(num_classes):
    model = torchvision.models.resnet152(pretrained=True)
    model.fc = torch.nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes
    )
    return model


def densenet121(num_classes):
    model = torchvision.models.densenet121(pretrained=True)
    model.classifier = torch.nn.Linear(
        in_features=model.classifier.in_features,
        out_features=num_classes
    )
    return model


def densenet161(num_classes):
    model = torchvision.models.densenet161(pretrained=True)
    model.classifier = torch.nn.Linear(
        in_features=model.classifier.in_features,
        out_features=num_classes
    )
    return model


def densenet169(num_classes):
    model = torchvision.models.densenet169(pretrained=True)
    model.classifier = torch.nn.Linear(
        in_features=model.classifier.in_features,
        out_features=num_classes
    )
    return model


def densenet201(num_classes):
    model = torchvision.models.densenet201(pretrained=True)
    model.classifier = torch.nn.Linear(
        in_features=model.classifier.in_features,
        out_features=num_classes
    )
    return model


def mobilenet_v2(num_classes):
    model = torchvision.models.mobilenet_v2(pretrained=True)
    model.classifier[1] = torch.nn.Linear(
        in_features=model.classifier[1].in_features,
        out_features=num_classes
    )
    return model


def mobilenet_v3_large(num_classes):
    model = torchvision.models.mobilenet_v3_large(pretrained=True)
    model.classifier[3] = torch.nn.Linear(
        in_features=model.classifier[3].in_features,
        out_features=num_classes
    )
    return model


def mobilenet_v3_small(num_classes):
    model = torchvision.models.mobilenet_v3_small(pretrained=True)
    model.classifier[3] = torch.nn.Linear(
        in_features=model.classifier[3].in_features,
        out_features=num_classes
    )
    return model


def mnasnet1_0(num_classes):
    model = torchvision.models.mnasnet1_0(pretrained=True)
    model.classifier[1] = torch.nn.Linear(
        in_features=model.classifier[1].in_features,
        out_features=num_classes
    )
    return model


def mnasnet0_5(num_classes):
    model = torchvision.models.mnasnet0_5(pretrained=True)
    model.classifier[1] = torch.nn.Linear(
        in_features=model.classifier[1].in_features,
        out_features=num_classes
    )
    return model


def shufflenet_v2_x1_0(num_classes):
    model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
    model.fc = torch.nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes
    )
    return model


def resnext101_32x8d(num_classes):
    model = torchvision.models.resnext101_32x8d(pretrained=True)
    model.fc = torch.nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes
    )
    return model


def resnext50_32x4d(num_classes):
    model = torchvision.models.resnext50_32x4d(pretrained=True)
    model.fc = torch.nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes
    )
    return model