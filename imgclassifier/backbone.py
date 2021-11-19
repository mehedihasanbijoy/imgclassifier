import torch, torchvision


def resnet18(num_classes):
    '''
        out_embd_size: size of the feature vector. i.e. embedding.shape[0] = 400
    '''
    model = torchvision.models.resnet18(pretrained=True)

    model.fc = torch.nn.Linear(
        in_features=model.fc.in_features,
        out_features=num_classes
    )

    return model