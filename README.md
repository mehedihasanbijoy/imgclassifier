<h1 align="center">imgclassifier</h1>
  <p align="center">
    It is a naive image classifier developed on top of PyTorch. One can train a model and evaluate the performance of the model on a custom dataset by simply calling the <b>train</b> function of this package. An overview of image classification using imgclassifier is available on [YouTube](https://www.youtube.com/watch?v=ou3MaTrHQ0k&ab_channel=MehediHasanBijoy).
  </p>

<h3 align="center">Current Version: :zero:.:zero:.:two:</h3>

:heavy_check_mark: **Available backbones:** :heavy_minus_sign: default (resnet18) :heavy_minus_sign: alexnet :heavy_minus_sign: vgg11 :heavy_minus_sign: vgg13 :heavy_minus_sign: vgg16 :heavy_minus_sign: vgg19 :heavy_minus_sign: resnet18 :heavy_minus_sign: resnet34 :heavy_minus_sign: resnet50 :heavy_minus_sign: resnet101 :heavy_minus_sign: resnet152 :heavy_minus_sign: densenet121 :heavy_minus_sign: densenet161 :heavy_minus_sign: densenet169 :heavy_minus_sign: densenet201 :heavy_minus_sign: mobilenet_v2 :heavy_minus_sign: mobilenet_v3_large :heavy_minus_sign: mobilenet_v3_small :heavy_minus_sign: mnasnet1_0 :heavy_minus_sign: mnasnet0_5 :heavy_minus_sign: shufflenet_v2_x1_0 :heavy_minus_sign: resnext101_32x8d :heavy_minus_sign: resnext50_32x4d.

:heavy_check_mark: **Available Directory Structures:** :heavy_minus_sign: ImageFolder :heavy_minus_sign: Custom.

| ImageFolder | Custom |
| :---         |     :---       |
|<p>*Dataset<br>├── Train<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └── classes<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └── images<br>├── Validation<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └── classes<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └── images<br>├── Test<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └── classes<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └── images<br>*</p>    |  *Dataset<br>└── Classes<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; └── images<br>* </p>   |

:heavy_check_mark: **Evaluation Criteria:** :heavy_minus_sign: Precision :heavy_minus_sign: Recall :heavy_minus_sign: F1 Score :heavy_minus_sign: Accuracy.

## Installation
```
!pip install git+https://github.com/mehedihasanbijoy/imgclassifier.git
```

## Usage

### Train and test the model using "Custom" directory structure
```
from imgclassifier import train

model, train_acc, train_loss, test_acc, targets, preds = train(
    data_root='/Your/Dataset/Path', 
    folder_structure='Custom', 
    df = df,
    backbone='resnet18',
    transform = transform,
    device='cuda' if torch.cuda.is_available() else 'cpu', 
    epochs=10
)
```

### Train and test the model using "ImageFolder" directory structure
```
from imgclassifier import train

model, train_acc, train_loss, test_acc, targets, preds = train(
    data_root='/Your/Dataset/Path', 
    folder_structure='ImageFolder', 
    backbone='resnet18',
    device='cuda' if torch.cuda.is_available() else 'cpu', 
    epochs=10
)
```

### Evaluation of the method using Precision, Recall, F1 Score, and Accuracy
```
from imgclassifier import evaluation_report

pr, re, f1, acc = evaluation_report(targets, preds)
```



## Example Notebooks
#### Bangla Handwritten Character Recognition using imgclassifier
```
https://colab.research.google.com/drive/13pu3Mw7FVPGV-f_g2VhVOMiE-wrhPo3F?usp=sharing
```
#### Animal Classification using imgclassifier
```
https://colab.research.google.com/drive/1BKg0sFqQsatqQUMT73KrRavhRYLPqsab?usp=sharing
```
