<h1 align="center">imgclassifier</h1>
  <p align="center">
    It is a naive image classifier developed on top of Pytorch. One can train and evaluate an image classifier on a custom dataset by simply calling the <b>train</b> function of this package.
  </p>

## Installation
```
!pip install git+https://github.com/mehedihasanbijoy/imgclassifier.git
```

## Usage
### Train and test the model
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

evaluation_report(targets, preds)
```

## Current Version : '0.0.1'


## Notebooks
#### Bangla Handwritten Character Recognition: https://colab.research.google.com/drive/13pu3Mw7FVPGV-f_g2VhVOMiE-wrhPo3F?usp=sharing
