import torch
import torchvision

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os
from sklearn.model_selection import train_test_split

from torchvision.models.resnet import resnet34
# from imgclassifier.backbone import resnet18
from imgclassifier.backbone import *
from imgclassifier.test import test
from imgclassifier.custom_dataset import *


def train(
	data_root, 
	folder_structure='ImageFolder',
	df = None,
	backbone='resnet18',
	transform = None,
	device='cuda', 
	epochs=20,
):	
	if folder_structure.lower()=='imagefolder':
		train_set_root = os.path.join(data_root, 'Train')
		test_set_root = os.path.join(data_root, 'Test')

		transform = torchvision.transforms.Compose([
		    # torchvision.transforms.ToPILImage(),
		    torchvision.transforms.Resize((40, 40)),
		    torchvision.transforms.ToTensor()
		])

		train_loader = DataLoader(
		    ImageFolder(train_set_root, transform=transform),
		    batch_size = 32, shuffle = True, pin_memory = True, drop_last = True, num_workers = 2
		)
		test_loader = DataLoader(
		    ImageFolder(test_set_root, transform=transform),
		    batch_size = 32, shuffle = True, pin_memory = True, drop_last = True, num_workers = 2
		)

		num_classes = len(os.listdir(os.path.join(data_root, 'Train')))


	elif folder_structure.lower()=='custom':
		# transform = torchvision.transforms.Compose([
		#     torchvision.transforms.ToPILImage(),
		#     torchvision.transforms.Resize((40, 40)),
		#     torchvision.transforms.ToTensor()
		# ])

		train_df, test_df = train_test_split(df, test_size=.2)

		dataset_train = CustomDataset(root_dir=data_root, df=train_df, transform=transform)
		dataset_test = CustomDataset(root_dir=data_root, df=test_df, transform=transform)

		train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=32, shuffle=True, drop_last=True, pin_memory=True)
		test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=32, shuffle=True, drop_last=False, pin_memory=True)

		num_classes = len(os.listdir(data_root))



	if backbone=='alexnet':
		model = alexnet(num_classes).to(device)
	elif backbone=='vgg11':
		model = vgg11(num_classes).to(device)
	elif backbone=='vgg13':
		model = vgg13(num_classes).to(device)
	elif backbone=='vgg16':
		model = vgg16(num_classes).to(device)
	elif backbone=='vgg19':
		model = vgg19(num_classes).to(device)
	elif backbone=='resnet18':
		model = resnet18(num_classes).to(device)
	elif backbone=='resnet34':
		model = resnet34(num_classes).to(device)
	elif backbone=='resnet50':
		model = resnet50(num_classes).to(device)
	elif backbone=='resnet101':
		model = resnet101(num_classes).to(device)
	elif backbone=='resnet152':
		model = resnet152(num_classes).to(device)
	elif backbone=='densenet121':
		model = densenet121(num_classes).to(device)
	elif backbone=='densenet161':
		model = densenet161(num_classes).to(device)
	elif backbone=='densenet169':
		model = densenet169(num_classes).to(device)
	elif backbone=='densenet201':
		model = densenet201(num_classes).to(device)
	elif backbone=='mobilenet_v2':
		model = mobilenet_v2(num_classes).to(device)
	elif backbone=='mobilenet_v3_large':
		model = mobilenet_v3_large(num_classes).to(device)
	elif backbone=='mobilenet_v3_small':
		model = mobilenet_v3_small(num_classes).to(device)
	elif backbone=='mnasnet1_0':
		model = mnasnet1_0(num_classes).to(device)
	elif backbone=='mnasnet0_5':
		model = mnasnet0_5(num_classes).to(device)
	elif backbone=='shufflenet_v2_x1_0':
		model = shufflenet_v2_x1_0(num_classes).to(device)
	elif backbone=='resnext101_32x8d':
		model = resnext101_32x8d(num_classes).to(device)
	elif backbone=='resnext50_32x4d':
		model = resnext50_32x4d(num_classes).to(device)
	else:
		print("WARNING - Model Name Does Not Exist\nDefault ResNet18 is being used\n\n")
		model = resnet18(num_classes).to(device)


	optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0)
	loss_fn = torch.nn.CrossEntropyLoss()

	
	current_loss = 10000000.
	for epoch in range(epochs):
		model.train()
		print(f'Epoch: {epoch}')
		train_count, correct_preds = 0, 0
		train_loss = 0.
		for i, (images, labels) in enumerate(train_loader):
			images, labels = images.to(device), labels.to(device)
			outputs = model(images)
			# outputs = torch.matmul(features.float(), embedding.float())
			optimizer.zero_grad()
			loss = loss_fn(outputs, labels)
			loss.backward()
			optimizer.step()
			if folder_structure.lower()=='custom':
				_, targets = torch.max(labels.data, 1)
			# _, targets = torch.max(labels.data, 1)
			_, preds = torch.max(outputs.data, 1)
			train_count += labels.shape[0]
			correct_preds += (preds == labels).sum().item()
			train_loss += loss.item() * labels.shape[0]

		train_acc = (correct_preds / train_count)
		train_loss = (train_loss / train_count)

		if train_loss < current_loss:
			current_loss = train_loss
			torch.save(model.state_dict(), 'model.pth')
			print('model saved')
		print(f'Train: Correct/Total: {correct_preds}/{train_count}, Train Loss: {train_loss:.2f}, Train Acc: {train_acc:.2f}')

		model.load_state_dict(torch.load('/content/model.pth'))
		test_acc, targets, preds = test(
			model=model,
			test_loader=test_loader,
			device=device
		)
		
	return model, train_acc, train_loss, test_acc, targets, preds