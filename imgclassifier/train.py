import torch
import torchvision

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os
from imgclassifier.backbone import resnet18
from imgclassifier.test import test


def train(
	data_root, 
	folder_structure='ImageFolder',
	backbone='resnet18',
	device='cuda', 
	epochs=20,
):	
	if folder_structure=='ImageFolder':
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

	if backbone=='resnet18':
		model = resnet18(num_classes).to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0)
	loss_fn = torch.nn.CrossEntropyLoss()

	print('training begins...')
	# current_loss = 10000000.
	for epoch in range(epochs):
	    model.train()
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
	        # _, targets = torch.max(labels.data, 1)
	        _, preds = torch.max(outputs.data, 1)
	        train_count += labels.shape[0]
	        correct_preds += (preds == labels).sum().item()
	        train_loss += loss.item() * labels.shape[0]
	    train_acc = (correct_preds / train_count)
	    train_loss = (train_loss / train_count)

	#     if train_loss < current_loss:
	#         current_loss = train_loss
	#         torch.save(model.state_dict(), '/content/model.pth')
	#         print('model saved')

	    print(f'Epoch: {epoch}, Correct/Total: {correct_preds}/{train_count}, Train Loss: {train_loss:.2f}, Train Acc: {train_acc:.2f}')
	print('training ends...')

	print('test begins...')
	test_acc, targets, preds = test(
		model=model,
		test_loader=test_loader,
		device=device
	)
	print('test ends...')
	return model, train_acc, train_loss, test_acc, targets, preds