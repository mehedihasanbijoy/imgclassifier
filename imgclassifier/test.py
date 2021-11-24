import torch
import numpy as np

def train(
	model,
	test_loader,
	device='cuda'
):	
	model.eval()
	correct_preds, test_count = 0, 0, 0
	actual, predictions = [], []
	for i, (images, labels) in enumerate(test_loader):
		images, labels = images.to(device), labels.to(device)
		outputs = model(images)
		_, preds = torch.max(outputs.data, 1)
		test_count += labels.shape[0]
		correct_preds += (preds == labels).sum().item()
		actual.append(labels.cpu().numpy())
		predictions.append(preds.cpu().numpy())
	test_acc = (correct_preds / test_count)
	print(f'Correct/Total: {correct_preds}/{test_count}, Test Accuracy: {test_acc:.4f}')
	return test_acc, np.array(actual).flatten(), np.array(predictions).flatten()
