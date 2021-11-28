import torch, torchvision, cv2, os

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, df, transform):
        self.root_dir = root_dir
        self.df = pd.concat([df, pd.get_dummies(df.iloc[:, 1])], axis=1)
        self.transform = transform
        if self.transform == None:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize((128, 128)),
                torchvision.transforms.Grayscale(num_output_channels=3),
                torchvision.transforms.ToTensor(),
            ])
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.df.iloc[index, 0])
        img = cv2.imread(img_path)
        img = self.transform(img)
        label = torch.from_numpy(self.df.iloc[index, 2:].values.astype(float))
        return (img, label)