from torchvision import transforms
from torch.utils.data import DataLoader

from dataset.dataset import CNNClassifierDataset

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

dataset = CNNClassifierDataset(
    rootDir="dataset",
    transform=transform
)

dataLoader = DataLoader(
    dataset,
    batchSize=32,
    shuffle=True
)

print(dataset.classToIdx)
