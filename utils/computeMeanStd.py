import torch
from os import path
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset.CNNClassifierDataset import CNNClassifierDataset
import configs.config as config



def computeMeanStd(dataLoader):
    channelSum = torch.zeros(3)
    channelSquaredSum = torch.zeros(3)
    numPixels = 0

    for images, _ in dataLoader:
        batchSize, channels, height, width = images.shape
        numPixels += batchSize * height * width

        channelSum += images.sum(dim=[0, 2, 3])
        channelSquaredSum += (images ** 2).sum(dim=[0, 2, 3])

    mean = channelSum / numPixels
    std = (channelSquaredSum / numPixels - mean ** 2).sqrt()

    return mean, std


transform = transforms.Compose([
    transforms.Resize(config.IMAGE_SIZE),
    transforms.ToTensor()
])

trainDataset = CNNClassifierDataset(
    rootDir=path.join(config.DATASET_PATH, "train"),
    transform=transform
)

testDataset = CNNClassifierDataset(
    rootDir=path.join(config.DATASET_PATH, "test"),
    transform=transform
)

trainDataLoader = DataLoader(
    dataset=trainDataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=config.NUM_WORKERS,
    pin_memory=config.PIN_MEMORY
)

testDataLoader = DataLoader(
    dataset=testDataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=config.NUM_WORKERS,
    pin_memory=config.PIN_MEMORY
)

meanTrain,stdTrain = computeMeanStd(trainDataLoader)
print(f"meanTrain = {meanTrain},stdTrain={stdTrain} ")

meanTest,stdTtest = computeMeanStd(testDataLoader)
print(f"meanTest = {meanTest},stdTtest={stdTtest} ")


