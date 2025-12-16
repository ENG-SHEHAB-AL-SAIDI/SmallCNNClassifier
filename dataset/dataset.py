import os
from PIL import Image
from torch.utils.data import Dataset

class CNNClassifierDataset(Dataset):
    def __init__(self, rootDir, transform=None):
        self.rootDir = rootDir
        self.transform = transform
        self.imagePaths = []
        self.labels = []
        self.classToIdx = {}

        self._scanDataset()

    def _scanDataset(self):
        classes = sorted(os.listdir(self.rootDir))

        for idx, className in enumerate(classes):
            classPath = os.path.join(self.rootDir, className)

            if not os.path.isdir(classPath):
                continue

            self.classToIdx[className] = idx

            for fileName in os.listdir(classPath):
                if fileName.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.imagePaths.append(
                        os.path.join(classPath, fileName)
                    )
                    self.labels.append(idx)

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, index):
        imagePath = self.imagePaths[index]
        label = self.labels[index]

        image = Image.open(imagePath).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
