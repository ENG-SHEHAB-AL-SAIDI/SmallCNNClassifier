from os import path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  
from torchvision import transforms
from torch.utils.data import DataLoader
from models.CNNClassifier import CNNClassifier 
from dataset.CNNClassifierDataset import CNNClassifierDataset
import configs.config as config


torch.manual_seed(config.RANDOM_SEED)

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

print(trainDataset.classToIdx)


device = config.DEVICE
model = CNNClassifier().to(device)

# ---------------------------
# 3. Loss & Optimizer
# ---------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

# ---------------------------
# 4. Training Loop with Progress Bar
# ---------------------------
def train(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    # Wrap loader with tqdm
    loop = tqdm(loader, desc="Training", leave=False)
    for x, y in loop:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

        loop.set_postfix(loss=running_loss/total, acc=100.*correct/total)

    epoch_loss = running_loss / total
    accuracy = 100. * correct / total
    return epoch_loss, accuracy

# ---------------------------
# 5. Evaluation Loop with Progress Bar
# ---------------------------
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    loop = tqdm(loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            running_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

            loop.set_postfix(loss=running_loss/total, acc=100.*correct/total)

    epoch_loss = running_loss / total
    accuracy = 100. * correct / total
    return epoch_loss, accuracy

# ---------------------------
# 6. Run Training with tqdm
# ---------------------------
epochs = config.NUM_EPOCHS
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    train_loss, train_acc = train(model, trainDataLoader, optimizer, criterion, device)
    test_loss, test_acc = evaluate(model, testDataLoader, criterion, device)
    print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}%")
