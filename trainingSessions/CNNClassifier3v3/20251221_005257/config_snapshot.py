import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#### session parameters
SESSION_DIR = "./trainingSessions/CNNClassifier3v3/"
CONFIG_PATH = "./configs/config.py"
# RESUME_SESSION_ID = None    # accept None , "last", "String ID"
RESUME_SESSION_ID = "last"

#### dataset parameters
NUM_CLASSES = 10 
IMAGE_SIZE = (32, 32)
DATASET_PATH = "dataset/rawData/cifar10"
MODEL_SAVE_PATH = "models/saved_model.pth"
RANDOM_SEED = 42
NUM_WORKERS = 2
PIN_MEMORY = True
DATA_MEAN = [0.4914, 0.4822, 0.4465];
DATA_STD = [0.2470, 0.2435, 0.2616];

#### hyperparameters 
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 50
