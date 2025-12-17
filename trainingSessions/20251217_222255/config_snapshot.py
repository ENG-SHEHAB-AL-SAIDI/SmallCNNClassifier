import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#### session parameters
SESSION_DIR = "./trainingSessions/"
CONFIG_PATH = "./configs/config.py"
RESUME_SESSION_ID = "last"    # accept None , "last", "String ID"

#### dataset parameters
NUM_CLASSES = 10 
IMAGE_SIZE = (32, 32)
DATASET_PATH = "dataset/rawData/cifar10"
MODEL_SAVE_PATH = "models/saved_model.pth"
RANDOM_SEED = 42
NUM_WORKERS = 2
PIN_MEMORY = True

#### hyperparameters 
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 30
