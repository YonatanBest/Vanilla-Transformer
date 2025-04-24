import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from model_architecture.decoder import Decoder
from data_preprocessing.preprocessing import get_data
import time
import os

config = {
    'batch_size': 32,
    'block_size': 128,      # Maximum sequence length
    'n_embd': 256,         # Embedding dimension
    'n_head': 8,           # Number of attention heads
    'n_layer': 6,          # Number of transformer blocks
    'dropout': 0.2,        # Dropout rate
    'learning_rate': 3e-4,
    'max_epochs': 10,
    'eval_interval': 500,  # Evaluate every this many steps
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'grad_clip': 1.0,
    'eval_iters': 200,     # Number of iterations to estimate loss
}
