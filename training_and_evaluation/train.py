import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_architecture.decoder import Decoder
from data_preprocessing.preprocessing import get_data
import time
import os

config = {
    'batch_size': 16,
    'block_size': 128,      # Maximum sequence length
    'n_embd': 128,         # Embedding dimension
    'n_head': 8,           # Number of attention heads
    'n_layer': 4,          # Number of transformer blocks
    'dropout': 0.3,        # Dropout rate to prevent overfitting
    'learning_rate': 1e-4,
    'max_epochs': 20,
    'eval_interval': 100,  # Evaluate every this many steps
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'grad_clip': 1.0,
    'eval_iters': 100,     # Number of iterations to estimate loss
    'ckpt_path': 'model_ckpt.pth'  # Path to save the checkpoint
}

def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(0, data.size(0) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, val_data, block_size, batch_size, device, eval_iters):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(val_data, block_size, batch_size, device)
        logits, loss = model(X, Y)
        losses[k] = loss.item()
    mean_loss = losses.mean()
    model.train()
    return mean_loss

def save_checkpoint(model, optimizer, config, epoch, step):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'epoch': epoch,
        'step': step
    }
    torch.save(checkpoint, config['ckpt_path'])
    print(f"Checkpoint saved at epoch {epoch+1}, step {step}")
    
def load_checkpoint(model, optimizer, config):
    if os.path.exists(config['ckpt_path']):
        checkpoint = torch.load(config['ckpt_path'], map_location=config['device'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        start_step = checkpoint['step'] + 1
        print(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}, step {checkpoint['step']}")
        return start_epoch, start_step
    else:
        return 0, 0

def train():
    train_data, val_data, _, vocab = get_data(config['batch_size'])
    vocab_size = len(vocab)
    print(f"Shape of train_data: {train_data.shape}")
    model = Decoder(vocab_size, config['n_embd'], config['n_head'], config['n_layer'], config['dropout'], config['block_size']).to(config['device'])
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    loss_fn = nn.CrossEntropyLoss()

    start_epoch, start_step = load_checkpoint(model, optimizer, config)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    steps_per_epoch = train_data.size(0) // config['block_size'] // config['batch_size']

    for epoch in range(start_epoch, config['max_epochs']):
        for step in range(start_step if epoch == start_epoch else 0, steps_per_epoch):
            model.train()
            xb, yb = get_batch(train_data, config['block_size'], config['batch_size'], config['device'])

            logits, loss = model(xb, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            optimizer.step()

            current_step = epoch * steps_per_epoch + step

            if current_step % config['eval_interval'] == 0:
                val_loss = estimate_loss(model, val_data, config['block_size'], config['batch_size'], config['device'], config['eval_iters'])
                print(f"Epoch {epoch+1}, Step {step}/{steps_per_epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")
                save_checkpoint(model, optimizer, config, epoch, current_step)

        start_step = 0 # Reset step for the new epoch

    print("Training finished!")
    
if __name__ == '__main__':
    train()