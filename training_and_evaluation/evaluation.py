import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_architecture.decoder import Decoder
from data_preprocessing.preprocessing import get_data

# Model configueration
config = {
    'batch_size': 16,
    'block_size': 128,
    'n_embd': 128,
    'n_head': 8,
    'n_layer': 4,
    'dropout': 0.3,
    'learning_rate': 1e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# Trained Model Path
final_model_path = 'models/transformer_final.pth'

@torch.no_grad()
def evaluate(model, test_data, block_size, batch_size, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    steps_per_epoch = (test_data.size(0) - block_size) // batch_size

    for i in range(0, test_data.size(0) - block_size, batch_size):
        xb = test_data[i:i + block_size].unsqueeze(0).to(device) # Add batch dimension
        yb = test_data[i + 1:i + block_size + 1].unsqueeze(0).to(device) # Add batch dimension

        logits, loss = model(xb, yb)
        total_loss += loss.item()
        num_batches += 1
        if num_batches % 100 == 0:
            print(f"Evaluated batch {num_batches}/{steps_per_epoch}, Loss: {loss.item():.4f}")

    mean_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(mean_loss))
    return mean_loss, perplexity

def main():
    _, _, test_data, vocab = get_data(config['batch_size'])
    vocab_size = len(vocab)
    device = config['device']
    block_size = config['block_size']
    batch_size = config['batch_size']

    # Initialize the model same architecture as during of the training
    model = Decoder(vocab_size, config['n_embd'], config['n_head'], config['n_layer'], config['dropout'], config['block_size']).to(device)

    # Load the state dictionary of the final trained model
    model.load_state_dict(torch.load(final_model_path, map_location=device))
    model.eval() 
    
    print("Starting evaluation...")
    test_loss, test_perplexity = evaluate(model, test_data, block_size, batch_size, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Perplexity: {test_perplexity:.4f}")

if __name__ == '__main__':
    main()