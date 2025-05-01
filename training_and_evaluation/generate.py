import torch
import torch.nn.functional as F
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_architecture.decoder import Decoder
from data_preprocessing.preprocessing import get_data

config = {'block_size': 128, 'n_embd': 128, 'n_head': 8, 'n_layer': 4, 'dropout': 0.3, 'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
final_model_path = 'models/transformer_final.pth'
max_new_tokens = 200

# Word-level tokenization in generation function
def generate(m, start_words, max_len, bsz, dev, vocab, itos):
    m.eval()
    with torch.no_grad():
        input_indices = torch.tensor([vocab.string_index[word] for word in start_words]).unsqueeze(0).to(dev)
        for _ in range(max_len):
            logits = m(input_indices[:, -bsz:])[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_index = torch.multinomial(probs, num_samples=1)
            input_indices = torch.cat((input_indices, next_index), dim=1)
        return " ".join([itos[i] for i in input_indices[0].tolist()])

def main():
    _, _, _, vocab = get_data(16)
    itos = vocab.index_string 
    vsz = len(vocab)
    dev = config['device']
    bsz = config['block_size']
    
    # Decoder setup
    m = Decoder(vsz, config['n_embd'], config['n_head'], config['n_layer'], config['dropout'], bsz).to(dev)
    
    try:
        m.load_state_dict(torch.load(final_model_path, map_location=dev))
        m.eval()
    except FileNotFoundError:
        print(f"Model not found at {final_model_path}")
        return

    print("Word-Level Text Generation")
    
    while True:
        prompt = input("Enter a word-based prompt (or 'quit' to exit): ")
        
        if prompt.lower() == 'quit':
            break

        valid_prompt = [word for word in prompt.split() if word in vocab.string_index]
        
        if not valid_prompt:
            print("Error: None of the words in your prompt are in the vocabulary. Please try again.")
            continue

        print(f"Generating text with prompt: {' '.join(valid_prompt)}")
        generated_text = generate(m, valid_prompt, max_new_tokens, bsz, dev, vocab, itos)
        print(f"\nGenerated Text: {generated_text}")
        print("-" * 40)

if __name__ == '__main__':
    main()
