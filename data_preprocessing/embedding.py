import torch.nn as nn
from preprocessing import get_data

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

    def forward(self, x):
        return self.embedding(x)


"""
batch_size = 32
embed_dim = 256

train_data, valid_data, test_data, vocab = get_data(batch_size)
embedding = TokenEmbedding(len(vocab), embed_dim, pad_idx=vocab.string_index['<pad>'])

sample_input = train_data[:35]
embedded = embedding(sample_input)

print("Input shape:", sample_input.shape)
print("Embedded shape:", embedded.shape)

"""
