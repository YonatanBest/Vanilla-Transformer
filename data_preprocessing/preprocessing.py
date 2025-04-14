import torch
from collections import Counter

def load(path):
    with open(path, 'r') as f:
        text = f.read().replace('\n', ' <eos> ')
    return text.strip().split()

class Vocab:
    def __init__(self, tokens, min_freq=1):
        counter = Counter(tokens)
        self.index_string = ['<pad>', '<unk>'] + sorted([tok for tok, freq in counter.items() if freq >= min_freq])
        self.string_index = {tok: i for i, tok in enumerate(self.index_string)}

    def __len__(self):
        return len(self.index_string)

    def encode(self, tokens):
        return [self.string_index.get(token, self.string_index['<unk>']) for token in tokens]

    def decode(self, indices):
        return [self.index_string[index] for index in indices]


def tokener(tokens, vocab):
    return torch.tensor(vocab.encode(tokens))


def batcher(data, batch_size):
    nbatch = data.size(0) // batch_size
    data = data.narrow(0, 0, nbatch * batch_size)
    return data.view(batch_size, -1).t().contiguous()

def get_data(batch_size):
    train = load('ptbdataset/ptb.train.txt')
    valid = load('ptbdataset/ptb.valid.txt')
    test  = load('ptbdataset/ptb.test.txt')

    vocab = Vocab(train)

    train_data = batcher(tokener(train, vocab), batch_size)
    valid_data = batcher(tokener(valid, vocab), batch_size)
    test_data  = batcher(tokener(test, vocab), batch_size)

    return train_data, valid_data, test_data, vocab
"""
if __name__ == '__main__':
    train_data, valid_data, test_data, vocab = get_data(32)
    print(f"Shape: {train_data.shape}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Examples: {list(vocab.index_string)}")
    print(vocab.string_index['<unk>'])
    print(test_data)
"""
