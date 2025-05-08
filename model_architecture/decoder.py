import torch
import torch.nn as nn
import torch.nn.functional as F
from .positional_encoding import PositionalEncoding
from .block import Block 
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_preprocessing.embedding import TokenEmbedding

class Decoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, dropout, block_size, pad_idx=0):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, n_embd, pad_idx=pad_idx)
        self.position_embedding = PositionalEncoding(n_embd, dropout, max_len=block_size)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.pad_idx = pad_idx
        self.block_size = block_size

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Embed tokens and add positional encoding
        tok_emb = self.token_embedding(idx) 
        # pos_emb = self.position_embedding(torch.arange(T, device=idx.device)) # (T, n_embd)
        pos_emb = self.position_embedding(tok_emb)
        x = self.dropout(tok_emb + pos_emb) 
        x = self.blocks(x)

        # Final layer normalization
        x = self.ln_f(x)
        # Output head
        logits = self.head(x) 

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            return logits, loss
        else:
            return logits

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
