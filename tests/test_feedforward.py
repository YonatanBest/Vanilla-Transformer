import torch
from model_architecture.feedforward import FeedFoward

def Test_feedforward_output_shape():
    batch_size = 2
    seq_len = 5
    embedding_dim = 128
    dropout_rate = 0.1
    dummy_input = torch.rand(batch_size, seq_len, embedding_dim)

    # let's initialize the feedforward network
    ffwd = FeedFoward(n_embd=embedding_dim, dropout= dropout_rate)
    
    output = ffwd(dummy_input)
    assert output.shape == (batch_size, seq_len, embedding_dim), \
        f"Expected output shape {(batch_size, seq_len, embedding_dim)}, but got {output.shape}"
    print("Test passed: Output shape is correct.")
    
def Test_Feedforward_dtype():
    batch_size = 2
    seq_len = 5
    embedding_dim = 128
    dropout_rate = 0.1
    
    dummy_input = torch.rand(batch_size, seq_len, embedding_dim, dtype=torch.float)
    
    ffwd = FeedFoward(n_embd=embedding_dim, dropout= dropout_rate)
    
    output = ffwd(dummy_input)

    assert output.dtype == dummy_input.dtype, \
        f"Expected output dtype {dummy_input.dtype}, but got {output.dtype}"
    print("Test passed: Output dtype is correct.")
    

if __name__ == "__main__":
    Test_feedforward_output_shape()
    Test_Feedforward_dtype()
    print("All FeedFoward tests passed!")