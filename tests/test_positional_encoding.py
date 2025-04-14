import unittest
import torch
import torch.nn as nn 
from model_architecture.positional_encoding import PositionalEncoding

class TestPositionalEncoding(unittest.TestCase):
    def test_output_shape_and_values(self):
        batch_size = 8
        seq_len = 20
        embedding_dim = 64
        max_len = 30
        input_embeddings = torch.randn(batch_size, seq_len, embedding_dim)
        pos_encoder = PositionalEncoding(d_model=embedding_dim, max_len=max_len)
        output = pos_encoder(input_embeddings)
        # Check output shape
        self.assertEqual(output.shape, input_embeddings.shape)

        # Check positional encoding shape
        pe_shape = pos_encoder.pe.squeeze(0).shape
        self.assertEqual(pe_shape, (max_len, embedding_dim))

        # Check if values are within expected range
        pe = pos_encoder.pe.squeeze(0).numpy()
        self.assertTrue(all(-1.0 <= val <= 1.0 + 1e-7 for row in pe for val in row),
                        "Positional encoding values should be within [-1, 1]")

        # Check for uniqueness of first few encodings
        if max_len >= 2:
            self.assertFalse(torch.allclose(torch.tensor(pe[0]), torch.tensor(pe[1]), atol=1e-6),
                             "First two positional encodings should be different")

        # Check if dropout layer exists
        self.assertIsInstance(pos_encoder.dropout, nn.Dropout)

        # Check forward pass dtype
        self.assertEqual(output.dtype, torch.float32)

if __name__ == '__main__':
    unittest.main()