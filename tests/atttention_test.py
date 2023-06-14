import torch
import math
import unittest
from torch.nn import functional as F

from mingpt.model import CausalSelfAttention


class Config:
    def __init__(self, n_embd, n_head, block_size, attn_pdrop, resid_pdrop):
        self.n_embd = n_embd
        self.n_head = n_head
        self.block_size = block_size
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop

class TestCausalSelfAttention(unittest.TestCase):
    def test_tril_and_view(self):
        block_size = 5
        tensor = torch.tril(torch.ones(block_size, block_size))
        self.assertTrue(torch.all(tensor == torch.tensor([[1, 0, 0, 0, 0],
                                                          [1, 1, 0, 0, 0],
                                                          [1, 1, 1, 0, 0],
                                                          [1, 1, 1, 1, 0],
                                                          [1, 1, 1, 1, 1]])))
        tensor_viewed = tensor.view(1, 1, block_size, block_size)
        print(tensor_viewed)
        print(tensor_viewed[:,:,:block_size-1,:block_size-1])

    def test_forward(self):
        batch_size = 16
        seq_length = 20
        config = Config(n_embd=32, n_head=2, block_size=40, attn_pdrop=0.1, resid_pdrop=0.1)
        attention_layer = CausalSelfAttention(config)

        x = torch.rand(batch_size, seq_length, config.n_embd)  # create a random tensor to simulate input
        y = attention_layer(x)

        # Check output shape
        self.assertEqual(y.shape, (batch_size, seq_length, config.n_embd))

        # Check causality by ensuring every position doesn't attend to future positions
        q, k, _ = attention_layer.c_attn(x).split(config.n_embd, dim=2)
        k = k.view(batch_size, seq_length, config.n_head, config.n_embd // config.n_head).transpose(1, 2)
        q = q.view(batch_size, seq_length, config.n_head, config.n_embd // config.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(attention_layer.bias[:, :, :seq_length, :seq_length] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        future_attend = (att.triu(diagonal=1)>0).any()
        self.assertFalse(future_attend.item(), "Causality violated!")

if __name__ == "__main__":
    x = torch.rand(2, 3)  # Random tensor of size (2, 3)
    x_softmax = F.softmax(x, dim=-1)  # Apply softmax along the last dimension
    print(x_softmax)
    print(x_softmax.sum(dim=-1))  # Verify that the sum along the last dimension is 1

    # unittest.main()
