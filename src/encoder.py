"""LSTM encoder for the vanilla (no-attention) seq2seq model.

`outputs` is computed but never handed to the decoder -- Milestone 3
deliberately has the decoder rely only on the final hidden/cell state, so
the fixed-context bottleneck Milestone 5-7 investigate is actually present.
"""
import torch.nn as nn

from src.vocab import Vocab


class EncoderLSTM(nn.Module):
    def __init__(self, src_vocab_size: int, hidden_size: int, dropout_p: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(
            src_vocab_size, hidden_size, padding_idx=Vocab.PAD_token
        )
        self.dropout = nn.Dropout(dropout_p)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, (hidden, cell)
