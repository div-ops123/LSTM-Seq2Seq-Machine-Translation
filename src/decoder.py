"""LSTM decoder for the vanilla (no-attention) seq2seq model.

Teacher forcing during training (target_tensor given), autoregressive
decoding during inference (target_tensor=None) -- per docs/TODO.md
Milestone 3. No beam search, no attention: only the encoder's final
(hidden, cell) seeds the decoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.vocab import Vocab


class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size: int, tgt_vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(
            tgt_vocab_size, hidden_size, padding_idx=Vocab.PAD_token
        )
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, tgt_vocab_size)

    def forward(self, encoder_hidden, encoder_cell, max_length, target_tensor=None):
        batch_size = encoder_hidden.size(1)
        device = encoder_hidden.device

        decoder_input = torch.full(
            (batch_size, 1), Vocab.SOS_token, dtype=torch.long, device=device
        )
        hidden, cell = encoder_hidden, encoder_cell
        decoder_outputs = []

        for i in range(max_length):
            output, hidden, cell = self.forward_step(decoder_input, hidden, cell)
            decoder_outputs.append(output)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)  # teacher forcing
            else:
                _, topi = output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # autoregressive

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, (hidden, cell)

    def forward_step(self, input, hidden, cell):
        embedded = self.embedding(input)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        output = self.out(output)
        return output, hidden, cell
