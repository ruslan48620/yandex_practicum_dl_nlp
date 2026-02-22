# from __future__ import annotations

# from typing import Optional

import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class LSTMAutocompleteModel(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim=256,
                 hidden_dim=256,
                 num_layers=1,
                 dropout=0.1,
                 pad_token_id=0
                 ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, lengths, hidden=None):
        emb = self.embedding(input_ids) # (batch_size, seq_len, embedding_dim)

        packed = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.lstm(packed, hidden)
        outputs, _ = pad_packed_sequence(packed_out, batch_first=True) # (batch_size, seq_len, hidden_dim)

        # logits = self.fc(hidden[-1])
        logits = self.fc(outputs) # (batch_size, seq_len, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self,
                 prefix_ids,
                 max_new_tokens=20,
                 eos_token_id=None,
                 ):
        was_training = self.training
        self.eval()

        if isinstance(prefix_ids, list):
            generated = torch.tensor(prefix_ids, dtype=torch.long).unsqueeze(0)
        else:
            generated = prefix_ids.clone().detach()
            if generated.dim() == 1:
                generated = generated.unsqueeze(0)

        device = next(self.parameters()).device
        generated = generated.to(device)

        for _ in range(max_new_tokens):
            lengths = torch.full(
                (generated.size(0),),
                generated.size(1),
                dtype=torch.int64,
                device='cpu'
            )

            logits = self.forward(generated, lengths)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and int(next_token.item()) == eos_token_id:
                break

        if was_training:
            self.train()

        return generated.squeeze(0)
