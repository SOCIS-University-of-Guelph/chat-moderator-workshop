import torch
import torch.nn as nn


class ChatModerator(nn.Module):
    """
    A small MLP that classifies a message as toxic / not toxic.

    Input:  a bag-of-uni+bigrams vector of length vocab_size
    Output: a single raw logit
    """

    def __init__(self, vocab_size, hidden_dim=64):
        super().__init__()

        self.fc1 = nn.Linear(vocab_size, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x  # raw logit