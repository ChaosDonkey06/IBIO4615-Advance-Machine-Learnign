import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from sru import SRU, SRUCell

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN_SRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN_SRU, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.sru = SRU(hidden_size, hidden_size)

    def forward(self, input, hidden ,cell ):
        
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.sru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    def initCell(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN_SRU(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN_SRU, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.sru = SRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden,cell):

        output = self.embedding(input).view(1, 1, -1)

        output = F.relu(output)
        hidden, cell = self.sru(hidden, cell)
        hidden = self.softmax(self.out(hidden[0]))
        return hidden, cell

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    def initCell(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN_SRU(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=10):
        super(AttnDecoderRNN_SRU, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.sru = SRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, cell, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        hidden, cell = self.sru(hidden, cell)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return hidden, cell , attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    def initCell(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
