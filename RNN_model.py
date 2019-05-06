import torch
import torch.nn.functional as F
from torch import nn

class RNN(nn.Module):   
    def __init__(self, n_vocab):
        super().__init__()
        self.n_vocab = n_vocab
        self.embedding_size = 100
        self.hidden_size = 32
        self.num_layers = 2
        self.word_embeddings = nn.Embedding(self.n_vocab, self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, dropout=0.5)
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, 1)
        self.relu = nn.ReLU()
        
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.permute(1,0,2))
        fc1_out = self.fc1(lstm_out[-1])
        fc2_out = self.fc2(self.relu(fc1_out))
        return fc2_out 