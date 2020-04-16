import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU_NN(nn.Module):

    def __init__(self, n_gru_hidden=256):
        super(GRU_NN, self).__init__()

        self.n_gru_hidden = n_gru_hidden
        self.fc1 = nn.Linear(self.n_gru_hidden, 512)
        self.fc2 = nn.Linear(512, 3)

        self.gru = nn.GRU(input_size=144, hidden_size=n_gru_hidden, num_layers=1)

    def forward(self, x):
        # PyTorch needs the channels first (n_samples, dim,  n_feature_vectors)
        x = x.transpose(0, 1)

        h0 = torch.autograd.Variable(torch.zeros(1, x.size(1), self.n_gru_hidden).cuda())
        x = self.gru(x, h0)

        # Keep the last output
        x = x[-1].squeeze()

        # Classifier
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        return x


class LSTM_NN(nn.Module):

    def __init__(self, n_gru_hidden=256):
        super(LSTM_NN, self).__init__()

        self.n_gru_hidden = n_gru_hidden
        self.fc1 = nn.Linear(self.n_gru_hidden, 512)
        self.fc2 = nn.Linear(512, 3)

        self.lstm = nn.LSTM(input_size=144, hidden_size=n_gru_hidden, num_layers=1)

    def forward(self, x):
        # PyTorch needs the channels first (n_samples, dim,  n_feature_vectors)
        x = x.transpose(0, 1)

        # h0 = torch.autograd.Variable(torch.zeros(1, x.size(1), self.n_gru_hidden).cuda())
        x, (hn, cn) = self.lstm(x)

        # Keep the last output
        x = x[-1].squeeze()

        # Classifier
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        return x


class CNN_NN(nn.Module):

    def __init__(self, pooling=5, n_conv=256):
        super(CNN_NN, self).__init__()

        self.fc1 = nn.Linear(n_conv, 512)
        self.fc2 = nn.Linear(512, 3)

        # Input Convolutional
        self.input_conv = nn.Conv1d(144, n_conv, kernel_size=5, padding=2)

    def forward(self, x):
        # PyTorch needs the channels first (n_samples, dim,  n_feature_vectors)
        x = x.transpose(1, 2)

        # Apply a convolutional layer
        x = self.input_conv(x)
        x = F.tanh(x)

        x = F.avg_pool1d(x, x.size(2)).squeeze()

        # Classifier
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)

        return x
