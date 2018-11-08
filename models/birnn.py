'''bidirection rnn by pytorch'''
import torch
import torch.nn as nn
import torch.nn.functional as F

input_size = 32
hidden_size = 64
num_layers = 3
num_classes = 10
batch_size = 16
channel = 3


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BidirectRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BidirectRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 4, num_classes)  # 2 for bidirection
        self.l1 = nn.Linear(hidden_size * 4 * 32, hidden_size * 4)

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        h1 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)  # 2 for bidirection
        c1 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        b = x.transpose(2, 3)
        c = x.reshape(batch_size, -1, input_size)
        c = c.transpose(1, 2)
        d = b.reshape(batch_size, -1, input_size)
        d = d.transpose(1, 2)

        # Forward propagate LSTM
        out1, _ = self.lstm1(c, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        out2, _ = self.lstm2(d, (h1, c1))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        #print(torch.cat((out1, out2), dim=2)[:,-1,:].size())
        #print(torch.cat((out1, out2), dim=2)[:,:,:].size())
        # Decode the hidden state of the last time step
        out = F.relu(self.l1(torch.cat((out1, out2), dim=2).view(batch_size, -1)))
        out = self.fc(out)
        #out = self.fc(out1)[:,-1,:]
        return out


def BiRNN():
    return BidirectRNN(input_size * channel, hidden_size, num_layers, num_classes)

def test():
    net = BidirectRNN(input_size * channel, hidden_size, num_layers, num_classes)
    x = torch.randn(1, 3, 32, 32)
    y = net(x)
    print(y)
