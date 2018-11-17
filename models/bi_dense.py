import math
import torch
import torch.nn as nn
import torch.nn.functional as F

input_size = 32
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 1
seq_len = 3
channel = 3

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BidirectRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, batch_size):
        super(BidirectRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # self.fc = nn.Linear(hidden_size * 4, num_classes)  # 2 for bidirection
        # self.l1 = nn.Linear(hidden_size * 4 * 32, hidden_size * 4)

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

        #print(c.size(), d.size())

        # Forward propagate LSTM
        out1, _ = self.lstm1(c, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        out2, _ = self.lstm2(d, (h1, c1))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        #print(torch.cat((out1, out2), dim=2)[:,-1,:].size())
        #print(torch.cat((out1, out2), dim=2)[:,:,:].size())
        # Decode the hidden state of the last time step
        out = torch.cat((out1, out2), dim=2).view(batch_size, -1) #out = hidden *4 *32
        #out = self.fc(out1)[:,-1,:]
        return out


class gru(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(gru, self).__init__()
        self.gru1 = nn.GRU(input_size, hidden_size * 2, 1, batch_first=True, bidirectional=False)
        self.gru2 = nn.GRU(hidden_size * 2, hidden_size, 1, batch_first=True, bidirectional=False)
        #self.fc = nn.Linear(hidden_size * 4, num_classes)  # 2 for bidirection
        #self.l1 = nn.Linear(hidden_size * 4 * 32, hidden_size * 4)

    def forward(self, x, h0, h1):
        out = x.view(batch_size, seq_len, -1)
        h00 = h0.view(batch_size, 1, -1)
        h11 = h1.view(batch_size, 1, -1)
        out, _ = self.gru1(out, h00.transpose(0, 1))  # out: tensor of shape (batch_size, seq_length, 96 * 8 * 8)
        out, _ = self.gru2(out, h11.transpose(0, 1))  # out: tensor of shape (batch_size, seq_length, 384 * 4 * 4)

        return out

class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes


        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes


        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        #self.linear = nn.Linear(num_planes, num_classes)
        self.birnn = BidirectRNN(input_size * channel, hidden_size, num_layers, num_classes, batch_size)
        #self.birnn1 = BidirectRNN(64*32, hidden_size, num_layers, num_classes, batch_size)
        #self.birnn2 = BidirectRNN(128*16, hidden_size, num_layers, num_classes, batch_size)
        self.linear = nn.Linear(hidden_size * 4 * 32, num_planes )
        self.linear1 = nn.Linear(num_planes + num_planes, num_classes)
        #self.linear3 = nn.Linear(hidden_size * 4 , hidden_size)
        #self.linear4 = nn.Linear(hidden_size * 4 , hidden_size)
        #self.linear5 = nn.Linear(hidden_size * 4 , num_planes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        #out1 = self.birnn1(out)
        #out1 = self.linear3(out1)
        out = self.trans1(self.dense1(out))
        #out2 = self.birnn2(out)
        #out2 = self.linear4(out2)
        out = self.trans2(self.dense2(out))
        #print(out.size())
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out1 = self.birnn(x)
        out1 = self.linear(out1)
        #out3 = self.linear(torch.cat((out1, out2, out3), dim=1))
        out1 = F.relu(out1)
        out = self.linear1(torch.cat((out, out1), dim=1))#
        return out

def bi_DenseNet121():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)

def bi_DenseNet169():
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

def bi_DenseNet201():
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

def bi_DenseNet161():
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)

def bi_densenet_cifar():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12)

def test():
    net = bi_densenet_cifar().to(device)
    x = torch.randn(25,3,32,32).to(device)
    y = net(x)
    print(y)

if __name__ == "__main__":
    test()
