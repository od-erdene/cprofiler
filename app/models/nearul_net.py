import torch

class Network(nn.Module):

    def __init__(self, input_dim, middle_dim, out_dim):

        super(Network, self).__init__()
        
        self.nn1 = nn.Linear(input_dim, middle_dim)
        self.batch1 = nn.BatchNorm1d(middle_dim)
        self.relu1 = nn.ReLU()
        
        self.nn2 = nn.Linear(middle_dim, middle_dim)
        self.batch2 = nn.BatchNorm1d(middle_dim)
        self.relu2 = nn.ReLU()
        
        self.nn3 = nn.Linear(middle_dim, middle_dim)
        self.batch3 = nn.BatchNorm1d(middle_dim)
        self.relu3 = nn.ReLU()
        
        self.nn_out = nn.Linear(middle_dim, out_dim)
        
    def forward(self, x):

        out1 =self.relu1(self.batch1(self.nn1(x)))
        out2 = self.relu2(self.batch2(self.nn2(out1)))
        out2 = out2 + out1

        out3 = self.relu3(self.batch3(self.nn3(out2)))
        out3 = out3 + out2

        out = self.nn_out(out3)

        return out

