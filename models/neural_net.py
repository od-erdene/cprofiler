import argparse
import numpy as np
import torch

import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import Adam
from torch.autograd import Variable

from sklearn.preprocessing import normalize

from make_data import PrepareData


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', type=str, default='')
    parser.add_argument('--test_path', type=str, default='')
    parser.add_argument('--data_cols', type=str, default='keyword')
    parser.add_argument('--target_column', type=str, default='SEX')

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--class_num', type=int, default=2)
    
    args = parser.parse_args()
    return args


def get_transformer():
    transformer_list =[]
    transformer_list.append(transforms.ToTensor())
    #transformer_list.append()
    transformer = transforms.Compose(transformer_list)
    return transformer


def get_columns(args):

    test_prepare = PrepareData(args.test_path)
    dat_cols = []
    columns = args.data_cols.split(',')

    if 'keyword' in columns:
        dat_cols = dat_cols + test_prepare.keyword_columns
        
    if 'time' in columns:
        dat_cols = dat_cols + test_prepare.time_columns

    if 'device' in columns:
        dat_cols = dat_cols + test_prepare.device_columns
    
    return dat_cols


class KeywordLoader(Dataset):

    def __init__(self, data_path, class_num, feature_columns,
                 target_column, transformer):

        dataPrep = PrepareData(data_path)
        self.data, self.label = dataPrep.return_features(feature_columns,
                                                    target_column)

        if target_column in ['SEX', 'CHILD', 'PREFECTURE', 'MARRIED', 'AREA']:
            self.label = self.label - 1
        
        self.data = normalize(self.data, norm='l1')
        
    def __getitem__(self, idx):
        #return self.transformer(self.data[idx]), self.transformer(self.label[idx])
        return self.data[idx], self.label[idx]
        
    def __len__(self):
        return len(self.data)

    def make_onehot(self, label_data, dimension):
        return np.eye(dimension)[label_data]

    
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

    
class Model:

    def __init__(self, network, train_loader, test_loader,
                 device):

            
        self.network = network
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.network.parameters(), lr=0.001, betas=(0.9, 0.999))

        self.device = device
        self.network.to(device)
                
    def train_one_epoch(self):

        self.network.train()
        
        for data, label in self.train_loader:

            data = data.to(self.device)
            label = label.to(self.device).long()
            out = self.network(data.float())
            loss = self.criterion(out, label)
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def test(self):

        self.network.eval()
        correct = 0
        total = 0
        
        for data, label in self.test_loader:
            
            data = data.to(self.device)
            label = label.to(self.device).long()
                        
            out = self.network(data.float())
            _, predicted = torch.max(out, 1)

            total += label.size(0)
            correct += (predicted == label).sum().item()
            
        print ('Accuracy of the model {}'.format(float(correct)/float(total)))

            
if __name__ == '__main__':

    args = get_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feature_columns = get_columns(args)
    
    transformer = get_transformer()

    train_set = KeywordLoader(args.train_path, args.class_num, feature_columns,
                              args.target_column, transformer)

    test_set = KeywordLoader(args.test_path, args.class_num, feature_columns,
                             args.target_column, transformer)

    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, args.batch_size, shuffle=True, drop_last=False)

    network = Network(1531, 12, 2)
    model = Model(network, train_loader, test_loader, device)

    for e in range(args.num_epochs):
        model.train_one_epoch()
        model.test()
