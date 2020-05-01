import argparse
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import Variable, transforms
from torch.optim import Adam

from make_data import PrepareData


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', type=str, default='')
    parser.add_argument('--test_path', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=5)
    
    args = parser.parse_args()

    return args


def get_transformer():
    transformer_list =[]
    transformer_list.append(transforms.ToTensor())
    #transformer_list.append()
    transformer = transforms.Compose(transformer_list)
    return transformer


class KeywordLoader(Dataset):

    def __init__(self, data_path, class_num):

        dataPrep = PrepareData(data_path)
        self.data, label = dataPrep.return_features(dataPrep.keyword_columns,
                                                         'SEX')

        self.label = self.make_onehot(label, class_num)
        
    def __getitem__(self, idx):

        return self.data[idx], self.label[idx]

    def len(self):

        return len(self.data)

    def make_onehot(self, label_data, dimension):

        return np.eye(dimension)[label_data]

    
class Network(nn.Module):

    def __init__(self, input_dim, middle_dim, out_dim):

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
        self.softmax = nn.Softmax()
        
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

        self.criterion = nn.CrossEntorpyLoss()
        self.optimizer = Adam(self.network.parameters(), lr=0.001, betas=(0.9, 0.999))

        self.device = device
        
    def train_one_epoch(self, num_epochs):

        self.network.train()
        
        for data, label in self.train_loader:
            
            out = model(data)
            loss = self.criterion(out, label)
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def test(self):

        self.network.eval()
        correct = 0
        total = 0
        
        for data, label in self.test_loader:

            out = self.network(data)
            _, predicted = torch.max(out, 1)

            total += label.size(0)
            correct += (predicted == labels).sum().item()
            
        print ('Accuracy of the model %f'.format(float(correct)/float(total)))

            
if __name__ == '__main__':

    args = get_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_set = KeywordLoader()
