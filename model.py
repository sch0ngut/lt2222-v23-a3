import pandas as pd
from torch import nn
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import LabelEncoder

class EmailDataset(Dataset):
    def __init__(self, path, split='train'):
        assert split in ['train', 'test']
        self.df = pd.read_csv(path)
        if split == 'train':
            self.df = self.df[self.df['train']]
        if split == 'test':
            self.df = self.df[~self.df['train']]
        label_encoder = LabelEncoder()
        label_encoded = label_encoder.fit_transform(self.df.iloc[:, -2])
        self.labels = torch.tensor(label_encoded, dtype=torch.long)
        self.features = torch.tensor(self.df.iloc[:, :-2].to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class Model(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, nonlinearity):
        super(Model, self).__init__()
        assert nonlinearity in ['tanh', 'relu', 'none']
        self.hidden_size = hidden_size
        self.nonlinearities = {'tanh': torch.tanh, 'relu': torch.relu, 'none': lambda x: x}
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.nonlinearity = self.nonlinearities[nonlinearity]

    def forward(self, x):
        hidden_output = self.nonlinearity(self.hidden(x))
        return self.logsoftmax(self.output(hidden_output))
    