import pandas as pd
from torch import nn
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class EmailDataset(Dataset):
    def __init__(self, path, split='train'):
        # Only allowed values for split are 'train' and 'test'
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
        # Extract authors
        # predicted_class_index = self.labels[[0]].argmax(dim=1).item()
        # predicted_class_label = label_encoder.inverse_transform([predicted_class_index])


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class Model(nn.Module):
    # A simple perceptron with no non-linearities other than thhe logsfotmax at the end. 
    # Takes the representation of each text as input and outputs a distribution (via logsoftmax) of probabilities over each possible author that the given author wrote that email
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        return self.logsoftmax(self.linear(x))
    

dataset = EmailDataset('/home/gusmacjob@GU.GU.SE/statistical-nlp/Assignment3/lt2222-v23-a3/emails.csv', split='test')
print(len(dataset))
print(dataset.labels)
print(dataset.features)