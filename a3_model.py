import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from model import EmailDataset, Model
# Whatever other imports you need

# You can implement classes and helper functions here too.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("--featurefile", type=str, default='emails.csv', help="The file containing the table of instances and features.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for.")
    # Add options here for part 3 -- hidden layer and nonlinearity,
    # and any other options you may think you want/need.  Document
    # everything.
    
    args = parser.parse_args()

    print("Reading {}...".format(args.featurefile))
    # Read the data
    train = EmailDataset(args.featurefile, split='train')
    test = EmailDataset(args.featurefile, split='test')
    train_dataloader = DataLoader(train, batch_size=32, shuffle=True)

    # Initialize the model
    n_features = train.features.shape[1]
    # The number of labels are the unique values in the labels column
    n_labels = len(train.labels.unique())
    breakpoint()
    model = Model(n_features, n_labels)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    loss_function = nn.CrossEntropyLoss()

    # Train the model
    for epoch in range(args.epochs):
        print("Epoch {}".format(epoch))
        for batch_features, batch_labels in train_dataloader:
            output = model(batch_features)
            loss = loss_function(output, batch_labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("Loss: {}".format(loss.item()))


    

    
