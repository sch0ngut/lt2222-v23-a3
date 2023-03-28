import argparse
import pandas as pd
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from model import EmailDataset, Model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("--featurefile", type=str, default='emails.csv', help="The file containing the table of instances and features.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for.")
    parser.add_argument("--n_hidden", type=int, default=100, help="Number of hidden units.")
    parser.add_argument("--nonlinearity", type=str, default='tanh', help="Nonlinearity to use.")
    
    args = parser.parse_args()

    print("Reading {}...".format(args.featurefile))
    # Read the data
    train = EmailDataset(args.featurefile, split='train')
    test = EmailDataset(args.featurefile, split='test')
    train_dataloader = DataLoader(train, batch_size=32, shuffle=True)

    # Initialize the model
    n_features = train.features.shape[1]
    n_labels = len(train.labels.unique())
    model = Model(n_features, n_labels, args.n_hidden, args.nonlinearity)
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

    # Test the model on the test set
    test_dataloader = DataLoader(test, batch_size=len(test), shuffle=True)
    test_features, test_labels = iter(test_dataloader).next()
    predictions = model(test_features).argmax(dim=1)
    # Compute the accuracy
    accuracy = (predictions == test_labels).numpy().mean()
    print("Accuracy: {}".format(accuracy))
    # Print the confusion matrix
    print(pd.crosstab(pd.Series(predictions.numpy(), name='Predicted'), pd.Series(test_labels, name='Actual')))



    

    
