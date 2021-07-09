import torch
import torch.nn as nn
import numpy as np

class Ann(nn.Module):
    def __init__(self, input_size):
        super(Ann, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, X):
        return self.model(X)

class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, X):
        return self.model(X)
    
    def fit(self, X_train, y_train, batch_size=64, num_epochs=20):
        n, d = X_train.shape
        optim = torch.optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        print(np.bincount(y_train))
        ind = np.arange(n)
        num_splits = int(n/batch_size) + 1
        ind_splits = np.array_split(ind, num_splits)

        for epoch in range(num_epochs):
            for it in range(len(ind_splits)):
                i = ind_splits[it]
                currX = torch.FloatTensor(X_train[i])
                curry = torch.LongTensor(y_train[i])
                optim.zero_grad()
                output = self(currX)
                loss = criterion(output, curry)
                loss.backward()
                optim.step()

    def predict(self, X):
        with torch.no_grad():
            X = torch.FloatTensor(X)
            output = torch.argmax(self(X).detach(), dim=1).numpy()
            return output

class Generator(nn.Module):
    def __init__(self, input_size, data_length, output_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(data_length+input_size, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, output_size),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        inp = torch.cat([x1, x2], dim=1)
        return self.model(inp)

