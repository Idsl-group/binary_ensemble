from ann import Ann
import torch
import numpy as np
import torch.nn as nn

class Single_Class():
    def __init__(self, input_size, label, lr=1e-4, k=2):
        self.models = []
        self.label = label
        self.k = k
        self.lr = lr
        for _ in range(k):
            model = Ann(input_size)
            self.models.append(model)

    def train(self, X_train, y_train, batch_size=64,
              num_epochs=10000, random_bag=False):
        n, _ = X_train.shape
        y = y_train.copy()
        X = X_train.copy()
        y[y_train==self.label] = 1
        y[y_train!=self.label] = 0
        num_splits = n/batch_size

        if not num_splits % 1 == 0:
            num_splits = int(num_splits) + 1
        else:
            num_splits = int(num_splits)

        Xtrain_split = np.array_split(X, num_splits)
        ytrain_split = np.array_split(y, num_splits)

        for model in self.models:
            optim = torch.optim.Adam(model.parameters(),
                                 lr=self.lr)
            criterion = nn.BCEWithLogitsLoss()
            for epoch in range(num_epochs):
                for i in range(len(Xtrain_split)):
                    currX = torch.FloatTensor(Xtrain_split[i])
                    curry = torch.FloatTensor(ytrain_split[i]).unsqueeze(1)
                    model.zero_grad()
                    y_pred = model(currX)
                    loss = criterion(y_pred, curry)
                    loss.backward()
                    optim.step()

    def predict(self, X):
        pred = []
        X = torch.FloatTensor(X)
        with torch.no_grad():
            for i in range(self.k):
                y_pred = self.models[i](X).detach().numpy()
                pred.append(y_pred)
        means = np.mean(pred, axis=0)
        means[means > 0.5] = 1
        means[means<=0.5] = 0
        return means

