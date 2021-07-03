import torch
import torch.nn as nn
import numpy as np
from ann import Ann

class BinaryModel():
    def __init__(self, input_size, lr=3e-4, weight_decay=0.05):
        self.model = Ann(input_size)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr,
                                      weight_decay=weight_decay)
        self.criterion = nn.BCELoss()

    def train(self, X_train, y_train, batch_size=64,
              num_epochs=10000):

        n, _ = X_train.shape
        y = y_train.copy()
        X = X_train.copy()
        num_splits = n/batch_size

        if not num_splits % 1 == 0:
            num_splits = int(num_splits) + 1
        else:
            num_splits = int(num_splits)

        Xtrain_split = np.array_split(X, num_splits)
        ytrain_split = np.array_split(y, num_splits)


        for epoch in range(num_epochs):
            for i in range(len(Xtrain_split)):
                currX = torch.FloatTensor(Xtrain_split[i])
                curry = torch.FloatTensor(ytrain_split[i]).unsqueeze(1)
                #calculate output
                output = self.model(currX)
            
                #calculate loss
                loss = self.criterion(output,curry.reshape(-1,1))
            
                #accuracy
                
                #backprop
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            
                
    def predict(self, X):
        predicted = self.model(torch.tensor(X,dtype=torch.float32))
        return predicted.reshape(-1).detach().cpu().numpy()
