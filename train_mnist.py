from utils import mnist_imbalanced
from models import Classifier
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import f1_score

X_train, y_train, X_test, y_test = mnist_imbalanced()

n, d = X_train.shape
model = Classifier(d, 3)
optim = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
num_epochs = 20
batch_size = 64
print(np.bincount(y_train))
ind = np.arange(n)
num_splits = n/batch_size
ind_splits = np.array_split(ind, num_splits)

for epoch in range(num_epochs):
    for it in range(len(ind_splits)):
        i = ind_splits[it]
        currX = torch.FloatTensor(X_train[i])
        curry = torch.LongTensor(y_train[i])
        optim.zero_grad()
        output = model(currX)
        loss = criterion(output, curry)
        loss.backward()
        optim.step()
    print(epoch)

with torch.no_grad():
    X = torch.FloatTensor(X_test)
    y = torch.IntTensor(y_test)
    output = torch.argmax(model(X).detach(), dim=1).numpy()
    print((output == y_test).mean())
    print(f1_score(y_test, output, average='weighted'))
