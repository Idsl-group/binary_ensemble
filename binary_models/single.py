from k_binary_models import KBinaryModels
from utils import process_wearable_dataset
from cgan_models import CGAN
from sklearn.model_selection import train_test_split
import numpy as np
import random
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

X, y = process_wearable_dataset()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)

n, d = X_train.shape

model = KBinaryModels(d)
y_copy = y_train.copy()
y_copy[y_train == 1] = 0
y_copy[y_train != 1] = 1
model.train(X_train, y_copy, num_epochs=1000)
y_pred = model.predict(X_train).round()
print((y_pred != y_copy).mean())

cgan = CGAN(d, num_classes=3)
cgan.train(X_train, y_train, num_epoch=200)

model = KBinaryModels(d)
y_new = y_train.copy()
X_new = X_train.copy()
y_train[y_new == 1] = 0
y_train[y_new != 1] = 1

print("*************************")
print('Random Forest')
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_test_copy = y_test.copy()
y_test[y_test_copy != 1] = 1
y_test[y_test_copy == 1] = 0
print(f1_score(y_test, y_pred))

counts = np.bincount(y_train)
toGen = counts[0] - counts[1]
noise = torch.randn(toGen, 64)
labels = np.zeros(toGen)
for l in range(len(labels)):
    labels[l] = random.choice([0,2])

print(labels)
labels = torch.IntTensor(labels)
new = cgan.gen.forward(noise, labels)
new = new.detach().numpy()
X_train = np.concatenate((X_train, new))
y_train = np.concatenate((y_train, np.ones(toGen).astype(int)))
n, d = X_train.shape
ind = np.random.choice(range(n), n, replace=False)
X_train = X_train[ind]
y_train = y_train[ind]
print("*************************")
print('Random Forest after gen')
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_test_copy = y_test.copy()
y_test[y_test_copy != 1] = 1
y_test[y_test_copy == 1] = 0
print(f1_score(y_test, y_pred))
print(np.bincount(y_train))
model = KBinaryModels(d)
model.train(X_train, y_train, num_epochs=1000)

print("***************************")
y_pred = model.predict(X_test).round()
print(f1_score(y_test, y_pred))

