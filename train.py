from sklearn.model_selection import train_test_split
import numpy as np
from utils import process_wearable_dataset
from binary_ensemble import BinaryEnsemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from cgan_models import CGAN
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

X, y = process_wearable_dataset()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)

n, d = X_train.shape

X = 0
y = 0

cgan = CGAN(d, num_classes=3)
cgan.train(X_train, y_train, num_epoch=200)


print('Random Forest without CGAN')
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Testing Accuracy')
print((y_pred == y_test).mean())
print('F1 Score')
print(f1_score(y_test, y_pred, average='weighted'))


#  model = RandomForestClassifier()
#  model.fit(X_train, y_train)
#  y_pred = model.predict(X_test)
#  print('RF error before gen')
#  print((y_pred != y_test).mean())
#  print('bincount of RF prediction')
#  print(np.bincount(y_pred))

#  cgan = CGAN(input_size=d, num_classes=3)
#  cgan.train(X_train, y_train, X_test, y_test, num_epoch=1000)
counts = np.bincount(y_train)
maxCount = np.max(counts)
for i in range(len(counts)):
    toGen = maxCount - counts[i]
    if toGen == 0:
        continue
    noise = torch.randn(toGen, 64)
    labels = np.zeros(toGen)
    labels += i
    labels = torch.IntTensor(labels)
    new = cgan.gen.forward(noise, labels)
    new = new.detach().numpy()
    labels = labels.numpy()
    X_train = np.concatenate((X_train, new))
    y_train = np.concatenate((y_train, labels))


print("*******************************")
print('Binary Ensemble Model')
model = BinaryEnsemble(d, cgan)
model.train(X_train, y_train, X_test, y_test, num_epochs=50)
print('Training Accuracy')
y_pred = model.predict(X_train)
print((y_pred == y_train).mean())
print('Testing Accuracy')
y_pred = model.predict(X_test)
print((y_pred == y_test).mean())
print('F1 Score')
print(f1_score(y_test, y_pred, average='weighted'))

print('******************************')
print('Random Forest train prediction with CGAN')
print('Bincount')
print(np.bincount(y_train))
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_train)
print('Train')
print((y_pred == y_train).mean())

print('Test')
y_pred = model.predict(X_test)
print((y_pred == y_test).mean())
print('F1')
print(f1_score(y_test, y_pred, average='weighted'))
