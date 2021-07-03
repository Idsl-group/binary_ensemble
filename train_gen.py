from sklearn.model_selection import train_test_split
import numpy as np
from utils import process_wearable_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from gen_class import GenClass

X, y = process_wearable_dataset()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)

n, d = X_train.shape

X = 0
y = 0
gen_class = GenClass(d, num_classes=3)
gen_class.train(X_train, y_train, X_test, y_test, num_epoch=100)

#  print('Random Forest without Gen')
#  model = RandomForestClassifier()
#  model.fit(X_train, y_train)
#  y_pred = model.predict(X_test)
#  print(np.bincount(y_train))
#  print('Testing Accuracy')
#  print((y_pred == y_test).mean())
#  print('F1 Score')
#  print(f1_score(y_test, y_pred, average='weighted'))

#  train_loss = []
#  test_loss = []
#  f1_scores = []
#  for epoch in range(400):
    #  print(epoch)
    #  gen_class = GenClass(d, num_classes=3)
    #  gen_class.train(X_train, y_train, num_epoch=epoch)

    #  counts = np.bincount(y_train)
    #  maxCount = np.max(counts)
    #  for i in range(len(counts)):
        #  toGen = maxCount - counts[i]
        #  if toGen == 0:
            #  continue
        #  noise = torch.randn(toGen, 64)
        #  selFrom = X_train[y_train == i]
        #  selN, _ = selFrom.shape
        #  X_comp = selFrom[np.random.choice(selN, size=toGen, replace=True)]
        #  X_comp = torch.FloatTensor(X_comp)
        #  new = gen_class.gen(noise, X_comp).detach().numpy()
        #  labels = np.zeros((toGen)).astype(int) + i
        #  X_train = np.concatenate((X_train, new))
        #  y_train = np.concatenate((y_train, labels))

    #  model = RandomForestClassifier()
    #  model.fit(X_train, y_train)
    #  y_pred = model.predict(X_test)
    #  print(np.bincount(y_train))
    #  test_loss.append((y_pred == y_test).mean())
    #  f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

#  plt.plot(range(len(test_loss)), test_loss, label='Test Accuracy')
#  plt.plot(range(len(f1_scores)), f1_scores, label='F1 Scores')
#  plt.legend(loc='best')
#  plt.show()
#  print(max(f1_scores))

