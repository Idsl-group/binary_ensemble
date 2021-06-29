import numpy as np
import matplotlib.pyplot as plt
from k_binary_models import KBinaryModels
from cgan_models import CGAN
import torch
import random
from sklearn.metrics import f1_score

class BinaryEnsemble():
    def __init__(self, input_size, cgan, num_classes=3, k=2, lr=3e-4, weight_decay=0.05):
        self.num_classes = num_classes
        self.models = []
        self.cgan = cgan
        for _ in range(num_classes):
            self.models.append(KBinaryModels(input_size, k=k, lr=lr, weight_decay=weight_decay))

    def train(self, X, y, X_test, y_test, batch_size=64,
              num_epochs=10000):

        train_err = []
        print("Training CGAN")
        all_data = []
        for i in range(self.num_classes):
            #  print("This is for class "+str(i))
            X_train = X.copy()
            y_train = y.copy()
            y_train[y==i] = 1
            y_train[y!=i] = 0
            #  counts = np.bincount(y_train)
            #  #  print("bincount before generating = " + str(counts))
            #  toGen = len(y_train[y_train == 0]) - len(y_train[y_train == 1])
            #  #  print("To Gen "+str(toGen))
            #  if toGen > 0:
                #  #  print("Generating ones")
                #  noise = torch.randn(toGen, 64)
                #  labels = np.zeros(toGen)
                #  labels += i
                #  #  print("Labels when 0 is majority")
                #  #  print(labels)
                #  labels = torch.IntTensor(labels)
                #  new = self.cgan.gen.forward(noise, labels)
                #  new = new.detach().numpy()
                #  X_train = np.concatenate((X_train, new))
                #  y_train = np.concatenate((y_train, np.ones(toGen).astype(int)))
                #  #  print("Bincount after Gen: "+str(np.bincount(y_train)))
            #  elif toGen < 0:
                #  #  print("Generating zeros")
                #  toGen = -toGen
                #  noise = torch.randn(toGen, 64)
                #  labels = np.zeros(toGen)
                #  choose = []
                #  for l in range(self.num_classes):
                    #  if l != i:
                        #  choose.append(l)
                #  for l in range(len(labels)):
                    #  labels[l] = random.choice(choose)
                #  #  print("Labels when 0 is majority")
                #  #  print(labels)
                #  labels = torch.IntTensor(labels)
                #  new = self.cgan.gen.forward(noise, labels)
                #  new = new.detach().numpy()
                #  X_train = np.concatenate((X_train, new))
                #  y_train = np.concatenate((y_train, np.zeros(toGen).astype(int)))

                #  #  print("Bincount after Gen: "+str(np.bincount(y_train)))
                                         
            Xn, _ = X_train.shape
            Yn = y_train.shape[0]
            assert(Xn == Yn)
            ind = np.random.choice(range(Xn), Xn, replace=False)
            X_train = X_train[ind]
            y_train = y_train[ind]
            all_data.append((X_train, y_train))

        train_accs = []
        test_accs = []
        f1_scores = []
        print("Training Models")
        for epoch in range(num_epochs):
            for i in range(self.num_classes):
                curr_X_train, curr_y_train = all_data[i]
                run = 1
                if i != 1:
                    run = 10
                self.models[i].train(curr_X_train, curr_y_train, num_epochs=run)
            y_pred = self.predict(X)
            train_accs.append((y_pred == y).mean())
            y_pred = self.predict(X_test)
            test_accs.append((y_pred == y_test).mean())
            f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
            print(epoch)
        plt.plot(range(len(train_accs)), train_accs, label='Train')
        plt.plot(range(len(train_accs)), test_accs, label='Test')
        plt.plot(range(len(train_accs)), f1_scores, label='F1')
        plt.legend(loc='best')
        #  plt.show()


    def predict(self, X):
        probs = []
        for model in self.models:
            probs.append(model.predict(X))
        probs = np.array(probs)
        return probs.argmax(axis=0)
