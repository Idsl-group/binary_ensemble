import torch
import numpy as np
import torch.nn as nn
from model import Generator, Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

class GenClass():
    def __init__(self, input_size, num_classes=10, lr=0.001):
        zdim = 64
        self.gen = Generator(zdim, input_size, input_size)
        self.classifier = Classifier(input_size, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        self.gen_optim = torch.optim.Adam(self.gen.parameters(), lr=lr)
        self.gen_criterion = nn.CrossEntropyLoss()
        self.gen_mse = nn.MSELoss()

    def train(self, X_train, y_train, X_test, y_test, num_epoch=200, batch_size=64):
        zdim = 64
        n, d = X_train.shape
        ind = np.arange(n)
        num_splits = n/batch_size
        ind_splits = np.array_split(ind, num_splits)
        test_losses = []
        f1_scores = []
        losses = []
        for epoch in range(num_epoch):
            for it in range(len(ind_splits)):
                i = ind_splits[it]
                currX = torch.FloatTensor(X_train[i])
                curry = torch.LongTensor(y_train[i])
                self.optim.zero_grad()
                output = self.classifier(currX)
                loss = self.criterion(output, curry)
                loss.backward()
                self.optim.step()

                if (it+1) % 1 == 0:
                    noise = torch.randn(batch_size, zdim)
                    sel = np.random.choice(X_train.shape[0], size=batch_size, replace=False)
                    X_comp = torch.FloatTensor(X_train[sel])
                    y_comp = torch.LongTensor(y_train[sel])
                    self.gen_optim.zero_grad()
                    generated_data = self.gen(noise, X_comp)
                    sim_output = self.classifier(generated_data)
                    gen_loss = self.gen_criterion(sim_output, y_comp)
                    gen_loss += 10*self.gen_mse(generated_data, X_comp)
                    gen_loss.backward()
                    self.gen_optim.step()

            with torch.no_grad():
                noise = torch.randn(X_test.shape[0], zdim)
                sel = np.random.choice(X_test.shape[0], size=X_test.shape[0], replace=False)
                X_comp = torch.FloatTensor(X_train[sel])
                generated_data = self.gen(noise, X_comp)
                loss = self.gen_mse(generated_data, X_comp)
                losses.append(loss)


            counts = np.bincount(y_train)
            maxCount = np.max(counts)
            X_train_new = X_train.copy()
            y_train_new = y_train.copy()
            for i in range(len(counts)):
                toGen = maxCount - counts[i]
                if toGen == 0:
                    continue
                noise = torch.randn(toGen, 64)
                selFrom = X_train[y_train == i]
                selN, _ = selFrom.shape
                X_comp = selFrom[np.random.choice(selN, size=toGen, replace=True)]
                X_comp = torch.FloatTensor(X_comp)
                new = self.gen(noise, X_comp).detach().numpy()
                labels = np.zeros((toGen)).astype(int) + i
                X_train_new = np.concatenate((X_train_new, new))
                y_train_new = np.concatenate((y_train_new, labels))

            model = KNeighborsClassifier()
            model.fit(X_train_new, y_train_new)
            y_pred = model.predict(X_test)
            test_losses.append((y_pred == y_test).mean())
            f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
            print(epoch)

        plt.plot(range(len(losses)), losses)
        plt.show()
        plt.plot(range(len(test_losses)), test_losses, label='Test Accuracy')
        plt.plot(range(len(f1_scores)), f1_scores, label='F1 Scores')
        print('F1 Score ******************')
        print(max(f1_scores))
        print(np.argmax(np.array(f1_scores)))
        print('Test Acc *********************')
        print(test_losses[np.argmax(np.array(f1_scores))])
        plt.legend(loc='best')
        plt.show()
