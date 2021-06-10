from single_class import Single_Class
import numpy as np
import torch

class Ensemble():
    def __init__(self, labels, input_size, k=2, lr=1e-4):
        self.models = []
        for i in labels:
            self.models.append(Single_Class(
                input_size, i, lr, k
            ))

    def train(self, X, y, num_epochs=1000):
        for i in range(num_epochs):
            print(i)
            for j in range(len(self.models)):
                self.models[j].epoch_iter(X, y)

    def predict(self, X):
        y_pred = []
        for i in X:
            ys = []
            run = torch.FloatTensor(i)
            for model in self.models:
                ys.append(model.predict(run))
            y_pred.append(np.argmax(ys))

        return np.array(y_pred)

