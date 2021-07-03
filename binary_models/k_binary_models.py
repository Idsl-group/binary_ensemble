import numpy as np
from binary_model import BinaryModel


class KBinaryModels():
    def __init__(self, input_size, k=2, lr=3e-4, weight_decay=0.05):
        self.models = []
        for _ in range(k):
            self.models.append(BinaryModel(input_size, lr=lr, weight_decay=weight_decay))

    def train(self, X_train, y_train, batch_size=64,
              num_epochs=10000):

        for epoch in range(num_epochs):
            for model in self.models:
                model.train(X_train, y_train, batch_size,
                  num_epochs=1)
            
    def predict(self, X):
        probs = []



        for model in self.models:
            probs.append(model.predict(X))
        probs = np.array(probs)
        return probs.mean(axis=0)
