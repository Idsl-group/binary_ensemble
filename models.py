import torch
import torch.nn as nn

class Ann(nn.Module):
    def __init__(self, input_size):
        super(Ann, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, X):
        return self.model(X)

class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, X):
        return self.model(X)

class Generator(nn.Module):
    def __init__(self, input_size, data_length, output_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(data_length+input_size, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, output_size),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        inp = torch.cat([x1, x2], dim=1)
        return self.model(inp)

