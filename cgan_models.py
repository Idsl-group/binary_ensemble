import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from models import Classifier
from sklearn.svm import SVC


class Discriminator(nn.Module):
    def __init__(self, input_size, num_classes=2):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        self.embed = nn.Embedding(num_classes, num_classes)
        self.main = nn.Sequential(
            #  nn.Linear(input_size+num_classes, 128),
            #  nn.LeakyReLU(0.1),
            #  nn.Linear(128, 64),
            #  nn.LeakyReLU(0.1),
            #  nn.Linear(64, 32),
            #  nn.LeakyReLU(0.1),
            #  nn.Linear(32, 1),
            nn.Linear(input_size+num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x, labels=0):
        new_labels = self.embed(labels)
        inp = torch.cat([x, new_labels], dim=1)
        return self.main(inp)


class Generator(nn.Module):
    def __init__(self, input_size, num_classes, output_size):
        super(Generator, self).__init__()
        self.num_classes = num_classes
        self.embed = nn.Embedding(num_classes, num_classes)
        self.main = nn.Sequential(
            #  nn.Linear(input_size+num_classes, 128),
            #  nn.LeakyReLU(0.1),
            #  nn.Linear(128, 64),
            #  nn.LeakyReLU(0.1),
            #  nn.Linear(64, output_size),
            #  nn.Sigmoid()
            nn.Linear(num_classes+input_size, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, output_size),
            nn.Sigmoid()
        )

    def forward(self, x, labels=0):
        new_labels = self.embed(labels)
        inp = torch.cat([x, new_labels], dim=1)
        return self.main(inp)


def grad_pen(disc, real, generated, labels):
    batch_size, n = real.shape
    eps = torch.rand((batch_size, 1)).repeat(1, n)
    interpolate = (real * eps + generated * (1-eps)).requires_grad_(True)
    output = disc(interpolate, labels)
    gradient = torch.autograd.grad(
          inputs=interpolate,
          outputs=output,
          grad_outputs=torch.ones_like(output),
          retain_graph=True,
          create_graph=True,
          only_inputs=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    norm = gradient.norm(2, dim=1)
    pen = torch.mean((norm-1)**2)
    return pen


class CGAN():
    def __init__(self, input_size, num_classes, lr=1e-4, zdim=64,
                ncrit=5):
        self.disc = Discriminator(input_size, num_classes)
        self.gen = Generator(zdim, num_classes, input_size)
        self.gen_optimizer = torch.optim.Adam(self.gen.parameters(), lr=lr,
                                              betas=(0.0, 0.9))
        self.disc_optimizer = torch.optim.Adam(self.disc.parameters(), lr=lr,
                                               betas=(0.0, 0.9))
        self.loss = nn.BCELoss()
        self.ncrit = ncrit
        self.zdim = zdim
        self.num_classes = num_classes

    def train(self, X_train, y_train, X_test, y_test, batch_size=64, num_epoch=1000):
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
        gen_errs = []
        epochs = []
        train_errs = []
        test_errs = []
        f1 = []
        for epoch in range(num_epoch):
            for i in range(len(Xtrain_split)):
                currX = torch.FloatTensor(Xtrain_split[i])
                curry = torch.IntTensor(ytrain_split[i])
                batch_size = (len(currX))

                noise = torch.randn(batch_size, self.zdim)

                generated_data = self.gen(noise, curry)
                disc_real_data = self.disc(currX, curry).view(-1)
                disc_gen_data = self.disc(generated_data, curry).view(-1)

                pen = grad_pen(self.disc, currX, generated_data, curry)
                disc_loss = -(torch.mean(disc_real_data) - torch.mean(
                    disc_gen_data)) + 10 * pen

                self.disc.zero_grad()
                disc_loss.backward()
                self.disc_optimizer.step()
                # We want this to be max for good gen
                gen_errs.append(-torch.mean(disc_gen_data).detach().numpy())
                epochs.append(epoch)

                if i % self.ncrit == 0:
                    noise = torch.randn(batch_size, self.zdim)
                    gen_labels = torch.randint(0, self.num_classes, (batch_size,))
                    generated_data = self.gen(noise, gen_labels)
                    disc_output = self.disc(generated_data, gen_labels).view(-1)
                    gen_loss = -torch.mean(disc_output)
                    self.gen.zero_grad()
                    gen_loss.backward()
                    self.gen_optimizer.step()

            t, f = self.getErrs(X_train, y_train, X_test, y_test)
            test_errs.append(t)
            f1.append(f)
            if (epoch + 1) % 25 == 0:
                print('Epoch: '+str(epoch+1))
        #  print('*************************************')
        #  print('CGAN')
        #  print('F1 Score: '+str(np.max(f1)))
        #  print('Test Accuracy: '+str(test_errs[np.argmax(f1)]))
        #  print('Epochs Required: '+str(np.argmax(f1)))
        f1_max = np.max(f1)
        test_for_f1 = test_errs[np.argmax(f1)]
        epoch_no = np.argmax(f1)
        return f1_max, test_for_f1, epoch_no

    def getErrs(self, X, y, X_test, y_test):
        X_train = X.copy()
        y_train = y.copy()
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
            new = self.gen.forward(noise, labels)
            new = new.detach().numpy()
            labels = labels.numpy()
            X_train = np.concatenate((X_train, new))
            y_train = np.concatenate((y_train, labels))

        n, d = X_train.shape
        #  model = Classifier(d, 3)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_err = (y_pred == y_test).mean()
        f = f1_score(y_test, y_pred, average='weighted')
        return test_err, f

