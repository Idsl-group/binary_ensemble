import numpy as np
import pandas
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import numpy as np
from sklearn.model_selection import train_test_split

def process_occutherm(class0, class1):
    train_url = 'https://raw.githubusercontent.com/Idsl-group/binary_ensemble/main/data/occutherm_train.csv'
    test_url = 'https://raw.githubusercontent.com/Idsl-group/binary_ensemble/main/data/occutherm_test.csv'
    train_dat = pandas.read_csv(train_url)
    n, d = train_dat.shape
    cols = train_dat.columns.to_numpy().astype(float)

    train_dat = np.concatenate((cols.reshape(1,d), train_dat.to_numpy()))

    X_train = train_dat[:, :-1]
    y_train = train_dat[:, -1].astype(int) + 1
    

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    print(X_train.shape)

    test_dat = pandas.read_csv(test_url)
    n, d = test_dat.shape
    cols = test_dat.columns.to_numpy().astype(float)

    test_dat = np.concatenate((cols.reshape(1,d), test_dat.to_numpy()))

    X_test = test_dat[:, :-1]
    y_test = test_dat[:, -1].astype(int) + 1
    

    scaler = MinMaxScaler()
    X_test = scaler.fit_transform(X_test)

    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    n, d = X.shape
    ind = np.random.choice(range(n), n, replace=False)
    X = X[ind]
    y = y[ind]
    print(np.bincount(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)

    class0 = min(class0, np.bincount(y_train)[0])
    majority = np.bincount(y_train)[1]
    class1 = min(class1, np.bincount(y_train)[2])

    X_sel = X_train[y_train == 0]
    n, _ = X_sel.shape
    size = class0
    ind = np.random.choice(range(n), size=size, replace=False)
    X_new = X_sel[ind]
    y_new = np.zeros(size).astype(int)

    X_sel = X_train[y_train == 1]
    n, _ = X_sel.shape
    size = majority
    ind = np.random.choice(range(n), size=size, replace=False)
    new = X_sel[ind]
    X_new = np.concatenate((X_new, new))
    new = np.zeros(size).astype(int) + 1
    y_new = np.concatenate((y_new, new))

    X_sel = X_train[y_train == 2]
    n, _ = X_sel.shape
    size = class1
    ind = np.random.choice(range(n), size=size, replace=False)
    new = X_sel[ind]
    X_new = np.concatenate((X_new, new))
    new = np.zeros(size).astype(int) + 2
    y_new = np.concatenate((y_new, new))


    n, d = X_new.shape
    indices = np.random.choice(range(n), n, replace=False)
    X_new = X_new[indices]
    y_new = y_new[indices]

    print(np.bincount(y_new))
    print(np.bincount(y_test))

    mcir = class1/class0 * class1/majority
    print(mcir)

    return X_new, y_new, X_test, y_test, mcir

