import numpy as np
import pandas
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import numpy as np
from sklearn.model_selection import train_test_split

def process_wearable_dataset():
    dat = pandas.read_csv('./data/wearable.csv')
    new_data = dat[['Weight', 'Age',
                    'mean.Temperature_480',
                    'mean.Temperature_60',
                    'mean.Humidity_60',
                    'mean.Humidity_480',
                    'mean.hr_5',
                    'mean.hr_15',
                    'mean.hr_60',
                    'mean.WristT_5',
                    'mean.WristT_15',
                    'mean.WristT_60',
                    'mean.PantT_5', 
                    'mean.PantT_60', 
                    'Height',
                    'Coffeeintake',                   
                    'mean.AnkleT_5',
                    'mean.AnkleT_15',
                    'mean.AnkleT_60',]]

    new_data = np.array(new_data)

    print('KNN Imputer')

    imputer = KNNImputer(n_neighbors=10)
    new_data = imputer.fit_transform(new_data)
    scaler = MinMaxScaler()
    new_data = scaler.fit_transform(new_data)

    y = dat['therm_pref']
    y += 1
    X = np.array(new_data, dtype=float)
    y = np.array(y, dtype=int)

    n, d = X.shape
    ind = np.random.choice(range(n), n, replace=False)
    X = X[ind]
    y = y[ind]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2)


    majority = np.bincount(y_train)[1]
    minority = 100

    X_sel = X_train[y_train == 0]
    n, _ = X_sel.shape
    size = minority
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
    size = minority
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
    return X_new, y_new, X_test, y_test


def mnist_imbalanced():
    X_train = np.load('./parsed/X_train.npy')
    y_train = np.load('./parsed/y_train.npy')
    X_test = np.load('./parsed/X_test.npy')
    y_test = np.load('./parsed/y_test.npy')

    X_sel = X_train[y_train == 0]
    n, _ = X_sel.shape
    size = 10
    ind = np.random.choice(range(n), size=size, replace=False)
    X_train_new = X_sel[ind]
    y_train_new = np.zeros(size).astype(int)

    X_sel = X_train[y_train == 1]
    n, _ = X_sel.shape
    size = 4000
    ind = np.random.choice(range(n), size=size, replace=False)
    new = X_sel[ind]
    X_train_new = np.concatenate((X_train_new, new))
    new = np.zeros(size).astype(int) + 1
    y_train_new = np.concatenate((y_train_new, new))

    X_sel = X_train[y_train == 2]
    n, _ = X_sel.shape
    size = 10
    ind = np.random.choice(range(n), size=size, replace=False)
    new = X_sel[ind]
    X_train_new = np.concatenate((X_train_new, new))
    new = np.zeros(size).astype(int) + 2
    y_train_new = np.concatenate((y_train_new, new))


    X_sel = X_test[y_test == 0]
    n, _ = X_sel.shape
    size = 100
    ind = np.random.choice(range(n), size=size, replace=False)
    X_test_new = X_sel[ind]
    y_test_new = np.zeros(size).astype(int)

    X_sel = X_test[y_test == 1]
    n, _ = X_sel.shape
    size = 100
    ind = np.random.choice(range(n), size=size, replace=False)
    new = X_sel[ind]
    X_test_new = np.concatenate((X_test_new, new))
    new = np.zeros(size).astype(int) + 1
    y_test_new = np.concatenate((y_test_new, new))

    X_sel = X_test[y_test == 2]
    n, _ = X_sel.shape
    size = 100
    ind = np.random.choice(range(n), size=size, replace=False)
    new = X_sel[ind]
    X_test_new = np.concatenate((X_test_new, new))
    new = np.zeros(size).astype(int) + 2
    y_test_new = np.concatenate((y_test_new, new))

    assert(X_train_new.shape[0] == y_train_new.shape[0])
    assert(X_test_new.shape[0] == y_test_new.shape[0])


    n, _ = X_train_new.shape
    ind = np.random.choice(range(n), n, replace=False)
    X_train_new = X_train_new[ind]
    y_train_new = y_train_new[ind]


    n, _ = X_test_new.shape
    ind = np.random.choice(range(n), n, replace=False)
    X_test_new = X_test_new[ind]
    y_test_new = y_test_new[ind]

    return X_train_new, y_train_new, X_test_new, y_test_new


def imaginary_dataset():
    X, y = make_classification(n_samples=12000, n_features=20, n_informative=10, n_classes=3)
    X_new = X.copy()
    y_new = y.copy()
    X_new = X_new[y == 1]
    y_new = y_new[y==1]
    sel = X[y==0]
    n, d = sel.shape
    ind = np.random.choice(range(n), size=600, replace=False)
    X_new = np.append(X_new, X[ind], axis=0)
    y_new = np.append(y_new, np.zeros(600).astype(int))
    sel = X[y==2]
    n, d = sel.shape
    ind = np.random.choice(range(n), size=600, replace=False)
    X_new = np.append(X_new, X[ind], axis=0)
    y_new = np.append(y_new, np.zeros(600).astype(int) + 2)
    n, d = X_new.shape
    ind = np.random.choice(range(n), n, replace=False)
    X_new = X_new[ind]
    y_new = y_new[ind]
    scaler = MinMaxScaler()
    X_new = scaler.fit_transform(X_new)
    X_train, X_test, y_train, y_test = train_test_split(
        X_new, y_new, test_size=0.2)

    return X_train, y_train, X_test, y_test

imaginary_dataset()
