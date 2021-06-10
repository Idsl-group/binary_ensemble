import numpy as np

def cross_validate(model, X, y, k=5):
    n, d = X.shape
    Xsplits = []
    ysplits = []


def train_and_test(model, Xtrain, ytrain, Xtest, ytest, k=3):
    n, _ = Xtrain.shape
    ind = np.random.choice(range(n), size=n, replace=False)
    Xtrain = Xtrain[ind]
    ytrain = ytrain[ind]
    n, _ = Xtest.shape
    ind = np.random.choice(range(n), size=n, replace=False)
    Xtest = Xtest[ind]
    ytest = ytest[ind]

    X_test_splits = np.array_split(Xtest, k)
    y_test_splits = np.array_split(ytest, k)
    train_score = []
    test_score = []
    for i in range(k):
        Xcurr_test = X_test_splits[i]
        ycurr_test = y_test_splits[i]
        X = Xtrain.copy()
        y = ytrain.copy()
        for j in range(k):
            if i == j:
                continue
            X = np.concatenate((X, X_test_splits[j]))
            y = np.concatenate((y, y_test_splits[j]))
        n, _ = X.shape
        ind = np.random.choice(range(n), size=n, replace=False)
        X = X[ind]
        y = y[ind]
        model.fit(X, y)
        y_pred = model.predict(X)
        train_score.append(np.mean(y_pred == y))
        y_test_pred = model.predict(Xcurr_test)
        test_score.append(np.mean(y_test_pred == ycurr_test))
    return {'train': train_score, 'test': test_score}
