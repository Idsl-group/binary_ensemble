import numpy as np
import pandas
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from ensemble import Ensemble
from single_class import Single_Class


def transform_one_hot(data, ind, labels):
    n, d = data.shape
    one_hot = np.zeros((n, len(labels)))
    for i in range(len(data[:,ind])):
        for j in range(len(labels)):
            if data[i, ind] == labels[j]:
                one_hot[i,j] = 1
                break

    new_data = np.concatenate((data[:,0:ind].reshape((n,ind)), one_hot, 
                                data[:,ind+1:]), axis = 1)
    return new_data

def process(dat):
    new_data = dat[['ID', 'Sex',
                    'Age', 'Height', 'location',
                    'Weight',
                    'mean.Temperature_60',
                    'grad.Temperature_60',
                    'sd.Temperature_60',
                    'mean.Temperature_480',
                    'grad.Temperature_480',
                    'sd.Temperature_480',
                    'mean.Humidity_60',
                    'grad.Humidity_60',
                    'sd.Humidity_60',
                    'mean.Humidity_480',
                    'grad.Humidity_480',
                    'sd.Humidity_480',
                    'mean.Solar_60',
                    'grad.Solar_60',
                    'sd.Solar_60',
                    'mean.Solar_480',
                    'grad.Solar_480',
                    'sd.Solar_480',
                    'mean.hr_5',
                    'grad.hr_5',
                    'sd.hr_5',
                    'mean.hr_15',
                    'grad.hr_15',
                    'sd.hr_15',
                    'mean.hr_60',
                    'grad.hr_60',
                    'sd.hr_60',
                    'mean.WristT_5',
                    'grad.WristT_5',
                    'sd.WristT_5',
                    'mean.WristT_15',
                    'grad.WristT_15',
                    'sd.WristT_15',
                    'mean.WristT_60',
                    'grad.WristT_60',
                    'sd.WristT_60',
                    'mean.AnkleT_5',
                    'grad.AnkleT_5',
                    'sd.AnkleT_5',
                    'mean.AnkleT_15',
                    'grad.AnkleT_15',
                    'sd.AnkleT_15',
                    'mean.AnkleT_60',
                    'grad.AnkleT_60',
                    'sd.AnkleT_60']]

    new_data = np.array(new_data)

    new_data = transform_one_hot(new_data, 4, [-1,1])
    new_data = transform_one_hot(new_data, 1, ['Male', 'Female'])


    # Indices in new data that does not
    # contain nan in its row. If false
    # then new data at the same index contains nan
    # in its row


    print('KNN Imputer')

    imputer = KNNImputer(n_neighbors=10)
    new_data = imputer.fit_transform(new_data)

    y = dat['therm_pref']
    y += 1
    X = np.array(new_data, dtype=float)
    y = np.array(y, dtype=int)
    return X, y

dat = pandas.read_csv('./data/wearable.csv')
X, y = process(dat)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)
n, d = X_train.shape

model = Single_Class(input_size=d, label=1)

model.train(X_train, y_train,
           num_epochs=10)

y_test[y_test!=1]=0
y_train[y_train!=1]=0

y_pred = model.predict(X_test)
print("************")
print("Accuracy:")
print(np.mean(y_pred == y_test))
