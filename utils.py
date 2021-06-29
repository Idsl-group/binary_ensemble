import numpy as np
import pandas
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

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
    return X, y
