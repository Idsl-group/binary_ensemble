from utils import process_wearable_dataset
from sklearn.metrics import f1_score
from gen_class import GenClass
from sklearn.model_selection import train_test_split
from cgan_models import CGAN
import numpy as np
from sklearn.model_selection import train_test_split


X, y = process_wearable_dataset()

n, d = X.shape
counts = np.bincount(y)
argmin = np.argmin(counts)
argmax = np.argmax(counts)
mcir = counts[argmin]/counts[argmax] * counts[argmin]/counts[2-argmin]

num_epoch = 200

ind = np.arange(n)
ind_split = np.array_split(ind, 5)
cgan_f1 = []
cgan_test = []
cgan_epoch = []
gen_f1 = []
gen_test = []
gen_epoch = []



#  for i in range(5):
    #  X_train = np.empty((0, d))
    #  y_train = np.empty((0,)).astype(int)
    #  X_test = X[ind_split[i]]
    #  y_test = y[ind_split[i]]

    #  for j in range(5):
        #  if i == j:
            #  continue
        #  X_train = np.concatenate((X_train, X[ind_split[j]]))
        #  y_train = np.concatenate((y_train, y[ind_split[j]]))

    #  f, t, e = 0, 0, 0

    #  cgan = CGAN(d, num_classes=3)
    #  f, t, e = cgan.train(X_train, y_train, X_test,
               #  y_test, num_epoch=num_epoch)
    #  cgan_f1.append(f)
    #  cgan_test.append(t)
    #  cgan_epoch.append(e)

    #  f, t, e = 0, 0, 0

    #  gen_class = GenClass(d, num_classes=3)
    #  f, t, e = gen_class.train(X_train,
                                        #  y_train, X_test,
                                        #  y_test,
                                        #  num_epoch=num_epoch)
    #  gen_f1.append(f)
    #  gen_test.append(t)
    #  gen_epoch.append(e)
    #  print('****************************')
    #  print(str(i+1)+' Iteration of cross validation complete')


#  print('***************************')
#  print('MCIR: '+str(mcir))
#  print('***************************')
#  print('Gen Class')
#  print('F1 Score: ' + str(np.mean(gen_f1)))
#  print('Test Accuracy: ' + str(np.mean(gen_test)))
#  print('Epochs Required: ' + str(np.mean(gen_epoch)))
#  print('***************************')
#  print('CGAN')
#  print('F1 Score: ' + str(np.mean(cgan_f1)))
#  print('Test Accuracy: ' + str(np.mean(cgan_test)))
#  print('Epochs Required: ' + str(np.mean(cgan_epoch)))
