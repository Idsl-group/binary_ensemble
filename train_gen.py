from sklearn.model_selection import train_test_split
import numpy as np
from utils import process_wearable_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from gen_class import GenClass
from utils import mnist_imbalanced, imaginary_dataset
from sklearn.svm import SVC
from cgan_models import CGAN
import time
import logging

#  X_train, y_train, X_test, y_test = mnist_imbalanced()
#  X_train, y_train, X_test, y_test = imaginary_dataset()


#  X_train, X_test, y_train, y_test = train_test_split(
    #  X, y64, test_size=0.2)
logging.basicConfig(level=logging.DEBUG, filename="logfile", filemode="a+",
                        format="%(message)s")
all_mcirs = [10, 100, 200, 300, 450]

for j in all_mcirs:
    print('####################################################################')
    print(j*50)
    all_b = np.zeros((5, 2))
    all_g = np.zeros((5, 3))
    all_c = np.zeros((5, 3))
    imb = []
    for i in range(5):
        start_time = time.time()
        X_train, y_train, X_test, y_test = process_wearable_dataset(j*50)

        n, d = X_train.shape
        imb.append(np.bincount(y_train)[1])

        num_epoch = 200

        X = 0
        y = 0
        
        baseline = RandomForestClassifier()
        baseline.fit(X_train, y_train)
        y_pred = baseline.predict(X_test)
        baseline_test = (y_pred == y_test).mean()
        baseline_f1 = f1_score(y_test, y_pred, average='weighted')
        all_b[i] = np.array([baseline_f1, baseline_test])

        gen_class = GenClass(d, num_classes=3)
        gen_scores = gen_class.train(X_train, y_train, X_test, y_test, num_epoch=num_epoch)
        all_g[i] = np.array(gen_scores)


        cgan = CGAN(d, num_classes=3)
        scores = cgan.train(X_train, y_train, X_test, y_test, num_epoch=num_epoch)
        all_c[i] = np.array(scores)
        
        #  logging.info('***********************')
        #  logging.info(i)
        #  logging.info('Baseline')
        #  logging.info('F1: '+str(baseline_f1))
        #  logging.info('Test: '+str(baseline_test))
        #  logging.info('*************************')
        #  logging.info('Gen Class')
        #  logging.info('F1: '+str(gen_scores[0]))
        #  logging.info('Test: '+str(gen_scores[1]))
        #  logging.info('Epoch: '+str(gen_scores[2]))
        #  logging.info('*******************')
        #  logging.info('CGAN')
        #  logging.info('F1: '+str(scores[0]))
        #  logging.info('Test: '+str(scores[1]))
        #  logging.info('Epoch: '+str(scores[2]))
        #  logging.info('Time taken to run in minutes: '+str((time.time()-start_time)/60))
        #  logging.info('')
        #  logging.info('')

    logging.info('*************************************************')
    logging.info('*************************************************')
    logging.info('Minority Class: '+str(j*50))
    logging.info('MCIR: '+str(j*50/np.mean(imb)))
    logging.info('##################################')
    logging.info('Baseline')
    logging.info('F1: '+str(np.mean(all_b[:, 0])))
    logging.info('Test: '+str(np.mean(all_b[:, 1])))
    logging.info('##################################')
    logging.info('Gen')
    logging.info('F1: '+str(np.mean(all_g[:, 0])))
    logging.info('Test: '+str(np.mean(all_g[:, 1])))
    logging.info('Epoch: '+str(np.mean(all_g[:, 2])))
    logging.info('##################################')
    logging.info('CGAN')
    logging.info('F1: '+str(np.mean(all_c[:, 0])))
    logging.info('Test: '+str(np.mean(all_c[:, 1])))
    logging.info('Epoch: '+str(np.mean(all_c[:, 2])))
    logging.info('##################################')
    logging.info('')
    logging.info('')
    break

#  model = SVC()
#  model.fit(X_train, y_train)
#  y_pred = model.predict(X_test)
#  print((y_pred == y_test).mean())
#  print(f1_score(y_test, y_pred, average='weighted'))

#  print('Random Forest without Gen')
#  model = RandomForestClassifier()
#  model.fit(X_train, y_train)
#  y_pred = model.predict(X_test)
#  print(np.bincount(y_train))
#  print('Testing Accuracy')
#  print((y_pred == y_test).mean())
#  print('F1 Score')
#  print(f1_score(y_test, y_pred, average='weighted'))

#  train_loss = []
#  test_loss = []
#  f1_scores = []
#  for epoch in range(400):
    #  print(epoch)
    #  gen_class = GenClass(d, num_classes=3)
    #  gen_class.train(X_train, y_train, num_epoch=epoch)

    #  counts = np.bincount(y_train)
    #  maxCount = np.max(counts)
    #  for i in range(len(counts)):
        #  toGen = maxCount - counts[i]
        #  if toGen == 0:
            #  continue
        #  noise = torch.randn(toGen, 64)
        #  selFrom = X_train[y_train == i]
        #  selN, _ = selFrom.shape
        #  X_comp = selFrom[np.random.choice(selN, size=toGen, replace=True)]
        #  X_comp = torch.FloatTensor(X_comp)
        #  new = gen_class.gen(noise, X_comp).detach().numpy()
        #  labels = np.zeros((toGen)).astype(int) + i
        #  X_train = np.concatenate((X_train, new))
        #  y_train = np.concatenate((y_train, labels))

    #  model = RandomForestClassifier()
    #  model.fit(X_train, y_train)
    #  y_pred = model.predict(X_test)
    #  print(np.bincount(y_train))
    #  test_loss.append((y_pred == y_test).mean())
    #  f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

#  plt.plot(range(len(test_loss)), test_loss, label='Test Accuracy')
#  plt.plot(range(len(f1_scores)), f1_scores, label='F1 Scores')
#  plt.legend(loc='best')
#  plt.show()
#  print(max(f1_scores))

