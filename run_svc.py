import time
from imblearn.over_sampling import ADASYN, SMOTE
import numpy as np
from cgan_models import CGAN
from gen_class import GenClass
from warnings import simplefilter
from sklearn.svm import SVC
from utils import process_wearable_dataset
from sklearn.metrics import f1_score
import logging
from process_occutherm import process_occutherm


# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
logging.basicConfig(level=logging.DEBUG, filename="svc_logfile_occutherm", filemode="a+",
                        format="%(message)s")
all_class0 = [10, 100, 200, 300, 400, 1000]
all_class1 = [10, 100, 200, 1000, 1000, 1000]

runs = 5

for j0, j1 in zip(all_class0, all_class1):
    all_b = np.zeros((runs, 2))
    all_g = np.zeros((runs, 3))
    all_c = np.zeros((runs, 3))
    all_smote = np.zeros((runs, 2))
    all_ada = np.zeros((runs, 2))
    imb = []
    print('****************')

    for i in range(runs):
        start_time = time.time()
        X_train, y_train, X_test, y_test, mcir = process_occutherm(j0, j1)
        #  X_train, y_train, X_test, y_test, mcir = process_wearable_dataset(j)

        n, d = X_train.shape

        imb.append(mcir)

        X = 0
        y = 0
        
        model = SVC()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        baseline_test = (y_pred == y_test).mean()
        baseline_f1 = f1_score(y_test, y_pred, average='weighted')
        all_b[i] = np.array([baseline_f1, baseline_test])

        gen_class = GenClass(d, num_classes=3)
        gen_scores = gen_class.train(X_train, y_train, X_test, y_test, SVC(), num_epoch=200)
        all_g[i] = np.array(gen_scores)
        print(':::::')


        cgan = CGAN(d, num_classes=3)
        scores = cgan.train(X_train, y_train, X_test, y_test, SVC(), num_epoch=2000)
        all_c[i] = np.array(scores)

        sm = SMOTE()
        X_smote, y_smote = sm.fit_resample(X_train.copy(), y_train.copy())
        print(np.bincount(y_smote))
        model = SVC()
        model.fit(X_smote, y_smote)
        y_pred = model.predict(X_test)
        smote_test = (y_pred == y_test).mean()
        smote_f1 = f1_score(y_test, y_pred, average='weighted')
        all_smote[i] = np.array([smote_f1, smote_test])

        sm = ADASYN()
        X_smote, y_smote = sm.fit_resample(X_train.copy(), y_train.copy())
        print(np.bincount(y_smote))
        model = SVC()
        model.fit(X_smote, y_smote)
        y_pred = model.predict(X_test)
        smote_test = (y_pred == y_test).mean()
        smote_f1 = f1_score(y_test, y_pred, average='weighted')
        all_ada[i] = np.array([smote_f1, smote_test])
        
        # print('*************************')
        # print(i)
        # print('Baseline')
        # print(baseline_f1)
        # print('*************************')
        print('Gen Class')
        print(gen_scores[0])
        # print('*************************')
        print('CGAN')
        print(scores[0])
        # print('Time taken to run in minutes: '+str((time.time()-start_time)/60))
        # print('*************************')
        # print()
        # print()


    logging.info('##################################')
    logging.info('MCIR: '+str(np.mean(imb)))
    logging.info('##################################')
    logging.info('F1 Scores')
    logging.info('Baseline: '+str(np.mean(all_b[:, 0])))
    logging.info('Gen: '+str(np.mean(all_g[:, 0])))
    logging.info('CGAN: '+str(np.mean(all_c[:, 0])))
    logging.info('SMOTE: '+str(np.mean(all_smote[:, 0])))
    logging.info('ADASYN: '+str(np.mean(all_ada[:, 0])))
    logging.info('##################################')
    logging.info('Test Accuracy')
    logging.info('Baseline: '+str(np.mean(all_b[:, 1])))
    logging.info('Gen: '+str(np.mean(all_g[:, 1])))
    logging.info('CGAN: '+str(np.mean(all_c[:, 1])))
    logging.info('SMOTE: '+str(np.mean(all_smote[:, 1])))
    logging.info('ADASYN: '+str(np.mean(all_ada[:, 1])))
    logging.info('##################################')
    logging.info('')

    # print('##################################')
    # print('Epochs')
    # print('Gen: '+str(np.mean(all_g[:, 2])))
    # print('CGAN: '+str(np.mean(all_c[:, 2])))

