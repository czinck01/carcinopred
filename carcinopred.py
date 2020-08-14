import os, time, datetime
import numpy as np

from sklearn.neural_network import MLPClassifier as MLP
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import confusion_matrix as CM
from sklearn.metrics import accuracy_score as AS
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier

import tensorflow as tf
import tensorflow_probability as tfp

from scipy.stats import mode

import src.bnn.hmc as hmc
import src.bnn.bnn as bnn

def get_data():
    mat = np.loadtxt('mat.csv', delimiter=',', dtype='float32')
    test_indices = range(2, len(mat), 5)
    train_indices = np.setdiff1d(range(len(mat)), test_indices)
    test = np.array([mat[i] for i in test_indices])
    train = np.array([mat[i] for i in train_indices])
    
    X_train = train[:, :-1]
    X_test = test[:, :-1]
    Y_train = train[:, -1]
    Y_test = test[:, -1]

    return X_train, X_test, Y_train, Y_test

def accuracy(Y_true, Y_pred):
    cm_mean, cm_std = confusion(Y_true, Y_pred)
    return np.trace(cm_mean) / np.sum(cm_mean), np.trace(cm_std) / np.sum(cm_mean)

def confusion(Y_true, Y_pred):
    cms = []
    for Y in Y_pred:
        cms.append(CM(Y_true, Y))
    cms = np.array(cms)
    cm_mean = np.mean(cms, axis=0)
    cm_std = np.std(cms, axis=0)
    return cm_mean, cm_std

def output(test_num, cm_mean, cm_std, acc_mean, acc_std):
    with open("results" + str(test_num) + ".txt", "w") as file:
        file.write(np.array2string(cm_mean) + "\n")
        file.write(np.array2string(cm_std) + "\n")
        file.write(str(acc_mean) + "\n")
        file.write(str(acc_std) + "\n")

# One-hot NN
def OHNN(architecture, X_train, X_test, Y_train, Y_test):
    clf = MLP(hidden_layer_sizes=architecture)
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    
    cm_mean = CM(Y_test, Y_pred)
    cm_std = np.zeros_like(cm_mean)
    acc_mean = AS(Y_test, Y_pred)
    acc_std = 0
    output(1, cm_mean, cm_std, acc_mean, acc_std)

# One vs Rest NN
def OVRNN(architecture, X_train, X_test, Y_train, Y_test):
    clf = OneVsRestClassifier(MLP(hidden_layer_sizes=architecture))
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    
    cm_mean = CM(Y_test, Y_pred)
    cm_std = np.zeros_like(cm_mean)
    acc_mean = AS(Y_test, Y_pred)
    acc_std = 0
    output(2, cm_mean, cm_std, acc_mean, acc_std)

# One vs One NN
def OVONN(architecture, X_train, X_test, Y_train, Y_test):
    clf = OneVsOneClassifier(MLP(hidden_layer_sizes=architecture))
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    
    cm_mean = CM(Y_test, Y_pred)
    cm_std = np.zeros_like(cm_mean)
    acc_mean = AS(Y_test, Y_pred)
    acc_std = 0
    output(3, cm_mean, cm_std, acc_mean, acc_std)

# One-hot BNN
def OHBNN(architecture, X_train, X_test, Y_train, Y_test):
    start = time.time()
    prior = tfp.distributions.Normal(0, 1.0)
    architecture = [len(X_train[0])] + list(architecture) + [int(np.max(Y_train)) + 1]
    init = bnn.get_random_initial_state(prior, prior, architecture, overdisp=1.0)
    Y_pred, trace, k, s = hmc.hmc_predict(prior, prior, init, X_train, Y_train, X_test)
    log_prob = trace[0].inner_results.accepted_results.target_log_prob.numpy()
    Y_pred = tf.math.argmax(Y_pred, axis=2)
    cm_mean, cm_std = confusion(Y_test, Y_pred)
    acc_mean, acc_std = accuracy(Y_test, Y_pred)
    end = time.time()
    print(str(datetime.timedelta(seconds=int(end-start))))

    output(4, cm_mean, cm_std, acc_mean, acc_std)

# One vs Rest BNN
def OVRBNN(architecture, X_train, X_test, Y_train, Y_test):
    prior = tfp.distributions.Normal(0, 1.0)
    architecture = [len(X_train[0])] + list(architecture) + [2]
    Y_pred = 0
    for i in range(int(np.max(Y_train)) + 1):
        Y_train_temp = np.copy(Y_train)
        Y_train_temp[Y_train_temp == i] = 1
        Y_train_temp[Y_train_temp != i] = 0
        init = bnn.get_random_initial_state(prior, prior, architecture, overdisp=1.0)
        Y_pred_temp, trace, k, s = hmc.hmc_predict(prior, prior, init, X_train, Y_train_temp, X_test)
        Y_pred_temp = np.array(tf.math.argmax(Y_pred_temp, axis=2))
        Y_pred_temp[Y_pred_temp == 1] = i
        if (i == 0):
            Y_pred = Y_pred_temp
        else:
            Y_pred[np.nonzero(Y_pred_temp)] = Y_pred_temp[np.nonzero(Y_pred_temp)] 
    print(Y_pred)
    cm_mean, cm_std = confusion(Y_test, Y_pred)
    acc_mean, acc_std = accuracy(Y_test, Y_pred)
    output(5, cm_mean, cm_std, acc_mean, acc_std)

# One vs One BNN
def OVOBNN(architecture, X_train, X_test, Y_train, Y_test):
    prior = tfp.distributions.Normal(0, 1.0)
    architecture = [len(X_train[0])] + list(architecture) + [2]
    Y_preds = []
    for i in range(int(np.max(Y_train)) + 1):
        for j in range(i+1, int(np.max(Y_train)) + 1):
            traini = np.where(Y_train == i)
            trainj = np.where(Y_train == j)
            X_train_temp = np.concatenate((X_train[traini], X_train[trainj]))
            Y_train_temp = np.concatenate((Y_train[traini], Y_train[trainj]))
            Y_train_temp[Y_train_temp == i] = 0
            Y_train_temp[Y_train_temp == j] = 1

            init = bnn.get_random_initial_state(prior, prior, architecture, overdisp=1.0)
            Y_pred_temp, trace, k, s = hmc.hmc_predict(prior, prior, init, X_train_temp, Y_train_temp, X_test)
            Y_pred_temp = np.array(tf.math.argmax(Y_pred_temp, axis=2))
            Y_pred_temp[Y_pred_temp == 0] = i
            Y_pred_temp[Y_pred_temp == 1] = j
            Y_preds.append(Y_pred_temp)
    Y_pred = mode(np.array(Y_preds), axis=0)[0][0]
    cm_mean, cm_std = confusion(Y_test, Y_pred)
    acc_mean, acc_std = accuracy(Y_test, Y_pred)
    output(6, cm_mean, cm_std, acc_mean, acc_std)

architecture = (300, 100)


X_train, X_test, Y_train, Y_test = get_data()
OHNN(architecture, X_train, X_test, Y_train, Y_test)
OVRNN(architecture, X_train, X_test, Y_train, Y_test)
OVONN(architecture, X_train, X_test, Y_train, Y_test)
OHBNN(architecture, X_train, X_test, Y_train, Y_test)
OVRBNN(architecture, X_train, X_test, Y_train, Y_test)
OVOBNN(architecture, X_train, X_test, Y_train, Y_test)
