import time
import numpy as np
from random import shuffle
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from eps_separation import find_eps_separated_set
from nn_attack_white_box import generate_adversarial_examples
import pickle

def mnist_1v7_data(n=1000, m=500):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    ones = [mnist.train.images[i] for i in range(len(mnist.train.labels)) if mnist.train.labels[i] == 1]
    sevens = [mnist.train.images[i] for i in range(len(mnist.train.labels)) if mnist.train.labels[i] == 7]
    onesTest = [mnist.validation.images[i] for i in range(len(mnist.validation.labels)) if mnist.validation.labels[i] == 1]
    sevensTest = [mnist.validation.images[i] for i in range(len(mnist.validation.labels)) if mnist.validation.labels[i] == 7]
    
    
    n_ones = len(ones)
    n_sevens = len(sevens)
    shuffle(ones)
    shuffle(sevens)
    X_train = np.array(ones[:n] + sevens[:n])
    X_test = np.array(onesTest[:m] + sevensTest[:m])
    X_extra = np.array(ones[n:2*n] + sevens[n:2*n])
    X_valid = np.array(onesTest[2*n:2*n+m] + sevensTest[2*n:2*n+m])
    y_train = np.array([1 for i in range(n)] + [-1 for i in range(n)])
    y_test = np.array([1 for i in range(m)] + [-1 for i in range(m)])
    y_extra = np.array([1 for i in range(n)] + [-1 for i in range(n)])
    y_valid = np.array([1 for i in range(m)] + [-1 for i in range(m)])
    assert (len(X_extra) == len(y_extra))
    return [X_train, X_test, X_extra, y_train, y_test, y_extra, X_valid, y_valid]

def mnist_avb_data(a=1, b=7, n=1000, m=500):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
    ones = [mnist.train.images[i] for i in range(len(mnist.train.labels)) if mnist.train.labels[i] == a]
    sevens = [mnist.train.images[i] for i in range(len(mnist.train.labels)) if mnist.train.labels[i] == b]
    onesTest = [mnist.validation.images[i] for i in range(len(mnist.validation.labels)) if mnist.validation.labels[i] == a]
    sevensTest = [mnist.validation.images[i] for i in range(len(mnist.validation.labels)) if mnist.validation.labels[i] == b]
    
    
    n_ones = len(ones)
    n_sevens = len(sevens)
    shuffle(ones)
    shuffle(sevens)
    X_train = np.array(ones[:n] + sevens[:n])
    X_test = np.array(onesTest[:m] + sevensTest[:m])
    X_extra = np.array(ones[n:2*n] + sevens[n:2*n])
    y_train = np.array([1 for i in range(n)] + [-1 for i in range(n)])
    y_test = np.array([1 for i in range(m)] + [-1 for i in range(m)])
    y_extra = np.array([1 for i in range(n)] + [-1 for i in range(n)])
    assert (len(X_extra) == len(y_extra))
    return [X_train, X_test, X_extra, y_train, y_test, y_extra]

from sklearn import datasets

def sample(n, sigma):
    [X, Y] = datasets.make_moons(n, True, sigma)
    Y = np.array([1 if i==1 else -1 for i in Y])
    return [X, Y]

def halfmoon_data(n=1000, m=500, sigma=0.2):
    [X_train, y_train] = sample(n, sigma)
    [X_test, y_test] = sample(m, sigma)
    [X_extra, y_extra] = sample(n, sigma)
    [X_valid, y_valid] = sample(m, sigma)
    return [X_train, X_test, X_extra, y_train, y_test, y_extra, X_valid, y_valid]


def abalone_data(n=500,m=100):
    data = np.genfromtxt('abalone.data', dtype='str', delimiter=',')
    data = [data[i] for i in range(len(data)) if data[i][0] == 'F']
    X = [data[i][1:8] for i in range(len(data))]
    Y = [int(data[i][8]) for i in range(len(data))]
    X = [map(float, X[i]) for i in range(len(X))]
    s = sum([Y[i] >= 11 for i in range(len(Y))])
    # half of the abalones are 11 years old and above, so the classification task is whether age >= 11
    Y = [int(Y[i] >= 11) for i in range(len(Y))]
    Y = [1 if i==1 else -1 for i in Y]
    Z = zip(X, Y)
    shuffle(Z)
    test = Z[:m]
    train = Z[m:m+n]
    extra = Z[m+n:m+2*n]
    valid = Z[m+2*n:2*m+2*n]
    Xtrain = np.array([train[i][0] for i in range(len(train))])
    Ytrain = np.array([train[i][1] for i in range(len(train))])
    Xtest = np.array([test[i][0] for i in range(len(test))])
    Ytest = np.array([test[i][1] for i in range(len(test))])
    Xextra = np.array([extra[i][0] for i in range(len(extra))])
    Yextra = np.array([extra[i][1] for i in range(len(extra))])
    Xvalid = np.array([test[i][0] for i in range(len(valid))])
    Yvalid = np.array([test[i][1] for i in range(len(valid))])
    return [Xtrain, Xtest, Xextra, Ytrain, Ytest, Yextra, Xvalid, Yvalid]

def shuffle_data(X, Y):
    n = X.shape[0]
    X_new = np.zeros(X.shape)
    Y_new = np.zeros(Y.shape)
    index = np.array([i for i in range(n)])
    np.random.shuffle(index)
    for i in range(n):
        j = index[i]
        X_new[i, :] = X[j, :]
        Y_new[i] = Y[j]
    return [X_new, Y_new]