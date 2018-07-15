from __future__ import division
import time
import sys
sys.path.insert(0, "./cleverhans")
import numpy as np
from random import shuffle
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from nn_attacks import generate_adversarial_examples 
from prepare_data import mnist_1v7_data, abalone_data, halfmoon_data, shuffle_data
from model_utils import neural_net_classifier, basic_mlp, prepare_tree, Kernel_Classifier
from robust_1nn import Robust_1NN

# This function generates the substitute model for black-box attacks.
def generate_sub(X_train, y_train, X_test, y_test, X_extra, y_extra_pred, neigh, eps):
    clf=None
    dt=None
    mask=None
    mapping='noMapping'
    
    n_extra = len(X_extra)
    n_test = len(X_test)
    if FLAG == 'dt':
        clf, dt = prepare_tree(X_extra, y_extra_pred, MAX_DEPTH)
    elif FLAG == 'lr':
        clf = LogisticRegression()
        clf.fit(X_extra, y_extra_pred)
    elif FLAG == 'kernel':
        clf = Kernel_Classifier(c=C)
        clf.fit(X_extra, y_extra_pred)
    elif FLAG == 'nn':
        clf = neural_net_classifier(task=TASK)
        clf.fit(X_extra, y_extra_pred, 
                target_model=neigh,
                model=model, 
                train_params=train_params, 
                shape=shape, num_rounds=2)
    elif FLAG == 'wb_kernel':
        clf = Kernel_Classifier(c=C)
        clf.fit(X_train, y_train)
    else:
        clf = neigh
    y_pred = clf.predict(X_test).reshape(y_test.shape)
    mask = y_pred==y_test
    acc_train = (n_test-sum(abs(y_pred-y_test))/2)/n_test
    print 'sub accuracy on true test', n_test-sum(abs(y_pred-y_test))/2
    
    y_sub_extra_pred = clf.predict(X_extra).reshape(y_extra_pred.shape)
    acc_extra = (n_extra-sum(abs(y_sub_extra_pred - y_extra_pred))/2)/n_extra
    print 'sub accuracy on sun train', n_extra-sum(abs(y_sub_extra_pred - y_extra_pred))/2
    
    y_true_pred = neigh.predict(X_test)
    acc_original = (n_test-sum(abs(y_true_pred - y_pred))/2)/n_test
    print 'sub agrees with original', n_test-sum(abs(y_true_pred - y_pred))/2
    if FLAG == 'dt':
        clf = dt
    return [clf, mask, [acc_train, acc_extra, acc_original]]


# creating adversarial examples for standard 1-NN
def standard_nn(X_train, y_train, X_test, y_test, X_extra, y_extra, ep):
    n_test = len(y_test)
    n_extra = len(y_extra)
    standard_acc = []
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(X_train, y_train)
    y_extra_pred = neigh.predict(X_extra)
    
    clf=None
    dt=None
    mask=None
    mapping='noMapping'

    [clf, mask, 
     [acc_train, 
      acc_extra, 
      acc_original]] = generate_sub(X_train, y_train, 
                                    X_test, y_test, 
                                    X_extra, y_extra_pred, neigh, ep[-1])                                                                              
    for i in range(len(ep)):
        eps = ep[i]
        adv_test = generate_adversarial_examples(FLAG, eps, 
                                                 X_train, X_test, 
                                                 y_train, y_test,
                                                 X_extra, y_extra_pred,
                                                 mapping, clf, mask)
        print adv_test - X_test
        y_adv = neigh.predict(adv_test)
        adv = sum(abs(y_test - y_adv))/2
        test_acc = 1-(adv*1.0/n_test)
        standard_acc.append(test_acc)
        
    return [standard_acc, [acc_train, acc_extra, acc_original]]


# create adversarial examples for RobustNN
def robust_nn(X_train, y_train, X_test, y_test, X_extra, y_extra, ep, eps_adv):
    n_test = len(y_test)
    n_extra = len(y_extra)
    robustnn_acc = []
    print ep
    robust_1nn = Robust_1NN(X_train, y_train, 0.45, 0.1, eps_adv)
    robust_1nn.find_confident_label()
    robust_1nn.find_red_points()
    robust_1nn.fit()
    [new_train, y_new_train] = robust_1nn.get_data()
    neigh = robust_1nn.get_clf()
    
    y_extra_pred = neigh.predict(X_extra)
    
    clf=None
    dt=None
    mask=None
    mapping='noMapping'

    [clf, mask, 
     [acc_train, 
      acc_extra, 
      acc_original]] = generate_sub(new_train, y_new_train, 
                                    X_test, y_test, 
                                    X_extra, y_extra_pred, neigh, ep[-1])
        
    for i in range(len(ep)):
        eps = ep[i]
        adv_test = generate_adversarial_examples(FLAG, eps, 
                                                 new_train, X_test, 
                                                 y_new_train, y_test,
                                                 X_extra, y_extra_pred,
                                                 mapping, clf, mask)
        #print adv_test
        y_adv = neigh.predict(adv_test)
        adv = sum(abs(y_test - y_adv))/2
        test_acc = 1-(adv*1.0/n_test)
        robustnn_acc.append(test_acc)
    return [robustnn_acc, [acc_train, acc_extra, acc_original]]

# Augmented learning
def ATNN(X_train, y_train, eps, X_aug=None, y_aug=None):
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(X_train, y_train)
    if X_aug and y_aug:
        new_train = np.concatenate((X_train, X_aug), axis = 0)
        y_new_train = np.concatenate((y_train, y_aug), axis = 0)
    else:
        clf=None
        dt=None
        mask=None
        mapping='noMapping'
        if FLAG == 'dt':
            clf, dt = prepare_tree(X_train, y_train, MAX_DEPTH)
        elif FLAG == 'lr':
            clf = LogisticRegression()
            clf.fit(X_train, y_train)
        elif FLAG == 'kernel' or FLAG == 'wb_kernel':
            clf = Kernel_Classifier(c=C)
            clf.fit(X_train, y_train)
        elif FLAG == 'nn':
            clf = neural_net_classifier(task=TASK)
            clf.fit(X_train, y_train,
                    target_model=neigh,
                    model=model, 
                    train_params=train_params, 
                    shape=shape)
        else:
            clf = neigh    
        y_pred = clf.predict(X_train).reshape(y_train.shape)
        mask = y_pred==y_train
        mask = [True for i in range(len(y_train))]
        if FLAG == 'dt':
            clf = dt
        aug = generate_adversarial_examples(FLAG, eps, 
                                            X_train, X_train, 
                                            y_train, y_train,
                                            X_train, y_train,
                                            mapping, clf, mask, 'attack') 
        new_train = np.concatenate((X_train, aug), axis = 0)
        y_new_train = np.concatenate((y_train, y_train), axis = 0)
    [new_train, y_new_train] = shuffle_data(new_train, y_new_train)
    neigh.fit(new_train, y_new_train)
    return [new_train, y_new_train, neigh]

# Augmented learning with adv. examples to all baseline methods.
def ATNN_all(X_train, y_train, eps, X_aug=None, y_aug=None):
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(X_train, y_train)
    if X_aug and y_aug:
        new_train = np.concatenate((X_train, X_aug), axis = 0)
        y_new_train = np.concatenate((y_train, y_aug), axis = 0)
    else:
        clf=None
        dt=None
        mask=None
        clfs = []
        masks = []
        #FLAGS = ['wb', 'dt', 'lr', 'kernel', 'nn']
        FLAGS = ['wb', 'kernel', 'nn']
        mapping='noMapping'
        
        new_train = np.copy(X_train)
        y_new_train = np.copy(y_train)
        
        clf = neigh
        y_pred = clf.predict(X_train).reshape(y_train.shape)
        mask = y_pred==y_train
        clfs.append(clf)
        masks.append(mask)
        
        '''
        clf, dt = prepare_tree(X_train, y_train, MAX_DEPTH)
        y_pred = clf.predict(X_train).reshape(y_train.shape)
        mask = y_pred==y_train
        clfs.append(dt)
        masks.append(mask)
        
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_train).reshape(y_train.shape)
        mask = y_pred==y_train
        clfs.append(clf)
        masks.append(mask)
        '''
        
        clf = Kernel_Classifier(c=C)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_train).reshape(y_train.shape)
        mask = y_pred==y_train
        clfs.append(clf)
        masks.append(mask)
    
        clf = neural_net_classifier(task=TASK)
        clf.fit(X_train, y_train,
                target_model=neigh,
                model=model, 
                train_params=train_params, 
                shape=shape)  
        y_pred = clf.predict(X_train).reshape(y_train.shape)
        mask = y_pred==y_train
        clfs.append(clf)
        masks.append(mask)
        augs = []
        for (clf, mask, flag) in zip(clfs, masks, FLAGS):
            mask = [True for i in range(len(y_train))]
            aug = generate_adversarial_examples(flag, eps, 
                                                X_train, X_train, 
                                                y_train, y_train,
                                                X_train, y_train,
                                                mapping, clf, mask, 'attack')
            augs.append(aug)
            new_train = np.concatenate((new_train, aug), axis = 0)
            y_new_train = np.concatenate((y_new_train, y_train), axis = 0)
    [new_train, y_new_train] = shuffle_data(new_train, y_new_train)    
    neigh.fit(new_train, y_new_train)
    return [new_train, y_new_train, neigh]

# create adversarial examples against ATNN
def at_nn(X_train, y_train, X_test, y_test, X_extra, y_extra, ep):
    n_test = len(y_test)
    n_extra = len(y_extra)
    atnn_acc = []
    [new_train, y_new_train, neigh] = ATNN(X_train, y_train, eps_aug)
    y_extra_pred = neigh.predict(X_extra)
    
    clf=None
    dt=None
    mask=None
    mapping='noMapping'

    [clf, mask, 
     [acc_train, 
      acc_extra, 
      acc_original]] = generate_sub(new_train, y_new_train, 
                                    X_test, y_test, 
                                    X_extra, y_extra_pred, neigh, ep[-1])
    for i in range(len(ep)):
        eps = ep[i]
        adv_test = generate_adversarial_examples(FLAG, eps, 
                                                 new_train, X_test, 
                                                 y_new_train, y_test,
                                                 X_extra, y_extra_pred,
                                                 mapping, clf, mask)
        y_adv = neigh.predict(adv_test)
        adv = sum(abs(y_test - y_adv))/2
        test_acc = 1-(adv*1.0/n_test)
        atnn_acc.append(test_acc)
    return [atnn_acc, [acc_train, acc_extra, acc_original]]


# create adversarial examples against ATNN
def at_nn_all(X_train, y_train, X_test, y_test, X_extra, y_extra, ep):
    n_test = len(y_test)
    n_extra = len(y_extra)
    atnn_acc = []
    [new_train, y_new_train, neigh] = ATNN_all(X_train, y_train, eps_aug_all)
    y_extra_pred = neigh.predict(X_extra)
    
    clf=None
    dt=None
    mask=None
    mapping='noMapping'

    [clf, mask, 
     [acc_train, 
      acc_extra, 
      acc_original]] = generate_sub(new_train, y_new_train, 
                                    X_test, y_test, 
                                    X_extra, y_extra_pred, neigh, ep[-1])
    for i in range(len(ep)):
        eps = ep[i]
        adv_test = generate_adversarial_examples(FLAG, eps, 
                                                 new_train, X_test, 
                                                 y_new_train, y_test,
                                                 X_extra, y_extra_pred,
                                                 mapping, clf, mask)
        y_adv = neigh.predict(adv_test)
        adv = sum(abs(y_test - y_adv))/2
        test_acc = 1-(adv*1.0/n_test)
        atnn_acc.append(test_acc)
    if clf:
        return [atnn_acc, [acc_train, acc_extra, acc_original]]
    else:
        return atnn_acc

if __name__ == "__main__":
    ### main script for the experiment.
    t = time.time()
    TASK = sys.argv[1]     # which dataset
    FLAG = sys.argv[2]     # which attack method
    '''
        Valid flags are:
        wb:    whitebox attack
        wb_kernel: whitebox with kernel sub
        kernel:  blackbox with kernel sub
        nn:      blackbox with neural net sub
        lr:      blackbox with logistic regression sub.
    '''

    if TASK == 'abalone':    
        ep = [0.002*(i) for i in range(21)]
        eps_aug = 0.01
        eps_aug_all = 0.01
        eps_adv = 0.072
        eps_max = 0.08
        num_exp = 10
        C = 1e-2
        n = 500
        m = 100
        train_params = {
            'nb_epochs': 20,
            'batch_size': 128,
            'learning_rate': 0.0005
        }
        shape=[None, 7, 1, 1]
        model=None
        MAX_DEPTH = 5
        
    elif TASK == 'mnist':
        ep = [0.2*(i) for i in range(21)]
        #ep = [3]
        eps_aug = 2.5
        eps_aug_all = 2.5
        eps_adv = 8
        eps_max = 8
        num_exp = 1
        C = 0.1
        n = 500
        m = 200
        train_params = {
            'nb_epochs': 6,
            'batch_size': 256,
            'learning_rate': 0.001
        }
        shape=[None, 28, 28, 1]
        model=None
        
    elif TASK == 'halfmoon':
        ep = [0.01*(i) for i in range(21)]
        eps_aug = 0.2
        eps_aug_all = 0.04
        eps_adv = 0.4
        eps_max = 0.4
        num_exp = 5
        C = 0.1
        n = 2000
        m = 1000
        train_params = {
            'nb_epochs': 100,
            'batch_size': 128,
            'learning_rate': 0.002
        }
        shape=[None, 2, 1, 1]
        model=None
        MAX_DEPTH = 3

    standard_acc = []
    robust_acc = []
    atnn_acc = []
    atnn_all_acc = []
    standard_stats = []
    robust_stats = []
    atnn_stats = []
    atnn_all_stats = []
    neigh = KNeighborsClassifier(n_neighbors=1)

    for i in range(num_exp):
        print "now processing Experiment %d" % i
        t = time.time()
        if TASK == 'mnist':
            [X_train, X_test, X_extra, y_train, y_test, y_extra, X_valid, y_valid] = mnist_1v7_data(n,m)
        elif TASK == 'abalone':
            [X_train, X_test, X_extra, y_train, y_test, y_extra, X_valid, y_valid] = abalone_data(n,m)
        elif TASK == 'halfmoon':
            [X_train, X_test, X_extra, y_train, y_test, y_extra, X_valid, y_valid] = halfmoon_data(n,m)

        print 'attack standard NN'
        standard_res = standard_nn(X_train,y_train,X_test,y_test,X_extra,y_extra,ep) 
        standard_acc.append(standard_res[0])
        standard_stats.append(standard_res[1])

        print 'attack robust NN'
        robust_res = robust_nn(X_train,y_train,X_test,y_test,X_extra,y_extra,ep,eps_adv)
        robust_acc.append(robust_res[0])
        robust_stats.append(robust_res[1])

        print 'attack ATNN'
        atnn_res = at_nn(X_train,y_train,X_test,y_test,X_extra,y_extra,ep)
        atnn_acc.append(atnn_res[0])
        atnn_stats.append(atnn_res[1])

        print 'attack ATNN-all'
        atnn_all_res = at_nn_all(X_train,y_train,X_test,y_test,X_extra,y_extra,ep)
        atnn_all_acc.append(atnn_all_res[0])
        atnn_all_stats.append(atnn_all_res[1])

        print time.time() - t
        
    dim = [num_exp, len(ep)]
    standard_acc = np.array(standard_acc).reshape(dim).sum(axis=0)*1.0/num_exp
    robust_acc = np.array(robust_acc).reshape(dim).sum(axis=0)*1.0/num_exp
    atnn_acc = np.array(atnn_acc).reshape(dim).sum(axis=0)*1.0/num_exp
    atnn_all_acc = np.array(atnn_all_acc).reshape(dim).sum(axis=0)*1.0/num_exp

    standard_stats = np.array(standard_stats)
    atnn_stats = np.array(atnn_stats)
    robust_stats = np.array(robust_stats)
    atnn_all_stats = np.array(atnn_all_stats)
    print time.time() - t


    data_folder = "./experiment_results/"
    f_name = data_folder+TASK+FLAG+'new'
    np.save(f_name, [standard_acc, robust_acc, atnn_acc, atnn_all_acc])
    np.save(f_name+'stats', [standard_stats, robust_stats, atnn_stats, atnn_all_stats])
    print robust_acc
    print standard_acc
    print robust_acc
    print atnn_all_acc, atnn_acc 
    print standard_stats,atnn_all_stats
