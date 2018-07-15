from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import logging
import sys
import pickle
from sklearn import tree
from sklearn.preprocessing import normalize

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval
from cleverhans.attacks import FastGradientMethod
from cleverhans_tutorials.tutorial_models import make_basic_cnn, MLP
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans_tutorials.tutorial_models import Layer, Flatten, Linear, ReLU, Softmax

class Sigmoid(Layer):

    def __init__(self):
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def get_output_shape(self):
        return self.output_shape

    def fprop(self, x):
        return tf.sigmoid(x)

def basic_mlp(nb_classes=2, input_shape=None):
    layers = [Flatten(),
              Linear(20),
              Sigmoid(),
              Linear(10),
              Sigmoid(),
              Flatten(),
              Linear(nb_classes),
              Softmax()
             ]
    model = MLP(layers, input_shape)
    return model

def abalone_mlp(nb_classes=2, input_shape=None):
    layers = [Flatten(),
              Linear(10),
              Sigmoid(),
              Linear(10),
              Sigmoid(),
              Flatten(),
              Linear(nb_classes),
              Softmax()
             ]
    model = MLP(layers, input_shape)
    return model

def make_model(task):
    if task == 'mnist':
        model = make_basic_cnn(nb_classes=2)
    elif task == 'abalone':
        model = abalone_mlp(nb_classes=2, input_shape=[None, 7, 1, 1])
    else:
        model = basic_mlp(nb_classes=2, input_shape=[None, 2, 1, 1])    
    return model

class neural_net_classifier():
    def __init__(self, X=None, y=None, model=None, task=None):
        self.X = X
        self.y = y
        self.model = model
        self.task = task
        
    def fit(self, X, y, target_model=None, model=None, train_params=None, shape=None, num_rounds=3):
        self.X = X
        self.y = y
        self.train_params = train_params
        self.shape = shape
        self.model = model
        #num_rounds=3
        eps_aug = 2
        train_params = self.train_params
        nb_epochs= train_params['nb_epochs']
        batch_size=train_params['batch_size']
        learning_rate=train_params['learning_rate']
        for i in range(num_rounds):
        # Set TF random seed to improve reproducibility
            tf.set_random_seed(1234)
            tf.reset_default_graph()
            # Create TF session
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

            # Define input TF placeholder
            x = tf.placeholder(tf.float32, shape=self.shape)
            y = tf.placeholder(tf.float32, shape=(None, 2))

            rng = np.random.RandomState([2017, 8, 30])

            task = self.task
            model = make_model(task)
            preds = model.get_probs(x) 

            def evaluate():
                pass

            X_train = self.X
            Y_train = self.y
            shape_original = X_train.shape
            X_train = np.reshape(X_train, [len(X_train)]+self.shape[1:])
            Y_train = np.array([[1,0] if i==1 else [0,1] for i in Y_train])

        

        
            model_train(sess, x, y, preds, X_train, Y_train, evaluate=evaluate,
                        args=train_params, rng=rng)
            eval_params = {'batch_size': batch_size}
            pred_results = sess.run(preds, feed_dict={x:X_train})
            fgsm_params = {'eps': eps_aug,
                           #'clip_min': 0.0,
                           #'clip_max': 1.0,
                           'ord': 2}   
            fgsm = FastGradientMethod(model, sess=sess)
            adv_x = fgsm.generate(x, **fgsm_params)
            preds_adv = model.get_probs(adv_x)
            X_adv = sess.run(adv_x, feed_dict={x: X_train})
            X_adv = np.reshape(X_adv, shape_original)
            print X_adv.shape
            X_adv = np.nan_to_num(np.clip(X_adv, -100, 100))
            y_adv = target_model.predict(X_adv)
            self.y = np.concatenate((self.y, y_adv), axis=0)
            self.X = np.concatenate((self.X, X_adv), axis=0)
            sess.close()
            del sess
            
    def predict(self, X_test):
        train_params = self.train_params
        nb_epochs= train_params['nb_epochs']
        batch_size=train_params['batch_size']
        learning_rate=train_params['learning_rate']
        tf.set_random_seed(1234)
        tf.reset_default_graph()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        
        # Define input TF placeholder
        x = tf.placeholder(tf.float32, shape=self.shape)
        y = tf.placeholder(tf.float32, shape=(None, 2))

        rng = np.random.RandomState([2017, 8, 30])

        task = self.task
        model = make_model(task)
        preds = model.get_probs(x) 

        def evaluate():
            pass
       
        X_train = self.X
        Y_train = self.y
        X_train = np.reshape(X_train, [len(X_train)]+self.shape[1:])
        X_test = np.reshape(X_test, [len(X_test)]+self.shape[1:])
        Y_train = np.array([[1,0] if i==1 else [0,1] for i in Y_train])
        
        model_train(sess, x, y, preds, X_train, Y_train, evaluate=evaluate,
                    args=train_params, rng=rng)
        eval_params = {'batch_size': batch_size}
        pred_results = sess.run(preds, feed_dict={x:X_test})
        sess.close()
        del sess
        pred_results = np.array([1 if a[0]>a[1] else -1 for a in pred_results])
        return pred_results
    
    def generate_adv(self, X_test, eps, mask=None):
        train_params = self.train_params
        nb_epochs= train_params['nb_epochs']
        batch_size=train_params['batch_size']
        learning_rate=train_params['learning_rate']

        tf.set_random_seed(1234)
        tf.reset_default_graph()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
        # Define input TF placeholder
        x = tf.placeholder(tf.float32, shape=self.shape)
        y = tf.placeholder(tf.float32, shape=(None, 2))

        rng = np.random.RandomState([2017, 8, 30])

        task = self.task
        model = make_model(task)
        preds = model.get_probs(x) 

        def evaluate():
            pass
       
        X_train = self.X
        Y_train = self.y
        X_train = np.reshape(X_train, [len(X_train)]+self.shape[1:])
        shape_original = X_test.shape
        X_test_original = np.copy(X_test)
        X_test = np.reshape(X_test, [len(X_test)]+self.shape[1:])
        Y_train = np.array([[1,0] if i==1 else [0,1] for i in Y_train])
        
        model_train(sess, x, y, preds, X_train, Y_train, evaluate=evaluate,
                    args=train_params, rng=rng)
        eval_params = {'batch_size': batch_size}
        pred_results = sess.run(preds, feed_dict={x:X_test})
        
        fgsm_params = {'eps': eps,
                       'ord': 2}   
        fgsm = FastGradientMethod(model, sess=sess)
        adv_x = fgsm.generate(x, **fgsm_params)
        preds_adv = model.get_probs(adv_x)
        X_adv = sess.run(adv_x, feed_dict={x: X_test})
        X_adv = np.reshape(X_adv, shape_original)
        
        for i in range(len(mask)):
            if mask[i]:
                pass
            else:
                X_adv[i] = X_test_original[i]
        
        sess.close()
        del sess
        return X_adv


    
###########################

### kernel classifier
class Kernel_Classifier():
    def __init__(self, X=None, y=None, c=None, eps=None, num_rounds=None):
        self.X = X
        self.y = y
        self.c = c
    
    def augment(self, eps, num_rounds):
        [X, y] = [self.X, self.y]
        for i in range(num_rounds):
            mask_pred = [True for x in X]
            X_adv = self.generate_adv(X, y, eps, mask_pred)
            X = np.concatenate([X, X_adv], axis=0)
            y = np.concatenate([y, y])
        self.X = X
        self.y = y
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        
    def predict(self, X_test):
        X_train_kernel = self.X
        y_train_kernel = self.y
        c = self.c
        n_test = len(X_test)
        n_train_kernel = len(X_train_kernel)
        d = len(X_test[0])
        y_pred = np.zeros([n_test, 1])

        for i in range(n_test):
            x = X_test[i].reshape(1,d)
            #y = y_test[i]

            x = np.repeat(x, n_train_kernel, axis=0)
            delta = (X_train_kernel - x)
            dist = np.zeros([n_train_kernel, 1])
            e = np.zeros([n_train_kernel, 1])

            for k in range(n_train_kernel):
                dist[k] = np.linalg.norm(x[k]-X_train_kernel[k])
                e[k] = np.exp(-dist[k]*1.0/c)


            mask = (y_train_kernel==1).reshape(n_train_kernel,1)

            total = sum(e)
            p_pos = sum(np.multiply(e, mask))
            p_neg = total - p_pos

            if p_pos > p_neg:
                y_pred[i] = 1
            else:
                y_pred[i] = -1
        return y_pred
    
    def generate_adv(self, X_test, y_test, eps, mask_pred):
        X_train_kernel = self.X
        y_train_kernel = self.y
        c = self.c
        n_test = len(X_test)
        n_train_kernel = len(X_train_kernel)
        d = len(X_test[0])
        adv = np.zeros([n_test, d])
        
        for i in range(n_test):
            x = X_test[i].reshape(1,d)
            y = y_test[i]

            x = np.repeat(x, n_train_kernel, axis=0)
            delta = (X_train_kernel - x)*2
            dist = np.zeros([n_train_kernel, 1])
            e = np.zeros([n_train_kernel, 1])

            for k in range(n_train_kernel):
                dist[k] = np.linalg.norm(x[k]-X_train_kernel[k])
                e[k] = np.exp(-dist[k]*1.0/c)

            if y==1:
                mask = (y_train_kernel==1)
            else:
                #mask = 1-y_train_kernel
                mask = 1-(y_train_kernel==1)
            mask = mask.reshape([len(mask),1])
            g_same = np.multiply(e, mask)
            delta_same = np.multiply(delta, mask)

            sum_e = sum(e)
            sum_g_same = sum(g_same)

            g_2 = sum_e**2
            t_same = np.repeat(g_same, d, axis=1)
            df_g = np.multiply(t_same, delta_same) * sum_e
            df_g = np.sum(df_g, axis=0).reshape(1, d)
            t = np.repeat(e, d, axis=1)
            dg_f = np.multiply(t, delta) * sum_g_same
            dg_f = np.sum(dg_f, axis=0).reshape(1,d)

            deriv = (df_g - dg_f)*1.0
            deriv = normalize(deriv)
            X_new = x[0]-eps*deriv

            if mask_pred[i]:
                adv[i] = X_new
            else:
                adv[i] = X_test[i]
        return adv


# DT attacks
class decisionTreeNode:
    def __init__(self, node_id=None, input_component=None, threshold=None,left=None,right=None,output=[], parent=None):
        self.node_id = node_id
        self.input_component = input_component
        self.threshold = threshold
        self.left = left
        self.right = right
        self.output = output
        self.parent = parent

def tree_parser(clf):
    t = clf.tree_
    n_nodes = t.node_count
    children_left = t.children_left
    children_right = t.children_right
    feature = t.feature
    threshold = t.threshold
    values = t.__getstate__()['values']

    t_dict = {}
    stack = [(0, None)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_id = stack.pop()
        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            left = children_left[node_id]
            right = children_right[node_id]
            input_component = feature[node_id]
            thres = threshold[node_id]
            output = values[node_id]

            stack.append((left, node_id))
            stack.append((right, node_id))
            if parent_id:
                t_dict[str(node_id)] = decisionTreeNode(str(node_id), input_component, 
                                                   thres, str(left), str(right), output, str(parent_id))
            else: 
                t_dict[str(node_id)] = decisionTreeNode(str(node_id), input_component, 
                                                   thres, str(left), str(right), output, None)
        
        else:
            input_component = feature[node_id]
            thres = threshold[node_id]
            output = values[node_id]
            t_dict[str(node_id)] = decisionTreeNode(str(node_id), input_component,
                                                   thres, None, None, output, str(parent_id))
    return t_dict

def prepare_tree(X, y, max_depth):
    clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=max_depth)
    clf.fit(X, y)
    dt = tree_parser(clf)
    return clf, dt