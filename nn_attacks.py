from __future__ import division

import time
import numpy as np
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

# DT attacks
def prediction(decisionTree_instance, sample, argmax=True, node_index=False):
    node = decisionTree_instance['0']
    while node.left or node.right:
        if float(sample[int(node.input_component)]) <= float(node.threshold):
            node = decisionTree_instance[node.left]
        else:
            node = decisionTree_instance[node.right]
    
    if argmax:
        return np.argmax(node.output)
    else:
        if node_index == False:
            return node.output
        else:
            return str(node.node_id)

def find_adv(decisionTree_instance, sample):
    legitimate_classification_node = decisionTree_instance[prediction(decisionTree_instance, sample, argmax=False, node_index=True)]
    legitimate_class = prediction(decisionTree_instance, sample, argmax=True)
    ancestor = legitimate_classification_node
    adv_node = legitimate_classification_node
    previous_ancestor = ancestor
    while np.argmax(adv_node.output) == legitimate_class and ancestor.parent:
        # is adv node on the left of its parent?
        list_components_left = [] #list of nodes where we went left
        list_components_right = [] #list of nodes where we went right
        if ancestor.node_id == decisionTree_instance[ancestor.parent].left:     
            list_components_right.append([decisionTree_instance[ancestor.parent].input_component,
                                          decisionTree_instance[ancestor.parent].threshold])
            adv_node = decisionTree_instance[decisionTree_instance[ancestor.parent].right]
        else: # no, it is on the right
            list_components_left.append([decisionTree_instance[ancestor.parent].input_component,
                                         decisionTree_instance[ancestor.parent].threshold])
            adv_node = decisionTree_instance[decisionTree_instance[ancestor.parent].left]
        if adv_node.input_component:
            list_components_left.append([adv_node.input_component,adv_node.threshold])
        while adv_node.left or adv_node.right:
            adv_node = decisionTree_instance[adv_node.left]
            if adv_node.input_component:
                list_components_left.append([adv_node.input_component,adv_node.threshold])
        previous_ancestor = ancestor
        ancestor = decisionTree_instance[ancestor.parent]
    return previous_ancestor, adv_node, list_components_left, list_components_right

def generate_dt_adv(X, mask, dt, eps):
    X_adv = np.copy(X)
    c = 0
    for i in range(len(X)):
        x = X_adv[i]
        previous_ancestor, adv_node, l, r = find_adv(dt, x)
        for a in l:
            [pixel, thres] = a
            if pixel>0:
                x[pixel] = min(x[pixel], thres - 1e-3)
        for a in r:
            [pixel, thres] = a
            if pixel>0:
                x[pixel] = max(x[pixel], thres + 1e-3)
        if mask[i]:
            delta = x - X[i]
            delta = normalize(delta.reshape(1,-1))*eps
            delta = delta[0]
            x = X[i]+delta
            X_adv[i] = X[i]+delta
            for a in l:
                [pixel, thres] = a
                if pixel>0:
                    x[pixel] = min(x[pixel], thres - 1e-3)
            for a in r:
                [pixel, thres] = a
                if pixel>0:
                    x[pixel] = max(x[pixel], thres + 1e-3)
            delta = x - X[i]
            if np.linalg.norm(delta) <= eps:
                X_adv[i] = x
        else:
            X_adv[i] = X[i]
    print X_adv.shape
    return X_adv

# LR attacks
def generate_lr_adv(X, eps, lr_clf, mask):
    X_adv = np.copy(X)
    w = lr_clf.coef_
    prob = lr_clf.predict_proba(X)[:,1]
    direction = np.array([1 if p<0.5 else -1 for p in prob])
    grad = (prob*(1-prob)*direction).reshape([len(X), 1])
    grad = np.repeat(grad, len(X[0]), axis=1)
    ws = np.repeat(w, len(X), axis=0)
    grad = grad * ws
    X_adv = X_adv+normalize(grad)*eps
    for i in range(len(X)):
        if mask[i]:
            pass
        else:
            X_adv[i] = X[i]
    return X_adv


# WB attacks

def find_adv_direction(y_train, y_pts, train, pts, mapping):
    '''
        This function finds the adversarail perturbation direction for each test point in pts.
        Args:
            train: the training set of the nn classifier
            y_train: the label of train
            pts:   the set of testing inputs
            y_pts: the label of pts
            mapping: the function that maps train to model input, e.g. ISOMAP, LLE.
    ''' 
    if mapping != 'noMapping':
        train2 = mapping(train)
        pts2 = mapping(pts)
    else:
        train2 = train
        pts2 = pts
        
    n_pts = len(pts)
    dim = len(pts[0])
    n_train = len(train)
    direction = np.zeros((n_pts, dim))
    
    distance = np.zeros(n_pts)
    sd1 = 0
    sd2 = 0
    c = 0
    
    for i in range(n_pts):
        p1 = pts2[i]
        min_d1 = 10000
        min_d2 = 10000
        close_same = p1
        close_opposite = p1
        for j in range(n_train):
            p2 = train2[j]
            d = np.linalg.norm(p1 - p2)
            if y_pts[i] != y_train[j]:
                if d < min_d2:
                    min_d2 = d
                    close_opposite = train[j]
            else:
                if d < min_d1:
                    min_d1 = d
                    close_same = train[j]
        c += 1
        sd1 += min_d1
        sd2 += min_d2
        temp = close_opposite - p1 # move to the closest oppositely labeled points.
        if np.linalg.norm(temp) != 0:
            temp /= np.linalg.norm(temp)
        if min_d2 > min_d1:
            direction[i] = temp
        distance[i] = min_d2
    
    return [np.array(direction), distance]

def generate_adv(eps, direction, pts, distance, mode='attack'):
    n = len(pts)
    dim = len(pts[0])
    adv_pts = np.zeros((n, dim))
    for i in range(len(pts)):
        if distance[i] > eps:
            adv_pts[i] = pts[i] + (direction[i] * eps)
        elif mode=='aug':
            adv_pts[i] = pts[i] + (direction[i] * eps)
        else:
            adv_pts[i] = pts[i] + (direction[i] * distance[i])
    return np.array(adv_pts)

def generate_wb_adv(eps, train, test, y_train, y_test, mapping, mode='attack'):
    [direction, distance] = find_adv_direction(y_train, y_test, train, test, mapping)
    return generate_adv(eps, direction, test, distance, mode)


def generate_adversarial_examples(FLAG, eps, X_train, X_test, y_train, y_test, X_train_bb, y_train_bb, mapping, clf=None, mask=None, mode='attack'):
    if FLAG == 'wb':
        res = generate_wb_adv(eps, X_train, X_test, y_train, y_test, mapping, mode)
    elif FLAG == 'dt':
        res = generate_dt_adv(X_test, mask, clf, eps)
    elif FLAG == 'lr':
        res = generate_lr_adv(X_test, eps, clf, mask)
    elif FLAG == 'kernel':
        res = clf.generate_adv(X_test, y_test, eps, mask)
    elif FLAG == 'nn':
        res = clf.generate_adv(X_test, eps, mask)
    elif FLAG == 'wb_kernel':
        res = clf.generate_adv(X_test, y_test, eps, mask)
    return np.nan_to_num(np.clip(res, -100, 100))

