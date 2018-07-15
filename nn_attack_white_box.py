import time
import numpy as np

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
                    #close_opposite = p2
                    close_opposite = train[j]
            else:
                if d < min_d1:
                    min_d1 = d
                    #close_same = p2
                    close_same = train[j]
        #if min_d1 < min_d2:
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

def generate_adv(eps, direction, pts, distance):
    n = len(pts)
    dim = len(pts[0])
    adv_pts = np.zeros((n, dim))
    for i in range(len(pts)):
        if distance[i] > eps:
            adv_pts[i] = pts[i] + (direction[i] * eps)
        else:
            adv_pts[i] = pts[i] + (direction[i] * distance[i])
    return np.array(adv_pts)

def generate_adversarial_examples(eps, train, test, y_train, y_test, mapping):
    [direction, distance] = find_adv_direction(y_train, y_test, train, test, mapping)
    return generate_adv(eps, direction, test, distance)


def generate_adv_aug(eps, direction, pts, distance):
    n = len(pts)
    dim = len(pts[0])
    adv_pts = np.zeros((n, dim))
    for i in range(len(pts)):
        if distance[i] > eps:
            adv_pts[i] = pts[i] + (direction[i] * eps)
        else:
            adv_pts[i] = pts[i] + (direction[i] * distance[i])
    return np.array(adv_pts)

def generate_adversarial_examples_aug(eps, train, test, y_train, y_test, mapping):
    [direction, distance] = find_adv_direction(y_train, y_test, train, test, mapping)
    return generate_adv_aug(eps, direction, test, distance)
