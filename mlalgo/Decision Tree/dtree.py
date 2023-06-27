import numpy as np
import random

from scipy import stats
from sklearn.metrics import r2_score, accuracy_score


class DecisionNode:
    
    def __init__(self, col, split, lchild, rchild):
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild

    def predict(self, x_test):
        if x_test[self.col] < self.split:
            return self.lchild.predict(x_test)
        return self.rchild.predict(x_test)
    
    
class LeafNode:
    
    def __init__(self, y, prediction):
        self.n = len(y)
        self.prediction = prediction

    def predict(self, x_test):
        return self.prediction


def gini(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1 - np.sum( p**2 )

def find_best_split(X, y, loss, min_samples_leaf):
    best = (-1, -1, loss(y))
    for i in range(X.shape[1]):
        col = X[:, i]
        if len(col) >= 11:
            splits = random.sample(list(col), 11)
        else:
            splits = col
        for j in splits:
            left = y[col < j]
            right = y[col >= j]
            if (len(left) < min_samples_leaf) or (len(right) < min_samples_leaf):
                continue
            l = ((len(left) * loss(left)) + (len(right) *loss(right)))/len(y)
            if l == 0:
                return i, j
            if l < best[2]:
                best = (i,j,l)
    return best[0], best[1]


class DecisionTree621:
    
    def __init__(self, min_samples_leaf=1, loss=None):
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss
        
    def fit(self, X, y):
        self.root = self.fit_(X,y)
        
    def fit_(self, X, y):
        if (X.shape[1] <= self.min_samples_leaf) or (np.unique(X, axis=0).shape[1] ==1): 
            return self.create_leaf(y)
        col, split = find_best_split(X, y, self.loss, self.min_samples_leaf)
        if col == -1:
            return self.create_leaf(y)
        lchild = self.fit_(X[X[:,col]<split],y[X[:,col]<split])
        rchild = self.fit_(X[X[:,col]>=split],y[X[:,col]>=split])
        return DecisionNode(col, split, lchild, rchild)
    
    def predict(self, X_test):
        return np.asarray([self.root.predict(i) for i in X_test])
    

class RegressionTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1):
        super().__init__(min_samples_leaf, loss=np.std)

    def score(self, X_test, y_test):
        return r2_score(y_test, self.predict(X_test))

    def create_leaf(self, y):
        return LeafNode(y, np.mean(y))


class ClassifierTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1):
        super().__init__(min_samples_leaf, loss=gini)

    def score(self, X_test, y_test):
        return accuracy_score(y_test, self.predict(X_test))

    def create_leaf(self, y):
        return LeafNode(y, stats.mode(y)[0][0])