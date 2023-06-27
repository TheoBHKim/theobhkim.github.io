import numpy as np
import random

from scipy import stats
from sklearn.metrics import r2_score, accuracy_score


def gini(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1 - np.sum( p**2 )

def find_best_split(X, y, loss, min_samples_leaf, max_features):
    best = {'col':-1, 'split':-1, 'loss':loss(y)}
    features = np.random.choice(np.arange(len(X[0])), size = max_features, replace=False)
    for col in features:
        candidates = np.random.choice(X[:, col], size = 11, 
                                      replace = True)
        for sp in candidates:
            lhs = X[:, col] < sp
            rhs = X[:, col] >= sp
            yl = y[lhs]
            yr = y[rhs]
            if len(yl) < min_samples_leaf or len(yr) < min_samples_leaf:
                continue
            l = (len(yl)*loss(yl) + len(yr)*loss(yr))/(len(y))
            if l == 0:
                return col, sp
            if l < best['loss']:
                best = {'col':col, 'split':sp, 'loss':l}
    return best['col'],best['split']


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
    
    def leaf(self, x_test):
        """
        Given a single test record, x_test, return the leaf node reached 
        by running it down the tree starting at this node.  This is just 
        like prediction, except we return the decision tree leaf rather 
        than the prediction from that leaf.
        """
        if x_test[self.col] < self.split:
            return self.lchild.leaf(x_test)
        return self.rchild.leaf(x_test)
    
    
class LeafNode:
    
    def __init__(self, y, prediction):
        self.n = len(y)
        self.y = y
        self.prediction = prediction
        
    def predict(self, x_test):
        return self.prediction
    
    def leaf(self, x_test):
        return self


class DecisionTree621: 
    
    def __init__(self, min_samples_leaf=1, max_features=3, loss=None):
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.loss = loss
        
    def fit(self, X, y):
        self.root = self.fit_(X,y)
        
    def fit_(self, X, y):
        if (X.shape[1] <= self.min_samples_leaf) or (np.unique(X, axis=0).shape[1] ==1): 
            return self.create_leaf(y) #LeafNode(y, np.mean(y))
        col, split = find_best_split(X, y, self.loss, self.min_samples_leaf, self.max_features)
        if col == -1:
            return self.create_leaf(y)
        lchild = self.fit_(X[X[:,col]<split],y[X[:,col]<split])
        rchild = self.fit_(X[X[:,col]>=split],y[X[:,col]>=split])
        return DecisionNode(col, split, lchild, rchild)  
    
    def predict(self, X_test):
        return np.asarray([self.root.predict(i) for i in X_test])
    
    def returnleaf(self, X_test):
        return self.root.leaf(X_test)
    

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