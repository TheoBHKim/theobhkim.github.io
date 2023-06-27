import numpy as np
from sklearn.utils import resample

from dtree import *

class RandomForest621:

    def __init__(self, n_estimators=10, oob_score=False):
        self.n_estimators = n_estimators
        self.oob_score = oob_score
        self.oob_score_ = np.nan
        self.reg = True
    
    def fit(self, X, y):

        self.nclasses = np.unique(y)
        trees = []
        
        if not self.oob_score:
            
            for i in range(self.n_estimators):
                Xboot, yboot = resample(X,y)
                i = RegressionTree621()
                i.fit(Xboot, yboot)
                trees.append(i)
            
        else:
            dic = {lis: [] for lis in range(len(y))}
            index = np.arange(len(y))
            ooblis =[]
            
            for i in range(self.n_estimators):
                Xboot, yboot, indboot = resample(X,y,index)
                ooblis.append(list(set(index).difference(indboot)))
                i = RegressionTree621()
                i.fit(Xboot, yboot)
                trees.append(i)
            
            for tree in range(len(trees)):
                for ind in ooblis[tree]:
                    dic[ind].append(tree)
            
            predictions = []
            nooob = []
            
            #for regressor
            if self.reg:
                for oob in range(len(dic)):
                    if len(dic[oob]) != 0:
                        obspres = []
                        for treeind in dic[oob]:
                            obspres.append(trees[treeind].returnleaf(X[oob]).predict(X))
                        prediction = sum(obspres)/len(obspres)
                        predictions.append(prediction)
                    else:
                        nooob.append(oob)
                actual = y
                for i in nooob:
                    actual = np.delete(actual, i)
                self.oob_score_ = r2_score(actual, predictions)
            
            #for classifier    
            else: 
                predictions = []
                nooob = []
                for oob in range(len(dic)):
                    if len(dic[oob]) != 0:
                        counts = np.zeros(len(self.nclasses))
                        for treeind in dic[oob]:
                            for value in trees[treeind].returnleaf(X[oob]).y:
                                counts[value] += 1
                        prediction = self.nclasses[np.argmax(counts)]
                        predictions.append(prediction)
                    else:
                        nooob.append(oob)
                actual = y
                for i in nooob:
                    actual = np.delete(actual, i)
                self.oob_score_ = accuracy_score(actual, predictions)                    
 
        self.trees = trees

            
class RandomForestRegressor621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, 
    max_features=3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)

    def predict(self, X_test) -> np.ndarray:
        """
        Given a 2D nxp array with one or more records, compute the 
        weighted average prediction from all trees in this forest. 
        Weight each trees prediction by the number of observations in 
        the leaf making that prediction.  Return a 1D vector
        with the predictions for each input record of X_test.
        """
        predictions = []
        for row in X_test:
            n, total = 0, 0
            for tree in self.trees:
                n += tree.returnleaf(row).n
                for value in tree.returnleaf(row).y:
                    total += value
            prediction = total/n
            predictions.append(prediction)
        return predictions
        
    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or 
        more records, collect the prediction for each record and then 
        compute R^2 on that and y_test.
        """
        return r2_score(y_test, self.predict(X_test))
        

class RandomForestClassifier621(RandomForest621):
    
    def __init__(self, n_estimators=10, min_samples_leaf=3, 
    max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.reg = False

    def predict(self, X_test) -> np.ndarray:
        predictions = []
        classes = self.nclasses
        for row in X_test:
            counts = np.zeros(len(classes))
            for tree in self.trees:
                for value in tree.returnleaf(row).y:
                    counts[value] += 1
            prediction = classes[np.argmax(counts)]
            predictions.append(prediction)
        return predictions
        
    def score(self, X_test, y_test) -> float:
        return accuracy_score(y_test, self.predict(X_test))