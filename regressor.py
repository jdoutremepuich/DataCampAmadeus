from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import numpy as np



class Regressor(BaseEstimator):
    def __init__(self):
        self.clf = AdaBoostRegressor(RandomForestRegressor(n_estimators=100, max_depth=40, max_features=25), n_estimators=100)
        #self.clf_Boost = GradientBoostingRegressor( n_estimators = 500 , max_features = 20 )
        #self.clf_Regression = LinearRegression()
        

    def fit(self, X, y):
        self.clf.fit(X,y)

    def predict(self, X):
        return self.clf.predict(X)
