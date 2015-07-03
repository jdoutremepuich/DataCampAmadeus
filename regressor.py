from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
import numpy as np

class Regressor(BaseEstimator):
    def __init__(self):
        self.clf = AdaBoostRegressor(RandomForestRegressor(n_estimators=100, max_depth=70, max_features=25), n_estimators=20)
        self.clf_Boost = GradientBoostingRegressor( n_estimators = 500 , max_features = 20 )
        self.clf_Regression = LinearRegression()
        

    def fit(self, X, y):
        self.clf.fit(X,y)
        self.clf_Boost.fit(X, y)
        self.clf_Regression.fit(X, y)

    def predict(self, X):
        return np.mean(self.clf.predict(X),self.clf_Boost.predict(X),self.clf_Regression.predict(X))
