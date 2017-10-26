import numpy as np
from sklearn import preprocessing, linear_model
import dataHandler as dh
import matplotlib.pyplot as plt


def stringToNpArr(arr_string):
    return np.fromstring(arr_string, dtype=int, sep=', ')

def parseDict(data):
    return dh.parseDict(data)

def parseQuery(data):
    query = []
    dh.X_columns_order = data.keys()
    try:
        for col in dh.X_columns_order:
            query.append(data[col])
    except KeyError as e:
        print "Query data does not contain the required columns"
        raise e
    return np.array(query)
 
class simpleRegression:
    # Constructor
    def __init__(self, indep_vars, dep_var):
        self.indep_vars = indep_vars
        self.dep_var = dep_var
        self.poly_fit_degree = 1
        self.predictions = 0

    def train(self):
        # training the linear regression model
        self.lin_regr_model = linear_model.LinearRegression()
        self.lin_regr_model.fit(self.indep_vars, self.dep_var)
        
    def predict(self, query):
        # calculating the prediction
        self.predictions = self.lin_regr_model.predict(query.reshape(1,-1))
        return self.predictions

    def getScore(self):
        # the R^2 score of the model (percentage of explained variance of the predictions)
        poly = preprocessing.PolynomialFeatures(degree=self.poly_fit_degree,interaction_only=False,include_bias=False)
        indep_vars_ = poly.fit_transform(self.indep_vars)
        #print(indep_vars_)
        #print(self.dep_var)
        # The line below throws the error ValueError: matrices are not aligned
        return self.lin_regr_model.score(indep_vars_, self.dep_var)

    def getCoef(self):
        # Slope or m
        return self.lin_regr_model.coef_

    def getIntercept(self):
        # Y intercept or b
        return self.lin_regr_model.intercept_

    def getEquation(self):
        return "y = {0}x + {1}".format(self.lin_regr_model.coef_, self.lin_regr_model.intercept_)

    # def showGraph(self):
    #     print self.indep_vars
    #     print self.dep_var
    #     print self.predictions[0,0]
    #     plt.scatter(self.indep_vars, self.dep_var,  color='black')
    #     plt.plot(self.indep_vars, self.predictions[0,0], color='blue',linewidth=3)
    #
    #     fit = polyfit(self.indep_vars, self.dep_var, 1)
    #     fit_fn = poly1d(fit)
    #     plt.plot(self.indep_vars, y, '*', self.indep_vars, fit_fn(self.indep_vars), 'k')
    #
    #     plt.show()

