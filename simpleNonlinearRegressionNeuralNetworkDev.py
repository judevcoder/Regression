import numpy as np
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
import dataHandler as dh
import matplotlib.pyplot as plt


def stringToNpArr(arr_string):
    return np.fromstring(arr_string, dtype=int, sep=', ')


def parseDict(data):
    return dh.parseDict(data)


def parseQuery(data):
    query = []
    try:
        for col in dh.X_columns_order:
            query.append(data[col])
    except KeyError as e:
        print "Query data does not contain the required columns"
        raise e
    return np.array(query)


class NonlinearRegression:
    # Constructor
    def __init__(self, indep_vars, dep_var):
        self.indep_vars = indep_vars
        self.dep_var = dep_var

    def train(self):
        # training the linear regression model
        self.mlp_regr_model = MLPRegressor(hidden_layer_sizes=(3,),shuffle=True)
        self.mlp_regr_model.fit(self.indep_vars, self.dep_var)

    def predict(self, query):
        # calculating the prediction
        self.predictions = self.mlp_regr_model.predict(query.reshape(1, -1))
        return self.predictions

    def getScore(self):
        # the R^2 score of the model (percentage of explained variance of the predictions)
        return self.mlp_regr_model.score(self.indep_vars, self.dep_var)


    def getParameters(self):
        return "Parameters: y = {0}x ".format(self.mlp_regr_model.get_params())

    def showGraph(self):
        print self.indep_vars
        print self.dep_var
        print self.predictions[0, 0]
        plt.scatter(self.indep_vars, self.dep_var, color='black')
        plt.plot(self.indep_vars, self.predictions[0, 0], color='blue', linewidth=3)

        fit = polyfit(self.indep_vars, self.dep_var, 1)
        fit_fn = poly1d(fit)
        plt.plot(self.indep_vars, y, '*', self.indep_vars, fit_fn(self.indep_vars), 'k')

        plt.show()
