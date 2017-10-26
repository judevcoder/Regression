import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import dataHandler as dh
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


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
    def __init__(self, indep_vars, dep_var, poly_fit_degree=2):
        self.poly_fit_degree = int(poly_fit_degree)
        self.poly = PolynomialFeatures(degree=poly_fit_degree,interaction_only=False,include_bias=False)
        self.indep_vars = self.poly.fit_transform(indep_vars)
        self.initial_vars = indep_vars
        self.dep_var = dep_var
        self.predictions = 0;

    def train(self):
        # training the linear regression model
        self.poly_regr_model = linear_model.LinearRegression()
        self.poly_regr_model.fit(self.indep_vars, self.dep_var)

    def predict(self, query):
        # calculating the prediction
        query_ = self.poly.fit_transform(query.reshape(1, -1))
        self.predictions = self.poly_regr_model.predict(query_)
        return self.predictions

    def getScore(self):
        # the R^2 score of the model (percentage of explained variance of the predictions)
        return self.poly_regr_model.score(self.indep_vars, self.dep_var)

    def getCoef(self):
        # Slope or m
        return self.poly_regr_model.coef_[0]

    def getIntercept(self):
        # Y intercept or b
        return self.poly_regr_model.intercept_

    def getEquation(self):
        return "y = {0}PolynomialFeatures(x) + {1}".format(self.getCoef(), self.getIntercept())

    def showGraph(self):
        print self.initial_vars
        print self.dep_var
        print self.predictions[0, 0]
        fig = Figure()
        plt.scatter(self.initial_vars, self.dep_var, color='black')
        #plt.plot(self.initial_vars, self.predictions[0, 0], color='blue', linewidth=3)

        fit = polyfit(self.initial_vars, self.dep_var, 1)
        fit_fn = poly1d(fit)
        plt.plot(self.initial_vars, y, '*', self.initial_vars, fit_fn(self.initial_vars), 'k')
        canvas = FigureCanvas(fig)
        output = StringIO.StringIO()
        canvas.print_png(output)
        response = make_response(output.getvalue())
        response.mimetype = 'image/png'

        plt.show()
        return output
