import numpy as np
# from sklearn import preprocessing
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn import linear_model
import dataHandler as dh
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import pandas as pd
from sklearn import svm, preprocessing, cross_validation, linear_model
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.model_selection import cross_val_score
# from sklearn.grid_search import GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, PolynomialFeatures
# import csv
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
# import seaborn as sns


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


def getPCA(X, n, pca):
    # get principal components
    print "GET PCAf"
    print X[:18]
    print X.shape
    Xmat = scale(X)
    print Xmat[:18]

    '''pca = PCA(n_components=n)
    X_r = pca.fit(Xmat).transform(Xmat)'''

    X_r = pca.transform(Xmat)

    print('PCA explained variance ratio (first few components): %s'
          % str(pca.explained_variance_ratio_))
    print X_r
    print type(X_r)

    Xpca = pd.DataFrame()
    for i in range(0,n):
        Xpca[i] = X_r[:,i]
    return Xpca

def returnPCA(queryX, X, n):
    pca = PCA(n_components=n)
    pca.fit(scale(X))
    print "getPCA Xpca"
    Xpca = getPCA(X,n, pca)
    print "getPCA queryXPCA", queryX[:18]
    queryXPCA = getPCA(queryX, n, pca)
    return queryXPCA, Xpca


class SVMRegression:
    # Constructor
    def __init__(self, indep_vars, dep_var):
        #self.indep_vars = self.poly.fit_transform(indep_vars)
        #self.initial_vars = indep_vars
        self.indep_vars = indep_vars
        self.dep_var = dep_var
        self.predictions = 0;

    def train(self, kerneltype):
        # training the SVM regression model
        self.poly_regr_model = svm.SVR(kernel=kerneltype, C=100000)
        self.poly_regr_model.fit(self.indep_vars, self.dep_var)



    def predict(self, query):
        # calculating the prediction
        self.predictions = self.poly_regr_model.predict(query)
        return self.predictions

    def getScore(self, pred):
        # the s-value score of the model (percentage of explained variance of the predictions)
        #return self.poly_regr_model.score(self.indep_vars, self.dep_var)
        r2 = self.poly_regr_model.score(self.indep_vars, self.dep_var)
        return np.std(pred)*np.sqrt(1-r2)

    def getSValue(self, pred, queryY):
        # Definition: http://mtweb.mtsu.edu/stats/regression/level3/indicator/useminitabinterp.htm
        # Examples:
        # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
        # https://machinelearningmastery.com/metrics-evaluate-machine-learning-algorithms-python/
        # https://stackoverflow.com/questions/17197492/root-mean-square-error-in-python
        meanSq = mean_squared_error(queryY, pred)
        sValue = np.sqrt(meanSq)
        #print "meanSq is ", meanSq
        #print "sValue is ", sValue
        return sValue  

    def getCoef(self):
        # Slope or m
        return self.poly_regr_model.coef_[0]

    def getIntercept(self):
        # Y intercept or b
        return self.poly_regr_model.intercept_

    def getEquation(self):
        return "y = {0}x + {1}".format(self.getCoef(), self.getIntercept())

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
