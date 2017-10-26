import numpy as np
import sklearn as skl
import sklearn.preprocessing
import sklearn.linear_model
import dataHandler as dh


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
        
 
class Regr:
    def __init__(self, indep_vars, dep_var, poly_fit_degree=1):
        self.poly_fit_degree = int(poly_fit_degree)
        self.indep_vars = indep_vars
        self.dep_var = dep_var
        self.poly_max_fig_degree = 1        

    def set_poly_degree(self, poly_fit_degree):
        self.poly_fit_degree = int(poly_fit_degree)
        
    def score(self, indep_vars, dep_var):
        poly = skl.preprocessing.PolynomialFeatures(degree=self.poly_fit_degree)
        indep_vars_ = poly.fit_transform(self.indep_vars)
        return self.lin_regr_model.score(indep_vars_, self.dep_var)

    def getScore(self):
        # the R^2 score of the model (percentage of explained variance of the predictions)
        poly = skl.preprocessing.PolynomialFeatures(degree=self.poly_fit_degree)
        indep_vars_ = poly.fit_transform(self.indep_vars)
        return self.lin_regr_model.score(indep_vars_, self.dep_var)

    def getEquation(self):
        return "y = {0}x + {1}".format(self.lin_regr_model.coef_, self.lin_regr_model.intercept_)

    def train(self):
        # preprocessing for the linear regression
        poly = skl.preprocessing.PolynomialFeatures(degree=self.poly_fit_degree)
        indep_vars_ = poly.fit_transform(self.indep_vars)
        # training the linear regression model
        lin_regr = sklearn.linear_model.LinearRegression()
        lin_regr.fit(indep_vars_, self.dep_var)
        self.lin_regr_model = lin_regr
        return lin_regr
        
    def predict(self, query):
        # preprocessing for the linear regression
        poly = skl.preprocessing.PolynomialFeatures(degree=self.poly_fit_degree)
        query_ = poly.fit_transform(query)
        # calculating the prediction
        prediction = self.lin_regr_model.predict(query_.reshape(1,-1))
        return prediction
        
    def train_with_poly_degree_optimization(self):
        max_score = 0
        max_score_degree = 0
        max_score_pred = 0
        
        for i in range(1,11):
            self.set_poly_degree(i)
            model = self.train()
            score = self.score(self.indep_vars, self.dep_var)
            
            if max_score < score:
                max_score = score
                max_score_degree = i
            
            print "fit degree:", i, "\t score:", score

        self.poly_max_fig_degree = max_score_degree 
        print
        print "max score:", max_score
        print "degree:", max_score_degree
        print

    def getSS(self):
        # Initiate logistic regression object        
        poly = skl.preprocessing.PolynomialFeatures (self.poly_max_fig_degree)
        X = poly.fit_transform (self.indep_vars)
        y = self.dep_var
        # training the linear regression model
        lm = sklearn.linear_model.LinearRegression ()
        lm.fit (X, y)        
        predictions = lm.predict(X)
        newX = np.append(np.ones((len(X),1)), X, axis=1)
        newX = np.array(X)
        mse = sklearn.metrics.mean_squared_error(y, predictions)        
        var_b = mse * ((np.dot (newX.T, newX)).diagonal ())        
        sd_b = np.sqrt (var_b)        
        return sd_b

