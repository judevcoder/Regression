
# A very simple Flask Hello World app for you to get started with...

from flask import Flask, url_for, redirect, request,make_response
from datetime import datetime
import json
import regr
import simpleRegression
import simpleNonlinearRegression
import simpleSVMRegression
from flask_cors import CORS, cross_origin
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

@app.route('/')

#-------------------------------------------------------
# Set up Error Pages
#-------------------------------------------------------
@app.errorhandler(403)
def pageForbidden(error):
    return json.dumps({'error':403,  'msg':'this is forbidden', 'data':{}})

@app.errorhandler(404)
def pageNotFound(error):
    return json.dumps({'error':404,  'msg':'this is not found', 'data':{}})

@app.errorhandler(410)
def pageGone(error):
    return json.dumps({'error':410,  'msg':'this is gone', 'data':{}})

@app.errorhandler(500)
def InternalServerError(error):
    return json.dumps({'error':500,  'msg':'this is not valid json', 'data':{}})

#-------------------------------------------------------
# Helper Functions
#-------------------------------------------------------
def parseRequest(request):
    data = request.json
    query = simpleRegression.parseQuery(data[u'query'])
    trainData = data[u'trainData']
    return trainData, query

#-------------------------------------------------------
# Our Routes
#-------------------------------------------------------

# Show our upload page
# @app.route('/test', methods=['GET'])
# #@cross_origin()
# def test():
#     return app.send_static_file('upload.html')

'''# Our Nonlinear Regression Call
@app.route('/nonlin', methods=['POST'])
def nonlin():
    trainData, query = parseRequest(request)
    X, Y = simpleNonlinearRegression.parseDict(trainData) # Convert our data in XY values
    poly = simpleNonlinearRegression.NonlinearRegression(X,Y)
    poly.train()
    # Your data goes here
    score = poly.getScore()
    pred = poly.predict(query)# Non-linear regression prediction
    pred = pred[0,0]
    coefs = poly.getCoef()# Non-linear regression coefficients
    formula =poly.getEquation() # Non-linear regression formula
    #chart = poly.showGraph()# The URL for the plotted chart, which is an IMG that was saved to the server

    return json.dumps({'error': 200, 'msg': 'success','data': {'prediction': pred,'score':score,'formula': formula}},sort_keys=True), 200
'''

# Our Nonlinear Regression Call
@app.route('/nonlin', methods=['POST'])
def nonlin():
    print "Nonlinear StatsModel Regression"
    trainData, query = parseRequest(request)
    print "query: ", query
    print "trainData: ", trainData
    X, Y = simpleSVMRegression.parseDict(trainData) # Convert our data in XY values
    print "X: ", X
    print "Y: ", Y
    pred = 0
    formula = ''
    r2 = 0
    sValue = 0

    # Your Code Goes here
    return json.dumps({'error': 200, 'msg': 'success','data': {'prediction': pred, 'formula':formula, 'r2':r2, 's_value':sValue}},sort_keys=True), 200

# Our Nonlinear Regression Call
@app.route('/nonlinold', methods=['POST'])
def nonlinold():
    print "SV Regression"
    trainData, query = parseRequest(request)
    print "query: ", query
    print "trainData: ", trainData
    X, Y = simpleSVMRegression.parseDict(trainData) # Convert our data in XY values
    print "X: ", X
    print "Y: ", Y

    queryY = np.array([query[1]])  # This should be the list price of the query, this sometimes changes so watch it
    # If the position is 0 then the below works
    # queryX = query[1:]
    # Since its not 0 we need to do something else
    print "queryY ", queryY
    firstPart = query[0]
    secondPart = query[2:]
    '''print type(query)
    print type(firstPart)
    print "1stpart", firstPart
    print "2ndpart ", secondPart[:18]
    queryX = firstPart + secondPart'''
    queryX = np.append(firstPart,secondPart)
    print "queryX ", queryX
    print "typeX", type(X)
    print X[0,:]
    print "y? ",X[:,1]
    print "newX"
    firstX = X[:,0]
    secondX = X[:, 2:]
    Xpd = pd.DataFrame(X)
    print "before ", Xpd.head()
    Xpd.drop(Xpd.columns[[1]], axis=1, inplace=True)
    print "after ",Xpd.head()
    X = Xpd.as_matrix()
    #X= np.append(firstX,secondX)
    print "shapeX ", X.shape, "ShapequeryX", queryX.shape
    queryPCA, XPCA = simpleSVMRegression.returnPCA(queryX, X, 1)
    poly = simpleSVMRegression.SVMRegression(XPCA, Y)
    poly.train('linear')

    '''X = simpleSVMRegression.getPCA(X, 1) #specify number of principal components with n_components
    poly = simpleSVMRegression.SVMRegression(X,Y)
    poly.train('linear')
    # Your data goes here
    #queryY=query['close_price']
    #locOfY = query.index("close_price")

    queryY=np.array([query[1]]) # This should be the list price of the query, this sometimes changes so watch it
    # If the position is 0 then the below works
    #queryX = query[1:]
    # Since its not 0 we need to do something else
    firstPart = query[0]
    secondPart = query[2:]
    queryX = firstPart + secondPart
    #queryX.drop(query.columns[[0]], axis=1, inplace=True)
    queryPCA = simpleSVMRegression.getPCA(queryX, 1)'''
    pred = poly.predict(queryPCA)# Non-linear regression prediction
    #score = poly.getScore(pred)
    print "query: ", query
    print "queryX: ", queryX
    print "pred: ", pred
    print "predtype ", pred.shape
    print "queryY: ", queryY
    print "queryYtype ", queryY.shape
    r2 = poly.getScore(pred)
    sValue = poly.getSValue(pred, queryY)
    pred = pred[0]
    coefs = poly.getCoef()# Non-linear regression coefficients
    formula =poly.getEquation() # Non-linear regression formula
    #chart = poly.showGraph()# The URL for the plotted chart, which is an IMG that was saved to the server
    print "pred is ", pred
    print "r2 is ", r2
    print "sValue is ", sValue
    print "coefs is ", coefs
    print "formula is ", formula
    return json.dumps({'error': 200, 'msg': 'success','data': {'prediction': pred, 'formula':formula, 'r2':r2, 's_value':sValue}},sort_keys=True), 200

# Our Linear Regression Call
@app.route('/lin', methods=['POST'])
def lin():
    trainData, query = parseRequest(request)
    X, Y = regr.parseDict(trainData)
    print "Y: ", Y
    reg = regr.Regr(X, Y, poly_fit_degree=1)
    reg.train()
    pred = reg.predict(query)
    pred = pred[0,0]
    print "pred: ", pred
    s_err = reg.getSS ()
    js_err = {'%d' % i: '%s' % s_err[i] for i in range (len (s_err))}
    #print(js_err)
    return json.dumps ({'error': 200, 'msg': 'success', 'data': {'prediction': pred, 'serror': js_err, 'r2': reg.getScore(), 'formula': reg.getEquation()}}, sort_keys=True), 200

# Our Linear Regression Call
@app.route('/linalt', methods=['POST'])
def linalt():
    # Pull our passed data and process it
    trainData, query = parseRequest(request)
    X, Y = simpleRegression.parseDict(trainData) # Convert our data in XY values
    regObj = simpleRegression.simpleRegression(X, Y)
    regObj.train()
    pred = regObj.predict(query)
    pred = pred[0,0]
    r2 = regObj.getScore()
    formula = regObj.getEquation()
    print(r2)
    print(pred)
    print( formula )
    #regObj.showGraph()

    #return str(np.round(pred[0,0],4)) + ' ' + regObj.getEquation()
    #s_err = regObj.getSS ()
    #js_err = {'%d' % i: '%s' % s_err[i] for i in range (len (s_err))}
    #print(js_err)
    return json.dumps ({'error': 200, 'msg': 'success', 'data': {'prediction': pred, 'r2': r2, 'formula': formula}}, sort_keys=True), 200

if __name__ == "__main__":
    app.run()