#!/usr/bin/python

import os

import cPickle as pickle
import gzip

import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from sklearn.svm import SVC, NuSVC
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.metrics import log_loss

from load_data import load_data

def scorer(estimator, X, y):
    yprob = estimator.predict_proba(X)
    return 1.0 / log_loss(y, yprob)

def score_model(model, xtrain, ytrain):
    randint = reduce(lambda x,y: x|y, [ord(x)<<(n*8) for (n,x) in enumerate(os.urandom(4))])
    xTrain, xTest, yTrain, yTest = \
      cross_validation.train_test_split(xtrain, ytrain, test_size=0.4,
                                        random_state=randint)
    #model.fit(xTrain, yTrain)
    model.fit(xTrain, yTrain[:,0])
    print model
    ytest_pred = model.predict(xTest)
    ytest_prob = model.predict_proba(xTest)
    print ytest_prob.shape, yTest[:,n].shape
    print 'logloss', log_loss(yTest[:,n], ytest_prob)
    with gzip.open('model.pkl.gz', 'wb') as mfile:
        pickle.dump(model, mfile, protocol=2)

def train_model_parallel(model, xtrain, ytrain, index):
    randint = reduce(lambda x,y: x|y, [ord(x)<<(n*8) for (n,x) in enumerate(os.urandom(4))])
    xTrain, xTest, yTrain, yTest = \
      cross_validation.train_test_split(xtrain, ytrain[:,index], test_size=0.4,
                                        random_state=randint)
    n_est = [10, 100, 200]
    m_dep = [5, 10, 40]
    #param_grid = [{'n_estimators': 10, 'max_depth': 5},
                  #{'n_estimators': 10, 'max_depth': 10},
                  #{'n_estimators': 10, 'max_depth': 40},
                  #{'n_estimators': 100, 'max_depth': 5},
                  #{'n_estimators': 100, 'max_depth': 10},
                  #{'n_estimators': 100, 'max_depth': 40},
                  #{'n_estimators': 200, 'max_depth': 5},
                  #{'n_estimators': 200, 'max_depth': 10},
                  #{'n_estimators': 200, 'max_depth': 40},]

    #select = RFECV(estimator=model, scoring=scorer, verbose=0, step=0.1)
    model = GridSearchCV(estimator=model,
                                param_grid=dict(n_estimators=n_est, max_depth=m_dep),
                                scoring=scorer,
                                n_jobs=-1, verbose=1)
    model.fit(xTrain, yTrain)
    print model
    
    ytest_prob = model.predict_proba(xTest)
    print 'logloss', log_loss(yTest, ytest_prob)
    with gzip.open('model_%d.pkl.gz' % index, 'wb') as mfile:
        pickle.dump(model, mfile, protocol=2)
    return

def test_model_parallel(xtrain, ytrain):
    randint = reduce(lambda x,y: x|y, [ord(x)<<(n*8) for (n,x) in enumerate(os.urandom(4))])
    xTrain, xTest, yTrain, yTest = \
      cross_validation.train_test_split(xtrain, ytrain, test_size=0.4,
                                        random_state=randint)
    ytest_prob = np.zeros((yTest.shape[0], yTest.shape[1], 2))
    sum_log_loss = 0
    for n in range(14):
        with gzip.open('model_%d.pkl.gz' % n, 'rb') as mfile:
            model = pickle.load(mfile)
            print 'grid scores', model.grid_scores_
            print 'best score', model.best_score_
            print 'best params', model.best_params_
            pyt = model.predict_proba(xTest)
            ytest_prob[:,n,:] = pyt
            sum_log_loss += log_loss(yTest[:,n], pyt)
    print sum_log_loss

def prepare_submission_parallel(xtrain, ytrain, xtest, ytest):
    for n in range(14):
        with gzip.open('model_%d.pkl.gz' % n, 'rb') as mfile:
            model = pickle.load(mfile)
            ytest_prob = model.predict_proba(xtest)
            label = 'service_%s' % (chr(ord('a')+n))
            ytest[label] = ytest_prob[:,1]
    ytest.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    xtrain, ytrain, xtest, ytest = load_data()


    #model = SGDClassifier(loss='log', n_jobs=-1, penalty='l1', verbose=1, n_iter=200)
    model = GradientBoostingClassifier(loss='deviance', verbose=1)

    index = -1
    for arg in os.sys.argv:
        try:
            index = int(arg)
            break
        except ValueError:
            continue
    if index == -1:
        score_model(model, xtrain, ytrain)
    elif index >= 0 and index < 14:
        train_model_parallel(model, xtrain, ytrain, index)
    elif index == 14:
        test_model_parallel(xtrain, ytrain)
    elif index == 15:
        prepare_submission_parallel(xtrain, ytrain, xtest, ytest)
