#!/usr/bin/python

import os

import matplotlib
matplotlib.use('Agg')
import pylab as pl
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation

def create_html_page_of_plots(list_of_plots):
    if not os.path.exists('html'):
        os.makedirs('html')
    os.system('mv *.png html')
    #print(list_of_plots)
    idx = 0
    htmlfile = open('html/index_0.html', 'w')
    htmlfile.write('<!DOCTYPE html><html><body><div>\n')
    for plot in list_of_plots:
        if idx > 0 and idx % 200 == 0:
            htmlfile.write('</div></html></html>\n')
            htmlfile.close()
            htmlfile = open('html/index_%d.html' % (idx//200), 'w')
            htmlfile.write('<!DOCTYPE html><html><body><div>\n')
        htmlfile.write('<p><img src="%s"></p>\n' % plot)
        idx += 1
    htmlfile.write('</div></html></html>\n')
    htmlfile.close()

def get_plots(in_df):
    list_of_plots = []
    #print in_df.columns

    for c in in_df.columns:
        pl.clf()
        v = in_df[c][in_df[c].astype(int) != -1]
        nent = len(v)
        if nent < 5:
            continue
        if v.dtype == np.object:
            v = v.map(ord)
        hmin, hmax = v.min(), v.max()
        if np.isnan(hmin) or np.isnan(hmax):
            continue
        if nent > 10000:
            nent = nent//500
        elif nent > 5000:
            nent = nent//250
        elif nent > 1000:
            nent = nent//50
        elif nent > 500:
            nent = nent//25
        if hmin < hmax:
            xbins = np.linspace(hmin,hmax,nent)
            a = v.values
            try:
                pl.hist(a, bins=xbins, histtype='step')
            except ValueError:
                print xbins, hmin, hmax, nent, v.dtype
                print np.isnan(hmin), np.isnan(hmax)
                exit(0)
            except IndexError:
                print xbins, hmin, hmax, nent, v.dtype
                print np.isnan(hmin), np.isnan(hmax)
                exit(0)
            pl.title(c)
            pl.savefig('%s.png' % c)
            list_of_plots.append('%s.png' % c)
    create_html_page_of_plots(list_of_plots)

def cleanup_data(in_df, drop_missing=True):
    drop_list = []
    for c in in_df.columns:
        if c.find('c_') == 0 or c == 'release':
            n = len(in_df[c])
            m = len(in_df[c][in_df[c].isnull()])
            if n == m:
                drop_list.append(c)
                continue
            if drop_missing:
                in_df[c] = in_df[c].fillna(chr(0x60))
            in_df[c] = in_df[c].map(lambda x: ord(x)-0x61).astype(np.int64)
            #if m > 0:
                #print c, n, m, in_df[c].min(), in_df[c].max()
        elif c.find('n_') == 0:
            n = len(in_df[c])
            m = len(in_df[c][in_df[c].isnull()])
            if n == m:
                drop_list.append(c)
                continue
            if drop_missing:
                in_df[c] = in_df[c].fillna(-1.0)
            in_df[c] = in_df[c].astype(np.float64)
            #if m > 0:
                #print c, n, m, in_df[c].min(), in_df[c].max()
        elif c.find('o_') == 0:
            n = len(in_df[c])
            m = len(in_df[c][in_df[c].isnull()])
            if n == m:
                drop_list.append(c)
                continue
            if drop_missing:
                in_df[c] = in_df[c].fillna(-1).astype(np.int64)
            #if m > 0:
                #print c, n, m, in_df[c].min(), in_df[c].max()
    return in_df, drop_list
    #return in_df.drop(labels=drop_list, axis=1)

def load_data():
    train_df_labels = pd.read_csv('train_labels.csv')
    train_df_values = pd.read_csv('train_values.csv', low_memory=False)
    test_df_labels = pd.read_csv('SubmissionFormat.csv')
    test_df_values = pd.read_csv('test_values.csv', low_memory=False)

    #print train_df_labels.columns
    #print train_df_values.columns
    #print test_df_labels.columns
    #print test_df_values.columns

    train_df_values, train_drop_list = cleanup_data(train_df_values)
    test_df_values, test_drop_list = cleanup_data(test_df_values)

    train_df_values = train_df_values.drop(labels=train_drop_list+test_drop_list, axis=1)
    test_df_values = test_df_values.drop(labels=train_drop_list+test_drop_list, axis=1)

    #get_plots(train_df_values)

    #print train_df_labels.columns
    #print test_df_labels.columns

    #print 'train', train_df_labels.shape, train_df_values.shape
    #print 'test', test_df_labels.shape, test_df_values.shape

    #for c in train_df_values.columns:
        #print c, train_df_values[c].dtype

    xtrain = train_df_values.values[:,1:]
    ytrain = train_df_labels.values[:,1]
    xtest = test_df_values.values[:,1:]
    ytest = test_df_labels

    print xtrain.shape, ytrain.shape, xtest.shape, ytest.shape

    return xtrain, ytrain, xtest, ytest

def calculate_log_loss(ypred, ytest):
    if ypred.shape != ytest.shape:
        return False
    n = ypred.shape[0]
    ypred = ypred - ypred.min() # make ypred.min() == 0
    ypred = ypred / ypred.max() # make ypred.max() == 1
    log_loss = (-1/n) * np.sum(ytest * np.log(ypred+1e-9) + (1-ytest) * np.log(1-ypred+1e-9))
    return log_loss

def score_model(model, xtrain, ytrain):
    randint = reduce(lambda x,y: x|y, [ord(x)<<(n*8) for (n,x) in enumerate(os.urandom(4))])
    xTrain, xTest, yTrain, yTest = \
      cross_validation.train_test_split(xtrain, ytrain, test_size=0.4,
                                        random_state=randint)
    model.fit(xTrain, yTrain)
    #cvAccuracy = np.mean(cross_val_score(model, xtrain, ytrain, cv=2))
    ytest_pred = model.predict(xTest)
    print 'logloss', calculate_log_loss(ytest_pred, yTest)
    print ytest_pred
    print yTest
    #print 'rmsle', calculate_rmsle(ytest_pred, yTest)
    return model.score(xTest, yTest)

def prepare_submission(model, xtrain, ytrain, xtest, ytest):
    model.fit(xtrain, ytrain)
    ytest_pred = model.predict(xtest)

    for idx in range(ord('a'), ord('n')+1):
        c = 'service_%s' % chr(idx)
        ytest[c] = ytest_pred[:,idx-ord('a')]

    ytest.to_csv('submit.csv', index=False)

if __name__ == '__main__':
    xtrain, ytrain, xtest, ytest = load_data()


    model = RandomForestClassifier(n_estimators=50, n_jobs=-1)
    #model = LogisticRegression(class_weight='auto')
    #model = SGDRegressor()
    print score_model(model, xtrain, ytrain)

    #prepare_submission(model, xtrain, ytrain, xtest, ytest)
