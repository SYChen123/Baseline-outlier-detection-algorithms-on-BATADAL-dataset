import numpy as np
import os
import pickle

from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.xgbod import XGBOD

from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# calculate roc curve and auc
def compute_roc_auc(y_true, x, model):
    prob = model.predict_proba(x)
    if len(prob.shape) > 1:
        prob = prob[:,1]
    fpr,tpr,thresholds = roc_curve(y_true, prob)
    auc = roc_auc_score(y_true, prob)
    return auc, fpr, tpr

def compute_scores(Y,Yhat):
    return [accuracy_score(Y,Yhat),f1_score(Y,Yhat),precision_score(Y,Yhat),recall_score(Y,Yhat)]

def save(model, model_dir, file_name):
    model_path = os.path.join(model_dir, file_name)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

def load(model_dir, file_name):
    model_path = os.path.join(model_dir, file_name)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def train_model(X, Y, contamination, name, from_scratch=True):
    model_dir = './model'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    file_name = name+'.pkl'

    if from_scratch:
        if name == 'ocsvm':
            model = OCSVM(contamination=contamination)
            model.fit(X)
        elif name == 'iforest':
            model = IForest(contamination=contamination)
            model.fit(X)
        elif name == 'lof':
            model = LOF(contamination=contamination)
            model.fit(X)
        elif name == 'knn':
            model = KNN(contamination=contamination)
            model.fit(X)
        elif name == 'xgbod':
            model = XGBOD(contamination=contamination)
            model.fit(X,Y)

        save(model, model_dir, file_name)

    else:
        model = load(model_dir, file_name)
    
    return model
    