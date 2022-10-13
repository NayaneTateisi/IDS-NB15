import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import warnings
import pickle
warnings.filterwarnings("ignore")


with open('normalization.pkl','rb') as f:

    X_train, y_train, X_test, y_test = pickle.load(f)


param_grid = {

    "C": [0.1,1, 10, 100],
    "gamma": [1,0.1,0.01,0.001]
}

cf = SVC(random_state=0,  max_iter=10000)
cf_cv = GridSearchCV(estimator=cf, param_grid=param_grid, cv=5, verbose=1)
cf_cv.fit(X_train, y_train)

cf_cv.best_params_