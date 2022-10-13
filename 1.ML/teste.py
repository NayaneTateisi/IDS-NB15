from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
import pickle
import warnings
import time
warnings.filterwarnings("ignore")

with open('dataset-nids.pkl','rb') as f:
    X_train, y_train, X_test, y_test = pickle.load(f)

space = {
"hidden_layer_sizes": [(4),(4,4),(4,3)],
"activation":['tanh','relu'],
"solver": ['sgd','adam'],
"tol": [0.001, 0.0001,0.0001],
"alpha": [0.001,0.5]
}

rna = MLPClassifier(verbose=True)
t1 = time.time()
clr = GridSearchCV(rna, space,n_jobs= -1, cv = 3)
clr.fit(X_train, y_train)

params = clr.best_params_
score = clr.best_score_
timeExec = time.time() - t1

print(params)
print(score)
print('Tempo de execução: {}segundos'.format(timeExec))


