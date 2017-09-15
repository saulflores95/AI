import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import tflearn
import tensorflow as tf
import numpy as np

def train_step(X, Y):
    #Modelo
    tnorm = tflearn.initializations.uniform(minval=-1., maxval=1.)
    input_ = tflearn.input_data(shape=[None, 6])
    hidden = tflearn.fully_connected(input_, n_units=6, activation='linear', weights_init=tnorm)
    output = tflearn.fully_connected(hidden, n_units=1, activation=None, name='output',  weights_init=tnorm)

    regression = tflearn.regression(output, optimizer='sgd', loss='mean_square',
                                    metric='R2', learning_rate=.7)

    #loss -> roc_auc_score

    #Entrenamiento
    m = tflearn.DNN(regression)
    m.fit(X, Y, n_epoch=500, show_metric=True, snapshot_epoch=False)
    return m

def test_step(model, x_test, y_true):
    y_pred = model.predict(x_test)
    score = r2_score(y_true, y_pred)
    print(score)
    return score

machine = pd.read_csv('C:\\Users\\Saul\\Documents\\GitHub\\AI\\9_06_2017\\machine.data', sep=",",
                names = ["Vendor", "Model", "MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"])
keep_col = ["ERP"]
y = machine[keep_col]
x = machine.drop(["Vendor", "Model", "ERP", "PRP"], axis=1)

kf = KFold(n_splits=5, random_state=None, shuffle=False)
error = []

for train_index, test_index in kf.split(x):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = x.loc[train_index].values, x.loc[test_index].values
    y_train, y_test = y.loc[train_index].values, y.loc[test_index].values
    model = train_step(X_train, y_train)
    acc = test_step(model, X_test, y_test)
    error.append(acc)
    tf.reset_default_graph()


print("Accuracy: ", np.mean(error))
