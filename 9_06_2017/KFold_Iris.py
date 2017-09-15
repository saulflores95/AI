import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import tflearn
from sklearn.model_selection import train_test_split
import pandas as pd
#from tflearn.metrics import accuracy_op
from sklearn.metrics import accuracy_score


iris = datasets.load_iris()
X = iris.data
y = iris.target

y = pd.get_dummies(y)

df_x = pd.DataFrame(X, columns=iris.feature_names)
print("Entradas\n", df_x.describe())
print(df_x.head())

df_y = pd.DataFrame(y)
print("Salidas\n", df_y.describe())
print(df_y.head())

train, test = train_test_split(df_x, test_size = 0.2)

print(df_x.loc[train.index])

def train_step(X, Y):

    #Modelo
    tnorm = tflearn.initializations.uniform(minval=-1., maxval=1.)
    input_ = tflearn.input_data(shape=[None, 4])
    hidden = tflearn.fully_connected(input_, n_units=1, activation='linear', weights_init=tnorm)
    output = tflearn.fully_connected(hidden, n_units=3, activation=None, name='output',  weights_init=tnorm)

    regression = tflearn.regression(output, optimizer='sgd', loss='mean_square',
                                    metric='accuracy', learning_rate=.7)

    #loss -> roc_auc_score

    #Entrenamiento
    m = tflearn.DNN(regression)
    m.fit(X, Y, n_epoch=500, show_metric=True, snapshot_epoch=False)
    return m

def test_step(model, x_test, y_true):
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)
    acc_op = accuracy_score(y_pred, y_true)
    return acc_op


model = train_step(df_x.loc[train.index].values, df_y.loc[train.index].values)
acc = test_step(model, df_x.loc[test.index].values, df_y.loc[test.index].values)
print("Accuracy: ", acc)
