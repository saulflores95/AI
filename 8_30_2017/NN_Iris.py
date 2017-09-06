import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
import tflearn
from P2RegresionLineal import mse

iris = datasets.load_iris()
X = iris.data
y = iris.target

n_sample = len(X)
print("shape X: ", X.shape)
print("shape Y: ", y.shape)

fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

for name, label in [('Setosa', 0),
                    ('Versicolour', 1),
                    ('Virginica', 2)]:
    ax.text3D(X[y == label, 3].mean(),
              X[y == label, 0].mean(),
              X[y == label, 2].mean() + 2, name,
              horizontalalignment='center',
              bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y, edgecolor='k')
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
ax.set_title('Iris Dataset')
plt.show()


#Modelo
tnorm = tflearn.initializations.uniform(minval=-1., maxval=1.)
input_ = tflearn.input_data(shape=[None, 4])
hidden = tflearn.fully_connected(input_, n_units=1, activation='tanh', weights_init=tnorm)
hidden2 = tflearn.fully_connected(hidden, n_units=1, activation='tanh', weights_init=tnorm)
hidden3 = tflearn.fully_connected(hidden2, n_units=1, activation='tanh', weights_init=tnorm)
hidden4 = tflearn.fully_connected(hidden3, n_units=11, activation='tanh', weights_init=tnorm)
hidden5 = tflearn.fully_connected(hidden4, n_units=11, activation='tanh', weights_init=tnorm)
output = tflearn.fully_connected(hidden5, n_units=1, activation=None, name='output',  weights_init=tnorm)

regression = tflearn.regression(output, optimizer='sgd', loss='mean_square',
                                metric='accuracy', learning_rate=0.25)


#print("MSE: ", mse(1, 3))

#Entrenamiento
m = tflearn.DNN(regression)
m.fit(X, y.reshape(len(y),1), n_epoch=500, show_metric=True, snapshot_epoch=False)
Y_ = m.predict(X)
