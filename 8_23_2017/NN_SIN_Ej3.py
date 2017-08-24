import tflearn
import numpy as np
import matplotlib.pyplot as plt


X = np.arange(-5, 5, 0.1)
Y = np.sin(X)
print("Numero de datos: "+str(len(X)))
print(X.shape)

plt.title("Sin(x)")
plt.plot(X, Y, 'k--', label='real points')
plt.show()

tnorm = tflearn.initializations.uniform(minval=-1., maxval=1.)
input_ = tflearn.input_data(shape=[None, 1])
hidden = tflearn.fully_connected(input_, n_units=100, activation='sigmoid', weights_init=tnorm)
hidden2 = tflearn.fully_connected(hidden, n_units=1000, activation='sigmoid', weights_init=tnorm)
hidden3 = tflearn.fully_connected(hidden2, n_units=100, activation='sigmoid', weights_init=tnorm)
output = tflearn.fully_connected(hidden3, n_units=1, activation=None, name='output',  weights_init=tnorm)

regression = tflearn.regression(output, optimizer='sgd', loss='mean_square',
                                metric='R2', learning_rate=.007)

m = tflearn.DNN(regression)

m.fit(X.reshape(100,1), Y.reshape(100,1), n_epoch=500, show_metric=True, snapshot_epoch=False)

Y_ = m.predict(X.reshape(len(X), 1))

plt.title("Modelo obtenido")
plt.plot(X, Y, 'k--', label='real points')
plt.plot(X, Y_, 'ro', label='predicted points')
plt.show()
