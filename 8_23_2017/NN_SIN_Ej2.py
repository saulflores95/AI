import tflearn
import numpy as np
import matplotlib.pyplot as plt
from P2RegresionLineal import mse

X = np.arange(-1.5, 1.5, 0.05)
Y = np.sin(X)

print("Numero de datos: "+str(len(X)))
plt.title("Sin(x)")
plt.plot(X, Y, 'k--', label='real points')
plt.show()

tnorm = tflearn.initializations.uniform(minval=-1., maxval=1.)
input_ = tflearn.input_data(shape=[None, 1])

hidden = tflearn.fully_connected(input_, n_units=5, activation='tanh', weights_init=tnorm)
hidden2 = tflearn.fully_connected(hidden, n_units=10, activation='tanh', weights_init=tnorm)
output = tflearn.fully_connected(hidden2, n_units=1, activation=None, name='output')

regression = tflearn.regression(output, optimizer='sgd', loss='mean_square',
                                metric='R2', learning_rate=.4)

m = tflearn.DNN(regression)
m.fit(X.reshape(len(X),1), Y.reshape(len(Y),1), n_epoch=100, show_metric=True, snapshot_epoch=False)
Y_ = m.predict(X.reshape(len(X), 1))
print("MSE Modelo Uno: ", mse(Y, Y_))
plt.plot(X, Y, 'k--', label='real points')
plt.plot(X, Y_, 'ro', label='real points')
plt.title('Ej.2 Configuracion 1')
plt.show()
