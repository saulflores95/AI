import tflearn
import numpy as np
import matplotlib.pyplot as plt
import P2RegresionLineal as LR
from PlotLR import y2MinimoCuadrado

X = [1, 2, 3, 4, 5, 6, 7]
Y = [0.50, 2.50, 2.00, 4.00, 3.50, 6.00, 5.50]

X = np.array(X).reshape(len(X), 1)
Y = np.array(Y).reshape(len(Y), 1)

input_ = tflearn.input_data(shape=[None, 1])
linear = tflearn.fully_connected (input_, n_units=1)
regression = tflearn.regression(linear, optimizer='sgd', loss='mean_square',
                                metric='R2', learning_rate=0.01)
m = tflearn.DNN(regression)
m.fit(X, Y, n_epoch=1000, show_metric=True, snapshot_epoch=False)

print("\nRegression result:")
print("Y = " + str(m.get_weights(linear.W)) + "*X + " + str(m.get_weights(linear.b)))

Y_ = m.get_weights(linear.W) * X + m.get_weights(linear.b)

print("MSE Modelo Uno: ", LR.mse(Y, Y_))
print("MSE Minimo Cuadrados: ", y2MinimoCuadrado)
plt.plot(X, Y, 'k^', label='puntos reales')
plt.plot(X, Y_, 'r--', label='puntos obtenidos')
plt.title('NN111')
plt.show()
