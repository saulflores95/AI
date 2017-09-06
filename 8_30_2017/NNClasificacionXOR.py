import tensorflow as tf
import tflearn

#Ejemplo de clasificacion de operacion XOR

X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
Y = [[0.], [1.], [1.], [0.]]


#Modelo
tnorm = tflearn.initializations.uniform(minval=-1., maxval=1.)

input_ = tflearn.input_data(shape=[None, 2])
hidden = tflearn.fully_connected(input_, n_units=10, activation='linear', weights_init=tnorm)
hidden2 = tflearn.fully_connected(hidden, n_units=1000, activation='linear', weights_init=tnorm)
output = tflearn.fully_connected(hidden2, n_units=1, activation=None, name='output',  weights_init=tnorm)

regression = tflearn.regression(output, optimizer='sgd', loss='mean_square',
                                metric='R2', learning_rate=.005)

#Entrenamiento
m = tflearn.DNN(regression)
m.fit(X, Y, n_epoch=500, show_metric=True, snapshot_epoch=False)


# Test
print("Operador XOR aprendido")
print("0 xor 0:", m.predict([[0., 1.]]))
print("0 xor 1:", m.predict([[0., 0.]]))
print("1 xor 0:", m.predict([[1., 0.]]))
print("1 xor 1:", m.predict([[1., 1.]]))
