import tflearn
import numpy as np
import matplotlib.pyplot as plt

#Ejemplo de clasificacion de operacion OR

X = [ [0,0], [1,0], [0,1], [1,1]]
Y = [[0], [1], [1], [1] ]

#Modelo
tnorm = tflearn.initializations.uniform(minval=-1., maxval=1.)
input_ = tflearn.input_data(shape=[None, 2])
hidden = tflearn.fully_connected(input_, n_units=1, activation='sigmoid', weights_init=tnorm)
output = tflearn.fully_connected(hidden, n_units=1, activation=None, name='output',  weights_init=tnorm)

regression = tflearn.regression(output, optimizer='sgd', loss='mean_square',
                                metric='R2', learning_rate=.7)

#Entrenamiento
m = tflearn.DNN(regression)
m.fit(X, Y, n_epoch=500, show_metric=True, snapshot_epoch=False)



# Test
print("Operador OR aprendido")
print("0 or 0:", m.predict([[0., 0.]]))
print("0 or 1:", m.predict([[0., 1.]]))
print("1 or 0:", m.predict([[1., 0.]]))
print("1 or 1:", m.predict([[1., 1.]]))
