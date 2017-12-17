#Propacacion hacia delante
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#Normalization
def minmax(x,min,max, reverse=False):
    if reverse:
        return(x*(max-min)+min)
    return (x-min)/(max-min)
def mapminmax(array, min_max=None, reverse=False):
    if(min_max and reverse):
        (min,max) = min_max
        vminmax = np.vectorize(minmax)
        return vminmax(array,min,max,True)
    min = np.min(array)
    max = np.max(array)
    vminmax = np.vectorize(minmax)
    return (vminmax(array,min,max),min,max)
def vminmax(vector, min_max=None, reverse=False):
    norm = []
    (min,max) = min_max
    for n in vector:
        norm.append(minmax(n,min,max,reverse))
    return np.array(norm)
#defined sigmoid
def sigmoid(z):
    s = 1.0 / (1.0 + np.exp(-1.0 * z))
    return s
#derivitate of sigmoid
def sigmoid_derivative(z):
    ans = sigmoid(z) * (1 - sigmoid(z))
    return ans
# AW + b
def z(inputs, weights, onesOne, onesTwo):
    #adds a vector(bia) to our inputs
    inputs = np.vstack((inputs, onesOne))
    #adds a vector (bias) to our weights
    weights = np.hstack((weights, onesTwo))
    #Z y es obtained
    z = np.dot(weights, inputs)
    return z
#a0 of our Neurol Netowrk
inputs = np.array([
    [4.7, 6.0],
    [6.1, 3.9],
    [2.9, 4.2],
    [7.0, 5.5]
])
#Our target matrix
targets = np.array([
    [3.52, 4.02],
    [5.43, 6.23],
    [4.95, 5.76],
    [4.70, 4.28]
])
xtrain, xtest, ytrain, ytest = train_test_split(inputs, targets, test_size=0.3, random_state=1)
#Declaring learningrate
learning_rate = 0.4;
#Randomly Generatoed matrix 3x2
w10 = np.random.uniform(low=0.0, high=1.0, size=(3, 2))
#Randomly Generatoed matrix of 2x3
w21 = np.random.uniform(low=0.0, high=1.0, size=(2, 3))
#Declaration of ones that are going to be appended to the matrixes
fourOnes = np.ones((1, 4))
threeOnes = np.ones((3, 1))
twoOnes = np.ones((2,1))
#declaring amount of iterations
epoch = 0
#transposing inputs to be able to append vector of ones
inputs = np.transpose(inputs)
(inputs, imin, imax) = mapminmax(inputs)
(targets,tmin, tmax) = mapminmax(targets)
while epoch < 500:
    #callin z funciton to get our z one
    z1 = z(inputs, w10, fourOnes, threeOnes)
    #se optiene a1 ingresando la z a la funciona de activacion
    a1 = sigmoid(z1)
    #se obtiene z2
    z2 = z(a1, w21, fourOnes, twoOnes)
    #se optiene a2 ingresando la z2 a la funciona de activacion d
    a2 = sigmoid(z2)
    #Using the subtract function from numpy we subtract the matrix
    error_2 = np.subtract(targets.T, a2)
    #obtaine de loss of the function
    loss = np.power(error_2, 2)
    #error mse
    error_mse = np.sum(loss)
    print('Error MSE:', error_mse)
    #define derivate function for sigmoid
    f1 = sigmoid_derivative(z2)
    #A1 is added a vertically an array of four ones
    a1Aum = np.vstack((a1, fourOnes))
    #define delta K and multiply the error with de derivative function
    deltaK = np.multiply(f1, error_2 * -2)
    #we finish the equation by using dot function to multiply the current
    #delta K with a1Aum
    deltaW21 = np.dot(deltaK, a1Aum)
    #print delta K
    W21Aumentada = np.hstack((w21, twoOnes))
    #obtaining new weights
    newW21 = np.subtract(W21Aumentada, learning_rate * deltaW21)
    #removing bias
    w21 = np.delete(newW21, 3, axis=1)
    #Define variables needed for delta J
    f2 = sigmoid_derivative(z1)
    #error signal
    error_1 = np.dot(w21.T, deltaK)
    #declaring delta
    deltaJ = np.multiply(f2, error_1)
    #declaring augmented A
    a0Aum = np.vstack((inputs, fourOnes))
    #declaring delta10
    delta10 = np.dot(deltaJ, a0Aum.T)
    #decllaring W10Aumentada
    W10Aumentada = np.hstack((w10, threeOnes))
    #creating the new w10
    newW10 = np.subtract(W10Aumentada, learning_rate * delta10)
    w10 = np.delete(newW10, 2, axis=1)
    epoch += 1
