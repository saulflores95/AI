import numpy as np
import matplotlib.pyplot as plt
#variables declaradas previas a su utilizacion
w10 = None
w21 = None
z1 = None
z2 = None
a1 = None
a2 = None
error = None
#defined sigmoid
def sigmoid(z):
    s = 1.0 / (1.0 + np.exp(-1.0 * z))
    return s
# AW + b
def z(inputs, weights, onesOne, onesTwo):
    #adds a vector(bia) to our inputs
    inputs = np.vstack((inputs, onesOne))
    #adds a vector (bias) to our weights
    weights = np.hstack((weights, onesTwo))
    #Z y es obtained
    z = np.dot(weights, inputs)
    return z
#a0 of our FFW network
inputs = np.array([
    [4.6, 6.0],
    [6.1, 3.9],
    [2.9, 4.2],
    [7.0, 5.5]
    ])
#our target matrix
targets = np.array([
    [3.52, 4.02],
    [5.43, 6.23],
    [4.95, 5.76],
    [4.70, 4.28]
    ])
#Randomly Generatoed matrix 3x2
w10 = np.random.uniform(low=0.0, high=1.0, size=(3, 2))
#Randomly Generatoed matrix of 2x3
w20 = np.random.uniform(low=0.0, high=1.0, size=(2, 3))
#Declaration of ones that are going to be appended to the matrixes
fourOnes = np.ones((1, 4))
threeOnes = np.ones((3, 1))
twoOnes = np.ones((2,1))
#transposing inputs to be able to append vector of ojnes
inputs = np.transpose(inputs)
#callin z funciton to get our z one
z1 = z(inputs, w10, fourOnes, threeOnes)
#se optiene a1 ingresando la z a la funciona de activacion
a1 = sigmoid(z1)
#se obtiene z2
z2 = z(a1, w20, fourOnes, twoOnes)
#se optiene a2 ingresando la z2 a la funciona de activacion d
a2 = sigmoid(z2)
#transpos 4x2 matrix to 2x4 so we can subtract a2 matrix
targets = np.transpose(targets)
#Using the subtract function from numpy we subtract the matrix
error = np.subtract(targets, a2)
#We Found our errors, lets go get a beer :D
print("Error", error)
