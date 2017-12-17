import numpy as np
import seaborn as sb
import pandas as pd
import tflearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
def train_step(X, Y):

    #Modelo
    tnorm = tflearn.initializations.uniform(minval=-1., maxval=1.)
    input_ = tflearn.input_data(shape=[None, 13])
    hidden = tflearn.fully_connected(input_, n_units=7, activation='linear', weights_init=tnorm)
    hidden1 = tflearn.fully_connected(hidden, n_units=7, activation='linear', weights_init=tnorm)
    output = tflearn.fully_connected(hidden, n_units=1, activation=None, name='output',  weights_init=tnorm)

    regression = tflearn.regression(output, optimizer='sgd', loss='mean_square',
                                    metric='accuracy', learning_rate=1)

    #Entrenamiento
    m = tflearn.DNN(regression)
    m.fit(X, Y, n_epoch=50, show_metric=True, snapshot_epoch=False)
    return m
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
#Suma de los errores al cuadrado
def sse(reales, obtenidos):
	result = 0
	for i in range(len(reales)):
		result += pow(reales[i] - obtenidos[i], 2)
	return result
def mse(reales, obtenidos):
	return sse(reales, obtenidos)/len(reales)

#Ejercicio 1
BOSTON_DATASET = "C:/Users/Saul/Documents/GitHub/AI/10_04_2017/housing\housing.data.txt"
columns=["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]
df = pd.read_csv(BOSTON_DATASET, delim_whitespace=True, names=columns)
df.head()
df.describe()
sb.set(style="white")
# Compute the correlation matrix
corr = df.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sb.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sb.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()

#Ejercicio 2
Y = df[['MEDV']]
X = df.drop('MEDV', axis=1)
(X, imin, imax) = mapminmax(X)
(Y,tmin, tmax) = mapminmax(Y)
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.3, random_state=1)
model = train_step(np.array(xtrain), np.array(ytrain))
ypred = model.predict(np.array(xtest))
error = mse(ytest, ypred)
print(error)

#Ejercicio 3
