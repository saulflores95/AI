metros = [5, 15, 20, 25]
precio = [375, 487, 450, 500]

#Funcion que regresa la media
def media(valores):
    return sum(valores) / len(valores) #sumatoria de la lista de datos, dividio por el tamano de la lista.

#funcion que regresa el valor de la prediccion y imprime b, eh interseccion
def regresionLineal(x, y, n):
    #n es el valore constante ej tiempo, precio que se quiere calcular
    multX = [] #Se inicializa una variale para agrega los valores que van se sumados
    multY = [] #Se inicializa una variale para agrega los valores que van se sumados
    sumatoriaMultiplicacion = 0 #Inicializa la sumatoria en y de la lista declarada previamente
    sumatoriaXCuadradaIndivudal = 0 #inicializa vairable de la sumatoria de los cuadrados
    sumatoriaX = sum(x) #sumatoria de los valores en x
    sumatoriaY = sum(y) #sumatoria de los valores en Y
    sumatoriaXCuadrada = pow(sumatoriaX, 2) #sumatoriaX al cuadrado
    for i in range(0, len(x)):
        multY.append(x[i] * y[i])#se agregan a la lista multY todos los productors de 'x' y 'y'
        sumatoriaMultiplicacion = sum(multY) #sumatoria de tdos los valores en Y
        multX.append(pow(x[i], 2))
        sumatoriaXCuadradaIndivudal = sum(multX)
    b = ((len(x)*sumatoriaMultiplicacion) - (sumatoriaX*sumatoriaY))/((len(y)*sumatoriaXCuadradaIndivudal)-sumatoriaXCuadrada) #se obtiene la pendiente
    mediaX = media(x) #media de los valores en x
    mediaY = media(y) #media de los valores en y
    interseccion = mediaY - b * mediaX # se calcula interesccion
    ans = interseccion + b * n #se consigue la respuesta
    print(['b: ', b, 'Interseccion: ', interseccion, 'Ans: ', ans])
    return ans

regresionLineal(metros, precio, 35)
#funcion de regresion logistica
