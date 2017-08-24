array = [1, 2, 3, 5, 10]
arrayTwo = [2, 3, 4, 14, 1]
#Funcion para evaluar si cada numero en el arreglo es par o impar
def par_o_impar(arreglo):
    for numero in arreglo:
        if(numero % 2 == 0):
            print (str(numero) + " es par")
        else:
            print (str(numero) + " es impar")
par_o_impar(array)
#funcion para calcular el numero maximo en un arreglo
def maximus_numbah(array):
    return max(array)
maximus_numbah(array)
#funcion para calcular la suma de dos vectores arreglos unidemsiales de misma longitud
def suma_de_vectores(vec1, vec2):
    return map(sum, zip(vec1, vec2))
suma_de_vectores(array, arrayTwo)
