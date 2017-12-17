#funciones en python

#No se requiere indicar tipos de datos
def suma_dos_numeros(num1, num2):
    return num1+num2

#funcion que itera valores en un arreglo
def muestra_valores(array):
    for valor in array:
        print valor
    print "Fin de muestra_valores"

def iterar_indice(array):
    n = len(array) #taman/o del arreglo
    for i in range(n):
        print array[i]
    print "Fin de iterar_indice"

#Uso de la funcion para sumar dos numeros
a = 5
b = 8
print "resultado de "+ str(a) +" mas "+ str(b) +" es: " + str(suma_dos_numeros(a,b))
c = [9,12,45,90]
muestra_valores(c)
iterar_indice(c)