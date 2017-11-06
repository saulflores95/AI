datos = [0, 2, 4, 5, 8, 10, 10, 15, 38]

estudiantes = [13, 14, 15, 15, 15, 16, 17, 18, 20]

def media(valores):
    return sum(valores) / len(valores)

def varianza(valores):
    ans = []
    for valor in valores:
        ans.append(pow(valor - media(valores), 2))
    return media(ans)

def desviacionE(valores):
    return pow(varianza(valores), 0.5)

#Se imprimen los resultados
#print('Media: ', media(datos))
print('Varianza: ', varianza(estudiantes))
print('Desviacion estandar: ', desviacionE(estudiantes))
