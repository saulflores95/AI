
#PlotLR.py

#Representacion grafica del ajuste lineal de un conjunto de datos (puntos)

"""
Comentario de multiples lineas

Sintaxis python
Al realizar un import de una biblioteca se le puede asignar a esta un sobrenombre.
Ejemplo
import misuperbiblioteca as msb

Ahora en lugar de acceder mediante misuperbiblioteca a metodos o clases, lo hariamos
mediante msb.

ejemplo:
msb.Clase()
msb.funcion()
msb.variable

"""



#Biblioteca para graficar
#Es la que usa matlab asi que les resultara familiar
import matplotlib.pyplot as plt
#Es el programa que realizaron (El archivo se llamo LinearRegression)
import P2RegresionLineal as LR



datosX = [1, 2, 3, 4, 5, 6, 7]
datosY = [0.50, 2.50, 2.00, 4.00, 3.50, 6.00, 5.50]

#Es como el String format


function_format = "f(x) = {:.2f} + {:.2f}x"

x = datosX
y = datosY

linealRegression = LR.RegresionLineal(x,y)
linealRegression.fit()
a0 = linealRegression.a0
a1 = linealRegression.a1
y2 = linealRegression.eval(x)

y2MinimoCuadrado = LR.mse(y, y2)

print("MSE Modelo Uno: ", LR.mse(y, y2))


fig, ax = plt.subplots()
ax.plot(x, y, 'ko', label='real points')
ax.plot(x, y2, 'r--', label=function_format.format(a0,a1))
plt.title('Least squares')
plt.ylabel('y')
plt.xlabel('x')
plt.legend()
plt.show()
