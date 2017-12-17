def fibonacci(n):
    sequence = []
    actual = 0
    anterior = 0
    sumatoria = 0
    for i in range(0, n):
        if(actual == 0):
            actual = 1
            sequence.append(actual)
        else:
            sumatoria = actual + anterior
            anterior = actual
            actual = sumatoria
            sequence.append(sumatoria)
    return sequence

print(fibonacci(100))
