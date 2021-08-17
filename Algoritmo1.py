

datos_entrenamiento = [
    [1, 1, 0, 1, 'Infección de garganta'],
    [0, 0, 1, 0, 'Alergia'],
    [0, 0, 1, 1, 'Resfriado'],
    [0, 0, 1, 0, 'Alergia'],
    [1, 1, 1, 1, 'Infección de garganta'],
    [0, 0, 1, 0, 'Resfriado'],
    [1, 0, 1, 1, 'Resfriado'],
    [1, 1, 0, 1, 'Infección de garganta'],
    [0, 0, 1, 1, 'Resfriado'],
]

#Síntomas
etiquetas = ["Fiebre", "Dolor de garganta", "Congestión", "Dolor de cabeza", "Diagnóstico"]

def unicos_valores(filas, columnas): return set([fila[columnas] for fila in filas])

print(unicos_valores(datos_entrenamiento, 4))

def class_contar(filas):
    contar = {}
    for fila in filas:
        contenido = fila[-1]
        if contenido not in contar:
            contar[contenido] = 0
        contar[contenido] += 1
    return contar

#contar etiquetas de los datos de entrenamiento
print(class_contar(datos_entrenamiento))

def es_numerico(valor): return isinstance(valor, int) or isinstance(valor, float)
print(es_numerico(7))

class Evaluar:
    
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def comparar(self, example):
        val = example[self.column]
        if es_numerico(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        condition = "=="
        if es_numerico(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            etiquetas[self.column], condition, str(self.value))

print(Evaluar(2, 3))

q = Evaluar(0, 1)
print(q)

example = datos_entrenamiento[0]
print(q.comparar(example))

def particionar(filas, evaluar):
    filas_acertadas, filas_erradas = [], []
    for fila in filas:
        if evaluar.comparar(fila):
            filas_acertadas.append(fila)
        else:
            filas_erradas.append(fila)
    return filas_acertadas, filas_erradas

filas_acertadas, filas_erradas = particionar(datos_entrenamiento, Evaluar(1, 1)) 
print(filas_acertadas)
print(filas_erradas)

def incertidumbre(filas):
    conteo = class_contar(filas)
    impureza = 1
    for datos in conteo:
        prob_de_datos = conteo[datos] / float(len(filas))
        impureza -= prob_de_datos**2
    return impureza

incertidumbre_actual = incertidumbre(datos_entrenamiento)
print(incertidumbre_actual)

def informacion_seguimiento(izquierda, derecha, incertidumbre_actual):
    p = float(len(izquierda)) / (len(izquierda) + len(derecha))
    return incertidumbre_actual - p * incertidumbre(izquierda) - (1 - p) * incertidumbre(derecha)

filas_acertadas, filas_erradas = particionar(datos_entrenamiento, Evaluar(1, 1)) 
informacion_seguimiento(filas_acertadas, filas_erradas, incertidumbre_actual)

def encontrar_mejor_ruta(filas):
    
    mejor_seguimiento = 0
    mejor_evaluacion = None
    incertidumbre_actual = incertidumbre(filas)
    numero_columnas = len(filas[0]) - 1

    for columnas in range(numero_columnas):

        valores = set([fila[columnas] for fila in filas])

        for valor in valores:

            evaluar = Evaluar(columnas, valor)
            filas_acertadas, filas_erradas = particionar(filas, evaluar)

            if len(filas_acertadas) == 0 or len(filas_erradas) == 0:
                continue

            info = informacion_seguimiento(filas_acertadas, filas_erradas, incertidumbre_actual)

            if info >= mejor_seguimiento:
                mejor_seguimiento, mejor_evaluacion = info, evaluar

    return mejor_seguimiento, mejor_evaluacion

mejor_seguimiento, mejor_evaluacion = encontrar_mejor_ruta(datos_entrenamiento)
print(mejor_seguimiento)

print(mejor_evaluacion)

class clasificacion:
       
    def __init__(self, filas):
        self.prediccion = class_contar(filas)

class decidir_nodo:
    
    def __init__(self, evaluar, rama_correcta, rama_incorrecta):
        self.evaluar = evaluar
        self.rama_correcta = rama_correcta
        self.rama_incorrecta = rama_incorrecta

def construir_arbol(filas):
    
    info, evaluar = encontrar_mejor_ruta(filas)

    if info == 0:
        return clasificacion(filas)

    filas_acertadas, filas_erradas = particionar(filas, evaluar)
    rama_correcta = construir_arbol(filas_acertadas)
    rama_incorrecta = construir_arbol(filas_erradas)

    return decidir_nodo(evaluar, rama_correcta, rama_incorrecta)

def imprimir_arbol(nodo, espacio=""):
    
    if isinstance(nodo, clasificacion):
        print (espacio + "Predict", nodo.prediccion)
        return

    print (espacio + str(nodo.evaluar))

    print (espacio + '--> True:')
    imprimir_arbol(nodo.rama_correcta, espacio + "  ")

    print (espacio + '--> False:')
    imprimir_arbol(nodo.rama_incorrecta, espacio + "  ")

mi_arbol = construir_arbol(datos_entrenamiento)

print(imprimir_arbol(mi_arbol))

def clasificar(fila, nodo):
    
    if isinstance(nodo, clasificacion):
        return nodo.prediccion

    if nodo.evaluar.comparar(fila):
        return clasificar(fila, nodo.rama_correcta)
    else:
        return clasificar(fila, nodo.rama_incorrecta)

print(clasificar(datos_entrenamiento[0], mi_arbol))

def imprimir_clasificacion(conteo):
    total = sum(conteo.values()) * 1.0
    probabilidades = {}
    for datos in conteo.keys():
        probabilidades[datos] = str(int(conteo[datos] / total * 100)) + "%"
    return probabilidades

print(imprimir_clasificacion(clasificar(datos_entrenamiento[1], mi_arbol)))

evaluar_datos = [
    [1, 0, 1, 1, '¿?'],
    [0, 0, 1, 0, '¿?'],
]

for fila in evaluar_datos:
    print ("Diagnóstico: %s" %
           (imprimir_clasificacion(clasificar(fila, mi_arbol))))