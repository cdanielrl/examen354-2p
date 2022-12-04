import numpy as np
import random

def leer_datos_csv(nom):
    if not nom.endswith(".csv"):
        nom=nom+".csv"
    with open(nom, 'r') as f:
        resultado = []
        for line in f:
            columnas = line.strip().split(',')
            fila=[]
            for i in range(len(columnas)):
                e=columnas[i]
                if isinstance(e, int) or (isinstance(e, str) and e.isnumeric()):
                    fila.append(int(e))
                else:
                    fila.append(e)
            resultado.append(fila)
        return resultado

def mutar(vector,num):
    resultado=vector.copy()
    for i in range(num):
        pos1=random.randrange(0, len(vector)-1)
        pos2=random.randrange(0, len(vector)-1)
        aux=resultado[pos1]
        resultado[pos1]=resultado[pos2]
        resultado[pos2]=aux
    return resultado

def costo(inicio,vector,matriz):
    a=inicio
    sum=0
    for i in vector:
        b=i
        sum=sum+matriz[a][b]
        a=b
    return sum

def cruzar(vector1,vector2):
    mezcla = []
    perte1 = []
    parte2 = []
    
    geneA = int(random.random() * len(vector1))
    geneB = int(random.random() * len(vector1))
    
    inicioGen = min(geneA, geneB)
    finGen = max(geneA, geneB)

    for i in range(inicioGen, finGen):
        perte1.append(vector1[i])
        
    parte2 = [item for item in vector2 if item not in perte1]

    mezcla = perte1 + parte2
    return mezcla

def genDescendientes(matriz,base,inicio,numIndividuos,numMutaciones):
    resultado = []
    for i in range(numIndividuos):
        individuo=mutar(base,numMutaciones)
        valor=costo(inicio,individuo,matriz)
        fila=[]
        fila.append(valor)
        fila.append(individuo)
        resultado.append(fila)
    return resultado

def seleccion(poblacion,numIndividuos):
    inicial=poblacion.copy()
    ordenado=sorted(inicial, key=lambda costo: costo[0])
    return ordenado[0:numIndividuos]

def genGeneraciones(base,numIndividuos,numElegidos,numGeneraciones,numMutaciones,inicio,matriz,puntos):
    gen1=genDescendientes(matriz,base,inicio,numIndividuos,numMutaciones)
    poblacion=gen1.copy()
    print("Elegidos Primera Generación:")
    for e in poblacion:
        mostrarRuta(e,inicio,puntos)
    final=[]
    for i in range(numGeneraciones-1):
        generacion=[]
        for j in range(0,len(poblacion),2):
            i1=poblacion[j][1]
            i2=poblacion[j][1]
            h=cruzar(i1,i2)
            descendientes=genDescendientes(matriz,h,inicio,numIndividuos,numMutaciones)
            for d in descendientes:
                generacion.append(d)
        poblacion=seleccion(generacion,numElegidos)
        print("Elegidos Generación ",str(i+2),":")
        for e in poblacion:
            mostrarRuta(e,inicio,puntos)
    final=poblacion

    return final

def mostrarRuta(elem,inicio,puntos):
    salida="Ruta: "+puntos[inicio]
    for i in elem[1]:
        salida=salida+" -> "+puntos[i]
    salida=salida+" Costo: "+str(elem[0])
    print(salida)

inicio=0  #El punto de inicio es el primer elemento (primera columna primera fila del csv)
tasa_mutaciones=1
num__descendientes=10
num__generaciones=100
num_elegidos=10

matriz=leer_datos_csv("grafo.csv")
puntos=matriz.pop(0)

base=np.arange(len(puntos)).tolist()
base.pop(inicio)

mejores=genGeneraciones(base,num__descendientes,num_elegidos,num__generaciones,tasa_mutaciones,inicio,matriz,puntos)
print("Elegidos Generación final:")
for e in mejores:
    mostrarRuta(e,inicio,puntos)