# -*- coding: utf-8 -*-

import numpy as np

class Neurona:
  def __init__(self, pesos, bias):
    self.pesos = pesos
    self.bias = bias

  def retroalimentacion(self, inputs):
    total = np.dot(self.pesos, inputs) + self.bias
    return sigmoid(total)

class redNeuronal:

  def __init__(self):

    self.neurona1 = Neurona(np.random.normal(size=(4)), np.random.normal())
    self.neurona2 = Neurona(np.random.normal(size=(4)), np.random.normal())
    self.neurona3 = Neurona(np.random.normal(size=(4)), np.random.normal())
    self.neurona4 = Neurona(np.random.normal(size=(4)), np.random.normal())
    self.neurona5 = Neurona(np.random.normal(size=(4)), np.random.normal())
    self.neurona6 = Neurona(np.random.normal(size=(4)), np.random.normal())
    self.neurona7 = Neurona(np.random.normal(size=(2)), np.random.normal())

  def retroalimentacion(self, x):
    salida_neurona1 = self.neurona1.retroalimentacion(x)
    salida_neurona2 = self.neurona2.retroalimentacion(x)
    salida_neurona3 = self.neurona3.retroalimentacion(x)
    salida_neurona4 = self.neurona4.retroalimentacion(x)

    salida_neurona5 = self.neurona5.retroalimentacion(np.array([salida_neurona1, salida_neurona2,salida_neurona3, salida_neurona4]))
    salida_neurona6 = self.neurona6.retroalimentacion(np.array([salida_neurona1, salida_neurona2,salida_neurona3, salida_neurona4]))

    salida_neurona7 = self.neurona7.retroalimentacion(np.array([salida_neurona5, salida_neurona6]))

    return salida_neurona7

  def train(self, datos, y_trues):
    tasa_aprendizaje = 0.1
    epocas = 1000 

    for epoca in range(epocas):
	
      for x, y_true in zip(datos, y_trues):
        sum_neurona1 = np.dot(self.neurona1.pesos,x)
        sigmoid_neurona1 = sigmoid(sum_neurona1)

        sum_neurona2 = np.dot(self.neurona2.pesos,x)
        sigmoid_neurona2 = sigmoid(sum_neurona2)

        sum_neurona3 =  np.dot(self.neurona3.pesos,x)
        sigmoid_neurona3 = sigmoid(sum_neurona3)

        sum_neurona4 =  np.dot(self.neurona4.pesos,x)
        sigmoid_neurona4 = sigmoid(sum_neurona4)

        sum_neurona5 = np.dot(self.neurona5.pesos,np.array([sigmoid_neurona1, sigmoid_neurona2,sigmoid_neurona3, sigmoid_neurona4]))
        sigmoid_neurona5 = sigmoid(sum_neurona5)

        sum_neurona6 = np.dot(self.neurona6.pesos,np.array([sigmoid_neurona1, sigmoid_neurona2,sigmoid_neurona3, sigmoid_neurona4]))
        sigmoid_neurona6 = sigmoid(sum_neurona6)

        sum_neurona7 = np.dot(self.neurona7.pesos,np.array([sigmoid_neurona5, sigmoid_neurona6]))
        sigmoid_neurona7 = sigmoid(sum_neurona7)

        y_pred = sigmoid_neurona7

        # derivada parcial
        d_L_d_ypred = -2 * (y_true - y_pred)

        # Neurona1
        d_neurona1_d_pesos1 = np.multiply(x,deriv_sigmoid(sum_neurona1))
        d_neurona1_d_b1 = deriv_sigmoid(sum_neurona1)

        # Neurona2
        d_neurona2_d_pesos2 = np.multiply(x,deriv_sigmoid(sum_neurona2))
        d_neurona2_d_b2 = deriv_sigmoid(sum_neurona2)

        # Neurona3
        d_neurona3_d_pesos3 = np.multiply(x,deriv_sigmoid(sum_neurona3))
        d_neurona3_d_b3 = deriv_sigmoid(sum_neurona3)

        # Neurona4
        d_neurona4_d_pesos4 = np.multiply(x,deriv_sigmoid(sum_neurona4))
        d_neurona4_d_b4 = deriv_sigmoid(sum_neurona4)

        # Neurona5
        d_ypred_d_neurona5=np.dot(np.array([sigmoid_neurona1,sigmoid_neurona2,sigmoid_neurona3,sigmoid_neurona4]),np.array([d_neurona1_d_b1,d_neurona2_d_b2,d_neurona3_d_b3,d_neurona4_d_b4]))
        d_ypred_d_b5 = deriv_sigmoid(sum_neurona5)
        
        d_neurona5_d_b5 = self.neurona7.pesos[0]*deriv_sigmoid(sum_neurona5)

        # Neurona6
        d_ypred_d_neurona6=np.dot(np.array([sigmoid_neurona1,sigmoid_neurona2,sigmoid_neurona3,sigmoid_neurona4]),np.array([d_neurona1_d_b1,d_neurona2_d_b2,d_neurona3_d_b3,d_neurona4_d_b4]))
        d_ypred_d_b6 = deriv_sigmoid(sum_neurona6)
        d_neurona6_d_b6 = self.neurona7.pesos[1]*deriv_sigmoid(sum_neurona6)

        d_ypred_d_neurona1 = self.neurona5.pesos[0] * deriv_sigmoid(sum_neurona5) * self.neurona6.pesos[0] * deriv_sigmoid(sum_neurona6)
        d_ypred_d_neurona2 = self.neurona5.pesos[1] * deriv_sigmoid(sum_neurona5) * self.neurona6.pesos[1] * deriv_sigmoid(sum_neurona6)
        d_ypred_d_neurona3 = self.neurona5.pesos[2] * deriv_sigmoid(sum_neurona5) * self.neurona6.pesos[2] * deriv_sigmoid(sum_neurona6)
        d_ypred_d_neurona4 = self.neurona5.pesos[3] * deriv_sigmoid(sum_neurona5) * self.neurona6.pesos[3] * deriv_sigmoid(sum_neurona6)

        # Neurona7
        d_ypred_d_neurona7=np.dot(np.array([sigmoid_neurona5,sigmoid_neurona6]),np.array([d_neurona5_d_b5,d_neurona6_d_b6]))
        
        d_ypred_d_b7 = deriv_sigmoid(sum_neurona7)

        d_ypred_d_neurona5 = self.neurona7.pesos[0] * deriv_sigmoid(sum_neurona7)
        d_ypred_d_neurona6 = self.neurona7.pesos[1] * deriv_sigmoid(sum_neurona7)

        # Actualizar
        # Neurona1
        self.neurona1.pesos=np.multiply(d_neurona1_d_pesos1,tasa_aprendizaje * d_L_d_ypred * d_ypred_d_neurona1)
        self.neurona1.bias-= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_neurona1 * d_neurona1_d_b1

        # Neurona2
        self.neurona2.pesos=np.multiply(d_neurona2_d_pesos2,tasa_aprendizaje * d_L_d_ypred * d_ypred_d_neurona2)
        self.neurona2.bias-= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_neurona2 * d_neurona2_d_b2

        # Neurona3
        self.neurona3.pesos=np.multiply(d_neurona3_d_pesos3,tasa_aprendizaje * d_L_d_ypred * d_ypred_d_neurona3)
        self.neurona3.bias-= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_neurona3 * d_neurona3_d_b3

        # Neurona4
        self.neurona4.pesos=np.multiply(d_neurona4_d_pesos4,tasa_aprendizaje * d_L_d_ypred * d_ypred_d_neurona4)
        self.neurona4.bias-= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_neurona4 * d_neurona4_d_b4

        # Neurona5
        self.neurona5.pesos=np.multiply(np.array([d_ypred_d_neurona1,d_ypred_d_neurona2,d_ypred_d_neurona3,d_ypred_d_neurona4]),tasa_aprendizaje * d_L_d_ypred)
        self.neurona5.bias-= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_neurona5 * d_ypred_d_b5

        # Neurona6
        self.neurona6.pesos=np.multiply(np.array([d_ypred_d_neurona1,d_ypred_d_neurona2,d_ypred_d_neurona3,d_ypred_d_neurona4]),tasa_aprendizaje * d_L_d_ypred)
        self.neurona6.bias-= tasa_aprendizaje * d_L_d_ypred * d_ypred_d_neurona6 * d_ypred_d_b6

        # Neurona7
        self.neurona7.pesos=np.multiply(np.array([d_ypred_d_neurona5,d_ypred_d_neurona6]),tasa_aprendizaje * d_ypred_d_neurona7)
        self.neurona7.bias-= np.dot(d_L_d_ypred,tasa_aprendizaje * d_ypred_d_neurona6 * d_ypred_d_b6)

      # perdida por cada epoca
      if epoca % 10 == 0:
        y_preds = np.apply_along_axis(self.retroalimentacion, 1, data)
        perdida = mse_perdida(y_trues, y_preds)
        print("epoca %d perdida: %.3f" % (epoca, perdida))

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  fx = sigmoid(x)
  return fx * (1 - fx)

def mse_perdida(y_true, y_pred):
  return ((y_true - y_pred) ** 2).mean()

# Entrenar
file = open("iris_csv.csv")
data = np. loadtxt(file, delimiter=",")
y_trues = np.tile([5.84333333333333, 3.054, 3.75866666666667, 1.19866666666667],(150,1))

mired = redNeuronal()
mired.train(data, y_trues)

# Predecir
data1 = np.array([2, 4, 1, 3])
print("data1: %s" % mired.retroalimentacion(data1))