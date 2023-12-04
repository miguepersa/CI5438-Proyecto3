from neuron import Neuron
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from random import random
import matplotlib.pyplot as plt
import os


class Network(object):

    def __init__(self, data: pd.DataFrame, ind, dep, neuron_layers):

        self.amount_neurons_layers = neuron_layers
        self.n_layers = len(neuron_layers)
        self.ind = ind
        self.dep = dep 
        self.data = data
        self.network = None
        self.test_y = None
        self.test_x = None
        self.error_medio = []
        self.error_maximo = []
        self.error_minimo = []
        self.iters = None
        self.learning_rate = None
        self.form_network()
    
    def get_training_test(self, data: pd.DataFrame):

        x_train, x_test, y_train, y_test = train_test_split(data[self.ind], data[self.dep], test_size=0.2, random_state=42)

        self.y_test = y_test
        self.x_test = x_test

        return x_train, x_test, y_train, y_test
    
    def test(self,species_classifier, testname, binary = False):
        '''
        Funcion que evalua los datos del conjunto de pruebas
        luego de entrenar el modelo
        '''

        if binary == True:

            df = pd.DataFrame(columns=["Species Classifier","Iters","Learning Rate","Mean Error",'Max Error',"Min Error",'False Negative','False Positive',"Positive","Negative"])
            
            # Se combinan los datos de prueba en un solo DataFrame
            data = pd.concat([self.x_test,self.y_test],axis=1)

            positivo = 0
            negativo = 0
            f_positivo = 0
            f_negativo = 0

            for _, row in data.iterrows():

                values = row[self.ind]
                result = row[self.dep]

                # Evaluacion
                prediction = self.evaluate(values)
                prediction = prediction[:-1]

                pred = prediction[0]

                

                if (pred >= 0.5) and result[0] == 1 :
                    positivo +=1
                elif (pred >= 0.5) and result[0] == 0 :
                    f_positivo += 1
                elif (pred < 0.5) and result[0] == 1:
                    f_negativo += 1
                elif (pred < 0.5) and result[0] == 0:
                    negativo += 1

            df = df.append({
                    'Species Classifier': species_classifier,
                    'Iters': self.iters,
                    'Learning Rate':self.learning_rate,
                    'Mean Error': sum(self.error_medio) / len(self.error_medio),
                    'Max Error': max(self.error_maximo),
                    'Min Error': min(self.error_minimo),
                    'False Negative': f_negativo,
                    'False Positive': f_positivo,
                    'Positive': positivo,
                    'Negative': negativo,
                    }, ignore_index=True)

            plt.xlabel("Iteraciones")
            plt.ylabel("Error")
            plt.plot([i for i in range(1,self.iters+1)], self.error_maximo, color='yellow', label='Error Maximo')
            plt.plot([i for i in range(1,self.iters+1)], self.error_medio, color='blue', label='Error Medio')
            plt.plot([i for i in range(1,self.iters+1)], self.error_minimo, color='red', label='Error Minimo')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join("graficos", testname))
            plt.show()

            return df
        
        else:

            df = pd.DataFrame(columns=["Hidden Layers","Neuron Per Layers","Iters","Learning Rate","Mean Error",'Max Error',"Min Error",'Correct Classifications',"Average Certainty",'Incorrect Classifications',"Average Incorrect Activation"])
            
            # Se combinan los datos de prueba en un solo DataFrame
            data = pd.concat([self.x_test,self.y_test],axis=1)

            c_class = 0
            i_class = 0
            activation_fetched = []
            activation_incorrect_fetched = []

            for _, row in data.iterrows():

                values = row[self.ind]
                result = row[self.dep]

                # Evaluacion
                prediction = self.evaluate(values)

                prediction = prediction[:-1]

                max_index = np.argmax(prediction)

                result_positive_index = np.argmax(result)



                if max_index == result_positive_index:
                    c_class += 1
                    activation_fetched.append(prediction[max_index])
                else: 
                    i_class += 1
                    activation_incorrect_fetched.append(prediction[max_index])
                    
            df = df.append({
                    'Hidden Layers': self.n_layers-1,
                    'Neuron Per Layers':self.amount_neurons_layers[:-1],
                    'Iters': self.iters,
                    'Learning Rate':self.learning_rate,
                    'Mean Error': sum(self.error_medio) / len(self.error_medio),
                    'Max Error': max(self.error_maximo),
                    'Min Error': min(self.error_minimo),
                    'Correct Classifications': c_class,
                    "Average Certainty": sum(activation_fetched) / len(activation_fetched),
                    'Incorrect Classifications': i_class,
                    "Average Incorrect Activation":sum(activation_incorrect_fetched) / len(activation_incorrect_fetched) 
                    }, ignore_index=True)

            plt.xlabel("Iteraciones")
            plt.ylabel("Error")
            plt.plot([i for i in range(1,self.iters+1)], self.error_maximo, color='yellow', label='Error Maximo')
            plt.plot([i for i in range(1,self.iters+1)], self.error_medio, color='blue', label='Error Medio')
            plt.plot([i for i in range(1,self.iters+1)], self.error_minimo, color='red', label='Error Minimo')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join("graficos", testname))
            plt.show()

            return df

    def form_network(self):

        ind = len(self.ind)
        second_layer = [Neuron(np.array(([random()] * ind) + [1.0])) for i in range(self.amount_neurons_layers[0])]
        network = [second_layer]

        for i in range(1, self.n_layers):
            layer = [Neuron(np.array(([random()] * len(network[-1])) + [1.0])) for n in range(self.amount_neurons_layers[i])]
            network.append(layer)

        self.network = network

    def evaluate(self, values):
        x = list(values) + [1.0]
        for layer in self.network:
            for neuron in layer:
                neuron.values = x

            x = [neuron.get_activation_value() for neuron in layer] + [1.0]

        return x
    
    def train_network(self, iters, learning_rate):

        self.learning_rate = learning_rate

        self.iters = iters

        x_train, self.x_test, y_train, self.y_test = self.get_training_test(self.data)
        
        data = x_train
        data = data.join(y_train)

        self.error_medio = []
        self.error_maximo = []
        self.error_minimo = []
    
        for iteration in range(iters):

            print(iteration)

            for _, row in data.iterrows():

                values = row[self.ind]
                result = row[self.dep]

                h = self.evaluate(values)

                error = [result[i] - h[i] for i in range(len(result))]
  
                delta_j = [neuron.activation_function_derivate() for neuron in self.network[-1]]
                delta_j = [delta_j[i] * error[i] for i in range(len(delta_j))]

                deltas = [delta_j]

                if(len(self.network) == 1):

                    layer = self.network[-1]
                    delta = []

                    for i in range(len(layer)):

                        neuron = layer[i]
                        d = 0.0
                        for k in range(len(layer)):
                            d += (layer[k].weights[i])*deltas[-1][k]

                        # delta[i] = 
                        delta.append(d*neuron.activation_function_derivate())

                    deltas.append(delta)

                    for i in range(len(layer)):
                            neuron = layer[i]
                            for j in range(len(neuron.weights)):
                                neuron.weights[j] += (learning_rate * neuron.values[j] * deltas[-2][i])

                else:
                    # Backpropagation
                    for l in range(len(self.network)-1,0,-1):

                        layer = self.network[l-1]
                        next_layer = self.network[l]
                        
                        delta = []

                        for i in range(len(layer)):

                            neuron = layer[i]
                            d = 0.0
                            for k in range(len(next_layer)):
                                d += (next_layer[k].weights[i])*deltas[-1][k]

                            # delta[i] = 
                            delta.append(d*neuron.activation_function_derivate())

                        deltas.append(delta)

                        for i in range(len(next_layer)):
                            neuron = next_layer[i]
                            for j in range(len(neuron.weights)):
                                neuron.weights[j] += (learning_rate * neuron.values[j] * deltas[-2][i]) 

            self.error_medio.append(abs(np.mean(error)))
            self.error_maximo.append(abs(np.max(error)))
            self.error_minimo.append(abs(np.min(error)))                      