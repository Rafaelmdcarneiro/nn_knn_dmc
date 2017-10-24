import numpy as np
import math
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

#dados
def dados(arquivo):
    #carregar vetor de dados 
    with open(arquivo) as fn:
        raw_data = np.loadtxt(fn, delimiter= ' ', dtype="float",
            skiprows=1, usecols=None)
        
    #inicializa vetores
    data  = []
    label = []
    
    #dados de entrada
    for row in raw_data:
        data.append(row[:-1])
        label.append(row[-1])
        
    #retornar os dados
    return np.array(data), np.array(label)


#main
def inicializa():
    print """ KNN """
    
    #contador de tempo de execução
    start = time.clock()

    #leitura vetor de teste e treinamento
    x_train, y_train   = dados('CCtest1.txt')
    #y_train           = dados('CCtest2.txt')
    x_test, y_test    = dados('CCtrain.txt')

    #k
    k = 15

    print "k: ", k
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(x_train, y_train)
    y_pred3=neigh.predict(x_test)
    neigh_score=neigh.score(x_test,y_test)
    print "Acerto: ", neigh_score*100, "%"
    cm3=confusion_matrix(y_test,y_pred3)
    print "Matrix de confusão"
    print cm3    
    
    #tempo de execucao
    run_time = time.clock() - start
    print "Tempo de Execução:", run_time


inicializa()
