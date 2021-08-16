import numpy as np
import pandas as pd
from numpy import load
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from load_dataframe import load_vetor
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier

def construir_base(arq1, arq2, arq3):
    rsa = load_vetor(arq1)
    elgamal = load_vetor(arq2)
    ec = load_vetor(arq3)
    #aleatorio = load_vetor(arq2)
    data = [rsa, elgamal, ec]
    base = np.concatenate(data)
    return base

def separação_x(dados):
    x = list(dados[:,0])
    return  x

def separação_y(dados):
    y = list(dados[:,1])
    return y
  
def normalização(dados):
    return preprocessing.normalize(dados, norm='l1')

def clf_svm(x_train, x_test, y_train, y_test):
    print("Classificação SVM")
    clf_svm_rbf = svm.SVC()
    clf_svm_rbf = clf_svm_rbf.fit(x_train, y_train)
    print("treinamento SVM ok")
    predição_svm_rbf = clf_svm_rbf.predict(x_test)
    print(classification_report(y_test, predição_svm_rbf))
    print (confusion_matrix(y_test,predição_svm_rbf))

def clf_tree(x_train, x_test, y_train, y_test):
    print("Classificação árvore de decisão")
    clf_tree = tree.DecisionTreeClassifier()
    clf_tree = clf_tree.fit(x_train, y_train)
    print("treinamento árvore ok")
    predição_tree = clf_tree.predict(x_test)
    print (classification_report(y_test, predição_tree))
    print (confusion_matrix(y_test,predição_tree))


def clf_random(x_train, x_test, y_train, y_test):
    print("Classificação Random Forest")
    clf_random = RandomForestClassifier()
    clf_random = clf_random.fit(x_train, y_train)
    print("treinamento random ok")
    predição_random = clf_random.predict(x_test)
    print (classification_report(y_test, predição_random))
    print (confusion_matrix(y_test,predição_random))

def clf_grad_dec(x_train, x_test, y_train, y_test):
    clf_grad_dec = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
    clf_grad_dec = clf_grad_dec.fit(x_train,y_train)
    predição_grad_dec = clf_grad_dec.predict(x_test)
    print(classification_report(y_test, predição_grad_dec))
    print (confusion_matrix(y_test,predição_grad_dec))

def clf_bayes(x_train, x_test, y_train, y_test):
    print("Classificação Naive Bayes")
    clf_bayes = GaussianNB()
    clf_bayes = clf_bayes.fit(x_train, y_train)
    predição_bayes = clf_bayes.predict(x_test)
    print(classification_report(y_test, predição_bayes))
    print (confusion_matrix(y_test,predição_bayes))

def clf_neural(x_train, x_test, y_train, y_test):
    print("Classificação MLP")
    clf_neural = MLPClassifier()
    clf_neural = clf_neural.fit(x_train,y_train)
    predição_neural = clf_neural.predict(x_test)
    print(classification_report(y_test, predição_neural))
    print (confusion_matrix(y_test,predição_neural))

def clf_knn(x_train, x_test, y_train, y_test, x_valid, y_valid):
    print("Classificação KNN")
    clf_KNN = KNeighborsClassifier()
    clf_KNN = clf_KNN.fit(x_train,y_train)
    predição_knn = clf_KNN.predict(x_test)
    #print("Predição_teste")
    #print(classification_report(y_test, predição_knn))
    #print (confusion_matrix(y_test,predição_knn))
    #predição_knn_valid = clf_KNN.predict(x_valid)
    #print("Predição_validação")
    #print(classification_report(y_valid, predição_knn_valid))
    #print (confusion_matrix(y_valid,predição_knn_valid))



def principal_ml(arq1, arq2, arq3, arq4, arq5, arq6):
    base_toda_trein_test = construir_base(arq1, arq2, arq3)
    base_toda_valid = construir_base(arq4, arq5, arq6)
    #print("leitura ok")
    x_trein_test = separação_x(base_toda_trein_test)
    y_trein_test = separação_y(base_toda_trein_test)
    x_valid = separação_x(base_toda_valid)
    y_valid = separação_y(base_toda_valid)
    x_train, x_test, y_train, y_test = train_test_split(x_trein_test, y_trein_test, test_size=0.33)
    x_valid, x_valid_2, y_valid, y_valid_2 = train_test_split(x_valid, y_valid, test_size=0.01)
    #print("split ok")
    #clf_bayes = GaussianNB()
    #clf_bayes = clf_bayes.fit(x_train, y_train)
    #print("bayes ok")
    clf_random = RandomForestClassifier()
    clf_random = clf_random.fit(x_train,y_train)
    #print("random ok")
    #clf_tree = tree.DecisionTreeClassifier()
    #clf_tree = clf_tree.fit(x_train, y_train)
    #print("tree ok")
    clf_svm = svm.SVC()
    clf_svm = clf_svm.fit(x_train, y_train)
    #print("treinamento SVM ok")
    clf_KNN = KNeighborsClassifier()
    clf_KNN = clf_KNN.fit(x_train,y_train)
    comite = VotingClassifier(
        estimators=[('knn', clf_KNN), ('random', clf_random), ('svm', clf_svm)],
        voting='hard', verbose=False)
    comite = comite.fit(x_train,y_train)
    predição_comite_trein = comite.predict(x_train)
    predição_comite_test = comite.predict(x_test)
    predição_comite_valid = comite.predict(x_valid)
    #print(classification_report(y_test, predição_comite))
    #print (confusion_matrix(y_test,predição_comite))
    matriz_treinamento =confusion_matrix(y_train, predição_comite_trein).ravel()
    matriz_teste = confusion_matrix(y_test,predição_comite_test).ravel()
    matriz_valid = confusion_matrix(y_valid,predição_comite_valid).ravel()
    alg = "Comite"
    return matriz_treinamento, matriz_teste, matriz_valid, alg
    
resultados_finais = []
for i in range(32):
    resultados_rodada = ['e) 150k', 'CE', 'ELG', 'RSA']
    matriz_trein_final, matriz_teste_final, matriz_valid_final, algoritmo_final = principal_ml('/home/kplo/Documentos/Mestrado/Vetores/Vetores_1024/RSA/RSA_base_1024_Trein_Valid_vetor_8_150kbits_2kamostras.npy',
            '/home/kplo/Documentos/Mestrado/Vetores/Vetores_1024/ELG/Elgamal_base_1024_Trein_Valid_vetor_8_150kbits_2kamostras.npy',
            '/home/kplo/Documentos/Mestrado/Vetores/Vetores_1024/EC/EC_base_160_Trein_Valid_vetor_8_150kbits_2kamostras.npy', 
            '/home/kplo/Documentos/Mestrado/Vetores/Vetores_1024/RSA/RSA_base_1024_Test_vetor_8_150kbits_2kamostras.npy',
            '/home/kplo/Documentos/Mestrado/Vetores/Vetores_1024/ELG/Elgamal_base_1024_Test_vetor_8_150kbits_2kamostras.npy',
            '/home/kplo/Documentos/Mestrado/Vetores/Vetores_1024/EC/EC_base_160_Test_vetor_8_150kbits_2kamostras.npy' )
    resultados_rodada.append(algoritmo_final)
    for j in matriz_trein_final:
        resultados_rodada.append(j)
    for j in matriz_teste_final:
        resultados_rodada.append(j)
    for j in matriz_valid_final:
        resultados_rodada.append(j)
    resultados_finais.append(resultados_rodada)
df_resultados = pd.DataFrame(data=resultados_finais, columns=['No_bits','Classe_1','Classe_2', 'Classe_3', 'Algoritmo_ML', 'Trein_A', 'Trein_B', 'Trein_C', 'Trein_D', 'Trein_E', 'Trein_F', 'Trein_G', 'Trein_H', 'Trein_I', 'Test_A', 'Test_B', 'Test_C', 'Test_D', 'Test_E', 'Test_F','Test_G', 'Test_H','Test_I', 'Valid_A', 'Valid_B', 'Valid_C', 'Valid_D', 'Valid_E', 'Valid_F','Valid_G', 'Valid_H','Valid_I'])
df_resultados.to_excel('/mnt/c/Users/kpl0/Documents/export_ubuntu/32x_comite_rsaxelgxce_150k.xlsx')
