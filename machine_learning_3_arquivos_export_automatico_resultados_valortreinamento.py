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
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump

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

def clf_svm(x_train, x_test, y_train, y_test, x_valid, y_valid):
    #print("Classificação SVM")
    clf_svm_rbf = svm.SVC()
    clf_svm_rbf = clf_svm_rbf.fit(x_train, y_train)
    #print("treinamento SVM ok")
    #dump(clf_svm_rbf, '/home/kplo/Documentos/Mestrado/Classificadores_salvos/clf_svm_rsaxelgxce_4076bits.joblib')
    predição_svm_rbf_test = clf_svm_rbf.predict(x_test)
    predição_svm_rbf_trein = clf_svm_rbf.predict(x_train)
    #print("Predição_teste")
    #print(classification_report(y_test, predição_svm_rbf))
    #print (confusion_matrix(y_test,predição_svm_rbf))
    predição_svm_rbf_valid = clf_svm_rbf.predict(x_valid)
    #print("Predição_validação")
    #print(classification_report(y_valid, predição_svm_rbf_valid))
    #print (confusion_matrix(y_valid,predição_svm_rbf_valid))
    matriz_treinamento =confusion_matrix(y_train, predição_svm_rbf_trein).ravel()
    matriz_teste = confusion_matrix(y_test,predição_svm_rbf_test).ravel()
    matriz_valid = confusion_matrix(y_valid,predição_svm_rbf_valid).ravel()
    alg = "SVM"
    return matriz_treinamento, matriz_teste, matriz_valid, alg

def clf_tree(x_train, x_test, y_train, y_test, x_valid, y_valid):
    print("Classificação árvore de decisão")
    clf_tree = tree.DecisionTreeClassifier()
    clf_tree = clf_tree.fit(x_train, y_train)
    print("treinamento árvore ok")
    predição_tree = clf_tree.predict(x_test)
    print("Predição_teste")
    print (classification_report(y_test, predição_tree))
    print (confusion_matrix(y_test,predição_tree))
    predição_tree_valid = clf_tree.predict(x_valid)
    print("Predição_validação")
    print(classification_report(y_valid, predição_tree_valid))
    print (confusion_matrix(y_valid,predição_tree_valid))

def clf_random(x_train, x_test, y_train, y_test, x_valid, y_valid):
    #print("Classificação Random Forest")
    clf_random = RandomForestClassifier()
    clf_random = clf_random.fit(x_train, y_train)
    #print("treinamento random ok")
    #dump(clf_random, '/home/kplo/Documentos/Mestrado/Classificadores_salvos/clf_random_rsaxelgxce_4076bits.joblib')
    predição_random_test = clf_random.predict(x_test)
    predição_random_trein = clf_random.predict(x_train)
    #print("Predição_teste")
    #print (classification_report(y_test, predição_random))
    #print (confusion_matrix(y_test,predição_random))
    predição_random_valid = clf_random.predict(x_valid)
    #print("Predição_validação")
    #print(classification_report(y_valid, predição_random_valid))
    #print (confusion_matrix(y_valid,predição_random_valid))
    matriz_treinamento =confusion_matrix(y_train, predição_random_trein).ravel()
    matriz_teste = confusion_matrix(y_test,predição_random_test).ravel()
    matriz_valid = confusion_matrix(y_valid,predição_random_valid).ravel()
    alg = "RF"
    return matriz_treinamento, matriz_teste, matriz_valid, alg

def clf_grad_dec(x_train, x_test, y_train, y_test, x_valid, y_valid):
    clf_grad_dec = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
    clf_grad_dec = clf_grad_dec.fit(x_train,y_train)
    predição_grad_dec = clf_grad_dec.predict(x_test)
    print(classification_report(y_test, predição_grad_dec))
    print (confusion_matrix(y_test,predição_grad_dec))

def clf_bayes(x_train, x_test, y_train, y_test, x_valid, y_valid):
    print("Classificação Naive Bayes")
    clf_bayes = GaussianNB()
    clf_bayes = clf_bayes.fit(x_train, y_train)
    predição_bayes = clf_bayes.predict(x_test)
    print("Predição_teste")
    print(classification_report(y_test, predição_bayes))
    print (confusion_matrix(y_test,predição_bayes))
    predição_bayes_valid = clf_bayes.predict(x_valid)
    print("Predição_validação")
    print(classification_report(y_valid, predição_bayes_valid))
    print (confusion_matrix(y_valid,predição_bayes_valid))

def clf_neural(x_train, x_test, y_train, y_test, x_valid, y_valid):
    print("Classificação MLP")
    clf_neural = MLPClassifier()
    clf_neural = clf_neural.fit(x_train,y_train)
    predição_neural = clf_neural.predict(x_test)
    print(classification_report(y_test, predição_neural))
    print (confusion_matrix(y_test,predição_neural))
    predição_neural_valid = clf_neural.predict(x_valid)
    print("Predição_validação")
    print(classification_report(y_valid, predição_neural_valid))
    print (confusion_matrix(y_valid,predição_neural_valid))



def clf_knn(x_train, x_test, y_train, y_test, x_valid, y_valid):
    #print("Classificação KNN")
    clf_KNN = KNeighborsClassifier()
    clf_KNN = clf_KNN.fit(x_train,y_train)
    #dump(clf_KNN, '/home/kplo/Documentos/Mestrado/Classificadores_salvos/clf_knn_rsaxelgxce_4076bits.joblib')
    predição_knn_test = clf_KNN.predict(x_test)
    predição_knn_trein = clf_KNN.predict(x_train)
    #print("Predição_teste")
    #print(classification_report(y_test, predição_knn))
    #print (confusion_matrix(y_test,predição_knn))
    predição_knn_valid = clf_KNN.predict(x_valid)
    #print("Predição_validação")
    #print(classification_report(y_valid, predição_knn_valid))
    #print (confusion_matrix(y_valid,predição_knn_valid))
    matriz_treinamento =confusion_matrix(y_train, predição_knn_trein).ravel()
    matriz_teste = confusion_matrix(y_test,predição_knn_test).ravel()
    matriz_valid = confusion_matrix(y_valid,predição_knn_valid).ravel()
    alg = "KNN"
    return matriz_treinamento, matriz_teste, matriz_valid, alg
    

def principal_ml(arq1, arq2, arq3, arq4, arq5, arq6):
    base_toda_trein_test = construir_base(arq1, arq2, arq3)
    base_toda_valid = construir_base(arq4, arq5, arq6)
    #print("leitura ok")
    x_trein_test = normalização(separação_x(base_toda_trein_test))
    y_trein_test = separação_y(base_toda_trein_test)
    x_valid = normalização(separação_x(base_toda_valid))
    y_valid = separação_y(base_toda_valid)
    x_train, x_test, y_train, y_test = train_test_split(x_trein_test, y_trein_test, test_size=0.33)
    x_valid, x_valid_2, y_valid, y_valid_2 = train_test_split(x_valid, y_valid, test_size=0.01)
    #print("split ok")
    #clf_tree(x_train, x_test, y_train, y_test, x_valid, y_valid)
    #clf_bayes(x_train, x_test, y_train, y_test, x_valid, y_valid)
    #matriz0, matriz1, matriz2, algoritmo = clf_random(x_train, x_test, y_train, y_test, x_valid, y_valid)   
    #matriz0, matriz1, matriz2, algoritmo = clf_svm(x_train, x_test, y_train, y_test, x_valid, y_valid)
    #clf_neural(x_train, x_test, y_train, y_test, x_valid, y_valid)
    matriz0, matriz1, matriz2, algoritmo = clf_knn(x_train, x_test, y_train, y_test, x_valid, y_valid)
    return matriz0, matriz1, matriz2, algoritmo
  
resultados_finais = []
for i in range(32):
    resultados_rodada = ['4076', 'CE', 'ELG', 'RSA']
    matriz_trein_final, matriz_teste_final, matriz_valid_final, algoritmo_final = principal_ml('/home/kplo/Documentos/Mestrado/Vetores/Vetores_1024/RSA/RSA_base_1024_Trein_Valid_vetor_8_4076bits_2kamostras.npy',
            '/home/kplo/Documentos/Mestrado/Vetores/Vetores_1024/ELG/Elgamal_base_1024_Trein_Valid_vetor_8_4076bits_2kamostras.npy',
            '/home/kplo/Documentos/Mestrado/Vetores/Vetores_1024/EC/EC_base_160_Trein_Valid_vetor_8_4076bits_2kamostras.npy', 
            '/home/kplo/Documentos/Mestrado/Vetores/Vetores_1024/RSA/RSA_base_1024_Test_vetor_8_4076bits_2kamostras.npy',
            '/home/kplo/Documentos/Mestrado/Vetores/Vetores_1024/ELG/Elgamal_base_1024_Test_vetor_8_4076bits_2kamostras.npy',
            '/home/kplo/Documentos/Mestrado/Vetores/Vetores_1024/EC/EC_base_160_Test_vetor_8_4076bits_2kamostras.npy')
    resultados_rodada.append(algoritmo_final)
    for j in matriz_trein_final:
        resultados_rodada.append(j)    
    for j in matriz_teste_final:
        resultados_rodada.append(j)
    for j in matriz_valid_final:
        resultados_rodada.append(j)
    resultados_finais.append(resultados_rodada)
df_resultados = pd.DataFrame(data=resultados_finais, columns=['No_bits','Classe_1','Classe_2', 'Classe_3', 'Algoritmo_ML', 'Trein_A', 'Trein_B', 'Trein_C', 'Trein_D', 'Trein_E', 'Trein_F', 'Trein_G', 'Trein_H', 'Trein_I', 'Test_A', 'Test_B', 'Test_C', 'Test_D', 'Test_E', 'Test_F','Test_G', 'Test_H','Test_I', 'Valid_A', 'Valid_B', 'Valid_C', 'Valid_D', 'Valid_E', 'Valid_F','Valid_G', 'Valid_H','Valid_I'])
df_resultados.to_excel('/mnt/c/Users/kpl0/Documents/export_ubuntu/32x_KNN_rsa_elg_ce_4076.xlsx')
