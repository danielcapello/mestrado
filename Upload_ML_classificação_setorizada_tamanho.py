import pandas as pd
import numpy as np
from scipy.sparse import base
from load_dataframe import load_criptogramas_rsa, load_criptogramas_elgamal, load_criptogramas_ec
from criptograma_em_bits import criptograma_em_bits_rsa_sem_salvar, criptograma_em_bits_elgamal_sem_salvar, criptograma_em_bits_ec_sem_salvar
from vetorização import dict_map_8bits, check_binario_8
from joblib import load
from load_dataframe import load_binarios, load_vetor
from sklearn import preprocessing
from machine_learning_2_arquivos import construir_base, separação_x, separação_y

def normalização(dados):
    return preprocessing.normalize(dados, norm='l1')

arquivo_1 = '/home/kplo/Documentos/Mestrado/Vetores/Vetores_1024/Elgamal_base_1024_Test_vetor_8.npy'
arquivo_2 = '/home/kplo/Documentos/Mestrado/Vetores/Vetores_1024/EC_base_160_Test_vetor_8.npy'
bases_juntas = construir_base(arquivo_1, arquivo_2)
vetores_originais = separação_x(bases_juntas)
vetores_normalizados = list(normalização(vetores_originais))
classes_originais = separação_y(bases_juntas)
dados = {'Vetor_original':vetores_originais, 'Soma_vetor': 0, 'Vetores_normalizados': vetores_normalizados, 'Classe_original': classes_originais, 'Predição':0}
data = pd.DataFrame(data=dados)
classificador = load('/home/kplo/Documentos/Mestrado/Classificadores_salvos/clf_svm_elgxce_norm.joblib')
for i in data.index:
    vetor = data['Vetor_original'][i]
    soma = 0
    for j in vetor:
        soma += j
    data['Soma_vetor'][i] = soma
    vetor_class = data['Vetores_normalizados'][i]
    vetor_class = np.array(vetor_class)
    vetor_class = vetor_class.reshape(1,-1)
    predição = classificador.predict(vetor_class)
    data['Predição'][i] = predição
data.to_excel('/home/kplo/Documentos/Mestrado/resultados_wikipedia_elgxce_norm.xlsx')

