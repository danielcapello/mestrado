import pandas as pd
import numpy as np
from numpy import load, save
from gerar_lista_binaria import lista_8bits, lista_16bits
from load_dataframe import load_binarios
from math import ceil
from io import StringIO
import random

def dict_map_8bits():
    vetor_mapeamento = {}
    lista = lista_8bits()
    for q in lista:
        vetor_mapeamento[q] = 0
    return vetor_mapeamento

def dict_map_16bits():
    vetor_mapeamento = {}
    lista = lista_16bits()
    for q in lista:
        vetor_mapeamento[q] = 0
    return vetor_mapeamento

def check_binario_8(string):
    tamanho = len(string)
    pad = 0
    if tamanho % 8 != 0:
        pad = (ceil(tamanho/8)*8) - tamanho
        string = string + "0"*pad
        return string
    else:
        return string

def check_binario_16(string):
    tamanho = len(string)
    pad = 0
    if tamanho % 16 != 0:
        pad = (ceil(tamanho/16)*16) - tamanho
        string = string + "0"*pad
        return string
    else:
        return string


def vetorização_8bits(arquivo, destino, label):
    df = load_binarios(arquivo)
    df['vetor'] = 0
    df['vetor'] = df['vetor'].astype(object)
    for i in df.index:
        vetor_map = dict_map_8bits()
        a_mapear = check_binario_8(df['binario'][i])
        for j in range(0,len(a_mapear),8):
            vetor_map[a_mapear[j:j+8]] += 1
        df['vetor'][i] = list(vetor_map.values())
    df['label'] = label
    save(destino,df.loc[:,'vetor':'label'].to_numpy())

def vetorização_16bits(arquivo, destino, label):
    df = load_binarios(arquivo)
    df['vetor'] = 0
    df['vetor'] = df['vetor'].astype(object)
    for i in df.index:
        vetor_map = dict_map_16bits()
        a_mapear = check_binario_16(df['binario'][i])
        for j in range(0,len(a_mapear),16):
            vetor_map[a_mapear[j:j+16]] += 1
        df['vetor'][i] = list(vetor_map.values())
    df['label'] = label
    save(destino,df.loc[:,'vetor':'label'].to_numpy())

def vetorização_16bits_reduzida(arquivo, destino, label):
    df = load_binarios(arquivo)
    df['vetor'] = 0
    df['vetor'] = df['vetor'].astype(object)
    for i in range(10000,30279):
        df = df.drop([i])
    for i in df.index:
        vetor_map = dict_map_16bits()
        a_mapear = check_binario_16(df['binario'][i])
        for j in range(0,len(a_mapear),16):
            vetor_map[a_mapear[j:j+16]] += 1
        df['vetor'][i] = list(vetor_map.values())
    df['label'] = label
    save(destino,df.loc[:,'vetor':'label'].to_numpy())

def vetorização_8bits_reduzida(arquivo, destino, label):
    df = load_binarios(arquivo)
    df['vetor'] = 0
    df['vetor'] = df['vetor'].astype(object)
    for i in range(81,30279):
        df = df.drop([i])
    for i in df.index:
        vetor_map = dict_map_8bits()
        a_mapear = check_binario_8(df['binario'][i])
        for j in range(0,len(a_mapear),8):
            vetor_map[a_mapear[j:j+8]] += 1
        df['vetor'][i] = list(vetor_map.values())
    df['label'] = label
    save(destino,df.loc[:,'vetor':'label'].to_numpy())

def vetorização_8bits_tamanho_minimo(arquivo, destino, label, tamanho):
    df = load_binarios(arquivo)
    df['bits_reduzidos'] = 0
    df['bits_reduzidos'] = df['bits_reduzidos'].astype(object)
    for i in df.index:
        df['bits_reduzidos'][i] = df['binario'][i][:tamanho]
    df['vetor'] = 0
    df['vetor'] = df['vetor'].astype(object)
    for i in df.index:
        vetor_map = dict_map_8bits()
        a_mapear = check_binario_8(df['bits_reduzidos'][i])
        for j in range(0,len(a_mapear),8):
            vetor_map[a_mapear[j:j+8]] += 1
        df['vetor'][i] = list(vetor_map.values())
    df['label'] = label
    save(destino,df.loc[:,'vetor':'label'].to_numpy())

def vetorização_8bits_sem_salvar(arquivo):
    df = pd.read_csv(StringIO(arquivo))
    df['vetor'] = 0
    df['vetor'] = df['vetor'].astype(object)
    for i in df.index:
        vetor_map = dict_map_8bits()
        a_mapear = check_binario_8(df['binario'][i])
        for j in range(0,len(a_mapear),8):
            vetor_map[a_mapear[j:j+8]] += 1
        df['vetor'][i] = list(vetor_map.values())
    return df

def vetorização_8bits_tamanho_minimo_aleatorio(arquivo, destino, label, tamanho):
    df = load_binarios(arquivo)
    df['bits_reduzidos'] = 0
    df['bits_reduzidos'] = df['bits_reduzidos'].astype(object)
    for i in df.index:
        tamanho_binario = len(df['binario'][i])
        começo = tamanho_binario - tamanho
        começo = random.randrange(começo)
        fim = começo + tamanho
        df['bits_reduzidos'][i] = df['binario'][i][começo:fim]
    df['vetor'] = 0
    df['vetor'] = df['vetor'].astype(object)
    for i in df.index:
        vetor_map = dict_map_8bits()
        a_mapear = check_binario_8(df['bits_reduzidos'][i])
        for j in range(0,len(a_mapear),8):
            vetor_map[a_mapear[j:j+8]] += 1
        df['vetor'][i] = list(vetor_map.values())
    df['label'] = label
    save(destino,df.loc[:,'vetor':'label'].to_numpy())

def vetorização_8bits_tamanho_minimo_aleatorio_filtro(arquivo, destino, label, tamanho):
    df = load_binarios(arquivo)
    df['tamanho'] = 0
    df['bits_reduzidos'] = 0
    df['bits_reduzidos'] = df['bits_reduzidos'].astype(object)
    for i in df.index:
        df['tamanho'][i] = len(df['binario'][i])
    df_mask = df['tamanho'] > tamanho
    df = df[df_mask]
    for i in df.index:
        tamanho_binario = len(df['binario'][i])
        começo = tamanho_binario - tamanho
        começo = random.randrange(começo)
        fim = começo + tamanho
        df['bits_reduzidos'][i] = df['binario'][i][começo:fim]
    df['vetor'] = 0
    df['vetor'] = df['vetor'].astype(object)
    for i in df.index:
        vetor_map = dict_map_8bits()
        a_mapear = check_binario_8(df['bits_reduzidos'][i])
        for j in range(0,len(a_mapear),8):
            vetor_map[a_mapear[j:j+8]] += 1
        df['vetor'][i] = list(vetor_map.values())
    df['label'] = label
    save(destino,df.loc[:,'vetor':'label'].to_numpy())

def vetorização_8bits_tamanho_minimo_aleatorio_filtro_amostrasiguais(arquivo, destino, label, tamanho):
    df = load_binarios(arquivo)
    df['tamanho'] = 0
    df['bits_reduzidos'] = 0
    df['bits_reduzidos'] = df['bits_reduzidos'].astype(object)
    for i in df.index:
        df['tamanho'][i] = len(df['binario'][i])
    df_mask = df['tamanho'] > tamanho
    df = df[df_mask]
    for i in df.index:
        tamanho_binario = len(df['binario'][i])
        começo = tamanho_binario - tamanho
        começo = random.randrange(começo)
        fim = começo + tamanho
        df['bits_reduzidos'][i] = df['binario'][i][começo:fim]
    df['vetor'] = 0
    df['vetor'] = df['vetor'].astype(object)
    for i in df.index:
        vetor_map = dict_map_8bits()
        a_mapear = check_binario_8(df['bits_reduzidos'][i])
        for j in range(0,len(a_mapear),8):
            vetor_map[a_mapear[j:j+8]] += 1
        df['vetor'][i] = list(vetor_map.values())
    df = df.reset_index(drop=True)
    tamanho_index = df['vetor'].count()
    lista = list(range(2000,tamanho_index))
    for j in lista:
        df = df.drop(j)
    df['label'] = label
    save(destino,df.loc[:,'vetor':'label'].to_numpy())

#vetorização_8bits('/home/kplo/Documentos/Mestrado/binarios/Elgamal_base_1024_Trein_Valid_bin_2_parte.npy',
                #'/home/kplo/Documentos/Mestrado/Vetores/Elgamal_base_1024_Trein_Valid_vetor_8_2_parte.npy',1)

#vetorização_8bits('/home/kplo/Documentos/Mestrado/binarios/EC_base_160_Trein_Valid_bin_2_parte.npy',
                #'/home/kplo/Documentos/Mestrado/Vetores/EC_base_160_Trein_Valid_vetor_8.npy_2_parte',0)

#vetorização_8bits_tamanho_minimo_aleatorio_filtro('/home/kplo/Documentos/Mestrado/binarios/Wikipedia/EC_base_160_Test_bin.npy',
                                #'/home/kplo/Documentos/Mestrado/Vetores/Vetores_1024/EC_base_160_Test_vetor_8_100kbits.npy',
                                #0,100000)

vetorização_8bits_tamanho_minimo_aleatorio_filtro_amostrasiguais('/home/kplo/Documentos/Mestrado/binarios/Wikipedia/RSA_base_1024_Test_bin.npy',
                                '/home/kplo/Documentos/Mestrado/Vetores/Vetores_1024/RSA/RSA_base_1024_Test_vetor_8_4076bits_2kamostras.npy',
                                2,4076)