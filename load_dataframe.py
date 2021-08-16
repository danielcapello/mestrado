import pandas as pd
import numpy as np
from numpy import load

def load_criptogramas_rsa(arquivo):
    df = pd.DataFrame(np.load(arquivo, allow_pickle=True))
    df.columns = ['criptograma_rsa']
    return df

def load_criptogramas_elgamal(arquivo):
    df = pd.DataFrame(np.load(arquivo, allow_pickle=True))
    df.columns = ['criptograma_elgamal_parte1','criptograma_elgamal_parte2']
    return df

def load_criptogramas_ec(arquivo):
    df = pd.DataFrame(np.load(arquivo, allow_pickle=True))
    df.columns = ['EC_c1','EC_c2']
    return df

def load_binarios(arquivo):
    df = pd.DataFrame(np.load(arquivo, allow_pickle=True))
    df.columns = ['binario']
    return df

def load_vetor(arquivo):
    vetor = np.load(arquivo, allow_pickle=True)
    return vetor


#data = load_criptogramas_ec('/home/kplo/Documentos/Mestrado/Criptogramas/1024/EC_base_160_Trein_Valid.npy')
#print(data['EC_c2'][5])
#print(len(data['EC_c2'][5]))

#data = load_criptogramas_rsa('/home/kplo/Documentos/Mestrado/Criptogramas/1024/RSA_base_1024_Trein_Valid.npy')
#print(data['criptograma_rsa'][5])
#print(len(data['criptograma_rsa'][5]))