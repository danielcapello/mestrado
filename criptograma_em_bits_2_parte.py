from numpy import save
from load_dataframe import load_criptogramas_rsa, load_criptogramas_elgamal, load_criptogramas_ec

def criptograma_em_bits_rsa(origem, destino):
    df = load_criptogramas_rsa(origem)
    df['criptograma_em_bits'] = 0
    df['criptograma_em_bits'] = df['criptograma_em_bits'].astype(object)
    for i in df.index:
        em_binario = []
        lista_recuperada = df['criptograma_rsa'][i]
        for j in range(0,len(lista_recuperada)):
            a_mapear = lista_recuperada[j]
            for k in range(0,len(a_mapear)): 
                em_binario.append("{0:>08b}".format(a_mapear[k]))
        df['criptograma_em_bits'][i] = ''.join(em_binario)
    save(destino,df.loc[:,'criptograma_em_bits'].to_numpy())

def criptograma_em_bits_elgamal(origem, destino):
    df = load_criptogramas_elgamal(origem)
    df['criptograma_em_bits'] = 0
    df['criptograma_em_bits'] = df['criptograma_em_bits'].astype(object)
    for i in df.index:
        em_binario = []
        #lista_recuperada = df['criptograma_elgamal_parte1'][i]
        #for j in range(0,len(lista_recuperada)):
            #a_mapear = lista_recuperada[j]
            #for k in range(0,len(a_mapear)): 
                #em_binario.append("{0:>08b}".format(a_mapear[k]))
        lista_recuperada = df['criptograma_elgamal_parte2'][i]        
        for j in range(0,len(lista_recuperada)):
            a_mapear = lista_recuperada[j]
            for k in range(0,len(a_mapear)): 
                em_binario.append("{0:>08b}".format(a_mapear[k]))
        df['criptograma_em_bits'][i] = ''.join(em_binario)
    save(destino,df.loc[:,'criptograma_em_bits'].to_numpy())

def criptograma_em_bits_ec(origem, destino):
    df = load_criptogramas_ec(origem)
    df['criptograma_em_bits'] = 0
    df['criptograma_em_bits'] = df['criptograma_em_bits'].astype(object)
    for i in df.index:
        em_binario = []
        #lista_recuperada = df['EC_c1'][i]
        #for j in range(0,len(lista_recuperada)):
            #a_mapear = lista_recuperada[j]
            #em_binario.append(bin(a_mapear[0])[2:])
            #em_binario.append(bin(a_mapear[1])[2:])
        lista_recuperada = df['EC_c2'][i]
        for j in range(0,len(lista_recuperada)):
            a_mapear = lista_recuperada[j]
            em_binario.append(bin(a_mapear[0])[2:])
            em_binario.append(bin(a_mapear[1])[2:])
        df['criptograma_em_bits'][i] = ''.join(em_binario)
    save(destino,df.loc[:,'criptograma_em_bits'].to_numpy())