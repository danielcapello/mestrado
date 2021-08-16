def lista_8bits():
    lista_bin_8bits = []
    for i in range(0,2**8):
        lista_bin_8bits.append("{0:>08b}".format(i))
    return lista_bin_8bits

def lista_16bits():
    lista_bin_16bits = []
    for i in range(0,2**8):
        for j in range(0,2**8):
            lista_bin_16bits.append("{0:>08b}".format(i)+"{0:>08b}".format(j))
    return lista_bin_16bits


