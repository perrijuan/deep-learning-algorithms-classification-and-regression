#algoritimos de data science 


import numpy as np 

def numpy_dimension():

    sort = np.array([1,2,3,4,5])


    #funcoes de nunpy -> para max,min,med e desvio padrao  usando o sort 
    max_valor = np.max(sort)
    min_valor = np.min(sort)
    media = np.mean(sort)
    mediana = np.median(sort)
    desvio_padrao = np.std(sort)

    print(f"O maior valor é: {max_valor}")
    print(f"O menor valor é: {min_valor}")
    print(f"A média é: {media}")


  

  #organizando o array  com os dados de um csv 

  
