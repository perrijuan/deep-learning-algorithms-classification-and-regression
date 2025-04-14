import matplotlib.pyplot as plt 



#plotando um grafico simples 2d 

def grafico_simples():
    x = [1,2,3,4,5]
    y = [1,4,9,16,25]

    plt.plot(x,y)
    plt.xlabel('Eixo X')
    plt.ylabel('Eixo Y')
    plt.title('Grafico Simples')
    plt.show()




    
    
