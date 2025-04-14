#modelo de arvore de decisao para classificacao 
# usando o dataset iris e sklearn como lib para criar o modelo 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree

# Função para carregar e preparar os dados
def carregar_dados(arquivo_csv):
    # Carregar o dataset
    dados = pd.read_csv(arquivo_csv)
    print("Informações do dataset:")
    print(dados.info())
    print("\nPrimeiras linhas do dataset:")
    print(dados.head())
    return dados

# Função para dividir os dados em conjunto de treino e teste
def dividir_dados(X, y, test_size=0.3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"Dimensões do conjunto de treino: {X_train.shape}")
    print(f"Dimensões do conjunto de teste: {X_test.shape}")
    return X_train, X_test, y_train, y_test

# Função para treinar e avaliar o modelo
def treinar_arvore_decisao(X_train, X_test, y_train, y_test, max_depth=None):
    # Criar e treinar o modelo
    modelo = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    modelo.fit(X_train, y_train)
    
    # Fazer previsões
    y_pred = modelo.predict(X_test)
    
    # Avaliar o modelo
    acuracia = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {acuracia:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão')
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Previsto')
    plt.show()
    
    return modelo

# Função para visualizar a árvore de decisão
def visualizar_arvore(modelo, feature_names, class_names):
    plt.figure(figsize=(15, 10))
    tree.plot_tree(modelo, feature_names=feature_names, class_names=class_names, filled=True)
    plt.title("Visualização da Árvore de Decisão")
    plt.show()

# Exemplo de uso
if __name__ == "__main__":
    # Substitua 'dataset.csv' pelo caminho do seu arquivo CSV
    # dados = carregar_dados('dataset.csv')
    
    # Exemplo com o dataset Iris (já vem com o scikit-learn)
    from sklearn.datasets import load_iris
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    
    # Dividir os dados
    X_train, X_test, y_train, y_test = dividir_dados(X, y)
    
    # Treinar o modelo
    modelo = treinar_arvore_decisao(X_train, X_test, y_train, y_test, max_depth=3)
    
    # Visualizar a árvore
    visualizar_arvore(modelo, feature_names=iris.feature_names, class_names=iris.target_names)
    
    # Importância das features
    importancia = pd.Series(modelo.feature_importances_, index=iris.feature_names)
    plt.figure(figsize=(10, 6))
    importancia.sort_values().plot(kind='barh')
    plt.title('Importância das Features')
    plt.show()






