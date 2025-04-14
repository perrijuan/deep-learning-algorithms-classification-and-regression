import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

# Implementação do algoritmo de soma de Kahan para reduzir erros de arredondamento. 

def kahan_sum(values):
    """
    Implementa o algoritmo de soma de Kahan para reduzir erros de arredondamento.
    """
    s = 0.0  # Soma
    c = 0.0  # Termo de correção
    
    for v in values:
        y = v - c  # O valor a ser adicionado corrigido
        t = s + y  # Próxima soma
        c = (t - s) - y  # Novo termo de correção
        s = t  # Atualiza soma
    
    return s

class KahanLinearRegression(BaseEstimator, RegressorMixin):
    """
    Regressão Linear usando o somatório de Kahan para cálculos mais precisos.
    """
    
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
    
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
        
        # Calculando (X^T * X)^-1 * X^T * y com somas de Kahan
        XtX = np.zeros((X.shape[1], X.shape[1]))
        
        # Calcular X^T * X com somatório de Kahan
        for i in range(X.shape[1]):
            for j in range(X.shape[1]):
                products = X[:, i] * X[:, j]
                XtX[i, j] = kahan_sum(products)
        
        Xty = np.zeros(X.shape[1])
        
        # Calcular X^T * y com somatório de Kahan
        for i in range(X.shape[1]):
            products = X[:, i] * y
            Xty[i] = kahan_sum(products)
        
        # Resolver o sistema de equações normais
        self.coef_ = np.linalg.solve(XtX, Xty)
        
        # Separar o intercepto dos coeficientes, se necessário
        if self.fit_intercept:
            self.intercept_ = self.coef_[0]
            self.coef_ = self.coef_[1:]
        else:
            self.intercept_ = 0.0
        
        return self
    
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        y_pred = np.zeros(X.shape[0])
        
        # Predição com somatório de Kahan
        for i in range(X.shape[0]):
            products = self.coef_ * X[i]
            y_pred[i] = kahan_sum(products) + self.intercept_
        
        return y_pred

# Exemplo de uso
if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt
    
    # Criar dados sintéticos
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Treinar modelo com soma de Kahan
    kahan_model = KahanLinearRegression()
    kahan_model.fit(X_train, y_train)
    kahan_pred = kahan_model.predict(X_test)
    kahan_mse = mean_squared_error(y_test, kahan_pred)
    
    # Comparar com a implementação padrão do sklearn
    sklearn_model = LinearRegression()
    sklearn_model.fit(X_train, y_train)
    sklearn_pred = sklearn_model.predict(X_test)
    sklearn_mse = mean_squared_error(y_test, sklearn_pred)
    
    print(f"MSE com Regressão Linear padrão: {sklearn_mse:.10f}")
    print(f"MSE com Regressão Linear usando soma de Kahan: {kahan_mse:.10f}")
    
    # Visualizar diferenças nas predições
    plt.figure(figsize=(10, 6))
    plt.scatter(sklearn_pred, kahan_pred, alpha=0.5)
    plt.plot([min(sklearn_pred), max(sklearn_pred)], [min(sklearn_pred), max(sklearn_pred)], 'r--')
    plt.xlabel('Predições com LinearRegression padrão')
    plt.ylabel('Predições com KahanLinearRegression')
    plt.title('Comparação entre implementações')
    plt.grid(True)
    plt.show()