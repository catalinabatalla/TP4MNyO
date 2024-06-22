import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1428)

def funcion_de_costo(A, x, b):
    c = A @ x - b 
    return c.T @ c

def gradiente_funcion_costo(A, x, b):
    gradiente = 2 * A.T @ (A @ x - b)
    return gradiente

def funcion_regularizacion(A, x, b, F, delta):
    norm_x = np.linalg.norm(x)**2
    F2 = F(A, x, b) + delta * norm_x
    return F2

def gradiente_funcion_regularizacion(A, x, b, delta):
    gradiente_F = 2 * A.T @ (A @ x - b)
    grad = gradiente_F + 2 * delta * x
    return grad 

def iterativo(A, x0, b, step, gradiente_func, epsilon, max_iteraciones):
    x = x0
    iteraciones = []
    valores = []
    for iter in range(max_iteraciones):
        grad = gradiente_func(A, x, b)
        x = x - step * grad
        iteraciones.append(iter)
        valores.append(x.copy())
        if np.linalg.norm(grad) < epsilon:
            break
    return x, iteraciones, valores

def valor_sing_max(A):
    U, S, Vt = np.linalg.svd(A)
    return S[0]

def valor_sing_min(A):
    U, S, Vt = np.linalg.svd(A)
    return S[-1]

def autovalor_max(A):
    Hessiana = 2 * A.T @ A
    lambda_max = np.linalg.eigvals(Hessiana).max()
    return lambda_max

def pca_manual(A, n_components):
    U, S, Vt = np.linalg.svd(A)
    U_k = U[:, :n_components]
    S_k = S[:n_components]
    Vt_k = Vt[:n_components, :]
    A_reduced = U_k @ np.diag(S_k) @ Vt_k
    return A_reduced, U_k, S_k, Vt_k

def pseudo_inverse(S):
    S_inv = np.zeros_like(S)
    for i in range(len(S)):
        if S[i] != 0:
            S_inv[i] = 1 / S[i]
    return np.diag(S_inv)

def calcular_x(U, S, Vt, b):
    S_inv = pseudo_inverse(S)
    beta = Vt.T @ S_inv @ U.T @ b
    return beta

def grafico_comparacion(x_F, x_F2, d):
    plt.figure(figsize=(10, 6))
    plt.scatter(range(d), x_F, label='F', color='plum', marker='s')
    plt.scatter(range(d), x_F2, label='F2', color='chartreuse', marker='^')
    plt.xlabel('Índice', fontsize=14)
    plt.ylabel('Valor de x', fontsize=14)
    plt.title('Comparación de soluciones por SVD y L2', fontsize=16)
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.show()

def grafico_diferentes_delta(deltas, x_F, d):
    plt.figure(figsize=(10, 6))

    labels = [r'$\delta^2 = 10^{-2}\sigma_{max}$', r'$\delta^2 = 10^{-2}\sigma_{min}$', r'$\delta^2 = 0$', r'$\delta^2 = 25$', r'$\delta^2 = 100$']

    for idx, x in enumerate(x_F):
        plt.plot(x, label=f'{labels[idx]}')
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel('Iteraciones', fontsize=14)
    plt.ylabel('Valor de F2', fontsize=14)
    plt.title('Evolución de F2 con diferentes valores de $\delta$', fontsize=16)
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.show()
    



def main():
    n = 5
    d = 100
    A = np.random.rand(n, d)
    b = np.random.rand(n)
    delta = 10**(-2) * valor_sing_max(A)
    s = 1 / autovalor_max(A)
    x0 = np.random.uniform(0, 1, d)
    epsilon = 10**(-6)
    max_iteraciones = 6000

    # Minimización F1
    x_F, iteraciones_F, valores_F = iterativo(A, x0, b, s, gradiente_funcion_costo, epsilon, max_iteraciones)
    valores_F_costo = [funcion_de_costo(A, x, b) for x in valores_F]
    
    # Minimización F2
    x_F2, iteraciones_F2, valores_F2 = iterativo(A, x0, b, s, lambda A, x, b: gradiente_funcion_regularizacion(A, x, b, delta), epsilon, max_iteraciones)
    valores_F2_costo = [funcion_regularizacion(A, x, b, funcion_de_costo, delta) for x in valores_F2]

    # Minimización con SVD
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    x_svd = calcular_x(U, S, Vt, b)
    
    grafico_comparacion(x_F, x_F2, d)

    deltas = []
    lista = []
    delta_lista = [10**(-2) * valor_sing_max(A), 10**(-2) * valor_sing_min(A), 0, 25, 100]
    for sub_delta in delta_lista:
        x_F2, iteraciones_F2, valores_F2 = iterativo(A, x0, b, s, lambda A, x, b: gradiente_funcion_regularizacion(A, x, b, sub_delta), epsilon, max_iteraciones)
        valores_F2_costo = [funcion_regularizacion(A, x, b, funcion_de_costo, sub_delta) for x in valores_F2]
        lista.append(valores_F2_costo)
    grafico_diferentes_delta(deltas, lista, d)

if __name__ == "__main__":
    main()
