import numpy as np
import matplotlib.pyplot as plt
import math

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

def grafico_comparacion(x_F, x_F2, d, x_svd):
    plt.figure(figsize=(10, 6))
    plt.scatter(range(d), x_svd, label='SVD', color='gold', marker='o')
    plt.scatter(range(d), x_F, label='F', color='plum', marker='s')
    plt.scatter(range(d), x_F2, label='F2', color='chartreuse', marker='^')
    plt.xlabel('Índice', fontsize=25)
    plt.ylabel('Valor de x', fontsize=25)
    plt.title('Comparación de las soluciones de F, F2, SVD', fontsize=20)
    plt.legend(fontsize=20)
    # plt.yticks(fontsize = 18)
    # plt.xticks(fontsize = 18)
    plt.grid(True)
    plt.show()

def grafico_diferentes_delta(x_F, b):
    plt.figure(figsize=(10, 6))

    labels = [r'$\delta^2 = 10^{-2}\sigma_{max}$', r'$\delta^2 = 10^{-2}\sigma_{min}$', r'$\delta^2 = 0$', r'$\delta^2 = 5$', r'$\delta^2 = 25$', r'$\delta^2 = 100$']

    for idx, x in enumerate(x_F):
        plt.plot(x, label=f'{labels[idx]}')
    n = np.linspace(0, len(b), len(b))
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel('Iteraciones', fontsize=25)
    plt.ylabel('Valor de F2', fontsize=25)
    plt.xlim(10**(-1), 10**3.7)
    plt.ylim(10**(-6.5), 10**3.5)
    plt.title('Evolución de F2 con diferentes valores de $\delta$', fontsize=20)
    plt.legend(fontsize=20)
    # plt.yticks(fontsize = 18)
    # plt.xticks(fontsize = 18)
    plt.grid(True)
    plt.show()

def grafico_errores_delta(errores_delta):
    labels = [r'$\delta^2 = 10^{-2}\sigma_{max}$', r'$\delta^2 = 10^{-2}\sigma_{min}$', r'$\delta^2 = 0$', r'$\delta^2 = 5$', r'$\delta^2 = 25$', r'$\delta^2 = 100$']
    colors = ['mediumpurple', 'cornflowerblue', 'palegreen', 'greenyellow', 'coral', 'tomato']
    errores_porcentuales = {key: value * 100 for key, value in errores_delta.items()}
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, errores_porcentuales.values(), color=colors)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}%', ha='center', va='bottom', fontsize=12)
    
    plt.xlabel('Valores de $\delta$', fontsize=20)
    plt.ylabel('Errores relativos', fontsize=20)
    plt.title('Error relativo entre SVD y F2 en función de $\delta$', fontsize=20)
    plt.yticks(fontsize = 18)
    plt.xticks(fontsize = 18)
    plt.grid(True)
    plt.show()



def grafico_diferentes_steps(lista_steps):
    plt.figure(figsize=(10, 6))

    labels = [r'$s = 1/\lambda_{max}$', r'$s = 0.001$', r'$s = 0.0004$', r'$s = 0.005$']

    for idx, x in enumerate(lista_steps):
        plt.plot(x, label=f'{labels[idx]}')
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel('Iteraciones', fontsize=20)
    plt.ylabel('Valor de F2', fontsize=20)
    plt.title('Evolución de F2 con diferentes valores de $s$', fontsize=20)
    plt.xlim(10**(-1), 10**3.7)
    plt.ylim(10**(-3), 10**5)
    plt.yticks(fontsize = 18)
    plt.xticks(fontsize = 18)
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.show()

def grafico_errores_step(errores_step):
    labels = [r'$s = 1/\lambda_{max}$', r'$s = 0.001$', r'$s = 0.0004$', r'$s = 0.005$']
    colors = ['mediumpurple', 'palegreen', 'greenyellow', 'coral']
    errores_porcentuales = {key: value * 100 for key, value in errores_step.items()}
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, errores_porcentuales.values(), color=colors)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}%', ha='center', va='bottom', fontsize=12)
    
    plt.xlabel('Valores de $s$', fontsize=20)
    plt.ylabel('Errores relativos', fontsize=20)
    plt.title('Error relativo entre SVD y F2 en función de $s (step)$', fontsize=20)
    plt.yticks(fontsize = 18)
    plt.xticks(fontsize = 18)
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
    grafico_comparacion(x_F, x_F2, d, x_svd)

    lista = []
    delta_lista = [10**(-2) * valor_sing_max(A), 10**(-2) * valor_sing_min(A), 0, 5, 25, 100]
    for sub_delta in delta_lista:
        x_F2, iteraciones_F2, valores_F2 = iterativo(A, x0, b, s, lambda A, x, b: gradiente_funcion_regularizacion(A, x, b, sub_delta), epsilon, max_iteraciones)
        valores_F2_costo = [funcion_regularizacion(A, x, b, funcion_de_costo, sub_delta) for x in valores_F2]
        lista.append(valores_F2_costo)
    grafico_diferentes_delta(lista, b)

    errores_delta = {}
    for idx, sub_delta in enumerate(delta_lista):
        x_F2, iteraciones_F2, valores_F2 = iterativo(A, x0, b, s, lambda A, x, b: gradiente_funcion_regularizacion(A, x, b, sub_delta), epsilon, max_iteraciones)
        error = np.linalg.norm(x_svd - x_F2)/np.linalg.norm(x_svd)
        errores_delta[sub_delta] = error
    grafico_errores_delta(errores_delta)

    lista_steps = []
    steps = [s, 0.001, 0.0004, 0.005]
    for step in steps:
        x_F2, iteraciones_F2, valores_F2 = iterativo(A, x0, b, step, lambda A, x, b: gradiente_funcion_regularizacion(A, x, b, delta), epsilon, max_iteraciones)
        valores_F2_costo = [funcion_regularizacion(A, x, b, funcion_de_costo, delta) for x in valores_F2]
        lista_steps.append(valores_F2_costo)
    grafico_diferentes_steps(lista_steps)

    errores_step = {}
    for idx, step in enumerate(steps):
        x_F2, iteraciones_F2, valores_F2 = iterativo(A, x0, b, step, lambda A, x, b: gradiente_funcion_regularizacion(A, x, b, delta), epsilon, max_iteraciones)
        error = np.linalg.norm(x_svd - x_F2)/np.linalg.norm(x_svd)
        errores_step[step] = error
    grafico_errores_step(errores_step)



if __name__ == "__main__":
    main()
