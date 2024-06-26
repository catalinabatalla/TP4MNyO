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
    F2 = F(A, x, b) + delta*norm_x
    return F2

def gradiente_funcion_regularizacion(A, x, b, delta):
    gradiente_F = 2 * A.T @ (A @ x - b)
    delta = 10**(-2) * valor_sing_max(A)
    grad = gradiente_F + 2*delta*x
    return grad 

def iterativo(A, x0, b, step, gradiente_func, epsilon, max_iteraciones):
    x = x0
    iteraciones = []
    valores = []
    for iter in range(1, max_iteraciones):
        grad = gradiente_func(A, x, b)
        x = x - step * grad
        iteraciones.append(iter)
        valores.append(x)
        # if np.linalg.norm(grad) < epsilon:
        #     break
    return x, iteraciones, valores

def valor_sing_max(A):
    U, S, Vt = np.linalg.svd(A)
    return S[0]

def autovalor_max(A):
    Hessiana = 2* A.T @ A
    lambda_max = np.linalg.eigvals(Hessiana).max()
    return lambda_max

def pca_manual(A, n_components):
    U, S, Vt = np.linalg.svd(A)
    U_k = U[:, :n_components]
    S_k = S[:n_components]
    Vt_k = Vt[:n_components, :]
    A_reduced = U_k @ np.diag(S_k) 
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

def main():
    n = 5
    d = 100
    A = np.random.rand(n,d)
    b = np.random.rand(n)
    delta = 10**(-2) * valor_sing_max(A)
    s = 1/autovalor_max(A)
    # x0 = np.random.rand(d)
    x0 = np.random.uniform(0, 1, d)
    epsilon = 10**(-6)
    max_iteraciones = 1000

    # Minimización F1
    x_F, iteraciones_F, valores_F = iterativo(A, x0, b, s, gradiente_funcion_costo, epsilon, max_iteraciones)
    valores_F_costo = [funcion_de_costo(A, x, b) for x in valores_F]
    
    # Minimización F2
    x_F2, iteraciones_F2, valores_F2 = iterativo(A, x0, b, s, lambda A, x, b: gradiente_funcion_regularizacion(A, x, b, delta), epsilon, max_iteraciones)
    valores_F2_costo = [funcion_regularizacion(A, x, b, funcion_de_costo, delta) for x in valores_F2]

    # Minimización F1 con x generados por F2
    valores_F1_con_F2 = [funcion_de_costo(A, x, b) for x in valores_F2]

    # Minimización con SVD
    # A_reduced U_k, S_k, Vt_k = pca_manual(A, 2)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    x_approx = calcular_x(U, S, Vt, b)
    print(x_approx)
    valores_F_costo_SVD = funcion_de_costo(A, x_approx, b) 
    print(f'el valor de la funcion de costo con la aproximacion usando SVD es {valores_F_costo_SVD}')
    
    # GRÁFICO 1: CONVERGENCIA DE LAS SOLUCIONES 

    # print("Solución minimizando F:")
    # print(x_F)
    # print("Solución minimizando F2:")
    # print(x_F2)

    plt.yscale("log")
    plt.plot(iteraciones_F, valores_F_costo, lw=2, label=r"Minimización de $F(x)$")
    plt.plot(iteraciones_F2, valores_F2_costo, lw=2, label=r"Minimización de $F_2(x)$", color="orange")
    # plt.plot(iteraciones_F2, valores_F1_con_F2, label="F1 con penalizacion")
    plt.axhline(valores_F_costo_SVD, color='green', linestyle='--', lw=3, label="Aproximación con SVD")
    plt.axhline(delta * np.linalg.norm(x_F2)**2, color='red', lw=3, linestyle='--', label=r"$\delta \|x\|^2$")
    plt.xlabel("Número de iteración", fontsize=25)
    plt.ylabel("Valor de la función de costo", fontsize=25)
    plt.legend(fontsize=20)
    #plt.title("Evolución de la función de costo")
    plt.show()

    # GRAFICO 2: NORMA DE X EN FUNCION DE LAS ITERACIONES 

    norm_x_F = [np.linalg.norm(x) for x in valores_F]
    norm_x_F2 = [np.linalg.norm(x) for x in valores_F2]

    plt.figure()
    plt.plot(iteraciones_F, norm_x_F, label=r"Norma 2 de x con $F(x)$")
    plt.plot(iteraciones_F2, norm_x_F2, label=r"Norma 2 de x con $F_2(x)$)", color="orange")
    plt.axhline(np.linalg.norm(x_approx), color='green', linestyle='--', lw=3, label=r"Norma 2 de x con SVD")
    plt.xlabel("Iteraciones", fontsize=30)
    plt.ylabel("Norma 2 de x", fontsize=30)
    plt.legend(fontsize=20)
    plt.show()

    # GRAFICO 2.1: ERRORES RELATIVOS

    erroresF1 = [np.linalg.norm(x - x_approx)/np.linalg.norm(x_approx) for x in valores_F]
    erroresF2 = [np.linalg.norm(x - x_approx)/np.linalg.norm(x_approx) for x in valores_F2]

    plt.figure()
    plt.plot(iteraciones_F, erroresF1, label=r"Error relativo de minimizar $F(x)$")
    plt.plot(iteraciones_F2, erroresF2, label=r"Error relativo de minimizar $F_2(x)$")
    plt.xlabel("Iteraciones", fontsize=30)
    plt.ylabel("Error relativo", fontsize=30)
    plt.legend(fontsize=20)
    plt.show()

    # GRAFICO 3: ERROR COMETIDO EN LAS APROXIMACIONES

    error_F = [np.linalg.norm(A @ x - b) for x in valores_F]
    error_F2 = [np.linalg.norm(A @ x - b) for x in valores_F2]
    
    plt.figure()
    plt.plot(iteraciones_F, error_F, label="Error en la aproximación (F)")
    plt.plot(iteraciones_F2, error_F2, label="Error en la aproximación (F2)", color="orange")
    plt.xlabel("Iteraciones")
    plt.ylabel("Error (norma de Ax - b)")
    plt.legend()
    plt.title("Evolución del error en las aproximaciones")
    plt.show()

if __name__ == "__main__":
    main()
