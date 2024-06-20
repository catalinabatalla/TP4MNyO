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
    for iter in range(max_iteraciones):
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

def svd_vs(A, b):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    #x = V S^(-1) Ut b 
    x_svd = Vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ b
    return x_svd

def reducir_dim(A, d):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    A_2D = U[:, :d] @ np.diag(S[:d]) @ Vt[:d, :]
    return A_2D

def main():
    n = 5
    d = 100
    A = np.random.rand(n,d)
    b = np.random.rand(n)
    delta = 10**(-2) * valor_sing_max(A)
    s = 1/autovalor_max(A)
    x0 = np.random.rand(d)
    epsilon = 10**(-6)
    max_iteraciones = 1000

    # Minimización F1
    x_F, iteraciones_F, valores_F = iterativo(A, x0, b, s, gradiente_funcion_costo, epsilon, max_iteraciones)
    valores_F_costo = [funcion_de_costo(A, x, b) for x in valores_F]
    
    # Minimización F2
    x_F2, iteraciones_F2, valores_F2 = iterativo(A, x0, b, s, lambda A, x, b: gradiente_funcion_regularizacion(A, x, b, delta), epsilon, max_iteraciones)
    valores_F2_costo = [funcion_regularizacion(A, x, b, funcion_de_costo, delta) for x in valores_F2]
    
    # GRÁFICO 1: CONVERGENCIA DE LAS SOLUCIONES 

    print("Solución minimizando F:")
    print(x_F)
    print("Solución minimizando F2:")
    print(x_F2)

    plt.plot(iteraciones_F, valores_F_costo, label="F")
    plt.plot(iteraciones_F2, valores_F2_costo, label="F2", color="orange")
    plt.axhline(0, color='grey', linestyle='--', label="F = 0")
    plt.axhline(delta * np.linalg.norm(x_F2)**2, color='red', linestyle='--', label=r"$\delta \|x\|^2$")
    plt.xlabel("Iteraciones")
    plt.ylabel("Valor de la función de costo")
    plt.legend()
    plt.title("Evolución de la función de costo")
    plt.show()

    # GRAFICO 2: NORMA DE X EN FUNCION DE LAS ITERACIONES 

    norm_x_F = [np.linalg.norm(x) for x in valores_F]
    norm_x_F2 = [np.linalg.norm(x) for x in valores_F2]

    plt.figure()
    plt.plot(iteraciones_F, norm_x_F, label="Norma 2 de x (F)")
    plt.plot(iteraciones_F2, norm_x_F2, label="Norma 2 de x (F2)", color="orange")
    plt.xlabel("Iteraciones")
    plt.ylabel("Norma 2 de x")
    plt.legend()
    plt.title("Evolución de la norma 2 de x")
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

    # GRAFICO 4: COMPARACION CON SVD
    # Solución mediante SVD
    x_SVD = svd_vs(A, b)

    # Reducción de dimensionalidad
    A_2D = reducir_dim(A, 2)
    
    # Graficar los datos y la aproximación por SVD
    plt.figure(figsize=(10, 6))
    plt.scatter(A_2D[:, 0], A_2D[:, 1], label='Datos', color='blue', alpha=0.5)

    # Calcular la pendiente y ordenada al origen para la línea de regresión
    slope = -x_SVD[0] / x_SVD[1]
    intercept = 0  # Asumiendo que la línea de regresión pasa por el origen en este ejemplo
    
    # Definir puntos para trazar la línea de regresión
    x_line = np.linspace(np.min(A_2D[:, 0]), np.max(A_2D[:, 0]), 100)
    y_line = slope * x_line + intercept
    
    # Graficar la línea de regresión
    plt.plot(x_line, y_line, label='Línea de Regresión SVD', color='orange', linewidth=2)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Aproximación de los datos comparando con SVD')
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    main()
