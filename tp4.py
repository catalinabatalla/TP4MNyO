import numpy as np 
np.random.seed(1428)
def funcion_de_costo(A, b, x):
    c = A @ x - b 
    return c.T @ c

def gradiente_funcion_costo(A,x, b):
    gradiente = 2 * A.T @ (A @ x - b)
    return gradiente

def regulizacion(F, gamma, x, A, b):
    norm_x = np.linalg.norm(x)**2
    F2 = F(A,b, x) + gamma*norm_x
    return F2

def regularizacion_gradiente(gamma, x, A, b):
    grad = gradiente_funcion_costo(A, x, b) + 2*gamma*x
    return grad 

def iterativo(A, b, x0, step, epsilon, max_iteraciones):
    x = x0
    grad = gradiente_funcion_costo(A, x, b)
    iter = 0
    while grad > epsilon and max_iteraciones > iter:
        x = x - step*(A.T @ (A @ x - b))
        grad = gradiente_funcion_costo(A, x, b)
        iter += 1
    return x

def valor_sing_max(A):
    U, S, Vt = np.linalg.svd(A)
    return S[0]

def autovalor_max(A):
    Hessiana = 2* A.T @ A
    lambda_max = np.linalg.eigvals(Hessiana).max()
    return lambda_max

def main():
    n = 5
    d = 100
    A = np.random.rand(n,d)
    b = np.random.rand(n)
    gamma = 10**(-2)*valor_sing_max(A)
    s = 1/autovalor_max(A)
    x0 = np.random.rand(d)
    epsilon = 10**(-6)
    max_iteraciones = 1000

    #hacer que sea iterativo



if __name__ == "__main__":
    main()
