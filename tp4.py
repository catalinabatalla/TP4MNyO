import numpy as np 
np.random.seed(1428)
def funcion_de_costo(A, b, x):
    c = A @ x - b 
    return c.T @ c

def iterativo(A, b, x0, step, epsilon, max_iteraciones):
    x = x0
    c = funcion_de_costo(A, b, x)
    iter = 0
    while c > epsilon and max_iteraciones > iter:
        x = x - step*(A.T @ (A @ x - b))
        c = funcion_de_costo(A, b, x)
        iter += 1
    return x

def regulizacion(F, gamma, x, A, b):
    norm_x = np.linalg.norm(x)**2
    F2 = F(A,b, x) + gamma*norm_x
    return F2

def valor_sing_max(A):
    U, S, Vt = np.linalg.svd(A)
    return S[0]

def autovalor_max(A):
    Hessiana = A.T @ A
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

    #minimizo la funcion de costo
    x_F = iterativo(A, b, x0, s, epsilon, max_iteraciones)
    costo_F = funcion_de_costo(A, b, x_F)

    #minimizo la funcion de costo regularizada
    x_F2 = iterativo(A, b, x0, s, epsilon, max_iteraciones)
    costo_F2 = regulizacion(funcion_de_costo, gamma, x_F2, A, b)

    x_svd = np.linalg.pinv(A) @ b



if __name__ == "__main__":
    main()
