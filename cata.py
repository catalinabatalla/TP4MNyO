import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

np.random.seed(1428)

# Función para obtener la solución por SVD
def svd_vs(A, b):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    x_svd = Vt.T @ np.linalg.inv(np.diag(S)) @ U.T @ b
    return x_svd

# Función para reducir la dimensión de A a d dimensiones mediante SVD
def reducir_dim(A, d):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    A_reducido = U[:, :d] @ np.diag(S[:d]) @ Vt[:d, :]
    return A_reducido

# Generar datos aleatorios
n = 5
d = 100
A = np.random.rand(n, d)
b = np.random.rand(n)

# Ajuste por mínimos cuadrados para comparación
A_extendida = np.hstack([A, np.ones((n, 1))])
x_ls = np.linalg.lstsq(A_extendida, b, rcond=None)[0]

# PCA para reducción de dimensionalidad a 2 componentes principales para comparación
pca = PCA(n_components=2)
A_pca = pca.fit_transform(A)

# Solución mediante SVD para comparación
x_SVD = svd_vs(A, b)
A_reducido = reducir_dim(A, 2)

# Graficar los datos y las líneas ajustadas
plt.figure(figsize=(10, 6))
plt.scatter(A_reducido[:, 0], A_reducido[:, 1], label='Datos', color='blue', alpha=0.5)

# Calcular la pendiente y ordenada al origen para la línea de regresión SVD
slope_svd = -x_SVD[0] / x_SVD[1]
intercept_svd = 0  # Asumiendo que la línea de regresión pasa por el origen en este ejemplo

# Definir puntos para trazar la línea de regresión SVD
x_line_svd = np.linspace(np.min(A_reducido[:, 0]), np.max(A_reducido[:, 0]), 100)
y_line_svd = slope_svd * x_line_svd + intercept_svd

# Graficar la línea de regresión SVD
plt.plot(x_line_svd, y_line_svd, label='Línea de Regresión SVD', color='orange', linewidth=2)

# Calcular la línea de regresión lineal por mínimos cuadrados
x_min = np.min(A[:, 0])
x_max = np.max(A[:, 0])
y_min = x_ls[d] + x_min * x_ls[0]
y_max = x_ls[d] + x_max * x_ls[0]

# Graficar la línea de regresión lineal por mínimos cuadrados
plt.plot([x_min, x_max], [y_min, y_max], label='Regresión Lineal (Mínimos Cuadrados)', color='green', linewidth=2)

# Graficar la proyección PCA para comparación
plt.scatter(A_pca[:, 0], A_pca[:, 1], label='Proyección PCA', color='red', marker='x')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Comparación de Aproximaciones')
plt.legend()
plt.grid(True)
plt.show()
