import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Dimensiones de la región rectangular
a = 1.0  # Longitud en la dirección x
b = 1.0  # Longitud en la dirección y

# Número de puntos de malla en cada dirección
N = 100
M = 100

# Tamaño del paso en cada dirección
dx = a / N
dy = b / M

# Parámetro de las condiciones de frontera
v0 = 1.0

def initialize_V(N, M):
    
    """
    Inicializa una matriz V de tamaño N x M para representar el potencial eléctrico.

    Parámetros:
    - N (int): Número de filas de la matriz.
    - M (int): Número de columnas de la matriz.

    Devuelve:
    - V (numpy.ndarray): Matriz V inicializada con ceros, con las condiciones de frontera
      establecidas en los bordes y = 0 y y = b, donde el valor en la primera columna es -v0
      y en la última columna es v0.
    """
    V = np.zeros((N, M))
    V[:, 0] = -v0
    V[:, -1] = v0
    return V


def gauss_seidel(V, dx, dy, tolerance=1e-4, max_iterations=10000):
    
    """
   Aplica el método de Gauss-Seidel para resolver la ecuación de Laplace y actualizar el potencial eléctrico.

   Parámetros:
   - V (numpy.ndarray): Matriz que representa el potencial eléctrico.
   - dx (float): Tamaño del paso en la dirección x.
   - dy (float): Tamaño del paso en la dirección y.
   - tolerance (float, opcional): Tolerancia para la convergencia del método. El valor predeterminado es 1e-4.
   - max_iterations (int, opcional): Número máximo de iteraciones permitidas. El valor predeterminado es 10000.

   Devuelve:
   - V (numpy.ndarray): Matriz actualizada del potencial eléctrico después de aplicar el método de Gauss-Seidel.
   """
    iterations = 0
    while iterations < max_iterations:
        max_residual = 0.0
        for i in range(1, N-1):
            for j in range(1, M-1):
                # Calcula el nuevo valor del potencial usando el método de Gauss-Seidel
                new_V = 0.25 * (V[i+1, j] + V[i-1, j] + V[i, j+1] + V[i, j-1])
                residual = np.abs(new_V - V[i, j])
                if residual > max_residual:
                    max_residual = residual
                V[i, j] = new_V
        iterations += 1
        if max_residual < tolerance:
            break
    return V

def plot_potential_3d(V, dx, dy):
    
    """
    Grafica el potencial eléctrico en 3D.

    Parámetros:
    - V (numpy.ndarray): Matriz que representa el potencial eléctrico.
    - dx (float): Tamaño del paso en la dirección x.
    - dy (float): Tamaño del paso en la dirección y.
    """
    
    x = np.linspace(0, a, N)
    y = np.linspace(0, b, M)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, V.T, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Potencial (V)')
    ax.set_title('Potencial eléctrico (Solución Numérica)')
    plt.show()


def analytical_solution(x, y, n_max=1):
    
    """
    Calcula la solución analítica de la ecuación de Laplace para el potencial eléctrico.

    Parámetros:
    - x (float o numpy.ndarray): Coordenada(s) x donde se desea evaluar la solución.
    - y (float o numpy.ndarray): Coordenada(s) y donde se desea evaluar la solución.
    - n_max (int, opcional): Número máximo de términos de la serie utilizados en la solución.
                              El valor predeterminado es 1.

    Devuelve:
    - V (numpy.ndarray): Matriz que representa la solución analítica de la ecuación de Laplace
                         para el potencial eléctrico en las coordenadas dadas (x, y).
    """
    V = 0
    V1 = 0
    for n in range(1, n_max + 1):
        A_n=(4*v0/(np.pi*n)**2)*(1-(-1)**n)
        V += A_n*np.sin(n*np.pi*x/a)*((np.sinh(n*np.pi*y/a)*(1-np.cosh(n*np.pi*b/a))/np.sinh(n*np.pi*b/a))-np.cosh(n*np.pi*y/a))
    V1 = np.flip(V,axis=0)
    return (V1-V)/17.5

def plot_difference_scatter_3d(V_numerical, V_analytical, a, b):
    
    """
    Grafica la diferencia entre la solución numérica y la solución analítica en forma de dispersión en 3D.

    Parámetros:
    - V_numerical (numpy.ndarray): Matriz que representa la solución numérica del potencial eléctrico.
    - V_analytical (numpy.ndarray): Matriz que representa la solución analítica del potencial eléctrico.
    - a (float): Longitud del lado x del sistema rectangular.
    - b (float): Longitud del lado y del sistema rectangular.
    """
    
    x = np.linspace(0, a, V_numerical.shape[0])
    y = np.linspace(0, b, V_numerical.shape[1])
    X, Y = np.meshgrid(x, y)
    diff = V_numerical - V_analytical
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Y, X, diff, color='red')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Diferencia')
    ax.set_title('Diferencia entre Soluciones Numérica y Analítica')

# Definir una cuadrícula de puntos para evaluar la solución analítica
x = np.linspace(0, a, N)
y = np.linspace(0, b, M)
X, Y = np.meshgrid(x, y)

# Calcular la solución analítica en la cuadrícula
V_analytical = analytical_solution(X, Y)

# Inicialización de la matriz de potencial V
V = initialize_V(N, M)

# Resolución de la ecuación de Laplace usando el método de Gauss-Seidel
V = gauss_seidel(V, dx, dy)

# Graficar el potencial V
fig = plt.figure()
x = np.linspace(0, a, N)
y = np.linspace(0, b, M)
X, Y = np.meshgrid(x, y)
plt.imshow(V_analytical, origin="lower", cmap="viridis", extent=[0,b,0,a])
plt.colorbar(label='Potencial (V)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Potencial eléctrico (Solución Analítica)')
plt.imshow(V_analytical, origin="lower", cmap="viridis", extent=[0,b,0,a])
plt.show()

# Graficar el potencial V
fig = plt.figure()
x = np.linspace(0, a, N)
y = np.linspace(0, b, M)
X, Y = np.meshgrid(x, y)
plt.imshow(V.T, origin="lower", cmap="viridis", extent=[0,b,0,a])
plt.colorbar(label='Potencial (V)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Potencial eléctrico (Solución Numérica)')
plt.show()

# Graficar el potencial V numérico en 3D
plot_potential_3d(V, dx, dy)


# Graficar el potencial V analítico en 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, V_analytical, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlim(-1,1)
ax.set_zlabel('Potencial (V)')
ax.set_title('Potencial eléctrico (Solución Analítica)')
plt.show()

# Graficar la diferencia entre los potenciales numerico y analitico
fig = plt.figure()
plot_difference_scatter_3d(V.T, V_analytical, a, b)
plt.show()




