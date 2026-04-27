# Funciones para la derivada fraccionaria y el operador de Caputo

"""
physics.py
Lógica de cálculo para la derivada fraccionaria de Caputo.
"""

import numpy as np

def calculate_caputo_weights(alpha, M):
    """
    Calcula los coeficientes del kernel de memoria (L1 approximation).
    alpha: Orden de la derivada (0 < alpha < 1).
    M: Tamaño de la ventana de memoria.
    """
    weights = np.zeros(M)
    for i in range(M):
        # Fórmula discreta para los pesos del operador L1
        weights[i] = (i**(1 - alpha) - (i - 1)**(1 - alpha)) if i > 0 else 1.0
    return weights

def compute_memory_term(p_buffer, weights):
    """
    Calcula el término de memoria usando el buffer deslizante.
    p_buffer: Matriz de (M, nx, nz) con los estados pasados.
    weights: Coeficientes de Caputo.
    """
    # Usamos np.tensordot para multiplicar los pesos por cada snapshot del buffer
    # Esto es mucho más rápido que un ciclo for en Python.
    # El resultado es una matriz (nx, nz) que representa la atenuación acumulada.
    
    # Invertimos el orden de weights para que coincida con el tiempo (t-tau)
    # y hacemos la suma ponderada
    memory_term = np.tensordot(weights[::-1], p_buffer, axes=(0, 0))
    
    return memory_term