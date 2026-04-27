# Constantes, propiedades del material, tamaño de malla (nx, nz)

"""
config.py
Configuración de parámetros físicos y numéricos para la simulación 
de propagación de ondas (Medio Viscoelástico).
"""

import numpy as np

# --- Dimensiones del Dominio ---
nx = 200          # Número de puntos en X
nz = 200          # Número de puntos en Z
dx = 10.0         # Espaciado espacial (metros) - Asegura resolución
dz = 10.0         # Espaciado espacial (metros)

# --- Propiedades del Material ---
# Estos valores determinan la física de la onda P
vp = 2500.0       # Velocidad de la onda P (m/s)
rho = 2200.0      # Densidad del medio (kg/m^3)

# --- Configuración Temporal ---
nt = 1000         # Total de pasos de tiempo
# La condición CFL: dt <= dx / (vp * sqrt(2)) 
# Para vp=2500, dx=10, el dt debe ser aprox < 0.0028
dt = 0.001        

# --- Parámetros de la Derivada Fraccionaria (Caputo) ---
alpha = 0.5       # Orden de la derivada (0 < alpha < 1)
# 0.5 es un buen punto de partida para viscoelasticidad (atenuación media)

# --- Configuración del Buffer de Memoria (Sliding Window) ---
M = 30            # Tamaño de la ventana de memoria (historial)

print(f"Configuración cargada: Malla {nx}x{nz} | dt: {dt}s")