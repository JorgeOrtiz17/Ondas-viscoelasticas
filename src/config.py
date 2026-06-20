"""
config.py
Parámetros físicos y numéricos para la simulación de propagación de ondas.
"""

import numpy as np

# Dimensiones del Dominio
nx = 200          # Nodos en X
nz = 200          # Nodos en Z
dx = 1.5          # Espaciado espacial (m) → dominio de 300 m × 300 m
dz = 1.5

# Propiedades del Material (valores por defecto)
vp  = 2600.0      # Velocidad de onda P (m/s)
rho = 2200.0      # Densidad (kg/m³)
Q   = 20.0        # Factor de calidad genérico
Qp  = 20.0        # Factor de calidad para onda P
Qs  = 10.0        # Factor de calidad para onda S

# Configuración Temporal
# Condición CFL: dt ≤ dx / (vp * sqrt(2)) = 1.5 / (2600 * 1.414) = 4.08e-4 s
dt  = 2.0e-4      # Paso temporal (s)  — conservador respecto al límite CFL
nt  = 250         # Pasos totales → 50 ms de simulación

# Derivada Fraccionaria de Caputo
alpha = 0.5       # Orden fraccionario (0 < alpha < 1)
M     = 30        # Ventana de memoria (historial de pasos)

# Coeficiente de atenuación: incluye dt^alpha para escala dimensional correcta
attenuation_coeff = 2.0 * alpha * np.pi * (dt ** alpha) / Q

# Fuente sísmica
f_peak = 300.0    # Frecuencia pico del wavelet Ricker (Hz)

# Posición de la fuente y receptor (en índices de malla)
src_x = nx // 2   # 100
src_z = nz // 2   # 100  — centro del dominio
rec_x = src_x      # mismo X que la fuente
rec_z = src_z + 33 # 33 celdas → 49.5 m ≈ 6 wavelengths (lambda=8.7m) → llegada ≈ 19 ms

print(f"Configuración: malla {nx}×{nz} | dx={dx}m | dt={dt*1000:.2f}ms | vp={vp}m/s | f_peak={f_peak}Hz")
