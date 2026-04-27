# El motor principal (loop de tiempo y actualización de matrices)
"""
solver.py
Motor de cálculo principal. Integra las condiciones físicas, 
la derivada fraccionaria y las capas PML.
"""

import numpy as np
import os
from src.config import nx, nz, dt, dx, dz, rho, vp, M
from src.physics import calculate_caputo_weights, compute_memory_term
from src.pml import get_pml_profile
from src.visualizer import plot_snapshot

def run_simulation():
    # 1. Inicialización de campos
    p = np.zeros((nx, nz))
    vx = np.zeros((nx, nz))
    vz = np.zeros((nx, nz))
    
    # Cargamos el perfil de amortiguamiento PML
    pml_damping = get_pml_profile(nx, nz)
    
    # Buffer de memoria (Sliding Window) para el término de Caputo
    p_buffer = np.zeros((M, nx, nz))
    
    # Pre-calculamos los pesos del kernel (alpha=0.5)
    weights = calculate_caputo_weights(0.5, M)
    
    # Se crea la carpeta de salida si no existe
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    
    print("Iniciando simulación de propagación de ondas...")
    
    # 2. Bucle de tiempo principal
    for t in range(1000): # nt = 1000
        # --- NUEVO: Inyección de fuente (Sismo en el centro) ---
        # Inyectamos energía solo en los primeros 50 pasos de tiempo
        if t < 50:
            p[nx // 2, nz // 2] += 500.0 * np.sin(2 * np.pi * 0.05 * t)
        
        # --- A. Actualización de Velocidades (Staggered Grid) ---
        # Ajustamos los índices para que el resultado de la resta sea (198, 200)
        # en lugar de (199, 200), coincidiendo con vx[1:-1, :]
        vx[1:-1, :] -= (dt / (rho * dx)) * (p[2:, :] - p[1:-1, :])
        vz[:, 1:-1] -= (dt / (rho * dz)) * (p[:, 2:] - p[:, 1:-1])
        
        # Aplicamos el PML
        vx *= (1 - pml_damping)
        vz *= (1 - pml_damping)
        
        # --- B. Cálculo del Término de Memoria (Caputo) ---
        # Este término modela la atenuación viscoelástica
        p_memory = compute_memory_term(p_buffer, weights)
        
        # --- C. Actualización de Presión ---
        divergence = (
            (vx[1:-1, 1:-1] - vx[:-2, 1:-1]) / dx + 
            (vz[1:-1, 1:-1] - vz[1:-1, :-2]) / dz
        )
        
        # Actualización integrando el término elástico y el de memoria
        p[1:-1, 1:-1] -= (vp**2 * rho * dt) * divergence + p_memory[1:-1, 1:-1]
        
        # Aplicamos el PML a la presión
        p *= (1 - pml_damping)
        
        # --- D. Actualización del Buffer (Sliding Window) ---
        p_buffer[:-1] = p_buffer[1:].copy()
        p_buffer[-1] = p.copy()
        
        # --- E. Visualización (Snapshots) ---
        if t % 50 == 0:
            snapshot_index = t // 50
            plot_snapshot(p, snapshot_index, t)
            print(f"Paso de tiempo {t} completado. Snapshot guardado.")
            
    print("Simulación finalizada exitosamente.")
    return p

if __name__ == "__main__":
    final_pressure = run_simulation()