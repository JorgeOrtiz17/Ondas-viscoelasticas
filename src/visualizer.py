# Scripts para generar los snapshots (plots) y sismogramas
"""
visualizer.py
Módulo de visualización para snapshots de la propagación de ondas.
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_snapshot(p, step, t):
    """
    Genera un heatmap del campo de presión.
    p: matriz de presión (nx, nz).
    step: número de paso para el nombre del archivo.
    """
    plt.figure(figsize=(8, 6))
    
    # 'RdBu' es el colormap estándar en geofísica para ver ondas (rojo/azul)
    # vmin/vmax ayuda a normalizar el contraste para ver bien la onda
    plt.imshow(p.T, cmap='RdBu', origin='lower', aspect='auto', vmin=-0.05, vmax=0.05)
    
    plt.colorbar(label='Amplitud de Presión')
    plt.title(f"Frente de onda en paso de tiempo {t}")
    plt.xlabel("Distancia X (nodos)")
    plt.ylabel("Distancia Z (nodos)")
    
    # Guardar la imagen para tu informe
    plt.savefig(f"outputs/snapshot_{step:04d}.png")
    plt.close() # Importante: cerrar para liberar memoria
    print(f"Snapshot guardado: outputs/snapshot_{step:04d}.png")