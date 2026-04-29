# Scripts para generar los snapshots (plots) y sismogramas
"""
visualizer.py
Módulo de visualización para snapshots de la propagación de ondas.
"""

import matplotlib.pyplot as plt

def create_snapshot_figure(p, t):
    """Genera y devuelve una figura de Matplotlib para un snapshot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(p.T, cmap='RdBu', origin='lower', aspect='auto', vmin=-0.05, vmax=0.05)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Amplitud de Presión')
    ax.set_title(f"Frente de onda en paso de tiempo {t}")
    ax.set_xlabel("Distancia X (nodos)")
    ax.set_ylabel("Distancia Z (nodos)")
    fig.tight_layout()
    return fig


def plot_snapshot(p, step, t):
    """Genera un heatmap del campo de presión y guarda la imagen."""
    fig = create_snapshot_figure(p, t)
    output_path = f"outputs/snapshot_{step:04d}.png"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Snapshot guardado: {output_path}")