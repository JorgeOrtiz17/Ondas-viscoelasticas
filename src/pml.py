import numpy as np

def get_pml_profile(nx, nz, thickness=20):
    """
    Genera un perfil de amortiguamiento (damping) para las fronteras.
    thickness: número de celdas de la capa PML.
    """
    damping = np.zeros((nx, nz))
    
    # Perfil suave (cuadrático) para evitar reflexiones bruscas
    for i in range(thickness):
        # Damping para los bordes izquierdo y derecho
        dist = (thickness - i) / thickness
        val = 0.15 * (dist**2)
        damping[i, :] += val
        damping[nx - 1 - i, :] += val
        
        # Damping para los bordes superior e inferior
        damping[:, i] += val
        damping[:, nz - 1 - i] += val
        
    return damping