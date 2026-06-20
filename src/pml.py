"""
pml.py
======
Capa de Ajuste Perfecto (PML) para absorción de ondas en los bordes del dominio.

Formulación
-----------
La PML utiliza un perfil de amortiguamiento σ(x) que sigue una ley polinomial
de orden n (Collino & Tsogka, 2001):

    σ(x) = σ_max · (d/L)^n

donde d es la distancia al borde interior de la PML y L es el espesor total.

El valor óptimo de σ_max se calibra para obtener una reflexión numérica
objetivo R (típicamente R = 10⁻³ a 10⁻⁴):

    σ_max = −(n+1) · vp · ln(R) / (2 · L)

En el esquema de diferencias finitas, el amortiguamiento se aplica como:
    v_new = v_old · (1 − σ · dt)  − (dt / ρ) · ∇p
    p_new = p_old · (1 − σ · dt)  − ρ v_P² · dt · ∇·v

Referencia
----------
Collino, F. & Tsogka, C. (2001). Application of the PML absorbing layer
model to the linear elastodynamic problem in anisotropic heterogeneous media.
Geophysics, 66(1), 294-307.
"""

import numpy as np


def pml_profile(nx: int, nz: int,
                thickness: int,
                vp: float,
                dt: float,
                R: float = 1e-3,
                order: int = 3) -> np.ndarray:
    """
    Genera el perfil de amortiguamiento σ (adimensional, ya multiplicado por dt)
    para ser aplicado directamente en las actualizaciones de velocidad y presión.

    El factor σ·dt retornado satisface: 0 ≤ σ·dt < 1 en toda la malla.

    Parámetros
    ----------
    nx, nz    : dimensiones de la malla
    thickness : espesor de la PML en número de celdas
    vp        : velocidad de onda P (m/s) — para calibrar σ_max
    dt        : paso de tiempo (s)
    R         : coeficiente de reflexión objetivo (default 10⁻³)
    order     : orden del perfil polinomial (default 3)

    Retorna
    -------
    damping : ndarray (nx, nz), valores en [0, 1)
    """
    L     = thickness          # espesor PML en celdas (unidades de dx)
    sigma_max = -(order + 1) * vp * np.log(R) / (2.0 * L)   # s⁻¹
    sigma_max_dt = sigma_max * dt                             # adimensional

    # Clamp para evitar inestabilidades (σ·dt < 0.5 por precaución)
    sigma_max_dt = min(sigma_max_dt, 0.49)

    damping = np.zeros((nx, nz))

    for i in range(thickness):
        # d = distancia normalizada al borde interior de la PML (0 en interior, 1 en borde)
        d_norm = (thickness - i) / thickness        # 1 → borde externo, 0 → interior
        val    = sigma_max_dt * (d_norm ** order)

        # Aplicar a los cuatro bordes
        damping[i, :]        = np.maximum(damping[i, :],        val)
        damping[nx-1-i, :]   = np.maximum(damping[nx-1-i, :],   val)
        damping[:, i]        = np.maximum(damping[:, i],        val)
        damping[:, nz-1-i]   = np.maximum(damping[:, nz-1-i],   val)

    return damping
