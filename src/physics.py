"""
physics.py
==========
Cálculo de la derivada fraccionaria de Caputo mediante el esquema L1
y funciones auxiliares de fuente sísmica.

Formulación
-----------
En el modelo viscoelástico fraccionario, la relación constitutiva
presión–divergencia de velocidad (acústico) es:

    p(t) + (τ^α / Q) · D_t^α p(t) = −ρ v_P² · ∇·v(t)

donde D_t^α es el operador de Caputo de orden α ∈ (0, 1) y τ es el
tiempo de relajación de referencia (aquí τ = dt).

Discretización L1 (Lin et al., 2007)
--------------------------------------
    D_t^α p(t_n) ≈ (1 / Γ(2−α)) · Σ_{j=0}^{M−1} b_j · Δp_{n−j}

donde  b_j = (j+1)^{1−α} − j^{1−α}  y  Δp_{n−j} = p_{n−j} − p_{n−j−1}.

Referencias
-----------
- Blanch, J. O., Robertsson, J. O. A., & Symes, W. W. (1993).
  Viscoelastic finite-difference modeling.
- Lin, Y., Xu, C. (2007). Finite difference/spectral approximations
  for the time-fractional diffusion equation. J. Comput. Phys.
- Caputo, M. (1967). Linear models of dissipation whose Q is almost
  frequency independent. Geoph. J. R. Astr. Soc.
"""

import numpy as np
from math import gamma as _gamma


# ── Pesos del esquema L1 ─────────────────────────────────────────────────────

def caputo_weights(alpha: float, M: int) -> np.ndarray:
    """
    Coeficientes b_j del esquema L1 normalizados por Γ(2−α).

    El operador de Caputo discrito queda:
        D_t^α p_n ≈ (1/Γ(2−α)) Σ_j b_j Δp_{n−j}  →  caputo_weights[j] = b_j / Γ(2−α)

    Parámetros
    ----------
    alpha : orden fraccionario (0 < alpha < 1)
    M     : longitud de la ventana de memoria (número de pasos retenidos)

    Retorna
    -------
    w : ndarray de longitud M, decreciente, con w[0] ≥ w[1] ≥ …
    """
    j = np.arange(M, dtype=np.float64)
    b = (j + 1.0) ** (1.0 - alpha) - j ** (1.0 - alpha)   # b_j > 0, decreciente para j ≥ 1
    return b / _gamma(2.0 - alpha)


# ── Término de memoria (contribución Caputo) ─────────────────────────────────

def caputo_memory(p_curr: np.ndarray,
                  p_history: np.ndarray,
                  weights: np.ndarray) -> np.ndarray:
    """
    Calcula D_t^α p(t_n) multiplicado por el coeficiente de atenuación.

    La historia está ordenada como p_history[0] = p_{n-1} (más reciente),
    p_history[1] = p_{n-2}, …, p_history[M-1] = p_{n-M}.

    Usa la forma de diferencias del esquema L1:
        Δp_j = p_{n-j} − p_{n-j-1}

    Parámetros
    ----------
    p_curr    : campo de presión en t_n              (nx, nz)
    p_history : historial [p_{n-1}, …, p_{n-M}]     (M, nx, nz)
    weights   : salida de caputo_weights(alpha, M)   (M,)

    Retorna
    -------
    mem : (nx, nz) — término de memoria ya normalizado por Γ(2−α)
    """
    M = len(weights)

    # Construir array contiguo [p_n, p_{n-1}, …, p_{n-M}] (M+1, nx, nz)
    stack = np.empty((M + 1,) + p_curr.shape, dtype=p_curr.dtype)
    stack[0]  = p_curr
    stack[1:] = p_history

    # Diferencias: Δp_j = stack[j] − stack[j+1]   (M, nx, nz)
    diffs = stack[:-1] - stack[1:]

    # Suma ponderada: Σ_j w_j · Δp_j
    return np.tensordot(weights, diffs, axes=([0], [0]))


# ── Fuente sísmica ───────────────────────────────────────────────────────────

def ricker_wavelet(f0: float, t: float, t0: float = None) -> float:
    """
    Wavelet Ricker (sombrero mexicano) normalizado al pico unitario.

    Definición:
        R(t) = (1 − 2τ²) exp(−τ²),   τ = π f₀ (t − t₀)

    Parámetros
    ----------
    f0 : frecuencia pico (Hz)
    t  : tiempo actual (s)
    t0 : retardo temporal (s); si None → 1.5 / f0  (centrado en t = t0)
    """
    if t0 is None:
        t0 = 1.5 / f0
    tau = np.pi * f0 * (t - t0)
    return (1.0 - 2.0 * tau ** 2) * np.exp(-(tau ** 2))


def ricker_spectrum(f0: float, freqs: np.ndarray) -> np.ndarray:
    """
    Espectro de amplitud del wavelet Ricker (para análisis de frecuencias).

    |R(f)| = (2/√π) · (f/f0)² · exp(−(f/f0)²)   [normalizado]
    """
    ratio = freqs / f0
    return (2.0 / np.sqrt(np.pi)) * ratio ** 2 * np.exp(-(ratio ** 2))
