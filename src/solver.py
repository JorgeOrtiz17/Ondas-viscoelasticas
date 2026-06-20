"""
solver.py
=========
Motor de cálculo principal para la propagación de ondas P viscoelásticas
en malla intercalada (staggered grid) con derivada fraccionaria de Caputo.

Esquema numérico
----------------
Variables en la malla intercalada (Virieux, 1986):
  • Presión   p  :  nodos completos  (i,   j  )
  • Velocidad vx :  nodos medios     (i+½, j  )
  • Velocidad vz :  nodos medios     (i,   j+½)

Actualización (orden 2 en espacio, 1 en tiempo):

  [A] vx(i+½,j)^{n+½} = vx(i+½,j)^{n-½} − (dt/ρ dx)·[p(i+1,j)^n − p(i,j)^n]
  [B] vz(i,j+½)^{n+½} = vz(i,j+½)^{n-½} − (dt/ρ dz)·[p(i,j+1)^n − p(i,j)^n]
  [C] p(i,j)^{n+1}    = p(i,j)^n − ρ vP² dt · ∇·v^{n+½}
                        − att_coeff · D_t^α p(i,j)^n   (término Caputo)

Amortiguamiento PML
-------------------
Aplicado como factor multiplicativo semi-implícito:
  v *= (1 − σ·dt)   ;   p *= (1 − σ·dt)
donde σ·dt = pml_profile(…) ya es adimensional.

Condición CFL
-------------
  dt ≤ dx / (vP · √2)

Referencias
-----------
Virieux, J. (1986). P-SV wave propagation in heterogeneous media:
velocity-stress finite-difference method. Geophysics, 51(4), 889-901.

Blanch, J. O. et al. (1993). Viscoelastic finite-difference modeling.
SEP Report 80, Rice University.
"""

import os
import numpy as np
from math import gamma as _gamma

from src.config  import (nx, nz, dt, dx, dz, rho, vp, M, nt,
                         alpha, f_peak, Q, src_x, src_z, rec_x, rec_z)
from src.physics import caputo_weights, caputo_memory, ricker_wavelet
from src.pml     import pml_profile
from src.visualizer import plot_snapshot, plot_seismogram


def _att_coeff(Qp: float, alpha_val: float) -> float:
    """
    Coeficiente de atenuación adimensional para el término de Caputo.

    Derivado de la relación constitutiva fraccionaria:
        att = (dt^α · Γ(2−α)) / Qp
    El factor Γ(2−α) cancela el Γ en los pesos de Caputo para dejar
    un coeficiente efectivo = dt^α / Qp (físicamente motivado).
    """
    return (dt ** alpha_val) * _gamma(2.0 - alpha_val) / Qp


def run_simulation(vp_value:    float = vp,
                   rho_value:   float = rho,
                   alpha_value: float = alpha,
                   nt_value:    int   = nt,
                   Qp_value:    float = None,
                   snapshot_step: int = None,
                   snapshot_interval: int = 50,
                   callback   = None,
                   save_snapshots: bool = True,
                   return_seismogram: bool = False,
                   extra_receivers: list = None):
    """
    Ejecuta la simulación de propagación de onda P viscoelástica.

    Parámetros
    ----------
    vp_value, rho_value, alpha_value : propiedades físicas (anula los defaults de config)
    nt_value          : número total de pasos de tiempo
    Qp_value          : factor de calidad P; si None usa Q de config
    snapshot_step     : paso en que se captura el campo de onda completo para comparación
    snapshot_interval : frecuencia de guardado de snapshots (y callback)
    callback          : función f(p, t) llamada cada snapshot_interval pasos (para UI)
    save_snapshots    : si True guarda imágenes en outputs/
    return_seismogram : si True incluye el sismograma en el retorno

    Retorna
    -------
    p                          : campo de presión final (nx, nz)
    seismogram (opcional)      : traza en receptor único (nt_value,)
    snapshot_field (opcional)  : campo en snapshot_step  (nx, nz) o None
    """
    if Qp_value is None:
        Qp_value = Q

    # ── Verificación CFL ────────────────────────────────────────────────────
    cfl = vp_value * dt * np.sqrt(1/dx**2 + 1/dz**2)
    if cfl > 1.0:
        raise ValueError(
            f"Condición CFL violada: CFL={cfl:.3f} > 1. "
            f"Reduzca dt o aumente dx/dz."
        )

    # ── Inicialización de campos ────────────────────────────────────────────
    p  = np.zeros((nx, nz), dtype=np.float64)
    vx = np.zeros((nx, nz), dtype=np.float64)
    vz = np.zeros((nx, nz), dtype=np.float64)

    # ── PML con calibración física ──────────────────────────────────────────
    damping = pml_profile(nx, nz, thickness=20,
                          vp=vp_value, dt=dt, R=1e-3, order=3)

    # ── Pesos L1 de Caputo y coeficiente de atenuación ─────────────────────
    w       = caputo_weights(alpha_value, M)        # (M,) normalizados por Γ(2−α)
    att     = _att_coeff(Qp_value, alpha_value)     # adimensional

    # Historial: p_history[0] = p_{n-1}, …, p_history[M-1] = p_{n-M}
    p_history = np.zeros((M, nx, nz), dtype=np.float64)

    # ── Sismograma en receptor único y multi-receptor (opcional) ────────────
    seismogram     = np.zeros(nt_value, dtype=np.float64)
    snapshot_field = None
    # extra_receivers: list of (rx, rz) tuples for multi-distance recording
    n_extra   = len(extra_receivers) if extra_receivers else 0
    extra_rec = np.zeros((n_extra, nt_value), dtype=np.float64) if n_extra else None

    os.makedirs('outputs', exist_ok=True)

    t0_src       = 1.5 / f_peak
    source_steps = int(4.0 * t0_src / dt)

    print(f"  CFL = {cfl:.3f} | att_coeff = {att:.5f} | source_steps = {source_steps}")

    # ── Bucle temporal ──────────────────────────────────────────────────────
    for t_idx in range(nt_value):
        time = t_idx * dt

        # [0] Inyección de fuente Ricker — nodo único (más cercano a fuente puntual 2D)
        if t_idx < source_steps:
            amp = ricker_wavelet(f_peak, time, t0_src)
            p[src_x, src_z] += 1000.0 * amp

        # [A] Actualización de velocidades — staggered grid (orden 2)
        vx[1:-1, :] -= (dt / (rho_value * dx)) * (p[2:, :] - p[1:-1, :])
        vz[:, 1:-1] -= (dt / (rho_value * dz)) * (p[:, 2:] - p[:, 1:-1])

        # PML en velocidades
        vx *= (1.0 - damping)
        vz *= (1.0 - damping)

        # [B] Término de memoria de Caputo usando historial correcto
        mem = att * caputo_memory(p, p_history, w)

        # [C] Actualización de presión
        div = ((vx[1:-1, 1:-1] - vx[:-2, 1:-1]) / dx +
               (vz[1:-1, 1:-1] - vz[1:-1, :-2]) / dz)

        # Guardamos p_n ANTES de actualizar (necesario para buffer)
        p_prev = p.copy()

        p[1:-1, 1:-1] -= (vp_value**2 * rho_value * dt) * div + mem[1:-1, 1:-1]

        # PML en presión
        p *= (1.0 - damping)

        # [D] Actualización del historial (FIFO, más reciente en [0])
        p_history[1:] = p_history[:-1]
        p_history[0]  = p_prev

        # [E] Grabación en receptor principal y receptores adicionales
        seismogram[t_idx] = p[rec_x, rec_z]
        if extra_rec is not None:
            for k, (rx, rz) in enumerate(extra_receivers):
                extra_rec[k, t_idx] = p[rx, rz]

        # [F] Captura de snapshot de comparación
        if snapshot_step is not None and t_idx == snapshot_step:
            snapshot_field = p.copy()

        # [G] Callback / guardado periódico
        if t_idx % snapshot_interval == 0:
            if callback is not None:
                callback(p.copy(), t_idx)
            if save_snapshots:
                plot_snapshot(p, t_idx // snapshot_interval, t_idx)

    # ── Guardado de sismograma ──────────────────────────────────────────────
    if return_seismogram or save_snapshots:
        plot_seismogram(seismogram, dt, 'outputs/seismogram.png')

    if return_seismogram:
        if snapshot_step is not None:
            if extra_rec is not None:
                return p, seismogram, snapshot_field, extra_rec
            return p, seismogram, snapshot_field
        if extra_rec is not None:
            return p, seismogram, extra_rec
        return p, seismogram

    return p


if __name__ == "__main__":
    p, seismo = run_simulation(return_seismogram=True, save_snapshots=True)
    print(f"Simulación completada. Amplitud máxima: {np.max(np.abs(seismo)):.4f}")
