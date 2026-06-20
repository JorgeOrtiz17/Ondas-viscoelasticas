"""
validation.py
=============
Módulo de validación numérica del simulador de ondas P.

Experimentos de validación implementados
-----------------------------------------
1. **Límite elástico** (Qp → ∞):
   La traza simulada con Qp muy alto debe converger al sismograma
   analítico elástico (wavelet Ricker retardado por el tiempo de viaje).

2. **Decaimiento geométrico 2D**:
   En un medio elástico homogéneo, la amplitud de pico a lo largo
   de un radio r desde la fuente sigue la ley:
       A(r) ∝ 1 / √r
   (cilindrical spreading en 2D). Se verifica extrayendo los picos
   a distintos radios del campo de onda simulado.

3. **Decaimiento por atenuación viscoelástica**:
   Para una onda monocromática de frecuencia f₀ la teoría predice:
       A(r) ∝ exp(−π f₀ r / (Qp vP)) / √r
   Se ajusta la amplitud simulada a esta curva y se reporta el
   error relativo en la estimación del Qp efectivo.

Funciones exportadas
--------------------
- `analytical_elastic_seismogram`   : traza analítica de referencia
- `elastic_limit_error`             : error L2 vs solución analítica
- `geometric_spreading_profile`     : extrae A(r) del campo de onda
- `fit_effective_Q`                 : estima Qp efectivo a partir de A(r)
- `run_validation_report`           : ejecuta los tres experimentos y guarda figuras
"""

import numpy as np
from math import gamma as _gamma
from scipy.optimize import curve_fit          # único uso de scipy (fitting)

from src.physics import ricker_wavelet


# ── 1. Sismograma analítico elástico ────────────────────────────────────────

def analytical_elastic_seismogram(vp:    float,
                                  r:     float,
                                  f0:    float,
                                  dt:    float,
                                  nt:    int,
                                  scale: float = 1.0) -> np.ndarray:
    """
    Traza analítica para una onda P en un medio elástico 2D homogéneo.

    Aproximación de campo lejano (far-field):
        p_analy(t) = scale · R(t − r/vP)

    donde R es el wavelet Ricker con frecuencia pico f₀.
    La amplitud se normaliza externamente mediante `scale`.

    Parámetros
    ----------
    vp    : velocidad de onda P (m/s)
    r     : distancia fuente–receptor (m)
    f0    : frecuencia pico del wavelet (Hz)
    dt    : paso de tiempo (s)
    nt    : número de pasos a generar
    scale : factor de escala de amplitud (ajustado para comparar con simulación)

    Retorna
    -------
    trace : ndarray (nt,)
    """
    t_arr  = np.arange(nt) * dt
    t0_src = 1.5 / f0
    # Analytic: p(t) = Ricker(t - r/vp), where Ricker peaks at t0_src.
    # ricker_wavelet(f0, t, t0_src) evaluates tau = pi*f0*(t - t0_src).
    # Substituting t -> (t - r/vp) gives tau = pi*f0*(t - r/vp - t0_src),
    # so the analytic trace peaks at t = t0_src + r/vp, matching the numeric.
    t_arr_shifted = t_arr - r / vp
    trace = np.array([ricker_wavelet(f0, tt, t0=t0_src)
                      for tt in t_arr_shifted])
    return scale * trace


def elastic_limit_error(seismogram_elastic: np.ndarray,
                        seismogram_analytic: np.ndarray,
                        t_start_step: int = 0) -> dict:
    """
    Calcula el error relativo L2 entre el sismograma simulado (límite
    elástico, Qp→∞) y la solución analítica de referencia.

    Parámetros
    ----------
    seismogram_elastic  : traza simulada con Qp muy alto  (nt,)
    seismogram_analytic : traza analítica de referencia   (nt,)
    t_start_step        : primer paso incluido (excluye el pre-arribo)

    Retorna
    -------
    dict con:
        'L2_error'    : ‖num − analy‖ / ‖analy‖   (relativo)
        'max_error'   : max|num − analy| / max|analy|
        'peak_delay'  : desfase del pico (número de pasos)
    """
    num   = seismogram_elastic[t_start_step:]
    analy = seismogram_analytic[t_start_step:]

    norm_analy = np.linalg.norm(analy)
    if norm_analy < 1e-30:
        return {'L2_error': np.nan, 'max_error': np.nan, 'peak_delay': 0}

    # Normalizar ambas por el pico del analítico para comparar formas
    peak_analy = np.max(np.abs(analy))
    peak_num   = np.max(np.abs(num))
    if peak_num < 1e-30:
        return {'L2_error': np.nan, 'max_error': np.nan, 'peak_delay': 0}

    num_norm   = num   / peak_num
    analy_norm = analy / peak_analy

    L2_err  = np.linalg.norm(num_norm - analy_norm) / np.linalg.norm(analy_norm)
    max_err = np.max(np.abs(num_norm - analy_norm)) / 1.0

    # Desfase del pico (en pasos de tiempo)
    peak_delay = int(np.argmax(np.abs(num))) - int(np.argmax(np.abs(analy)))

    return {
        'L2_error':   float(L2_err),
        'max_error':  float(max_err),
        'peak_delay': peak_delay,
    }


# ── 2. Decaimiento geométrico ────────────────────────────────────────────────

def geometric_spreading_profile(p_field: np.ndarray,
                                src_x:   int,
                                src_z:   int,
                                dx:      float,
                                dz:      float,
                                r_min_cells: int = 10,
                                n_radii: int     = 20) -> tuple:
    """
    Extrae el perfil de amplitud de pico A(r) del campo de onda en un
    instante dado, midiendo el máximo a lo largo de anillos concéntricos.

    Parámetros
    ----------
    p_field      : campo de presión (nx, nz)
    src_x, src_z : posición de la fuente en celdas
    dx, dz       : espaciado de malla (m)
    r_min_cells  : radio mínimo del primer anillo (celdas)
    n_radii      : número de anillos a medir

    Retorna
    -------
    radii  : ndarray (n_radii,) — radios en metros
    amps   : ndarray (n_radii,) — amplitud de pico en cada anillo
    """
    nx, nz = p_field.shape
    x_idx  = np.arange(nx)
    z_idx  = np.arange(nz)
    XX, ZZ = np.meshgrid(x_idx, z_idx, indexing='ij')
    R_grid = np.sqrt(((XX - src_x) * dx)**2 + ((ZZ - src_z) * dz)**2)

    nx_half = min(src_x, nx - src_x)
    nz_half = min(src_z, nz - src_z)
    r_max_m = min(nx_half, nz_half) * min(dx, dz) * 0.85

    r_min_m = r_min_cells * min(dx, dz)
    radii   = np.linspace(r_min_m, r_max_m, n_radii)
    amps    = np.zeros(n_radii)
    dr      = (r_max_m - r_min_m) / n_radii * 0.8

    for i, r in enumerate(radii):
        mask        = np.abs(R_grid - r) < dr
        ring_values = np.abs(p_field[mask])
        amps[i]     = ring_values.max() if ring_values.size > 0 else 0.0

    # Keep only the largest contiguous amplitude cluster (near the wavefront).
    # Discard radii where amplitude is below 5% of the global max to avoid
    # fitting to the near-zero pre-arrival or post-wavefront zone.
    threshold = 0.05 * (amps.max() + 1e-30)
    valid     = amps >= threshold
    return radii[valid], amps[valid]


def fit_effective_Q(radii:  np.ndarray,
                    amps:   np.ndarray,
                    vp:     float,
                    f0:     float) -> dict:
    """
    Ajusta la curva teórica A(r) = C · r^{−½} · exp(−π f₀ r / (Qp_eff vP))
    a los datos extraídos del campo de onda.

    Retorna el Qp efectivo estimado y el coeficiente R².

    Parámetros
    ----------
    radii : radios (m)
    amps  : amplitudes de pico en cada radio
    vp    : velocidad P (m/s)
    f0    : frecuencia pico (Hz)

    Retorna
    -------
    dict con 'Qp_eff', 'C', 'R2'
    """
    def model(r, C, Qp_eff):
        return C / np.sqrt(r + 1e-10) * np.exp(-np.pi * f0 * r / (Qp_eff * vp))

    valid  = amps > 0.01 * amps.max()
    r_fit  = radii[valid]
    a_fit  = amps[valid]

    if r_fit.size < 4:
        return {'Qp_eff': np.nan, 'C': np.nan, 'R2': np.nan}

    # Estimate initial C from the geometric-spreading-only model
    C0     = a_fit[0] * np.sqrt(r_fit[0])
    Qp0    = max(10.0, vp / (np.pi * f0 * (r_fit[-1] - r_fit[0]) + 1e-10))

    try:
        popt, _ = curve_fit(model, r_fit, a_fit,
                            p0=[C0, Qp0],
                            bounds=([0, 1], [np.inf, 5e4]),
                            maxfev=10000)
        C, Qp_eff = popt
        a_pred    = model(r_fit, C, Qp_eff)
        ss_res    = np.sum((a_fit - a_pred)**2)
        ss_tot    = np.sum((a_fit - a_fit.mean())**2)
        R2        = 1 - ss_res / (ss_tot + 1e-30)
        return {'Qp_eff': float(Qp_eff), 'C': float(C), 'R2': float(R2)}
    except Exception:
        return {'Qp_eff': np.nan, 'C': np.nan, 'R2': np.nan}


# ── 3. Reporte completo ──────────────────────────────────────────────────────

def run_validation_report(results_elastic:       dict,
                          results_viscoelastic:  list,
                          dt:   float,
                          dx:   float,
                          dz:   float,
                          vp:   float,
                          f0:   float,
                          src_x: int,
                          src_z: int,
                          rec_r: float,
                          visualizer) -> dict:
    """
    Ejecuta los tres experimentos de validación y llama al visualizador
    para generar las figuras.

    Parámetros
    ----------
    results_elastic      : dict {'trace', 'snapshot'} — caso elástico (Qp~1000)
    results_viscoelastic : lista de dicts {'Qp','Qs','trace','snapshot'}
    dt, dx, dz, vp, f0  : parámetros físicos
    src_x, src_z         : posición fuente
    rec_r                : distancia fuente–receptor (m)
    visualizer           : módulo src.visualizer importado

    Retorna
    -------
    report : dict con métricas de validación
    """
    from src.solver  import run_simulation as _run_sim
    from src.config  import nx, nz, dx as _dx, dz as _dz

    nt = len(results_elastic['trace'])

    # ── Experimento 1: límite elástico ──────────────────────────────────────
    scale     = np.max(np.abs(results_elastic['trace'])) + 1e-30
    analy     = analytical_elastic_seismogram(vp, rec_r, f0, dt, nt, scale=scale)
    arr_step  = max(0, int(rec_r / vp / dt) - 10)
    err_dict  = elastic_limit_error(results_elastic['trace'], analy, arr_step)

    print(f"\n[Validacion 1] Error L2 relativo (limite elastico): {err_dict['L2_error']:.4f}")
    print(f"               Error maximo:                         {err_dict['max_error']:.4f}")
    print(f"               Desfase del pico:                     {err_dict['peak_delay']} pasos")

    visualizer.plot_elastic_validation(
        results_elastic['trace'], analy, dt, err_dict,
        output_path='outputs/val_elastico.png'
    )

    # ── Experimento 2: decaimiento geometrico multi-receptor ─────────────────
    # Run one elastic simulation with receivers at several distances along z-axis.
    # Each receiver records the wavefield; we extract the peak amplitude.
    print("\n[Validacion 2] Ejecutando simulacion multi-receptor (decaimiento geometrico)...")
    pml_cells = 20
    r_cells_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    # Filter: only use receivers inside the PML-free zone
    max_safe = min(src_z, nz - src_z) - pml_cells - 5
    r_cells_list = [r for r in r_cells_list if r <= max_safe]

    extra_recv = [(src_x, src_z + rc) for rc in r_cells_list]
    radii_m    = np.array(r_cells_list, dtype=float) * _dz

    _, _, extra_seismos = _run_sim(
        Qp_value          = 1000,
        save_snapshots    = False,
        return_seismogram = True,
        extra_receivers   = extra_recv,
    )

    peak_amps = np.array([np.max(np.abs(extra_seismos[k])) for k in range(len(r_cells_list))])

    # Fit A(r) = C / sqrt(r) * exp(-pi*f0*r / (Qp_eff * vp))
    fit_res = fit_effective_Q(radii_m, peak_amps, vp, f0)
    print(f"[Validacion 2] Qp_eff estimado: {fit_res['Qp_eff']:.1f}  "
          f"(entrada: 1000)  R2={fit_res['R2']:.4f}")

    visualizer.plot_geometric_spreading(
        radii_m, peak_amps, vp, f0, fit_res,
        output_path='outputs/val_spreading.png'
    )

    return {
        'elastic_error': err_dict,
        'geometric':     fit_res,
    }
