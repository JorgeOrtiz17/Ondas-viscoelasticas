"""
visualizer.py
=============
Módulo de visualización para el simulador de ondas P viscoelásticas.

Incluye:
 • Snapshots en escala de grises (campo de presión 2D)
 • Sismograma de traza única normalizado
 • Figura de comparación multi-Q (estilo Figura 4 del anteproyecto)
 • Figura de snapshots comparativos (estilo Figura 3 del anteproyecto)
 • Animación del campo de ondas exportada a GIF
 • Figuras de validación (límite elástico, decaimiento geométrico)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')                          # backend sin pantalla (compatibilidad)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

# ── Estilo global de publicación ─────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         11,
    'axes.labelsize':    12,
    'axes.titlesize':    12,
    'legend.fontsize':   9,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'figure.dpi':        120,
    'lines.linewidth':   1.6,
    'axes.grid':         True,
    'grid.alpha':        0.25,
    'grid.linestyle':    '--',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'savefig.bbox':      'tight',
    'savefig.dpi':       150,
})

# Paleta de colores para las distintas combinaciones Q
_Q_COLORS  = ['#d62728', '#2ca02c', '#1f77b4', '#ff7f0e']
_Q_MARKERS = ['o', 's', '^', 'D']


# ── Utilidades internas ───────────────────────────────────────────────────────

def _normalize(arr: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(arr))
    return arr / peak if peak > 1e-30 else arr


def _physical_extent(nx, nz, dx, dz):
    """Retorna extent para imshow con ejes en metros."""
    return [0, nx * dx, nz * dz, 0]


# ── Funciones de UI en tiempo real (Streamlit) ────────────────────────────────

def create_snapshot_figure(p: np.ndarray, t: int) -> plt.Figure:
    """Figura de snapshot para la UI en tiempo real."""
    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = max(np.max(np.abs(p)) * 0.5, 1e-10)
    im   = ax.imshow(p.T, cmap='gray', origin='upper',
                     vmin=-vmax, vmax=vmax, aspect='equal')
    fig.colorbar(im, ax=ax, label='Amplitud de presión')
    ax.set_title(f"Campo de onda — paso {t}")
    ax.set_xlabel("X (nodos)")
    ax.set_ylabel("Z (nodos)")
    fig.tight_layout()
    return fig


def plot_snapshot(p: np.ndarray, step: int, t: int) -> None:
    """Guarda un snapshot en outputs/."""
    fig = create_snapshot_figure(p, t)
    fig.savefig(f"outputs/snapshot_{step:04d}.png")
    plt.close(fig)


def plot_seismogram(seismogram: np.ndarray, dt: float, output_path: str) -> None:
    """Guarda el sismograma de traza única normalizado."""
    nt   = len(seismogram)
    time = np.arange(nt) * dt * 1000.0

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time, _normalize(seismogram), color='#1f77b4')
    ax.axhline(0, color='gray', lw=0.6, ls='--')
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title("Sismograma sintético — receptor único")
    fig.savefig(output_path)
    plt.close(fig)


# ── Figuras de comparación multi-Q ───────────────────────────────────────────

def plot_experiment_seismogram(results:         list,
                               dt:              float,
                               n_display_steps: int,
                               output_path:     str) -> None:
    """
    Figura 4 del anteproyecto.
    Sismograma sintético normalizado para distintas combinaciones (Qp, Qs).

    Parámetros
    ----------
    results         : lista de dicts {'Qp','Qs','trace'}
    dt              : paso de tiempo (s)
    n_display_steps : pasos a mostrar en el eje de tiempo
    output_path     : ruta de salida
    """
    time_ms = np.arange(n_display_steps) * dt * 1000.0

    # Normaliza por el pico del caso menos atenuado (mayor Qp = último de la lista)
    ref_peak = np.max(np.abs(results[-1]['trace'][:n_display_steps])) + 1e-30

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, res in enumerate(results):
        trace = res['trace'][:n_display_steps] / ref_peak
        label = f"$(Q_p,\\,Q_s)=({res['Qp']},\\,{res['Qs']})$"
        ax.plot(time_ms, trace,
                color=_Q_COLORS[i % len(_Q_COLORS)],
                label=label)

    ax.axhline(0, color='black', lw=0.6, ls='--')
    ax.set_xlim(0, time_ms[-1])
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title("Synthetic seismogram of the pressure wave\n"
                 r"for different $(Q_p,\,Q_s)$ combinations in Experiment 1")
    ax.legend(loc='upper right')
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Sismograma de comparación guardado: {output_path}")


def plot_experiment_snapshots(results:    list,
                              dx:         float,
                              dz:         float,
                              t_ms:       float,
                              output_path: str) -> None:
    """
    Figura 3 del anteproyecto.
    Snapshots del campo de onda para distintas combinaciones Q.

    Parámetros
    ----------
    results     : lista de dicts {'Qp','Qs','snapshot'}
    dx, dz      : espaciado de malla (m)
    t_ms        : tiempo del snapshot (ms) — para el supertítulo
    output_path : ruta de salida
    """
    n   = len(results)
    fig = plt.figure(figsize=(5.5, 4.0 * n))
    fig.suptitle(
        f"Wave-field snapshots for different $(Q_p,\\,Q_s)$,  "
        f"$t \\approx {t_ms:.1f}$ ms",
        fontsize=12, y=1.01
    )
    gs = gridspec.GridSpec(n, 2, width_ratios=[20, 1], hspace=0.45, wspace=0.08)

    for row, res in enumerate(results):
        field = res.get('snapshot')
        if field is None:
            continue
        nx_, nz_ = field.shape
        vmax      = max(np.max(np.abs(field)) * 0.4, 1e-10)
        extent    = _physical_extent(nx_, nz_, dx, dz)

        ax  = fig.add_subplot(gs[row, 0])
        cax = fig.add_subplot(gs[row, 1])

        im = ax.imshow(field.T, cmap='gray', origin='upper',
                       extent=extent, vmin=-vmax, vmax=vmax,
                       aspect='equal', interpolation='bilinear')
        ax.set_title(f"$Q_p={res['Qp']}$,  $Q_s={res['Qs']}$", fontsize=10)
        ax.set_xlabel("x (m)", fontsize=9)
        ax.set_ylabel("z (m)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))

        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Amplitude", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    fig.savefig(output_path)
    plt.close(fig)
    print(f"Snapshots de comparación guardados: {output_path}")


# ── Animación ────────────────────────────────────────────────────────────────

def save_animation(frames:      list,
                   dt:          float,
                   snapshot_interval: int,
                   dx:          float,
                   dz:          float,
                   output_path: str = 'outputs/animation.gif',
                   fps:         int = 8) -> None:
    """
    Exporta los frames capturados durante la simulación a un GIF animado.

    Parámetros
    ----------
    frames            : lista de arrays (nx, nz) — uno por snapshot_interval
    dt                : paso de tiempo (s)
    snapshot_interval : pasos entre frames
    dx, dz            : espaciado de malla (m)
    output_path       : ruta del GIF de salida
    fps               : fotogramas por segundo
    """
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:
        print("Pillow no disponible — animación omitida.")
        return

    if not frames:
        return

    nx_, nz_  = frames[0].shape
    extent    = _physical_extent(nx_, nz_, dx, dz)
    global_vmax = max(np.max(np.abs(f)) for f in frames) * 0.5 + 1e-10

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(frames[0].T, cmap='gray', origin='upper',
                   extent=extent, vmin=-global_vmax, vmax=global_vmax,
                   aspect='equal', interpolation='bilinear')
    fig.colorbar(im, ax=ax, label="Amplitude")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    time_text = ax.set_title("")

    def _update(i):
        im.set_data(frames[i].T)
        t_ms = i * snapshot_interval * dt * 1000.0
        time_text.set_text(f"$t = {t_ms:.1f}$ ms")
        return [im, time_text]

    anim = FuncAnimation(fig, _update, frames=len(frames), blit=True)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Animación guardada: {output_path}")


# ── Figuras de validación ────────────────────────────────────────────────────

def plot_elastic_validation(seismo_num:   np.ndarray,
                            seismo_analy: np.ndarray,
                            dt:           float,
                            error_dict:   dict,
                            output_path:  str) -> None:
    """
    Compara el sismograma numérico (límite elástico) con la solución analítica.
    """
    nt   = len(seismo_num)
    time = np.arange(nt) * dt * 1000.0

    peak_n = np.max(np.abs(seismo_num))   + 1e-30
    peak_a = np.max(np.abs(seismo_analy)) + 1e-30

    fig, axes = plt.subplots(2, 1, figsize=(9, 7),
                             gridspec_kw={'height_ratios': [3, 1]})

    ax = axes[0]
    ax.plot(time, seismo_analy / peak_a,
            color='#1f77b4', lw=2.0, label='Analítico (elástico)', zorder=3)
    ax.plot(time, seismo_num   / peak_n,
            color='#d62728', lw=1.4, ls='--', label='Numérico ($Q_p \\to \\infty$)', zorder=2)
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Normalized amplitude")
    ax.set_title(
        f"Validation — Elastic limit ($Q_p \\to \\infty$)\n"
        f"$L_2$ error = {error_dict['L2_error']:.4f}  |  "
        f"Max error = {error_dict['max_error']:.4f}  |  "
        f"Peak delay = {error_dict['peak_delay']} steps"
    )
    ax.legend()

    # Residual
    n_min = min(len(seismo_num), len(seismo_analy))
    resid = seismo_num[:n_min] / peak_n - seismo_analy[:n_min] / peak_a
    axes[1].plot(time[:n_min], resid, color='#2ca02c', lw=1.2)
    axes[1].axhline(0, color='gray', lw=0.5, ls=':')
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Diferencia numérico − analítico")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Figura de validación elástica guardada: {output_path}")


def plot_geometric_spreading(radii:      np.ndarray,
                             amps:       np.ndarray,
                             vp:         float,
                             f0:         float,
                             fit_result: dict,
                             output_path: str) -> None:
    """
    Decaimiento geométrico 2D: A(r) vs. r, con ajuste teórico.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(radii, amps, color='#1f77b4', s=25, zorder=3,
               label='Amplitud extraída del campo de onda')

    Qp_eff = fit_result.get('Qp_eff', np.nan)
    C      = fit_result.get('C',      np.nan)
    R2     = fit_result.get('R2',     np.nan)

    if not np.isnan(Qp_eff):
        r_fine  = np.linspace(radii.min(), radii.max(), 300)
        a_fit   = C / np.sqrt(r_fine) * np.exp(-np.pi * f0 * r_fine / (Qp_eff * vp))
        ax.plot(r_fine, a_fit, color='#d62728', lw=2,
                label=f'Ajuste: $Q_p^{{\\rm eff}}={Qp_eff:.1f}$, $R^2={R2:.4f}$')

    # Curva pura de spreading 1/√r (normalizada)
    a_geo = amps[0] * np.sqrt(radii[0]) / np.sqrt(radii)
    ax.plot(radii, a_geo, color='gray', lw=1.2, ls='--',
            label=r'Decaimiento geométrico $\propto r^{-1/2}$')

    ax.set_xlabel("Distancia desde la fuente $r$ (m)")
    ax.set_ylabel("Amplitud de pico")
    ax.set_title(r"Decaimiento de amplitud: geométrico vs. viscoelástico")
    ax.legend()
    ax.set_yscale('log')
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Figura de decaimiento geométrico guardada: {output_path}")


def plot_Q_sensitivity(q_values:     list,
                       peak_amps:    list,
                       output_path:  str) -> None:
    """
    Amplitud de pico en el receptor vs. Qp (curva de sensibilidad).

    Parámetros
    ----------
    q_values  : lista de Qp ensayados
    peak_amps : lista de amplitudes máximas en el receptor
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(q_values, peak_amps,
            'o-', color='#1f77b4', ms=7, lw=1.8, label='Amplitud numérica')

    # Curva analítica: A ∝ exp(−π f₀ r / (Q vP))  [normalizada al Q más alto]
    from src.config import f_peak, vp, dx, rec_z, src_z
    r = abs(rec_z - src_z) * dx
    q_arr   = np.linspace(min(q_values), max(q_values), 200)
    a_analy = np.exp(-np.pi * f_peak * r / (q_arr * vp))
    a_analy /= a_analy.max()
    a_ref    = np.array(peak_amps)
    a_ref   /= a_ref.max()
    ax.plot(q_arr, a_analy * a_ref.max() / a_analy[-1],
            '--', color='#d62728', lw=1.5, label='Predicción analítica (campo lejano)')

    ax.set_xlabel("$Q_p$")
    ax.set_ylabel("Amplitud de pico en el receptor (Pa)")
    ax.set_title("Sensibilidad de la amplitud al factor de calidad $Q_p$")
    ax.legend()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Figura de sensibilidad Q guardada: {output_path}")
