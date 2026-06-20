"""
main.py - Experimentos principales del trabajo de grado.

Experimento 1: Comparacion multi-Q
    outputs/figura3_snapshots.png  - campo de onda a t ~ 37.9 ms
    outputs/figura4_sismograma.png - sismogramas normalizados

Experimento 2: Validacion - limite elastico
    outputs/val_elastico.png  - numerico vs. analitico
    outputs/val_spreading.png - decaimiento geometrico 2D

Experimento 3: Sensibilidad - barrido de Qp
    outputs/sensibilidad_Q.png - amplitud vs. Qp
"""

import os
import numpy as np

import src.visualizer as vis
from src.solver     import run_simulation
from src.config     import dt, dx, dz, src_x, src_z, rec_x, rec_z, vp, f_peak
from src.validation import run_validation_report

os.makedirs('outputs', exist_ok=True)

# ---------------------------------------------------------------------------
# Parametros globales del experimento
# ---------------------------------------------------------------------------
TARGET_T_MS     = 37.9
SNAPSHOT_STEP   = int(TARGET_T_MS * 1e-3 / dt)
N_DISPLAY_MS    = 40.0
N_DISPLAY_STEPS = int(N_DISPLAY_MS * 1e-3 / dt)

RECEPTOR_DIST_M = abs(rec_z - src_z) * dz

Q_COMBINATIONS = [
    ( 6,  3),
    (10,  5),
    (20, 10),
    (50, 25),
]

ANIMATE = False   # True para generar GIFs (mas lento)


# ===========================================================================
# EXPERIMENTO 1: Comparacion multi-Q
# ===========================================================================
print("\n" + "="*65)
print(" EXPERIMENTO 1: Comparacion de atenuacion para distintos (Qp, Qs)")
print("="*65)

results_exp1 = []
frames_store = {}

for Qp, Qs in Q_COMBINATIONS:
    print(f"\n  Simulando Qp={Qp}, Qs={Qs}  (snapshot en paso {SNAPSHOT_STEP})")

    captured_frames = []

    def _frame_callback(p_snap, t_step, _frames=captured_frames):
        _frames.append(p_snap.copy())

    _, seismogram, snapshot = run_simulation(
        Qp_value          = Qp,
        snapshot_step     = SNAPSHOT_STEP,
        save_snapshots    = False,
        return_seismogram = True,
        callback          = _frame_callback if ANIMATE else None,
    )

    peak = np.max(np.abs(seismogram))
    print(f"     Amplitud maxima en receptor: {peak:.4f} Pa")

    results_exp1.append({
        'Qp':       Qp,
        'Qs':       Qs,
        'trace':    seismogram,
        'snapshot': snapshot,
    })

    if ANIMATE:
        frames_store[(Qp, Qs)] = captured_frames

print("\n  Generando figuras del Experimento 1...")

vis.plot_experiment_seismogram(
    results_exp1,
    dt              = dt,
    n_display_steps = N_DISPLAY_STEPS,
    output_path     = 'outputs/figura4_sismograma.png',
)

vis.plot_experiment_snapshots(
    results_exp1,
    dx          = dx,
    dz          = dz,
    t_ms        = TARGET_T_MS,
    output_path = 'outputs/figura3_snapshots.png',
)

if ANIMATE:
    for (Qp, Qs), frames in frames_store.items():
        vis.save_animation(frames, dt, snapshot_interval=50,
                           dx=dx, dz=dz,
                           output_path=f'outputs/animation_Qp{Qp}.gif')


# ===========================================================================
# EXPERIMENTO 2: Validacion - limite elastico
# ===========================================================================
print("\n" + "="*65)
print(" EXPERIMENTO 2: Validacion - limite elastico (Qp = 1000)")
print("="*65)

_, seismo_elastic, snap_elastic = run_simulation(
    Qp_value          = 1000,
    snapshot_step     = SNAPSHOT_STEP,
    save_snapshots    = False,
    return_seismogram = True,
)

result_elastic = {
    'trace':    seismo_elastic,
    'snapshot': snap_elastic,
}

report = run_validation_report(
    results_elastic      = result_elastic,
    results_viscoelastic = results_exp1,
    dt        = dt,
    dx        = dx,
    dz        = dz,
    vp        = vp,
    f0        = f_peak,
    src_x     = src_x,
    src_z     = src_z,
    rec_r     = RECEPTOR_DIST_M,
    visualizer = vis,
)


# ===========================================================================
# EXPERIMENTO 3: Sensibilidad - barrido de Qp
# ===========================================================================
print("\n" + "="*65)
print(" EXPERIMENTO 3: Sensibilidad - barrido de Qp")
print("="*65)

Q_SWEEP   = [5, 8, 12, 20, 30, 50, 80, 150, 300, 1000]
peak_amps = []

for Qp_sw in Q_SWEEP:
    print(f"  Qp = {Qp_sw:4d}", end='', flush=True)
    _, seismo_q = run_simulation(
        Qp_value          = Qp_sw,
        save_snapshots    = False,
        return_seismogram = True,
    )
    pk = np.max(np.abs(seismo_q))
    peak_amps.append(pk)
    print(f"   pico = {pk:.4f}")

vis.plot_Q_sensitivity(Q_SWEEP, peak_amps,
                       output_path='outputs/sensibilidad_Q.png')


# ===========================================================================
# Resumen final
# ===========================================================================
print("\n" + "="*65)
print(" EXPERIMENTOS COMPLETADOS")
print("="*65)
print("\n Figuras generadas en outputs/:")
for f in sorted(os.listdir('outputs')):
    if f.endswith(('.png', '.gif')):
        print(f"   {f}")
print()
