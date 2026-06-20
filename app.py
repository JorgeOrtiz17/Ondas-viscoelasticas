"""
app.py — Interfaz web interactiva (Streamlit)

Pestañas
--------
1. Simulación    : parámetros libres, visualización en tiempo real
2. Experimento Q : comparación de múltiples combinaciones (Qp, Qs)
3. Validación    : límite elástico vs. solución analítica
"""

import os
import numpy as np
import streamlit as st
from PIL import Image

from src.config import dx, dz, rho, vp, alpha, nt, f_peak, Qp, Qs
from src.solver import run_simulation
import src.visualizer as vis
from src.validation import (analytical_elastic_seismogram,
                             elastic_limit_error,
                             geometric_spreading_profile,
                             fit_effective_Q)

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Simulación de Ondas P Viscoelásticas",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Simulación de Propagación de Ondas P en Medios Viscoelásticos")
st.markdown(
    "**Método:** Diferencias finitas en malla intercalada (staggered grid) · "
    "**Física:** Derivada fraccionaria de Caputo · "
    "**Fronteras:** Capa de ajuste perfecto (PML)\n\n"
    "*Universidad de Pamplona — Trabajo de Grado Ingeniería de Sistemas*"
)

os.makedirs('outputs', exist_ok=True)

# ── Pestañas ──────────────────────────────────────────────────────────────────
tab_sim, tab_exp, tab_val = st.tabs([
    "🔬 Simulación",
    "📊 Experimento Q",
    "✅ Validación",
])


# ═══════════════════════════════════════════════════════════════════════════════
# PESTAÑA 1 — Simulación interactiva
# ═══════════════════════════════════════════════════════════════════════════════
with tab_sim:
    with st.sidebar:
        st.header("⚙️ Parámetros físicos")

        st.subheader("Material")
        vp_val    = st.slider("Velocidad P  (m/s)",   1000,  6000,  int(vp),    100)
        rho_val   = st.slider("Densidad     (kg/m³)", 1000,  3000,  int(rho),    50)
        Qp_val    = st.slider("Factor Qp",               5,   300,  int(Qp),      5)

        st.subheader("Numérico")
        alpha_val = st.slider("Orden α  (Caputo)",    0.10,  0.90, float(alpha), 0.05)
        nt_val    = st.number_input("Pasos de tiempo", 100, 2000, int(nt), 50)
        fpeak_val = st.slider("Frecuencia pico  (Hz)", 50,  500,  int(f_peak),  50)
        snap_int  = st.number_input("Intervalo snapshot", 10, nt_val, 50, 10)
        save_snap = st.checkbox("Guardar snapshots en outputs/", value=True)

        run_btn = st.button("▶️ Iniciar simulación", use_container_width=True)

    status = st.empty()
    pbar   = st.progress(0)

    col_plot, col_log = st.columns(2)
    with col_plot:
        plot_ph = st.empty()
    with col_log:
        log_ph  = st.empty()

    def _callback(field, t_step):
        pbar.progress(int(100 * (t_step + 1) / nt_val))
        status.markdown(f"**t = {t_step * 1e3 * 1.0 / (nt_val / (nt_val * 0.001 + 1)):.2f} ms** "
                        f"| paso {t_step}")
        plot_ph.pyplot(vis.create_snapshot_figure(field, t_step))
        log_ph.text(f"Snapshot en paso {t_step}")

    if run_btn:
        status.info("⏳ Ejecutando… puede tardar 1–2 minutos.")
        try:
            p_fin, seismo = run_simulation(
                vp_value   = vp_val,
                rho_value  = rho_val,
                Qp_value   = Qp_val,
                alpha_value = alpha_val,
                nt_value   = nt_val,
                snapshot_interval = snap_int,
                callback   = _callback,
                save_snapshots = save_snap,
                return_seismogram = True,
            )

            pbar.progress(100)
            status.success("✅ Simulación finalizada.")
            st.balloons()

            st.subheader("📊 Resultados")
            c1, c2 = st.columns(2)

            with c1:
                snaps = sorted([f for f in os.listdir('outputs')
                                if f.startswith('snapshot') and f.endswith('.png')])
                if snaps:
                    sel = st.selectbox("Ver snapshot:", snaps)
                    if sel:
                        st.image(f"outputs/{sel}", use_container_width=True)

            with c2:
                if os.path.exists('outputs/seismogram.png'):
                    st.image('outputs/seismogram.png', use_container_width=True)

        except ValueError as e:
            status.error(f"❌ {e}")
        except Exception as e:
            status.error(f"❌ Error inesperado: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PESTAÑA 2 — Experimento de comparación Q
# ═══════════════════════════════════════════════════════════════════════════════
with tab_exp:
    st.subheader("Comparación de atenuación para distintas combinaciones (Qp, Qs)")
    st.markdown(
        "Ejecuta cuatro simulaciones con distintos factores de calidad y genera\n"
        "las figuras de comparación del anteproyecto (Figuras 3 y 4)."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        t_snap_ms = st.number_input("Tiempo del snapshot (ms)", 5.0, 100.0, 37.9, 1.0)
        disp_ms   = st.number_input("Duración sismograma (ms)",  5.0,  50.0, 10.0, 1.0)
    with col_b:
        st.markdown("**Combinaciones (Qp, Qs) a simular:**")
        q_pairs = [
            (st.number_input("Qp₁", 2, 500,  6, 1, key='qp1'),
             st.number_input("Qs₁", 2, 500,  3, 1, key='qs1')),
            (st.number_input("Qp₂", 2, 500, 10, 1, key='qp2'),
             st.number_input("Qs₂", 2, 500,  5, 1, key='qs2')),
            (st.number_input("Qp₃", 2, 500, 20, 1, key='qp3'),
             st.number_input("Qs₃", 2, 500, 10, 1, key='qs3')),
            (st.number_input("Qp₄", 2, 500, 50, 1, key='qp4'),
             st.number_input("Qs₄", 2, 500, 25, 1, key='qs4')),
        ]

    run_exp_btn = st.button("▶️ Ejecutar Experimento Q", use_container_width=True)

    if run_exp_btn:
        snap_step   = int(t_snap_ms * 1e-3 / 2e-4)
        n_disp      = int(disp_ms   * 1e-3 / 2e-4)
        results_q   = []
        prog_exp    = st.progress(0)
        st_exp      = st.empty()

        for idx, (qp_i, qs_i) in enumerate(q_pairs):
            st_exp.info(f"Simulando Qp={qp_i}, Qs={qs_i}…")
            _, seismo_i, snap_i = run_simulation(
                Qp_value      = qp_i,
                snapshot_step = snap_step,
                save_snapshots = False,
                return_seismogram = True,
            )
            results_q.append({'Qp': qp_i, 'Qs': qs_i,
                               'trace': seismo_i, 'snapshot': snap_i})
            prog_exp.progress((idx + 1) * 25)

        vis.plot_experiment_seismogram(results_q, dt=2e-4,
                                       n_display_steps=n_disp,
                                       output_path='outputs/figura4_sismograma.png')
        vis.plot_experiment_snapshots(results_q, dx=dx, dz=dz,
                                      t_ms=t_snap_ms,
                                      output_path='outputs/figura3_snapshots.png')

        st_exp.success("✅ Experimento Q completado.")
        c_seis, c_snap = st.columns(2)
        with c_seis:
            st.image('outputs/figura4_sismograma.png', caption="Figura 4 — Sismograma",
                     use_container_width=True)
        with c_snap:
            st.image('outputs/figura3_snapshots.png', caption="Figura 3 — Snapshots",
                     use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PESTAÑA 3 — Validación
# ═══════════════════════════════════════════════════════════════════════════════
with tab_val:
    st.subheader("Validación numérica — Límite elástico")
    st.markdown(
        "Compara el sismograma simulado con Qp → ∞ contra la solución analítica\n"
        "(wavelet Ricker retardado por el tiempo de viaje) y calcula el error L₂."
    )

    from src.config import rec_z, src_z, dz as _dz

    rec_dist = abs(rec_z - src_z) * _dz
    st.info(f"Distancia fuente–receptor: **{rec_dist:.1f} m** "
            f"| Tiempo de arribo esperado: **{rec_dist/vp*1000:.2f} ms**")

    run_val_btn = st.button("▶️ Ejecutar Validación", use_container_width=True)

    if run_val_btn:
        st_val = st.empty()
        st_val.info("Ejecutando simulación elástica (Qp = 1000)…")

        _, seismo_el, snap_el = run_simulation(
            Qp_value      = 1000,
            snapshot_step = int(37.9e-3 / 2e-4),
            save_snapshots = False,
            return_seismogram = True,
        )

        nt_sim = len(seismo_el)
        scale  = np.max(np.abs(seismo_el)) + 1e-30
        analy  = analytical_elastic_seismogram(
            vp=vp, r=rec_dist, f0=f_peak, dt=2e-4,
            nt=nt_sim, scale=scale
        )

        arr_step = max(0, int(rec_dist / vp / 2e-4) - 10)
        err      = elastic_limit_error(seismo_el, analy, arr_step)

        vis.plot_elastic_validation(seismo_el, analy, 2e-4, err,
                                    output_path='outputs/val_elastico.png')

        if snap_el is not None:
            from src.config import src_x as _sx, src_z as _sz
            radii, amps = geometric_spreading_profile(snap_el, _sx, _sz, dx, dz)
            fit_res     = fit_effective_Q(radii, amps, vp, f_peak)
            vis.plot_geometric_spreading(radii, amps, vp, f_peak, fit_res,
                                         output_path='outputs/val_spreading.png')

        st_val.success("✅ Validación completada.")

        m1, m2, m3 = st.columns(3)
        m1.metric("Error L₂ relativo",    f"{err['L2_error']:.4f}")
        m2.metric("Error máximo",          f"{err['max_error']:.4f}")
        m3.metric("Desfase del pico",      f"{err['peak_delay']} pasos")

        c_v1, c_v2 = st.columns(2)
        with c_v1:
            st.image('outputs/val_elastico.png',
                     caption="Numérico vs. Analítico", use_container_width=True)
        with c_v2:
            if os.path.exists('outputs/val_spreading.png'):
                st.image('outputs/val_spreading.png',
                         caption="Decaimiento geométrico 2D", use_container_width=True)

# ── Pie de página ─────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Configuración de malla**\n"
    f"- Dominio: {200}×{200} nodos × {dx} m = **{200*dx:.0f} m × {200*dz:.0f} m**\n"
    f"- dt = 0.20 ms | CFL ≈ 0.49\n"
    f"- Fuente Ricker: {f_peak:.0f} Hz\n\n"
    "_Jorge Sebastián Ortiz — Universidad de Pamplona, 2026_"
)
