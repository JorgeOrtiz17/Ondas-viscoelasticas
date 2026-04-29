import streamlit as st
from src.config import dx, dz, rho, vp, alpha, nt
from src.solver import run_simulation
from src.visualizer import create_snapshot_figure

st.set_page_config(page_title="Simulación de Ondas", layout="wide")
st.title("Simulación de Propagación de Ondas")

with st.sidebar:
    st.header("Parámetros de la simulación")
    vp_value = st.slider("Velocidad P (m/s)", 1000, 6000, int(vp), step=100)
    rho_value = st.slider("Densidad (kg/m³)", 1000, 3000, int(rho), step=50)
    alpha_value = st.slider("Orden α", 0.1, 0.9, float(alpha), step=0.05)
    nt_value = st.number_input("Pasos de tiempo", min_value=100, max_value=5000, value=int(nt), step=100)
    snapshot_interval = st.number_input("Intervalo de snapshot", min_value=10, max_value=nt_value, value=50, step=10)
    save_snapshots = st.checkbox("Guardar snapshots en outputs/", value=True)
    run_button = st.button("Iniciar simulación")

status_text = st.empty()
progress_bar = st.progress(0)
plot_area = st.empty()
log_area = st.empty()


def update_ui(field, t):
    progress = int(100 * (t + 1) / nt_value)
    progress_bar.progress(progress)
    status_text.markdown(f"**Paso de tiempo:** {t} / {nt_value}")
    fig = create_snapshot_figure(field, t)
    plot_area.pyplot(fig)
    log_area.text(f"Actualizado snapshot en t={t}")


if run_button:
    status_text.info("Iniciando simulación... Esto puede tardar unos minutos.")
    run_simulation(
        vp_value=vp_value,
        rho_value=rho_value,
        alpha_value=alpha_value,
        nt_value=nt_value,
        snapshot_interval=snapshot_interval,
        callback=update_ui,
        save_snapshots=save_snapshots,
    )
    progress_bar.progress(100)
    status_text.success("Simulación finalizada.")
    st.balloons()
    st.markdown("### Resultados finales")
    st.write("Los snapshots también se guardan en la carpeta `outputs/` si está habilitado.")
