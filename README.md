# Modelado de Propagación de Ondas P en Medios Viscoelásticos

Este proyecto implementa un motor numérico de diferencias finitas para simular ondas en medios complejos, utilizando el modelo de **Derivada Fraccionaria de Caputo**.

## Características principales
- **Física:** Modelado viscoelástico mediante cálculo fraccionario.
- **Estabilidad:** Implementación en Mallas Intercaladas (Staggered Grid).
- **Fronteras:** Capas absorbentes PML para evitar reflexiones espurias.
- **Visualización:** Generación de snapshots y frontend interactivo con Streamlit.
- **Configuración:** Parámetros físicos y numéricos centralizados en `src/config.py`.

## Requisitos
- Python 3.12
- Dependencias definidas en `requirements.txt`

## Instalación
```bash
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Uso
### Ejecutar la simulación desde consola
```bash
python main.py
```

### Iniciar el frontend con Streamlit
```bash
streamlit run app.py
```

## Estructura del proyecto
- `app.py`: Interfaz web para controlar y visualizar la simulación.
- `main.py`: Entrada de consola para ejecutar el solver.
- `requirements.txt`: Dependencias del proyecto.
- `src/`
  - `config.py`: Parámetros de malla y propiedades físicas.
  - `physics.py`: Cálculo del término de memoria con la derivada de Caputo.
  - `pml.py`: Perfil de amortiguamiento para la capa absorbente.
  - `solver.py`: Bucle de tiempo y dinámica principal de la simulación.
  - `visualizer.py`: Funciones de creación de gráficos y snapshots.
- `outputs/`: Carpeta donde se guardan imágenes de snapshots generadas por la simulación.

## Flujo del frontend
1. Abre Streamlit con `streamlit run app.py`.
2. Ajusta los parámetros de velocidad, densidad, orden fraccionario y pasos de tiempo.
3. Haz clic en "Iniciar simulación".
4. Observa el progreso y los snapshots que se actualizan en pantalla.

## Notas
- Si activas `Guardar snapshots en outputs/`, las imágenes se guardarán automáticamente en la carpeta `outputs/`.
- El frontend es una forma sencilla de visualizar cómo evoluciona el campo de presión durante la simulación.
