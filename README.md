# Modelado de Propagación de Ondas P en Medios Viscoelásticos

Este proyecto implementa un motor numérico de diferencias finitas para simular ondas en medios complejos, utilizando el modelo de **Derivada Fraccionaria de Caputo**.

## Características principales
- **Física:** Modelado viscoelástico mediante cálculo fraccionario.
- **Estabilidad:** Implementación en Mallas Intercaladas (Staggered Grid).
- **Fronteras:** Capas absorbentes PML para evitar reflexiones espurias.
- **Rendimiento:** Computación vectorizada con NumPy.

## Requisitos
- Python 3.x
- NumPy, Matplotlib, SciPy

## Ejecución
```bash
python -m src.solver
