# Segmentación de Jurisdicciones Argentinas mediante K-Means

**Autor:** Lucas Nahuel Llanes  
**Materia:** Aprendizaje Automático  
**Carrera:** Tecnicatura en Ciencia de Datos e Inteligencia Artificial  
**Institución:** Instituto Superior de Formación Técnica N° 197  
**Año:** 2026

---

## 📌 Descripción

Este repositorio contiene un notebook de Jupyter que aplica el algoritmo de clustering **K-Means** para agrupar las 24 jurisdicciones de Argentina (23 provincias + CABA) según la cantidad de viviendas registradas. El objetivo es identificar conglomerados naturales que permitan clasificar el territorio en categorías de tamaño habitacional.

El flujo de trabajo incluye:
- Carga y exploración del conjunto de datos.
- Normalización de variables con `StandardScaler`.
- Determinación del número óptimo de clusters mediante el **método del codo**.
- Entrenamiento del modelo con `K=3`.
- Visualización e interpretación de los resultados.

---

## ⚙️ Requisitos

- Python 3.10 o superior
- pandas
- scikit-learn
- matplotlib
- numpy

Instalación rápida:

```bash
pip install pandas scikit-learn matplotlib numpy# K-means-A
