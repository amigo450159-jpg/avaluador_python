# Avaluador Multimodal

## Descripción
Aplicación Streamlit para valuar computadores usados combinando datos tabulares y fotos. La UI muestra solo el monto de préstamo al cliente y aplica reglas de negocio (restricciones por CPU, penalización por estado/calidad de imagen) y límites por histórico.

## Características
- Dropdowns estandarizados para marca, RAM, tipo/capacidad de disco, procesador y GPU
- Carga de múltiples fotos con guía visual y barra de pasos
- Penalizaciones por estado y calidad de imagen
- Ajuste del avalúo por mínimos/máximos del histórico (`PRECIOS PCS.csv`)
- Mensajes de restricción cuando no se cumplen requisitos
- Registro interno en consola del servidor (no visible al cliente)

## Estructura
- `app.py`: interfaz y cálculo de valoración
- `PRECIOS PCS.csv`: histórico con columnas `MARCA, DISCO, RAM, PROCESADOR, GRAFICA GAMER, MINIMO, MAXIMO`
- `artifacts/sklearn_multimodal/model.joblib` (opcional): modelo con embeddings de imagen y tabular
- `train_multimodal.py` (opcional): script de entrenamiento
- `requirements.txt`: dependencias

## Uso local
1. Crear entorno y instalar dependencias:
   - `pip install -r requirements.txt`
2. Ejecutar la interfaz:
   - `streamlit run app.py`
3. Colocar `PRECIOS PCS.csv` en la raíz del proyecto.
4. (Opcional) Si cuentas con `artifacts/sklearn_multimodal/model.joblib`, la app lo usará; de lo contrario, funciona con el histórico.

## Entrenamiento opcional
- `python train_multimodal.py --csv data/train.csv --label price --val_size 0.2 --seed 42`
- Guarda `model.joblib` en `artifacts/sklearn_multimodal/`.

## Despliegue (Streamlit Cloud)
1. Subir el repo a GitHub con `app.py`, `requirements.txt` y `PRECIOS PCS.csv`.
2. En Streamlit Cloud → New app → seleccionar el repo y `app.py`.
3. Abrir la URL pública y validar.

## Despliegue (Hugging Face Spaces)
1. Crear un Space tipo Streamlit.
2. Subir archivos (o conectar GitHub): `app.py`, `requirements.txt`, `PRECIOS PCS.csv`.
3. Usar la URL pública para test.

## Notas
- El valor mostrado al cliente es solo el monto de préstamo.
- Los detalles de avalúo y logs se imprimen en consola del servidor para auditoría.
