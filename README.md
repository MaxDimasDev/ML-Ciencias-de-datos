# Sistema ML Completo: Regresión Logística (Bank Marketing)

Este proyecto implementa un sistema completo de ML para predecir si un cliente contratará un producto bancario, usando el dataset Bank Marketing (UCI). Incluye:

- API REST con FastAPI para predicción, métricas y reentrenamiento.
- Modelo de Regresión Logística con `scikit-learn`, pipeline y métricas almacenadas.
- Base de datos PostgreSQL para predicciones, métricas y versiones del modelo.
- Dashboard web en Streamlit para ingresar datos, ver métricas y reentrenar.
 - Dashboard web en Streamlit para ingresar datos, modo Chat asistente, ver métricas y reentrenar.
- Listo para despliegue gratuito: API en Render, BD en Neon, Dashboard en Streamlit Community Cloud.

## Estructura

```
.
├── api/
│   ├── main.py          # Entrypoint FastAPI
│   ├── db.py            # Conexión SQLAlchemy
│   ├── models.py        # Tablas: model_versions, predictions, labeled_examples
│   ├── crud.py          # Operaciones DB
│   ├── schemas.py       # Pydantic schemas
│   └── ml.py            # Entrenamiento, métricas y serialización
├── dashboard/
│   └── app.py           # App Streamlit (UI)
├── requirements.txt
├── render.yaml          # Config para Render (API)
└── README.md
```

## Requerimientos

- Python 3.10+
- Postgres (local o en la nube). Para desarrollo local, por defecto usa SQLite (`local.db`). Para producción, usar `DATABASE_URL` (Postgres).

## Variables de Entorno y Secrets

- Crea un archivo `.env` (no se versiona) a partir de `.env.example` y define al menos:
  - `DATABASE_URL` → cadena de conexión Postgres (por ejemplo, Neon) con `sslmode=require`.
  - `BANK_DATASET_PATH` (opcional) → ruta a un CSV local; si se omite, la API descargará el dataset UCI automáticamente.
  - `AUTO_RETRAIN` (opcional) → `true/false`. Si true, la API reentrena en background con feedback etiquetado.
  - `RETRAIN_MIN_FEEDBACK` (opcional) → entero N; reentrena cada N feedbacks (por defecto 1).
  - `API_BASE_URL` (para el dashboard, si no usas localhost).
- El Dashboard en Streamlit Community Cloud debe usar `Secrets` y definir `API_BASE_URL` ahí.
- Los datasets locales `bank-full.csv` y `bank-additional-full.csv` están ignorados por `.gitignore` para evitar subirlos al repo.

## Correr Localmente

1) Crear entorno e instalar dependencias:

```
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

2) (Opcional) Configurar Postgres local exportando `DATABASE_URL`, ejemplo:

```
set DATABASE_URL=postgresql+psycopg2://USER:PASSWORD@HOST:5432/DBNAME
```

3) Levantar la API (crea tablas y entrena modelo inicial automáticamente):

```
uvicorn api.main:app --reload
```

- Salud: http://localhost:8000/health
- Predicción: POST http://localhost:8000/predict
- Métricas: GET http://localhost:8000/metrics
- Reentrenar: POST http://localhost:8000/retrain (multipart con CSV opcional con columna `y`)

4) Levantar el dashboard (requiere `API_BASE_URL` si la API no está en localhost):

```
set API_BASE_URL=http://localhost:8000
streamlit run dashboard/app.py
```

## Despliegue Gratuito

### 1. Base de Datos: Neon (free Postgres)
- Crear una cuenta en https://neon.tech
- Crear proyecto y base de datos
- Obtener la cadena `postgresql://...`
- Convertir a formato SQLAlchemy: `postgresql+psycopg2://USER:PASSWORD@HOST/DB`
 - Añadir `?sslmode=require` si tu proveedor lo solicita (Neon lo requiere).

### 2. API en Render (Free Web Service)
- Conectar repo de GitHub a Render
- Crear un servicio Web:
  - Build Command: `pip install -r requirements.txt`
  - Start Command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
  - Añadir variable `DATABASE_URL` con la cadena de Neon
  - Alternativamente, Render detecta `render.yaml` en la raíz.

> Nota: La API detecta automáticamente un CSV local (`BANK_DATASET_PATH` o `bank-full.csv`/`bank-additional-full.csv`) si existe en el proyecto. En producción (Render) es habitual no subir datasets; la API descargará el dataset UCI y entrenará al inicio.

### 3. Dashboard en Streamlit Community Cloud (gratis)
- Ir a https://share.streamlit.io/
- Conectar el repo
- Entry point: `dashboard/app.py`
- Configurar en Secrets: `API_BASE_URL: https://<tu-servicio-render>.onrender.com`

## Flujo del Sistema

1) Entrenamiento inicial: al iniciar la API, descarga el dataset de UCI, limpia columnas (descarta `duration`) y entrena una Regresión Logística con pipeline (`OneHotEncoder` + `StandardScaler`). Calcula métricas y almacena una versión `vN` del modelo en la BD (incluye artifact binario y métricas en JSON).

2) Predicción (dashboard → API): el usuario ingresa datos, el dashboard envía JSON a la API, la API carga el modelo más reciente, predice, devuelve probabilidad y guarda el registro en la tabla `predictions` con timestamp y versión.

3) Reentrenamiento: el dashboard permite subir un CSV etiquetado (con columna `y`) y ahora también enviar feedback por ejemplo individual desde el Chat. La API reentrena automáticamente en segundo plano cuando se alcanza el umbral `RETRAIN_MIN_FEEDBACK` o inmediatamente si vale 1.

## Endpoints (resumen)

- `GET /health` → estado del servicio
- `GET /model/latest` → versión y métricas del modelo de producción
- `POST /predict` → body `{ "features": { ... } }` → probabilidad, predicción, timestamp, versión
- `GET /metrics?limit=5` → historial de métricas por versión
- `POST /retrain` → `multipart/form-data` con `labeled_csv` opcional
- `POST /feedback` → body `{ "features": { ... }, "y": 0|1 }` → guarda ejemplo etiquetado y puede disparar reentrenamiento en background

## Dashboard: Predicción, Chat y Métricas

- Accuracy, Precision, Recall, F1, ROC-AUC, (PR-AUC), Matriz de confusión
- Curva ROC, Curva Precision-Recall, Distribución de `y`
- Tendencia histórica por versión (líneas)
 - Pestaña "Chat": escribe en lenguaje natural (p. ej., "Tengo 45 años, saldo 1200, hipoteca sí"), el sistema extrae rasgos básicos y predice. Puedes enviar feedback del resultado real para mejorar el modelo.

## Notas

- La API por defecto permite CORS `*` para facilidad de demo.
- Para reentrenar con datos nuevos reales, el CSV debe tener las mismas columnas del dataset base y columna `y` (yes/no o 1/0). Si viene como yes/no, el procesamiento lo convertirá internamente.
- El pipeline maneja categorías desconocidas y realiza escalado en numéricos.

## Licencia

Uso educativo y demostrativo.