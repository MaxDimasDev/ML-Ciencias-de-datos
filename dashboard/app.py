import os
import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


st.set_page_config(page_title="Bank Marketing - Logistic Regression", layout="wide")


def get_api_base() -> str:
    # Prefer Streamlit secrets, fallback to env var, then localhost
    api = st.secrets.get("API_BASE_URL", None) if hasattr(st, "secrets") else None
    if not api:
        api = os.getenv("API_BASE_URL", "http://localhost:8000")
    return api.rstrip("/")


API_BASE = get_api_base()


@st.cache_data(ttl=60)
def get_latest_model():
    try:
        r = requests.get(f"{API_BASE}/model/latest", timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=60)
def get_metrics(limit: int = 10):
    try:
        r = requests.get(f"{API_BASE}/metrics", params={"limit": limit}, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def predict(features: dict):
    try:
        r = requests.post(f"{API_BASE}/predict", json={"features": features}, timeout=30)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def retrain(file_bytes: bytes | None):
    files = None
    if file_bytes:
        files = {"labeled_csv": ("new_data.csv", file_bytes, "text/csv")}
    try:
        r = requests.post(f"{API_BASE}/retrain", files=files, timeout=600)
        r.raise_for_status()
        return r.json(), None
    except Exception as e:
        return None, str(e)


def section_header(title: str):
    st.markdown(f"### {title}")


def draw_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión")
    st.pyplot(fig)


def draw_roc(roc_curve_data):
    fpr = roc_curve_data.get("fpr", [])
    tpr = roc_curve_data.get("tpr", [])
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(fpr, tpr, label="ROC")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("Curva ROC")
    ax.legend()
    st.pyplot(fig)


def draw_pr(pr_curve_data):
    precision = pr_curve_data.get("precision", [])
    recall = pr_curve_data.get("recall", [])
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(recall, precision, label="P-R")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curva Precision-Recall")
    ax.legend()
    st.pyplot(fig)


def draw_y_dist(y_dist):
    labels = list(y_dist.keys())
    values = list(y_dist.values())
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(labels, values, color=["#1f77b4", "#ff7f0e"])
    ax.set_title("Distribución de y")
    st.pyplot(fig)


st.title("Predicción de Contratación de Producto Bancario")
st.caption("Modelo de Regresión Logística + API FastAPI + BD PostgreSQL")

tab_pred, tab_metrics, tab_retrain = st.tabs(["Predicción", "Métricas", "Reentrenamiento"])


with tab_pred:
    st.subheader("Ingresar datos para predicción")
    col1, col2, col3 = st.columns(3)

    # Valores por defecto sensatos; el pipeline imputará faltantes si se omiten
    with col1:
        age = st.number_input("age", min_value=18, max_value=100, value=35)
        job = st.selectbox("job", [
            "admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired",
            "self-employed", "services", "student", "technician", "unemployed", "unknown"
        ], index=0)
        marital = st.selectbox("marital", ["single", "married", "divorced", "unknown"], index=1)
        education = st.selectbox("education", [
            "basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
            "professional.course", "university.degree", "unknown"
        ], index=6)
        default = st.selectbox("default", ["yes", "no", "unknown"], index=2)
        balance = st.number_input("balance", value=0, step=100)
        housing = st.selectbox("housing", ["yes", "no", "unknown"], index=1)
    with col2:
        loan = st.selectbox("loan", ["yes", "no", "unknown"], index=1)
        contact = st.selectbox("contact", ["cellular", "telephone"], index=0)
        day = st.number_input("day", min_value=1, max_value=31, value=15)
        month = st.selectbox("month", ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"], index=7)
        day_of_week = st.selectbox("day_of_week", ["mon","tue","wed","thu","fri"], index=2)
        campaign = st.number_input("campaign", min_value=1, max_value=60, value=1)
    with col3:
        pdays = st.number_input("pdays", min_value=-1, max_value=999, value=999)
        previous = st.number_input("previous", min_value=0, max_value=100, value=0)
        poutcome = st.selectbox("poutcome", ["failure","nonexistent","success"], index=1)
        # Campos del dataset 'bank-additional' (opcionales, se ignorarán si no aplican al modelo actual)
        emp_var_rate = st.number_input("emp.var.rate", value=1.1, format="%0.2f")
        cons_price_idx = st.number_input("cons.price.idx", value=93.75, format="%0.2f")
        cons_conf_idx = st.number_input("cons.conf.idx", value=-40.0, format="%0.1f")
        euribor3m = st.number_input("euribor3m", value=4.0, format="%0.3f")
        nr_employed = st.number_input("nr.employed", value=5191.0, format="%0.1f")

    features = {
        "age": int(age),
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": int(balance),
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": int(day),
        "month": month,
        "day_of_week": day_of_week,
        "campaign": int(campaign),
        "pdays": int(pdays),
        "previous": int(previous),
        "poutcome": poutcome,
        "emp.var.rate": float(emp_var_rate),
        "cons.price.idx": float(cons_price_idx),
        "cons.conf.idx": float(cons_conf_idx),
        "euribor3m": float(euribor3m),
        "nr.employed": float(nr_employed),
    }

    if st.button("Predecir", type="primary"):
        with st.spinner("Solicitando a la API..."):
            resp, err = predict(features)
        if err:
            st.error(f"Error en predicción: {err}")
        elif resp:
            st.success("¡Predicción generada!")
            colr1, colr2, colr3, colr4 = st.columns(4)
            colr1.metric("Predicción (y)", str(resp.get("predicted")))
            colr2.metric("Probabilidad", f"{resp.get('probability'):.3f}")
            colr3.metric("Modelo", resp.get("model_version"))
            colr4.metric("Timestamp", str(resp.get("timestamp")))


with tab_metrics:
    st.subheader("Métricas del modelo y gráficas")
    data = get_metrics(limit=10)
    if "error" in data:
        st.warning("No se pudieron obtener métricas de la API. Mostrando UI.")
    else:
        history = data.get("history", [])
        if not history:
            st.info("Aún no hay métricas disponibles.")
        else:
            latest = history[0]
            m = latest.get("metrics", {})

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Accuracy", f"{m.get('accuracy', 0):.3f}")
            c2.metric("Precision", f"{m.get('precision', 0):.3f}")
            c3.metric("Recall", f"{m.get('recall', 0):.3f}")
            c4.metric("F1", f"{m.get('f1', 0):.3f}")
            c5.metric("ROC-AUC", f"{m.get('roc_auc', 0):.3f}")

            st.markdown("---")
            gc1, gc2, gc3, gc4 = st.columns(4)
            with gc1:
                draw_confusion_matrix(m.get("confusion_matrix", [[0,0],[0,0]]))
            with gc2:
                draw_roc(m.get("roc_curve", {}))
            with gc3:
                draw_pr(m.get("pr_curve", {}))
            with gc4:
                draw_y_dist(m.get("y_distribution", {"0":0, "1":0}))

            # Tendencia histórica de métricas
            st.markdown("#### Tendencia histórica por versión")
            hist_df = pd.DataFrame([
                {
                    "version": h.get("version"),
                    "created_at": h.get("created_at"),
                    **{k: h.get("metrics", {}).get(k) for k in ["accuracy","precision","recall","f1","roc_auc"]},
                }
                for h in history
            ])
            if not hist_df.empty:
                st.line_chart(hist_df.set_index("version")[ ["accuracy","precision","recall","f1","roc_auc"] ])


with tab_retrain:
    st.subheader("Reentrenar Modelo")
    st.write("Opcionalmente sube un CSV con datos etiquetados (debe incluir columna 'y').")
    uploaded = st.file_uploader("CSV etiquetado", type=["csv"], accept_multiple_files=False)

    if st.button("Reentrenar", type="primary"):
        file_bytes = uploaded.read() if uploaded is not None else None
        with st.spinner("Reentrenando en la API (puede tardar unos minutos)..."):
            resp, err = retrain(file_bytes)
        if err:
            st.error(f"Error al reentrenar: {err}")
        else:
            st.success("¡Reentrenamiento completo!")
            st.json(resp)
            st.cache_data.clear()
            st.toast("Métricas y modelo actualizados. Refresca las secciones.")

st.sidebar.info("API base: " + API_BASE)
st.sidebar.caption("Configura API_BASE_URL en Secrets o variables de entorno.")