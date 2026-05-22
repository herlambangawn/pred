import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from tensorflow.keras.models import load_model
from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)

st.title("📈 Evaluasi Model Hybrid ARIMA-GRU")

if "data" not in st.session_state:
    st.warning("Upload data terlebih dahulu")
    st.stop()

# =========================
# DATA
# =========================

df = st.session_state["data"].copy()

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date")

series = (
    df
    .set_index("Date")["Unit"]
    .asfreq("D")
)

# =========================
# SPLIT
# =========================

split_index = int(len(series)*0.8)

train = series.iloc[:split_index]

test = series.iloc[split_index:]

# =========================
# MODEL
# =========================

gru_model = load_model(
    "models/gru_model.h5",
    compile=False
)

scaler = joblib.load(
    "models/scaler.pkl"
)

WINDOW_SIZE = 15

# =========================
# ARIMA
# =========================

with st.spinner("Menjalankan evaluasi model..."):

    arima_model = ARIMA(
        train,
        order=(4,1,4)
    )

    arima_result = arima_model.fit()

    arima_pred = arima_result.forecast(
        steps=len(test)
    )

    # =====================
    # RESIDUAL TRAIN
    # =====================

    residual = (
        train -
        arima_result.fittedvalues
    )

    residual = residual.dropna()

    # =====================
    # SCALING
    # =====================

    scaled_residual = scaler.transform(
        residual.values.reshape(-1,1)
    )

    # =====================
    # WINDOW
    # =====================

    x_input = scaled_residual[-WINDOW_SIZE:]

    x_input = x_input.reshape(
        1,
        WINDOW_SIZE,
        1
    )

    # =====================
    # FORECAST RESIDUAL
    # =====================

    future_residual = []

    for _ in range(len(test)):

        pred = gru_model.predict(
            x_input,
            verbose=0
        )

        value = pred[0][0]

        future_residual.append(
            value
        )

        x_input = np.append(
            x_input[:,1:,:],
            [[[value]]],
            axis=1
        )

    future_residual = scaler.inverse_transform(
        np.array(
            future_residual
        ).reshape(-1,1)
    ).flatten()

    # =====================
    # HYBRID
    # =====================

    hybrid_pred = (
        arima_pred.values +
        future_residual
    )

# =========================
# METRIK
# =========================

rmse_arima = np.sqrt(
    mean_squared_error(
        test,
        arima_pred
    )
)

mae_arima = mean_absolute_error(
    test,
    arima_pred
)

mape_arima = np.mean(
    np.abs(
        (test-arima_pred)/test
    )
)*100

rmse_hybrid = np.sqrt(
    mean_squared_error(
        test,
        hybrid_pred
    )
)

mae_hybrid = mean_absolute_error(
    test,
    hybrid_pred
)

mape_hybrid = np.mean(
    np.abs(
        (test-hybrid_pred)/test
    )
)*100

# =========================
# TABEL
# =========================

result_df = pd.DataFrame({
    "Model":[
        "ARIMA",
        "Hybrid ARIMA-GRU"
    ],
    "RMSE":[
        rmse_arima,
        rmse_hybrid
    ],
    "MAE":[
        mae_arima,
        mae_hybrid
    ],
    "MAPE":[
        mape_arima,
        mape_hybrid
    ]
})

st.subheader(
    "Perbandingan Kinerja Model"
)

st.dataframe(
    result_df.round(4),
    use_container_width=True
)

# =========================
# CARD
# =========================

col1,col2,col3 = st.columns(3)

col1.metric(
    "RMSE Hybrid",
    round(rmse_hybrid,3)
)

col2.metric(
    "MAE Hybrid",
    round(mae_hybrid,3)
)

col3.metric(
    "MAPE Hybrid (%)",
    round(mape_hybrid,2)
)

# =========================
# PLOT
# =========================

st.subheader(
    "Aktual vs Prediksi"
)

fig, ax = plt.subplots(
    figsize=(12,5)
)

ax.plot(
    test.index,
    test.values,
    label="Aktual"
)

ax.plot(
    test.index,
    arima_pred,
    label="ARIMA"
)

ax.plot(
    test.index,
    hybrid_pred,
    label="Hybrid ARIMA-GRU"
)

ax.legend()

ax.set_xlabel("Tanggal")

ax.set_ylabel("Unit")

st.pyplot(fig)

# =========================
# INTERPRETASI
# =========================

st.subheader(
    "Interpretasi"
)

st.markdown(
f"""
Model Hybrid ARIMA-GRU menghasilkan:

- RMSE = **{rmse_hybrid:.3f}**
- MAE = **{mae_hybrid:.3f}**
- MAPE = **{mape_hybrid:.2f}%**

Dibandingkan model ARIMA, model Hybrid
ARIMA-GRU menunjukkan kemampuan yang lebih baik
dalam mengikuti pola data aktual apabila nilai
RMSE, MAE, dan MAPE lebih kecil.

Nilai MAPE sebesar **{mape_hybrid:.2f}%**
menunjukkan rata-rata kesalahan prediksi
terhadap data aktual.
"""
)