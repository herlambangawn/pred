import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Prediksi Penjualan Sepeda Listrik",
    page_icon="🚲",
    layout="wide"
)

st.title(
    "🚲 Prediksi Penjualan Sepeda Listrik Menggunakan Hybrid ARIMA-GRU"
)

st.markdown("""
### Tentang Website

Website ini digunakan untuk:

- Analisis data penjualan sepeda listrik
- Visualisasi time series
- Preprocessing data
- Evaluasi model ARIMA dan Hybrid ARIMA-GRU
- Forecast penjualan 7 hari ke depan

Model yang digunakan:

- ARIMA(4,1,4)
- GRU Residual Learning
- Hybrid ARIMA-GRU
""")

st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload File Excel (.xlsx)",
    type=["xlsx"]
)

if uploaded_file is not None:

    try:

        df = pd.read_excel(uploaded_file)

        if "Date" not in df.columns:
            st.error(
                "Kolom 'Date' tidak ditemukan"
            )
            st.stop()

        if "Unit" not in df.columns:
            st.error(
                "Kolom 'Unit' tidak ditemukan"
            )
            st.stop()

        df["Date"] = pd.to_datetime(
            df["Date"]
        )

        df = df.sort_values(
            "Date"
        )

        st.session_state["data"] = df

        st.success(
            "Data berhasil diupload"
        )

        col1,col2,col3 = st.columns(3)

        col1.metric(
            "Jumlah Data",
            len(df)
        )

        col2.metric(
            "Tanggal Awal",
            df["Date"].min().date()
        )

        col3.metric(
            "Tanggal Akhir",
            df["Date"].max().date()
        )

        st.subheader(
            "Preview Data"
        )

        st.dataframe(
            df.head(),
            use_container_width=True
        )

        st.info(
            """
Gunakan menu di sidebar untuk membuka:

📊 Analisis Data

⚙️ Preprocessing

📈 Evaluasi Model

🔮 Prediksi 7 Hari
"""
        )

    except Exception as e:

        st.error(
            f"Terjadi kesalahan: {e}"
        )