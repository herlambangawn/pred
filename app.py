import streamlit as st
import pandas as pd

# ==================================================
# KONFIGURASI HALAMAN
# ==================================================

st.set_page_config(
    page_title="Prediksi Penjualan Sepeda Listrik",
    page_icon="🚲",
    layout="wide"
)

# ==================================================
# HEADER
# ==================================================

st.title(
    "🚲 Prediksi Penjualan Sepeda Listrik Menggunakan Hybrid ARIMA-GRU"
)

st.markdown("""
### Tentang Website

Website ini digunakan untuk:

- Analisis data penjualan sepeda listrik
- Visualisasi data time series
- Preprocessing data
- Evaluasi model ARIMA dan Hybrid ARIMA-GRU
- Prediksi penjualan 7 hari ke depan

### Model yang Digunakan

- ARIMA (4,1,4)
- GRU (Residual Learning)
- Hybrid ARIMA-GRU

### Format Data

Pastikan file Excel memiliki kolom:

| Date | Unit |
|--------|--------|
| 2021-01-01 | 15 |
| 2021-01-02 | 18 |

Nama kolom harus persis:

- Date
- Unit
""")

st.divider()

# ==================================================
# UPLOAD FILE
# ==================================================

uploaded_file = st.file_uploader(
    "Upload File Excel (.xlsx)",
    type=["xlsx"]
)

# ==================================================
# PROSES FILE
# ==================================================

if uploaded_file is not None:

    try:

        # ------------------------------------------
        # BACA FILE EXCEL
        # ------------------------------------------

        df = pd.read_excel(
            uploaded_file,
            engine="openpyxl"
        )

        # ------------------------------------------
        # VALIDASI KOLOM
        # ------------------------------------------

        required_columns = [
            "Date",
            "Unit"
        ]

        missing_columns = [
            col
            for col in required_columns
            if col not in df.columns
        ]

        if len(missing_columns) > 0:

            st.error(
                f"Kolom tidak ditemukan: {missing_columns}"
            )

            st.stop()

        # ------------------------------------------
        # KONVERSI DATE
        # ------------------------------------------

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        # hapus tanggal invalid

        df = df.dropna(
            subset=["Date"]
        )

        # ------------------------------------------
        # SORTING
        # ------------------------------------------

        df = df.sort_values(
            "Date"
        )

        # ------------------------------------------
        # SIMPAN KE SESSION
        # ------------------------------------------

        st.session_state["data"] = df

        # ------------------------------------------
        # INFO BERHASIL
        # ------------------------------------------

        st.success(
            "✅ Data berhasil diupload"
        )

        # ------------------------------------------
        # METRIK
        # ------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Jumlah Data",
            len(df)
        )

        col2.metric(
            "Tanggal Awal",
            str(
                df["Date"]
                .min()
                .date()
            )
        )

        col3.metric(
            "Tanggal Akhir",
            str(
                df["Date"]
                .max()
                .date()
            )
        )

        col4.metric(
            "Total Penjualan",
            int(
                df["Unit"]
                .sum()
            )
        )

        # ------------------------------------------
        # PREVIEW
        # ------------------------------------------

        st.subheader(
            "Preview Data"
        )

        st.dataframe(
            df.head(20),
            use_container_width=True
        )

        # ------------------------------------------
        # STATISTIK DESKRIPTIF
        # ------------------------------------------

        st.subheader(
            "Statistik Deskriptif"
        )

        st.dataframe(
            df["Unit"]
            .describe()
            .round(2)
        )

        # ------------------------------------------
        # PETUNJUK
        # ------------------------------------------

        st.info(
            """
Gunakan menu sidebar untuk membuka:

📊 Analisis Data

⚙️ Preprocessing

📈 Evaluasi Model

🔮 Prediksi 7 Hari
"""
        )

    except ImportError:

        st.error(
            """
Package openpyxl belum tersedia.

Tambahkan pada requirements.txt:

openpyxl

Kemudian lakukan redeploy aplikasi.
"""
        )

    except Exception as e:

        st.error(
            f"Terjadi kesalahan: {str(e)}"
        )

        st.exception(e)

# ==================================================
# JIKA BELUM ADA FILE
# ==================================================

else:

    st.warning(
        "Silakan upload file Excel terlebih dahulu."
    )
