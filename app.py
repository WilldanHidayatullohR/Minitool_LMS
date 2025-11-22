# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from math import pi

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Opsional: reportlab hanya digunakan jika tersedia (untuk PDF lokal)
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# =====================================================
# KONFIGURASI HALAMAN
# =====================================================
st.set_page_config(
    page_title="Mini Tool Analisis Kepuasan",
    layout="wide"
)

st.title("Mini Tool Analisis Kepuasan Berbasis Data")
st.caption("Versi demo Kerja Praktek – dapat digunakan untuk berbagai survei kepuasan (sistem, layanan, aplikasi, layanan, dan lain-lain).")

with st.sidebar:
    st.header("Informasi Aplikasi")
    st.write(
        """
        Mini tool ini dirancang untuk:
        - Mengunggah data survei kepuasan dalam format CSV
        - Melihat statistik deskriptif dan visualisasi
        - Membangun model prediksi sederhana (Linear Regression, Decision Tree, dan Random Forest)
        - Menghasilkan insight otomatis dari data (siap digunakan di laporan)
        """
    )
    st.markdown("---")
    st.write("Pengembang: Wildan Hidayatulloh")
    st.write("Program Studi: Informatika, FTI UNSAP")


# =====================================================
# FUNGSI BANTU
# =====================================================
def hitung_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def klasifikasi_level_mean(mean_value: float) -> str:
    """
    Label sederhana berdasarkan nilai rata-rata skala 1–5.
    """
    if mean_value >= 4.0:
        return "sangat baik"
    elif mean_value >= 3.0:
        return "cukup baik"
    else:
        return "perlu perhatian"


# =====================================================
# 1. UPLOAD DATA
# =====================================================
st.subheader("1. Upload Data Survei")

uploaded_file = st.file_uploader(
    "Upload file CSV hasil survei (skala Likert 1–5 atau angka).",
    type=["csv"]
)

if uploaded_file is not None:
    # Simpan ke session_state supaya tidak hilang saat rerun
    st.session_state["df"] = pd.read_csv(uploaded_file)

if "df" not in st.session_state:
    st.info("Silakan upload file CSV terlebih dahulu untuk melanjutkan.")
    st.stop()

df = st.session_state["df"]

# =====================================================
# 2. PREVIEW DATA & INFO UMUM
# =====================================================
st.subheader("2. Preview Data dan Informasi Umum")

st.write("Preview 5 baris pertama:")
st.dataframe(df.head())

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Jumlah Baris", df.shape[0])
with col2:
    st.metric("Jumlah Kolom", df.shape[1])
with col3:
    st.metric("Total Nilai Kosong", int(df.isna().sum().sum()))

# =====================================================
# 3. KONFIGURASI ANALISIS (PILIH TARGET)
# =====================================================
st.subheader("3. Konfigurasi Analisis")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) == 0:
    st.error("Tidak ditemukan kolom numerik. Pastikan file berisi nilai angka (misalnya skala Likert 1–5).")
    st.stop()

st.write("Kolom numerik yang terdeteksi:")
st.write(", ".join(numeric_cols))

# Coba tebak target kepuasan jika ada nama yang umum
default_target = None
for cand in ["Kepuasan", "Overall_Satisfaction", "Overall Satisfaction", "Avg_Satisfaction"]:
    if cand in numeric_cols:
        default_target = cand
        break

target_col = st.selectbox(
    "Pilih kolom target kepuasan yang akan dianalisis:",
    options=numeric_cols,
    index=numeric_cols.index(default_target) if default_target in numeric_cols else 0
)

feature_cols = [c for c in numeric_cols if c != target_col]

if len(feature_cols) == 0:
    st.error("Tidak ada fitur numerik selain kolom target. Tambahkan kolom fitur lain (misalnya SQ1, SQ2, UX1, dan sebagainya).")
    st.stop()

st.success(
    f"Target yang dipilih: {target_col}. "
    f"Jumlah fitur yang digunakan dalam model: {len(feature_cols)}."
)

# =====================================================
# 4. TABS UTAMA
# =====================================================
tab_data, tab_stat, tab_analisis = st.tabs(
    ["Data & Ringkasan", "Statistik & Visualisasi", "Analisis Lanjutan"]
)

# =====================================================
# TAB 1: DATA & RINGKASAN
# =====================================================
with tab_data:
    st.write("### Ringkasan Statistik Dasar (Kolom Numerik)")
    st.dataframe(df[numeric_cols].describe().T)

    st.write("### Distribusi Nilai Target")
    fig_t, ax_t = plt.subplots(figsize=(6, 3))
    df[target_col].hist(bins=5, ax=ax_t)
    ax_t.set_xlabel(f"Nilai {target_col}")
    ax_t.set_ylabel("Frekuensi")
    ax_t.set_title(f"Distribusi {target_col}")
    plt.tight_layout()
    st.pyplot(fig_t)

# =====================================================
# TAB 2: STATISTIK & VISUALISASI (DENGAN SUB-TABS)
# =====================================================
with tab_stat:
    st.markdown("### Statistik & Visualisasi")

    tab_bar, tab_box, tab_corr, tab_radar = st.tabs(
        ["Rata-rata Fitur", "Sebaran (Boxplot)", "Korelasi", "Radar Aspek"]
    )

    # ------------------------------
    # SUB-TAB: RATA-RATA FITUR
    # ------------------------------
    with tab_bar:
        st.subheader("Rata-rata Skor per Fitur")

        mean_scores = df[feature_cols].mean().sort_values(ascending=False)

        fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
        mean_scores.plot(kind="bar", ax=ax_bar)
        ax_bar.set_ylabel("Rata-rata Skor")
        ax_bar.set_xlabel("Fitur")
        ax_bar.set_title("Rata-rata Skor Fitur")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig_bar)

        st.markdown("**Ringkasan:**")
        top3 = mean_scores.head(3)
        bottom3 = mean_scores.tail(3)

        st.write("Aspek dengan rata-rata skor tertinggi:")
        for idx, val in top3.items():
            st.write(f"- {idx}: rata-rata {val:.2f}")

        st.write("Aspek dengan rata-rata skor terendah:")
        for idx, val in bottom3.items():
            st.write(f"- {idx}: rata-rata {val:.2f}")

        st.caption(
            "Nilai rata-rata tinggi menunjukkan area yang sudah berjalan baik dan perlu dipertahankan. "
            "Sebaliknya, rata-rata rendah dapat menjadi fokus perbaikan."
        )

    # ------------------------------
    # SUB-TAB: BOXPLOT
    # ------------------------------
    with tab_box:
        st.subheader("Sebaran Skor per Fitur (Boxplot)")

        fig_box, ax_box = plt.subplots(figsize=(10, 4))
        df[feature_cols].plot(kind="box", ax=ax_box)
        ax_box.set_ylabel("Skor")
        ax_box.set_title("Sebaran Skor Fitur")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig_box)

        st.caption(
            "Boxplot menunjukkan median, sebaran, dan potensi outlier. "
            "Sebaran yang lebar menandakan persepsi pengguna yang beragam terhadap fitur tersebut."
        )

    # ------------------------------
    # SUB-TAB: KORELASI
    # ------------------------------
    with tab_corr:
        st.subheader("Korelasi Antar Fitur dan Target")

        corr_cols = feature_cols + [target_col]
        corr = df[corr_cols].corr()

        st.write("Matriks Korelasi:")
        st.dataframe(corr.style.background_gradient(cmap="coolwarm"))

        fig_corr, ax_corr = plt.subplots(figsize=(7, 5))
        cax = ax_corr.imshow(corr, interpolation="nearest", cmap="coolwarm")
        ax_corr.set_xticks(range(len(corr_cols)))
        ax_corr.set_yticks(range(len(corr_cols)))
        ax_corr.set_xticklabels(corr_cols, rotation=45, ha="right")
        ax_corr.set_yticklabels(corr_cols)
        fig_corr.colorbar(cax)
        ax_corr.set_title("Heatmap Korelasi")
        plt.tight_layout()
        st.pyplot(fig_corr)

        st.caption(
            "Korelasi mendekati +1 atau -1 menandakan hubungan kuat antara dua variabel. "
            "Fitur dengan korelasi tinggi terhadap target berpotensi kuat memengaruhi kepuasan."
        )

    # ------------------------------
    # SUB-TAB: RADAR ASPEK
    # ------------------------------
    with tab_radar:
        st.subheader("Profil Aspek Kepuasan (Radar Chart)")

        # Mapping contoh grup aspek → prefix kolom (opsional, hanya jika ada)
        aspect_groups = {
            "System Quality": ["SQ1", "SQ2", "SQ3", "SQ4"],
            "Information Quality": ["IQ1", "IQ2", "IQ3", "IQ4"],
            "Service Quality": ["SVQ1", "SVQ2", "SVQ3", "SVQ4"],
            "User Experience": ["UX1", "UX2", "UX3", "UX4"],
            "Expected Satisfaction": ["ES1", "ES2", "ES3", "ES4"],
        }

        labels = []
        means = []

        for asp, cols in aspect_groups.items():
            valid_cols = [c for c in cols if c in df.columns]
            if len(valid_cols) == 0:
                continue
            labels.append(asp)
            means.append(df[valid_cols].mean().mean())

        if len(labels) < 3:
            st.info(
                "Radar chart membutuhkan minimal tiga aspek. "
                "Jika ingin memanfaatkan radar, gunakan penamaan kolom seperti SQ1–SQ4, IQ1–IQ4, dan seterusnya."
            )
        else:
            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
            values = np.array(means)
            # Tutup loop radar
            values = np.concatenate((values, [values[0]]))
            angles = np.concatenate((angles, [angles[0]]))

            fig_rad, ax_rad = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            ax_rad.plot(angles, values, "o-", linewidth=2)
            ax_rad.fill(angles, values, alpha=0.25)
            ax_rad.set_thetagrids(angles[:-1] * 180 / np.pi, labels)
            ax_rad.set_ylim(1, 5)  # asumsi skala 1–5
            ax_rad.set_title("Profil Rata-rata per Aspek", pad=20)
            st.pyplot(fig_rad)

            st.caption(
                "Radar chart memberikan gambaran cepat mengenai aspek mana yang relatif kuat "
                "dan mana yang lebih lemah berdasarkan nilai rata-rata."
            )

# =====================================================
# TAB 3: ANALISIS LANJUTAN (MODEL + INSIGHT)
# =====================================================
with tab_analisis:
    st.markdown("### Analisis Lanjutan")

    tab_model, tab_insight = st.tabs(["Model Prediksi", "Insight & Laporan"])

    # -------------------------------------------------
    # SUB-TAB: MODEL PREDIKSI
    # -------------------------------------------------
    with tab_model:
        st.subheader("Model Prediksi Kepuasan")

        X = df[feature_cols]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Model 1: Linear Regression
        lin_reg = LinearRegression()
        lin_reg.fit(X_train, y_train)
        y_pred_lin = lin_reg.predict(X_test)
        r2_lin = r2_score(y_test, y_pred_lin)
        rmse_lin = hitung_rmse(y_test, y_pred_lin)

        # Model 2: Decision Tree
        dt_model = DecisionTreeRegressor(
            max_depth=None,
            random_state=42
        )
        dt_model.fit(X_train, y_train)
        y_pred_dt = dt_model.predict(X_test)
        r2_dt = r2_score(y_test, y_pred_dt)
        rmse_dt = hitung_rmse(y_test, y_pred_dt)

        # Model 3: Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        r2_rf = r2_score(y_test, y_pred_rf)
        rmse_rf = hitung_rmse(y_test, y_pred_rf)

        # Tabel perbandingan model
        perf_df = pd.DataFrame({
            "Model": ["Linear Regression", "Decision Tree", "Random Forest"],
            "R2": [r2_lin, r2_dt, r2_rf],
            "RMSE": [rmse_lin, rmse_dt, rmse_rf],
        })

        st.write("Tabel Perbandingan Kinerja Model:")
        st.dataframe(perf_df)

        fig_p, ax_p = plt.subplots(figsize=(6, 3))
        ax_p.bar(perf_df["Model"], perf_df["R2"])
        ax_p.set_ylabel("R²")
        ax_p.set_title("Perbandingan R² Antar Model")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        st.pyplot(fig_p)

        st.caption(
            "Model dengan nilai R² paling tinggi dan RMSE paling rendah dianggap memiliki performa terbaik. "
            "Random Forest dan Decision Tree biasanya lebih baik untuk pola non-linear, sementara Linear Regression "
            "digunakan sebagai baseline."
        )

        # Feature importance berdasarkan Random Forest
        st.write("Pentingnya Fitur menurut Random Forest:")

        importances = rf_model.feature_importances_
        fi_series = pd.Series(importances, index=feature_cols).sort_values(ascending=True)

        fig_fi, ax_fi = plt.subplots(figsize=(8, 5))
        fi_series.plot(kind="barh", ax=ax_fi)
        ax_fi.set_xlabel("Importance")
        ax_fi.set_title("Pentingnya Fitur (Random Forest)")
        plt.tight_layout()
        st.pyplot(fig_fi)

        st.write("Lima fitur paling berpengaruh berdasarkan Random Forest:")
        for feat, val in fi_series.sort_values(ascending=False).head(5).items():
            st.write(f"- {feat}: importance {val:.3f}")

    # -------------------------------------------------
    # SUB-TAB: INSIGHT & LAPORAN
    # -------------------------------------------------
    with tab_insight:
        st.subheader("Insight Otomatis dari Data")

        # 1) Rata-rata dan label kualitas
        mean_scores = df[feature_cols].mean()
        top3 = mean_scores.sort_values(ascending=False).head(3)
        bottom3 = mean_scores.sort_values(ascending=True).head(3)

        st.write("Aspek dengan skor rata-rata tertinggi:")
        top_df = pd.DataFrame({
            "Fitur": top3.index,
            "Rata-rata": top3.values,
            "Kategori": [klasifikasi_level_mean(v) for v in top3.values]
        })
        st.dataframe(top_df, hide_index=True)

        st.write("Aspek dengan skor rata-rata terendah:")
        bottom_df = pd.DataFrame({
            "Fitur": bottom3.index,
            "Rata-rata": bottom3.values,
            "Kategori": [klasifikasi_level_mean(v) for v in bottom3.values]
        })
        st.dataframe(bottom_df, hide_index=True)

        st.markdown("---")

        # 2) Korelasi fitur dengan target
        st.write("Fitur dengan korelasi tertinggi terhadap target:")

        corr_with_target = df[feature_cols + [target_col]].corr()[target_col].drop(target_col)
        corr_sorted = corr_with_target.sort_values(ascending=False)
        top_corr = corr_sorted.head(5)

        corr_df = pd.DataFrame({
            "Fitur": top_corr.index,
            "Korelasi_dengan_Target": top_corr.values
        })
        st.dataframe(corr_df, hide_index=True)

        st.caption(
            "Fitur dengan korelasi positif tinggi cenderung memiliki hubungan searah dengan kepuasan. "
            "Semakin besar nilainya, semakin kuat pengaruh potensial fitur tersebut terhadap target."
        )

        st.markdown("---")

        # 3) Prioritas perbaikan (mean rendah + korelasi tinggi)
        st.write("Prioritas Perbaikan (Skor Rendah tetapi Pengaruh Tinggi):")

        if (mean_scores.max() - mean_scores.min()) > 0:
            mean_norm = (mean_scores - mean_scores.min()) / (mean_scores.max() - mean_scores.min())
        else:
            mean_norm = mean_scores - mean_scores  # semua nol jika datanya konstan

        corr_abs = corr_with_target.abs()
        priority_score = (1 - mean_norm) * corr_abs
        priority_rank = priority_score.sort_values(ascending=False).head(5)

        if len(priority_rank) == 0:
            st.info("Belum dapat menghitung prioritas perbaikan. Cek kembali data dan kolom target.")
        else:
            priority_df = pd.DataFrame({
                "Fitur": priority_rank.index,
                "Rata-rata": [mean_scores[f] for f in priority_rank.index],
                "Korelasi_dengan_Target": [corr_with_target[f] for f in priority_rank.index],
                "Skor_Prioritas": priority_rank.values
            })
            st.dataframe(priority_df, hide_index=True)

            st.caption(
                "Fitur di atas memiliki kombinasi nilai rata-rata relatif rendah dan korelasi tinggi "
                "terhadap kepuasan, sehingga layak menjadi kandidat prioritas perbaikan."
            )

        st.markdown("---")

        # 4) Evaluasi singkat model (refit ringan untuk insight naratif)
        st.write("Evaluasi Singkat Model (Random Forest) untuk Ringkasan:")

        X = df[feature_cols]
        y = df[target_col]

        X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        rf_i = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
        rf_i.fit(X_train_i, y_train_i)
        y_pred_rf_i = rf_i.predict(X_test_i)
        r2_rf_i = r2_score(y_test_i, y_pred_rf_i)

        st.write(f"Nilai R² Random Forest (refit cepat): {r2_rf_i:.3f}")

        if r2_rf_i >= 0.75:
            st.write(
                "Interpretasi: Model memiliki kualitas sangat baik (R² ≥ 0.75). "
                "Variasi kepuasan cukup baik dijelaskan oleh fitur-fitur yang digunakan."
            )
        elif r2_rf_i >= 0.5:
            st.write(
                "Interpretasi: Model memiliki kualitas sedang (0.50 ≤ R² < 0.75). "
                "Masih ada faktor lain di luar data yang mungkin berpengaruh terhadap kepuasan."
            )
        else:
            st.write(
                "Interpretasi: Model memiliki kualitas rendah (R² < 0.50). "
                "Hasil model perlu ditafsirkan dengan hati-hati, dan disarankan menambah variabel lain."
            )

        st.markdown("---")

        # 5) Ringkasan naratif (siap copas)
        st.subheader("Ringkasan Naratif (Siap Digunakan di Laporan)")

        best_feat = top_corr.index[0] if len(top_corr) > 0 else None
        worst_feat = bottom3.index[0] if len(bottom3) > 0 else None

        summary_text = ""
        summary_text += f"- Secara umum, tingkat kepuasan yang diukur melalui variabel **{target_col}** dipengaruhi oleh beberapa fitur utama.\n"

        if best_feat:
            summary_text += (
                f"- Fitur dengan pengaruh paling kuat terhadap kepuasan adalah **{best_feat}** "
                f"dengan nilai korelasi {corr_with_target[best_feat]:.2f} terhadap target.\n"
            )
        if worst_feat:
            summary_text += (
                f"- Di sisi lain, **{worst_feat}** memiliki skor rata-rata terendah "
                f"({mean_scores[worst_feat]:.2f}), sehingga layak menjadi fokus perbaikan.\n"
            )

        summary_text += (
            "- Berdasarkan kombinasi nilai rata-rata dan korelasi, beberapa fitur dapat diidentifikasi "
            "sebagai prioritas utama perbaikan karena memiliki skor relatif rendah namun pengaruh yang signifikan "
            "terhadap kepuasan.\n"
        )
        summary_text += (
            f"- Hasil pemodelan menggunakan Random Forest menunjukkan nilai R² sekitar {r2_rf_i:.2f}, "
            "sehingga model dapat dijadikan dasar analisis awal, namun tetap perlu dikombinasikan "
            "dengan pertimbangan konteks dan kebijakan di lapangan."
        )

        st.write(summary_text)

        st.markdown("---")

        # 6) Export PDF (jika reportlab tersedia)
        st.subheader("Export Laporan ke PDF")

        if not REPORTLAB_AVAILABLE:
            st.info(
                "Fitur export PDF hanya tersedia jika aplikasi dijalankan secara lokal "
                "dan pustaka `reportlab` terinstal.\n\n"
                "Untuk menjalankan secara lokal:\n"
                "1. Instal reportlab: `pip install reportlab`\n"
                "2. Jalankan: `streamlit run app.py`"
            )
        else:
            st.caption(
                "Klik tombol di bawah ini untuk mengunduh ringkasan laporan singkat dalam format PDF."
            )

            pdf_buffer = BytesIO()
            c = canvas.Canvas(pdf_buffer, pagesize=A4)
            width, height = A4

            textobject = c.beginText()
            textobject.setTextOrigin(40, height - 50)
            textobject.setFont("Helvetica", 11)

            judul = "Laporan Ringkas Analisis Kepuasan"
            textobject.textLine(judul)
            textobject.textLine("")
            textobject.textLine(f"Target kepuasan : {target_col}")
            textobject.textLine(f"Jumlah responden: {df.shape[0]}")
            textobject.textLine(f"Jumlah fitur    : {len(feature_cols)}")
            textobject.textLine(f"R² Random Forest: {r2_rf_i:.2f}")
            textobject.textLine("")
            textobject.textLine("Ringkasan Insight:")
            textobject.textLine("")

            for line in summary_text.split("\n"):
                while len(line) > 110:
                    textobject.textLine(line[:110])
                    line = line[110:]
                textobject.textLine(line)

            c.drawText(textobject)
            c.showPage()
            c.save()
            pdf_buffer.seek(0)

            st.download_button(
                label="Download Laporan PDF",
                data=pdf_buffer,
                file_name="laporan_kepuasan.pdf",
                mime="application/pdf"
            )
