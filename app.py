# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas



from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ===== KONFIGURASI HALAMAN =====
st.set_page_config(
    page_title="Mini Tool Analisis Kepuasan",
    layout="wide"
)

# ===== HEADER & SIDEBAR =====
st.title("Mini Tool Analisis Kepuasan Berbasis Data")
st.caption("Versi demo Kerja Praktek – dapat digunakan untuk berbagai survei kepuasan (sistem, layanan, aplikasi, dan lain-lain).")

with st.sidebar:
    st.header("Informasi Aplikasi")
    st.write(
        """
        Mini tool ini dirancang untuk:
        - Mengunggah data survei kepuasan dalam format CSV
        - Melihat statistik deskriptif dan visualisasi
        - Membangun model prediksi sederhana (Linear Regression dan Random Forest)
        - Menghasilkan insight otomatis dari data
        """
    )
    st.markdown("---")
    st.write("Pengembang: Wildan Hidayatulloh")
    st.write("Program Studi: Informatika, FTI UNSAP")

# ===== FUNGSI BANTU =====

def hitung_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def klasifikasi_level_mean(mean_value):
    """
    Mengembalikan label sederhana berdasarkan nilai rata-rata Likert 1–5.
    """
    if mean_value >= 4.0:
        return "sangat baik"
    elif mean_value >= 3.0:
        return "cukup baik"
    else:
        return "perlu perhatian"

# ===== 1. UPLOAD DATA =====
st.subheader("1. Upload Data Survei")

uploaded_file = st.file_uploader("Upload file CSV hasil survei (skala Likert 1–5 atau angka).", type=["csv"])

if uploaded_file is not None:
    # Simpan ke session_state agar tidak hilang saat rerun
    if "df" not in st.session_state:
        st.session_state["df"] = pd.read_csv(uploaded_file)
    else:
        # Jika user upload file baru, update df
        new_df = pd.read_csv(uploaded_file)
        st.session_state["df"] = new_df

if "df" not in st.session_state:
    st.info("Silakan upload file CSV terlebih dahulu untuk melanjutkan.")
    st.stop()

df = st.session_state["df"]

# ===== 2. PREVIEW DATA & INFO UMUM =====
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

# ===== 3. PEMILIHAN KOLUMN TARGET =====
st.subheader("3. Konfigurasi Analisis")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) == 0:
    st.error("Tidak ditemukan kolom numerik. Pastikan file berisi nilai angka (misalnya skala Likert 1–5).")
    st.stop()

st.write("Kolom numerik yang terdeteksi:")
st.write(", ".join(numeric_cols))

# Tebak target secara otomatis jika memungkinkan
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

st.success(f"Target yang dipilih: {target_col}. Jumlah fitur yang digunakan dalam model: {len(feature_cols)}.")


# ==== 4. STATISTIK & VISUALISASI ====
st.markdown("### 4. Statistik & Visualisasi")

# Tabs Visualisasi (5 Tab)
tab_bar, tab_box, tab_corr, tab_scatter, tab_radar = st.tabs(
    ["Rata-rata Fitur", "Sebaran (Boxplot)", "Korelasi", "Scatter Plot", "Radar Aspek"]
)

# -------------------------------------------------
# TAB 1: BAR CHART RATA-RATA PER FITUR + INSIGHT
# -------------------------------------------------
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

    st.write("Aspek dengan rata-rata skor **tertinggi**:")
    for idx, val in top3.items():
        st.write(f"- **{idx}** (rata-rata: {val:.2f})")

    st.write("Aspek dengan rata-rata skor **terendah**:")
    for idx, val in bottom3.items():
        st.write(f"- **{idx}** (rata-rata: {val:.2f})")

    st.caption(
        "Nilai rata-rata tinggi menunjukkan area yang sudah berjalan baik. "
        "Sebaliknya, rata-rata rendah dapat menjadi fokus perbaikan."
    )

# -------------------------------------------------
# TAB 2: BOXPLOT – SEBARAN NILAI PER FITUR
# -------------------------------------------------
with tab_box:
    st.subheader("Sebaran Skor per Fitur (Boxplot)")

    fig_box, ax_box = plt.subplots(figsize=(10, 4))
    df[feature_cols].plot(kind='box', ax=ax_box)
    ax_box.set_ylabel("Skor")
    ax_box.set_title("Sebaran Skor Fitur")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig_box)

    st.caption(
        "Boxplot menunjukkan median, sebaran, dan potensi outlier. "
        "Sebaran luas = persepsi pengguna lebih beragam."
    )

# -------------------------------------------------
# TAB 3: HEATMAP KORELASI
# -------------------------------------------------
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
        "Korelasi tinggi (+ / -) menandakan hubungan kuat antara fitur dan target. "
        "Fitur dengan korelasi tinggi berpotensi memengaruhi kepuasan."
    )

# -------------------------------------------------
# TAB 4: SCATTER PLOT FITUR vs TARGET
# -------------------------------------------------
with tab_scatter:
    st.subheader("Scatter Plot Fitur vs Target")

    selected_feature = st.selectbox(
        "Pilih fitur untuk dibandingkan dengan target:",
        options=feature_cols,
        index=0
    )

    fig_scatter, ax_scatter = plt.subplots(figsize=(6, 4))
    ax_scatter.scatter(df[selected_feature], df[target_col], alpha=0.7)
    ax_scatter.set_xlabel(selected_feature)
    ax_scatter.set_ylabel(target_col)
    ax_scatter.set_title(f"{selected_feature} vs {target_col}")
    plt.tight_layout()
    st.pyplot(fig_scatter)

    st.caption(
        "Scatter plot membantu melihat tren: "
        "apakah nilai fitur naik → kepuasan ikut naik?"
    )

# -------------------------------------------------
# TAB 5: RADAR CHART PER ASPEK
# -------------------------------------------------
with tab_radar:
    st.subheader("Profil Aspek Kepuasan (Radar Chart)")

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
        st.info("Radar chart butuh minimal 3 aspek. Pastikan nama kolom sesuai format SQ1–ES4.")
    else:
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        values = np.array(means)
        values = np.concatenate((values, [values[0]]))
        angles = np.concatenate((angles, [angles[0]]))

        fig_rad, ax_rad = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax_rad.plot(angles, values, "o-", linewidth=2)
        ax_rad.fill(angles, values, alpha=0.25)
        ax_rad.set_thetagrids(angles[:-1] * 180 / np.pi, labels)
        ax_rad.set_ylim(1, 5)
        ax_rad.set_title("Profil Rata-rata per Aspek", pad=20)
        st.pyplot(fig_rad)

        st.caption(
            "Radar chart menunjukkan kekuatan & kelemahan tiap aspek. "
            "Aspek dengan nilai lebih tinggi → kinerja lebih baik."
        )

# ==========================================================
# ============= TAB 4: INSIGHT OTOMATIS (UPGRADED) =========
# ==========================================================
with tab_insight:
    st.write("### Ringkasan Insight Otomatis")

    # --- 1. Statistika dasar: rata-rata ---
    mean_scores = df[feature_cols].mean()
    top3 = mean_scores.sort_values(ascending=False).head(3)
    bottom3 = mean_scores.sort_values(ascending=True).head(3)

    st.subheader("1) Gambaran Umum Aspek Kepuasan")

    st.write("**Aspek dengan skor rata-rata tertinggi:**")
    for feat, val in top3.items():
        level = klasifikasi_level_mean(val)
        st.write(f"- **{feat}** → rata-rata {val:.2f} ({level})")

    st.write("**Aspek dengan skor rata-rata terendah:**")
    for feat, val in bottom3.items():
        level = klasifikasi_level_mean(val)
        st.write(f"- **{feat}** → rata-rata {val:.2f} ({level})")

    st.markdown("---")

    # --- 2. Korelasi dengan target: fitur yang paling memengaruhi ---
    st.subheader("2) Fitur yang Paling Berkorelasi dengan Kepuasan")

    corr_with_target = df[feature_cols + [target_col]].corr()[target_col].drop(target_col)
    corr_sorted = corr_with_target.sort_values(ascending=False)

    top_corr = corr_sorted.head(5)
    st.write("**Top 5 fitur dengan korelasi positif tertinggi terhadap target:**")
    for feat, val in top_corr.items():
        st.write(f"- **{feat}** → korelasi {val:.2f}")

    st.caption(
        "Semakin tinggi nilai korelasi positif, semakin besar kecenderungan "
        "bahwa kenaikan nilai fitur diikuti kenaikan nilai kepuasan."
    )

    st.markdown("---")

    # --- 3. Analisis gabungan: skor rendah + korelasi tinggi = prioritas ---
    st.subheader("3) Prioritas Perbaikan (Skor Rendah tetapi Pengaruh Tinggi)")

    # Normalisasi skor rata-rata ke 0–1 dan korelasi ke |r|
    mean_norm = (mean_scores - mean_scores.min()) / (mean_scores.max() - mean_scores.min() + 1e-9)
    corr_abs = corr_with_target.abs()

    # Kita definisikan "prioritas" = skor relatif rendah tapi korelasi absolut tinggi
    priority_score = (1 - mean_norm) * corr_abs

    priority_rank = priority_score.sort_values(ascending=False).head(5)

    if len(priority_rank) == 0:
        st.info("Belum dapat menghitung prioritas perbaikan (cek data dan kolom target).")
    else:
        st.write(
            "Fitur di bawah ini memiliki kombinasi **skor rata-rata relatif rendah** "
            "namun **pengaruh besar terhadap kepuasan**, sehingga layak jadi prioritas perbaikan:"
        )
        for feat, val in priority_rank.items():
            st.write(
                f"- **{feat}** → skor rata-rata {mean_scores[feat]:.2f}, "
                f"korelasi {corr_with_target[feat]:.2f}"
            )

    st.markdown("---")

    # --- 4. Analisis model prediksi (quick re-fit) ---
    st.subheader("4) Evaluasi Singkat Model Prediksi")

    X = df[feature_cols]
    y = df[target_col]

    # Split ulang cepat (data biasanya kecil)
    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Baseline: Linear Regression
    lin_reg_i = LinearRegression()
    lin_reg_i.fit(X_train_i, y_train_i)
    y_pred_lin_i = lin_reg_i.predict(X_test_i)
    r2_lin_i = r2_score(y_test_i, y_pred_lin_i)

    # Model non-linear: Random Forest
    rf_i = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    rf_i.fit(X_train_i, y_train_i)
    y_pred_rf_i = rf_i.predict(X_test_i)
    r2_rf_i = r2_score(y_test_i, y_pred_rf_i)

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.write("**Linear Regression (baseline):**")
        st.write(f"- R²: {r2_lin_i:.3f}")
    with col_m2:
        st.write("**Random Forest:**")
        st.write(f"- R²: {r2_rf_i:.3f}")

    # Komentar otomatis soal kualitas model
    st.write("**Interpretasi kualitas model:**")
    if r2_rf_i >= 0.75:
        st.write(
            "- Model Random Forest memiliki *kualitas sangat baik* (R² ≥ 0.75). "
            "Artinya, variasi kepuasan cukup baik dijelaskan oleh fitur-fitur yang digunakan."
        )
    elif r2_rf_i >= 0.5:
        st.write(
            "- Model Random Forest memiliki *kualitas sedang* (0.50 ≤ R² < 0.75). "
            "Artinya, masih ada faktor lain di luar data yang mungkin berpengaruh terhadap kepuasan."
        )
    else:
        st.write(
            "- Model Random Forest memiliki *kualitas rendah* (R² < 0.50). "
            "Hasil model perlu ditafsirkan dengan hati-hati, dan disarankan menambah variabel lain."
        )

    st.markdown("---")

    # --- 5. Narasi akhir sebagai summary (bisa dicopas ke laporan) ---
    st.subheader("5) Ringkasan Naratif (Siap Copas ke Laporan)")

    best_feat = top_corr.index[0] if len(top_corr) > 0 else None
    worst_feat = bottom3.index[0] if len(bottom3) > 0 else None

    summary_text = ""

    summary_text += f"- Secara umum, tingkat kepuasan yang diukur melalui **{target_col}** dipengaruhi oleh beberapa fitur utama.\n"
    if best_feat:
        summary_text += (
            f"- Fitur dengan pengaruh paling kuat terhadap kepuasan adalah **{best_feat}**, "
            f"dengan korelasi {corr_with_target[best_feat]:.2f}. "
        )
    if worst_feat:
        summary_text += (
            f"- Meskipun demikian, **{worst_feat}** tercatat memiliki skor rata-rata terendah "
            f"({mean_scores[worst_feat]:.2f}), sehingga layak menjadi fokus perbaikan.\n"
        )
    summary_text += (
        "- Berdasarkan analisis gabungan antara skor rata-rata dan korelasi, beberapa fitur "
        "dapat diidentifikasi sebagai prioritas utama perbaikan karena memiliki skor relatif rendah "
        "namun pengaruh yang signifikan terhadap kepuasan.\n"
    )
    summary_text += (
        "- Hasil pemodelan menggunakan Random Forest menunjukkan bahwa nilai R² berada pada kisaran "
        f"{r2_rf_i:.2f}, sehingga model dapat digunakan sebagai dasar analisis awal, "
        "namun tetap perlu dikombinasikan dengan pertimbangan kebijakan dan konteks lapangan."
    )

    st.write(summary_text)
    
    # -------------------------------------------------
    # 6) Export Laporan ke PDF
    # -------------------------------------------------
    st.subheader("6) Ekspor Laporan ke PDF")

    st.caption(
        "Tombol di bawah ini akan menghasilkan ringkasan laporan singkat dalam bentuk PDF "
        "berdasarkan insight otomatis di atas."
    )

    # Siapkan buffer untuk file PDF
    pdf_buffer = BytesIO()

    # Buat kanvas PDF
    c = canvas.Canvas(pdf_buffer, pagesize=A4)
    width, height = A4

    # Set posisi awal teks
    textobject = c.beginText()
    textobject.setTextOrigin(40, height - 50)
    textobject.setFont("Helvetica", 11)

    # Judul laporan
    judul = "Laporan Ringkas Analisis Kepuasan"
    textobject.textLine(judul)
    textobject.textLine("")  # baris kosong

    # Info target
    textobject.textLine(f"Target kepuasan     : {target_col}")
    textobject.textLine(f"Jumlah responden    : {df.shape[0]}")
    textobject.textLine(f"Jumlah fitur        : {len(feature_cols)}")
    textobject.textLine(f"R² Random Forest    : {r2_rf_i:.2f}")
    textobject.textLine("")
    textobject.textLine("Ringkasan Insight:")
    textobject.textLine("")

    # Pecah summary_text jadi beberapa baris
    for line in summary_text.split("\n"):
        # Pecah lagi kalau terlalu panjang (supaya tidak keluar margin)
        while len(line) > 110:
            textobject.textLine(line[:110])
            line = line[110:]
        textobject.textLine(line)

    c.drawText(textobject)
    c.showPage()
    c.save()

    # Kembalikan pointer ke awal buffer
    pdf_buffer.seek(0)

    st.download_button(
        label="📄 Download Laporan PDF",
        data=pdf_buffer,
        file_name="laporan_kepuasan.pdf",
        mime="application/pdf"
    )
