import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import pytz
from datetime import datetime

# === Load model dan komponen ===
model = joblib.load('Linear_SVM_Original_model_Learn Uke Kala.pkl')
vectorizer = joblib.load('tfidf_vectorizer_Learn Uke Kala.pkl')
label_encoder = joblib.load('label_encoder_Learn Uke Kala.pkl')

# === Judul Aplikasi ===
st.title("🎵 Aplikasi Analisis Sentimen – Kala: Learn Ukulele & Tuner")

# === Pilih Mode Input ===
st.header("📌 Pilih Metode Input")
input_mode = st.radio("Pilih salah satu:", ["📝 Input Manual", "📁 Upload File CSV"])

# === Zona waktu WIB ===
wib = pytz.timezone("Asia/Jakarta")
now_wib = datetime.now(wib)

# === Mapping label dan warna ===
label_map = {'positive': 'Positif', 'negative': 'Negatif'}
color_map = {'Positif': 'blue', 'Negatif': 'red'}

# === MODE 1: INPUT MANUAL ===
if input_mode == "📝 Input Manual":
    st.subheader("🧾 Masukkan Satu Review Pengguna")

    name = st.text_input("👤 Nama Pengguna:")
    star_rating = st.selectbox("⭐ Rating Bintang:", [1, 2, 3, 4, 5])
    user_review = st.text_area("💬 Tulis Review Pengguna:")

    review_day = st.date_input("📅 Tanggal:", value=now_wib.date())
    review_time = st.time_input("⏰ Waktu:", value=now_wib.time())

    review_datetime = datetime.combine(review_day, review_time)
    review_datetime_wib = wib.localize(review_datetime)
    review_date_str = review_datetime_wib.strftime("%Y-%m-%d %H:%M")

    if st.button("🚀 Prediksi Sentimen"):
        if user_review.strip() == "":
            st.warning("⚠️ Silakan isi review terlebih dahulu.")
        else:
            vec = vectorizer.transform([user_review])
            pred = model.predict(vec)
            label = label_encoder.inverse_transform(pred)[0]

            result_df = pd.DataFrame([{
                "name": name if name else "(Anonim)",
                "star_rating": star_rating,
                "date": review_date_str,
                "review": user_review,
                "predicted_sentiment": label
            }])

            st.success(f"✅ Sentimen terdeteksi: **{label.upper()}**")
            st.dataframe(result_df, use_container_width=True, height=200)

            csv_manual = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Unduh Hasil sebagai CSV",
                data=csv_manual,
                file_name="hasil_prediksi_manual_learn_uke_kala.csv",
                mime="text/csv"
            )

# === MODE 2: UPLOAD CSV ===
else:
    st.subheader("📤 Unggah File CSV Review")
    uploaded_file = st.file_uploader(
        "Pilih file CSV (harus memiliki kolom: 'name', 'star_rating', 'date', 'review')",
        type=['csv']
    )

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

            required_cols = {'name', 'star_rating', 'date', 'review'}
            if not required_cols.issubset(df.columns):
                st.error(f"❌ File harus memiliki kolom: {', '.join(required_cols)}.")
            else:
                # Prediksi
                X_vec = vectorizer.transform(df['review'].fillna(""))
                y_pred = model.predict(X_vec)
                df['predicted_sentiment'] = label_encoder.inverse_transform(y_pred)

                st.success("✅ Prediksi berhasil!")

                # === Filter Tanggal ===
                min_date = df['date'].min().date()
                max_date = df['date'].max().date()

                st.subheader("📅 Filter Rentang Tanggal")
                start_date = st.date_input("Mulai", min_value=min_date, max_value=max_date, value=min_date)
                end_date = st.date_input("Selesai", min_value=min_date, max_value=max_date, value=max_date)

                filtered_df = df[(df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)]

                # === Filter Sentimen ===
                sentiment_option = st.selectbox("🎯 Filter Sentimen:", ["Semua", "Positif", "Negatif"])
                if sentiment_option == "Positif":
                    filtered_df = filtered_df[filtered_df['predicted_sentiment'] == "positive"]
                elif sentiment_option == "Negatif":
                    filtered_df = filtered_df[filtered_df['predicted_sentiment'] == "negative"]

                # === Tampilkan Tabel ===
                st.dataframe(
                    filtered_df[['name', 'star_rating', 'date', 'review', 'predicted_sentiment']],
                    use_container_width=True,
                    height=400
                )

                # === Bar Chart ===
                st.subheader("📊 Distribusi Sentimen – Diagram Batang: Kala: Learn Ukulele & Tuner")
                sentimen_bahasa = filtered_df['predicted_sentiment'].map(label_map)
                bar_data = sentimen_bahasa.value_counts().reset_index()
                bar_data.columns = ['Sentimen', 'Jumlah']
                colors = [color_map.get(sent, 'gray') for sent in bar_data['Sentimen']]

                fig_bar, ax_bar = plt.subplots()
                bars = ax_bar.bar(bar_data['Sentimen'], bar_data['Jumlah'], color=colors)

                for bar in bars:
                    height = bar.get_height()
                    ax_bar.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f'{int(height)}',
                                ha='center', va='bottom', fontsize=10)

                ax_bar.set_ylabel("Jumlah")
                ax_bar.set_xlabel("Sentimen")
                ax_bar.set_title("Distribusi Sentimen Pengguna – Kala: Learn Ukulele & Tuner")
                st.pyplot(fig_bar)

                # === Pie Chart ===
                st.subheader("🧁 Distribusi Sentimen – Diagram Pai: Kala: Learn Ukulele & Tuner")
                pie_data = sentimen_bahasa.value_counts()
                pie_colors = [color_map.get(sent, 'gray') for sent in pie_data.index]

                def autopct_format(pct, allvals):
                    absolute = int(round(pct / 100. * sum(allvals)))
                    return f"{pct:.1f}%\n({absolute})"

                fig_pie, ax_pie = plt.subplots()
                ax_pie.pie(
                    pie_data,
                    labels=pie_data.index,
                    colors=pie_colors,
                    autopct=lambda pct: autopct_format(pct, pie_data),
                    startangle=90
                )
                ax_pie.axis('equal')
                st.pyplot(fig_pie)

                # === Unduh CSV ===
                csv_result = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Unduh Hasil CSV",
                    data=csv_result,
                    file_name="hasil_prediksi_learn_uke_kala.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat membaca file: {e}")
