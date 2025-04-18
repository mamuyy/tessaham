import streamlit as st
import yfinance as yf
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns, plotting
import matplotlib.pyplot as plt
from cvxpy import ECOS
import io

st.set_page_config(layout="wide")
st.title("📈 Optimasi Portofolio Saham Indonesia (IDX) 🇮🇩")

# Input dari pengguna
tickers_input = st.text_input("Masukkan ticker saham (pisahkan dengan koma)", "BBCA.JK, BBRI.JK, BMRI.JK, ANTM.JK")
tickers = [ticker.strip() for ticker in tickers_input.split(",") if ticker.strip()]

start = st.date_input("Tanggal mulai", pd.to_datetime("2020-01-01"))
end = st.date_input("Tanggal akhir", pd.to_datetime("2024-12-31"))

if st.button("🔍 Optimasi Max Sharpe"):
    try:
        # Ambil data harga
        data = yf.download(tickers, start=start, end=end)["Close"]
        data = data.dropna(axis=0)  # Buang baris yang mengandung NaN

        # Cek lagi setelah dropna
        if data.empty:
            st.error("❌ Data kosong setelah menghapus NaN. Periksa ticker atau periode waktu.")
        else:
            # Hitung return dan kovarians
            mu = expected_returns.mean_historical_return(data)
            S = risk_models.sample_cov(data)

            # Optimasi
            ef = EfficientFrontier(mu, S, solver=ECOS)
            weights = ef.max_sharpe()
            cleaned_weights = ef.clean_weights()
            df = pd.DataFrame.from_dict(cleaned_weights, orient='index', columns=['Weight'])

            # Tampilkan hasil
            st.subheader("📊 Rekomendasi Alokasi Portofolio:")
            st.dataframe(df)

            # Tombol download CSV
            csv = df.to_csv().encode("utf-8")
            st.download_button("📥 Download Bobot ke Excel", csv, "bobot_portofolio.csv", "text/csv")

            # Plot bobot
            st.subheader("📉 Plot Bobot Saham")
            fig, ax = plt.subplots()
            plotting.plot_weights(cleaned_weights, ax=ax)
            st.pyplot(fig)

            # Efficient Frontier
            st.subheader("🌀 Efficient Frontier")
            fig2, ax2 = plt.subplots()
            ef_plot = EfficientFrontier(mu, S)
            plotting.plot_efficient_frontier(ef_plot, ax=ax2, show_assets=True)
            ret_tangent, std_tangent, _ = ef.portfolio_performance()
            ax2.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")
            ax2.legend()
            st.pyplot(fig2)

            # Ringkasan performa
            st.subheader("📈 Ringkasan Kinerja Portofolio")
            perf = ef.portfolio_performance(verbose=True)
            st.text(f"Expected Annual Return: {perf[0]*100:.2f}%")
            st.text(f"Annual Volatility: {perf[1]*100:.2f}%")
            st.text(f"Sharpe Ratio: {perf[2]:.2f}")

    except Exception as e:
        st.error(f"❌ Terjadi error saat proses optimasi: {e}")
