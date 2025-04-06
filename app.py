import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from cvxpy import SCS
from pypfopt import EfficientFrontier, risk_models, expected_returns, plotting

st.set_page_config(page_title="Tessaham IDX", layout="wide")
st.title("📈 Optimasi Portofolio Saham Indonesia (IDX) 🇮🇩")

# Input
tickers = st.text_input(
    "Masukkan ticker saham (pisahkan dengan koma)",
    "BBCA.JK, BBRI.JK, BMRI.JK, ANTM.JK"
).split(',')

start = st.date_input("Tanggal mulai", pd.to_datetime("2020-01-01"))
end = st.date_input("Tanggal akhir", pd.to_datetime("2024-12-31"))

# Tombol Optimasi
if st.button("🔍 Optimasi Max Sharpe"):
    try:
        # Ambil data historis Close Price
        data = yf.download(tickers, start=start, end=end)["Close"]

        # Hitung expected return & covarian matrix
        mu = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)

        # Optimasi
        ef = EfficientFrontier(mu, S, solver="ECOS")
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()

        # Output Alokasi
        st.subheader("📊 Rekomendasi Alokasi Portofolio:")
        st.dataframe(pd.DataFrame.from_dict(cleaned_weights, orient='index', columns=['Weight']))

        # Plot Bobot
        st.subheader("📉 Plot Bobot Saham")
        fig, ax = plt.subplots()
        plotting.plot_weights(cleaned_weights, ax=ax)
        st.pyplot(fig)

        # Plot Efficient Frontier
        st.subheader("🌀 Efficient Frontier")
        fig2, ax2 = plt.subplots()
        ef_plot = EfficientFrontier(mu, S)
        plotting.plot_efficient_frontier(ef_plot, ax=ax2, show_assets=True)
        ret_tangent, std_tangent, _ = ef.portfolio_performance()
        ax2.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")
        ax2.legend()
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"❌ Terjadi error saat proses optimasi: {e}")
