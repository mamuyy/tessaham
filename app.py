import streamlit as st
import yfinance as yf
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns, plotting
import matplotlib.pyplot as plt

st.title("📈 Optimasi Portofolio Saham Indonesia (IDX) 🇮🇩")

tickers = st.text_input("Masukkan ticker saham (pisahkan dengan koma)", "BBCA.JK, BBRI.JK, BMRI.JK, ANTM.JK").split(',')

start = st.date_input("Tanggal mulai", pd.to_datetime("2020-01-01"))
end = st.date_input("Tanggal akhir", pd.to_datetime("2024-12-31"))

if st.button("Optimasi Max Sharpe"):
    data = yf.download(tickers, start=start, end=end)["Close"]
    mu = expected_returns.mean_historical_return(data)
    S = risk_models.sample_cov(data)
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe(solver="SCS")
    cleaned_weights = ef.clean_weights()
    st.subheader("📊 Rekomendasi Alokasi Portofolio:")
    st.dataframe(pd.DataFrame.from_dict(cleaned_weights, orient='index', columns=['Weight']))
    st.subheader("📉 Plot Bobot Saham")
    fig, ax = plt.subplots()
    plotting.plot_weights(cleaned_weights, ax=ax)
    st.pyplot(fig)
    st.subheader("🌀 Efficient Frontier")
    fig2, ax2 = plt.subplots()
    ef_plot = EfficientFrontier(mu, S)
    plotting.plot_efficient_frontier(ef_plot, ax=ax2, show_assets=True)
    ret_tangent, std_tangent, _ = ef.portfolio_performance()
    ax2.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")
    ax2.legend()
    st.pyplot(fig2)
