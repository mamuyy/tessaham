import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import datetime

st.set_page_config(page_title="Dashboard Saham", layout="wide")
st.title("📊 Dashboard Analisis Saham - mamuyy")

# ----------------------------------
# SECTION 1: CHART SAHAM REALTIME
# ----------------------------------
st.header("📈 Grafik Harga Saham Realtime")
ticker = st.text_input("Masukkan kode saham (misal BBCA.JK, TLKM.JK):", "BBCA.JK")
period = st.selectbox("Periode:", ["1mo", "3mo", "6mo", "1y", "2y", "5y"])
interval = st.selectbox("Interval:", ["1d", "1h", "30m"])

if st.button("Tampilkan Grafik"):
    df = yf.download(ticker, period=period, interval=interval)
    if not df.empty:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"]
        ))
        fig.update_layout(title=f"Harga Saham {ticker}", xaxis_title="Tanggal", yaxis_title="Harga")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Data tidak ditemukan untuk kode saham tersebut.")

# ----------------------------------
# SECTION 2: ANALISIS PORTOFOLIO CSV
# ----------------------------------
st.header("📁 Analisis Portofolio dari CSV")
uploaded_file = st.file_uploader("Upload file CSV (kolom: Ticker, Qty, BuyPrice)", type="csv")

if uploaded_file:
    df_porto = pd.read_csv(uploaded_file)
    result = []

    for i, row in df_porto.iterrows():
        ticker = row["Ticker"]
        qty = row["Qty"]
        buy_price = row["BuyPrice"]

        try:
            harga_terkini = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
            nilai_akhir = harga_terkini * qty
            nilai_awal = buy_price * qty
            gain = nilai_akhir - nilai_awal
            persentase = (gain / nilai_awal) * 100

            result.append({
                "Ticker": ticker,
                "Qty": qty,
                "BuyPrice": buy_price,
                "LastPrice": round(harga_terkini, 2),
                "Gain/Loss": round(gain, 2),
                "Return (%)": round(persentase, 2)
            })
        except:
            continue

    df_result = pd.DataFrame(result)
    st.dataframe(df_result)

    total_gain = df_result["Gain/Loss"].sum()
    total_return = (df_result["Gain/Loss"].sum() / df_result["BuyPrice"].sum()) * 100
    st.success(f"📌 Total Gain/Loss: Rp {total_gain:,.0f} | Total Return: {total_return:.2f}%")

# ----------------------------------
# SECTION 3: Sinyal Beli/Jual Sederhana (MA Crossover)
# ----------------------------------
st.header("⚡ Sinyal Beli / Jual Sederhana")

ticker_ma = st.text_input("Kode saham untuk sinyal MA (misal: ASII.JK)", "ASII.JK")
data = yf.download(ticker_ma, period="3mo")

if not data.empty:
    data["MA5"] = data["Close"].rolling(5).mean()
    data["MA20"] = data["Close"].rolling(20).mean()

    latest_ma5 = data["MA5"].iloc[-1]
    latest_ma20 = data["MA20"].iloc[-1]

if data[["MA5", "MA20"]].dropna().empty:
    st.warning("Data terlalu sedikit untuk menghitung MA5 dan MA20.")
else:
    st.line_chart(data[["Close", "MA5", "MA20"]])
else:
    st.line_chart(data[["Close", "MA5", "MA20"]])


    if latest_ma5 > latest_ma20:
        st.success("✅ Sinyal: BELI (MA5 > MA20)")
    else:
        st.error("❌ Sinyal: JUAL (MA5 < MA20)")
