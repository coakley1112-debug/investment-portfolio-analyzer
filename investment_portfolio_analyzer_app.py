
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta

try:
    import yfinance as yf
except ImportError:
    yf = None

st.set_page_config(page_title="Investment Portfolio Analyzer Pro", layout="wide")

st.title("Investment Portfolio Analyzer Pro")
st.caption("Real tickers, fixed-income/short-term assets, Monte Carlo simulation, portfolio optimization, and efficient frontier")

DEFAULT_CAPITAL = 200000
DEFAULT_RISK_FREE_RATE = 4.0
DEFAULT_LOOKBACK_YEARS = 3
TRADING_DAYS = 252

DEFAULT_TICKERS = pd.DataFrame(
    {
        "Ticker": ["SPY", "QQQ", "AGG", "BIL", "VNQ", "VIG"],
        "Asset Type": ["Equity ETF", "Growth ETF", "Bond ETF", "T-Bill ETF", "REIT ETF", "Dividend ETF"],
        "Allocation %": [30.0, 15.0, 20.0, 15.0, 10.0, 10.0],
    }
)


def validate_inputs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Ticker"] = out["Ticker"].astype(str).str.upper().str.strip()
    out["Asset Type"] = out["Asset Type"].astype(str).str.strip()
    out["Allocation %"] = pd.to_numeric(out["Allocation %"], errors="coerce").fillna(0)
    out = out[out["Ticker"] != ""]
    return out


@st.cache_data(show_spinner=False)
def download_prices(tickers: tuple[str, ...], start_date: date, end_date: date) -> pd.DataFrame:
    if yf is None:
        raise ImportError("yfinance is not installed. Run: pip install yfinance")

    raw = yf.download(list(tickers), start=start_date, end=end_date + timedelta(days=1), auto_adjust=True, progress=False)
    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw.to_frame(name=tickers[0]) if len(tickers) == 1 else raw.copy()

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    return prices.dropna(how="all")


def compute_portfolio(daily_returns: pd.DataFrame, prices: pd.DataFrame, allocations: pd.Series, capital: float, risk_free_rate: float):
    weights = (allocations / 100).reindex(prices.columns).fillna(0)

    annual_returns = daily_returns.mean() * TRADING_DAYS
    annual_volatility = daily_returns.std() * np.sqrt(TRADING_DAYS)

    portfolio_daily_returns = daily_returns.mul(weights, axis=1).sum(axis=1)
    portfolio_annual_return = portfolio_daily_returns.mean() * TRADING_DAYS
    portfolio_annual_vol = portfolio_daily_returns.std() * np.sqrt(TRADING_DAYS)

    sharpe_ratio = np.nan
    if portfolio_annual_vol > 0:
        sharpe_ratio = (portfolio_annual_return - risk_free_rate / 100) / portfolio_annual_vol

    latest_prices = prices.ffill().iloc[-1]
    dollar_allocations = capital * weights
    shares = dollar_allocations / latest_prices

    metrics_df = pd.DataFrame(
        {
            "Ticker": prices.columns,
            "Allocation %": (weights * 100).values,
            "Latest Price": latest_prices.values,
            "Dollar Allocation": dollar_allocations.values,
            "Estimated Shares": shares.values,
            "Annual Return %": (annual_returns * 100).reindex(prices.columns).values,
            "Annual Volatility %": (annual_volatility * 100).reindex(prices.columns).values,
        }
    )

    cumulative_growth = (1 + portfolio_daily_returns).cumprod()

    summary = {
        "portfolio_annual_return": portfolio_annual_return * 100,
        "portfolio_annual_vol": portfolio_annual_vol * 100,
        "sharpe_ratio": sharpe_ratio,
        "portfolio_daily_returns": portfolio_daily_returns,
    }

    return metrics_df, cumulative_growth, summary


def future_value(capital: float, annual_return_pct: float, years: int) -> float:
    return capital * (1 + annual_return_pct / 100) ** years


def monte_carlo_simulation(capital: float, annual_return_pct: float, annual_vol_pct: float, years: int, simulations: int = 3000) -> np.ndarray:
    mu = annual_return_pct / 100
    sigma = annual_vol_pct / 100
    results = []

    for _ in range(simulations):
        value = capital
        for _year in range(years):
            simulated_return = np.random.normal(mu, sigma)
            value *= 1 + simulated_return
        results.append(value)

    return np.array(results)


def random_portfolio_optimizer(daily_returns: pd.DataFrame, risk_free_rate: float, simulations: int = 5000):
    tickers = daily_returns.columns.tolist()
    annual_returns = daily_returns.mean() * TRADING_DAYS
    cov_matrix = daily_returns.cov() * TRADING_DAYS

    rows = []

    for _ in range(simulations):
        weights = np.random.random(len(tickers))
        weights /= weights.sum()

        port_return = np.dot(weights, annual_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = np.nan

        if port_vol > 0:
            sharpe = (port_return - risk_free_rate / 100) / port_vol

        row = {
            "Return %": port_return * 100,
            "Volatility %": port_vol * 100,
            "Sharpe": sharpe,
        }

        for i, ticker in enumerate(tickers):
            row[ticker] = weights[i] * 100

        rows.append(row)

    optimizer_df = pd.DataFrame(rows)
    best_row = optimizer_df.loc[optimizer_df["Sharpe"].idxmax()]
    min_vol_row = optimizer_df.loc[optimizer_df["Volatility %"].idxmin()]

    return optimizer_df, best_row, min_vol_row


def make_current_portfolio_point(portfolio_return: float, portfolio_vol: float, sharpe_ratio: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Volatility %": [portfolio_vol],
            "Return %": [portfolio_return],
            "Sharpe": [sharpe_ratio],
        }
    )


with st.sidebar:
    st.header("Portfolio Setup")
    capital = st.number_input("Starting capital ($)", min_value=1000, value=DEFAULT_CAPITAL, step=1000)
    lookback_years = st.slider("Historical lookback (years)", min_value=1, max_value=10, value=DEFAULT_LOOKBACK_YEARS)
    projection_years = st.slider("Projection years", min_value=1, max_value=30, value=10)
    risk_free_rate = st.number_input("Risk-free rate %", min_value=0.0, max_value=10.0, value=DEFAULT_RISK_FREE_RATE, step=0.1)
    mc_sims = st.slider("Monte Carlo simulations", min_value=500, max_value=10000, value=3000, step=500)
    optimizer_sims = st.slider("Optimizer / frontier portfolios", min_value=1000, max_value=20000, value=5000, step=1000)

    end_date = date.today()
    start_date = end_date - timedelta(days=365 * lookback_years)

st.subheader("Enter Real Tickers and Allocations")
st.markdown(
    "Examples: **SPY** = S&P 500, **QQQ** = Nasdaq growth, **AGG** = bonds, **BIL/SGOV** = short-term T-bills, **VIG** = dividend ETF, **VNQ** = REIT."
)

portfolio_input = st.data_editor(
    DEFAULT_TICKERS,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Ticker": st.column_config.TextColumn("Ticker"),
        "Asset Type": st.column_config.TextColumn("Asset Type"),
        "Allocation %": st.column_config.NumberColumn("Allocation %", format="%.2f"),
    },
)

portfolio_input = validate_inputs(portfolio_input)
allocation_total = portfolio_input["Allocation %"].sum()

c1, c2 = st.columns(2)
c1.metric("Allocation Total", f"{allocation_total:.2f}%")

if abs(allocation_total - 100) > 0.01:
    c2.error("Allocations should add up to 100%.")
    st.stop()
else:
    c2.success("Allocations add up to 100%.")

if portfolio_input.empty:
    st.warning("Enter at least one ticker.")
    st.stop()

try:
    tickers = tuple(portfolio_input["Ticker"].tolist())
    prices = download_prices(tickers, start_date, end_date)
except Exception as exc:
    st.error(f"Could not load price data: {exc}")
    st.stop()

if prices.empty:
    st.error("No historical price data found for the selected tickers.")
    st.stop()

valid_columns = [col for col in prices.columns if col in tickers]
prices = prices[valid_columns].dropna(how="all")
daily_returns = prices.pct_change().dropna()
correlation = daily_returns.corr()

alloc_series = portfolio_input.set_index("Ticker")["Allocation %"]
asset_type_map = portfolio_input.set_index("Ticker")["Asset Type"].to_dict()

metrics_df, cumulative_growth, analysis = compute_portfolio(daily_returns, prices, alloc_series, capital, risk_free_rate)
metrics_df["Asset Type"] = metrics_df["Ticker"].map(asset_type_map)

portfolio_return = analysis["portfolio_annual_return"]
portfolio_vol = analysis["portfolio_annual_vol"]
sharpe_ratio = analysis["sharpe_ratio"]

st.subheader("Portfolio Summary")
s1, s2, s3, s4 = st.columns(4)
s1.metric("Expected Annual Return", f"{portfolio_return:.2f}%")
s2.metric("Annual Volatility", f"{portfolio_vol:.2f}%")
s3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}" if pd.notna(sharpe_ratio) else "N/A")
s4.metric(f"Projected Value ({projection_years} yrs)", f"${future_value(capital, portfolio_return, projection_years):,.0f}")

st.subheader("Holdings and Historical Metrics")
metrics_display = metrics_df.copy()
for col in ["Latest Price", "Dollar Allocation"]:
    metrics_display[col] = metrics_display[col].map(lambda x: f"${x:,.2f}")
metrics_display["Estimated Shares"] = metrics_display["Estimated Shares"].map(lambda x: f"{x:,.3f}")
metrics_display["Annual Return %"] = metrics_display["Annual Return %"].map(lambda x: f"{x:,.2f}%")
metrics_display["Annual Volatility %"] = metrics_display["Annual Volatility %"].map(lambda x: f"{x:,.2f}%")
metrics_display["Allocation %"] = metrics_display["Allocation %"].map(lambda x: f"{x:,.2f}%")

st.dataframe(
    metrics_display[
        [
            "Ticker",
            "Asset Type",
            "Allocation %",
            "Latest Price",
            "Dollar Allocation",
            "Estimated Shares",
            "Annual Return %",
            "Annual Volatility %",
        ]
    ],
    use_container_width=True,
)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Portfolio Growth Over Time")
    st.line_chart(cumulative_growth)
with col2:
    st.subheader("Allocation by Ticker")
    alloc_chart = metrics_df.set_index("Ticker")["Allocation %"]
    st.bar_chart(alloc_chart)

st.subheader("Correlation Matrix")
st.dataframe(correlation.style.format("{:.2f}"), use_container_width=True)

st.subheader("Scenario Analysis")
bear_return = portfolio_return - portfolio_vol
base_return = portfolio_return
bull_return = portfolio_return + portfolio_vol / 2
scenario_df = pd.DataFrame(
    {
        "Scenario": ["Bear Case", "Base Case", "Bull Case"],
        "Annual Return %": [bear_return, base_return, bull_return],
        "Projected Value": [
            future_value(capital, bear_return, projection_years),
            future_value(capital, base_return, projection_years),
            future_value(capital, bull_return, projection_years),
        ],
    }
).set_index("Scenario")

st.dataframe(
    scenario_df.style.format({"Annual Return %": "{:.2f}", "Projected Value": "${:,.0f}"}),
    use_container_width=True,
)
st.bar_chart(scenario_df[["Projected Value"]])

st.subheader("Monte Carlo Simulation")
mc_results = monte_carlo_simulation(capital, portfolio_return, portfolio_vol, projection_years, mc_sims)
mc_series = pd.Series(mc_results, name="Ending Value")

mc1, mc2, mc3 = st.columns(3)
mc1.metric("5th Percentile", f"${np.percentile(mc_results, 5):,.0f}")
mc2.metric("Median Outcome", f"${np.median(mc_results):,.0f}")
mc3.metric("95th Percentile", f"${np.percentile(mc_results, 95):,.0f}")
st.bar_chart(mc_series)

st.subheader("Portfolio Optimizer")
optimizer_df, best_portfolio, min_vol_portfolio = random_portfolio_optimizer(daily_returns, risk_free_rate, optimizer_sims)

opt1, opt2, opt3 = st.columns(3)
opt1.metric("Best Sharpe Return", f"{best_portfolio['Return %']:.2f}%")
opt2.metric("Best Sharpe Volatility", f"{best_portfolio['Volatility %']:.2f}%")
opt3.metric("Best Sharpe Ratio", f"{best_portfolio['Sharpe']:.2f}")

best_weights = {ticker: best_portfolio[ticker] for ticker in prices.columns}
best_df = pd.DataFrame({"Ticker": list(best_weights.keys()), "Suggested Allocation %": list(best_weights.values())})
best_df["Asset Type"] = best_df["Ticker"].map(asset_type_map)
st.dataframe(best_df[["Ticker", "Asset Type", "Suggested Allocation %"]].style.format({"Suggested Allocation %": "{:.2f}"}), use_container_width=True)

st.subheader("Efficient Frontier Chart")
st.markdown(
    "Each dot is a randomly generated portfolio. The higher and more left a point is, the better: higher return with lower risk. The highlighted point is the highest Sharpe ratio portfolio."
)

frontier_df = optimizer_df[["Volatility %", "Return %", "Sharpe"]].copy()
frontier_df["Portfolio Type"] = "Random Portfolio"

best_point = pd.DataFrame(
    {
        "Volatility %": [best_portfolio["Volatility %"]],
        "Return %": [best_portfolio["Return %"]],
        "Sharpe": [best_portfolio["Sharpe"]],
        "Portfolio Type": ["Best Sharpe Portfolio"],
    }
)

min_vol_point = pd.DataFrame(
    {
        "Volatility %": [min_vol_portfolio["Volatility %"]],
        "Return %": [min_vol_portfolio["Return %"]],
        "Sharpe": [min_vol_portfolio["Sharpe"]],
        "Portfolio Type": ["Minimum Volatility Portfolio"],
    }
)

current_point = pd.DataFrame(
    {
        "Volatility %": [portfolio_vol],
        "Return %": [portfolio_return],
        "Sharpe": [sharpe_ratio],
        "Portfolio Type": ["Your Current Portfolio"],
    }
)

frontier_plot_df = pd.concat([frontier_df, best_point, min_vol_point, current_point], ignore_index=True)

st.scatter_chart(
    frontier_plot_df,
    x="Volatility %",
    y="Return %",
    color="Portfolio Type",
    size="Sharpe",
)

st.subheader("Minimum Volatility Portfolio")
st.write("This portfolio tries to reduce risk as much as possible, even if return is lower.")
min_vol_weights = {ticker: min_vol_portfolio[ticker] for ticker in prices.columns}
min_vol_df = pd.DataFrame({"Ticker": list(min_vol_weights.keys()), "Suggested Allocation %": list(min_vol_weights.values())})
min_vol_df["Asset Type"] = min_vol_df["Ticker"].map(asset_type_map)
st.dataframe(min_vol_df[["Ticker", "Asset Type", "Suggested Allocation %"]].style.format({"Suggested Allocation %": "{:.2f}"}), use_container_width=True)

st.subheader("How to talk about this project in an interview")
st.markdown(
    """
- Built a Streamlit investment analyzer using real market tickers, including equities, bond ETFs, REITs, dividend ETFs, and short-term Treasury-bill ETFs.
- Pulled historical data and calculated annualized return, volatility, correlation, and Sharpe ratio.
- Added Monte Carlo simulation to estimate the distribution of possible future portfolio values.
- Built a random portfolio optimizer to search for high-Sharpe allocations across a multi-asset portfolio.
- Added an efficient frontier chart to visualize the risk-return tradeoff across thousands of possible portfolios.
"""
)

st.subheader("Requirements")
st.code("pip install streamlit pandas numpy yfinance")



