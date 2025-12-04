import math
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm
import plotly.express as px
import plotly.graph_objects as go

CONTRACT_SIZE = 100


# -------- Black Scholes Basics --------

def year_fraction(expiry_date: dt.date, valuation_date: dt.date = None) -> float:
    if valuation_date is None:
        valuation_date = dt.date.today()
    return max((expiry_date - valuation_date).days, 0) / 365.0


def bs_d1_d2(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return np.nan, np.nan
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def bs_price(S, K, T, r, sigma, option_type="C"):
    d1, d2 = bs_d1_d2(S, K, T, r, sigma)
    if np.isnan(d1):
        return np.nan
    if option_type.upper() == "C":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S, K, T, r, sigma, option_type="C"):
    d1, _ = bs_d1_d2(S, K, T, r, sigma)
    if np.isnan(d1):
        return np.nan
    if option_type.upper() == "C":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1.0


def bs_gamma(S, K, T, r, sigma):
    d1, _ = bs_d1_d2(S, K, T, r, sigma)
    if np.isnan(d1):
        return np.nan
    return norm.pdf(d1) / (S * sigma * math.sqrt(T))


def bs_vega(S, K, T, r, sigma):
    d1, _ = bs_d1_d2(S, K, T, r, sigma)
    if np.isnan(d1):
        return np.nan
    return S * norm.pdf(d1) * math.sqrt(T)


def bs_theta(S, K, T, r, sigma, option_type="C"):
    d1, d2 = bs_d1_d2(S, K, T, r, sigma)
    if np.isnan(d1):
        return np.nan
    first = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
    if option_type.upper() == "C":
        second = -r * K * math.exp(-r * T) * norm.cdf(d2)
        return first + second
    second = r * K * math.exp(-r * T) * norm.cdf(-d2)
    return first + second


# -------- Beispielportfolio --------

def load_example_portfolio():
    today = dt.date.today()
    return pd.DataFrame(
        {
            "underlying": ["SPX", "SPX", "SPX", "NDX", "NDX"],
            "option_type": ["C", "P", "C", "P", "C"],
            "strike": [5000, 4800, 5200, 17500, 18500],
            "expiry": [
                today + dt.timedelta(days=30),
                today + dt.timedelta(days=45),
                today + dt.timedelta(days=60),
                today + dt.timedelta(days=35),
                today + dt.timedelta(days=70),
            ],
            "quantity": [10, -15, 8, 5, -12],
            "side": ["long", "short", "long", "long", "short"],
            "spot": [5100, 5100, 5100, 18000, 18000],
            "iv": [0.18, 0.20, 0.22, 0.25, 0.23],
            "r": [0.04, 0.04, 0.04, 0.04, 0.04],
        }
    )


def compute_greeks_for_row(row, valuation_date=None):
    if valuation_date is None:
        valuation_date = dt.date.today()

    S = float(row["spot"])
    K = float(row["strike"])
    expiry = row["expiry"]
    if isinstance(expiry, str):
        expiry = dt.datetime.strptime(expiry, "%Y-%m-%d").date()
    T = year_fraction(expiry, valuation_date)
    r = float(row["r"])
    sigma = float(row["iv"])
    option_type = row["option_type"]

    price = bs_price(S, K, T, r, sigma, option_type)
    delta = bs_delta(S, K, T, r, sigma, option_type)
    gamma = bs_gamma(S, K, T, r, sigma)
    vega = bs_vega(S, K, T, r, sigma)
    theta = bs_theta(S, K, T, r, sigma, option_type)

    return price, delta, gamma, vega, theta, T


def apply_sign_and_size(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sign = np.where(df["side"].str.lower() == "short", -1.0, 1.0)
    df["effective_qty"] = df["quantity"].astype(float) * sign

    df["position_value"] = df["price"] * df["effective_qty"] * CONTRACT_SIZE
    df["delta_pos"] = df["delta"] * df["effective_qty"] * CONTRACT_SIZE
    df["gamma_pos"] = df["gamma"] * df["effective_qty"] * CONTRACT_SIZE
    df["vega_pos"] = df["vega"] * df["effective_qty"] * CONTRACT_SIZE
    df["theta_pos"] = df["theta"] * df["effective_qty"] * CONTRACT_SIZE
    return df


def aggregate_portfolio(df: pd.DataFrame):
    return {
        "portfolio_value": df["position_value"].sum(),
        "delta": df["delta_pos"].sum(),
        "gamma": df["gamma_pos"].sum(),
        "vega": df["vega_pos"].sum(),
        "theta": df["theta_pos"].sum(),
    }


# -------- Szenario Engine --------

def run_scenario(
    df: pd.DataFrame,
    spot_shock: float = 0.0,
    iv_shock: float = 0.0,
    days_forward: int = 0,
    valuation_date=None,
):
    if valuation_date is None:
        valuation_date = dt.date.today()

    df_scen = df.copy()

    df_scen["spot"] = df_scen["spot"] * (1.0 + spot_shock)
    df_scen["iv"] = np.maximum(df_scen["iv"] + iv_shock, 0.0001)

    if days_forward != 0:
        valuation_date_scen = valuation_date + dt.timedelta(days=days_forward)
    else:
        valuation_date_scen = valuation_date

    results = df_scen.apply(
        lambda row: compute_greeks_for_row(row, valuation_date=valuation_date_scen),
        axis=1,
        result_type="expand",
    )
    results.columns = ["price", "delta", "gamma", "vega", "theta", "T"]
    df_scen[["price", "delta", "gamma", "vega", "theta", "T"]] = results

    df_scen = apply_sign_and_size(df_scen)
    agg = aggregate_portfolio(df_scen)
    return df_scen, agg


def build_scenario_set():
    return {
        "Base": {"spot_shock": 0.0, "iv_shock": 0.0, "days_forward": 0},
        "S -2%": {"spot_shock": -0.02, "iv_shock": 0.0, "days_forward": 0},
        "S -5%": {"spot_shock": -0.05, "iv_shock": 0.0, "days_forward": 0},
        "S +2%": {"spot_shock": 0.02, "iv_shock": 0.0, "days_forward": 0},
        "IV +5 Punkte": {"spot_shock": 0.0, "iv_shock": 0.05, "days_forward": 0},
        "IV +10 Punkte": {"spot_shock": 0.0, "iv_shock": 0.10, "days_forward": 0},
        "S -3%, IV +10": {"spot_shock": -0.03, "iv_shock": 0.10, "days_forward": 0},
        "+1 Tag": {"spot_shock": 0.0, "iv_shock": 0.0, "days_forward": 1},
        "+5 Tage": {"spot_shock": 0.0, "iv_shock": 0.0, "days_forward": 5},
    }


def compute_greeks_at_risk(base_agg, scenario_aggs: dict):
    rows = []
    for name, agg in scenario_aggs.items():
        if name == "Base":
            continue
        rows.append(
            {
                "scenario": name,
                "delta_change": agg["delta"] - base_agg["delta"],
                "gamma_change": agg["gamma"] - base_agg["gamma"],
                "vega_change": agg["vega"] - base_agg["vega"],
                "theta_change": agg["theta"] - base_agg["theta"],
                "pnl": agg["portfolio_value"] - base_agg["portfolio_value"],
            }
        )
    return pd.DataFrame(rows)


# -------- Visualisierungen --------

def gamma_heatmap(df: pd.DataFrame):
    df = df.copy()
    df["days_to_expiry"] = df["T"] * 365.0
    bins = [0, 30, 60, 90, 180, 365, 10000]
    labels = ["0-30", "31-60", "61-90", "91-180", "181-365", ">365"]
    df["maturity_bucket"] = pd.cut(df["days_to_expiry"], bins=bins, labels=labels)

    pivot = df.pivot_table(
        index="maturity_bucket",
        columns="strike",
        values="gamma_pos",
        aggfunc="sum",
        fill_value=0.0,
    )
    if pivot.empty:
        return None

    fig = px.imshow(
        pivot.values,
        x=pivot.columns.astype(str),
        y=pivot.index.astype(str),
        aspect="auto",
        labels={"x": "Strike", "y": "Maturity Bucket", "color": "Gamma Exposure"},
    )
    fig.update_layout(
        title="Gamma Exposure Heatmap",
        xaxis_title="Strike",
        yaxis_title="Maturity Bucket",
    )
    return fig


def scenario_barplot(df_scen: pd.DataFrame):
    fig = go.Figure()
    fig.add_bar(x=df_scen["scenario"], y=df_scen["pnl"], name="PnL")
    fig.update_layout(
        title="Scenario PnL",
        xaxis_title="Scenario",
        yaxis_title="PnL Change",
    )
    return fig


# -------- Monte Carlo Greeks at Risk Funktionen --------

def portfolio_greeks(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    records = []

    for _, row in portfolio_df.iterrows():
        S0 = row["spot"]
        K = row["strike"]
        T = year_fraction(row["expiry"])
        sigma0 = row["iv"]
        r = row["r"]
        qty = row["quantity"]
        side = 1 if row["side"] == "long" else -1
        opt_type = row["option_type"]

        price = bs_price(S0, K, T, r, sigma0, opt_type)
        delta = bs_delta(S0, K, T, r, sigma0, opt_type)
        gamma = bs_gamma(S0, K, T, r, sigma0)  # nur 5 Argumente
        vega = bs_vega(S0, K, T, r, sigma0)
        theta = bs_theta(S0, K, T, r, sigma0, opt_type)

        records.append(
            {
                "underlying": row["underlying"],
                "S0": S0,
                "sigma0": sigma0,
                "delta": delta * qty * side * CONTRACT_SIZE,
                "gamma": gamma * qty * side * CONTRACT_SIZE,
                "vega": vega * qty * side * CONTRACT_SIZE,
                "theta": theta * qty * side * CONTRACT_SIZE,
                "position_value": price * qty * side * CONTRACT_SIZE,
            }
        )

    return pd.DataFrame(records)


def simulate_greeks_at_risk(
    greeks_df: pd.DataFrame,
    horizon_days=1,
    n_sims=10000,
    spot_vol_scale=1.0,
    iv_vol_scale=0.3,
    corr_matrix=None,
    random_seed=42,
):
    np.random.seed(random_seed)

    underlyings = greeks_df["underlying"].unique()
    n_u = len(underlyings)

    S0_vec = np.zeros(n_u)
    sigma0_vec = np.zeros(n_u)

    for i, u in enumerate(underlyings):
        sub = greeks_df[greeks_df["underlying"] == u]
        S0_vec[i] = sub["S0"].iloc[0]
        sigma0_vec[i] = sub["sigma0"].mean()

    dt_years = horizon_days / 252.0
    sigma_S = sigma0_vec * spot_vol_scale
    sigma_S_dt = sigma_S * np.sqrt(dt_years)

    sigma_iv = sigma0_vec * iv_vol_scale * np.sqrt(dt_years)

    if corr_matrix is None:
        corr_matrix = np.eye(n_u)

    cov_S = np.outer(sigma_S_dt, sigma_S_dt) * corr_matrix

    dS_rel = np.random.multivariate_normal(
        mean=np.zeros(n_u),
        cov=cov_S,
        size=n_sims,
    )
    dS = dS_rel * S0_vec

    dSigma = np.random.normal(
        loc=0.0,
        scale=sigma_iv,
        size=(n_sims, n_u),
    )

    dT = -dt_years

    pnl = np.zeros(n_sims)

    for i, u in enumerate(underlyings):
        sub = greeks_df[greeks_df["underlying"] == u]
        delta_u = sub["delta"].values
        gamma_u = sub["gamma"].values
        vega_u = sub["vega"].values
        theta_u = sub["theta"].values

        dS_u = dS[:, i].reshape(-1, 1)
        dSigma_u = dSigma[:, i].reshape(-1, 1)

        dP = (
            delta_u * dS_u
            + 0.5 * gamma_u * (dS_u ** 2)
            + vega_u * dSigma_u
            + theta_u * dT
        )

        pnl += dP.sum(axis=1)

    return pnl


def var_es_from_pnl(pnl, alpha=0.95):
    pnl_sorted = np.sort(pnl)
    n = len(pnl_sorted)
    var_index = int((1 - alpha) * n)

    var_level = -pnl_sorted[var_index]
    es_level = -pnl_sorted[:var_index].mean()

    return var_level, es_level


# -------- Streamlit App --------

def main():
    st.set_page_config(page_title="Options Greeks at Risk Dashboard", layout="wide")

    st.title("Options Greeks at Risk Dashboard by Elias Benhachmi")
    st.caption("Delta / Gamma / Vega / Theta unter Spot und Vol Schocks.")

    st.sidebar.header("Portfolio Input")
    upload = st.sidebar.file_uploader("Portfolio CSV hochladen", type=["csv"])

    if upload is not None:
        df_port = pd.read_csv(upload)
        if "expiry" in df_port.columns:
            try:
                df_port["expiry"] = pd.to_datetime(df_port["expiry"]).dt.date
            except Exception:
                pass
    else:
        st.sidebar.info("Kein File hochgeladen, verwende Beispielportfolio.")
        df_port = load_example_portfolio()

    st.subheader("Portfolio Input")
    st.dataframe(df_port)

    valuation_date = dt.date.today()
    results = df_port.apply(
        lambda row: compute_greeks_for_row(row, valuation_date=valuation_date),
        axis=1,
        result_type="expand",
    )
    results.columns = ["price", "delta", "gamma", "vega", "theta", "T"]
    df_base = df_port.copy()
    df_base[["price", "delta", "gamma", "vega", "theta", "T"]] = results
    df_base = apply_sign_and_size(df_base)
    base_agg = aggregate_portfolio(df_base)

    st.subheader("Basis Greeks aggregiert")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Portfolio Value", f"{base_agg['portfolio_value']:.2f}")
    c2.metric("Delta", f"{base_agg['delta']:.2f}")
    c3.metric("Gamma", f"{base_agg['gamma']:.4f}")
    c4.metric("Vega", f"{base_agg['vega']:.2f}")
    c5.metric("Theta / Jahr", f"{base_agg['theta']:.2f}")

    with st.expander("Details je Option (Basis)"):
        st.dataframe(
            df_base[
                [
                    "underlying",
                    "option_type",
                    "strike",
                    "expiry",
                    "quantity",
                    "side",
                    "spot",
                    "iv",
                    "price",
                    "delta",
                    "gamma",
                    "vega",
                    "theta",
                    "position_value",
                    "delta_pos",
                    "gamma_pos",
                    "vega_pos",
                    "theta_pos",
                ]
            ]
        )

    st.subheader("Szenario Analyse und Greeks at Risk")
    scenario_dict = build_scenario_set()
    scenario_aggs = {"Base": base_agg}
    scenario_details = {"Base": df_base}

    for name, params in scenario_dict.items():
        if name == "Base":
            continue
        df_s, agg_s = run_scenario(
            df_port,
            spot_shock=params["spot_shock"],
            iv_shock=params["iv_shock"],
            days_forward=params["days_forward"],
            valuation_date=valuation_date,
        )
        scenario_aggs[name] = agg_s
        scenario_details[name] = df_s

    df_gar = compute_greeks_at_risk(base_agg, scenario_aggs)
    st.write("Greeks at Risk relativ zur Basis")
    st.dataframe(df_gar)

    fig_scen = scenario_barplot(df_gar)
    st.plotly_chart(fig_scen, use_container_width=True)

    st.subheader("Gamma Exposure Heatmap (Basis)")
    fig_gamma = gamma_heatmap(df_base)
    if fig_gamma is not None:
        st.plotly_chart(fig_gamma, use_container_width=True)
    else:
        st.info("Nicht genug Daten für Gamma Heatmap.")

    st.subheader("Szenario Details")
    selected = st.selectbox("Szenario wählen", list(scenario_aggs.keys()))
    agg_sel = scenario_aggs[selected]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Portfolio Value", f"{agg_sel['portfolio_value']:.2f}")
    c2.metric("Delta", f"{agg_sel['delta']:.2f}")
    c3.metric("Gamma", f"{agg_sel['gamma']:.4f}")
    c4.metric("Vega", f"{agg_sel['vega']:.2f}")
    c5.metric("Theta / Jahr", f"{agg_sel['theta']:.2f}")

    with st.expander(f"Optionsdetails – {selected}"):
        st.dataframe(
            scenario_details[selected][
                [
                    "underlying",
                    "option_type",
                    "strike",
                    "expiry",
                    "quantity",
                    "side",
                    "spot",
                    "iv",
                    "price",
                    "delta",
                    "gamma",
                    "vega",
                    "theta",
                    "position_value",
                    "delta_pos",
                    "gamma_pos",
                    "vega_pos",
                    "theta_pos",
                ]
            ]
        )

    # Monte Carlo Block
    with st.expander("Monte Carlo Greeks at Risk"):
        horizon_days = st.slider("Risk Horizont (Tage)", 1, 20, 1, key="mc_horizon")
        n_sims = st.selectbox(
            "Anzahl Simulationen", [1000, 5000, 10000], index=2, key="mc_nsims"
        )

        if st.button("Greeks at Risk simulieren", key="mc_button"):
            current_portfolio_df = scenario_details[selected].copy()
            greeks_df = portfolio_greeks(current_portfolio_df)
            pnl = simulate_greeks_at_risk(
                greeks_df,
                horizon_days=horizon_days,
                n_sims=n_sims,
            )

            var95, es95 = var_es_from_pnl(pnl, alpha=0.95)
            var99, es99 = var_es_from_pnl(pnl, alpha=0.99)

            st.write(f"VaR 95%: {var95:,.0f}")
            st.write(f"ES 95%: {es95:,.0f}")
            st.write(f"VaR 99%: {var99:,.0f}")
            st.write(f"ES 99%: {es99:,.0f}")

            st.plotly_chart(
                px.histogram(x=pnl, nbins=50, title="PnL Verteilung Monte Carlo"),
                use_container_width=True,
            )


if __name__ == "__main__":
    main()

# python -m streamlit run greeks_at_risk_dashboard.py

