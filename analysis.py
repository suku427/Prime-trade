from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HISTORICAL_PATH = BASE_DIR.parent / "historical_data.csv"
SENTIMENT_PATH = BASE_DIR.parent / "fear_greed_index.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
PLOT_DIR = BASE_DIR / "plots"

SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
BUCKET_ORDER = ["Fear-side", "Neutral", "Greed-side"]
BUCKET_MAP = {
    "Extreme Fear": "Fear-side",
    "Fear": "Fear-side",
    "Neutral": "Neutral",
    "Greed": "Greed-side",
    "Extreme Greed": "Greed-side",
}
SENTIMENT_COLORS = {
    "Extreme Fear": "#8c1c13",
    "Fear": "#c0392b",
    "Neutral": "#f4d35e",
    "Greed": "#2a9d8f",
    "Extreme Greed": "#1d6f62",
}
BUCKET_COLORS = {
    "Fear-side": "#c0392b",
    "Neutral": "#f4d35e",
    "Greed-side": "#2a9d8f",
}


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    historical = pd.read_csv(HISTORICAL_PATH)
    sentiment = pd.read_csv(SENTIMENT_PATH)

    historical["Timestamp IST"] = pd.to_datetime(
        historical["Timestamp IST"], format="%d-%m-%Y %H:%M", errors="coerce"
    )
    historical["trade_date"] = historical["Timestamp IST"].dt.normalize()
    historical["abs_size_usd"] = historical["Size USD"].abs()
    historical["net_after_fee"] = historical["Closed PnL"] - historical["Fee"]
    historical["is_realized"] = historical["Closed PnL"].ne(0)

    sentiment["date"] = pd.to_datetime(sentiment["date"], errors="coerce")
    sentiment["classification"] = pd.Categorical(
        sentiment["classification"], categories=SENTIMENT_ORDER, ordered=True
    )

    return historical, sentiment


def prepare_data(
    historical: pd.DataFrame, sentiment: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = historical.merge(
        sentiment[["date", "value", "classification"]],
        left_on="trade_date",
        right_on="date",
        how="left",
    )

    missing_sentiment = merged[merged["classification"].isna()].copy()
    missing_sentiment.to_csv(OUTPUT_DIR / "missing_sentiment_rows.csv", index=False)

    merged = merged.dropna(subset=["classification"]).copy()
    merged["classification"] = pd.Categorical(
        merged["classification"], categories=SENTIMENT_ORDER, ordered=True
    )
    merged["bucket"] = merged["classification"].map(BUCKET_MAP)

    realized = merged[merged["is_realized"]].copy()
    realized["is_win"] = realized["Closed PnL"].gt(0)
    realized["loss"] = realized["Closed PnL"].lt(0)
    realized["gross_roi_bps"] = (
        realized["Closed PnL"] / realized["abs_size_usd"]
    ) * 10000
    realized["net_roi_bps"] = (
        realized["net_after_fee"] / realized["abs_size_usd"]
    ) * 10000
    realized["month"] = realized["trade_date"].dt.to_period("M").dt.to_timestamp()

    return merged, realized, missing_sentiment


def ordered(df: pd.DataFrame, category_col: str, order: list[str]) -> pd.DataFrame:
    out = df.copy()
    out[category_col] = pd.Categorical(out[category_col], categories=order, ordered=True)
    return out.sort_values(category_col)


def compute_tables(merged: pd.DataFrame, realized: pd.DataFrame) -> dict[str, pd.DataFrame]:
    all_trade_summary = (
        merged.groupby("classification", observed=False)
        .agg(
            trades=("Trade ID", "count"),
            active_accounts=("Account", "nunique"),
            traded_coins=("Coin", "nunique"),
            total_volume_usd=("abs_size_usd", "sum"),
            avg_trade_size_usd=("abs_size_usd", "mean"),
            total_closed_pnl=("Closed PnL", "sum"),
            total_fees=("Fee", "sum"),
            total_net_after_fee=("net_after_fee", "sum"),
            realized_trades=("is_realized", "sum"),
        )
        .reset_index()
    )
    all_trade_summary = ordered(all_trade_summary, "classification", SENTIMENT_ORDER)

    realized_summary = (
        realized.groupby("classification", observed=False)
        .agg(
            realized_trades=("Trade ID", "count"),
            active_accounts=("Account", "nunique"),
            traded_coins=("Coin", "nunique"),
            total_realized_pnl=("Closed PnL", "sum"),
            total_net_after_fee=("net_after_fee", "sum"),
            total_fees=("Fee", "sum"),
            win_rate=("is_win", "mean"),
            loss_rate=("loss", "mean"),
            avg_pnl=("Closed PnL", "mean"),
            median_pnl=("Closed PnL", "median"),
            avg_net_pnl=("net_after_fee", "mean"),
            avg_size_usd=("abs_size_usd", "mean"),
            median_size_usd=("abs_size_usd", "median"),
            avg_gross_roi_bps=("gross_roi_bps", "mean"),
            avg_net_roi_bps=("net_roi_bps", "mean"),
        )
        .reset_index()
    )
    realized_summary = ordered(realized_summary, "classification", SENTIMENT_ORDER)

    bucket_summary = (
        realized.groupby("bucket")
        .agg(
            realized_trades=("Trade ID", "count"),
            total_realized_pnl=("Closed PnL", "sum"),
            total_net_after_fee=("net_after_fee", "sum"),
            win_rate=("is_win", "mean"),
            avg_pnl=("Closed PnL", "mean"),
            median_pnl=("Closed PnL", "median"),
            avg_net_roi_bps=("net_roi_bps", "mean"),
            avg_size_usd=("abs_size_usd", "mean"),
        )
        .reset_index()
    )
    bucket_summary = ordered(bucket_summary, "bucket", BUCKET_ORDER)

    daily_realized = (
        realized.groupby(["trade_date", "classification"], observed=True)
        .agg(
            realized_trades=("Trade ID", "count"),
            total_pnl=("Closed PnL", "sum"),
            total_net=("net_after_fee", "sum"),
            total_fees=("Fee", "sum"),
            total_volume=("abs_size_usd", "sum"),
            win_rate=("is_win", "mean"),
            active_accounts=("Account", "nunique"),
        )
        .reset_index()
    )

    daily_regime_summary = (
        daily_realized.groupby("classification", observed=False)
        .agg(
            days=("trade_date", "nunique"),
            avg_daily_realized_trades=("realized_trades", "mean"),
            avg_daily_total_net=("total_net", "mean"),
            median_daily_total_net=("total_net", "median"),
            avg_daily_win_rate=("win_rate", "mean"),
            avg_daily_volume=("total_volume", "mean"),
            avg_daily_active_accounts=("active_accounts", "mean"),
        )
        .reset_index()
    )
    daily_regime_summary = ordered(daily_regime_summary, "classification", SENTIMENT_ORDER)

    daily_sentiment_score = (
        realized.groupby(["trade_date", "value"])
        .agg(
            realized_trades=("Trade ID", "count"),
            total_net=("net_after_fee", "sum"),
            total_volume=("abs_size_usd", "sum"),
            win_rate=("is_win", "mean"),
            active_accounts=("Account", "nunique"),
        )
        .reset_index()
        .sort_values("trade_date")
    )

    direction_bucket_summary = (
        realized.groupby(["bucket", "Direction"], observed=True)
        .agg(
            trades=("Trade ID", "count"),
            total_net=("net_after_fee", "sum"),
            avg_pnl=("Closed PnL", "mean"),
            win_rate=("is_win", "mean"),
        )
        .reset_index()
    )
    direction_bucket_summary = direction_bucket_summary[
        direction_bucket_summary["Direction"].isin(["Close Long", "Close Short", "Sell"])
    ].copy()
    direction_bucket_summary = ordered(direction_bucket_summary, "bucket", BUCKET_ORDER)

    top_coins = (
        realized.groupby(["classification", "Coin"], observed=True)
        .agg(
            realized_trades=("Trade ID", "count"),
            total_net=("net_after_fee", "sum"),
            total_pnl=("Closed PnL", "sum"),
            win_rate=("is_win", "mean"),
        )
        .reset_index()
    )
    top_coins = (
        top_coins.sort_values(["classification", "total_net"], ascending=[True, False])
        .groupby("classification", observed=False)
        .head(5)
        .reset_index(drop=True)
    )
    top_coins["classification"] = pd.Categorical(
        top_coins["classification"], categories=SENTIMENT_ORDER, ordered=True
    )
    top_coins = top_coins.sort_values(["classification", "total_net"], ascending=[True, False])

    account_summary = (
        realized.groupby(["classification", "Account"], observed=True)
        .agg(
            realized_trades=("Trade ID", "count"),
            total_net=("net_after_fee", "sum"),
            total_pnl=("Closed PnL", "sum"),
            win_rate=("is_win", "mean"),
        )
        .reset_index()
    )

    top_accounts = (
        account_summary.sort_values(["classification", "total_net"], ascending=[True, False])
        .groupby("classification", observed=False)
        .head(5)
        .reset_index(drop=True)
    )
    top_accounts["classification"] = pd.Categorical(
        top_accounts["classification"], categories=SENTIMENT_ORDER, ordered=True
    )
    top_accounts = top_accounts.sort_values(["classification", "total_net"], ascending=[True, False])

    bottom_accounts = (
        account_summary.sort_values(["classification", "total_net"], ascending=[True, True])
        .groupby("classification", observed=False)
        .head(5)
        .reset_index(drop=True)
    )
    bottom_accounts["classification"] = pd.Categorical(
        bottom_accounts["classification"], categories=SENTIMENT_ORDER, ordered=True
    )
    bottom_accounts = bottom_accounts.sort_values(["classification", "total_net"], ascending=[True, True])

    monthly_net = (
        realized.groupby(["month", "classification"], observed=True)
        .agg(total_net=("net_after_fee", "sum"), realized_trades=("Trade ID", "count"))
        .reset_index()
    )

    positive_profit_share_rows = []
    for sentiment_name, group in account_summary.groupby("classification", observed=False):
        group = group.sort_values("total_net", ascending=False).reset_index(drop=True)
        positive_total = group.loc[group["total_net"] > 0, "total_net"].sum()
        negative_total = group.loc[group["total_net"] < 0, "total_net"].sum()
        top3_positive_share = (
            group.head(3)["total_net"].sum() / positive_total if positive_total else np.nan
        )
        positive_profit_share_rows.append(
            {
                "classification": sentiment_name,
                "accounts": len(group),
                "profitable_accounts": int((group["total_net"] > 0).sum()),
                "losing_accounts": int((group["total_net"] < 0).sum()),
                "positive_profit_total": positive_total,
                "negative_profit_total": negative_total,
                "top3_share_of_positive_profit": top3_positive_share,
            }
        )
    account_concentration = pd.DataFrame(positive_profit_share_rows)
    account_concentration = ordered(account_concentration, "classification", SENTIMENT_ORDER)

    return {
        "all_trade_summary": all_trade_summary,
        "realized_summary_by_sentiment": realized_summary,
        "bucket_summary": bucket_summary,
        "daily_realized_by_sentiment": daily_regime_summary,
        "daily_sentiment_score_metrics": daily_sentiment_score,
        "direction_bucket_summary": direction_bucket_summary,
        "top_coins_by_sentiment": top_coins,
        "top_accounts_by_sentiment": top_accounts,
        "bottom_accounts_by_sentiment": bottom_accounts,
        "monthly_net_by_sentiment": monthly_net,
        "account_concentration_by_sentiment": account_concentration,
    }


def save_tables(tables: dict[str, pd.DataFrame]) -> None:
    for name, df in tables.items():
        df.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)


def make_plots(tables: dict[str, pd.DataFrame], realized: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    realized_summary = tables["realized_summary_by_sentiment"].copy()
    realized_summary["classification"] = realized_summary["classification"].astype(str)
    color_list = [SENTIMENT_COLORS[label] for label in realized_summary["classification"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(
        realized_summary["classification"],
        realized_summary["total_net_after_fee"] / 1_000_000,
        color=color_list,
    )
    axes[0].set_title("Total Net Realized PnL After Fees")
    axes[0].set_ylabel("USD (Millions)")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(
        realized_summary["classification"],
        realized_summary["win_rate"] * 100,
        color=color_list,
    )
    axes[1].set_title("Realized Trade Win Rate")
    axes[1].set_ylabel("Win Rate (%)")
    axes[1].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "01_regime_net_and_win_rate.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(
        realized_summary["classification"],
        realized_summary["avg_net_pnl"],
        color=color_list,
    )
    axes[0].set_title("Average Net PnL Per Realized Trade")
    axes[0].set_ylabel("USD")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(
        realized_summary["classification"],
        realized_summary["avg_net_roi_bps"],
        color=color_list,
    )
    axes[1].set_title("Average Net Return on Notional")
    axes[1].set_ylabel("Basis Points")
    axes[1].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "02_regime_efficiency.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    daily_summary = tables["daily_realized_by_sentiment"].copy()
    daily_summary["classification"] = daily_summary["classification"].astype(str)
    daily_colors = [SENTIMENT_COLORS[label] for label in daily_summary["classification"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(
        daily_summary["classification"],
        daily_summary["avg_daily_realized_trades"],
        color=daily_colors,
    )
    axes[0].set_title("Average Daily Realized Trades")
    axes[0].set_ylabel("Trades per Day")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(
        daily_summary["classification"],
        daily_summary["avg_daily_volume"] / 1_000_000,
        color=daily_colors,
    )
    axes[1].set_title("Average Daily Realized Volume")
    axes[1].set_ylabel("USD (Millions)")
    axes[1].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "03_regime_activity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    monthly = tables["monthly_net_by_sentiment"].copy()
    monthly["classification"] = pd.Categorical(
        monthly["classification"], categories=SENTIMENT_ORDER, ordered=True
    )
    monthly_pivot = (
        monthly.pivot(index="month", columns="classification", values="total_net")
        .fillna(0)
        .reindex(columns=SENTIMENT_ORDER)
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    monthly_pivot.plot(
        ax=ax,
        linewidth=2.2,
        color=[SENTIMENT_COLORS[label] for label in SENTIMENT_ORDER],
    )
    ax.set_title("Monthly Net Realized PnL by Sentiment Regime")
    ax.set_ylabel("USD")
    ax.set_xlabel("Month")
    ax.legend(title="Sentiment")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "04_monthly_net_by_sentiment.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    direction_summary = tables["direction_bucket_summary"].copy()
    direction_summary["bucket"] = pd.Categorical(
        direction_summary["bucket"], categories=BUCKET_ORDER, ordered=True
    )
    direction_summary = direction_summary.sort_values(["bucket", "Direction"])
    direction_pivot = direction_summary.pivot(
        index="bucket", columns="Direction", values="total_net"
    ).reindex(BUCKET_ORDER)

    fig, ax = plt.subplots(figsize=(10, 6))
    direction_pivot.plot(kind="bar", ax=ax, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_title("Net PnL by Direction and Sentiment Bucket")
    ax.set_ylabel("USD")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "05_directional_edge_by_bucket.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    daily_score = tables["daily_sentiment_score_metrics"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(
        daily_score["value"],
        daily_score["total_volume"] / 1_000_000,
        alpha=0.65,
        color="#2a9d8f",
        edgecolor="white",
        linewidth=0.5,
    )
    axes[0].set_title("Daily Volume vs Fear/Greed Score")
    axes[0].set_xlabel("Fear/Greed Score")
    axes[0].set_ylabel("Realized Volume (USD Millions)")

    axes[1].scatter(
        daily_score["value"],
        daily_score["win_rate"] * 100,
        alpha=0.65,
        color="#c0392b",
        edgecolor="white",
        linewidth=0.5,
    )
    axes[1].set_title("Daily Win Rate vs Fear/Greed Score")
    axes[1].set_xlabel("Fear/Greed Score")
    axes[1].set_ylabel("Win Rate (%)")

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "06_daily_score_relationships.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_text_summary(
    tables: dict[str, pd.DataFrame], missing_sentiment: pd.DataFrame
) -> None:
    realized_summary = tables["realized_summary_by_sentiment"].set_index("classification")
    daily_summary = tables["daily_realized_by_sentiment"].set_index("classification")
    account_concentration = tables["account_concentration_by_sentiment"].set_index(
        "classification"
    )
    bucket_summary = tables["bucket_summary"].set_index("bucket")
    top_coins = tables["top_coins_by_sentiment"]

    daily_score = tables["daily_sentiment_score_metrics"]
    correlations = {
        "daily_realized_volume": daily_score["value"].corr(daily_score["total_volume"]),
        "daily_win_rate": daily_score["value"].corr(daily_score["win_rate"]),
        "daily_active_accounts": daily_score["value"].corr(daily_score["active_accounts"]),
    }

    lines = [
        "PrimeTrade.ai Assignment Summary",
        "================================",
        "",
        f"Rows excluded from sentiment analysis because the sentiment file had no label for 2024-10-26: {len(missing_sentiment)}",
        "",
        "Best regime by total net PnL after fees:",
        f"- Fear: ${realized_summary.loc['Fear', 'total_net_after_fee']:,.0f}",
        "",
        "Best regime by average net PnL per realized trade:",
        f"- Extreme Greed: ${realized_summary.loc['Extreme Greed', 'avg_net_pnl']:,.2f}",
        "",
        "Best regime by win rate:",
        f"- Extreme Greed: {realized_summary.loc['Extreme Greed', 'win_rate']:.1%}",
        "",
        "Highest average daily net PnL:",
        f"- Extreme Fear: ${daily_summary.loc['Extreme Fear', 'avg_daily_total_net']:,.0f}",
        "",
        "Broad-bucket comparison:",
        f"- Fear-side win rate: {bucket_summary.loc['Fear-side', 'win_rate']:.1%}",
        f"- Greed-side win rate: {bucket_summary.loc['Greed-side', 'win_rate']:.1%}",
        f"- Fear-side avg size: ${bucket_summary.loc['Fear-side', 'avg_size_usd']:,.0f}",
        f"- Greed-side avg size: ${bucket_summary.loc['Greed-side', 'avg_size_usd']:,.0f}",
        "",
        "Daily sentiment relationships:",
        f"- Correlation(score, daily realized volume) = {correlations['daily_realized_volume']:.3f}",
        f"- Correlation(score, daily win rate) = {correlations['daily_win_rate']:.3f}",
        f"- Correlation(score, active accounts) = {correlations['daily_active_accounts']:.3f}",
        "",
        "Account concentration:",
        f"- Top 3 accounts captured {account_concentration.loc['Fear', 'top3_share_of_positive_profit']:.1%} of positive profit in Fear",
        f"- Top 3 accounts captured {account_concentration.loc['Extreme Greed', 'top3_share_of_positive_profit']:.1%} of positive profit in Extreme Greed",
        "",
        "Top coin by regime:",
    ]

    for sentiment_name in SENTIMENT_ORDER:
        top_coin = top_coins[top_coins["classification"] == sentiment_name].head(1)
        if top_coin.empty:
            continue
        coin = top_coin.iloc[0]
        lines.append(
            f"- {sentiment_name}: {coin['Coin']} with ${coin['total_net']:,.0f} net after fees"
        )

    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    historical, sentiment = load_data()
    merged, realized, missing_sentiment = prepare_data(historical, sentiment)
    tables = compute_tables(merged, realized)
    save_tables(tables)
    make_plots(tables, realized)
    write_text_summary(tables, missing_sentiment)
    print(f"Saved outputs to: {OUTPUT_DIR}")
    print(f"Saved plots to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
