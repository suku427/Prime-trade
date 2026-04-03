# Trader Performance vs Bitcoin Market Sentiment

## Objective

Analyze whether Hyperliquid trader behavior and realized performance change across Bitcoin market sentiment regimes, then turn those findings into practical trading recommendations.

## Data Used

- `historical_data.csv`: 211,224 executions across 32 accounts and 246 coins, covering `2023-05-01` to `2025-05-01`.
- `fear_greed_index.csv`: Daily Bitcoin sentiment labels from `2018-02-01` to `2025-05-02`.
- Six trader rows on `2024-10-26` had no matching sentiment label in the fear/greed file and were excluded from regime analysis.

## Method

1. Parse Hyperliquid timestamps to calendar dates in IST.
2. Join each execution to the same-day Fear/Greed label.
3. Separate activity metrics from realized performance:
   - Activity: all executions, volume, account participation.
   - Performance: rows with `Closed PnL != 0`.
4. Measure performance by sentiment using:
   - net PnL after fees
   - win rate
   - average and median realized PnL
   - return on traded notional
   - direction-specific edge (`Close Long`, `Close Short`, `Sell`)

## Key Findings

### 1. Fear regimes trigger much heavier trading activity

- `Fear` had the largest average realized ticket size at about `$8,041` per trade.
- `Extreme Fear` averaged `743` realized trades per day, versus only `151` in `Greed`.
- Daily sentiment score had a negative correlation with realized volume (`-0.287`) and active accounts (`-0.323`).

Interpretation: when sentiment deteriorates, traders become more active and deploy more capital.

### 2. Extreme Greed is the most efficient regime

- `Extreme Greed` produced the highest win rate: `89.2%`.
- It also had the highest average net PnL per realized trade: about `$129.55`.
- It delivered the best average net return on notional: `765 bps`.
- These results were achieved with the smallest average realized trade size, roughly `$2,780`.

Interpretation: traders appear to capture cleaner, higher-quality opportunities when the market is strongly bullish.

### 3. Fear produced the largest absolute profit pool

- `Fear` generated the highest total net realized PnL after fees: about `$3.31M`.
- `Fear` also averaged around `$44.1K` net PnL per day.
- `Extreme Fear` had an even higher average daily net PnL, about `$51.9K`, but only across `14` labeled days.

Interpretation: fearful markets create the richest dollar opportunity set, even if trade efficiency is lower than in extreme greed.

### 4. Plain Greed is the weakest regime

- `Greed` had the lowest median daily net PnL at about `$898`.
- It also contained the worst single day in the sample: `2025-04-23`, with about `-$423.5K` net after fees.
- Its realized win rate (`76.9%`) lagged both `Fear` (`87.3%`) and `Extreme Greed` (`89.2%`).

Interpretation: moderate optimism looks less reliable than either strong fear or strong bullish momentum.

### 5. Trade direction matters by regime

- In the combined `Fear-side` bucket, `Close Short` contributed about `$2.28M` net after fees and `Close Long` added about `$1.90M`.
- In the combined `Greed-side` bucket, `Sell` was dominant at about `$2.84M` net, followed by `Close Long` at about `$1.13M`.

Interpretation: short covering matters more in fearful conditions, while profit-taking and long exits dominate in bullish conditions.

### 6. Profit is concentrated in a handful of accounts

- The top 3 accounts captured about `60.9%` of all positive profit in `Fear`.
- The top 3 accounts captured about `64.9%` of all positive profit in `Extreme Greed`.
- Concentration was still high even in `Greed`, where the top 3 accounts contributed `48.2%` of positive profits.

Interpretation: a relatively small number of traders account for a large share of edge, so account-level monitoring matters.

### 7. Coin leadership rotates by regime

- `Fear` was led by `HYPE`, `SOL`, `ETH`, and `BTC`.
- `Extreme Greed` was dominated by `@107`, which contributed about `$1.99M` net after fees.
- `HYPE` was consistently strong during fear-driven regimes.

Interpretation: regime-aware coin selection is likely as important as direction selection.

## Recommendations

1. Allocate more risk budget during fear-driven markets, but pair it with tighter loss controls because dispersion also rises.
2. Favor momentum and profit-taking setups during extreme greed, where smaller positions still produced the best return efficiency.
3. Build regime-specific playbooks by coin, especially for `HYPE`, `SOL`, `BTC`, and `@107`.
4. Track account-level concentration so strategy decisions are not overly influenced by a few standout wallets.
5. Treat moderate `Greed` as a caution regime rather than a high-conviction regime, because it showed the weakest consistency and the worst drawdown day.

## Deliverables Generated

- Summary tables in `outputs/`
- Charts in `plots/`
- Reproducible script: `analysis.py`

## Caveat

This analysis measures realized PnL on the execution where the PnL is booked. It does not reconstruct full position life cycles from entry to exit, so the results are best interpreted as realized trading performance by sentiment regime rather than complete strategy attribution.
