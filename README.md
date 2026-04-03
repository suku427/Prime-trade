# PrimeTrade.ai Sentiment vs Trader Performance Analysis

This folder contains a reproducible analysis of how Hyperliquid trader performance changes across Bitcoin fear/greed regimes.

## Files

- `analysis.py`: Loads the two raw CSVs, joins trader activity to daily sentiment, exports summary tables, and saves charts.
- `outputs/`: CSV exports for the main tables plus a short text summary.
- `plots/`: PNG charts used in the report.
- `report.md`: Concise write-up of the methods, findings, and recommendations.

## Assumptions

- Performance is measured on rows where `Closed PnL != 0`, because those rows represent realized outcomes.
- Trade activity metrics such as volume and trade count still use the full execution stream where helpful.
- Six Hyperliquid rows on `2024-10-26` have no matching sentiment label in the provided fear/greed file, so they are excluded from regime-specific comparisons.

## How To Run

From `C:\Users\sukum\Downloads\prime_trade_assignment`:

```powershell
python analysis.py
```

The script expects the raw files at:

- `C:\Users\sukum\Downloads\historical_data.csv`
- `C:\Users\sukum\Downloads\fear_greed_index.csv`
