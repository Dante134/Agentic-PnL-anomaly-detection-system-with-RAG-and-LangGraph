"""
Synthetic trade data generator.
Produces realistic daily PnL records with planted anomalies so we have
something interesting for the agent to detect.
"""

import random
import json
from datetime import date, timedelta


DESKS = ["Rates", "Credit", "Equity", "FX"]
INSTRUMENTS = ["IRS", "CDS", "Equity Swap", "FX Forward", "Bond", "Option"]

random.seed(42)


def normal_pnl(desk: str) -> float:
    """Typical daily PnL ranges by desk (USD thousands)."""
    ranges = {
        "Rates":  (-80,  80),
        "Credit": (-50,  50),
        "Equity": (-120, 120),
        "FX":     (-40,  40),
    }
    lo, hi = ranges[desk]
    return round(random.gauss((lo + hi) / 2, (hi - lo) / 4), 2)


def generate_trades(n_days: int = 10, n_trades_per_day: int = 8) -> list[dict]:
    trades = []
    trade_id = 1000
    start = date(2024, 11, 1)

    for day_offset in range(n_days):
        trade_date = start + timedelta(days=day_offset)
        if trade_date.weekday() >= 5:          # skip weekends
            continue
        for _ in range(n_trades_per_day):
            desk = random.choice(DESKS)
            pnl = normal_pnl(desk)

            # Plant anomalies in ~15% of trades
            anomaly_type = None
            roll = random.random()
            if roll < 0.05:
                # Large loss spike
                pnl = round(random.uniform(-900, -500), 2)
                anomaly_type = "large_loss_spike"
            elif roll < 0.08:
                # Zero PnL on active position
                pnl = 0.0
                anomaly_type = "zero_pnl"
            elif roll < 0.12:
                # Extreme gain (could be booking error)
                pnl = round(random.uniform(600, 1200), 2)
                anomaly_type = "large_gain_spike"
            elif roll < 0.15:
                # Sudden reversal (large negative after recent positives)
                pnl = round(random.uniform(-400, -200), 2)
                anomaly_type = "sudden_reversal"

            trades.append({
                "trade_id":     f"TRD-{trade_id}",
                "trade_date":   trade_date.isoformat(),
                "desk":         desk,
                "instrument":   random.choice(INSTRUMENTS),
                "notional_usd": random.choice([1_000_000, 5_000_000, 10_000_000, 50_000_000]),
                "daily_pnl":    pnl,        # in USD thousands
                "anomaly_type": anomaly_type,   # None for normal trades
            })
            trade_id += 1

    return trades


if __name__ == "__main__":
    trades = generate_trades()
    with open("trades.json", "w") as f:
        json.dump(trades, f, indent=2)
    total     = len(trades)
    anomalies = sum(1 for t in trades if t["anomaly_type"])
    print(f"Generated {total} trades, {anomalies} anomalies planted.")
    print(json.dumps(trades[:3], indent=2))
