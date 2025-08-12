#!/usr/bin/env python3
"""
crypto_monitor: fetch Kraken market data, compute indicators, and publish JSON snapshots.

- No API keys required (public OHLCV only).
- Writes docs/snapshot.json and docs/changes.json
- Config via env:
    EXCHANGE    (default: kraken)
    TICKERS     (comma-separated; if empty auto-pick liquid spot pairs quoted in USDT/USD/EUR)
    TIMEFRAMES  (comma-separated; default: 1h,15m)
    LIMIT       (default: 200)
    MAX_PAIRS   (used only when TICKERS not set; default: 15)
"""
import os, json
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange

UTC = timezone.utc

def now_iso():
    return datetime.now(UTC).isoformat()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def pick_markets(ex, max_pairs=15):
    """Auto-pick liquid spot pairs quoted in USDT/USD/EUR."""
    quote_whitelist = {"USDT", "USD", "EUR"}
    markets = ex.load_markets()
    syms = []
    for sym, m in markets.items():
        try:
            if m.get("spot") and m.get("quote") in quote_whitelist and m.get("active", True):
                syms.append(sym)
        except Exception:
            continue
    syms = sorted(set(syms))
    return syms[:max_pairs]

def compute_indicators(df: pd.DataFrame):
    df = df.copy()
    df["ema_20"] = EMAIndicator(close=df["close"], window=20).ema_indicator()
    df["ema_50"] = EMAIndicator(close=df["close"], window=50).ema_indicator()

    rsi = RSIIndicator(close=df["close"], window=14)
    df["rsi_14"] = rsi.rsi()

    macd = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["atr_14"] = atr.average_true_range()

    adx = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["adx_14"] = adx.adx()

    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["bb_mid"]

    return df

def derive_signals(df: pd.DataFrame):
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last

    if float(last["rsi_14"]) >= 70:
        rsi_zone = "overbought"
    elif float(last["rsi_14"]) <= 30:
        rsi_zone = "oversold"
    else:
        rsi_zone = "neutral"

    macd_cross = (
        "bullish"
        if (last["macd"] > last["macd_signal"] and prev["macd"] <= prev["macd_signal"])
        else "bearish"
        if (last["macd"] < last["macd_signal"] and prev["macd"] >= prev["macd_signal"])
        else "none"
    )

    trend = "trending" if float(last["adx_14"]) >= 25 else "choppy"
    breakout_up = bool(last["close"] > last["bb_high"])
    breakout_dn = bool(last["close"] < last["bb_low"])
    ema_state = (
        "bullish" if last["ema_20"] > last["ema_50"] else "bearish" if last["ema_20"] < last["ema_50"] else "flat"
    )

    return {
        "rsi_zone": rsi_zone,
        "macd_cross": macd_cross,
        "trend": trend,
        "breakout_up": breakout_up,
        "breakout_dn": breakout_dn,
        "ema_state": ema_state,
    }

def ohlcv_to_df(ohlcv):
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    df = pd.DataFrame(ohlcv, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df

def fetch_ohlcv_safe(ex, symbol, timeframe, limit):
    try:
        data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not data:
            return None
        return data
    except Exception as e:
        print(f"[WARN] fetch_ohlcv failed for {symbol} {timeframe}: {e}")
        return None

def build_snapshot():
    exchange_id = os.getenv("EXCHANGE", "kraken")
    timeframes = [t.strip() for t in os.getenv("TIMEFRAMES", "1h,15m").split(",") if t.strip()]
    limit = int(os.getenv("LIMIT", "200"))
    max_pairs = int(os.getenv("MAX_PAIRS", "15"))
    tickers_env = os.getenv("TICKERS", "").strip()

    ex_class = getattr(ccxt, exchange_id)
    ex = ex_class({"enableRateLimit": True})
    ex.load_markets()

    if tickers_env:
        symbols = [t.strip() for t in tickers_env.split(",") if t.strip()]
    else:
        symbols = pick_markets(ex, max_pairs=max_pairs)

    out = {
        "exchange": exchange_id,
        "generated_at": now_iso(),
        "timeframes": timeframes,
        "limit": limit,
        "symbols": symbols,
        "data": {},
    }

    for sym in symbols:
        out["data"][sym] = {}
        for tf in timeframes:
            ohlcv = fetch_ohlcv_safe(ex, sym, tf, limit)
            if ohlcv is None or len(ohlcv) < 60:
                continue
            df = ohlcv_to_df(ohlcv)
            df = compute_indicators(df)
            sig = derive_signals(df)
            last = df.iloc[-1]
            row = {
                "last_candle": {
                    "timestamp": last["timestamp"].isoformat(),
                    "open": float(last["open"]),
                    "high": float(last["high"]),
                    "low": float(last["low"]),
                    "close": float(last["close"]),
                    "volume": float(last["volume"]),
                },
                "indicators": {
                    "ema_20": float(last["ema_20"]),
                    "ema_50": float(last["ema_50"]),
                    "rsi_14": float(last["rsi_14"]),
                    "macd": float(last["macd"]),
                    "macd_signal": float(last["macd_signal"]),
                    "macd_hist": float(last["macd_hist"]),
                    "atr_14": float(last["atr_14"]),
                    "adx_14": float(last["adx_14"]),
                    "bb_high": float(last["bb_high"]),
                    "bb_low": float(last["bb_low"]),
                    "bb_mid": float(last["bb_mid"]),
                    "bb_width": float(last["bb_width"]),
                },
                "signals": sig,
            }
            out["data"][sym][tf] = row
    return out

def compare_snapshots(prev, curr):
    changes = {"generated_at": now_iso(), "notes": [], "changes": []}
    if not prev:
        changes["notes"].append("first_run")
        return changes

    for sym, tf_data in curr.get("data", {}).items():
        for tf, row in tf_data.items():
            curr_sig = row.get("signals", {})
            prev_sig = prev.get("data", {}).get(sym, {}).get(tf, {}).get("signals", {})
            diffs = {}
            for k, v in curr_sig.items():
                if prev_sig.get(k) != v:
                    diffs[k] = {"from": prev_sig.get(k), "to": v}
            if diffs:
                changes["changes"].append(
                    {
                        "symbol": sym,
                        "timeframe": tf,
                        "diff": diffs,
                        "price": row.get("last_candle", {}).get("close"),
                    }
                )
    return changes

def main():
    root = Path(__file__).resolve().parent
    docs = root / "docs"
    ensure_dir(docs)

    prev_path = docs / "snapshot.json"
    prev = None
    if prev_path.exists():
        try:
            prev = json.loads(prev_path.read_text())
        except Exception:
            prev = None

    snapshot = build_snapshot()

    # Write snapshot
    with (docs / "snapshot.json").open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    # Write changes
    changes = compare_snapshots(prev, snapshot)
    with (docs / "changes.json").open("w", encoding="utf-8") as f:
        json.dump(changes, f, ensure_ascii=False, indent=2)

    print(f"Wrote docs/snapshot.json and docs/changes.json at {now_iso()}")

if __name__ == "__main__":
    main()
