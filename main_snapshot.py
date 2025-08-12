#!/usr/bin/env python3
"""
crypto_monitor: fetch Kraken market data, compute indicators, and publish JSON snapshots.

- No API keys required (public OHLCV only).
- Saves docs/snapshot.json and docs/changes.json
- Configurable via env vars:
    EXCHANGE      (default: kraken)
    TICKERS       (comma-separated; default: auto-pick spot/perp pairs quoted in USDT)
    TIMEFRAMES    (comma-separated; default: 1M,1w,1d,4h,1h,15m,5m,1m)
    LIMIT         (candles per timeframe; default: 200)
    MAX_PAIRS     (only when TICKERS not set; default: 15)
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
    # Auto-pick liquid spot pairs quoted in USDT (solo USDT come richiesto)
    quote_whitelist = {"USDT"}
    markets = ex.load_markets()
    syms = []
    for sym, m in markets.items():
        try:
            if m.get('spot') and m.get('quote') in quote_whitelist and m.get('active', True):
                syms.append(sym)
        except Exception:
            continue
    syms = sorted(set(syms))
    return syms[:max_pairs]

def compute_indicators(df: pd.DataFrame):
    df = df.copy()
    df['ema_20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
    df['ema_50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()

    # === AGGIUNTA: indicatori extra per L/S/S ===
    df['ema_9']   = EMAIndicator(close=df['close'], window=9).ema_indicator()
    df['ema_21']  = EMAIndicator(close=df['close'], window=21).ema_indicator()
    df['ema_200'] = EMAIndicator(close=df['close'], window=200).ema_indicator()
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    # === FINE AGGIUNTA ===

    rsi = RSIIndicator(close=df['close'], window=14)
    df['rsi_14'] = rsi.rsi()

    macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()

    atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['atr_14'] = atr.average_true_range()

    adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['adx_14'] = adx.adx()

    bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_mid'] = bb.bollinger_mavg()
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']

    return df

def derive_signals(df: pd.DataFrame):
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last

    if last['rsi_14'] >= 70:
        rsi_zone = 'overbought'
    elif last['rsi_14'] <= 30:
        rsi_zone = 'oversold'
    else:
        rsi_zone = 'neutral'

    macd_cross = 'bullish' if (last['macd'] > last['macd_signal'] and prev['macd'] <= prev['macd_signal']) else \
                 'bearish' if (last['macd'] < last['macd_signal'] and prev['macd'] >= prev['macd_signal']) else \
                 'none'

    trend = 'trending' if last['adx_14'] >= 25 else 'choppy'
    breakout_up = bool(last['close'] > last['bb_high'])
    breakout_dn = bool(last['close'] < last['bb_low'])
    ema_state = 'bullish' if last['ema_20'] > last['ema_50'] else 'bearish' if last['ema_20'] < last['ema_50'] else 'flat'

    # === AGGIUNTA: segnali derivati per L/S/S ===
    ema_alignment = (
        'bullish' if (last['ema_20'] > last['ema_50'] > last['ema_200'])
        else 'bearish' if (last['ema_20'] < last['ema_50'] < last['ema_200'])
        else 'mixed'
    )
    long_regime = 'above_200' if last['close'] > last['ema_200'] else 'below_200'
    scalp_ema = 'bullish' if last['ema_9'] > last['ema_21'] else 'bearish'
    # === FINE AGGIUNTA ===

    return {
        'rsi_zone': rsi_zone,
        'macd_cross': macd_cross,
        'trend': trend,
        'breakout_up': breakout_up,
        'breakout_dn': breakout_dn,
        'ema_state': ema_state,
        # aggiunte
        'ema_alignment': ema_alignment,
        'long_regime': long_regime,
        'scalp_ema': scalp_ema,
    }

def ohlcv_to_df(ohlcv):
    cols = ['timestamp','open','high','low','close','volume']
    df = pd.DataFrame(ohlcv, columns=cols)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
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
    exchange_id = os.getenv("EXCHANGE","kraken")
    # === MODIFICA: includiamo 1d/1w/1M come default ===
    timeframes = [t.strip() for t in os.getenv("TIMEFRAMES","1M,1w,1d,4h,1h,15m,5m,1m").split(",") if t.strip()]
    limit = int(os.getenv("LIMIT","200"))
    max_pairs = int(os.getenv("MAX_PAIRS","15"))
    tickers_env = os.getenv("TICKERS", "").strip()

    ex_class = getattr(ccxt, exchange_id)
    ex = ex_class({"enableRateLimit": True})
    ex.load_markets()

    if tickers_env:
        symbols = [t.strip() for t in tickers_env.split(",") if t.strip()]
    else:
        symbols = pick_markets(ex, max_pairs=max_pairs)

    out = {
        'exchange': exchange_id,
        'generated_at': now_iso(),
        'timeframes': timeframes,
        'limit': limit,
        'symbols': symbols,
        'data': {}
    }

    for sym in symbols:
        out['data'][sym] = {}
        for tf in timeframes:
            ohlcv = fetch_ohlcv_safe(ex, sym, tf, limit)
            if ohlcv is None or len(ohlcv) < 60:
                continue
            df = ohlcv_to_df(ohlcv)
            df = compute_indicators(df)
            sig = derive_signals(df)
            last = df.iloc[-1]
            row = {
                'last_candle': {
                    'timestamp': last['timestamp'].isoformat(),
                    'open': float(last['open']),
                    'high': float(last['high']),
                    'low': float(last['low']),
                    'close': float(last['close']),
                    'volume': float(last['volume']),
                },
                'indicators': {
                    'ema_20': float(last['ema_20']),
                    'ema_50': float(last['ema_50']),
                    # === AGGIUNTA: pubblica anche i nuovi indicatori ===
                    'ema_9': float(last['ema_9']),
                    'ema_21': float(last['ema_21']),
                    'ema_200': float(last['ema_200']),
                    'vol_sma_20': float(last['vol_sma_20']) if pd.notna(last['vol_sma_20']) else None,
                    # === FINE AGGIUNTA ===
                    'rsi_14': float(last['rsi_14']),
                    'macd': float(last['macd']),
                    'macd_signal': float(last['macd_signal']),
                    'macd_hist': float(last['macd_hist']),
                    'atr_14': float(last['atr_14']),
                    'adx_14': float(last['adx_14']),
                    'bb_high': float(last['bb_high']),
                    'bb_low': float(last['bb_low']),
                    'bb_mid': float(last['bb_mid']),
                    'bb_width': float(last['bb_width']),
                },
                'signals': sig,
            }
            out['data'][sym][tf] = row
    return out

def compare_snapshots(prev, curr):
    changes = {'generated_at': now_iso(), 'notes': [], 'changes': []}
    if not prev:
        changes['notes'].append('first_run')
        return changes

    for sym, tf_data in curr.get('data', {}).items():
        for tf, row in tf_data.items():
            curr_sig = row.get('signals', {})
            prev_sig = prev.get('data', {}).get(sym, {}).get(tf, {}).get('signals', {})
            diffs = {}
            for k, v in curr_sig.items():
                if prev_sig.get(k) != v:
                    diffs[k] = {'from': prev_sig.get(k), 'to': v}
            if diffs:
                changes['changes'].append({
                    'symbol': sym,
                    'timeframe': tf,
                    'diff': diffs,
                    'price': row.get('last_candle', {}).get('close')
                })
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

    with (docs / "snapshot.json").open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    changes = compare_snapshots(prev, snapshot)
    with (docs / "changes.json").open("w", encoding="utf-8") as f:
        json.dump(changes, f, ensure_ascii=False, indent=2)

    # === TXT (estesi con i nuovi campi) ===
    signals_lines = [f"generated_at={snapshot.get('generated_at','N/A')}"]
    for sym, tf_data in snapshot.get("data", {}).items():
        for tf, row in tf_data.items():
            ind = row.get("indicators", {})
            sig = row.get("signals", {})
            lc = row.get("last_candle", {})
            signals_lines.append(
                f"{sym},{tf}"
                f",price={lc.get('close')}"
                f",rsi={round(ind.get('rsi_14',0),2)}"
                f",macd={round(ind.get('macd',0),6)}"
                f",macd_sig={round(ind.get('macd_signal',0),6)}"
                f",macd_hist={round(ind.get('macd_hist',0),6)}"
                f",ema9={round(ind.get('ema_9',0),6)}"
                f",ema21={round(ind.get('ema_21',0),6)}"
                f",ema20={round(ind.get('ema_20',0),6)}"
                f",ema50={round(ind.get('ema_50',0),6)}"
                f",ema200={round(ind.get('ema_200',0),6)}"
                f",atr={round(ind.get('atr_14',0),6)}"
                f",bbw={round(ind.get('bb_width',0),6)}"
                f",vol={lc.get('volume')}"
                f",vol_sma20={round(ind.get('vol_sma_20',0),6) if ind.get('vol_sma_20') is not None else 'nan'}"
                f",macd_cross={sig.get('macd_cross')}"
                f",ema_state={sig.get('ema_state')}"
                f",ema_align={sig.get('ema_alignment')}"
                f",long_regime={sig.get('long_regime')}"
                f",scalp_ema={sig.get('scalp_ema')}"
                f",trend={sig.get('trend')}"
                f",breakout_up={sig.get('breakout_up')}"
                f",breakout_dn={sig.get('breakout_dn')}"
                f",adx={round(ind.get('adx_14',0),1)}"
            )
    (docs / "signals_feed.txt").write_text("\n".join(signals_lines) + "\n", encoding="utf-8")

    changes_lines = [f"generated_at={changes.get('generated_at','N/A')}"]
    for ch in changes.get("changes", []):
        sym = ch.get("symbol")
        tf = ch.get("timeframe")
        price = ch.get("price")
        for k, v in ch.get("diff", {}).items():
            changes_lines.append(
                f"{sym},{tf},{k},{v.get('from')},{v.get('to')},price={price}"
            )
    (docs / "changes_feed.txt").write_text("\n".join(changes_lines) + "\n", encoding="utf-8")
    # === FINE TXT ===

    # === Creazione pagina HTML per signals (no-cache) ===
    ts = snapshot.get('generated_at','N/A')
    html_signals = [
        "<html><head><meta charset='utf-8'>",
        "<meta http-equiv='Cache-Control' content='no-cache, no-store, must-revalidate'>",
        "<meta http-equiv='Pragma' content='no-cache'>",
        "<meta http-equiv='Expires' content='0'>",
        "<title>Signals Feed</title></head><body>"
    ]
    html_signals.append(f"<h1>Signals Feed - generated at {ts}</h1>")
    html_signals.append("<table border='1' cellpadding='5'><tr><th>Symbol</th><th>Timeframe</th><th>Price</th><th>RSI</th><th>MACD Cross</th><th>EMA State</th><th>Trend</th><th>Breakout Up</th><th>Breakout Down</th><th>ADX</th></tr>")
    for sym, tf_data in snapshot.get("data", {}).items():
        for tf, row in tf_data.items():
            ind = row.get("indicators", {})
            sig = row.get("signals", {})
            lc = row.get("last_candle", {})
            html_signals.append(
                f"<tr><td>{sym}</td><td>{tf}</td>"
                f"<td>{lc.get('close')}</td>"
                f"<td>{round(ind.get('rsi_14',0),2)}</td>"
                f"<td>{sig.get('macd_cross')}</td>"
                f"<td>{sig.get('ema_state')}</td>"
                f"<td>{sig.get('trend')}</td>"
                f"<td>{sig.get('breakout_up')}</td>"
                f"<td>{sig.get('breakout_dn')}</td>"
                f"<td>{round(ind.get('adx_14',0),1)}</td></tr>"
            )
    html_signals.append("</table></body></html>")
    (docs / "signals_feed.html").write_text("\n".join(html_signals), encoding="utf-8")

    # === Creazione pagina HTML per changes (no-cache) ===
    ts_changes = changes.get('generated_at','N/A')
    html_changes = [
        "<html><head><meta charset='utf-8'>",
        "<meta http-equiv='Cache-Control' content='no-cache, no-store, must-revalidate'>",
        "<meta http-equiv='Pragma' content='no-cache'>",
        "<meta http-equiv='Expires' content='0'>",
        "<title>Changes Feed</title></head><body>"
    ]
    html_changes.append(f"<h1>Changes Feed - generated at {ts_changes}</h1>")
    html_changes.append("<table border='1' cellpadding='5'><tr><th>Symbol</th><th>Timeframe</th><th>Indicator</th><th>From</th><th>To</th><th>Price</th></tr>")
    for ch in changes.get("changes", []):
        sym = ch.get("symbol")
        tf = ch.get("timeframe")
        price = ch.get("price")
        for k, v in ch.get("diff", {}).items():
            html_changes.append(
                f"<tr><td>{sym}</td><td>{tf}</td><td>{k}</td>"
                f"<td>{v.get('from')}</td><td>{v.get('to')}</td><td>{price}</td></tr>"
            )
    html_changes.append("</table></body></html>")
    (docs / "changes_feed.html").write_text("\n".join(html_changes), encoding="utf-8")

    # === Fine HTML ===

    print(f"Wrote docs/snapshot.json, docs/changes.json, docs/signals_feed.txt and docs/changes_feed.txt at {now_iso()}")

if __name__ == "__main__":
    main()
