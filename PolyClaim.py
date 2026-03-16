import os
import json
import asyncio
import logging
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

import requests
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs

# web3 is only needed in LIVE mode for on-chain redemption.
# Install with:  pip install web3
try:
    from web3 import Web3
    from web3.middleware import geth_poa_middleware
    _WEB3_AVAILABLE = True
except ImportError:
    _WEB3_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("PolyBTC")


class Config:
    # ┌─────────────────────────────────────────────────────────────────────┐
    # │                        SANDBOX MODE TOGGLE                          │
    # │                                                                     │
    # │  True  → Simulates $10 trades. No real money spent. Prints a full   │
    # │           financial report when you stop the bot (Ctrl+C).          │
    # │                                                                     │
    # │  False → Uses POLYGON_PRIVATE_KEY to place real on-chain orders.    │
    # │           REAL MONEY WILL BE SPENT. Only set False when ready.      │
    # └─────────────────────────────────────────────────────────────────────┘
    SANDBOX_MODE: bool = True

    # --- Wallet (only used when SANDBOX_MODE = False) ---
    PK: str = os.getenv("POLYGON_PRIVATE_KEY", "")

    # --- Risk ---
    MAX_USD_PER_TRADE: Decimal = Decimal("10.0")

    # --- Signal ---
    # Fires when market price (= probability) >= CONFIDENCE_THRESHOLD.
    # e.g. UP price = 0.72 means 72% chance UP wins → buy if >= 0.70
    CONFIDENCE_THRESHOLD: Decimal = Decimal("0.70")

    # Maximum entry price — avoids buying into moves that already happened.
    # At 0.90 you risk $9.00 to make $1.00 (terrible risk/reward).
    # At 0.80 you risk $8.00 to make $2.00 — still 25% upside minimum.
    # Set to Decimal("1.0") to disable.
    MAX_ENTRY_PRICE: Decimal = Decimal("0.85")

    # --- Spread gate ---
    MAX_SPREAD_ALLOWED: Decimal = Decimal("0.03")   # 3%

    # --- Time gate ---
    # Only enter after 2:30 has elapsed in the current 5-min window.
    WINDOW_SECONDS: int = 300
    MIN_ELAPSED_SECONDS: int = 90     # 90s  = 1 min 30 sec

    # --- Loop ---
    CHECK_INTERVAL: int = 15          # seconds between scans

    # --- Persistence ---
    STATE_FILE:   str = "trading_state.json"
    SESSION_FILE: str = "live_sessions.json"       # cumulative session history
    WINDOWS_FILE: str = "windows.json"        # per-window log (all windows, traded or not)

    # --- Auto-Claim (LIVE mode only) ---
    # When True, the bot calls redeemPositions() on-chain after a market resolves,
    # sending winnings back to your wallet automatically.
    # Requires:  pip install web3
    AUTO_CLAIM: bool = True

    # Polymarket CTF Exchange — standard Polygon mainnet address.
    CTF_ADDRESS: str = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"

    # --- API ---
    CLOB_HOST: str = "https://clob.polymarket.com"
    GAMMA_API: str = "https://gamma-api.polymarket.com"
    CHAIN_ID: int = 137               # Polygon mainnet
    POLYGON_RPC: str = "https://polygon-rpc.com"  # Public RPC used for claim tx


# ---------------------------------------------------------------------------
# Time-gate helpers
# ---------------------------------------------------------------------------

def seconds_elapsed_in_window(now: datetime | None = None) -> int:
    """
    Seconds elapsed since the start of the current 5-min UTC window.

    Polymarket BTC 5-min windows align to UTC boundaries:
    00:00, 00:05, 00:10, …

    Example: UTC 14:32:45 → window started 14:30:00 → elapsed = 165s (2:45)
    """
    if now is None:
        now = datetime.now(timezone.utc)
    total = now.hour * 3600 + now.minute * 60 + now.second
    return total % Config.WINDOW_SECONDS


def is_past_time_gate(now: datetime | None = None) -> bool:
    """True once >= 2:30 has elapsed in the current 5-min window."""
    return seconds_elapsed_in_window(now) >= Config.MIN_ELAPSED_SECONDS


def seconds_until_gate(now: datetime | None = None) -> int:
    """Seconds until the time-gate opens (0 if already open)."""
    remaining = Config.MIN_ELAPSED_SECONDS - seconds_elapsed_in_window(now)
    return max(0, remaining)


def window_start_iso(now: datetime | None = None) -> str:
    """ISO timestamp of the start of the current 5-min window."""
    if now is None:
        now = datetime.now(timezone.utc)
    elapsed = seconds_elapsed_in_window(now)
    start = now.replace(microsecond=0) - timedelta(seconds=elapsed)
    return start.isoformat()


# ---------------------------------------------------------------------------
# Persistent state — keyed per condition_id + window
# ---------------------------------------------------------------------------

class State:
    """
    Prevents double-entering the same market in the same 5-min window.

    Key format: "<condition_id>:<window_start_iso>"
    Each new 5-min window is treated as a fresh opportunity.
    """

    def __init__(self):
        self._data: dict = self._load()

    def _load(self) -> dict:
        if os.path.exists(Config.STATE_FILE):
            with open(Config.STATE_FILE, "r") as f:
                return json.load(f)
        return {}

    def _save(self):
        with open(Config.STATE_FILE, "w") as f:
            json.dump(self._data, f, indent=4)

    def _key(self, condition_id: str, now: datetime | None = None) -> str:
        return f"{condition_id}:{window_start_iso(now)}"

    def is_traded(self, condition_id: str, now: datetime | None = None) -> bool:
        return self._key(condition_id, now) in self._data

    def record_trade(
        self,
        condition_id: str,
        side: str,
        price: Decimal,
        shares: Decimal,
        now: datetime | None = None,
        sandbox: bool = False,
    ):
        key = self._key(condition_id, now)
        cost       = float((shares * price).quantize(Decimal("0.01")))
        max_payout = float((shares * Decimal("1.0")).quantize(Decimal("0.01")))
        self._data[key] = {
            "timestamp":           datetime.now(timezone.utc).isoformat(),
            "side":                side,
            "entry_price":         float(price),
            "shares":              float(shares),
            "cost_usd":            cost,
            "max_payout_usd":      max_payout,
            "amount_gained_if_win": round(max_payout - cost, 2),
            "sandbox":             sandbox,
        }
        self._save()


# ---------------------------------------------------------------------------
# Sandbox ledger — accumulates simulated trades for the final report
# ---------------------------------------------------------------------------

class SandboxLedger:
    """
    Tracks all simulated trades during a SANDBOX_MODE session.

    P&L settlement:
    When a 5-min window closes (the bot detects a window rollover), it calls
    settle_window() which hits the Gamma API for the final outcomePrices.
    A price of 1.0 means that outcome won; 0.0 means it lost.

    Payout = shares * 1.0  if you picked the winning side
    P&L    = payout - cost
    """

    def __init__(self):
        self._trades: list[dict] = []
        self._session_start: datetime = datetime.now(timezone.utc)
        self._settled_windows: set = set()  # window_ts values already settled

    def record(
        self,
        question: str,
        side: str,
        price: Decimal,
        shares: Decimal,
        cost: Decimal,
        up_prob: Decimal,
        dn_prob: Decimal,
        elapsed_s: int,
        condition_id: str = "",
        window_ts: int = 0,
        market_slug: str = "",
    ):
        self._trades.append({
            "time": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "side": side,
            "price": float(price),
            "shares": float(shares),
            "cost_usd": float(cost),
            "up_prob_pct": float(up_prob * 100),
            "dn_prob_pct": float(dn_prob * 100),
            "window_elapsed_s": elapsed_s,
            "condition_id": condition_id,
            "window_ts": window_ts,
            "market_slug": market_slug,
            # Filled in after settlement:
            "outcome": None,      # "WIN" | "LOSS" | "PUSH" | "PENDING"
            "payout_usd": None,
            "pnl_usd": None,
        })

    def settle_window(self, window_ts: int, gamma_api: str) -> bool:
        """
        Attempts to settle all trades in window_ts.

        Returns True  if every trade resolved to WIN or LOSS (fully done).
        Returns False if any trade is still PENDING (caller should retry).

        The window is only added to _settled_windows once fully resolved,
        so this method is safe to call multiple times until it returns True.

        outcomePrices = ["1.0", "0.0"] means UP won, DOWN lost.
        outcomePrices = ["0.0", "1.0"] means DOWN won, UP lost.
        Prices between 0.01-0.99 mean the market has not resolved yet.
        """
        # Already fully resolved — nothing to do
        if window_ts in self._settled_windows:
            return True

        # Collect trades that still need a final outcome
        pending = [
            t for t in self._trades
            if t["window_ts"] == window_ts
            and t.get("outcome") not in ("WIN", "LOSS")
        ]
        if not pending:
            self._settled_windows.add(window_ts)
            return True

        slug = f"btc-updown-5m-{window_ts}"
        url  = f"{gamma_api}/markets?slug={slug}"

        try:
            resp = requests.get(url, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            markets = data if isinstance(data, list) else [data]
            if not markets:
                raise ValueError("Empty response")
            m = markets[0]
            raw_prices = m.get("outcomePrices") or "[]"
            import json as _json
            prices = _json.loads(raw_prices) if isinstance(raw_prices, str) else raw_prices
            up_price = float(prices[0]) if len(prices) > 0 else None
            dn_price = float(prices[1]) if len(prices) > 1 else None
        except Exception as exc:
            logger.warning(f"Settlement fetch failed for {slug}: {exc}")
            return False   # transient error — let caller retry

        all_resolved = True
        for t in pending:
            side      = t["side"]
            shares    = t["shares"]
            cost      = t["cost_usd"]
            win_price = up_price if side == "UP" else dn_price

            if win_price is None:
                t["outcome"] = "PENDING"
                all_resolved = False
                continue

            if win_price >= 0.99:
                payout       = round(shares * 1.0, 2)
                t["outcome"] = "WIN"
            elif win_price <= 0.01:
                payout       = 0.0
                t["outcome"] = "LOSS"
            else:
                # Mid-market — not yet resolved; store best estimate but keep retrying
                payout       = round(shares * win_price, 2)
                t["outcome"] = "PENDING"
                all_resolved = False

            t["payout_usd"] = payout
            t["pnl_usd"]    = round(payout - cost, 2)

            sign = "+" if t["pnl_usd"] >= 0 else ""
            logger.info(
                f"  {'✅' if t['outcome'] == 'WIN' else '❌' if t['outcome'] == 'LOSS' else '⏳'} "
                f"SETTLEMENT [{slug}]  "
                f"{side}  outcome={t['outcome']}  "
                f"payout=${payout:.2f}  P&L={sign}${t['pnl_usd']:.2f}"
            )

        if all_resolved:
            self._settled_windows.add(window_ts)

        return all_resolved

    def print_window_report(self, window_ts: int):
        """
        Compact inline report printed immediately when a window closes.
        Shows only trades from that specific window + their P&L.
        """
        trades = [t for t in self._trades if t.get("window_ts") == window_ts]
        if not trades:
            logger.info(f"  No trades were placed in window {window_ts}.")
            return

        sep = "─" * 60
        print(f"\n{sep}")
        print(f"  📋 WINDOW REPORT  {window_ts}")
        print(sep)
        print(f"  {'Side':<6} {'Entry':>6} {'Shares':>6} {'Cost':>7}  "
              f"{'Conf':>6}  {'El':>5}  {'Result':>7}  {'P&L':>8}")
        print(f"  {sep}")

        window_pnl = 0.0
        for t in trades:
            conf    = max(t["up_prob_pct"], t["dn_prob_pct"])
            el_s    = t["window_elapsed_s"]
            el_fmt  = f"{el_s//60}:{el_s%60:02d}"
            outcome = t.get("outcome") or "PENDING"
            pnl     = t.get("pnl_usd")

            if pnl is not None:
                window_pnl += pnl
                pnl_str   = f"{'+' if pnl >= 0 else ''}${pnl:.2f}"
            else:
                pnl_str   = "  ——"

            if outcome == "WIN":
                tag = "✓ WIN"
            elif outcome == "LOSS":
                tag = "✗ LOSS"
            else:
                tag = "? PEND"

            print(f"  {t['side']:<6} {t['price']:>6.4f} {t['shares']:>6.1f} "
                  f"${t['cost_usd']:>5.2f}  {conf:>5.1f}%  {el_fmt:>5}  "
                  f"{tag:>7}  {pnl_str:>8}")

        pnl_sign = "+" if window_pnl >= 0 else ""
        status   = "settled" if any(t.get("pnl_usd") is not None for t in trades) else "pending"
        print(f"  {sep}")
        print(f"  Window P&L: {pnl_sign}${window_pnl:.2f}  ({status})")
        print(f"  Now scanning next window →\n")

    def print_report(self):
        import csv as _csv
        sep  = "=" * 70
        thin = "-" * 70
        session_end = datetime.now(timezone.utc)
        duration    = session_end - self._session_start

        # ── Header ──────────────────────────────────────────────────────────
        print(f"\n{sep}")
        print(f"  {'':1}SANDBOX SESSION REPORT  —  PolyBTC Bot  v2")
        print(sep)
        print(f"  Session start : {self._session_start.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"  Session end   : {session_end.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"  Duration      : {str(duration).split('.')[0]}")
        print(f"  Total trades  : {len(self._trades)}")

        if not self._trades:
            print(f"\n  No trades were simulated this session.")
            print(f"  Possible reasons:")
            print(f"    • No market reached {Config.CONFIDENCE_THRESHOLD:.0%} confidence")
            print(f"    • Time gate ({Config.MIN_ELAPSED_SECONDS}s) was not yet open")
            print(f"    • Spread exceeded {Config.MAX_SPREAD_ALLOWED:.0%} limit")
            print(f"{sep}\n")
            return

        total_cost  = sum(Decimal(str(t["cost_usd"])) for t in self._trades)
        up_trades   = [t for t in self._trades if t["side"] == "UP"]
        dn_trades   = [t for t in self._trades if t["side"] == "DOWN"]
        avg_conf    = sum(
            max(t["up_prob_pct"], t["dn_prob_pct"]) for t in self._trades
        ) / len(self._trades)
        avg_elapsed = sum(t["window_elapsed_s"] for t in self._trades) / len(self._trades)
        avg_price   = sum(t["price"] for t in self._trades) / len(self._trades)

        settled   = [t for t in self._trades if t.get("pnl_usd") is not None]
        unsettled = [t for t in self._trades if t.get("pnl_usd") is None]
        total_pnl = sum(t["pnl_usd"] for t in settled) if settled else 0.0
        wins      = sum(1 for t in settled if t.get("outcome") == "WIN")
        losses    = sum(1 for t in settled if t.get("outcome") == "LOSS")

        # ── Summary ─────────────────────────────────────────────────────────
        print(thin)
        print("  SUMMARY")
        print(thin)
        print(f"  Total simulated spend  : ${total_cost:.2f}")
        print(f"  UP  bets               : {len(up_trades)}")
        print(f"  DOWN bets              : {len(dn_trades)}")
        print(f"  Avg entry price        : {avg_price:.4f}  "
              f"(implied prob ≈ {avg_price * 100:.1f}%)")
        print(f"  Avg confidence         : {avg_conf:.1f}%")
        print(f"  Avg window elapsed     : {avg_elapsed:.0f}s  "
              f"({int(avg_elapsed)//60}:{int(avg_elapsed)%60:02d} into window)")
        print(f"  Avg $ per trade        : ${float(total_cost) / len(self._trades):.2f}")

        # ── P&L Summary ─────────────────────────────────────────────────────
        print(f"\n{thin}")
        print("  P&L SUMMARY (settled trades)")
        print(thin)
        if settled:
            pnl_sign = "+" if total_pnl >= 0 else ""
            roi = total_pnl / float(total_cost) * 100 if total_cost else 0.0
            roi_sign = "+" if roi >= 0 else ""
            print(f"  Settled trades         : {len(settled)}  ({wins} WIN / {losses} LOSS)")
            print(f"  Total P&L              : {pnl_sign}${total_pnl:.2f}")
            print(f"  ROI                    : {roi_sign}{roi:.1f}%")
            print(f"  Win rate               : {wins/len(settled)*100:.0f}%")
        else:
            print(f"  Settled trades         : 0  (0 WIN / 0 LOSS)")
            print(f"  Total P&L              : $0.00")
            print(f"  ROI                    : 0.0%")
            print(f"  Win rate               : 0%")
        if unsettled:
            print(f"  Pending (unresolved)   : {len(unsettled)} trade(s)")

        # ── Trade Log ───────────────────────────────────────────────────────
        print(f"\n{thin}")
        print("  TRADE LOG")
        print(thin)
        print(
            f"  {'#':<4} {'Time':<8} {'Side':<5} {'Entry':>6} {'Shares':>6} "
            f"{'Cost':>6}  {'Conf':>5}  {'El':>5}  {'Result':>7}  {'P&L':>8}"
        )
        print(f"  {thin}")

        running_pnl = 0.0
        for i, t in enumerate(self._trades, 1):
            ts      = t["time"][11:19]
            conf    = max(t["up_prob_pct"], t["dn_prob_pct"])
            el_s    = t["window_elapsed_s"]
            el_fmt  = f"{el_s//60}:{el_s%60:02d}"
            outcome = t.get("outcome") or "PENDING"
            pnl     = t.get("pnl_usd")

            if pnl is not None:
                running_pnl += pnl
                pnl_str = f"{'+' if pnl >= 0 else ''}${pnl:.2f}"
            else:
                pnl_str = "——"

            if outcome == "WIN":
                result_tag = "WIN"
            elif outcome == "LOSS":
                result_tag = "LOSS"
            else:
                result_tag = "PENDING"

            print(
                f"  {i:<4} {ts:<8} {t['side']:<5} "
                f"{t['price']:>6.4f} {t['shares']:>6.1f} "
                f"${t['cost_usd']:>5.2f}  "
                f"{conf:>4.1f}%  {el_fmt:>5}  "
                f"{result_tag:>7}  {pnl_str:>8}"
            )

        print(f"\n{thin}")
        print("  NOTE: All figures are SIMULATED. No real money was spent.")
        print(f"  To go live: set Config.SANDBOX_MODE = False")
        print(f"{sep}\n")

        # ── CSV Export ──────────────────────────────────────────────────────
        self._export_csv(session_end)

    def _export_csv(self, session_end: datetime):
        """
        Appends this session's trades to sessions_all.csv (created if absent).

        File layout mirrors the visual table:
          - Column headers on row 1 (written once, on first session)
          - One session-header row per session (grey separator in spreadsheet)
          - One trade row per trade

        Columns: # | Time (UTC) | Market Window | Side | Market Title |
                 Entry Price | Shares | Cost ($) | Confidence (%) |
                 Elapsed | Outcome | P&L ($)
        """
        import csv as _csv
        import os as _os

        filename  = "sessions_all.csv"
        is_new    = not _os.path.exists(filename)

        total_cost = sum(t["cost_usd"] for t in self._trades)
        settled    = [t for t in self._trades if t.get("pnl_usd") is not None]
        unsettled  = [t for t in self._trades if t.get("pnl_usd") is None]
        total_pnl  = sum(t["pnl_usd"] for t in settled) if settled else 0.0
        wins       = sum(1 for t in settled if t.get("outcome") == "WIN")
        losses     = sum(1 for t in settled if t.get("outcome") == "LOSS")
        roi        = (total_pnl / total_cost * 100) if total_cost else 0.0

        session_start_str = self._session_start.strftime("%Y-%m-%d %H:%M:%S UTC")
        session_end_str   = session_end.strftime("%Y-%m-%d %H:%M:%S UTC")
        duration          = str(session_end - self._session_start).split(".")[0]

        # Build session label identical to the widget header row
        roi_str = f"{'+' if roi >= 0 else ''}{roi:.1f}%"
        session_label = (
            f"Session {session_start_str}  ·  {len(self._trades)} trades  ·  "
            f"ROI {roi_str}  ·  {wins}W / {losses}L  ·  "
            f"{duration}  ·  total cost ${total_cost:.2f}  ·  "
            f"P&L {('+' if total_pnl >= 0 else '')}${total_pnl:.2f}  ·  "
            f"end {session_end_str}"
        )

        HEADERS = [
            "#", "Time (UTC)", "Market Window", "Side", "Market Title",
            "Entry Price", "Shares", "Cost ($)", "Confidence (%)",
            "Elapsed", "Outcome", "P&L ($)",
        ]

        with open(filename, "a", newline="", encoding="utf-8") as fh:
            w = _csv.writer(fh)

            # Write column headers only on the very first write
            if is_new:
                w.writerow(HEADERS)

            # Session separator row (all columns collapsed into first cell)
            w.writerow([session_label] + [""] * (len(HEADERS) - 1))

            # One row per trade
            for i, t in enumerate(self._trades, 1):
                ts      = t["time"][11:19]
                conf    = max(t["up_prob_pct"], t["dn_prob_pct"])
                el_s    = t["window_elapsed_s"]
                el_fmt  = f"{el_s//60}:{el_s%60:02d}"
                outcome = t.get("outcome") or "PENDING"
                pnl     = t.get("pnl_usd")
                pnl_val = f"{pnl:.2f}" if pnl is not None else ""

                # Derive short window label from market slug e.g. "1:00-1:05AM"
                slug    = t.get("market_slug", "")
                title   = t.get("question", "")

                w.writerow([
                    i,
                    ts,
                    slug,
                    t["side"],
                    title,
                    f"{t['price']:.4f}",
                    f"{t['shares']:.1f}",
                    f"{t['cost_usd']:.2f}",
                    f"{conf:.1f}",
                    el_fmt,
                    outcome,
                    pnl_val,
                ])

        logger.info(f"  📄 CSV updated → {filename}  [{len(self._trades)} trade(s)]")

    def save_session_json(self):
        """
        Appends a full session summary to live_sessions.json.

        Each entry is keyed by session_start timestamp so the file grows
        across runs — a permanent record of every bot session.
        Includes config snapshot, P&L summary, and full trade list.
        """
        session_end = datetime.now(timezone.utc)
        duration_s  = int((session_end - self._session_start).total_seconds())
        session_key = self._session_start.strftime("%Y-%m-%dT%H:%M:%SZ")

        settled   = [t for t in self._trades if t.get("pnl_usd") is not None]
        unsettled = [t for t in self._trades if t.get("pnl_usd") is None]
        wins      = sum(1 for t in settled if t.get("outcome") == "WIN")
        losses    = sum(1 for t in settled if t.get("outcome") == "LOSS")
        total_pnl = round(sum(t["pnl_usd"] for t in settled), 2) if settled else None
        total_cost = round(sum(t["cost_usd"] for t in self._trades), 2)

        record = {
            "session_start":    session_key,
            "session_end":      session_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "duration_seconds": duration_s,
            "sandbox":          Config.SANDBOX_MODE,
            "config": {
                "confidence_threshold": float(Config.CONFIDENCE_THRESHOLD),
                "max_entry_price":      float(Config.MAX_ENTRY_PRICE),
                "min_elapsed_seconds":  Config.MIN_ELAPSED_SECONDS,
                "max_usd_per_trade":    float(Config.MAX_USD_PER_TRADE),
                "max_spread":           float(Config.MAX_SPREAD_ALLOWED),
            },
            "summary": {
                "total_trades":   len(self._trades),
                "total_cost_usd": total_cost,
                "settled":        len(settled),
                "unsettled":      len(unsettled),
                "wins":           wins,
                "losses":         losses,
                "win_rate_pct":   round(wins / len(settled) * 100, 1) if settled else None,
                "total_pnl_usd":  total_pnl,
                "roi_pct":        round(total_pnl / total_cost * 100, 1)
                                  if (total_pnl is not None and total_cost > 0) else None,
            },
            "trades": [
                {
                    "time":             t["time"],
                    "market_title":     t["question"],
                    "market_slug":      t.get("market_slug"),
                    "side":             t["side"],
                    "entry_price":      t["price"],
                    "shares":           t["shares"],
                    "cost_usd":         t["cost_usd"],
                    "confidence_pct":   round(max(t["up_prob_pct"], t["dn_prob_pct"]), 1),
                    "window_elapsed_s": t["window_elapsed_s"],
                    "outcome":          t.get("outcome"),
                    "max_payout_usd":   round(t["shares"] * 1.0, 2),
                    "amount_gained_if_win": round(t["shares"] * 1.0 - t["cost_usd"], 2),
                    "payout_usd":       t.get("payout_usd"),
                    "pnl_usd":          t.get("pnl_usd"),
                    "amount_gained":    t.get("pnl_usd"),
                    "window_ts":        t.get("window_ts"),
                }
                for t in self._trades
            ],
        }

        # Load existing file or start fresh
        sessions = {}
        if os.path.exists(Config.SESSION_FILE):
            try:
                with open(Config.SESSION_FILE, "r") as fh:
                    sessions = json.load(fh)
            except Exception:
                sessions = {}

        sessions[session_key] = record

        with open(Config.SESSION_FILE, "w") as fh:
            json.dump(sessions, fh, indent=2)

        logger.info(
            f"Session saved → {Config.SESSION_FILE}  "
            f"[{len(self._trades)} trade(s)  "
            f"P&L: {'N/A' if total_pnl is None else f'+${total_pnl:.2f}' if total_pnl >= 0 else f'-${abs(total_pnl):.2f}'}]"
        )


# ---------------------------------------------------------------------------
# Window tracker — records every 5-min window, traded or not
# ---------------------------------------------------------------------------

class WindowTracker:
    """
    Writes one record per 5-min window to windows.json, regardless of
    whether a trade was placed. Lets you analyse:
      - How often the gate opened but no signal reached 70%
      - Which windows had signals that were above the cap
      - Overall market activity across sessions
    """

    def __init__(self):
        self._windows: dict = {}   # window_ts → record (updated live)

    def open_window(self, window_ts: int):
        """Called when a new window is first seen."""
        if window_ts not in self._windows:
            self._windows[window_ts] = {
                "window_ts":        window_ts,
                "window_start_utc": datetime.fromtimestamp(window_ts, tz=timezone.utc).isoformat(),
                "market_title":     None,   # e.g. "Bitcoin Up or Down - March 14, 5:00AM-5:05AM ET"
                "market_slug":      f"btc-updown-5m-{window_ts}",
                "scans":            0,
                "gate_open_scans":  0,
                "no_signal_scans":  0,
                "above_cap_scans":  0,
                "traded":           False,
                "skip_reason":      None,
                "best_signal_pct":  None,
                "trade":            None,
            }

    def record_scan(self, window_ts: int, gate_open: bool, signal_pct: float | None,
                    skip_reason: str | None, market_title: str | None = None):
        """Called after each price evaluation for this window."""
        if window_ts not in self._windows:
            self.open_window(window_ts)
        w = self._windows[window_ts]
        if market_title and not w["market_title"]:
            w["market_title"] = market_title
        w["scans"] += 1
        if gate_open:
            w["gate_open_scans"] += 1
        if signal_pct is not None:
            prev = w["best_signal_pct"] or 0
            w["best_signal_pct"] = round(max(prev, signal_pct), 1)
        if skip_reason == "no_signal":
            w["no_signal_scans"] += 1
        elif skip_reason == "above_cap":
            w["above_cap_scans"] += 1
        if skip_reason:
            w["skip_reason"] = skip_reason

    def record_trade(self, window_ts: int, side: str, price: float,
                     shares: float, cost: float, market_title: str | None = None):
        """Called when a trade fires in this window."""
        if window_ts not in self._windows:
            self.open_window(window_ts)
        w = self._windows[window_ts]
        w["traded"]      = True
        w["skip_reason"] = None
        if market_title:
            w["market_title"] = market_title
        w["trade"] = {
            "timestamp":           datetime.now(timezone.utc).isoformat(),
            "market_title":        market_title,
            "market_slug":         f"btc-updown-5m-{window_ts}",
            "side":                side,
            "entry_price":         price,
            "shares":              shares,
            "cost_usd":            cost,
            "amount_gained_if_win": round(shares - cost, 2),
        }

    def save_window(self, window_ts: int):
        """Appends/updates this window's record in windows.json."""
        if window_ts not in self._windows:
            return
        record = self._windows[window_ts]

        existing = {}
        if os.path.exists(Config.WINDOWS_FILE):
            try:
                with open(Config.WINDOWS_FILE, "r") as f:
                    loaded = json.load(f)
                # Guard: file must be a dict — reset if it got corrupted to a list
                existing = loaded if isinstance(loaded, dict) else {}
            except Exception:
                existing = {}

        existing[str(window_ts)] = record

        with open(Config.WINDOWS_FILE, "w") as f:
            json.dump(existing, f, indent=2)



# ---------------------------------------------------------------------------
# Auto-Claimer — on-chain redemption of winning positions (LIVE mode only)
# ---------------------------------------------------------------------------

# Minimal ABI for the Polymarket CTF redeemPositions function.
_CTF_ABI = [
    {
        "inputs": [
            {"internalType": "address",   "name": "collateralToken", "type": "address"},
            {"internalType": "bytes32",   "name": "parentCollectionId", "type": "bytes32"},
            {"internalType": "bytes32",   "name": "conditionId",        "type": "bytes32"},
            {"internalType": "uint256[]", "name": "indexSets",          "type": "uint256[]"},
        ],
        "name": "redeemPositions",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
]

# USDC on Polygon mainnet
_USDC_POLYGON = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"


class AutoClaimer:
    """
    Calls redeemPositions() on the Polymarket CTF Exchange contract after a
    market resolves to WIN.

    Polymarket binary markets have two outcomes (index sets 1 and 2):
      index set 1 = outcome[0] (UP)
      index set 2 = outcome[1] (DOWN)

    redeemPositions burns your winning shares and sends USDC back to your wallet.
    Losing positions pay out nothing — no point claiming them.

    Usage:
        claimer = AutoClaimer(private_key="0x...")
        await claimer.claim(condition_id="0xabc...", side="UP")
    """

    def __init__(self, private_key: str):
        if not _WEB3_AVAILABLE:
            raise ImportError(
                "web3 is required for auto-claim. Install it with:\n"
                "  pip install web3"
            )
        self._pk = private_key
        self._w3 = Web3(Web3.HTTPProvider(Config.POLYGON_RPC))
        # Polygon uses PoA — this middleware fixes extraData length issues
        self._w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        self._account = self._w3.eth.account.from_key(private_key)
        self._ctf = self._w3.eth.contract(
            address=Web3.to_checksum_address(Config.CTF_ADDRESS),
            abi=_CTF_ABI,
        )
        logger.info(
            f"AutoClaimer ready | wallet={self._account.address[:10]}… | "
            f"RPC={Config.POLYGON_RPC}"
        )

    async def claim(self, condition_id: str, side: str) -> bool:
        """
        Calls redeemPositions for the winning side of a resolved market.

        condition_id : hex string e.g. "0xabc123..."
        side         : "UP" or "DOWN"

        Returns True if the tx confirmed, False on any error.
        """
        # index set 1 = UP (outcome[0]), index set 2 = DOWN (outcome[1])
        index_set = 1 if side.upper() == "UP" else 2

        # condition_id must be bytes32
        try:
            cid_bytes = bytes.fromhex(condition_id.removeprefix("0x"))
            if len(cid_bytes) != 32:
                raise ValueError(f"condition_id must be 32 bytes, got {len(cid_bytes)}")
        except Exception as exc:
            logger.error(f"AutoClaimer: invalid condition_id {condition_id!r}: {exc}")
            return False

        loop = asyncio.get_event_loop()
        try:
            success = await loop.run_in_executor(
                None,
                lambda: self._send_redeem(cid_bytes, index_set),
            )
            return success
        except Exception as exc:
            logger.error(f"AutoClaimer: unexpected error: {exc}")
            return False

    def _send_redeem(self, cid_bytes: bytes, index_set: int) -> bool:
        """Builds, signs, and broadcasts the redeemPositions transaction."""
        try:
            nonce = self._w3.eth.get_transaction_count(self._account.address)
            gas_price = self._w3.eth.gas_price

            tx = self._ctf.functions.redeemPositions(
                Web3.to_checksum_address(_USDC_POLYGON),   # collateral token
                b"\x00" * 32,                              # parentCollectionId = 0 (root)
                cid_bytes,                                  # conditionId
                [index_set],                                # winning index set only
            ).build_transaction({
                "from":     self._account.address,
                "nonce":    nonce,
                "gas":      200_000,       # generous limit; actual use is ~80k
                "gasPrice": gas_price,
                "chainId":  Config.CHAIN_ID,
            })

            signed = self._w3.eth.account.sign_transaction(tx, self._pk)
            tx_hash = self._w3.eth.send_raw_transaction(signed.rawTransaction)
            receipt = self._w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

            if receipt["status"] == 1:
                logger.info(
                    f"  💰 CLAIMED  index_set={index_set}  "
                    f"tx={tx_hash.hex()[:20]}…  "
                    f"gas={receipt['gasUsed']}"
                )
                return True
            else:
                logger.error(
                    f"  ❌ CLAIM REVERTED  tx={tx_hash.hex()[:20]}…"
                )
                return False

        except Exception as exc:
            logger.error(f"  💥 Claim tx error: {exc}")
            return False

# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class PolymarketEngine:

    def __init__(self):
        self.sandbox        = Config.SANDBOX_MODE
        self.state          = State()
        self.ledger         = SandboxLedger()
        self.window_tracker = WindowTracker()

        if self.sandbox:
            logger.info(
                "🟡 SANDBOX MODE — trades are SIMULATED. No real money spent."
            )
            self.client = ClobClient(
                host=Config.CLOB_HOST,
                key=Config.PK or "0x" + "0" * 64,  # dummy key for read-only calls
                chain_id=Config.CHAIN_ID,
            )
        else:
            if not Config.PK:
                raise EnvironmentError(
                    "SANDBOX_MODE is False but POLYGON_PRIVATE_KEY is not set."
                )
            logger.warning(
                "🔴 LIVE MODE — REAL MONEY WILL BE SPENT."
            )
            self.client = ClobClient(
                host=Config.CLOB_HOST,
                key=Config.PK,
                chain_id=Config.CHAIN_ID,
            )

        self._authenticated = False

        # Auto-claimer — only active in LIVE mode with AUTO_CLAIM enabled
        self.claimer: AutoClaimer | None = None
        if not self.sandbox and Config.AUTO_CLAIM:
            if not _WEB3_AVAILABLE:
                logger.warning(
                    "AUTO_CLAIM is True but web3 is not installed. "
                    "Run:  pip install web3   — claiming disabled for this session."
                )
            else:
                try:
                    self.claimer = AutoClaimer(Config.PK)
                    logger.info("✅ Auto-claim ENABLED — winnings claimed automatically.")
                except Exception as exc:
                    logger.warning(f"AutoClaimer init failed: {exc} — claiming disabled.")

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    async def _ensure_auth(self):
        if self.sandbox:
            logger.info("Sandbox: skipping on-chain auth (read-only).")
            return
        if self._authenticated:
            return
        loop = asyncio.get_event_loop()
        # create_or_derive_api_creds() handles both first-run and returning users.
        creds = await loop.run_in_executor(
            None, self.client.create_or_derive_api_creds
        )
        self.client.set_api_creds(creds)
        self._authenticated = True
        logger.info("Auth complete.")

    # ------------------------------------------------------------------
    # Market discovery — deterministic slug from the clock
    #
    # Polymarket slugs are NOT searchable via tag. Instead they follow a
    # fixed pattern:  btc-updown-5m-{unix_window_start}
    #
    # The window_start is the Unix epoch (seconds) of the most recent
    # 5-min boundary, i.e.  int(time.time()) // 300 * 300
    #
    # We also try the NEXT window (+300s) in case Polymarket opens it
    # slightly early, and the PREVIOUS window (-300s) as a fallback.
    # The events endpoint returns the market's token IDs directly.
    # ------------------------------------------------------------------

    @staticmethod
    def _current_window_timestamps() -> list[int]:
        """Return [prev_window, current_window, next_window] Unix timestamps."""
        import time
        now_ts   = int(time.time())
        current  = (now_ts // 300) * 300
        return [current - 300, current, current + 300]

    def _fetch_btc_5min_markets(self) -> list[dict]:
        """
        Constructs slugs deterministically from the clock and fetches
        the event (which contains the Up/Down market tokens) directly.

        Falls back to a broad search if the slug lookup returns nothing,
        e.g. if Polymarket ever changes their naming convention.
        """
        markets_found = []

        for window_ts in self._current_window_timestamps():
            slug = f"btc-updown-5m-{window_ts}"
            url  = f"{Config.GAMMA_API}/events?slug={slug}"
            logger.info(f"Trying slug: {slug}")
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                data = resp.json()

                # /events returns a list; each event has a 'markets' array
                events = data if isinstance(data, list) else [data]
                for event in events:
                    if not event:
                        continue
                    # Flatten: each event can contain multiple markets
                    sub_markets = event.get("markets", [])
                    if sub_markets:
                        for m in sub_markets:
                            # Attach window timestamp so we can use it later
                            m["_window_ts"] = window_ts
                            markets_found.append(m)
                    else:
                        # Some responses return the market fields at the top level
                        event["_window_ts"] = window_ts
                        markets_found.append(event)

            except Exception as exc:
                logger.debug(f"Slug {slug} not found or error: {exc}")

        if markets_found:
            logger.info(
                f"Found {len(markets_found)} BTC 5-min market(s) via slug lookup."
            )
            return markets_found

        # ── Fallback: broad search ────────────────────────────────────
        logger.warning(
            "Slug lookup returned nothing — falling back to broad search."
        )
        url = f"{Config.GAMMA_API}/markets?active=true&closed=false&limit=50"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            all_markets = resp.json()
        except Exception as exc:
            logger.error(f"Fallback search error: {exc}")
            return []

        filtered = []
        for m in all_markets:
            q = (m.get("question") or m.get("title") or "").lower()
            if (
                ("5 min" in q or "5-min" in q or "5min" in q)
                and ("up" in q or "down" in q)
                and ("bitcoin" in q or "btc" in q)
            ):
                filtered.append(m)

        logger.info(f"Fallback found {len(filtered)} candidate market(s).")
        return filtered

    # ------------------------------------------------------------------
    # Signal helper
    # ------------------------------------------------------------------

    @staticmethod
    def _implied_prob(ask_price: Decimal) -> Decimal:
        """
        On Polymarket, the price IS the probability.

        A price of 0.53 means the market assigns a 53% chance of that
        outcome winning — you pay 53¢ to win $1.00 if correct.

        Screenshot example (March 13 session):
          UP   ask = 0.53  →  UP   prob = 53%  → skip (< 70%)
          DOWN ask = 0.48  →  DOWN prob = 48%  → skip (< 70%)

        Signal fires when:
          UP   ask >= 0.70  (market says 70%+ chance BTC goes UP)
          DOWN ask >= 0.70  (market says 70%+ chance BTC goes DOWN)
        """
        return ask_price

    # ------------------------------------------------------------------
    # Per-market analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_tokens(market: dict) -> list[dict]:
        """
        Normalises token extraction across multiple schemas returned by
        the Gamma /events and /markets endpoints.

        Schema A — /markets endpoint:
            market["tokens"] = [{"token_id": "21345", "outcome": "Up"}, ...]

        Schema B — /events > markets[], clobTokenIds as real list:
            market["clobTokenIds"] = ["21345", "67890"]
            market["outcomes"]     = ["Up", "Down"]

        Schema C — /events > markets[], clobTokenIds as JSON string:
            market["clobTokenIds"] = "[\"21345\", \"67890\"]"
            market["outcomes"]     = "[\"Up\", \"Down\"]"
            (Gamma sometimes double-encodes these as strings)
        """
        import json as _json

        def _unwrap(val):
            """If val is a JSON-encoded string, decode it; else return as-is."""
            if isinstance(val, str):
                try:
                    decoded = _json.loads(val)
                    return decoded if isinstance(decoded, list) else [val]
                except Exception:
                    return [val]
            return val  # already a list or None

        # Schema A
        tokens = market.get("tokens") or []
        if tokens and isinstance(tokens[0], dict) and "token_id" in tokens[0]:
            return tokens

        # Schema B / C
        clob_ids = _unwrap(
            market.get("clobTokenIds")
            or market.get("clob_token_ids")
            or []
        )
        outcomes = _unwrap(
            market.get("outcomes") or ["Up", "Down"]
        )

        if clob_ids and len(clob_ids) >= 2:
            return [
                {"token_id": str(tid).strip(), "outcome": str(out).strip()}
                for tid, out in zip(clob_ids, outcomes)
            ]

        logger.warning(
            f"_extract_tokens: could not find token IDs. "
            f"Market keys: {list(market.keys())} | "
            f"clobTokenIds raw: {repr(market.get('clobTokenIds'))[:80]}"
        )
        return []

    # Tracks whether we've logged a raw market dump this session
    _debug_dumped: bool = False

    async def _analyze_and_trade(self, market: dict):
        condition_id = (
            market.get("condition_id")
            or market.get("conditionId")
            or ""
        )
        question = (
            market.get("question")
            or market.get("title")
            or "unknown"
        )
        tokens = self._extract_tokens(market)

        # One-time debug dump so we can see the raw API shape
        if not PolymarketEngine._debug_dumped:
            import json as _json
            safe = {k: repr(v)[:80] for k, v in market.items()}
            logger.info(f"[DEBUG] Raw market fields: {_json.dumps(safe, indent=2)}")
            PolymarketEngine._debug_dumped = True

        if not condition_id or len(tokens) < 2:
            logger.warning(
                f"Skipping — missing condition_id or <2 tokens. "
                f"Keys present: {list(market.keys())} | "
                f"tokens extracted: {tokens}"
            )
            return

        # Skip markets that are already closed or ended — these produce 404s
        if market.get("closed") is True or market.get("active") is False:
            logger.info(f"Market is closed/inactive — skip. ({question[:40]})")
            return
        end_date_str = market.get("endDate") or market.get("endDateIso") or ""
        if end_date_str:
            try:
                from dateutil.parser import parse as _parse_dt
                end_dt = _parse_dt(end_date_str)
                if end_dt.tzinfo is None:
                    end_dt = end_dt.replace(tzinfo=timezone.utc)
                if end_dt < datetime.now(timezone.utc):
                    logger.info(f"Market end date passed ({end_date_str[:19]}) — skip.")
                    return
            except Exception:
                pass  # dateutil not available; skip the check

        now = datetime.now(timezone.utc)

        # 1. Time gate
        elapsed    = seconds_elapsed_in_window(now)
        window_ts  = market.get("_window_ts") or (int(now.timestamp()) // 300 * 300)
        self.window_tracker.open_window(window_ts)

        if not is_past_time_gate(now):
            wait = seconds_until_gate(now)
            logger.info(
                f"⏳ Gate CLOSED [{elapsed}s/{Config.MIN_ELAPSED_SECONDS}s] "
                f"opens in {wait}s | {question[:40]}"
            )
            self.window_tracker.record_scan(window_ts, gate_open=False,
                                             signal_pct=None, skip_reason="gate_closed",
                                             market_title=question)
            return

        logger.info(
            f"✅ Gate OPEN [{elapsed}s = {elapsed//60}:{elapsed%60:02d}] "
            f"| {question[:40]}"
        )

        # 2. Duplicate guard
        if self.state.is_traded(condition_id, now):
            logger.info("Already traded this window — skip.")
            return

        # 3. Fetch fresh prices directly from Gamma /markets API.
        #
        #    The market dict passed here was fetched during discovery and may be
        #    up to 15s stale. We re-fetch the single market by condition_id to
        #    get the most current outcomePrices, bestBid, and bestAsk.
        #
        #    CLOB is abandoned: get_order_book() returns phantom 0.99/0.99 data
        #    on these markets consistently. Gamma prices match the UI Buy buttons
        #    and update frequently enough for our 15s scan interval.
        import json as _json
        up_token_id = tokens[0]["token_id"]
        dn_token_id = tokens[1]["token_id"]

        # Re-fetch this market fresh using conditionId as the query key.
        # The market 'id' field is an event ID — useless for /markets lookup.
        # conditionId is the stable per-market identifier that works reliably.
        # URL: /markets?condition_id=0xabc...  returns a list with one market.
        loop = asyncio.get_event_loop()
        up_ask = dn_ask = None

        cid = (
            market.get("conditionId")
            or market.get("condition_id")
            or condition_id
        )

        def _parse_prices(fm: dict):
            """Extract (up, dn) Decimals from a market dict. Returns None if zero or missing."""
            raw = fm.get("outcomePrices") or "[]"
            try:
                op = _json.loads(raw) if isinstance(raw, str) else raw
                if len(op) >= 2:
                    u, d = Decimal(str(op[0])), Decimal(str(op[1]))
                    # Reject zeros — Gamma returns 0/0 for wrong market or settling state
                    if u > 0 and d > 0:
                        return u, d
            except Exception:
                pass
            return None

        # Attempt 0: get_midpoint() — live CLOB mid-market price (fastest, most accurate)
        # Hits /midpoint endpoint, NOT the broken /book endpoint.
        # Returns {"mid": "0.72"} matching exactly what the UI Buy button shows.
        # Falls through silently if unavailable (404 or ghost data).
        async def _get_mid(token_id: str):
            try:
                raw = await loop.run_in_executor(
                    None, self.client.get_midpoint, token_id
                )
                val = raw.get("mid") if isinstance(raw, dict) else str(raw)
                d = Decimal(str(val))
                return d if d > 0 else None
            except Exception:
                return None

        up_mid, dn_mid = await asyncio.gather(
            _get_mid(up_token_id),
            _get_mid(dn_token_id),
        )

        if (up_mid is not None and dn_mid is not None
                and (up_mid + dn_mid) <= Decimal("1.15")
                and (up_mid + dn_mid) >= Decimal("0.85")):
            up_ask, dn_ask = up_mid, dn_mid
            logger.info(f"  [midpoint]  UP={up_ask} DN={dn_ask}")

        # Attempt 1: fresh fetch by conditionId
        if cid:
            try:
                r = await loop.run_in_executor(
                    None,
                    lambda u=f"{Config.GAMMA_API}/markets?condition_id={cid}": requests.get(u, timeout=5)
                )
                if r.status_code == 200:
                    data = r.json()
                    fm = data[0] if isinstance(data, list) and data else (data if isinstance(data, dict) else {})
                    result = _parse_prices(fm)
                    if result:
                        up_ask, dn_ask = result
                        logger.info(f"  [Gamma cid]  UP={up_ask} DN={dn_ask}")
            except Exception as exc:
                logger.debug(f"cid fetch error: {exc}")

        # Attempt 2: fresh fetch by UP token_id
        if (up_ask is None or dn_ask is None) and up_token_id:
            try:
                r = await loop.run_in_executor(
                    None,
                    lambda u=f"{Config.GAMMA_API}/markets?clob_token_id={up_token_id}": requests.get(u, timeout=5)
                )
                if r.status_code == 200:
                    data = r.json()
                    fm = data[0] if isinstance(data, list) and data else (data if isinstance(data, dict) else {})
                    result = _parse_prices(fm)
                    if result:
                        up_ask, dn_ask = result
                        logger.info(f"  [Gamma token]  UP={up_ask} DN={dn_ask}")
            except Exception as exc:
                logger.debug(f"token fetch error: {exc}")

        # Attempt 3: cached outcomePrices from the discovery scan (may be 15s stale)
        if up_ask is None or dn_ask is None:
            result = _parse_prices(market)
            if result:
                up_ask, dn_ask = result
                logger.info(f"  [Gamma cached]  UP={up_ask} DN={dn_ask}")
            else:
                logger.warning("  All price sources failed or returned 0/0 — skip.")
                return

        up_prob = self._implied_prob(up_ask)
        dn_prob = self._implied_prob(dn_ask)

        # Sanity: UP + DN must sum to ~1.00 in a valid binary market
        price_sum = up_ask + dn_ask
        if price_sum > Decimal("1.15") or price_sum < Decimal("0.85"):
            logger.warning(f"  Price sanity FAIL: UP={up_ask}+DN={dn_ask}={price_sum:.3f} — skip.")
            return

        logger.info(
            f"  UP={up_ask:.2f} ({up_prob:.0%}) | DN={dn_ask:.2f} ({dn_prob:.0%}) | "
            f"sum={price_sum:.3f} | need ≥{Config.CONFIDENCE_THRESHOLD:.0%} to trade"
        )

        # 4. Spread gate — if Gamma shows prices are too close to call, skip
        # (markets stuck at exactly 0.50/0.50 are pre-market defaults, not real signals)
        if up_ask == Decimal("0.5") and dn_ask == Decimal("0.5"):
            logger.info("  Market at default 50/50 — not yet priced — skip.")
            return

        # 5. Confidence gate + max-entry-price gate
        #    Lower bound:  price >= CONFIDENCE_THRESHOLD  (signal strong enough)
        #    Upper bound:  price <= MAX_ENTRY_PRICE        (move not already over)
        def _in_range(price: Decimal) -> bool:
            return Config.CONFIDENCE_THRESHOLD <= price <= Config.MAX_ENTRY_PRICE

        best_pct = float(max(up_prob, dn_prob) * 100)

        if _in_range(up_prob):
            logger.info(
                f"  🎯 UP signal {up_prob:.0%} — in range "
                f"[{Config.CONFIDENCE_THRESHOLD:.0%}–{Config.MAX_ENTRY_PRICE:.0%}]"
            )
            self.window_tracker.record_scan(window_ts, gate_open=True,
                                             signal_pct=best_pct, skip_reason=None,
                                             market_title=question)
            await self._execute(
                token_id=up_token_id, ask_price=up_ask,
                condition_id=condition_id, side="UP", question=question,
                up_prob=up_prob, dn_prob=dn_prob, elapsed_s=elapsed, now=now,
            )
        elif _in_range(dn_prob):
            logger.info(
                f"  🎯 DN signal {dn_prob:.0%} — in range "
                f"[{Config.CONFIDENCE_THRESHOLD:.0%}–{Config.MAX_ENTRY_PRICE:.0%}]"
            )
            self.window_tracker.record_scan(window_ts, gate_open=True,
                                             signal_pct=best_pct, skip_reason=None,
                                             market_title=question)
            await self._execute(
                token_id=dn_token_id, ask_price=dn_ask,
                condition_id=condition_id, side="DOWN", question=question,
                up_prob=up_prob, dn_prob=dn_prob, elapsed_s=elapsed, now=now,
            )
        elif up_prob > Config.MAX_ENTRY_PRICE or dn_prob > Config.MAX_ENTRY_PRICE:
            dominant = "UP" if up_prob > dn_prob else "DN"
            dominant_p = up_prob if up_prob > dn_prob else dn_prob
            logger.info(
                f"  🚫 {dominant} {dominant_p:.0%} — move already happened, "
                f"above max entry {Config.MAX_ENTRY_PRICE:.0%} — skip."
            )
            self.window_tracker.record_scan(window_ts, gate_open=True,
                                             signal_pct=best_pct, skip_reason="above_cap",
                                             market_title=question)
        else:
            logger.info(
                f"  ⚪ No signal — UP={up_prob:.0%} DN={dn_prob:.0%} "
                f"(need ≥{Config.CONFIDENCE_THRESHOLD:.0%} to trade)"
            )
            self.window_tracker.record_scan(window_ts, gate_open=True,
                                             signal_pct=best_pct, skip_reason="no_signal",
                                             market_title=question)

    # ------------------------------------------------------------------
    # Execution — sandbox (simulated) or live (real money)
    # ------------------------------------------------------------------

    async def _execute(
        self,
        token_id: str,
        ask_price: Decimal,
        condition_id: str,
        side: str,
        question: str,
        up_prob: Decimal,
        dn_prob: Decimal,
        elapsed_s: int,
        now: datetime,
    ):
        shares = (Config.MAX_USD_PER_TRADE / ask_price).quantize(
            Decimal("0.1"), rounding=ROUND_DOWN
        )
        cost = (shares * ask_price).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        if shares <= 0:
            logger.warning("0 shares calculated — skip.")
            return

        if self.sandbox:
            # ── SANDBOX ──────────────────────────────────────────────
            logger.info(
                f"  🟡 [SANDBOX] {side}  "
                f"shares={shares}  @{ask_price}  cost=${cost}"
            )
            import time as _time
            _wts = int(_time.time()) // 300 * 300
            self.ledger.record(
                question=question, side=side, price=ask_price,
                shares=shares, cost=cost,
                up_prob=up_prob, dn_prob=dn_prob, elapsed_s=elapsed_s,
                condition_id=condition_id,
                window_ts=_wts,
                market_slug=f"btc-updown-5m-{_wts}",
            )
            self.state.record_trade(
                condition_id, side, ask_price, shares, now, sandbox=True
            )
            self.window_tracker.record_trade(
                window_ts=_wts, side=side,
                price=float(ask_price), shares=float(shares), cost=float(cost),
                market_title=question,
            )

        else:
            # ── LIVE ─────────────────────────────────────────────────
            logger.info(
                f"  🔴 [LIVE] {side}  "
                f"shares={shares}  @{ask_price}  cost≈${cost}"
            )
            loop = asyncio.get_event_loop()
            try:
                order_args = OrderArgs(
                    price=float(ask_price),
                    size=float(shares),
                    side="BUY",
                    token_id=token_id,
                )
                signed = await loop.run_in_executor(
                    None, self.client.create_order, order_args
                )
                resp = await loop.run_in_executor(
                    None, self.client.post_order, signed
                )
                if resp.get("success"):
                    logger.info(f"  ✅ FILLED {side} {shares}sh @{ask_price} ${cost}")
                    self.state.record_trade(
                        condition_id, side, ask_price, shares, now, sandbox=False
                    )
                    import time as _time
                    _wts = int(_time.time()) // 300 * 300
                    # Record in ledger so auto-claim can find condition_id + side after settlement
                    self.ledger.record(
                        question=question, side=side, price=ask_price,
                        shares=shares, cost=cost,
                        up_prob=up_prob, dn_prob=dn_prob, elapsed_s=elapsed_s,
                        condition_id=condition_id,
                        window_ts=_wts,
                        market_slug=f"btc-updown-5m-{_wts}",
                    )
                    self.window_tracker.record_trade(
                        window_ts=_wts, side=side,
                        price=float(ask_price), shares=float(shares), cost=float(cost),
                        market_title=question,
                    )
                else:
                    logger.error(f"  ❌ REJECTED: {resp}")
            except Exception as exc:
                logger.error(f"  💥 Exception: {exc}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self):
        import time as _time
        await self._ensure_auth()
        mode = "SANDBOX (simulated $)" if self.sandbox else "LIVE (real $)"
        logger.info(
            f"\n{'='*60}\n"
            f"  PolyBTC Bot | {mode}\n"
            f"  Scan every {Config.CHECK_INTERVAL}s | "
            f"Time gate {Config.MIN_ELAPSED_SECONDS}s | "
            f"Confidence {Config.CONFIDENCE_THRESHOLD:.0%} | "
            f"Max ${Config.MAX_USD_PER_TRADE}/trade\n"
            f"{'='*60}"
        )

        last_window_ts = int(_time.time()) // 300 * 300

        while True:
            try:
                current_window_ts = int(_time.time()) // 300 * 300

                # ── Window rollover ───────────────────────────────────────
                # Gate just closed on the old window. Do three things:
                #   1. Settle P&L for trades in the old window
                #   2. Print a mid-session per-window report immediately
                #   3. Skip the sleep and scan the new window right away
                if current_window_ts != last_window_ts:
                    logger.info(
                        f"\n{'─'*60}\n"
                        f"  🔔 Window closed: {last_window_ts} → {current_window_ts}\n"
                        f"{'─'*60}"
                    )
                    if self.sandbox:
                        # Settlement runs as a background task — does NOT block the
                        # scan loop. The new window is scanned immediately while
                        # the old window settles concurrently after a 90s delay.
                        settled_wts = last_window_ts  # capture before it changes

                        async def _settle_background(wts=settled_wts):
                            # Wait for Polymarket to resolve the market before
                            # first attempt. Resolution usually takes 1-3 min
                            # after the window closes.
                            INITIAL_DELAY  = 90   # seconds before first try
                            RETRY_INTERVAL = 30   # seconds between retries
                            MAX_ATTEMPTS   = 10   # give up after ~5 min total

                            await asyncio.sleep(INITIAL_DELAY)

                            _loop = asyncio.get_event_loop()
                            for attempt in range(1, MAX_ATTEMPTS + 1):
                                resolved = await _loop.run_in_executor(
                                    None,
                                    lambda: self.ledger.settle_window(wts, Config.GAMMA_API)
                                )
                                if resolved:
                                    logger.info(
                                        f"  ✅ Window {wts} fully settled "
                                        f"(attempt {attempt})"
                                    )
                                    break
                                logger.info(
                                    f"  ⏳ Window {wts} still PENDING — "
                                    f"retry {attempt}/{MAX_ATTEMPTS} "
                                    f"in {RETRY_INTERVAL}s"
                                )
                                if attempt < MAX_ATTEMPTS:
                                    await asyncio.sleep(RETRY_INTERVAL)
                            else:
                                logger.warning(
                                    f"  ⚠️  Window {wts} could not be fully "
                                    f"settled after {MAX_ATTEMPTS} attempts — "
                                    f"leaving remaining trades as PENDING."
                                )

                            self.ledger.print_window_report(wts)
                            self.ledger.save_session_json()
                            self.window_tracker.save_window(wts)

                        asyncio.create_task(_settle_background())

                    else:
                        # ── LIVE: settle + auto-claim winning positions ────
                        # Runs as a background task so the main scan continues
                        # uninterrupted while we wait for market resolution.
                        live_wts = last_window_ts

                        async def _live_settle_and_claim(wts=live_wts):
                            INITIAL_DELAY  = 90
                            RETRY_INTERVAL = 30
                            MAX_ATTEMPTS   = 10

                            await asyncio.sleep(INITIAL_DELAY)

                            _loop = asyncio.get_event_loop()
                            for attempt in range(1, MAX_ATTEMPTS + 1):
                                resolved = await _loop.run_in_executor(
                                    None,
                                    lambda: self.ledger.settle_window(wts, Config.GAMMA_API)
                                )
                                if resolved:
                                    logger.info(
                                        f"  ✅ [LIVE] Window {wts} settled "
                                        f"(attempt {attempt})"
                                    )
                                    break
                                logger.info(
                                    f"  ⏳ [LIVE] Window {wts} PENDING — "
                                    f"retry {attempt}/{MAX_ATTEMPTS} in {RETRY_INTERVAL}s"
                                )
                                if attempt < MAX_ATTEMPTS:
                                    await asyncio.sleep(RETRY_INTERVAL)
                            else:
                                logger.warning(
                                    f"  ⚠️  [LIVE] Window {wts} unsettled after "
                                    f"{MAX_ATTEMPTS} attempts."
                                )
                                return

                            # Auto-claim every WIN trade in this window
                            if self.claimer:
                                win_trades = [
                                    t for t in self.ledger._trades
                                    if t.get("window_ts") == wts
                                    and t.get("outcome") == "WIN"
                                    and t.get("condition_id")
                                ]
                                if win_trades:
                                    logger.info(
                                        f"  💰 Auto-claiming {len(win_trades)} "
                                        f"WIN position(s) for window {wts}…"
                                    )
                                    for t in win_trades:
                                        await self.claimer.claim(
                                            condition_id=t["condition_id"],
                                            side=t["side"],
                                        )
                                else:
                                    logger.info(
                                        f"  ℹ️  No WIN positions to claim in window {wts}."
                                    )

                            self.ledger.save_session_json()
                            self.window_tracker.save_window(wts)

                        asyncio.create_task(_live_settle_and_claim())

                    last_window_ts = current_window_ts
                    # Fall through to normal scan immediately — no sleep needed

                markets = self._fetch_btc_5min_markets()
                if markets:
                    await asyncio.gather(
                        *[self._analyze_and_trade(m) for m in markets]
                    )
                else:
                    logger.info("No markets found this scan.")
            except Exception as exc:
                logger.error(f"Loop error: {exc}")
            await asyncio.sleep(Config.CHECK_INTERVAL)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    engine = PolymarketEngine()
    try:
        asyncio.run(engine.run())
    except KeyboardInterrupt:
        logger.info("Stopped.")
        if engine.sandbox:
            # Best-effort: settle any trades that are still PENDING before
            # printing the final report. Retries up to 5 times with a 30s gap.
            import time as _time
            pending_wts = {
                t["window_ts"] for t in engine.ledger._trades
                if t.get("outcome") not in ("WIN", "LOSS") and t.get("window_ts")
            }
            if pending_wts:
                logger.info(
                    f"  Attempting final settlement for "
                    f"{len(pending_wts)} window(s) before exit..."
                )
                for wts in sorted(pending_wts):
                    for attempt in range(1, 6):
                        resolved = engine.ledger.settle_window(wts, Config.GAMMA_API)
                        if resolved:
                            logger.info(f"  ✅ Window {wts} settled on exit (attempt {attempt})")
                            break
                        logger.info(
                            f"  ⏳ Window {wts} still PENDING on exit — "
                            f"retry {attempt}/5 in 30s"
                        )
                        if attempt < 5:
                            _time.sleep(30)
                    else:
                        logger.warning(
                            f"  ⚠️  Window {wts} could not be settled on exit — "
                            f"marked PENDING in report."
                        )
            engine.ledger.print_report()
            engine.ledger.save_session_json()