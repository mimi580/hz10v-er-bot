"""
1HZ10V EXPIRYRANGE BOT
========================
Single symbol. Single focus. No distractions.

ARCHITECTURE DECISIONS
─────────────────────
Trade lock:
  state.in_trade is set True the moment a signal fires and is only
  cleared after the contract fully settles (won/lost). The bot will
  not evaluate any signal while a trade is open. No exceptions.

Buy response — direct recv():
  The message loop is "async for raw in ws: await _handle(msg)".
  _handle → _evaluate → _place_trade is ONE suspended call stack.
  While _place_trade awaits a buy confirmation, the message loop
  is frozen and CANNOT read new WS messages. So the buy response
  sits in the buffer unread until timeout.

  Fix: after sending the buy, _place_trade reads the WebSocket
  directly with ws.recv() in a tight loop, matching by req_id.
  Non-buy messages are buffered and replayed after _handle returns.
  This is the only correct pattern for request-response over a
  single WebSocket where the reader is also the caller.

Breathing cooldown:
  After every settled trade, the bot waits a random 120–180 seconds
  before evaluating the next signal. This forces the bot into
  genuinely different market conditions — different volatility cluster,
  different OU fit, different autocorrelation regime. Prevents
  immediately re-entering conditions that just produced a loss.

Market condition change detection:
  After the cooldown, the bot also checks that the current market
  state differs meaningfully from the state at the last trade entry.
  If EWMA vol, tick ACF, and BB width are all within 5% of their
  pre-trade values, the bot skips and waits another 30 seconds.
  This is the "break after a trade to avoid similar conditions".
"""

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import asyncio, json, random, time, uuid
from collections import deque
from typing import Optional

import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

import settings as S
from models  import evaluate, ERSignal
from risk    import RiskManager
from logger  import setup_logger, TradeLogger, Telegram

log = setup_logger()


# ─────────────────────────────────────────────────────────────────
# MARKET STATE SNAPSHOT — for condition-change detection
# ─────────────────────────────────────────────────────────────────

class MarketSnapshot:
    """Lightweight snapshot of market conditions at trade entry."""
    def __init__(self, prices: np.ndarray, barrier: float):
        if len(prices) < 25:
            self.valid    = False
            return
        self.valid        = True
        rets              = np.diff(prices[-60:])
        alpha             = 1.0 - np.exp(-np.log(2) / 20.0)
        var               = float(rets[-1] ** 2)
        for r in reversed(rets[:-1]):
            var           = alpha * r**2 + (1.0 - alpha) * var
        self.ewma_vol     = float(np.sqrt(var))
        n                 = len(rets)
        mu                = float(np.mean(rets))
        vv                = float(np.var(rets, ddof=1))
        self.acf1         = (float(np.mean((rets[:-1] - mu) * (rets[1:] - mu))) / vv
                             if vv > 0 and n >= 3 else 0.0)
        arr               = prices[-20:]
        self.bb_width     = float(4 * np.std(arr, ddof=1) / (np.mean(arr) + 1e-8))

    def conditions_changed(self, other: "MarketSnapshot",
                           threshold: float = 0.05) -> bool:
        """
        Returns True if the market looks meaningfully different
        from when the last trade was placed.
        All three metrics must have changed by at least threshold (5%)
        before we consider conditions genuinely different.
        """
        if not self.valid or not other.valid:
            return True   # can't compare — assume changed
        vol_changed  = abs(self.ewma_vol - other.ewma_vol) > threshold * other.ewma_vol
        acf_changed  = abs(self.acf1    - other.acf1)    > 0.05     # absolute
        bbw_changed  = abs(self.bb_width - other.bb_width) > threshold * other.bb_width
        return vol_changed or acf_changed or bbw_changed


# ─────────────────────────────────────────────────────────────────
# MAIN BOT
# ─────────────────────────────────────────────────────────────────

class HZ10VBot:

    def __init__(self):
        self.ws:            Optional[websockets.WebSocketClientProtocol] = None
        self.connected      = False
        self.balance        = 0.0
        self.risk:          Optional[RiskManager] = None
        self.tlog           = TradeLogger()
        self.tg             = Telegram()

        # Tick buffer
        self._prices        = deque(maxlen=500)
        self._tick_count    = 0

        # Trade state
        self.in_trade       = False
        self._cooldown_until = 0.0        # epoch time — don't evaluate before this
        self._last_snapshot: Optional[MarketSnapshot] = None  # conditions at last trade

        # WS messaging
        self._req_id        = 0
        self._msg_buffer    = []          # messages buffered during direct recv

    def _nid(self) -> int:
        self._req_id += 1
        return self._req_id

    def _prices_arr(self) -> np.ndarray:
        return np.array(self._prices, dtype=np.float64)

    # ─── Reconnect loop ───────────────────────────────────────────

    async def run(self):
        log.info("=" * 65)
        log.info(f"1HZ10V EXPIRYRANGE BOT")
        log.info(f"Barriers: {S.BARRIER_LOWER} / {S.BARRIER_UPPER} | Expiry: {S.EXPIRY_MINUTES}min")
        log.info(f"Martingale: ×{S.MARTINGALE_FACTOR} | max {S.MARTINGALE_MAX_STEP} steps | after {S.MARTINGALE_AFTER} losses")
        log.info(f"Breathing: {S.COOLDOWN_MIN}–{S.COOLDOWN_MAX}s random cooldown")
        log.info("=" * 65)

        delay, attempt = S.RECONNECT_BASE, 0
        while True:
            try:
                attempt += 1
                log.info(f"Connecting (attempt {attempt})...")
                await self._connect()
                delay = S.RECONNECT_BASE; attempt = 0
            except (ConnectionClosed, WebSocketException) as e:
                log.warning(f"WS closed: {e}")
            except asyncio.TimeoutError:
                log.warning("Connection timed out")
            except Exception as e:
                log.error(f"Unexpected error: {e}", exc_info=True)

            self.connected = False
            if self.in_trade:
                log.warning("Disconnected mid-trade — releasing lock")
                self.in_trade = False

            log.info(f"Reconnecting in {delay}s...")
            await asyncio.sleep(delay)
            delay = min(delay * 2, S.RECONNECT_MAX)

    # ─── Connection ───────────────────────────────────────────────

    async def _connect(self):
        url = f"{S.DERIV_WS_URL}?app_id={S.DERIV_APP_ID}"
        async with websockets.connect(
            url, ping_interval=30, ping_timeout=20, close_timeout=5
        ) as ws:
            self.ws        = ws
            self.connected = True

            # Auth
            await self._send_raw({"authorize": S.DERIV_API_TOKEN})
            auth = json.loads(await asyncio.wait_for(ws.recv(), 15.0))
            if auth.get("error"):
                raise Exception(f"Auth failed: {auth['error']['message']}")

            self.balance = float(auth["authorize"]["balance"])
            log.info(f"Authorized | Balance: ${self.balance:.2f}")

            if self.risk is None:
                self.risk = RiskManager(self.balance)
            else:
                self.risk.sync(self.balance)

            # Subscribe
            await self._send_raw({"balance": 1, "subscribe": 1})
            await self._send_raw({"ticks": S.SYMBOL, "subscribe": 1})
            log.info(f"Subscribed to {S.SYMBOL}")

            self.tg.send(
                f"⚡ <b>1HZ10V ER Bot STARTED</b>\n"
                f"Balance: <b>${self.balance:.2f}</b>\n"
                f"Barriers: {S.BARRIER_LOWER}/{S.BARRIER_UPPER} | {S.EXPIRY_MINUTES}min"
            )

            # Main message loop
            async for raw in ws:
                if not self.connected:
                    break
                try:
                    await self._handle(json.loads(raw))
                except Exception as e:
                    log.error(f"Handle error: {e}", exc_info=True)

                # Drain buffer (messages captured during direct recv in _place_trade)
                while self._msg_buffer:
                    msg = self._msg_buffer.pop(0)
                    try:
                        await self._handle(msg)
                    except Exception as e:
                        log.error(f"Buffer handle error: {e}", exc_info=True)

    async def _send_raw(self, payload: dict) -> int:
        rid = self._nid()
        payload["req_id"] = rid
        if self.ws and self.connected:
            try:
                await asyncio.wait_for(self.ws.send(json.dumps(payload)), 10.0)
            except Exception as e:
                log.warning(f"Send error: {e}")
        return rid

    # ─── Message router ───────────────────────────────────────────

    async def _handle(self, msg: dict):
        t = msg.get("msg_type")
        if   t == "tick":
            await self._on_tick(msg["tick"])
        elif t == "balance":
            self.balance = float(msg["balance"]["balance"])
            if self.risk:
                self.risk.sync(self.balance)
        elif t == "proposal_open_contract":
            await self._on_contract(msg["proposal_open_contract"])
        elif t == "buy":
            # Stray buy message reaching the main loop — ignore safely
            log.debug(f"Stray buy in main loop: req_id={msg.get('req_id')}")
        elif t == "error":
            log.error(f"API error: {msg.get('error', {}).get('message', '?')}")

    # ─── Tick processing ──────────────────────────────────────────

    async def _on_tick(self, tick: dict):
        price = float(tick["quote"])
        epoch = int(tick["epoch"])
        self._prices.append(price)
        self._tick_count += 1

        # Warmup
        if self._tick_count < S.WARMUP_TICKS:
            if self._tick_count % 50 == 0:
                log.info(f"Warming up {self._tick_count}/{S.WARMUP_TICKS}")
            return

        # Hard trade lock — no evaluation while a trade is open
        if self.in_trade:
            return

        # Risk active
        if not self.risk or not self.risk.bot_active:
            return

        # Breathing cooldown
        now = time.time()
        if now < self._cooldown_until:
            remaining = self._cooldown_until - now
            if self._tick_count % 30 == 0:
                log.debug(f"Breathing... {remaining:.0f}s remaining")
            return

        # Market condition change check — only after cooldown expires
        if self._last_snapshot is not None:
            current = MarketSnapshot(self._prices_arr(), S.BARRIER_VALUE)
            if not current.conditions_changed(self._last_snapshot):
                if self._tick_count % 30 == 0:
                    log.info("Conditions unchanged from last trade — waiting 30s")
                self._cooldown_until = now + 30.0
                return

        await self._evaluate()

    # ─── Signal evaluation ────────────────────────────────────────

    async def _evaluate(self):
        prices = self._prices_arr()
        signal = evaluate(prices)

        if not signal.tradeable:
            # Log meaningful skips at INFO, noise at DEBUG
            sr = signal.skip_reason
            if any(k in sr for k in ["LOW_CONF", "NO_STRONG", "DISAGREE"]):
                log.info(f"SKIP | {sr} | {' '.join(signal.reasons[:2])}")
            else:
                log.debug(f"SKIP | {sr}")
            return

        stake = self.risk.stake()
        if stake == 0:
            return

        # Lock immediately — before any await
        self.in_trade = True

        log.info(
            f"SIGNAL ✅ | conf={signal.confidence:.4f} | "
            f"score={signal.score}/4 | stake=${stake:.2f} | "
            f"{' | '.join(signal.reasons)}"
        )

        # Snapshot conditions at entry for post-trade comparison
        self._last_snapshot = MarketSnapshot(prices, S.BARRIER_VALUE)

        await self._place_trade(signal, stake)

    # ─── Trade placement (direct recv — the only correct approach) ─

    async def _place_trade(self, signal: ERSignal, stake: float):
        """
        Send buy and read the response DIRECTLY from the WebSocket.

        The message loop is frozen inside _handle → _evaluate → here.
        We cannot wait for the loop to deliver the buy response because
        it can't run — it's waiting for us to return. So we call
        ws.recv() directly in a loop until we see our buy response,
        buffering everything else for replay after we return.
        """
        trade_id = str(uuid.uuid4())[:8].upper()
        rid      = self._nid()

        payload = {
            "buy": 1,
            "price": stake,
            "req_id": rid,
            "parameters": {
                "contract_type": "EXPIRYRANGE",
                "symbol":        S.SYMBOL,
                "duration":      S.EXPIRY_MINUTES,
                "duration_unit": S.EXPIRY_UNIT,
                "basis":         "stake",
                "amount":        stake,
                "currency":      "USD",
                "barrier":       S.BARRIER_UPPER,
                "barrier2":      S.BARRIER_LOWER,
            },
        }

        try:
            await asyncio.wait_for(
                self.ws.send(json.dumps(payload)), timeout=10.0)
        except Exception as e:
            log.error(f"Send failed: {e} — releasing lock")
            self._release_lock()
            return

        # Read directly — buffer everything that isn't our buy response
        buy_resp = None
        deadline = time.time() + 15.0

        while time.time() < deadline:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                raw = await asyncio.wait_for(
                    self.ws.recv(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            except Exception as e:
                log.error(f"WS recv error: {e}")
                break

            msg      = json.loads(raw)
            msg_type = msg.get("msg_type")
            msg_rid  = msg.get("req_id")

            if msg_type == "buy" and msg_rid == rid:
                buy_resp = msg
                break
            else:
                # Buffer for replay after _handle returns
                self._msg_buffer.append(msg)

        if buy_resp is None:
            log.error("Buy response timeout — lock released. "
                      "Trade may have been placed on Deriv — check account.")
            self._release_lock()
            return

        if "error" in buy_resp:
            err = buy_resp["error"].get("message", "unknown")
            log.error(f"Buy rejected by Deriv: {err}")
            self._release_lock()
            return

        buy_data    = buy_resp.get("buy", {})
        contract_id = buy_data.get("contract_id")
        if not contract_id:
            log.error("No contract_id in buy response")
            self._release_lock()
            return

        log.info(
            f"PLACED {trade_id} | cid={contract_id} | "
            f"${stake:.2f} | EXPIRYRANGE ±{S.BARRIER_VALUE} | {S.EXPIRY_MINUTES}min"
        )

        self.tg.send(
            f"📈 <b>TRADE PLACED</b> [{S.SYMBOL}]\n"
            f"Stake: <b>${stake:.2f}</b> | conf={signal.confidence:.3f}\n"
            f"{' | '.join(signal.reasons[:3])}"
        )

        # Store trade meta for settlement
        self._pending = {
            "trade_id":  trade_id,
            "contract_id": contract_id,
            "stake":     stake,
            "signal":    signal,
            "mg_step":   self.risk._mg_step,
        }

        # Subscribe for settlement push
        await self._send_raw({
            "proposal_open_contract": 1,
            "contract_id": contract_id,
            "subscribe": 1,
        })

        # Fallback poller in case push never comes
        asyncio.create_task(self._poll(contract_id))

    def _release_lock(self):
        """Release trade lock with minimum cooldown on error paths."""
        self.in_trade        = False
        self._cooldown_until = time.time() + S.COOLDOWN_MIN

    # ─── Settlement push ──────────────────────────────────────────

    async def _on_contract(self, contract: dict):
        cid    = contract.get("contract_id")
        status = contract.get("status")
        if status not in ("won", "lost", "sold"):
            return
        meta = getattr(self, "_pending", {})
        if not meta or meta.get("contract_id") != cid:
            return
        await self._settle(contract, meta)

    # ─── Fallback poller ──────────────────────────────────────────

    async def _poll(self, contract_id: str):
        """
        Poll for settlement every 10s after expiry time.
        Fires only if the push subscription didn't deliver.
        """
        try:
            # Wait for contract to expire + buffer
            await asyncio.sleep(S.EXPIRY_MINUTES * 60 + 15)
            for attempt in range(1, 13):
                meta = getattr(self, "_pending", {})
                if not meta or meta.get("contract_id") != contract_id:
                    return   # already settled via push
                if not self.connected:
                    return
                log.info(f"Polling settlement {contract_id} ({attempt}/12)")
                await self._send_raw({
                    "proposal_open_contract": 1,
                    "contract_id": contract_id,
                })
                await asyncio.sleep(10)

            # Never settled
            log.error(f"Contract {contract_id} never settled after polling")
            self._pending = {}
            self._release_lock()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error(f"Poll error: {e}", exc_info=True)
            self._release_lock()

    # ─── Settlement handler ───────────────────────────────────────

    async def _settle(self, contract: dict, meta: dict):
        """Full settlement — update risk, log, breathe, unlock."""
        profit = float(contract.get("profit", 0))
        win    = contract.get("status") == "won"
        cid    = meta["contract_id"]
        stake  = meta["stake"]
        signal = meta["signal"]

        alerts = self.risk.record(profit, win)

        self.tlog.record(
            profit=profit, win=win, stake=stake,
            confidence=signal.confidence,
            score=signal.score,
            models=signal.models,
            mg_step=meta["mg_step"],
            balance=self.balance,
            reasons=signal.reasons,
        )

        icon    = "✅" if win else "❌"
        summary = self.risk.summary_line()
        log.info(
            f"{icon} {'WIN' if win else 'LOSS'} | "
            f"P&L ${profit:+.2f} | {summary}"
        )

        self.tg.send(
            f"{icon} <b>{'WIN' if win else 'LOSS'}</b> [{S.SYMBOL}]\n"
            f"P&L: <b>${profit:+.2f}</b> | Bal: ${self.balance:.2f}\n"
            f"{summary}"
        )

        for kind, msg in alerts.items():
            self.tg.send(f"🛑 <b>{kind}</b>: {msg}")

        # ── Breathing cooldown (random 2–3 min) ───────────────────
        cooldown = random.uniform(S.COOLDOWN_MIN, S.COOLDOWN_MAX)
        self._cooldown_until = time.time() + cooldown
        log.info(
            f"Breathing {cooldown:.0f}s — market conditions will "
            f"shift before next evaluation"
        )

        self._pending = {}
        self.in_trade = False   # unlock AFTER everything is done

        # Print scoreboard
        self._scoreboard(win, profit)

    # ─── Terminal scoreboard ──────────────────────────────────────

    def _scoreboard(self, win: bool, profit: float):
        G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"
        B = "\033[94m"; M = "\033[90m"; X = "\033[0m"; BO = "\033[1m"

        r = self.risk
        wr    = r.wins / r.total_trades * 100 if r.total_trades else 0
        wc    = G if wr >= 62 else Y if wr >= 55 else R
        pc    = G if r.daily_pnl >= 0 else R
        mgc   = R if r._mg_step >= 2 else Y if r._mg_step == 1 else M
        res   = f"{G}WIN ✅{X}" if win else f"{R}LOSS ❌{X}"
        pnl_s = f"{G}+${profit:.2f}{X}" if profit >= 0 else f"{R}-${abs(profit):.2f}{X}"

        print(f"\n{BO}{'─'*65}{X}")
        print(f"  {BO}[1HZ10V]{X}  {res}  P&L {pnl_s}  Bal {B}${r.balance:.2f}{X}")
        print(f"  {M}{'─'*63}{X}")
        print(f"  {'Trades':>8}  {'Wins':>6}  {'Losses':>7}  "
              f"{'WR':>6}  {'P&L':>9}  {'Stake':>7}  {'MG':>4}")
        print(f"  {r.total_trades:>8}  "
              f"{G}{r.wins:>6}{X}  "
              f"{R}{r.losses:>7}{X}  "
              f"{wc}{wr:>5.1f}%{X}  "
              f"{pc}{r.daily_pnl:>+9.2f}{X}  "
              f"{Y}${r.current_stake:>5.2f}{X}  "
              f"{mgc}{'s'+str(r._mg_step) if r._mg_step else 'base':>4}{X}")
        print(f"{BO}{'─'*65}{X}\n", flush=True)


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not S.DERIV_API_TOKEN:
        print("ERROR: Set DERIV_API_TOKEN in settings.py")
        sys.exit(1)
    try:
        asyncio.run(HZ10VBot().run())
    except KeyboardInterrupt:
        log.info("Stopped by user.")
