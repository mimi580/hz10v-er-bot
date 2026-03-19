"""
RISK MANAGER
============
Martingale: factor 1.89, max 3 steps, starts after 2 consecutive losses.
Stakes always computed from base (not compounding previous) — avoids float drift.
All values rounded to 2dp before use (Deriv requirement).

Step table:
  Base   $0.35
  Step 1 $0.35 × 1.89¹ = $0.66
  Step 2 $0.35 × 1.89² = $1.25
  Step 3 $0.35 × 1.89³ = $2.36  ← cap, then reset
"""

import json, logging
from pathlib import Path
import settings as S

log = logging.getLogger("Risk")


class RiskManager:
    def __init__(self, balance: float):
        self.balance          = balance
        self.start_balance    = balance
        self.daily_pnl        = 0.0
        self.total_trades     = 0
        self.wins             = 0
        self.losses           = 0
        self.consec_loss      = 0
        self.max_consec_loss  = 0
        self.current_stake    = round(S.FIRST_STAKE, 2)
        self._mg_step         = 0
        self.bot_active       = True
        self.stop_reason      = ""

    # ── Public ────────────────────────────────────────────────────

    def stake(self) -> float:
        return self.current_stake if self.bot_active else 0.0

    def sync(self, balance: float):
        self.balance = balance

    def record(self, profit: float, win: bool) -> dict:
        """Update state after trade settles. Returns any alert dict."""
        self.total_trades += 1
        self.daily_pnl    += profit
        self.balance      += profit
        alerts = {}

        if win:
            self.wins        += 1
            self.consec_loss  = 0
            self._mg_step     = 0
            self.current_stake = round(S.FIRST_STAKE, 2)
        else:
            self.losses          += 1
            self.consec_loss     += 1
            self.max_consec_loss  = max(self.max_consec_loss, self.consec_loss)
            self._apply_martingale()

        # Stop guards
        if self.balance <= 0 or (self.start_balance - self.balance) >= S.STOP_LOSS:
            self.bot_active  = False
            self.stop_reason = f"Stop loss hit (≥${S.STOP_LOSS:.2f})"
            alerts["STOP"]   = self.stop_reason
            log.critical(f"BOT STOPPED — {self.stop_reason}")

        if self.daily_pnl >= S.TARGET_PROFIT:
            self.bot_active  = False
            self.stop_reason = f"Target reached (+${S.TARGET_PROFIT:.2f})"
            alerts["TARGET"] = self.stop_reason
            log.info(f"BOT STOPPED — {self.stop_reason}")

        self._save()
        return alerts

    def summary_line(self) -> str:
        wr = self.wins / self.total_trades * 100 if self.total_trades else 0
        return (f"{self.total_trades}T {self.wins}W/{self.losses}L "
                f"WR={wr:.1f}% P&L=${self.daily_pnl:+.2f} "
                f"Stake=${self.current_stake:.2f} MG={self._mg_step} "
                f"Bal=${self.balance:.2f}")

    # ── Internal ──────────────────────────────────────────────────

    def _apply_martingale(self):
        if self.consec_loss < S.MARTINGALE_AFTER:
            return   # not enough consecutive losses yet
        self._mg_step += 1
        if self._mg_step <= S.MARTINGALE_MAX_STEP:
            raw = S.FIRST_STAKE * (S.MARTINGALE_FACTOR ** self._mg_step)
            self.current_stake = round(raw, 2)
            log.info(f"MG step {self._mg_step}/{S.MARTINGALE_MAX_STEP} "
                     f"→ ${self.current_stake:.2f}")
        else:
            self._mg_step      = 0
            self.current_stake = round(S.FIRST_STAKE, 2)
            log.info(f"MG cap reached → reset to ${S.FIRST_STAKE:.2f}")

    def _save(self):
        try:
            Path(S.STATS_FILE).write_text(json.dumps({
                "balance":        self.balance,
                "daily_pnl":      self.daily_pnl,
                "total_trades":   self.total_trades,
                "wins":           self.wins,
                "losses":         self.losses,
                "consec_loss":    self.consec_loss,
                "max_consec":     self.max_consec_loss,
                "current_stake":  self.current_stake,
                "mg_step":        self._mg_step,
                "bot_active":     self.bot_active,
                "stop_reason":    self.stop_reason,
            }, indent=2))
        except Exception:
            pass
