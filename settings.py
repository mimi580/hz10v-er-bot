# ═══════════════════════════════════════════════════════════════
# 1HZ10V EXPIRYRANGE BOT — SETTINGS
# ═══════════════════════════════════════════════════════════════

# ── Connection ───────────────────────────────────────────────────
DERIV_APP_ID    = "1089"
import os
DERIV_API_TOKEN = os.environ.get("DERIV_API_TOKEN", "")
DERIV_WS_URL    = "wss://ws.derivws.com/websockets/v3"

# ── Contract ─────────────────────────────────────────────────────
SYMBOL          = "1HZ10V"
BARRIER_UPPER   = "+1.6"
BARRIER_LOWER   = "-1.6"
BARRIER_VALUE   = 1.6
EXPIRY_MINUTES  = 2
EXPIRY_UNIT     = "m"
EXPIRY_TICKS    = 120          # 2 min × 60 ticks/min

# ── Warmup ───────────────────────────────────────────────────────
WARMUP_TICKS    = 250          # ticks before any signal is evaluated

# ── Model gates — DO NOT LOOSEN ─────────────────────────────────
# Normal dist: only votes when bollinger also votes OR ou+autocorr vote
# (fixes the nD_OK-alone = 28.6% WR problem from trade data)
NORMAL_DIST_MIN_PROB        = 0.58   # minimum P(inside barriers)
NORMAL_DIST_STANDALONE_PROB = 0.72   # prob required when nD is the ONLY vol model passing
BB_COMPRESSION_PCT          = 0.55   # percentile rank threshold for compression
BB_MIDZONE_LO               = 0.25   # %B lower bound for "near midline"
BB_MIDZONE_HI               = 0.75   # %B upper bound for "near midline"
OU_HALFLIFE_MAX_RATIO       = 2.5    # max half_life = expiry_minutes × ratio × 60
OU_THETA_STABILITY_RATIO    = 2.0    # max ratio between first/second half theta
OU_MIN_THETA                = 0.05   # floor — near-zero reversion = skip
ACF_LAG1_THRESHOLD          = -0.05  # must be more negative than this
ACF_LAG2_MAX                = 0.15   # lag-2 must not show strong trend

# Confluence
MODELS_REQUIRED             = 3      # votes needed out of 4
CONFIDENCE_MIN              = 0.75   # avg confidence of passing models
TOP_CONF_MIN                = 0.70   # at least one model must exceed this

# ── Martingale ───────────────────────────────────────────────────
# Step table:  $0.35 → $0.66 → $1.25 → $2.36 → reset
FIRST_STAKE         = 0.35
MARTINGALE_FACTOR   = 2.5
MARTINGALE_AFTER    = 1      # consecutive losses before step-up
MARTINGALE_MAX_STEP = 3      # hard cap — resets to base after

# ── Risk ─────────────────────────────────────────────────────────
TARGET_PROFIT   = 35.0       # daily profit target ($)
STOP_LOSS       = 40.0       # daily loss limit ($)

# ── Breathing — market condition change after each trade ─────────
# Random cooldown forces the bot into genuinely different market
# conditions before the next trade. Prevents re-entering the same
# volatility cluster that just produced a loss.
#
# Data shows: gaps 6-10 min → 66.7% WR vs gaps 3-6 min → 56.5% WR
# Raising cooldown to 5-8 min targets the higher-WR bucket.
COOLDOWN_MIN    = 210        # 5 minutes minimum
COOLDOWN_MAX    = 314        # 8 minutes maximum

# ── Vol stability window ─────────────────────────────────────────
# Before firing a signal, EWMA vol must have been stable for this
# many ticks. Compares vol now vs vol N ticks ago — if it moved
# more than VOL_STABILITY_THRESHOLD it means the market just had
# a spike or regime change. Wait it out.
VOL_STABILITY_TICKS     = 30    # lookback window (ticks)
VOL_STABILITY_THRESHOLD = 0.15  # max allowed relative vol change (15%)

# ── Reconnect ────────────────────────────────────────────────────
RECONNECT_BASE  = 3
RECONNECT_MAX   = 60

# ── Telegram alerts (optional) ───────────────────────────────────
TELEGRAM_TOKEN   = ""
TELEGRAM_CHAT_ID = ""

# ── Logging ──────────────────────────────────────────────────────
LOG_FILE        = "/tmp/bot.log"
TRADES_FILE     = "/tmp/trades.csv"
STATS_FILE      = "/tmp/stats.json"
