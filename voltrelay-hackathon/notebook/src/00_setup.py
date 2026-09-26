# %% [markdown]
# # VoltRelay Energy — What is really driving failures, churn and margin?
#
# **Gradient Learnings Data Analytics Hackathon 2026** · Team: **Monish Kannaha S A** and **Lokeshkumar K**
#
# **How to run:** in Google Colab choose *Runtime → Run all*. The first code cell downloads the eight
# organiser CSVs from the hackathon Google Drive folder (about 2–3 minutes); a full run takes roughly
# 10–15 minutes on a standard Colab CPU runtime. Nothing needs to be uploaded by hand.
#
# **How this notebook is organised**
#
# | Part | What it answers |
# |---|---|
# | 1 · Setup and loading | Download, typed loading of all eight tables |
# | 2 · Data understanding | Shape, coverage, keys and relationships |
# | 3 · Data cleaning | Every documented data-quality issue: detected, measured, handled, re-checked |
# | 4 · Core question 1 | Network performance over time — do the metrics tell the same story? |
# | 5 · Core question 2 | Service failures and customer experience |
# | 6 · Core question 3 | Station and geographic patterns |
# | 7 · Core question 4 | Battery and equipment performance |
# | 8 · Core question 5 | Pricing and partner economics |
# | 9 · Core question 6 | Root causes of new-rider churn (primary vs secondary drivers) |
# | 10 · Decision layer | The ₹ cost of a failed swap, and the four budget proposals scored on one scale |
# | 11 · Key findings | What leadership should do, in priority order |
#
# **Conventions.** Money is in Indian rupees (₹; L = lakh, Cr = crore). A *service failure* is an attempt
# the network failed to serve: no charged battery, abandoned in the queue, or a system error. Rider
# cancellations are reported separately because they are not clearly the network's fault. Every number
# quoted in the report is printed by a cell in this notebook.

# %% [markdown]
# ## 1 · Setup and loading

# %%
import os, re, glob, json, math, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, PercentFormatter
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 160)
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")

DRIVE_FOLDER = "https://drive.google.com/drive/folders/1qsjGWrgmEvOaf2Is2fh7RgVEkKd1V3df"
DATA_DIR = Path(os.environ.get("VOLTRELAY_DATA", "/content/voltrelay_data"))
FIG_DIR = Path(os.environ.get("VOLTRELAY_FIGS", "figures"))
OUT_DIR = Path(os.environ.get("VOLTRELAY_OUT", "outputs"))
for d in (DATA_DIR, FIG_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

TABLES = ["swap_events", "station_hourly_status", "riders", "batteries",
          "support_tickets", "stations", "city_daily_context", "fleet_partners"]


def find_table(name):
    """Return every file for a table (plain, gzipped, or split into parts), searched recursively."""
    hits = sorted(p for p in DATA_DIR.rglob(f"{name}*") if p.suffix in {".csv", ".gz"})
    return [p for p in hits if re.match(rf"{name}(_part\d+|\-\d+|_\d+)?\.csv(\.gz)?$", p.name)]


if not all(find_table(t) for t in TABLES):
    import gdown
    print("Downloading the organiser dataset from Google Drive ...")
    gdown.download_folder(DRIVE_FOLDER, output=str(DATA_DIR), quiet=False, remaining_ok=True)

missing = [t for t in TABLES if not find_table(t)]
assert not missing, f"Missing tables after download: {missing}"
for t in TABLES:
    files = find_table(t)
    print(f"{t:<24} {len(files)} file(s), {sum(f.stat().st_size for f in files)/1e6:8.1f} MB")

# %% [markdown]
# **Chart style.** One quiet style for every figure: thin marks, hairline solid grids, one highlight
# colour when the story is about one thing, and a fixed colour per entity (a city or charger generation
# keeps its colour in every chart). No chart uses two y-axes.

# %%
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
DEEMPH = "#c3c2b7"          # de-emphasised marks
CRITICAL, GOOD = "#d03b3b", "#0ca30c"
SEQ = LinearSegmentedColormap.from_list("seq_blue", [
    "#f3f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_blue_red", ["#1c5cab", "#86b6ef", "#f0efec", "#f0a3a2", "#c23636"])

CITY_ORDER = ["Bengaluru", "Delhi NCR", "Hyderabad", "Pune", "Mumbai", "Jaipur"]
CITY_COLOR = dict(zip(CITY_ORDER, [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN]))
GEN_ORDER = ["Gen1", "Gen2", "Gen3"]
GEN_COLOR = {"Gen1": ORANGE, "Gen2": BLUE, "Gen3": AQUA}

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlepad": 22, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.size": 0, "ytick.major.size": 0, "lines.linewidth": 2, "lines.solid_capstyle": "round",
    "legend.frameon": False, "legend.fontsize": 9, "figure.dpi": 110, "savefig.dpi": 160,
    "savefig.bbox": "tight",
})


def titled(ax, title, subtitle=None):
    ax.set_title(title)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK2, fontsize=9, va="bottom")


def save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png")
    plt.show()


def inr(x, digits=1):
    """Format rupees the Indian way: crore (Cr) and lakh (L)."""
    if pd.isna(x):
        return "n/a"
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e7:
        return f"{sign}₹{x/1e7:,.{digits}f} Cr"
    if x >= 1e5:
        return f"{sign}₹{x/1e5:,.{digits}f} L"
    return f"{sign}₹{x:,.{0 if x >= 100 else 2}f}"


def pct(x, digits=1):
    return "n/a" if pd.isna(x) else f"{100*x:.{digits}f}%"


pct_axis = PercentFormatter(1.0, decimals=0)
RESULTS = {}      # every headline number used in the report is recorded here and exported at the end


def record(key, value, note=""):
    RESULTS[key] = {"value": (float(value) if isinstance(value, (int, float, np.floating, np.integer)) else value),
                    "note": note}
    return value

# %% [markdown]
# **Loading.** Types are set explicitly so that IDs stay text, low-cardinality text becomes categorical
# (the 3.9M-row swap table fits comfortably in Colab memory) and timestamps are parsed once.

# %%
def read_table(name, dtypes=None, dates=()):
    frames = []
    for f in find_table(name):
        header = pd.read_csv(f, nrows=0).columns
        dt = {c: t for c, t in (dtypes or {}).items() if c in header}
        frames.append(pd.read_csv(f, dtype=dt, low_memory=False))
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    for c in dates:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


cat, f32 = "category", "float32"
swaps = read_table("swap_events", {
    "event_id": "string", "rider_id": cat, "station_id": cat, "event_type": cat, "attempt_seq": f32,
    "queue_wait_sec": f32, "battery_in_id": cat, "battery_out_id": cat, "soc_in_pct": f32,
    "soh_in_pct": f32, "soc_out_pct": f32, "soh_out_pct": f32, "km_since_last_swap": f32,
    "tariff_code": cat, "list_price_inr": f32, "discount_inr": f32, "amount_charged_inr": f32,
    "payment_mode": cat, "energy_to_recharge_kwh": f32, "station_firmware": cat, "sync_mode": cat,
}, dates=["event_ts"])
hourly = read_table("station_hourly_status", {
    "station_id": cat, "charged_2w_avg": f32, "charged_2w_min": f32, "charged_3w_min": f32,
    "packs_charging": f32, "packs_quarantined": f32, "chargers_online": f32, "ambient_temp_c": f32,
    "cabinet_temp_c": f32, "avg_charge_minutes": f32, "outage_minutes": f32, "grid_kwh": f32,
    "telemetry_status": cat,
}, dates=["hour_start"])
riders = read_table("riders", {"rider_id": "string", "partner_id": "string"}, dates=["signup_date"])
batteries = read_table("batteries", {"battery_id": "string"},
                       dates=["manufacture_date", "commission_date", "retired_date"])
tickets = read_table("support_tickets", {"ticket_id": "string", "rider_id": "string", "station_id": "string",
                                         "battery_id": "string"}, dates=["created_ts"])
stations = read_table("stations", {"station_id": "string"},
                      dates=["commissioned_date", "decommissioned_date", "firmware_updated_date",
                             "competitor_within_1_5km_since"])
city_daily = read_table("city_daily_context", dates=["date"])
partners = read_table("fleet_partners", {"partner_id": "string"}, dates=["contract_start_date", "amendment_date"])

RAW_ROWS = {n: len(df) for n, df in [("swap_events", swaps), ("station_hourly_status", hourly),
            ("riders", riders), ("batteries", batteries), ("support_tickets", tickets),
            ("stations", stations), ("city_daily_context", city_daily), ("fleet_partners", partners)]}
print(pd.Series(RAW_ROWS, name="rows").to_frame().T)
print(f"swap_events in memory: {swaps.memory_usage(deep=True).sum()/1e9:.2f} GB")
