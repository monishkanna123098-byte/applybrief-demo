# %% [markdown]
# ## 4 · Core question 1 — Network performance over time
#
# *How have completed swaps, revenue, failure rates and contribution margin per swap trended? Do they
# tell the same story?*
#
# **Contribution margin per swap** needs a cost model, because the data has costs in three places:
#
# | Cost | How it is computed | Source |
# |---|---|---|
# | Energy | kWh drawn to recharge the returned pack × that station's grid tariff | swap_events, stations |
# | Battery wear | share of the pack's usable life used up per swap × pack purchase cost | batteries, swap_events |
# | Site | monthly rent + maintenance of each open station, spread over that station's swaps that month | stations |
#
# *Usable life* runs from the pack's initial state of health (SoH) down to the end-of-life threshold below.
# Revenue is the net amount charged, so partner discounts are already netted out.

# %% [markdown]
# **Battery wear per swap.** A pack's SoH lost during the data window, divided by the swaps it delivered
# in the window, is the health it spends per swap. The end-of-life threshold is taken from the data: the
# median SoH of packs retired for health reasons, falling back to the industry-standard 70% if there are
# too few.

# %%
issued = sw.loc[sw["completed"] & sw["battery_out_id"].notna(), ["battery_out_id", "event_ts", "soh_out_pct"]]
issued = issued.assign(battery_id=issued["battery_out_id"].astype(str)).sort_values("event_ts")
per_pack = issued.groupby("battery_id").agg(swaps_in_window=("event_ts", "size"),
                                            first_soh_seen=("soh_out_pct", "first"),
                                            first_seen=("event_ts", "min"))
bat = bat.merge(per_pack, left_on="battery_id", right_index=True, how="left")
health_retired = bat["retired_date"].notna() & bat["retirement_reason"].astype(str).str.contains(
    "soh|health|degrad|capacity|end", case=False, regex=True)
EOL_SOH = float(bat.loc[health_retired, "current_soh_pct"].median()) if health_retired.sum() >= 20 else 70.0
EOL_SOH = min(EOL_SOH, 80.0)
start_soh = bat["first_soh_seen"].fillna(bat["initial_soh_pct"])
bat["soh_lost_in_window"] = (start_soh - bat["current_soh_pct"]).clip(lower=0)
bat["soh_per_swap"] = bat["soh_lost_in_window"] / bat["swaps_in_window"]
usable = (bat["initial_soh_pct"] - EOL_SOH).clip(lower=5)
bat["wear_inr_per_swap"] = bat["soh_per_swap"] / usable * bat["purchase_cost_inr"]
print(f"End-of-life SoH threshold used: {EOL_SOH:.1f}% "
      f"({'from ' + str(int(health_retired.sum())) + ' health retirements' if health_retired.sum() >= 20 else 'default'})")
display(bat.groupby("supplier")[["soh_per_swap", "wear_inr_per_swap", "purchase_cost_inr"]].median())
record("eol_soh", EOL_SOH)

# %%
sw["battery_out_str"] = sw["battery_out_id"].astype(str)
sw["wear_inr"] = sw["battery_out_str"].map(bat.set_index("battery_id")["wear_inr_per_swap"]).astype(float)
sw.loc[~sw["completed"], "wear_inr"] = np.nan
sw["energy_inr"] = sw["energy_to_recharge_kwh"].astype(float) * sw["grid_tariff_inr_kwh"].astype(float)

months = pd.period_range(sw["month"].min(), sw["month"].max(), freq="M")
st_open = st[~st["is_test"]].copy()
site_rows = []
for m in months:
    m0, m1 = m.start_time, m.end_time
    open_ = (st_open["commissioned_date"] <= m1) & (st_open["decommissioned_date"].isna() | (st_open["decommissioned_date"] >= m0))
    end = st_open["decommissioned_date"].fillna(m1).clip(upper=m1)
    start = st_open["commissioned_date"].clip(lower=m0)
    days_open = ((end - start).dt.days + 1).clip(lower=0)
    frac = days_open / m.days_in_month
    cost = (st_open["monthly_rent_inr"] + st_open["monthly_maintenance_inr"]) * frac
    site_rows.append(pd.DataFrame({"station_id": st_open["station_id"], "month": m, "site_cost_inr": cost.where(open_, 0)}))
site = pd.concat(site_rows, ignore_index=True)
site = site[site["site_cost_inr"] > 0]

done = sw[sw["completed"]]
stm = done.groupby(["station_id", "month"], observed=True).agg(swaps=("completed", "size")).reset_index()
stm["station_id"] = stm["station_id"].astype(str)
site = site.merge(stm, on=["station_id", "month"], how="left").fillna({"swaps": 0})
site_per_swap = site.assign(v=site["site_cost_inr"] / site["swaps"].replace(0, np.nan)).set_index(["station_id", "month"])["v"]
key = pd.MultiIndex.from_arrays([sw["station_id"].astype(str), sw["month"]])
sw["site_inr"] = np.where(sw["completed"], site_per_swap.reindex(key).to_numpy(), np.nan)
sw["contribution_inr"] = sw["amount_charged_inr"] - sw["energy_inr"] - sw["wear_inr"] - sw["site_inr"]
sw["variable_margin_inr"] = sw["amount_charged_inr"] - sw["energy_inr"] - sw["wear_inr"]
idle_site_cost = site.loc[site["swaps"] == 0, "site_cost_inr"].sum()
print(f"Site cost of station-months with zero swaps (not attributable to any swap): {inr(idle_site_cost)}")

# %% [markdown]
# **Monthly network scorecard.** One row per month; the four headline metrics are charted below.

# %%
g = sw.groupby("month")
monthly = pd.DataFrame({
    "attempts": g.size(),
    "completed": g["completed"].sum(),
    "revenue_inr": g["amount_charged_inr"].sum(),
    "service_failures": g["service_failure"].sum(),
    "stockouts": g["stockout"].sum(),
    "abandoned": g["abandoned"].sum(),
    "system_errors": g["system_error"].sum(),
    "cancelled": g["cancelled"].sum(),
    "energy_inr": g["energy_inr"].sum(),
    "wear_inr": g["wear_inr"].sum(),
    "site_inr": g["site_inr"].sum(),
    "active_riders": g["rider_id"].nunique(),
    "stations_active": g["station_id"].nunique(),
    "median_wait_min": g["queue_wait_sec"].median() / 60,
})
monthly["service_failure_rate"] = monthly["service_failures"] / monthly["attempts"]
monthly["stockout_rate"] = monthly["stockouts"] / monthly["attempts"]
monthly["abandon_rate"] = monthly["abandoned"] / monthly["attempts"]
for c in ["revenue", "energy", "wear", "site"]:
    monthly[f"{c}_per_swap"] = monthly[f"{c}_inr"] / monthly["completed"]
monthly["contribution_per_swap"] = monthly["revenue_per_swap"] - monthly["energy_per_swap"] - monthly["wear_per_swap"] - monthly["site_per_swap"]
monthly["variable_margin_per_swap"] = monthly["revenue_per_swap"] - monthly["energy_per_swap"] - monthly["wear_per_swap"]
monthly.index = monthly.index.astype(str)
display(monthly[["attempts", "completed", "revenue_inr", "service_failure_rate", "stockout_rate", "abandon_rate",
                 "revenue_per_swap", "energy_per_swap", "wear_per_swap", "site_per_swap", "contribution_per_swap"]])

# %% [markdown]
# **When did the business decisions happen?** Rather than hard-coding dates, each decision is located
# in the data: the base-price change from the standard list price, the pricing pilot from the first
# peak/off-peak tariffs, the new supplier from pack commission dates, the expansion waves from station
# commission dates, and the contract amendment from the partner table.

# %%
std = sw[sw["completed"] & sw["tariff_code"].astype(str).eq("STD")]
wk_price = std.groupby([std["event_ts"].dt.to_period("W"), "vehicle_class"], observed=True)["list_price_inr"].median().unstack()
price_change = {}
for vc in wk_price.columns:
    s = wk_price[vc].dropna()
    jumps = s[s.diff().abs() > 0.5]
    price_change[vc] = [(str(p.start_time.date()), float(s.shift(1)[p]), float(v)) for p, v in jumps.items()]
print("Standard list-price changes (week, before, after):", price_change)

tc = sw["tariff_code"].astype(str)
pilot_rows = sw[tc.isin(["PEAK", "OFFPEAK"])]
PILOT_CITIES = sorted(pilot_rows["city"].astype(str).unique())
PILOT_START = pilot_rows["event_ts"].min().normalize()
PILOT_END = pilot_rows["event_ts"].max().normalize()
new_supplier = bat.groupby("supplier")["commission_date"].min().sort_values()
waves = st_open.groupby("expansion_wave")["commissioned_date"].agg(["min", "max", "size"])
amend = partners.loc[partners["amendment_date"].notna(),
                     ["partner_name", "amendment_date", "discount_pct", "discount_pct_after_amendment"]]
print(f"Peak/off-peak pilot: cities {PILOT_CITIES}, {PILOT_START.date()} → {PILOT_END.date()}")
print("First commission date by supplier:", new_supplier.dt.date.to_dict())
display(waves)
display(amend)

BASE_PRICE_DATE = pd.Timestamp(min((c[0][0] for c in price_change.values() if c), default="2024-07-01"))
NEW_SUPPLIER = new_supplier.index[-1]
NEW_SUPPLIER_DATE = new_supplier.iloc[-1]
EVENTS = {"Base price change": BASE_PRICE_DATE, f"{NEW_SUPPLIER} packs arrive": NEW_SUPPLIER_DATE,
          "Pricing pilot": PILOT_START}
for w_name, row in waves.iterrows():
    if not str(w_name).lower().startswith("launch"):
        EVENTS[str(w_name).replace("_", " ")] = row["min"]
for _, r in amend.iterrows():
    EVENTS[f"{r['partner_name']} amendment"] = r["amendment_date"]
record("pilot_cities", ", ".join(PILOT_CITIES))
record("pilot_start", str(PILOT_START.date()))
record("base_price_change", str(BASE_PRICE_DATE.date()))

# %%
def add_events(ax, y_frac=0.97):
    for i, (name, d) in enumerate(sorted(EVENTS.items(), key=lambda kv: kv[1])):
        x = pd.Period(d, "M").to_timestamp()
        ax.axvline(x, color=AXIS, lw=1, zorder=0)
        ax.text(x, y_frac - 0.09 * (i % 3), " " + name, transform=ax.get_xaxis_transform(),
                fontsize=7.5, color=MUTED, va="top")


mts = pd.PeriodIndex(monthly.index, freq="M").to_timestamp()
fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), sharex=True)
panels = [("completed", "Completed swaps per month", lambda v, _: f"{v/1e3:,.0f}K"),
          ("revenue_inr", "Revenue per month", lambda v, _: inr(v, 1)),
          ("service_failure_rate", "Service-failure rate", lambda v, _: f"{100*v:.1f}%"),
          ("contribution_per_swap", "Contribution margin per swap (after energy, wear, site)", lambda v, _: f"₹{v:,.0f}")]
for ax, (col, ttl, fmt) in zip(axes.ravel(), panels):
    ax.plot(mts, monthly[col], color=BLUE)
    ax.scatter(mts[[0, -1]], monthly[col].iloc[[0, -1]], color=BLUE, s=28, zorder=3, edgecolor=SURFACE, linewidth=2)
    for i in (0, -1):
        ax.annotate(fmt(monthly[col].iloc[i], None), (mts[i], monthly[col].iloc[i]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8.5, color=INK)
    ax.yaxis.set_major_formatter(FuncFormatter(fmt))
    if col == "contribution_per_swap":
        ax.axhline(0, color=INK2, lw=1)
    add_events(ax)
    ax.set_title(ttl, fontsize=10.5, pad=8)
fig.suptitle("Four headline metrics, month by month", x=0.01, ha="left",
             fontsize=13, fontweight="bold")
fig.tight_layout()
save(fig, "q1_four_metrics")

# %%
q = monthly.copy()
base = q.iloc[:3][["completed", "revenue_inr", "service_failures"]].mean()
idx = q[["completed", "revenue_inr", "service_failures"]] / base * 100
fig, ax = plt.subplots(figsize=(10, 4.2))
for col, color, label in [("service_failures", RED, "Service failures"), ("revenue_inr", BLUE, "Revenue"),
                          ("completed", AQUA, "Completed swaps")]:
    ax.plot(mts, idx[col], color=color, label=label)
    ax.annotate(f"{label}  {idx[col].iloc[-1]/100:.1f}×", (mts[-1], idx[col].iloc[-1]), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=9, color=INK)
ax.axhline(100, color=AXIS, lw=1)
ax.set_xlim(right=mts[-1] + pd.Timedelta(days=110))
ax.legend(loc="upper left")
titled(ax, "Growth of swaps, revenue and service failures", "Index, Jan–Mar 2024 average = 100")
save(fig, "q1_indexed_growth")
growth = (q.iloc[-3:][["completed", "revenue_inr", "service_failures"]].mean() / base)
record("growth_completed_x", growth["completed"], "last 3 months vs first 3 months")
record("growth_revenue_x", growth["revenue_inr"])
record("growth_failures_x", growth["service_failures"])
record("failure_rate_first3", q.iloc[:3]["service_failures"].sum() / q.iloc[:3]["attempts"].sum())
record("failure_rate_last3", q.iloc[-3:]["service_failures"].sum() / q.iloc[-3:]["attempts"].sum())
record("contribution_per_swap_first3", (q.iloc[:3]["revenue_inr"] - q.iloc[:3][["energy_inr", "wear_inr", "site_inr"]].sum(axis=1)).sum() / q.iloc[:3]["completed"].sum())
record("contribution_per_swap_last3", (q.iloc[-3:]["revenue_inr"] - q.iloc[-3:][["energy_inr", "wear_inr", "site_inr"]].sum(axis=1)).sum() / q.iloc[-3:]["completed"].sum())
print(growth.round(2).to_dict())

# %% [markdown]
# **Where did the margin go?** A bridge from the first three months to the last three months, per
# completed swap. Each bar is the change in one line of the per-swap P&L.

# %%
def per_swap(block):
    c = block["completed"].sum()
    return pd.Series({k: block[f"{k}_inr"].sum() / c for k in ["revenue", "energy", "wear", "site"]})


p0, p1 = per_swap(q.iloc[:3]), per_swap(q.iloc[-3:])
m0 = p0["revenue"] - p0[["energy", "wear", "site"]].sum()
m1 = p1["revenue"] - p1[["energy", "wear", "site"]].sum()
steps = [("Margin, Jan–Mar 2024", m0, "total"), ("Revenue per swap", p1["revenue"] - p0["revenue"], "delta"),
         ("Energy cost", -(p1["energy"] - p0["energy"]), "delta"), ("Battery wear", -(p1["wear"] - p0["wear"]), "delta"),
         ("Site cost", -(p1["site"] - p0["site"]), "delta"), ("Margin, Apr–Jun 2025", m1, "total")]
fig, ax = plt.subplots(figsize=(9.5, 4.2))
run = 0.0
for i, (lab, v, kind) in enumerate(steps):
    if kind == "total":
        ax.bar(i, v, width=0.55, color=INK2)
        run = v
        y = v
    else:
        ax.bar(i, v, bottom=run, width=0.55, color=GOOD if v >= 0 else CRITICAL)
        run += v
        y = run if v >= 0 else run - v
    ax.annotate(f"{'+' if (kind == 'delta' and v > 0) else ''}₹{v:,.1f}", (i, max(y, run if kind == 'delta' else v)),
                xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, color=INK)
ax.axhline(0, color=INK2, lw=1)
ax.set_xticks(range(len(steps)), [s[0] for s in steps], fontsize=8.5)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"₹{v:,.0f}"))
titled(ax, "Per-swap margin bridge: what moved between the first and last quarter",
       "₹ per completed swap; green raised margin, red reduced it")
save(fig, "q1_margin_bridge")
bridge = pd.DataFrame(steps, columns=["step", "inr_per_swap", "kind"])
display(bridge)
for lab, v, kind in steps:
    record(f"bridge::{lab}", v)
