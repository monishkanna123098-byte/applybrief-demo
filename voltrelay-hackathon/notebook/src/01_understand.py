# %% [markdown]
# ## 2 · Data understanding
#
# Before cleaning anything: what does each table cover, do the keys join, and what does a swap attempt
# look like? The swap table is the fact table; everything else describes the stations, riders, packs,
# partners and conditions around each attempt.

# %%
def coverage(df, col):
    s = df[col].dropna()
    return f"{s.min():%Y-%m-%d} → {s.max():%Y-%m-%d}" if len(s) else "n/a"


overview = pd.DataFrame([
    ("swap_events", len(swaps), swaps.shape[1], "event_id", coverage(swaps, "event_ts")),
    ("station_hourly_status", len(hourly), hourly.shape[1], "station_id + hour_start", coverage(hourly, "hour_start")),
    ("riders", len(riders), riders.shape[1], "rider_id", coverage(riders, "signup_date")),
    ("batteries", len(batteries), batteries.shape[1], "battery_id", coverage(batteries, "commission_date")),
    ("support_tickets", len(tickets), tickets.shape[1], "ticket_id", coverage(tickets, "created_ts")),
    ("stations", len(stations), stations.shape[1], "station_id", coverage(stations, "commissioned_date")),
    ("city_daily_context", len(city_daily), city_daily.shape[1], "city + date", coverage(city_daily, "date")),
    ("fleet_partners", len(partners), partners.shape[1], "partner_id", coverage(partners, "contract_start_date")),
], columns=["table", "rows", "columns", "primary key", "date coverage"])
overview

# %% [markdown]
# **Do the keys join?** Every foreign key is checked against its parent table. A key that does not
# resolve would silently drop rows in a join, so it is better to know now.

# %%
def fk_share(child, col, parent, pcol):
    vals = child[col].dropna().astype(str)
    return vals.isin(set(parent[pcol].astype(str))).mean() if len(vals) else np.nan


fk = pd.DataFrame([
    ("swap_events.rider_id → riders", fk_share(swaps, "rider_id", riders, "rider_id")),
    ("swap_events.station_id → stations", fk_share(swaps, "station_id", stations, "station_id")),
    ("swap_events.battery_in_id → batteries", fk_share(swaps, "battery_in_id", batteries, "battery_id")),
    ("swap_events.battery_out_id → batteries", fk_share(swaps, "battery_out_id", batteries, "battery_id")),
    ("riders.partner_id → fleet_partners", fk_share(riders, "partner_id", partners, "partner_id")),
    ("support_tickets.rider_id → riders", fk_share(tickets, "rider_id", riders, "rider_id")),
    ("support_tickets.station_id → stations", fk_share(tickets, "station_id", stations, "station_id")),
    ("support_tickets.battery_id → batteries", fk_share(tickets, "battery_id", batteries, "battery_id")),
    ("station_hourly_status.station_id → stations", fk_share(hourly, "station_id", stations, "station_id")),
], columns=["relationship", "share of non-blank keys that resolve"])
fk

# %% [markdown]
# **What happens at a cabinet.** The outcome mix of all 3.9M attempts, and which fields are filled for
# each outcome (the data dictionary says pricing and battery fields depend on the outcome — this checks it).

# %%
outcome = swaps["event_type"].value_counts().rename("attempts").to_frame()
outcome["share"] = outcome["attempts"] / outcome["attempts"].sum()
fill = swaps.groupby("event_type", observed=True)[
    ["battery_in_id", "battery_out_id", "soh_in_pct", "soh_out_pct", "km_since_last_swap",
     "amount_charged_inr", "energy_to_recharge_kwh"]].agg(lambda s: s.notna().mean())
display(outcome)
display(fill.style.format("{:.0%}"))
charged_on_fail = swaps.loc[swaps["event_type"] != "swap_completed", "amount_charged_inr"].fillna(0)
print(f"Revenue recorded on non-completed attempts: ₹{charged_on_fail.sum():,.0f} "
      f"across {int((charged_on_fail > 0).sum()):,} attempts")

# %%
print("Categorical fields in swap_events:")
for c in ["tariff_code", "payment_mode", "sync_mode", "station_firmware", "attempt_seq"]:
    if c in swaps.columns:
        print(f"  {c:<18}", swaps[c].value_counts(dropna=False).head(8).to_dict())
print("\nStations:")
for c in ["city", "charger_generation", "location_type", "host_type", "expansion_wave", "connectivity_tier"]:
    print(f"  {c:<20}", stations[c].value_counts(dropna=False).to_dict())
print("\nBatteries by supplier:", batteries["supplier"].value_counts().to_dict())
print("Riders by vehicle class:", riders["vehicle_class"].value_counts().to_dict(),
      "| by plan:", riders["plan_type"].value_counts().to_dict())
display(partners)
