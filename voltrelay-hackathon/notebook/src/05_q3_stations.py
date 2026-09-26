# %% [markdown]
# ## 6 · Core question 3 — Station and geographic patterns
#
# *Stations differ in charger generation, location type and commissioning date, and are unevenly spread
# across cities. How do these differences relate to service, battery turnaround and complaints?*
#
# Each attempt is matched to its station's telemetry for that hour (ambient temperature, charged stock,
# charge time), so the heat a cabinet was working in is known for every swap.

# %%
tel_cols = ["station_id", "hour_start", "ambient_temp_c", "cabinet_temp_c", "charged_2w_min", "charged_3w_min",
            "packs_quarantined", "chargers_online", "avg_charge_minutes", "outage_minutes"]
sw["hour_start"] = sw["event_ts"].dt.floor("h")
sw = sw.merge(hr[tel_cols].assign(station_id=lambda d: d["station_id"].astype("category")),
              on=["station_id", "hour_start"], how="left")
TEMP_BINS = [-50, 25, 30, 35, 40, 45, 80]
TEMP_LABELS = ["<25°C", "25–30", "30–35", "35–40", "40–45", "≥45°C"]
sw["temp_band"] = pd.cut(sw["ambient_temp_c"], TEMP_BINS, labels=TEMP_LABELS, right=False)
hr["temp_band"] = pd.cut(hr["ambient_temp_c"], TEMP_BINS, labels=TEMP_LABELS, right=False)
print(f"Attempts matched to a measured station-hour temperature: {sw['ambient_temp_c'].notna().mean():.1%}")

# %%
def seg_table(col):
    t = sw.groupby(col, observed=True).agg(stations=("station_id", "nunique"), attempts=("service_failure", "size"),
                                           service_failure_rate=("service_failure", "mean"),
                                           stockout_rate=("stockout", "mean"),
                                           median_wait_min=("queue_wait_sec", lambda s: s.median() / 60))
    return t


tk_st = tk[tk["station_id"].notna()].merge(st[["station_id", "charger_generation", "location_type", "expansion_wave", "city"]],
                                             on="station_id", how="left")
for col in ["charger_generation", "location_type", "expansion_wave", "city"]:
    t = seg_table(col)
    t["tickets_per_1k_attempts"] = tk_st.groupby(col).size().reindex(t.index) / t["attempts"] * 1000
    display(t.style.format({"attempts": "{:,.0f}", "service_failure_rate": "{:.2%}", "stockout_rate": "{:.2%}",
                            "median_wait_min": "{:.1f}", "tickets_per_1k_attempts": "{:.1f}"}))

# %% [markdown]
# **Where are the charger generations?** If older cabinets sit in particular cities, a "city problem"
# might really be an equipment problem — or the reverse.

# %%
gen_city = pd.crosstab(st_open["city"], st_open["charger_generation"]).reindex(CITY_ORDER).fillna(0).astype(int)
heat_city = city_daily.groupby("city")["max_temp_c"].agg(
    summer_avg_max=lambda s: s[city_daily.loc[s.index, "date"].dt.month.isin([4, 5, 6])].mean(),
    days_above_40=lambda s: (s >= 40).sum())
display(gen_city.join(heat_city))

# %% [markdown]
# **Turnaround and stock in the heat.** Hour-level telemetry shows what happens inside the cabinet as it
# gets hotter: how long a pack takes to charge, and how often the cabinet has no charged 2W pack at all.

# %%
measured = hr[hr["avg_charge_minutes"].notna() & hr["temp_band"].notna()]
turn = measured.groupby(["temp_band", "charger_generation"], observed=True)["avg_charge_minutes"].median().unstack()
stock_meas = hr[hr["charged_2w_min"].notna() & hr["temp_band"].notna() & (hr["slots_2w"] > 0)]
empty = stock_meas.assign(empty=stock_meas["charged_2w_min"] <= 0).groupby(
    ["temp_band", "charger_generation"], observed=True)["empty"].mean().unstack()
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for gen in GEN_ORDER:
    if gen in turn.columns:
        axes[0].plot(range(len(turn)), turn[gen], color=GEN_COLOR[gen], marker="o", ms=5, label=gen,
                     markeredgecolor=SURFACE, markeredgewidth=1.5)
        axes[1].plot(range(len(empty)), empty[gen], color=GEN_COLOR[gen], marker="o", ms=5, label=gen,
                     markeredgecolor=SURFACE, markeredgewidth=1.5)
for ax in axes:
    ax.set_xticks(range(len(turn)), turn.index.astype(str))
    ax.set_xlabel("Ambient temperature")
    ax.legend(loc="upper left")
axes[1].yaxis.set_major_formatter(pct_axis)
titled(axes[0], "Median charge time (minutes)", "By ambient temperature and charger generation")
titled(axes[1], "Hours with zero charged 2W packs", "Share of measured station-hours")
fig.tight_layout()
save(fig, "q3_heat_by_generation")
display(turn.round(0))
display(empty.style.format("{:.1%}"))
for gen in turn.columns:
    record(f"charge_min::{gen}::<25", turn[gen].iloc[0])
    record(f"charge_min::{gen}::hottest", turn[gen].dropna().iloc[-1])
    record(f"empty_hours::{gen}::hottest", empty[gen].dropna().iloc[-1] if gen in empty else np.nan)

# %% [markdown]
# **Separating equipment, heat and city.** Gen1 cabinets, hot weather and certain cities overlap, so a
# simple table cannot say which one matters. A logistic model of service failure on all of them together
# (plus hour of day and vehicle class) holds the others constant. Attempts are aggregated into cells
# first, which gives the same estimates as a row-level model but runs in seconds.

# %%
sw["hour_bucket"] = pd.cut(sw["hour"], [-1, 5, 9, 12, 16, 20, 23],
                           labels=["00–05", "06–09", "10–12", "13–16", "17–20", "21–23"])
sw["hot"] = sw["temp_band"].isin(["40–45", "≥45°C"])
cells = (sw.dropna(subset=["temp_band"])
         .groupby(["charger_generation", "hot", "city", "hour_bucket", "vehicle_class", "location_type"], observed=True)
         .agg(n=("service_failure", "size"), k=("service_failure", "sum")).reset_index())
cells = cells[cells["n"] > 0]
cells["rate"] = cells["k"] / cells["n"]
for c in ["charger_generation", "city", "hour_bucket", "vehicle_class", "location_type"]:
    cells[c] = cells[c].astype(str)
cells["hot"] = cells["hot"].astype(int)
fit = smf.glm("rate ~ C(charger_generation, Treatment('Gen2')) * hot + C(city) + C(hour_bucket) + C(vehicle_class)"
              " + C(location_type)", data=cells, family=sm.families.Binomial(), freq_weights=cells["n"]).fit()
orr = pd.DataFrame({"odds_ratio": np.exp(fit.params), "ci_low": np.exp(fit.conf_int()[0]),
                    "ci_high": np.exp(fit.conf_int()[1]), "p_value": fit.pvalues})
display(orr[orr.index.str.contains("generation|hot")].style.format("{:.3f}"))

# %%
def pred_rate(gen, hot):
    d = cells.copy()
    d["charger_generation"], d["hot"] = gen, hot
    return np.average(fit.predict(d), weights=d["n"])


grid = pd.DataFrame({h: {g: pred_rate(g, int(h == "Hot (≥40°C)")) for g in GEN_ORDER}
                     for h in ["Normal (<40°C)", "Hot (≥40°C)"]})
display(grid.style.format("{:.2%}"))
record("adj_fail::Gen1::hot", grid.loc["Gen1", "Hot (≥40°C)"], "model-adjusted service-failure rate")
record("adj_fail::Gen1::normal", grid.loc["Gen1", "Normal (<40°C)"])
record("adj_fail::Gen2::hot", grid.loc["Gen2", "Hot (≥40°C)"])
record("adj_fail::Gen2::normal", grid.loc["Gen2", "Normal (<40°C)"])

# %% [markdown]
# **Expansion waves: did new stations go where riders needed them?** For every zone, the service-failure
# rate at existing stations in the three months *before* a wave is compared between zones that received
# new stations and zones that did not. If the waves targeted pain, the receiving zones should have been
# the ones failing.

# %%
wave_rows = []
for wave in [w for w in st_open["expansion_wave"].dropna().unique() if not str(w).lower().startswith("launch")]:
    new = st_open[st_open["expansion_wave"] == wave]
    w0 = new["commissioned_date"].min()
    pre = sw[(sw["event_ts"] >= w0 - pd.DateOffset(months=3)) & (sw["event_ts"] < w0) &
             (sw["commissioned_date"] < w0)]
    z = pre.groupby("zone", observed=True).agg(attempts=("service_failure", "size"), rate=("service_failure", "mean"))
    z["got_new_station"] = z.index.isin(new["zone"])
    post_new = sw[sw["station_id"].astype(str).isin(new["station_id"]) & (sw["event_ts"] < w0 + pd.DateOffset(months=4))]
    days = max(1, (min(sw["event_ts"].max(), w0 + pd.DateOffset(months=4)) - w0).days)
    slots = new["slots_2w"].sum() + new["slots_3w"].sum()
    wave_rows.append({
        "wave": wave, "new stations": len(new), "first opened": w0.date(),
        "pre-wave failure rate, zones that got stations": np.average(z.loc[z["got_new_station"], "rate"], weights=z.loc[z["got_new_station"], "attempts"]) if z["got_new_station"].any() else np.nan,
        "pre-wave failure rate, zones that did not": np.average(z.loc[~z["got_new_station"], "rate"], weights=z.loc[~z["got_new_station"], "attempts"]) if (~z["got_new_station"]).any() else np.nan,
        "new stations in top-quartile failing zones": int(new["zone"].isin(z[z["rate"] >= z["rate"].quantile(0.75)].index).sum()),
        "swaps per slot per day, first 4 months": post_new["completed"].sum() / max(1, slots) / days,
        "host types": new["host_type"].value_counts().to_dict(),
    })
wave_eval = pd.DataFrame(wave_rows).set_index("wave")
display(wave_eval)
network_util = sw["completed"].sum() / (st_open["slots_2w"].sum() + st_open["slots_3w"].sum()) / sw["date"].nunique()
print(f"For comparison, network-wide swaps per slot per day: {network_util:.2f}")
