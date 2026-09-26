# %% [markdown]
# ## 3 · Data cleaning
#
# The organiser documents nine data-quality characteristics. Each one is handled the same way:
# **detect** it in the data, **measure** how big it is, **decide** what to do and why, **apply** the fix,
# and **re-check** that the fix worked. Every decision is logged in `CLEAN_LOG`, which is printed at the
# end of this section so the whole cleaning trail can be audited in one table.

# %%
CLEAN_LOG = []


def log(step, rows, decision):
    CLEAN_LOG.append({"issue": step, "rows affected": int(rows), "decision": decision})


st = stations.copy()
st["station_id"] = st["station_id"].astype(str)
st["is_test"] = st["station_id"].str.startswith("STN-TST")
sw = swaps.copy()
sw["station_id"] = sw["station_id"].astype(str).astype("category")

# %% [markdown]
# ### 3.1 Test stations
# Internal test cabinets (`STN-TST…`) are not customer-facing, so they must not count towards network
# performance. First, how big are they really?

# %%
test_ids = set(st.loc[st["is_test"], "station_id"])
is_test_ev = sw["station_id"].isin(test_ids)
test_view = (sw[is_test_ev].groupby("station_id", observed=True)
             .agg(attempts=("event_type", "size"), revenue_inr=("amount_charged_inr", "sum"),
                  first_event=("event_ts", "min"), last_event=("event_ts", "max")))
display(test_view)
print("Hour-of-day of test-station events:",
      sw.loc[is_test_ev, "event_ts"].dt.hour.value_counts().sort_index().to_dict())
record("test_station_events", int(is_test_ev.sum()), "events at STN-TST stations (excluded)")
record("test_station_revenue_inr", float(sw.loc[is_test_ev, "amount_charged_inr"].sum()))
log("Test stations (STN-TST*)", is_test_ev.sum(), "Excluded from every network metric; kept aside for the anomaly list")
sw = sw.loc[~is_test_ev].copy()
sw["station_id"] = sw["station_id"].cat.remove_unused_categories()

# %% [markdown]
# ### 3.2 Offline-sync near-duplicates
# A cabinet with a weak connection can upload the same attempt twice. A duplicate is the same rider,
# station, returned pack, issued pack and outcome, logged again within a few minutes, where at least one
# copy came through `offline_batch` sync. A genuine retry after a failure would differ in outcome or pack,
# so it is not caught by this rule.

# %%
sw = sw.sort_values(["rider_id", "event_ts"], kind="mergesort").reset_index(drop=True)
codes = {c: sw[c].cat.codes.to_numpy() for c in ["rider_id", "station_id", "battery_in_id", "battery_out_id", "event_type"]}
ts_sec = sw["event_ts"].to_numpy().astype("datetime64[s]").astype(np.int64)
gap = np.diff(ts_sec)
same = np.ones(len(gap), dtype=bool)
for c in codes.values():
    same &= c[1:] == c[:-1]
offline = sw["sync_mode"].astype(str).eq("offline_batch").to_numpy()
either_offline = offline[1:] | offline[:-1]
DUP_WINDOW_SEC = 180
cand = same & (gap >= 0) & (gap <= 600)
print("Identical-attempt pairs by gap (seconds), split by whether a copy was offline-synced:")
gap_tab = pd.crosstab(pd.cut(gap[cand], [-1, 10, 30, 60, 120, 180, 300, 600]), either_offline[cand],
                      colnames=["offline copy"])
display(gap_tab)
is_dup = np.r_[False, same & either_offline & (gap >= 0) & (gap <= DUP_WINDOW_SEC)]
dup_by_tier = (pd.DataFrame({"tier": sw["station_id"].map(st.set_index("station_id")["connectivity_tier"]).astype(str),
                             "dup": is_dup}).groupby("tier")["dup"].agg(["sum", "mean"]))
display(dup_by_tier.rename(columns={"sum": "duplicates", "mean": "share of attempts"}))
record("duplicates_removed", int(is_dup.sum()))
log("Offline-sync near-duplicates", is_dup.sum(),
    f"Dropped the later copy when identical within {DUP_WINDOW_SEC}s and one copy was offline-synced")
sw = sw.loc[~is_dup].reset_index(drop=True)

# %% [markdown]
# ### 3.3 Firmware v3.2.0 clock bug
# Stations on firmware v3.2.0 logged times about 5 h 30 m early between 10 March and 14 April 2025 —
# exactly the IST offset, i.e. local time was written as if it were UTC. If uncorrected, a morning rush
# looks like a 3 a.m. rush. The check below compares the hour-of-day profile of those events with the
# same stations' profile in the four weeks before, and with all other stations in the same window.

# %%
fw = sw["station_firmware"].astype(str)
in_window = (sw["event_ts"] >= "2025-03-10") & (sw["event_ts"] < "2025-04-15")
affected = fw.str.contains("3.2.0", regex=False) & in_window
aff_stations = set(sw.loc[affected, "station_id"].astype(str))
before = sw["station_id"].astype(str).isin(aff_stations) & (sw["event_ts"] >= "2025-02-10") & (sw["event_ts"] < "2025-03-10")
others = ~sw["station_id"].astype(str).isin(aff_stations) & in_window


def hour_profile(mask, shift=pd.Timedelta(0)):
    h = (sw.loc[mask, "event_ts"] + shift).dt.hour
    return h.value_counts(normalize=True).reindex(range(24), fill_value=0)


prof = pd.DataFrame({
    "v3.2.0 stations, logged time": hour_profile(affected),
    "v3.2.0 stations, corrected (+5h30m)": hour_profile(affected, pd.Timedelta(hours=5, minutes=30)),
    "same stations, 4 weeks before": hour_profile(before),
    "all other stations, same window": hour_profile(others),
})
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(prof.index, prof.iloc[:, 0], color=RED, label=prof.columns[0])
ax.plot(prof.index, prof.iloc[:, 1], color=BLUE, label=prof.columns[1])
ax.plot(prof.index, prof.iloc[:, 3], color=DEEMPH, label=prof.columns[3])
ax.yaxis.set_major_formatter(pct_axis)
ax.set_xticks(range(0, 24, 3))
ax.set_xlabel("Hour of day")
ax.legend(loc="upper left", ncol=3, bbox_to_anchor=(0, -0.18))
titled(ax, "Hour-of-day profile at v3.2.0 stations: as logged vs corrected",
       "Share of attempts by hour of day, 10 Mar – 14 Apr 2025")
save(fig, "03_clock_bug")
corr_raw = prof.iloc[:, 0].corr(prof.iloc[:, 3])
corr_fix = prof.iloc[:, 1].corr(prof.iloc[:, 3])
print(f"{affected.sum():,} events at {len(aff_stations)} stations affected. Correlation with the normal "
      f"hourly profile: {corr_raw:.2f} as logged → {corr_fix:.2f} after the +5h30m correction.")
record("clock_bug_events", int(affected.sum()))
record("clock_bug_stations", len(aff_stations))
sw.loc[affected, "event_ts"] = sw.loc[affected, "event_ts"] + pd.Timedelta(hours=5, minutes=30)
sw["clock_corrected"] = affected.to_numpy()
log("Firmware v3.2.0 timestamp bug", affected.sum(), "Shifted +5h30m; verified by the hourly-profile match")

# %% [markdown]
# ### 3.4 City spellings
# `riders.home_city` spells the same city several ways. Each variant is mapped to one of the six
# network cities with explicit rules, and the full mapping is printed so it can be checked by eye.

# %%
CITY_RULES = [
    ("Bengaluru", ("beng", "bang", "blr", "blore")),
    ("Delhi NCR", ("delhi", "ncr", "gurg", "noida", "dilli", "ndls", "del")),
    ("Hyderabad", ("hyd", "secunderabad", "cyberabad")),
    ("Pune", ("pune", "poona", "pnq")),
    ("Mumbai", ("mum", "bom", "bombay", "thane", "navimumbai")),
    ("Jaipur", ("jaipur", "jaipr", "jpr", "jai")),
]


def canon_city(raw):
    key = re.sub(r"[^a-z]", "", str(raw).lower())
    if not key or key == "nan":
        return np.nan
    for city, keys in CITY_RULES:
        if any(key.startswith(k) or k in key for k in keys):
            return city
    return np.nan


city_map = riders["home_city"].value_counts(dropna=False).rename("riders").to_frame()
city_map["standardised"] = [canon_city(v) for v in city_map.index]
display(city_map)
unmapped = city_map["standardised"].isna() & city_map.index.notna()
if unmapped.any():
    print("Spellings left unmapped (kept as missing):", list(city_map.index[unmapped]))
riders["home_city_clean"] = riders["home_city"].map(canon_city)
variants = riders["home_city"].nunique()
record("city_spelling_variants", variants)
log("Inconsistent city spellings", (riders["home_city"] != riders["home_city_clean"]).sum(),
    f"{variants} spellings mapped to 6 cities by explicit rules")
print("Station and city-context tables use clean names:",
      sorted(st["city"].unique()) == sorted(city_daily["city"].unique()))

# %% [markdown]
# ### 3.5 Outliers in distance and charge readings
# Negative distances and implausibly long ones are odometer resets, not real riding; charge readings a
# little above 100% are sensor drift. The plausible ceiling for distance is set from the data itself
# (1.5× the 99th percentile for each vehicle class), not guessed.

# %%
vclass = sw["rider_id"].map(riders.set_index("rider_id")["vehicle_class"]).astype(str)
km = sw["km_since_last_swap"]
km_cap = km[km > 0].groupby(vclass[km > 0]).quantile(0.99) * 1.5
bad_km = (km < 0) | (km > vclass.map(km_cap).astype(float))
print("Distance ceiling by vehicle class (km):", km_cap.round(1).to_dict())
print(f"Invalid distances set to missing: {bad_km.sum():,} ({bad_km.mean():.2%} of attempts)")
sw.loc[bad_km, "km_since_last_swap"] = np.nan
over100 = 0
for c in ["soc_in_pct", "soh_in_pct", "soc_out_pct", "soh_out_pct"]:
    n = int((sw[c] > 100).sum())
    over100 += n
    sw[c] = sw[c].clip(upper=100)
neg_wait = int((sw["queue_wait_sec"] < 0).sum())
sw.loc[sw["queue_wait_sec"] < 0, "queue_wait_sec"] = np.nan
log("Distance outliers (odometer resets)", bad_km.sum(), "Negative or > 1.5× p99 set to missing, not dropped")
log("Charge readings above 100%", over100, "Clipped to 100 (calibration drift)")
if neg_wait:
    log("Negative queue waits", neg_wait, "Set to missing")

# %% [markdown]
# ### 3.6 Missing telemetry
# Blank telemetry means *not measured*, not zero. Treating a blank stock reading as zero would invent
# stockouts. The table shows where blanks sit: they concentrate at poor-connectivity stations, so every
# telemetry metric below is computed over measured hours only.

# %%
hr = hourly.copy()
hr["station_id"] = hr["station_id"].astype(str)
hr = hr[~hr["station_id"].isin(test_ids)].merge(
    st[["station_id", "city", "charger_generation", "location_type", "expansion_wave", "connectivity_tier",
        "slots_2w", "slots_3w", "inventory_target_2w", "inventory_target_3w"]], on="station_id", how="left")
tele = pd.crosstab(hr["connectivity_tier"], hr["telemetry_status"], normalize="index")
display(tele.style.format("{:.1%}"))
blank_stock = hr["charged_2w_min"].isna()
log("Missing telemetry", blank_stock.sum(), "Never zero-filled; metrics use measured hours only")

# %% [markdown]
# ### 3.7 Satisfaction scores are not missing at random
# CSAT is recorded far more often when a ticket is resolved quickly. A plain average therefore
# over-represents the happiest tickets. The check below shows the response-rate gradient and compares the
# naive average with one re-weighted so that each resolution-time band counts in proportion to its tickets.

# %%
tk = tickets.copy()
tk["res_band"] = pd.cut(tk["resolution_hours"], [-0.01, 4, 12, 24, 48, 96, np.inf],
                        labels=["≤4h", "4–12h", "12–24h", "1–2d", "2–4d", ">4d"])
csat_tab = tk.groupby("res_band", observed=True).agg(
    tickets=("ticket_id", "size"), csat_response_rate=("csat_score", lambda s: s.notna().mean()),
    mean_csat_when_given=("csat_score", "mean"))
display(csat_tab)
w = csat_tab["tickets"] / csat_tab["tickets"].sum()
naive = tk["csat_score"].mean()
reweighted = (csat_tab["mean_csat_when_given"] * w).sum() / w[csat_tab["mean_csat_when_given"].notna()].sum()
print(f"CSAT recorded on {tk['csat_score'].notna().mean():.1%} of tickets. Naive mean {naive:.2f} vs "
      f"re-weighted {reweighted:.2f}; unresolved tickets have no CSAT at all.")
record("csat_naive", naive)
record("csat_reweighted", reweighted)
log("CSAT missing not at random", tk["csat_score"].isna().sum(),
    "Not used as a KPI; quoted only re-weighted by resolution time")

# %% [markdown]
# ### 3.8 Ticket categories are not always right
# Agents file some battery and range complaints under *other* or *app_issue*, and riders often describe
# a weak battery as a problem with the vehicle ("gaadi ka pickup kam hai"). A transparent keyword rule
# (English and Hinglish) flags comments that are really about range or charge. It is deliberately simple
# so it can be audited; sample matches are printed.

# %%
BATTERY_TERMS = [
    r"\bbatt?(e|a)ry\b", r"\brange\b", r"\bcharg", r"\bbackup\b", r"\bdrain", r"\bsoc\b", r"\bpercent|%",
    r"\bkm\b", r"\bkilomet", r"\bkhatam\b", r"\bjaldi\b.*\b(khatam|down|low|ho)", r"\bdischarg",
    r"\bpick ?up\b", r"\bpower\b", r"\bdead\b", r"\bweak\b", r"\bjaldi\b", r"\bkam\b.*\b(chal|range|km)",
    r"\b(ruk|band) (gayi|gaya|ho)", r"\bnahi chal", r"\bchal nahi", r"\bslow\b", r"\bheat|garam|hot\b",
]
battery_re = re.compile("|".join(BATTERY_TERMS), flags=re.IGNORECASE)
tk["mentions_battery"] = tk["rider_comment"].fillna("").str.contains(battery_re)
cat_check = tk.groupby("category").agg(tickets=("ticket_id", "size"),
                                       with_comment=("rider_comment", lambda s: s.notna().mean()),
                                       mentions_battery_or_range=("mentions_battery", "mean"))
display(cat_check.sort_values("tickets", ascending=False))
for c in ["other", "app_issue"]:
    ex = tk.loc[(tk["category"] == c) & tk["mentions_battery"], "rider_comment"].head(4).tolist()
    print(f"Sample '{c}' comments flagged as battery/range: {ex}")
generic = tk["category"].isin(["other", "app_issue"])
tk["category_adj"] = np.where(generic & tk["mentions_battery"], "battery_or_range (re-labelled)", tk["category"])
record("tickets_relabelled", int((generic & tk["mentions_battery"]).sum()))
log("Ticket category mislabelling", (generic & tk["mentions_battery"]).sum(),
    "Generic tickets whose comment is about battery/range re-labelled for analysis (original kept)")

# %% [markdown]
# ### 3.9 Outcome definitions and further checks
# Non-completed attempts are kept (they *are* the failures) but never counted as revenue or delivered
# range. The final cell looks for problems the organiser did **not** document.

# %%
et = sw["event_type"].astype(str)
sw["completed"] = et.eq("swap_completed")
sw["stockout"] = et.eq("failed_no_charged_battery")
sw["abandoned"] = et.eq("abandoned_queue")
sw["system_error"] = et.eq("failed_system_error")
sw["cancelled"] = et.eq("cancelled_by_rider")
sw["service_failure"] = sw["stockout"] | sw["abandoned"] | sw["system_error"]
sw.loc[~sw["completed"], ["amount_charged_inr", "energy_to_recharge_kwh"]] = np.nan

bat = batteries.copy()
bat["battery_id"] = bat["battery_id"].astype(str)
retired_on = sw["battery_out_id"].astype(str).map(bat.set_index("battery_id")["retired_date"])
issued_after_retire = sw["completed"] & retired_on.notna() & (sw["event_ts"] > retired_on)
decom = sw["station_id"].astype(str).map(st.set_index("station_id")["decommissioned_date"])
after_decom = decom.notna() & (sw["event_ts"] > decom)
comm = sw["station_id"].astype(str).map(st.set_index("station_id")["commissioned_date"])
before_comm = sw["event_ts"] < comm
signup = sw["rider_id"].astype(str).map(riders.set_index("rider_id")["signup_date"])
before_signup = sw["event_ts"].dt.normalize() < signup
zero_value = sw["completed"] & (sw["amount_charged_inr"].fillna(0) == 0) & (sw["tariff_code"].astype(str) != "PROMO_FREE")
extra = pd.Series({
    "packs issued after their recorded retirement date": int(issued_after_retire.sum()),
    "attempts after the station's decommission date": int(after_decom.sum()),
    "attempts before the station's commission date": int(before_comm.sum()),
    "attempts before the rider's signup date": int(before_signup.sum()),
    "completed paid swaps charged ₹0 (non-promo)": int(zero_value.sum()),
}, name="rows")
display(extra.to_frame())
record("retired_packs_issued", int(issued_after_retire.sum()), "completed swaps issuing a pack after its retired_date")
sw["retired_pack_issued"] = issued_after_retire.to_numpy()
for k, v in extra.items():
    if v:
        log(f"Undocumented: {k}", v, "Flagged and analysed in the anomaly section; rows kept")

# %% [markdown]
# **Analysis-ready tables.** Station, rider and calendar attributes are joined once onto the cleaned
# swap table, so every later cell works from the same base.

# %%
sw["station_id"] = sw["station_id"].astype(str)
st_cols = ["station_id", "city", "zone", "location_type", "host_type", "expansion_wave", "charger_generation",
           "connectivity_tier", "commissioned_date", "grid_tariff_inr_kwh"]
rd = riders.copy()
rd["rider_id"] = rd["rider_id"].astype(str)
rd_cols = ["rider_id", "partner_id", "vehicle_class", "plan_type", "signup_date", "signup_channel", "home_city_clean"]
sw["rider_id"] = sw["rider_id"].astype(str)
sw = sw.merge(st[st_cols], on="station_id", how="left").merge(rd[rd_cols], on="rider_id", how="left")
for c in ["station_id", "rider_id", "city", "zone", "location_type", "host_type", "expansion_wave",
          "charger_generation", "connectivity_tier", "partner_id", "vehicle_class", "plan_type", "signup_channel"]:
    sw[c] = sw[c].astype("category")
sw["date"] = sw["event_ts"].dt.normalize()
sw["month"] = sw["event_ts"].dt.to_period("M")
sw["hour"] = sw["event_ts"].dt.hour.astype("int8")
sw["dow"] = sw["event_ts"].dt.dayofweek.astype("int8")
sw["is_partner"] = sw["partner_id"].notna()
cd = city_daily.copy()
sw = sw.merge(cd[["city", "date", "max_temp_c", "rainfall_mm", "heat_alert", "is_public_holiday",
                  "competitor_promo_active", "ecommerce_sale_event"]].assign(city=lambda d: d["city"].astype("category")),
              on=["city", "date"], how="left")
hr["date"] = hr["hour_start"].dt.normalize()
hr["month"] = hr["hour_start"].dt.to_period("M")
hr["hour"] = hr["hour_start"].dt.hour

clean_log = pd.DataFrame(CLEAN_LOG)
display(clean_log)
print(f"Cleaned swap table: {len(sw):,} attempts ({RAW_ROWS['swap_events'] - len(sw):,} removed), "
      f"{sw['completed'].sum():,} completed swaps.")
record("attempts_clean", len(sw))
record("completed_clean", int(sw["completed"].sum()))
