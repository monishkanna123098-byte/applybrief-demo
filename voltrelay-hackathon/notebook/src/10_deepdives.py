# %% [markdown]
# ### 10.4 Support tickets: does the free text add anything?
# The re-labelling in 3.8 found battery and range complaints hiding under generic categories. Two checks:
# do those complaints track the outlier battery lots, and how does the complaint mix move over time?

# %%
tk["battery_lot"] = tk["battery_id"].map(bat.set_index("battery_id")["manufacturing_lot"])
with_pack = tk[tk["battery_lot"].notna()]
lot_view = with_pack.assign(bad_lot=with_pack["battery_lot"].isin(BAD_LOTS)).groupby("bad_lot").agg(
    tickets=("ticket_id", "size"), battery_or_range_share=("mentions_battery", "mean"))
display(lot_view.rename(index={True: "outlier-lot pack", False: "other pack"}))
theme = tk.groupby([tk["created_ts"].dt.to_period("Q"), "category_adj"]).size().unstack(fill_value=0)
theme = theme.div(theme.sum(axis=1), axis=0)
theme.index = theme.index.astype(str)
display(theme.style.format("{:.0%}"))

# %% [markdown]
# ### 10.5 Anomalies worth flagging separately
# Station-months whose failure rate is extreme relative to the network that month (robust z-score above
# 4), plus the irregularities already found during cleaning.

# %%
sm_ = sw.groupby(["station_id", "month"], observed=True).agg(attempts=("service_failure", "size"),
                                                             rate=("service_failure", "mean")).reset_index()
sm_ = sm_[sm_["attempts"] >= 200]
med_m = sm_.groupby("month")["rate"].transform("median")
mad_m = sm_.groupby("month")["rate"].transform(lambda s: (s - s.median()).abs().median() * 1.4826).replace(0, np.nan)
sm_["z"] = (sm_["rate"] - med_m) / mad_m
extreme = sm_[sm_["z"] > 4].merge(st[["station_id", "city", "charger_generation", "location_type", "connectivity_tier"]],
                                  on="station_id", how="left").sort_values("z", ascending=False)
print(f"{len(extreme)} extreme station-months across {extreme['station_id'].nunique()} stations")
display(extreme.head(15))
anomalies = pd.DataFrame([
    ("Test stations booking transactions", RESULTS["test_station_events"]["value"], "Exclude from reporting; audit why revenue is booked"),
    ("Firmware v3.2.0 clock bug", RESULTS["clock_bug_events"]["value"], "Timestamps corrected; fix firmware time zone"),
    ("Offline-sync duplicate records", RESULTS["duplicates_removed"]["value"], "De-duplicated; add idempotent event IDs at sync"),
    ("Packs issued after recorded retirement", RESULTS["retired_packs_issued"]["value"], "Safety check: block retired IDs at the cabinet"),
    ("Extreme station-months (z > 4)", len(extreme), "Review with operations; most are listed above"),
], columns=["anomaly", "rows / cases", "action"])
display(anomalies)
anomalies.to_csv(OUT_DIR / "anomalies.csv", index=False)
