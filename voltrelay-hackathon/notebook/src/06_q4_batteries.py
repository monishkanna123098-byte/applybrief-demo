# %% [markdown]
# ## 7 · Core question 4 — Battery and equipment performance
#
# *How does battery health (state of health, charge cycles, supplier, manufacturing lot) relate to
# delivered range and swap frequency? Do any equipment or supplier cohorts stand out?*

# %%
bat["months_in_service"] = ((pd.Timestamp("2025-06-30") - bat["commission_date"]).dt.days / 30.44).clip(lower=0.5)
bat["soh_loss_per_month"] = (bat["initial_soh_pct"] - bat["current_soh_pct"]) / bat["months_in_service"]
lots = bat.groupby(["supplier", "manufacturing_lot"]).agg(
    packs=("battery_id", "size"), commissioned=("commission_date", "min"),
    soh_per_100_swaps=("soh_per_swap", lambda s: 100 * s.median()),
    soh_loss_per_month=("soh_loss_per_month", "median"),
    current_soh=("current_soh_pct", "median"), wear_inr_per_swap=("wear_inr_per_swap", "median")).reset_index()
med = lots["soh_per_100_swaps"].median()
mad = (lots["soh_per_100_swaps"] - med).abs().median() * 1.4826
lots["robust_z"] = (lots["soh_per_100_swaps"] - med) / mad
lots["outlier"] = lots["robust_z"] > 3.5
BAD_LOTS = set(lots.loc[lots["outlier"], "manufacturing_lot"])
display(lots.sort_values("soh_per_100_swaps", ascending=False).head(15).style.format(
    {"soh_per_100_swaps": "{:.3f}", "soh_loss_per_month": "{:.2f}", "current_soh": "{:.1f}",
     "wear_inr_per_swap": "₹{:.2f}", "robust_z": "{:.1f}"}))
print(f"Lots degrading abnormally fast (robust z > 3.5): {sorted(BAD_LOTS)}")
record("bad_lots", ", ".join(sorted(BAD_LOTS)))
record("bad_lot_packs", int(bat["manufacturing_lot"].isin(BAD_LOTS).sum()))
ratio = lots.loc[lots["outlier"], "soh_per_100_swaps"].median() / lots.loc[~lots["outlier"], "soh_per_100_swaps"].median()
record("bad_lot_degradation_multiple", ratio, "median SoH loss per swap, outlier lots vs all other lots")

# %%
fig, ax = plt.subplots(figsize=(10, 4.4))
ls = lots.sort_values("soh_per_100_swaps").reset_index(drop=True)
colors = [ORANGE if o else DEEMPH for o in ls["outlier"]]
ax.scatter(range(len(ls)), ls["soh_per_100_swaps"], c=colors, s=36, edgecolor=SURFACE, linewidth=1.5, zorder=3)
for i, r in ls[ls["outlier"]].iterrows():
    ax.annotate(f"{r['manufacturing_lot']} ({r['supplier']})", (i, r["soh_per_100_swaps"]), xytext=(-8, 0),
                textcoords="offset points", ha="right", va="center", fontsize=8.5)
ax.set_xticks([])
ax.set_xlabel(f"{len(ls)} manufacturing lots, ranked")
ax.set_ylabel("SoH points lost per 100 swaps")
titled(ax, "Battery degradation rate by manufacturing lot", "Median per lot; highlighted lots are statistical outliers")
save(fig, "q4_lot_degradation")

# %% [markdown]
# **Degradation curves.** The state of health of packs *as they are issued to riders*, by months in
# service. This uses the swap log itself, so it shows what riders actually received.

# %%
iss = sw.loc[sw["completed"], ["battery_out_str", "event_ts", "soh_out_pct"]].merge(
    bat[["battery_id", "supplier", "manufacturing_lot", "commission_date"]], left_on="battery_out_str",
    right_on="battery_id", how="left")
iss["age_m"] = ((iss["event_ts"] - iss["commission_date"]).dt.days // 30).clip(lower=0)
iss["cohort"] = np.where(iss["manufacturing_lot"].isin(BAD_LOTS), "Outlier lots", iss["supplier"].astype(str))
curve = iss[iss["age_m"] <= 18].groupby(["cohort", "age_m"])["soh_out_pct"].median().unstack(0)
fig, ax = plt.subplots(figsize=(9.5, 4.2))
palette = {"Cellora": BLUE, "Amptek": AQUA, "Kyron": VIOLET, "Outlier lots": ORANGE}
for c in curve.columns:
    ax.plot(curve.index, curve[c], color=palette.get(c, DEEMPH), label=c)
ax.set_xlabel("Months in service")
ax.set_ylabel("Median SoH at issue (%)")
ax.legend(loc="lower left")
titled(ax, "State of health of packs handed to riders, by age", "Median SoH at issue")
save(fig, "q4_degradation_curves")

# %% [markdown]
# **Health → range → swap frequency.** A worn pack carries less energy, so the rider covers fewer
# kilometres before needing the next swap — and comes back sooner, adding load to the stations.

# %%
ret = sw[sw["battery_in_id"].notna() & sw["km_since_last_swap"].notna() & sw["soh_in_pct"].notna()]
ret = ret.assign(soh_bin=pd.cut(ret["soh_in_pct"], [0, 75, 80, 85, 90, 95, 101],
                                labels=["<75", "75–80", "80–85", "85–90", "90–95", "95–100"]))
rng = ret.groupby(["vehicle_class", "soh_bin"], observed=True)["km_since_last_swap"].agg(["median", "size"]).unstack(0)
display(rng)
sw_sorted = sw[["rider_id", "event_ts", "completed", "soh_out_pct"]].sort_values(["rider_id", "event_ts"])
nxt = sw_sorted.groupby("rider_id", observed=True)["event_ts"].shift(-1)
gap_h = (nxt - sw_sorted["event_ts"]).dt.total_seconds() / 3600
freq = pd.DataFrame({"soh_out": sw_sorted["soh_out_pct"], "gap_h": gap_h})[sw_sorted["completed"].to_numpy()]
freq = freq[(freq["gap_h"] > 0) & (freq["gap_h"] < 72)]
freq["soh_bin"] = pd.cut(freq["soh_out"], [0, 75, 80, 85, 90, 95, 101], labels=["<75", "75–80", "80–85", "85–90", "90–95", "95–100"])
gap_tab = freq.groupby("soh_bin", observed=True)["gap_h"].median()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for vc, color in [("2W", BLUE), ("3W", ORANGE)]:
    if ("median", vc) in rng.columns:
        axes[0].plot(range(len(rng)), rng[("median", vc)], color=color, marker="o", ms=5, label=vc,
                     markeredgecolor=SURFACE, markeredgewidth=1.5)
axes[0].set_xticks(range(len(rng)), rng.index.astype(str))
axes[0].set_xlabel("SoH of returned pack (%)")
axes[0].legend(loc="upper left")
titled(axes[0], "Median km covered on the pack", "By its state of health")
axes[1].plot(range(len(gap_tab)), gap_tab.values, color=BLUE, marker="o", ms=5, markeredgecolor=SURFACE, markeredgewidth=1.5)
axes[1].set_xticks(range(len(gap_tab)), gap_tab.index.astype(str))
axes[1].set_xlabel("SoH of pack received (%)")
titled(axes[1], "Median hours until the rider's next attempt", "By the health of the pack they received")
fig.tight_layout()
save(fig, "q4_range_frequency")
lo, hi = rng.iloc[0], rng.iloc[-1]
record("range_2w_low_soh", rng[("median", "2W")].dropna().iloc[0] if ("median", "2W") in rng else np.nan)
record("range_2w_high_soh", rng[("median", "2W")].dropna().iloc[-1] if ("median", "2W") in rng else np.nan)
record("gap_h_low_soh", gap_tab.dropna().iloc[0])
record("gap_h_high_soh", gap_tab.dropna().iloc[-1])

# %% [markdown]
# **Share of swaps served by the outlier lots over time**, and the packs still handed out after their
# recorded retirement date (a safety and asset-control question, not only a margin one).

# %%
iss["bad"] = iss["manufacturing_lot"].isin(BAD_LOTS)
bad_share = iss.groupby(iss["event_ts"].dt.to_period("M"))["bad"].mean()
bad_share.index = bad_share.index.astype(str)
print("Share of completed swaps using outlier-lot packs, by month:")
print((bad_share * 100).round(1).to_dict())
record("bad_lot_swap_share_last3", float(bad_share.iloc[-3:].mean()))
ret_iss = sw[sw["retired_pack_issued"]]
print(f"Swaps that issued a pack after its retirement date: {len(ret_iss):,} "
      f"({len(ret_iss)/sw['completed'].sum():.2%} of completed swaps), "
      f"involving {ret_iss['battery_out_str'].nunique():,} packs")
if len(ret_iss):
    display(ret_iss.merge(bat[["battery_id", "supplier", "manufacturing_lot", "retirement_reason"]],
                          left_on="battery_out_str", right_on="battery_id")
            .groupby(["supplier", "retirement_reason"]).size().rename("swaps").to_frame())
print("BMS firmware vs degradation (sanity check):")
display(bat.groupby("bms_firmware")["soh_per_swap"].median().to_frame("median SoH lost per swap"))
