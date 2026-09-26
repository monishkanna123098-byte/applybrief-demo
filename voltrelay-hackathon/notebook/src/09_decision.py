# %% [markdown]
# ## 10 · Decision layer — one currency for every problem
#
# Leadership is choosing where the next operating budget goes. Findings in different units (failure
# rates, SoH points, odds ratios) cannot be ranked against each other, so this section converts them into
# one currency: **rupees per year**. It answers three questions:
#
# 1. **What does one failed swap cost?** Lost revenue when the rider does not come back within the hour,
#    plus the future margin lost when an early failure makes a new rider quit.
# 2. **How many failures does each root cause create?** Counterfactual comparison against the better
#    equipment in the same city, hour and conditions.
# 3. **What is each fix worth, and what should VoltRelay be willing to pay for it?** Expressed as a
#    *break-even* value, because the data contains no vendor quotes — inventing capex numbers would be
#    guessing.

# %% [markdown]
# ### 10.1 The cost of a failed swap
# **Direct loss.** After a service failure, did the same rider complete a swap anywhere within 60 minutes?
# If yes, the revenue was recovered (late, at the rider's cost). If not, that swap's margin was lost.

# %%
seq = sw[["rider_id", "event_ts", "completed", "service_failure", "variable_margin_inr", "amount_charged_inr"]].sort_values(
    ["rider_id", "event_ts"]).reset_index(drop=True)
nxt_done = seq["event_ts"].where(seq["completed"])
nxt_done = nxt_done.groupby(seq["rider_id"], observed=True).bfill()
# bfill finds the next completed swap at or after this row; for a failure row that is strictly later
wait_to_recover = (nxt_done - seq["event_ts"]).dt.total_seconds() / 60
fails = seq[seq["service_failure"]].assign(recover_min=wait_to_recover[seq["service_failure"]])
recovered_1h = (fails["recover_min"] <= 60).mean()
margin_per_swap = sw.loc[sw["completed"], "variable_margin_inr"].mean()
revenue_per_swap = sw.loc[sw["completed"], "amount_charged_inr"].mean()
direct_cost = (1 - recovered_1h) * margin_per_swap
print(f"Failures recovered within 60 min: {recovered_1h:.1%}; median time to recover "
      f"{fails['recover_min'].median():.0f} min. Direct margin lost per failure: ₹{direct_cost:.2f}")
record("failures_recovered_1h", recovered_1h)
record("direct_cost_per_failure_inr", direct_cost)

# %% [markdown]
# **Churn loss.** From the retention model: how much does one extra early failure raise a new rider's
# probability of leaving, and what is a retained rider worth over the following year? The yearly value is
# the observed contribution of retained riders in their second to fourth months, annualised — a
# conservative choice because it ignores any growth in usage.

# %%
d_up = X.copy()
d_up["service_failure_share_14d"] = X["service_failure_share_14d"] + (
    1 / cohort["attempts_14d"].clip(lower=1)) / (cohort["service_failure_share_14d"].std() or 1)
d_up["any_stockout_14d"] = 1
churn_lift = float((logit.predict(d_up) - logit.predict(X))[cohort["any_stockout_14d"] == 0].mean())
ret_ids = cohort.index[cohort["retained"]]
later = sw[sw["rider_id"].isin(ret_ids) & (sw["days_in"] >= 30) & (sw["days_in"] < 120) & sw["completed"]]
value_3m = later.groupby("rider_id", observed=True)["contribution_inr"].sum().reindex(ret_ids).fillna(0).mean()
rider_year_value = max(value_3m, 0) * 4
churn_cost = churn_lift * rider_year_value
record("early_failure_churn_lift", churn_lift, "extra churn probability from a first stockout")
record("retained_rider_year_value_inr", rider_year_value)
record("churn_cost_per_early_failure_inr", churn_cost)
print(f"A first stockout raises churn probability by {churn_lift:.1%}. A retained rider is worth "
      f"{inr(rider_year_value)} a year in contribution, so the churn cost of that failure is {inr(churn_cost)}.")

# %%
early_fail_share = len(early[early["service_failure"]]) / max(1, sw["service_failure"].sum())
blended_cost = direct_cost + early_fail_share * churn_cost
record("blended_cost_per_failure_inr", blended_cost)
cost_table = pd.DataFrame({
    "component": ["Margin lost when the rider does not recover within 1 hour",
                  "Future margin lost when a new rider quits after an early failure (per early failure)",
                  "Blended cost of an average service failure"],
    "₹ per failure": [direct_cost, churn_cost, blended_cost]})
display(cost_table.style.format({"₹ per failure": "₹{:,.2f}"}))

# %% [markdown]
# ### 10.2 Failures by root cause
# **Equipment × heat.** For every Gen1 attempt, the model from Core question 3 predicts the failure rate
# the same attempt would have had on a Gen2 cabinet (same city, hour, vehicle, location type and heat).
# The difference, summed, is the number of failures attributable to Gen1 hardware.

# %%
g1 = sw[sw["charger_generation"].astype(str).eq("Gen1") & sw["temp_band"].notna()]
g1_cells = (g1.groupby(["hot", "city", "hour_bucket", "vehicle_class", "location_type"], observed=True)
            .agg(n=("service_failure", "size"), k=("service_failure", "sum")).reset_index())
for c in ["city", "hour_bucket", "vehicle_class", "location_type"]:
    g1_cells[c] = g1_cells[c].astype(str)
g1_cells["hot"] = g1_cells["hot"].astype(int)
as_gen1 = fit.predict(g1_cells.assign(charger_generation="Gen1"))
as_gen2 = fit.predict(g1_cells.assign(charger_generation="Gen2"))
excess = ((as_gen1 - as_gen2) * g1_cells["n"])
excess_hot = excess[g1_cells["hot"] == 1].sum()
excess_all = excess.sum()
n_gen1 = int(st_open["charger_generation"].eq("Gen1").sum())
years = (sw["event_ts"].max() - sw["event_ts"].min()).days / 365.25
record("gen1_excess_failures", excess_all, "failures vs Gen2 counterfactual, whole period")
record("gen1_excess_failures_hot", excess_hot)
record("gen1_excess_share_of_all_failures", excess_all / sw["service_failure"].sum())
gen1_value_per_cabinet_year = excess_all / years / max(1, n_gen1) * blended_cost
record("gen1_breakeven_inr_per_cabinet_year", gen1_value_per_cabinet_year)
print(f"Gen1 cabinets caused ~{excess_all:,.0f} excess failures ({excess_all / sw['service_failure'].sum():.0%} of all "
      f"service failures); {excess_hot / max(excess_all, 1):.0%} of them in ≥40°C hours. Across {n_gen1} Gen1 cabinets that "
      f"is worth {inr(gen1_value_per_cabinet_year)} per cabinet per year — the most VoltRelay should pay per year to fix one.")

# %% [markdown]
# **Battery lots.** The excess wear cost of the outlier lots compared with every other pack of the same
# type, and what replacing them early would save.

# %%
bat["pack_type"] = bat["pack_type"].astype(str)
normal_wear = bat[~bat["manufacturing_lot"].isin(BAD_LOTS)].groupby("pack_type")["wear_inr_per_swap"].median()
bb = bat[bat["manufacturing_lot"].isin(BAD_LOTS)].copy()
bb["excess_per_swap"] = bb["wear_inr_per_swap"] - bb["pack_type"].map(normal_wear)
bad_swaps = iss[iss["bad"]].groupby("battery_out_str").size()
bb["swaps"] = bb["battery_id"].map(bad_swaps).fillna(0)
excess_wear_total = float((bb["excess_per_swap"].clip(lower=0) * bb["swaps"]).sum())
recent_rate = iss[iss["bad"] & (iss["event_ts"] >= iss["event_ts"].max() - pd.DateOffset(months=3))].shape[0] * 4
excess_wear_year = float(bb["excess_per_swap"].clip(lower=0).median() * recent_rate)
record("bad_lot_excess_wear_total_inr", excess_wear_total)
record("bad_lot_excess_wear_run_rate_year_inr", excess_wear_year)
record("bad_lot_breakeven_inr_per_pack_year", excess_wear_year / max(1, len(bb)))
print(f"Outlier-lot packs cost {inr(excess_wear_total)} in excess wear over the period, and at the current run-rate "
      f"{inr(excess_wear_year)} a year ({inr(excess_wear_year / max(1, len(bb)))} per pack per year) — before counting "
      f"their effect on range and retention.")

# %% [markdown]
# **Partner terms.** What the largest partner's current terms cost relative to the network's other
# partners, per year, at its current volume.

# %%
others_cps = part.drop(index=["Independent", largest])
peer_cps = np.average(others_cps["contribution_per_swap"], weights=others_cps["swaps"])
lp = pp[pp["partner_id"] == largest]
lp_year_swaps = lp[lp["event_ts"] >= lp["event_ts"].max() - pd.DateOffset(months=3)].shape[0] * 4
gap_year = (peer_cps - part.loc[largest, "contribution_per_swap"]) * lp_year_swaps
list_per_swap = lp["list_price_inr"].mean()
record("largest_partner_gap_vs_peers_inr_year", gap_year)
record("largest_partner_discount_breakeven_pct",
       float(part.loc[largest, "effective_discount"] - (peer_cps - part.loc[largest, "contribution_per_swap"]) / list_per_swap))
print(f"{part.loc[largest, 'partner_name']} earns {inr(part.loc[largest, 'contribution_per_swap'])} per swap vs "
      f"{inr(peer_cps)} for other partners — {inr(gap_year)} a year at current volume. A discount of about "
      f"{RESULTS['largest_partner_discount_breakeven_pct']['value']:.0%} would bring it level with the other partners.")

# %% [markdown]
# ### 10.3 The four proposals, scored on one scale
# Each proposal is tested against the evidence above: does it address a cause the data points to, how much
# of the measured problem could it touch, and what does the evidence say about the alternative?

# %%
capacity_fail = sw.loc[~sw["charger_generation"].astype(str).eq("Gen1"), "service_failure"].sum()
scorecard = pd.DataFrame([
    {"proposal": "More stations",
     "what the evidence says": f"Failures are concentrated (worst 10% of stations = {RESULTS['top10pct_stations_failure_share']['value']:.0%} "
                               f"of failures) and {RESULTS['gen1_excess_share_of_all_failures']['value']:.0%} of all failures are "
                               f"excess Gen1 failures in the same conditions; see the expansion-wave table for where new sites went.",
     "addresses a root cause?": "Partly — only where demand exceeds capacity",
     "better alternative": f"Fix or cool the {n_gen1} Gen1 cabinets first; worth up to {inr(gen1_value_per_cabinet_year)} per cabinet per year"},
    {"proposal": "More batteries",
     "what the evidence says": f"Outlier lots {', '.join(sorted(BAD_LOTS)) or 'n/a'} degrade "
                               f"{RESULTS['bad_lot_degradation_multiple']['value']:.1f}× faster; excess wear {inr(excess_wear_year)}/year.",
     "addresses a root cause?": "Only if it replaces the outlier lots",
     "better alternative": "Replace (and claim warranty on) outlier-lot packs rather than adding more of the same"},
    {"proposal": "Network-wide pricing rollout",
     "what the evidence says": f"Pilot DiD on peak share: {RESULTS['did::peak_share']['value']:+.2%}; on peak failure rate: "
                               f"{RESULTS['did::peak_fail']['value']:+.2%} (see 8.2 for confidence intervals).",
     "addresses a root cause?": "Only where peak congestion is real",
     "better alternative": "Targeted peak pricing at congested stations and hours, not network-wide"},
    {"proposal": f"Long-term exclusive with {part.loc[largest, 'partner_name']}",
     "what the evidence says": f"Largest partner by volume ({RESULTS['largest_partner_swap_share']['value']:.0%} of swaps) earns "
                               f"{inr(part.loc[largest, 'contribution_per_swap'])} per swap vs {inr(peer_cps)} for other partners.",
     "addresses a root cause?": "No — it locks in the weakest terms",
     "better alternative": "Renegotiate first: discount floor near break-even, no exclusivity until margin is level"},
]).set_index("proposal")
display(scorecard)
scorecard.to_csv(OUT_DIR / "budget_scorecard.csv")
