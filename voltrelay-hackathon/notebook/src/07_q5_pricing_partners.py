# %% [markdown]
# ## 8 · Core question 5 — Pricing and partner economics
#
# *How do the base price change, the peak/off-peak pilot and individual partner contract terms relate to
# swap timing, revenue and contribution margin? Do they affect every rider segment the same way?*

# %% [markdown]
# ### 8.1 The base price change
# Three months before versus three months after the change, by plan type. Partner-billed riders pay a
# contracted discount on the list price; pay-as-you-go riders pay list price directly.

# %%
b0, b1 = BASE_PRICE_DATE - pd.DateOffset(months=3), BASE_PRICE_DATE + pd.DateOffset(months=3)
win = sw[(sw["event_ts"] >= b0) & (sw["event_ts"] < b1) & ~sw["city"].astype(str).isin(PILOT_CITIES)].copy()
win["period"] = np.where(win["event_ts"] < BASE_PRICE_DATE, "3 months before", "3 months after")
bp = win.groupby(["plan_type", "period"], observed=True).apply(lambda d: pd.Series({
    "list_price": d.loc[d["completed"], "list_price_inr"].mean(),
    "paid_per_swap": d.loc[d["completed"], "amount_charged_inr"].mean(),
    "swaps_per_rider_week": d["completed"].sum() / d["rider_id"].nunique() / 13,
    "service_failure_rate": d["service_failure"].mean(),
    "margin_per_swap": d.loc[d["completed"], "contribution_inr"].mean(),
})).unstack("period")
display(bp.round(2))

# %% [markdown]
# ### 8.2 The peak/off-peak pilot — a difference-in-differences test
# The pilot changed prices in two cities only, so the other cities show what would have happened anyway.
# The estimate is *(change in pilot cities) − (change in other cities)* over the same weeks. Peak hours are
# read from the tariff codes themselves (the hours in which PEAK tariffs were charged).

# %%
PEAK_HOURS = sorted(sw.loc[sw["tariff_code"].astype(str) == "PEAK", "hour"].value_counts()
                    .loc[lambda s: s > s.max() * 0.05].index.tolist())
OFFPEAK_HOURS = sorted(sw.loc[sw["tariff_code"].astype(str) == "OFFPEAK", "hour"].value_counts()
                       .loc[lambda s: s > s.max() * 0.05].index.tolist())
print("Peak hours:", PEAK_HOURS, "| Off-peak hours:", OFFPEAK_HOURS)
sw["is_peak_hour"] = sw["hour"].isin(PEAK_HOURS)
sw["pilot_city"] = sw["city"].astype(str).isin(PILOT_CITIES)
pre_start = PILOT_START - pd.DateOffset(weeks=12)
dd = sw[(sw["event_ts"] >= pre_start) & (sw["event_ts"] <= PILOT_END)].copy()
dd["post"] = dd["event_ts"] >= PILOT_START
dd["week"] = dd["event_ts"].dt.to_period("W").astype(str)
cw = dd.groupby(["city", "week"], observed=True).apply(lambda d: pd.Series({
    "peak_share": d["is_peak_hour"].mean(),
    "peak_fail": d.loc[d["is_peak_hour"], "service_failure"].mean(),
    "peak_wait_min": d.loc[d["is_peak_hour"], "queue_wait_sec"].median() / 60,
    "rev_per_swap": d.loc[d["completed"], "amount_charged_inr"].mean(),
    "margin_per_swap": d.loc[d["completed"], "contribution_inr"].mean(),
    "attempts": len(d)})).reset_index()
cw["city"] = cw["city"].astype(str)
cw["treated"] = cw["city"].isin(PILOT_CITIES).astype(int)
cw["post"] = (pd.PeriodIndex(cw["week"], freq="W").start_time >= PILOT_START).astype(int)
did_rows = []
for y in ["peak_share", "peak_fail", "peak_wait_min", "rev_per_swap", "margin_per_swap"]:
    m = smf.wls(f"{y} ~ treated:post + C(city) + C(week)", data=cw.dropna(subset=[y]),
                weights=cw.dropna(subset=[y])["attempts"]).fit(cov_type="HC1")
    pre_mean = cw.loc[(cw["treated"] == 1) & (cw["post"] == 0), y].mean()
    did_rows.append({"outcome": y, "pilot-city baseline": pre_mean, "DiD estimate": m.params["treated:post"],
                     "95% CI low": m.conf_int().loc["treated:post", 0], "95% CI high": m.conf_int().loc["treated:post", 1],
                     "p": m.pvalues["treated:post"]})
did = pd.DataFrame(did_rows).set_index("outcome")
display(did.style.format("{:.4f}"))
for y, r in did.iterrows():
    record(f"did::{y}", r["DiD estimate"], f"baseline {r['pilot-city baseline']:.4f}, p={r['p']:.3f}")

# %%
trend = cw.groupby(["treated", "week"]).apply(lambda d: np.average(d["peak_share"], weights=d["attempts"])).unstack(0)
trend.index = pd.PeriodIndex(trend.index, freq="W").start_time
fig, ax = plt.subplots(figsize=(9.5, 3.8))
ax.plot(trend.index, trend[1], color=BLUE, label="Pilot cities")
ax.plot(trend.index, trend[0], color=DEEMPH, label="Other cities")
ax.axvline(PILOT_START, color=AXIS, lw=1)
ax.text(PILOT_START, 0.98, " Pilot starts", transform=ax.get_xaxis_transform(), fontsize=8, color=MUTED, va="top")
ax.yaxis.set_major_formatter(pct_axis)
ax.legend(loc="lower left")
titled(ax, "Share of attempts made in peak hours", "Weekly, pilot cities vs other cities")
save(fig, "q5_pilot_did")

# %% [markdown]
# **Did the pilot reach the riders it was meant to move?** The surcharge only applies to riders who pay
# it: independents, and partner riders whose contract makes the surcharge billable. Riders whose employer
# absorbs it should not react — a built-in placebo group.

# %%
pmap = partners.set_index("partner_id")["peak_surcharge_billable"].astype(str)
sw["surcharge_exposed"] = np.where(sw["is_partner"], sw["partner_id"].astype(str).map(pmap).eq("Y"), True)
dd["surcharge_exposed"] = np.where(dd["is_partner"], dd["partner_id"].astype(str).map(pmap).eq("Y"), True)
seg = dd.groupby(["surcharge_exposed", "pilot_city", "post"])["is_peak_hour"].mean().unstack("post")
seg["change"] = seg[True] - seg[False]
seg_did = seg["change"].unstack("pilot_city")
seg_did["pilot minus other cities"] = seg_did[True] - seg_did[False]
seg_did.index = seg_did.index.map({True: "Pays the surcharge", False: "Surcharge absorbed by employer"})
display(seg_did.style.format("{:+.2%}"))

base_cong = sw[(sw["event_ts"] < PILOT_START) & sw["is_peak_hour"]].groupby("city", observed=True).agg(
    peak_failure_rate=("service_failure", "mean"), peak_p90_wait_min=("queue_wait_sec", lambda s: s.quantile(0.9) / 60))
base_cong["pilot city"] = base_cong.index.astype(str).isin(PILOT_CITIES)
display(base_cong.sort_values("peak_failure_rate", ascending=False).style.format(
    {"peak_failure_rate": "{:.2%}", "peak_p90_wait_min": "{:.1f}"}))

# %% [markdown]
# ### 8.3 Partner economics
# Revenue is not value. Each partner's swaps carry the same energy, battery-wear and site costs as any
# other swap, but earn a discounted price — and some partners pay months later. Contribution per swap is
# computed per partner (before and after any contract amendment) with independent riders as a benchmark.
# Payment terms are priced at a 12% annual cost of capital (an explicit assumption, shown separately).

# %%
COST_OF_CAPITAL = 0.12
pp = sw[sw["completed"]].copy()
pp["partner_id"] = pp["partner_id"].astype(str).replace({"nan": "Independent", "<NA>": "Independent"})
pinfo = partners.set_index("partner_id")
pp["amended"] = pp["partner_id"].map(pinfo["amendment_date"]).notna() & (
    pp["event_ts"] >= pp["partner_id"].map(pinfo["amendment_date"]))
pp["terms_days"] = pp["partner_id"].map(pinfo["payment_terms_days"]).fillna(0).astype(float)
pp["wc_cost_inr"] = pp["amount_charged_inr"] * COST_OF_CAPITAL * pp["terms_days"] / 365
part = pp.groupby("partner_id").agg(swaps=("completed", "size"), revenue_inr=("amount_charged_inr", "sum"),
                                    list_inr=("list_price_inr", "sum"), discount_inr=("discount_inr", "sum"),
                                    energy_inr=("energy_inr", "sum"), wear_inr=("wear_inr", "sum"),
                                    site_inr=("site_inr", "sum"), wc_inr=("wc_cost_inr", "sum"))
part["contribution_per_swap"] = (part["revenue_inr"] - part[["energy_inr", "wear_inr", "site_inr"]].sum(axis=1)) / part["swaps"]
part["after_payment_terms"] = part["contribution_per_swap"] - part["wc_inr"] / part["swaps"]
part["effective_discount"] = part["discount_inr"] / part["list_inr"]
part = part.join(pinfo[["partner_name", "partner_segment", "vehicle_class", "discount_pct",
                        "discount_pct_after_amendment", "payment_terms_days", "peak_surcharge_billable"]], how="left")
part.loc["Independent", "partner_name"] = "Independent riders"
part = part.sort_values("swaps", ascending=False)
display(part[["partner_name", "partner_segment", "vehicle_class", "swaps", "revenue_inr", "effective_discount",
              "contribution_per_swap", "after_payment_terms", "payment_terms_days"]].style.format(
    {"swaps": "{:,.0f}", "revenue_inr": "₹{:,.0f}", "effective_discount": "{:.1%}",
     "contribution_per_swap": "₹{:.2f}", "after_payment_terms": "₹{:.2f}"}))
largest = part.drop(index="Independent").index[0]
record("largest_partner", str(part.loc[largest, "partner_name"]))
record("largest_partner_swap_share", part.loc[largest, "swaps"] / part["swaps"].sum())
record("largest_partner_contribution_per_swap", part.loc[largest, "contribution_per_swap"])
record("independent_contribution_per_swap", part.loc["Independent", "contribution_per_swap"])

amended_ids = partners.loc[partners["amendment_date"].notna(), "partner_id"].astype(str)
if len(amended_ids):
    ba = pp[pp["partner_id"].isin(amended_ids)].groupby(["partner_id", "amended"]).agg(
        swaps=("completed", "size"), paid_per_swap=("amount_charged_inr", "mean"),
        contribution_per_swap=("contribution_inr", "mean"))
    display(ba)
    for (pid, am), r in ba.iterrows():
        record(f"amend::{pid}::{'after' if am else 'before'}::contribution_per_swap", r["contribution_per_swap"])

# %%
fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(part) + 1.2))
pv = part.sort_values("after_payment_terms")
ypos = np.arange(len(pv))
cols = [BLUE if i == largest else (INK2 if i == "Independent" else DEEMPH) for i in pv.index]
ax.barh(ypos, pv["after_payment_terms"], height=0.55, color=cols)
for y, (i, r) in zip(ypos, pv.iterrows()):
    ax.annotate(f"₹{r['after_payment_terms']:.1f}", (r["after_payment_terms"], y), xytext=(4 if r["after_payment_terms"] >= 0 else -4, 0),
                textcoords="offset points", va="center", ha="left" if r["after_payment_terms"] >= 0 else "right", fontsize=8.5)
ax.set_yticks(ypos, [f"{r['partner_name']}  ({r['swaps']/1e3:,.0f}K swaps)" for _, r in pv.iterrows()], fontsize=8.5)
ax.axvline(0, color=INK2, lw=1)
ax.grid(axis="x")
ax.grid(axis="y", visible=False)
titled(ax, "Contribution per swap by customer", "₹ after energy, battery wear, site cost and payment terms; largest partner highlighted")
save(fig, "q5_partner_value")
