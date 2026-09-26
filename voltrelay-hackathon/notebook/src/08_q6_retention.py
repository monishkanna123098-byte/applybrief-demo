# %% [markdown]
# ## 9 · Core question 6 — Why don't new riders come back?
#
# **Definitions.** A *new rider* is anyone who signed up in the data window and made a first attempt on or
# before 30 April 2025, so everyone has at least 60 days of follow-up. A rider is **retained** if they made
# at least one completed swap in days 31–60 after their first attempt, and **churned** otherwise.
#
# **Early experience** is measured over each rider's first 14 days: failures they hit, how long they
# waited, what kind of cabinets and packs they got, whether they had to raise a ticket, what they paid.
# Rider attributes (vehicle, plan, partner, channel, city) and competitive context are included too.
#
# **Method.** Three views that must agree before a factor is called *primary*:
# 1. a logistic regression (effect of each factor holding the others constant, with confidence intervals);
# 2. a gradient-boosting model with permutation importance (catches non-linear effects; a robustness check);
# 3. **attributable churn** — how many percentage points of churn the model says would disappear if that
#    factor were at its good level for every rider. This combines *how strong* an effect is with *how many
#    riders it touches*, which is exactly what separates a primary driver from a secondary one.

# %%
first = sw.groupby("rider_id", observed=True)["event_ts"].min().rename("first_ts")
sw["days_in"] = (sw["event_ts"] - sw["rider_id"].map(first).astype("datetime64[ns]")).dt.days
riders_in = rd.set_index("rider_id")
cohort = first[(first <= "2025-04-30")].to_frame()
cohort = cohort.join(riders_in[["signup_date", "vehicle_class", "plan_type", "partner_id", "signup_channel",
                                "kyc_verified", "age_band", "declared_shift", "home_city_clean"]], how="left")
cohort = cohort[cohort["signup_date"] >= "2024-01-01"]
m2 = sw[(sw["days_in"] >= 30) & (sw["days_in"] < 60) & sw["completed"]].groupby("rider_id", observed=True).size()
cohort["retained"] = cohort.index.isin(m2.index)
cohort["churned"] = ~cohort["retained"]
print(f"New-rider cohort: {len(cohort):,} riders; month-2 retention {cohort['retained'].mean():.1%}")
record("cohort_riders", len(cohort))
record("m2_retention_overall", cohort["retained"].mean())

# %%
early = sw[(sw["days_in"] < 14) & sw["rider_id"].isin(cohort.index)].copy()
early["bad_lot_pack"] = early["battery_out_str"].map(bat.set_index("battery_id")["manufacturing_lot"]).isin(BAD_LOTS)
early["gen1"] = early["charger_generation"].astype(str).eq("Gen1")
eg = early.groupby("rider_id", observed=True)
feat = pd.DataFrame({
    "attempts_14d": eg.size(),
    "service_failures_14d": eg["service_failure"].sum(),
    "any_stockout_14d": eg["stockout"].max().astype(int),
    "any_abandon_14d": eg["abandoned"].max().astype(int),
    "first_attempt_failed": eg["service_failure"].first().astype(int),
    "median_wait_min_14d": eg["queue_wait_sec"].median() / 60,
    "gen1_share_14d": eg["gen1"].mean(),
    "hot_share_14d": eg["hot"].mean(),
    "km_per_pack_14d": eg["km_since_last_swap"].median(),
    "soh_received_14d": eg["soh_out_pct"].mean(),
    "bad_lot_share_14d": eg.apply(lambda d: d.loc[d["completed"], "bad_lot_pack"].mean()),
    "paid_per_swap_14d": eg["amount_charged_inr"].mean(),
    "peak_exposed_share_14d": eg.apply(lambda d: (d["tariff_code"].astype(str) == "PEAK").mean()),
    "competitor_promo_share_14d": eg["competitor_promo_active"].mean(),
    "main_station": eg["station_id"].agg(lambda s: s.value_counts().index[0]),
})
tk14 = tk.merge(first.rename("first_ts").reset_index().assign(rider_id=lambda d: d["rider_id"].astype(str)), on="rider_id")
tk14 = tk14[(tk14["created_ts"] >= tk14["first_ts"]) & (tk14["created_ts"] < tk14["first_ts"] + pd.Timedelta(days=14))]
feat["ticket_14d"] = feat.index.astype(str).isin(set(tk14["rider_id"])).astype(int)
comp_since = st.set_index("station_id")["competitor_within_1_5km_since"]
feat["competitor_nearby"] = (feat["main_station"].astype(str).map(comp_since) <=
                             cohort["first_ts"].reindex(feat.index) + pd.Timedelta(days=14)).astype(int)
feat["main_city"] = feat["main_station"].astype(str).map(st.set_index("station_id")["city"])
cohort = cohort.join(feat, how="inner")
cohort["is_partner"] = cohort["partner_id"].notna().astype(int)
cohort["signup_q"] = cohort["first_ts"].dt.to_period("Q").astype(str)
cohort["service_failure_share_14d"] = cohort["service_failures_14d"] / cohort["attempts_14d"]
display(cohort.groupby("any_stockout_14d")["retained"].agg(["size", "mean"]).rename(
    index={0: "no stockout in first 14 days", 1: "≥1 stockout in first 14 days"}, columns={"size": "riders", "mean": "retention"}))

# %% [markdown]
# **Retention by signup month**, split by whether the rider hit a stockout in their first two weeks.

# %%
cohort["signup_m"] = cohort["first_ts"].dt.to_period("M")
rt = cohort.groupby(["signup_m", "any_stockout_14d"])["retained"].mean().unstack()
rt_all = cohort.groupby("signup_m")["retained"].mean()
exp_share = cohort.groupby("signup_m")["any_stockout_14d"].mean()
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
x = rt.index.to_timestamp()
axes[0].plot(x, rt[0], color=BLUE, label="No early stockout")
axes[0].plot(x, rt[1], color=ORANGE, label="≥1 early stockout")
axes[0].plot(x, rt_all.values, color=INK2, lw=1.2, label="All new riders")
axes[0].yaxis.set_major_formatter(pct_axis)
axes[0].legend(loc="lower left")
titled(axes[0], "Month-2 retention by signup month", "Share of new riders still swapping in days 31–60")
axes[1].plot(x, exp_share.values, color=ORANGE)
axes[1].yaxis.set_major_formatter(pct_axis)
titled(axes[1], "New riders hitting a stockout in their first 14 days", "Share of each signup month")
fig.tight_layout()
save(fig, "q6_retention_trend")
record("m2_retention_first_q", cohort[cohort["signup_q"] == cohort["signup_q"].min()]["retained"].mean())
record("m2_retention_last_q", cohort[cohort["signup_q"] == cohort["signup_q"].max()]["retained"].mean())
record("retention_no_stockout", cohort.loc[cohort["any_stockout_14d"] == 0, "retained"].mean())
record("retention_with_stockout", cohort.loc[cohort["any_stockout_14d"] == 1, "retained"].mean())

# %% [markdown]
# **The model.** Continuous factors are standardised, so each odds ratio is the effect of a one-standard-
# deviation change; binary factors compare yes vs no. An odds ratio above 1 means *more* churn.

# %%
CONT = ["attempts_14d", "service_failure_share_14d", "median_wait_min_14d", "gen1_share_14d", "hot_share_14d",
        "km_per_pack_14d", "soh_received_14d", "bad_lot_share_14d", "paid_per_swap_14d", "peak_exposed_share_14d",
        "competitor_promo_share_14d"]
BIN = ["any_stockout_14d", "any_abandon_14d", "first_attempt_failed", "ticket_14d", "competitor_nearby"]
CATS = ["vehicle_class", "plan_type", "signup_channel", "main_city", "signup_q"]
X = cohort[CONT + BIN + CATS].copy()
for c in CONT:
    X[c] = X[c].astype(float).fillna(X[c].median())
    X[c] = (X[c] - X[c].mean()) / (X[c].std() or 1)
for c in CATS:
    X[c] = X[c].astype(str)
X["kyc_verified"] = cohort["kyc_verified"].astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)
formula = "churned ~ " + " + ".join(CONT + BIN + ["kyc_verified"] + [f"C({c})" for c in CATS])
data_m = X.assign(churned=cohort["churned"].astype(int))
logit = smf.logit(formula, data=data_m).fit(disp=False, maxiter=200, method="bfgs")
ors = pd.DataFrame({"odds_ratio": np.exp(logit.params), "ci_low": np.exp(logit.conf_int()[0]),
                    "ci_high": np.exp(logit.conf_int()[1]), "p": logit.pvalues}).drop(index="Intercept")
display(ors.sort_values("odds_ratio", ascending=False).style.format("{:.3f}"))
print(f"Pseudo R²: {logit.prsquared:.3f}")

# %% [markdown]
# **Robustness: gradient boosting with permutation importance.** If a factor only matters in the linear
# model, it is not trusted as a primary driver.

# %%
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

Xg = pd.get_dummies(X, columns=CATS, drop_first=False, dtype=float)
yg = cohort["churned"].astype(int).to_numpy()
Xtr, Xte, ytr, yte = train_test_split(Xg, yg, test_size=0.3, random_state=59500, stratify=yg)
gbm = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=300, random_state=59500).fit(Xtr, ytr)
auc = roc_auc_score(yte, gbm.predict_proba(Xte)[:, 1])
pi = permutation_importance(gbm, Xte, yte, scoring="roc_auc", n_repeats=8, random_state=59500)
imp = pd.Series(pi.importances_mean, index=Xg.columns)
imp_grouped = imp.groupby(lambda c: next((k for k in CATS if c.startswith(k + "_")), c)).sum().sort_values(ascending=False)
try:
    logit_tr = smf.logit(formula, data=data_m.loc[Xtr.index]).fit(disp=False, maxiter=200)
    auc_logit = roc_auc_score(yte, logit_tr.predict(data_m.loc[Xte.index]))
except Exception as e:          # a rare category absent from the training split
    auc_logit = np.nan
print(f"Held-out ROC AUC — gradient boosting: {auc:.3f}, logistic regression: {auc_logit:.3f}")
display(imp_grouped.head(15).to_frame("drop in AUC when shuffled"))
record("gbm_auc", auc)

# %% [markdown]
# **Attributable churn.** For each factor that VoltRelay can influence, every rider's churn probability is
# re-predicted with that factor moved to its good level (no failure; the best-decile wait, range or pack
# health), and the drop in average churn is recorded.

# %%
def attributable(col, good):
    d = X.copy()
    d[col] = good
    return float(logit.predict(X).mean() - logit.predict(d).mean())


def z(col, raw_value):
    raw = cohort[col].astype(float)
    return (raw_value - raw.fillna(raw.median()).mean()) / (raw.fillna(raw.median()).std() or 1)


levers = {
    "Stockout in first 14 days": ("any_stockout_14d", 0),
    "Abandoned in queue in first 14 days": ("any_abandon_14d", 0),
    "Very first attempt failed": ("first_attempt_failed", 0),
    "Share of early attempts that failed": ("service_failure_share_14d", z("service_failure_share_14d", 0)),
    "Long waits (to best decile)": ("median_wait_min_14d", z("median_wait_min_14d", cohort["median_wait_min_14d"].quantile(0.1))),
    "Low range per pack (to best decile)": ("km_per_pack_14d", z("km_per_pack_14d", cohort["km_per_pack_14d"].quantile(0.9))),
    "Worn packs received (to best decile)": ("soh_received_14d", z("soh_received_14d", cohort["soh_received_14d"].quantile(0.9))),
    "Outlier-lot packs received": ("bad_lot_share_14d", z("bad_lot_share_14d", 0)),
    "Gen1 cabinets used": ("gen1_share_14d", z("gen1_share_14d", 0)),
    "Raised a ticket in first 14 days": ("ticket_14d", 0),
    "Price paid (to cheapest decile)": ("paid_per_swap_14d", z("paid_per_swap_14d", cohort["paid_per_swap_14d"].quantile(0.1))),
    "Competitor promotions": ("competitor_promo_share_14d", z("competitor_promo_share_14d", 0)),
    "Competitor site nearby": ("competitor_nearby", 0),
}
base_churn = cohort["churned"].mean()
attr = pd.DataFrame([{"factor": k, "variable": v[0], "churn points removed": attributable(*v),
                      "prevalence": (cohort[v[0]] > 0).mean() if v[0] in BIN else np.nan} for k, v in levers.items()])
attr["share of all churn"] = attr["churn points removed"] / base_churn
attr = attr.merge(ors[["odds_ratio", "ci_low", "ci_high", "p"]], left_on="variable", right_index=True, how="left")
attr["gbm importance rank"] = attr["variable"].map(imp_grouped.rank(ascending=False))


def tier(r):
    sig = (r["p"] < 0.01) and (r["ci_low"] > 1 or r["ci_high"] < 1)
    if sig and r["churn points removed"] >= 0.02 and r["gbm importance rank"] <= 8:
        return "Primary"
    if sig and r["churn points removed"] > 0.003:
        return "Secondary"
    return "Not supported"


attr["verdict"] = attr.apply(tier, axis=1)
attr = attr.sort_values("churn points removed", ascending=False)
display(attr.style.format({"churn points removed": "{:+.1%}", "prevalence": "{:.0%}", "share of all churn": "{:.0%}",
                           "odds_ratio": "{:.2f}", "ci_low": "{:.2f}", "ci_high": "{:.2f}", "p": "{:.3g}",
                           "gbm importance rank": "{:.0f}"}))
for _, r in attr.iterrows():
    record(f"attr::{r['factor']}", r["churn points removed"], r["verdict"])

# %%
fig, ax = plt.subplots(figsize=(9.5, 0.45 * len(attr) + 1.2))
av = attr.sort_values("churn points removed")
tier_color = {"Primary": BLUE, "Secondary": "#86b6ef", "Not supported": DEEMPH}
ax.barh(range(len(av)), av["churn points removed"] * 100, height=0.55, color=[tier_color[t] for t in av["verdict"]])
for i, (_, r) in enumerate(av.iterrows()):
    ax.annotate(f"{r['churn points removed']*100:+.1f} pts · {r['verdict']}", (max(r["churn points removed"] * 100, 0), i),
                xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.5)
ax.set_yticks(range(len(av)), av["factor"], fontsize=9)
ax.axvline(0, color=INK2, lw=1)
ax.grid(axis="x")
ax.grid(axis="y", visible=False)
ax.set_xlabel("Percentage points of new-rider churn attributable")
titled(ax, "What drives new-rider churn: primary vs secondary factors",
       f"Churn removed if the factor were at its good level · base churn {base_churn:.0%}")
save(fig, "q6_attributable_churn")

# %% [markdown]
# **Counterfactual trend.** If every signup month had experienced the early-failure exposure of the
# first quarter, what would retention have been? This isolates how much of the retention decline the
# service failures explain.

# %%
q0 = cohort["signup_q"].min()
SVC = ["any_stockout_14d", "any_abandon_14d", "first_attempt_failed", "service_failure_share_14d", "median_wait_min_14d"]
ref_rows = X.loc[cohort["signup_q"] == q0, SVC].to_numpy()
cf = X.copy()
cf[SVC] = ref_rows[np.random.default_rng(59500).integers(0, len(ref_rows), size=len(cf))]
comp = pd.DataFrame({"actual": 1 - cohort["churned"].astype(float),
                     "model": 1 - logit.predict(X), "q1_service_levels": 1 - logit.predict(cf),
                     "q": cohort["signup_q"]}).groupby("q").mean()
display(comp.style.format("{:.1%}"))
record("retention_last_q_actual", comp["actual"].iloc[-1])
record("retention_last_q_if_q1_service", comp["q1_service_levels"].iloc[-1])
