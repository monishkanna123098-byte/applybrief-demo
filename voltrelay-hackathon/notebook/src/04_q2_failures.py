# %% [markdown]
# ## 5 · Core question 2 — Service failures and customer experience
#
# *How do queue wait, failed attempts and abandoned swaps relate to station, hour of day, season and
# vehicle class? Is service quality spread evenly, or concentrated somewhere specific?*

# %%
fail_mix = sw.groupby(sw["event_ts"].dt.to_period("Q"))[["stockout", "abandoned", "system_error", "cancelled"]].mean()
fail_mix.index = fail_mix.index.astype(str)
display(fail_mix.style.format("{:.2%}"))
by_type = sw[["stockout", "abandoned", "system_error"]].sum()
record("share_failures_stockout", by_type["stockout"] / by_type.sum(), "stockouts as share of service failures")

# %%
heat = sw.groupby(["hour", "month"])["service_failure"].mean().unstack()
fig, ax = plt.subplots(figsize=(12, 5.2))
im = ax.imshow(heat.values, aspect="auto", cmap=SEQ, origin="upper")
ax.set_yticks(range(0, 24, 2), [f"{h:02d}:00" for h in range(0, 24, 2)])
ax.set_xticks(range(len(heat.columns)), [p.strftime("%b\n%y") if p.month in (1, 4, 7, 10) else "" for p in heat.columns])
ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, format=PercentFormatter(1.0, decimals=0))
cb.outline.set_visible(False)
titled(ax, "Service-failure rate by hour of day and month",
       "Service-failure rate by hour of day (rows) and month (columns)")
save(fig, "q2_hour_month_heatmap")
hot = heat.stack().sort_values(ascending=False)
print("Worst hour × month cells:", [(f"{h:02d}:00", str(m), f"{v:.1%}") for (h, m), v in hot.head(5).items()])

# %% [markdown]
# **Vehicle class and season.** 3W riders use bigger packs, fewer slots exist for them, and a 3W stockout
# strands a cargo vehicle — so the two classes are compared separately. Seasons are grouped as Indian
# operating seasons: summer (Apr–Jun), monsoon (Jul–Sep), post-monsoon (Oct–Nov) and winter (Dec–Mar).

# %%
season_map = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Winter", 4: "Summer", 5: "Summer", 6: "Summer",
              7: "Monsoon", 8: "Monsoon", 9: "Monsoon", 10: "Post-monsoon", 11: "Post-monsoon"}
sw["season"] = pd.Categorical(sw["event_ts"].dt.month.map(season_map),
                              ["Winter", "Summer", "Monsoon", "Post-monsoon"], ordered=True)
vc_season = sw.groupby(["vehicle_class", "season"], observed=True).agg(
    attempts=("service_failure", "size"), service_failure_rate=("service_failure", "mean"),
    stockout_rate=("stockout", "mean"), abandon_rate=("abandoned", "mean"),
    median_wait_min=("queue_wait_sec", lambda s: s.median() / 60),
    p90_wait_min=("queue_wait_sec", lambda s: s.quantile(0.9) / 60))
display(vc_season.style.format({"attempts": "{:,.0f}", "service_failure_rate": "{:.2%}", "stockout_rate": "{:.2%}",
                                "abandon_rate": "{:.2%}", "median_wait_min": "{:.1f}", "p90_wait_min": "{:.1f}"}))

fig, ax = plt.subplots(figsize=(9, 3.8))
seasons = ["Winter", "Summer", "Monsoon", "Post-monsoon"]
width = 0.36
for i, (vc, color) in enumerate([("2W", BLUE), ("3W", ORANGE)]):
    vals = [vc_season.loc[(vc, s), "service_failure_rate"] if (vc, s) in vc_season.index else np.nan for s in seasons]
    xs = np.arange(len(seasons)) + (i - 0.5) * (width + 0.03)
    ax.bar(xs, vals, width=width, color=color, label=vc)
    for x, v in zip(xs, vals):
        ax.annotate(f"{v:.1%}", (x, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5)
ax.set_xticks(range(len(seasons)), seasons)
ax.yaxis.set_major_formatter(pct_axis)
ax.legend(loc="upper right")
titled(ax, "Service-failure rate by season and vehicle class", "Service-failure rate by season")
save(fig, "q2_season_vehicle")

# %% [markdown]
# **How concentrated are failures?** If failures were an even, network-wide capacity problem, the
# worst stations would carry about the same share of failures as of traffic. The curve compares the two.

# %%
stn = sw.groupby("station_id", observed=True).agg(attempts=("service_failure", "size"),
                                                  failures=("service_failure", "sum"),
                                                  stockouts=("stockout", "sum"))
stn["rate"] = stn["failures"] / stn["attempts"]
stn = stn.sort_values("failures", ascending=False)
cum_fail = stn["failures"].cumsum() / stn["failures"].sum()
cum_att = stn["attempts"].cumsum() / stn["attempts"].sum()
share_st = np.arange(1, len(stn) + 1) / len(stn)
top10 = max(1, int(round(0.10 * len(stn))))
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(np.r_[0, share_st], np.r_[0, cum_fail.values], color=RED, label="Share of service failures")
ax.plot(np.r_[0, share_st], np.r_[0, cum_att.values], color=DEEMPH, label="Share of attempts")
ax.scatter([share_st[top10 - 1]], [cum_fail.iloc[top10 - 1]], color=RED, s=30, zorder=3, edgecolor=SURFACE, linewidth=2)
ax.annotate(f"Worst 10% of stations: {cum_fail.iloc[top10 - 1]:.0%} of failures,\n"
            f"but only {cum_att.iloc[top10 - 1]:.0%} of attempts",
            (share_st[top10 - 1], cum_fail.iloc[top10 - 1]), xytext=(12, -34), textcoords="offset points", fontsize=9)
ax.xaxis.set_major_formatter(pct_axis)
ax.yaxis.set_major_formatter(pct_axis)
ax.set_xlabel("Stations, worst first")
ax.legend(loc="lower right")
ax.grid(axis="x")
titled(ax, "How concentrated are service failures across stations?", "Cumulative share, stations ranked by failures")
save(fig, "q2_concentration")
record("top10pct_stations_failure_share", cum_fail.iloc[top10 - 1])
record("top10pct_stations_attempt_share", cum_att.iloc[top10 - 1])

worst = stn.head(12).join(st.set_index("station_id")[["city", "charger_generation", "location_type", "expansion_wave"]])
display(worst.style.format({"attempts": "{:,.0f}", "failures": "{:,.0f}", "stockouts": "{:,.0f}", "rate": "{:.1%}"}))

# %% [markdown]
# **Waiting.** Queue wait is the experience riders feel even when the swap succeeds. The chart compares
# the typical (median) wait with the bad-day wait (90th percentile) through the day.

# %%
wait_hr = sw.groupby("hour")["queue_wait_sec"].describe(percentiles=[0.5, 0.9])[["50%", "90%"]] / 60
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(wait_hr.index, wait_hr["90%"], color=BLUE, label="90th percentile")
ax.plot(wait_hr.index, wait_hr["50%"], color=DEEMPH, label="Median")
ax.set_xticks(range(0, 24, 3))
ax.set_xlabel("Hour of day")
ax.set_ylabel("Minutes")
ax.legend(loc="upper left")
titled(ax, "Queue wait through the day", "Queue wait in minutes, all attempts")
save(fig, "q2_wait_by_hour")
