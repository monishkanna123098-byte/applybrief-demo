# %% [markdown]
# ## 11 · Key findings
#
# The table below is generated from the numbers computed above, so it cannot drift from the analysis.
# The written report and the video use exactly these values.

# %%
def v(k):
    return RESULTS[k]["value"] if k in RESULTS else np.nan


findings = pd.DataFrame([
    ("Performance", f"Completed swaps grew {v('growth_completed_x'):.1f}× and revenue {v('growth_revenue_x'):.1f}×, "
                    f"but service failures grew {v('growth_failures_x'):.1f}× (rate {pct(v('failure_rate_first3'))} → "
                    f"{pct(v('failure_rate_last3'))}); contribution per swap went from ₹{v('contribution_per_swap_first3'):.1f} "
                    f"to ₹{v('contribution_per_swap_last3'):.1f}."),
    ("Failures", f"The worst 10% of stations produce {pct(v('top10pct_stations_failure_share'), 0)} of service failures "
                 f"on {pct(v('top10pct_stations_attempt_share'), 0)} of attempts."),
    ("Equipment × heat", f"Model-adjusted failure rate: Gen1 {pct(v('adj_fail::Gen1::normal'))} normally vs "
                         f"{pct(v('adj_fail::Gen1::hot'))} in ≥40°C hours; Gen2 {pct(v('adj_fail::Gen2::normal'))} vs "
                         f"{pct(v('adj_fail::Gen2::hot'))}. Gen1 accounts for {v('gen1_excess_failures'):,.0f} excess failures "
                         f"({pct(v('gen1_excess_share_of_all_failures'), 0)} of all)."),
    ("Batteries", f"Lots {v('bad_lots')} degrade {v('bad_lot_degradation_multiple'):.1f}× faster than all other lots; "
                  f"excess wear {inr(v('bad_lot_excess_wear_total_inr'))} over the period."),
    ("Pricing", f"Pilot difference-in-differences: peak-hour share {v('did::peak_share'):+.2%}, peak failure rate "
                f"{v('did::peak_fail'):+.2%}, revenue per swap ₹{v('did::rev_per_swap'):+.2f}."),
    ("Partners", f"Largest partner {v('largest_partner')} ({pct(v('largest_partner_swap_share'), 0)} of swaps) earns "
                 f"₹{v('largest_partner_contribution_per_swap'):.1f} per swap vs ₹{v('independent_contribution_per_swap'):.1f} "
                 f"from independent riders."),
    ("Retention", f"Month-2 retention {pct(v('retention_no_stockout'))} without an early stockout vs "
                  f"{pct(v('retention_with_stockout'))} with one; latest-quarter retention {pct(v('retention_last_q_actual'))} "
                  f"would be {pct(v('retention_last_q_if_q1_service'))} at first-quarter service levels."),
    ("Cost of a failure", f"A service failure costs ₹{v('direct_cost_per_failure_inr'):.1f} directly; an early failure for a "
                          f"new rider costs {inr(v('churn_cost_per_early_failure_inr'))} in lost future margin."),
], columns=["area", "finding"])
with pd.option_context("display.max_colwidth", 300):
    display(findings)

# %%
out = {k: v_["value"] for k, v_ in RESULTS.items()}
(OUT_DIR / "results.json").write_text(json.dumps(out, indent=1, default=str))
monthly.to_csv(OUT_DIR / "monthly_network.csv")
attr.to_csv(OUT_DIR / "churn_drivers.csv", index=False)
part.to_csv(OUT_DIR / "partner_economics.csv")
lots.to_csv(OUT_DIR / "battery_lots.csv", index=False)
print("RESULTS_BLOCK_START")
print(json.dumps(out, default=str))
print("RESULTS_BLOCK_END")
print(f"\nFigures saved to {FIG_DIR.resolve()} ({len(list(FIG_DIR.glob('*.png')))} files); tables to {OUT_DIR.resolve()}")
