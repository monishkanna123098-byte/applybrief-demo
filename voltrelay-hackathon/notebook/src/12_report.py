# %% [markdown]
# ## 12 · Build the report and the video script from these results
#
# This cell writes `VoltRelay_Report.html` (open it and print to PDF) and `VoltRelay_Video_Script.md`
# using only the numbers computed above, with every chart embedded.

# %%
import base64, html as _html


def V(k, default=np.nan):
    return RESULTS[k]["value"] if k in RESULTS else default


def img(name, caption):
    p = FIG_DIR / f"{name}.png"
    if not p.exists():
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{b64}"/><figcaption>{_html.escape(caption)}</figcaption></figure>'


gen1_ratio = V("adj_fail::Gen1::hot") / max(V("adj_fail::Gen2::hot"), 1e-9)
primary = attr.loc[attr["verdict"] == "Primary", "factor"].tolist()
secondary = attr.loc[attr["verdict"] == "Secondary", "factor"].tolist()
not_supported = attr.loc[attr["verdict"] == "Not supported", "factor"].tolist()
d_fail, d_share = did.loc["peak_fail"], did.loc["peak_share"]
pilot_cut_peak = (d_share["DiD estimate"] < 0) and (d_share["p"] < 0.05)
pilot_cut_fail = (d_fail["DiD estimate"] < 0) and (d_fail["p"] < 0.05)
lp_name = str(V("largest_partner"))
lp_gap = V("largest_partner_gap_vs_peers_inr_year")
bad_lots_txt = str(V("bad_lots")) if BAD_LOTS else ""
cohort_per_year = len(cohort) / max(1, (cohort["first_ts"].max() - cohort["first_ts"].min()).days / 365.25)
gen1_year = V("gen1_excess_failures") / years * V("blended_cost_per_failure_inr")
top_churn = attr.iloc[0]
churn_year = top_churn["churn points removed"] * cohort_per_year * V("retained_rider_year_value_inr")

concentrated = V("top10pct_stations_failure_share", 0) >= 1.8 * V("top10pct_stations_attempt_share", 1)
gen1_major = gen1_ratio >= 1.5 and V("gen1_excess_share_of_all_failures", 0) >= 0.2
failures_outgrew = V("growth_failures_x", 0) > V("growth_completed_x", 0)
retention_fell = V("m2_retention_last_q", 0) < V("m2_retention_first_q", 0)
conc_txt = (f"Failures are not network-wide: the worst ten percent of stations produce {pct(V('top10pct_stations_failure_share'), 0)} of them."
            if concentrated else
            f"Failures are fairly spread out: the worst ten percent of stations produce {pct(V('top10pct_stations_failure_share'), 0)} of them.")
gen1_txt = (f"The biggest single cause is old Gen1 cabinets in the heat: above 40 degrees they fail {gen1_ratio:.1f} times as often as Gen2 "
            f"in the same conditions, about {pct(V('gen1_excess_share_of_all_failures'), 0)} of all failures."
            if gen1_major else
            f"Charger generation matters less than expected: in hot hours Gen1 fails {gen1_ratio:.1f} times as often as Gen2, "
            f"about {pct(V('gen1_excess_share_of_all_failures'), 0)} of all failures.")
growth_txt = (f"But service failures grew {V('growth_failures_x'):.1f} times" if failures_outgrew
              else f"Service failures grew {V('growth_failures_x'):.1f} times")
ret_txt = (f"new-rider retention fell from {pct(V('m2_retention_first_q'), 0)} to {pct(V('m2_retention_last_q'), 0)}" if retention_fell
           else f"new-rider month-2 retention was {pct(V('m2_retention_overall'), 0)}")
cost_txt = (f"One failed swap costs {inr(V('direct_cost_per_failure_inr'))} right away, and {inr(V('churn_cost_per_early_failure_inr'))} "
            f"when it drives a new rider away." if V("direct_cost_per_failure_inr", 0) > 0 and V("churn_cost_per_early_failure_inr", 0) > 0
            else f"{pct(1 - V('failures_recovered_1h', 0), 0)} of riders who hit a failure did not complete a swap within the hour.")

levers_inr = pd.DataFrame([
    ("Fix or cool Gen1 cabinets in the hot hours", gen1_year, f"{V('gen1_excess_failures'):,.0f} excess failures over the period"),
    (f"Replace outlier battery lots ({bad_lots_txt or 'none found'})", V("bad_lot_excess_wear_run_rate_year_inr", 0), "excess wear at current run-rate"),
    (f"Renegotiate {lp_name}'s terms before any exclusive", max(lp_gap, 0), "margin gap vs other partners at current volume"),
    (f"Protect new riders from: {top_churn['factor'].lower()}", churn_year, "retained riders' yearly contribution"),
], columns=["action", "inr_per_year", "basis"]).sort_values("inr_per_year", ascending=False)
levers_inr.to_csv(OUT_DIR / "levers_ranked.csv", index=False)
display(levers_inr.assign(value=levers_inr["inr_per_year"].map(inr)))

verdicts = [
    ("More stations",
     "Not as proposed" if V("gen1_excess_share_of_all_failures", 0) >= 0.2 else "Only where capacity is short",
     f"{pct(V('gen1_excess_share_of_all_failures'), 0)} of failures are excess Gen1 failures at existing sites; "
     f"the worst 10% of stations hold {pct(V('top10pct_stations_failure_share'), 0)} of failures."),
    ("More batteries",
     "Replace, don't expand" if BAD_LOTS else "Not supported by the evidence",
     f"Outlier lots {bad_lots_txt} degrade {V('bad_lot_degradation_multiple'):.1f}× faster." if BAD_LOTS
     else "No lot degrades abnormally; stockouts track heat and equipment, not pack count."),
    ("Network-wide pricing",
     "Targeted only" if pilot_cut_peak and not pilot_cut_fail else ("Supported" if pilot_cut_fail else "Not supported"),
     f"Pilot moved peak share by {d_share['DiD estimate']:+.2%} (p={d_share['p']:.2g}) and peak failure rate by "
     f"{d_fail['DiD estimate']:+.2%} (p={d_fail['p']:.2g})."),
    (f"Exclusive with {lp_name}",
     "Not supported" if lp_gap > 0 else "Supported",
     f"{lp_name} earns {inr(V('largest_partner_contribution_per_swap'))} per swap vs other partners' average; "
     f"gap worth {inr(lp_gap)} a year."),
]
verdict_df = pd.DataFrame(verdicts, columns=["proposal", "verdict", "evidence"])
display(verdict_df)

css = """body{font-family:Arial,Helvetica,sans-serif;max-width:900px;margin:32px auto;color:#0b0b0b;line-height:1.5;padding:0 16px}
h1{font-size:26px;margin-bottom:4px}h2{font-size:19px;margin-top:30px;border-bottom:1px solid #e1e0d9;padding-bottom:4px}
h3{font-size:15px;margin-bottom:4px}.sub{color:#52514e}table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}
td,th{border-bottom:1px solid #e1e0d9;padding:6px 8px;text-align:left;vertical-align:top}th{background:#f3f3f0}
figure{margin:14px 0}img{max-width:100%}figcaption{font-size:12px;color:#52514e}.kpi{display:flex;gap:12px;flex-wrap:wrap}
.k{flex:1;min-width:150px;border:1px solid #e1e0d9;border-radius:6px;padding:10px}.k b{font-size:22px;display:block}
@media print{h2{page-break-after:avoid}figure{page-break-inside:avoid}}"""
rows = lambda df: "".join("<tr>" + "".join(f"<td>{_html.escape(str(c))}</td>" for c in r) + "</tr>" for r in df.itertuples(index=False))
report = f"""<!doctype html><html><head><meta charset="utf-8"><title>VoltRelay Analysis Report</title><style>{css}</style></head><body>
<h1>VoltRelay Energy: fix before you grow</h1>
<p class="sub">Gradient Learnings Data Analytics Hackathon 2026 · Team: Monish Kannaha S A, Lokeshkumar K</p>
<div class="kpi">
<div class="k"><b>{V('growth_failures_x'):.1f}×</b>growth in service failures vs {V('growth_completed_x'):.1f}× in swaps</div>
<div class="k"><b>{pct(V('gen1_excess_share_of_all_failures'), 0)}</b>of failures are excess Gen1 failures</div>
<div class="k"><b>{pct(V('retention_with_stockout'), 0)} vs {pct(V('retention_no_stockout'), 0)}</b>month-2 retention with vs without an early stockout</div>
<div class="k"><b>{inr(V('blended_cost_per_failure_inr'))}</b>blended cost of one service failure</div></div>

<h2>1. Executive summary</h2>
<p>Between January 2024 and June 2025 VoltRelay's completed swaps grew {V('growth_completed_x'):.1f}× and revenue
{V('growth_revenue_x'):.1f}×, but service failures grew {V('growth_failures_x'):.1f}× (failure rate {pct(V('failure_rate_first3'))} →
{pct(V('failure_rate_last3'))}) and contribution per swap moved from ₹{V('contribution_per_swap_first3'):.1f} to
₹{V('contribution_per_swap_last3'):.1f}. {conc_txt} Converting every problem into rupees per year ranks the fixes:</p>
<table><tr><th>Priority</th><th>Action</th><th>Value per year</th><th>Basis</th></tr>
{''.join(f"<tr><td>{i + 1}</td><td>{_html.escape(r.action)}</td><td>{inr(r.inr_per_year)}</td><td>{_html.escape(r.basis)}</td></tr>" for i, r in enumerate(levers_inr.itertuples()))}</table>

<h2>2. Problem understanding</h2>
<p>VoltRelay runs unmanned battery-swap cabinets for gig and logistics riders in six cities. Over 18 months it opened two
expansion waves, changed base prices, piloted peak/off-peak pricing in {', '.join(PILOT_CITIES)}, added {NEW_SUPPLIER} as a
battery supplier and amended its largest fleet contract. Leadership sees three symptoms — failures rising faster than
volume, falling new-rider retention and eroding per-swap margin — and must choose between four budget proposals: more
stations, more batteries, network-wide pricing, or a long-term exclusive with the largest partner.</p>

<h2>3. Analytical approach</h2>
<ol><li><b>Clean deliberately.</b> All nine documented data-quality issues were detected, measured and fixed: {int(V('test_station_events')):,}
test-station events excluded, {int(V('duplicates_removed')):,} offline-sync duplicates removed, {int(V('clock_bug_events')):,} v3.2.0
timestamps corrected by +5h30m, {int(V('city_spelling_variants'))} city spellings standardised, telemetry gaps never zero-filled,
CSAT not used as a KPI (naive {V('csat_naive'):.2f} vs re-weighted {V('csat_reweighted'):.2f}), and {int(V('tickets_relabelled')):,}
mislabelled tickets re-classified from their text. We also found {int(V('retired_packs_issued')):,} swaps that issued a pack after its
recorded retirement date.</li>
<li><b>Build a per-swap P&amp;L:</b> revenue minus energy (kWh × local tariff), battery wear (share of usable life × pack cost) and
site cost (rent + maintenance spread over each station's swaps).</li>
<li><b>Separate causes that overlap</b> with a model of failure on charger generation, heat, city, hour, vehicle and location together;
a difference-in-differences test for the pricing pilot; and a churn model cross-checked with gradient boosting.</li>
<li><b>Put everything in one currency:</b> the rupee cost of a failed swap and the rupee value of each fix, stated as a break-even
rather than invented capex.</li></ol>

<h2>4. Key insights</h2>
<h3>4.1 {"The metrics disagree: growth hid a quality problem" if failures_outgrew else "Network performance over time"}</h3>
{img('q1_four_metrics', 'Monthly completed swaps, revenue, service-failure rate and contribution per swap.')}
{img('q1_margin_bridge', 'Per-swap margin bridge, first quarter to last quarter.')}
<h3>4.2 Failures are concentrated in time and place</h3>
{img('q2_hour_month_heatmap', 'Service-failure rate by hour of day and month.')}
{img('q2_concentration', 'Cumulative share of failures vs attempts, stations ranked worst first.')}
<h3>4.3 Equipment × heat</h3>
<p>Holding city, hour, vehicle and location constant, a Gen1 cabinet fails {pct(V('adj_fail::Gen1::normal'))} of attempts in normal hours
and {pct(V('adj_fail::Gen1::hot'))} in ≥40°C hours, versus {pct(V('adj_fail::Gen2::normal'))} and {pct(V('adj_fail::Gen2::hot'))} for Gen2
({gen1_ratio:.1f}× in the heat). Gen1 hardware accounts for about {V('gen1_excess_failures'):,.0f} excess failures,
{pct(V('gen1_excess_share_of_all_failures'), 0)} of all service failures.</p>
{img('q3_heat_by_generation', 'Charge time and empty-cabinet hours by temperature and charger generation.')}
<h3>4.4 Batteries</h3>
<p>{('Lots ' + bad_lots_txt + f" degrade {V('bad_lot_degradation_multiple'):.1f}× faster than every other lot, costing {inr(V('bad_lot_excess_wear_total_inr'))} in excess wear over the period. ") if BAD_LOTS else 'No manufacturing lot degrades abnormally. '}
Worn packs deliver less range (2W median {V('range_2w_low_soh'):.0f} km at the lowest health band vs {V('range_2w_high_soh'):.0f} km at the
highest), and riders who receive them come back sooner ({V('gap_h_low_soh'):.1f} h vs {V('gap_h_high_soh'):.1f} h to the next attempt),
which adds load to the stations.</p>
{img('q4_lot_degradation', 'Degradation rate by manufacturing lot.')}
{img('q4_range_frequency', 'Range and time to next swap by pack health.')}
<h3>4.5 Pricing and partners</h3>
<p>The pilot moved the peak-hour share of attempts by {d_share['DiD estimate']:+.2%} and the peak-hour failure rate by
{d_fail['DiD estimate']:+.2%} relative to other cities (p = {d_share['p']:.2g} and {d_fail['p']:.2g}). {lp_name}, the largest partner with
{pct(V('largest_partner_swap_share'), 0)} of swaps, earns {inr(V('largest_partner_contribution_per_swap'))} per swap against
{inr(V('independent_contribution_per_swap'))} from independent riders.</p>
{img('q5_pilot_did', 'Peak-hour share of attempts, pilot cities vs other cities.')}
{img('q5_partner_value', 'Contribution per swap by customer after costs and payment terms.')}
<h3>4.6 Why new riders leave</h3>
<p>Month-2 retention is {pct(V('retention_no_stockout'))} for new riders with no stockout in their first 14 days and
{pct(V('retention_with_stockout'))} for those who hit one. <b>Primary drivers:</b> {', '.join(primary) or 'none met all three tests'}.
<b>Secondary:</b> {', '.join(secondary) or 'none'}. <b>Not supported by the data:</b> {', '.join(not_supported) or 'none'}.
At first-quarter service levels, the latest quarter's retention would be {pct(V('retention_last_q_if_q1_service'))} instead of
{pct(V('retention_last_q_actual'))}.</p>
{img('q6_attributable_churn', 'Percentage points of churn attributable to each factor.')}
{img('q6_retention_trend', 'Month-2 retention by signup month.')}

<h2>5. Business findings: the four budget proposals</h2>
<table><tr><th>Proposal</th><th>Verdict</th><th>Evidence</th></tr>{rows(verdict_df)}</table>
<p>A failed swap costs {inr(V('direct_cost_per_failure_inr'))} in immediate margin ({pct(V('failures_recovered_1h'), 0)} of failed riders
recover within an hour) and an early failure costs {inr(V('churn_cost_per_early_failure_inr'))} in a new rider's future margin.</p>

<h2>6. Recommendations</h2>
<ol>{''.join(f"<li><b>{_html.escape(r.action)}</b> — worth about {inr(r.inr_per_year)} a year ({_html.escape(r.basis)}).</li>" for r in levers_inr.itertuples())}
<li><b>Fix the data plumbing:</b> block retired pack IDs at the cabinet, correct the v3.2.0 time zone, add idempotent event IDs for offline sync,
and remove test stations from revenue reporting.</li>
<li><b>Track five leading indicators weekly:</b> Gen1 hours above 40°C with an empty cabinet, lot-level SoH slope, share of new riders with
a failure in their first 14 days, contribution per swap by partner, and packs issued past retirement.</li></ol>
<p class="sub">All figures come from the accompanying Colab notebook; nothing in this report is hand-entered.</p>
</body></html>"""
(OUT_DIR / "VoltRelay_Report.html").write_text(report, encoding="utf-8")

script = f"""# VoltRelay — 3-minute video script
Speakers: **Monish Kannaha S A** (M) and **Lokeshkumar K** (L). About 420 words ≈ 3 minutes. Show the report charts named in brackets.

**[0:00–0:30] M — The business problem** [q1_four_metrics]
VoltRelay's swaps grew {V('growth_completed_x'):.1f} times and revenue {V('growth_revenue_x'):.1f} times in eighteen months. {growth_txt},
{ret_txt}, and contribution per swap went from ₹{V('contribution_per_swap_first3'):.0f} to ₹{V('contribution_per_swap_last3'):.0f}. Leadership has four budget proposals on the table. Our job: find what is really driving this before they spend.

**[0:30–1:05] L — Our approach**
We cleaned all nine data issues first — for example {int(V('clock_bug_events')):,} timestamps from a firmware bug were shifted by five and a half hours,
and {int(V('duplicates_removed')):,} duplicate records removed. Then we built a profit-and-loss for every swap — energy, battery wear and site cost —
used models to separate overlapping causes, and converted every problem into one currency: rupees per year.

**[1:05–2:15] M and L — What we found** [q2_concentration, q3_heat_by_generation, q4_lot_degradation, q6_attributable_churn]
M: {conc_txt}
L: {gen1_txt}
M: {('Battery lots ' + bad_lots_txt + f" wear out {V('bad_lot_degradation_multiple'):.1f} times faster, eating margin and range.") if BAD_LOTS else 'Battery wear is steady across lots.'}
L: And new riders who hit a stockout in their first two weeks stay only {pct(V('retention_with_stockout'), 0)} of the time, versus
{pct(V('retention_no_stockout'), 0)} without one. {('Primary churn drivers: ' + ', '.join(primary) + '.') if primary else ''}
M: {cost_txt}

**[2:15–3:00] L — Recommendations** [levers table]
""" + "\n".join(f"{i + 1}. {r.action} — about {inr(r.inr_per_year)} a year." for i, r in enumerate(levers_inr.itertuples())) + f"""

M: So: {verdict_df.iloc[0]['verdict'].lower()} on new stations, {verdict_df.iloc[1]['verdict'].lower()} on batteries,
{verdict_df.iloc[2]['verdict'].lower()} on pricing, and {verdict_df.iloc[3]['verdict'].lower()} on the exclusive. Fix before you grow. Thank you.
"""
(OUT_DIR / "VoltRelay_Video_Script.md").write_text(script, encoding="utf-8")
print(script)
print(f"Report written to {(OUT_DIR / 'VoltRelay_Report.html').resolve()}")
try:
    from google.colab import files
    import shutil
    shutil.make_archive("VoltRelay_submission", "zip", root_dir=".", base_dir=str(OUT_DIR))
    files.download(str(OUT_DIR / "VoltRelay_Report.html"))
    files.download(str(OUT_DIR / "VoltRelay_Video_Script.md"))
    files.download("VoltRelay_submission.zip")
except Exception as e:
    print("Not running in Colab — files are in", OUT_DIR.resolve())
