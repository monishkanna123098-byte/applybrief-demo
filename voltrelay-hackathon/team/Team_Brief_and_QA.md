# VoltRelay — Team brief (read this before recording)

**Team:** Monish Kannaha S A, Lokeshkumar K
**One-line story:** *Growth hid a quality problem. Fix the causes (old cabinets in the heat, bad battery lots, bad partner terms, early failures for new riders) before spending on growth.*

All numbers come from the notebook's final `RESULTS` block and the generated report. Never quote a number you can't point to in the report.

---

## How the analysis works (say it in your own words)

1. **Cleaning first.** The organiser planted 9 data problems. We fixed each one and can say how many rows it touched:
   - **Test stations** (`STN-TST…`) removed: they aren't real customers.
   - **Duplicates**: the same swap uploaded twice by cabinets with weak internet (`offline_batch`). We removed the second copy.
   - **Clock bug**: firmware v3.2.0 wrote times 5h30m early (10 Mar–14 Apr 2025). We added 5h30m back and proved it by showing that the hour-of-day pattern then matches normal stations.
   - **City spellings** ("Bangalore", "BLR", …) mapped to 6 cities.
   - **Outliers**: negative/huge km set to missing; charge above 100% capped at 100.
   - **Missing telemetry** is *not measured*, never treated as zero.
   - **CSAT** is mostly missing, and missing more when tickets were slow, so a plain average looks too happy. We don't use it as a KPI.
   - **Ticket categories** are sometimes wrong. We re-labelled battery/range complaints hidden under "other"/"app_issue" using English + Hinglish keywords ("battery jaldi khatam", "pickup kam").
   - **Failures kept, not deleted.** Non-completed attempts *are* the service failures, but they're never counted as revenue.
2. **Per-swap profit (contribution margin).** Price paid − energy (kWh × local electricity price) − battery wear (how much of the pack's life one swap uses × pack cost) − site cost (rent + maintenance spread over that station's swaps).
3. **Separating causes.** Old cabinets, hot weather and certain cities overlap. A **logistic regression** estimates each one's effect *while holding the others fixed*.
4. **Pricing pilot = difference-in-differences.** Change in pilot cities minus change in other cities over the same weeks. That removes seasonality and growth that affected everyone.
5. **Churn drivers.** New riders are *retained* if they swap again in days 31–60. We tested early-experience factors (first 14 days) with a logistic regression plus a gradient-boosting check. We then computed **attributable churn**: how much churn disappears if that factor were fixed, which combines "how strong" with "how many riders it hits".
   - **Primary** = significant, big attributable share, and important in both models.
   - **Secondary** = real but smaller.
6. **One currency.** Every problem is converted to **₹ per year**, so the fixes can be ranked against the four budget proposals. We give **break-even values** ("worth up to ₹X per cabinet per year") instead of inventing costs the data doesn't have.

## Key definitions (judges may ask)

| Term | Meaning |
|---|---|
| Service failure | No charged battery, abandoned in the queue, or system error. Rider cancellations are counted separately. |
| SoH (state of health) | Battery capacity vs new (100%). Lower = less range. |
| SoC (state of charge) | How full the battery is right now. |
| Gen1 / Gen2 / Gen3 | Charger cabinet generations; Gen1 is the oldest. |
| Contribution margin per swap | Revenue − energy − battery wear − site cost, per completed swap. |
| Odds ratio | >1 means the factor raises the chance of churn/failure; <1 lowers it. |
| p-value | Chance of seeing an effect this big if there were really none. We call p < 0.05 "significant". |
| Difference-in-differences | (Pilot after − before) − (others after − before). |

## Likely judge questions and how to answer

1. **"Why should we trust your numbers?"** Every issue was measured before it was fixed, and each fix was re-checked (e.g. the clock-bug fix restores the normal hourly pattern). The report is generated from the notebook, so nothing is typed by hand.
2. **"Is this causation?"** No. It's observational data, as the brief says. We compare like with like (same city, hour, heat) and check with two different models, so it's a well-evidenced case, not proof.
3. **"Why not just build more stations?"** Failures are concentrated at specific stations and hours and are explained by old cabinets in the heat. New stations don't fix an existing Gen1 cabinet failing at 3 pm in May. Point to the Gen1 excess-failure share.
4. **"How did you pick 'primary' drivers?"** Three tests must agree: statistical significance, large attributable churn, and high importance in the second model.
5. **"Where does the ₹ cost of a failure come from?"**
   - *Direct:* if the rider doesn't complete a swap within 60 minutes, that swap's margin is lost.
   - *Churn:* the model's increase in churn from a first stockout × what a retained rider earns us in a year.
6. **"Why a break-even and not an ROI?"** The data has no vendor prices for cooling kits or new cabinets. Break-even tells leadership the maximum worth paying, so they can compare it with real quotes.
7. **"What about competitors?"** We tested competitor promotions and nearby competitor sites. Say whatever the attributable-churn chart shows for them (usually small).
8. **"What would you do first on Monday?"** Recommendation #1 in the report, which is ranked by ₹ per year.
9. **"Anything unusual?"** Packs issued after their retirement date (a safety issue), test stations booking revenue, and the firmware clock bug.

## Recording the 3-minute video (fastest way)

1. Open `VoltRelay_Report.html` from the Colab download in Chrome, so the charts are visible.
2. Screen-record while you talk. Options: Windows **Win+Alt+R** (Game Bar) or **Win+Shift+S → Record**, Mac **Cmd+Shift+5**, or record a Google Meet with both of you presenting.
3. Follow `VoltRelay_Video_Script.md`. Keep the required order: **business problem → approach → key insights → recommendations**. Both speakers take turns (the script marks M / L).
4. Rehearse once, record once or twice, keep it under 3:00.

## Submission checklist (Drive folder: Devengers_hackathon)

- [ ] **Notebook**: in Colab, after Run all, **File → Download → .ipynb** (keeps outputs) and upload it. Also share the Colab link (Share → Anyone with the link).
- [ ] **Report**: open `VoltRelay_Report.html` in Chrome → **Ctrl+P → Save as PDF** → upload the PDF.
- [ ] **Video**: upload the MP4.
- [ ] Set the folder to **Anyone with the link → Viewer** and submit that folder link.
