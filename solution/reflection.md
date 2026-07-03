# Reflection

**Which fault types were hardest to catch, and why?**

The hardest cases were subtle numerical faults: near-normal freshness/runtime
delays, feature skew, and embedding drift. The published limits are calibrated
near three standard deviations, so they catch obvious faults but can miss
smaller shifts. Tightening every limit globally increased false positives.
The final defense therefore combines exact invariants for contracts and
lineage with asymmetric inner numerical bands. For one-sided metrics it learns
a robust median from the stream, while refusing to add already out-of-band
observations to that baseline. This prevents a fault-heavy period from
teaching the detector that broken behavior is normal. The private-oriented
band is intentionally sensitive (about 0.95 sigma for most numerical signals);
contract freshness stays conservative because its exact contract violations
already provide strong coverage.

**What would you change about your cost/coverage tradeoff, if you had another pass?**

I chose one metered call per event. That gives complete private-phase coverage
for about 300 of the 320 available credits, with no redundant calls or cost
overage. Because a missed private fault costs roughly 4.5 times as much as a
false positive, I accepted a higher FPR to reach 94.44% TPR; the final score
was 40.65 with zero cost penalty. With another pass I would calibrate separate
per-metric likelihood thresholds from a larger clean reference sample instead
of sharing one sensitivity factor. Under the tighter public budget I would
selectively sample expensive AI checks; under the private budget, skipping
them saves credits that have no score value while risking missed faults.
