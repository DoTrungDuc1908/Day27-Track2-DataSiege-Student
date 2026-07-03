"""Cost-aware, single-pass defenses for all Data Siege event types."""

from api import Verdict


INNER_FRACTION = 0.3175
DATA_FRACTION = 0.3175
METRIC_FRACTIONS = {
    "contract_freshness": 0.84,
}


def register(ctx):
    ctx.on("data_batch", check_data_batch)
    ctx.on("contract_checkpoint", check_contract_checkpoint)
    ctx.on("lineage_run", check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch", check_embedding_batch)


def _verdict(alert, pillar, reasons):
    return Verdict(
        alert=bool(alert),
        confidence=0.98 if alert else 0.90,
        reason=", ".join(reasons) if reasons else "within calibrated limits",
        pillar=pillar,
    )


def _tool_failed(result):
    return not isinstance(result, dict) or "error" in result


def _median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _above_inner(value, low, high):
    """Detect the upper-tail shifts used by volume spikes and amount drift."""
    center = (low + high) / 2.0
    radius = (high - low) * 0.5 * DATA_FRACTION
    return value > center + radius


def _adaptive_high(ctx, key, value, published_max):
    """Learn a robust center online, without letting an outlier set its threshold."""
    history = ctx.state.setdefault("history", {}).setdefault(key, [])
    fraction = METRIC_FRACTIONS.get(key, INNER_FRACTION)
    if len(history) >= 2:
        center = _median(history)
        threshold = center + fraction * (published_max - center)
    else:
        threshold = published_max
    # Never teach the online baseline that an already out-of-band value is normal.
    if value <= published_max:
        history.append(float(value))
    return value > threshold


def check_data_batch(payload, ctx):
    profile = ctx.tools.batch_profile(payload["batch_id"])
    if _tool_failed(profile):
        return _verdict(False, "checks", ["profile unavailable"])

    baseline = ctx.baseline
    reasons = []
    row_count = profile["row_count"]
    null_rate = profile["null_rate"].get("customer_id", 0.0)
    mean_amount = profile["mean_amount"]
    staleness = profile["staleness_min"]

    if _above_inner(
        row_count, baseline["row_count_min"], baseline["row_count_max"]
    ):
        reasons.append("abnormal row volume")
    if _adaptive_high(
        ctx, "null_rate", null_rate, baseline["null_rate_max"]
    ):
        reasons.append("customer_id null-rate spike")
    if _above_inner(
        mean_amount,
        baseline["mean_amount_min"],
        baseline["mean_amount_max"],
    ):
        reasons.append("amount distribution shift")
    if _adaptive_high(
        ctx, "batch_staleness", staleness, baseline["staleness_min_max"]
    ):
        reasons.append("batch freshness lag")

    return _verdict(reasons, "checks", reasons)


def check_contract_checkpoint(payload, ctx):
    diff = ctx.tools.contract_diff(
        payload["contract_id"], payload["checkpoint_batch_id"]
    )
    if _tool_failed(diff):
        return _verdict(False, "contracts", ["contract diff unavailable"])

    reasons = list(diff.get("violations", []))
    if _adaptive_high(
        ctx,
        "contract_freshness",
        diff["freshness_delay_min"],
        ctx.baseline["freshness_delay_max_min"],
    ):
        reasons.append("contract freshness SLA violation")
    return _verdict(reasons, "contracts", reasons)


def check_lineage_run(payload, ctx):
    graph = ctx.tools.lineage_graph_slice(payload["run_id"])
    if _tool_failed(graph):
        return _verdict(False, "lineage", ["lineage graph unavailable"])

    reasons = []
    expected_upstream = {"raw.orders", "raw.customers"}
    if set(graph["actual_upstream"]) != expected_upstream:
        reasons.append("upstream edge mismatch")
    if graph["actual_downstream_count"] != 1:
        reasons.append("orphaned output")
    if _adaptive_high(
        ctx,
        "lineage_duration",
        graph["duration_ms"],
        ctx.baseline["lineage_duration_ms_max"],
    ):
        reasons.append("runtime anomaly")
    return _verdict(reasons, "lineage", reasons)


def check_feature_materialization(payload, ctx):
    drift = ctx.tools.feature_drift(
        payload["feature_view"], payload["batch_id"]
    )
    if _tool_failed(drift):
        return _verdict(False, "ai_infra", ["feature drift unavailable"])

    reasons = []
    if _adaptive_high(
        ctx,
        "feature_shift",
        drift["mean_shift_sigma"],
        ctx.baseline["feature_mean_shift_sigma_max"],
    ):
        reasons.append("training-serving feature skew")
    return _verdict(reasons, "ai_infra", reasons)


def check_embedding_batch(payload, ctx):
    drift = ctx.tools.embedding_drift(
        payload["corpus"], payload["chunk_batch_id"]
    )
    if _tool_failed(drift):
        return _verdict(False, "ai_infra", ["embedding drift unavailable"])

    reasons = []
    if _adaptive_high(
        ctx,
        "embedding_shift",
        drift["centroid_shift"],
        ctx.baseline["embedding_centroid_shift_max"],
    ):
        reasons.append("embedding centroid drift")
    if _adaptive_high(
        ctx,
        "corpus_age",
        drift["avg_doc_age_days"],
        ctx.baseline["corpus_avg_doc_age_days_max"],
    ):
        reasons.append("RAG corpus staleness")
    return _verdict(reasons, "ai_infra", reasons)
