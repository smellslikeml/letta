"""Cache-aware eviction planning for conversation compaction.

Adapted from the *Lifecycle-Aware Eviction* idea in TokenPilot: Cache-Efficient
Context Management for LLM Agents (https://arxiv.org/abs/2606.17016v1).

Every compaction rewrites the in-context message sequence and inserts a fresh
summary message, which invalidates the prompt-cache prefix from that point on.
The paper observes a trade-off between *text sparsity* (evicting aggressively to
shrink the token footprint) and *prompt-cache continuity* (mutating the sequence
as rarely as possible). Its remedy is a "conservative batch-turn schedule": when
an eviction does happen, reclaim a large enough batch that the next eviction is
deferred as long as possible, so the (expensive) cache-invalidating event is
amortized over many turns instead of firing repeatedly while reclaiming little.

Letta's sliding-window summarizers already accept a single eviction knob
(``CompactionSettings.sliding_window_percentage``) that determines how much
context a compaction reclaims. This module turns that knob into a cache-aware
batch target: given the current context usage relative to the compaction trigger
threshold, it picks an eviction fraction that lands post-compaction usage at a
"low-water mark" comfortably below the trigger, leaving headroom so the agent can
run many more turns before the next cache-breaking compaction. The adjustment is
conservative and bounded — it only ever evicts *more* than the configured
fraction (never less) and is capped so we keep recent, task-relevant context.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from letta.services.summarizer.summarizer_config import CompactionSettings

# Modes whose eviction is governed by ``sliding_window_percentage``. Whole-history
# modes ("all") ignore the knob, so cache-aware batching does not apply to them.
SLIDING_WINDOW_MODES = frozenset({"sliding_window", "self_compact_sliding_window"})

# Fraction of the context window to leave free *below the trigger threshold* after
# a compaction. Larger headroom => fewer, larger compactions => fewer cache breaks.
DEFAULT_HEADROOM_FRACTION = 0.25

# Never let cache-aware batching evict more than this fraction of context in one
# pass. Keeps the schedule "conservative" so recent, task-relevant turns survive.
DEFAULT_MAX_EVICTION_FRACTION = 0.6

# Floor for the post-compaction low-water mark, so a low/odd trigger threshold can
# never drive us to evict almost everything.
MIN_LOW_WATER_FRACTION = 0.4


@dataclass(frozen=True)
class CacheAwareEvictionPlan:
    """The outcome of cache-aware eviction planning for a single compaction."""

    eviction_fraction: float
    """Recommended ``sliding_window_percentage`` to use for this compaction."""

    original_fraction: float
    """The configured eviction fraction before any cache-aware adjustment."""

    adjusted: bool
    """True if the recommended fraction differs from the configured one."""

    target_low_water_fraction: float
    """Post-compaction context usage (as a fraction of the window) we aimed for."""

    reason: str
    """Human-readable explanation, suitable for logging/telemetry."""


def plan_cache_aware_eviction(
    *,
    current_eviction_fraction: float,
    context_tokens_before: Optional[int],
    trigger_threshold: Optional[int],
    context_window: Optional[int],
    headroom_fraction: float = DEFAULT_HEADROOM_FRACTION,
    max_eviction_fraction: float = DEFAULT_MAX_EVICTION_FRACTION,
) -> CacheAwareEvictionPlan:
    """Pick a cache-aware eviction fraction for an upcoming compaction.

    The recommended fraction targets a post-compaction "low-water mark" of
    ``(trigger_fraction - headroom_fraction)`` of the context window, so the agent
    has room to keep accumulating context for many turns before the next
    cache-invalidating compaction. The result is clamped to never evict *less*
    than ``current_eviction_fraction`` and never more than
    ``max_eviction_fraction``.

    When inputs are missing or out of range (no token estimate, no threshold, or
    usage below the trigger), this is a no-op and returns the configured fraction
    unchanged — the caller's existing behavior is preserved.

    Args:
        current_eviction_fraction: Configured ``sliding_window_percentage``.
        context_tokens_before: Estimated in-context tokens prior to compaction.
        trigger_threshold: Token count at/above which compaction is triggered.
        context_window: The agent model's context window in tokens.
        headroom_fraction: Free fraction to leave below the trigger after eviction.
        max_eviction_fraction: Upper bound on how much to evict in one pass.

    Returns:
        A :class:`CacheAwareEvictionPlan`.
    """
    if (
        context_window is None
        or context_window <= 0
        or trigger_threshold is None
        or trigger_threshold <= 0
        or context_tokens_before is None
    ):
        return CacheAwareEvictionPlan(
            eviction_fraction=current_eviction_fraction,
            original_fraction=current_eviction_fraction,
            adjusted=False,
            target_low_water_fraction=1.0 - current_eviction_fraction,
            reason="insufficient signal for cache-aware batching; using configured eviction fraction",
        )

    # Only batch up when we are actually at/over the trigger. Below it, compaction
    # is proactive/forced and over-evicting would needlessly drop fresh context.
    if context_tokens_before < trigger_threshold:
        return CacheAwareEvictionPlan(
            eviction_fraction=current_eviction_fraction,
            original_fraction=current_eviction_fraction,
            adjusted=False,
            target_low_water_fraction=1.0 - current_eviction_fraction,
            reason=(
                f"usage {context_tokens_before} below trigger {trigger_threshold}; "
                "no cache-aware batching needed"
            ),
        )

    trigger_fraction = trigger_threshold / context_window
    low_water_fraction = max(MIN_LOW_WATER_FRACTION, trigger_fraction - headroom_fraction)

    # eviction_fraction is the fraction reclaimed; (1 - it) is the post-compaction target.
    desired_fraction = 1.0 - low_water_fraction
    eviction_fraction = min(max(desired_fraction, current_eviction_fraction), max_eviction_fraction)

    # Re-derive the achievable low-water mark after clamping (max_eviction_fraction
    # may keep us above the desired mark; that is fine and stays correct/conservative).
    achieved_low_water = 1.0 - eviction_fraction
    adjusted = eviction_fraction > current_eviction_fraction + 1e-9

    if adjusted:
        reason = (
            f"batching eviction {current_eviction_fraction:.2f}->{eviction_fraction:.2f} "
            f"to target {achieved_low_water:.0%} post-compaction usage "
            f"(trigger {trigger_fraction:.0%}, headroom {headroom_fraction:.0%}) "
            "and amortize prompt-cache invalidation over more turns"
        )
    else:
        reason = (
            f"configured eviction {current_eviction_fraction:.2f} already meets the "
            f"cache-aware batch target ({achieved_low_water:.0%} post-compaction usage)"
        )

    return CacheAwareEvictionPlan(
        eviction_fraction=eviction_fraction,
        original_fraction=current_eviction_fraction,
        adjusted=adjusted,
        target_low_water_fraction=achieved_low_water,
        reason=reason,
    )


def apply_cache_aware_eviction(
    summarizer_config: "CompactionSettings",
    *,
    context_tokens_before: Optional[int],
    trigger_threshold: Optional[int],
    context_window: Optional[int],
    headroom_fraction: float = DEFAULT_HEADROOM_FRACTION,
    max_eviction_fraction: float = DEFAULT_MAX_EVICTION_FRACTION,
) -> Tuple["CompactionSettings", CacheAwareEvictionPlan]:
    """Return a (possibly batched) copy of ``summarizer_config`` plus the plan.

    For sliding-window modes the returned config carries the cache-aware
    ``sliding_window_percentage``; for whole-history modes (and any no-op plan)
    the original config is returned unchanged so behavior is identical to before.
    """
    plan = plan_cache_aware_eviction(
        current_eviction_fraction=summarizer_config.sliding_window_percentage,
        context_tokens_before=context_tokens_before,
        trigger_threshold=trigger_threshold,
        context_window=context_window,
        headroom_fraction=headroom_fraction,
        max_eviction_fraction=max_eviction_fraction,
    )

    if summarizer_config.mode not in SLIDING_WINDOW_MODES or not plan.adjusted:
        return summarizer_config, plan

    batched_config = summarizer_config.model_copy(update={"sliding_window_percentage": plan.eviction_fraction})
    return batched_config, plan
