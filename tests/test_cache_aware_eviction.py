"""Tests for cache-aware (Lifecycle-Aware) eviction batching in compaction.

These exercise the planner against the *real* CompactionSettings model and the
existing compaction-threshold helper, so they cover the wiring used by
``letta.services.summarizer.compact.compact_messages``.
"""

from letta.schemas.llm_config import LLMConfig
from letta.services.summarizer.cache_aware_eviction import (
    DEFAULT_MAX_EVICTION_FRACTION,
    apply_cache_aware_eviction,
    plan_cache_aware_eviction,
)

# Non-new modules: the config type and threshold helper the integration relies on.
from letta.services.summarizer.summarizer_config import CompactionSettings
from letta.services.summarizer.thresholds import get_compaction_trigger_threshold


def _llm_config(context_window: int = 100_000) -> LLMConfig:
    return LLMConfig(
        model="claude-haiku-4-5",
        model_endpoint_type="anthropic",
        model_endpoint="https://api.anthropic.com/v1",
        context_window=context_window,
    )


def test_batches_eviction_when_over_trigger():
    """Over the trigger, eviction is increased to leave headroom below it."""
    plan = plan_cache_aware_eviction(
        current_eviction_fraction=0.30,
        context_tokens_before=100_000,
        trigger_threshold=100_000,  # trigger at 100% of window
        context_window=100_000,
        headroom_fraction=0.40,
    )
    assert plan.adjusted
    # low-water = trigger(1.0) - headroom(0.40) = 0.60 kept -> evict 0.40 (> 0.30).
    assert abs(plan.eviction_fraction - 0.40) < 1e-9
    # Post-compaction usage target is below the trigger fraction.
    assert plan.target_low_water_fraction < 1.0


def test_batches_more_aggressively_with_larger_headroom():
    plan = plan_cache_aware_eviction(
        current_eviction_fraction=0.30,
        context_tokens_before=90_000,
        trigger_threshold=90_000,
        context_window=100_000,  # trigger fraction = 0.9
        headroom_fraction=0.35,
    )
    # low-water = 0.9 - 0.35 = 0.55 kept -> evict 0.45 (> configured 0.30)
    assert plan.adjusted
    assert abs(plan.eviction_fraction - 0.45) < 1e-9
    assert abs(plan.target_low_water_fraction - 0.55) < 1e-9


def test_eviction_is_capped_for_conservatism():
    plan = plan_cache_aware_eviction(
        current_eviction_fraction=0.30,
        context_tokens_before=100_000,
        trigger_threshold=100_000,
        context_window=100_000,
        headroom_fraction=0.9,  # would want to evict ~0.9
        max_eviction_fraction=DEFAULT_MAX_EVICTION_FRACTION,
    )
    assert plan.eviction_fraction <= DEFAULT_MAX_EVICTION_FRACTION + 1e-9


def test_no_op_below_trigger():
    """Below the trigger, do not over-evict fresh context."""
    plan = plan_cache_aware_eviction(
        current_eviction_fraction=0.30,
        context_tokens_before=10_000,
        trigger_threshold=90_000,
        context_window=100_000,
    )
    assert not plan.adjusted
    assert plan.eviction_fraction == 0.30


def test_no_op_when_signal_missing():
    plan = plan_cache_aware_eviction(
        current_eviction_fraction=0.30,
        context_tokens_before=None,
        trigger_threshold=None,
        context_window=100_000,
    )
    assert not plan.adjusted
    assert plan.eviction_fraction == 0.30


def test_never_reduces_configured_eviction():
    """A high configured eviction fraction is preserved (we only ever batch up)."""
    plan = plan_cache_aware_eviction(
        current_eviction_fraction=0.55,
        context_tokens_before=100_000,
        trigger_threshold=90_000,
        context_window=100_000,
        headroom_fraction=0.25,  # would only want ~0.35 eviction
    )
    assert not plan.adjusted
    assert plan.eviction_fraction == 0.55


def test_apply_adjusts_sliding_window_config():
    cfg = CompactionSettings(mode="sliding_window", sliding_window_percentage=0.30)
    llm_config = _llm_config(100_000)
    trigger = get_compaction_trigger_threshold(llm_config)

    batched, plan = apply_cache_aware_eviction(
        cfg,
        context_tokens_before=trigger,
        trigger_threshold=trigger,
        context_window=llm_config.context_window,
        headroom_fraction=0.35,
    )

    assert plan.adjusted
    assert batched is not cfg  # returns a copy, original untouched
    assert cfg.sliding_window_percentage == 0.30
    assert batched.sliding_window_percentage == plan.eviction_fraction
    assert batched.sliding_window_percentage > 0.30
    # Other fields are carried over unchanged.
    assert batched.mode == "sliding_window"


def test_apply_is_noop_for_whole_history_mode():
    """'all' mode ignores the eviction knob, so the config is returned unchanged."""
    cfg = CompactionSettings(mode="all", sliding_window_percentage=0.30)

    batched, _ = apply_cache_aware_eviction(
        cfg,
        context_tokens_before=100_000,
        trigger_threshold=90_000,
        context_window=100_000,
        headroom_fraction=0.35,
    )
    assert batched is cfg


def test_apply_to_self_compact_sliding_window_mode():
    cfg = CompactionSettings(mode="self_compact_sliding_window", sliding_window_percentage=0.30)

    batched, plan = apply_cache_aware_eviction(
        cfg,
        context_tokens_before=95_000,
        trigger_threshold=90_000,
        context_window=100_000,
        headroom_fraction=0.30,
    )
    assert plan.adjusted
    assert batched.mode == "self_compact_sliding_window"
    assert batched.sliding_window_percentage > 0.30
