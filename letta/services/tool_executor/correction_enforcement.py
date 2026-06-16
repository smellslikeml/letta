"""Runtime enforcement of user corrections on proposed tool calls.

Adapted from TRACE — "Getting Better at Working With You: Compiling User
Corrections into Runtime Enforcement for Coding Agents"
(arXiv:2606.13174, https://github.com/YujunZhou/tellonce).

TRACE observes that agents can *recall* a user preference and still *violate*
it: memory recall is not the same as compliance. Its remedy is to compile a
user's own corrections into atomic rules and run them as runtime checks that a
proposed agent action must pass before the task is allowed to complete.

This module implements the "compiled enforcement" half of that loop for Letta's
tool-execution forward path. Mining free-text corrections and rewriting them
into atomic rules (TRACE's acquisition half) is intentionally left to an
upstream/LLM step; here we accept *already-compiled* rules stored in the
agent's block_manager-backed memory and enforce them deterministically, with no
model call, against each proposed tool call.

Rule grammar (one rule per line, stored in any memory block; non-matching lines
such as prose or comments are ignored)::

    <mode> | <tool-glob> | <field> | <regex> | <repair message>

* ``mode``      -- ``forbid`` (block when the pattern matches) or ``require``
                   (block when the pattern is absent).
* ``tool-glob`` -- ``fnmatch`` pattern over the tool name; ``*`` matches any.
* ``field``     -- argument key to inspect, or ``*`` for the whole call's
                   JSON-serialized arguments.
* ``regex``     -- Python regular expression evaluated against the target text.
* ``repair``    -- guidance surfaced back to the agent so it can revise.

Example (the "always use snake_case" correction from the paper's motivation)::

    forbid | run_code | code | [a-z][a-zA-Z0-9]*[A-Z] | use snake_case for variable names, not camelCase
"""

import fnmatch
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from letta.log import get_logger

logger = get_logger(__name__)

# A rule line must begin with one of these tokens, which keeps prose lines that
# happen to contain "|" from being mis-parsed as rules.
_VALID_MODES = ("forbid", "require")


@dataclass
class CorrectionRule:
    """A single compiled correction, derived from a user's own chat correction."""

    mode: str  # "forbid" or "require"
    tool_glob: str  # fnmatch pattern over the tool name; "*" matches any
    field: str  # argument key to inspect, or "*" for all serialized args
    pattern: "re.Pattern[str]"
    repair: str

    def applies_to(self, function_name: str) -> bool:
        return fnmatch.fnmatch(function_name or "", self.tool_glob)

    def _target_text(self, function_args: Dict[str, Any]) -> Optional[str]:
        if self.field == "*":
            return json.dumps(function_args, default=str, sort_keys=True)
        if function_args and self.field in function_args:
            value = function_args[self.field]
            return value if isinstance(value, str) else json.dumps(value, default=str)
        return None

    def violated_by(self, function_name: str, function_args: Dict[str, Any]) -> bool:
        """Return True when a proposed call breaches this correction."""
        if not self.applies_to(function_name):
            return False
        target = self._target_text(function_args or {})
        if target is None:
            # The field this rule cares about is not present in this call.
            return False
        found = self.pattern.search(target) is not None
        return found if self.mode == "forbid" else not found


@dataclass
class ComplianceReport:
    """Outcome of checking a proposed tool call against compiled corrections."""

    allowed: bool = True
    violations: List[str] = field(default_factory=list)

    def format_guidance(self, function_name: str) -> str:
        """Human-readable block message handed back to the agent for repair."""
        header = (
            f"Blocked call to `{function_name}`: it violates {len(self.violations)} "
            f"correction(s) you previously gave. Revise and try again."
        )
        bullets = "\n".join(f"  - {v}" for v in self.violations)
        return f"{header}\n{bullets}"


def parse_correction_rules(text: Optional[str]) -> List[CorrectionRule]:
    """Parse compiled correction rules out of a memory block's text.

    Lines that do not match the grammar (prose, comments, blanks) are skipped,
    so a correction block can hold both rules and human notes.
    """
    rules: List[CorrectionRule] = []
    if not text:
        return rules
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 5:
            continue
        mode, tool_glob, fld, pattern_src, repair = parts
        mode = mode.lower()
        if mode not in _VALID_MODES or not pattern_src or not repair:
            continue
        try:
            compiled = re.compile(pattern_src)
        except re.error as e:
            logger.warning(f"Skipping correction rule with invalid regex {pattern_src!r}: {e}")
            continue
        rules.append(
            CorrectionRule(
                mode=mode,
                tool_glob=tool_glob or "*",
                field=fld or "*",
                pattern=compiled,
                repair=repair,
            )
        )
    return rules


def extract_correction_rules(agent_state: Any) -> List[CorrectionRule]:
    """Collect compiled correction rules from an agent's in-context memory.

    Duck-typed against ``agent_state.memory.get_blocks()`` so it works for a
    real ``AgentState`` as well as lightweight stand-ins. Any failure to read
    memory degrades to "no rules" rather than breaking tool execution.
    """
    if agent_state is None:
        return []
    try:
        memory = getattr(agent_state, "memory", None)
        if memory is None:
            return []
        blocks = memory.get_blocks()
    except Exception as e:  # never let enforcement break the forward path
        logger.warning(f"Could not read agent memory for correction enforcement: {e}")
        return []

    rules: List[CorrectionRule] = []
    for block in blocks:
        rules.extend(parse_correction_rules(getattr(block, "value", None)))
    return rules


def check_tool_call(function_name: str, function_args: Dict[str, Any], rules: List[CorrectionRule]) -> ComplianceReport:
    """Check a proposed tool call against compiled corrections (allow/block)."""
    violations = [rule.repair for rule in rules if rule.violated_by(function_name, function_args)]
    return ComplianceReport(allowed=not violations, violations=violations)


def enforce_corrections(function_name: str, function_args: Dict[str, Any], agent_state: Any) -> ComplianceReport:
    """Top-level enforcement hook for the tool-execution forward path.

    Returns an allowing report (zero overhead, no behavior change) when the
    agent has no compiled corrections in memory; otherwise blocks any call that
    breaches one, attaching repair guidance for the agent.
    """
    rules = extract_correction_rules(agent_state)
    if not rules:
        return ComplianceReport(allowed=True)
    return check_tool_call(function_name, function_args, rules)
